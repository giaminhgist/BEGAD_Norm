"""Absolute features + cross-task contrasts (methodology 2).

18 contrast features are built from the raw wide matrix before imputation:
ratio-log contrasts log(x_KD2/x_KD1), log(x_KD3/x_KD1) for D023/A002/A003
(guarded: both components must be positive) and difference contrasts
x_KD2 - x_KD1, x_KD3 - x_KD1 for D025/D026/D035/D036/D017/D022. Missing
components propagate as NaN and are imputed per fold (median, fitted on the
training split only). Three representations share the identical outer splits,
inner CV seeds and tuning grids: abs132 (132 absolute), abs150 (132 + 18
contrasts), compact84 (33 KD1 + 33 READING + 18 contrasts).

Per (problem, model_type, representation, fold): hyperparameters are selected
by inner CV (4 folds, seed 42, mean log-loss) on the training fold and the
model is refitted on the full training fold; inner OOF probabilities
(inner-inner seed 7) provide the training-level Youden threshold and the
inner log-loss. Models are unweighted; threshold 0.5 for headline binary
metrics.

Usage:
    PYTHONPATH=src .venv/bin/python -m contrasts.run_contrasts \
        --config src/contrasts/config.yaml --smoke
    PYTHONPATH=src .venv/bin/python -m contrasts.run_contrasts \
        --config src/contrasts/config.yaml
"""

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

# penalty='elasticnet' is deprecated in sklearn 1.9 but fully supported in the
# pinned environment (1.9.1); the warning is recorded here and filtered to keep
# run logs readable.
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="sklearn.linear_model._logistic")

import numpy as np
import yaml

from expcommon.data import build_problem, fit_prep, fold_masks, load_split, \
    load_wide, task_columns
from expcommon.metrics import BIN_METRICS, MC_METRICS, binary_metrics, \
    multiclass_metrics, youden_threshold
from expcommon.models import fit_predict_proba, inner_oof, logloss, tune_params
from expcommon.util import append_run_log, sha256_file, write_provenance

ROOT = Path(__file__).resolve().parents[2]

# "bacc" (balanced accuracy = macro recall) is shared by both problem types;
# it is the only metric appearing in both BIN_METRICS and MC_METRICS, so the
# MC columns below exclude it to keep the CSV schema unambiguous.
MC_DETAIL_COLS = [m for m in MC_METRICS if m != "bacc"]
DETAIL_COLS = (["problem", "model_type", "representation", "fold",
                "n_train", "n_val", "thr"] + BIN_METRICS
               + ["thr_y", "f1_y", "bacc_y", "sen_y", "spec_y"] + MC_DETAIL_COLS
               + ["inner_logloss", "best_params", "fit_s"])


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["config_path"] = str(Path(path).resolve())
    cfg["config_sha256"] = sha256_file(Path(path))
    return cfg


def build_contrast_matrix(X, feats, contrast_cfg):
    """Build the 18 contrast columns from the raw (pre-imputation) matrix.

    Returns (column_names, values). For ratio-log contrasts the result is
    finite only when both components are finite and positive; for difference
    contrasts both components must be finite. Otherwise NaN, to be handled by
    the per-fold imputer.
    """
    cols, vals = [], []
    for feat in contrast_cfg["ratio_log"]:
        for suffix, (t2, t1) in contrast_cfg["pairs"].items():
            i1, i2 = feats.index(f"{feat}__{t1}"), feats.index(f"{feat}__{t2}")
            x1, x2 = X[:, i1], X[:, i2]
            v = np.full(len(X), np.nan)
            ok = np.isfinite(x1) & np.isfinite(x2) & (x1 > 0) & (x2 > 0)
            v[ok] = np.log(x2[ok] / x1[ok])
            cols.append(f"{feat}__contrast_{suffix}")
            vals.append(v)
    for feat in contrast_cfg["difference"]:
        for suffix, (t2, t1) in contrast_cfg["pairs"].items():
            i1, i2 = feats.index(f"{feat}__{t1}"), feats.index(f"{feat}__{t2}")
            x1, x2 = X[:, i1], X[:, i2]
            v = np.full(len(X), np.nan)
            ok = np.isfinite(x1) & np.isfinite(x2)
            v[ok] = x2[ok] - x1[ok]
            cols.append(f"{feat}__contrast_{suffix}")
            vals.append(v)
    assert len(cols) == contrast_cfg["expected_n"] == len(vals)
    return cols, np.column_stack(vals)


def contrast_manifest(cols, V, contrast_cfg):
    rows = []
    for j, c in enumerate(cols):
        v = V[:, j]
        feat = c.split("__")[0]
        rows.append({
            "feature": feat, "contrast": c.split("__")[1],
            "type": "ratio_log" if feat in contrast_cfg["ratio_log"] else "difference",
            "n_finite": int(np.isfinite(v).sum()),
            "n_nan": int(np.isnan(v).sum()),
            "min": None if np.isnan(v).all() else f"{np.nanmin(v):.6f}",
            "max": None if np.isnan(v).all() else f"{np.nanmax(v):.6f}",
        })
    return rows


def build_representation(X, feats, rep_spec, C_cols, C_vals):
    idx = []
    for t in rep_spec["abs_tasks"]:
        idx += task_columns(feats, t)
    names = [feats[i] for i in idx]
    Xrep = X[:, idx]
    if rep_spec["add_contrasts"]:
        names += C_cols
        Xrep = np.hstack([Xrep, C_vals])
    assert Xrep.shape[1] == rep_spec["n_features"], \
        (Xrep.shape[1], f"expected {rep_spec['n_features']}")
    return Xrep, names


def detail_row(problem, model_type, rep, fold, n_tr, n_va, y_va, y_tr,
               Pv, Poof, best_params, fit_s, threshold):
    row = {"problem": problem["name"], "model_type": model_type,
           "representation": rep, "fold": fold, "n_train": n_tr,
           "n_val": n_va, "best_params": json.dumps(best_params, sort_keys=True),
           "fit_s": f"{fit_s:.1f}"}
    if problem["positive"] is not None:
        row["thr"] = f"{threshold:.2f}"
        m = binary_metrics(y_va, Pv[:, 1], threshold)
        row.update({k: f"{m[k]:.4f}" for k in BIN_METRICS})
        thr_y = youden_threshold(y_tr, Poof[:, 1])
        my = binary_metrics(y_va, Pv[:, 1], thr_y)
        row["thr_y"] = f"{thr_y:.4f}"
        row["f1_y"] = f"{my['f1']:.4f}"
        row["bacc_y"] = f"{my['bacc']:.4f}"
        row["sen_y"] = f"{my['sen']:.4f}"
        row["spec_y"] = f"{my['spec']:.4f}"
    else:
        row["thr"] = "argmax"
        m = multiclass_metrics(y_va, Pv, Pv.argmax(axis=1))
        row["bacc"] = f"{m['bacc']:.4f}"
        row.update({k: f"{m[k]:.4f}" for k in MC_DETAIL_COLS})
    row["inner_logloss"] = f"{logloss(y_tr, Poof):.4f}"
    return row


def evaluate_rep(cfg, Xrep, y_global, groups, pid_arr, fold_of, held_ids,
                 problem, model_type, rep, nf, fold, proba_rows):
    tr, va = fold_masks(fold_of, held_ids, groups, problem, fold, pid_arr)
    y_tr, y_va = y_global[tr], y_global[va]
    n_tr, n_va = len(tr), len(va)
    if problem["positive"] is None:
        assert (y_va == 2).sum() == cfg["split"]["expect"]["ad_val"], \
            (fold, "AD in 3-class validation = fold AD + heldout AD")
        assert (y_tr == 2).sum() == cfg["split"]["expect"]["ad_train"]
    elif problem["name"] == "co_vs_ad":
        assert (y_va == 1).sum() == cfg["split"]["expect"]["ad_val"]

    prep = fit_prep(Xrep, tr)
    Zt, Zv = prep(Xrep[tr]), prep(Xrep[va])
    K = 3 if problem["positive"] is None else 2
    t0 = time.time()
    oof, _ = inner_oof(Zt, y_tr, model_type, nf,
                       cfg["cv"]["inner_seed"], cfg["cv"]["inner_inner_seed"],
                       cfg["cv"]["inner_folds"])
    params, score = tune_params(Zt, y_tr, model_type, nf,
                                cfg["cv"]["inner_seed"], cfg["cv"]["inner_folds"])
    Pv = fit_predict_proba(Zt, y_tr, Zv, model_type, params)
    fit_s = time.time() - t0
    best = {**params, "inner_cv_logloss": round(score, 6)}

    for i, gi in enumerate(va):
        row = {"subject_id": pid_arr[gi], "group": groups[gi], "fold": fold,
               "role": "heldout_ad" if pid_arr[gi] in held_ids else "cv",
               "y": int(y_va[i])}
        for c in range(K):
            row[f"p{c}_{rep}"] = f"{Pv[i, c]:.8f}"
        proba_rows.append(row)

    return detail_row(problem, model_type, rep, fold, n_tr, n_va, y_va, y_tr,
                      Pv, oof, best, fit_s, cfg["threshold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--smoke", action="store_true",
                    help="first fold only -> results/smoke_detail.csv")
    ap.add_argument("--problems", help="comma-separated problem names")
    ap.add_argument("--models", help="comma-separated model types")
    ap.add_argument("--folds", help="comma-separated folds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    X, pid_arr, groups, feats = load_wide(cfg)
    fold_of, held_ids = load_split(cfg)
    problems = ([p for p in cfg["problems"] if p["name"] in args.problems.split(",")]
                if args.problems else list(cfg["problems"]))
    model_types = args.models.split(",") if args.models else list(cfg["models"].keys())
    folds = [int(f) for f in args.folds.split(",")] if args.folds else \
        list(range(1, cfg["cv"]["n_folds"] + 1))
    if args.smoke:
        folds = folds[:1]

    # contrasts from the raw matrix, before any imputation
    C_cols, C_vals = build_contrast_matrix(X, feats, cfg["contrasts"])
    reps = cfg["representations"]
    rep_matrices = {name: build_representation(X, feats, spec, C_cols, C_vals)
                    for name, spec in reps.items()}

    out_dir = ROOT / cfg["output"]["dir"]
    detail_rel = cfg["output"]["smoke_csv"] if args.smoke else cfg["output"]["detail_csv"]
    detail_path = out_dir / detail_rel
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if detail_path.exists():
        sys.exit(f"refusing to overwrite existing results: {detail_path}")
    manifest_path = out_dir / cfg["output"]["contrast_manifest_csv"]
    if manifest_path.exists() and not args.smoke:
        sys.exit(f"refusing to overwrite existing results: {manifest_path}")
    if args.smoke:
        manifest_path = out_dir / "results/smoke_contrast_manifest.csv"
    proba_prefix = out_dir / cfg["output"]["proba_prefix"]

    run_start = time.time()
    n_rows = 0
    proba_by_key = {}
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        w.writeheader()
        for prob in problems:
            y_global = build_problem(groups, prob)
            for mtype in model_types:
                for fold in folds:
                    for rep, (Xrep, names) in rep_matrices.items():
                        proba_rows = []
                        row = evaluate_rep(cfg, Xrep, y_global, groups, pid_arr,
                                           fold_of, held_ids, prob, mtype, rep,
                                           len(names), fold, proba_rows)
                        w.writerow(row)
                        f.flush()
                        n_rows += 1
                        for r in proba_rows:
                            proba_by_key.setdefault((prob["name"], mtype), []).append(r)
                        disp = row.get("acc") or row.get("auc") or row.get("ovr_auc")
                        disp2 = row.get("auc") or row.get("ovr_auc") or ""
                        print(f"  {prob['name']} {mtype} {rep} fold={fold} "
                              f"acc={disp} auc={disp2} "
                              f"({row['fit_s']}s) {row['best_params']}", flush=True)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "contrast", "type",
                                          "n_finite", "n_nan", "min", "max"])
        w.writeheader()
        w.writerows(contrast_manifest(C_cols, C_vals, cfg["contrasts"]))

    for (pname, mtype), rows in sorted(proba_by_key.items()):
        tag = f"{pname}_{mtype}"
        if args.smoke:
            tag = f"smoke_{tag}"
        ppath = Path(f"{proba_prefix}_{tag}.csv")
        ppath.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(ppath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    duration = time.time() - run_start
    append_run_log(out_dir, "smoke" if args.smoke else "full",
                   f"problems={','.join(p['name'] for p in problems)} "
                   f"models={','.join(model_types)} folds={','.join(map(str, folds))} "
                   f"reps={','.join(reps)}", n_rows, duration)
    protocol = {
        "class_weight": cfg["data"].get("class_weight"),
        "tuning_metric": "mean CV log-loss",
        "grids": {k: v["grid"] for k, v in cfg["models"].items()},
        "svm_calibration": cfg["models"]["svm_rbf"]["calibrate"],
        "threshold": cfg["threshold"],
        "youden_source": cfg["youden"]["source"],
        "inner_cv": {"folds": cfg["cv"]["inner_folds"],
                     "seed": cfg["cv"]["inner_seed"],
                     "inner_inner_seed": cfg["cv"]["inner_inner_seed"]},
        "contrasts": cfg["contrasts"],
        "representations": {k: {kk: vv for kk, vv in v.items() if kk != "n_features"}
                            for k, v in reps.items()},
        "note": ("contrast features are computed from the raw matrix before "
                 "imputation; missing components propagate as NaN and are "
                 "imputed per fold within the training pipeline; identical "
                 "outer splits, inner seeds and tuning grids as the fusion "
                 "experiment"),
    }
    write_provenance(out_dir, cfg, {"argv": sys.argv, "protocol": protocol,
                                    "n_rows": n_rows})
    print(f"done: {n_rows} rows -> {detail_path}")
    print(f"contrast manifest -> {manifest_path}")
    print(f"provenance -> {out_dir / 'provenance.json'}")
    print(f"duration {duration:.1f}s")


if __name__ == "__main__":
    main()
