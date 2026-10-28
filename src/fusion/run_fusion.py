"""Task-specific models + late probability fusion (methodology 1).

For every (problem, model_type, fold): each task model (KD1, KD2, KD3,
READING) is evaluated on its own 33 features. Validation probabilities come
from a model whose hyperparameters were selected by inner CV on the full
training fold (4 folds, seed 42, mean log-loss). Inner out-of-fold
probabilities for every training subject come from per-inner-fold models whose
fit, tuning (inner-inner CV, seed 7) and calibration never used the predicted
subject. Those inner OOF probabilities are used to learn the fusion weights
(log-loss + lambda_reg * KL(w || uniform), SLSQP on the simplex) and to derive
the Youden-J threshold; neither the outer validation subjects nor their labels
are used for weights or thresholds.

Variants written per (problem, model_type, fold):
task_kd1..task_reading (individual models), fusion_equal4, fusion_learned,
fusion_equal3kd (KD1/KD2/KD3 only, READING excluded) and concat132 (the
132-feature concatenated model with the identical tuning protocol).

Usage:
    PYTHONPATH=src .venv/bin/python -m fusion.run_fusion \
        --config src/fusion/config.yaml --smoke
    PYTHONPATH=src .venv/bin/python -m fusion.run_fusion \
        --config src/fusion/config.yaml
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
from scipy.optimize import minimize

from expcommon.data import (TASKS, build_problem, fit_prep, fold_masks,
                            load_split, load_wide, task_columns)
from expcommon.metrics import (BIN_METRICS, MC_METRICS, binary_metrics,
                               multiclass_metrics, youden_threshold)
from expcommon.models import fit_predict_proba, inner_oof, logloss, tune_params
from expcommon.util import append_run_log, sha256_file, write_provenance

ROOT = Path(__file__).resolve().parents[2]

FUSION_VARIANTS = ["task_kd1", "task_kd2", "task_kd3", "task_reading",
                   "fusion_equal4", "fusion_learned", "fusion_equal3kd",
                   "concat132"]
BIN_YOUDEN_COLS = ["thr_y", "f1_y", "bacc_y", "sen_y", "spec_y"]
# "bacc" (balanced accuracy = macro recall) is shared by both problem types;
# it is the only metric appearing in both BIN_METRICS and MC_METRICS, so the
# MC columns below exclude it to keep the CSV schema unambiguous.
MC_DETAIL_COLS = [m for m in MC_METRICS if m != "bacc"]
DETAIL_COLS = (["problem", "model_type", "variant", "fold", "n_train", "n_val",
                "thr"] + BIN_METRICS + BIN_YOUDEN_COLS + MC_DETAIL_COLS
               + ["inner_logloss", "best_params", "fit_s"])
WEIGHTS_COLS = ["problem", "model_type", "fold",
                "w_kd1", "w_kd2", "w_kd3", "w_reading", "lambda_reg",
                "inner_ce_learned", "inner_ce_equal", "inner_ce_uniform_prior",
                "ll_kd1", "ll_kd2", "ll_kd3", "ll_reading", "ll_concat132",
                "n_starts", "status"]


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


def fuse(P_list, w):
    """Weighted average of probability matrices; w on the simplex."""
    return sum(wi * Pi for wi, Pi in zip(w, P_list))


def learn_weights(P_list, y, lam, n_starts, seed):
    """Learn one shared task-weight vector (used for every class) on the
    simplex by minimizing mean log-loss + lam * KL(w || uniform)."""
    T = len(P_list)
    u = np.full(T, 1.0 / T)

    def objective(w):
        ce = logloss(y, fuse(P_list, w))
        kl = float(np.sum(w * np.log(np.clip(w, 1e-12, None) / u)))
        return ce + lam * kl

    rng = np.random.default_rng(seed)
    starts = [u] + list(np.eye(T)) + \
        [np.diff(np.sort(np.r_[0, rng.random(T - 1), 1])) for _ in range(n_starts)]
    best_w, best_val, status = None, np.inf, "no start converged"
    for w0 in starts:
        res = minimize(objective, w0, method="SLSQP",
                       bounds=[(0.0, 1.0)] * T,
                       constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
                       options={"maxiter": 500, "ftol": 1e-10})
        if res.success and res.fun < best_val:
            best_w, best_val, status = res.x, res.fun, res.message
    if best_w is None:
        return u.copy(), float("nan"), status
    best_w = np.clip(best_w, 0.0, 1.0)
    return best_w / best_w.sum(), best_val, status


def variant_row(problem, model_type, variant, fold, n_tr, n_va, y_va, y_tr,
                Pv, Poof, best_params, fit_s, threshold):
    row = {"problem": problem["name"], "model_type": model_type,
           "variant": variant, "fold": fold, "n_train": n_tr, "n_val": n_va,
           "best_params": json.dumps(best_params, sort_keys=True),
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


def evaluate_fold(cfg, X, feats, y_global, groups, pid_arr, fold_of, held_ids,
                  problem, model_type, fold, proba_rows):
    tr, va = fold_masks(fold_of, held_ids, groups, problem, fold, pid_arr)
    y_tr, y_va = y_global[tr], y_global[va]
    n_tr, n_va = len(tr), len(va)
    if problem["positive"] is None:
        assert (y_va == 2).sum() == cfg["split"]["expect"]["ad_val"], \
            (fold, "AD in 3-class validation = fold AD + heldout AD")
        assert (y_tr == 2).sum() == cfg["split"]["expect"]["ad_train"]
    elif problem["name"] == "co_vs_ad":
        assert (y_va == 1).sum() == cfg["split"]["expect"]["ad_val"]

    K = 3 if problem["positive"] is None else 2
    nf_task = cfg["data"]["n_features_per_task"]
    nf_full = len(feats)

    # --- per-task models: inner OOF (for weights/Youden) + tuned val models
    task_oof, task_val, task_params = {}, {}, {}
    task_fit = {}
    for task in TASKS:
        cols = task_columns(feats, task)
        prep = fit_prep(X[:, cols], tr)
        Zt, Zv = prep(X[:, cols][tr]), prep(X[:, cols][va])
        t0 = time.time()
        oof, _ = inner_oof(Zt, y_tr, model_type, nf_task,
                           cfg["cv"]["inner_seed"], cfg["cv"]["inner_inner_seed"],
                           cfg["cv"]["inner_folds"])
        params, score = tune_params(Zt, y_tr, model_type, nf_task,
                                    cfg["cv"]["inner_seed"], cfg["cv"]["inner_folds"])
        Pv = fit_predict_proba(Zt, y_tr, Zv, model_type, params)
        task_oof[task], task_val[task] = oof, Pv
        task_params[task] = {**params, "inner_cv_logloss": round(score, 6)}
        task_fit[task] = time.time() - t0

    # --- concatenated 132-feature reference, identical protocol
    prep_all = fit_prep(X, tr)
    Zt_all, Zv_all = prep_all(X[tr]), prep_all(X[va])
    t0 = time.time()
    oof_all, _ = inner_oof(Zt_all, y_tr, model_type, nf_full,
                           cfg["cv"]["inner_seed"], cfg["cv"]["inner_inner_seed"],
                           cfg["cv"]["inner_folds"])
    params_all, score_all = tune_params(Zt_all, y_tr, model_type, nf_full,
                                        cfg["cv"]["inner_seed"],
                                        cfg["cv"]["inner_folds"])
    Pv_all = fit_predict_proba(Zt_all, y_tr, Zv_all, model_type, params_all)
    fit_concat = time.time() - t0
    concat_params = {**params_all, "inner_cv_logloss": round(score_all, 6)}

    # --- fusion weights from inner OOF (training subjects only)
    t0 = time.time()
    w_learned, w_obj, w_status = learn_weights(
        [task_oof[t] for t in TASKS], y_tr, cfg["fusion"]["lambda_reg"],
        cfg["fusion"]["n_random_starts"], cfg["fusion"]["start_seed"])
    fit_weights = time.time() - t0
    # fusion rows share the upstream task/concat fits; report that wall time
    fit_fusion = fit_weights + sum(task_fit.values()) + fit_concat
    w_equal = np.asarray(cfg["fusion"]["equal_weights"], dtype=float)
    w_3kd = np.asarray(cfg["fusion"]["kd3_weights"], dtype=float)
    weights_def = {"fusion_equal4": w_equal, "fusion_learned": w_learned,
                   "fusion_equal3kd": w_3kd}
    oof_fused = {name: fuse([task_oof[t] for t in TASKS], w)
                 for name, w in weights_def.items()}
    val_fused = {name: fuse([task_val[t] for t in TASKS], w)
                 for name, w in weights_def.items()}

    # --- write rows
    rows, weights_row = [], None
    for task in TASKS:
        rows.append(variant_row(
            problem, model_type, f"task_{task.lower()}", fold, n_tr, n_va,
            y_va, y_tr, task_val[task], task_oof[task], task_params[task],
            task_fit[task], cfg["threshold"]))
    for name in ["fusion_equal4", "fusion_learned", "fusion_equal3kd"]:
        w = weights_def[name]
        params = {"weights": [round(float(x), 6) for x in w]}
        if name == "fusion_learned":
            params["lambda_reg"] = cfg["fusion"]["lambda_reg"]
            params["objective"] = round(float(w_obj), 6)
        rows.append(variant_row(
            problem, model_type, name, fold, n_tr, n_va, y_va, y_tr,
            val_fused[name], oof_fused[name], params,
            fit_fusion, cfg["threshold"]))
    rows.append(variant_row(
        problem, model_type, "concat132", fold, n_tr, n_va, y_va, y_tr,
        Pv_all, oof_all, concat_params, fit_concat, cfg["threshold"]))

    weights_row = {"problem": problem["name"], "model_type": model_type,
                   "fold": fold}
    for i, t in enumerate(TASKS):
        weights_row[f"w_{t.lower()}"] = f"{w_learned[i]:.6f}"
    weights_row["lambda_reg"] = cfg["fusion"]["lambda_reg"]
    weights_row["inner_ce_learned"] = f"{logloss(y_tr, oof_fused['fusion_learned']):.6f}"
    weights_row["inner_ce_equal"] = f"{logloss(y_tr, oof_fused['fusion_equal4']):.6f}"
    weights_row["inner_ce_uniform_prior"] = f"{logloss(y_tr, np.full_like(task_oof['KD1'], 1.0 / K)):.6f}"
    for t in TASKS:
        weights_row[f"ll_{t.lower()}"] = f"{logloss(y_tr, task_oof[t]):.6f}"
    weights_row["ll_concat132"] = f"{logloss(y_tr, oof_all):.6f}"
    weights_row["n_starts"] = cfg["fusion"]["n_random_starts"] + 5
    weights_row["status"] = w_status

    # --- per-subject probability rows
    val_names = {f"task_{t.lower()}": task_val[t] for t in TASKS}
    val_names.update(val_fused)
    val_names["concat132"] = Pv_all
    for i, gi in enumerate(va):
        row = {"subject_id": pid_arr[gi], "group": groups[gi], "fold": fold,
               "role": "heldout_ad" if pid_arr[gi] in held_ids else "cv",
               "y": int(y_va[i])}
        for name in FUSION_VARIANTS:
            P = val_names[name]
            if K == 2:
                row[f"p_{name}"] = f"{P[i, 1]:.8f}"
            else:
                for c in range(3):
                    row[f"p{c}_{name}"] = f"{P[i, c]:.8f}"
        proba_rows.append(row)

    return rows, weights_row


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

    out_dir = ROOT / cfg["output"]["dir"]
    detail_rel = cfg["output"]["smoke_csv"] if args.smoke else cfg["output"]["detail_csv"]
    detail_path = out_dir / detail_rel
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if detail_path.exists():
        sys.exit(f"refusing to overwrite existing results: {detail_path}")
    weights_rel = cfg["output"]["weights_csv"]
    if args.smoke:
        weights_rel = "results/smoke_fusion_weights.csv"
    weights_path = out_dir / weights_rel
    if weights_path.exists():
        sys.exit(f"refusing to overwrite existing results: {weights_path}")
    proba_prefix = out_dir / cfg["output"]["proba_prefix"]

    run_start = time.time()
    n_rows = 0
    weights_rows = []
    proba_by_key = {}
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        w.writeheader()
        for prob in problems:
            y_global = build_problem(groups, prob)
            for mtype in model_types:
                for fold in folds:
                    proba_rows = []
                    rows, weights_row = evaluate_fold(
                        cfg, X, feats, y_global, groups, pid_arr, fold_of,
                        held_ids, prob, mtype, fold, proba_rows)
                    w.writerows(rows)
                    f.flush()
                    weights_rows.append(weights_row)
                    n_rows += len(rows)
                    for r in proba_rows:
                        proba_by_key.setdefault((prob["name"], mtype), []).append(r)
                    head = [r for r in rows if r["variant"] == "fusion_learned"][0]
                    disp = head.get("acc") or head.get("auc") or head.get("ovr_auc")
                    disp2 = head.get("auc") or head.get("ovr_auc") or ""
                    print(f"  {prob['name']} {mtype} fold={fold} acc={disp} "
                          f"auc={disp2} "
                          f"w={weights_row['w_kd1']},{weights_row['w_kd2']},"
                          f"{weights_row['w_kd3']},{weights_row['w_reading']} "
                          f"({weights_row['status']})", flush=True)
                    print(f"    {head['best_params']}", flush=True)

    with open(weights_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=WEIGHTS_COLS)
        w.writeheader()
        w.writerows(weights_rows)

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
                   f"models={','.join(model_types)} folds={','.join(map(str, folds))}",
                   n_rows, duration)
    protocol = {
        "class_weight": cfg["data"].get("class_weight"),
        "tuning_metric": "mean CV log-loss",
        "grids": {k: v["grid"] for k, v in cfg["models"].items()},
        "svm_calibration": cfg["models"]["svm_rbf"]["calibrate"],
        "fusion": {k: v for k, v in cfg["fusion"].items()
                   if k not in ("variants", "equal_weights", "kd3_weights")},
        "threshold": cfg["threshold"],
        "youden_source": cfg["youden"]["source"],
        "inner_cv": {"folds": cfg["cv"]["inner_folds"],
                     "seed": cfg["cv"]["inner_seed"],
                     "inner_inner_seed": cfg["cv"]["inner_inner_seed"]},
        "note": ("task models and concat132 share the identical nested tuning "
                 "protocol; fusion weights are learned on inner OOF predictions "
                 "of the training fold only, never on validation data"),
    }
    write_provenance(out_dir, cfg, {"argv": sys.argv, "protocol": protocol,
                                    "n_rows": n_rows})
    print(f"done: {n_rows} rows -> {detail_path}")
    print(f"weights -> {weights_path}")
    print(f"provenance -> {out_dir / 'provenance.json'}")
    print(f"duration {duration:.1f}s")


if __name__ == "__main__":
    main()
