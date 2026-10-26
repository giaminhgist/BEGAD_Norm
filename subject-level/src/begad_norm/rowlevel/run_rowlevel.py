"""Row-level classification on T1235_clean.xlsx / T5_clean.xlsx.

Protocol (see config_t1235.yaml / config_t5.yaml):
  - data: one row per trial-level sample (pat x task x ROI); cleaned files
    (pupil <= 0 rows excluded, zero-fix totalFixTime imputed); pupil NaN
    cells remain and are median-imputed per fold on the training split.
  - split seed fixed at 42, 10 folds:
      grouped — participant-level stratified round-robin (baseline's
        stratified_folds): no participant in both train and test.
        Evaluated at row level AND subject level (mean predicted
        probability per participant, 0.5 / argmax decision rule).
      random  — row-level StratifiedKFold(10, shuffle, random_state=42),
        stratify by y; row-level metrics only. Rows of one participant can
        appear in both train and test (leakage), which is exactly what
        this arm quantifies.
  - 5 model seeds [42, 2026, 1, 0, 7] feed the random state of RF / LGBM /
    FNN and the torch initialization; LR and SVM-RBF are deterministic so
    their repeats are identical (std = 0 in the reports).
  - problems: co_vs_mci (MCI positive), co_vs_ad (AD positive),
    co_mci_ad (3-class; Sen/Precision/Spec/F1 macro, AUC macro OVR).
  - models: logistic_regression, svm_rbf, random_forest, lightgbm, fnn
    (sklearn, balanced class weights on row counts) + torch_multimodal
    (PyTorch port of Multimodal.ipynb, no class weights).
  - metrics: acc, sen, precision, spec, auc, f1 (baseline semantics);
    fixed 0.5 threshold for binary problems.
  - fold info: folds/fold_info.csv (role=test rows only; train is the
    complement).

Detail CSV schema: dataset, problem, model, seed, fold, n_train, n_test,
split, eval_level + acc/sen/precision/spec/auc/f1. n_train/n_test are row
counts for eval_level="row" and participant counts for "subject".

Usage:
    python -m begad_norm.rowlevel.run_rowlevel \
        --config src/begad_norm/rowlevel/config_t1235.yaml --smoke
    python -m begad_norm.rowlevel.run_rowlevel \
        --config src/begad_norm/rowlevel/config_t1235.yaml
    ... --problems co_vs_ad --models lightgbm --seeds 42 --folds 1,2

Note: the venv's begad_norm.pth points to a stale path; run with
PYTHONPATH=<repo>/subject-level/src (recorded in provenance.json).
"""
import argparse
import csv
import fcntl
import importlib.metadata
import json
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from begad_norm.baseline.run_baselines import (LABEL_MAP, METRIC_COLS,
                                               load_config, make_model,
                                               metrics_binary,
                                               metrics_multiclass,
                                               sha256_file,
                                               stratified_folds)
from begad_norm.config import PROJECT_ROOT
from begad_norm.rowlevel.torch_model import (carve_val_mask, fit_torch_model,
                                             torch_branch_inputs)

INDEX_COLS = ["group", "pat", "task", "ROI"]
SPLITS = ("grouped", "random")
EVAL_LEVELS = {"grouped": ["row", "subject"], "random": ["row"]}
TORCH_MODEL_NAME = "torch_multimodal"
DETAIL_COLS = ["dataset", "problem", "model", "seed", "fold", "n_train",
               "n_test", "split", "eval_level"] + METRIC_COLS
FIT_TIMES_COLS = ["dataset", "problem", "model", "seed", "fold", "split",
                  "fit_s"]
REPO_ROOT = PROJECT_ROOT.parent
CODE_FILES = {"run_rowlevel.py": Path(__file__),
              "torch_model.py": Path(__file__).with_name("torch_model.py"),
              "make_report.py": Path(__file__).with_name("make_report.py"),
              "aggregate_results.py": Path(__file__).parents[1] / "baseline"
              / "aggregate_results.py"}


def load_dataset(xlsx_path, data_cfg):
    """Load a cleaned xlsx with structure guards; returns df with
    group_label and a positional 0-based index."""
    df = pd.read_excel(xlsx_path, engine="openpyxl").reset_index(drop=True)
    if list(df.columns) != data_cfg["columns"]:
        raise AssertionError(f"unexpected columns: {list(df.columns)}")
    if not df["group"].isin({0, 1, 2}).all():
        raise AssertionError("group column contains values outside {0,1,2}")
    if df.duplicated(subset=["pat", "task", "ROI"]).sum() != 0:
        raise AssertionError("duplicate (pat, task, ROI) rows present")
    if df["pat"].nunique() != data_cfg["n_patients"]:
        raise AssertionError(f"expected {data_cfg['n_patients']} "
                             f"participants, got {df['pat'].nunique()}")
    df["group_label"] = df["group"].map({0: "CO", 1: "MCI", 2: "AD"})
    return df


def build_problem_rows(df, feats, problem):
    """Row-level X, y, pat list, pat->group dict, and positional row
    indices for one problem (include_groups/positive semantics as in the
    baseline)."""
    groups = problem["include_groups"]
    d = df[df["group_label"].isin(groups)]
    X = d[feats].to_numpy(dtype=float)
    pat = d["pat"].to_numpy()
    gmap = dict(zip(d["pat"], d["group_label"]))
    y = np.array([LABEL_MAP[gmap[p]] if problem["positive"] is None
                  else int(gmap[p] == problem["positive"])
                  for p in pat])
    pid_group = {p: gmap[p] for p in np.unique(pat)}
    idx = d.index.to_numpy()
    return X, y, pat, pid_group, idx


def grouped_fold_masks(pat, pid_group, seed, fold, n_folds):
    """Participant-level stratified split: no participant in both train
    and test (baseline's stratified_folds)."""
    assign = stratified_folds(pid_group, seed, n_folds)
    in_fold = np.array([assign[str(p)] == fold for p in pat])
    return ~in_fold, in_fold


def random_fold_masks(X, y, seed, fold, n_folds):
    """Row-level StratifiedKFold(shuffle, random_state=seed)."""
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tr_idx, te_idx = list(kf.split(X, y))[fold - 1]
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[tr_idx] = True
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[te_idx] = True
    return train_mask, test_mask


def aggregate_subject(pat_te, y_te, proba_te):
    """Per-participant aggregation of test-row predictions: mean predicted
    probability per pat; label = 0.5 threshold (binary) / argmax (3-class).
    Returns (pats, y_sub, proba_sub, pred_sub); pats sorted ascending."""
    sub = pd.DataFrame(proba_te, index=pat_te).groupby(level=0).mean()
    pats = sub.index.to_numpy()
    y_sub = np.array([y_te[np.flatnonzero(pat_te == p)[0]] for p in pats])
    proba_sub = sub.to_numpy(dtype=float)
    if proba_sub.shape[1] == 2:
        pred_sub = (proba_sub[:, 1] >= 0.5).astype(int)
    else:
        pred_sub = np.argmax(proba_sub, axis=1)
    return pats, y_sub, proba_sub, pred_sub


def write_fold_info(out_dir, df, problems, cfg):
    """Write folds/fold_info.csv (split, seed, fold, problem, unit,
    unit_id, group, role; role=test rows only, train is the complement)
    and return the masks used by _fit_predict, keyed
    (split, problem, fold) -> (train_mask, test_mask) over the problem's
    rows. The grouped participant assignment is computed once on the full
    cohort (problem="all" rows in the CSV); each problem's grouped masks
    are that assignment restricted to the problem's participants."""
    cv = cfg["cv"]
    split_seed = cv["split_seed"]
    n_folds = cv["n_folds"]
    feats = cfg["data"]["feature_columns"]
    rows = []
    masks = {}
    assign = stratified_folds(
        dict(zip(df["pat"], df["group_label"])), split_seed, n_folds)
    pat_group = dict(zip(df["pat"], df["group"]))
    for fold in range(1, n_folds + 1):
        for p in sorted(assign, key=lambda x: (assign[x], x)):
            if assign[p] == fold:
                rows.append({"split": "grouped", "seed": split_seed,
                             "fold": fold, "problem": "all",
                             "unit": "participant", "unit_id": p,
                             "group": int(pat_group[p]), "role": "test"})
    for pcfg in problems:
        X, y, pat, pid_group, idx = build_problem_rows(df, feats, pcfg)
        for fold in range(1, n_folds + 1):
            masks[("grouped", pcfg["name"], fold)] = grouped_fold_masks(
                pat, pid_group, split_seed, fold, n_folds)
            masks[("random", pcfg["name"], fold)] = random_fold_masks(
                X, y, split_seed, fold, n_folds)
            te = masks[("random", pcfg["name"], fold)][1]
            for i in np.flatnonzero(te):
                rows.append({"split": "random", "seed": split_seed,
                             "fold": fold, "problem": pcfg["name"],
                             "unit": "row", "unit_id": int(idx[i]),
                             "group": int(df["group"].iloc[idx[i]]),
                             "role": "test"})
    out = pd.DataFrame(rows)
    folds_dir = out_dir / cfg["output"]["folds_dir"]
    folds_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(folds_dir / "fold_info.csv", index=False)
    return masks


def _append_rows(path, fieldnames, rows):
    """Append rows to a CSV under an exclusive lock (safe for concurrent
    workers); the header is written once by whoever creates the file."""
    lock = str(path) + ".lock"
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new:
                w.writeheader()
            w.writerows(rows)


def _fit_predict(cfg, X, y, pat, pid_group, problem, model_name, seed, fold,
                 split, train_mask, test_mask, X_torch, torch_cols,
                 detail_path=None, ft_path=None):
    """One (problem, model, seed, fold, split) evaluation: detail rows
    (1 for random / row-level, 2 for grouped: row + subject) + fit time.
    Imputation and scaling are fitted on the training split only; class
    weights come from training row counts. The torch arm scales its own
    branch columns (+ ROI) and uses no class weights. When detail_path /
    ft_path are given, the rows and the fit time are appended immediately
    after the fit finishes (incremental reporting under a file lock)."""
    import threadpoolctl
    with threadpoolctl.threadpool_limits(1):
        imputer = SimpleImputer(strategy="median")
        X_imp = np.nan_to_num(imputer.fit_transform(X[train_mask]), nan=0.0)
        scaler = StandardScaler().fit(X_imp)
        X_tr = scaler.transform(X_imp)
        X_te = scaler.transform(np.nan_to_num(
            imputer.transform(X[test_mask]), nan=0.0))
        if model_name == TORCH_MODEL_NAME:
            imp_t = SimpleImputer(strategy="median")
            Xt = np.nan_to_num(imp_t.fit_transform(X_torch[train_mask]),
                               nan=0.0)
            sc_t = StandardScaler().fit(Xt)
            Xt_full = sc_t.transform(np.nan_to_num(
                imp_t.transform(X_torch), nan=0.0))
            Xv, Xs, Xp = torch_branch_inputs(
                Xt_full, torch_cols, cfg["data"]["branch_features"],
                cfg["data"]["roi_column"])
            val_mask = carve_val_mask(
                train_mask, pat, pid_group, y, split, seed,
                cfg["cv"]["torch"]["val_fraction"])
            out = fit_torch_model(Xv, Xs, Xp, y, train_mask, val_mask,
                                  test_mask, seed, cfg["cv"]["torch"])
            proba = out["proba_test"]
            pred = ((proba[:, 1] >= 0.5).astype(int)
                    if problem["positive"] is not None
                    else np.argmax(proba, axis=1))
            fit_s = out["fit_s"]
        else:
            priors = None
            sample_weights = None
            if cfg["data"].get("class_weight") == "balanced":
                counts = np.bincount(y[train_mask],
                                     minlength=len(np.unique(y)))
                if (counts > 0).all():
                    inv = 1.0 / counts
                    priors = inv / inv.sum()
                    w = len(y[train_mask]) / (len(counts) * counts)
                    sample_weights = w[y[train_mask]]
            model = make_model(model_name, cfg["models"][model_name], seed,
                               cfg["data"].get("class_weight"), priors)
            t0 = time.time()
            if model_name == "fnn" and sample_weights is not None:
                model.fit(X_tr, y[train_mask], sample_weight=sample_weights)
            else:
                model.fit(X_tr, y[train_mask])
            fit_s = time.time() - t0
            pred = model.predict(X_te)
            proba = model.predict_proba(X_te)

    y_te = y[test_mask]
    base = {"dataset": cfg["data"]["dataset_name"],
            "problem": problem["name"], "model": model_name,
            "seed": seed, "fold": fold,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "split": split}
    if problem["positive"] is not None:
        m = metrics_binary(y_te, pred, proba)
    else:
        m = metrics_multiclass(y_te, pred, proba)
    rows = [dict(base, eval_level="row", **m)]
    if split == "grouped":
        pats, y_sub, proba_sub, pred_sub = aggregate_subject(
            pat[test_mask], y_te, proba)
        if problem["positive"] is not None:
            m_sub = metrics_binary(y_sub, pred_sub, proba_sub)
        else:
            m_sub = metrics_multiclass(y_sub, pred_sub, proba_sub)
        rows.append(dict(base, eval_level="subject", **m_sub,
                         n_train=len(np.unique(pat[train_mask])),
                         n_test=len(pats)))
    if detail_path is not None:
        _append_rows(detail_path, DETAIL_COLS, rows)
    if ft_path is not None:
        _append_rows(ft_path, FIT_TIMES_COLS,
                     [{"dataset": base["dataset"],
                       "problem": base["problem"], "model": model_name,
                       "seed": seed, "fold": fold, "split": split,
                       "fit_s": fit_s}])
    return rows, fit_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(Path(__file__).with_name("config_t1235.yaml")))
    ap.add_argument("--smoke", action="store_true",
                    help="1 problem x all models x 1 seed x 1 fold x both splits")
    ap.add_argument("--problems", help="comma-separated problem names")
    ap.add_argument("--models", help="comma-separated model names")
    ap.add_argument("--seeds", help="comma-separated model seeds")
    ap.add_argument("--folds", help="comma-separated folds")
    ap.add_argument("--splits", help="comma-separated split styles")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    dataset_name = data["dataset_name"]
    out_dir = REPO_ROOT / cfg["output"]["dir"]
    detail_rel = cfg["output"]["smoke_csv"] if args.smoke else \
        cfg["output"]["detail_csv"]
    detail_path = out_dir / detail_rel
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if detail_path.exists():
        sys.exit(f"refusing to overwrite existing results: {detail_path}")
    # incremental reporting: workers append their rows as soon as each fit
    # finishes (file-locked); start with empty files
    ft_path = out_dir / cfg["output"]["fit_times_csv"]
    open(detail_path, "w").close()
    open(ft_path, "w").close()

    problems = args.problems.split(",") if args.problems else \
        [p["name"] for p in cfg["problems"]]
    models = list(cfg["models"].keys()) + [TORCH_MODEL_NAME]
    if args.models:
        wanted = set(args.models.split(","))
        models = [m for m in models if m in wanted]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else \
        cfg["cv"]["model_seeds"]
    folds = [int(f) for f in args.folds.split(",")] if args.folds else \
        list(range(1, cfg["cv"]["n_folds"] + 1))
    splits = tuple(args.splits.split(",")) if args.splits else SPLITS
    if args.smoke:
        problems, seeds, folds = [problems[0]], [seeds[0]], [folds[0]]
    n_jobs = cfg["execution"]["n_jobs"]

    # code copies into <out_dir>/code/
    code_dir = out_dir / cfg["output"]["code_dir"]
    code_dir.mkdir(parents=True, exist_ok=True)
    for name, src in CODE_FILES.items():
        shutil.copy(src, code_dir / name)
    shutil.copy(args.config, code_dir / Path(args.config).name)

    run_start = time.time()
    df = load_dataset(REPO_ROOT / data["xlsx_path"], data)
    print(f"dataset {dataset_name}: {len(df)} rows, "
          f"{df['pat'].nunique()} participants")

    # missingness over all rows
    miss_path = out_dir / cfg["output"]["missingness_csv"]
    miss_path.parent.mkdir(parents=True, exist_ok=True)
    feats = data["feature_columns"]
    n_missing = df[feats].isna().sum()
    with open(miss_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "column", "n_missing",
                                          "pct_missing"])
        w.writeheader()
        for col in feats:
            w.writerow({"dataset": dataset_name, "column": col,
                        "n_missing": int(n_missing[col]),
                        "pct_missing": n_missing[col] / len(df) * 100.0})

    branch_feats = cfg["data"]["branch_features"]
    torch_cols = (branch_feats["visual"] + branch_feats["speech"]
                  + branch_feats["pupil"] + [data["roi_column"]])
    problems_data = {}
    for pname in problems:
        pcfg = next(p for p in cfg["problems"] if p["name"] == pname)
        X, y, pat, pid_group, idx = build_problem_rows(df, feats, pcfg)
        X_torch = df.loc[idx, torch_cols].to_numpy(dtype=float)
        problems_data[pname] = (pcfg, X, y, pat, pid_group, X_torch)
        print(f"problem {pname}: {len(pat)} rows, "
              f"classes {dict(zip(*np.unique(y, return_counts=True)))}")

    masks = write_fold_info(out_dir, df, cfg["problems"], cfg)
    print(f"fold info -> {out_dir / cfg['output']['folds_dir'] / 'fold_info.csv'}")

    tasks = []
    for pname in problems:
        pcfg, X, y, pat, pid_group, X_torch = problems_data[pname]
        for mname in models:
            for seed in seeds:
                for fold in folds:
                    for split in splits:
                        tr, te = masks[(split, pname, fold)]
                        tasks.append((pcfg, X, y, pat, pid_group, mname,
                                      seed, fold, split, tr, te, X_torch))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_fit_predict)(cfg, X, y, pat, pid_group, pcfg, mname,
                              seed, fold, split, tr, te, X_torch, torch_cols,
                              detail_path, ft_path)
        for (pcfg, X, y, pat, pid_group, mname, seed, fold, split, tr, te,
             X_torch) in tasks
    )

    rows = []
    fit_times = []
    for (_, _, _, _, _, mname, seed, fold, split, _, _, _), (r, fit_s) in \
            zip(tasks, results):
        rows.extend(r)
        fit_times.append({"dataset": dataset_name,
                          "problem": r[0]["problem"], "model": mname,
                          "seed": seed, "fold": fold, "split": split,
                          "fit_s": fit_s})

    expected = (len(problems) * len(models) * len(seeds) * len(folds)
                * sum(len(EVAL_LEVELS[s]) for s in splits))
    if len(rows) != expected:
        sys.exit(f"unexpected detail row count {len(rows)} != {expected}")

    ft = pd.DataFrame(fit_times)
    print("fit times (s) per model:")
    print(ft.groupby("model")["fit_s"].agg(["mean", "median", "max"])
          .round(1).to_string())

    # run log (append-mode, same schema as the baseline)
    log_path = out_dir / cfg["output"]["run_log_csv"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["timestamp_utc", "mode", "datasets", "problems",
                        "models", "seeds", "folds", "n_rows", "duration_s",
                        "status"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "smoke" if args.smoke else "full", dataset_name,
                    ",".join(problems), ",".join(models),
                    ",".join(map(str, seeds)), ",".join(map(str, folds)),
                    len(rows), f"{time.time() - run_start:.1f}", "ok"])

    # provenance
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "command": " ".join(sys.argv),
        "pythonpath": __import__("os").environ.get("PYTHONPATH", ""),
        "git_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or "unknown",
        "git_dirty": bool(subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True).stdout.strip()),
        "config_sha256": cfg["config_sha256"],
        "data_sha256": {f"{dataset_name}_clean_xlsx":
                        sha256_file(REPO_ROOT / data["xlsx_path"])},
        "packages": {p: importlib.metadata.version(p)
                     for p in ["numpy", "pandas", "scikit-learn", "scipy",
                               "PyYAML", "joblib", "openpyxl",
                               "threadpoolctl", "torch", "lightgbm"]},
        "n_rows": len(rows),
        "n_jobs": n_jobs,
        "split_seed": cfg["cv"]["split_seed"],
        "model_seeds": seeds,
        "splits": list(splits),
        "eval_levels": EVAL_LEVELS,
        "torch_spec": cfg["cv"]["torch"],
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"done: {len(rows)} rows -> {detail_path}")
    print(f"fit times -> {ft_path}")
    print(f"missingness -> {miss_path}")
    print(f"provenance -> {out_dir / 'provenance.json'}")


if __name__ == "__main__":
    main()
