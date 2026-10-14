"""BEGAD baseline classifiers — runs every (dataset, problem, model, seed,
fold) evaluation defined in config.yaml and writes one row per evaluation to
the detail CSV with Acc / AUC / Sen / Spec / Precision / F1.

Protocol (see config.yaml):
  - samples: one row per participant; each dataset's features x 11 tasks
    flattened to wide columns, feature-major (25x11=275 for the processed
    dataset), median-imputed and standardized per fold on the training
    split only.
  - CV: participant-level, stratified by group, 5 folds, 20 seeds
    (42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337, 2, 3, 5, 10, 21, 777,
    2020, 2021, 2022, 2023) -> 100 train/test evaluations per
    (dataset, problem, model).
  - datasets: processed_dataset (25 features); the feature list of each
    dataset is read from its features.csv columns.
  - problems: co_vs_mci (MCI positive), co_vs_ad (AD positive),
    co_mci_ad (3-class: CO=0, MCI=1, AD=2; Sen/Precision/Spec/F1 are macro
    averages, AUC is macro OVR).

Usage:
    python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml --smoke
    python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml
    python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml \
        --datasets processed_dataset --problems co_vs_mci --models gnb --seeds 1,2 --folds 1

Reproducibility: config + seeds + data-file hashes + environment versions are
written to outputs/baseline/provenance.json.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from begad_norm.config import PROJECT_ROOT

MODELS = {
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "MLPClassifier": MLPClassifier,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "GaussianNB": GaussianNB,
}
RANDOM_STATE_MODELS = {"logistic_regression", "svm_rbf", "svm_linear", "fnn"}
METRIC_COLS = ["acc", "sen", "precision", "spec", "auc", "f1"]
DETAIL_COLS = ["dataset", "problem", "model", "seed", "fold", "n_train",
               "n_test"] + METRIC_COLS
LABEL_MAP = {"CO": 0, "MCI": 1, "AD": 2}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg["config_path"] = str(Path(path).resolve())
    cfg["config_sha256"] = sha256_file(Path(path))
    return cfg


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_features(features_csv):
    """Feature list of a dataset = its features.csv columns minus the index
    columns (participant_id, task)."""
    cols = pd.read_csv(features_csv, nrows=1).columns
    return [c for c in cols if c not in ("participant_id", "task")]


def stratified_folds(pid_group, seed, n_folds):
    """Participant-level stratified split: within each group, permute with
    rng(seed) and assign round-robin to folds 1..n_folds (same method as the
    fold/ folder, which used seed 42).

    pid_group: dict participant_id -> group label ("CO"/"MCI"/"AD").
    Returns dict participant_id -> fold (1..n_folds)."""
    rng = np.random.default_rng(seed)
    assign = {}
    for g in sorted(set(pid_group.values())):
        ids = [p for p in pid_group if pid_group[p] == g]
        for i, pid in enumerate(rng.permutation(np.array(ids))):
            assign[str(pid)] = int(i % n_folds) + 1
    return assign


def build_problem_data(feat, meta, feats, n_tasks, problem):
    """Returns X, y, pid list, pid->group dict.

    One row per participant: len(feats) x n_tasks wide (feature-major,
    columns `{feature}_t{task}`); missing (participant, task) cells stay NaN
    and are median-imputed per fold on the training split.
    """
    groups = problem["include_groups"]
    d = feat.merge(meta[["participant_id", "group", "group_label"]],
                   on="participant_id", how="left")
    d = d[d["group_label"].isin(groups)].copy()
    piv = d.pivot(index="participant_id", columns="task", values=feats)
    piv.columns = [f"{f}_t{t}" for f, t in piv.columns]
    pid = piv.index.to_numpy()
    X = piv.to_numpy(dtype=float)  # NaN where (participant, task) missing
    gmap = dict(zip(d["participant_id"], d["group_label"]))
    y = np.array([LABEL_MAP[gmap[str(p)]] if problem["positive"] is None
                  else int(gmap[str(p)] == problem["positive"])
                  for p in pid])
    pid_group = {str(p): gmap[str(p)] for p in pid}
    return X, y, pid, pid_group


def metrics_binary(y, pred, proba):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "acc": accuracy_score(y, pred),
        "sen": recall_score(y, pred, pos_label=1),
        "precision": precision_score(y, pred, pos_label=1, zero_division=0.0),
        "spec": (tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "auc": roc_auc_score(y, proba[:, 1]),
        "f1": f1_score(y, pred, pos_label=1, zero_division=0.0),
    }


def metrics_multiclass(y, pred, proba):
    """3-class CO/MCI/AD: Sen / Precision / Spec / F1 are macro averages
    over the three classes (Spec = mean per-class TN/(TN+FP)), AUC is the
    one-vs-rest macro AUC, Acc is the overall 3-class accuracy."""
    cm = confusion_matrix(y, pred)
    specs = []
    for c in range(cm.shape[0]):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        specs.append((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    return {
        "acc": accuracy_score(y, pred),
        "sen": recall_score(y, pred, average="macro", zero_division=0.0),
        "precision": precision_score(y, pred, average="macro", zero_division=0.0),
        "spec": float(np.mean(specs)),
        "auc": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "f1": f1_score(y, pred, average="macro", zero_division=0.0),
    }


def make_model(name, spec, seed, class_weight=None, priors=None):
    """Build a model. With class_weight='balanced': LR/SVM get balanced
    class weights; QDA/GNB get per-fold inverse-frequency priors (the
    generative analogue), passed via `priors`. MLPClassifier has no
    class_weight argument, so the FNN is weighted via sample_weight at
    fit time instead (see evaluate)."""
    klass = MODELS[spec["class"]]
    kwargs = dict(spec["kwargs"])
    if name in RANDOM_STATE_MODELS:
        kwargs["random_state"] = seed
    if class_weight == "balanced":
        if name in ("logistic_regression", "svm_rbf", "svm_linear"):
            kwargs["class_weight"] = "balanced"
        elif name in ("qda", "gnb") and priors is not None:
            kwargs["priors"] = priors
    return klass(**kwargs)


def youden_threshold(proba_tr, y_tr):
    """Binary threshold maximizing Youden's J (sen + spec - 1) on the
    training split's predicted probabilities. In-sample (optimistic):
    the threshold is selected on the same data the model was fitted on.
    Sweeps the midpoints between consecutive unique probabilities; ties
    keep the lowest threshold (highest sensitivity)."""
    ps = np.sort(np.unique(proba_tr))
    if len(ps) < 2:
        return 0.5
    best_thr, best_j = 0.5, -np.inf
    for t in (ps[:-1] + ps[1:]) / 2.0:
        pred = (proba_tr >= t).astype(int)
        cm = confusion_matrix(y_tr, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sen + spec - 1.0
        if j > best_j:
            best_j, best_thr = j, float(t)
    return best_thr


def evaluate(cfg, dataset_name, X, y, pid, pid_group, problem, model_name,
             seed, fold, n_folds):
    assign = stratified_folds(pid_group, seed, n_folds)
    test_mask = np.array([assign[str(p)] == fold for p in pid])
    train_mask = ~test_mask
    y_tr = y[train_mask]
    # median imputation + scaling, both fitted on the training split only
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X[train_mask])
    scaler = StandardScaler().fit(X_imp)
    X_tr = scaler.transform(X_imp)
    X_te = scaler.transform(imputer.transform(X[test_mask]))

    # class weighting: QDA/GNB priors and FNN sample weights from the
    # training split only (LR/SVM get class_weight inside make_model)
    priors = None
    sample_weights = None
    if cfg["data"].get("class_weight") == "balanced":
        counts = np.bincount(y_tr, minlength=len(np.unique(y)))
        if (counts > 0).all():
            inv = 1.0 / counts
            priors = inv / inv.sum()
            w = len(y_tr) / (len(counts) * counts)
            sample_weights = w[y_tr]
    model = make_model(model_name, cfg["models"][model_name], seed,
                       cfg["data"].get("class_weight"), priors)
    t0 = time.time()
    if model_name == "fnn" and sample_weights is not None:
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
    else:
        model.fit(X_tr, y_tr)
    fit_s = time.time() - t0
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    if problem["positive"] is not None:
        if cfg["data"].get("threshold") == "youden_train":
            thr = youden_threshold(model.predict_proba(X_tr)[:, 1], y_tr)
            pred = (proba[:, 1] >= thr).astype(int)
        m = metrics_binary(y[test_mask], pred, proba)
    else:
        m = metrics_multiclass(y[test_mask], pred, proba)
    row = {"dataset": dataset_name, "problem": problem["name"],
           "model": model_name, "seed": seed, "fold": fold,
           "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum())}
    row.update(m)
    return row, fit_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--smoke", action="store_true",
                    help="all datasets x 1 problem x 1 seed x 1 fold, all models")
    ap.add_argument("--datasets", help="comma-separated dataset names")
    ap.add_argument("--problems", help="comma-separated problem names")
    ap.add_argument("--models", help="comma-separated model names")
    ap.add_argument("--seeds", help="comma-separated seeds")
    ap.add_argument("--folds", help="comma-separated folds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = cfg["data"]
    all_datasets = data["datasets"]
    datasets = ([d for d in all_datasets if d["name"] in args.datasets.split(",")]
                if args.datasets else all_datasets)
    problems = args.problems.split(",") if args.problems else \
        [p["name"] for p in cfg["problems"]]
    models = args.models.split(",") if args.models else \
        list(cfg["models"].keys())
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else \
        cfg["cv"]["seeds"]
    folds = [int(f) for f in args.folds.split(",")] if args.folds else \
        list(range(1, cfg["cv"]["n_folds"] + 1))
    if args.smoke:
        problems, seeds, folds = [problems[0]], [seeds[0]], [folds[0]]
    n_folds = cfg["cv"]["n_folds"]
    n_tasks = data["n_tasks"]

    out_dir = PROJECT_ROOT / cfg["output"]["dir"]
    detail_rel = cfg["output"]["smoke_csv"] if args.smoke else \
        cfg["output"]["detail_csv"]
    detail_path = out_dir / detail_rel
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if detail_path.exists():
        sys.exit(f"refusing to overwrite existing results: {detail_path}")

    run_start = time.time()
    n_rows = 0
    missingness_rows = []
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        w.writeheader()
        for ds in datasets:
            feat = pd.read_csv(PROJECT_ROOT / ds["features_csv"])
            meta = pd.read_csv(PROJECT_ROOT / ds["metadata_csv"])
            feats = dataset_features(PROJECT_ROOT / ds["features_csv"])
            expected_cols = len(feats) * n_tasks

            # missing-cell counts of the wide matrix over all 157 participants
            full = feat.merge(meta[["participant_id", "group_label"]],
                              on="participant_id", how="left")
            piv = full.pivot(index="participant_id", columns="task",
                             values=feats)
            piv.columns = [f"{f}_t{t}" for f, t in piv.columns]
            X_full = piv.to_numpy(dtype=float)
            n_missing = np.isnan(X_full).sum(axis=0)
            for col, m in zip(piv.columns, n_missing):
                missingness_rows.append({
                    "dataset": ds["name"], "column": col,
                    "n_missing": int(m),
                    "pct_missing": m / X_full.shape[0] * 100.0})
            print(f"dataset {ds['name']}: {len(feats)} features x {n_tasks} "
                  f"tasks = {expected_cols} columns, "
                  f"{X_full.shape[0]} participants, "
                  f"{int(n_missing.sum())} missing cells")

            for pname in problems:
                pcfg = next(p for p in cfg["problems"] if p["name"] == pname)
                X, y, pid, pid_group = build_problem_data(
                    feat, meta, feats, n_tasks, pcfg)
                assert X.shape[1] == expected_cols, (X.shape, expected_cols)
                print(f"problem {pname}: {len(pid)} samples, X {X.shape}, "
                      f"classes {dict(zip(*np.unique(y, return_counts=True)))}")
                for mname in models:
                    for seed in seeds:
                        for fold in folds:
                            row, fit_s = evaluate(
                                cfg, ds["name"], X, y, pid, pid_group, pcfg,
                                mname, seed, fold, n_folds)
                            w.writerow(row)
                            f.flush()
                            n_rows += 1
                            print(f"  {ds['name']} {pname} {mname} "
                                  f"seed={seed} fold={fold} "
                                  f"acc={row['acc']:.4f} auc={row['auc']:.4f} "
                                  f"({fit_s:.1f}s)", flush=True)

    # missingness
    miss_path = out_dir / cfg["output"]["missingness_csv"]
    miss_path.parent.mkdir(parents=True, exist_ok=True)
    with open(miss_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "column", "n_missing",
                                          "pct_missing"])
        w.writeheader()
        w.writerows(missingness_rows)

    # run log
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
                    "smoke" if args.smoke else "full",
                    ",".join(d["name"] for d in datasets),
                    ",".join(problems), ",".join(models),
                    ",".join(map(str, seeds)), ",".join(map(str, folds)),
                    n_rows, f"{time.time() - run_start:.1f}", "ok"])

    # provenance
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "command": " ".join(sys.argv),
        "git_commit": subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or "unknown",
        "git_dirty": bool(subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
            capture_output=True, text=True).stdout.strip()),
        "config_sha256": cfg["config_sha256"],
        "data_sha256": {
            f"{d['name']}_features": sha256_file(PROJECT_ROOT / d["features_csv"])
            for d in datasets
        } | {
            f"{d['name']}_metadata": sha256_file(PROJECT_ROOT / d["metadata_csv"])
            for d in datasets
        },
        "packages": {p: importlib.metadata.version(p)
                     for p in ["numpy", "pandas", "scikit-learn", "scipy",
                               "PyYAML"]},
        "n_rows": n_rows,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"done: {n_rows} rows -> {detail_path}")
    print(f"missingness -> {miss_path}")
    print(f"provenance -> {out_dir / 'provenance.json'}")


if __name__ == "__main__":
    main()
