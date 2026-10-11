"""BEGAD baseline classifiers — runs every (problem, model, seed, fold)
evaluation defined in config.yaml and writes one row per evaluation to the
detail CSV with Acc / Sen / Spec / AUC / F1.

Protocol (see config.yaml):
  - samples: one row per participant; 25 selected features x 11 tasks
    flattened to 275 columns (feature-major), median-imputed and standardized
    per fold on the training split only.
  - CV: participant-level, stratified by group, 4 folds, 10 seeds (1-10)
    -> 40 train/test evaluations per (problem, model).
  - problems: co_vs_mci (MCI positive), co_vs_ad (AD positive),
    co_mci_ad (3-class: CO=0, MCI=1, AD=2).

Usage:
    python run_baselines.py --config config.yaml --smoke
    python run_baselines.py --config config.yaml
    python run_baselines.py --config config.yaml --problems co_vs_mci \
        --models gnb --seeds 1,2 --folds 1

Reproducibility: config + seed + data-file hashes + environment versions are
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = {
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "MLPClassifier": MLPClassifier,
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis,
    "GaussianNB": GaussianNB,
}
RANDOM_STATE_MODELS = {"logistic_regression", "svm_rbf", "svm_linear", "fnn"}
METRIC_COLS = ["acc", "sen", "spec", "auc", "f1"]
DETAIL_COLS = ["problem", "model", "seed", "fold", "n_train", "n_test"] + METRIC_COLS
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


def build_problem_data(feat, meta, cfg, problem):
    """Returns X, y, pid list, pid->group dict.

    Two sample modes (config data.sample_mode):
      - pool_task_rows (default): one row per (participant, task); 25 features
        + optional 11 task one-hot indicators.
      - participant_wide: one row per participant; 25 features x 11 tasks
        flattened to 275 columns (feature-major, names `{feature}_t{task}`),
        missing (participant, task) cells stay NaN and are median-imputed
        per fold on the training split.
    """
    groups = problem["include_groups"]
    d = feat.merge(meta[["participant_id", "group", "group_label"]],
                   on="participant_id", how="left")
    d = d[d["group_label"].isin(groups)].copy()
    feats = cfg["data"]["features"]

    if cfg["data"].get("sample_mode", "pool_task_rows") == "participant_wide":
        n_tasks = cfg["data"].get("n_tasks", 11)
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

    d = d.dropna(subset=feats)
    X_feat = d[feats].to_numpy(dtype=float)
    n_tasks = cfg["data"].get("n_tasks", 11)
    task_cols = np.zeros((len(d), n_tasks))
    task_cols[np.arange(len(d)), d["task"].to_numpy(dtype=int)] = 1.0
    X = np.hstack([X_feat, task_cols]) if cfg["data"]["add_task_indicator"] \
        else X_feat
    if problem["positive"] is not None:
        y = (d["group_label"] == problem["positive"]).to_numpy(dtype=int)
    else:
        y = d["group_label"].map(LABEL_MAP).to_numpy(dtype=int)
    pid = d["participant_id"].to_numpy()
    pid_group = dict(zip(d["participant_id"], d["group_label"]))
    return X, y, pid, pid_group


def metrics_binary(y, pred, proba):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "acc": accuracy_score(y, pred),
        "sen": recall_score(y, pred, pos_label=1),
        "spec": (tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "auc": roc_auc_score(y, proba[:, 1]),
        "f1": f1_score(y, pred, pos_label=1),
    }


def metrics_multiclass(y, pred, proba):
    cm = confusion_matrix(y, pred)
    specs = []
    for c in range(cm.shape[0]):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        specs.append((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    return {
        "acc": accuracy_score(y, pred),
        "sen": recall_score(y, pred, average="macro"),
        "spec": float(np.mean(specs)),
        "auc": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
        "f1": f1_score(y, pred, average="macro"),
    }


def make_model(name, spec, seed):
    klass = MODELS[spec["class"]]
    kwargs = dict(spec["kwargs"])
    if name in RANDOM_STATE_MODELS:
        kwargs["random_state"] = seed
    return klass(**kwargs)


def evaluate(cfg, X, y, pid, pid_group, problem, model_name, seed, fold,
             n_folds):
    assign = stratified_folds(pid_group, seed, n_folds)
    test_mask = np.array([assign[str(p)] == fold for p in pid])
    train_mask = ~test_mask
    wide = cfg["data"].get("sample_mode", "pool_task_rows") == "participant_wide"
    if wide:
        # median imputation + scaling, both fitted on the training split only
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X[train_mask])
        scaler = StandardScaler().fit(X_imp)
        X_tr = scaler.transform(X_imp)
        X_te = scaler.transform(imputer.transform(X[test_mask]))
    else:
        n_feat = len(cfg["data"]["features"])
        scaler = StandardScaler()
        X_tr = np.hstack([scaler.fit_transform(X[train_mask][:, :n_feat]),
                          X[train_mask][:, n_feat:]])
        X_te = np.hstack([scaler.transform(X[test_mask][:, :n_feat]),
                          X[test_mask][:, n_feat:]])
    model = make_model(model_name, cfg["models"][model_name], seed)
    t0 = time.time()
    model.fit(X_tr, y[train_mask])
    fit_s = time.time() - t0
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    if problem["positive"] is not None:
        m = metrics_binary(y[test_mask], pred, proba)
    else:
        m = metrics_multiclass(y[test_mask], pred, proba)
    row = {"problem": problem["name"], "model": model_name, "seed": seed,
           "fold": fold, "n_train": int(train_mask.sum()),
           "n_test": int(test_mask.sum())}
    row.update(m)
    return row, fit_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--smoke", action="store_true",
                    help="1 problem x 1 seed x 1 fold, all models")
    ap.add_argument("--problems", help="comma-separated problem names")
    ap.add_argument("--models", help="comma-separated model names")
    ap.add_argument("--seeds", help="comma-separated seeds")
    ap.add_argument("--folds", help="comma-separated folds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[2]
    data = cfg["data"]
    feat = pd.read_csv(root / data["features_csv"])
    meta = pd.read_csv(root / data["metadata_csv"])

    # sanity: the configured feature list must equal the v2 dataset columns
    v2_cols = [c for c in pd.read_csv(
        root / "processed_dataset_v2/features.csv", nrows=1).columns
        if c not in ("participant_id", "task")]
    assert set(v2_cols) == set(data["features"]), \
        f"config feature list mismatch: {len(v2_cols)} vs {len(data['features'])}"

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

    out_dir = root / cfg["output"]["dir"]
    detail_rel = cfg["output"]["smoke_csv"] if args.smoke else \
        cfg["output"]["detail_csv"]
    detail_path = out_dir / detail_rel
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if detail_path.exists():
        sys.exit(f"refusing to overwrite existing results: {detail_path}")

    run_start = time.time()
    n_rows = 0
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        w.writeheader()
        for pname in problems:
            pcfg = next(p for p in cfg["problems"] if p["name"] == pname)
            X, y, pid, pid_group = build_problem_data(feat, meta, cfg, pcfg)
            expected_cols = len(cfg["data"]["features"]) * \
                cfg["data"].get("n_tasks", 11) \
                if cfg["data"].get("sample_mode") == "participant_wide" \
                else None
            if expected_cols:
                assert X.shape[1] == expected_cols, (X.shape, expected_cols)
            print(f"problem {pname}: {len(pid)} samples, X {X.shape}, "
                  f"classes {dict(zip(*np.unique(y, return_counts=True)))}")
            for mname in models:
                for seed in seeds:
                    for fold in folds:
                        row, fit_s = evaluate(cfg, X, y, pid, pid_group, pcfg,
                                              mname, seed, fold, n_folds)
                        w.writerow(row)
                        f.flush()
                        n_rows += 1
                        print(f"  {pname} {mname} seed={seed} fold={fold} "
                              f"acc={row['acc']:.4f} auc={row['auc']:.4f} "
                              f"({fit_s:.1f}s)", flush=True)

    # run log
    log_path = out_dir / cfg["output"]["run_log_csv"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["timestamp_utc", "mode", "problems", "models",
                        "seeds", "folds", "n_rows", "duration_s", "status"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "smoke" if args.smoke else "full",
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
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or "unknown",
        "git_dirty": bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True).stdout.strip()),
        "config_sha256": cfg["config_sha256"],
        "data_sha256": {
            "features_csv": sha256_file(root / data["features_csv"]),
            "metadata_csv": sha256_file(root / data["metadata_csv"]),
            "v2_features_csv": sha256_file(
                root / "processed_dataset_v2/features.csv"),
        },
        "packages": {p: importlib.metadata.version(p)
                     for p in ["numpy", "pandas", "scikit-learn", "scipy",
                               "PyYAML"]},
        "n_rows": n_rows,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"done: {n_rows} rows -> {detail_path}")
    print(f"provenance -> {out_dir / 'provenance.json'}")


if __name__ == "__main__":
    main()
