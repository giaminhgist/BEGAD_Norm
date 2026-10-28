"""BEGAD baseline classifiers on the processed 37-feature dataset (RAW157).

Runs every (problem, model, seed, fold) evaluation defined in config.yaml and
writes one row per evaluation to the detail CSV.

Split (fixed, from processed_dataset/splits/): the 4 held-out AD subjects are
appended to EVERY validation fold and never enter training; the remaining 153
subjects are 4 stratified folds (seed 42). Seeds are model random_state only:
LR/SVM/GNB are deterministic, so their 10 seed-rows repeat identically.

Class imbalance: balanced class weights (LR/SVM/RF/LGBM), inverse-frequency
priors (GNB) and sample weights (MLP), all fitted on the training split only.
Features are standardized per fold on the training split (no imputation: the
dataset is complete-case by construction).

Usage:
    PYTHONPATH=src .venv/bin/python -m baseline.run_baselines \
        --config src/baseline/config.yaml --smoke
    PYTHONPATH=src .venv/bin/python -m baseline.run_baselines \
        --config src/baseline/config.yaml
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
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "RandomForestClassifier": RandomForestClassifier,
    "LGBMClassifier": LGBMClassifier,
    "GaussianNB": GaussianNB,
    "MLPClassifier": MLPClassifier,
}
RANDOM_STATE_MODELS = {"random_forest", "lightgbm", "mlp"}
BALANCED_WEIGHT_MODELS = {"logistic_regression", "svm_rbf", "random_forest", "lightgbm"}
LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
CLASS_ORDER = ["CN", "MCI", "AD"]
METRIC_COLS = ["acc", "sen", "precision", "spec", "auc", "f1"]
PER_CLASS_COLS = [f"{m}_{c.lower()}" for m in ("recall", "precision", "f1")
                  for c in CLASS_ORDER]
DETAIL_COLS = (["problem", "model", "seed", "fold", "n_train", "n_val"]
               + METRIC_COLS + PER_CLASS_COLS)


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


def load_data(cfg):
    df = pd.read_csv(ROOT / cfg["data"]["wide_csv"])
    id_col, label_col = cfg["data"]["id_col"], cfg["data"]["label_col"]
    feats = [c for c in df.columns if c not in (id_col, label_col)]
    assert len(feats) == 96, (len(feats), "expected 24 features x 4 tasks")
    assert df[feats].isna().to_numpy().sum() == 0, "dataset must be complete-case"
    return (df[feats].to_numpy(dtype=float),
            df[id_col].astype(str).to_numpy(),
            df[label_col].astype(str).to_numpy(),
            feats)


def load_split(cfg):
    assign = pd.read_csv(ROOT / cfg["split"]["assignments_csv"])
    held = pd.read_csv(ROOT / cfg["split"]["heldout_csv"])
    fold_of = dict(zip(assign["subject_id"].astype(str), assign["fold"].astype(int)))
    held_ids = sorted(held["subject_id"].astype(str))
    assert len(fold_of) == 153 and len(held_ids) == 4
    assert not (set(held_ids) & set(fold_of)), "heldout AD must not be in CV assignments"
    return fold_of, held_ids


def build_problem(groups, problem):
    """Global label array: label for included subjects, -1 elsewhere."""
    inc = set(problem["include_groups"])
    y = np.full(len(groups), -1, dtype=int)
    for i, g in enumerate(groups):
        if g in inc:
            y[i] = LABEL_MAP[g] if problem["positive"] is None else int(g == problem["positive"])
    return y


def global_split_masks(fold_of, held_ids, fold, n):
    """Global train/val index arrays for one fold (before problem filtering).

    Train = the 153 subjects whose assigned fold differs (held-out AD are not
    in fold_of, so they are automatically excluded from training). Val = the
    fold's subjects + the 4 held-out AD subjects."""
    val_ids = {p for p, f in fold_of.items() if f == fold} | set(held_ids)
    val = np.array([i for i, p in enumerate(_PID_GLOBAL) if p in val_ids], dtype=int)
    train = np.array([i for i, p in enumerate(_PID_GLOBAL)
                      if p in fold_of and fold_of[p] != fold], dtype=int)
    return train, val


def metrics_binary(y_true, pred, proba):
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "acc": accuracy_score(y_true, pred),
        "sen": recall_score(y_true, pred, pos_label=1, zero_division=0.0),
        "precision": precision_score(y_true, pred, pos_label=1, zero_division=0.0),
        "spec": (tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "auc": roc_auc_score(y_true, proba[:, 1]),
        "f1": f1_score(y_true, pred, pos_label=1, zero_division=0.0),
    }, cm


def metrics_multiclass(y_true, pred, proba):
    cm = confusion_matrix(y_true, pred, labels=[0, 1, 2])
    specs = []
    for c in range(cm.shape[0]):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        specs.append((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    m = {
        "acc": accuracy_score(y_true, pred),
        "sen": recall_score(y_true, pred, average="macro", zero_division=0.0),
        "precision": precision_score(y_true, pred, average="macro", zero_division=0.0),
        "spec": float(np.mean(specs)),
        "auc": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "f1": f1_score(y_true, pred, average="macro", zero_division=0.0),
    }
    for i, c in enumerate(CLASS_ORDER):
        m[f"recall_{c.lower()}"] = float(recall_score(
            y_true, pred, labels=[i], average=None, zero_division=0.0)[0])
        m[f"precision_{c.lower()}"] = float(precision_score(
            y_true, pred, labels=[i], average=None, zero_division=0.0)[0])
        m[f"f1_{c.lower()}"] = float(f1_score(
            y_true, pred, labels=[i], average=None, zero_division=0.0)[0])
    return m, cm


def make_model(name, spec, seed, class_weight, priors):
    if name == "svm_rbf":
        # SVC(probability=True) is deprecated in sklearn 1.9 and its libsvm
        # Platt scaling is nondeterministic; a fixed 5-fold calibration makes
        # probabilities (and AUC) reproducible across seeds.
        est = SVC(C=1.0, kernel="rbf", gamma="scale")
        if class_weight == "balanced":
            est.set_params(class_weight="balanced")
        return CalibratedClassifierCV(est, cv=5, method="sigmoid")
    klass = MODELS[spec["class"]]
    kwargs = dict(spec["kwargs"])
    if name in RANDOM_STATE_MODELS:
        kwargs["random_state"] = seed
    if class_weight == "balanced":
        if name in BALANCED_WEIGHT_MODELS:
            kwargs["class_weight"] = "balanced"
        elif name == "gaussian_nb" and priors is not None:
            kwargs["priors"] = priors
    return klass(**kwargs)


def evaluate(cfg, X, y_global, groups, fold_of, held_ids, problem, model_name, seed, fold):
    g_train, g_val = global_split_masks(fold_of, held_ids, fold, len(groups))
    inc = set(problem["include_groups"])
    tr = np.array([i for i in g_train if groups[i] in inc], dtype=int)
    va = np.array([i for i in g_val if groups[i] in inc], dtype=int)
    y_tr, y_va = y_global[tr], y_global[va]

    if problem["name"] == "co_mci_ad":
        assert (y_va == 2).sum() == 7, (fold, "expected 3 fold AD + 4 heldout AD")
        assert (y_tr == 2).sum() == 9, (fold, "expected 12 AD - 3 fold AD in training")
    elif problem["name"] == "co_vs_ad":
        assert (y_va == 1).sum() == 7, (fold, "expected 7 AD in AD-positive validation")

    scaler = StandardScaler().fit(X[tr])
    X_tr = scaler.transform(X[tr])
    X_val = scaler.transform(X[va])

    priors, sample_weights = None, None
    if cfg["data"].get("class_weight") == "balanced":
        n_classes = int(np.unique(y_tr).size)
        counts = np.bincount(y_tr, minlength=n_classes)
        if (counts > 0).all():
            inv = 1.0 / counts
            priors = inv / inv.sum()
            w = len(y_tr) / (n_classes * counts)
            sample_weights = w[y_tr]

    t0 = time.time()
    model = make_model(model_name, cfg["models"][model_name], seed,
                       cfg["data"].get("class_weight"), priors)
    if model_name == "mlp" and sample_weights is not None:
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
    else:
        model.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)
    if problem["positive"] is not None:
        m, cm = metrics_binary(y_va, pred, proba)
    else:
        m, cm = metrics_multiclass(y_va, pred, proba)

    row = {"problem": problem["name"], "model": model_name, "seed": seed,
           "fold": fold, "n_train": int(len(tr)), "n_val": int(len(va))}
    row.update({k: float(m[k]) for k in METRIC_COLS})
    for c in PER_CLASS_COLS:
        row[c] = m.get(c, "")
    return row, cm, fit_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--smoke", action="store_true",
                    help="3-class x 1 seed x 1 fold, all models -> smoke_detail.csv")
    ap.add_argument("--problems", help="comma-separated problem names")
    ap.add_argument("--models", help="comma-separated model names")
    ap.add_argument("--seeds", help="comma-separated seeds")
    ap.add_argument("--folds", help="comma-separated folds")
    args = ap.parse_args()

    cfg = load_config(args.config)
    X, pid_arr, groups, feats = load_data(cfg)
    fold_of, held_ids = load_split(cfg)
    global _PID_GLOBAL
    _PID_GLOBAL = pid_arr

    problems = ([p for p in cfg["problems"] if p["name"] in args.problems.split(",")]
                if args.problems else list(cfg["problems"]))
    models = args.models.split(",") if args.models else list(cfg["models"].keys())
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else cfg["cv"]["seeds"]
    folds = [int(f) for f in args.folds.split(",")] if args.folds else \
        list(range(1, cfg["cv"]["n_folds"] + 1))
    if args.smoke:
        problems, seeds, folds = [problems[0]], [seeds[0]], [folds[0]]

    out_dir = ROOT / cfg["output"]["dir"]
    detail_rel = cfg["output"]["smoke_csv"] if args.smoke else cfg["output"]["detail_csv"]
    detail_path = out_dir / detail_rel
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    if detail_path.exists():
        sys.exit(f"refusing to overwrite existing results: {detail_path}")

    run_start = time.time()
    n_rows = 0
    cm_rows = []
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        w.writeheader()
        for prob in problems:
            y_global = build_problem(groups, prob)
            for mname in models:
                for seed in seeds:
                    for fold in folds:
                        row, cm, fit_s = evaluate(cfg, X, y_global, groups, fold_of,
                                                  held_ids, prob, mname, seed, fold)
                        w.writerow(row)
                        f.flush()
                        labels = CLASS_ORDER if cm.shape[0] == 3 else \
                            ["CN", prob["positive"]]
                        for ti in range(cm.shape[0]):
                            for pi in range(cm.shape[1]):
                                cm_rows.append({"problem": prob["name"], "model": mname,
                                                "seed": seed, "true": labels[ti],
                                                "pred": labels[pi], "n": int(cm[ti, pi])})
                        n_rows += 1
                        print(f"  {prob['name']} {mname} seed={seed} fold={fold} "
                              f"acc={row['acc']:.4f} auc={row['auc']:.4f} "
                              f"n_train={row['n_train']} n_val={row['n_val']} "
                              f"({fit_s:.1f}s)", flush=True)

    cm_path = out_dir / cfg["output"]["confusion_csv"]
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cm_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "model", "seed", "true", "pred", "n"])
        w.writeheader()
        w.writerows(cm_rows)

    log_path = out_dir / cfg["output"]["run_log_csv"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["timestamp_utc", "mode", "problems", "models", "seeds",
                        "folds", "n_rows", "duration_s", "status"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "smoke" if args.smoke else "full",
                    ",".join(p["name"] for p in problems), ",".join(models),
                    ",".join(map(str, seeds)), ",".join(map(str, folds)),
                    n_rows, f"{time.time() - run_start:.1f}", "ok"])

    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "command": " ".join(sys.argv),
        "git_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip() or "unknown",
        "git_dirty": bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                         capture_output=True, text=True).stdout.strip()),
        "config_sha256": cfg["config_sha256"],
        "data_sha256": {
            "wide": sha256_file(ROOT / cfg["data"]["wide_csv"]),
            "fold_assignments": sha256_file(ROOT / cfg["split"]["assignments_csv"]),
            "heldout_ad": sha256_file(ROOT / cfg["split"]["heldout_csv"]),
        },
        "class_weight": cfg["data"].get("class_weight"),
        "split": {"n_folds": cfg["cv"]["n_folds"], "seeds": seeds,
                  "heldout_ad_appended_to_every_val_fold": True},
        "packages": {p: importlib.metadata.version(p)
                     for p in ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm",
                               "PyYAML"]},
        "n_rows": n_rows,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"done: {n_rows} rows -> {detail_path}")
    print(f"confusion -> {cm_path}")
    print(f"provenance -> {out_dir / 'provenance.json'}")


if __name__ == "__main__":
    main()
