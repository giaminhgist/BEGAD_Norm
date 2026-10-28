"""Shared data, split and preprocessing helpers.

Both experiments read the same wide CSV (processed_dataset_33feat_wide.csv,
33 features x 4 tasks = 132 columns, 157 subjects) and the same fixed split
(processed_dataset/splits/: 4 held-out AD subjects appended to every validation
fold, 153 subjects in 4 stratified folds, seed 42). Imputation and scaling are
fitted on the training subjects of the current level only and then applied to
the validation subjects (no leakage).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]

LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
CLASS_ORDER = ["CN", "MCI", "AD"]
TASKS = ["KD1", "KD2", "KD3", "READING"]


def load_wide(cfg):
    df = pd.read_csv(ROOT / cfg["data"]["wide_csv"])
    id_col, label_col = cfg["data"]["id_col"], cfg["data"]["label_col"]
    feats = [c for c in df.columns if c not in (id_col, label_col)]
    X = df[feats].to_numpy(dtype=float)
    assert X.shape[1] == cfg["data"]["n_feature_columns"], \
        (X.shape[1], f"expected {cfg['data']['n_feature_columns']} feature columns")
    return X, df[id_col].astype(str).to_numpy(), \
        df[label_col].astype(str).to_numpy(), feats


def task_columns(feats, task):
    """Column indices of one task's 33 features (order as in the wide CSV)."""
    return [i for i, c in enumerate(feats) if c.split("__")[1] == task]


def load_split(cfg):
    assign = pd.read_csv(ROOT / cfg["split"]["assignments_csv"])
    held = pd.read_csv(ROOT / cfg["split"]["heldout_csv"])
    fold_of = dict(zip(assign["subject_id"].astype(str), assign["fold"].astype(int)))
    held_ids = sorted(held["subject_id"].astype(str))
    assert len(fold_of) == cfg["split"]["n_cv_subjects"]
    assert len(held_ids) == cfg["split"]["n_heldout_ad"]
    assert not (set(held_ids) & set(fold_of)), "heldout AD must not be in CV assignments"
    return fold_of, held_ids


def build_problem(groups, problem):
    """Global label array: encoded label for included subjects, -1 elsewhere."""
    inc = set(problem["include_groups"])
    y = np.full(len(groups), -1, dtype=int)
    for i, g in enumerate(groups):
        if g in inc:
            y[i] = LABEL_MAP[g] if problem["positive"] is None else int(g == problem["positive"])
    return y


def fold_masks(fold_of, held_ids, groups, problem, fold, pid_arr):
    """Global train/val index arrays for one outer fold of one problem.

    Train = subjects whose assigned fold differs (held-out AD are not in
    fold_of, so they never enter training). Val = the fold's subjects plus the
    4 held-out AD subjects. Rows whose group is not in the problem are dropped.
    """
    inc = set(problem["include_groups"])
    val_ids = {p for p, f in fold_of.items() if f == fold} | set(held_ids)
    va = np.array([i for i, p in enumerate(pid_arr)
                   if p in val_ids and groups[i] in inc], dtype=int)
    tr = np.array([i for i, p in enumerate(pid_arr)
                   if p in fold_of and fold_of[p] != fold and groups[i] in inc],
                  dtype=int)
    return tr, va


def fit_prep(X, idx):
    """Fit per-level median imputation + standardization on X[idx].

    Returns a transform function applicable to any subset (the imputer and
    scaler were fitted on the training indices only).
    """
    imp = SimpleImputer(strategy="median").fit(X[idx])
    scaler = StandardScaler().fit(imp.transform(X[idx]))

    def transform(Xin):
        Z = imp.transform(Xin)
        Z = scaler.transform(Z)
        assert not np.isnan(Z).any(), "imputation must remove all NaN cells"
        return Z

    return transform
