"""Tabular (25-feature) access for the normative pipeline.

Per-participant task matrices X (S x D) + presence masks, plus leakage-safe
statistics: everything is fitted ONLY on the given train participants.

Missingness rules (ported from LNGModel src/data/tabular.py):
  - A (participant, task) row whose features are ALL NaN is a *missing task*:
    masked everywhere (pooling, bank, losses). The mask is defined from the
    all-NaN rows BEFORE anything else.
  - A cell that is NaN while its row is otherwise valid is filled with the
    train-fitted per-feature mean, so it contributes ZERO deviation after
    standardization.
  - Zero-variance features: std is clamped to EPS, so a constant feature
    contributes (x - mu) / EPS = 0 deviation everywhere.
"""
import numpy as np
import pandas as pd
import torch

EPS = 1e-6

# Problem -> (included group labels, label mapping). CO is the normative
# reference group in every problem.
PROBLEMS = {
    "co_vs_mci": {"groups": ["CO", "MCI"], "positive": "MCI",
                  "labels": {"CO": 0, "MCI": 1}},
    "co_vs_ad": {"groups": ["CO", "AD"], "positive": "AD",
                 "labels": {"CO": 0, "AD": 1}},
    "co_mci_ad": {"groups": ["CO", "MCI", "AD"], "positive": None,
                  "labels": {"CO": 0, "MCI": 1, "AD": 2}},
}


def build_matrices(participants, features_csv, feature_cols, n_tasks=11):
    """Return {participant_id: (X (S, D) float32, mask (S,) bool)}.

    X rows are NaN for missing (participant, task) pairs or missing feature
    cells; mask marks rows that are not entirely NaN.
    """
    feat = pd.read_csv(features_csv)
    out = {}
    by_pid = {pid: sub for pid, sub in feat.groupby("participant_id")}
    task_ids = list(range(n_tasks))
    for pid in participants:
        if pid in by_pid:
            sub = by_pid[pid].set_index("task").reindex(task_ids)
            X = sub[list(feature_cols)].to_numpy(dtype=np.float32)
        else:  # participant with no rows at all (defensive)
            X = np.full((n_tasks, len(feature_cols)), np.nan, dtype=np.float32)
        mask = ~np.isnan(X).all(axis=1)
        out[str(pid)] = (X, mask)
    return out


def feature_norm_stats(train_pids, matrices):
    """Per-feature mean/std over the given participants' valid rows.

    Used to scale raw inputs and to fill missing feature cells. Fits on the
    passed participants only — call with the training fold to avoid leakage.
    """
    X = np.concatenate([matrices[p][0][matrices[p][1]] for p in train_pids],
                       axis=0)
    mean = np.nanmean(X, axis=0).astype(np.float32)
    std = np.nanstd(X, axis=0).astype(np.float32)
    std = np.maximum(std, EPS)
    return mean, std


class NormativeDataset:
    """Participants with standardized per-task feature tensors + masks.

    Inputs are standardized per-feature with stats fitted on
    `train_participants` only (unsupervised, leakage-safe); the encoder +
    latent bank do the deviation work inside the model.
    """

    def __init__(self, participants, train_participants, features_csv,
                 metadata_csv, problem, n_tasks=11):
        if problem not in PROBLEMS:
            raise ValueError(f"unknown problem {problem!r}; "
                             f"expected one of {sorted(PROBLEMS)}")
        spec = PROBLEMS[problem]
        self.problem = problem
        self.positive = spec["positive"]
        meta = pd.read_csv(metadata_csv).set_index("participant_id")
        self.subjects = [str(p) for p in participants
                         if meta.loc[str(p), "group_label"] in spec["groups"]]
        self.group_labels = [meta.loc[s, "group_label"] for s in self.subjects]
        self.labels = np.array([spec["labels"][g] for g in self.group_labels])
        feature_cols = [c for c in pd.read_csv(features_csv, nrows=1).columns
                        if c not in ("participant_id", "task")]
        # Matrices cover the union of this dataset's subjects and the train
        # participants (both restricted to the problem's groups): stats need
        # the train fold even when this dataset holds only val subjects, and
        # a train list may still contain out-of-problem ids.
        all_ids = list(dict.fromkeys(
            self.subjects + [str(p) for p in train_participants]))
        all_ids = [p for p in all_ids
                   if meta.loc[p, "group_label"] in spec["groups"]]
        mats = build_matrices(all_ids, features_csv, feature_cols, n_tasks)
        train = [str(p) for p in train_participants if str(p) in mats]
        if not train:
            raise ValueError("train_participants has no overlap with the "
                             "problem groups")
        mu, sigma = feature_norm_stats(train, mats)
        self.X, self.mask = [], []
        for s in self.subjects:
            X, mask = mats[s]
            X = np.where(np.isnan(X), mu, X).astype(np.float32)  # mean-fill
            X = (X - mu) / sigma                                  # standardize
            X = X * mask[:, None].astype(np.float32)              # mask tasks
            self.X.append(torch.from_numpy(X))
            self.mask.append(torch.from_numpy(mask.astype(np.float32)))
        self.X = torch.stack(self.X, 0)        # (P, S, D)
        self.mask = torch.stack(self.mask, 0)  # (P, S)

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return (self.X[idx], self.mask[idx]), self.labels[idx], self.subjects[idx]

    def collate(self, items):
        """Batched x from a list of dataset items (first elements)."""
        return (torch.stack([it[0][0] for it in items], 0),
                torch.stack([it[0][1] for it in items], 0))

    def co_indices(self):
        """Indices of CO (normative reference) subjects in this dataset."""
        return [i for i, g in enumerate(self.group_labels) if g == "CO"]
