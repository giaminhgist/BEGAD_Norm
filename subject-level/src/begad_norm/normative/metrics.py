"""Evaluation metrics for the normative pipeline.

Binary semantics match begad_norm.baseline.run_baselines.metrics_binary
(positive class = 1, fixed threshold 0.5, no tuning); 3-class semantics
match metrics_multiclass (macro Sen/Precision/F1, mean per-class Spec,
one-vs-rest macro AUC).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

THRESHOLD = 0.5


def _binary_auc(y, prob):
    if len(np.unique(y)) > 1:
        return float(roc_auc_score(y, prob)), None
    return float("nan"), f"single class in labels ({int(np.unique(y)[0])})"


def compute_metrics(y_true, y_score, n_classes, positive=None):
    """Return the baseline metric set for one evaluation.

    y_score: 1-d positive-class probability (binary) or (N, C) class
    probabilities (multiclass). AUC is NaN when it cannot be computed
    (single class in labels); callers must record the reason, never
    silently replace it.
    """
    y = np.asarray(y_true)
    if n_classes == 1:
        prob = np.asarray(y_score, dtype=np.float64).ravel()
        pred = (prob >= THRESHOLD).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        auc, auc_reason = _binary_auc(y, prob)
        return {
            "acc": accuracy_score(y, pred),
            "sen": recall_score(y, pred, pos_label=1),
            "precision": precision_score(y, pred, pos_label=1,
                                         zero_division=0.0),
            "spec": (tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "auc": auc,
            "f1": f1_score(y, pred, pos_label=1, zero_division=0.0),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "threshold": float(THRESHOLD), "auc_reason": auc_reason,
        }
    proba = np.asarray(y_score, dtype=np.float64)
    pred = proba.argmax(1)
    cm = confusion_matrix(y, pred)
    specs = []
    for c in range(cm.shape[0]):
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fp = cm[:, c].sum() - cm[c, c]
        specs.append((tn / (tn + fp)) if (tn + fp) > 0 else 0.0)
    if len(np.unique(y)) > 1:
        try:
            auc = float(roc_auc_score(y, proba, multi_class="ovr",
                                      average="macro"))
            auc_reason = None
        except ValueError as exc:  # e.g. a class missing from the labels
            auc, auc_reason = float("nan"), str(exc)
    else:
        auc = float("nan")
        auc_reason = f"single class in labels ({int(np.unique(y)[0])})"
    return {
        "acc": accuracy_score(y, pred),
        "sen": recall_score(y, pred, average="macro", zero_division=0.0),
        "precision": precision_score(y, pred, average="macro",
                                     zero_division=0.0),
        "spec": float(np.mean(specs)),
        "auc": auc,
        "f1": f1_score(y, pred, average="macro", zero_division=0.0),
        "auc_reason": auc_reason,
    }
