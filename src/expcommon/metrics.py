"""Evaluation metrics per the protocol.

Binary (CN vs MCI, CN vs AD): ROC-AUC, average precision, F1, balanced
accuracy, sensitivity and specificity at the decision threshold (default 0.5);
the Youden-J threshold is derived from training-level inner OOF probabilities
and applied to the validation fold as supplementary columns.

3-class (CN/MCI/AD): accuracy, balanced accuracy, macro-F1, one-vs-rest macro
AUC and per-class recall, all at the argmax decision rule.
"""

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, recall_score, roc_auc_score, roc_curve)

BIN_METRICS = ["auc", "ap", "f1", "bacc", "sen", "spec"]
MC_METRICS = ["acc", "bacc", "macro_f1", "ovr_auc",
              "recall_cn", "recall_mci", "recall_ad"]


def binary_metrics(y, p1, thr):
    pred = (p1 >= thr).astype(int)
    return {
        "auc": float(roc_auc_score(y, p1)),
        "ap": float(average_precision_score(y, p1)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0.0)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "sen": float(recall_score(y, pred, pos_label=1, zero_division=0.0)),
        "spec": float(recall_score(y, pred, pos_label=0, zero_division=0.0)),
    }


def youden_threshold(y, p1):
    """Threshold maximizing sensitivity + specificity - 1 on training data.

    sklearn's roc_curve pairs point (fpr[i], tpr[i]) with threshold th[i];
    th[0] corresponds to the all-negative rule (inf in roc_curve, mapped here
    to max(score) + 1e-6 so the value stays finite while keeping the
    all-negative semantics). On ties argmax keeps the first (most
    conservative, fewest positives) rule.
    """
    try:
        fpr, tpr, th = roc_curve(y, p1)
    except ValueError:
        return 0.5
    j = tpr - fpr
    k = int(np.argmax(j))
    if k == 0:
        return float(np.max(p1) + 1e-6)   # all-negative rule, finite sentinel
    return float(th[k])


def multiclass_metrics(y, P, pred):
    m = {
        "acc": float(np.mean(pred == y)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0.0)),
        "ovr_auc": float(roc_auc_score(y, P, multi_class="ovr", average="macro")),
    }
    for i, c in enumerate(["CN", "MCI", "AD"]):
        m[f"recall_{c.lower()}"] = float(recall_score(
            y, pred, labels=[i], average=None, zero_division=0.0)[0])
    return m
