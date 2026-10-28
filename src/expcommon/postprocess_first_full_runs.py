"""Documented post-processing of the 2026-10-28 first full runs.

Two lossless, rule-based completions; both operate on artifacts the runners
themselves wrote, add no new computation beyond applying the documented
decision rules to stored per-subject probabilities, and are recorded in this
file for provenance:

1. The contrasts proba CSVs were written in long layout (one row per
   representation and subject). They are merged here into one row per
   (fold, subject) by union of the per-representation probability columns;
   no value is changed (this is the same wide layout the fusion proba CSVs
   already use). The runner code was fixed to write wide directly.

2. The first full runs left the shared "bacc" cell empty for 3-class rows
   (every other 3-class metric was stored). Balanced accuracy is macro recall
   under the argmax rule, so it is recomputed here from the stored
   per-subject probabilities. Validation built in: the recomputed per-class
   recalls must match the already-stored recall_cn/recall_mci/recall_ad cells
   exactly, otherwise nothing is written for that row. The runner code was
   fixed to store bacc directly.

Usage:
    PYTHONPATH=src .venv/bin/python -m expcommon.postprocess_first_full_runs
"""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = Path(__file__).resolve().parents[2]


def merge_long_proba_rows(path):
    """Merge long-layout proba rows (one row per representation and subject)
    into one row per (fold, subject). Lossless union of columns."""
    rows = list(csv.DictReader(path.open()))
    keys = sorted({k for r in rows for k in r})
    meta = ["subject_id", "group", "fold", "role", "y"]
    out, seen = [], {}
    for r in rows:
        idx = (r["subject_id"], r["fold"])
        if idx not in seen:
            seen[idx] = {m: r[m] for m in meta}
            out.append(seen[idx])
        for k, v in r.items():
            if k not in meta and v != "":
                assert k not in seen[idx] or seen[idx][k] == v
                seen[idx][k] = v
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"merged {len(rows)} -> {len(out)} rows: {path}")
    return out


def fill_bacc(out_dir, variant_key):
    detail = out_dir / "results/results_detail.csv"
    rows = list(csv.DictReader(detail.open()))
    n_filled = 0
    for r in rows:
        if r["problem"] != "co_mci_ad" or r.get("bacc"):
            continue   # binary rows already store bacc; skip completed 3-class rows
        proba_path = out_dir / f"results/proba_co_mci_ad_{r['model_type']}.csv"
        pr = list(csv.DictReader(proba_path.open()))
        preds, ys = [], []
        for row in pr:
            if row["fold"] != r["fold"]:
                continue
            ys.append(int(row["y"]))
            preds.append(max(range(3), key=lambda c: float(row[f"p{c}_{r[variant_key]}"])))
        preds, ys = np.array(preds), np.array(ys)
        assert len(preds) == int(r["n_val"]), (r, len(preds))
        bacc = balanced_accuracy_score(ys, preds)
        recalls = recall_score(ys, preds, labels=[0, 1, 2], average=None,
                               zero_division=0.0)
        # built-in validation against the stored per-class recalls
        for c, col in zip(range(3), ["recall_cn", "recall_mci", "recall_ad"]):
            assert abs(recalls[c] - float(r[col])) < 5e-5, \
                (r[variant_key], r["fold"], col, recalls[c], r[col])
        r["bacc"] = f"{bacc:.4f}"
        n_filled += 1
    with open(detail, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{out_dir}: filled {n_filled} bacc cells, all validated against "
          f"stored per-class recalls")
    return n_filled


def main():
    out = ROOT / "output"
    for path in sorted((out / "contrasts/results").glob("proba_co_mci_ad_*.csv")):
        merge_long_proba_rows(path)
    n = fill_bacc(out / "fusion", "variant")
    n += fill_bacc(out / "contrasts", "representation")
    print(f"total {n} bacc cells completed")


if __name__ == "__main__":
    main()
