"""Aggregate the baseline detail CSV into a summary table (mean +/- std over
the 40 evaluations per problem/model = 10 seeds x 4 folds) and keep the
per-seed/fold detail untouched.

Usage:
    PYTHONPATH=src .venv/bin/python -m baseline.aggregate_results \
        --config src/baseline/config.yaml [--smoke]
"""

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRIC_COLS = ["acc", "sen", "precision", "spec", "auc", "f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = ROOT / cfg["output"]["dir"]
    detail = out_dir / (cfg["output"]["smoke_csv"] if args.smoke else cfg["output"]["detail_csv"])
    summary = out_dir / cfg["output"]["summary_csv"]
    summary.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(detail.open()))
    groups = {}
    for r in rows:
        groups.setdefault((r["problem"], r["model"]), []).append(r)

    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "model", "n_eval", "n_train", "n_val"]
                           + [f"{m}_mean" for m in METRIC_COLS]
                           + [f"{m}_std" for m in METRIC_COLS]
                           + ["recall_ad_mean", "recall_ad_std",
                              "f1_ad_mean", "f1_ad_std", "note"])
        w.writeheader()
        for (problem, model), rs in sorted(groups.items()):
            row = {"problem": problem, "model": model, "n_eval": len(rs),
                   "n_train": rs[0]["n_train"], "n_val": rs[0]["n_val"],
                   "note": ""}
            for m in METRIC_COLS:
                vals = np.array([float(r[m]) for r in rs])
                row[f"{m}_mean"] = f"{vals.mean():.4f}"
                row[f"{m}_std"] = f"{vals.std():.4f}"
            if rs[0]["recall_ad"]:
                ra = np.array([float(r["recall_ad"]) for r in rs])
                fa = np.array([float(r["f1_ad"]) for r in rs])
                row["recall_ad_mean"] = f"{ra.mean():.4f}"
                row["recall_ad_std"] = f"{ra.std():.4f}"
                row["f1_ad_mean"] = f"{fa.mean():.4f}"
                row["f1_ad_std"] = f"{fa.std():.4f}"
            else:
                row["recall_ad_mean"] = row["recall_ad_std"] = ""
                row["f1_ad_mean"] = row["f1_ad_std"] = ""
            if model in ("logistic_regression", "svm_rbf", "gaussian_nb"):
                row["note"] = "deterministic: 10 seed-rows repeat identically"
            w.writerow(row)
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
