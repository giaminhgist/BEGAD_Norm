"""Aggregate results_detail.csv into summary.csv.

For each (dataset, problem, model, metric): mean and std over the 20 seeds x
4 folds (= 80 evaluations). Refuses to overwrite an existing summary.

Usage:
    python -m begad_norm.baseline.aggregate_results \
        --detail outputs/baseline/results/results_detail.csv \
        --summary outputs/baseline/results/summary.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

METRICS = ["acc", "sen", "spec", "auc", "f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", default="outputs/baseline/results/results_detail.csv")
    ap.add_argument("--summary", default="outputs/baseline/results/summary.csv")
    ap.add_argument("--expected-rows", type=int, default=1440,
                    help="1 dataset x 3 problems x 6 models x 20 seeds x 4 folds")
    ap.add_argument("--per-group", type=int, default=80,
                    help="evaluations per (dataset, problem, model): 20 seeds x 4 folds")
    args = ap.parse_args()

    detail_path = Path(args.detail)
    summary_path = Path(args.summary)
    if not detail_path.exists():
        sys.exit(f"detail file not found: {detail_path}")
    if summary_path.exists():
        sys.exit(f"refusing to overwrite existing summary: {summary_path}")

    df = pd.read_csv(detail_path)
    if len(df) != args.expected_rows:
        sys.exit(f"detail has {len(df)} rows, expected {args.expected_rows} — "
                 f"is the full run complete?")

    rows = []
    for (dataset, problem, model), g in df.groupby(
            ["dataset", "problem", "model"]):
        if len(g) != args.per_group:
            sys.exit(f"{dataset}/{problem}/{model}: {len(g)} rows, "
                     f"expected {args.per_group}")
        for m in METRICS:
            rows.append({"dataset": dataset, "problem": problem,
                         "model": model, "metric": m,
                         "mean": g[m].mean(), "std": g[m].std(),
                         "n_runs": len(g)})
    out = pd.DataFrame(rows).sort_values(
        ["dataset", "problem", "model", "metric"]).reset_index(drop=True)
    out.to_csv(summary_path, index=False)
    print(f"summary written: {summary_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
