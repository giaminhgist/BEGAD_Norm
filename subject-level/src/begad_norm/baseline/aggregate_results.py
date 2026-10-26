"""Aggregate results_detail.csv into summary.csv.

For each group of the detail rows (default: dataset, problem, model;
overridable with --group-cols), computes per metric the mean and std over
the group's evaluations. Refuses to overwrite an existing summary.

Usage:
    python -m begad_norm.baseline.aggregate_results \
        --detail outputs/baseline/results/results_detail.csv \
        --summary outputs/baseline/results/summary.csv
    # row-level experiment: one summary row per (problem, model, split,
    # eval_level) x metric, 50 evaluations each (5 model seeds x 10 folds):
    python -m begad_norm.baseline.aggregate_results \
        --detail T1235/results/results_detail.csv \
        --summary T1235/results/summary.csv \
        --group-cols dataset,problem,model,split,eval_level \
        --expected-rows 2700 --per-group 50
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

METRICS = ["acc", "sen", "precision", "spec", "auc", "f1"]


def aggregate(df, group_cols, per_group):
    """Group the detail rows by group_cols and return one summary row per
    (group, metric) with mean/std/n_runs; every group must contain exactly
    per_group evaluations."""
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"--group-cols not in detail columns: {missing}")
    rows = []
    for key, g in df.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        if len(g) != per_group:
            raise ValueError(f"{'/'.join(map(str, key))}: {len(g)} rows, "
                             f"expected {per_group}")
        for m in METRICS:
            rows.append({**dict(zip(group_cols, key)), "metric": m,
                         "mean": g[m].mean(), "std": g[m].std(),
                         "n_runs": len(g)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", default="outputs/baseline_balanced/results/results_detail.csv")
    ap.add_argument("--summary", default="outputs/baseline_balanced/results/summary.csv")
    ap.add_argument("--expected-rows", type=int, default=1800,
                    help="1 dataset x 3 problems x 6 models x 20 seeds x 5 folds")
    ap.add_argument("--per-group", type=int, default=100,
                    help="evaluations per group: 20 seeds x 5 folds")
    ap.add_argument("--group-cols", default="dataset,problem,model",
                    help="comma-separated detail columns to group by "
                         "(default: dataset,problem,model)")
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

    group_cols = [c.strip() for c in args.group_cols.split(",")]
    out = aggregate(df, group_cols, args.per_group)
    out = out.sort_values(group_cols + ["metric"]).reset_index(drop=True)
    out.to_csv(summary_path, index=False)
    print(f"summary written: {summary_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
