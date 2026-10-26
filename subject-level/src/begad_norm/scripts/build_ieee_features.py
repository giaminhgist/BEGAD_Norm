"""Build the IEEE-mapped feature table (16 of 20 features) per (participant, task).

Mirrors build_features.py: same roster loop, parsing, fixation cleaning and
cohort guards; then computes the IEEE feature set (features/ieee.py) instead
of the 25 selected features. Outputs (processed_dataset/):
ieee_features.csv, ieee_feature_names.txt.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from begad_norm.config import DATA_DIR, MIN_FIX, OUT_DIR, PARTICIPANTS_CSV, TASK_IDS
from begad_norm.features.ieee import IEEE_FEATURE_NAMES, compute_ieee_features
from begad_norm.rawdata import (
    clean_fixations,
    parse_decimal_columns,
    participant_id_from_filename,
    read_participant_file,
)

TABLE_COLUMNS = ["participant_id", "task"] + IEEE_FEATURE_NAMES


def build_all(participants_subset=None):
    """Build the IEEE feature table for the roster cohort.

    Returns (features, stats): features has one row per (participant, task);
    stats is keyed by participant_id with the cleaning counts.
    """
    roster = pd.read_csv(PARTICIPANTS_CSV)
    if participants_subset:
        roster = roster[roster["participant_id"].isin(participants_subset)].copy()
    if roster.empty:
        raise ValueError(f"no roster participants match {participants_subset}")

    rows, stats = [], {}
    for rec in roster.itertuples(index=False):
        path = DATA_DIR / rec.filename
        df = parse_decimal_columns(read_participant_file(path))
        file_groups = df["Group"].dropna().unique()
        file_followups = df["IsFollowup"].dropna().unique()
        assert file_groups.size == 1 and float(file_groups[0]) == float(rec.group), (
            f"{rec.filename}: Group {file_groups} != roster group {rec.group}")
        assert file_followups.size == 1 and float(file_followups[0]) == 0.0, (
            f"{rec.filename}: IsFollowup {file_followups} != 0")

        clean, per_file_stats = clean_fixations(df, return_stats=True)
        stats[rec.participant_id] = per_file_stats
        bl_l = df["PupilLeftBaseline"].dropna()
        bl_r = df["PupilRightBaseline"].dropna()
        baseline_l = float(bl_l.iloc[0]) if not bl_l.empty else np.nan
        baseline_r = float(bl_r.iloc[0]) if not bl_r.empty else np.nan

        for task in TASK_IDS:
            g = clean[clean["task"] == task]
            feat = compute_ieee_features(g, baseline_l, baseline_r)
            feat["participant_id"] = rec.participant_id
            feat["task"] = task
            rows.append(feat)

    features = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    return features, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the IEEE-mapped feature table (16 features).")
    parser.add_argument("--participants", type=str, default=None,
                        help="comma-separated participant subset (default: full roster)")
    args = parser.parse_args()
    subset = (args.participants.split(",") if args.participants else None)

    t0 = time.time()
    features, stats = build_all(participants_subset=subset)
    elapsed = time.time() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "ieee_features.csv"
    features.to_csv(out_csv, index=False)
    (OUT_DIR / "ieee_feature_names.txt").write_text(
        "\n".join(IEEE_FEATURE_NAMES) + "\n")

    n_nan = features[IEEE_FEATURE_NAMES].isna().sum()
    print(f"wrote {out_csv}: {len(features)} rows "
          f"({features.participant_id.nunique()} participants x {len(TASK_IDS)} tasks) "
          f"in {elapsed:.1f}s")
    print("rows with < MIN_FIX clean fixations (all-NaN rows):",
          int((features[IEEE_FEATURE_NAMES].isna().all(axis=1)).sum()))
    print("NaN counts per feature (worst 5):")
    print(n_nan.sort_values(ascending=False).head(5).to_string())


if __name__ == "__main__":
    main()
