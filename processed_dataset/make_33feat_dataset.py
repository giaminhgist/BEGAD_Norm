#!/usr/bin/env python3
"""Build the 33-feature RAW157 dataset from the extraction long table.

The 33-feature set = the 24 kept features (feature_columns.csv) + the 9 features
dropped from v1/v2 because they are not complete for all 157 subjects
(D017, D022, D025, D026, D030, D034, D035, D036, D037; see README.md). All 157
subjects are kept; cells a subject cannot support (insufficient_within_subject_support)
are left as NaN. In total 51 cells are missing, all of them in the 9 added features.

The wide table carries those NaNs; the baseline pipeline imputes them per fold
(median per feature x task column, fitted on the training split only, see
src/baseline/run_baselines.py).

Writes:
    processed_dataset_33feat_wide.csv   subject_id, group, {feature}__{task} (132 cols; 51 NaN)
    processed_dataset_33feat_long.csv   subject_id, group, task, 33 feature cols (51 NaN)
    feature_columns_33feat.csv          feature_id, feature_name, unit (33 rows)

Deterministic: rebuilds from the same long table produce identical files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LONG = HERE / "raw157_feature_values_long_full.csv"
KEPT_CSV = HERE / "feature_columns.csv"
OUT_WIDE = HERE / "processed_dataset_33feat_wide.csv"
OUT_LONG = HERE / "processed_dataset_33feat_long.csv"
OUT_COLS = HERE / "feature_columns_33feat.csv"

DROPPED_9 = ["D017", "D022", "D025", "D026", "D030", "D034", "D035", "D036", "D037"]
TASK_ORDER = ["KD1", "KD2", "KD3", "READING"]
VALID_STATUS = {"observed", "observed_zero"}
EXPECT_SUBJECTS = 157
EXPECT_GROUPS = {"CN": 100, "MCI": 41, "AD": 16}
EXPECT_MISSING_CELLS = 51  # total NaN cells across the 33 features x 4 tasks


def main():
    raw = pd.read_csv(LONG)
    assert set(raw["cohort_id"].unique()) == {"RAW157"}

    kept = pd.read_csv(KEPT_CSV)
    features = sorted(set(kept["feature_id"]) | set(DROPPED_9))
    assert len(features) == 33, len(features)

    sub = raw[raw["feature_id"].isin(features)]
    assert sub.duplicated(["subject_id", "feature_id", "scope_id"]).sum() == 0

    valid = sub[sub["measurement_status"].isin(VALID_STATUS)]
    assert valid["value"].isna().sum() == 0, "valid rows must have a value"

    # --- wide: one row per subject, columns {feature}__{task}; NaN where unsupported ---
    wide = valid.pivot(index="subject_id", columns=["feature_id", "scope_id"], values="value")
    wide.columns = [f"{f}__{t}" for f, t in wide.columns]
    assert wide.shape == (EXPECT_SUBJECTS, 132), wide.shape

    missing = wide.isna()
    n_missing = int(missing.to_numpy().sum())
    print(f"missing cells: {n_missing} (expected {EXPECT_MISSING_CELLS})")
    assert n_missing == EXPECT_MISSING_CELLS
    # all NaNs must sit in the 9 added features (the 24 kept features are complete)
    bad_cols = {c for c in wide.columns[missing.any(axis=0)]
                if c.split("__")[0] not in DROPPED_9}
    assert not bad_cols, bad_cols
    subj_with_missing = wide.index[missing.any(axis=1)].tolist()
    print(f"subjects with >=1 missing cell: {len(subj_with_missing)}")

    groups = raw.groupby("subject_id")["group"].first()
    counts = groups.value_counts().to_dict()
    assert counts == EXPECT_GROUPS, counts
    print("group counts:", counts)

    wide = wide.reset_index().rename(columns={"index": "subject_id"})
    wide.insert(1, "group", wide["subject_id"].map(groups))

    # --- cross-check: the 24 shared features must match v1 wide ---
    # v1 wide was serialized with shorter float reprs than the long table, so
    # identical real numbers can differ by <= 7e-15; compare with tolerance.
    v1 = pd.read_csv(HERE / "processed_dataset_v1_wide.csv")
    shared = [f"{f}__{t}" for f in kept["feature_id"] for t in TASK_ORDER]
    cmp = wide.merge(v1[["subject_id", *shared]], on="subject_id", suffixes=("_33f", "_v1"))
    for col in shared:
        assert np.isclose(cmp[f"{col}_33f"], cmp[f"{col}_v1"], atol=1e-9).all(), \
            f"mismatch on {col}"
    print(f"cross-check: {len(shared)} shared feature columns match v1 wide (atol=1e-9)")

    # --- long: one row per (subject, task), feature columns ---
    long_df = valid.pivot_table(index=["subject_id", "scope_id"], columns="feature_id",
                                values="value", aggfunc="first")
    long_df = long_df.reset_index()
    long_df.insert(1, "group", long_df["subject_id"].map(groups))
    long_df = long_df.rename(columns={"scope_id": "task"})
    long_df["task"] = pd.Categorical(long_df["task"], TASK_ORDER)
    long_df = long_df.sort_values(["subject_id", "task"]).reset_index(drop=True)
    long_df = long_df[["subject_id", "group", "task", *features]]
    assert long_df.shape == (EXPECT_SUBJECTS * 4, 3 + 33), long_df.shape
    assert long_df[features].isna().to_numpy().sum() == EXPECT_MISSING_CELLS

    # --- feature_columns_33feat.csv ---
    meta = sub.groupby("feature_id")[["feature_name", "unit"]].first().loc[features]
    meta.reset_index().to_csv(OUT_COLS, index=False)

    wide.to_csv(OUT_WIDE, index=False)
    long_df.to_csv(OUT_LONG, index=False)
    print(f"wrote {OUT_WIDE.name} ({wide.shape}, {n_missing} NaN cells)")
    print(f"wrote {OUT_LONG.name} ({long_df.shape})")
    print(f"wrote {OUT_COLS.name} ({len(meta)} rows)")


if __name__ == "__main__":
    main()
