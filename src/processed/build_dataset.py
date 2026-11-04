#!/usr/bin/env python3
"""Build the 33-feature per-task dataset from raw Tobii exports.

One row per (subject, task). Columns: subject_id, task, group (0/1/2),
then F01..F33 (see feature_selection.md for definitions).

Restores the original feature-discovery extraction code (event_contracts.py +
extract_macro.py from git commit ffa36c7) and applies it to all 157 roster
participants over the four tasks KD1/KD2/KD3/READING.

Deterministic: same inputs -> identical output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from event_contracts import RAW_TASK_TO_CANONICAL, build_fixations, build_saccades, build_transitions, read_raw  # noqa
from extract_macro import task_features  # noqa

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ROSTER = ROOT / "dataset" / "participants_20250305.csv"
RAW_DIR = ROOT / "dataset" / "20250305"
OUT = ROOT / "processed_data" / "processed_dataset_long.csv"

# 33 selected RAW157 features in canonical order (matches feature_selection.md and
# the original feature_columns_33feat.csv order A002..A046, D017..D047 with the 9
# dropped IDs removed). F01..F33 = the renumbered names.
SELECTED = [
    "A002", "A003", "A021", "A023", "A024", "A026", "A027", "A028",
    "A029", "A030", "A031", "A033", "A035", "A046",
    "D017", "D018", "D021", "D022", "D023", "D024", "D025", "D026",
    "D028", "D029", "D030", "D034", "D035", "D036", "D037", "D039",
    "D040", "D041", "D047",
]
assert len(SELECTED) == 33, len(SELECTED)
RENAMED = {fid: f"F{i:02d}" for i, fid in enumerate(SELECTED, start=1)}

GROUP_CODE_TO_LABEL = {0: "CN", 1: "MCI", 2: "AD"}
CANONICAL_TASK_LABEL = {v: k for k, v in RAW_TASK_TO_CANONICAL.items()}  # noqa (kept for clarity)

EXPECT_SUBJECTS = 157
EXPECT_GROUPS = {"CN": 100, "MCI": 41, "AD": 16}
TASK_ORDER = ["KD1", "KD2", "KD3", "READING"]


def main() -> None:
    roster = pd.read_csv(ROSTER).rename(columns={"participant_id": "subject_id", "group": "group_code"})
    assert set(roster.group_code) == {0, 1, 2}, roster.group_code.unique()
    roster["group"] = roster["group_code"].map(GROUP_CODE_TO_LABEL)

    records = []
    for r in roster.itertuples():
        path = RAW_DIR / r.filename
        df = read_raw(path)
        fix = build_fixations(df)
        sac = build_saccades(df)
        tr = build_transitions(fix, sac)
        keep_rows = ["Task", "ValidityLeft", "ValidityRight", "PupilLeft", "PupilRight"]
        rows = df[keep_rows].copy()

        per_task = {}
        for code, canon in RAW_TASK_TO_CANONICAL.items():
            per_task[canon] = task_features(fix, sac, tr, rows, code)

        for canon in TASK_ORDER:
            feats = per_task[canon]
            rec = {"subject_id": r.subject_id, "task": canon, "group": r.group,
                   "group_code": r.group_code}
            for fid in SELECTED:
                v, st, n = feats[fid]
                rec[RENAMED[fid]] = v
                rec[f"{RENAMED[fid]}_status"] = st
                rec[f"{RENAMED[fid]}_n"] = n
            records.append(rec)

    out = pd.DataFrame(records)
    assert len(out) == EXPECT_SUBJECTS * 4, len(out)

    counts = out.drop_duplicates("subject_id")["group"].value_counts().to_dict()
    assert counts == EXPECT_GROUPS, counts
    print("group counts:", counts)

    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows)")

    # report per-feature missingness across the 628 subject-task cells
    feat_cols = list(RENAMED.values())
    nm = out[feat_cols].isna().sum()
    print("\nper-feature missing cells (of %d):" % len(out))
    print(nm[nm > 0].to_string())


if __name__ == "__main__":
    main()