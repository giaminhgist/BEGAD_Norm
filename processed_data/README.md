# processed_data — 33-feature per-task dataset

Eye-tracking feature values for the BEGAD cohort: **157 participants × 4 tasks = 628 rows**, one row per (subject, task).

## Files

| File | Description |
|---|---|
| `processed_dataset_long.csv` | The 33-feature dataset (described below). |

## Schema

`processed_dataset_long.csv` has 103 columns.

### Identity columns

| Column | Type | Description |
|---|---|---|
| `subject_id` | string | Participant identifier (e.g. `BEG-201`, `BEGMCI-006`). |
| `task` | string | One of `KD1`, `KD2`, `KD3`, `READING`. |
| `group` | string | Diagnostic group: `CN` (cognitively normal), `MCI`, `AD`. |
| `group_code` | int | Numeric diagnostic code: `0` = CN, `1` = MCI, `2` = AD. |

### Feature columns

The 33 features are named `F01`–`F33` (renumbered from the original `A002…A046` / `D017…D047` IDs). Each feature has three columns:

| Suffix | Meaning |
|---|---|
| *(none)* — `F01` | The feature value (float). `NaN` means the feature could not be supported for that subject-task. |
| `_status` — `F01_status` | Measurement status: `observed`, `observed_zero`, or `insufficient_within_subject_support`. |
| `_n` — `F01_n` | Number of underlying events used to compute the value. |

Definitions and formulas for all 33 features are in [`../feature_selection.md`](../feature_selection.md) (same numbering, `F01` = feature #1 … `F33` = feature #33).

Feature → original-ID mapping:

| New | Old | New | Old | New | Old |
|---|---|---|---|---|---|
| F01 | A002 | F12 | A033 | F23 | D028 |
| F02 | A003 | F13 | A035 | F24 | D029 |
| F03 | A021 | F14 | A046 | F25 | D030 |
| F04 | A023 | F15 | D017 | F26 | D034 |
| F05 | A024 | F16 | D018 | F27 | D035 |
| F06 | A026 | F17 | D021 | F28 | D036 |
| F07 | A027 | F18 | D022 | F29 | D037 |
| F08 | A028 | F19 | D023 | F30 | D039 |
| F09 | A029 | F20 | D024 | F31 | D040 |
| F10 | A030 | F21 | D025 | F32 | D041 |
| F11 | A031 | F22 | D026 | F33 | D047 |

## Cohort composition

| Group | Code | Subjects |
|---|---|---|
| CN | 0 | 100 |
| MCI | 1 | 41 |
| AD | 2 | 16 |
| **Total** | | **157** |

Each subject appears once per task, so every task has 157 rows (CN 100 / MCI 41 / AD 16).

## Missing values

51 cells are `NaN` (0.25% of 628 × 33 = 20 724 cells), affecting **20 subjects** and confined to 9 features (the serial-correlation / coupling / slope features that need a minimum number of matched events):

| Feature | Original ID | NaN cells |
|---|---|---|
| F15 | D017 | 4 |
| F18 | D022 | 4 |
| F21 | D025 | 1 |
| F22 | D026 | 1 |
| F25 | D030 | 6 |
| F26 | D034 | 13 |
| F27 | D035 | 4 |
| F28 | D036 | 4 |
| F29 | D037 | 14 |

These cells carry status `insufficient_within_subject_support`. The other 24 features are complete for all 628 subject-task cells.

## How it was produced

Prepared by [`../src/processed/build_dataset.py`](../src/processed/build_dataset.py), which restores the original feature-discovery extraction code (`event_contracts.py` + `extract_macro.py`) and applies it to the raw Tobii exports in [`../dataset/20250305/`](../dataset/20250305/). Run it from the repo root:

```bash
.venv/bin/python src/processed/build_dataset.py
```

The result is deterministic: identical inputs reproduce this file exactly.