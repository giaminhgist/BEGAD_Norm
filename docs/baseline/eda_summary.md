# BEGAD EDA Summary — dataset, features, and group comparison

> Synthesized 2026-10-12 from the former `EDA/` folder (feature dictionary,
> group-comparison summary, and per-task statistics) during the repository
> cleanup. The EDA figures and the comparison notebook were removed; they are
> recoverable from git history. The dataset was reduced to the 25 selected
> features on 2026-10-13; the screening evidence is preserved at
> `processed_dataset/selection.csv`.

## 1. Dataset and cohort

- **Session**: 2025-03-05, export tag `BEGAD_ETRA25`; 205 raw eye-tracking
  CSVs (`dataset/20250305/`, `dataset_v2/20250305/`).
- **Cohort filter**: first-visit participants in groups CO / MCI / AD only.
  Excluded: 36 follow-up visits, 10 NPH participants (Group 3), 2 recordings
  with Group −11.
- **Participants**: 157 — CO 100, MCI 41, AD 16.
- **Protocol**: 11 tasks per participant (0–10):
  instruction screens (0, 2, 6, 8, 10), King–Devick demo/cards (1, 3, 4, 5),
  reading practice/task (7, 9). Media geometry in `TASK_GEOMETRY`
  (`src/begad_norm/config.py`).

## 2. Cleaning pipeline

Computed per (participant, task) from `dataset_v2` raw exports
(`src/begad_norm/rawdata.py`):

1. Keep rows: `Final == True`, `HasNulls == False`,
   `GazeEventType == "Fixation"`, `FixationIndex > 0` (cohort filter applied).
2. EMS duration window: `40 ms < GazeEventDuration ≤ 2000 ms` (drops 2 rows).
3. Per-task media bounds on `FixationPointX/Y (MCSpx)` (drops 0 rows).
4. Pupil outliers: per (participant, task), drop rows where the binocular mean
   pupil deviates from the group median by more than `4 × max(std, 1e-6)`
   (drops 381 rows).

Net clean fixations: **116,274**. Decimal-comma columns are parsed to float.
Fixations are ordered by `RecordingTimestamp` within each task.

## 3. Feature dictionary (25 features)

25 features per (participant, task) in `processed_dataset/features.csv`
(1727 rows = 157 × 11; 2 all-NaN rows: `BEGCO-029`, `BEGMCI-019` × task 10).
Notation: `n` = clean fixations; `x, y` = fixation centers (px, media-relative);
`dur` = fixation duration (ms); `pup` = binocular mean pupil (mm); `W, H` =
task media size; `cx, cy` = media center `(W/2, H/2)`. Sample SD (`ddof=1`);
pairs with `n < 2` get an all-NaN row (`MIN_FIX = 2`).

### `spa_` — spatial position & dispersion (4)

| Feature | Formula | Notes |
|---|---|---|
| `spa_fix_count` | `n` | never NaN |
| `spa_dispersion` | `mean(sqrt((x−mean(x))² + (y−mean(y))²))` | px, distance to centroid |
| `spa_center_dist_std` | std of fixations' distance to `(cx, cy)` | px |
| `spa_q2` | fraction of fixations with `x ≥ cx` and `y < cy` | quadrant fraction around the media center |

### `geo_` — scanpath geometry (4)

Inferred saccades between consecutive fixations: `dx = diff(x)`,
`amp = sqrt(dx² + dy²)`.

| Feature | Formula | Notes |
|---|---|---|
| `geo_scanpath_len` | `sum(amp)` | px; NaN if `n < 2` |
| `geo_sacc_amp_mean` | `mean(amp)` | px |
| `geo_revisit_rate` | fraction of fixations with a later fixation within 60 px | forward-looking (`triu`, k=1) |
| `geo_nn_dist_mean` | mean distance to nearest other fixation | px |

### `tem_` — temporal (2)

Half-dwell IFI approximation (no onset timestamps): `half_gap = (dur[:-1] + dur[1:]) / 2`.

| Feature | Formula | Notes |
|---|---|---|
| `tem_dur_total` | `sum(dur)` | ms |
| `tem_velocity_mean` | `mean(amp / half_gap)` over finite terms | px/ms |

### `pup_`, `pupbl_` — pupil (6 + 2)

`pup_` = binocular mean `(PupilLeft + PupilRight) / 2`;
`pupbl_` = baseline-corrected binocular mean (constant per-recording
baselines).

| Feature | Formula |
|---|---|
| `pup_mean` | `mean(series)` |
| `pup_std` | `std(series)` — NaN if `n < 2` |
| `pup_min` / `pup_max` | min / max |
| `pup_median` | median |
| `pup_slope` | linear fit (`np.polyfit`) over fixation order 0…n−1 |
| `pupbl_std` | `std` of the baseline-corrected series — NaN if `n < 2` |
| `pupbl_slope` | linear fit (`np.polyfit`) over fixation order 0…n−1 |

`pupbl_*` is NaN for the 16 participants whose pupil baselines are missing.

### `sac_` — measured saccades (7)

BEGAD records saccade events directly (EMS only infers them). Per-fixation
properties: `SaccadicAmplitude` (deg), `AbsoluteSaccadicDirection` (deg),
`RelativeSaccadicDirection` (deg). Final saccade event rows (which always have
`HasNulls == True`, so the HasNulls filter is not applied) are collapsed into
*runs* sharing the same landing fixation; `m` = number of runs.

| Feature | Formula | Notes |
|---|---|---|
| `sac_count` | `m` | 0 valid |
| `sac_dur_mean` / `_total` / `_max` | run durations | ms; NaN if `m = 0` (mean: `m < 2`) |
| `sac_amp_mean` / `_std` | per-fixation saccade amplitudes | deg |
| `sac_turn_std` | std of `min(reldir, 360 − reldir)` | deg, folded to [0, 180] |

## 4. Group comparison and feature screening

**Method**: one Kruskal–Wallis test per (task, feature) over the extracted
feature pool. Unit: one value per (participant, task); within a task the three
groups are independent samples. `p` = uncorrected p-value. Screening is
exploratory; multiple testing was not corrected at the selection stage.

### Selected 25 features (p < 0.1 in ≥ 3 tasks)

`processed_dataset/`; selection evidence in `processed_dataset/selection.csv`.

| Feature | n_tasks | Tasks (p<0.1) |
|---|---|---|
| `spa_fix_count` | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `spa_dispersion` | 3 | 3, 8, 9 |
| `spa_center_dist_std` | 5 | 1, 2, 4, 6, 9 |
| `spa_q2` | 3 | 0, 3, 10 |
| `geo_scanpath_len` | 4 | 0, 2, 4, 5 |
| `geo_sacc_amp_mean` | 5 | 3, 4, 5, 7, 8 |
| `geo_revisit_rate` | 6 | 0, 1, 2, 3, 4, 5 |
| `geo_nn_dist_mean` | 6 | 0, 2, 3, 4, 5, 9 |
| `tem_dur_total` | 4 | 0, 2, 4, 5 |
| `tem_velocity_mean` | 3 | 3, 5, 8 |
| `pup_mean` | 9 | 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_std` | 4 | 1, 2, 9, 10 |
| `pup_min` | 7 | 2, 5, 6, 7, 8, 9, 10 |
| `pup_max` | 7 | 0, 2, 5, 7, 8, 9, 10 |
| `pup_median` | 9 | 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_slope` | 3 | 1, 2, 10 |
| `pupbl_std` | 4 | 1, 2, 9, 10 |
| `pupbl_slope` | 3 | 1, 2, 10 |
| `sac_count` | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `sac_dur_mean` | 4 | 1, 3, 4, 9 |
| `sac_dur_total` | 9 | 0, 1, 2, 3, 4, 5, 6, 7, 10 |
| `sac_dur_max` | 4 | 0, 2, 7, 10 |
| `sac_amp_mean` | 5 | 3, 5, 7, 8, 10 |
| `sac_amp_std` | 3 | 3, 5, 10 |
| `sac_turn_std` | 3 | 0, 8, 9 |

**8 features** are significant (p < 0.1) in all three King–Devick cards
(tasks 3, 4, 5): `spa_fix_count`, `geo_sacc_amp_mean`, `geo_revisit_rate`,
`geo_nn_dist_mean`, `pup_mean`, `pup_median`, `sac_count`, `sac_dur_total`.

### Missingness

- 2 all-NaN pairs (`BEGCO-029`, `BEGMCI-019` × task 10): fewer than 2 clean
  fixations.
- `pupbl_std`, `pupbl_slope`: NaN for the 16 participants whose pupil
  baselines are missing in the raw CSVs (10.3% of rows); the `pup_` features
  are unaffected.
- All other features: 0.116% NaN (the 2 all-NaN rows).

## 5. Related files

| File | Content |
|---|---|
| `processed_dataset/features.csv`, `metadata.csv` | 25 features + metadata for 157 participants |
| `processed_dataset/selection.csv` | screening evidence (n_tasks, tasks per selected feature) |
| `processed_dataset/quality_report.txt` | build quality report (rows, all-NaN pairs, stage drops, NaN fractions) |
| `src/begad_norm/features/stimulus.py`, `features/saccade.py` | reference implementations |
| `docs/baseline/fold_split.md` | participant-level 5-fold split |

Regenerate features: `uv run python -m begad_norm.scripts.build_features`.
