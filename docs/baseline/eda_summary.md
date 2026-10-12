# BEGAD EDA Summary — dataset, features, and group comparison

> Synthesized 2026-10-12 from the former `EDA/` folder (feature dictionary,
> group-comparison summary, and per-task statistics) during the repository
> cleanup. The EDA figures and the comparison notebook were removed; they are
> recoverable from git history. The full per-task statistics table is
> preserved at `processed_dataset/group_comparison_stats_per_task.csv`.

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

## 3. Feature dictionary (72 features)

72 features per (participant, task) in `processed_dataset/features.csv`
(1727 rows = 157 × 11; 2 all-NaN rows: `BEGCO-029`, `BEGMCI-019` × task 10).
Notation: `n` = clean fixations; `x, y` = fixation centers (px, media-relative);
`dur` = fixation duration (ms); `pup` = binocular mean pupil (mm); `W, H` =
task media size. Sample SD (`ddof=1`); pairs with `n < 2` get an all-NaN row
(`MIN_FIX = 2`).

### `spa_` — spatial position & dispersion (17)

| Feature | Formula | Notes |
|---|---|---|
| `spa_fix_count` | `n` | never NaN |
| `spa_mean_x`, `spa_mean_y` | `mean(x)`, `mean(y)` | px |
| `spa_std_x`, `spa_std_y` | `std(x)`, `std(y)` | NaN if `n < 2` |
| `spa_dispersion` | `mean(sqrt((x−mean(x))² + (y−mean(y))²))` | px, distance to centroid |
| `spa_bbox_area` | `(max(x)−min(x)) × (max(y)−min(y))` | px² |
| `spa_center_dist_mean` / `_std` | distance of fixations to media center `(W/2, H/2)` | px |
| `spa_center_frac` | fraction inside central quarter (`|x−cx|<W/4`, `|y−cy|<H/4`) | |
| `spa_q1…spa_q4` | quadrant fractions around the media center | sum = 1 |
| `spa_entropy` | Shannon entropy of 8×6 grid histogram / `log(48)` | 0 = one cell, 1 = uniform |
| `spa_max_grid_frac` | peak cell density of the 8×6 histogram | |
| `spa_skew_x` | `mean((x−mean(x))³) / std(x)³` | NaN if `n ≤ 2` or `std = 0` |

### `geo_` — scanpath geometry (10)

Inferred saccades between consecutive fixations: `dx = diff(x)`,
`amp = sqrt(dx² + dy²)`.

| Feature | Formula | Notes |
|---|---|---|
| `geo_scanpath_len` | `sum(amp)` | px; NaN if `n < 2` |
| `geo_sacc_amp_mean` / `_std` / `_max` | mean / std / max of `amp` | px |
| `geo_dx_mean`, `geo_dy_mean` | `mean(dx)`, `mean(dy)` | net drift |
| `geo_angle_var` | `1 − R`, `R = sqrt(mean(cos θ)² + mean(sin θ)²)`, `θ = atan2(dy, dx)` | 0 = one direction, 1 = uniform |
| `geo_revisit_rate` | fraction of fixations with a later fixation within 60 px | forward-looking (`triu`, k=1) |
| `geo_nn_dist_mean` | mean distance to nearest other fixation | px |
| `geo_hull_area` | convex hull area of `(x, y)` | px²; NaN if `n < 4` or degenerate |

### `tem_` — temporal (11)

Half-dwell IFI approximation (no onset timestamps): `half_gap = (dur[:-1] + dur[1:]) / 2`.

| Feature | Formula | Notes |
|---|---|---|
| `tem_dur_mean` / `_std` / `_max` | mean / std / max of `dur` | ms |
| `tem_dur_total` | `sum(dur)` | ms |
| `tem_first_dur`, `tem_last_dur` | first / last fixation duration in time order | |
| `tem_ifi_mean` / `_std` | mean / std of `half_gap` | ms; NaN if `n < 2` |
| `tem_velocity_mean` | `mean(amp / half_gap)` over finite terms | px/ms |
| `tem_fix_rate` | `n / total_time × 1000`, `total_time = sum(dur) + tem_ifi_mean × (n−1)` | 1/s |
| `tem_trans_entropy` | entropy of transition start points on a 4×3 grid / `log(12)` | NaN if `n < 2` |

### `pup_`, `pupbl_`, `pupdiff_` — pupil (7 + 7 + 7)

Same 7 formulas on three series per pair:
`pup_` = binocular mean `(PupilLeft + PupilRight) / 2`;
`pupbl_` = baseline-corrected binocular mean (constant per-recording
baselines); `pupdiff_` = left − right difference.

| Feature | Formula |
|---|---|
| `<pre>_mean` | `mean(series)` |
| `<pre>_std` | `std(series)` — NaN if `n < 2` |
| `<pre>_min` / `<pre>_max` | min / max |
| `<pre>_median` | median |
| `<pre>_slope` | linear fit (`np.polyfit`) over fixation order 0…n−1 |
| `<pre>_first_last_diff` | `series[-1] − series[0]` |

`pupbl_*` is NaN for the 16 participants whose pupil baselines are missing.

### `sac_` — measured saccades (13)

BEGAD records saccade events directly (EMS only infers them). Per-fixation
properties: `SaccadicAmplitude` (deg), `AbsoluteSaccadicDirection` (deg),
`RelativeSaccadicDirection` (deg). Final saccade event rows (which always have
`HasNulls == True`, so the HasNulls filter is not applied) are collapsed into
*runs* sharing the same landing fixation; `m` = number of runs.

| Feature | Formula | Notes |
|---|---|---|
| `sac_count` | `m` | 0 valid |
| `sac_rate` | `m / total_time × 1000` | events/s |
| `sac_dur_mean` / `_std` / `_total` / `_max` | run durations | ms; NaN if `m = 0` (std: `m < 2`) |
| `sac_amp_mean` / `_std` / `_max` | per-fixation saccade amplitudes | deg |
| `sac_absdir_mean` | circular mean of absolute directions | deg, 0–360 |
| `sac_absdir_var` | `1 − R` of the circular mean | 0 = same direction |
| `sac_turn_mean` / `_std` | mean / std of `min(reldir, 360 − reldir)` | deg, folded to [0, 180] |

## 4. Group comparison and feature screening

**Method**: one Kruskal–Wallis test per (task, feature): 11 tasks × 72
features = **792 tests**. Unit: one value per (participant, task); within a
task the three groups are independent samples. `p` = uncorrected p-value;
`q_bh` = Benjamini–Hochberg over all 792 tests. Screening is exploratory —
multiple testing is expected.

### Selected 25 features (p < 0.1 in ≥ 3 tasks)

`finalized_dataset/`; selection evidence in
`finalized_dataset/selection.csv`.

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

### Extended 39 features (p < 0.2 in ≥ 3 tasks)

`processed_dataset_v3/`; selection evidence in
`processed_dataset_v3/selection.csv`. The 25 features above are a subset.
Under the global null, ~28 of 72 features would meet this criterion by chance
(792 tests), so this list is only a broad candidate set. Notable additions
(all 11 tasks at p < 0.2): `pup_max`, `pup_mean`, `pup_median`;
`pup_min` (10); `sac_dur_total` (9); `geo_nn_dist_mean`, `sac_amp_mean` (8);
`sac_count`, `sac_dur_std`, `sac_turn_std`, `spa_fix_count` (7); plus 20 more
features with 3–6 tasks — full list in `processed_dataset_v3/selection.csv`.

### Missingness

- 2 all-NaN pairs (`BEGCO-029`, `BEGMCI-019` × task 10): fewer than 2 clean
  fixations.
- `pupbl_*`: NaN for the 16 participants whose pupil baselines are missing in
  the raw CSVs; `pup_` and `pupdiff_` are unaffected.
- All other features: < 1% NaN over the 1725 non-empty rows.

## 5. Related files

| File | Content |
|---|---|
| `processed_dataset/group_comparison_stats_per_task.csv` | full per-(task, feature) stats (H, p, q_bh, group means, n) |
| `processed_dataset/features.csv`, `metadata.csv` | 72 features + metadata for 157 participants |
| `finalized_dataset/`, `processed_dataset_v3/` | final working set (25 features) and extended subset (39) |
| `src/begad_norm/features/stimulus.py`, `features/saccade.py` | reference implementations |
| `docs/baseline/fold_split.md` | participant-level 4-fold split |

Regenerate features: `uv run python -m begad_norm.scripts.build_features`.
