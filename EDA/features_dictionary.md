# BEGAD Feature Dictionary — dataset_v2, per-(participant, task)

This document describes the 72 features in `processed_dataset/features.csv`
(1727 rows = 157 participants × 11 tasks; 2 all-NaN rows: `BEGCO-029` and
`BEGMCI-019`, task 10). Rows are indexed by `participant_id` (RealTag) and
`task` (0–10). Features are grouped as in the table: `spa_` (17), `geo_` (10),
`tem_` (11), `pup_` (7), `pupbl_` (7), `pupdiff_` (7), `sac_` (13).

## Data sources and cleaning

Every feature is computed from the **clean fixations** of one (participant, task)
pair, extracted from `dataset_v2/20250305/BEGAD_ETRA25_<RealTag>.csv` by
`src/begad_norm/rawdata.py`:

1. Rows kept: `Final == True`, `HasNulls == False`, `GazeEventType == "Fixation"`,
   `FixationIndex > 0` (cohort: `Group ∈ {0,1,2}`, `IsFollowup == 0`, roster of 157
   participants from `dataset_v2/participants_20250305.csv`).
2. EMS duration window: `40 ms < GazeEventDuration ≤ 2000 ms` (drops 2 rows total).
3. Per-task media bounds on `FixationPointX/Y (MCSpx)` (inclusive, drops 0 rows).
4. Pupil outliers: per (participant, task), drop rows where the binocular mean
   pupil deviates from the group median by more than `4 × max(std, 1e-6)`
   (EMS-exact; drops 381 rows total).

Net clean fixations: 116,274 (116,657 after step 1, minus 2 duration + 381 pupil).
Decimal-comma columns (`"3,98"`) are parsed to float (`str.replace(',', '.')`).
Fixations are ordered by `RecordingTimestamp` within each task.

Notation: `n` = number of clean fixations in the pair; `x, y` = fixation centers
(`FixationPointX/Y (MCSpx)`, px, media-relative); `dur` = fixation duration
(`GazeEventDuration`, ms); `pup` = binocular mean pupil
`(PupilLeft + PupilRight) / 2` (mm). `W, H` = task media size from
`TASK_GEOMETRY` (config.py); center `(cx, cy) = (W/2, H/2)`. All stds use
`ddof=1` (sample SD). Pairs with `n < 2` get an all-NaN row (EMS `MIN_FIX = 2`).
`NaN` propagates wherever an input is undefined (see the NaN column).

## `spa_` — spatial position & dispersion (17)

Reference implementation: `compute_stimulus_features` section A–B in
`src/begad_norm/features/stimulus.py` (verbatim port of EMS-Projects).

| Feature | Formula | Source (dataset_v2 columns) | NaN / notes |
|---|---|---|---|
| `spa_fix_count` | `n` | rows after cleaning | never NaN |
| `spa_mean_x` | `mean(x)` | `FixationPointX (MCSpx)` | px |
| `spa_mean_y` | `mean(y)` | `FixationPointY (MCSpx)` | px |
| `spa_std_x` | `std(x, ddof=1)` | same | NaN if `n < 2` |
| `spa_std_y` | `std(y, ddof=1)` | same | NaN if `n < 2` |
| `spa_dispersion` | `mean(sqrt((x−mean(x))² + (y−mean(y))²))` | both fixation-point columns | mean distance to the centroid, px |
| `spa_bbox_area` | `(max(x)−min(x)) × (max(y)−min(y))` | both | px²; 0 for a single point |
| `spa_center_dist_mean` | `mean(sqrt((x−cx)² + (y−cy)²))` | fixation points + `TASK_GEOMETRY` | px to media center |
| `spa_center_dist_std` | `std(center distances, ddof=1)` | same | NaN if `n < 2` |
| `spa_center_frac` | `mean(\|x−cx\| < W/4 AND \|y−cy\| < H/4)` | same | fraction inside the central quarter of the media |
| `spa_q1…spa_q4` | quadrant fractions: q1 `x<cx & y<cy`; q2 `x≥cx & y<cy`; q3 `x<cx & y≥cy`; q4 `x≥cx & y≥cy` | same | sum = 1 |
| `spa_entropy` | Shannon entropy of the 8×6 grid histogram over `(0,W)×(0,H)`, divided by `log(48)` | same | 0 = one cell, 1 = uniform |
| `spa_max_grid_frac` | `max(8×6 histogram) / n` | same | peak cell density |
| `spa_skew_x` | `mean((x−mean(x))³) / std(x)³` (no bias correction) | `FixationPointX (MCSpx)` | NaN if `n ≤ 2` or `std == 0` |

## `geo_` — scanpath geometry (10)

Consecutive-fixation transitions: `dx = diff(x)`, `dy = diff(y)`,
`amp = sqrt(dx² + dy²)` — the *inferred* saccades between consecutive fixations
(EMS convention, kept even though BEGAD also records measured saccades — the
measured ones live in `sac_`). Reference: `compute_stimulus_features` section C.

| Feature | Formula | Source | NaN / notes |
|---|---|---|---|
| `geo_scanpath_len` | `sum(amp)` | fixation points | px; NaN if `n < 2` |
| `geo_sacc_amp_mean` | `mean(amp)` | same | inferred saccade amplitude, px |
| `geo_sacc_amp_std` | `std(amp, ddof=1)` | same | NaN if `n < 2` |
| `geo_sacc_amp_max` | `max(amp)` | same | |
| `geo_dx_mean` | `mean(dx)` | same | net horizontal drift, px |
| `geo_dy_mean` | `mean(dy)` | same | net vertical drift, px |
| `geo_angle_var` | `1 − R`, `R = sqrt(mean(cos θ)² + mean(sin θ)²)`, `θ = atan2(dy, dx)` | same | 0 = all one direction, 1 = uniform; zero-length steps count as `θ=0` (straight) |
| `geo_revisit_rate` | fraction of fixations with a *later* fixation within 60 px (forward-looking, `triu(dist<60, k=1)`) | same, `REVISIT_RADIUS = 60` px | last fixation never counted but stays in denominator |
| `geo_nn_dist_mean` | mean over fixations of distance to the nearest *other* fixation | same | px |
| `geo_hull_area` | area of `scipy.spatial.ConvexHull(x, y)` | same | px²; NaN if `n < 4` or degenerate (collinear) |

## `tem_` — temporal (11)

Half-dwell IFI approximation (onset timestamps are not recorded):
`half_gap = (dur[:-1] + dur[1:]) / 2`. Reference: `compute_stimulus_features`
section D.

| Feature | Formula | Source | NaN / notes |
|---|---|---|---|
| `tem_dur_mean` | `mean(dur)` | `GazeEventDuration` (ms) | |
| `tem_dur_std` | `std(dur, ddof=1)` | same | NaN if `n < 2` |
| `tem_dur_total` | `sum(dur)` | same | ms |
| `tem_dur_max` | `max(dur)` | same | |
| `tem_first_dur` | `dur[0]` (first fixation in time order) | same, order by `RecordingTimestamp` | |
| `tem_last_dur` | `dur[-1]` | same | |
| `tem_ifi_mean` | `mean(half_gap)` | `GazeEventDuration` pairs | ms; NaN if `n < 2` |
| `tem_ifi_std` | `std(half_gap, ddof=1)` | same | NaN if `n < 2` |
| `tem_velocity_mean` | `mean(amp / half_gap)` over finite terms only | fixation points + durations | px/ms of the inferred saccades; NaN if no finite term |
| `tem_fix_rate` | `n / total_time × 1000`, `total_time = sum(dur) + tem_ifi_mean × (n−1)` | durations | 1/s; NaN if `total_time ≤ 0` |
| `tem_trans_entropy` | Shannon entropy of transition *starting* points `(x[:-1], y[:-1])` on a 4×3 grid, divided by `log(12)` | fixation points | NaN if `n < 2`; 0 if all starts share one cell |

## `pup_`, `pupbl_`, `pupdiff_` — pupil (7 + 7 + 7)

The same 7 formulas are applied to three pupil series per pair:

- `pup_`: raw binocular mean `pup_i = (PupilLeft_i + PupilRight_i) / 2` (mm).
- `pupbl_`: baseline-corrected binocular mean
  `((PupilLeft_i − PupilLeftBaseline) + (PupilRight_i − PupilRightBaseline)) / 2`
  (baselines are constant per recording).
- `pupdiff_`: raw left-minus-right difference `PupilLeft_i − PupilRight_i` (mm).

| Feature | Formula | Source | NaN / notes |
|---|---|---|---|
| `<pre>_mean` | `mean(series)` | `PupilLeft`, `PupilRight` (+ `PupilLeftBaseline`, `PupilRightBaseline` for `pupbl_`) | mm; `pupbl_*` is NaN for the 16 participants whose baseline is missing in the raw data |
| `<pre>_std` | `std(series, ddof=1)` | same | NaN if `n < 2` |
| `<pre>_min` / `<pre>_max` | `min` / `max` of the series | same | |
| `<pre>_median` | `median(series)` | same | |
| `<pre>_slope` | linear fit (`np.polyfit`) of the series over fixation order 0…n−1 | same, order by `RecordingTimestamp` | per fixation index; NaN if `n < 2` |
| `<pre>_first_last_diff` | `series[-1] − series[0]` | same | NaN if `n < 2` |

`<pre>` ∈ {`pup`, `pupbl`, `pupdiff`} — e.g. `pup_mean`, `pupbl_std`,
`pupdiff_first_last_diff`. Reference: `compute_stimulus_features` section E.

## `sac_` — measured saccades (13)

New group (BEGAD has measured saccade events; EMS only infers them). Sources:
(a) saccade properties attached to each clean fixation — `SaccadicAmplitude`
(deg), `AbsoluteSaccadicDirection` (deg, 0–360), `RelativeSaccadicDirection`
(deg, turn relative to the previous saccade direction); (b) `Final` saccade
event rows (they always have `HasNulls == True`, so the HasNulls filter is NOT
applied to them) collapsed into *runs*: consecutive saccade rows in time order
sharing the same landing fixation (first clean fixation with timestamp greater
than the saccade timestamp). `run_dur` = max `GazeEventDuration` in the run.
`m` = number of runs; `total_time` = EMS total time (see `tem_fix_rate`).

Reference: `src/begad_norm/features/saccade.py` and
`extract_saccade_runs` in `rawdata.py`.

| Feature | Formula | Source | NaN / notes |
|---|---|---|---|
| `sac_count` | `m` | saccade runs | 0 is valid |
| `sac_rate` | `m / total_time × 1000` | runs + durations | events/s; NaN if `total_time ≤ 0` |
| `sac_dur_mean` | `mean(run_durs)` | `GazeEventDuration` of Final saccade rows | ms; NaN if `m = 0` |
| `sac_dur_std` | `std(run_durs, ddof=1)` | same | NaN if `m < 2` |
| `sac_dur_total` | `sum(run_durs)` | same | NaN if `m = 0` |
| `sac_dur_max` | `max(run_durs)` | same | NaN if `m = 0` |
| `sac_amp_mean` | `mean(amp)` over the per-fixation amplitudes | `SaccadicAmplitude` (deg, decimal-comma) | amplitude of the saccade landing on each fixation |
| `sac_amp_std` | `std(amp, ddof=1)` | same | NaN if `n < 2` |
| `sac_amp_max` | `max(amp)` | same | |
| `sac_absdir_mean` | circular mean of absolute directions: `deg(atan2(mean(sin θ), mean(cos θ))) mod 360` | `AbsoluteSaccadicDirection` (deg) | values within 1e-12 of 360 snap to 0 |
| `sac_absdir_var` | `1 − R`, `R = sqrt(mean(cos θ)² + mean(sin θ)²)` | same | 0 = all same direction; NaN if `n < 2` |
| `sac_turn_mean` | `mean(min(reldir, 360 − reldir))` | `RelativeSaccadicDirection` (deg, folded to [0, 180]) | mean turn magnitude between consecutive saccades |
| `sac_turn_std` | `std(turn, ddof=1)` | same | NaN if `n < 2` |

## Output files

- `processed_dataset/features.csv` — 1727 rows × 74 columns (`participant_id`,
  `task`, 72 features; order = `spa_` 17, `geo_` 10, `tem_` 11, `pup_` 7,
  `pupbl_` 7, `pupdiff_` 7, `sac_` 13).
- `processed_dataset/metadata.csv` — per-participant group, file, fixations,
  tasks present, missing-baseline flag.
- `processed_dataset/feature_names.txt`, `quality_report.txt`.
- Regenerate: `uv run python -m begad_norm.scripts.build_features`.
- Tests: `uv run python -m unittest discover -s tests` (34 tests, incl. a
  verbatim-EMS probe proving the 45-feature port and an end-to-end golden test).

## Missingness

- 2 all-NaN pairs (`BEGCO-029`, `BEGMCI-019` × task 10): fewer than 2 clean
  fixations.
- `pupbl_*`: NaN for the 16 participants whose pupil baselines are missing in
  the raw CSVs (documented in `quality_report.txt`); `pup_` and `pupdiff_` are
  unaffected.
- All other features: < 1% NaN over the 1725 non-empty rows.
