# Processed dataset: 37-feature task-level values (RAW157)

Feature values for the 37-feature set (see `../feature_engineer/FEATURES_37_SUMMARY.md`), restricted to the
**RAW157** cohort (raw Tobii event exports). The four ROI64_T1235 features (D007, D010, D011, D015) are
excluded by design: only RAW157 features are computed. Of the 33 RAW157 features, 24 have valid values for
**all 157 subjects in all 4 tasks** and are kept; 9 features are dropped because their within-subject support
rules leave 1-6 subjects without a value in at least one task.

- Subjects: **157** baseline participants (CN 100 / MCI 41 / AD 16).
- Tasks: KD1, KD2, KD3, READING (one value per subject and task).
- Re-extracted 2026-10-28 with the original pipeline code (git commit `ffa36c7`, freeze
  `freeze_35a60cb67109`, config `configs/feature_discovery.yaml`) and the original dependency versions
  (Python 3.10, pandas 2.3.3, numpy 2.2.6, scipy 1.15.3). Spot-checked against the published table:
  6/6 p-values reproduced exactly (e.g. A002@KD3 CN-AD p = 9.5e-05; D023@KD3 CN-AD p = 0.000136).

## Files

| File | Layout |
|---|---|
| `processed_dataset_v1_wide.csv` | **V1 wide.** 1 row per subject (157 rows). Columns: `subject_id`, `group`, then 24 feature columns per task in the order KD1, KD2, KD3, READING, named `{feature_id}__{task}` (96 feature columns, 98 total). No missing values. |
| `processed_dataset_v2_long.csv` | **V2 long.** 4 rows per subject, one per task (628 rows). Columns: `subject_id`, `group`, `task`, then the 24 feature columns (`A002`...`D047`). No missing values. |
| `feature_columns.csv` | Kept features with `feature_id`, `feature_name`, `unit`. |
| `raw157_feature_values_long_full.csv` | Full RAW157 long table: all 33 features x 4 tasks with `value`, `measurement_status` and `n_events` (33,284 rows). Dropped features are present here with their per-task status (`observed` / `observed_zero` / `insufficient_within_subject_support`), so per-task versions can be rebuilt without re-running extraction. |

Both versions carry `subject_id` and `group` (label: CN / MCI / AD) for later cross-validation; use
`subject_id` as the split group so no participant appears in more than one fold.

## Kept features (24, complete 157/157 in every task)

A002 `total_fixation_time`, A003 `task_fixation_count`, A021 `task_saccade_count`, A023 `amplitude_median`,
A024 `amplitude_iqr`, A026 `saccade_duration_median`, A027 `saccade_duration_iqr`, A028
`saccade_mean_speed_median`, A029 `task_tiny_saccade_fraction`, A030 `task_long_saccade_fraction`, A031
`angular_scanpath_length`, A033 `normalized_screen_dispersion`, A035 `task_spatial_entropy`, A046
`task_absolute_pupil_median`, D018 `fixation_duration_quartile_dispersion`, D021 `interfixation_interval_median`,
D023 `gaze_event_span`, D024 `saccadic_time_fraction`, D028 `leftward_transition_fraction`, D029
`rightward_saccade_amplitude_median`, D039 `fixation_xy_spread_log_anisotropy`, D040
`vertical_progression_monotonicity`, D041 `off_media_fixation_time_fraction`, D047 `viewing_distance_iqr`.

Definitions, formulas and units: `../feature_engineer/FEATURES_37_SUMMARY.md` and
`../feature_engineer/feature_registry.csv`.

## Dropped features (9) and valid subject counts per task

A feature is kept only if it has valid values for all 157 subjects in all 4 tasks; these fail that rule
(valid subjects per task, KD1/KD2/KD3/READING):

| Feature | KD1 | KD2 | KD3 | READING | Reason |
|---|---|---|---|---|---|
| D017 fixation_duration_serial_correlation | 155 | 156 | 157 | 156 | needs >= 30 gap-free transitions |
| D022 fixation_onset_burstiness | 155 | 156 | 157 | 156 | needs >= 30 inter-onset intervals |
| D025 main_sequence_duration_slope | 156 | 157 | 157 | 157 | needs >= 10 matched saccades |
| D026 amplitude_duration_coupling | 156 | 157 | 157 | 157 | needs >= 10 matched saccades |
| D030 leftward_saccade_amplitude_median | 154 | 155 | 156 | 157 | needs >= 1 leftward saccade |
| D034 saccade_amplitude_serial_correlation | 153 | 154 | 152 | 156 | needs >= 30 consecutive transitions |
| D035 fixation_duration_outgoing_amplitude_coupling | 155 | 156 | 157 | 156 | needs >= 30 transitions |
| D036 incoming_amplitude_fixation_duration_coupling | 155 | 156 | 157 | 156 | needs >= 30 transitions |
| D037 post_leftward_fixation_duration_log_ratio | 151 | 155 | 152 | 156 | needs >= 5 leftward and rightward saccades |

If per-task versions of any dropped feature are wanted (e.g. D025 for KD2/KD3/READING, where all 157 subjects
are present), they can be recovered from the extraction long table with the per-task completeness rule applied
instead of the all-tasks rule.

## Provenance

- Extraction inputs: `dataset/20250305/*.csv` (raw Tobii event exports) and `dataset/participants_20250305.csv`
  (roster with group codes 0/1/2 -> CN/MCI/AD).
- Extraction code: `src/begad_norm/feature_discovery/extract_macro.py` (git commit `ffa36c7`, recoverable from
  git history); values are `observed`/`observed_zero` measurements only.
- Re-extraction run directory: `/tmp/begad_run_260` (scratch; `macro_features_long.csv` holds the long table).
- The two CSVs are complete-case by construction: no NaN in any feature column.
