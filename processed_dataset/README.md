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

## 33-feature variant (157 subjects, missing cells imputed at model time)

The 9 dropped features can be used if unsupported cells are imputed instead of requiring completeness.
`processed_dataset_33feat_wide.csv` keeps **all 157 subjects**; the 51 cells a subject cannot support
(insufficient within-subject support, all in the 9 added features) are left as NaN. 20 subjects have at
least one such cell (CN 10 / MCI 6 / AD 4: BEG-202, BEG-220, BEG-231, BEG-234, BEG-247, BEG-260, BEG-275,
BEG-277, BEG-282, BEG-283, BEGAD-002, BEGAD-005, BEGAD-007, BEGAD-019, BEGCO-004, BEGCO-020, BEGCO-029,
BEGCO-032, BEGCO-035, BEGMCI-006).

Imputation is done at model time, not baked into the tables: the baseline pipeline fills each missing
cell with the **per-column median fitted on the training split of the fold only** (`SimpleImputer`,
`strategy="median"` in `src/baseline/run_baselines.py`), so no validation or held-out value leaks into
training. With 51/20724 missing cells (0.25%) the choice of imputer is expected to matter little;
alternatives (feature-specific zero-fill for D030, missingness indicator columns) are possible and
would require a design change.

| File | Layout |
|---|---|
| `processed_dataset_33feat_wide.csv` | 1 row per subject (157 rows). Columns: `subject_id`, `group`, then 33 feature columns per task in the order KD1, KD2, KD3, READING, named `{feature_id}__{task}` (132 feature columns, 134 total). 51 NaN cells, all in the 9 added features. |
| `processed_dataset_33feat_long.csv` | 4 rows per subject, one per task (628 rows). Columns: `subject_id`, `group`, `task`, then the 33 feature columns (`A002`...`D047`). 51 NaN cells. |
| `feature_columns_33feat.csv` | The 33 features with `feature_id`, `feature_name`, `unit`. |
| `make_33feat_dataset.py` | Rebuild script. Reads `raw157_feature_values_long_full.csv`, keeps every subject, writes NaN for `insufficient_within_subject_support` cells (asserts 51, all in the 9 added features), and cross-checks the 24 shared feature columns against `processed_dataset_v1_wide.csv` (1e-9 tolerance; the two files serialize floats with different precision, differences are <= 7e-15). |

Because no subject is dropped, the 33-feature baseline reuses the **same splits as the 24-feature
baseline**: `splits/cv_fold_assignments.csv`, `splits/heldout_ad_subjects.csv` (4 held-out AD:
BEGAD-001, BEGAD-005, BEGAD-006, BEGAD-016). Config: `src/baseline/config_33feat.yaml`
(`output/baseline_33feat/`).

## Provenance

- Extraction inputs: `dataset/20250305/*.csv` (raw Tobii event exports) and `dataset/participants_20250305.csv`
  (roster with group codes 0/1/2 -> CN/MCI/AD).
- Extraction code: `src/begad_norm/feature_discovery/extract_macro.py` (git commit `ffa36c7`, recoverable from
  git history); values are `observed`/`observed_zero` measurements only.
- Re-extraction run directory: `/tmp/begad_run_260` (scratch; `macro_features_long.csv` holds the long table).
- The two CSVs are complete-case by construction: no NaN in any feature column.
