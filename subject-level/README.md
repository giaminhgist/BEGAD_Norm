# BEGAD_Norm

Eye-tracking feature extraction and group comparison (CO / MCI / AD) for the
BEGAD battery.

## 1. Dataset

- **Session**: 2025-03-05, export tag `BEGAD_ETRA25`.
- **Raw exports**: 205 CSV files in `dataset/20250305/` and
  `dataset_v2/20250305/` (export schema with ADCSpx/MCSpx gaze coordinates,
  fixation/saccade event indices, pupil size, and per-row `Task`, `MediaName`,
  `Group`, `IsFollowup` columns).
- **Cohort filter** for the processed dataset: first-visit participants in
  groups CO / MCI / AD only. Excluded: 36 follow-up visits, 10 NPH
  participants (Group 3), 2 recordings with Group −11.
- **Participants**: 157 — CO 100, MCI 41, AD 16
  (`processed_dataset/metadata.csv`).
- **Tasks**: 11 per participant (0–10) → `processed_dataset/features.csv` holds
  1727 rows (157 × 11; 2 all-NaN rows: `BEGCO-029` and `BEGMCI-019`, task 10).

### Protocol tasks

Media names are read from the `MediaName` column of the raw exports; geometry
from `TASK_GEOMETRY` in `src/begad_norm/config.py`.

| Task | Media (raw `MediaName` column) | Geometry on 1920×1080 | Description |
|---|---|---|---|
| 0 | `Instruction Element` | 1920×1080 (full screen) | Instruction screen |
| 1 | `KD_demo_highres.png` | 748×564 | King–Devick demo card (practice) |
| 2 | `Instruction Element 2` | 1920×1080 (full screen) | Instruction screen |
| 3 | `KD_1_highres.png` | 746×566 | King–Devick card 1 |
| 4 | `KD_2_highres.png` | 744×525 | King–Devick card 2 |
| 5 | `KD_3_highres.png` | 737×508 | King–Devick card 3 |
| 6 | `Instruction Element` | 1920×1080 (full screen) | Instruction screen |
| 7 | `lukeminen_harj.png` | 950×417 | Reading practice |
| 8 | `Instruction Element` | 1920×1080 (full screen) | Instruction screen |
| 9 | `lukeminen_1.png` | 918×917 | Reading task |
| 10 | `Instruction Element 2` | 1920×1080 (full screen) | Closing instruction screen |

The King–Devick / reading interpretation follows the protocol notebook
(`notebooks/Analysisbase_for_Ali.ipynb` labels tasks 3–5 "KD tasks" and task 9
"reading task") and the stimulus file names ("lukeminen" = reading in Finnish).
Tasks 0, 2, 6, 8, 10 are full-screen instruction screens, not test stimuli.

## 2. Features (25)

25 features are computed per (participant, task) from the clean fixations
(see `docs/baseline/eda_summary.md` for the cleaning pipeline, formulas and units;
Vietnamese version: `docs/baseline/eda_summary_vi.md`). Groups by prefix:

| Prefix | n | Meaning |
|---|---|---|
| `spa_` | 4 | spatial position & dispersion |
| `geo_` | 4 | scanpath geometry |
| `tem_` | 2 | temporal dynamics |
| `pup_` | 6 | pupil size |
| `pupbl_` | 2 | pupil baseline |
| `sac_` | 7 | measured saccades |

Full list:

| Group | Feature |
|---|---|
| spa — spatial position & dispersion | `spa_fix_count` |
| spa — spatial position & dispersion | `spa_dispersion` |
| spa — spatial position & dispersion | `spa_center_dist_std` |
| spa — spatial position & dispersion | `spa_q2` |
| geo — scanpath geometry | `geo_scanpath_len` |
| geo — scanpath geometry | `geo_sacc_amp_mean` |
| geo — scanpath geometry | `geo_revisit_rate` |
| geo — scanpath geometry | `geo_nn_dist_mean` |
| tem — temporal dynamics | `tem_dur_total` |
| tem — temporal dynamics | `tem_velocity_mean` |
| pup — pupil size | `pup_mean` |
| pup — pupil size | `pup_std` |
| pup — pupil size | `pup_min` |
| pup — pupil size | `pup_max` |
| pup — pupil size | `pup_median` |
| pup — pupil size | `pup_slope` |
| pupbl — pupil baseline | `pupbl_std` |
| pupbl — pupil baseline | `pupbl_slope` |
| sac — measured saccades | `sac_count` |
| sac — measured saccades | `sac_dur_mean` |
| sac — measured saccades | `sac_dur_total` |
| sac — measured saccades | `sac_dur_max` |
| sac — measured saccades | `sac_amp_mean` |
| sac — measured saccades | `sac_amp_std` |
| sac — measured saccades | `sac_turn_std` |

## 3. Feature selection

The 25 features were screened from the extracted feature pool with
Kruskal–Wallis tests per (feature, task): features with p < 0.1 (uncorrected)
in **at least 3 of the 11 tasks** were retained. Screening was exploratory and
multiple testing was not corrected at the selection stage. Evidence:
`processed_dataset/selection.csv`.

| Feature | n_tasks (p<0.1) | Tasks |
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

Among these, **8 features** are significant (p < 0.1) in all three King–Devick
cards (tasks 3, 4, 5): `spa_fix_count`, `geo_sacc_amp_mean`, `geo_revisit_rate`, `geo_nn_dist_mean`, `pup_mean`, `pup_median`, `sac_count`, `sac_dur_total`.

## 4. Baseline experiments

Classical baselines (logistic regression, SVM RBF/linear, FNN, QDA, GNB) on
one sample per participant (25 features × 11 tasks = 275 columns), 5-fold
participant-level CV × 20 seeds, with class-imbalance handling (balanced
class weights / inverse-frequency priors, fixed 0.5 decision threshold):

- Protocol and results: `docs/baseline/baseline_balanced.md`
  (Vietnamese: `docs/baseline/baseline_balanced_vi.md`).
- Machine-readable outputs: `outputs/baseline_balanced/` (results,
  missingness, provenance, run log).
- Re-run: `uv run python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml`.

## Repository layout

| Path | Content |
|---|---|
| `dataset/`, `dataset_v2/` | raw eye-tracker exports (2025-03-05) |
| `processed_dataset/` | the processed dataset (25 features) + metadata for 157 participants + selection evidence |
| `fold1/` | participant-level 5-fold assignment (CSV) |
| `fold/` | earlier 4-fold reference split (CSV) |
| `docs/baseline/` | project documentation (EN + VI): EDA summary, fold split, balanced baseline |
| `outputs/` | experiment outputs (baseline results, provenance, run log) |
| `src/begad_norm/` | feature-extraction code, config, and baseline experiments |
| `tests/` | unit tests (config, raw data, features) |

## Notes

- The p-values used for screening are uncorrected Kruskal–Wallis p-values;
  the 25 features were screened on the same cohort, so downstream CV
  performance carries in-sample selection bias.
- Feature values on instruction-screen tasks (0, 2, 6, 8, 10) describe
  instruction viewing, not the test stimuli.
