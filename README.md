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

## 2. Features (72)

72 features are computed per (participant, task) from the clean fixations
(see `EDA/features_dictionary.md` for formulas and units; Vietnamese version:
`EDA/features_dictionary_vi.md`). Groups by prefix:

| Prefix | n | Meaning |
|---|---|---|
| `spa_` | 17 | spatial position & dispersion |
| `geo_` | 10 | scanpath geometry |
| `tem_` | 11 | temporal dynamics |
| `pup_` | 7 | pupil size |
| `pupbl_` | 7 | pupil baseline |
| `pupdiff_` | 7 | pupil difference |
| `sac_` | 13 | measured saccades |

Full list:

| Group | Feature |
|---|---|
| spa — spatial position & dispersion | `spa_fix_count` |
| spa — spatial position & dispersion | `spa_mean_x` |
| spa — spatial position & dispersion | `spa_mean_y` |
| spa — spatial position & dispersion | `spa_std_x` |
| spa — spatial position & dispersion | `spa_std_y` |
| spa — spatial position & dispersion | `spa_dispersion` |
| spa — spatial position & dispersion | `spa_bbox_area` |
| spa — spatial position & dispersion | `spa_center_dist_mean` |
| spa — spatial position & dispersion | `spa_center_dist_std` |
| spa — spatial position & dispersion | `spa_center_frac` |
| spa — spatial position & dispersion | `spa_q1` |
| spa — spatial position & dispersion | `spa_q2` |
| spa — spatial position & dispersion | `spa_q3` |
| spa — spatial position & dispersion | `spa_q4` |
| spa — spatial position & dispersion | `spa_entropy` |
| spa — spatial position & dispersion | `spa_max_grid_frac` |
| spa — spatial position & dispersion | `spa_skew_x` |
| geo — scanpath geometry | `geo_scanpath_len` |
| geo — scanpath geometry | `geo_sacc_amp_mean` |
| geo — scanpath geometry | `geo_sacc_amp_std` |
| geo — scanpath geometry | `geo_sacc_amp_max` |
| geo — scanpath geometry | `geo_dx_mean` |
| geo — scanpath geometry | `geo_dy_mean` |
| geo — scanpath geometry | `geo_angle_var` |
| geo — scanpath geometry | `geo_revisit_rate` |
| geo — scanpath geometry | `geo_nn_dist_mean` |
| geo — scanpath geometry | `geo_hull_area` |
| tem — temporal dynamics | `tem_dur_mean` |
| tem — temporal dynamics | `tem_dur_std` |
| tem — temporal dynamics | `tem_dur_total` |
| tem — temporal dynamics | `tem_dur_max` |
| tem — temporal dynamics | `tem_first_dur` |
| tem — temporal dynamics | `tem_last_dur` |
| tem — temporal dynamics | `tem_ifi_mean` |
| tem — temporal dynamics | `tem_ifi_std` |
| tem — temporal dynamics | `tem_velocity_mean` |
| tem — temporal dynamics | `tem_fix_rate` |
| tem — temporal dynamics | `tem_trans_entropy` |
| pup — pupil size | `pup_mean` |
| pup — pupil size | `pup_std` |
| pup — pupil size | `pup_min` |
| pup — pupil size | `pup_max` |
| pup — pupil size | `pup_median` |
| pup — pupil size | `pup_slope` |
| pup — pupil size | `pup_first_last_diff` |
| pupbl — pupil baseline | `pupbl_mean` |
| pupbl — pupil baseline | `pupbl_std` |
| pupbl — pupil baseline | `pupbl_min` |
| pupbl — pupil baseline | `pupbl_max` |
| pupbl — pupil baseline | `pupbl_median` |
| pupbl — pupil baseline | `pupbl_slope` |
| pupbl — pupil baseline | `pupbl_first_last_diff` |
| pupdiff — pupil difference | `pupdiff_mean` |
| pupdiff — pupil difference | `pupdiff_std` |
| pupdiff — pupil difference | `pupdiff_min` |
| pupdiff — pupil difference | `pupdiff_max` |
| pupdiff — pupil difference | `pupdiff_median` |
| pupdiff — pupil difference | `pupdiff_slope` |
| pupdiff — pupil difference | `pupdiff_first_last_diff` |
| sac — measured saccades | `sac_count` |
| sac — measured saccades | `sac_rate` |
| sac — measured saccades | `sac_dur_mean` |
| sac — measured saccades | `sac_dur_std` |
| sac — measured saccades | `sac_dur_total` |
| sac — measured saccades | `sac_dur_max` |
| sac — measured saccades | `sac_amp_mean` |
| sac — measured saccades | `sac_amp_std` |
| sac — measured saccades | `sac_amp_max` |
| sac — measured saccades | `sac_absdir_mean` |
| sac — measured saccades | `sac_absdir_var` |
| sac — measured saccades | `sac_turn_mean` |
| sac — measured saccades | `sac_turn_std` |

## 3. Selected features (25)

Screening from `EDA/group_comparison_stats_per_task.csv`: features with
Kruskal–Wallis p < 0.1 (uncorrected) in **at least 3 of the 11 tasks**.
Exploratory only — 792 tests were run, so multiple testing is expected.

| Feature | n_tasks (p<0.1) | Tasks |
|---|---|---|
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

- Full statistics (H, p, q_bh, group means) for all 72 features × 11 tasks:
  `EDA/comparison_summary.md` (Vietnamese: `EDA/comparison_summary_vi.md`).
- Dataset restricted to the 25 selected features: `processed_dataset_v2/`.

## 4. Extended selection (39 features, p < 0.2)

A looser screening threshold: features with Kruskal–Wallis p < 0.2
(uncorrected) in **at least 3 of the 11 tasks**. The 25 features from
Section 3 are a subset of these 39. Statistical caveat: with 792 tests, ~28 of
72 features would meet this criterion by chance alone under the global null, so
this list is only a broad candidate set.

| Feature | n_tasks (p<0.2) | Tasks |
|---|---|---|
| `pup_max` | 11 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_mean` | 11 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_median` | 11 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_min` | 10 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `sac_dur_total` | 9 | 0, 1, 2, 3, 4, 5, 6, 7, 10 |
| `geo_nn_dist_mean` | 8 | 0, 1, 2, 3, 4, 5, 6, 9 |
| `sac_amp_mean` | 8 | 3, 4, 5, 6, 7, 8, 9, 10 |
| `sac_count` | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `sac_dur_std` | 7 | 0, 1, 2, 6, 7, 9, 10 |
| `sac_turn_std` | 7 | 0, 3, 4, 6, 7, 8, 9 |
| `spa_fix_count` | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `geo_revisit_rate` | 6 | 0, 1, 2, 3, 4, 5 |
| `geo_sacc_amp_mean` | 6 | 3, 4, 5, 6, 7, 8 |
| `geo_scanpath_len` | 6 | 0, 2, 4, 5, 6, 9 |
| `sac_dur_max` | 6 | 0, 2, 6, 7, 8, 10 |
| `sac_dur_mean` | 6 | 1, 2, 3, 4, 9, 10 |
| `spa_center_dist_std` | 6 | 0, 1, 2, 4, 6, 9 |
| `tem_dur_total` | 6 | 0, 2, 4, 5, 6, 10 |
| `pup_std` | 5 | 1, 2, 7, 9, 10 |
| `pupdiff_std` | 5 | 1, 7, 8, 9, 10 |
| `sac_amp_std` | 5 | 3, 5, 7, 9, 10 |
| `spa_q2` | 5 | 0, 3, 4, 5, 10 |
| `spa_skew_x` | 5 | 0, 6, 7, 8, 9 |
| `geo_sacc_amp_max` | 4 | 0, 5, 6, 9 |
| `pupbl_slope` | 4 | 1, 2, 7, 10 |
| `pupbl_std` | 4 | 1, 2, 9, 10 |
| `pupdiff_slope` | 4 | 2, 4, 7, 9 |
| `sac_absdir_var` | 4 | 1, 2, 3, 6 |
| `spa_bbox_area` | 4 | 0, 3, 4, 6 |
| `spa_dispersion` | 4 | 3, 6, 8, 9 |
| `geo_angle_var` | 3 | 1, 3, 5 |
| `pup_first_last_diff` | 3 | 1, 2, 10 |
| `pup_slope` | 3 | 1, 2, 10 |
| `pupbl_first_last_diff` | 3 | 1, 2, 10 |
| `pupbl_max` | 3 | 1, 7, 8 |
| `spa_center_frac` | 3 | 1, 3, 9 |
| `spa_std_x` | 3 | 3, 6, 8 |
| `tem_dur_max` | 3 | 5, 8, 10 |
| `tem_velocity_mean` | 3 | 3, 5, 8 |

- Dataset restricted to these 39 features: `processed_dataset_v3/`.

## Repository layout

| Path | Content |
|---|---|
| `dataset/`, `dataset_v2/` | raw eye-tracker exports (2025-03-05) |
| `processed_dataset/` | features (72) + metadata for 157 participants |
| `processed_dataset_v2/` | subset with the 25 selected features |
| `processed_dataset_v3/` | subset with the 39 features (p < 0.2 in ≥ 3 tasks) |
| `EDA/` | notebooks, feature dictionary, group-comparison summary |
| `report/` | reports |
| `src/begad_norm/` | feature-extraction code and config |
| `tests/` | unit tests (config, raw data, features) |

## Notes

- The p-values used for screening are uncorrected Kruskal–Wallis p-values;
  `q_bh` (Benjamini–Hochberg over all 792 tests) is available in
  `EDA/comparison_summary.md`.
- Feature values on instruction-screen tasks (0, 2, 6, 8, 10) describe
  instruction viewing, not the test stimuli.
