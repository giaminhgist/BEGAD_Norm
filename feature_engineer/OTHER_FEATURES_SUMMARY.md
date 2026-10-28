# Other feature results: archived summary (files deleted 2026-10-28)

Source run: `2026-10-27_v2_expanded260` (freeze `freeze_35a60cb67109`, seed 20261027, no classifier training).
Cohorts: **RAW157** = raw Tobii event exports, 157 baseline participants (CN 100 / MCI 41 / AD 16), macro only,
tasks KD1/KD2/KD3/READING; **ROI64_T1235** = original ROI workbook, 64 participants (CN 32 / MCI 17 / AD 15),
micro and macro, KD1-3 + reading; **ROI64_T5** = same 64 participants, reading only, sensitivity analysis.
Cohorts were never pooled.

This file preserves the key results of every feature family **not** covered by
`FEATURES_37_SUMMARY.md`: the main macro screen, the micro screen, task-ROI rankings,
posthoc region evidence, audits, and figures. The underlying files were deleted on 2026-10-28 during project
cleanup; all numbers below are taken from them. Full raw data remain recoverable from git history.

## 1. Catalogue and screening

- 160 base definitions + 49 dataset-derived = **209** (target 260; extension partial: 49/100). 78/209 have
  sources (30 base, 48 new).
- Macro contexts eligible for the main screen: RAW157 **214**, ROI64_T1235 **39**, ROI64_T5 **10**.
  Micro contexts eligible: ROI64_T1235 **1338**, ROI64_T5 **526**.
- Main screen = feasible AND p_CN_AD < 0.20 AND p_CN_MCI < 0.20 (raw two-sided Mann-Whitney U).
- Passing the main screen: macro RAW157 **40**, ROI64_T1235 **7**, ROI64_T5 **0**; with FDR support
  (joint BH q <= 0.10): RAW157 **4**, others **0**; in the compact panel: RAW157 **18**, ROI64_T1235 **4**,
  ROI64_T5 **0**.
- One definition instantiates many hypotheses (per task, ROI, contrast), so 209 definitions is not the number
  of tests. Selection was exploratory; any predictive use must repeat selection inside training data or use an
  independent cohort.

## 2. Macro results

### FDR-supported (joint BH q <= 0.10), all RAW157

| ID@scope | Name | p CN-AD / CN-MCI / MCI-AD | delta CN-AD [CI] | delta CN-MCI [CI] | delta MCI-AD | q joint (BH) | N CN/MCI/AD | bootstrap selection / direction | LOO-AD kept |
|---|---|---|---|---|---|---|---|---|---|
| D023@KD3 | gaze_event_span | 0.000136 / 0.000246 / 0.040 | 0.60 [0.29, 0.85] | 0.39 [0.21, 0.58] | 0.35 | 0.053 | 100/41/16 | 0.99 / 1.00 | 1.00 |
| A002@KD3 | total_fixation_time | 9.5e-05 / 0.000868 / 0.077 | 0.61 [0.31, 0.85] | 0.36 [0.16, 0.56] | 0.30 | 0.093 | 100/41/16 | 0.97 / 1.00 | 1.00 |
| A003@KD3 | task_fixation_count | 0.000181 / 0.002 / 0.033 | 0.58 [0.32, 0.80] | 0.34 [0.15, 0.51] | 0.37 | 0.100 | 100/41/16 | 0.98 / 1.00 | 1.00 |
| D023@KD2 | gaze_event_span | 0.002 / 0.00082 / 0.419 | 0.49 [0.16, 0.77] | 0.36 [0.16, 0.55] | 0.14 | 0.100 | 100/41/16 | 0.95 / 0.99 | 1.00 |

BY retains none of these. A002, A003 and D023 are members of the 37-feature set (see
FEATURES_37_SUMMARY.md); D023/A002/A003/A021 are strongly correlated (longer card reading = more fixations
and saccades), so they are not independent confirmations.

### Compact macro panel (22 rows, `macro_compact_panel.csv`)

RAW157 (18): D023@KD3, D023@KD2, A003@KD3, D036@KD3, D023@READING, A003@KD2, A003@KD1, A021@KD2, A027@KD3,
A046@READING, D029@KD3, A021@KD3, D023@KD1, A002@KD2, A030@KD1, D030@KD3, D034@READING, D025@KD1.
ROI64_T1235 (4): D007@KD1, D007@KD2, D011@KD1, D010@KD3.

### Selected macro features (47 rows, `macro_selected_main.csv`; headline rows only)

RAW157 stable candidates beyond the FDR set include D036@KD3 `incoming_amplitude_fixation_duration_coupling`
(0.008 / 0.000121 / 0.729, delta -0.42/-0.41), D023@READING (0.008 / 0.016 / 0.245), A003@KD2
(0.027 / 4.5e-05 / 0.972), A046@READING (0.036 / 0.028 / 0.467, delta -0.33/-0.24), D029@KD3
(0.002 / 0.039 / 0.169, delta -0.48), A027@KD3 (0.040 / 0.027 / 0.003), A021@KD2/KD3, D030@KD3, D023@KD1,
A003@KD1, A002@KD2, A030@KD1, D034@READING, D025@KD1; exploratory-only rows: D040@KD2 (0.004 / 0.063),
D031@KD2/KD3, A046@KD2/KD3, A023@KD2/KD3/READING, A030@READING, A035@READING/KD2, D022@KD3, D038@KD3,
D039@READING, D026@KD1, A027@READING, A021@KD1, A028@READING, A029@KD1, A046@KD1, D029@KD1, A023@KD1.

ROI64_T1235 macro (7, none FDR-supported): D007@KD1 (0.015 / 0.005 / 0.806, delta 0.45/0.49, stable),
D010@KD3 (0.021 / 0.037 / 1.000, delta -0.45, stable), D007@KD2 (0.062 / 0.011 / 0.566, stable),
D011@KD1 (0.020 / 0.062 / 0.509, delta 0.42, stable), D009@KD3 (0.084 / 0.151 / 0.549, exploratory),
D012@KD2 (0.185 / 0.094 / 0.890, exploratory), D013@KD1 (0.189 / 0.123 / 1.000, exploratory).

### Selected-by-origin counts

Macro: RAW157 base160 23 + dataset100 17; ROI64_T1235 0 + 7. Micro: ROI64_T1235 base160 50 + dataset100 57;
ROI64_T5 14 + 27.

## 3. Micro results (ROI64_T1235; ROI64_T5 = reading-only sensitivity)

- 1338 eligible micro contexts in ROI64_T1235 (526 in ROI64_T5). No micro context reached joint BH q <= 0.10
  and no region reached q_region <= 0.10, so micro results are exploratory or bootstrap-stable only.
- 65 task-ROI pairs passed the main rule in ROI64_T1235 (20 shown; `roi_ranking_all.csv`, `roi_top20.csv`).
  Top 5 (anchor feature, p CN-AD / CN-MCI / MCI-AD; delta; region p; top-20 bootstrap frequency):

| Rank | Task | ROI | Anchor | Anchor p (CN-AD / CN-MCI / MCI-AD) | p_region | top-20 freq | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | KD1 | ROI 16 | D002 | 0.011 / 0.016 / 0.982 | 0.131 | 0.58 | stable_candidate |
| 2 | READING | ROI 138 | D001 | 0.019 / 0.024 / 1.000 | 0.196 | 0.41 | exploratory_only |
| 3 | KD1 | ROI 23 | M001 | 0.001 / 0.026 / 0.276 | 0.207 | 0.45 | stable_candidate |
| 4 | KD1 | ROI 15 | M001 | 0.006 / 0.028 / 0.610 | 0.221 | 0.45 | stable_candidate |
| 5 | KD3 | ROI 16 | M051 | 0.011 / 0.028 / 0.296 | 0.226 | 0.34 | stable_candidate |

- Reading-only T5 sensitivity (`roi_top20.csv` secondary rows; 25 regions, 20 shown). Top:
  READING ROI 134 (M001: 0.013 / 0.008 / 0.948, p_region 0.090, freq 0.54, stable),
  READING ROI 88 (D005: 0.012 / 0.024 / 0.815, p_region 0.166, freq 0.73, stable),
  READING ROI 50 (D003: 0.037 / 0.000929 / 0.706, freq 0.48, exploratory).
- ROI text/geometry/reading order was unavailable for all regions (`text_unavailable`).

## 4. Task-ROI rankings and pair-specific reserves

Four-task summary (`task_ranking.csv`): | Cohort | Task | ROI expected | testable | main-selected | regions passing CN-AD / CN-MCI / MCI-AD |
| ROI64_T1235 | KD1 | 40 | 40 | 18 | 36 / 34 / 32 |
| ROI64_T1235 | KD2 | 40 | 40 | 16 | 33 / 29 / 26 |
| ROI64_T1235 | KD3 | 40 | 38 | 15 | 33 / 29 / 24 |
| ROI64_T1235 | READING | 159 | 57 | 16 | 60 / 33 / 51 |
| ROI64_T5 | READING | 159 | 80 | 25 | 73 / 67 / 58 |

- 211 reserve regions in `roi_pair_specific_reserves.csv`. MCI-AD passes by cohort: RAW157 40,
  ROI64_T1235 252, ROI64_T5 109. Top MCI-AD hits: RAW157 D047@KD3 (p=0.000717, delta 0.58);
  ROI64_T1235 D004@READING ROI 068 (p=0.002, delta 0.79); RAW157 A024@KD3 (p=0.003, delta -0.52);
  RAW157 D047@KD2 (p=0.003, delta 0.50); ROI64_T1235 D004@READING ROI 015 (p=0.004, delta 0.78);
  ROI64_T5 D005@READING ROI 056 (p=0.004, delta 0.67); ROI64_T1235 D004@KD1 ROI 034 (p=0.006, delta 0.64);
  ROI64_T1235 D003@READING ROI 041 (p=0.007, delta -0.71); ROI64_T5 D004@READING ROI 138 (p=0.013, delta 0.59);
  ROI64_T1235 D004@KD2 ROI 027 (p=0.014, delta -0.54).

## 5. Stability, influence and permutation

- Permutation checks (Monte Carlo +1, 9999 permutations): **381**; disagreements with the primary test at
  p < 0.20: **8** (`permutation_validation.csv`).
- Bootstrap (500 replicates, `bootstrap_stability.csv`): D023@KD3 selection frequency main = 0.99; A002@KD3
  0.97; A003@KD3 0.98; D023@KD2 0.95; direction agreement was 1.00 for all FDR-supported rows.
- Coverage sensitivity (selected counts, `sensitivity_coverage.csv`): RAW157 macro 40 at cov 0.5 / 0.6 / 0.75;
  ROI64_T1235 macro 8 / 7 / 6; ROI64_T5 macro 0; ROI64_T1235 micro 129 / 107 / 64; ROI64_T5 micro 50 / 41 / 6.
- AD influence (leave-one-AD-out, `ad_influence.csv`): most stable candidates had prop_main_screen_preserved
  = 1.0 and no sign flips; e.g. D023@KD3 prop_main_screen_preserved = 1.0, sign_flips CN-AD = 0.

## 6. Quality and redundancy audits

- `quality_features.csv` (948 rows): per-subject-task data-quality metrics (tracking/pupil validity fractions,
  gap and off-media fractions, missing-ROI fraction, calibration status).
- `redundancy_audit.csv` (2411 rows): pairwise Spearman correlations between macro features per task,
  including within-group-centered correlations (e.g. A003@KD1 vs A021@KD1 rho = 0.68, centered 0.65).
- `proposal_audit.csv` (88 rows): acceptance decisions for the 49 dataset-derived definitions (D-series),
  with evidence from the paper Table II (TNSRE 2025, p. 1452) and label-blind workbook profiles.
- `dataset_proposed_features.csv`: full support rules of the 49 D-series definitions (kept).

## 7. Posthoc region evidence (files deleted; headline lists below)

`control_regions_KD20_READING15.csv` (35 rows: KD top-20 + reading top-15 control regions),
`mci_ad_regions_KD10_READING10.csv` (20 rows: KD top-10 + reading top-10 MCI-AD regions),
`mci_ad_region_evidence_all.csv` (193 rows: full region evidence).

KD control regions (top 10 of 20; anchor feature, anchor p CN-AD): KD1 ROI 016 (D002, 0.011),
KD1 ROI 023 (M001, 0.001), KD1 ROI 015 (M001, 0.006), KD3 ROI 016 (M051, 0.011), KD2 ROI 006 (D003, 0.028),
KD1 ROI 005 (M051, 0.030), KD1 ROI 027 (M001, 0.035), KD1 ROI 028 (D004, 0.002), KD3 ROI 002 (D002, 0.041),
KD2 ROI 038 (D005, 0.003).
Reading control regions (top 10 of 15): ROI 138 (D001, 0.019), ROI 088 (M051, 0.012), ROI 005 (M001, 0.036),
ROI 131 (D003, 0.043), ROI 134 (D004, 0.057), ROI 156 (D003, 0.081), ROI 079 (M001, 0.081),
ROI 105 (D003, 0.023), ROI 007 (D005, 0.089), ROI 000 (D002, 0.109).

KD MCI-AD regions (top 10; anchor, p MCI-AD, delta): KD3 ROI 029 (D001, 0.010, 0.75),
KD1 ROI 034 (D004, 0.047, 0.64), KD3 ROI 032 (M051, 0.092, 0.62), KD2 ROI 027 (D004, 0.112, -0.54),
KD1 ROI 026 (D003, 0.147, 0.49), KD1 ROI 008 (D002, 0.163, -0.47), KD2 ROI 010 (D003, 0.175, 0.41),
KD3 ROI 022 (D004, 0.197, 0.50), KD2 ROI 012 (D003, 0.236, 0.46), KD2 ROI 034 (D003, 0.247, -0.48).
Reading MCI-AD regions (top 10): ROI 068 (D004, 0.013, 0.79), ROI 015 (D004, 0.027, 0.78),
ROI 041 (D003, 0.052, -0.71), ROI 045 (M001, 0.091, 0.62), ROI 105 (D001, 0.120, -0.61),
ROI 010 (D001, 0.181, 0.59), ROI 057 (D003, 0.202, -0.57), ROI 079 (M001, 0.209, 0.52),
ROI 096 (D005, 0.221, -0.54), ROI 003 (D001, 0.260, 0.44).

None of the region p-values survived BH/BY (all q = 1.0); region evidence is exploratory.

## 8. Near-miss task-level features (in the 45-feature union list, outside the 37)

These 8 features met at most one of the three posthoc conditions and were excluded from
FEATURES_37_SUMMARY.md. p-values: CN-AD / CN-MCI / MCI-AD per task (KD1/KD2/KD3/READING).

| Cohort | ID | Name | Condition met | Best evidence |
|---|---|---|---|---|
| RAW157 | A036 | task_transition_entropy | cond2 only | MCI-AD KD2 = 0.139 |
| RAW157 | D020 | fixation_duration_successive_change_median | cond2 only | MCI-AD READING = 0.143 |
| RAW157 | D031 | transition_direction_entropy | cond1 only | CN-AD KD2 = 0.077; CN-MCI KD2 = 0.057 |
| RAW157 | D033 | post_large_saccade_correction_fraction | cond2 only | MCI-AD KD2 = 0.143 |
| RAW157 | D038 | immediate_return_fraction | cond2 only | CN-AD KD3 = 0.063; MCI-AD KD2 = 0.200 |
| RAW157 | D046 | binocular_horizontal_disparity_iqr | cond2 only | MCI-AD READING = 0.169 |
| ROI64_T1235 | D008 | refixated_target_fraction | cond2 only | MCI-AD KD3 = 0.183 |
| ROI64_T1235 | D012 | leftward_saccade_target_fraction | cond2 only | MCI-AD READING = 0.138 |

## 9. Figures (deleted)

- `fig1_availability_heatmap` (png/pdf): availability of feature contexts per cohort and level.
- `fig2_macro_forest` (png/pdf): effect sizes (Cliff's delta with CI) of selected macro features.
- `fig3_task_roi_evidence_ROI64_T1235` and `fig3_task_roi_evidence_ROI64_T5` (png/pdf): task-ROI evidence
  ranking of the micro screen.
- `fig4_top20_bootstrap` (png/pdf): bootstrap selection frequency of the top-20 regions.
- `fig5_participant_distributions` (png/pdf): participant-level value distributions.

## 10. Deleted file inventory

- Results: RESULTS_EN.md, RESULTS_VI.md.
- Macro: macro_stats_all.csv (881 rows), macro_selected_main.csv (47), macro_selected_three_pairs.csv (47),
  macro_compact_panel.csv (22), macro_pair_specific.csv (237), macro_features_long.csv (37659).
- Micro: micro_stats_all.csv (10113), micro_selected_main.csv (148), micro_selected_top20.csv (80),
  micro_pair_specific.csv (1319), micro_features_long.csv (214080).
- ROI/task: roi_ranking_all.csv (438), roi_top20.csv (40), roi_pair_specific_reserves.csv (211),
  task_ranking.csv (5), feature_origin_summary.csv.
- Audits/QA: quality_features.csv (948), redundancy_audit.csv (2411), proposal_audit.csv (88),
  sensitivity_coverage.csv, sensitivity_coverage_selected.csv (343), permutation_validation.csv (381),
  bootstrap_stability.csv (3030), ad_influence.csv (195).
- Manifests: multiplicity_manifest.json, stage_state.json.
- Posthoc: top45_features.csv (45), task_level_union_conditions.csv (45),
  control_regions_KD20_READING15.csv (35), mci_ad_regions_KD10_READING10.csv (20),
  mci_ad_region_evidence_all.csv (193).
- Cache (intermediate pickles, ~140 MB): long.pkl, raw_events.pkl, report.pkl, stability.pkl, stats.pkl,
  workbooks.pkl.
- Diagnosis profiles: raw_event_profile.csv, t1235_column_profile.csv, t5_column_profile.csv.
- Figures: 5 figures, png + pdf each.

## 11. What was kept (37-feature bundle)

`FEATURES_37_SUMMARY.md` and `.vi.md`,
`task_level_union_conditions_strict.csv`, `feature_registry.csv`,
`DATASET_PROPOSED_100_FEATURES.md` and `.vi.md`, `dataset_proposed_features.csv`, `semantic_audit.md` and
`semantic_audit.csv`, `candidate_freeze.json`, and run provenance (`run_config.yaml`, `run_metadata.json`,
`source_manifest.json`, `subject_manifest.csv`, `task_roi_manifest.csv`).

## 12. Main limitations (from RESULTS_EN.md)

No ROI geometry/text/reading order; no audio/TextGrid/error annotations; totalFixTime is not summed dwell;
workbook saccade counts are inconsistent; T1235 pupil fields are sign-inverted relative to an unidentified
constant; raw pupil only at event boundaries; no demographics; AD n = 15-16. These are results of this cohort
only, not confirmation or refutation of literature hypotheses.
