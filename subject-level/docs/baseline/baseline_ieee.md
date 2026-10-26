# IEEE feature-set baseline — BEGAD group classification

Trial of the 20-feature set from the IEEE multimodal AD/MCI paper (Table II)
mapped onto BEGAD, run under the exact protocol of the balanced baseline
(`baseline_balanced.md`): participant-wide wide layout (features x 11 tasks,
feature-major), 3 problems, 6 models, 20 fixed seeds, 5-fold participant CV
stratified by group, median imputation + `StandardScaler` per fold on the
training split only, balanced class weights/priors, fixed 0.5 threshold.
Same seed schedule as `baseline_balanced` -> matched folds per seed.

## 1. Feature mapping: 16 of 20 features available

| # | IEEE feature | BEGAD mapping | Status |
|---|---|---|---|
| 1 | ROI | word/number AOI of the reading task | **unavailable** (no AOI annotations in BEGAD) |
| 2 | Number of fixations | `vis_n_fix` = clean fixations of the task | available |
| 3 | Number of saccades | `vis_n_sac` = fixations with an attached saccade | available |
| 4 | Number of short saccades (< 1 deg) | `vis_n_sac_short` | available |
| 5 | Number of long saccades (> 2 deg) | `vis_n_sac_long` | available |
| 6 | Number of regressive saccades | `vis_n_sac_reg` (turn angle > 90 deg) | available |
| 7 | Saccade amplitude sum | `vis_sac_amp_sum` (deg) | available |
| 8 | Total fixation time | `vis_fix_time_total` (ms) | available |
| 9 | Left pupil at first fixation | `pup_l_first` (baseline-corrected) | available |
| 10 | Left pupil at last fixation | `pup_l_last` | available |
| 11 | Left pupil difference | `pup_l_diff` = last - first | available |
| 12 | Right pupil at first fixation | `pup_r_first` | available |
| 13 | Right pupil at last fixation | `pup_r_last` | available |
| 14 | Right pupil difference | `pup_r_diff` | available |
| 15 | Minimum left pupil size | `pup_l_min` | available |
| 16 | Maximum left pupil size | `pup_l_max` | available |
| 17 | Min-max difference | `pup_l_minmax_diff` | available |
| 18 | Total spoken time | - | **unavailable** (no audio in BEGAD) |
| 19 | Delay first fixation -> utterance onset | - | **unavailable** |
| 20 | Delay last fixation -> utterance end | - | **unavailable** |

Conventions follow the project pipeline (`rawdata.clean_fixations`): saccade
amplitude/direction attached to the clean fixation they land on, amplitude in
degrees, regressive = turn angle `min(reldir, 360 - reldir) > 90`. Pupil
features are baseline-corrected per the paper (per-participant
`PupilLeftBaseline` / `PupilRightBaseline`); 16 of 157 participants lack the
baseline for one or both eyes, so their absolute pupil features of the
affected eye stay NaN and are median-imputed per fold (same treatment as the
`pupbl` features of the main baseline). The differences (`pup_l_diff`,
`pup_r_diff`, `pup_l_minmax_diff`) are baseline-invariant and always computed.

Feature table: `processed_dataset/ieee_features.csv` (1,727 rows = 157
participants x 11 tasks, 1,020 missing cells of 27,632). Builder:
`src/begad_norm/features/ieee.py` + `scripts/build_ieee_features.py`. Run
artifacts: `outputs/baseline_ieee/` (1,800 evaluations).

## 2. Results — mean ± std over 100 evaluations (20 seeds x 5 folds)

### co_vs_mci (positive = MCI)

| Model | IEEE AUC | balanced AUC | delta | IEEE acc | bal acc | IEEE sen/spec | bal sen/spec |
|---|---|---|---|---|---|---|---|
| fnn | 0.564±0.112 | 0.695±0.101 | -0.132 | 0.650 | 0.718 | 0.305/0.791 | 0.457/0.824 |
| gnb | 0.548±0.108 | 0.643±0.103 | -0.095 | 0.555 | 0.630 | 0.480/0.586 | 0.591/0.645 |
| logistic_regression | 0.602±0.114 | 0.636±0.087 | -0.034 | 0.633 | 0.657 | 0.389/0.733 | 0.447/0.743 |
| qda | 0.431±0.097 | 0.553±0.094 | -0.123 | 0.613 | 0.696 | 0.093/0.826 | 0.093/0.944 |
| svm_linear | 0.581±0.127 | 0.598±0.097 | -0.017 | 0.632 | 0.645 | 0.371/0.740 | 0.407/0.742 |
| svm_rbf | 0.500±0.115 | 0.738±0.094 | -0.238 | 0.631 | 0.728 | 0.286/0.773 | 0.458/0.839 |

### co_vs_ad (positive = AD)

| Model | IEEE AUC | balanced AUC | delta | IEEE acc | bal acc | IEEE sen/spec | bal sen/spec |
|---|---|---|---|---|---|---|---|
| fnn | 0.726±0.156 | 0.714±0.156 | +0.012 | 0.850 | 0.857 | 0.321/0.935 | 0.307/0.945 |
| gnb | 0.695±0.170 | 0.797±0.159 | -0.101 | 0.854 | 0.847 | 0.418/0.924 | 0.598/0.887 |
| logistic_regression | 0.754±0.145 | 0.857±0.100 | -0.103 | 0.839 | 0.859 | 0.393/0.911 | 0.376/0.936 |
| qda | 0.495±0.196 | 0.594±0.223 | -0.099 | 0.861 | 0.861 | 0.000/0.999 | 0.000/0.999 |
| svm_linear | 0.736±0.152 | 0.864±0.099 | -0.128 | 0.836 | 0.869 | 0.343/0.916 | 0.348/0.952 |
| svm_rbf | 0.746±0.148 | 0.843±0.090 | -0.096 | 0.866 | 0.862 | 0.101/0.989 | 0.056/0.991 |

### co_mci_ad (3-class, macro metrics)

| Model | IEEE AUC | balanced AUC | delta | IEEE acc | bal acc | IEEE sen/spec | bal sen/spec |
|---|---|---|---|---|---|---|---|
| fnn | 0.604±0.086 | 0.701±0.086 | -0.097 | 0.576 | 0.647 | 0.397/0.715 | 0.475/0.765 |
| gnb | 0.566±0.095 | 0.657±0.089 | -0.092 | 0.475 | 0.568 | 0.387/0.693 | 0.470/0.756 |
| logistic_regression | 0.631±0.081 | 0.698±0.072 | -0.067 | 0.574 | 0.591 | 0.438/0.725 | 0.452/0.741 |
| qda | 0.455±0.106 | 0.564±0.107 | -0.109 | 0.552 | 0.620 | 0.309/0.650 | 0.341/0.674 |
| svm_linear | 0.629±0.082 | 0.691±0.076 | -0.062 | 0.568 | 0.574 | 0.424/0.711 | 0.427/0.721 |
| svm_rbf | 0.604±0.091 | 0.758±0.071 | -0.155 | 0.562 | 0.648 | 0.374/0.707 | 0.459/0.776 |

## 3. Findings

1. **The IEEE set underperforms the balanced baseline in 17 of 18 cells.**
   The only positive AUC delta is fnn on co_vs_ad (+0.012, within seed
   variability, std 0.156). The largest drops are svm_rbf on co_vs_mci
   (-0.238, 0.500 vs 0.738) and svm_rbf on co_mci_ad (-0.155). Median AUC
   delta across the 18 cells: -0.099.
2. **The comparison is not a fair fight in one direction only.** The
   balanced baseline's 25 features were selected on the labels (Kruskal-Wallis
   p < 0.1 in >= 3 tasks, screening on all subjects), which inflates its CV
   estimate. The IEEE set is prescribed a priori (no selection bias) and
   still has fewer features (16 vs 25, 176 vs 275 wide columns). Both
   directions of advantage should be kept in mind: the delta partly reflects
   label-driven feature selection, and the IEEE set is missing its strongest
   modality.
3. **Four IEEE features could not be mapped.** ROI (word-level AOIs of the
   reading task) and all three speech features have no BEGAD counterpart
   (no AOI annotations, no audio). The paper's fusion network put speech on
   an equal footing with eye movements; on BEGAD the IEEE set is effectively
   eye-movements-only, which plausibly explains part of the gap.
4. **Task heterogeneity is ignored here, as in the main baseline.** The
   patch-level comparison showed that pooling the 11 tasks dilutes
   task-specific signal; the same caveat applies to both runs in this
   comparison, so it does not change the relative result.

## 4. Caveats

- Matched comparison: same problems, models, seeds, folds (seed-42
  round-robin assignment), imputation and threshold as `baseline_balanced`.
- 16 of 157 participants lack a pupil baseline in one or both eyes: their
  absolute pupil features are NaN and median-imputed per fold (1,020 of
  27,632 cells missing in total).
- The comparator's feature set was label-selected (optimistic CV); the IEEE
  set is prescribed (unbiased but missing ROI + speech).
- QDA on co_vs_ad is degenerate in both runs (sensitivity 0.0), so its AUC
  columns carry little information.

## 5. Next steps

- Per-task IEEE runs (the patch-level result showed single tasks beat the
  pooled run; the IEEE set may behave the same).
- Fair head-to-head: label-selected 16-feature shortlist vs the prescribed
  IEEE set, both under the same CV.
- Add a ROI-like proxy (stimulus-region coverage) only if a principled AOI
  definition for the BEGAD stimuli is adopted first.
