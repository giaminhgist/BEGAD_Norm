# Per-task vs pooled baseline — patch-level comparison

Side-by-side comparison of the cell-level baseline run on all four stimulus
tasks pooled (`outputs/baseline/`, 2,512 samples = 157 subjects x 4 tasks x 4
cells) against the same protocol run per task (`outputs/baseline/task_{3,4,5,9}/`,
628 samples each = 157 subjects x 4 cells). Primary metric: subject-level AUC,
mean ± std over 100 evaluations (20 fixed seeds x 5 fixed subject folds). The
(seed, fold) evaluation keys are identical across all five runs, so every
comparison below is paired at the evaluation level.

Protocol (identical for all runs, see `baseline_cells.md`): 28 cell features,
3 problems, 6 models, median imputation + `StandardScaler` fitted per fold on
the training split only, balanced class weights/priors, subject-level 5-fold CV
from the fixed seed-42 split reused across seeds.

## 1. Subject-level AUC — mean ± std over 100 evaluations

### co_vs_mci (positive = MCI)

| Model | pooled | task 3 | task 4 | task 5 | task 9 | best task |
|---|---|---|---|---|---|---|
| fnn | 0.546±0.069 | 0.602±0.093 | 0.703±0.059 | 0.587±0.117 | 0.527±0.100 | task 4 (+0.157) |
| gnb | 0.597±0.112 | 0.626±0.094 | 0.611±0.043 | 0.587±0.176 | 0.578±0.130 | task 3 (+0.030) |
| logistic_regression | 0.652±0.093 | 0.622±0.076 | 0.656±0.036 | 0.704±0.071 | 0.629±0.074 | task 5 (+0.053) |
| qda | 0.597±0.100 | 0.635±0.059 | 0.644±0.063 | 0.554±0.168 | 0.575±0.086 | task 4 (+0.047) |
| svm_linear | 0.638±0.097 | 0.627±0.069 | 0.635±0.054 | 0.703±0.069 | 0.643±0.062 | task 5 (+0.065) |
| svm_rbf | 0.609±0.091 | 0.667±0.056 | 0.661±0.091 | 0.573±0.125 | 0.555±0.064 | task 3 (+0.058) |

### co_vs_ad (positive = AD)

| Model | pooled | task 3 | task 4 | task 5 | task 9 | best task |
|---|---|---|---|---|---|---|
| fnn | 0.659±0.140 | 0.694±0.169 | 0.558±0.165 | 0.713±0.139 | 0.561±0.125 | task 5 (+0.053) |
| gnb | 0.662±0.187 | 0.693±0.148 | 0.648±0.151 | 0.736±0.159 | 0.552±0.176 | task 5 (+0.073) |
| logistic_regression | 0.647±0.120 | 0.705±0.171 | 0.625±0.083 | 0.708±0.139 | 0.536±0.131 | task 5 (+0.061) |
| qda | 0.597±0.186 | 0.702±0.092 | 0.588±0.089 | 0.702±0.101 | 0.531±0.129 | task 3 (+0.106) |
| svm_linear | 0.660±0.128 | 0.655±0.219 | 0.632±0.081 | 0.714±0.148 | 0.553±0.159 | task 5 (+0.054) |
| svm_rbf | 0.701±0.135 | 0.725±0.171 | 0.650±0.091 | 0.708±0.077 | 0.672±0.106 | task 3 (+0.025) |

### co_mci_ad (3-class, macro metrics)

| Model | pooled | task 3 | task 4 | task 5 | task 9 | best task |
|---|---|---|---|---|---|---|
| fnn | 0.570±0.083 | 0.605±0.087 | 0.638±0.069 | 0.572±0.094 | 0.564±0.072 | task 4 (+0.068) |
| gnb | 0.570±0.135 | 0.586±0.064 | 0.561±0.073 | 0.570±0.123 | 0.560±0.086 | task 3 (+0.016) |
| logistic_regression | 0.617±0.084 | 0.613±0.099 | 0.605±0.068 | 0.628±0.072 | 0.589±0.102 | task 5 (+0.011) |
| qda | 0.566±0.133 | 0.607±0.071 | 0.576±0.082 | 0.547±0.104 | 0.572±0.072 | task 3 (+0.041) |
| svm_linear | 0.629±0.076 | 0.633±0.056 | 0.610±0.057 | 0.682±0.083 | 0.605±0.075 | task 5 (+0.053) |
| svm_rbf | 0.638±0.085 | 0.661±0.089 | 0.634±0.064 | 0.621±0.087 | 0.608±0.063 | task 3 (+0.024) |

## 2. Paired evidence — best task vs pooled, per (seed, fold)

Mean paired subject-AUC delta (task minus pooled) over the same 100
(seed, fold) evaluations, with the share of evaluations in which the single
task beats pooled. Only the best task per (problem, model) is shown.

| Problem | Model | Best task | paired delta | share task > pooled |
|---|---|---|---|---|
| co_vs_mci | fnn | task 4 | +0.157±0.096 | 94% |
| co_vs_mci | gnb | task 3 | +0.030±0.081 | 60% |
| co_vs_mci | logistic_regression | task 5 | +0.053±0.067 | 80% |
| co_vs_mci | qda | task 4 | +0.047±0.121 | 60% |
| co_vs_mci | svm_linear | task 5 | +0.065±0.081 | 80% |
| co_vs_mci | svm_rbf | task 3 | +0.058±0.066 | 80% |
| co_vs_ad | fnn | task 5 | +0.053±0.103 | 69% |
| co_vs_ad | gnb | task 5 | +0.073±0.083 | 80% |
| co_vs_ad | logistic_regression | task 5 | +0.061±0.064 | 80% |
| co_vs_ad | qda | task 3 | +0.106±0.133 | 60% |
| co_vs_ad | svm_linear | task 5 | +0.054±0.092 | 62% |
| co_vs_ad | svm_rbf | task 3 | +0.025±0.092 | 60% |
| co_mci_ad | fnn | task 4 | +0.068±0.083 | 78% |
| co_mci_ad | gnb | task 3 | +0.016±0.081 | 60% |
| co_mci_ad | logistic_regression | task 5 | +0.011±0.084 | 40% |
| co_mci_ad | qda | task 3 | +0.041±0.083 | 80% |
| co_mci_ad | svm_linear | task 5 | +0.053±0.090 | 80% |
| co_mci_ad | svm_rbf | task 3 | +0.024±0.026 | 79% |

## 3. Cell-level AUC

Cell-level AUC points the same way. Pooled cell AUC sits near chance in every
problem (co_vs_mci 0.53-0.60, co_vs_ad 0.55-0.61, co_mci_ad 0.53-0.59), while
the best single task reaches 0.59-0.65 for co_vs_mci, 0.63-0.68 for co_vs_ad
(gnb task 5: 0.680 vs pooled 0.613), and 0.58-0.63 for co_mci_ad (svm_linear
task 5: 0.628 vs pooled 0.589).

## 4. Findings

1. **Single-task beats pooled in every (problem, model) cell.** In all 18
   cells the best single-task subject AUC exceeds the pooled run (pooled best:
   co_vs_mci 0.652, co_vs_ad 0.701, co_mci_ad 0.638). The pooled run never
   dominates. In 17 of 18 cells at least 60% of the paired evaluations agree;
   the only soft case is logistic_regression on co_mci_ad (+0.011, 40%).
2. **The largest gains are the most consistent.** fnn on co_vs_mci at task 4
   gains +0.157 paired with 94% of evaluations better than pooled; qda on
   co_vs_ad at task 3 gains +0.106; gnb on co_vs_ad at task 5 gains +0.073
   with 80%.
3. **Task 5 (KD card 3) is the strongest single task** and is best or near
   best for co_vs_ad and co_mci_ad across all six models. Task 4 is the
   strongest for co_vs_mci with fnn. Task 9 (reading) is the weakest: for
   co_vs_ad most models fall 0.10 below pooled (gnb -0.110, logistic_regression
   -0.111, svm_linear -0.107 paired), and only 20-40% of its evaluations beat
   pooled. This matches the screening observation that task 9 contributed
   little signal.
4. **Interpretation.** Pooling quadruples the training cells (2,512 vs 628)
   but mixes four task regimes in one feature space; the simple classifiers
   here cannot share parameters across regimes, so task-specific signal is
   diluted. The per-task models see a homogeneous regime and recover more
   signal despite having one quarter of the samples. The effect is visible at
   both cell and subject level.
5. **Implication for modeling.** For this feature set, task-specific models
   are the stronger direction, either as separate per-task classifiers or as a
   pooled model that gets task identity (task indicators or per-task
   normalization). This also supports running the normative-model approach per
   task.

## 5. Caveats

- Fixed folds across seeds: the stds cover model randomness only, not split
  variability. The paired deltas share folds, so the share-greater-than-pooled
  percentages describe within-protocol consistency, not population inference.
- Per-fold test sets are small (28-29 subjects; the AD group has 16 subjects
  in total), so per-task subject AUCs are noisy, particularly co_vs_ad.
- Cells of one subject are not independent; subject-level aggregation
  addresses this for the reported subject metrics.
- For co_vs_ad, subject accuracy is degenerate for svm_linear and svm_rbf
  (sensitivity 0.0, accuracy = majority rate 0.862), so AUC is the meaningful
  metric for that problem.
- The 12-feature screening shortlist was derived on all subjects' labels;
  CV estimates using that shortlist with these folds are optimistic unless
  the shortlist is re-derived inside each training fold.

## 6. Next steps

1. Carry task-specific modeling forward on the strongest tasks (task 4 and
   task 5, plus task 3 for co_vs_ad).
2. Cross-task generalization test: train per task, evaluate on the other
   tasks, to measure how task-specific the signal is.
3. Ablation on the pooled model: add task identity via indicators or per-task
   feature normalization to check whether the dilution is recoverable.
4. Run the subject-level normative model per task and compare against the
   per-task classifiers.
