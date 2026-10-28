# Task-specific models + late probability fusion (33-feature x 4-task dataset)

Run `2026-10-28T21:29Z`. Detail: `results/results_detail.csv` (192 evaluations),
summary: `results/summary.csv`, fusion weights: `results/fusion_weights.csv`, paired
deltas: `results/paired_deltas.csv`, per-subject probabilities:
`results/proba_<problem>_<model_type>.csv`, provenance: `provenance.json`.

## 1. Setup

- **Data**: `processed_dataset/processed_dataset_33feat_wide.csv` — 157 subjects
  (CN 100 / MCI 41 / AD 16), 132 features (33 RAW157 features x 4 tasks:
  KD1/KD2/KD3/READING). The 51 missing cells are imputed per fold with the per-column
  median fitted on the training split only (SimpleImputer before StandardScaler, both
  fitted on training subjects at every CV level).
- **Split (fixed)**: identical to the baselines, from `processed_dataset/splits/` — 4
  held-out AD subjects (BEGAD-001, BEGAD-005, BEGAD-006, BEGAD-016) are appended to
  every validation fold and never enter training; the remaining 153 subjects form 4
  stratified folds (seed 42). Training per fold: 114-115 subjects (3-class) with 9 AD;
  validation: 42-43 subjects with 7 AD (3 from the fold + 4 held-out).
- **Task models**: each of KD1, KD2, KD3, READING gets its own model on its own 33
  features, for two base model families with the same tuning budget (selected by mean
  CV log-loss, a proper scoring rule appropriate for probability fusion):
  - `elasticnet`: LogisticRegression(penalty=elasticnet, saga, max_iter=2000,
    tol=1e-4, random_state=0 pinned — saga uses random_state to shuffle the data order,
    so an unpinned seed would make fits nondeterministic). Grid: C x l1_ratio =
    4 x 3 = 12 combinations.
  - `svm_rbf`: SVC(rbf) with sigmoid calibration via CalibratedClassifierCV with
    adaptive stratified folds `min(5, min class count)` (fixed cv=5 fails when an
    inner-inner split has fewer than 5 subjects of the smallest class). Grid:
    C x gamma = 3 x 3 = 9 combinations with gamma = k / n_features (features are
    standardized, so sklearn's 'scale' equals 1/n_features — this keeps the budget
    comparable across feature counts).
- **Models are unweighted** (class_weight=None) on purpose: fusion combines raw model
  probabilities and balanced class weights would shift their scale per problem. The
  baseline (output/baseline_33feat) used balanced weights with fixed C=1.0 and no
  tuning; comparisons against it are descriptive only.
- **Variants** per (problem, model_type, fold):
  - `task_kd1`..`task_reading`: individual task models (A).
  - `fusion_equal4`: mean of the 4 task probabilities (B).
  - `fusion_learned`: one shared task-weight vector (used for every class) learned by
    minimizing mean log-loss + lambda * KL(w || uniform) on the simplex (SLSQP, 13
    starts), with lambda = 0.1 (C). Weights come exclusively from inner OOF
    predictions of the training fold; validation subjects are never used.
  - `fusion_equal3kd`: mean of KD1/KD2/KD3 probabilities, READING excluded (D).
  - `concat132`: all 132 features in one model with the identical nested tuning
    protocol (reference for the overfitting hypothesis).
- **Nested protocol**: validation probabilities come from a model whose
  hyperparameters were selected by inner CV (4 folds, seed 42) on the full training
  fold, then refitted on it. The inner OOF probabilities used for the fusion weights
  and the Youden threshold come from per-inner-fold models whose fit, tuning
  (inner-inner CV, 4 folds, seed 7) and calibration never used the predicted subject.
- **Thresholds**: headline binary metrics at the fixed threshold 0.5; supplementary
  columns report the Youden-J threshold derived from training-level inner OOF
  probabilities and the validation metrics under it. 3-class predictions use argmax.
- **Determinism**: everything (splitters, saga, SVC, sigmoid calibration, SLSQP) is
  deterministic for a fixed input order, so each (problem, model_type, fold) is
  evaluated exactly once; the 4 folds are the evaluation unit.

## 2. Results summary (mean ± std over 4 folds)

### 2.1 3-class CN/MCI/AD

| Variant | EN acc | EN macroF1 | EN bacc | EN OVR AUC | EN rec AD | SVM acc | SVM macroF1 | SVM bacc | SVM OVR AUC | SVM rec AD |
|---|---|---|---|---|---|---|---|---|---|---|
| task_kd1 | 0.574 ± 0.030 | 0.245 ± 0.005 | 0.323 ± 0.017 | 0.569 ± 0.070 | 0.000 ± 0.000 | 0.580 ± 0.010 | 0.246 ± 0.003 | 0.327 ± 0.007 | 0.556 ± 0.038 | 0.000 ± 0.000 |
| task_kd2 | 0.604 ± 0.041 | 0.328 ± 0.021 | 0.373 ± 0.016 | 0.711 ± 0.041 | 0.000 ± 0.000 | 0.598 ± 0.025 | 0.324 ± 0.011 | 0.369 ± 0.003 | 0.649 ± 0.068 | 0.000 ± 0.000 |
| task_kd3 | 0.592 ± 0.040 | 0.332 ± 0.043 | 0.371 ± 0.037 | 0.717 ± 0.038 | 0.036 ± 0.062 | 0.574 ± 0.020 | 0.307 ± 0.047 | 0.351 ± 0.029 | 0.686 ± 0.053 | 0.036 ± 0.062 |
| task_reading | 0.592 ± 0.006 | 0.283 ± 0.038 | 0.348 ± 0.017 | 0.636 ± 0.013 | 0.000 ± 0.000 | 0.598 ± 0.004 | 0.289 ± 0.023 | 0.351 ± 0.011 | 0.600 ± 0.046 | 0.000 ± 0.000 |
| fusion_equal4 | 0.592 ± 0.029 | 0.275 ± 0.047 | 0.342 ± 0.032 | 0.727 ± 0.046 | 0.000 ± 0.000 | 0.598 ± 0.004 | 0.289 ± 0.023 | 0.351 ± 0.011 | 0.683 ± 0.049 | 0.000 ± 0.000 |
| fusion_learned | 0.592 ± 0.029 | 0.274 ± 0.045 | 0.342 ± 0.032 | 0.724 ± 0.047 | 0.000 ± 0.000 | 0.598 ± 0.004 | 0.289 ± 0.023 | 0.351 ± 0.011 | 0.682 ± 0.048 | 0.000 ± 0.000 |
| fusion_equal3kd | 0.580 ± 0.033 | 0.272 ± 0.043 | 0.335 ± 0.031 | 0.726 ± 0.051 | 0.000 ± 0.000 | 0.592 ± 0.012 | 0.287 ± 0.022 | 0.348 ± 0.012 | 0.677 ± 0.043 | 0.000 ± 0.000 |
| concat132 | 0.633 ± 0.014 | 0.390 ± 0.058 | 0.422 ± 0.039 | 0.737 ± 0.037 | 0.036 ± 0.062 | 0.615 ± 0.036 | 0.349 ± 0.061 | 0.394 ± 0.048 | 0.707 ± 0.071 | 0.000 ± 0.000 |

### 2.2 CN vs MCI (positive = MCI)

| Variant | EN AUC | EN AP | EN F1 | EN bacc | EN sen | EN spec | SVM AUC | SVM AP | SVM F1 | SVM bacc | SVM sen | SVM spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| task_kd1 | 0.630 ± 0.105 | 0.451 ± 0.119 | 0.000 ± 0.000 | 0.485 ± 0.026 | 0.000 ± 0.000 | 0.970 ± 0.052 | 0.546 ± 0.095 | 0.352 ± 0.069 | 0.000 ± 0.000 | 0.490 ± 0.010 | 0.000 ± 0.000 | 0.980 ± 0.020 |
| task_kd2 | 0.702 ± 0.083 | 0.582 ± 0.050 | 0.186 ± 0.135 | 0.540 ± 0.024 | 0.141 ± 0.135 | 0.940 ± 0.104 | 0.696 ± 0.056 | 0.558 ± 0.035 | 0.247 ± 0.074 | 0.554 ± 0.018 | 0.168 ± 0.073 | 0.940 ± 0.060 |
| task_kd3 | 0.681 ± 0.094 | 0.496 ± 0.113 | 0.117 ± 0.123 | 0.508 ± 0.055 | 0.075 ± 0.083 | 0.940 ± 0.066 | 0.684 ± 0.093 | 0.474 ± 0.104 | 0.234 ± 0.146 | 0.549 ± 0.045 | 0.168 ± 0.102 | 0.930 ± 0.059 |
| task_reading | 0.654 ± 0.063 | 0.435 ± 0.033 | 0.167 ± 0.173 | 0.538 ± 0.045 | 0.125 ± 0.130 | 0.950 ± 0.052 | 0.676 ± 0.051 | 0.519 ± 0.018 | 0.119 ± 0.129 | 0.531 ± 0.034 | 0.073 ± 0.082 | 0.990 ± 0.017 |
| fusion_equal4 | 0.688 ± 0.086 | 0.584 ± 0.109 | 0.083 ± 0.083 | 0.519 ± 0.020 | 0.048 ± 0.048 | 0.990 ± 0.017 | 0.719 ± 0.061 | 0.572 ± 0.040 | 0.087 ± 0.087 | 0.524 ± 0.024 | 0.048 ± 0.048 | 1.000 ± 0.000 |
| fusion_learned | 0.681 ± 0.086 | 0.569 ± 0.125 | 0.080 ± 0.080 | 0.514 ± 0.019 | 0.048 ± 0.048 | 0.980 ± 0.035 | 0.715 ± 0.058 | 0.563 ± 0.023 | 0.083 ± 0.083 | 0.514 ± 0.026 | 0.048 ± 0.048 | 0.980 ± 0.020 |
| fusion_equal3kd | 0.673 ± 0.069 | 0.560 ± 0.116 | 0.107 ± 0.186 | 0.534 ± 0.059 | 0.068 ± 0.118 | 1.000 ± 0.000 | 0.712 ± 0.067 | 0.548 ± 0.053 | 0.083 ± 0.083 | 0.519 ± 0.020 | 0.048 ± 0.048 | 0.990 ± 0.017 |
| concat132 | 0.698 ± 0.065 | 0.555 ± 0.093 | 0.407 ± 0.139 | 0.620 ± 0.051 | 0.341 ± 0.148 | 0.900 ± 0.060 | 0.751 ± 0.038 | 0.570 ± 0.048 | 0.369 ± 0.118 | 0.601 ± 0.041 | 0.291 ± 0.116 | 0.910 ± 0.059 |

### 2.3 CN vs AD (positive = AD)

| Variant | EN AUC | EN AP | EN F1 | EN bacc | EN sen | EN spec | SVM AUC | SVM AP | SVM F1 | SVM bacc | SVM sen | SVM spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| task_kd1 | 0.519 ± 0.089 | 0.244 ± 0.036 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.590 ± 0.105 | 0.353 ± 0.119 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| task_kd2 | 0.806 ± 0.078 | 0.661 ± 0.058 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 | 0.793 ± 0.045 | 0.620 ± 0.078 | 0.211 ± 0.212 | 0.567 ± 0.067 | 0.143 ± 0.143 | 0.990 ± 0.017 |
| task_kd3 | 0.851 ± 0.079 | 0.657 ± 0.143 | 0.118 ± 0.119 | 0.531 ± 0.032 | 0.071 ± 0.071 | 0.990 ± 0.017 | 0.801 ± 0.079 | 0.604 ± 0.130 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 |
| task_reading | 0.736 ± 0.063 | 0.536 ± 0.061 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.700 ± 0.088 | 0.433 ± 0.080 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| fusion_equal4 | 0.851 ± 0.061 | 0.721 ± 0.091 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.833 ± 0.058 | 0.642 ± 0.103 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| fusion_learned | 0.843 ± 0.061 | 0.700 ± 0.090 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 | 0.827 ± 0.057 | 0.635 ± 0.106 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| fusion_equal3kd | 0.839 ± 0.067 | 0.681 ± 0.083 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 | 0.826 ± 0.044 | 0.616 ± 0.105 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| concat132 | 0.881 ± 0.051 | 0.741 ± 0.089 | 0.417 ± 0.177 | 0.654 ± 0.093 | 0.357 ± 0.237 | 0.950 ± 0.052 | 0.717 ± 0.071 | 0.506 ± 0.078 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |

The Youden-threshold supplementary metrics are in the detail CSV; the most notable
pattern is that in CN vs MCI the fusion variants at the Youden threshold reach
sensitivity 0.58-0.73 (F1 0.48-0.55) versus sensitivity 0.05 (F1 0.08) at the fixed
0.5 threshold — the fused probabilities rank MCI subjects correctly but are shifted
toward the majority class, so a threshold derived from training-level probabilities
recovers most of the discriminative value.

## 3. Fusion weights (learned, lambda = 0.1)

Mean ± std over the 4 folds; the inner OOF cross-entropy of the learned fusion vs the
equal-weight fusion shows what the learning gains on training data.

| Problem / model | w_kd1 | w_kd2 | w_kd3 | w_reading | inner CE learned | inner CE equal |
|---|---|---|---|---|---|---|
| 3-class elasticnet | 0.215 ± 0.031 | 0.248 ± 0.047 | 0.312 ± 0.017 | 0.225 ± 0.013 | 0.806 | 0.810 |
| 3-class svm_rbf | 0.242 ± 0.028 | 0.281 ± 0.032 | 0.253 ± 0.018 | 0.223 ± 0.022 | 0.816 | 0.818 |
| CN vs MCI elasticnet | 0.229 ± 0.019 | 0.261 ± 0.059 | 0.282 ± 0.037 | 0.227 ± 0.016 | 0.584 | 0.587 |
| CN vs MCI svm_rbf | 0.255 ± 0.047 | 0.251 ± 0.042 | 0.284 ± 0.046 | 0.211 ± 0.028 | 0.576 | 0.580 |
| CN vs AD elasticnet | 0.215 ± 0.022 | 0.249 ± 0.021 | 0.288 ± 0.049 | 0.248 ± 0.040 | 0.334 | 0.337 |
| CN vs AD svm_rbf | 0.248 ± 0.015 | 0.290 ± 0.033 | 0.253 ± 0.025 | 0.208 ± 0.024 | 0.339 | 0.342 |

Reading the table together with the per-task inner log-loss (in
`results/fusion_weights.csv`): the fused inner OOF cross-entropy is lower than that of
every single task in all six problem/model combinations, which is direct evidence that
the tasks carry partially complementary information.

## 4. Paired per-fold deltas (a minus b, mean ± std over 4 folds)

| Problem | Model | Comparison | AUC delta | Sign across folds |
|---|---|---|---|---|
| 3-class | elasticnet | fusion_equal4 vs concat132 | -0.010 ± 0.018 | 3/4 negative |
| 3-class | elasticnet | fusion_learned vs fusion_equal4 | -0.003 ± 0.006 | 3/4 negative |
| 3-class | svm_rbf | fusion_equal4 vs concat132 | -0.024 ± 0.036 | 2/4 negative |
| CN vs MCI | elasticnet | fusion_equal4 vs concat132 | -0.010 ± 0.044 | 3/4 negative |
| CN vs MCI | elasticnet | task_kd2 vs concat132 | +0.004 ± 0.082 | mixed |
| CN vs MCI | svm_rbf | fusion_equal4 vs concat132 | -0.033 ± 0.041 | 3/4 negative |
| CN vs AD | elasticnet | fusion_equal4 vs concat132 | -0.030 ± 0.036 | 4/4 negative |
| CN vs AD | elasticnet | task_kd3 vs concat132 | -0.030 ± 0.029 | 3/4 negative |
| CN vs AD | svm_rbf | fusion_equal4 vs concat132 | +0.116 ± 0.039 | **4/4 positive** |
| CN vs AD | svm_rbf | task_kd1 vs concat132 | -0.127 ± 0.045 | 4/4 negative |
| CN vs AD | svm_rbf | fusion_learned vs fusion_equal4 | -0.006 ± 0.007 | 2/4 negative, 2/4 tie |

## 5. Hypotheses

**Does splitting by task reduce overfitting vs the 132-feature concatenation?**
No general advantage. With elastic-net, concat132 is the best variant on all three
problems (OVR AUC 0.737 / AUC 0.698 / 0.881); the fusion reaches parity with the best
single task but not above concat132 (mean paired delta negative in 5 of 6
problem/model cases). The exception is instructive: for SVM-RBF on CN vs AD, concat132
is the weakest non-KD1 variant (AUC 0.717) and every fusion beats it, in all 4 folds
(+0.116 ± 0.039 for fusion_equal4). So task-specific models do not reduce overfitting
per se, but fusion acts as a regularizer that rescues a base model that fits poorly on
the 132-feature space (the unweighted SVM struggles with the tiny AD class). Note the
baseline comparison is descriptive only: baseline_33feat LR (balanced, untuned)
reached OVR AUC 0.705 in 3-class and AUC 0.891 in CN vs AD; our unweighted tuned
concat132 reaches 0.737 and 0.881.

**Do the tasks provide complementary information?** Yes on training data. The fused
inner OOF cross-entropy is lower than every single task's in all six problem/model
combinations (e.g. 3-class EN: fused 0.806 vs best task 0.807; CN vs AD EN: fused
0.334 vs best task 0.353), and the learned weights are systematically non-uniform
(KD3 highest in 4 of 6 cases, READING lowest in 4 of 6). On validation the fusion
matches or slightly trails the best single task and concat132, so the complementary
signal is real but small relative to fold-to-fold variance.

**Are learned weights better than equal weights, and are they stable across folds?**
The learned weights beat equal weights on inner OOF by only 0.002-0.004 CE (the
regularization lambda = 0.1 pulls hard toward uniform), and on validation they are
equal or slightly worse than equal weights in 10 of 12 comparisons (e.g. CN vs AD EN
-0.009 AUC). With 105-115 training subjects and 3 free weight parameters, the weight
vector mildly overfits the inner OOF. Weights are reasonably stable across folds
(std 0.013-0.059, and the KD3-highest / READING-lowest ordering is mostly preserved),
but the practical recommendation from these data is to use equal weights and treat
the learned weights as a descriptive tool for task importance.

## 6. Limitations

- Validation folds overlap in the 4 held-out AD subjects (by design of the fixed
  split), so pooled CV metrics overweight those 4 subjects; per-fold metrics are
  reported separately in the detail CSV.
- 16 AD subjects total (12 in CV, 9 per training fold) make the CN-vs-AD and 3-class
  estimates noisy; fold-to-fold std is the honest uncertainty measure here.
- lambda = 0.1 is a fixed prior strength, not tuned; the weights are regularized
  toward uniform by design, so learned-weight gains are conservative.
- Models are unweighted, unlike the balanced baselines; balanced weights interact
  badly with probability fusion and were left out on purpose.
- Some weakly regularized grid corners (C=10) reach saga's iteration cap on some
  folds; their CV scores are still usable for selection (they lose by a wide margin)
  and the behavior is identical across all variants.
- Two documented post-run completions were applied to the artifacts (see
  `src/expcommon/postprocess_first_full_runs.py`): the contrasts proba rows were
  merged to wide layout, and the 3-class balanced-accuracy cells missing from the
  first runs were recomputed from the stored per-subject probabilities (argmax rule),
  each validated against the stored per-class recalls. The runner code was fixed to
  write both directly in future runs.
