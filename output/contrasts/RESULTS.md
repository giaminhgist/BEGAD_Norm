# Absolute features + cross-task contrasts (33-feature x 4-task dataset)

Run `2026-10-28T21:57Z`. Detail: `results/results_detail.csv` (72 evaluations),
summary: `results/summary.csv`, paired deltas: `results/paired_deltas.csv`, contrast
manifest: `results/contrast_manifest.csv`, per-subject probabilities:
`results/proba_<problem>_<model_type>.csv`, provenance: `provenance.json`.

## 1. Setup

- **Data and split**: identical to the fusion experiment and the baselines —
  `processed_dataset/processed_dataset_33feat_wide.csv` (157 subjects, 132 features),
  fixed 4-fold split with the 4 held-out AD subjects appended to every validation fold
  and never in training (see output/fusion/RESULTS.md section 1 for details).
- **Contrast features (18)**, computed from the raw matrix BEFORE imputation so that
  missingness propagates into the per-fold imputer:
  - ratio-log for the time/count features D023, A002, A003 (all strictly positive in
    this dataset; a non-positive component would yield a missing contrast):
    `contrast_21 = log(x_KD2/x_KD1)`, `contrast_31 = log(x_KD3/x_KD1)`.
  - difference for the signed saccade/coupling/temporal features D025, D026, D035,
    D036, D017, D022 (no log transform on signed features):
    `contrast_21 = x_KD2 - x_KD1`, `contrast_31 = x_KD3 - x_KD1`.
  - The manifest (`results/contrast_manifest.csv`) records per-contrast finite counts,
    NaN counts and ranges: all 6 ratio contrasts are finite for all 157 subjects; the
    12 difference contrasts have 1-2 missing cells each (from the unsupported cells in
    their components), imputed per fold inside the training pipeline.
  - READING is kept as its own task, never treated as a fourth KD level; the contrasts
    are not interpreted as cognitive-load responses (they can reflect layout
    differences and task demands as well).
- **Representations** (identical outer splits, inner seeds and tuning budget):
  - `abs132`: all 132 absolute features (33 x 4 tasks).
  - `abs150`: abs132 + 18 contrasts = 150.
  - `compact84`: 33 KD1 + 33 READING + 18 contrasts = 84.
- **Models and protocol**: elastic-net logistic regression (primary) and calibrated
  SVM-RBF (control), with the exact same grids, log-loss tuning, per-fold
  imputation/scaling, inner CV (4 folds, seed 42) and inner-inner OOF (seed 7)
  protocol as the fusion experiment; `abs132` is bit-for-bit identical to the fusion
  experiment's `concat132`. Models are unweighted; threshold 0.5 for headline binary
  metrics, Youden-J from inner OOF as supplementary columns, argmax for 3-class.

## 2. Results summary (mean ± std over 4 folds)

### 2.1 3-class CN/MCI/AD

| Repr | EN acc | EN macroF1 | EN bacc | EN OVR AUC | EN rec AD | SVM acc | SVM macroF1 | SVM bacc | SVM OVR AUC | SVM rec AD |
|---|---|---|---|---|---|---|---|---|---|---|
| abs132 | 0.633 ± 0.014 | 0.390 ± 0.058 | 0.422 ± 0.039 | 0.737 ± 0.037 | 0.036 ± 0.062 | 0.615 ± 0.036 | 0.349 ± 0.061 | 0.394 ± 0.048 | 0.707 ± 0.071 | 0.000 ± 0.000 |
| abs150 | 0.621 ± 0.022 | 0.390 ± 0.087 | 0.419 ± 0.053 | 0.729 ± 0.041 | 0.071 ± 0.071 | 0.621 ± 0.017 | 0.384 ± 0.040 | 0.411 ± 0.023 | 0.693 ± 0.076 | 0.036 ± 0.062 |
| compact84 | 0.592 ± 0.027 | 0.328 ± 0.069 | 0.370 ± 0.043 | 0.684 ± 0.054 | 0.036 ± 0.062 | 0.597 ± 0.033 | 0.282 ± 0.059 | 0.349 ± 0.044 | 0.629 ± 0.076 | 0.000 ± 0.000 |

### 2.2 CN vs MCI (positive = MCI)

| Repr | EN AUC | EN AP | EN F1 | EN bacc | EN sen | EN spec | SVM AUC | SVM AP | SVM F1 | SVM bacc | SVM sen | SVM spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs132 | 0.698 ± 0.065 | 0.555 ± 0.093 | 0.407 ± 0.139 | 0.620 ± 0.051 | 0.341 ± 0.148 | 0.900 ± 0.060 | 0.751 ± 0.038 | 0.570 ± 0.048 | 0.369 ± 0.118 | 0.601 ± 0.041 | 0.291 ± 0.116 | 0.910 ± 0.059 |
| abs150 | 0.677 ± 0.082 | 0.569 ± 0.102 | 0.348 ± 0.129 | 0.593 ± 0.050 | 0.266 ± 0.122 | 0.920 ± 0.049 | 0.731 ± 0.091 | 0.577 ± 0.081 | 0.377 ± 0.060 | 0.597 ± 0.026 | 0.293 ± 0.072 | 0.900 ± 0.035 |
| compact84 | 0.640 ± 0.096 | 0.517 ± 0.091 | 0.202 ± 0.140 | 0.537 ± 0.045 | 0.143 ± 0.103 | 0.930 ± 0.071 | 0.670 ± 0.093 | 0.498 ± 0.089 | 0.240 ± 0.172 | 0.563 ± 0.058 | 0.166 ± 0.134 | 0.960 ± 0.028 |

### 2.3 CN vs AD (positive = AD)

| Repr | EN AUC | EN AP | EN F1 | EN bacc | EN sen | EN spec | SVM AUC | SVM AP | SVM F1 | SVM bacc | SVM sen | SVM spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs132 | 0.881 ± 0.051 | 0.741 ± 0.089 | 0.417 ± 0.177 | 0.654 ± 0.093 | 0.357 ± 0.237 | 0.950 ± 0.052 | 0.717 ± 0.071 | 0.506 ± 0.078 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| abs150 | 0.906 ± 0.076 | 0.767 ± 0.125 | 0.466 ± 0.222 | 0.689 ± 0.126 | 0.429 ± 0.303 | 0.950 ± 0.052 | 0.787 ± 0.100 | 0.595 ± 0.126 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 |
| compact84 | 0.821 ± 0.066 | 0.674 ± 0.062 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 | 0.681 ± 0.064 | 0.466 ± 0.115 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |

Inner-CV log-loss (training-level fit) improves with contrasts in every problem
(3-class EN 0.836 → 0.809; CN vs MCI EN 0.649 → 0.624; CN vs AD EN 0.371 → 0.345), so
the contrasts carry genuine training signal everywhere; the validation columns above
show where that signal generalizes.

## 3. Paired per-fold deltas (a minus b, same fold, same model type)

| Problem | Model | Comparison | AUC delta | Sign across folds |
|---|---|---|---|---|
| 3-class | elasticnet | abs150 vs abs132 | -0.008 ± 0.011 | 3/4 negative |
| 3-class | svm_rbf | abs150 vs abs132 | -0.015 ± 0.026 | 3/4 negative |
| 3-class | elasticnet | compact84 vs abs132 | -0.054 ± 0.018 | 4/4 negative |
| 3-class | svm_rbf | compact84 vs abs132 | -0.079 ± 0.033 | 4/4 negative |
| CN vs MCI | elasticnet | abs150 vs abs132 | -0.020 ± 0.020 | 3/4 negative |
| CN vs MCI | svm_rbf | abs150 vs abs132 | -0.021 ± 0.073 | 3/4 negative |
| CN vs MCI | elasticnet | compact84 vs abs132 | -0.058 ± 0.055 | 3/4 negative |
| CN vs MCI | svm_rbf | compact84 vs abs132 | -0.082 ± 0.084 | 3/4 negative |
| CN vs AD | elasticnet | abs150 vs abs132 | +0.024 ± 0.039 | 3/4 positive |
| CN vs AD | svm_rbf | abs150 vs abs132 | +0.070 ± 0.080 | **4/4 positive** |
| CN vs AD | elasticnet | compact84 vs abs132 | -0.060 ± 0.057 | 3/4 negative |
| CN vs AD | svm_rbf | compact84 vs abs132 | -0.036 ± 0.023 | 4/4 negative |

Per-fold AUC for the headline comparison (abs150 vs abs132, CN vs AD): elastic-net
0.777 vs 0.800, 0.966 vs 0.880, 0.954 vs 0.937, 0.926 vs 0.909; svm_rbf 0.777 vs
0.766, 0.891 vs 0.686, 0.851 vs 0.800, 0.629 vs 0.617.

## 4. Hypotheses

**Do contrasts add information beyond the absolute values?** It depends on the
problem. CN vs AD: yes — elastic-net AUC 0.881 → 0.906 (AP 0.741 → 0.767, F1
0.417 → 0.466) and SVM-RBF AUC 0.717 → 0.787, with the SVM gain positive in all 4
folds and the elastic-net gain in 3 of 4. CN vs MCI and 3-class: no — small negative
deltas for both model types (3/4 folds negative). The contrasts therefore improve the
problem where the disease signal is strongest and the reference task (KD1) alone is
weak, but do not help the harder MCI distinction or the 3-class problem, where they
fit the training folds better (inner log-loss drops everywhere) yet generalize
slightly worse.

**Does the compact representation keep or improve performance with fewer variables?**
No. compact84 (84 features) loses to abs132 in all 6 problem/model comparisons, by
0.036-0.082 AUC (3/4 or 4/4 folds negative in every case), and to abs150 by
0.037-0.106.
The dropped absolute features of KD2 and KD3 carry discriminative information that
the 18 contrasts do not recover; in CN vs AD the compact representation even loses
most of the contrast benefit (0.821 vs 0.906 for abs150 EN).

**Are the benefits consistent across folds and problems?** Within a problem the sign
is consistent across folds: the CN-vs-AD contrast gain holds in 4/4 SVM folds and 3/4
elastic-net folds, and the CN-vs-MCI / 3-class losses hold in 3/4 folds. Across
problems the effect flips sign, so a single global statement about the value of the
contrasts would be misleading; the benefit is specific to CN vs AD.

## 5. Limitations

- Same fixed-split caveats as the fusion experiment (overlapping held-out AD in every
  validation fold; small AD class).
- The ratio contrasts use KD1 as the reference task; KD1 values are part of both the
  numerator and denominator scales, so contrast coefficients should not be
  interpreted causally.
- Contrasts are raw cross-task differences; no attempt was made to separate task
  difficulty, layout or practice effects.
- Some weakly regularized grid corners (C=10) reach saga's iteration cap on some
  folds; selection is unaffected (they lose by a wide margin) and identical across
  representations.
- One fold (co_vs_ad, elasticnet, compact84, fold 3) has a Youden threshold of "inf"
  in the detail CSV: on that fold's inner OOF no threshold beats the all-negative
  rule, so the Youden-threshold supplementary metrics for that row describe
  all-negative predictions.
- Two documented post-run completions were applied to the artifacts (see
  `src/expcommon/postprocess_first_full_runs.py`): the long-layout proba rows were
  merged to wide layout, and the 3-class balanced-accuracy cells missing from the
  first runs were recomputed from the stored per-subject probabilities (argmax rule),
  each validated against the stored per-class recalls. The runner code was fixed to
  write both directly in future runs.
