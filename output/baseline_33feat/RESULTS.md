# Baseline results: 6 classifiers on the 33-feature RAW157 dataset (157 subjects, imputed)

Run `2026-10-28T19:09Z`. Detail: `results/results_detail.csv` (720 evaluations),
summary: `results/summary.csv`, confusion counts: `results/confusion_summary.csv`,
provenance: `provenance.json`.

## 1. Setup

- **Data**: `processed_dataset/processed_dataset_33feat_wide.csv` — 157 subjects (CN 100 / MCI 41 / AD 16),
  132 features (33 RAW157 features x 4 tasks: KD1/KD2/KD3/READING). The 9 features dropped from the
  24-feature version (D017, D022, D025, D026, D030, D034, D035, D036, D037) are included; their
  unsupported cells (51 of 20724, 0.25%, spread over 20 subjects) are left as NaN and **imputed per
  fold with the per-column median fitted on the training split only** (SimpleImputer before
  StandardScaler; no value from validation or held-out AD leaks into training).
- **Split (fixed)**: identical to the 24-feature baseline, from `processed_dataset/splits/` — 4
  held-out AD subjects (BEGAD-001, BEGAD-005, BEGAD-006, BEGAD-016) are appended to every validation
  fold and never enter training; the remaining 153 subjects form 4 stratified folds (seed 42).
  Training per fold: 114-115 subjects for the 3-class problem (75 CN / 30-31 MCI / 9 AD); validation:
  42-43 subjects with 7 AD (3 from the fold + 4 held-out).
- **Models**: LogisticRegression, SVC-RBF, RandomForest, LightGBM, GaussianNB, MLP.
- **Imbalance handling**: balanced class weights (LR/SVM/RF/LGBM), inverse-frequency priors (GNB),
  sample weights (MLP) — all computed from the training split of each fold.
- **Protocol**: per-fold median imputation then StandardScaler, both fitted on the training split
  only; no hyperparameter tuning, fixed decision threshold 0.5; 10 model seeds
  `[42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337]` x 4 folds = 40 evaluations per (problem, model).
- **Metrics**: 3-class = accuracy, macro sensitivity/precision/F1, mean per-class specificity,
  one-vs-rest macro AUC, plus per-class recall/precision/F1; binary = positive-class metrics
  (MCI or AD as positive).

Because the subject set, splits and seeds are identical to the 24-feature baseline, the two runs are
**paired evaluation by evaluation**; section 3 reports the paired deltas (33 features minus 24).

## 2. Results summary (mean ± std over 40 evaluations)

### 3-class CN/MCI/AD

| Model | Acc | Sen (macro) | Prec (macro) | Spec | AUC (OVR) | F1 (macro) | Recall AD | F1 AD |
|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.562 ± 0.023 | 0.467 ± 0.021 | 0.505 ± 0.011 | 0.756 ± 0.025 | 0.705 ± 0.025 | 0.473 ± 0.016 | 0.286 ± 0.000 | 0.356 ± 0.013 |
| svm_rbf | 0.598 ± 0.027 | 0.372 ± 0.039 | 0.272 ± 0.076 | 0.706 ± 0.040 | 0.705 ± 0.092 | 0.310 ± 0.063 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| random_forest | 0.607 ± 0.047 | 0.469 ± 0.064 | 0.584 ± 0.110 | 0.761 ± 0.036 | 0.741 ± 0.050 | 0.469 ± 0.071 | 0.182 ± 0.096 | 0.276 ± 0.127 |
| lightgbm | 0.632 ± 0.045 | 0.473 ± 0.050 | 0.538 ± 0.135 | 0.773 ± 0.030 | 0.732 ± 0.043 | 0.462 ± 0.061 | 0.125 ± 0.112 | 0.186 ± 0.153 |
| gaussian_nb | 0.544 ± 0.013 | 0.479 ± 0.031 | 0.495 ± 0.037 | 0.758 ± 0.015 | 0.618 ± 0.033 | 0.471 ± 0.027 | 0.286 ± 0.101 | 0.325 ± 0.108 |
| mlp | 0.603 ± 0.051 | 0.465 ± 0.056 | 0.550 ± 0.119 | 0.762 ± 0.032 | 0.698 ± 0.052 | 0.462 ± 0.061 | 0.171 ± 0.091 | 0.253 ± 0.120 |

Per-class recall (mean ± std): LGBM CN 0.852 ± 0.080 / MCI 0.442 ± 0.080 / AD 0.125 ± 0.112;
MLP CN 0.795 / MCI 0.427 / AD 0.171; RF CN 0.799 / MCI 0.425 / AD 0.182; LR CN 0.700 /
MCI 0.414 / AD 0.286; GNB CN 0.620 / MCI 0.532 / AD 0.286; SVM CN 0.940 / MCI 0.175 /
AD 0.000 (predicts almost everything as CN).

### Binary CN vs MCI (positive = MCI)

| Model | Acc | Sen (MCI) | Prec | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.688 ± 0.021 | 0.486 ± 0.053 | 0.468 ± 0.035 | 0.770 ± 0.044 | 0.670 ± 0.034 | 0.474 ± 0.030 |
| svm_rbf | 0.752 ± 0.012 | 0.289 ± 0.144 | 0.740 ± 0.151 | 0.940 ± 0.045 | 0.719 ± 0.085 | 0.379 ± 0.142 |
| random_forest | 0.712 ± 0.053 | 0.465 ± 0.192 | 0.496 ± 0.082 | 0.812 ± 0.057 | 0.697 ± 0.121 | 0.468 ± 0.130 |
| lightgbm | 0.744 ± 0.063 | 0.486 ± 0.101 | 0.592 ± 0.118 | 0.849 ± 0.078 | 0.687 ± 0.099 | 0.525 ± 0.085 |
| gaussian_nb | 0.653 ± 0.061 | 0.607 ± 0.107 | 0.433 ± 0.073 | 0.670 ± 0.071 | 0.625 ± 0.099 | 0.504 ± 0.081 |
| mlp | 0.705 ± 0.043 | 0.437 ± 0.112 | 0.502 ± 0.088 | 0.814 ± 0.067 | 0.668 ± 0.065 | 0.458 ± 0.074 |

### Binary CN vs AD (positive = AD)

| Model | Acc | Sen (AD) | Prec | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.875 ± 0.044 | 0.607 ± 0.186 | 0.817 ± 0.185 | 0.950 ± 0.052 | 0.891 ± 0.066 | 0.671 ± 0.129 |
| svm_rbf | 0.781 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.874 ± 0.037 | 0.000 ± 0.000 |
| random_forest | 0.817 ± 0.039 | 0.207 ± 0.139 | 0.787 ± 0.365 | 0.988 ± 0.018 | 0.914 ± 0.061 | 0.316 ± 0.188 |
| lightgbm | 0.806 ± 0.050 | 0.200 ± 0.152 | 0.654 ± 0.428 | 0.976 ± 0.031 | 0.868 ± 0.035 | 0.296 ± 0.212 |
| gaussian_nb | 0.727 ± 0.041 | 0.393 ± 0.062 | 0.387 ± 0.083 | 0.820 ± 0.045 | 0.639 ± 0.102 | 0.388 ± 0.067 |
| mlp | 0.817 ± 0.030 | 0.279 ± 0.096 | 0.784 ± 0.224 | 0.968 ± 0.037 | 0.813 ± 0.085 | 0.393 ± 0.107 |

## 3. Paired comparison with the 24-feature baseline (delta = 33f minus 24f)

Identical 157 subjects, identical splits and seeds; only the feature set differs (plus 51 imputed
cells). Mean ± std over the 40 paired (seed, fold) evaluations; descriptive, not significance-tested.

| Problem | Model | Acc delta | AUC delta |
|---|---|---|---|
| 3-class | logistic_regression | +0.029 ± 0.020 | +0.032 ± 0.015 |
| 3-class | svm_rbf | -0.012 ± 0.012 | +0.004 ± 0.034 |
| 3-class | random_forest | +0.045 ± 0.041 | +0.031 ± 0.023 |
| 3-class | lightgbm | +0.033 ± 0.054 | +0.051 ± 0.029 |
| 3-class | gaussian_nb | +0.083 ± 0.040 | +0.006 ± 0.025 |
| 3-class | mlp | +0.035 ± 0.059 | +0.047 ± 0.051 |
| CN vs MCI | logistic_regression | +0.028 ± 0.035 | +0.062 ± 0.044 |
| CN vs MCI | svm_rbf | +0.036 ± 0.032 | +0.072 ± 0.036 |
| CN vs MCI | random_forest | +0.037 ± 0.034 | +0.054 ± 0.038 |
| CN vs MCI | lightgbm | +0.060 ± 0.049 | +0.087 ± 0.073 |
| CN vs MCI | gaussian_nb | +0.078 ± 0.037 | +0.029 ± 0.012 |
| CN vs MCI | mlp | +0.049 ± 0.057 | +0.083 ± 0.051 |
| CN vs AD | logistic_regression | +0.016 ± 0.016 | +0.003 ± 0.017 |
| CN vs AD | svm_rbf | -0.008 ± 0.014 | +0.010 ± 0.021 |
| CN vs AD | random_forest | -0.002 ± 0.024 | +0.015 ± 0.024 |
| CN vs AD | lightgbm | -0.026 ± 0.025 | +0.008 ± 0.028 |
| CN vs AD | gaussian_nb | +0.008 ± 0.079 | -0.036 ± 0.026 |
| CN vs AD | mlp | +0.001 ± 0.046 | -0.012 ± 0.079 |

The 9 added features help most in **3-class** and **CN vs MCI** (all six models gain accuracy in
CN vs MCI; AUC gains of 0.03-0.09). CN vs AD is essentially unchanged.

## 4. Aggregated confusion matrices (summed over 10 seeds x 4 folds)

CN vs AD, LogisticRegression (1,280 validation samples): CN→CN 950, CN→AD 50, AD→CN 110, AD→AD 170
(AD recall 61%, AD precision 77%).
3-class, LightGBM (1,690 samples): CN→CN 852 / CN→MCI 127 / CN→AD 21; MCI→CN 220 / MCI→MCI 181 /
MCI→AD 9; AD→CN 78 / AD→MCI 167 / AD→AD 35 (AD is mostly confused with MCI).
3-class, GaussianNB: AD→AD 80 / AD→MCI 160 / AD→CN 40 — best MCI recall (0.53) and tied-best AD
recall (0.29, with LR) among the six models, at the cost of large CN misclassification.
3-class, SVM-RBF: AD→AD 0 — the model never predicts AD at threshold 0.5.

## 5. Key observations

- **CN vs AD is the most learnable problem**: LR reaches AUC 0.891 and the best F1 (0.671) with AD
  sensitivity 0.61 at threshold 0.5; RF has the best AUC (0.914) but its 0.5-threshold sensitivity is
  only 0.21.
- **The 9 added features improve 3-class and CN vs MCI across models** (paired deltas, section 3):
  every model gains CN vs MCI accuracy and AUC, 5 of 6 gain 3-class accuracy, and all six gain
  3-class AUC. CN vs AD is flat.
- **SVM-RBF does not predict disease classes at threshold 0.5** (3-class recall AD 0.000, co_vs_ad
  sensitivity 0.000, co_vs_mci 0.29) despite AUC 0.71 / 0.87 / 0.72: its sigmoid calibration compresses
  probabilities below 0.5 for the positive classes. Its high accuracy is almost entirely CN recall.
  Do not interpret its accuracy as usable detection; threshold tuning would be required.
- **LR and GNB tie for the best 3-class AD recall (0.286)**; GNB additionally has the best MCI recall
  (0.53) via inverse-frequency priors. GNB's CN vs AD AUC dropped slightly with the added features
  (paired -0.036).
- LR/SVM/GNB are deterministic: their 10 seed-rows are identical; the reported std for those models
  is fold-to-fold variation only. RF/LGBM/MLP vary across seeds.

## 6. Caveats

- The same 4 held-out AD subjects appear in **every** validation fold, so the 40 evaluations per
  (problem, model) are **not independent** and pooled means overweight those 4 subjects; the 4
  held-out AD are also **never seen in training**. Report per-fold metrics when comparing models.
- Validation AD n is small (7 per fold, 3 of which are new to the model) and the training AD n is 9;
  AD estimates have high variance (see std columns).
- 51 cells (0.25%) are imputed with the per-column training median; with such a small share, the
  choice of imputer is expected to matter little, but the imputation is part of the pipeline and
  was not ablated here.
- Paired deltas in section 3 are descriptive; no statistical tests were run.
- No hyperparameter tuning and no threshold tuning (0.5 fixed); results are a lower bound for tuned
  models.

## 7. Reproducibility

- Code: `src/baseline/` (`run_baselines.py`, `config_33feat.yaml`, `aggregate_results.py`); rerun with
  `PYTHONPATH=src .venv/bin/python -m baseline.run_baselines --config src/baseline/config_33feat.yaml`
  (refuses to overwrite existing detail results).
- Dataset builder: `processed_dataset/make_33feat_dataset.py` (157 subjects, 51 NaN cells).
- Recorded in `provenance.json`: git commit `56834bd` (working tree dirty: the 33-feature dataset and
  config files were not yet committed at run time), config sha256 `3679074a...`, data sha256 (wide
  CSV `cbcecbc7...`, fold assignments, held-out AD list), package versions (Python 3.12.3,
  numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, scipy 1.18.1, lightgbm 4.7.0).
