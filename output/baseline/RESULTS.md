# Baseline results: 6 classifiers on the processed 37-feature dataset (RAW157)

Run `2026-10-28T09:27Z`. Detail: `results/results_detail.csv` (720 evaluations),
summary: `results/summary.csv`, confusion counts: `results/confusion_summary.csv`,
provenance: `provenance.json`.

## 1. Setup

- **Data**: `processed_dataset/processed_dataset_v1_wide.csv` — 157 subjects (CN 100 / MCI 41 / AD 16),
  96 features (24 RAW157 features x 4 tasks: KD1/KD2/KD3/READING), complete-case (no missing values).
- **Split (fixed)**: from `processed_dataset/splits/` — 4 held-out AD subjects are appended to every
  validation fold and never enter training; the remaining 153 subjects form 4 stratified folds.
  Training per fold: 115 subjects for the 3-class problem (75 CN / 31-30 MCI / 9 AD); validation:
  42-43 subjects with 7 AD (3 from the fold + 4 held-out).
- **Models**: LogisticRegression, SVC-RBF, RandomForest, LightGBM, GaussianNB, MLP.
- **Imbalance handling**: balanced class weights (LR/SVM/RF/LGBM), inverse-frequency priors (GNB),
  sample weights (MLP) — all computed from the training split of each fold.
- **Protocol**: StandardScaler fitted on the training split only; no imputation, no hyperparameter
  tuning, fixed decision threshold 0.5; 10 model seeds `[42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337]`
  x 4 folds = 40 evaluations per (problem, model).
- **Metrics**: 3-class = accuracy, macro sensitivity/precision/F1, mean per-class specificity,
  one-vs-rest macro AUC, plus per-class recall/precision/F1; binary = positive-class metrics
  (MCI or AD as positive).

## 2. Results summary (mean ± std over 40 evaluations)

### 3-class CN/MCI/AD

| Model | Acc | Sen (macro) | Prec (macro) | Spec | AUC (OVR) | F1 (macro) | Recall AD | F1 AD |
|---|---|---|---|---|---|---|---|---|
| lightgbm | 0.599 ± 0.037 | 0.464 ± 0.044 | 0.615 ± 0.103 | 0.755 ± 0.029 | 0.682 ± 0.026 | 0.469 ± 0.052 | 0.189 ± 0.087 | 0.292 ± 0.113 |
| svm_rbf | 0.610 ± 0.033 | 0.367 ± 0.058 | 0.324 ± 0.218 | 0.695 ± 0.044 | 0.700 ± 0.071 | 0.304 ± 0.096 | 0.036 ± 0.062 | 0.062 ± 0.108 |
| mlp | 0.567 ± 0.050 | 0.436 ± 0.041 | 0.558 ± 0.106 | 0.735 ± 0.025 | 0.652 ± 0.048 | 0.446 ± 0.044 | 0.214 ± 0.085 | 0.311 ± 0.096 |
| random_forest | 0.562 ± 0.046 | 0.434 ± 0.044 | 0.549 ± 0.108 | 0.740 ± 0.029 | 0.710 ± 0.043 | 0.436 ± 0.051 | 0.182 ± 0.084 | 0.264 ± 0.097 |
| logistic_regression | 0.533 ± 0.039 | 0.424 ± 0.038 | 0.452 ± 0.043 | 0.737 ± 0.019 | 0.673 ± 0.036 | 0.425 ± 0.039 | 0.214 ± 0.071 | 0.268 ± 0.071 |
| gaussian_nb | 0.462 ± 0.035 | 0.418 ± 0.032 | 0.423 ± 0.040 | 0.719 ± 0.021 | 0.611 ± 0.040 | 0.407 ± 0.030 | 0.321 ± 0.118 | 0.307 ± 0.113 |

Per-class recall (mean ± std): LGBM CN 0.788 ± 0.054 / MCI 0.415 ± 0.070 / AD 0.189 ± 0.087;
RF CN 0.744 / MCI 0.375 / AD 0.182; LR CN 0.690 / MCI 0.368 / AD 0.214; GNB CN 0.520 /
MCI 0.411 / AD 0.321; SVM CN 0.990 / MCI 0.075 / AD 0.036 (predicts almost everything as CN).

### Binary CN vs MCI (positive = MCI)

| Model | Acc | Sen (MCI) | Prec | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| svm_rbf | 0.716 ± 0.035 | 0.096 ± 0.064 | 0.667 ± 0.408 | 0.970 ± 0.033 | 0.647 ± 0.062 | 0.162 ± 0.103 |
| lightgbm | 0.684 ± 0.044 | 0.439 ± 0.091 | 0.460 ± 0.056 | 0.784 ± 0.064 | 0.600 ± 0.058 | 0.445 ± 0.060 |
| random_forest | 0.675 ± 0.047 | 0.403 ± 0.118 | 0.431 ± 0.087 | 0.785 ± 0.044 | 0.643 ± 0.090 | 0.414 ± 0.101 |
| logistic_regression | 0.660 ± 0.026 | 0.461 ± 0.063 | 0.423 ± 0.035 | 0.740 ± 0.045 | 0.608 ± 0.034 | 0.440 ± 0.041 |
| mlp | 0.656 ± 0.061 | 0.343 ± 0.088 | 0.404 ± 0.111 | 0.783 ± 0.076 | 0.586 ± 0.065 | 0.366 ± 0.090 |
| gaussian_nb | 0.575 ± 0.064 | 0.534 ± 0.092 | 0.352 ± 0.067 | 0.590 ± 0.087 | 0.596 ± 0.096 | 0.423 ± 0.071 |

### Binary CN vs AD (positive = AD)

| Model | Acc | Sen (AD) | Prec | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.859 ± 0.035 | 0.607 ± 0.156 | 0.747 ± 0.159 | 0.930 ± 0.052 | 0.889 ± 0.061 | 0.649 ± 0.091 |
| lightgbm | 0.832 ± 0.047 | 0.307 ± 0.177 | 0.810 ± 0.260 | 0.979 ± 0.027 | 0.860 ± 0.025 | 0.423 ± 0.205 |
| random_forest | 0.820 ± 0.026 | 0.232 ± 0.094 | 0.835 ± 0.212 | 0.984 ± 0.020 | 0.899 ± 0.056 | 0.353 ± 0.117 |
| mlp | 0.816 ± 0.031 | 0.307 ± 0.104 | 0.740 ± 0.220 | 0.959 ± 0.037 | 0.825 ± 0.082 | 0.415 ± 0.110 |
| svm_rbf | 0.789 ± 0.014 | 0.036 ± 0.062 | 0.250 ± 0.433 | 1.000 ± 0.000 | 0.864 ± 0.038 | 0.063 ± 0.108 |
| gaussian_nb | 0.719 ± 0.054 | 0.536 ± 0.062 | 0.402 ± 0.074 | 0.770 ± 0.052 | 0.676 ± 0.121 | 0.458 ± 0.072 |

## 3. Aggregated confusion matrices (summed over 10 seeds x 4 folds)

CN vs AD, LogisticRegression (1,280 validation samples): CN→CN 930, CN→AD 70, AD→CN 110, AD→AD 170
(AD recall 61%, AD precision 71%).
3-class, LightGBM (1,690 samples): CN→CN 788 / CN→MCI 206 / CN→AD 6; MCI→CN 228 / MCI→MCI 171 /
MCI→AD 11; AD→CN 83 / AD→MCI 144 / AD→AD 53 (AD is mostly confused with MCI).
3-class, GaussianNB: AD→AD 90 / AD→MCI 160 / AD→CN 30 — highest AD recall (0.32) among the six
models, at the cost of large CN misclassification.

## 4. Key observations

- **CN vs AD is the most learnable problem**: LR reaches AUC 0.889 and F1 0.649 with the highest AD
  recall (0.61); RF has the best AUC (0.899) but its 0.5-threshold sensitivity is only 0.23.
- **SVM-RBF ranks well but does not predict disease classes at threshold 0.5** (3-class recall AD
  0.04, co_vs_ad sensitivity 0.04, co_vs_mci sensitivity 0.10) despite AUC 0.70 / 0.86 / 0.65: its
  sigmoid calibration compresses probabilities, so almost no sample crosses 0.5. Its high accuracy
  (0.61 in 3-class) is almost entirely CN recall (0.99). Do not interpret its accuracy as usable
  detection; threshold tuning would be required.
- **GNB is the most AD-sensitive model** (3-class recall AD 0.32; co_vs_ad sensitivity 0.54) thanks
  to inverse-frequency priors, but at weak overall accuracy (3-class acc 0.46).
- CN vs MCI is the hardest problem (best AUC 0.65, all models' macro F1 <= 0.45).
- LR/SVM/GNB are deterministic: their 10 seed-rows are identical; the reported std for those models
  is fold-to-fold variation only. RF/LGBM/MLP vary across seeds.

## 5. Caveats

- The same 4 held-out AD subjects appear in **every** validation fold, so the 40 evaluations per
  (problem, model) are **not independent** and pooled means overweight those 4 subjects; the 4
  held-out AD are also **never seen in training**. Report per-fold metrics when comparing models.
- Validation AD n is small (7 per fold, 3 of which are new to the model) and the training AD n is 9;
  AD estimates have high variance (see std columns).
- No hyperparameter tuning and no threshold tuning (0.5 fixed); results are a lower bound for tuned
  models.
- Models are baselines on the 24 complete features; 9 further 37-feature candidates were dropped for
  subject completeness (see `processed_dataset/README.md`).

## 6. Reproducibility

- Code: `src/baseline/` (`run_baselines.py`, `config.yaml`, `aggregate_results.py`); rerun with
  `PYTHONPATH=src .venv/bin/python -m baseline.run_baselines --config src/baseline/config.yaml`
  (refuses to overwrite existing detail results).
- Recorded in `provenance.json`: git commit `ffa36c7`, config sha256, data sha256 (wide CSV, fold
  assignments, held-out AD list), package versions (Python 3.12.3, numpy 2.5.4, pandas 3.1.0,
  scikit-learn 1.9.1, scipy 1.18.1, lightgbm 4.7.0).
