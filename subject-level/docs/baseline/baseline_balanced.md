# Balanced Baseline — processed_dataset, Class Weights, Fixed 0.5 Threshold

Baseline experiment on the processed_dataset (25 features) with
class-imbalance handling: balanced class weights / inverse-frequency priors.
No threshold tuning: binary predictions use the fixed default decision
threshold 0.5 (`model.predict`). Run date: 2026-10-13. Local CPU run
(no GPU / paid compute).

## 1. Protocol

- **Samples**: one row per participant; the 25 features are flattened across
  all 11 tasks in feature-major order (columns `{feature}_t{task}`) →
  **275 columns**.
- **Problems (3)** — sample counts (participants):
  - `co_vs_mci` — binary, positive = MCI (141: CO 100 / MCI 41)
  - `co_vs_ad` — binary, positive = AD (116: CO 100 / AD 16)
  - `co_mci_ad` — 3-class CO/MCI/AD (157: 100 / 41 / 16)
- **CV**: participant-level, stratified by group, 5 folds × **20 seeds**
  (fixed before results were seen: `[42, 2026, 1, 0, 7, 123, 1234, 2024,
  2025, 1337, 2, 3, 5, 10, 21, 777, 2020, 2021, 2022, 2023]`) →
  **100 train/test evaluations per (problem, model)**; 3 × 6 × 100 =
  **1800 evaluations** total.
- **Missing data**: 402 / 43,175 cells (0.93%), 45 / 275 columns affected
  (`pupbl_*` of the 16 participants without a pupil baseline, and the 2
  all-NaN task-10 rows). **Median imputation per fold on the training split
  only**, then `StandardScaler` fitted on the training split only.
- **Class weights** (`class_weight: balanced`): Logistic Regression, SVM RBF
  and SVM Linear get scikit-learn balanced class weights; the FNN gets
  balanced `sample_weight` at fit time (MLPClassifier has no `class_weight`
  argument); QDA and GNB get **inverse-frequency priors** computed from the
  training split of each fold (the generative analogue of balanced weights).
- **Decision threshold** (`threshold: 0.5`): binary predictions use the
  fixed default threshold 0.5 (`model.predict`); no threshold tuning. The
  3-class problem uses argmax (no threshold). AUC is unaffected by the
  threshold.
- **Models** (fixed hyperparameters, no tuning): Logistic Regression
  (`max_iter=2000`), SVM RBF (`C=1, gamma='scale', probability=True`),
  SVM Linear (`C=1, probability=True`), FNN (MLP 64-32, relu, adam,
  lr=1e-3, alpha=1e-4, batch=32, max_iter=300), QDA (`solver='eigen',
  shrinkage=0.5`), GNB (`var_smoothing=1e-9`). `random_state = seed` for
  LR / SVM / FNN.
- **Metrics** (per evaluation, on the held-out fold): Acc; Sen
  (sensitivity of the disease class: binary = the positive class, 3-class
  = macro recall); Spec (binary: TN/(TN+FP); 3-class: mean per-class
  TN/(TN+FP)); Precision (binary: disease class; 3-class: macro); AUC
  (binary: ROC-AUC; 3-class: OVR macro); F1 (binary: disease class;
  3-class: macro). Acc is the overall 3-class accuracy.
- **Primary metric**: AUC. **Chance references** (majority-class accuracy):
  CO vs MCI 0.709, CO vs AD 0.862, 3-class 0.637; AUC chance = 0.5.

## 2. Results — mean ± std over 100 evaluations (20 seeds × 5 folds)

### co_vs_mci (positive = MCI)

| Model | Acc | AUC | Sen | Spec | Precision | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.6572±0.0720 | 0.6361±0.0873 | 0.4472±0.1622 | 0.7430±0.0935 | 0.4222±0.1332 | 0.4244±0.1267 |
| svm_rbf | 0.7282±0.0706 | 0.7384±0.0945 | 0.4585±0.1819 | 0.8390±0.0863 | 0.5534±0.1645 | 0.4829±0.1492 |
| svm_linear | 0.6447±0.0736 | 0.5975±0.0968 | 0.4068±0.1721 | 0.7420±0.0937 | 0.3954±0.1389 | 0.3906±0.1363 |
| fnn | 0.7178±0.0763 | 0.6954±0.1013 | 0.4572±0.1590 | 0.8245±0.0886 | 0.5349±0.1787 | 0.4796±0.1429 |
| qda | 0.6962±0.0460 | 0.5535±0.0943 | 0.0932±0.0957 | 0.9435±0.0563 | 0.3493±0.3688 | 0.1401±0.1378 |
| gnb | 0.6301±0.0916 | 0.6429±0.1028 | 0.5910±0.1596 | 0.6455±0.1148 | 0.4141±0.1099 | 0.4808±0.1161 |

### co_vs_ad (positive = AD)

| Model | Acc | AUC | Sen | Spec | Precision | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.8588±0.0539 | 0.8571±0.0996 | 0.3758±0.2886 | 0.9365±0.0517 | 0.4455±0.3286 | 0.3839±0.2636 |
| svm_rbf | 0.8619±0.0266 | 0.8428±0.0903 | 0.0558±0.1330 | 0.9910±0.0218 | 0.1300±0.3157 | 0.0750±0.1753 |
| svm_linear | 0.8687±0.0474 | 0.8638±0.0993 | 0.3483±0.2544 | 0.9520±0.0443 | 0.4977±0.3617 | 0.3861±0.2575 |
| fnn | 0.8569±0.0532 | 0.7141±0.1562 | 0.3067±0.2515 | 0.9450±0.0505 | 0.4257±0.3569 | 0.3374±0.2618 |
| qda | 0.8610±0.0158 | 0.5937±0.2230 | 0.0000±0.0000 | 0.9985±0.0086 | 0.0000±0.0000 | 0.0000±0.0000 |
| gnb | 0.8465±0.0819 | 0.7965±0.1592 | 0.5975±0.3004 | 0.8870±0.0751 | 0.4780±0.2719 | 0.5120±0.2552 |

### co_mci_ad (3-class, macro metrics)

| Model | Acc | AUC | Sen | Spec | Precision | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.5915±0.0753 | 0.6982±0.0715 | 0.4519±0.1000 | 0.7408±0.0486 | 0.4567±0.1258 | 0.4440±0.1025 |
| svm_rbf | 0.6478±0.0735 | 0.7583±0.0706 | 0.4588±0.0764 | 0.7755±0.0495 | 0.4359±0.1201 | 0.4364±0.0828 |
| svm_linear | 0.5736±0.0751 | 0.6913±0.0757 | 0.4265±0.0937 | 0.7208±0.0492 | 0.4374±0.1316 | 0.4212±0.1010 |
| fnn | 0.6471±0.0709 | 0.7008±0.0865 | 0.4751±0.0916 | 0.7650±0.0507 | 0.4815±0.1193 | 0.4681±0.0970 |
| qda | 0.6203±0.0447 | 0.5645±0.1068 | 0.3411±0.0347 | 0.6735±0.0278 | 0.3105±0.1129 | 0.2955±0.0479 |
| gnb | 0.5685±0.0912 | 0.6575±0.0889 | 0.4697±0.1081 | 0.7564±0.0557 | 0.4763±0.1223 | 0.4574±0.1053 |

### Best mean AUC per problem

| Problem | Best model | AUC |
|---|---|---|
| co_vs_mci | SVM RBF | 0.7384±0.0945 |
| co_vs_ad | SVM Linear | 0.8638±0.0993 |
| co_mci_ad | SVM RBF | 0.7583±0.0706 |

## 3. Notes

- **With the fixed 0.5 threshold, probability-calibrated models keep useful
  hard predictions**: SVM RBF reaches Sen 0.459 and F1 0.483 on co_vs_mci;
  GNB has the highest binary Sen (0.591 on co_vs_mci, 0.598 on co_vs_ad) at
  the cost of Spec (0.646 / 0.887). QDA's binary predictions still collapse
  on co_vs_ad (Sen = 0).
- **3-class metrics are macro averages**: Sen/Precision/Spec/F1 are
  averaged over the three classes and AUC is the one-vs-rest macro AUC; no
  class is singled out as the disease class there (unlike the binary
  problems). Acc is the overall 3-class accuracy.
- **AUC is threshold-independent**; class weights and priors barely move
  the ranking. SVM Linear's solution is exactly invariant to class weights
  here (the p > n standardized training folds are linearly separable, where
  the max-margin solution does not depend on class weights); QDA/GNB binary
  AUC is likewise unaffected by the priors, which shift scores by a
  constant and preserve the ROC ranking.
- Accuracy is within one standard deviation of the majority-class chance
  level in every problem; AUC is the informative metric. Large standard
  deviations reflect small test folds (23–33 participants) and the small
  AD class (16 participants).
- The 25 features were screened on the same cohort (Kruskal–Wallis,
  p < 0.1 in ≥ 3 tasks), so CV performance carries in-sample selection
  bias.
- **The archived Youden run** (`outputs/baseline_balanced_youden/`) used
  per-fold Youden-threshold tuning. AUC and all 3-class metrics are
  identical between the two runs; only the binary hard-decision metrics
  (Acc/Sen/Spec/Precision/F1) changed.

## 4. Reproducibility

- Code: `src/begad_norm/baseline/run_baselines.py`
  (+ `aggregate_results.py`), config: `src/begad_norm/baseline/config.yaml`.
- `outputs/baseline_balanced/provenance.json`: git commit, config sha256,
  data-file sha256, package versions (numpy 2.5.4, pandas 3.0.6,
  scikit-learn 1.9.1, scipy 1.18.1), host, command, duration. The recorded
  config sha256 is that of `config.yaml` at re-run time. Data-file sha256
  values are unchanged from the first run of this experiment; the protocol
  changed after the first run (`recall` is now `sen`; the 3-class metrics
  returned to macro after an intermediate AD-vs-rest protocol; Youden
  threshold tuning was replaced by the fixed 0.5 threshold), so the
  experiment was re-run. Archived runs: `outputs/baseline_balanced_macro/`
  (first run, identical metrics, `recall` column),
  `outputs/baseline_balanced_adpos/` (superseded intermediate run with
  AD-vs-rest 3-class metrics), `outputs/baseline_balanced_youden/`
  (superseded run with per-fold Youden threshold tuning; identical AUC and
  3-class metrics, different binary hard metrics). AUC and all 3-class
  metrics are identical across all four runs; the binary hard-decision
  metrics differ only between the Youden run and the fixed-0.5 runs.
- Re-run:
  `uv run python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml`
  (refuses to overwrite existing results), then
  `uv run python -m begad_norm.baseline.aggregate_results --detail outputs/baseline_balanced/results/results_detail.csv --summary outputs/baseline_balanced/results/summary.csv`.
- Smoke runs (6 evaluations each) and two verification mini-runs (6
  evaluations each, 3-class problem) were run first; their detail CSVs
  were deleted after use (`run_log.csv` records them).

## 5. Files

| File | Content |
|---|---|
| `outputs/baseline_balanced/results/results_detail.csv` | 1800 rows, one per (problem, model, seed, fold) with all 6 metrics |
| `outputs/baseline_balanced/results/summary.csv` | mean / std / n_runs per (problem, model, metric) |
| `outputs/baseline_balanced/results/missingness.csv` | missing-cell counts per column |
| `outputs/baseline_balanced/run_log.csv` | run log (smoke + verification runs + full) |
| `outputs/baseline_balanced/full_run.log` | stdout of the full run |
| `outputs/baseline_balanced/provenance.json` | reproducibility record |
| `outputs/baseline_balanced_macro/` | archived first run (identical metrics, `recall` column) |
| `outputs/baseline_balanced_adpos/` | superseded intermediate run with AD-vs-rest 3-class metrics |
| `outputs/baseline_balanced_youden/` | superseded run with per-fold Youden threshold tuning |
