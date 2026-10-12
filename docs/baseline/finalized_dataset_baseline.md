# Finalized Dataset Baseline — 25 Features × 11 Tasks, 20 Seeds

Dedicated baseline experiment on the final working feature set
`finalized_dataset/` (25 selected features, Kruskal–Wallis p < 0.1 in ≥ 3
tasks). Run date: 2026-10-12. Local CPU run (no GPU / paid compute).

Dataset choice: see `docs/baseline/baseline_dataset_comparison.md` (all72 vs
selected25 vs extended39). The 3-dataset comparison run is archived in
`docs/baseline/baseline_experiment.md` and
`outputs/baseline/comparison_3datasets/`.

## 1. Protocol

- **Samples**: one row per participant; the 25 features are flattened across
  all 11 tasks in feature-major order (columns `{feature}_t{task}`) →
  **275 columns**.
- **Problems (3)** — sample counts (participants):
  - `co_vs_mci` — binary, positive = MCI (141: CO 100 / MCI 41)
  - `co_vs_ad` — binary, positive = AD (116: CO 100 / AD 16)
  - `co_mci_ad` — 3-class CO/MCI/AD (157: 100 / 41 / 16)
- **CV**: participant-level, stratified by group, 4 folds × **20 seeds** →
  **80 train/test evaluations per (problem, model)**; 3 × 6 × 80 =
  **1440 evaluations** total. Split method: within each group, permute with
  `numpy.random.default_rng(seed)` and assign round-robin to folds.
- **Seeds** (fixed before results were seen): the 10 seeds of the comparison
  run — `[42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337]` — plus a 10-seed
  extension — `[2, 3, 5, 10, 21, 777, 2020, 2021, 2022, 2023]`.
- **Missing data**: 402 / 43,175 cells (0.93%), 45 / 275 columns affected
  (`pupbl_*` of the 16 participants without a pupil baseline, and the 2
  all-NaN task-10 rows). **Median imputation per fold on the training split
  only**, then `StandardScaler` fitted on the training split only.
- **Models** (fixed hyperparameters, no tuning): Logistic Regression
  (`max_iter=2000`), SVM RBF (`C=1, gamma='scale', probability=True`),
  SVM Linear (`C=1, probability=True`), FNN (MLP 64-32, relu, adam,
  lr=1e-3, alpha=1e-4, batch=32, max_iter=300), QDA (`solver='eigen',
  shrinkage=0.5`), GNB (`var_smoothing=1e-9`). `random_state = seed` for
  LR / SVM / FNN.
- **Metrics** (per evaluation, on the held-out fold): Acc; Sen (binary:
  recall of the positive class, 3-class: macro recall); Spec (binary:
  TN/(TN+FP), 3-class: macro per-class specificity); AUC (binary: ROC-AUC,
  3-class: OVR macro); F1 (binary: positive class, 3-class: macro).
- **Primary metric**: AUC. **Chance references** (majority-class accuracy):
  CO vs MCI 0.709, CO vs AD 0.862, 3-class 0.637; AUC chance = 0.5.

## 2. Results — mean ± std over 80 evaluations (20 seeds × 4 folds)

### co_vs_mci (positive = MCI)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.6822±0.0628 | 0.4276±0.1459 | 0.7870±0.0824 | 0.6659±0.0897 | 0.4318±0.1183 |
| svm_rbf | 0.7154±0.0212 | 0.0359±0.0638 | 0.9940±0.0202 | 0.7482±0.0856 | 0.0607±0.1035 |
| svm_linear | 0.6566±0.0724 | 0.4336±0.1426 | 0.7485±0.0912 | 0.6147±0.1081 | 0.4193±0.1131 |
| fnn | 0.7433±0.0631 | 0.4578±0.1545 | 0.8610±0.0779 | 0.7173±0.0801 | 0.5011±0.1318 |
| qda | 0.7058±0.0373 | 0.0858±0.0689 | 0.9600±0.0455 | 0.5880±0.0999 | 0.1392±0.1071 |
| gnb | 0.6575±0.0815 | 0.5780±0.1249 | 0.6905±0.0981 | 0.6534±0.0928 | 0.4964±0.1025 |

### co_vs_ad (positive = AD)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.8612±0.0459 | 0.2656±0.2227 | 0.9565±0.0455 | 0.8510±0.0871 | 0.3145±0.2399 |
| svm_rbf | 0.8621±0.0000 | 0.0000±0.0000 | 1.0000±0.0000 | 0.8492±0.1003 | 0.0000±0.0000 |
| svm_linear | 0.8672±0.0494 | 0.3594±0.2173 | 0.9485±0.0486 | 0.8561±0.0862 | 0.4092±0.2212 |
| fnn | 0.8642±0.0404 | 0.2437±0.2352 | 0.9635±0.0398 | 0.7146±0.1758 | 0.2898±0.2530 |
| qda | 0.8608±0.0066 | 0.0000±0.0000 | 0.9985±0.0076 | 0.5989±0.1982 | 0.0000±0.0000 |
| gnb | 0.8556±0.0685 | 0.5563±0.2352 | 0.9035±0.0682 | 0.7897±0.1510 | 0.5156±0.1982 |

### co_mci_ad (3-class)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.6137±0.0630 | 0.4427±0.0837 | 0.7399±0.0427 | 0.6956±0.0665 | 0.4410±0.0965 |
| svm_rbf | 0.6453±0.0196 | 0.3458±0.0203 | 0.6770±0.0152 | 0.7530±0.0679 | 0.2851±0.0367 |
| svm_linear | 0.5888±0.0666 | 0.4338±0.0867 | 0.7301±0.0434 | 0.6979±0.0649 | 0.4318±0.0947 |
| fnn | 0.6629±0.0647 | 0.4653±0.0786 | 0.7623±0.0458 | 0.6891±0.0819 | 0.4634±0.0935 |
| qda | 0.6322±0.0353 | 0.3480±0.0300 | 0.6780±0.0229 | 0.5794±0.0785 | 0.3017±0.0455 |
| gnb | 0.5804±0.0753 | 0.4645±0.0971 | 0.7554±0.0475 | 0.6628±0.0887 | 0.4576±0.0988 |

### Best mean AUC per problem

| Problem | Best model | AUC |
|---|---|---|
| co_vs_mci | SVM RBF | 0.7482±0.0856 |
| co_vs_ad | SVM Linear | 0.8561±0.0862 |
| co_mci_ad | SVM RBF | 0.7530±0.0679 |

## 3. Notes

- The 20-seed estimates agree with the 10-seed comparison run on the same
  feature set (e.g. co_vs_ad SVM Linear 0.8518±0.0885 at 10 seeds vs
  0.8561±0.0862 at 20 seeds); extending the seeds did not change the ranking
  of models or problems.
- SVM RBF tops AUC on co_vs_mci and the 3-class problem, but its binary
  predictions collapse to the majority class (Sen ≈ 0); the ranking comes
  from probability scores, not hard predictions.
- Accuracy is at or below the majority-class chance level in every problem;
  AUC is the informative metric. Large standard deviations reflect small test
  folds (29–41 participants) and the small AD class (16 participants).
- The 25 features were screened on the same cohort (Kruskal–Wallis,
  p < 0.1 in ≥ 3 tasks), so CV performance carries in-sample selection bias;
  a nested-selection run would be needed for an unbiased estimate against
  the full 72-feature set (see
  `docs/baseline/baseline_dataset_comparison.md`).

## 4. Reproducibility

- Code: `src/begad_norm/baseline/run_baselines.py` (+ `aggregate_results.py`),
  config: `src/begad_norm/baseline/config.yaml`.
- `outputs/baseline/provenance.json`: git commit, config sha256, data-file
  sha256, package versions (numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1,
  scipy 1.18.1), host, command, duration.
- Re-run:
  `uv run python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml`
  (refuses to overwrite existing results), then
  `uv run python -m begad_norm.baseline.aggregate_results`.
- A smoke run (6 evaluations) was run first and deleted after use
  (`run_log.csv` records it).

## 5. Files

| File | Content |
|---|---|
| `outputs/baseline/results/results_detail.csv` | 1440 rows, one per (problem, model, seed, fold) with all 5 metrics |
| `outputs/baseline/results/summary.csv` | mean / std / n_runs per (problem, model, metric) |
| `outputs/baseline/results/missingness.csv` | missing-cell counts per column of the 275-column matrix |
| `outputs/baseline/run_log.csv` | run log (smoke + full) |
| `outputs/baseline/full_run.log` | stdout of the full run |
| `outputs/baseline/provenance.json` | reproducibility record |
