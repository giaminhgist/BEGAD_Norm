# Baseline Experiment — BEGAD Group Classification (participant-wide 275 features)

Classical baselines on one sample per participant: 25 selected eye-tracking
features × 11 tasks = **275 input features**. Local CPU run (no GPU / paid
compute).

## 1. Experiment description

- **Objective**: characterize how well simple baselines separate CO / MCI / AD
  from participant-wide eye-tracking features. Exploratory baseline — no
  mechanism claim, no clinical interpretation.
- **Samples**: one row per participant; the 25 selected features
  (Kruskal–Wallis p < 0.1 in ≥ 3 tasks) are flattened across the 11 tasks in
  feature-major order → 275 columns named `{feature}_t{task}`.
- **Problems (3)** — sample counts (participants):
  - `co_vs_mci` — binary, positive = MCI (141: CO 100 / MCI 41)
  - `co_vs_ad` — binary, positive = AD (116: CO 100 / AD 16)
  - `co_mci_ad` — 3-class CO/MCI/AD (157: 100 / 41 / 16)
- **CV**: participant-level, stratified by group, 4 folds × 10 seeds (1–10)
  → **40 train/test evaluations per (problem, model)**; 720 evaluations total.
  Split method: within each group, permute with
  `numpy.random.default_rng(seed)` and assign round-robin to folds (same
  method as `fold/`, which used seed 42).
- **Missing data**: 402 / 43175 cells (0.93%) are missing — 18/157
  participants have ≥ 1 missing cell, 45/275 columns affected (driven by
  `pupbl_*` of participants without a pupil baseline, and the 2 all-NaN
  task-10 rows). **Median imputation, fitted per fold on the training split
  only.** Details: `missingness.csv`.
- **Preprocessing**: `StandardScaler` on the 275 features, fitted per fold on
  the training split only (after imputation).
- **Models** (fixed hyperparameters, no tuning):

| Model | Hyperparameters |
|---|---|
| Logistic Regression | `max_iter=2000` |
| SVM RBF | `SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)` |
| SVM Linear | `SVC(C=1.0, kernel='linear', probability=True)` |
| FNN (MLP 64-32) | `MLPClassifier(hidden=(64,32), relu, adam, lr=1e-3, alpha=1e-4, batch=32, max_iter=300)` |
| QDA | `solver='eigen', shrinkage=0.5` (p=275 > n per class; svd solver is infeasible) |
| GNB | `var_smoothing=1e-9` |

`random_state = seed` for LR / SVM / FNN. QDA and GNB are deterministic.

- **Metrics** (per evaluation, on the held-out fold):
  - `Acc` — accuracy.
  - `Sen` — binary: recall of the positive class; 3-class: macro recall.
  - `Spec` — binary: TN/(TN+FP) of the negative class; 3-class: macro
    per-class specificity.
  - `AUC` — binary: ROC-AUC of the positive class; 3-class: OVR macro AUC.
  - `F1` — binary: F1 of the positive class; 3-class: macro F1.
- **Primary metric**: AUC. **Chance reference** (majority-class accuracy):
  CO vs MCI 0.709, CO vs AD 0.862, 3-class 0.637.

## 2. Results — mean ± std over 40 evaluations (10 seeds × 4 folds)

### CO vs MCI (positive = MCI)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.6680±0.0631 | 0.4020±0.1537 | 0.7780±0.0915 | 0.6632±0.0902 | 0.4030±0.1275 |
| SVM RBF | 0.7143±0.0170 | 0.0250±0.0494 | 0.9970±0.0140 | 0.7334±0.0818 | 0.0435±0.0833 |
| SVM Linear | 0.6383±0.0713 | 0.4023±0.1566 | 0.7360±0.1000 | 0.6248±0.1028 | 0.3843±0.1249 |
| FNN (MLP 64-32) | 0.7384±0.0593 | 0.4286±0.1634 | 0.8660±0.0750 | 0.7052±0.0825 | 0.4747±0.1483 |
| QDA | 0.6981±0.0417 | 0.0857±0.0804 | 0.9490±0.0462 | 0.5678±0.0815 | 0.1345±0.1212 |
| GNB | 0.6534±0.0911 | 0.5748±0.1540 | 0.6860±0.1074 | 0.6462±0.1016 | 0.4909±0.1194 |

### CO vs AD (positive = AD)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.8698±0.0394 | 0.2687±0.2147 | 0.9660±0.0346 | 0.8518±0.0868 | 0.3309±0.2490 |
| SVM RBF | 0.8621±0.0000 | 0.0000±0.0000 | 1.0000±0.0000 | 0.8588±0.1026 | 0.0000±0.0000 |
| SVM Linear | 0.8733±0.0458 | 0.3563±0.2184 | 0.9560±0.0403 | 0.8578±0.0919 | 0.4166±0.2338 |
| FNN (MLP 64-32) | 0.8741±0.0439 | 0.2812±0.2414 | 0.9690±0.0430 | 0.7305±0.1617 | 0.3374±0.2677 |
| QDA | 0.8612±0.0055 | 0.0000±0.0000 | 0.9990±0.0063 | 0.6045±0.1742 | 0.0000±0.0000 |
| GNB | 0.8586±0.0702 | 0.5625±0.2640 | 0.9060±0.0699 | 0.7925±0.1447 | 0.5189±0.2214 |

### CO / MCI / AD (3-class)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.6120±0.0772 | 0.4502±0.0941 | 0.7394±0.0523 | 0.6976±0.0688 | 0.4495±0.1027 |
| SVM RBF | 0.6415±0.0188 | 0.3429±0.0190 | 0.6746±0.0147 | 0.7560±0.0729 | 0.2810±0.0346 |
| SVM Linear | 0.5840±0.0682 | 0.4245±0.0899 | 0.7261±0.0477 | 0.6982±0.0807 | 0.4216±0.0952 |
| FNN (MLP 64-32) | 0.6561±0.0631 | 0.4558±0.0686 | 0.7568±0.0415 | 0.6807±0.0785 | 0.4525±0.0811 |
| QDA | 0.6300±0.0330 | 0.3448±0.0270 | 0.6754±0.0212 | 0.5677±0.0907 | 0.2971±0.0410 |
| GNB | 0.5593±0.0781 | 0.4485±0.1009 | 0.7450±0.0477 | 0.6560±0.0865 | 0.4397±0.1009 |

**Best mean AUC per problem**: CO vs MCI — SVM RBF
(0.7334±0.0818); CO vs AD —
SVM RBF (0.8588±0.1026);
3-class — SVM RBF
(0.7560±0.0729).

## 3. Reproducibility

- Code: `src/baseline/run_baselines.py` (+ `aggregate_results.py`),
  config: `src/baseline/config.yaml`.
- `provenance.json`: git commit, config sha256, data-file sha256, package
  versions, host, command.
- Re-run: `python src/baseline/run_baselines.py --config src/baseline/config.yaml`
  (refuses to overwrite existing results), then
  `python src/baseline/aggregate_results.py`.
- Development notes (also in `run_log.csv`): first smoke attempt failed on
  QDA with `solver='svd'` (p > n); fixed with `solver='eigen', shrinkage=0.5`.

## 4. Files

| File | Content |
|---|---|
| `results/results_detail.csv` | 720 rows, one per (problem, model, seed, fold) with all 5 metrics |
| `results/summary.csv` | mean / std / n_runs per (problem, model, metric) |
| `results/smoke_detail.csv` | 6-row smoke run |
| `missingness.csv` | missing-cell counts per column of the 275-feature matrix |
| `run_log.csv` | run log incl. the failed QDA smoke attempt |
| `full_run.log` | stdout of the full run |
| `provenance.json` | reproducibility record |

## 5. Limitations

- AD is small (16 participants) and classes are imbalanced (CO 63.7–86.2%) →
  macro metrics and AUC are the informative ones.
- p (275) > n per class → heavily regularized models (QDA shrinkage); all
  models are at risk of overfitting the training folds.
- Missing cells are median-imputed (18/157 participants affected, 0.93% of
  cells).
- No hyperparameter tuning — fixed config for all models.
- Single site, single session (2025-03-05); no external validation.
- The 25 features were selected on the same dataset (p < 0.1 screen) —
  selection bias is possible; results are exploratory.
