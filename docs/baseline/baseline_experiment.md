# Baseline Experiment — BEGAD Group Classification (participant-wide features × 11 tasks)

Classical baselines on one sample per participant: each dataset's features are
flattened across all 11 tasks in feature-major order (columns
`{feature}_t{task}`), giving 792 / 275 / 429 input features. Local CPU run (no
GPU / paid compute). Run date: 2026-10-12.

> Archived record of the 3-dataset comparison run (2026-10-12). The
> `selected25` dataset was afterwards renamed `finalized_dataset` and adopted
> as the final working set; the dedicated 20-seed experiment on it is
> documented in `docs/baseline/finalized_dataset_baseline.md`. The run's
> outputs are archived under `outputs/baseline/comparison_3datasets/`.

## 1. Experiment description

- **Objective**: characterize how well simple baselines separate CO / MCI / AD
  from participant-wide eye-tracking features. Exploratory baseline — no
  mechanism claim, no clinical interpretation.
- **Samples**: one row per participant; the feature set of each dataset is
  combined across all 11 tasks:
  - `all72` — full 72 features × 11 tasks = **792 columns**
  - `selected25` — 25 selected features (Kruskal–Wallis p < 0.1 in ≥ 3 tasks)
    × 11 tasks = **275 columns**
  - `extended39` — 39 features (p < 0.2 in ≥ 3 tasks) × 11 tasks =
    **429 columns**
- **Problems (3)** — sample counts (participants):
  - `co_vs_mci` — binary, positive = MCI (141: CO 100 / MCI 41)
  - `co_vs_ad` — binary, positive = AD (116: CO 100 / AD 16)
  - `co_mci_ad` — 3-class CO/MCI/AD (157: 100 / 41 / 16)
- **CV**: participant-level, stratified by group, 4 folds × 10 seeds
  `[42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337]` (10 common seeds, fixed
  before results were seen) → **40 train/test evaluations per
  (dataset, problem, model)**; 3 × 3 × 6 × 40 = **2160 evaluations** total.
  Split method: within each group, permute with
  `numpy.random.default_rng(seed)` and assign round-robin to folds (same
  method as `fold/`, which used seed 42).
- **Missing data**: missing (participant, task) cells stay NaN in the wide
  matrix and are **median-imputed per fold on the training split only**.
  - `all72`: 1,389 / 124,344 cells (1.12%), 142 / 792 columns affected
  - `selected25`: 402 / 43,175 cells (0.93%), 45 / 275 columns
  - `extended39`: 785 / 67,353 cells (1.17%), 79 / 429 columns
  - Driven by `pupbl_*` of the 16 participants without a pupil baseline, and
    the 2 all-NaN task-10 rows (`BEGCO-029`, `BEGMCI-019`).
  Details: `outputs/baseline/comparison_3datasets/missingness.csv`.
- **Preprocessing**: `StandardScaler` on the wide features, fitted per fold on
  the training split only (after imputation).
- **Models** (fixed hyperparameters, no tuning):

| Model | Hyperparameters |
|---|---|
| Logistic Regression | `max_iter=2000` |
| SVM RBF | `SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)` |
| SVM Linear | `SVC(C=1.0, kernel='linear', probability=True)` |
| FNN (MLP 64-32) | `MLPClassifier(hidden=(64,32), relu, adam, lr=1e-3, alpha=1e-4, batch=32, max_iter=300)` |
| QDA | `solver='eigen', shrinkage=0.5` (wide layouts have p > n per class; svd solver is infeasible) |
| GNB | `var_smoothing=1e-9` |

`random_state = seed` for LR / SVM / FNN. QDA and GNB are deterministic.

- **Metrics** (per evaluation, on the held-out fold):
  - `Acc` — accuracy.
  - `Sen` — binary: recall of the positive class; 3-class: macro recall.
  - `Spec` — binary: TN/(TN+FP) of the negative class; 3-class: macro
    per-class specificity.
  - `AUC` — binary: ROC-AUC of the positive class; 3-class: OVR macro AUC.
  - `F1` — binary: F1 of the positive class; 3-class: macro F1.
- **Primary metric**: AUC. **Chance references** (majority-class accuracy):
  CO vs MCI 0.709, CO vs AD 0.862, 3-class 0.637; AUC chance = 0.5.

## 2. Results — mean ± std over 40 evaluations (10 seeds × 4 folds)

### all72 (72 features × 11 tasks = 792 columns)

#### co_vs_mci

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.7212±0.0762 | 0.4032±0.1380 | 0.8520±0.0925 | 0.6840±0.1106 | 0.4533±0.1329 |
| svm_rbf | 0.7093±0.0087 | 0.0000±0.0000 | 1.0000±0.0000 | 0.6845±0.1532 | 0.0000±0.0000 |
| svm_linear | 0.7035±0.0736 | 0.4275±0.1340 | 0.8170±0.0945 | 0.6759±0.1105 | 0.4533±0.1180 |
| fnn | 0.7234±0.0642 | 0.3700±0.1447 | 0.8690±0.0713 | 0.6839±0.1208 | 0.4304±0.1413 |
| qda | 0.7015±0.0210 | 0.0148±0.0356 | 0.9830±0.0270 | 0.4923±0.1011 | 0.0255±0.0617 |
| gnb | 0.6143±0.0839 | 0.3848±0.1455 | 0.7090±0.1060 | 0.5639±0.1118 | 0.3640±0.1184 |

#### co_vs_ad

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.8716±0.0292 | 0.1812±0.1789 | 0.9820±0.0300 | 0.7602±0.1250 | 0.2466±0.2290 |
| svm_rbf | 0.8621±0.0000 | 0.0000±0.0000 | 1.0000±0.0000 | 0.7183±0.1656 | 0.0000±0.0000 |
| svm_linear | 0.8578±0.0400 | 0.2313±0.2071 | 0.9580±0.0470 | 0.7492±0.1280 | 0.2746±0.2293 |
| fnn | 0.8483±0.0339 | 0.1187±0.1497 | 0.9650±0.0353 | 0.6318±0.1562 | 0.1576±0.1937 |
| qda | 0.8621±0.0000 | 0.0000±0.0000 | 1.0000±0.0000 | 0.4494±0.1653 | 0.0000±0.0000 |
| gnb | 0.8526±0.0406 | 0.2313±0.1735 | 0.9520±0.0436 | 0.6590±0.1640 | 0.2838±0.1968 |

#### co_mci_ad (3-class)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.6477±0.0592 | 0.4525±0.0847 | 0.7504±0.0398 | 0.6977±0.0728 | 0.4497±0.1020 |
| svm_rbf | 0.6370±0.0070 | 0.3333±0.0000 | 0.6667±0.0000 | 0.6791±0.0660 | 0.2594±0.0018 |
| svm_linear | 0.6227±0.0620 | 0.4346±0.0846 | 0.7419±0.0405 | 0.6834±0.0723 | 0.4305±0.0966 |
| fnn | 0.6381±0.0555 | 0.4181±0.0607 | 0.7390±0.0373 | 0.6659±0.0793 | 0.4109±0.0718 |
| qda | 0.6287±0.0174 | 0.3318±0.0131 | 0.6662±0.0122 | 0.4724±0.0585 | 0.2654±0.0208 |
| gnb | 0.5536±0.0771 | 0.4013±0.0802 | 0.7176±0.0484 | 0.5909±0.0871 | 0.3933±0.0870 |

### selected25 (25 features × 11 tasks = 275 columns)

#### co_vs_mci

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.6793±0.0675 | 0.4275±0.1482 | 0.7830±0.0815 | 0.6628±0.0939 | 0.4305±0.1259 |
| svm_rbf | 0.7164±0.0238 | 0.0370±0.0701 | 0.9950±0.0206 | 0.7525±0.0936 | 0.0621±0.1126 |
| svm_linear | 0.6566±0.0793 | 0.4320±0.1378 | 0.7490±0.0954 | 0.6272±0.0961 | 0.4208±0.1148 |
| fnn | 0.7418±0.0607 | 0.4498±0.1556 | 0.8620±0.0784 | 0.7167±0.0822 | 0.4952±0.1245 |
| qda | 0.7079±0.0364 | 0.0930±0.0664 | 0.9600±0.0462 | 0.5944±0.1005 | 0.1506±0.1039 |
| gnb | 0.6539±0.0814 | 0.5725±0.1361 | 0.6880±0.0947 | 0.6470±0.1044 | 0.4900±0.1057 |

#### co_vs_ad

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.8578±0.0513 | 0.2625±0.2189 | 0.9530±0.0503 | 0.8430±0.0874 | 0.3142±0.2333 |
| svm_rbf | 0.8621±0.0000 | 0.0000±0.0000 | 1.0000±0.0000 | 0.8455±0.1029 | 0.0000±0.0000 |
| svm_linear | 0.8681±0.0529 | 0.3750±0.2193 | 0.9470±0.0523 | 0.8518±0.0885 | 0.4249±0.2167 |
| fnn | 0.8681±0.0382 | 0.2500±0.2532 | 0.9670±0.0338 | 0.7232±0.1904 | 0.2953±0.2636 |
| qda | 0.8603±0.0076 | 0.0000±0.0000 | 0.9980±0.0088 | 0.5855±0.2165 | 0.0000±0.0000 |
| gnb | 0.8517±0.0729 | 0.5437±0.2525 | 0.9010±0.0690 | 0.7798±0.1638 | 0.5022±0.2169 |

#### co_mci_ad (3-class)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.6114±0.0666 | 0.4382±0.0865 | 0.7383±0.0418 | 0.6921±0.0698 | 0.4358±0.1004 |
| svm_rbf | 0.6491±0.0206 | 0.3502±0.0234 | 0.6802±0.0172 | 0.7468±0.0681 | 0.2926±0.0423 |
| svm_linear | 0.5853±0.0707 | 0.4318±0.0906 | 0.7282±0.0439 | 0.6934±0.0667 | 0.4304±0.1011 |
| fnn | 0.6630±0.0625 | 0.4661±0.0795 | 0.7626±0.0442 | 0.6887±0.0875 | 0.4618±0.0939 |
| qda | 0.6254±0.0367 | 0.3431±0.0312 | 0.6739±0.0233 | 0.5701±0.0836 | 0.2959±0.0458 |
| gnb | 0.5720±0.0839 | 0.4519±0.1023 | 0.7503±0.0526 | 0.6589±0.1003 | 0.4439±0.1054 |

### extended39 (39 features × 11 tasks = 429 columns)

#### co_vs_mci

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.7134±0.0718 | 0.4302±0.1237 | 0.8300±0.0900 | 0.6853±0.1024 | 0.4648±0.1133 |
| svm_rbf | 0.7136±0.0164 | 0.0200±0.0464 | 0.9980±0.0088 | 0.7609±0.1010 | 0.0350±0.0791 |
| svm_linear | 0.6858±0.0736 | 0.4348±0.1276 | 0.7890±0.0933 | 0.6683±0.1037 | 0.4444±0.1054 |
| fnn | 0.7439±0.0823 | 0.4568±0.1492 | 0.8620±0.1004 | 0.7269±0.1015 | 0.5064±0.1428 |
| qda | 0.7072±0.0274 | 0.0416±0.0586 | 0.9800±0.0314 | 0.5661±0.1062 | 0.0707±0.0981 |
| gnb | 0.6405±0.0959 | 0.5509±0.1623 | 0.6780±0.1064 | 0.6232±0.1092 | 0.4701±0.1265 |

#### co_vs_ad

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.8638±0.0366 | 0.1688±0.1825 | 0.9750±0.0296 | 0.8273±0.1070 | 0.2270±0.2403 |
| svm_rbf | 0.8621±0.0000 | 0.0000±0.0000 | 1.0000±0.0000 | 0.8313±0.1091 | 0.0000±0.0000 |
| svm_linear | 0.8612±0.0532 | 0.2625±0.2464 | 0.9570±0.0438 | 0.8257±0.1054 | 0.3149±0.2692 |
| fnn | 0.8543±0.0394 | 0.1625±0.1839 | 0.9650±0.0376 | 0.7202±0.1859 | 0.2074±0.2305 |
| qda | 0.8621±0.0000 | 0.0000±0.0000 | 1.0000±0.0000 | 0.5570±0.2081 | 0.0000±0.0000 |
| gnb | 0.8629±0.0521 | 0.3812±0.2467 | 0.9400±0.0444 | 0.7410±0.1647 | 0.4110±0.2470 |

#### co_mci_ad (3-class)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.6407±0.0690 | 0.4587±0.0893 | 0.7515±0.0489 | 0.7167±0.0703 | 0.4580±0.1037 |
| svm_rbf | 0.6402±0.0132 | 0.3383±0.0131 | 0.6711±0.0094 | 0.7484±0.0688 | 0.2705±0.0244 |
| svm_linear | 0.6184±0.0769 | 0.4474±0.1033 | 0.7422±0.0526 | 0.7076±0.0735 | 0.4461±0.1102 |
| fnn | 0.6572±0.0574 | 0.4507±0.0778 | 0.7593±0.0457 | 0.7048±0.0860 | 0.4408±0.0842 |
| qda | 0.6421±0.0292 | 0.3452±0.0290 | 0.6765±0.0219 | 0.5419±0.0845 | 0.2865±0.0466 |
| gnb | 0.5695±0.0771 | 0.4449±0.1001 | 0.7444±0.0565 | 0.6444±0.0928 | 0.4355±0.1032 |

### Best mean AUC per problem

| Problem | Best model | Dataset | AUC |
|---|---|---|---|
| co_vs_mci | SVM RBF | extended39 | 0.7609±0.1010 |
| co_vs_ad | SVM Linear | selected25 | 0.8518±0.0885 |
| co_mci_ad | SVM RBF | extended39 | 0.7484±0.0688 |

## 3. Observations

- The screened feature sets (selected25, extended39) outperform the full
  72-feature set on AUC in all three problems; extended39 is best for
  co_vs_mci and the 3-class problem, selected25 for co_vs_ad.
- SVM RBF reaches the top AUC on co_vs_mci and the 3-class problem, but its
  binary predictions collapse to the majority class (Sen ≈ 0, Spec ≈ 1):
  the ranking comes from the probability scores, not from the hard
  predictions.
- QDA degrades as the width grows (all72 AUC 0.45–0.49, at or below chance)
  and is dominated by the screened sets.
- Standard deviations are large (AUC ± 0.07–0.21) because the test folds are
  small (29–41 participants) and the AD class has only 16 participants.
- Mean accuracy is at or below the majority-class chance level in every
  problem; AUC is the informative metric here.

## 4. Reproducibility

- Code: `src/begad_norm/baseline/run_baselines.py` (+ `aggregate_results.py`),
  config: `src/begad_norm/baseline/config.yaml`.
- `outputs/baseline/comparison_3datasets/provenance.json`: git commit, config sha256, per-dataset
  data-file sha256, package versions (numpy 2.5.4, pandas 3.0.6,
  scikit-learn 1.9.1, scipy 1.18.1), host, command, duration.
- Re-run:
  `uv run python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml`
  (refuses to overwrite existing results), then
  `uv run python -m begad_norm.baseline.aggregate_results`.
- A smoke run (18 evaluations, one per dataset × model) was run first and
  validated the pipeline; its output file was deleted after use
  (`run_log.csv` records it).

## 5. Files

| File | Content |
|---|---|
| `outputs/baseline/comparison_3datasets/results_detail.csv` | 2160 rows, one per (dataset, problem, model, seed, fold) with all 5 metrics |
| `outputs/baseline/comparison_3datasets/summary.csv` | mean / std / n_runs per (dataset, problem, model, metric) |
| `outputs/baseline/comparison_3datasets/missingness.csv` | missing-cell counts per column of each wide matrix |
| `outputs/baseline/comparison_3datasets/run_log.csv` | run log (smoke + full) |
| `outputs/baseline/comparison_3datasets/full_run.log` | stdout of the full run |
| `outputs/baseline/comparison_3datasets/provenance.json` | reproducibility record |

## 6. Limitations

- AD is small (16 participants) and classes are imbalanced (CO 63.7–86.2%) →
  macro metrics and AUC are the informative ones; per-fold estimates are noisy.
- The wide layouts have p > n per class; all models risk overfitting the
  training folds (QDA shrinkage, LR / SVM defaults were not tuned).
- Missing cells are median-imputed per fold (≤ 1.2% of cells).
- No hyperparameter tuning — fixed config for all models; per-seed
  differences reflect training variability, not tuning.
- Single site, single session (2025-03-05); no external validation.
- The selected25 / extended39 features were screened on the same dataset
  (Kruskal–Wallis p thresholds) — selection bias is possible; results are
  exploratory.
