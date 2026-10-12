# Feature Set Comparison — all72 vs selected25 vs extended39

Analysis date: 2026-10-12. Source: `outputs/baseline/comparison_3datasets/results_detail.csv`
(2160 evaluations; see `docs/baseline/baseline_experiment.md`). Machine-readable
comparison: `outputs/baseline/comparison_3datasets/dataset_comparison.csv`.

> The `selected25` dataset was afterwards renamed `finalized_dataset` and
> adopted as the final working set for all experiments.

## 1. Method

- **Metric**: AUC (primary metric of the experiment).
- **Paired design**: for a given problem and (seed, fold), all three feature
  sets are evaluated on the identical train/test split, so comparisons are
  paired at the (seed, fold) unit (40 units per problem).
- **Pooling**: per unit, AUC is averaged over the 6 models, giving one paired
  observation per dataset per (seed, fold) — 40 per problem.
- **Tests**: Friedman test across the 3 feature sets (40 blocks), followed by
  pairwise Wilcoxon signed-rank tests on the 40 paired differences.
  Exploratory; the (seed, fold) units share training data across seeds, so
  p-values are slightly optimistic.

## 2. Results

### Pooled AUC (mean over 6 models, 40 seed-fold units)

| Problem | all72 (792 cols) | selected25 (275 cols) | extended39 (429 cols) | Friedman p |
|---|---|---|---|---|
| co_vs_mci | 0.6307±0.0903 | 0.6668±0.0775 | 0.6718±0.0834 | < 0.0001 |
| co_vs_ad | 0.6613±0.1052 | 0.7715±0.1071 | 0.7504±0.1112 | < 0.0001 |
| co_mci_ad | 0.6316±0.0554 | 0.6750±0.0621 | 0.6773±0.0620 | < 0.0001 |

### Best mean AUC per feature set (any model, mean over 40 evaluations)

| Problem | all72 | selected25 | extended39 |
|---|---|---|---|
| co_vs_mci | 0.6845 (SVM RBF) | 0.7525 (SVM RBF) | **0.7609 (SVM RBF)** |
| co_vs_ad | 0.7602 (LR) | **0.8518 (SVM Linear)** | 0.8313 (SVM RBF) |
| co_mci_ad | 0.6977 (LR) | 0.7468 (SVM RBF) | **0.7484 (SVM RBF)** |

### Pairwise Wilcoxon tests (40 paired units per problem)

| Problem | selected25 vs all72 | extended39 vs all72 | extended39 vs selected25 |
|---|---|---|---|
| co_vs_mci | +0.0360, p < 0.0001 | +0.0410, p < 0.0001 | +0.0050, p = 0.204 |
| co_vs_ad | +0.1101, p < 0.0001 | +0.0891, p < 0.0001 | −0.0210, p = 0.0007 |
| co_mci_ad | +0.0434, p < 0.0001 | +0.0457, p < 0.0001 | −0.0023, p = 0.618 |

Mean paired AUC difference; positive favors the second-named set.

## 3. Interpretation

- **all72 is clearly dominated.** Both screened sets beat it significantly in
  all three problems (Friedman p < 0.0001; pairwise p < 0.0001). Its QDA
  degrades below chance (AUC 0.45–0.49) and its best AUC never exceeds 0.76.
- **selected25 vs extended39**:
  - co_vs_ad: selected25 is significantly better (p = 0.0007); all 6 models
    individually favor selected25 on this problem.
  - co_vs_mci and co_mci_ad: no significant difference (p = 0.204 and
    p = 0.618); extended39 is nominally higher in pooled mean, driven by
    SVM RBF and FNN.
- **Feature-selection circularity caveat**: selected25 and extended39 were
  screened on the same dataset (Kruskal–Wallis over the full cohort,
  including test-fold participants), so their CV performance is inflated
  relative to all72, which is the only set without in-sample selection. The
  gap between all72 and the screened sets is therefore overstated; a fair
  comparison would require nested selection (screening inside each training
  fold). The 25-vs-39 comparison is less affected (both are in-sample
  selections, 25 being a subset of 39).

## 4. Recommendation

**Use the 25-feature set (275 columns), now named `finalized_dataset`, as the
working feature set.** Rationale:

- Statistically on par with extended39 in 2 of 3 problems and significantly
  better in co_vs_ad.
- Lower dimensionality (275 vs 429 columns): less overfitting pressure in the
  p > n regime and a cleaner, better-interpretable set.
- Screened at the stricter threshold (p < 0.1 vs p < 0.2), so it contains the
  more robust discriminative features; extended39 adds mostly weaker
  candidates.
- extended39 remains a defensible alternative if the focus is specifically
  co_vs_mci or the 3-class problem, where it holds a small, non-significant
  nominal edge. all72 should not be used as-is for classification without
  dimensionality reduction or feature selection within the CV loop.
