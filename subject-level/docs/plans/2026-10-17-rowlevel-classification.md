# Plan: Row-level classification on T1235_clean and T5_clean (5 ML models + multimodal torch)

Date: 2026-10-17. Status: approved and executed.

## Goal

Classify CO / MCI / AD from the cleaned row-level eye-tracking datasets
`T1235_clean.xlsx` (12,486 rows, 23 columns, tasks 1/2/3/5) and
`T5_clean.xlsx` (6,217 rows, 20 columns, task 5) with 6 models: logistic
regression, SVM-RBF, random forest, LightGBM, sklearn FNN, and a PyTorch
port of the multimodal model from `subject-level/Multimodal.ipynb`.

## Protocol

- Problems: co_vs_mci (positive MCI), co_vs_ad (positive AD), co_mci_ad
  (3-class).
- Split seed fixed at 42; 10 folds. `grouped`: participant-level stratified
  round-robin (reuses `stratified_folds` from the baseline), evaluated at
  row and subject level (subject = mean predicted probability per
  participant, 0.5 threshold / argmax). `random`: row-level
  `StratifiedKFold(10, shuffle, random_state=42)` stratified by y, row-level
  metrics only (deliberately leaky arm to quantify leakage).
- 5 model seeds [42, 2026, 1, 0, 7] for RF / LGBM / FNN / torch
  initialization. LR and SVM-RBF are deterministic, so their 5 repeats are
  identical (std = 0 in the reports).
- Metrics: acc, sen, precision, spec, auc, f1 (baseline semantics). Per-fold
  median imputation and standardization fitted on the train split only;
  balanced class weights on row counts for all ML models; the torch model
  uses no class weights (faithful to the notebook).
- Torch model: three branches (visual 7d, speech 5d, pupil 7d for T5;
  pupil 10d for T1235, which adds maxPupLB/minPupLB/diffXpupL; ROI is the
  last column of every branch), Dense 128-64 with L2 1e-3 and dropout 0.2,
  fusion 192-128-64, softmax with 2 units for binary problems and 3 for
  3-class. Adam lr 1e-3, batch 32, max 300 epochs, early stopping
  patience 15 on a validation set carved from the train split
  (participant-disjoint for the grouped split).
- Volume per dataset: ML 3 problems x 5 models x 5 seeds x 10 folds x
  3 eval configs = 2,250 detail rows; torch 3 x 5 x 10 x 3 = 450;
  total 2,700. Torch trainings per dataset = 300.

## Outputs (per dataset folder at repo root)

`T1235/` and `T5/`: `code/` (runner, torch model, config, aggregate and
report scripts), `results/results_detail.csv` (2,700 rows),
`results/summary.csv` (324 rows), `results/fit_times.csv`,
`results/missingness.csv`, `folds/fold_info.csv` (seed 42; grouped: 640
participant rows, problem = all; random: test rows per problem), plus
`provenance.json`, `run_log.csv`, and `report.md` / `report_vi.md`.

## Execution order

Smoke T1235 -> full T1235 -> aggregate -> report T1235 (delivered
immediately) -> smoke T5 -> full T5 -> aggregate -> report T5.

## Verification

Full pytest suite; per-run invariants (2,700 detail rows; 50 per problem,
model, split, eval level; 324 summary rows without NaN; fold-info schema;
provenance sha256 matches config and data; one-task determinism spot
check). Reports ship with AUC / Sen / Spec tables in EN and VI.
