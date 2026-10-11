# Fold split — 4-fold cross-validation (participant level)

## Configuration

| Setting | Value |
|---|---|
| n_folds | 4 |
| Split unit | participant (157 participants) |
| Stratification | by group (CO / MCI / AD) |
| Seed | 42 (NumPy `default_rng`) |
| Assignment | within each group: random permutation, round-robin to folds 1–4 |

## Files

- `fold_assignments.csv` — one row per participant:
  `participant_id, group, group_label, fold` (fold ∈ 1–4).
- `fold_summary.csv` — participants per fold and group.

| fold | n_total | n_CO | n_MCI | n_AD |
|---|---|---|---|---|
| 1 | 40 | 25 | 11 | 4 |
| 2 | 39 | 25 | 10 | 4 |
| 3 | 39 | 25 | 10 | 4 |
| 4 | 39 | 25 | 10 | 4 |

## Usage notes

- The **participant is the independent unit**: all rows of one participant
  (every task, every feature dataset — `processed_dataset/`, `_v2/`, `_v3/`)
  belong to the same fold. Never split (participant, task) rows across folds.
- Join on `participant_id` to attach the fold to any feature table.
- Group proportions are preserved in every fold (stratified split).
- The split does not account for any within-participant structure beyond group
  (no other metadata was used).
