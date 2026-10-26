# Fold split — 5-fold cross-validation (participant level)

## Configuration

| Setting | Value |
|---|---|
| n_folds | 5 |
| Split unit | participant (157 participants) |
| Stratification | by group (CO / MCI / AD) |
| Seed | 42 (NumPy `default_rng`) |
| Assignment | within each group: random permutation, round-robin to folds 1–5 |

## Files (in `fold1/`)

- `fold_assignments.csv` — one row per participant:
  `participant_id, group, group_label, fold` (fold ∈ 1–5).
- `fold_summary.csv` — participants per fold and group.

| fold | n_total | n_CO | n_MCI | n_AD |
|---|---|---|---|---|
| 1 | 33 | 20 | 9 | 4 |
| 2 | 31 | 20 | 8 | 3 |
| 3 | 31 | 20 | 8 | 3 |
| 4 | 31 | 20 | 8 | 3 |
| 5 | 31 | 20 | 8 | 3 |

## Usage notes

- The **participant is the independent unit**: all rows of one participant
  (every task of `processed_dataset/`) belong to the same fold. Never split
  (participant, task) rows across folds.
- Join on `participant_id` to attach the fold to any feature table.
- Group proportions are preserved in every fold (stratified split).
- The split does not account for any within-participant structure beyond group
  (no other metadata was used).
- The baseline experiments generate per-seed 5-fold splits with the same
  method (`numpy.random.default_rng(seed)`, round-robin within group); the
  CSVs here are the seed-42 reference split. The earlier 4-fold reference
  split (seed 42) remains in `fold/`.
