# BEGAD_Norm

Eye-tracking analyses of the BEGAD battery (CN / MCI / AD). The older subject-level pipeline, baselines and
normative model live in `subject-level/` (own README). This top-level package holds the feature-discovery
analysis.

## Feature discovery (statistical screening, no classifier training)

Implements `Running_Phase/BEGAD_feature_selection_pack/`: a combined registry of the 160 base definitions plus
dataset-derived D definitions, audited extraction, participant-level Mann–Whitney screening of CN_AD, CN_MCI and
MCI_AD, BH/BY multiplicity, participant bootstrap/leave-one-AD/permutation checks, macro selection and task–ROI
ranking.

One reproducible command (Python 3.10, `uv`):

```bash
uv sync
uv run python -m begad_norm.feature_discovery.cli all \
  --config configs/feature_discovery.yaml \
  --run-dir outputs/feature_discovery/2026-10-27_v2_expanded260
```

Stages can be run one at a time (`audit`, `propose`, `freeze`, `extract`, `stats`, `stability`, `report`);
`stats` refuses to run if the frozen catalogue, configuration or extraction code changed after `freeze`.
Use a new `--run-dir` for any changed configuration; `--resume` skips completed stages of the same frozen run.

Inputs (read-only): `T1235.xlsx`, `T5.xlsx` (original ROI workbooks; the `*_clean.xlsx` files are inventoried but
not used because they impute values), `dataset/participants_20250305.csv` and `dataset/20250305/*.csv`.

Outputs in the run directory: `RESULTS_VI.md` (Vietnamese memo) and `RESULTS_EN.md`, `macro_selected_main.csv`,
`roi_top20.csv`, all tables listed in pack file 04, `figures/`, `semantic_audit.md`, `candidate_freeze.json`.

Tests: `uv run pytest tests/feature_discovery -q`.

Code: `src/begad_norm/feature_discovery/` (`registry.py`, `dataset_features.py` for D001–D049, `event_contracts.py`
for raw event reconstruction, `extract_*.py`, `statistics.py`, `multiplicity.py`, `ranking.py`, `stability.py`,
`reporting.py`, `pipeline.py`, `cli.py`).
