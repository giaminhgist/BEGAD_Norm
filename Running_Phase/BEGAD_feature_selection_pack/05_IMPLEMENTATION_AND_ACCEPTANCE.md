# Implementation instructions and acceptance checks for Claude Code

Execute this pack in the actual BEGAD_Norm checkout. The deliverable is working
feature extraction/statistical reporting code plus results from available data.
This pack itself does not claim that those results have already been computed.

## 1. Inspect and preserve the repository

Read applicable AGENTS.md files, inspect `git status`, current HEAD, dependency
configuration and the existing subject/row pipelines. Preserve unrelated edits,
raw datasets, prior experiment outputs and existing model implementations.
Record the analysed commit and any differences from the snapshot in file 00.

Inspect the source and metadata files before deciding which existing functions
are reusable. In particular, do not reuse existing velocity, transition entropy,
reading regression or imputation logic until it passes the semantic contract in
file 02. Keep corrected extraction under a new namespace/version. Do not modify
raw clinical files to make assertions pass.

No classifier training, hyperparameter search, SMOTE, synthetic patients,
cross-validation benchmark, normative neural model or external clinical-data
upload is part of this task. A statistical bootstrap is not synthetic cohort
augmentation. Do not run unrelated repository experiments as a prerequisite.

## 2. Suggested structure

Adapt module paths to the inspected package layout rather than creating a
second incompatible project. A suitable structure is:

```text
src/begad_norm/feature_discovery/
    registry.py
    source_audit.py
    proposal_audit.py
    candidate_freeze.py
    event_contracts.py
    extract_micro.py
    extract_macro.py
    eligibility.py
    statistics.py
    multiplicity.py
    stability.py
    ranking.py
    reporting.py
    cli.py
configs/feature_discovery.yaml
tests/feature_discovery/
outputs/feature_discovery/<run_id>/
```

The registry preserves the 160 stable base IDs and adds the D001–D100 definitions
that Claude independently authors under file 06, targeting 260 definitions.
Record origin, micro/macro level, source requirements, formula version, support
rules and applicable scopes. Defined but unsupported IDs retain status/reason;
an undefined placeholder does not count as a proposal. Do not replace a blocked
feature with a convenient proxy. Document a genuine extension shortfall explicitly.

Use the repository's environment/package tooling. Prefer existing NumPy, pandas,
SciPy, statsmodels, YAML and plotting dependencies; add only necessary packages
and record versions. A GPU, additional LLM API, speech-recognition model and new classifier
framework are not needed for this task. CSV/JSON/Markdown outputs are mandatory;
Parquet caching is optional. Pin the numerical methods rather than relying on
changing `auto` test behavior.

## 3. CLI and execution stages

Provide idempotent stages with explicit run directories and saved manifests:

```text
audit       inventory sources, identities, semantics and availability
propose     export diagnosis-hidden profiles; validate Claude-authored D definitions
freeze      hash the combined catalogue, scopes, support rules and parameters
extract     produce micro/macro observations and quality/status tables
stats       eligibility, all pairwise tests, effect sizes, multiplicity
stability   whole-person bootstrap, AD influence, permutation validation
report      macro selections, ROI/task ranking, figures, Vietnamese memo
all         execute those stages in dependency order
```

Claude performs the dataset-informed reasoning and writes the actual additional
catalogue/specification as part of this task. The `propose` stage need not call
another LLM service; it must validate those authored definitions and source
evidence. Do not implement a CLI that merely prints 100 empty feature names.
The `stats` stage must refuse an unfrozen or silently changed candidate universe.

For example, if this package layout fits the repo:

```bash
python -m begad_norm.feature_discovery.cli all \
  --config configs/feature_discovery.yaml \
  --run-dir outputs/feature_discovery/2026-10-27_v2_expanded260
```

Resolve input paths through the configuration. Never hard-code this instruction
pack's temporary workspace paths as clinical-data locations. Support a validated
`--resume` mode keyed by input/config/code hashes; refuse incompatible cached
results. Reuse deterministic feature extraction if only reporting changes, but
do not reuse statistics from a different eligible population or hypothesis family.

Run all supported stages. Missing P/V/O sources should block dependent feature
families, not prevent a valid fixation/count analysis from completing. A partial
analysis must have `run_status=partial_sources` and identify actual coverage.
If no clinical source is testable, finish the implementation/tests/availability
report and set `run_status=blocked_no_testable_sources`.

## 4. Meaningful tests required before interpreting results

Use small deterministic synthetic fixtures with known event order and answers.
Synthetic data are for software tests only and must never appear in the clinical
selection report. Test inference integrity, not merely that a helper returns the
same values as a copy of its implementation.

### Registry and extraction

- Preserve exactly 160 base IDs, M001–M080 and A001–A080. A complete extension
  contains exactly D001–D100 and 260 unique definitions total. A documented partial
  extension has 160+N fully defined entries and cannot claim completion. All
  definitions need formulas, units, scope, dependencies, origin and explicit level.
- Proposal duplication checks reject renamed/base rank aliases, fixed unit
  conversions and task/ROI instances counted as extra definitions. Label shuffling
  cannot change diagnosis-hidden profiles, frozen formulas or raw extracted values.
- A change to D formulas/parameters after freezing invalidates dependent results;
  provenance records the amendment rather than silently reusing earlier q-values.
- Event IDs are deduplicated; a saccade's duration/amplitude remain matched after
  invalid-event filtering. Known amplitude/duration gives the expected speed;
  fixation dwell is never substituted as its denominator.
- A short known sequence distinguishes initial visit, return visit, reading-order
  regression, normal next-line return, outside-ROI transition and unknown gap.
  Missing samples do not create a fictitious reentry or regression.
- Go-past timing excludes a censored endpoint; elapsed visit span and summed
  fixation dwell differ correctly when a within-visit saccade is present.
- Fixation-time intersections with speech intervals are exact and deduplicated.
  Negative signed delays remain valid. Missing audio is not a speech omission.
- ROI-normalized coordinates use real bounds; physical units cannot change from
  pixels to degrees without calibration. A010 is not a mean over all pooled
  fixation rows. Macro incident counts do not double-count saccades.
- Subtraction of a constant leaves IQR and time slope unchanged. Endpoint pupil
  differences are not declared independent repeats of raw endpoints; true
  rank-equivalent aliases are recognized with their support masks.
- Missing ROI observations, genuine observed zero, censored values and imputed
  values retain distinct statuses through extraction and export.

### Statistical unit and tests

- Duplicate person/task/ROI/feature keys raise a clear error, or exact duplicate
  ingestion is resolved by a separately audited source-deduplication step. Adding
  repeated rows never increases the person N used by a test.
- CN/CO mapping is explicit; an unknown diagnosis cannot silently become CN.
- Minimum N and coverage use the original expected-person manifest. Removing a
  failed recording does not improve its own coverage denominator.
- Known rank arrays verify U/delta orientation, ties, reversed labels, constant
  pooled data and one constant group with a differing other group.
- A feature with p_CN_AD=0.10 and p_CN_MCI=0.30 fails the main AND rule; a feature
  with 0.10/0.10/0.30 passes main but fails all-three; exactly 0.20 fails `<0.20`.
- Permutation validation preserves group sizes, gives a documented nonzero
  randomization p-value and is never substituted with the smaller of two tests.

### Multiplicity, rankings and resampling

- Multiplicity families include all eligible losing and winning hypotheses;
  selecting top regions first changes no stored original q-value. Alias pruning
  occurs before inference, correlation pruning after original correction.
- Base and D hypotheses share their applicable families. Adding eligible D
  hypotheses updates family membership, q-values, region `m_r` and ranking.
  Origin tags must not silently split correction or reserve top-20 positions.
- `p_joint=max(p_CN_AD,p_CN_MCI)`. Test a case where two different features each
  pass only one contrast: the region is reserve-only, not main-selected.
- Region evidence uses all `m_r` eligible features. Two regions with equal best
  joint p but different numbers of tested features must reflect the specified
  multiplicity penalty, rather than both using their unadjusted minima.
- Top-20 keys are unique by task/stimulus/ROI; 7 qualifying regions produce 7
  output rows, 0 produce headers/reason, and >20 produce exactly 20 in a
  deterministic order. Different tasks sharing an ROI number remain distinct.
- Bootstrap draws carry all a person's records together. Duplicated draw IDs
  preserve with-replacement weighting. The original unique-N eligibility rule
  is not incorrectly reapplied to distinct IDs in every bootstrap replicate.
- Every discovery replicate re-evaluates all original eligible candidates and
  regions, including original nonwinners; the bootstrap ranking does not use its
  own future top-20 frequency. Noncomputable draws are handled as in file 03.
- Fixed seeds/hashes reproduce selection and ranking. A shuffled-label synthetic
  fixture must not preserve a deliberately constructed perfect group effect;
  do not require a random null dataset to have exactly zero false positives.

## 5. Run and review actual supported data

After the tests pass, run audit/extraction on the actual available datasets.
Before statistical interpretation, review:

1. Actual group Ns, source cohorts, baseline visits and duplicate identity checks.
2. Task-number mapping and expected ROI counts, including T5/T1235 overlap.
3. Reproduced semantic anomalies and what remains blocked.
4. Counts of base/new definitions (target 160+100), source feasibility, actual
   micro/macro split and aliases, plus the much larger number of testable hypotheses.
5. Group-specific measurement and conditioning-event coverage.

Then run the fixed primary statistical protocol, ranking and stability analysis.
Checkpoint bootstrap/permutation computations with seeds and completed replicate
IDs if runtime is substantial. Never silently cut 500 to 20 replicates and label
the result complete. A smoke run is allowed for debugging but must carry
`debug_only=true` and cannot supply the final clinical conclusions.

For substantial runtime, optimize reusable sorted values/arrays and cache audited
extraction rather than changing the scientific specification after seeing results.
If the environment truly prevents completion, preserve the exact resume command
and mark only the unfinished outputs as pending. Do not fabricate final stability
values or claim that unexecuted code has passed.

## 6. Acceptance checklist

The work is complete when all applicable items hold:

- [ ] Registry preserves the base 80 micro/80 macro definitions and adds 100
      dataset-grounded D definitions, or transparently documents a genuine shortfall.
- [ ] Actual D catalogue, proposal audit, dataset evidence and combined freeze
      manifest exist; no renamed features or ROI instances fill the quota.
- [ ] Original data remain unchanged and sources/configuration are hashed.
- [ ] All supported candidates are extracted under verified semantics; missing
      dependencies have actionable reasons rather than approximate substitutes.
- [ ] Person-level eligibility and all three contrasts are implemented correctly.
- [ ] Main AND screen, strict-three subset and pair-specific reserves are distinct.
- [ ] Full-family pairwise/joint FDR includes both origins; effects, CIs and coverage
      are exported, with the origin contribution summary.
- [ ] Subject bootstrap, leave-one-AD influence and secondary permutation checks
      are completed or explicitly marked pending for an evidenced blocker.
- [ ] Macro selected/compact tables and unique task–ROI top-20 table are produced
      without quotas, invented results or hidden failures.
- [ ] Actual available tasks have a separate task summary/ranking.
- [ ] Empty/partial outputs have headers and honest reasons.
- [ ] Tests pass, figures render and machine-readable schemas are consistent.
- [ ] Vietnamese `RESULTS_VI.md` answers the two requested conclusions and limits.
- [ ] README/run instructions identify one reproducible command and output folder.

Do not report “all 260 tested” if only 24 definitions have the necessary sources,
or “100 new features proposed” when some D slots are undefined or duplicated.
Do not report “20 selected ROIs” if only 6 meet the main criterion. Do not report
independent confirmation from overlapping T5 and T1235 participants.

## 7. Final message to the researcher

Report briefly, in Vietnamese:

1. What was implemented and which real datasets/tasks actually ran.
2. Number of available definitions, selected macro contexts and qualifying
   task–ROI pairs, distinguishing raw-p screening from FDR/stability support.
3. Links/paths to `RESULTS_VI.md`, selected macro table and `roi_top20.csv`.
4. Tests/checks run and any concrete missing sources or unresolved semantics.

Do not pivot to recommending models. The next discussion is which interpretable
features/regions deserve confirmation and what data are needed to establish it.
