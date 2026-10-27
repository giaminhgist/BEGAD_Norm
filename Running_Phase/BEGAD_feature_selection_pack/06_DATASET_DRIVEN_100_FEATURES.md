# Claude-generated extension: 100 additional dataset-grounded features

## 1. Required task

In addition to the unchanged 160 base definitions in
`01_FEATURE_CATALOG_160.md`, **Claude Code must inspect the actual available
datasets and independently propose 100 additional feature definitions**.
Do not ask the researcher to supply their names. The intended combined catalogue
is **260 definitions: 160 base + 100 dataset-derived**.

This file specifies how to generate those additional features; it does not
pretend that their 100 definitions have already been written or validated.
Perform this step after the data audit in file 02 and before diagnostic
statistical screening in file 03. Implement and test the supported new features,
then include them in the same macro selection and micro task–ROI ranking.

Use IDs `D001` through `D100`. Each definition has an explicit `level` of `micro`
or `macro`; the D prefix identifies its origin, not a third measurement level.
Keep M001–M080 and A001–A080 unchanged. The additional micro/macro split is
determined by actual source support and scientific rationale, not a mandatory
50/50 quota. Seek useful coverage of both levels and report the resulting split.

## 2. Inspect what the dataset can actually measure

Use the source manifests and semantic audit to inspect:

- Actual columns, units, event types, timestamps, sampling rate and valid segments.
- Fixation/saccade chronology, within-person event distributions and trial length.
- ROI bounds, reading order, line structure, stimulus versions and target identity.
- Which modalities are genuinely synchronized, and whether speech intervals and
  correctness/repetition annotations exist.
- Which pupil signals/baselines are observed and which are only endpoint exports.
- Shared participants and paired tasks across KD1/KD2/KD3/READING, keeping cohorts
  and visits separate until identity/source compatibility is verified.
- Pooled measurement support, missingness, range, resolution and deterministic
  dependencies. Do not infer unavailable samples from summary columns.

Use a diagnosis-hidden view for **feature invention**: remove group labels,
diagnostic scores, outcome columns and prior p-values/model importances from
the profiles used to propose formulas. The preceding cohort audit may verify
diagnosis codes/counts, but disease-group differences must not be the criterion
used to invent, tune or retain the 100 definitions.

Do not consult old winning-feature lists to manufacture variants of their
smallest p-values. Existing labels or group-coded filenames may be visible in
the repository; record that exposure honestly and keep the proposal process
independent of observed diagnostic separation as far as possible.

## 3. What qualifies as an additional feature?

A new definition must have all of the following:

1. A dataset observation that motivates it: actual source path/columns/events,
   their verified semantics, and a measurable behavior not adequately covered
   by the existing definition.
2. An exact reproducible formula, unit, domain, task scope, input dependencies,
   event/ROI support minimum, zero/NA/censoring rules and interpretation.
3. A reason it adds information beyond the closest base or new definition.
4. An extractor that produces one scalar per independent participant and
   context, preserving the file-02 data contract.
5. A source/semantic feasibility status, distinct from later statistical success.

Do **not** count these as additional definitions:

- Renaming an existing feature, changing seconds to milliseconds or changing sign.
- A fixed positive rescaling or monotone transform that gives the same within-ROI
  rank comparison on the same observed participants.
- Instantiating the same formula on another ROI, KD card, text version or cohort.
- Repeating the same feature for left/right eyes without a separate defensible
  measurement question and an audit of its relation to the binocular feature.
- Arbitrarily multiplying every mean/SD/min/max, threshold or window combination
  simply to reach 100. A statistic/window variant needs a distinct rationale.
- Metadata such as participant ID, diagnosis, MMSE/MoCA used to define diagnosis,
  ROI number, word identity or task label used directly as a disease feature.
- Tracking failure, imputation flags or missing-file indicators renamed as
  behavioral biomarkers. Keep these in the separate quality/retention audit.

Derived ratios, interactions and composites are allowed when their numerator,
denominator and physiological interpretation are explicit. Distinguish genuine
person-varying normalization from division by a word length that is constant
within one ROI. Do not use label-fitted coefficients or an AD/CN-specific reference
to generate the primary extension. No classifier, embedding model or hidden
supervised feature search is needed.

## 4. Candidate search directions, conditional on source support

These are directions to investigate, **not a prefilled list of 100 features**.
Claude must choose and define measurements supported by the inspected data.

| Available evidence | Possible additional measurement questions | Feasibility check |
|---|---|---|
| Ordered fixation/event sequences | Serial dependence, burstiness, duration alternation, local continuity and recovery after a regression | Enough consecutive events; valid segment boundaries; no bridging gaps |
| Matched saccade events or sufficiently sampled raw trajectories | Amplitude–duration coupling, directional organization, speed-profile shape where genuinely measured | Match event IDs; physical units; adequate temporal resolution; no speed from fixation dwell |
| ROI geometry and line/order metadata | Neighbor-context contrasts, line-entry/exit behavior, layout-adjusted transition organization and traversal consistency | Real bounds/order; distinguish next-line return from regression; enough transitions |
| Rich per-ROI distributions | Robust tail asymmetry, concentration or distribution shape beyond the base location/spread summaries | Adequate event count; not a disguised rank alias or tiny-N artifact |
| Reliable pupil time series | Time-dependent coupling to verified events, fluctuation structure or recovery under a fixed window definition | Sampling, response latency, luminance/gaze confounds and baseline quality |
| Synchronized gaze and utterances | Coordination variability, recovery after annotated speech errors, sequence coupling and local gaze/speech disagreement | Actual clocks, alignment, complete intervals and verified error labels |
| Observed word metadata and repeated contexts | Responses to lexical/layout categories or neighboring context within the same person | Actual language/word metadata, enough items per condition, no invented frequency or syllables |
| Matched tasks in the same participants | Cross-task consistency, dispersion or contrasts with a specified interpretation | All required paired tasks; no cohort mixing; no automatic enumeration of every pair |

A feature may be scientifically plausible yet infeasible in this dataset. Reject
or defer it transparently. Do not create a fake raw time series or label an
unobserved eye movement as measured in order to use one of these directions.

## 5. Proposal workflow and freeze

1. Produce diagnosis-hidden dataset profiles and a source-capability inventory.
2. Draft new candidates with formulas and source evidence. Compare each against
   all 160 base definitions and the other new candidates.
3. Check algebraic/rank aliases and perform label-blind feasibility spot checks.
   Test formulas on observed examples without inspecting disease separation.
4. Retain 100 defensible additional definitions, assign D001–D100, and write the
   complete catalogue and registry extension described below. Do not silently
   replace any of the original 160 to make room for them.
5. Freeze definitions, context scope, thresholds, support rules and extraction
   parameters; save their hashes/time in `candidate_freeze.json`.
6. Only then run group-specific eligibility, all three diagnostic contrasts,
   multiplicity correction, stability checks and macro/ROI/task reporting.

The target is 100 genuine new definitions, **not 100 statistically selected
features**. Make a sustained attempt across all supported sources. If available
data cannot justify 100 distinct definitions, report the actual N and the
concrete source/redundancy blockers, mark `extension_status=partial`, and continue
the valid analysis. Do not fill D slots with placeholders, duplicate formulas
or unmeasured signals and claim success. A complete extension has exactly 100
defined candidates; fewer computable/testable candidates may remain after
participant coverage and semantic checks.

Once frozen, do not replace a failed candidate by another p-value-driven variant
while continuing to call the search only 100 features. If a later semantic bug
requires a change, version the formula, record exposure to previous results and
retain a search/change log. Label outcome-informed additions post hoc; a new
FDR calculation alone cannot undo adaptive selection based on seen outcomes.

If an extraction parameter is estimated from pooled participants, state that
dependence. Re-estimate such nuisance parameters inside subject resampling when
needed to propagate uncertainty; keep the already-frozen definition fixed.
For any later predictive evaluation, fit data-dependent extraction parameters
and perform feature/ROI selection using training participants only.

## 6. Required extension deliverables

Claude must generate these analysis outputs, alongside files required elsewhere:

- `DATASET_PROPOSED_100_FEATURES.md`: the actual D catalogue, with an opening
  summary of how the inspected dataset led to these proposals, the micro/macro
  split, and feasibility limitations. It must contain real definitions, not this
  instruction text repeated or 100 empty numbered headings.
- `dataset_proposed_features.csv`: machine-readable D definitions, joined into
  the main `feature_registry.csv` with `origin=dataset100`. Base definitions use
  `origin=base160`. Keep `level` separate from `origin`.
- `proposal_audit.csv`: proposed/rejected/aliased/deferred candidates, closest
  existing feature, source evidence and reasons. Preserve failed attempts rather
  than hiding a larger search behind a final count of 100.
- `candidate_freeze.json`: base/new definition counts, extension status, catalogue
  and source hashes, freeze time, formula/configuration versions, label-exposure
  declaration and any later amendments.
- `feature_origin_summary.csv`: available, testable, selected, FDR-supported and
  stability-preferred counts by origin and level, from the same combined analysis.

Each new definition must record at least:

```text
feature_id,origin,level,family,name,formula,unit,scope,allowed_tasks,
source_paths,source_columns,source_semantics_evidence,dataset_motivation,
closest_existing_ids,incremental_information,within_subject_support,
conditioning,zero_rule,missing_rule,censoring_rule,denominator_guard,
parameters,parameter_origin,alias_status,feasibility_status,blocked_reason,
extraction_version,freeze_id
```

Use one row per definition in the proposal registry. Instantiated task/ROI
hypotheses belong in the usual long observation/statistics tables, not in the
count of 100 definitions.

## 7. Integration with statistical selection and conclusions

All supported D micro features are tested **separately at each eligible task–ROI**.
All supported D macro features are tested at their declared participant-level
scope. Apply the same CN_AD, CN_MCI and MCI_AD tests, eligibility, p<0.20 AND
screen, effect sizes, confidence intervals, FDR and resampling rules as the base
features. Do not give the new proposals a more permissive threshold.

Within each existing cohort/level/contrast family, combine eligible **base and
new hypotheses before correction**. Origin is a reporting tag, not a reason to
split the multiple-testing family. Likewise include both origins when computing
the region search size `m_r`, region/task evidence, redundancy, bootstrap
selection and top-20 membership. Recompute those quantities if the base160
pipeline has already run; stale base-only q-values/rankings are not the expanded
analysis. Raw statistics may be cached only when their inputs/methods are unchanged.

The final outputs still answer the same two questions:

1. **Các macro features được chọn** from the full expanded catalogue.
2. **Top 20 cặp task–ROI và micro features được chọn** from the full expanded
   catalogue, or fewer when fewer regions qualify.

Mark each selected feature as `base160` or `dataset100`. Summarize what the
additional proposals contributed without claiming an independent validation
cohort or improved classifier performance. Report all-three-pairs and
MCI_AD-specific reserve findings for both origins. No quota reserves a fixed
number of selected features or top-20 positions for either origin.
