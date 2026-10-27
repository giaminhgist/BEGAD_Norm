# Data audit and extraction contract

## 1. Inventory first

Inspect repository instructions and source versions. Locate original and cleaned
T5/T1235 workbooks, raw participant CSVs, stimulus images/text, ROI bounding boxes,
reading-order maps, pupil baselines, event definitions, TextGrid/audio alignment,
error annotations and demographics. Missing sources are a status, not permission
to invent data. Do not infer ROI geometry, words or audio from participant labels.

Create `source_manifest.json` with path, SHA256, size, cohort, visit, source type,
units, column mapping and parser version. Record the installed package versions.
Record CN=CO=HC only after verifying the repository's actual diagnostic codes.
Do not pool subjective memory complaints, NPH, other dementias or follow-ups into
the requested baseline CN/MCI/AD cohort without a separately declared analysis.

Build `subject_manifest.csv` and `task_roi_manifest.csv`. The latter includes
canonical task, raw task code, stimulus version, ROI ID, actual text/digit when
known, reading rank, line index, bounds and source of each metadata item.
ROI identity is `(stimulus_version, canonical_task, roi_id)`, never ROI number alone.
Missing lexical metadata stays missing. Stimulus metadata is conditioning/context,
not a diagnostic feature to test merely because missing ROI composition differs.

## 2. Mandatory semantic checks

Reproduce or explain these previous observations; they are prompts for inspection,
not hard-coded assertions that every future export must match:

| Observation in inspected snapshot | Required action |
|---|---|
| T5 diffPupL/R = endPupL/R − startPupL/R to floating-point tolerance | Record exact dependency; avoid presenting all three as independent evidence |
| regSaccades > nTobiiSaccades for 2,662/6,217 T5 rows | Establish event populations; block ratios using these fields until counts are reconciled |
| Many totalFixTime values around 0.002–0.003 | Verify seconds/ms and whether this is summed duration, elapsed span or an export artifact; do not relabel blindly |
| Single-fixation totalFixTime was globally imputed in cleaning.py | Prefer original observations/reconstruction; exclude imputed values from primary statistical testing and retain provenance |
| start/end pupil <=0 excluded in cleaning.py, including B fields | Establish raw vs subtractive-baseline coding; negative corrected values may be valid; raw invalid coding is different |
| tem_ifi uses half of adjacent fixation durations; tem_velocity uses that denominator | Do not treat these as measured inter-fixation gaps or measured saccade speeds |
| tem_trans_entropy uses a spatial histogram | Recompute actual transition entropy from transitions if supported, otherwise block that proposed feature |
| pup_slope uses fixation index rather than timestamp | Recompute per second; preserve the old feature only as legacy diagnostic output |
| IEEE mapper calls relative turning angle >90 degrees a regression | Recompute reading-order regressions; turns and regressions are different |
| pupil baseline subtracts a participant constant | SD, IQR and time slope are invariant to the same constant; mark duplicate variants |

Original paper Table II has a special definition of the last fixation linked to
an outgoing saccade. Do not assume it equals the last fixation inside the ROI.
Similarly, never equate endSpeechDelay with standard offset eye–voice span without
verifying the endpoint conventions. Store both explicit definitions if necessary;
an incompatible workbook field cannot populate a catalogue ID with another meaning.

Produce `semantic_audit.csv` with check ID, affected sources/columns, measured
evidence, severity, resolution and blocked feature IDs. Provide a short Markdown
explanation. Correct known issues in a new extraction path rather than altering
original files. No automatic winsorization or deletion solely because a value
appears to favor a diagnostic group.

## 3. Event and time semantics

- Deduplicate samples into true events using event IDs and verified onset/offset
  conventions. `Final` rows and `RecordingTimestamp` need inspection: timestamp
  may describe the recorded row rather than fixation onset.
- Units: all durations/gaps in seconds, measured amplitude in degrees and speed
  in degrees/second. Pixel distance cannot be relabelled degrees without viewing
  geometry. Store source units and conversion parameters.
- Link saccade amplitude and duration by event ID or validated timing, not array
  position after independent filtering. Mean speed = amplitude / actual matched
  saccade duration; peak speed needs adequately sampled trajectories and is not
  among the base160 definitions. A new D feature may use it only if those inputs
  genuinely support a validated measurement.
- Never connect points across a blink, long tracking gap, task boundary or dropped
  interval to manufacture a new saccade/transition. Preserve segment IDs.
- ROI assignment needs actual bounding boxes or validated annotations. Define a
  deterministic policy for overlap/boundary fixations and record it. A centroid
  assignment is acceptable if documented; it is not interchangeable with sample
  dwell allocation. Never count the same fixation duration fully in several ROIs.
- A visit is a contiguous sequence of fixations in one ROI with verified tracking
  continuity. Missing/outside-ROI observations break continuity; unknown gaps must
  not manufacture a return visit or a regression.
- First visit means the first observed visit, not necessarily first-pass traversal
  of the text. Report first-pass eligibility when that distinction matters.
- A reading-order regression is a valid transition to a lower ROI reading rank.
  Normal end-of-line to next-line returns increase reading rank and are not
  regressions. Relative turning angles are not reading order.
- A go-past interval starts at first fixation onset in ROI r and ends at the first
  fixation onset on a later-ranked ROI. No observed later-ranked ROI => censored,
  not zero. A gap within the interval makes it unresolved unless timing remains
  reliably observed under the frozen gap rule.
- Count saccades at ROI level using a declared incoming/outgoing/unique-union
  definition. Sum of incident ROI counts is not a count of distinct task saccades.
- Keep fixation/gaze validity separate from pupil validity. A missing pupil value
  should not silently delete otherwise valid gaze events for gaze-only features.

## 4. Missingness and support

Use explicit statuses: `observed`, `observed_zero`, `not_applicable`, `unavailable`,
`censored`, `invalid_measurement`, `insufficient_within_subject_support`, `imputed`.
No event/absence may be encoded as zero only when the opportunity was observed.
Missing ROI row is not proof of word skipping or speech omission.

For a feature at a fixed task–ROI, denominator for between-person coverage is all
eligible enrolled subjects in that diagnostic group expected to complete that
stimulus, including subjects with failed/missing measurement. Do not redefine it
to the subjects with successful data. Conditional features, such as return-visit
latency, must additionally report the conditioning event rate by group. A value
test among returners does not represent all participants; retain this limitation.

Predeclared technical defaults, adjustable only before viewing group p-values:

- Single-event measurements: at least one valid contributing event.
- Median of a distribution: at least one event, but record event count; flag
  one-event estimates for sensitivity analysis.
- IQR/dispersion: at least three observations, unless the exact formula requires
  more; no artificial zeros for insufficient support.
- P90 and P90−P10: at least ten observations.
- Spatial entropy: at least ten valid fixations; task transition entropy: at
  least thirty valid transitions. Record occupied bins/source states. Adequately
  observed one-bin or deterministic transitions have genuine entropy zero;
  missing observations must not be represented by zero entropy.
- Pupil time slope/variation: at least ten valid samples spanning 0.5 seconds;
  report sampling rate and gap handling. Short-word pupil changes do not establish
  a stimulus-specific physiological response.
- Spearman length response: at least twenty eligible distinct words and at least
  three distinct length values. Long/short contrasts need five words per bin.
- Ratios: denominator must be genuinely positive and sufficiently supported.
  For articulation-duration ratios use >=0.05 seconds, with sensitivity to this
  fixed threshold. Do not add arbitrary epsilon to make unavailable ratios finite.

All defaults are operational screening rules, not clinically validated cutoffs.
Export eligibility counts at each rule. Include a predefined secondary coverage
analysis at 50% and 75%; never select the threshold giving the nicest p-values.

## 5. Extraction levels and compatibility

Every catalogue row must have `available_status` and `reason`:

- `supported`: exact requested definition computable from inspected sources;
- `blocked_semantics`: field exists but meaning/units/endpoints unresolved;
- `blocked_source`: required raw/ROI/audio/metadata absent;
- `insufficient_support`: definition valid but too few observations;
- `exact_alias`: algebraically identical to another feature in this context.

The current ROI workbooks offer counts, timing summaries and pupil endpoints.
They do not preserve visit order, first-pass structure, continuous pupil response,
silence timing or correct/incorrect utterance labels. Do not fabricate these.
Raw157 events do not automatically provide word AOIs or aligned speech.

Catalogue macro medians over ROI summaries use equal weight per valid ROI unless
the definition explicitly states an event/time-weighted quantity. Report both
number of contributing ROIs and expected ROI coverage. Compute task metrics from
unique events when possible. Task duration is actual onset-to-offset elapsed time,
not summed fixation duration and not the EMS half-dwell approximation.

For macro summaries intended to represent the task's ROI set (e.g. A008–A010
and A061–A070), require valid values from at least 60% of expected target ROIs
within that participant, in addition to the catalogue's minimum ROI count.
This is distinct from between-person group coverage. Use the predefined 50%/75%
coverage sensitivity runs for this gate too, and label both levels in outputs.
For explicitly conditional event summaries such as regression jump length,
use the stated event denominator instead; do not demand that 60% of words
contain a regression. Report conditioning rates and support. Verified unvisited
ROIs may contribute genuine zeros only to definitions for which zero is meaningful.
Low coverage caused by observed behavior and low coverage caused by tracking
failure must be distinguished wherever annotations allow.

For combined analyses, keep separate task columns. Cross-task features use only
subjects observed in both required tasks and preserve the paired within-person
contrast. KD mean means the unweighted mean of the three valid card summaries,
not a mean over every ROI from all cards. Do not pool incompatible cohorts,
stimulus versions or seconds/pixels/degrees.

## 6. Required long-table schemas

`micro_features_long.csv` (one value per key):

```text
dataset_id,cohort_id,subject_id,visit_id,group,canonical_task,stimulus_version,
roi_id,roi_text,feature_id,origin,feature_name,value,unit,measurement_status,
n_events,n_valid_samples,valid_span_seconds,source_hash,extraction_version
```

`macro_features_long.csv`:

```text
dataset_id,cohort_id,subject_id,visit_id,group,scope_id,feature_id,origin,feature_name,
value,unit,measurement_status,n_events,n_rois,expected_rois,source_hash,
extraction_version
```

Primary baseline analysis permits one visit per subject. If repeats are retained
for a secondary reliability analysis, keep them identified and separate from the
independent-subject case-control tests. Nonunique primary keys are fatal errors;
do not silently mean duplicate rows to make a test run.

`feature_registry.csv` must include all 160 base IDs plus every fully defined D
candidate from file 06 (target 100 additional, 260 combined), even if some later
fail support checks. Do not create undefined D placeholders. Include origin
(`base160` or `dataset100`), level, family, name, formula, unit,
source_requirements, allowed_tasks, scope, within_subject_support, conditioning,
alias_group, supported_datasets, status, extraction_version and freeze_id.
The D prefix does not determine micro/macro level; read the explicit level field.
Use the file-06 proposal/audit/freeze before diagnostic screening.

`quality_features.csv` is separate from the expanded behavioral catalogue: tracking-valid
fraction, pupil-valid fraction, recording duration completeness, calibration
quality when observed, missing ROI fraction, imputation fraction and number of
available modalities. They are confound/retention checks, not automatic diagnostic
features. Behavioral inability to complete a task must be distinguished from
instrument failure whenever the annotations permit.
