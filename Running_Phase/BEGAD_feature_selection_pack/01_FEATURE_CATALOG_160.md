# Catalogue: 160 candidate definitions, not 160 proven biomarkers

Exactly **80 micro definitions (M001–M080)** and **80 macro definitions
(A001–A080)** are proposed below. Every ID must exist in the registry, including
blocked/unavailable candidates. This is a hypothesis-driven catalogue for
implementation and screening; it is not a claim that all definitions can be
reconstructed from T5/T1235 summaries or that they discriminate disease.

These remain the **base160** definitions. In addition, Claude must follow
`06_DATASET_DRIVEN_100_FEATURES.md` to propose 100 dataset-grounded definitions
with IDs D001–D100, targeting 260 in the combined registry. Do not renumber,
replace or count task/ROI instantiations of this base list as the extra 100.

A micro definition is instantiated separately at each eligible task–ROI. A macro
definition is one value per participant and specified scope, usually one task.
Thus the actual number of statistical hypotheses can greatly exceed 160.
Preserve scope in names, e.g. `M008@READING@ROI_42`, `A005@KD1`.

## Reading the catalogue

Source codes:

| Code | Required source |
|---|---|
| E | Existing ROI export, usable only after exact meaning/units/endpoints are verified |
| F | True fixation events with validated onset/offset, position, event ID and validity |
| S | True saccade events with matched amplitude, duration, direction and validity |
| O | Real ROI geometry, stimulus version, expected target list, line and reading order |
| P | Valid pupil samples/timestamps, eye-specific calibration/units and baseline where needed |
| V | Validated speech alignment and, for errors, actual correctness/repetition annotations |
| L | Actual lexical metadata: Unicode letter counts and language-appropriate syllable counts |
| T | Actual task onset/offset and continuous timing/validity metadata |

`E?` marks a potential direct field mapping, **not permission to use a similarly
named column**. If the export definition differs, the candidate remains blocked
unless it can be reconstructed exactly. Multiple source codes mean all required
sources. `B` means both KD and reading; `R` means reading only. Most B macro
definitions are instantiated separately for KD1, KD2, KD3 and READING. Exceptions
are the explicit GLOBAL/CROSS scopes in the last table.

Notation and common rules:

- A fixation assigned to ROI r contributes its actual duration once. A visit is
  a continuous sequence of fixations on r with verified intervening tracking.
  Return visits are observed revisits after a verified departure. Unknown gaps
  must not manufacture visits. First observed visit is not necessarily first pass.
- `D(r)` is the sum of valid fixation durations on r; `V1(r)` is its first visit.
  `rank(r)` is stimulus reading order, never an arbitrary numerical ROI ID.
- An incoming saccade lands on r; an outgoing saccade starts on r. An incident
  saccade is the unique union of these sets. Same-ROI saccades count once in the
  incident set. Reconstructed coordinate jumps are not automatically measured S.
- Normalized ROI coordinates are `(x-left)/width`, `(y-top)/height`. Screen
  coordinates divide by known screen width/height. Degree-valued geometry requires
  actual viewing geometry; pixels cannot be relabeled degrees.
- Durations/gaps are seconds. Pupil values use verified mm, otherwise the actual
  documented source unit. Keep incompatible pupil units in different contexts.
  A signed gaze–speech delay may validly be negative.
- Default binocular pupil is the mean of simultaneously valid left/right values;
  require both eyes for that definition. Do not switch between binocular and
  single-eye values across participants. A separately labeled single-eye
  sensitivity analysis is optional and adds hypotheses if used inferentially.
- Baseline-corrected pupil subtracts each eye's own verified instruction-period
  baseline before binocular averaging. Prefer the original protocol's first ten
  valid instruction fixations when those events are identifiable; record the
  exact baseline rule and block an unidentifiable baseline. Do not substitute
  task mean or infer baseline from the diagnostic group.
- For gaze–speech measures, the utterances are the annotations assigned to the
  target ROI, including correctly linked repetitions. Overlapping intervals are
  merged before calculating duration. Unspoken/ungazed targets have explicit
  absence/conditional statuses, not invented event times.
- A rate/fraction needs an observed opportunity and a supported positive
  denominator. Zero counts are valid only with adequate observation. Apply file
  02's minimum event/sample rules and file 03's participant-coverage rules.
- Micro features need actual ROI identity even if O is implicit in an E export.
  Macro summaries over ROIs are equal-ROI-weighted unless explicitly event/time
  weighted. Store contributing N and expected coverage for every summary.

## Micro 1: fixation burden and timing

| ID | Feature | Exact operational definition | Unit | Sources | Task / support |
|---|---|---|---|---|---|
| M001 | fixation_count | Number of distinct valid fixation events assigned to r | count | F+O; E? nFixations | B; 0 only for a reliably observed unvisited target |
| M002 | first_fixation_duration | Duration of the first observed fixation on r | s | F+O | B; at least one fixation |
| M003 | mean_fixation_duration | D(r) divided by M001 for M001>0 | s | F+O; E? verified totalFixTime/nFixations | B; record event count |
| M004 | median_fixation_duration | Median duration of fixations on r | s | F+O | B; record event count |
| M005 | iqr_fixation_duration | Q75−Q25 of fixation durations on r | s | F+O | B; at least 3 fixations |
| M006 | first_visit_dwell | Sum of fixation durations in V1(r), excluding intervening saccade duration | s | F+O | B; observed first visit |
| M007 | first_visit_fixation_count | Number of fixations in V1(r) | count | F+O | B; observed first visit |
| M008 | total_roi_dwell | D(r), summed across all observed visits | s | F+O; E? totalFixTime | B; observed zero only with verified full opportunity |
| M009 | last_visit_dwell | Sum of fixation durations in the last observed visit to r | s | F+O | B; observed last visit, record completeness |
| M010 | first_to_last_gaze_span | Last fixation offset on r minus first fixation onset on r; includes time away | s | F+O | B; observed endpoints; not identical to dwell |

Hypothesis: burden and heterogeneity of visual processing. Longer duration is
not inherently pathological; lexical difficulty, visual acuity and task demands
can produce it. M003 cannot be calculated until totalFixTime is verified as a
sum of genuine fixation durations in the same event population.

## Micro 2: revisiting and reading-order control

| ID | Feature | Exact operational definition | Unit | Sources | Task / support |
|---|---|---|---|---|---|
| M011 | visit_count | Number of verified visits to r | count | F+O | B; distinguish no visit from unknown observation |
| M012 | return_visit_count | max(M011−1,0), with verified departures and returns | count | F+O | B; flag rank-equivalent alias of M011 only where proven |
| M013 | reread_dwell | Sum of fixation durations in visits after V1(r) | s | F+O | B; 0 if observed visited ROI has no return |
| M014 | reread_dwell_fraction | M013/M008 | fraction | F+O | B; M008>0 |
| M015 | first_reentry_latency | First return-visit onset minus first-visit last fixation offset | s | F+O | B; conditional on an observed return and resolved interval |
| M016 | go_past_elapsed | First fixation onset on r to first subsequent fixation onset on an ROI ranked later than r | s | F+O | B; censored if no later ROI is observed |
| M017 | go_past_extra_elapsed | M016 minus the elapsed onset-to-offset span of V1(r), including its internal saccades | s | F+O | B; complete go-past interval; do not subtract M006 instead |
| M018 | first_exit_backward | 1 if the first valid ROI-changing exit from V1(r) lands at a lower reading rank, otherwise 0 | binary | F+O | B; needs an observed exit to a known ROI |
| M019 | regressions_out_count | Number of valid ROI-changing transitions r→j with rank(j)<rank(r) | count | F+O | B; observed transition opportunity |
| M020 | regressions_in_count | Number of valid ROI-changing transitions j→r with rank(j)>rank(r) | count | F+O | B; observed transition opportunity |

These are reading-order measures. Normal return sweeps to the next line are not
regressions. For KD they describe serial number scanning, not linguistic rereading.
M015/M016/M017 are conditional/censored quantities; report the conditioning rates.

## Micro 3: measured saccade behavior

| ID | Feature | Exact operational definition | Unit | Sources | Task / support |
|---|---|---|---|---|---|
| M021 | incoming_amplitude_median | Median measured amplitude among incoming saccades | deg | S+O | B; at least 1 matched event |
| M022 | incoming_amplitude_iqr | Q75−Q25 among incoming amplitudes | deg | S+O | B; at least 3 events |
| M023 | outgoing_amplitude_median | Median measured amplitude among outgoing saccades | deg | S+O | B; at least 1 matched event |
| M024 | outgoing_amplitude_iqr | Q75−Q25 among outgoing amplitudes | deg | S+O | B; at least 3 events |
| M025 | incoming_duration_median | Median actual duration among incoming saccades | s | S+O | B; matched event duration |
| M026 | outgoing_duration_median | Median actual duration among outgoing saccades | s | S+O | B; matched event duration |
| M027 | incoming_mean_speed_median | Median of event amplitude/event duration among incoming saccades | deg/s | S+O | B; positive actual duration per event |
| M028 | outgoing_mean_speed_median | Median of event amplitude/event duration among outgoing saccades | deg/s | S+O | B; positive actual duration per event |
| M029 | tiny_saccade_fraction | Incident saccades with amplitude<1 degree divided by all valid incident saccades | fraction | S+O; E? only with reconciled event populations | B; supported nonzero denominator |
| M030 | long_saccade_fraction | Incident saccades with amplitude>2 degrees divided by all valid incident saccades | fraction | S+O; E? only with reconciled event populations | B; supported nonzero denominator |

The 1/2-degree thresholds are fixed descriptive bins motivated by the existing
feature vocabulary, not universal physiological categories. Do not call M029 a
microsaccade measure. Mean speed is not peak velocity. Do not use the existing
half-fixation-duration velocity approximation for M027/M028.

## Micro 4: landing position and within-ROI geometry

| ID | Feature | Exact operational definition | Unit | Sources | Task / support |
|---|---|---|---|---|---|
| M031 | normalized_fixation_dispersion | RMS Euclidean distance from the mean of ROI-normalized fixation coordinates | normalized distance | F+O | B; at least 3 fixation positions |
| M032 | initial_landing_x | Normalized x coordinate of first fixation on r | ROI widths | F+O | B; actual bounds and first fixation |
| M033 | initial_landing_y | Normalized y coordinate of first fixation on r | ROI heights | F+O | B; actual bounds and first fixation |
| M034 | final_landing_x | Normalized x coordinate of last fixation on r | ROI widths | F+O | B; record endpoint completeness |
| M035 | final_landing_y | Normalized y coordinate of last fixation on r | ROI heights | F+O | B; record endpoint completeness |
| M036 | normalized_convex_hull_area | Convex-hull area of fixation positions divided by ROI bounding-box area | fraction | F+O | B; at least 3 observed positions; a collinear/degenerate hull is a genuine 0 with adequate support |
| M037 | within_roi_spatial_entropy | −sum(p_b log(p_b))/log(9) over a fixed 3x3 grid of normalized ROI fixation positions | unitless | F+O | B; at least 10 fixations; one occupied bin gives genuine entropy 0 |
| M038 | within_roi_scanpath_length | Sum of consecutive fixation-position distances in normalized ROI coordinates, within verified visits only | normalized distance | F+O | B; at least 1 valid within-visit transition |
| M039 | return_landing_displacement | Median normalized Euclidean distance from each return visit's first fixation to the initial first fixation | normalized distance | F+O | B; conditional on observed return |
| M040 | final_exit_rank_displacement | rank(destination)−rank(r) for the last observed valid exit from r | target positions | F+O | B; observed known destination; signed |

ROI normalization does not remove calibration error or ROI-size uncertainty.
Tiny text ROIs may be too small for reliable within-ROI geometry; record this in
the semantic/QC audit and block candidates if the measurement is unsupported.

## Micro 5: pupil level and change

| ID | Feature | Exact operational definition | Unit | Sources | Task / support |
|---|---|---|---|---|---|
| M041 | corrected_pupil_median | Median valid binocular baseline-subtracted pupil while gaze is on r | pupil unit | P+O | B; identified baseline and pupil validity |
| M042 | corrected_pupil_iqr | Q75−Q25 of the same valid pupil observations | pupil unit | P+O | B; at least 3; identical to uncorrected IQR under constant subtraction |
| M043 | pupil_p90_p10 | P90−P10 of the same observations | pupil unit | P+O | B; at least 10; baseline-invariant |
| M044 | first_fixation_pupil | Mean binocular pupil during the first fixation on r | pupil unit | P+F+O; E? startPupL/R | B; fixation-average definition must match |
| M045 | last_fixation_pupil | Mean binocular pupil during the last fixation actually inside r | pupil unit | P+F+O; E? endPupL/R | B; block incompatible outgoing-saccade endpoint convention |
| M046 | exit_entry_pupil_change | M045−M044 using the same eye-validity/units convention | pupil unit | P+F+O; E? reconciled endpoints | B; both endpoint measurements |
| M047 | pupil_time_slope | OLS slope of valid binocular pupil on actual seconds while gaze is on r | pupil unit/s | P+O | B; at least 10 samples spanning 0.5 s; actual time, not fixation index |
| M048 | interocular_pupil_difference | Median of simultaneous left pupil minus right pupil while on r | pupil unit | P+O | B; paired valid eyes and compatible eye calibration |
| M049 | upper_pupil_excursion_from_entry | P90 of pupil while on r minus M044 | pupil unit | P+F+O | B; at least 10 samples and valid entry value |
| M050 | return_vs_first_visit_pupil | Mean pupil during first return visit minus mean pupil during V1(r) | pupil unit | P+F+O | B; conditional on a verified return and valid pupil in both visits |

These are descriptive ocular/autonomic candidates. Short word dwell is generally
insufficient to isolate a word-specific pupil response; illumination, gaze angle,
blink artifacts, medication and response latency can confound interpretation.
M048 is especially sensitive to ocular/device factors. Do not manufacture a
pupil time series from start/end columns. Baseline correction alone does not
remove lighting effects, and negative corrected values may be valid.

## Micro 6: speech production at the target

| ID | Feature | Exact operational definition | Unit | Sources | Task / support |
|---|---|---|---|---|---|
| M051 | total_articulation_time | Duration of the union of all actual utterance intervals assigned to r | s | V+O; E? totalSpokenTime | B; verify whether E excludes intervening silence |
| M052 | first_utterance_duration | Offset−onset of the first actual utterance assigned to r | s | V+O | B; observed utterance |
| M053 | extra_utterance_count | max(number of verified utterances for r−1,0) | count | V+O | B; complete annotation; repetition interpretation needs annotation |
| M054 | speech_omission | 1 if expected target r is explicitly verified unspoken in a completed/annotated trial, else 0 | binary | V+O | B; missing audio or missing ROI row is not omission |
| M055 | speech_substitution | 1 if an annotated incorrect spoken replacement occurs for r, else 0 | binary | V+O | B; manual/validated errors, not unverified ASR disagreement |
| M056 | self_correction_count | Number of annotated self-correction events linked to r | count | V+O | B; requires correction tags |
| M057 | preceding_interutterance_gap | First utterance onset for r minus offset of the immediately preceding chronological utterance | s | V+O | B; preceding utterance observed; report true overlaps as signed |
| M058 | following_interutterance_gap | Onset of the utterance immediately after r's final utterance minus that final offset | s | V+O | B; subsequent utterance observed |
| M059 | first_to_last_vocal_span | Last utterance offset for r minus first utterance onset for r | s | V+O | B; includes intervening pauses/other speech |
| M060 | articulation_time_per_syllable | Duration of first correct utterance for r divided by its verified spoken syllable count | s/syllable | V+O+L | B; correct utterance and positive syllable count |

Use actual utterances, not only the expected script. Reading ability, education,
language and oral motor differences can influence these measures. Audio or a
TextGrid without validated error annotations does not support M054–M056.

## Micro 7: gaze–speech coordination

| ID | Feature | Exact operational definition | Unit | Sources | Task / support |
|---|---|---|---|---|---|
| M061 | onset_gaze_speech_delay | First speech onset for r minus first fixation onset on r | s | F+O+V; E? speechDelay | B; both observed onsets; signed |
| M062 | final_offset_gaze_speech_delay | Final speech offset for r minus last fixation offset actually on r | s | F+O+V; E? endSpeechDelay | B; exact endpoint audit required; signed |
| M063 | offset_referenced_eye_voice_span | First speech onset for r minus last fixation offset of V1(r) | s | F+O+V | B; operational offset reference; not M062 |
| M064 | gaze_visits_before_speech | Number of visits to r whose onset precedes first speech onset for r | count | F+O+V | B; valid synchronized timing |
| M065 | gaze_dwell_before_speech | Fixation duration on r overlapping times before its first speech onset | s | F+O+V | B; split intervals at the speech boundary |
| M066 | gaze_speech_overlap_time | Duration of intersection between fixation-on-r intervals and the union of utterance intervals for r | s | F+O+V | B; exact interval intersection, not minimum of totals |
| M067 | gaze_dwell_after_speech | Fixation duration on r occurring after its final speech offset | s | F+O+V | B; observed trial continuation |
| M068 | articulation_gaze_overlap_fraction | M066/M051 | fraction | F+O+V | B; M051>=0.05 s |
| M069 | post_speech_return | 1 if a verified return visit to r begins after final speech offset, else 0 | binary | F+O+V | B; requires adequate post-speech observation |
| M070 | spatial_eye_voice_span | rank(ROI currently fixated at r's first speech onset)−rank(r) | target positions | F+O+V | B; actual gaze at onset; saccade/gap/off-ROI onset is NA |

Synchronize eye and speech clocks from recorded alignment, not from correlations
chosen to maximize diagnostic separation. A gaze–speech lag can reflect naming,
lexical processing, coordination or ordinary preview; it is not a specific marker
of AD by definition. Preserve the explicit endpoint convention in feature names.

## Micro 8: context-scaled and conditional measures

| ID | Feature | Exact operational definition | Unit | Sources | Task / support |
|---|---|---|---|---|---|
| M071 | first_fixation_per_letter | M002 divided by verified Unicode letter count of the word | s/letter | F+O+L | R; positive letter count |
| M072 | first_visit_dwell_per_letter | M006 divided by letter count | s/letter | F+O+L | R; positive letter count |
| M073 | total_dwell_per_letter | M008 divided by letter count | s/letter | F+O+L; E? verified M008 | R; positive letter count |
| M074 | fixations_per_letter | M001 divided by letter count | count/letter | F+O+L; E? verified M001 | R; positive letter count |
| M075 | total_dwell_per_syllable | M008 divided by verified word syllable count | s/syllable | F+O+L; E? verified M008 | R; positive syllable count |
| M076 | onset_delay_per_articulation | M061 divided by M052 | unitless | F+O+V | B; M052>=0.05 s; signed numerator |
| M077 | dwell_per_return_visit | M013 divided by M012 | s/return | F+O | B; conditional on M012>0 |
| M078 | median_regression_jump_out | Median of rank(r)−rank(destination) among backward ROI transitions out of r | target positions | F+O | B; conditional on at least one valid regression |
| M079 | incoming_amplitude_relative_to_geometry | Median of incoming saccade amplitude divided by the angular distance between its source-ROI and r centroids | unitless | S+O | B; known distinct source ROI and positive distance in degrees |
| M080 | first_visit_dwell_per_articulation | M006 divided by M052 | unitless | F+O+V | B; M052>=0.05 s |

Dividing by word length is a descriptive scaling, not statistical adjustment for
word difficulty. At one fixed word ROI, length is constant across participants:
M071/M072/M073/M074/M075 can be strictly rank-equivalent to their unscaled parent.
Keep the definitions for cross-ROI summaries and interpretability, but collapse
proven same-support rank aliases **before** within-ROI multiplicity correction.
Do not claim extra independent disease evidence from those divisions.

## Macro 1: task-wide timing and fixation distribution

| ID | Feature | Exact operational definition | Unit | Sources | Scope / support |
|---|---|---|---|---|---|
| A001 | task_elapsed_time | Actual task offset−onset | s | T | B per task; exclude unresolved truncation |
| A002 | total_fixation_time | Sum of durations of unique valid within-task fixation events | s | F+T | B per task; do not double-count events across ROIs |
| A003 | task_fixation_count | Number of unique valid within-task fixation events | count | F+T | B per task |
| A004 | fixations_per_minute | 60*A003/A001 | count/min | F+T | B per task; actual elapsed time>0 |
| A005 | task_fixation_duration_median | Median of actual fixation durations | s | F+T | B per task |
| A006 | task_fixation_duration_iqr | Q75−Q25 of actual fixation durations | s | F+T | B per task; at least 3 |
| A007 | task_fixation_duration_p90 | P90 of actual fixation durations | s | F+T | B per task; at least 10 |
| A008 | roi_first_visit_dwell_median | Equal-ROI median of valid M006 values | s | F+O+T | B per task; report expected ROI coverage |
| A009 | roi_first_visit_dwell_iqr | Q75−Q25 of valid M006 values | s | F+O+T | B per task; at least 3 eligible ROIs |
| A010 | roi_total_dwell_median | Equal-ROI median of valid M008 values | s | F+O+T; E? verified M008 | B per task; report ROI coverage |

## Macro 2: global regressions and rereading

| ID | Feature | Exact operational definition | Unit | Sources | Scope / support |
|---|---|---|---|---|---|
| A011 | revisited_roi_fraction | Visited ROIs with >=2 verified visits divided by all observed visited ROIs | fraction | F+O+T | B per task; conditional on visited set |
| A012 | reread_time_fraction | Sum(M013) divided by sum(M008) over the same eligible ROIs | fraction | F+O+T | B per task; positive denominator |
| A013 | reading_order_regression_rate | Backward reading-rank ROI changes divided by all valid ROI changes | fraction | F+O+T | B per task; normal next-line returns excluded from numerator |
| A014 | backward_jump_median | Median positive reading-rank distance of backward ROI transitions | target positions | F+O+T | B per task; at least 1 regression |
| A015 | backward_jump_iqr | Q75−Q25 of backward jump distances | target positions | F+O+T | B per task; at least 3 regressions |
| A016 | long_regression_fraction | Backward jumps spanning >=3 target positions divided by all valid backward jumps | fraction | F+O+T | B per task; fixed operational threshold |
| A017 | within_line_regression_rate | Backward same-line ROI changes divided by all valid same-line ROI changes | fraction | F+O+T | B per task; line metadata |
| A018 | cross_line_regression_rate | ROI changes to an earlier line divided by all valid cross-line ROI changes | fraction | F+O+T | B per task; line metadata |
| A019 | normal_return_sweep_fraction | Leftward ROI changes to the immediately next line divided by all valid line changes | fraction | F+O+T | B per task; real x and line identity |
| A020 | rereading_episodes_per_100_targets | 100*number of completed backward episodes/number of expected targets | episodes/100 targets | F+O+T | B per task; episode definition below |

For A020, an episode begins on a valid backward transition from the current
furthest reached reading rank and ends when gaze reaches or exceeds that rank
again. Nested backward movements belong to the same episode. A gap or task end
before recovery censors the episode; report censored counts separately and block
the complete-episode rate if observation completeness is inadequate. Do not
reinterpret the count as memory failure without additional evidence.

## Macro 3: task saccade distribution

| ID | Feature | Exact operational definition | Unit | Sources | Scope / support |
|---|---|---|---|---|---|
| A021 | task_saccade_count | Count of unique validated within-task saccade events | count | S+T | B per task; not sum of incident ROI counts |
| A022 | saccades_per_second | A021/A001 | count/s | S+T | B per task; positive elapsed time |
| A023 | amplitude_median | Median measured saccade amplitude | deg | S+T | B per task |
| A024 | amplitude_iqr | Q75−Q25 of measured saccade amplitudes | deg | S+T | B per task; at least 3 |
| A025 | amplitude_p90 | P90 of measured saccade amplitudes | deg | S+T | B per task; at least 10 |
| A026 | saccade_duration_median | Median actual saccade duration | s | S+T | B per task |
| A027 | saccade_duration_iqr | Q75−Q25 of actual saccade durations | s | S+T | B per task; at least 3 |
| A028 | saccade_mean_speed_median | Median of event amplitude divided by its actual matched duration | deg/s | S+T | B per task; positive durations |
| A029 | task_tiny_saccade_fraction | Saccades with amplitude<1 degree divided by all valid task saccades | fraction | S+T | B per task; same event population |
| A030 | task_long_saccade_fraction | Saccades with amplitude>2 degrees divided by all valid task saccades | fraction | S+T | B per task; same event population |

## Macro 4: task scanpath organization

| ID | Feature | Exact operational definition | Unit | Sources | Scope / support |
|---|---|---|---|---|---|
| A031 | angular_scanpath_length | Sum of unique valid within-task measured saccade amplitudes | deg | S+T | B per task; no bridging missing intervals |
| A032 | path_length_relative_to_layout | A031 divided by sum of centroid-to-centroid angular distances between consecutive expected targets | unitless | S+O+T | B per task; positive fixed layout path; same-support rank alias of A031 within one layout |
| A033 | normalized_screen_dispersion | RMS distance from mean normalized screen fixation position | normalized distance | F+T | B per task; at least 3 positions; known screen dimensions |
| A034 | normalized_screen_hull_area | Convex-hull area of task fixation positions divided by screen area | fraction | F+T | B per task; at least 3 positions, record degeneracy |
| A035 | task_spatial_entropy | −sum(p_b log(p_b))/log(48) over a fixed 8x6 screen grid | unitless | F+T | B per task; at least 10 fixations; one occupied bin gives genuine 0 |
| A036 | task_transition_entropy | −sum_i pi_i sum_j P(j given i) log(P(j given i))/log(12) on a fixed 4x3 screen grid | unitless | F+T | B per task; at least 30 observed transitions; deterministic transitions can give genuine 0 |
| A037 | roi_dwell_entropy | −sum_r p_r log(p_r)/log(N_expected), where p_r=D(r)/sum D(r) | unitless | F+O+T; E? verified complete D(r) | B per task; N_expected>1, distinguish unvisited from missing |
| A038 | off_text_dwell_fraction | Valid onscreen fixation duration outside all target ROIs divided by all valid onscreen fixation duration | fraction | F+O+T | B per task; requires unclipped gaze, not ROI-only export |
| A039 | initial_landing_center_offset | Equal-ROI mean of M032−0.5 | ROI widths | F+O+T | B per task; valid first landings and bounds |
| A040 | return_sweep_landing_error | Median absolute horizontal landing error from next line's first-target center divided by that target width | ROI widths | F+O+T | B per task; verified normal return sweeps |

For A036, `pi_i` is the fraction of observed transition origins in state i and
`P(j given i)` is its observed destination distribution. Include same-state
successive fixation transitions consistently. Never infer this from a spatial
histogram. For A037, reliably observed unvisited ROIs contribute zero dwell;
unknown missing ROIs invalidate a claimed complete-layout entropy.

## Macro 5: task/global pupil summaries

| ID | Feature | Exact operational definition | Unit | Sources | Scope / support |
|---|---|---|---|---|---|
| A041 | task_corrected_pupil_median | Median of valid binocular baseline-subtracted pupil in the task | pupil unit | P+T | B per task; identified instruction baseline |
| A042 | task_pupil_iqr | Q75−Q25 of valid task pupil values | pupil unit | P+T | B per task; at least 3; baseline-invariant |
| A043 | task_pupil_p90_p10 | P90−P10 of valid task pupil values | pupil unit | P+T | B per task; at least 10 |
| A044 | task_pupil_time_slope | Theil–Sen slope of median pupil in fixed nonoverlapping 100-ms time bins versus bin-center seconds | pupil unit/s | P+T | B per task; at least 10 supported bins; do not interpolate gaps |
| A045 | late_early_pupil_change | Median pupil in final elapsed-time third minus median in initial elapsed-time third | pupil unit | P+T | B per task; both windows supported |
| A046 | task_absolute_pupil_median | Median valid uncorrected binocular pupil | pupil unit | P+T | B per task; units/illumination documented |
| A047 | pupil_above_baseline_fraction | Valid time with binocular pupil above its verified baseline divided by total valid pupil time | fraction | P+T | B per task; sample/time weights from actual timestamps |
| A048 | last_first_second_pupil_change | Mean pupil in last 1 second minus mean in first 1 second of the task | pupil unit | P+T | B per task; duration>=2 s and both fixed windows supported |
| A049 | pupil_variation_rate | Median absolute successive change per second after a centered 3-bin rolling median of supported 100-ms pupil-bin medians | pupil unit/s | P+T | B per task; at least 10 valid adjacent differences; never smooth across missing bins |
| A050 | instruction_baseline_pupil | Verified binocular instruction-period baseline used for subtraction | pupil unit | P | GLOBAL once per baseline participant, not repeated per task |

For A044 accept a 100-ms bin only when at least half its expected samples are
valid, using the actual acquisition rate. For A045/A048 require at least 60%
valid expected samples in each window. A049 uses the same supported 100-ms bins
as A044, requires all three bins for each centered rolling median, and takes
differences only between adjacent valid smoothed bin centers. Its derivative
is artifact-sensitive; retain the raw validity mask and document acquisition
filtering. If those semantics are unresolved, mark it blocked. Do not turn an
imputed smooth trace into observed data. Pupil summaries are
strongly confounded by luminance and ocular factors; retain these limitations.

## Macro 6: speech fluency and errors

| ID | Feature | Exact operational definition | Unit | Sources | Scope / support |
|---|---|---|---|---|---|
| A051 | correct_syllables_per_articulation_second | Correctly spoken syllable count divided by union duration of those corresponding correct utterances | syllables/s | V+L+T | B per task; actual spoken content, positive duration |
| A052 | correct_targets_per_elapsed_minute | 60*number of correctly spoken distinct expected targets/A001 | targets/min | V+O+T | B per task; word/digit target unit explicit |
| A053 | interutterance_pause_fraction | Union duration of positive interutterance gaps divided by first-speech-onset to last-speech-offset span | fraction | V+T | B per task; complete utterance stream; do not double-count gaps |
| A054 | interutterance_pause_median | Median nonnegative gap between successive utterances | s | V+T | B per task; use max(gap,0), report overlaps separately |
| A055 | interutterance_pause_iqr | Q75−Q25 of nonnegative interutterance gaps | s | V+T | B per task; at least 3 gaps |
| A056 | long_pause_fraction | Successive utterance gaps>0.5 s divided by all observed successive utterance gaps | fraction | V+T | B per task; fixed 0.5-s threshold |
| A057 | target_omission_rate | Verified omitted expected targets divided by all expected targets | fraction | V+O+T | B per task; complete correctness annotation |
| A058 | target_substitution_rate | Expected targets with at least one verified incorrect substitution divided by all expected targets | fraction | V+O+T | B per task; complete error annotation |
| A059 | extra_utterances_per_target | Sum(M053) divided by number of expected targets | count/target | V+O+T | B per task; distinguish repetition from correction |
| A060 | self_corrections_per_target | Sum(M056) divided by number of expected targets | count/target | V+O+T | B per task; correction tags |

The speech stream must cover the complete intended task. Do not treat a truncated
recording as many verified omissions. Rate denominators and correct/repeated
utterance rules must remain the same across groups.

## Macro 7: gaze–speech coordination across targets

| ID | Feature | Exact operational definition | Unit | Sources | Scope / support |
|---|---|---|---|---|---|
| A061 | onset_delay_median | Equal-ROI median of M061 | s | F+O+V; E? verified speechDelay | B per task; report eligible ROI coverage |
| A062 | onset_delay_iqr | Q75−Q25 of M061 | s | F+O+V; E? verified speechDelay | B per task; at least 3 ROIs |
| A063 | onset_delay_p90 | P90 of M061 | s | F+O+V; E? verified speechDelay | B per task; at least 10 ROIs |
| A064 | final_offset_delay_median | Equal-ROI median of M062 | s | F+O+V; E? verified endSpeechDelay | B per task; endpoint convention must match |
| A065 | final_offset_delay_iqr | Q75−Q25 of M062 | s | F+O+V; E? verified endSpeechDelay | B per task; at least 3 ROIs |
| A066 | negative_final_offset_fraction | ROIs with M062<0 divided by all ROIs with valid M062 | fraction | F+O+V; E? verified endSpeechDelay | B per task; not an error code |
| A067 | offset_referenced_evs_median | Equal-ROI median of M063 | s | F+O+V | B per task; operational endpoint convention |
| A068 | negative_offset_referenced_evs_fraction | ROIs with M063<0 divided by ROIs with valid M063 | fraction | F+O+V | B per task; conditional on observable timing |
| A069 | spatial_evs_median | Equal-ROI median of M070 | target positions | F+O+V | B per task; gaze observed at speech onset |
| A070 | task_gaze_articulation_overlap | Sum(M066)/sum(M051) over exactly the same paired eligible ROIs | fraction | F+O+V | B per task; no duplicated intervals across ROI assignments |

A task average of gaze–speech lag is not interchangeable with each word's lag.
Always retain per-ROI coverage and the distribution of contributing words/digits.

## Macro 8: context response and paired task contrasts

| ID | Feature | Exact operational definition | Unit | Sources | Scope / support |
|---|---|---|---|---|---|
| A071 | word_length_dwell_association | Within-person Spearman correlation between letter count and M006 across observed word ROIs | correlation | F+O+L | READING; >=20 distinct word ROIs and >=3 length values; nonconstant dwell |
| A072 | long_short_word_dwell_difference | Median M008 for words>=8 letters minus median for words<=4 letters | s | F+O+L; E? verified dwell | READING; >=5 eligible word ROIs per bin |
| A073 | long_short_word_onset_delay_difference | Median M061 for words>=8 letters minus median for words<=4 letters | s | F+O+V+L; E? verified delay | READING; >=5 eligible word ROIs per bin |
| A074 | text_position_fixation_difference | Median M001 in final third of expected text positions minus median in initial third | count | F+O; E? verified counts plus real reading ranks | READING; >=5 observed target ROIs per third |
| A075 | elapsed_time_fixation_duration_change | Median fixation duration for events with midpoint in last elapsed-time third minus first third | s | F+T | B per task; >=5 valid events per window |
| A076 | kd3_kd1_fixation_duration_difference | A005@KD3−A005@KD1 for the same participant | s | F+T | CROSS_KD; both cards observed |
| A077 | kd3_kd1_elapsed_log_ratio | log(A001@KD3)−log(A001@KD1) | log ratio | T | CROSS_KD; both positive complete durations |
| A078 | kd3_kd1_regression_rate_difference | A013@KD3−A013@KD1 | fraction difference | F+O+T | CROSS_KD; both valid transition denominators |
| A079 | reading_kd_onset_delay_difference | A061@READING−mean(A061@KD1,A061@KD2,A061@KD3) | s | F+O+V; E? verified delays | CROSS_BOTH; all four task summaries observed, equal card weight |
| A080 | reading_kd_time_per_correct_target_log_ratio | log(reading elapsed/correct distinct targets)−log(mean of KD1/2/3 elapsed/correct distinct targets) | log ratio | V+O+T | CROSS_BOTH; positive complete times/counts in all four tasks |

A071 is a participant-level descriptive correlation, not a separate significance
test over their words. A072/A073 do not match lexical frequency or sentence
context. A074 is spatial text position, not chronological fatigue. A075 can be
influenced by different content at different times. KD card order/layout may
confound A076–A078; differences are not pure executive-load effects. A079/A080
compare tasks with different stimulus units and speech demands, so interpret
them as paired task profiles rather than a calibrated cognitive-load scale.

## Priorities, aliases and feasibility

Implement by dependency rather than by promising all 160 values:

1. Audit and reuse exact supported E fields and true F/S/T events first.
2. Add ROI chronology/geometry candidates only with actual O and timing.
3. Add pupil samples/baseline candidates only with verified P.
4. Add speech/coordination candidates only with V and clock alignment.
5. Add lexical and cross-task candidates only with complete required metadata.

Potential workbook mappings worth auditing include M001, M003/M008, M044–M046,
M051, M061/M062 and their explicitly defined macro aggregates. This is a starting
audit list, not a guarantee that those fields are valid. Existing `regSaccades`,
`tinySaccades`, `longSaccades` and `nTobiiSaccades` must not yield ratios until
their incompatible observed counts are reconciled.

Report the 160 base definitions, the actual number of additional D definitions
and the supported/nonalias testable counts separately. The combined target is
260 definitions, not 260 guaranteed measurements. Do not add every mean/SD/min/max
permutation to inflate the catalogue. Follow file 06 for additional IDs, source
rationale and the freeze; expand the multiplicity universe before inference.

## Scientific anchors and limits

The supplied multimodal paper, *A Multimodal Approach for Early Identification
of Mild Cognitive Impairment and Alzheimer's Disease With Fusion Network Using
Eye Movements and Speech*, motivates the existing ROI fixation/saccade, pupil
endpoint and speech-delay vocabulary. Its Table II definitions must be checked
against the actual export; the present catalogue also proposes additional
chronological/context features that the summary workbook may not support.
The paper's DOI is https://doi.org/10.1109/TNSRE.2025.3561043.

Relevant primary papers for interpreting, rather than guaranteeing, the proposed
features include:

- Shah et al. (2026), *Alterations in fine-grained eye movements during reading
  as early indicators of cognitive decline in mild cognitive impairment*;
  publisher-hosted accepted manuscript, CN–MCI study, not AD validation:
  https://doi.org/10.1186/s12877-026-08171-6
- Mathôt et al. (2018), *Safe and sensible preprocessing and baseline correction
  of pupil-size data*; measurement methodology:
  https://doi.org/10.3758/s13428-017-1007-2
- *When the Eyes No Longer Lead: Familiarity and Length Effects on Eye-Voice
  Span* (2016); endpoint/coordination interpretation, not AD validation:
  https://doi.org/10.3389/fpsyg.2016.01720
- Repository being implemented:
  https://github.com/giaminhgist/BEGAD_Norm

Read the actual methods before using a paper to justify a particular endpoint,
threshold or clinical interpretation. Several catalogue definitions are proposed
engineering hypotheses; neither publication elsewhere nor a raw p<0.20 result
establishes validity in this cohort.
