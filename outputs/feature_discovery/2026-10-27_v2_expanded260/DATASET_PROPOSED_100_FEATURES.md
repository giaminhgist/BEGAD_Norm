# Dataset-proposed additional features (D001-D049)

Freeze ID: `freeze_35a60cb67109` (2026-10-27T20:51:43+00:00). Extension status: **partial**
(49 of the 100-definition target; combined catalogue 209 of 260).

## How the inspected data led to these proposals

The repository contains (i) two original ROI summary workbooks (T1235: KD1-3 and reading, 64 participants;
T5: reading, same 64 participants, a different export), with 19 per-ROI columns whose meaning is given by
Table II of the TNSRE 2025 paper, and (ii) 205 raw Tobii Studio event exports (157 baseline CN/MCI/AD
participants) holding fixation and saccade events as onset/offset row pairs with timestamps, fixation
positions, fixation-to-fixation amplitudes, per-eye gaze, eye distance and pupil at event boundaries.
There are no ROI bounding boxes, stimulus text, reading order, audio, TextGrids, error annotations, lexical
metadata, continuous pupil samples or media onset events.

The D definitions therefore use only what these sources measure, after the label-blind checks in
`semantic_audit.md`: constant-free workbook pupil contrasts (D001, D002), the export's direction-based and
amplitude-sum saccade fields as stand-alone counts (D003, D004), a joint gaze-speech ratio (D005),
target-set summaries and within-person couplings of the workbook fields (D007-D016), and raw event-sequence,
saccade dynamics, direction, spatial layout, event-sampled pupil, binocular and postural measures
(D017-D049). Profiles used for invention exclude diagnosis (`diagnosis_hidden_profiles/`).

Level split: {'macro': 43, 'micro': 6}. Supported definitions: 48; defined but blocked: 1
(D006, exit-landing speech offset delay, because the fixation time point is undocumented).

**Shortfall.** Only 49 defensible definitions could be authored. The repository has no ROI geometry, stimulus text, reading order, audio/TextGrid or error annotations, and the raw exports keep pupil only at event boundaries; the ROI workbooks expose 19 summary columns of which several are semantically unreconciled. Candidates that would need those sources, or that were renamed/rank-equivalent variants, were rejected (see proposal_audit.csv).

Rejected, aliased and deferred candidates are kept in `proposal_audit.csv`. Feasibility here is a source and
semantics statement, not a statistical result.

## Definitions

### D001 `roi_entry_to_exit_landing_pupil_change` (micro, roi_pupil_export)
- Formula: Mean over eyes of (pupil at the exit-landing fixation - pupil at the first fixation on r): T1235 (diffPupL+diffPupR)/2 (= startB-endB, sign verified); T5 (diffPupL+diffPupR)/2 with both start/end >0
- Unit: mm; scope: task_roi
- Sources: T1235.xlsx; T5.xlsx [diffPupL,diffPupR,startPup*,endPup*]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv; T5 diff=end-start exactly; T1235 diff=startB-endB exactly and B=c-pupil (sum of T5 raw and T1235 B constant within participant, median SD 0.028), so the difference is baseline-invariant
- Dataset motivation: Workbooks export endpoint pupil per ROI under the Table II convention ('last fixation' = landing after the last saccade leaving r); M045/M046 need the last fixation inside r and are blocked
- Closest existing: M046,M044; incremental: Different endpoint (post-exit landing fixation); M046 stays blocked
- Support: both eyes valid at both endpoints
- Zero/missing/censoring: n/a / NaN or T5 value <=0 in any endpoint -> invalid_measurement / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D002 `roi_left_pupil_range` (micro, roi_pupil_export)
- Formula: diffXpupL = max - min left pupil over fixations on r (minPupLB - maxPupLB; labels inverted with B=c-pupil)
- Unit: mm; scope: task_roi
- Sources: T1235.xlsx [diffXpupL,maxPupLB,minPupLB,startPupLB]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv; diffXpupL = minPupLB - maxPupLB exactly; maxPupLB <= startPupLB <= minPupLB in 97.2% of rows
- Dataset motivation: Only within-ROI pupil dispersion exported; constant-free (range)
- Closest existing: M042,M043; incremental: Range over exported fixation values of one eye, not sample IQR/P90-P10 (blocked)
- Support: >=1 fixation; range 0 is genuine when one fixation
- Zero/missing/censoring: n/a / NaN -> unavailable; start value outside [min,max] -> invalid_measurement / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235

### D003 `roi_leftward_saccade_count` (micro, roi_saccade_export)
- Formula: regSaccades: number of saccades with leftward gaze direction associated with r (Table II 'regressive saccades (gaze direction left)')
- Unit: count; scope: task_roi
- Sources: T1235.xlsx; T5.xlsx [regSaccades]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv; direction-left definition, not reading-order regressions; exceeds nTobiiSaccades in 2,850/6,691 T5 rows so it is used as a stand-alone count, never in a ratio
- Dataset motivation: Only exported directional count per ROI
- Closest existing: M019,M020; incremental: Direction-based leftward count, not rank-based
- Support: row present
- Zero/missing/censoring: 0 is an observed count when the ROI row exists / missing ROI row -> unavailable / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D004 `roi_summed_saccade_amplitude` (micro, roi_saccade_export)
- Formula: saccadeTotLength: sum of amplitudes of saccades on r (Table II)
- Unit: deg; scope: task_roi
- Sources: T1235.xlsx; T5.xlsx [saccadeTotLength]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv; strictly positive in every row; population not reconciled with nTobiiSaccades (count 0 with positive length in 3,307 T5 rows)
- Dataset motivation: Amplitude burden of oculomotor activity around one target
- Closest existing: M021,M023; incremental: Summed amplitude rather than median incoming/outgoing amplitude (blocked)
- Support: row present
- Zero/missing/censoring: n/a / missing ROI row -> unavailable / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D005 `roi_fixations_per_articulation_second` (micro, roi_gaze_speech_export)
- Formula: nFixations / totalSpokenTime
- Unit: count/s; scope: task_roi
- Sources: T1235.xlsx; T5.xlsx [nFixations,totalSpokenTime]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv; totalSpokenTime >= 0.048 s in all rows
- Dataset motivation: Visual sampling effort normalised by the time taken to articulate the same target (both person-varying)
- Closest existing: M001,M051,M080; incremental: Joint gaze-speech ratio; neither parent alone; reciprocal (articulation per fixation) rejected as rank alias
- Support: row present
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: articulation guard 0.05 s (file 02 default) (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D006 `roi_exit_landing_speech_offset_delay` (micro, roi_gaze_speech_export)
- Formula: endSpeechDelay: speech offset for r minus the exit-landing fixation (Table II caption convention)
- Unit: s; scope: task_roi
- Sources: T1235.xlsx; T5.xlsx [endSpeechDelay]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv; capped at 10.000 s; 7% (T5) / 17% (T1235) negative
- Dataset motivation: Exported coordination endpoint distinct from M062
- Closest existing: M062,M063; incremental: Different gaze endpoint than M062
- Support: row present
- Zero/missing/censoring: n/a / unavailable if support rule fails / 10 s cap
- Parameters: none (fixed a priori)
- Feasibility: blocked_semantics (fixation time point (onset/offset) of the exit-landing fixation is unspecified); datasets: none

### D007 `target_mean_fixation_count` (macro, roi_target_summary)
- Formula: Equal-ROI mean of M001 over observed target ROIs
- Unit: count/target; scope: KD1,KD2,KD3,READING
- Sources: T1235.xlsx; T5.xlsx [nFixations]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv
- Dataset motivation: Per-target fixation load summarised over the card/text
- Closest existing: A003,A010; incremental: Target-restricted mean, unlike unique task fixations (A003) or dwell (A010, blocked)
- Support: >=60% of expected ROIs observed
- Zero/missing/censoring: n/a / coverage gate fails -> insufficient_within_subject_support / none
- Parameters: ROI coverage 0.60 (sensitivity 0.50/0.75) (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D008 `refixated_target_fraction` (macro, roi_target_summary)
- Formula: Observed target ROIs with nFixations >= 2 / observed target ROIs
- Unit: fraction; scope: KD1,KD2,KD3,READING
- Sources: T1235.xlsx; T5.xlsx [nFixations]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv; 2 is the modal count, 1 occurs in ~5% of rows
- Dataset motivation: Prevalence of multiple fixations per target (refixation, not verified revisits)
- Closest existing: A011; incremental: Counts fixations not visits; A011 needs chronology (blocked)
- Support: >=60% of expected ROIs observed
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: ROI coverage 0.60 (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D009 `target_articulation_time_median` (macro, roi_target_summary)
- Formula: Equal-ROI median of M051
- Unit: s; scope: KD1,KD2,KD3,READING
- Sources: T1235.xlsx; T5.xlsx [totalSpokenTime]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv
- Dataset motivation: Typical per-target articulation time
- Closest existing: A051,A053; incremental: Per-target articulation; A051 needs syllables (blocked)
- Support: >=60% of expected ROIs observed
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: ROI coverage 0.60 (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D010 `target_pupil_change_median` (macro, roi_target_summary)
- Formula: Equal-ROI median of D001
- Unit: mm; scope: KD1,KD2,KD3,READING
- Sources: T1235.xlsx; T5.xlsx [diffPupL,diffPupR]
- Semantics evidence: see D001
- Dataset motivation: Typical entry-to-exit-landing pupil change
- Closest existing: A045,A048; incremental: Target-anchored change; time-window changes need continuous pupil (blocked)
- Support: >=60% of expected ROIs with valid D001
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: ROI coverage 0.60 (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D011 `target_left_pupil_range_median` (macro, roi_target_summary)
- Formula: Equal-ROI median of D002
- Unit: mm; scope: KD1,KD2,KD3,READING
- Sources: T1235.xlsx [diffXpupL]
- Semantics evidence: see D002
- Dataset motivation: Typical within-target pupil range
- Closest existing: A042,A043; incremental: Target-level range of exported fixation values
- Support: >=60% of expected ROIs with valid D002
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: ROI coverage 0.60 (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235

### D012 `leftward_saccade_target_fraction` (macro, roi_target_summary)
- Formula: Observed target ROIs with regSaccades >= 1 / observed target ROIs
- Unit: fraction; scope: KD1,KD2,KD3,READING
- Sources: T1235.xlsx; T5.xlsx [regSaccades]
- Semantics evidence: see D003
- Dataset motivation: Prevalence of leftward movement around targets
- Closest existing: A013; incremental: Direction-defined prevalence; A013 needs reading order (blocked)
- Support: >=60% of expected ROIs observed
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: ROI coverage 0.60 (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D013 `fixation_count_onset_delay_coupling` (macro, roi_coupling)
- Formula: Within-person Spearman rho between M001 and M061 across target ROIs
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: T1235.xlsx; T5.xlsx [nFixations,speechDelay]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv
- Dataset motivation: Whether targets that attract more fixations are named later
- Closest existing: A061,D007; incremental: Within-person association, not a location summary
- Support: >=10 ROIs with both valid; both non-constant
- Zero/missing/censoring: n/a / undefined correlation -> insufficient support / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D014 `articulation_onset_delay_coupling` (macro, roi_coupling)
- Formula: Within-person Spearman rho between M051 and M061 across target ROIs
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: T1235.xlsx; T5.xlsx [totalSpokenTime,speechDelay]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv
- Dataset motivation: Articulation time indexes target length/difficulty within the person (no lexical metadata); coupling describes latency sensitivity to it
- Closest existing: A071,A073; incremental: Uses observed articulation instead of unavailable letter counts
- Support: >=10 ROIs; both non-constant
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D015 `articulation_fixation_count_coupling` (macro, roi_coupling)
- Formula: Within-person Spearman rho between M051 and M001 across target ROIs
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: T1235.xlsx; T5.xlsx [totalSpokenTime,nFixations]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv
- Dataset motivation: Refixation sensitivity to target articulation length
- Closest existing: A071; incremental: Count-based length effect using articulation as within-person length index
- Support: >=10 ROIs; both non-constant
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235,ROI64_T5

### D016 `reading_kd_target_fixation_count_difference` (macro, roi_cross_task)
- Formula: D007@READING - mean(D007@KD1, D007@KD2, D007@KD3)
- Unit: count/target; scope: CROSS_BOTH
- Sources: T1235.xlsx [nFixations]
- Semantics evidence: Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles in semantic_audit.csv
- Dataset motivation: Word vs digit per-target fixation load in the same person
- Closest existing: A079; incremental: Paired task contrast of fixation load
- Support: all four task values valid
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: ROI64_T1235

### D017 `fixation_duration_serial_correlation` (macro, fixation_sequence)
- Formula: Spearman rho between durations of consecutive fixations joined by a gap-free transition
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [GazeEventDuration,RecordingTimestamp,SaccadeIndex]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Ordered fixation chronology is available; serial dependence of processing time is unmeasured by base
- Closest existing: A005,A006; incremental: Order dependence, invariant to the duration distribution
- Support: >=30 consecutive pairs
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D018 `fixation_duration_quartile_dispersion` (macro, fixation_distribution)
- Formula: (Q3 - Q1) / (Q3 + Q1) of fixation durations
- Unit: unitless; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [GazeEventDuration]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration
- Dataset motivation: Scale-free relative dispersion
- Closest existing: A006,A005; incremental: Not rank-equivalent to A006 across people (depends on location)
- Support: >=3 fixations; Q3+Q1>0
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D019 `fixation_duration_bowley_skewness` (macro, fixation_distribution)
- Formula: (Q3 + Q1 - 2*Q2) / (Q3 - Q1) of fixation durations
- Unit: unitless; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [GazeEventDuration]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration
- Dataset motivation: Right-skewed duration distributions observed pooled; tail asymmetry
- Closest existing: A007; incremental: Shape, not location or spread
- Support: >=10 fixations; Q3>Q1
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D020 `fixation_duration_successive_change_median` (macro, fixation_sequence)
- Formula: Median |d(k+1) - d(k)| over consecutive gap-free fixation pairs
- Unit: s; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [GazeEventDuration]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Local moment-to-moment variability
- Closest existing: A006; incremental: Local (successive) rather than global spread
- Support: >=10 pairs
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D021 `interfixation_interval_median` (macro, fixation_sequence)
- Formula: Median of onset(k+1) - offset(k) over gap-free transitions
- Unit: s; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [RecordingTimestamp,GazeEventDuration]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: ~30% of transitions contain several I-VT saccade events; the full movement interval is measurable
- Closest existing: A026; incremental: Whole fixation-to-fixation interval including split saccades
- Support: >=1 transition
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D022 `fixation_onset_burstiness` (macro, fixation_sequence)
- Formula: (sd - mean)/(sd + mean) of inter-onset intervals over gap-free transitions
- Unit: unitless; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [RecordingTimestamp]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Temporal regularity vs bursts of sampling
- Closest existing: A006; incremental: Regularity of the onset process
- Support: >=30 intervals
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D023 `gaze_event_span` (macro, fixation_sequence)
- Formula: Offset of last valid fixation - onset of first valid fixation within the task label
- Unit: s; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [RecordingTimestamp,GazeEventDuration,Task]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration
- Dataset motivation: Media onset is not exported but the gaze-engaged period is
- Closest existing: A001; incremental: Gaze-anchored span; excludes pre-first-fixation latency; A001 remains blocked
- Support: >=2 fixations
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D024 `saccadic_time_fraction` (macro, fixation_sequence)
- Formula: Sum of gap-free interfixation intervals / (that sum + durations of the fixations they join), within continuous segments
- Unit: fraction; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [RecordingTimestamp,GazeEventDuration]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Share of engaged time spent moving
- Closest existing: A002,A026; incremental: Gap-excluded time budget
- Support: >=10 transitions
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D025 `main_sequence_duration_slope` (macro, saccade_dynamics)
- Formula: Theil-Sen slope of matched saccade duration (s) on amplitude (deg)
- Unit: s/deg; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [SaccadeIndex,GazeEventDuration,SaccadicAmplitude]
- Semantics evidence: Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Matched duration and amplitude available
- Closest existing: A026,A028; incremental: Amplitude-duration relation, not either marginal
- Support: >=10 matched; >=3 distinct amplitudes
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D026 `amplitude_duration_coupling` (macro, saccade_dynamics)
- Formula: Spearman rho(amplitude, duration) over matched saccades
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [SaccadicAmplitude,GazeEventDuration]
- Semantics evidence: Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Strength of main-sequence coupling
- Closest existing: D025; incremental: Coupling strength vs slope
- Support: >=10 matched; both non-constant
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D027 `split_saccade_transition_fraction` (macro, saccade_dynamics)
- Formula: Gap-free transitions containing >=2 I-VT saccade events / gap-free transitions with >=1 saccade event
- Unit: fraction; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [SaccadeIndex,RecordingTimestamp]
- Semantics evidence: Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Observed 1 event in ~68% and >=2 in ~30% of intervals (label-blind)
- Closest existing: A021; incremental: Velocity-profile interruption (I-VT split) per movement
- Support: >=10 transitions with a saccade
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: I-VT 30 deg/s from export (export filter settings (paper))
- Feasibility: supported; datasets: RAW157

### D028 `leftward_transition_fraction` (macro, saccade_direction)
- Formula: Gap-free transitions with horizontal-dominant leftward displacement (dx<0, |dx|>|dy|) / gap-free transitions with displacement >= 1 deg
- Unit: fraction; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [FixationPointX/Y (MCSpx)]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Direction is measurable without ROI order
- Closest existing: A013,D003; incremental: Leftward movements (regressions and return sweeps together); not reading-order regressions
- Support: >=10 transitions
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: 1 deg minimum displacement (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D029 `rightward_saccade_amplitude_median` (macro, saccade_direction)
- Formula: Median amplitude of matched saccades with horizontal-dominant rightward displacement
- Unit: deg; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [SaccadicAmplitude,FixationPointX/Y (MCSpx)]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Forward (progressive) saccade length
- Closest existing: A023; incremental: Direction-specific amplitude
- Support: >=1 event (count recorded)
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D030 `leftward_saccade_amplitude_median` (macro, saccade_direction)
- Formula: Median amplitude of matched saccades with horizontal-dominant leftward displacement
- Unit: deg; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [SaccadicAmplitude,FixationPointX/Y (MCSpx)]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Backward movement size (short regressions vs long sweeps)
- Closest existing: A014,A023; incremental: Direction-specific amplitude
- Support: >=1 event (count recorded); conditioning: conditional on >=1 leftward matched saccade
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D031 `transition_direction_entropy` (macro, saccade_direction)
- Formula: -sum p_s log p_s / log 8 over 8 equal displacement-angle sectors of gap-free transitions >= 1 deg
- Unit: unitless; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [FixationPointX/Y (MCSpx)]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box
- Dataset motivation: Directional organisation of scanning
- Closest existing: A036; incremental: Direction distribution, not spatial-state transitions
- Support: >=30 transitions
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: 8 sectors, 1 deg minimum (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D032 `direction_reversal_fraction` (macro, saccade_direction)
- Formula: Consecutive gap-free transition pairs (both >= 1 deg) with turning angle > 150 deg / all such pairs
- Unit: fraction; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [FixationPointX/Y (MCSpx)]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box; turning angles, not reading-order regressions
- Dataset motivation: Back-and-forth checking
- Closest existing: A013; incremental: Geometric reversal without ROI order
- Support: >=30 pairs
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: 150 deg, 1 deg minimum (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D033 `post_large_saccade_correction_fraction` (macro, saccade_dynamics)
- Formula: Gap-free transitions >= 5 deg followed (via a fixation < 0.25 s) by a gap-free transition < 1 deg / gap-free transitions >= 5 deg with an observed next transition
- Unit: fraction; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [SaccadicAmplitude,GazeEventDuration]
- Semantics evidence: Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Small corrective movements after large ones
- Closest existing: A029; incremental: Sequential conditional pattern
- Support: >=5 large transitions; conditioning: large saccades
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: 5 deg, 1 deg, 0.25 s (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D034 `saccade_amplitude_serial_correlation` (macro, saccade_dynamics)
- Formula: Spearman rho between amplitudes of consecutive gap-free transitions
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [SaccadicAmplitude]
- Semantics evidence: Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Rhythmic alternation of short/long movements
- Closest existing: A024; incremental: Order dependence of amplitudes
- Support: >=30 pairs
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D035 `fixation_duration_outgoing_amplitude_coupling` (macro, fixation_saccade_coupling)
- Formula: Spearman rho(fixation duration, amplitude of the following gap-free transition)
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [GazeEventDuration,SaccadicAmplitude]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Planning time before larger movements
- Closest existing: A005,A023; incremental: Pre-saccadic coupling
- Support: >=30 pairs
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D036 `incoming_amplitude_fixation_duration_coupling` (macro, fixation_saccade_coupling)
- Formula: Spearman rho(amplitude of the preceding gap-free transition, fixation duration)
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [GazeEventDuration,SaccadicAmplitude]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Post-landing processing after larger movements
- Closest existing: D035; incremental: Post-saccadic coupling
- Support: >=30 pairs
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D037 `post_leftward_fixation_duration_log_ratio` (macro, fixation_saccade_coupling)
- Formula: log(median duration of fixations entered by leftward transitions / median entered by rightward)
- Unit: log ratio; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [GazeEventDuration,FixationPointX/Y (MCSpx)]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Processing after backward vs forward movements
- Closest existing: D028; incremental: Recovery-cost contrast within person
- Support: >=5 fixations in each direction
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D038 `immediate_return_fraction` (macro, fixation_sequence)
- Formula: Fixations k (with gap-free k-2 -> k-1 -> k) landing within 1 deg of fixation k-2 after a k-1 displacement >= 1 deg / all such fixations
- Unit: fraction; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [FixationPointX/Y (MCSpx),DistanceLeft/Right]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box; deg from mm/px and eye distance
- Dataset motivation: A->B->A ping-pong rechecking
- Closest existing: D032; incremental: Positional return, not turning angle
- Support: >=30 triplets
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: 1 deg radius; 0.2676 mm/px (screen scale verified from ADCS px/mm columns)
- Feasibility: supported; datasets: RAW157

### D039 `fixation_xy_spread_log_anisotropy` (macro, spatial_layout)
- Formula: log(SD of screen x / SD of screen y) over in-media fixation positions (px)
- Unit: log ratio; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [FixationPointX/Y (MCSpx)]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box
- Dataset motivation: Horizontal vs vertical coverage of the layout
- Closest existing: A033; incremental: Shape of spread rather than magnitude
- Support: >=10 fixations; SD_y>0
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D040 `vertical_progression_monotonicity` (macro, spatial_layout)
- Formula: Spearman rho(fixation onset time, screen y) over in-media fixations
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [FixationPointY (MCSpx),RecordingTimestamp]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box; Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration
- Dataset motivation: Top-to-bottom progression through a line-structured card/text
- Closest existing: A013,A074; incremental: Global ordering without ROI ranks
- Support: >=10 fixations
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D041 `off_media_fixation_time_fraction` (macro, spatial_layout)
- Formula: Duration of fixations whose gaze lies outside the stimulus media box / duration of all located fixations
- Unit: fraction; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [FixationPointX (MCSpx),GazePointX/Y (ADCSpx),MediaPos*]
- Semantics evidence: FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN exactly when the fixation's gaze samples lie outside the media box
- Dataset motivation: Gaze on screen but off the stimulus
- Closest existing: A038; incremental: Media-box based; A038 needs text ROIs (blocked)
- Support: >=10 located fixations
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D042 `within_fixation_pupil_change_median` (macro, pupil_event)
- Formula: Median over fixations of binocular pupil at the Final row minus binocular pupil at the onset row
- Unit: mm; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [PupilLeft,PupilRight,ValidityLeft,ValidityRight]
- Semantics evidence: PupilLeft/Right exported only on event onset/Final rows; validity code 4 = missing (NaN), 0 = valid; binocular value requires both eyes valid
- Dataset motivation: Two pupil samples bracket each fixation
- Closest existing: A045,A048; incremental: Intra-fixation change; baseline-invariant
- Support: >=10 fixations with both samples valid
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D043 `pupil_change_fixation_duration_coupling` (macro, pupil_event)
- Formula: Spearman rho(fixation duration, within-fixation binocular pupil change)
- Unit: correlation; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [PupilLeft,PupilRight,GazeEventDuration]
- Semantics evidence: PupilLeft/Right exported only on event onset/Final rows; validity code 4 = missing (NaN), 0 = valid; binocular value requires both eyes valid
- Dataset motivation: Dilation dynamics vs dwell
- Closest existing: D042; incremental: Coupling, not level
- Support: >=20 fixations
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D044 `fixation_pupil_successive_change_median` (macro, pupil_event)
- Formula: Median |mean pupil(k+1) - mean pupil(k)| for consecutive gap-free fixations (mean of valid binocular samples of each fixation)
- Unit: mm; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [PupilLeft,PupilRight]
- Semantics evidence: PupilLeft/Right exported only on event onset/Final rows; validity code 4 = missing (NaN), 0 = valid; binocular value requires both eyes valid; Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Event-sampled fluctuation (A049 needs 100-ms bins)
- Closest existing: A049; incremental: Fixation-indexed variability
- Support: >=10 pairs
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D045 `event_sampled_pupil_time_slope` (macro, pupil_event)
- Formula: Theil-Sen slope of fixation mean binocular pupil on fixation onset time (s)
- Unit: mm/s; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [PupilLeft,PupilRight,RecordingTimestamp]
- Semantics evidence: PupilLeft/Right exported only on event onset/Final rows; validity code 4 = missing (NaN), 0 = valid; binocular value requires both eyes valid
- Dataset motivation: Legacy pup_slope used fixation index; per-second slope requested by file 02
- Closest existing: A044; incremental: Event-sampled per-second trend; A044 bin rule unsatisfiable
- Support: >=10 fixations spanning >=2 s
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D046 `binocular_horizontal_disparity_iqr` (macro, binocular)
- Formula: IQR of (GazePointLeftX - GazePointRightX) in ADCSmm over fixation samples with both eyes valid
- Unit: mm; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv [GazePointLeftX (ADCSmm),GazePointRightX (ADCSmm)]
- Semantics evidence: Per-eye gaze mm columns exist in dataset/ (not dataset_v2); per-person disparity offsets large (calibration), so only within-task spread is used
- Dataset motivation: Stability of binocular alignment
- Closest existing: M048; incremental: Gaze (not pupil) binocular measure; offset-invariant
- Support: >=10 samples
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D047 `viewing_distance_iqr` (macro, posture)
- Formula: IQR of mean(DistanceLeft, DistanceRight) over fixation samples with both eyes valid
- Unit: mm; scope: KD1,KD2,KD3,READING
- Sources: dataset/20250305/*.csv [DistanceLeft,DistanceRight]
- Semantics evidence: No headrest (paper); distances 530-750 mm across files
- Dataset motivation: Head/posture stability during the task
- Closest existing: none; incremental: No base postural measure
- Support: >=10 samples
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D048 `reading_kd_fixation_duration_difference` (macro, raw_cross_task)
- Formula: A005@READING - mean(A005@KD1, A005@KD2, A005@KD3)
- Unit: s; scope: CROSS_BOTH
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [GazeEventDuration]
- Semantics evidence: Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms (300 Hz); offset = onset + GazeEventDuration
- Dataset motivation: Word vs digit processing time in the same person
- Closest existing: A076; incremental: Reading-vs-KD paired contrast
- Support: all four task values valid
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157

### D049 `kd3_kd1_saccade_count_log_ratio` (macro, raw_cross_task)
- Formula: log(A021@KD3) - log(A021@KD1)
- Unit: log ratio; scope: CROSS_KD
- Sources: dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv) [SaccadeIndex]
- Semantics evidence: Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle (r=0.99 vs geometry from ADCSmm and eye distance)
- Dataset motivation: Scanning effort scaling with card difficulty
- Closest existing: A077,A076; incremental: Count-based card contrast
- Support: both counts > 0
- Zero/missing/censoring: n/a / unavailable if support rule fails / none
- Parameters: none (fixed a priori)
- Feasibility: supported; datasets: RAW157
