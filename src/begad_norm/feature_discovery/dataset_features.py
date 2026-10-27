"""Dataset-derived D definitions (authored from diagnosis-hidden source inspection).

Authored 2026-10-27 after the source audit and before any diagnostic contrast
was computed. Evidence for each definition is recorded in ``source_semantics_evidence``
and ``dataset_motivation``; the inspection steps are summarised in
DATASET_PROPOSED_100_FEATURES.md. The extension is PARTIAL: see SHORTFALL_NOTE.
"""

from __future__ import annotations

import pandas as pd

from begad_norm.feature_discovery import EXTRACTION_VERSION

PER_TASK = "KD1,KD2,KD3,READING"

D_COLUMNS = [
    "feature_id", "origin", "level", "family", "name", "formula", "unit", "scope",
    "allowed_tasks", "source_paths", "source_columns", "source_semantics_evidence",
    "dataset_motivation", "closest_existing_ids", "incremental_information",
    "within_subject_support", "conditioning", "zero_rule", "missing_rule",
    "censoring_rule", "denominator_guard", "parameters", "parameter_origin",
    "alias_status", "feasibility_status", "blocked_reason", "supported_datasets",
    "extraction_version", "freeze_id",
]

SHORTFALL_NOTE = (
    "Only 49 defensible definitions could be authored. The repository has no ROI geometry, stimulus text, "
    "reading order, audio/TextGrid or error annotations, and the raw exports keep pupil only at event "
    "boundaries; the ROI workbooks expose 19 summary columns of which several are semantically unreconciled. "
    "Candidates that would need those sources, or that were renamed/rank-equivalent variants, were rejected "
    "(see proposal_audit.csv)."
)

WB = "T1235.xlsx; T5.xlsx"
RAW = "dataset/20250305/*.csv (baseline roster dataset/participants_20250305.csv)"
EV_WB = ("Paper Table II (TNSRE 2025, p.1452, inspected full text) defines the field; label-blind workbook profiles "
         "in semantic_audit.csv")
EV_FIX = ("Raw export: each fixation = onset row + Final row; Final ts - onset ts = duration - 3 ms "
          "(300 Hz); offset = onset + GazeEventDuration")
EV_TRANS = ("Gap-free transition: uncovered time between fixation k offset and k+1 onset after removing I-VT "
            "saccade events <= 10 ms; SaccadicAmplitude on landing fixation = fixation-to-fixation angle "
            "(r=0.99 vs geometry from ADCSmm and eye distance)")
EV_POS = ("FixationPoint (MCSpx) + MediaPos gives screen px; 0.2676 mm/px constant across files; MCS is NaN "
          "exactly when the fixation's gaze samples lie outside the media box")
EV_PUP = ("PupilLeft/Right exported only on event onset/Final rows; validity code 4 = missing (NaN), 0 = valid; "
          "binocular value requires both eyes valid")


def _d(fid, level, family, name, formula, unit, scope, allowed, paths, cols, evidence, motivation,
       closest, incremental, support, conditioning="", zero="", missing="unavailable if support rule fails",
       censor="none", guard="", params="", porigin="fixed a priori", alias="checked: no proven alias",
       status="supported", blocked="", datasets=""):
    return dict(
        feature_id=fid, origin="dataset100", level=level, family=family, name=name, formula=formula,
        unit=unit, scope=scope, allowed_tasks=allowed, source_paths=paths, source_columns=cols,
        source_semantics_evidence=evidence, dataset_motivation=motivation, closest_existing_ids=closest,
        incremental_information=incremental, within_subject_support=support, conditioning=conditioning,
        zero_rule=zero, missing_rule=missing, censoring_rule=censor, denominator_guard=guard,
        parameters=params, parameter_origin=porigin, alias_status=alias, feasibility_status=status,
        blocked_reason=blocked, supported_datasets=datasets,
    )


ROI = "ROI64_T1235,ROI64_T5"
T12 = "ROI64_T1235"
R157 = "RAW157"

D_DEFINITIONS = [
    # ---------------- micro (task-ROI) from the ROI workbooks ----------------
    _d("D001", "micro", "roi_pupil_export", "roi_entry_to_exit_landing_pupil_change",
       "Mean over eyes of (pupil at the exit-landing fixation - pupil at the first fixation on r): "
       "T1235 (diffPupL+diffPupR)/2 (= startB-endB, sign verified); T5 (diffPupL+diffPupR)/2 with both start/end >0",
       "mm", "task_roi", PER_TASK, WB, "diffPupL,diffPupR,startPup*,endPup*",
       EV_WB + "; T5 diff=end-start exactly; T1235 diff=startB-endB exactly and B=c-pupil (sum of T5 raw and "
       "T1235 B constant within participant, median SD 0.028), so the difference is baseline-invariant",
       "Workbooks export endpoint pupil per ROI under the Table II convention ('last fixation' = landing after "
       "the last saccade leaving r); M045/M046 need the last fixation inside r and are blocked",
       "M046,M044", "Different endpoint (post-exit landing fixation); M046 stays blocked",
       "both eyes valid at both endpoints", missing="NaN or T5 value <=0 in any endpoint -> invalid_measurement",
       datasets=ROI),
    _d("D002", "micro", "roi_pupil_export", "roi_left_pupil_range",
       "diffXpupL = max - min left pupil over fixations on r (minPupLB - maxPupLB; labels inverted with B=c-pupil)",
       "mm", "task_roi", PER_TASK, "T1235.xlsx", "diffXpupL,maxPupLB,minPupLB,startPupLB",
       EV_WB + "; diffXpupL = minPupLB - maxPupLB exactly; maxPupLB <= startPupLB <= minPupLB in 97.2% of rows",
       "Only within-ROI pupil dispersion exported; constant-free (range)",
       "M042,M043", "Range over exported fixation values of one eye, not sample IQR/P90-P10 (blocked)",
       ">=1 fixation; range 0 is genuine when one fixation",
       missing="NaN -> unavailable; start value outside [min,max] -> invalid_measurement",
       guard="consistency check startPupLB within [maxPupLB, minPupLB]", datasets=T12),
    _d("D003", "micro", "roi_saccade_export", "roi_leftward_saccade_count",
       "regSaccades: number of saccades with leftward gaze direction associated with r (Table II 'regressive "
       "saccades (gaze direction left)')", "count", "task_roi", PER_TASK, WB, "regSaccades",
       EV_WB + "; direction-left definition, not reading-order regressions; exceeds nTobiiSaccades in 2,850/6,691 "
       "T5 rows so it is used as a stand-alone count, never in a ratio",
       "Only exported directional count per ROI", "M019,M020", "Direction-based leftward count, not rank-based",
       "row present", zero="0 is an observed count when the ROI row exists",
       missing="missing ROI row -> unavailable", datasets=ROI),
    _d("D004", "micro", "roi_saccade_export", "roi_summed_saccade_amplitude",
       "saccadeTotLength: sum of amplitudes of saccades on r (Table II)", "deg", "task_roi", PER_TASK, WB,
       "saccadeTotLength", EV_WB + "; strictly positive in every row; population not reconciled with "
       "nTobiiSaccades (count 0 with positive length in 3,307 T5 rows)",
       "Amplitude burden of oculomotor activity around one target", "M021,M023",
       "Summed amplitude rather than median incoming/outgoing amplitude (blocked)", "row present",
       missing="missing ROI row -> unavailable", datasets=ROI),
    _d("D005", "micro", "roi_gaze_speech_export", "roi_fixations_per_articulation_second",
       "nFixations / totalSpokenTime", "count/s", "task_roi", PER_TASK, WB, "nFixations,totalSpokenTime",
       EV_WB + "; totalSpokenTime >= 0.048 s in all rows", "Visual sampling effort normalised by the time "
       "taken to articulate the same target (both person-varying)", "M001,M051,M080",
       "Joint gaze-speech ratio; neither parent alone; reciprocal (articulation per fixation) rejected as rank alias",
       "row present", guard="totalSpokenTime >= 0.05 s else insufficient_within_subject_support",
       params="articulation guard 0.05 s (file 02 default)", datasets=ROI),
    _d("D006", "micro", "roi_gaze_speech_export", "roi_exit_landing_speech_offset_delay",
       "endSpeechDelay: speech offset for r minus the exit-landing fixation (Table II caption convention)",
       "s", "task_roi", PER_TASK, WB, "endSpeechDelay",
       EV_WB + "; capped at 10.000 s; 7% (T5) / 17% (T1235) negative",
       "Exported coordination endpoint distinct from M062", "M062,M063",
       "Different gaze endpoint than M062", "row present", censor="10 s cap",
       status="blocked_semantics",
       blocked="fixation time point (onset/offset) of the exit-landing fixation is unspecified", datasets=""),
    # ---------------- macro from ROI workbooks (equal-ROI weighting) ----------------
    _d("D007", "macro", "roi_target_summary", "target_mean_fixation_count",
       "Equal-ROI mean of M001 over observed target ROIs", "count/target", PER_TASK, PER_TASK, WB, "nFixations",
       EV_WB, "Per-target fixation load summarised over the card/text", "A003,A010",
       "Target-restricted mean, unlike unique task fixations (A003) or dwell (A010, blocked)",
       ">=60% of expected ROIs observed", missing="coverage gate fails -> insufficient_within_subject_support",
       params="ROI coverage 0.60 (sensitivity 0.50/0.75)", datasets=ROI),
    _d("D008", "macro", "roi_target_summary", "refixated_target_fraction",
       "Observed target ROIs with nFixations >= 2 / observed target ROIs", "fraction", PER_TASK, PER_TASK, WB,
       "nFixations", EV_WB + "; 2 is the modal count, 1 occurs in ~5% of rows",
       "Prevalence of multiple fixations per target (refixation, not verified revisits)", "A011",
       "Counts fixations not visits; A011 needs chronology (blocked)", ">=60% of expected ROIs observed",
       params="ROI coverage 0.60", datasets=ROI),
    _d("D009", "macro", "roi_target_summary", "target_articulation_time_median",
       "Equal-ROI median of M051", "s", PER_TASK, PER_TASK, WB, "totalSpokenTime", EV_WB,
       "Typical per-target articulation time", "A051,A053", "Per-target articulation; A051 needs syllables (blocked)",
       ">=60% of expected ROIs observed", params="ROI coverage 0.60", datasets=ROI),
    _d("D010", "macro", "roi_target_summary", "target_pupil_change_median",
       "Equal-ROI median of D001", "mm", PER_TASK, PER_TASK, WB, "diffPupL,diffPupR",
       "see D001", "Typical entry-to-exit-landing pupil change", "A045,A048", "Target-anchored change; "
       "time-window changes need continuous pupil (blocked)", ">=60% of expected ROIs with valid D001",
       params="ROI coverage 0.60", datasets=ROI),
    _d("D011", "macro", "roi_target_summary", "target_left_pupil_range_median",
       "Equal-ROI median of D002", "mm", PER_TASK, PER_TASK, "T1235.xlsx", "diffXpupL", "see D002",
       "Typical within-target pupil range", "A042,A043", "Target-level range of exported fixation values",
       ">=60% of expected ROIs with valid D002", params="ROI coverage 0.60", datasets=T12),
    _d("D012", "macro", "roi_target_summary", "leftward_saccade_target_fraction",
       "Observed target ROIs with regSaccades >= 1 / observed target ROIs", "fraction", PER_TASK, PER_TASK, WB,
       "regSaccades", "see D003", "Prevalence of leftward movement around targets", "A013",
       "Direction-defined prevalence; A013 needs reading order (blocked)", ">=60% of expected ROIs observed",
       params="ROI coverage 0.60", datasets=ROI),
    _d("D013", "macro", "roi_coupling", "fixation_count_onset_delay_coupling",
       "Within-person Spearman rho between M001 and M061 across target ROIs", "correlation", PER_TASK, PER_TASK,
       WB, "nFixations,speechDelay", EV_WB, "Whether targets that attract more fixations are named later",
       "A061,D007", "Within-person association, not a location summary",
       ">=10 ROIs with both valid; both non-constant", missing="undefined correlation -> insufficient support",
       datasets=ROI),
    _d("D014", "macro", "roi_coupling", "articulation_onset_delay_coupling",
       "Within-person Spearman rho between M051 and M061 across target ROIs", "correlation", PER_TASK, PER_TASK,
       WB, "totalSpokenTime,speechDelay", EV_WB, "Articulation time indexes target length/difficulty within "
       "the person (no lexical metadata); coupling describes latency sensitivity to it", "A071,A073",
       "Uses observed articulation instead of unavailable letter counts", ">=10 ROIs; both non-constant",
       datasets=ROI),
    _d("D015", "macro", "roi_coupling", "articulation_fixation_count_coupling",
       "Within-person Spearman rho between M051 and M001 across target ROIs", "correlation", PER_TASK, PER_TASK,
       WB, "totalSpokenTime,nFixations", EV_WB, "Refixation sensitivity to target articulation length",
       "A071", "Count-based length effect using articulation as within-person length index",
       ">=10 ROIs; both non-constant", datasets=ROI),
    _d("D016", "macro", "roi_cross_task", "reading_kd_target_fixation_count_difference",
       "D007@READING - mean(D007@KD1, D007@KD2, D007@KD3)", "count/target", "CROSS_BOTH", "CROSS_BOTH",
       "T1235.xlsx", "nFixations", EV_WB, "Word vs digit per-target fixation load in the same person",
       "A079", "Paired task contrast of fixation load", "all four task values valid", datasets=T12),
    # ---------------- macro from raw157 events ----------------
    _d("D017", "macro", "fixation_sequence", "fixation_duration_serial_correlation",
       "Spearman rho between durations of consecutive fixations joined by a gap-free transition", "correlation",
       PER_TASK, PER_TASK, RAW, "GazeEventDuration,RecordingTimestamp,SaccadeIndex", EV_FIX + "; " + EV_TRANS,
       "Ordered fixation chronology is available; serial dependence of processing time is unmeasured by base",
       "A005,A006", "Order dependence, invariant to the duration distribution", ">=30 consecutive pairs",
       datasets=R157),
    _d("D018", "macro", "fixation_distribution", "fixation_duration_quartile_dispersion",
       "(Q3 - Q1) / (Q3 + Q1) of fixation durations", "unitless", PER_TASK, PER_TASK, RAW, "GazeEventDuration",
       EV_FIX, "Scale-free relative dispersion", "A006,A005", "Not rank-equivalent to A006 across people "
       "(depends on location)", ">=3 fixations; Q3+Q1>0", datasets=R157),
    _d("D019", "macro", "fixation_distribution", "fixation_duration_bowley_skewness",
       "(Q3 + Q1 - 2*Q2) / (Q3 - Q1) of fixation durations", "unitless", PER_TASK, PER_TASK, RAW,
       "GazeEventDuration", EV_FIX, "Right-skewed duration distributions observed pooled; tail asymmetry",
       "A007", "Shape, not location or spread", ">=10 fixations; Q3>Q1", datasets=R157),
    _d("D020", "macro", "fixation_sequence", "fixation_duration_successive_change_median",
       "Median |d(k+1) - d(k)| over consecutive gap-free fixation pairs", "s", PER_TASK, PER_TASK, RAW,
       "GazeEventDuration", EV_FIX + "; " + EV_TRANS, "Local moment-to-moment variability", "A006",
       "Local (successive) rather than global spread", ">=10 pairs", datasets=R157),
    _d("D021", "macro", "fixation_sequence", "interfixation_interval_median",
       "Median of onset(k+1) - offset(k) over gap-free transitions", "s", PER_TASK, PER_TASK, RAW,
       "RecordingTimestamp,GazeEventDuration", EV_FIX + "; " + EV_TRANS,
       "~30% of transitions contain several I-VT saccade events; the full movement interval is measurable",
       "A026", "Whole fixation-to-fixation interval including split saccades", ">=1 transition", datasets=R157),
    _d("D022", "macro", "fixation_sequence", "fixation_onset_burstiness",
       "(sd - mean)/(sd + mean) of inter-onset intervals over gap-free transitions", "unitless", PER_TASK,
       PER_TASK, RAW, "RecordingTimestamp", EV_FIX + "; " + EV_TRANS, "Temporal regularity vs bursts of sampling",
       "A006", "Regularity of the onset process", ">=30 intervals", datasets=R157),
    _d("D023", "macro", "fixation_sequence", "gaze_event_span",
       "Offset of last valid fixation - onset of first valid fixation within the task label", "s", PER_TASK,
       PER_TASK, RAW, "RecordingTimestamp,GazeEventDuration,Task", EV_FIX,
       "Media onset is not exported but the gaze-engaged period is", "A001",
       "Gaze-anchored span; excludes pre-first-fixation latency; A001 remains blocked",
       ">=2 fixations", datasets=R157),
    _d("D024", "macro", "fixation_sequence", "saccadic_time_fraction",
       "Sum of gap-free interfixation intervals / (that sum + durations of the fixations they join), within "
       "continuous segments", "fraction", PER_TASK, PER_TASK, RAW, "RecordingTimestamp,GazeEventDuration",
       EV_FIX + "; " + EV_TRANS, "Share of engaged time spent moving", "A002,A026",
       "Gap-excluded time budget", ">=10 transitions", datasets=R157),
    _d("D025", "macro", "saccade_dynamics", "main_sequence_duration_slope",
       "Theil-Sen slope of matched saccade duration (s) on amplitude (deg)", "s/deg", PER_TASK, PER_TASK, RAW,
       "SaccadeIndex,GazeEventDuration,SaccadicAmplitude", EV_TRANS, "Matched duration and amplitude available",
       "A026,A028", "Amplitude-duration relation, not either marginal", ">=10 matched; >=3 distinct amplitudes",
       datasets=R157),
    _d("D026", "macro", "saccade_dynamics", "amplitude_duration_coupling",
       "Spearman rho(amplitude, duration) over matched saccades", "correlation", PER_TASK, PER_TASK, RAW,
       "SaccadicAmplitude,GazeEventDuration", EV_TRANS, "Strength of main-sequence coupling", "D025",
       "Coupling strength vs slope", ">=10 matched; both non-constant", datasets=R157),
    _d("D027", "macro", "saccade_dynamics", "split_saccade_transition_fraction",
       "Gap-free transitions containing >=2 I-VT saccade events / gap-free transitions with >=1 saccade event",
       "fraction", PER_TASK, PER_TASK, RAW, "SaccadeIndex,RecordingTimestamp", EV_TRANS,
       "Observed 1 event in ~68% and >=2 in ~30% of intervals (label-blind)", "A021",
       "Velocity-profile interruption (I-VT split) per movement", ">=10 transitions with a saccade",
       params="I-VT 30 deg/s from export", porigin="export filter settings (paper)", datasets=R157),
    _d("D028", "macro", "saccade_direction", "leftward_transition_fraction",
       "Gap-free transitions with horizontal-dominant leftward displacement (dx<0, |dx|>|dy|) / gap-free "
       "transitions with displacement >= 1 deg", "fraction", PER_TASK, PER_TASK, RAW, "FixationPointX/Y (MCSpx)",
       EV_POS + "; " + EV_TRANS, "Direction is measurable without ROI order", "A013,D003",
       "Leftward movements (regressions and return sweeps together); not reading-order regressions",
       ">=10 transitions", params="1 deg minimum displacement", datasets=R157),
    _d("D029", "macro", "saccade_direction", "rightward_saccade_amplitude_median",
       "Median amplitude of matched saccades with horizontal-dominant rightward displacement", "deg", PER_TASK,
       PER_TASK, RAW, "SaccadicAmplitude,FixationPointX/Y (MCSpx)", EV_POS + "; " + EV_TRANS,
       "Forward (progressive) saccade length", "A023", "Direction-specific amplitude", ">=1 event (count recorded)",
       datasets=R157),
    _d("D030", "macro", "saccade_direction", "leftward_saccade_amplitude_median",
       "Median amplitude of matched saccades with horizontal-dominant leftward displacement", "deg", PER_TASK,
       PER_TASK, RAW, "SaccadicAmplitude,FixationPointX/Y (MCSpx)", EV_POS + "; " + EV_TRANS,
       "Backward movement size (short regressions vs long sweeps)", "A014,A023", "Direction-specific amplitude",
       ">=1 event (count recorded)", conditioning="conditional on >=1 leftward matched saccade", datasets=R157),
    _d("D031", "macro", "saccade_direction", "transition_direction_entropy",
       "-sum p_s log p_s / log 8 over 8 equal displacement-angle sectors of gap-free transitions >= 1 deg",
       "unitless", PER_TASK, PER_TASK, RAW, "FixationPointX/Y (MCSpx)", EV_POS,
       "Directional organisation of scanning", "A036", "Direction distribution, not spatial-state transitions",
       ">=30 transitions", params="8 sectors, 1 deg minimum", datasets=R157),
    _d("D032", "macro", "saccade_direction", "direction_reversal_fraction",
       "Consecutive gap-free transition pairs (both >= 1 deg) with turning angle > 150 deg / all such pairs",
       "fraction", PER_TASK, PER_TASK, RAW, "FixationPointX/Y (MCSpx)", EV_POS + "; turning angles, not "
       "reading-order regressions", "Back-and-forth checking", "A013", "Geometric reversal without ROI order",
       ">=30 pairs", params="150 deg, 1 deg minimum", datasets=R157),
    _d("D033", "macro", "saccade_dynamics", "post_large_saccade_correction_fraction",
       "Gap-free transitions >= 5 deg followed (via a fixation < 0.25 s) by a gap-free transition < 1 deg / "
       "gap-free transitions >= 5 deg with an observed next transition", "fraction", PER_TASK, PER_TASK, RAW,
       "SaccadicAmplitude,GazeEventDuration", EV_TRANS, "Small corrective movements after large ones",
       "A029", "Sequential conditional pattern", ">=5 large transitions", conditioning="large saccades",
       params="5 deg, 1 deg, 0.25 s", datasets=R157),
    _d("D034", "macro", "saccade_dynamics", "saccade_amplitude_serial_correlation",
       "Spearman rho between amplitudes of consecutive gap-free transitions", "correlation", PER_TASK, PER_TASK,
       RAW, "SaccadicAmplitude", EV_TRANS, "Rhythmic alternation of short/long movements", "A024",
       "Order dependence of amplitudes", ">=30 pairs", datasets=R157),
    _d("D035", "macro", "fixation_saccade_coupling", "fixation_duration_outgoing_amplitude_coupling",
       "Spearman rho(fixation duration, amplitude of the following gap-free transition)", "correlation",
       PER_TASK, PER_TASK, RAW, "GazeEventDuration,SaccadicAmplitude", EV_FIX + "; " + EV_TRANS,
       "Planning time before larger movements", "A005,A023", "Pre-saccadic coupling", ">=30 pairs",
       datasets=R157),
    _d("D036", "macro", "fixation_saccade_coupling", "incoming_amplitude_fixation_duration_coupling",
       "Spearman rho(amplitude of the preceding gap-free transition, fixation duration)", "correlation",
       PER_TASK, PER_TASK, RAW, "GazeEventDuration,SaccadicAmplitude", EV_FIX + "; " + EV_TRANS,
       "Post-landing processing after larger movements", "D035", "Post-saccadic coupling", ">=30 pairs",
       datasets=R157),
    _d("D037", "macro", "fixation_saccade_coupling", "post_leftward_fixation_duration_log_ratio",
       "log(median duration of fixations entered by leftward transitions / median entered by rightward)",
       "log ratio", PER_TASK, PER_TASK, RAW, "GazeEventDuration,FixationPointX/Y (MCSpx)",
       EV_POS + "; " + EV_TRANS, "Processing after backward vs forward movements", "D028",
       "Recovery-cost contrast within person", ">=5 fixations in each direction", datasets=R157),
    _d("D038", "macro", "fixation_sequence", "immediate_return_fraction",
       "Fixations k (with gap-free k-2 -> k-1 -> k) landing within 1 deg of fixation k-2 after a k-1 "
       "displacement >= 1 deg / all such fixations", "fraction", PER_TASK, PER_TASK, RAW,
       "FixationPointX/Y (MCSpx),DistanceLeft/Right", EV_POS + "; deg from mm/px and eye distance",
       "A->B->A ping-pong rechecking", "D032", "Positional return, not turning angle", ">=30 triplets",
       params="1 deg radius; 0.2676 mm/px", porigin="screen scale verified from ADCS px/mm columns",
       datasets=R157),
    _d("D039", "macro", "spatial_layout", "fixation_xy_spread_log_anisotropy",
       "log(SD of screen x / SD of screen y) over in-media fixation positions (px)", "log ratio", PER_TASK,
       PER_TASK, RAW, "FixationPointX/Y (MCSpx)", EV_POS, "Horizontal vs vertical coverage of the layout",
       "A033", "Shape of spread rather than magnitude", ">=10 fixations; SD_y>0", datasets=R157),
    _d("D040", "macro", "spatial_layout", "vertical_progression_monotonicity",
       "Spearman rho(fixation onset time, screen y) over in-media fixations", "correlation", PER_TASK, PER_TASK,
       RAW, "FixationPointY (MCSpx),RecordingTimestamp", EV_POS + "; " + EV_FIX,
       "Top-to-bottom progression through a line-structured card/text", "A013,A074",
       "Global ordering without ROI ranks", ">=10 fixations", datasets=R157),
    _d("D041", "macro", "spatial_layout", "off_media_fixation_time_fraction",
       "Duration of fixations whose gaze lies outside the stimulus media box / duration of all located fixations",
       "fraction", PER_TASK, PER_TASK, RAW, "FixationPointX (MCSpx),GazePointX/Y (ADCSpx),MediaPos*",
       EV_POS, "Gaze on screen but off the stimulus", "A038", "Media-box based; A038 needs text ROIs (blocked)",
       ">=10 located fixations", datasets=R157),
    _d("D042", "macro", "pupil_event", "within_fixation_pupil_change_median",
       "Median over fixations of binocular pupil at the Final row minus binocular pupil at the onset row", "mm",
       PER_TASK, PER_TASK, RAW, "PupilLeft,PupilRight,ValidityLeft,ValidityRight", EV_PUP,
       "Two pupil samples bracket each fixation", "A045,A048", "Intra-fixation change; baseline-invariant",
       ">=10 fixations with both samples valid", datasets=R157),
    _d("D043", "macro", "pupil_event", "pupil_change_fixation_duration_coupling",
       "Spearman rho(fixation duration, within-fixation binocular pupil change)", "correlation", PER_TASK,
       PER_TASK, RAW, "PupilLeft,PupilRight,GazeEventDuration", EV_PUP, "Dilation dynamics vs dwell", "D042",
       "Coupling, not level", ">=20 fixations", datasets=R157),
    _d("D044", "macro", "pupil_event", "fixation_pupil_successive_change_median",
       "Median |mean pupil(k+1) - mean pupil(k)| for consecutive gap-free fixations (mean of valid binocular "
       "samples of each fixation)", "mm", PER_TASK, PER_TASK, RAW, "PupilLeft,PupilRight", EV_PUP + "; " + EV_TRANS,
       "Event-sampled fluctuation (A049 needs 100-ms bins)", "A049", "Fixation-indexed variability",
       ">=10 pairs", datasets=R157),
    _d("D045", "macro", "pupil_event", "event_sampled_pupil_time_slope",
       "Theil-Sen slope of fixation mean binocular pupil on fixation onset time (s)", "mm/s", PER_TASK,
       PER_TASK, RAW, "PupilLeft,PupilRight,RecordingTimestamp", EV_PUP,
       "Legacy pup_slope used fixation index; per-second slope requested by file 02", "A044",
       "Event-sampled per-second trend; A044 bin rule unsatisfiable", ">=10 fixations spanning >=2 s",
       datasets=R157),
    _d("D046", "macro", "binocular", "binocular_horizontal_disparity_iqr",
       "IQR of (GazePointLeftX - GazePointRightX) in ADCSmm over fixation samples with both eyes valid", "mm",
       PER_TASK, PER_TASK, "dataset/20250305/*.csv", "GazePointLeftX (ADCSmm),GazePointRightX (ADCSmm)",
       "Per-eye gaze mm columns exist in dataset/ (not dataset_v2); per-person disparity offsets large "
       "(calibration), so only within-task spread is used", "Stability of binocular alignment",
       "M048", "Gaze (not pupil) binocular measure; offset-invariant", ">=10 samples", datasets=R157),
    _d("D047", "macro", "posture", "viewing_distance_iqr",
       "IQR of mean(DistanceLeft, DistanceRight) over fixation samples with both eyes valid", "mm", PER_TASK,
       PER_TASK, "dataset/20250305/*.csv", "DistanceLeft,DistanceRight",
       "No headrest (paper); distances 530-750 mm across files", "Head/posture stability during the task",
       "", "No base postural measure", ">=10 samples", datasets=R157),
    _d("D048", "macro", "raw_cross_task", "reading_kd_fixation_duration_difference",
       "A005@READING - mean(A005@KD1, A005@KD2, A005@KD3)", "s", "CROSS_BOTH", "CROSS_BOTH", RAW,
       "GazeEventDuration", EV_FIX, "Word vs digit processing time in the same person", "A076",
       "Reading-vs-KD paired contrast", "all four task values valid", datasets=R157),
    _d("D049", "macro", "raw_cross_task", "kd3_kd1_saccade_count_log_ratio",
       "log(A021@KD3) - log(A021@KD1)", "log ratio", "CROSS_KD", "CROSS_KD", RAW, "SaccadeIndex", EV_TRANS,
       "Scanning effort scaling with card difficulty", "A077,A076", "Count-based card contrast",
       "both counts > 0", datasets=R157),
]


def dataset_registry(freeze_id: str = "unfrozen") -> pd.DataFrame:
    rows = []
    for d in D_DEFINITIONS:
        rows.append({**d, "extraction_version": EXTRACTION_VERSION, "freeze_id": freeze_id})
    df = pd.DataFrame(rows, columns=D_COLUMNS)
    validate_d_definitions(df)
    # registry view
    out = pd.DataFrame({
        "feature_id": df.feature_id, "origin": df.origin, "level": df.level, "family": df.family,
        "name": df.name, "formula": df.formula, "unit": df.unit,
        "source_requirements": df.source_columns, "allowed_tasks": df.allowed_tasks, "scope": df.scope,
        "within_subject_support": df.within_subject_support, "conditioning": df.conditioning,
        "alias_group": "", "supported_datasets": df.supported_datasets,
        "status": df.feasibility_status, "status_reason": df.blocked_reason, "semantic_flags": "",
        "extraction_version": df.extraction_version, "freeze_id": df.freeze_id,
    })
    return out


def d_table(freeze_id: str = "unfrozen") -> pd.DataFrame:
    df = pd.DataFrame([{**d, "extraction_version": EXTRACTION_VERSION, "freeze_id": freeze_id}
                       for d in D_DEFINITIONS], columns=D_COLUMNS)
    validate_d_definitions(df)
    return df


REQUIRED_NONEMPTY = ["feature_id", "level", "family", "name", "formula", "unit", "scope", "source_columns",
                     "source_semantics_evidence", "dataset_motivation", "incremental_information",
                     "within_subject_support", "feasibility_status"]


def validate_d_definitions(df: pd.DataFrame) -> None:
    ids = list(df.feature_id)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate D IDs")
    expected = [f"D{i:03d}" for i in range(1, len(ids) + 1)]
    if ids != expected:
        raise ValueError("D IDs must be contiguous from D001 without placeholders")
    if len(ids) > 100:
        raise ValueError("at most 100 D definitions")
    if not set(df.level) <= {"micro", "macro"}:
        raise ValueError("D level must be micro or macro")
    for c in REQUIRED_NONEMPTY:
        empty = df[c].astype(str).str.strip().eq("")
        if empty.any():
            raise ValueError(f"D definitions with empty {c}: {list(df.feature_id[empty])}")
    names = df.name.str.lower()
    if names.duplicated().any():
        raise ValueError("duplicate D names")
    supported = df.feasibility_status.eq("supported")
    if (supported & df.supported_datasets.eq("")).any():
        raise ValueError("supported D definitions need supported_datasets")
