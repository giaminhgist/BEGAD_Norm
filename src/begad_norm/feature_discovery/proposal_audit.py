"""Proposal decisions for the dataset-derived extension (accepted, rejected, aliased, deferred)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from begad_norm.feature_discovery.dataset_features import d_table

# Candidates considered during the label-blind proposal step and NOT assigned a D ID.
REJECTED = [
    ("roi_articulation_time_per_fixation", "aliased", "D005",
     "totalSpokenTime/nFixations", "Reciprocal of D005: strictly rank-equivalent on the same support"),
    ("roi_export_incident_saccade_count", "rejected", "M029;M030",
     "nTobiiSaccades", "0 with positive saccadeTotLength in 3,307 T5 rows; population undefined"),
    ("roi_tiny_or_long_saccade_fraction", "rejected", "M029;M030",
     "tinySaccades,longSaccades,nTobiiSaccades", "Numerators exceed denominator in thousands of rows"),
    ("roi_leftward_saccade_fraction", "rejected", "D003",
     "regSaccades/nTobiiSaccades", "Ratio of unreconciled counts (file 02 forbids)"),
    ("roi_saccade_amplitude_per_fixation", "rejected", "D004",
     "saccadeTotLength/nFixations", "Ratio across unreconciled event populations"),
    ("roi_total_fixation_time", "rejected", "M008",
     "totalFixTime", "Not a summed duration (0 for single fixations; ~1.7 ms per fixation in ~45% of rows)"),
    ("roi_first_fixation_pupil_left_and_right", "rejected", "M044",
     "startPupL/R(B)", "Per-eye duplicates of blocked M044; T1235 constant/sign unresolved"),
    ("roi_exit_landing_pupil", "rejected", "M045",
     "endPupL/R(B)", "Absolute level under unresolved constant; only the constant-free change D001 kept"),
    ("roi_interocular_entry_pupil_difference", "rejected", "M048",
     "startPupL-startPupR", "Per-eye constants differ in T1235; device/ocular confound; left/right duplication"),
    ("roi_onset_delay_per_articulation", "rejected", "M076",
     "speechDelay/totalSpokenTime", "No defensible interpretation without first-utterance intervals"),
    ("roi_onset_minus_end_speech_delay", "rejected", "M062",
     "speechDelay-endSpeechDelay", "Depends on endSpeechDelay endpoint (D006 blocked)"),
    ("target_long_latency_fraction", "rejected", "A063",
     "speechDelay>2 s", "Pooled label-blind distribution has a multi-second mode consistent with early "
     "preview fixations; threshold meaning ambiguous; tail covered by A063"),
    ("target_articulation_iqr", "rejected", "D009", "totalSpokenTime IQR", "Statistic variant without new question"),
    ("target_onset_delay_position_trend", "deferred", "A074",
     "speechDelay vs ROI ID", "ROI ID order not verified as reading order"),
    ("raw_return_sweep_rate", "rejected", "A019",
     "large left-down saccades", "Would infer line layout from gaze (pseudo-ROI geometry)"),
    ("raw_fixations_per_span_second", "rejected", "A004",
     "A003/D023", "Proxy for blocked A004"),
    ("raw_kd3_kd1_span_log_ratio", "rejected", "A077", "D023 ratio", "Proxy for blocked A077"),
    ("raw_late_early_fixation_duration_by_order", "rejected", "A075",
     "fixation order thirds", "Proxy for blocked A075 (elapsed-time windows need task onset)"),
    ("raw_prolonged_fixation_fraction", "rejected", "A007",
     ">500 ms fraction", "Threshold variant of the tail already covered by A007"),
    ("raw_nearest_neighbour_fixation_distance", "rejected", "A033",
     "pixel NN distance", "Reproduces a legacy screened feature (geo_nn_dist_mean); spatial redundancy with A033"),
    ("raw_spatial_revisit_rate_60px", "rejected", "A011",
     "60 px radius", "Reproduces a legacy screened feature; arbitrary pixel radius"),
    ("raw_binocular_disparity_median", "rejected", "D046",
     "GazeLeftX-GazeRightX median", "Dominated by per-person calibration offsets (medians -0.1 to 39 mm)"),
    ("raw_viewing_distance_median", "rejected", "D047",
     "DistanceLeft/Right median", "Set-up/posture nuisance; kept as quality/confound only"),
    ("raw_task_interocular_pupil_difference", "rejected", "M048", "PupilLeft-PupilRight", "Device/ocular confound"),
    ("raw_saccade_pupil_change", "rejected", "D042", "pupil across saccade rows", "Pupil foreshortening artifact"),
    ("raw_pupil_gaze_position_coupling", "rejected", "", "pupil vs gaze y", "Measurement confound, not behaviour"),
    ("raw_tracking_gap_fraction", "rejected", "", "uncovered transition time",
     "Tracking/retention quality; reported in quality_features.csv"),
    ("raw_task_onset_pupil_vs_baseline", "rejected", "A050", "first fixations minus baseline",
     "Baseline rule not reproduced (S23)"),
    ("raw_saccade_mean_speed_iqr", "rejected", "A028", "speed IQR", "Statistic variant"),
    ("raw_downward_transition_fraction", "rejected", "D040", "dy>0 fraction", "Overlaps D028/D031/D040"),
    ("raw_kd_cross_card_cv", "rejected", "A076", "CV of 3 card medians", "Three-value dispersion; unstable"),
    ("raw_peak_saccade_velocity", "rejected", "A028", "", "Only 2 rows per saccade; trajectory not sampled"),
    ("raw_microsaccade_rate", "rejected", "M029", "", "I-VT export drops <60 ms fixations; no raw samples"),
    ("raw_blink_rate", "rejected", "", "", "Blinks not exported as events; validity only at event rows"),
    ("speech_pause_and_error_measures", "rejected", "A051-A060", "", "No audio/TextGrid/annotations (blocked source)"),
    ("lexical_length_frequency_responses", "rejected", "A071-A073", "", "No word metadata (blocked source)"),
    ("raw_within_roi_landing_geometry", "rejected", "M031-M040", "", "No ROI bounding boxes (blocked source)"),
    ("practice_and_instruction_task_features", "rejected", "", "Tasks 0,1,2,6,7,8,10",
     "Instantiation of the same formulas on non-canonical tasks, not new definitions"),
    ("per_eye_versions_of_D042_D046", "rejected", "D042", "", "Left/right duplication without separate question"),
]


def proposal_audit() -> pd.DataFrame:
    d = d_table()
    acc = pd.DataFrame({
        "candidate": d["name"], "decision": np.where(d["feasibility_status"] == "supported", "accepted",
                                                     "accepted_defined_but_blocked"),
        "assigned_id": d["feature_id"], "level": d["level"], "closest_existing_ids": d["closest_existing_ids"],
        "source_evidence": d["source_semantics_evidence"], "reason": d["incremental_information"],
    })
    rej = pd.DataFrame([dict(candidate=c, decision=dec, assigned_id="", level="",
                             closest_existing_ids=close, source_evidence=src, reason=why)
                        for c, dec, close, src, why in REJECTED])
    return pd.concat([acc, rej], ignore_index=True)
