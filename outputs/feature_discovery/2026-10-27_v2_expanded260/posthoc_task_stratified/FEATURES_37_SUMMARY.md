# 37 task-level features: definitions and p-values

Source run: `2026-10-27_v2_expanded260` (freeze in `candidate_freeze.json`). Cohorts: **RAW157** = raw Tobii event exports, 157 baseline participants (CN 100 / MCI 41 / AD 16); **ROI64_T1235** = original ROI workbook, 64 participants (CN 32 / MCI 17 / AD 15). The two cohorts are never pooled.

Selection rule (post hoc, requested after results were seen): a feature is listed if it meets at least one of (1) at least 2 tasks with p < 0.10 for both CN vs MCI and CN vs AD; (2) at least 1 task with MCI vs AD p < 0.10; (3) at least 1 task with any pairwise p < 0.05. 37 of 65 testable task-level features qualify; all 37 meet condition 3, which by itself is expected for roughly half of features under no true difference (12 tests per feature). The list is exploratory, not confirmed.

Tests: two-sided Mann-Whitney U (asymptotic, tie and continuity corrected), one value per participant and task. p values are raw (uncorrected). Durations in seconds, amplitudes in degrees, pupil in mm.

## 1. Definitions

| ID | Feature | Cohort | Origin | Meaning | Formula | Unit | Conditions met |
|---|---|---|---|---|---|---|---|
| A002 | `total_fixation_time` | RAW157 | base160 | Total time spent fixating during the task. | Sum of durations of unique valid within-task fixation events | s | 1, 2, 3 |
| A003 | `task_fixation_count` | RAW157 | base160 | Number of fixations in the task. | Number of unique valid within-task fixation events | count | 1, 2, 3 |
| A021 | `task_saccade_count` | RAW157 | base160 | Number of saccade events in the task. | Count of unique validated within-task saccade events | count | 1, 3 |
| A023 | `amplitude_median` | RAW157 | base160 | Typical saccade size. | Median measured saccade amplitude | deg | 3 |
| A024 | `amplitude_iqr` | RAW157 | base160 | Spread of saccade sizes. | Q75−Q25 of measured saccade amplitudes | deg | 2, 3 |
| A026 | `saccade_duration_median` | RAW157 | base160 | Typical saccade duration. | Median actual saccade duration | s | 2, 3 |
| A027 | `saccade_duration_iqr` | RAW157 | base160 | Spread of saccade durations. | Q75−Q25 of actual saccade durations | s | 2, 3 |
| A028 | `saccade_mean_speed_median` | RAW157 | base160 | Typical average saccade speed (not peak velocity). | Median of event amplitude divided by its actual matched duration | deg/s | 3 |
| A029 | `task_tiny_saccade_fraction` | RAW157 | base160 | Share of very small saccades (<1 degree). | Saccades with amplitude<1 degree divided by all valid task saccades | fraction | 3 |
| A030 | `task_long_saccade_fraction` | RAW157 | base160 | Share of long saccades (>2 degrees). | Saccades with amplitude>2 degrees divided by all valid task saccades | fraction | 3 |
| A031 | `angular_scanpath_length` | RAW157 | base160 | Total eye-movement path length in degrees. | Sum of unique valid within-task measured saccade amplitudes | deg | 3 |
| A033 | `normalized_screen_dispersion` | RAW157 | base160 | How widely fixations spread over the screen. | RMS distance from mean normalized screen fixation position | normalized distance | 3 |
| A035 | `task_spatial_entropy` | RAW157 | base160 | How evenly fixations cover an 8x6 screen grid. | −sum(p_b log(p_b))/log(48) over a fixed 8x6 screen grid | unitless | 2, 3 |
| A046 | `task_absolute_pupil_median` | RAW157 | base160 | Typical absolute pupil size (sensitive to lighting). | Median valid uncorrected binocular pupil | pupil unit | 1, 3 |
| D017 | `fixation_duration_serial_correlation` | RAW157 | dataset100 | Whether consecutive fixation durations resemble each other (serial dependence). | Spearman rho between durations of consecutive fixations joined by a gap-free transition | correlation | 3 |
| D018 | `fixation_duration_quartile_dispersion` | RAW157 | dataset100 | Relative (scale-free) spread of fixation durations. | (Q3 - Q1) / (Q3 + Q1) of fixation durations | unitless | 3 |
| D021 | `interfixation_interval_median` | RAW157 | dataset100 | Typical gap between one fixation ending and the next starting (movement time). | Median of onset(k+1) - offset(k) over gap-free transitions | s | 2, 3 |
| D022 | `fixation_onset_burstiness` | RAW157 | dataset100 | Regular vs bursty timing of fixation onsets. | (sd - mean)/(sd + mean) of inter-onset intervals over gap-free transitions | unitless | 3 |
| D023 | `gaze_event_span` | RAW157 | dataset100 | Time from the first to the last fixation of the task (gaze-engaged duration). | Offset of last valid fixation - onset of first valid fixation within the task label | s | 1, 2, 3 |
| D024 | `saccadic_time_fraction` | RAW157 | dataset100 | Share of engaged time spent moving the eyes rather than fixating. | Sum of gap-free interfixation intervals / (that sum + durations of the fixations they join), within continuous segments | fraction | 2, 3 |
| D025 | `main_sequence_duration_slope` | RAW157 | dataset100 | Extra saccade duration per extra degree of amplitude (main-sequence slope). | Theil-Sen slope of matched saccade duration (s) on amplitude (deg) | s/deg | 2, 3 |
| D026 | `amplitude_duration_coupling` | RAW157 | dataset100 | How tightly saccade duration follows amplitude. | Spearman rho(amplitude, duration) over matched saccades | correlation | 2, 3 |
| D028 | `leftward_transition_fraction` | RAW157 | dataset100 | Share of eye movements going leftward (regressions and line returns together). | Gap-free transitions with horizontal-dominant leftward displacement (dx<0, /dx/>/dy/) / gap-free transitions with displacement >= 1 deg | fraction | 2, 3 |
| D029 | `rightward_saccade_amplitude_median` | RAW157 | dataset100 | Typical size of forward (rightward) saccades. | Median amplitude of matched saccades with horizontal-dominant rightward displacement | deg | 3 |
| D030 | `leftward_saccade_amplitude_median` | RAW157 | dataset100 | Typical size of backward (leftward) saccades. | Median amplitude of matched saccades with horizontal-dominant leftward displacement | deg | 3 |
| D034 | `saccade_amplitude_serial_correlation` | RAW157 | dataset100 | Whether consecutive saccade sizes resemble each other. | Spearman rho between amplitudes of consecutive gap-free transitions | correlation | 3 |
| D035 | `fixation_duration_outgoing_amplitude_coupling` | RAW157 | dataset100 | Link between fixation duration and the size of the next saccade (planning). | Spearman rho(fixation duration, amplitude of the following gap-free transition) | correlation | 3 |
| D036 | `incoming_amplitude_fixation_duration_coupling` | RAW157 | dataset100 | Link between the size of the incoming saccade and the following fixation duration. | Spearman rho(amplitude of the preceding gap-free transition, fixation duration) | correlation | 3 |
| D037 | `post_leftward_fixation_duration_log_ratio` | RAW157 | dataset100 | Fixation duration after leftward vs rightward movements (cost of going back). | log(median duration of fixations entered by leftward transitions / median entered by rightward) | log ratio | 3 |
| D039 | `fixation_xy_spread_log_anisotropy` | RAW157 | dataset100 | Horizontal vs vertical spread of fixations. | log(SD of screen x / SD of screen y) over in-media fixation positions (px) | log ratio | 3 |
| D040 | `vertical_progression_monotonicity` | RAW157 | dataset100 | How consistently gaze moves from top to bottom over time. | Spearman rho(fixation onset time, screen y) over in-media fixations | correlation | 3 |
| D041 | `off_media_fixation_time_fraction` | RAW157 | dataset100 | Share of fixation time on the screen but outside the stimulus image. | Duration of fixations whose gaze lies outside the stimulus media box / duration of all located fixations | fraction | 3 |
| D047 | `viewing_distance_iqr` | RAW157 | dataset100 | Variability of eye-to-screen distance (head/posture stability). | IQR of mean(DistanceLeft, DistanceRight) over fixation samples with both eyes valid | mm | 2, 3 |
| D007 | `target_mean_fixation_count` | ROI64_T1235 | dataset100 | Average number of fixations per target (digit/word). | Equal-ROI mean of M001 over observed target ROIs | count/target | 1, 2, 3 |
| D010 | `target_pupil_change_median` | ROI64_T1235 | dataset100 | Typical pupil change from first fixation on a target to the landing after leaving it. | Equal-ROI median of D001 | mm | 2, 3 |
| D011 | `target_left_pupil_range_median` | ROI64_T1235 | dataset100 | Typical within-target range of left pupil size. | Equal-ROI median of D002 | mm | 2, 3 |
| D015 | `articulation_fixation_count_coupling` | ROI64_T1235 | dataset100 | Within-person link between how long a target is spoken and how many fixations it gets. | Within-person Spearman rho between M051 and M001 across target ROIs | correlation | 3 |

Notes on sources and definitions:
- RAW157 fixation and saccade events are rebuilt from the onset and final export rows; offset = onset + duration. A 'gap-free transition' joins two consecutive fixations of the same task whose interval is covered by saccade events up to 10 ms; gaps are never bridged. A 'matched saccade' is a gap-free transition with exactly one saccade event, so amplitude and duration belong to the same movement.
- Amplitude is the Tobii fixation-to-fixation angle (verified r = 0.99 against geometry). Direction (left/right) comes from fixation positions on screen; horizontal-dominant means |dx| > |dy|.
- Pupil in RAW157 exists only at event boundary samples, not as a continuous trace; lighting was not controlled.
- ROI64 features (D007, D010, D011, D015) summarise per-ROI workbook values over the targets of a task, requiring valid values for at least 60% of expected targets (couplings: at least 10 targets). D010 and D011 follow the export's Table II pupil convention (the 'exit' value is the landing fixation after leaving the target).
- Full support rules, zero/missing rules and evidence: `feature_registry.csv`, `DATASET_PROPOSED_100_FEATURES.md`, `semantic_audit.md`.

## 2. p-values: 4 tasks x 3 comparisons

**Bold** = p < 0.05; *italic* = 0.05 <= p < 0.10; NA = not testable (fewer than 8 valid participants or below 60% coverage in a group). Raw p values; see section 3 for group sizes.

| ID | Feature | KD1 CN vs MCI | KD1 CN vs AD | KD1 MCI vs AD | KD2 CN vs MCI | KD2 CN vs AD | KD2 MCI vs AD | KD3 CN vs MCI | KD3 CN vs AD | KD3 MCI vs AD | READING CN vs MCI | READING CN vs AD | READING MCI vs AD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A002 | total_fixation_time | *0.090* | 0.376 | 0.811 | **0.002** | *0.069* | 0.866 | **8.7e-04** | **9.5e-05** | *0.077* | 0.908 | 0.412 | 0.483 |
| A003 | task_fixation_count | **0.010** | *0.061* | 0.817 | **4.5e-05** | **0.027** | 0.972 | **0.002** | **1.8e-04** | **0.033** | 0.661 | 0.261 | 0.419 |
| A021 | task_saccade_count | 0.107 | 0.182 | 0.852 | **0.015** | *0.056* | 0.824 | *0.058* | **0.004** | 0.155 | **0.023** | 0.300 | 0.534 |
| A023 | amplitude_median | **0.019** | 0.262 | 0.600 | **0.022** | 0.124 | 0.986 | 0.101 | **0.046** | 0.360 | 0.116 | *0.074* | 0.763 |
| A024 | amplitude_iqr | 0.497 | 0.347 | 0.880 | 0.459 | 0.813 | 0.894 | 0.583 | **0.008** | **0.003** | 0.280 | 0.870 | 0.644 |
| A026 | saccade_duration_median | 0.557 | 0.374 | 0.669 | 0.797 | 0.445 | 0.443 | 0.279 | *0.092* | **0.048** | 0.857 | 0.913 | 0.828 |
| A027 | saccade_duration_iqr | 0.201 | 0.506 | 0.769 | *0.098* | 0.319 | *0.061* | **0.027** | **0.040** | **0.003** | 0.146 | 0.173 | 0.599 |
| A028 | saccade_mean_speed_median | **0.009** | 0.511 | 0.245 | **0.002** | 0.611 | 0.324 | **0.045** | 0.216 | 0.811 | *0.072* | 0.147 | 0.922 |
| A029 | task_tiny_saccade_fraction | **0.003** | 0.145 | 0.683 | 0.181 | 0.792 | 0.564 | 0.780 | 0.816 | 0.670 | 0.260 | 0.200 | 0.638 |
| A030 | task_long_saccade_fraction | **0.014** | 0.102 | 0.922 | *0.100* | 0.246 | 0.965 | 0.290 | 0.740 | 0.494 | 0.108 | *0.099* | 0.887 |
| A031 | angular_scanpath_length | 0.829 | 0.484 | 0.450 | **0.009** | 0.257 | 0.742 | 0.532 | **0.038** | 0.158 | *0.096* | 0.782 | 0.419 |
| A033 | normalized_screen_dispersion | 0.524 | 0.459 | 0.908 | 0.372 | 0.335 | 0.252 | 0.955 | 0.908 | 0.950 | **0.010** | 0.202 | 0.517 |
| A035 | task_spatial_entropy | 0.819 | 0.583 | 0.663 | 0.200 | 0.112 | *0.093* | 0.356 | 0.746 | 0.450 | **0.024** | 0.108 | 0.908 |
| A046 | task_absolute_pupil_median | *0.053* | 0.135 | 0.763 | *0.055* | 0.101 | 0.600 | **0.033** | *0.084* | 0.689 | **0.028** | **0.036** | 0.467 |
| D017 | fixation_duration_serial_correlation | **0.018** | 0.272 | 0.839 | **0.042** | 0.671 | 0.409 | *0.078* | 0.372 | 0.663 | *0.085* | 0.843 | 0.333 |
| D018 | fixation_duration_quartile_dispersion | 0.259 | 0.844 | 0.419 | 0.840 | 0.407 | 0.316 | 0.327 | 0.254 | 0.908 | **0.041** | 0.331 | 0.880 |
| D021 | interfixation_interval_median | 0.637 | **0.026** | **0.019** | 0.350 | 0.183 | *0.080* | 0.453 | **0.026** | **0.022** | 0.826 | 0.464 | 0.521 |
| D022 | fixation_onset_burstiness | 0.346 | 0.841 | 0.395 | 0.186 | 0.381 | 0.124 | **0.005** | 0.158 | 0.676 | 0.213 | 0.318 | 0.651 |
| D023 | gaze_event_span | *0.060* | **0.020** | 0.316 | **8.2e-04** | **0.002** | 0.419 | **2.5e-04** | **1.4e-04** | **0.040** | **0.016** | **0.008** | 0.245 |
| D024 | saccadic_time_fraction | 0.858 | *0.059* | *0.080* | 0.529 | 0.153 | 0.138 | 0.613 | **0.016** | *0.090* | 0.532 | 0.231 | 0.129 |
| D025 | main_sequence_duration_slope | **6.6e-04** | 0.119 | 0.494 | **0.008** | 0.600 | *0.069* | **0.005** | 0.484 | 0.307 | **0.003** | 0.335 | 0.252 |
| D026 | amplitude_duration_coupling | **0.046** | 0.199 | 0.941 | 0.489 | 0.241 | 0.540 | 0.390 | 0.204 | 0.540 | 0.275 | 0.562 | *0.090* |
| D028 | leftward_transition_fraction | 0.876 | 0.639 | 0.638 | **0.047** | 0.540 | 0.163 | 0.915 | 0.313 | 0.389 | 0.584 | 0.133 | *0.060* |
| D029 | rightward_saccade_amplitude_median | *0.075* | 0.128 | 0.894 | 0.505 | 0.645 | 0.929 | **0.039** | **0.002** | 0.169 | 0.127 | 0.329 | 0.908 |
| D030 | leftward_saccade_amplitude_median | 0.364 | 0.149 | 0.698 | 0.159 | 0.255 | 0.724 | *0.059* | **0.019** | 0.347 | 0.693 | 0.904 | 0.670 |
| D034 | saccade_amplitude_serial_correlation | 0.644 | 0.300 | 0.177 | 0.455 | 0.416 | 0.741 | 0.560 | 0.427 | 0.167 | **0.003** | 0.102 | 0.613 |
| D035 | fixation_duration_outgoing_amplitude_coupling | *0.093* | 0.712 | 0.229 | 0.690 | 0.660 | 0.894 | 0.597 | **0.028** | 0.108 | 0.365 | 0.756 | 0.440 |
| D036 | incoming_amplitude_fixation_duration_coupling | **0.037** | 0.603 | 0.579 | 0.125 | 0.881 | 0.389 | **1.2e-04** | **0.008** | 0.729 | 0.819 | 0.875 | 0.894 |
| D037 | post_leftward_fixation_duration_log_ratio | 0.987 | 0.616 | 0.877 | 0.506 | 0.774 | 0.644 | 0.631 | 0.363 | 0.257 | 0.530 | **0.022** | 0.133 |
| D039 | fixation_xy_spread_log_anisotropy | 0.515 | 0.180 | 0.588 | 0.937 | 0.251 | 0.360 | 0.911 | *0.098* | 0.169 | **0.006** | 0.199 | 0.351 |
| D040 | vertical_progression_monotonicity | 0.739 | 0.430 | 0.186 | *0.063* | **0.004** | 0.259 | 0.246 | 0.244 | 0.703 | 0.119 | 0.421 | 0.564 |
| D041 | off_media_fixation_time_fraction | 0.280 | 0.424 | 0.167 | 0.277 | 0.756 | 0.398 | **0.017** | 0.365 | 0.493 | 0.178 | 0.343 | 0.943 |
| D047 | viewing_distance_iqr | 0.998 | 0.153 | 0.153 | 0.401 | **2.7e-04** | **0.003** | 0.521 | **0.001** | **7.2e-04** | 0.214 | **0.003** | *0.087* |
| D007 | target_mean_fixation_count | **0.005** | **0.015** | 0.806 | **0.011** | *0.062* | 0.566 | 0.520 | **0.003** | **0.017** | NA | NA | 0.277 |
| D010 | target_pupil_change_median | 0.462 | *0.055* | 0.260 | 0.100 | 0.403 | **0.049** | **0.037** | **0.021** | 1.000 | NA | NA | NA |
| D011 | target_left_pupil_range_median | *0.062* | **0.020** | 0.509 | 0.323 | 0.101 | 0.409 | 0.969 | **0.011** | **0.026** | NA | NA | *0.073* |
| D015 | articulation_fixation_count_coupling | 0.958 | **0.035** | 0.104 | 0.387 | 0.303 | 0.859 | 0.494 | **0.046** | 0.199 | 0.636 | 0.262 | 0.442 |

## 3. Valid participants per task (CN/MCI/AD)

| ID | KD1 | KD2 | KD3 | READING |
|---|---|---|---|---|
| A002 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A003 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A021 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A023 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A024 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A026 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A027 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A028 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A029 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A030 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A031 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A033 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A035 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A046 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D017 | 99/41/15 | 99/41/16 | 100/41/16 | 99/41/16 |
| D018 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D021 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D022 | 99/41/15 | 99/41/16 | 100/41/16 | 99/41/16 |
| D023 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D024 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D025 | 100/41/15 | 100/41/16 | 100/41/16 | 100/41/16 |
| D026 | 100/41/15 | 100/41/16 | 100/41/16 | 100/41/16 |
| D028 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D029 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D030 | 98/41/15 | 99/40/16 | 99/41/16 | 100/41/16 |
| D034 | 98/40/15 | 99/40/15 | 97/39/16 | 99/41/16 |
| D035 | 99/41/15 | 99/41/16 | 100/41/16 | 99/41/16 |
| D036 | 99/41/15 | 99/41/16 | 100/41/16 | 99/41/16 |
| D037 | 99/38/14 | 99/40/16 | 96/40/16 | 99/41/16 |
| D039 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D040 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D041 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D047 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D007 | 31/17/15 | 30/16/15 | 29/15/13 | 19/12/10 |
| D010 | 31/17/13 | 29/16/13 | 28/15/13 | 18/12/7 |
| D011 | 31/17/15 | 29/16/13 | 29/15/13 | 19/12/9 |
| D015 | 32/17/15 | 31/16/15 | 31/16/15 | 30/16/14 |

## 4. Caveats

- p values are uncorrected. Within the pre-registered families, only D023@KD3, A002@KD3, A003@KD3 and D023@KD2 reach joint BH q <= 0.10 (none survive BY).
- D023, A002, A003 and A021 are strongly correlated (longer reading of the card = more fixations and saccades); they are not independent confirmations.
- The selection thresholds were chosen after inspecting results; any predictive use must repeat the selection inside training data or use an independent cohort.
- AD groups are small (15-16), and no demographic adjustment was possible (no age/sex/education data).
