# Feature Selection — 33 Features

Definitions and formulas for the 33 selected task-level features.

**Conventions**
- Features are computed per participant per task, one value per task.
- Durations in seconds (s), amplitudes in degrees (deg), pupil in mm, distances in normalized screen units.
- "Valid within-task events" = events inside the task label that pass validity flags.
- "Gap-free transition" = two consecutive fixations of the same task whose interval is covered by saccade events up to 10 ms; gaps are never bridged.
- "Matched saccade" = a gap-free transition with exactly one saccade event, so amplitude and duration belong to the same movement.
- Fixation and saccade events are rebuilt from onset and final export rows: `offset = onset + duration`.
- Amplitude is the Tobii fixation-to-fixation angle. Direction (left/right) comes from fixation positions on screen; horizontal-dominant means `|dx| > |dy|`.

## Feature definitions

| # | Name | Meaning | Formula | Unit |
|---|---|---|---|---|
| 1 | total_fixation_time | Total time spent fixating during the task. | Sum of durations of unique valid within-task fixation events | s |
| 2 | task_fixation_count | Number of fixations in the task. | Number of unique valid within-task fixation events | count |
| 3 | task_saccade_count | Number of saccade events in the task. | Count of unique validated within-task saccade events | count |
| 4 | amplitude_median | Typical saccade size. | Median measured saccade amplitude | deg |
| 5 | amplitude_iqr | Spread of saccade sizes. | Q75 − Q25 of measured saccade amplitudes | deg |
| 6 | saccade_duration_median | Typical saccade duration. | Median actual saccade duration | s |
| 7 | saccade_duration_iqr | Spread of saccade durations. | Q75 − Q25 of actual saccade durations | s |
| 8 | saccade_mean_speed_median | Typical average saccade speed (not peak velocity). | Median of event amplitude divided by its actual matched duration | deg/s |
| 9 | task_tiny_saccade_fraction | Share of very small saccades (< 1 degree). | Saccades with amplitude < 1 degree ÷ all valid task saccades | fraction |
| 10 | task_long_saccade_fraction | Share of long saccades (> 2 degrees). | Saccades with amplitude > 2 degrees ÷ all valid task saccades | fraction |
| 11 | angular_scanpath_length | Total eye-movement path length in degrees. | Sum of unique valid within-task measured saccade amplitudes | deg |
| 12 | normalized_screen_dispersion | How widely fixations spread over the screen. | RMS distance from mean normalized screen fixation position | normalized distance |
| 13 | task_spatial_entropy | How evenly fixations cover an 8×6 screen grid. | −Σ(p_b · log p_b) / log(48) over a fixed 8×6 screen grid | unitless |
| 14 | task_absolute_pupil_median | Typical absolute pupil size (sensitive to lighting). | Median valid uncorrected binocular pupil | pupil unit |
| 15 | fixation_duration_serial_correlation | Whether consecutive fixation durations resemble each other (serial dependence). | Spearman ρ between durations of consecutive fixations joined by a gap-free transition | correlation |
| 16 | fixation_duration_quartile_dispersion | Relative (scale-free) spread of fixation durations. | (Q3 − Q1) / (Q3 + Q1) of fixation durations | unitless |
| 17 | interfixation_interval_median | Typical gap between one fixation ending and the next starting (movement time). | Median of onset(k+1) − offset(k) over gap-free transitions | s |
| 18 | fixation_onset_burstiness | Regular vs bursty timing of fixation onsets. | (sd − mean) / (sd + mean) of inter-onset intervals over gap-free transitions | unitless |
| 19 | gaze_event_span | Time from the first to the last fixation of the task (gaze-engaged duration). | Offset of last valid fixation − onset of first valid fixation within the task label | s |
| 20 | saccadic_time_fraction | Share of engaged time spent moving the eyes rather than fixating. | Sum of gap-free interfixation intervals ÷ (that sum + durations of the fixations they join), within continuous segments | fraction |
| 21 | main_sequence_duration_slope | Extra saccade duration per extra degree of amplitude (main-sequence slope). | Theil–Sen slope of matched saccade duration (s) on amplitude (deg) | s/deg |
| 22 | amplitude_duration_coupling | How tightly saccade duration follows amplitude. | Spearman ρ(amplitude, duration) over matched saccades | correlation |
| 23 | leftward_transition_fraction | Share of eye movements going leftward (regressions and line returns together). | Gap-free transitions with horizontal-dominant leftward displacement (dx < 0, |dx| > |dy|) ÷ gap-free transitions with displacement ≥ 1 deg | fraction |
| 24 | rightward_saccade_amplitude_median | Typical size of forward (rightward) saccades. | Median amplitude of matched saccades with horizontal-dominant rightward displacement | deg |
| 25 | leftward_saccade_amplitude_median | Typical size of backward (leftward) saccades. | Median amplitude of matched saccades with horizontal-dominant leftward displacement | deg |
| 26 | saccade_amplitude_serial_correlation | Whether consecutive saccade sizes resemble each other. | Spearman ρ between amplitudes of consecutive gap-free transitions | correlation |
| 27 | fixation_duration_outgoing_amplitude_coupling | Link between fixation duration and the size of the next saccade (planning). | Spearman ρ(fixation duration, amplitude of the following gap-free transition) | correlation |
| 28 | incoming_amplitude_fixation_duration_coupling | Link between the size of the incoming saccade and the following fixation duration. | Spearman ρ(amplitude of the preceding gap-free transition, fixation duration) | correlation |
| 29 | post_leftward_fixation_duration_log_ratio | Fixation duration after leftward vs rightward movements (cost of going back). | log(median duration of fixations entered by leftward transitions ÷ median entered by rightward) | log ratio |
| 30 | fixation_xy_spread_log_anisotropy | Horizontal vs vertical spread of fixations. | log(SD of screen x ÷ SD of screen y) over in-media fixation positions (px) | log ratio |
| 31 | vertical_progression_monotonicity | How consistently gaze moves from top to bottom over time. | Spearman ρ(fixation onset time, screen y) over in-media fixations | correlation |
| 32 | off_media_fixation_time_fraction | Share of fixation time on the screen but outside the stimulus image. | Duration of fixations whose gaze lies outside the stimulus media box ÷ duration of all located fixations | fraction |
| 33 | viewing_distance_iqr | Variability of eye-to-screen distance (head/posture stability). | IQR of mean(DistanceLeft, DistanceRight) over fixation samples with both eyes valid | mm |

## Notes

- Pupil exists only at event boundary samples, not as a continuous trace; lighting was not controlled.
- Serial-correlation and coupling features use Spearman ρ, computed only over gap-free / matched transitions; they require a minimum number of events to be defined.