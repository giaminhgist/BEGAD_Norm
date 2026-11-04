"""Metadata for the 33-feature figure generation."""

from __future__ import annotations

# Feature display names and units, in F01..F33 order (from feature_selection.md).
FEATURES = {
    "F01": ("total_fixation_time", "s"),
    "F02": ("task_fixation_count", "count"),
    "F03": ("task_saccade_count", "count"),
    "F04": ("amplitude_median", "deg"),
    "F05": ("amplitude_iqr", "deg"),
    "F06": ("saccade_duration_median", "s"),
    "F07": ("saccade_duration_iqr", "s"),
    "F08": ("saccade_mean_speed_median", "deg/s"),
    "F09": ("task_tiny_saccade_fraction", "fraction"),
    "F10": ("task_long_saccade_fraction", "fraction"),
    "F11": ("angular_scanpath_length", "deg"),
    "F12": ("normalized_screen_dispersion", "norm. distance"),
    "F13": ("task_spatial_entropy", "unitless"),
    "F14": ("task_absolute_pupil_median", "pupil unit"),
    "F15": ("fixation_duration_serial_correlation", "correlation"),
    "F16": ("fixation_duration_quartile_dispersion", "unitless"),
    "F17": ("interfixation_interval_median", "s"),
    "F18": ("fixation_onset_burstiness", "unitless"),
    "F19": ("gaze_event_span", "s"),
    "F20": ("saccadic_time_fraction", "fraction"),
    "F21": ("main_sequence_duration_slope", "s/deg"),
    "F22": ("amplitude_duration_coupling", "correlation"),
    "F23": ("leftward_transition_fraction", "fraction"),
    "F24": ("rightward_saccade_amplitude_median", "deg"),
    "F25": ("leftward_saccade_amplitude_median", "deg"),
    "F26": ("saccade_amplitude_serial_correlation", "correlation"),
    "F27": ("fixation_duration_outgoing_amplitude_coupling", "correlation"),
    "F28": ("incoming_amplitude_fixation_duration_coupling", "correlation"),
    "F29": ("post_leftward_fixation_duration_log_ratio", "log ratio"),
    "F30": ("fixation_xy_spread_log_anisotropy", "log ratio"),
    "F31": ("vertical_progression_monotonicity", "correlation"),
    "F32": ("off_media_fixation_time_fraction", "fraction"),
    "F33": ("viewing_distance_iqr", "mm"),
}

FEATURE_IDS = [f"F{i:02d}" for i in range(1, 34)]

TASKS = ["KD1", "KD2", "KD3", "READING"]

# Group order used consistently across all figures and the stats CSV.
GROUPS = ["CN", "MCI", "AD"]

# The three pairwise comparisons.
PAIRS = [("CN", "MCI"), ("CN", "AD"), ("MCI", "AD")]

# Okabe-Ito-inspired swarm colors (distinct from the light box tints).
GROUP_COLOR = {"CN": "#0072B2", "MCI": "#D55E00", "AD": "#CC79A7"}
BOX_FACECOLOR = {"CN": "#b8d4ee", "MCI": "#f2c9a8", "AD": "#eebfd9"}

SEED = 20261027

# Bracket levels as fractions of the figure y-span above ymax, per pair index.
BRACKET_LEVELS = [0.10, 0.20, 0.30]

OUTPUT_DIR = "output/figures"