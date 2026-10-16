"""Measured-saccade feature group (sac_, 13 features) for one (participant, task).

Sources: (a) per-fixation attached saccade properties on clean fixations
(SaccadicAmplitude / AbsoluteSaccadicDirection / RelativeSaccadicDirection,
degrees; the saccade landing on each fixation), and (b) saccade event rows
collapsed into runs by rawdata.extract_saccade_runs (run_dur = max duration
of the run, ms). total_time follows the EMS definition:
total_time = dur.sum() + ifi_mean * (n - 1), with the half-dwell ifi_mean.
"""

import numpy as np


def _circular_mean_deg(angles_deg):
    theta = np.deg2rad(angles_deg)
    mean_deg = float(np.degrees(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())))
    mean_deg %= 360.0
    # float noise can land just below 360 for a true 0 mean (e.g. [350, 10])
    return 0.0 if mean_deg >= 360.0 - 1e-12 else mean_deg


def _circular_var(angles_deg):
    theta = np.deg2rad(angles_deg)
    R = np.sqrt(np.cos(theta).mean() ** 2 + np.sin(theta).mean() ** 2)
    return float(1.0 - R)


def compute_saccade_features(amp, absdir, reldir, run_durs, total_time):
    """Compute the 13 measured-saccade features.

    amp, absdir, reldir: 1-D arrays over clean fixations (n = len(amp)).
    run_durs: 1-D array of saccade-run durations (m = len(run_durs)).
    total_time: task total time in ms (EMS formula, see module docstring).
    Returns dict feature_name -> float (NaN where undefined).
    """
    amp = np.asarray(amp, dtype=np.float64)
    absdir = np.asarray(absdir, dtype=np.float64)
    reldir = np.asarray(reldir, dtype=np.float64)
    run_durs = np.asarray(run_durs, dtype=np.float64)
    n, m = len(amp), len(run_durs)

    f = {}
    f["sac_count"] = float(m)
    f["sac_rate"] = float(m / total_time * 1000.0) if total_time > 0 else np.nan
    f["sac_dur_mean"] = float(run_durs.mean()) if m > 0 else np.nan
    f["sac_dur_std"] = float(run_durs.std(ddof=1)) if m > 1 else np.nan
    f["sac_dur_total"] = float(run_durs.sum()) if m > 0 else np.nan
    f["sac_dur_max"] = float(run_durs.max()) if m > 0 else np.nan

    f["sac_amp_mean"] = float(amp.mean()) if n > 0 else np.nan
    f["sac_amp_std"] = float(amp.std(ddof=1)) if n > 1 else np.nan
    f["sac_amp_max"] = float(amp.max()) if n > 0 else np.nan

    f["sac_absdir_mean"] = _circular_mean_deg(absdir) if n > 0 else np.nan
    f["sac_absdir_var"] = _circular_var(absdir) if n > 1 else np.nan

    turn = np.minimum(reldir, 360.0 - reldir)  # fold 0.22-359.97 to [0, 180]
    f["sac_turn_mean"] = float(turn.mean()) if n > 0 else np.nan
    f["sac_turn_std"] = float(turn.std(ddof=1)) if n > 1 else np.nan
    return f
