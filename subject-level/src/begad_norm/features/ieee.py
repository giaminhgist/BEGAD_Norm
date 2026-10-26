"""IEEE-paper feature set (Table II of the multimodal AD/MCI fusion paper)
mapped to BEGAD, per (participant, task).

The paper's 20 features: 8 visual, 9 pupillary, 3 speech. On BEGAD we can
compute 16 of them:

- visual (7 of 8): number of fixations, number of saccades, short saccades
  (< 1 deg), long saccades (> 2 deg), regressive saccades, saccade amplitude
  sum, total fixation time. ROI (feature 1) is not available: BEGAD has no
  word/number AOI annotations, the paper's ROI definition.
- pupillary (9 of 9): left/right pupil at the first and last fixation of the
  task, their differences, min/max left pupil, min-max difference. The paper
  subtracts the participant's pupil baseline from the pupillary metrics;
  BEGAD's PupilLeftBaseline / PupilRightBaseline (one value per participant)
  are used the same way. Differences are baseline-invariant.
- speech (0 of 3): BEGAD has no speech/audio data (total spoken time,
  first-fixation-to-utterance-onset, last-fixation-to-utterance-end are all
  unavailable).

Conventions follow the project pipeline (rawdata.clean_fixations): saccade
amplitude and direction are attached to the clean fixation they land on,
amplitude in degrees, regressive = turn angle min(reldir, 360 - reldir) > 90.
Pairs with < MIN_FIX clean fixations yield all-NaN rows (EMS convention),
except vis_n_fix which keeps the real count.
"""

import numpy as np

from begad_norm.config import MIN_FIX

SAC_SHORT_DEG = 1.0
SAC_LONG_DEG = 2.0
REG_TURN_DEG = 90.0

IEEE_FEATURE_NAMES = [
    # visual (ROI of the paper is not available on BEGAD)
    "vis_n_fix", "vis_n_sac", "vis_n_sac_short", "vis_n_sac_long",
    "vis_n_sac_reg", "vis_sac_amp_sum", "vis_fix_time_total",
    # pupillary (baseline-corrected, per eye)
    "pup_l_first", "pup_l_last", "pup_l_diff",
    "pup_r_first", "pup_r_last", "pup_r_diff",
    "pup_l_min", "pup_l_max", "pup_l_minmax_diff",
]


def compute_ieee_features(g, baseline_l: float, baseline_r: float) -> dict:
    """The 16 IEEE-mapped features for one (participant, task) pair.

    g: clean fixations of the pair, ts-sorted (rawdata.clean_fixations
       columns: dur, pup_left, pup_right, amp, reldir).
    baseline_l/r: the participant's PupilLeftBaseline / PupilRightBaseline
       (NaN when the file lacks a baseline for that eye; the absolute pupil
       features of that eye then become NaN, matching the pipeline's
       treatment of pupbl features, while the differences stay computable).
    """
    n = len(g)
    if n < MIN_FIX:
        out = {name: np.nan for name in IEEE_FEATURE_NAMES}
        out["vis_n_fix"] = float(n)
        return out

    amp_valid = g["amp"].notna()
    amp = g.loc[amp_valid, "amp"].to_numpy(dtype=np.float64)
    turn = np.minimum(g["reldir"].to_numpy(dtype=np.float64),
                      360.0 - g["reldir"].to_numpy(dtype=np.float64))

    out = {
        # visual
        "vis_n_fix": float(n),
        "vis_n_sac": float(len(amp)),
        "vis_n_sac_short": float(np.sum(amp < SAC_SHORT_DEG)),
        "vis_n_sac_long": float(np.sum(amp > SAC_LONG_DEG)),
        "vis_n_sac_reg": float(np.sum(turn[amp_valid] > REG_TURN_DEG)),
        "vis_sac_amp_sum": float(amp.sum()),
        "vis_fix_time_total": float(g["dur"].sum()),
    }
    # pupillary: differences are baseline-invariant; absolute values need the
    # baseline and stay NaN when it is missing
    for eye, base in (("l", baseline_l), ("r", baseline_r)):
        pup = g[{"l": "pup_left", "r": "pup_right"}[eye]].to_numpy(
            dtype=np.float64)
        if np.isnan(base):
            for name in (f"pup_{eye}_first", f"pup_{eye}_last",
                         f"pup_{eye}_min", f"pup_{eye}_max"):
                out[name] = np.nan
        else:
            pup = pup - base
            out[f"pup_{eye}_first"] = float(pup[0])
            out[f"pup_{eye}_last"] = float(pup[-1])
            out[f"pup_{eye}_min"] = float(pup.min())
            out[f"pup_{eye}_max"] = float(pup.max())
        out[f"pup_{eye}_diff"] = float(pup[-1] - pup[0])
    out["pup_l_minmax_diff"] = out["pup_l_max"] - out["pup_l_min"] \
        if not (np.isnan(out["pup_l_max"]) or np.isnan(out["pup_l_min"])) \
        else np.nan
    return out
