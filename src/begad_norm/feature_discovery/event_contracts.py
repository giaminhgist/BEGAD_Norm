"""Reconstruct fixation/saccade events and gap-free transitions from raw Tobii exports.

Verified export semantics (see semantic_audit.md):
- Each event is exported as an onset row (Final=False) and a last-sample row
  (Final=True). Fixation offset = onset RecordingTimestamp + GazeEventDuration (ms).
- SaccadicAmplitude/AbsoluteSaccadicDirection on a fixation row describe the
  fixation-to-fixation movement that landed on it (degrees).
- Pupil and per-eye gaze exist only on these event rows (no continuous trace).
- FixationPoint (MCSpx) is NaN when the fixation lies outside the media box.

Nothing here bridges a gap: a transition is "gap-free" only when I-VT saccade
events cover the interval between two consecutive same-task fixations up to
GAP_TOL_MS of uncovered time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DECIMAL_COMMA_COLS = (
    "SaccadicAmplitude", "AbsoluteSaccadicDirection", "RelativeSaccadicDirection",
    "PupilLeft", "PupilRight", "GazePointLeftX (ADCSmm)", "GazePointRightX (ADCSmm)",
    "GazePointLeftY (ADCSmm)", "GazePointRightY (ADCSmm)", "DistanceLeft", "DistanceRight",
    "StrictAverageGazePointX (ADCSmm)", "StrictAverageGazePointY (ADCSmm)",
)
MM_PER_PX = 0.2676          # verified constant (ADCSmm vs ADCSpx regression in every inspected file)
SCREEN_W, SCREEN_H = 1920.0, 1080.0
GAP_TOL_MS = 10.0           # max uncovered time inside a gap-free transition (3 samples at 300 Hz)
ONSET_TOL_MS = 5.0          # saccade onset may precede the computed fixation offset by < 2 samples
OVERLAP_TOL_MS = 4.0

RAW_TASK_TO_CANONICAL = {3: "KD1", 4: "KD2", 5: "KD3", 9: "READING"}


def parse_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in DECIMAL_COMMA_COLS:
        if c in out.columns and out[c].dtype == object:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False)
                                   .replace({"nan": np.nan, "": np.nan}), errors="coerce")
        elif c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def read_raw(path) -> pd.DataFrame:
    return parse_decimal_columns(pd.read_csv(path, low_memory=False))


def _eye_valid(df):
    return (df["ValidityLeft"] == 0) & (df["ValidityRight"] == 0)


def build_fixations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fixation event with onset/offset (ms), position and endpoint samples."""
    fx = df[df["GazeEventType"] == "Fixation"].copy()
    fx = fx[fx["FixationIndex"].notna()]
    both = _eye_valid(fx)
    fx["pup_bin"] = np.where(both, (fx["PupilLeft"] + fx["PupilRight"]) / 2.0, np.nan)
    if "GazePointLeftX (ADCSmm)" in fx:
        fx["disp_mm"] = np.where(both, fx["GazePointLeftX (ADCSmm)"] - fx["GazePointRightX (ADCSmm)"], np.nan)
        fx["dist_mm"] = np.where(both, (fx["DistanceLeft"] + fx["DistanceRight"]) / 2.0, np.nan)
    else:
        fx["disp_mm"] = np.nan
        fx["dist_mm"] = np.nan
    on = fx[~fx["Final"].astype(bool)].drop_duplicates("FixationIndex", keep="first").set_index("FixationIndex")
    fin = fx[fx["Final"].astype(bool)].drop_duplicates("FixationIndex", keep="last").set_index("FixationIndex")
    idx = on.index.union(fin.index)
    out = pd.DataFrame(index=idx)
    out["onset_ms"] = on["RecordingTimestamp"]
    out["final_ms"] = fin["RecordingTimestamp"]
    out["dur_ms"] = fin["GazeEventDuration"].combine_first(on["GazeEventDuration"])
    out["task_on"] = on["Task"]
    out["task_fin"] = fin["Task"]
    out["mcs_x"] = fin["FixationPointX (MCSpx)"].combine_first(on["FixationPointX (MCSpx)"])
    out["mcs_y"] = fin["FixationPointY (MCSpx)"].combine_first(on["FixationPointY (MCSpx)"])
    out["amp_in"] = fin["SaccadicAmplitude"].combine_first(on["SaccadicAmplitude"])
    out["pup_on"] = on["pup_bin"]
    out["pup_fin"] = fin["pup_bin"]
    out["disp_on"] = on["disp_mm"]
    out["disp_fin"] = fin["disp_mm"]
    out["dist_on"] = on["dist_mm"]
    out["dist_fin"] = fin["dist_mm"]
    gx = fx.groupby("FixationIndex")["GazePointX (ADCSpx)"].mean()
    gy = fx.groupby("FixationIndex")["GazePointY (ADCSpx)"].mean()
    out["gaze_x"] = gx
    out["gaze_y"] = gy
    for c in ("MediaPosX (ADCSpx)", "MediaPosY (ADCSpx)", "MediaWidth", "MediaHeight"):
        out[c] = fin[c].combine_first(on[c]) if c in fin else np.nan
    out["has_both_rows"] = out["onset_ms"].notna() & out["final_ms"].notna()
    out["straddles_task"] = out["has_both_rows"] & (out["task_on"] != out["task_fin"])
    out["valid"] = out["has_both_rows"] & ~out["straddles_task"] & (out["dur_ms"] > 0)
    out["task"] = out["task_on"].where(out["valid"])
    out["offset_ms"] = out["onset_ms"] + out["dur_ms"]
    out["screen_x"] = out["mcs_x"] + out["MediaPosX (ADCSpx)"]
    out["screen_y"] = out["mcs_y"] + out["MediaPosY (ADCSpx)"]
    out["in_media"] = out["mcs_x"].notna() & out["mcs_y"].notna()
    # located: either in-media position, or gaze samples outside the media box
    out["located"] = out["in_media"] | (out["gaze_x"].notna() & out["gaze_y"].notna())
    out = out.reset_index().rename(columns={"index": "fixation_id", "FixationIndex": "fixation_id"})
    return out.sort_values("onset_ms").reset_index(drop=True)


def build_saccades(df: pd.DataFrame) -> pd.DataFrame:
    sc = df[(df["GazeEventType"] == "Saccade") & df["SaccadeIndex"].notna()]
    g = sc.groupby("SaccadeIndex")
    out = pd.DataFrame({
        "onset_ms": g["RecordingTimestamp"].min(),
        "dur_ms": g["GazeEventDuration"].first(),
        "task": g["Task"].first(),
        "n_rows": g.size(),
    }).reset_index().rename(columns={"SaccadeIndex": "saccade_id"})
    out["offset_ms"] = out["onset_ms"] + out["dur_ms"]
    return out.sort_values("onset_ms").reset_index(drop=True)


def _covered(intervals, a, b):
    """Total length of the union of intervals clipped to [a, b]."""
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        s, e = max(s, a), min(e, b)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def build_transitions(fix: pd.DataFrame, sac: pd.DataFrame) -> pd.DataFrame:
    """Consecutive valid same-task fixation pairs with gap/saccade matching."""
    rows = []
    sac_on = sac["onset_ms"].to_numpy(float)
    sac_off = sac["offset_ms"].to_numpy(float)
    sac_dur = sac["dur_ms"].to_numpy(float)
    order = np.argsort(sac_on)
    sac_on, sac_off, sac_dur = sac_on[order], sac_off[order], sac_dur[order]
    for task, f in fix[fix["valid"]].groupby("task", sort=False):
        f = f.sort_values("onset_ms")
        on = f["onset_ms"].to_numpy(float)
        off = f["offset_ms"].to_numpy(float)
        ids = f["fixation_id"].to_numpy()
        for k in range(len(f) - 1):
            a, b = off[k], on[k + 1]
            interval = b - a
            lo = np.searchsorted(sac_on, a - ONSET_TOL_MS, side="left")
            hi = np.searchsorted(sac_on, b, side="left")
            n_sac = int(hi - lo)
            if interval < -OVERLAP_TOL_MS:
                status, uncovered = "overlap_anomaly", np.nan
            else:
                ivs = list(zip(sac_on[lo:hi], sac_off[lo:hi]))
                uncovered = max(interval, 0.0) - _covered(ivs, a, b) if interval > 0 else 0.0
                status = "gap_free" if uncovered <= GAP_TOL_MS else "gap"
            matched = status == "gap_free" and n_sac == 1
            rows.append({
                "task": task, "from_id": ids[k], "to_id": ids[k + 1], "k": k,
                "interval_ms": interval, "uncovered_ms": uncovered, "n_saccades": n_sac,
                "status": status, "matched": matched,
                "sacc_dur_ms": float(sac_dur[lo]) if matched else np.nan,
            })
    tr = pd.DataFrame(rows, columns=["task", "from_id", "to_id", "k", "interval_ms", "uncovered_ms",
                                     "n_saccades", "status", "matched", "sacc_dur_ms"])
    if tr.empty:
        return tr
    fx = fix.set_index("fixation_id")
    tr["amp_deg"] = np.where(tr["status"] == "gap_free", fx.loc[tr["to_id"], "amp_in"].to_numpy(), np.nan)
    for side, col in (("from", "from_id"), ("to", "to_id")):
        tr[f"{side}_x"] = fx.loc[tr[col], "screen_x"].to_numpy()
        tr[f"{side}_y"] = fx.loc[tr[col], "screen_y"].to_numpy()
        tr[f"{side}_dur_ms"] = fx.loc[tr[col], "dur_ms"].to_numpy()
    tr["dx_px"] = tr["to_x"] - tr["from_x"]
    tr["dy_px"] = tr["to_y"] - tr["from_y"]
    return tr


def px_to_deg(dist_px, eye_dist_mm):
    d_mm = np.asarray(dist_px, float) * MM_PER_PX
    return np.degrees(2.0 * np.arctan(d_mm / (2.0 * np.asarray(eye_dist_mm, float))))


def build_participant_events(path):
    df = read_raw(path)
    fix = build_fixations(df)
    sac = build_saccades(df)
    tr = build_transitions(fix, sac)
    return df, fix, sac, tr
