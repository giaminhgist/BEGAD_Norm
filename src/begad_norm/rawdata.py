"""Loading, parsing and cleaning of BEGAD dataset_v2 participant CSVs.

Cleaning order per file (protocol notebook Analysisbase_for_Ali.ipynb + EMS):
1. keep only Final & !HasNulls fixation events (FixationIndex > 0),
2. EMS duration window (DUR_MIN, DUR_MAX],
3. per-task media bounds (inclusive, MCSpx is media-relative),
4. per-(participant, task) pupil outliers on the binocular mean
   (± PUPIL_SD * max(std, PUPIL_EPS) of the group median, EMS-exact).
The cohort filter (Group in {0,1,2}, IsFollowup==False) is enforced by the
roster-driven iteration in scripts/build_features.py.
"""

import numpy as np
import pandas as pd

from begad_norm.config import (
    DECIMAL_COMMA_COLS,
    DUR_MAX,
    DUR_MIN,
    PUPIL_EPS,
    PUPIL_SD,
    TASK_GEOMETRY,
)

CLEAN_COLUMNS = ["task", "ts", "dur", "x", "y", "pup", "pup_corrected", "pupdiff",
                 "amp", "absdir", "reldir"]


def participant_id_from_filename(name: str) -> str:
    """'BEGAD_ETRA25_BEG-201.csv' -> 'BEG-201' (RealTag)."""
    return name.removeprefix("BEGAD_ETRA25_").removesuffix(".csv")


def read_participant_file(path) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse European decimal-comma columns ('3,98' -> 3.98) into float64.

    Applied manually (never read_csv(decimal=',')): that option corrupts this
    mixed-format file. Dot-formatted values and missing cells survive verbatim.
    """
    out = df.copy()
    for col in DECIMAL_COMMA_COLS:
        if col not in out.columns:
            continue
        s = out[col].astype("str").str.replace(",", ".")
        out[col] = s.replace("", np.nan).astype(np.float64)
    return out


def clean_fixations(df: pd.DataFrame, return_stats: bool = False):
    """Return cleaned fixations with columns CLEAN_COLUMNS, sorted by ts.

    df must already have decimal-comma columns parsed (parse_decimal_columns).
    With return_stats=True, returns (out, stats) where stats counts the rows
    dropped by each stage for the quality report.
    """
    before = len(df)
    df = df[
        (df["Final"] == True)
        & (~df["HasNulls"])
        & (df["GazeEventType"] == "Fixation")
        & (df["FixationIndex"] > 0)
    ].copy()
    after_fixation = len(df)

    # duration window (EMS: (DUR_MIN, DUR_MAX])
    df = df[(df["GazeEventDuration"] > DUR_MIN) & (df["GazeEventDuration"] <= DUR_MAX)]
    after_duration = len(df)

    # per-task media bounds on MCSpx (media-relative, inclusive)
    df["media_w"] = df["Task"].map(lambda t: TASK_GEOMETRY.get(t, (np.nan,) * 4)[2])
    df["media_h"] = df["Task"].map(lambda t: TASK_GEOMETRY.get(t, (np.nan,) * 4)[3])
    df = df[df["media_w"].notna()]
    x = df["FixationPointX (MCSpx)"]
    y = df["FixationPointY (MCSpx)"]
    df = df[(x >= 0) & (x <= df["media_w"]) & (y >= 0) & (y <= df["media_h"])]
    after_bounds = len(df)

    # per-(participant, task) pupil outliers on the binocular mean (EMS-exact:
    # transform median / transform std with fillna(0.0), ddof=1 as pandas default)
    df["pup"] = (df["PupilLeft"] + df["PupilRight"]) / 2.0
    med = df.groupby("Task")["pup"].transform("median")
    sd = df.groupby("Task")["pup"].transform("std").fillna(0.0)
    df = df[np.abs(df["pup"] - med) <= PUPIL_SD * np.maximum(sd, PUPIL_EPS)]

    df["pup_corrected"] = (
        (df["PupilLeft"] - df["PupilLeftBaseline"])
        + (df["PupilRight"] - df["PupilRightBaseline"])
    ) / 2.0
    df["pupdiff"] = df["PupilLeft"] - df["PupilRight"]

    out = df.rename(
        columns={
            "Task": "task",
            "RecordingTimestamp": "ts",
            "GazeEventDuration": "dur",
            "FixationPointX (MCSpx)": "x",
            "FixationPointY (MCSpx)": "y",
            "SaccadicAmplitude": "amp",
            "AbsoluteSaccadicDirection": "absdir",
            "RelativeSaccadicDirection": "reldir",
        }
    )
    out = out[CLEAN_COLUMNS].sort_values(["task", "ts"]).reset_index(drop=True)
    if return_stats:
        stats = {
            "raw": before,
            "fixation_filter_dropped": before - after_fixation,
            "duration_dropped": after_fixation - after_duration,
            "bounds_dropped": after_duration - after_bounds,
            "pupil_dropped": after_bounds - len(out),
            "clean": len(out),
        }
        return out, stats
    return out


def extract_saccade_runs(df: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Collapse Final saccade event rows into runs, one row per run.

    Saccade event rows always have HasNulls==True (their amp/direction columns
    are null by design), so the HasNulls filter must NOT be applied here.
    A run = consecutive (in time) saccade rows sharing the same landing
    fixation: the first clean fixation (same task) with ts > saccade ts.
    Rows without a landing fixation are dropped. run_dur = max duration.
    """
    sac = df[
        (df["Final"] == True)
        & (df["GazeEventType"] == "Saccade")
        & (df["SaccadeIndex"] > 0)
    ][["Task", "RecordingTimestamp", "GazeEventDuration"]]

    rows = []
    for task, g in sac.groupby("Task", sort=False):
        clean_ts = clean.loc[clean["task"] == task, "ts"].to_numpy()
        if len(clean_ts) == 0:
            continue
        g = g.sort_values("RecordingTimestamp")
        cur_run = []
        prev_landing = None
        for ts, dur in zip(g["RecordingTimestamp"], g["GazeEventDuration"]):
            landing = int(np.searchsorted(clean_ts, ts, side="right"))
            if landing >= len(clean_ts):
                continue
            if landing != prev_landing:
                if cur_run:
                    rows.append((task, max(cur_run)))
                cur_run = [float(dur)]
                prev_landing = landing
            else:
                cur_run.append(float(dur))
        if cur_run:
            rows.append((task, max(cur_run)))
    return pd.DataFrame(rows, columns=["task", "run_dur"])
