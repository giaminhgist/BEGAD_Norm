"""End-to-end golden test: build_features on one real cohort participant (BEG-201)."""

import unittest

import numpy as np
import pandas as pd

from begad_norm.config import (
    DATA_DIR,
    DUR_MAX,
    DUR_MIN,
    PUPIL_EPS,
    PUPIL_SD,
    TASK_GEOMETRY,
    TASK_IDS,
)
from begad_norm.scripts.build_features import SELECTED_FEATURE_NAMES, build_all

FEATURE_COLS = ["participant_id", "task"] + SELECTED_FEATURE_NAMES


def reference_clean_fixations():
    """Independent recomputation of clean fixations with plain pandas."""
    df = pd.read_csv(DATA_DIR / "BEGAD_ETRA25_BEG-201.csv")
    for col in ["SaccadicAmplitude", "AbsoluteSaccadicDirection",
                "RelativeSaccadicDirection", "PupilLeft", "PupilRight"]:
        df[col] = df[col].astype("str").str.replace(",", ".").replace("", np.nan).astype(np.float64)
    df = df[
        (df["Final"] == True)
        & (~df["HasNulls"])
        & (df["GazeEventType"] == "Fixation")
        & (df["FixationIndex"] > 0)
        & (df["GazeEventDuration"] > DUR_MIN)
        & (df["GazeEventDuration"] <= DUR_MAX)
    ].copy()
    df["w"] = df["Task"].map(lambda t: TASK_GEOMETRY[t][2])
    df["h"] = df["Task"].map(lambda t: TASK_GEOMETRY[t][3])
    x, y = df["FixationPointX (MCSpx)"], df["FixationPointY (MCSpx)"]
    df = df[(x >= 0) & (x <= df["w"]) & (y >= 0) & (y <= df["h"])]
    df["pup"] = (df["PupilLeft"] + df["PupilRight"]) / 2.0
    med = df.groupby("Task")["pup"].transform("median")
    sd = df.groupby("Task")["pup"].transform("std").fillna(0.0)
    df = df[np.abs(df["pup"] - med) <= PUPIL_SD * np.maximum(sd, PUPIL_EPS)]
    return df


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (DATA_DIR / "BEGAD_ETRA25_BEG-201.csv").exists():
            raise unittest.SkipTest("dataset_v2 not present")
        cls.features, cls.metadata, cls.stats = build_all(
            participants_subset=["BEG-201"]
        )

    def test_table_shape_and_columns(self):
        self.assertEqual(self.features.shape, (11, 27))
        self.assertEqual(list(self.features.columns), FEATURE_COLS)
        self.assertEqual(set(self.features["task"]), set(TASK_IDS))
        self.assertTrue((self.features["participant_id"] == "BEG-201").all())

    def test_values_match_independent_recompute_for_task_1(self):
        ref = reference_clean_fixations()
        g = ref[ref["Task"] == 1].sort_values("RecordingTimestamp")
        row = self.features[self.features["task"] == 1].iloc[0]
        n = len(g)
        self.assertAlmostEqual(row["spa_fix_count"], float(n))
        x = g["FixationPointX (MCSpx)"].to_numpy()
        y = g["FixationPointY (MCSpx)"].to_numpy()
        self.assertAlmostEqual(
            row["spa_dispersion"],
            np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2).mean(),
        )
        dx, dy = np.diff(x), np.diff(y)
        self.assertAlmostEqual(row["geo_scanpath_len"],
                               np.sqrt(dx ** 2 + dy ** 2).sum())
        self.assertAlmostEqual(row["tem_dur_total"], g["GazeEventDuration"].sum())
        self.assertAlmostEqual(row["pup_mean"], g["pup"].mean())
        self.assertAlmostEqual(
            row["pupbl_std"],
            (((g["PupilLeft"] - g["PupilLeftBaseline"])
              + (g["PupilRight"] - g["PupilRightBaseline"])) / 2.0).std(),
        )
        self.assertAlmostEqual(
            row["sac_amp_mean"], g["SaccadicAmplitude"].mean()
        )

    def test_metadata_and_stats(self):
        meta = self.metadata.iloc[0]
        self.assertEqual(meta["participant_id"], "BEG-201")
        self.assertEqual(meta["group_label"], "CO")
        self.assertEqual(meta["n_tasks_present"], 11)
        clean_total = sum(
            s["clean"] for s in self.stats.values()
        )
        self.assertEqual(meta["n_clean_fixations"], clean_total)


if __name__ == "__main__":
    unittest.main()
