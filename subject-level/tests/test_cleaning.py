"""Tests for begad_norm.cleaning — T1235/T5 row-cleaning pipeline.

Contract (user decision 2026-10-17):
  - exclude rows with any start/end pupil column <= 0 (impossible pupil size);
  - impute totalFixTime for zero-fix rows (totalFixTime == 0, nFixations == 1;
    in both datasets totalFixTime is recorded only for nFixations >= 2 rows):
    median per-fixation duration (totalFixTime / nFixations) of reference
    rows (nFixations >= 2, totalFixTime > 0), per (task, ROI) with fallback
    to task, then to the whole dataset, times the row's 1 fixation;
  - each dataset is cleaned independently (no pooling).
"""
import unittest

import numpy as np
import pandas as pd

from begad_norm.config import PROJECT_ROOT
from begad_norm.cleaning import (clean_dataset, impute_zero_fix,
                                 pupil_cols, pupil_exclusion_cols)

T1235_COLS = ["group", "pat", "task", "ROI", "nFixations", "nTobiiSaccades",
              "regSaccades", "longSaccades", "tinySaccades", "saccadeTotLength",
              "totalFixTime", "totalSpokenTime", "speechDelay", "endSpeechDelay",
              "startPupLB", "startPupRB", "endPupLB", "endPupRB", "diffPupL",
              "diffPupR", "maxPupLB", "minPupLB", "diffXpupL"]
T5_COLS = ["group", "pat", "task", "ROI", "nFixations", "nTobiiSaccades",
           "regSaccades", "longSaccades", "tinySaccades", "saccadeTotLength",
           "totalFixTime", "totalSpokenTime", "speechDelay", "endSpeechDelay",
           "startPupL", "startPupR", "endPupL", "endPupR", "diffPupL", "diffPupR"]


class TestPupilCols(unittest.TestCase):
    def test_t1235(self):
        self.assertEqual(pupil_cols(T1235_COLS),
                         ["startPupLB", "startPupRB", "endPupLB", "endPupRB",
                          "maxPupLB", "minPupLB"])
        self.assertEqual(pupil_exclusion_cols(T1235_COLS),
                         ["startPupLB", "startPupRB", "endPupLB", "endPupRB"])

    def test_t5(self):
        self.assertEqual(pupil_cols(T5_COLS),
                         ["startPupL", "startPupR", "endPupL", "endPupR"])
        self.assertEqual(pupil_exclusion_cols(T5_COLS),
                         ["startPupL", "startPupR", "endPupL", "endPupR"])


class TestImputeZeroFix(unittest.TestCase):
    def _frame(self):
        # reference ratios: (5,1) -> [0.5, 0.3] median 0.4 ; (5,2) -> [0.6]
        #   ; global median of [0.5, 0.3, 0.6] = 0.5
        # row0 (5,1) zero-fix -> 0.4 ; row4 (5,2) zero-fix -> 0.6 ;
        # row5 (3,1) zero-fix, no (3,1) or task-3 refs -> global 0.5
        return pd.DataFrame({
            "task": [5, 5, 5, 5, 5, 3],
            "ROI": [1, 1, 1, 2, 2, 1],
            "nFixations": [1, 2, 4, 3, 1, 1],
            "totalFixTime": [0.0, 1.0, 1.2, 1.8, 0.0, 0.0],
        })

    def test_grouped_median_and_fallbacks(self):
        out, stats = impute_zero_fix(self._frame())
        np.testing.assert_allclose(out["totalFixTime"],
                                   [0.4, 1.0, 1.2, 1.8, 0.6, 0.5])
        self.assertEqual(stats["n_zero_fix"], 3)
        self.assertEqual(stats["n_imputed"], 3)
        self.assertEqual(stats["n_by_level"],
                         {"roi": 2, "task": 0, "global": 1})

    def test_non_zero_fix_rows_untouched(self):
        df = self._frame()
        df.loc[1, "totalFixTime"] = 0.7  # nFixations>=2, >0: reference row
        out, _ = impute_zero_fix(df)
        self.assertEqual(out.loc[1, "totalFixTime"], 0.7)


class TestCleanDatasetRealData(unittest.TestCase):
    """End-to-end invariants on the real xlsx files (skip if absent)."""

    def _check(self, name):
        xlsx = PROJECT_ROOT.parent / f"{name}.xlsx"
        if not xlsx.exists():
            self.skipTest(f"{name}.xlsx not present")
        dfc, excl, stats = clean_dataset(xlsx)
        raw = pd.read_excel(xlsx, engine="openpyxl")
        self.assertEqual(len(dfc) + len(excl), len(raw))
        self.assertEqual(set(dfc.columns), set(raw.columns))
        # exclusion invariant: no start/end pupil <= 0 remains
        for c in pupil_exclusion_cols(dfc.columns):
            self.assertEqual(int((dfc[c] <= 0).sum()), 0, c)
        # imputation invariant: no zero-fix rows remain
        self.assertEqual(
            int(((dfc["totalFixTime"] == 0) & (dfc["nFixations"] > 0)).sum()), 0)
        self.assertEqual(int((dfc["totalFixTime"] <= 0).sum()), 0)
        # structure invariants preserved
        self.assertEqual(dfc.duplicated(subset=["pat", "task", "ROI"]).sum(), 0)
        self.assertEqual(dfc["pat"].nunique(), 64)
        self.assertEqual(int(dfc.groupby("pat")["group"].nunique().max()), 1)
        self.assertEqual(int(dfc["group"].isin({0, 1, 2}).all()), 1)
        # imputation applied only where intended
        self.assertTrue((dfc["nFixations"] >= 1).all())
        # excluded rows are exactly the pupil<=0 rows (reason column)
        self.assertEqual(set(excl["reason"].unique()),
                         {"pupil_start_end_le0"})

    def test_t1235(self):
        self._check("T1235")

    def test_t5(self):
        self._check("T5")


if __name__ == "__main__":
    unittest.main()
