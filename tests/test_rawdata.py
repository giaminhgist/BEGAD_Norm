"""Tests for BEGAD data loading/cleaning (begad_norm/rawdata.py)."""

import math
import unittest

import numpy as np
import pandas as pd

from begad_norm.config import TASK_GEOMETRY
from begad_norm.rawdata import (
    clean_fixations,
    extract_saccade_runs,
    parse_decimal_columns,
    participant_id_from_filename,
)


def make_row(
    task=1,
    ts=100.0,
    event="Fixation",
    final=True,
    has_nulls=False,
    fix_index=1.0,
    dur=200.0,
    x=100.0,
    y=100.0,
    pupil_l=3.0,
    pupil_r=3.2,
    pupil_lb=2.5,
    pupil_rb=2.7,
    amp=5.0,
    absdir=45.0,
    reldir=10.0,
    sacc_index=np.nan,
    group=0,
    is_followup=0,
):
    return {
        "RecordingTimestamp": ts,
        "FixationIndex": fix_index,
        "SaccadeIndex": sacc_index,
        "GazeEventType": event,
        "GazeEventDuration": dur,
        "FixationPointX (MCSpx)": x,
        "FixationPointY (MCSpx)": y,
        "SaccadicAmplitude": amp,
        "AbsoluteSaccadicDirection": absdir,
        "RelativeSaccadicDirection": reldir,
        "PupilLeft": pupil_l,
        "PupilRight": pupil_r,
        "PupilLeftBaseline": pupil_lb,
        "PupilRightBaseline": pupil_rb,
        "Task": task,
        "Final": final,
        "HasNulls": has_nulls,
        "Group": group,
        "RealTag": "BEG-999",
        "IsFollowup": is_followup,
    }


def make_df(rows):
    return pd.DataFrame(rows)


class ParseDecimalColumnsTests(unittest.TestCase):
    def test_parses_comma_decimals_and_preserves_dots(self):
        df = pd.DataFrame(
            {
                "SaccadicAmplitude": pd.Series(["3,98", "2.38333", np.nan], dtype="str"),
                "PupilLeft": pd.Series(["0,88", "3,40", np.nan], dtype="str"),
            }
        )
        out = parse_decimal_columns(df)
        self.assertAlmostEqual(out["SaccadicAmplitude"].iloc[0], 3.98)
        self.assertAlmostEqual(out["SaccadicAmplitude"].iloc[1], 2.38333)
        self.assertTrue(math.isnan(out["SaccadicAmplitude"].iloc[2]))
        self.assertAlmostEqual(out["PupilLeft"].iloc[0], 0.88)
        self.assertAlmostEqual(out["PupilLeft"].iloc[1], 3.4)
        self.assertTrue(math.isnan(out["PupilLeft"].iloc[2]))

    def test_empty_string_becomes_nan(self):
        df = pd.DataFrame({"PupilLeft": pd.Series(["", "3,2"], dtype="str")})
        out = parse_decimal_columns(df)
        self.assertTrue(math.isnan(out["PupilLeft"].iloc[0]))
        self.assertAlmostEqual(out["PupilLeft"].iloc[1], 3.2)


class CleanFixationsTests(unittest.TestCase):
    def _clean(self, rows):
        df = parse_decimal_columns(make_df(rows))
        return clean_fixations(df)

    def test_keeps_only_final_clean_fixations(self):
        rows = [
            make_row(),                                        # kept
            make_row(event="Saccade", final=True),             # dropped: saccade
            make_row(final=False),                             # dropped: not final
            make_row(has_nulls=True),                          # dropped: nulls
            make_row(fix_index=np.nan),                        # dropped: no fixation index
            make_row(group=3),                                 # note: cohort filtered at roster level, not here
        ]
        out = self._clean(rows)
        # group filter is not row-level here; the saccade/final/nulls/index rows drop
        self.assertEqual(len(out), 2)

    def test_duration_window_boundaries(self):
        w, h = TASK_GEOMETRY[1][2], TASK_GEOMETRY[1][3]
        rows = [
            make_row(dur=40.0),                                # dropped (<= DUR_MIN)
            make_row(dur=41.0, ts=110.0),                      # kept
            make_row(dur=2000.0, ts=120.0),                    # kept
            make_row(dur=2001.0, ts=130.0),                    # dropped
        ]
        out = self._clean(rows)
        self.assertEqual(sorted(out["dur"].tolist()), [41.0, 2000.0])

    def test_bounds_are_inclusive_on_media(self):
        w, h = TASK_GEOMETRY[1][2], TASK_GEOMETRY[1][3]
        rows = [
            make_row(x=-1.0),                                  # dropped
            make_row(x=0.0, y=0.0, ts=110.0),                  # kept (inclusive)
            make_row(x=w, y=h, ts=120.0),                      # kept (inclusive)
            make_row(x=w + 1.0, ts=130.0),                     # dropped
        ]
        out = self._clean(rows)
        self.assertEqual(len(out), 2)
        self.assertTrue((out["x"] <= w).all())
        self.assertTrue((out["x"] >= 0).all())

    def test_pupil_outlier_filter_on_binocular_mean(self):
        # 99 points at 3.0/3.2 plus one extreme 100 -> extreme dropped
        rows = []
        for i in range(99):
            rows.append(make_row(ts=100.0 + i, pupil_l=3.0, pupil_r=3.2))
        rows.append(make_row(ts=300.0, pupil_l=100.0, pupil_r=100.2))
        out = self._clean(rows)
        self.assertEqual(len(out), 99)

    def test_pupil_filter_keeps_single_fixation_group(self):
        rows = [make_row(pupil_l=50.0, pupil_r=50.0)]
        out = self._clean(rows)
        self.assertEqual(len(out), 1)

    def test_output_columns_and_sorting(self):
        rows = [
            make_row(ts=300.0),
            make_row(ts=100.0),
        ]
        out = self._clean(rows)
        self.assertEqual(
            list(out.columns),
            ["task", "ts", "dur", "x", "y", "pup", "pup_corrected", "pupdiff",
             "amp", "absdir", "reldir"],
        )
        self.assertEqual(out["ts"].tolist(), [100.0, 300.0])
        # binocular mean pupil and baseline-corrected pupil
        self.assertAlmostEqual(out["pup"].iloc[0], (3.0 + 3.2) / 2.0)
        self.assertAlmostEqual(
            out["pup_corrected"].iloc[0],
            ((3.0 - 2.5) + (3.2 - 2.7)) / 2.0,
        )


class CleanFixationsStatsTests(unittest.TestCase):
    def test_return_stats_reports_stage_drops(self):
        rows = [
            make_row(),                          # kept
            make_row(dur=40.0, ts=110.0),        # duration drop
            make_row(x=-5.0, ts=120.0),          # bounds drop
            make_row(event="Saccade", final=True),  # fixation-filter drop
        ]
        df = parse_decimal_columns(make_df(rows))
        out, stats = clean_fixations(df, return_stats=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(stats["raw"], 4)
        self.assertEqual(stats["fixation_filter_dropped"], 1)
        self.assertEqual(stats["duration_dropped"], 1)
        self.assertEqual(stats["bounds_dropped"], 1)
        self.assertEqual(stats["pupil_dropped"], 0)
        self.assertEqual(stats["clean"], 1)

    def test_stats_pupil_drop_counted(self):
        rows = [make_row(ts=100.0 + i, pupil_l=3.0, pupil_r=3.2) for i in range(99)]
        rows.append(make_row(ts=300.0, pupil_l=100.0, pupil_r=100.2))
        df = parse_decimal_columns(make_df(rows))
        out, stats = clean_fixations(df, return_stats=True)
        self.assertEqual(stats["pupil_dropped"], 1)
        self.assertEqual(len(out), 99)


class ExtractSaccadeRunsTests(unittest.TestCase):
    def _runs(self, saccade_rows, clean_rows):
        df = make_df(saccade_rows + clean_rows)
        clean = clean_fixations(df)
        return extract_saccade_runs(df, clean)

    def test_groups_overlapping_saccades_by_landing_fixation(self):
        sac = [
            make_row(event="Saccade", final=True, has_nulls=True, fix_index=np.nan,
                     sacc_index=1.0, ts=50.0, dur=20.0),
            make_row(event="Saccade", final=True, has_nulls=True, fix_index=np.nan,
                     sacc_index=2.0, ts=60.0, dur=30.0),
            make_row(event="Saccade", final=True, has_nulls=True, fix_index=np.nan,
                     sacc_index=3.0, ts=250.0, dur=25.0),
        ]
        clean = [
            make_row(ts=100.0),
            make_row(ts=300.0),
        ]
        runs = self._runs(sac, clean)
        # two runs: max(20,30) landing on ts=100, and 25 landing on ts=300
        self.assertEqual(sorted(runs["run_dur"].tolist()), [25.0, 30.0])

    def test_saccade_without_landing_fixation_is_dropped(self):
        sac = [
            make_row(event="Saccade", final=True, has_nulls=True, fix_index=np.nan,
                     sacc_index=1.0, ts=400.0, dur=15.0),  # no fixation after -> dropped
            make_row(event="Saccade", final=True, has_nulls=True, fix_index=np.nan,
                     sacc_index=2.0, ts=50.0, dur=22.0),
        ]
        clean = [make_row(ts=100.0)]
        runs = self._runs(sac, clean)
        self.assertEqual(runs["run_dur"].tolist(), [22.0])

    def test_non_final_saccade_rows_are_ignored(self):
        sac = [
            make_row(event="Saccade", final=False, fix_index=np.nan,
                     sacc_index=1.0, ts=50.0, dur=20.0),
        ]
        clean = [make_row(ts=100.0)]
        runs = self._runs(sac, clean)
        self.assertEqual(len(runs), 0)

    def test_runs_are_split_by_task(self):
        sac = [
            make_row(event="Saccade", final=True, has_nulls=True, fix_index=np.nan,
                     sacc_index=1.0, ts=50.0, dur=20.0, task=1),
            make_row(event="Saccade", final=True, has_nulls=True, fix_index=np.nan,
                     sacc_index=1.0, ts=50.0, dur=40.0, task=2),
        ]
        clean = [make_row(ts=100.0, task=1), make_row(ts=100.0, task=2)]
        runs = self._runs(sac, clean)
        self.assertEqual(sorted(runs["run_dur"].tolist()), [20.0, 40.0])


class ParticipantIdTests(unittest.TestCase):
    def test_filename_to_participant_id(self):
        self.assertEqual(
            participant_id_from_filename("BEGAD_ETRA25_BEG-201.csv"), "BEG-201"
        )
        self.assertEqual(
            participant_id_from_filename("BEGAD_ETRA25_BEGCO-101.csv"), "BEGCO-101"
        )


if __name__ == "__main__":
    unittest.main()
