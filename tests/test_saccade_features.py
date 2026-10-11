"""Tests for the measured-saccade feature group (begad_norm/features/saccade.py)."""

import math
import unittest

import numpy as np

from begad_norm.features.saccade import compute_saccade_features


class SaccadeFeatureTests(unittest.TestCase):
    def test_four_fixation_probe(self):
        # n=4, m=3 runs; total_time = 1100 + 275*3 = 1925
        f = compute_saccade_features(
            amp=np.array([2.0, 3.0, 4.0]),
            absdir=np.array([0.0, 90.0, 180.0]),
            reldir=np.array([0.0, 90.0, 180.0]),
            run_durs=np.array([20.0, 30.0, 25.0]),
            total_time=1925.0,
        )
        self.assertEqual(f["sac_count"], 3.0)
        self.assertAlmostEqual(f["sac_rate"], 3.0 / 1925.0 * 1000.0, places=9)
        self.assertAlmostEqual(f["sac_dur_mean"], 25.0, places=9)
        self.assertAlmostEqual(f["sac_dur_std"], 5.0, places=9)
        self.assertAlmostEqual(f["sac_dur_total"], 75.0, places=9)
        self.assertAlmostEqual(f["sac_dur_max"], 30.0, places=9)
        self.assertAlmostEqual(f["sac_amp_mean"], 3.0, places=9)
        self.assertAlmostEqual(f["sac_amp_std"], 1.0, places=9)
        self.assertAlmostEqual(f["sac_amp_max"], 4.0, places=9)
        self.assertAlmostEqual(f["sac_absdir_mean"], 90.0, places=9)
        self.assertAlmostEqual(f["sac_absdir_var"], 2.0 / 3.0, places=9)
        self.assertAlmostEqual(f["sac_turn_mean"], 90.0, places=9)
        self.assertAlmostEqual(f["sac_turn_std"], 90.0, places=9)

    def test_no_runs_case(self):
        f = compute_saccade_features(
            amp=np.array([2.0, 3.0]),
            absdir=np.array([0.0, 90.0]),
            reldir=np.array([0.0, 90.0]),
            run_durs=np.array([]),
            total_time=1000.0,
        )
        self.assertEqual(f["sac_count"], 0.0)
        self.assertEqual(f["sac_rate"], 0.0)
        self.assertTrue(math.isnan(f["sac_dur_mean"]))
        self.assertTrue(math.isnan(f["sac_dur_std"]))
        self.assertTrue(math.isnan(f["sac_dur_total"]))
        self.assertTrue(math.isnan(f["sac_dur_max"]))

    def test_zero_total_time_rate_is_nan(self):
        f = compute_saccade_features(
            amp=np.array([2.0]),
            absdir=np.array([0.0]),
            reldir=np.array([0.0]),
            run_durs=np.array([20.0]),
            total_time=0.0,
        )
        self.assertTrue(math.isnan(f["sac_rate"]))

    def test_circular_wrap(self):
        f = compute_saccade_features(
            amp=np.array([2.0, 2.0]),
            absdir=np.array([350.0, 10.0]),
            reldir=np.array([0.0, 0.0]),
            run_durs=np.array([20.0]),
            total_time=1000.0,
        )
        # 350 and 10 wrap to circular mean 0
        self.assertAlmostEqual(f["sac_absdir_mean"], 0.0, places=9)
        cos10 = math.cos(math.radians(10.0))
        self.assertAlmostEqual(f["sac_absdir_var"], 1.0 - cos10, places=9)

    def test_single_fixation_nan_stds(self):
        f = compute_saccade_features(
            amp=np.array([2.0]),
            absdir=np.array([45.0]),
            reldir=np.array([30.0]),
            run_durs=np.array([20.0]),
            total_time=1000.0,
        )
        self.assertAlmostEqual(f["sac_amp_mean"], 2.0, places=9)
        self.assertTrue(math.isnan(f["sac_amp_std"]))
        self.assertAlmostEqual(f["sac_absdir_mean"], 45.0, places=9)
        self.assertTrue(math.isnan(f["sac_absdir_var"]))
        self.assertAlmostEqual(f["sac_turn_mean"], 30.0, places=9)
        self.assertTrue(math.isnan(f["sac_turn_std"]))

    def test_turn_folds_over_360(self):
        f = compute_saccade_features(
            amp=np.array([2.0, 2.0]),
            absdir=np.array([0.0, 0.0]),
            reldir=np.array([359.97, 0.0]),
            run_durs=np.array([20.0]),
            total_time=1000.0,
        )
        self.assertAlmostEqual(f["sac_turn_mean"], 0.03 / 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
