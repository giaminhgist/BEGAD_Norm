"""Tests for the EMS 45-feature port (begad_norm/features/stimulus.py).

Expected constants were produced by RUNNING the original EMS code
(/root/EMS-Projects/src/features/stimulus.py) on the same probe inputs,
so these tests prove the port reproduces EMS exactly.
"""

import math
import unittest

import numpy as np

from begad_norm.features.stimulus import FEATURE_NAMES, compute_stimulus_features

# Probe 1: the EMS 3-point probe (x, y, dur, pup), screen 1024x768.
X3 = np.array([512.0, 514.0, 520.0])
Y3 = np.array([384.0, 390.0, 400.0])
DUR3 = np.array([200.0, 250.0, 300.0])
PUP3 = np.array([1300.0, 1320.0, 1350.0])

EXPECTED3 = {
    "spa_fix_count": 3.0,
    "spa_mean_x": 515.3333333333334,
    "spa_mean_y": 391.3333333333333,
    "spa_std_x": 4.163331998932265,
    "spa_std_y": 8.082903768654761,
    "spa_dispersion": 6.59473247968315,
    "spa_bbox_area": 128.0,
    "spa_center_dist_mean": 8.071033046778359,
    "spa_center_dist_std": 9.071253404944516,
    "spa_center_frac": 1.0,
    "spa_q1": 0.0,
    "spa_q2": 0.0,
    "spa_q3": 0.0,
    "spa_q4": 1.0,
    "spa_entropy": 0.0,
    "spa_max_grid_frac": 1.0,
    "spa_skew_x": 0.28740950682962535,
    "geo_scanpath_len": 17.98645911002736,
    "geo_sacc_amp_mean": 8.99322955501368,
    "geo_sacc_amp_std": 3.774075296235742,
    "geo_sacc_amp_max": 11.661903789690601,
    "geo_dx_mean": 4.0,
    "geo_dy_mean": 8.0,
    "geo_angle_var": 0.005971061743182182,
    "geo_revisit_rate": 0.6666666666666666,
    "geo_nn_dist_mean": 8.103671476788039,
    "geo_hull_area": np.nan,
    "tem_dur_mean": 250.0,
    "tem_dur_std": 50.0,
    "tem_dur_total": 750.0,
    "tem_dur_max": 300.0,
    "tem_first_dur": 200.0,
    "tem_last_dur": 300.0,
    "tem_ifi_mean": 250.0,
    "tem_ifi_std": 35.35533905932738,
    "tem_velocity_mean": 0.03525802881432723,
    "tem_fix_rate": 2.4,
    "tem_trans_entropy": 0.0,
    "pup_mean": 1323.3333333333333,
    "pup_std": 25.166114784235834,
    "pup_min": 1300.0,
    "pup_max": 1350.0,
    "pup_median": 1320.0,
    "pup_slope": 25.00000000000004,
    "pup_first_last_diff": 50.0,
}


class StimulusFeatureTests(unittest.TestCase):
    def test_feature_names_match_ems_order(self):
        self.assertEqual(len(FEATURE_NAMES), 45)
        self.assertEqual(FEATURE_NAMES, list(EXPECTED3.keys()))

    def test_three_point_probe_matches_ems_run(self):
        f = compute_stimulus_features(X3, Y3, DUR3, PUP3)
        self.assertEqual(set(f.keys()), set(EXPECTED3.keys()))
        for name, expected in EXPECTED3.items():
            with self.subTest(feature=name):
                if expected is not None and math.isnan(expected):
                    self.assertTrue(math.isnan(f[name]), f"{name}: {f[name]}")
                else:
                    self.assertAlmostEqual(f[name], expected, places=9, msg=name)

    def test_four_point_quadrant_hull_probe(self):
        x = np.array([128.0, 900.0, 128.0, 900.0])
        y = np.array([128.0, 128.0, 600.0, 600.0])
        dur = np.array([100.0, 200.0, 300.0, 400.0])
        pup = np.array([3.0, 3.1, 3.2, 3.3])
        f = compute_stimulus_features(x, y, dur, pup)
        self.assertAlmostEqual(f["spa_q1"], 0.25, places=9)
        self.assertAlmostEqual(f["spa_q2"], 0.25, places=9)
        self.assertAlmostEqual(f["spa_q3"], 0.25, places=9)
        self.assertAlmostEqual(f["spa_q4"], 0.25, places=9)
        self.assertAlmostEqual(f["spa_bbox_area"], 364384.0, places=6)
        self.assertAlmostEqual(f["geo_hull_area"], 364384.0, places=6)
        self.assertAlmostEqual(f["spa_skew_x"], 0.0, places=9)
        self.assertAlmostEqual(f["spa_mean_x"], 514.0, places=9)
        self.assertAlmostEqual(f["spa_mean_y"], 364.0, places=9)
        self.assertAlmostEqual(f["spa_center_frac"], 0.0, places=9)

    def test_single_fixation_edge_case(self):
        f = compute_stimulus_features(
            np.array([512.0]), np.array([384.0]),
            np.array([200.0]), np.array([1300.0]),
        )
        self.assertEqual(f["spa_fix_count"], 1.0)
        self.assertEqual(f["spa_mean_x"], 512.0)
        self.assertTrue(math.isnan(f["spa_std_x"]))
        self.assertEqual(f["spa_dispersion"], 0.0)
        self.assertEqual(f["spa_bbox_area"], 0.0)
        self.assertEqual(f["spa_center_dist_mean"], 0.0)
        self.assertEqual(f["spa_center_frac"], 1.0)
        self.assertEqual(f["spa_q1"], 0.0)
        self.assertEqual(f["spa_q2"], 0.0)
        self.assertEqual(f["spa_q3"], 0.0)
        self.assertEqual(f["spa_q4"], 1.0)
        self.assertEqual(f["spa_entropy"], 0.0)
        self.assertEqual(f["spa_max_grid_frac"], 1.0)
        self.assertTrue(math.isnan(f["spa_skew_x"]))
        for k in ["geo_scanpath_len", "geo_sacc_amp_mean", "geo_sacc_amp_std",
                  "geo_sacc_amp_max", "geo_dx_mean", "geo_dy_mean",
                  "geo_angle_var", "geo_revisit_rate", "geo_nn_dist_mean",
                  "geo_hull_area"]:
            self.assertTrue(math.isnan(f[k]), k)
        self.assertEqual(f["tem_dur_mean"], 200.0)
        self.assertTrue(math.isnan(f["tem_dur_std"]))
        self.assertEqual(f["tem_dur_total"], 200.0)
        self.assertEqual(f["tem_first_dur"], 200.0)
        self.assertEqual(f["tem_last_dur"], 200.0)
        self.assertTrue(math.isnan(f["tem_ifi_mean"]))
        self.assertTrue(math.isnan(f["tem_velocity_mean"]))
        self.assertEqual(f["tem_fix_rate"], 5.0)  # 1000 / 200
        self.assertTrue(math.isnan(f["tem_trans_entropy"]))
        self.assertEqual(f["pup_mean"], 1300.0)
        self.assertTrue(math.isnan(f["pup_std"]))
        self.assertEqual(f["pup_min"], 1300.0)
        self.assertEqual(f["pup_max"], 1300.0)
        self.assertEqual(f["pup_median"], 1300.0)
        self.assertTrue(math.isnan(f["pup_slope"]))
        self.assertTrue(math.isnan(f["pup_first_last_diff"]))

    def test_two_point_skew_is_nan(self):
        f = compute_stimulus_features(
            np.array([100.0, 200.0]), np.array([100.0, 200.0]),
            np.array([100.0, 100.0]), np.array([3.0, 3.0]),
        )
        self.assertTrue(math.isnan(f["spa_skew_x"]))
        self.assertFalse(math.isnan(f["spa_std_x"]))
        self.assertTrue(math.isnan(f["geo_sacc_amp_std"]))  # n=2 -> ddof=1 std of 1 value

    def test_zero_std_x_gives_nan_skew(self):
        f = compute_stimulus_features(
            np.array([100.0, 100.0, 100.0]), np.array([100.0, 200.0, 300.0]),
            np.array([100.0, 100.0, 100.0]), np.array([3.0, 3.0, 3.0]),
        )
        self.assertTrue(math.isnan(f["spa_skew_x"]))
        self.assertEqual(f["spa_bbox_area"], 0.0)

    def test_collinear_points_give_nan_hull(self):
        f = compute_stimulus_features(
            np.array([0.0, 10.0, 20.0, 30.0]), np.array([0.0, 10.0, 20.0, 30.0]),
            np.array([100.0] * 4), np.array([3.0] * 4),
        )
        self.assertTrue(math.isnan(f["geo_hull_area"]))

    def test_zero_gap_velocity_is_nan(self):
        f = compute_stimulus_features(
            np.array([0.0, 10.0]), np.array([0.0, 0.0]),
            np.array([0.0, 0.0]), np.array([3.0, 3.0]),
        )
        self.assertTrue(math.isnan(f["tem_velocity_mean"]))

    def test_two_cell_entropy(self):
        # two points in distinct 128-px cells of the 8x6 grid -> -H/log(48) = ln2/ln48
        f = compute_stimulus_features(
            np.array([0.0, 200.0]), np.array([0.0, 0.0]),
            np.array([100.0, 100.0]), np.array([3.0, 3.0]),
        )
        self.assertAlmostEqual(f["spa_entropy"], math.log(2) / math.log(48), places=9)


if __name__ == "__main__":
    unittest.main()
