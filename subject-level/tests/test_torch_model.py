"""Tests for begad_norm.rowlevel.torch_model — PyTorch port of the
multimodal model in Multimodal.ipynb (3 branches, L2 + dropout, softmax
output, early stopping on a carved validation set)."""
import unittest

import numpy as np
import torch

from begad_norm.rowlevel.torch_model import (
    MultimodalNet, carve_val_mask, fit_torch_model, torch_branch_inputs)

TORCH_CFG = {"max_epochs": 20, "patience": 3, "batch_size": 32,
             "lr": 1e-3, "l2": 1e-3, "dropout": 0.2}

COLS = ["group", "pat", "task", "ROI", "nFixations", "nTobiiSaccades",
        "regSaccades", "longSaccades", "tinySaccades", "saccadeTotLength",
        "totalFixTime", "totalSpokenTime", "speechDelay", "endSpeechDelay",
        "startPupL", "startPupR", "endPupL", "endPupR", "diffPupL",
        "diffPupR"]
BRANCHES = {
    "visual": ["nFixations", "nTobiiSaccades", "regSaccades", "longSaccades",
               "tinySaccades", "saccadeTotLength"],
    "speech": ["totalFixTime", "totalSpokenTime", "speechDelay",
               "endSpeechDelay"],
    "pupil": ["startPupL", "startPupR", "endPupL", "endPupR", "diffPupL",
              "diffPupR"],
}


class TestBranchInputs(unittest.TestCase):
    def test_t5_layout(self):
        rng = np.random.default_rng(0)
        cols = COLS[4:] + ["ROI"]
        X = rng.uniform(size=(50, len(cols)))
        Xv, Xs, Xp = torch_branch_inputs(X, cols, BRANCHES, "ROI")
        self.assertEqual(Xv.shape, (50, 7))
        self.assertEqual(Xs.shape, (50, 5))
        self.assertEqual(Xp.shape, (50, 7))
        # ROI appended last in every branch
        roi_idx = cols.index("ROI")
        np.testing.assert_allclose(Xv[:, -1], X[:, roi_idx])
        np.testing.assert_allclose(Xs[:, -1], X[:, roi_idx])
        np.testing.assert_allclose(Xp[:, -1], X[:, roi_idx])
        # branch columns in declared order
        v_first = cols.index(BRANCHES["visual"][0])
        np.testing.assert_allclose(Xv[:, 0], X[:, v_first])

    def test_t1235_pupil_extra(self):
        cols2 = COLS[4:] + ["maxPupLB", "minPupLB", "diffXpupL", "ROI"]
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(20, len(cols2)))
        branches = dict(BRANCHES)
        branches["pupil"] = BRANCHES["pupil"] + ["maxPupLB", "minPupLB",
                                                 "diffXpupL"]
        Xv, Xs, Xp = torch_branch_inputs(X, cols2, branches, "ROI")
        self.assertEqual((Xv.shape, Xs.shape, Xp.shape),
                         ((20, 7), (20, 5), (20, 10)))


class TestMultimodalNet(unittest.TestCase):
    def test_forward_shapes(self):
        m = MultimodalNet(7, 5, 7, 3)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(8, 7), torch.randn(8, 5), torch.randn(8, 7))
        self.assertEqual(tuple(out.shape), (8, 3))
        p = torch.softmax(out, dim=1)
        np.testing.assert_allclose(p.sum(1).numpy(), 1.0, atol=1e-6)

    def test_binary_head(self):
        m = MultimodalNet(7, 5, 10, 2)
        with torch.no_grad():
            out = m(torch.randn(4, 7), torch.randn(4, 5), torch.randn(4, 10))
        self.assertEqual(tuple(out.shape), (4, 2))

    def test_l2_penalty_groups(self):
        m = MultimodalNet(7, 5, 7, 3, l2=1e-3)
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
            # pupil Dense(128) is NOT regularized (notebook quirk)
            m.pupil_net[0].weight.fill_(1.0)
            self.assertEqual(m.l2_penalty().item(), 0.0)
            m.pupil_net[0].weight.zero_()
            # output layer is NOT regularized
            m.fusion[-1].weight.fill_(1.0)
            self.assertEqual(m.l2_penalty().item(), 0.0)
            m.fusion[-1].weight.zero_()
            # visual Dense(128) IS regularized: 7 x 128 weights
            m.visual_net[0].weight.fill_(1.0)
            self.assertEqual(m.l2_penalty().item(), 7 * 128)
            # biases are excluded
            m.visual_net[0].bias.fill_(1.0)
            self.assertEqual(m.l2_penalty().item(), 7 * 128)


class TestCarveValMask(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.pat = np.repeat([f"p{i}" for i in range(20)], 4)
        self.pid_group = {f"p{i}": ["CO", "MCI", "AD"][i % 3]
                          for i in range(20)}
        self.y = np.array([["CO", "MCI", "AD"][i % 3]
                           for i in range(20) for _ in range(4)])
        self.y = np.array([{"CO": 0, "MCI": 1, "AD": 2}[g] for g in self.y])
        self.train = np.ones(len(self.pat), dtype=bool)

    def test_grouped_disjoint(self):
        test_mask = np.zeros(len(self.pat), dtype=bool)
        test_mask[::10] = True  # some rows held out
        val = carve_val_mask(~test_mask, self.pat, self.pid_group, self.y,
                             "grouped", 42)
        val_pats = set(self.pat[val])
        test_pats = set(self.pat[test_mask])
        self.assertFalse(val_pats & test_pats)
        self.assertTrue(val.sum() > 0)
        self.assertLessEqual(val.sum() / (~test_mask).sum(), 0.25)

    def test_random_fraction(self):
        val = carve_val_mask(self.train, self.pat, self.pid_group, self.y,
                             "random", 42)
        self.assertAlmostEqual(val.mean(), 0.2, delta=0.05)


class TestFitTorchModel(unittest.TestCase):
    def _data(self, n=96, separable=True):
        rng = np.random.default_rng(0)
        Xv = rng.normal(size=(n, 7))
        Xs = rng.normal(size=(n, 5))
        Xp = rng.normal(size=(n, 7))
        y = rng.integers(0, 2, n)
        if separable:
            Xv[y == 0] += np.array([2.0, 0, 0, 0, 0, 0, 0])
        train = np.arange(n) < n // 2
        val = np.zeros(n, dtype=bool)
        val[n // 2:n // 2 + n // 8] = True
        test = np.arange(n) >= n // 2 + n // 8
        return Xv.astype(np.float32), Xs.astype(np.float32), \
            Xp.astype(np.float32), y, train, val, test

    def test_early_stopping_restores_best(self):
        Xv, Xs, Xp, y, train, val, test = self._data()
        cfg = dict(TORCH_CFG, max_epochs=20, patience=3)
        out = fit_torch_model(Xv, Xs, Xp, y, train, val, test, 42, cfg)
        self.assertEqual(out["proba_test"].shape, (test.sum(), 2))
        self.assertTrue(np.isfinite(out["proba_test"]).all())
        self.assertLessEqual(out["n_epochs"], 20)
        self.assertGreaterEqual(out["n_epochs"], 1)
        self.assertLessEqual(out["best_epoch"], out["n_epochs"])

    def test_patience_triggers_on_noise(self):
        Xv, Xs, Xp, y, train, val, test = self._data(separable=False)
        cfg = dict(TORCH_CFG, max_epochs=30, patience=3)
        out = fit_torch_model(Xv, Xs, Xp, y, train, val, test, 42, cfg)
        self.assertLess(out["n_epochs"], 30)

    def test_determinism(self):
        Xv, Xs, Xp, y, train, val, test = self._data()
        cfg = dict(TORCH_CFG, max_epochs=5, patience=10)
        a = fit_torch_model(Xv, Xs, Xp, y, train, val, test, 42, cfg)
        b = fit_torch_model(Xv, Xs, Xp, y, train, val, test, 42, cfg)
        np.testing.assert_allclose(a["proba_test"], b["proba_test"],
                                   atol=1e-6)

    def test_different_seeds_differ(self):
        Xv, Xs, Xp, y, train, val, test = self._data(separable=False)
        cfg = dict(TORCH_CFG, max_epochs=5, patience=10)
        a = fit_torch_model(Xv, Xs, Xp, y, train, val, test, 42, cfg)
        b = fit_torch_model(Xv, Xs, Xp, y, train, val, test, 2026, cfg)
        self.assertFalse(np.allclose(a["proba_test"], b["proba_test"],
                                     atol=1e-6))


if __name__ == "__main__":
    unittest.main()
