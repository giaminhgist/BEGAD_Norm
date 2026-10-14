"""Tests for the normative modeling pipeline (learned CO-normative deviations).

Follows the unittest style of the existing suite (tests/test_rawdata.py, ...).
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from begad_norm.config import PROJECT_ROOT
from begad_norm.normative import seeding

FEATURES_CSV = PROJECT_ROOT / "processed_dataset" / "features.csv"
METADATA_CSV = PROJECT_ROOT / "processed_dataset" / "metadata.csv"


def load_participants():
    return sorted(pd.read_csv(METADATA_CSV)["participant_id"].tolist())


def load_feature_cols():
    cols = pd.read_csv(FEATURES_CSV, nrows=1).columns
    return [c for c in cols if c not in ("participant_id", "task")]


class TestSeeding(unittest.TestCase):
    def test_seed_all_deterministic(self):
        """Same seed -> identical torch weights; also seeds numpy."""
        seeding.seed_all(42)
        w1 = torch.nn.Linear(4, 4).weight.detach().clone()
        np1 = np.random.rand(4)
        seeding.seed_all(42)
        w2 = torch.nn.Linear(4, 4).weight.detach().clone()
        np2 = np.random.rand(4)
        self.assertTrue(torch.equal(w1, w2))
        self.assertTrue(np.allclose(np1, np2))


class TestNormativeData(unittest.TestCase):
    """Data layer: matrices, masks, leakage-safe stats, labels, dataset."""

    def setUp(self):
        from begad_norm.normative import data
        self.data = data
        self.pids = load_participants()
        self.feature_cols = load_feature_cols()
        self.n_tasks = 11

    def test_build_matrices_shapes_and_mask(self):
        mats = self.data.build_matrices(self.pids, FEATURES_CSV,
                                        self.feature_cols, self.n_tasks)
        self.assertEqual(len(mats), 157)
        for pid, (X, mask) in mats.items():
            self.assertEqual(X.shape, (11, 25))
            self.assertEqual(mask.shape, (11,))
            # Mask rule = feature table: row not all-NaN. BEGCO-029 and
            # BEGMCI-019 have an all-NaN task-10 row in features.csv
            # (README §1; metadata n_tasks_present counts the raw recording
            # and differs for BEGCO-029).
            expected = 10 if pid in {"BEGCO-029", "BEGMCI-019"} else 11
            self.assertEqual(int(mask.sum()), expected)

    def test_dataset_no_nan_and_standardized(self):
        ds = self.data.NormativeDataset(
            self.pids, self.pids, FEATURES_CSV, METADATA_CSV, "co_vs_mci")
        self.assertEqual(ds.X.shape, (141, 11, 25))  # 100 CO + 41 MCI
        self.assertFalse(np.isnan(ds.X.numpy()).any())
        # standardized with train stats -> train valid cells have mean ~0
        mu = ds.X.numpy()[ds.mask.numpy().astype(bool)].mean(axis=0)
        self.assertTrue(np.allclose(mu, 0.0, atol=1e-4))

    def test_dataset_stats_use_train_only(self):
        """Adding val participants to the dataset must not change any X."""
        train = self.pids[:110]
        val_a = self.pids[110:135]
        val_b = self.pids[110:150]
        ds_a = self.data.NormativeDataset(
            train + val_a, train, FEATURES_CSV, METADATA_CSV, "co_vs_mci")
        ds_b = self.data.NormativeDataset(
            train + val_b, train, FEATURES_CSV, METADATA_CSV, "co_vs_mci")
        for i in range(len(ds_a.subjects)):
            self.assertTrue(np.allclose(ds_a.X[i], ds_b.X[i]))
            self.assertTrue(np.allclose(ds_a.mask[i], ds_b.mask[i]))

    def test_train_ids_outside_problem_ignored(self):
        """Train ids outside the problem's groups do not change the stats."""
        meta = pd.read_csv(METADATA_CSV).set_index("participant_id")
        ad_pids = [p for p in self.pids if meta.loc[p, "group_label"] == "AD"]
        self.assertTrue(ad_pids)
        ds_a = self.data.NormativeDataset(
            self.pids, self.pids, FEATURES_CSV, METADATA_CSV, "co_vs_mci")
        ds_b = self.data.NormativeDataset(
            self.pids, [p for p in self.pids if p not in set(ad_pids)],
            FEATURES_CSV, METADATA_CSV, "co_vs_mci")
        for i in range(len(ds_a.subjects)):
            self.assertTrue(np.allclose(ds_a.X[i], ds_b.X[i]))

    def test_labels_binary(self):
        meta = pd.read_csv(METADATA_CSV).set_index("participant_id")
        ds = self.data.NormativeDataset(
            self.pids, self.pids, FEATURES_CSV, METADATA_CSV, "co_vs_mci")
        self.assertEqual(set(np.unique(ds.labels)), {0, 1})
        for i, pid in enumerate(ds.subjects):
            expected = 1 if meta.loc[pid, "group_label"] == "MCI" else 0
            self.assertEqual(int(ds.labels[i]), expected)
        ds2 = self.data.NormativeDataset(
            self.pids, self.pids, FEATURES_CSV, METADATA_CSV, "co_vs_ad")
        for i, pid in enumerate(ds2.subjects):
            expected = 1 if meta.loc[pid, "group_label"] == "AD" else 0
            self.assertEqual(int(ds2.labels[i]), expected)

    def test_labels_three_class(self):
        meta = pd.read_csv(METADATA_CSV).set_index("participant_id")
        ds = self.data.NormativeDataset(
            self.pids, self.pids, FEATURES_CSV, METADATA_CSV, "co_mci_ad")
        self.assertEqual(set(np.unique(ds.labels)), {0, 1, 2})
        for i, pid in enumerate(ds.subjects):
            expected = {"CO": 0, "MCI": 1, "AD": 2}[meta.loc[pid, "group_label"]]
            self.assertEqual(int(ds.labels[i]), expected)

    def test_co_indices(self):
        meta = pd.read_csv(METADATA_CSV).set_index("participant_id")
        ds = self.data.NormativeDataset(
            self.pids, self.pids, FEATURES_CSV, METADATA_CSV, "co_mci_ad")
        idx = ds.co_indices()
        self.assertEqual(len(idx), 100)
        for i in idx:
            self.assertEqual(meta.loc[ds.subjects[i], "group_label"], "CO")

    def test_getitem_and_collate(self):
        ds = self.data.NormativeDataset(
            self.pids, self.pids, FEATURES_CSV, METADATA_CSV, "co_vs_mci")
        (X, mask), label, sid = ds[0]
        self.assertEqual(X.shape, (11, 25))
        self.assertEqual(mask.shape, (11,))
        self.assertIn(label, (0, 1))
        (Xb, maskb) = ds.collate([ds[0], ds[1], ds[2]])
        self.assertEqual(Xb.shape, (3, 11, 25))
        self.assertEqual(maskb.shape, (3, 11))


class TestNormativeModel(unittest.TestCase):
    """Model: forward shapes, CO-only latent bank, buffers in state_dict."""

    def setUp(self):
        from begad_norm.normative import model
        self.model = model
        self.ds = None  # built lazily (real data) in the bank test

    def _dummy_input(self, B=4):
        D = torch.randn(B, 11, 25)
        mask = torch.ones(B, 11)
        return (D, mask)

    def test_forward_binary(self):
        # Binary problems use a single sigmoid output (spec §1.4).
        m = self.model.NormativeModel(n_classes=1)
        prob, aux = m(self._dummy_input())
        self.assertEqual(prob.shape, (4, 1))
        self.assertTrue(((prob >= 0) & (prob <= 1)).all())
        self.assertIn("emb", aux)

    def test_forward_three_class(self):
        m = self.model.NormativeModel(n_classes=3)
        prob, aux = m(self._dummy_input())
        self.assertEqual(prob.shape, (4, 3))
        self.assertTrue(np.allclose(prob.detach().numpy().sum(axis=1), 1.0,
                                    atol=1e-5))
        self.assertIn("emb", aux)

    def test_refresh_bank_matches_manual_co_stats(self):
        from begad_norm.normative.data import NormativeDataset
        pids = load_participants()
        ds = NormativeDataset(pids, pids, FEATURES_CSV, METADATA_CSV,
                              "co_vs_mci")
        m = self.model.NormativeModel(n_classes=2)
        m.refresh_bank(ds)
        # manual recomputation over the CO rows only
        with torch.no_grad():
            z = m.encoder(ds.X.view(-1, 25)).view(ds.X.shape[0], 11, 128)
        co = ds.co_indices()
        zc = z[co]
        mc = ds.mask[co][..., None]
        mu = (zc * mc).sum(0) / mc.sum(0).clamp(min=1)
        var = ((zc - mu[None]) ** 2 * mc).sum(0) / mc.sum(0).clamp(min=1)
        sigma = torch.sqrt(var.clamp(min=1e-6)) + 1e-6
        self.assertTrue(np.allclose(mu.numpy(), m.bank_mu.numpy(), atol=1e-5))
        self.assertTrue(np.allclose(sigma.numpy(), m.bank_sigma.numpy(),
                                    atol=1e-5))

    def test_refresh_bank_ignores_non_co_rows(self):
        """Zeroing non-CO inputs must leave the CO-only bank unchanged."""
        from begad_norm.normative.data import NormativeDataset
        pids = load_participants()
        ds = NormativeDataset(pids, pids, FEATURES_CSV, METADATA_CSV,
                              "co_vs_mci")
        m = self.model.NormativeModel(n_classes=2)
        m.refresh_bank(ds)
        mu_before = m.bank_mu.clone()
        sigma_before = m.bank_sigma.clone()
        non_co = [i for i in range(len(ds.subjects)) if i not in ds.co_indices()]
        ds.X[non_co] = 0.0
        m.refresh_bank(ds)
        self.assertTrue(torch.allclose(mu_before, m.bank_mu))
        self.assertTrue(torch.allclose(sigma_before, m.bank_sigma))

    def test_bank_buffers_in_state_dict(self):
        m = self.model.NormativeModel(n_classes=2)
        keys = m.state_dict().keys()
        self.assertIn("bank_mu", keys)
        self.assertIn("bank_sigma", keys)


def _small_datasets(problem="co_vs_mci", train_spec=None, val_spec=None):
    """Tiny real-data train/val dataset pair with both classes present."""
    from begad_norm.normative.data import NormativeDataset
    meta = pd.read_csv(METADATA_CSV).set_index("participant_id")
    pids = load_participants()
    groups = {}
    for g in ("CO", "MCI", "AD"):
        groups[g] = [p for p in pids if meta.loc[p, "group_label"] == g]
    if problem == "co_vs_mci":
        train_pids = groups["CO"][:20] + groups["MCI"][:8]
        val_pids = groups["CO"][20:25] + groups["MCI"][8:10]
    else:  # co_mci_ad
        train_pids = (groups["CO"][:20] + groups["MCI"][:8] + groups["AD"][:4])
        val_pids = (groups["CO"][20:24] + groups["MCI"][8:10] + groups["AD"][4:6])
    if train_spec is not None:  # allow tests to override (e.g. CO-only val)
        train_pids, val_pids = train_spec, val_spec
    ds_tr = NormativeDataset(train_pids, train_pids, FEATURES_CSV,
                             METADATA_CSV, problem)
    ds_va = NormativeDataset(val_pids, train_pids, FEATURES_CSV, METADATA_CSV,
                             problem)
    return ds_tr, ds_va


class TestMetrics(unittest.TestCase):
    """compute_metrics must match the baseline's metric semantics."""

    def test_metrics_binary_matches_sklearn(self):
        from begad_norm.normative.metrics import compute_metrics
        from sklearn.metrics import (accuracy_score, confusion_matrix,
                                     f1_score, precision_score,
                                     recall_score, roc_auc_score)
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 60)
        prob = rng.random(60)
        m = compute_metrics(y, prob, n_classes=1)
        pred = (prob >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        self.assertAlmostEqual(m["acc"], accuracy_score(y, pred))
        self.assertAlmostEqual(m["sen"], recall_score(y, pred, pos_label=1))
        self.assertAlmostEqual(m["precision"],
                               precision_score(y, pred, pos_label=1,
                                               zero_division=0.0))
        self.assertAlmostEqual(m["spec"], tn / (tn + fp))
        self.assertAlmostEqual(m["auc"], roc_auc_score(y, prob))
        self.assertAlmostEqual(m["f1"],
                               f1_score(y, pred, pos_label=1,
                                        zero_division=0.0))

    def test_metrics_multiclass_matches_sklearn(self):
        from begad_norm.normative.metrics import compute_metrics
        from sklearn.metrics import (accuracy_score, f1_score,
                                     precision_score, recall_score,
                                     roc_auc_score)
        rng = np.random.default_rng(1)
        y = rng.integers(0, 3, 60)
        p = rng.random((60, 3))
        p = p / p.sum(1, keepdims=True)
        m = compute_metrics(y, p, n_classes=3)
        pred = p.argmax(1)
        self.assertAlmostEqual(m["acc"], accuracy_score(y, pred))
        self.assertAlmostEqual(m["sen"],
                               recall_score(y, pred, average="macro",
                                            zero_division=0.0))
        self.assertAlmostEqual(m["precision"],
                               precision_score(y, pred, average="macro",
                                               zero_division=0.0))
        self.assertAlmostEqual(m["f1"],
                               f1_score(y, pred, average="macro",
                                        zero_division=0.0))
        self.assertAlmostEqual(m["auc"],
                               roc_auc_score(y, p, multi_class="ovr",
                                             average="macro"))
        # spec = mean per-class TN/(TN+FP), recomputed independently
        cm = np.zeros((3, 3), dtype=int)
        for a, b in zip(y, pred):
            cm[a, b] += 1
        specs = []
        for c in range(3):
            tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
            fp = cm[:, c].sum() - cm[c, c]
            specs.append(tn / (tn + fp))
        self.assertAlmostEqual(m["spec"], float(np.mean(specs)))


class TestTrainer(unittest.TestCase):
    """Seeded training loop: determinism, checkpoint restore, early stop."""

    def setUp(self):
        from begad_norm.normative.trainer import NormativeConfig
        self.trainer_cfg_cls = NormativeConfig
        self.tmp = tempfile.mkdtemp(prefix="normative_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cfg(self, **kw):
        from begad_norm.normative.trainer import NormativeConfig
        base = dict(experiment="test", problem="co_vs_mci",
                    model="normative_mlp_deepset", seed=7, fold=1,
                    epochs=3, patience=5, lr=1e-3, weight_decay=1e-4,
                    batch_size=16, dropout=0.3, n_classes=1,
                    out_root=self.tmp)
        base.update(kw)
        return NormativeConfig(**base)

    def _factory(self, n_classes):
        from begad_norm.normative.model import NormativeModel
        return lambda: NormativeModel(n_classes=n_classes)

    def test_deterministic_same_seed(self):
        from begad_norm.normative.trainer import train_model
        ds_tr, ds_va = _small_datasets()
        cfg = self._cfg(out_root=os.path.join(self.tmp, "a"))
        s1 = train_model(self._factory(1), ds_tr, ds_va, cfg, "cpu")
        cfg = self._cfg(out_root=os.path.join(self.tmp, "b"))
        s2 = train_model(self._factory(1), ds_tr, ds_va, cfg, "cpu")
        self.assertEqual(s1["best_val_auc"], s2["best_val_auc"])
        self.assertEqual(s1["best_epoch"], s2["best_epoch"])
        # per-epoch logs must match except elapsed_s (wall-clock timing)
        import json as _json
        rows1 = [_json.loads(l) for l in
                 (Path(s1["run_dir"]) / "metrics.jsonl").read_text().splitlines()]
        rows2 = [_json.loads(l) for l in
                 (Path(s2["run_dir"]) / "metrics.jsonl").read_text().splitlines()]
        for r in rows1 + rows2:
            r.pop("elapsed_s", None)
        self.assertEqual(rows1, rows2)

    def test_checkpoint_restore_reproduces_val(self):
        from begad_norm.normative.trainer import train_model
        from begad_norm.normative.model import NormativeModel
        ds_tr, ds_va = _small_datasets()
        cfg = self._cfg(epochs=5)
        s = train_model(self._factory(1), ds_tr, ds_va, cfg, "cpu")
        ckpt = torch.load(Path(s["run_dir"]) / "best.pt", weights_only=False)
        m = NormativeModel(n_classes=1)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        prob, _ = m((ds_va.X, ds_va.mask))
        best_probs = np.array(ckpt["val_probs"], dtype=np.float64)
        self.assertTrue(np.allclose(prob.detach().numpy().ravel(), best_probs,
                                    atol=1e-6))
        self.assertEqual(ckpt["val_auc"], s["best_val_auc"])

    def test_early_stopping_mechanism(self):
        """lr=0 -> val AUC constant -> stops after `patience` stale epochs."""
        from begad_norm.normative.trainer import train_model
        ds_tr, ds_va = _small_datasets()
        cfg = self._cfg(lr=0.0, epochs=8, patience=3)
        s = train_model(self._factory(1), ds_tr, ds_va, cfg, "cpu")
        self.assertEqual(s["stopping_reason"], "early_stopping")
        self.assertEqual(s["epochs_trained"], 4)  # epochs 0..3

    def test_single_class_val_raises(self):
        from begad_norm.normative.trainer import train_model
        meta = pd.read_csv(METADATA_CSV).set_index("participant_id")
        pids = load_participants()
        co = [p for p in pids if meta.loc[p, "group_label"] == "CO"]
        mci = [p for p in pids if meta.loc[p, "group_label"] == "MCI"]
        ds_tr, ds_va = _small_datasets(
            train_spec=co[:10] + mci[:8], val_spec=co[10:15])  # CO-only val
        cfg = self._cfg(epochs=2)
        with self.assertRaises(RuntimeError) as ctx:
            train_model(self._factory(1), ds_tr, ds_va, cfg, "cpu")
        self.assertIn("NaN", str(ctx.exception))

    def test_three_class_run(self):
        from begad_norm.normative.trainer import train_model
        ds_tr, ds_va = _small_datasets(problem="co_mci_ad")
        cfg = self._cfg(problem="co_mci_ad", n_classes=3, epochs=3)
        s = train_model(self._factory(3), ds_tr, ds_va, cfg, "cpu")
        self.assertTrue(np.isfinite(s["best_val_metrics"]["auc"]))
        preds = pd.read_csv(Path(s["run_dir"]) / "predictions.csv")
        self.assertEqual(len(preds), len(ds_va.subjects))

    def test_singleton_tail_batch(self):
        """17 train subjects = 16 + 1: the singleton tail batch must not
        crash BatchNorm1d (real folds have 113 = 7*16 + 1 train subjects)."""
        from begad_norm.normative.trainer import train_model
        meta = pd.read_csv(METADATA_CSV).set_index("participant_id")
        pids = load_participants()
        co = [p for p in pids if meta.loc[p, "group_label"] == "CO"]
        mci = [p for p in pids if meta.loc[p, "group_label"] == "MCI"]
        ds_tr, ds_va = _small_datasets(
            train_spec=co[:10] + mci[:7], val_spec=co[10:14] + mci[7:9])
        cfg = self._cfg(epochs=2, batch_size=16)
        s = train_model(self._factory(1), ds_tr, ds_va, cfg, "cpu")
        self.assertEqual(s["stopping_reason"], "max_epochs")


if __name__ == "__main__":
    unittest.main()
