"""Tests for begad_norm.rowlevel.run_rowlevel — row-level classification
runner (split seed 42, 5 model seeds, grouped + random splits, ML + torch).

Contract (approved plan 2026-10-17):
  - data: T1235_clean.xlsx / T5_clean.xlsx, guards on columns/group/dups;
  - problems: co_vs_mci (pos MCI), co_vs_ad (pos AD), co_mci_ad (3-class);
  - splits: grouped = participant-level stratified_folds(seed=42), eval at
    row + subject level; random = row-level StratifiedKFold(10, shuffle,
    random_state=42), eval at row level only;
  - 5 model seeds [42, 2026, 1, 0, 7] applied to RF/LGBM/FNN/torch only;
  - fold info written to folds/fold_info.csv (role=test only).
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from begad_norm.baseline.aggregate_results import aggregate
from begad_norm.baseline.run_baselines import LABEL_MAP
from begad_norm.config import PROJECT_ROOT
from begad_norm.rowlevel.run_rowlevel import (
    DETAIL_COLS, TORCH_MODEL_NAME, _append_rows, _fit_predict,
    aggregate_subject, build_problem_rows, grouped_fold_masks, load_dataset,
    random_fold_masks, write_fold_info)

METRICS = ["acc", "sen", "precision", "spec", "auc", "f1"]


def _frame(n_pats=12, rows_per_pat=3):
    """Synthetic row-level frame: groups 0/1/2, tasks, ROIs, one feature
    column per branch plus extras."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_pats):
        pat = f"BEGP{i:03d}"
        g = i % 3
        for r in range(rows_per_pat):
            rows.append({
                "group": g, "pat": pat, "task": 1 + (i + r) % 2,
                "ROI": r, "nFixations": rng.uniform(1, 10),
                "totalFixTime": rng.uniform(0, 5),
                "startPupL": rng.uniform(2, 6),
                "speechDelay": rng.uniform(0, 3),
                "endSpeechDelay": rng.uniform(-1, 4)})
    return pd.DataFrame(rows)


CFG = {
    "data": {
        "dataset_name": "synth",
        "columns": ["group", "pat", "task", "ROI", "nFixations",
                    "totalFixTime", "startPupL", "speechDelay",
                    "endSpeechDelay"],
        "n_patients": 12,
        "feature_columns": ["nFixations", "totalFixTime", "startPupL",
                            "speechDelay", "endSpeechDelay"],
        "roi_column": "ROI",
    },
    "problems": [
        {"name": "co_vs_mci", "include_groups": ["CO", "MCI"],
         "positive": "MCI"},
        {"name": "co_mci_ad", "include_groups": ["CO", "MCI", "AD"],
         "positive": None},
    ],
    "cv": {"n_folds": 3, "split_seed": 42, "model_seeds": [42, 2026]},
    "output": {"folds_dir": "folds"},
}


class TestLoadDataset(unittest.TestCase):
    def _write(self, df):
        tmp = Path(tempfile.mkdtemp()) / "synth.xlsx"
        df.to_excel(tmp, index=False)
        return tmp

    def test_ok(self):
        df = _frame()
        out = load_dataset(self._write(df), CFG["data"])
        self.assertEqual(len(out), len(df))
        self.assertEqual(out["pat"].nunique(), 12)
        self.assertEqual(set(out["group_label"]), {"CO", "MCI", "AD"})

    def test_wrong_columns(self):
        df = _frame().rename(columns={"nFixations": "X"})
        with self.assertRaises(AssertionError):
            load_dataset(self._write(df), CFG["data"])

    def test_group_outside(self):
        df = _frame()
        df.loc[0, "group"] = 7
        with self.assertRaises(AssertionError):
            load_dataset(self._write(df), CFG["data"])

    def test_duplicate_pat_task_roi(self):
        df = _frame()
        df.loc[1, ["pat", "task", "ROI"]] = df.loc[0, ["pat", "task", "ROI"]]
        with self.assertRaises(AssertionError):
            load_dataset(self._write(df), CFG["data"])

    def test_wrong_n_patients(self):
        with self.assertRaises(AssertionError):
            load_dataset(self._write(_frame(10)), CFG["data"])


class TestBuildProblemRows(unittest.TestCase):
    def setUp(self):
        self.df = _frame()
        self.df["group_label"] = self.df["group"].map(
            {0: "CO", 1: "MCI", 2: "AD"})

    def test_binary(self):
        X, y, pat, pid_group, _ = build_problem_rows(
            self.df, CFG["data"]["feature_columns"],
            CFG["problems"][0])
        n = len(self.df) * 2 // 3  # CO + MCI groups only (8 of 12 pats)
        self.assertEqual(X.shape, (n, 5))
        self.assertEqual(set(y), {0, 1})
        self.assertEqual(len(pat), n)
        self.assertEqual(set(pid_group.values()), {"CO", "MCI"})

    def test_three_class(self):
        X, y, pat, pid_group, _ = build_problem_rows(
            self.df, CFG["data"]["feature_columns"],
            CFG["problems"][1])
        self.assertEqual(set(y), {0, 1, 2})
        self.assertEqual(set(pid_group.values()), {"CO", "MCI", "AD"})
        # y matches LABEL_MAP of each participant's group
        for p in np.unique(pat):
            g = pid_group[p]
            self.assertEqual(set(y[pat == p]), {LABEL_MAP[g]})


class TestFoldMasks(unittest.TestCase):
    def setUp(self):
        self.df = _frame(12, 4)
        self.df["group_label"] = self.df["group"].map(
            {0: "CO", 1: "MCI", 2: "AD"})
        _, _, pat, pid_group, _ = build_problem_rows(
            self.df, CFG["data"]["feature_columns"],
            CFG["problems"][1])
        self.pat, self.pid_group = pat, pid_group

    def test_grouped_disjoint_per_group(self):
        for fold in range(1, 4):
            tr, te = grouped_fold_masks(self.pat, self.pid_group, 42, fold, 3)
            tr_pats = set(self.pat[tr])
            te_pats = set(self.pat[te])
            self.assertFalse(tr_pats & te_pats)
            self.assertEqual(tr_pats | te_pats,
                             set(self.pid_group.keys()))
            for g in ("CO", "MCI", "AD"):
                g_pats = {p for p in self.pid_group
                          if self.pid_group[p] == g}
                n_te = len(g_pats & te_pats)
                self.assertIn(n_te, (1, 2))  # round-robin of 4 over 3 folds

    def test_grouped_deterministic(self):
        a = grouped_fold_masks(self.pat, self.pid_group, 42, 2, 3)
        b = grouped_fold_masks(self.pat, self.pid_group, 42, 2, 3)
        np.testing.assert_array_equal(a[0], b[0])

    def test_random_partition_and_determinism(self):
        X, y, _, _, _ = build_problem_rows(
            self.df, CFG["data"]["feature_columns"],
            CFG["problems"][1])
        seen = np.zeros(len(y), dtype=bool)
        for fold in range(1, 4):
            tr, te = random_fold_masks(X, y, 42, fold, 3)
            self.assertFalse((tr & te).any())
            seen |= te
            for c in (0, 1, 2):
                prop_tr = (y[tr] == c).mean()
                self.assertAlmostEqual(prop_tr, 1 / 3, delta=0.1)
        self.assertTrue(seen.all())
        a = random_fold_masks(X, y, 42, 1, 3)
        b = random_fold_masks(X, y, 42, 1, 3)
        np.testing.assert_array_equal(a[0], b[0])


class TestAggregateSubject(unittest.TestCase):
    def test_binary_threshold(self):
        pat = np.array(["a", "a", "b"])
        y = np.array([0, 0, 1])
        proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.3, 0.7]])
        pats, y_sub, proba_sub, pred_sub = aggregate_subject(pat, y, proba)
        self.assertEqual(list(pats), ["a", "b"])
        self.assertEqual(list(y_sub), [0, 1])
        np.testing.assert_allclose(proba_sub, [[0.65, 0.35], [0.3, 0.7]])
        self.assertEqual(list(pred_sub), [0, 1])

    def test_multiclass_argmax(self):
        pat = np.array(["a", "a"])
        y = np.array([0, 0])
        proba = np.array([[0.1, 0.6, 0.3], [0.5, 0.1, 0.4]])
        _, _, proba_sub, pred_sub = aggregate_subject(pat, y, proba)
        np.testing.assert_allclose(proba_sub, [[0.3, 0.35, 0.35]])
        self.assertEqual(list(pred_sub), [1])  # argmax ties -> first index


class TestWriteFoldInfo(unittest.TestCase):
    def test_schema_and_counts(self):
        df = _frame(12, 4)
        df["group_label"] = df["group"].map({0: "CO", 1: "MCI", 2: "AD"})
        tmp = Path(tempfile.mkdtemp())
        masks = write_fold_info(tmp, df, CFG["problems"], CFG)
        csv = tmp / "folds" / "fold_info.csv"
        self.assertTrue(csv.exists())
        fi = pd.read_csv(csv)
        self.assertEqual(list(fi.columns),
                         ["split", "seed", "fold", "problem", "unit",
                          "unit_id", "group", "role"])
        self.assertEqual(set(fi["role"]), {"test"})
        gr = fi[fi.split == "grouped"]
        self.assertEqual(len(gr), 12)  # each pat held out in exactly 1 fold
        self.assertEqual(set(gr["problem"]), {"all"})
        self.assertEqual(set(gr["seed"]), {42})
        # round-robin of 4 pats per group over 3 folds -> sizes 2,1,1 per
        # group -> fold sizes 6,3,3 overall
        sizes = sorted(gr.groupby("fold")["unit_id"].nunique())
        self.assertEqual(sizes, [3, 3, 6])
        rnd = fi[fi.split == "random"]
        self.assertEqual(set(rnd["problem"]), {"co_vs_mci", "co_mci_ad"})
        self.assertEqual(set(rnd["unit"]), {"row"})
        self.assertEqual(set(rnd["seed"]), {42})
        # masks keys exist for every (split, problem, fold)
        for split in ("grouped", "random"):
            for p in ("co_vs_mci", "co_mci_ad"):
                for fold in range(1, 4):
                    tr, te = masks[(split, p, fold)]
                    self.assertEqual(tr.dtype, bool)
                    self.assertFalse((tr & te).any())


class TestFitPredictTiny(unittest.TestCase):
    """Regression: X_torch must be sized to the problem rows, not the full
    dataset (IndexError 'boolean index did not match' otherwise)."""

    def test_torch_problem_sized_inputs(self):
        df = _frame(12, 4)
        df["group_label"] = df["group"].map({0: "CO", 1: "MCI", 2: "AD"})
        cfg = {
            "data": {"dataset_name": "synth", "class_weight": "balanced",
                     "branch_features": {
                         "visual": ["nFixations"],
                         "speech": ["totalFixTime", "speechDelay",
                                    "endSpeechDelay"],
                         "pupil": ["startPupL"]},
                     "roi_column": "ROI"},
            "models": {},
            "cv": {"torch": {"max_epochs": 2, "patience": 5,
                             "batch_size": 32, "lr": 1e-3, "l2": 1e-3,
                             "dropout": 0.2, "val_fraction": 0.2}},
        }
        pcfg = CFG["problems"][0]
        X, y, pat, pid_group, idx = build_problem_rows(
            df, CFG["data"]["feature_columns"], pcfg)
        torch_cols = ["nFixations", "totalFixTime", "speechDelay",
                      "endSpeechDelay", "startPupL", "ROI"]
        X_torch = df.loc[idx, torch_cols].to_numpy(dtype=float)
        train = np.ones(len(y), dtype=bool)
        train[:len(y) // 3] = False
        rows, fit_s = _fit_predict(cfg, X, y, pat, pid_group, pcfg,
                                   TORCH_MODEL_NAME, 42, 1, "grouped",
                                   train, ~train, X_torch, torch_cols)
        self.assertEqual(len(rows), 2)  # row + subject for grouped
        self.assertGreater(fit_s, 0)
        for r in rows:
            self.assertTrue(set(r) <= set(DETAIL_COLS))
            self.assertTrue(np.isfinite([r[m] for m in METRICS]).all())


class TestAppendRows(unittest.TestCase):
    def test_header_once_and_rows_appended(self):
        tmp = Path(tempfile.mkdtemp()) / "detail.csv"
        _append_rows(tmp, ["a", "b"], [{"a": 1, "b": 2}])
        _append_rows(tmp, ["a", "b"], [{"a": 3, "b": 4}, {"a": 5, "b": 6}])
        out = pd.read_csv(tmp)
        self.assertEqual(list(out.columns), ["a", "b"])
        self.assertEqual(out["a"].tolist(), [1, 3, 5])


class TestAggregate(unittest.TestCase):
    def _detail(self):
        n = 12
        return pd.DataFrame({
            "dataset": ["a"] * n, "problem": ["p"] * n,
            "model": ["m1"] * 6 + ["m2"] * 6,
            "split": ["s"] * n, "eval_level": ["l"] * n,
            **{m: list(range(n)) for m in
               ["acc", "sen", "precision", "spec", "auc", "f1"]},
        })

    def test_group_cols(self):
        out = aggregate(self._detail(),
                        ["dataset", "problem", "model", "split",
                         "eval_level"], 6)
        self.assertEqual(len(out), 2 * 6)  # 2 models x 6 metrics
        g = out[out.model == "m1"].set_index("metric")
        self.assertAlmostEqual(g.loc["acc", "mean"], 2.5)
        self.assertAlmostEqual(g.loc["acc", "std"],
                               1.8708286933869707)  # std of 0..5 (ddof=1)

    def test_wrong_count_raises(self):
        with self.assertRaises(ValueError):
            aggregate(self._detail(), ["model"], 5)

    def test_missing_col_raises(self):
        with self.assertRaises(ValueError):
            aggregate(self._detail(), ["nope"], 6)


class TestDetailCols(unittest.TestCase):
    def test_schema(self):
        self.assertEqual(
            DETAIL_COLS,
            ["dataset", "problem", "model", "seed", "fold", "n_train",
             "n_test", "split", "eval_level"] + METRICS)


class TestRealDataIfPresent(unittest.TestCase):
    def test_clean_files(self):
        repo = PROJECT_ROOT.parent
        for name, cols, n_rows in (("T1235", 23, 12486), ("T5", 20, 6217)):
            xlsx = repo / f"{name}_clean.xlsx"
            if not xlsx.exists():
                self.skipTest(f"{name}_clean.xlsx not present")
            data_cfg = {"dataset_name": name,
                        "columns": None, "n_patients": 64}
            df = pd.read_excel(xlsx, engine="openpyxl")
            data_cfg["columns"] = list(df.columns)
            out = load_dataset(xlsx, data_cfg)
            self.assertEqual(len(out), n_rows)
            self.assertEqual(len(out.columns), cols + 1)  # + group_label
            self.assertEqual(out["pat"].nunique(), 64)


if __name__ == "__main__":
    unittest.main()
