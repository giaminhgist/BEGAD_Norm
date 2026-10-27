import numpy as np
import pandas as pd
import pytest

from begad_norm.feature_discovery import stability as stb
from begad_norm.feature_discovery.multiplicity import apply_pairwise, wide_table
from begad_norm.feature_discovery.statistics import contrast_rows


def _setup(effect=True, n=(20, 12, 10), seed=0, shuffle=False):
    rng = np.random.default_rng(seed)
    groups = np.array(["CN"] * n[0] + ["MCI"] * n[1] + ["AD"] * n[2])
    if shuffle:
        groups = rng.permutation(groups)
    true = np.array(["CN"] * n[0] + ["MCI"] * n[1] + ["AD"] * n[2])
    base = rng.normal(0, 1, len(true))
    strong = base + np.where(true == "CN", 0, 3.0) if effect else base
    vals = {("KD1|v|ROI_000", "F1"): strong, ("KD1|v|ROI_000", "F2"): rng.normal(0, 1, len(true))}
    mat = pd.DataFrame(vals).T
    mat.index = pd.MultiIndex.from_tuples(mat.index)
    mat.columns = [f"s{i}" for i in range(len(true))]
    g = pd.Series(groups, index=mat.columns)
    st = apply_pairwise(contrast_rows(mat, g, 8, 0.6, "ctx"), "fam")
    return mat, g, wide_table(st, "j", "s3")


def test_bootstrap_draws_whole_people_and_keeps_duplicates():
    g = np.array(["CN"] * 5 + ["MCI"] * 4 + ["AD"] * 3)
    idx = stb._draw(g, np.random.default_rng(0))
    assert all((g[idx[k]] == k).all() for k in idx)
    assert len(idx["CN"]) == 5 and len(np.unique(idx["CN"])) <= 5


def test_computability_does_not_reapply_unique_n_gate():
    A = np.array([[1., 2, 3, 4, 5, 6, 7, 8]])
    p, d, cov = stb._contrast_on_draw(A, np.array([0, 0, 1]), np.array([4, 5, 5]), 0.6)
    assert np.isfinite(p[0])                                    # 2 distinct per group is enough
    p, d, cov = stb._contrast_on_draw(A, np.array([0, 0, 0]), np.array([4, 5, 5]), 0.6)
    assert np.isnan(p[0])                                       # one distinct source participant


def test_discovery_bootstrap_reevaluates_all_and_is_seeded():
    mat, g, w = _setup()
    fs1, _ = stb.discovery_bootstrap(mat, g, w, 60, 7, 0.6, "t")
    fs2, _ = stb.discovery_bootstrap(mat, g, w, 60, 7, 0.6, "t")
    pd.testing.assert_frame_equal(fs1, fs2)
    assert set(fs1.feature_id) == {"F1", "F2"}                  # non-winners are re-evaluated too
    assert fs1.set_index("feature_id").loc["F1", "selection_freq_main"] > 0.9


def test_shuffled_labels_do_not_preserve_constructed_effect():
    mat, g, w = _setup(shuffle=True, seed=4)
    fs, _ = stb.discovery_bootstrap(mat, g, w, 60, 7, 0.6, "t")
    mat0, g0, w0 = _setup(seed=4)
    fs0, _ = stb.discovery_bootstrap(mat0, g0, w0, 60, 7, 0.6, "t")
    assert fs.set_index("feature_id").loc["F1", "selection_freq_main"] < fs0.set_index("feature_id").loc["F1", "selection_freq_main"]


def test_permutation_nonzero_group_sizes_and_exact_small_case():
    rng = np.random.default_rng(0)
    p, how = stb.permutation_p(np.arange(10.), np.arange(10.) + 100, 999, rng)
    assert how == "monte_carlo_plus_one" and p == pytest.approx(1 / 1000)
    p, how = stb.permutation_p([1., 2.], [3., 4., 5.], 999, rng)
    assert how == "exact_enumeration" and p == pytest.approx(0.2)   # 2 of 10 splits as extreme


def test_leave_one_ad_reports_ranges_and_screen_preservation():
    mat, g, w = _setup()
    loo = stb.leave_one_ad(mat, g, w[w.feature_id == "F1"], 8, 0.6)
    assert loo.n_omissions.iloc[0] == 10
    assert loo.prop_main_screen_preserved.iloc[0] == 1.0
    assert loo.sign_flips_CN_AD.iloc[0] == 0
