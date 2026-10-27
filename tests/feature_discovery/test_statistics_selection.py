import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from begad_norm.feature_discovery.multiplicity import apply_pairwise, wide_table
from begad_norm.feature_discovery.ranking import pair_specific_reserves, rank_regions, region_evidence
from begad_norm.feature_discovery.statistics import (
    contrast_rows, map_group, mwu_matrix, mwu_primary,
)


def test_delta_orientation_ties_and_reversal():
    _, _, d = mwu_primary([1, 2, 3], [4, 5, 6])
    assert d == 1.0                                   # second group larger -> positive
    _, _, d = mwu_primary([4, 5, 6], [1, 2, 3])
    assert d == -1.0
    _, _, d = mwu_primary([1, 2, 2], [2, 2, 3])
    # P(x2>x1)-P(x2<x1): pairs (9): greater 5 (2>1,2>1,3>1,3>2,3>2), less 0, ties 4
    assert d == pytest.approx(5 / 9)


def test_vectorised_mwu_equals_scipy_with_ties_and_missing():
    rng = np.random.default_rng(3)
    X1 = rng.integers(0, 4, (50, 14)).astype(float)
    X2 = rng.integers(0, 4, (50, 9)).astype(float)
    X1[rng.random(X1.shape) < .2] = np.nan
    r = mwu_matrix(X1, X2)
    for i in range(50):
        a, b = X1[i][np.isfinite(X1[i])], X2[i][np.isfinite(X2[i])]
        p = stats.mannwhitneyu(b, a, alternative="two-sided", method="asymptotic", use_continuity=True).pvalue
        assert r["p"][i] == pytest.approx(p, abs=1e-12)


def test_unknown_diagnosis_never_becomes_cn():
    assert map_group(0) == "CN"
    with pytest.raises(ValueError):
        map_group(3)
    with pytest.raises(ValueError):
        map_group(None)


def _mat(values: dict, groups):
    mat = pd.DataFrame(values).T
    mat.index = pd.MultiIndex.from_tuples(mat.index, names=["ctx", "feature_id"])
    mat.columns = [f"s{i}" for i in range(len(groups))]
    return mat, pd.Series(groups, index=mat.columns)


def test_eligibility_uses_expected_denominator_constant_rules():
    groups = ["CN"] * 10 + ["MCI"] * 10 + ["AD"] * 10
    full = np.r_[np.arange(10), np.arange(10) + 5, np.arange(10) + 10].astype(float)
    sparse = full.copy()
    sparse[20:25] = np.nan               # AD: 5 of 10 valid -> coverage 0.5 < 0.6
    const = np.ones(30)
    one_const = np.r_[np.ones(10), np.arange(10.), np.arange(10.)]
    mat, g = _mat({("c", "F1"): full, ("c", "F2"): sparse, ("c", "F3"): const, ("c", "F4"): one_const}, groups)
    st = contrast_rows(mat, g, 8, 0.6, "ctx").set_index(["feature_id", "contrast"])
    assert st.loc[("F1", "CN_AD"), "eligible"]
    assert not st.loc[("F2", "CN_AD"), "eligible"] and st.loc[("F2", "CN_AD"), "coverage2"] == 0.5
    assert st.loc[("F2", "CN_AD"), "expected_n2"] == 10   # failed recordings stay in the denominator
    assert st.loc[("F3", "CN_AD"), "status"] == "constant_not_informative"
    assert st.loc[("F4", "CN_MCI"), "eligible"]           # one constant group is still testable


def _wide_from_p(rows):
    """rows: list of (ctx, feature, p_ad, p_mci, p_mciad, d_ad, d_mci)."""
    recs = []
    for ctx, f, pa, pm, pma, da, dm in rows:
        for c, p, d in (("CN_AD", pa, da), ("CN_MCI", pm, dm), ("MCI_AD", pma, 0.1)):
            recs.append(dict(scope_or_region_key=ctx, feature_id=f, contrast=c, eligible=p is not None,
                             status="tested", exclusion_reason="", p_primary=p if p is not None else np.nan,
                             cliffs_delta=d, n1=20, n2=12, coverage1=1.0, coverage2=0.9))
    st = apply_pairwise(pd.DataFrame(recs), "fam")
    return st, wide_table(st, "joint", "s3")


def test_main_and_strict_screens_exact_rules():
    _, w = _wide_from_p([("KD1|v|ROI_000", "A", .10, .30, .5, .3, .3),
                         ("KD1|v|ROI_000", "B", .10, .10, .30, .3, .3),
                         ("KD1|v|ROI_000", "C", .20, .05, .01, .3, .3),
                         ("KD1|v|ROI_000", "D", .05, .05, .05, .3, .3)])
    w = w.set_index("feature_id")
    assert not w.loc["A", "selected_main"]
    assert w.loc["B", "selected_main"] and not w.loc["B", "selected_three_pairs"]
    assert not w.loc["C", "selected_main"]             # exactly 0.20 fails strict <
    assert w.loc["D", "selected_three_pairs"]
    assert w.loc["B", "p_joint"] == 0.10


def test_bh_family_contains_losers_and_expands_with_new_hypotheses():
    rows = [("r", f"F{i}", p, p, p, .1, .1) for i, p in enumerate([.001, .04, .3, .6, .9])]
    st, w = _wide_from_p(rows)
    el = st[st.eligible]
    expect = multipletests(el.p_primary, method="fdr_bh")[1]
    np.testing.assert_allclose(el.q_bh.to_numpy(), expect)
    assert (st.pairwise_family_size == 15).all()
    st2, w2 = _wide_from_p(rows + [("r", "D001", .5, .5, .5, .1, .1)])
    assert (st2.pairwise_family_size == 18).all()
    assert w2.set_index("feature_id").loc["F0", "q_joint_bh"] > w.set_index("feature_id").loc["F0", "q_joint_bh"]


def test_region_penalty_reserves_and_topk():
    _, w = _wide_from_p([
        ("KD1|v|ROI_000", "A", .01, .01, .5, .4, .4),                                   # m_r = 1
        ("KD1|v|ROI_001", "A", .01, .01, .5, .4, .4), ("KD1|v|ROI_001", "B", .9, .9, .9, 0, 0),  # m_r = 2
        ("KD2|v|ROI_000", "A", .05, .5, .5, .4, .1), ("KD2|v|ROI_000", "B", .5, .05, .5, .1, .4),  # complementary
    ])
    w["min_group_coverage"] = 1.0
    regions = sorted(w.scope_or_region_key.unique())
    ev = region_evidence(w, regions).set_index("region_key")
    assert ev.loc["KD1|v|ROI_000", "p_region"] == pytest.approx(0.01)
    assert ev.loc["KD1|v|ROI_001", "p_region"] == pytest.approx(0.02)
    assert not ev.loc["KD2|v|ROI_000", "main_candidate"]
    res = pair_specific_reserves(w, ev.reset_index())
    assert res.set_index("region_key").loc["KD2|v|ROI_000", "reserve_type"].startswith("complementary")
    rk = rank_regions(ev.reset_index(), 20)
    assert rk.base_rank.notna().sum() == 2 and rk.in_top_k.sum() == 2


def test_topk_counts_unique_and_deterministic():
    rows = []
    for t in ("KD1", "KD2"):
        for r in range(15):
            rows.append((f"{t}|v|ROI_{r:03d}", "A", .01 + r / 1000, .01, .5, .3, .3))
    _, w = _wide_from_p(rows)
    w["min_group_coverage"] = 1.0
    ev = region_evidence(w, sorted(w.scope_or_region_key.unique()))
    a = rank_regions(ev, 20)
    b = rank_regions(ev.sample(frac=1, random_state=1), 20)
    assert a.in_top_k.sum() == 20
    assert set(a.loc[a.in_top_k, "region_key"]) == set(b.loc[b.in_top_k, "region_key"])
    assert a.loc[a.in_top_k, "region_key"].is_unique
    assert {"KD1|v|ROI_000", "KD2|v|ROI_000"} <= set(a.loc[a.in_top_k, "region_key"])  # same ROI ID, distinct
    none = rank_regions(ev.assign(main_candidate=False), 20)
    assert none.in_top_k.sum() == 0
    seven = rank_regions(ev.assign(main_candidate=ev.index < 7), 20)
    assert seven.in_top_k.sum() == 7
