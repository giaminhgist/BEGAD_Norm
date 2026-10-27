"""Participant-level resampling: effect CIs, discovery stability, AD influence, permutation.

All resampling draws whole participants within diagnostic groups; every record
of a drawn participant comes along (columns of the hypothesis matrix).
Bootstrap computability: >=2 distinct source participants and >=2 valid draws
per compared group, and drawn valid coverage >= min_cov. The original
8-unique-participant gate is NOT re-applied inside replicates.
"""

from __future__ import annotations

import hashlib
import itertools
from math import comb

import numpy as np
import pandas as pd

from begad_norm.feature_discovery.ranking import rank_regions, region_evidence
from begad_norm.feature_discovery.statistics import CONTRASTS, mwu_matrix

SCREEN_P = 0.20


def rng_for(seed: int, name: str) -> np.random.Generator:
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return np.random.default_rng([seed, h])


def _draw(groups: np.ndarray, rng) -> dict:
    return {g: rng.choice(np.flatnonzero(groups == g), size=int((groups == g).sum()), replace=True)
            for g in ("CN", "MCI", "AD")}


def _contrast_on_draw(A: np.ndarray, idx1, idx2, min_cov):
    X1, X2 = A[:, idx1], A[:, idx2]
    r = mwu_matrix(X1, X2)
    v1, v2 = np.isfinite(X1), np.isfinite(X2)
    valid_src = np.isfinite(A)
    d1 = np.zeros(A.shape[1], bool)
    d1[np.unique(idx1)] = True
    d2 = np.zeros(A.shape[1], bool)
    d2[np.unique(idx2)] = True
    dist1 = (valid_src & d1).sum(1)
    dist2 = (valid_src & d2).sum(1)
    ok = ((v1.sum(1) >= 2) & (v2.sum(1) >= 2) & (dist1 >= 2) & (dist2 >= 2)
          & (v1.mean(1) >= min_cov) & (v2.mean(1) >= min_cov))
    p = np.where(ok, r["p"], np.nan)
    d = np.where(ok, r["delta"], np.nan)
    return p, d, np.minimum(v1.mean(1), v2.mean(1))


def bootstrap_ci(mat: pd.DataFrame, groups: pd.Series, hyps: pd.DataFrame, B: int, seed: int, min_cov: float,
                 name: str):
    """Percentile CIs of Cliff's delta for (hypothesis, contrast) rows in hyps (eligible rows only)."""
    rng = rng_for(seed, "ci:" + name)
    g = groups.reindex(mat.columns).to_numpy()
    if hyps.empty:
        return pd.DataFrame(columns=["scope_or_region_key", "feature_id", "contrast", "ci_low", "ci_high",
                                     "ci_valid_reps", "ci_status"])
    out = []
    keys = list(zip(hyps["scope_or_region_key"], hyps["feature_id"]))
    A = mat.loc[keys].to_numpy(float)
    deltas = {c: np.full((B, len(keys)), np.nan) for c in CONTRASTS}
    for b in range(B):
        idx = _draw(g, rng)
        for c, (g1, g2) in CONTRASTS.items():
            _, d, _ = _contrast_on_draw(A, idx[g1], idx[g2], min_cov)
            deltas[c][b] = d
    for c in CONTRASTS:
        el = hyps[f"eligible_{c}"].to_numpy(bool)
        D = deltas[c]
        nv = np.isfinite(D).sum(0)
        with np.errstate(all="ignore"):
            lo = np.nanpercentile(np.where(np.isfinite(D), D, np.nan), 2.5, axis=0)
            hi = np.nanpercentile(np.where(np.isfinite(D), D, np.nan), 97.5, axis=0)
        for i, (ctx, fid) in enumerate(keys):
            if not el[i]:
                status = "ci_not_computed_ineligible"
                lo_i = hi_i = np.nan
            else:
                lo_i, hi_i = lo[i], hi[i]
                status = "ok" if nv[i] >= 0.9 * B else "bootstrap_unreliable"
            out.append(dict(scope_or_region_key=ctx, feature_id=fid, contrast=c, ci_low=lo_i, ci_high=hi_i,
                            ci_valid_reps=int(nv[i]), ci_status=status))
    return pd.DataFrame(out)


def discovery_bootstrap(mat: pd.DataFrame, groups: pd.Series, w: pd.DataFrame, B: int, seed: int, min_cov: float,
                        name: str, regions: list[str] | None = None, top_k: int = 20):
    """Re-run the fixed eligible search in each replicate.

    Returns (feature_stability, region_stability).
    """
    rng = rng_for(seed, "discovery:" + name)
    g = groups.reindex(mat.columns).to_numpy()
    univ = w[w["eligible_CN_AD"] | w["eligible_CN_MCI"] | w["eligible_MCI_AD"]].reset_index(drop=True)
    keys = list(zip(univ["scope_or_region_key"], univ["feature_id"]))
    A = mat.loc[keys].to_numpy(float) if keys else np.zeros((0, mat.shape[1]))
    el = {c: univ[f"eligible_{c}"].to_numpy(bool) for c in CONTRASTS}
    sign0 = {c: np.sign(univ[f"delta_{c}"].to_numpy(float)) for c in CONTRASTS}
    sel_main = np.zeros(len(univ))
    sel_three = np.zeros(len(univ))
    sel_pair = {c: np.zeros(len(univ)) for c in CONTRASTS}
    agree = np.zeros(len(univ))
    agree_den = np.zeros(len(univ))
    computable_main = np.zeros(len(univ))
    m_r = None
    top_counts = {}
    if regions is not None:
        base = univ[univ["eligible_main"]]
        m_r = base.groupby("scope_or_region_key").size().to_dict()
        top_counts = {r: 0 for r in regions if m_r.get(r, 0) > 0}
    for b in range(B):
        idx = _draw(g, rng)
        P, D, C = {}, {}, {}
        for c, (g1, g2) in CONTRASTS.items():
            p = np.full(len(univ), np.nan)
            d = np.full(len(univ), np.nan)
            cv = np.full(len(univ), np.nan)
            if el[c].any():
                pp, dd, cc = _contrast_on_draw(A[el[c]], idx[g1], idx[g2], min_cov)
                p[el[c]], d[el[c]], cv[el[c]] = pp, dd, cc
            P[c], D[c], C[c] = p, d, cv
            sel_pair[c] += el[c] & (np.nan_to_num(p, nan=1.0) < SCREEN_P)
        main = el["CN_AD"] & el["CN_MCI"] & (np.nan_to_num(P["CN_AD"], nan=1) < SCREEN_P) & (
            np.nan_to_num(P["CN_MCI"], nan=1) < SCREEN_P)
        sel_main += main
        sel_three += main & el["MCI_AD"] & (np.nan_to_num(P["MCI_AD"], nan=1) < SCREEN_P)
        defined = el["CN_AD"] & el["CN_MCI"] & np.isfinite(D["CN_AD"]) & np.isfinite(D["CN_MCI"])
        computable_main += defined
        ag = defined & (np.sign(D["CN_AD"]) == sign0["CN_AD"]) & (np.sign(D["CN_MCI"]) == sign0["CN_MCI"]) & (
            sign0["CN_AD"] != 0) & (sign0["CN_MCI"] != 0)
        agree += ag
        agree_den += defined
        if regions is not None and top_counts:
            rw = pd.DataFrame({
                "scope_or_region_key": univ["scope_or_region_key"], "feature_id": univ["feature_id"],
                "eligible_main": univ["eligible_main"].to_numpy(bool),
                "p_joint": np.where(univ["eligible_main"], np.nan_to_num(np.maximum(P["CN_AD"], P["CN_MCI"]),
                                                                           nan=1.0), np.nan),
                "selected_main": main,
                "min_abs_control_delta": np.minimum(np.abs(D["CN_AD"]), np.abs(D["CN_MCI"])),
                "min_group_coverage": np.fmin(C["CN_AD"], C["CN_MCI"]),
            })
            rw = rw[rw["eligible_main"]]
            ev = region_evidence(rw, list(top_counts), m_r=m_r)
            ranked = rank_regions(ev, top_k)
            for r in ranked.loc[ranked["in_top_k"], "region_key"]:
                top_counts[r] += 1
    fs = univ[["scope_or_region_key", "feature_id"]].copy()
    fs["bootstrap_replicates"] = B
    fs["selection_freq_main"] = sel_main / B
    fs["selection_freq_three_pairs"] = sel_three / B
    for c in CONTRASTS:
        fs[f"selection_freq_{c}"] = sel_pair[c] / B
    fs["direction_agreement_both_controls"] = np.where(agree_den > 0, agree / np.maximum(agree_den, 1), np.nan)
    fs["direction_defined_reps"] = agree_den.astype(int)
    fs["direction_stable"] = fs["direction_agreement_both_controls"] >= 0.80
    fs["selection_stable"] = fs["selection_freq_main"] >= 0.60
    rs = pd.DataFrame({"region_key": list(top_counts), "top_k_frequency": [v / B for v in top_counts.values()],
                       "bootstrap_replicates": B})
    return fs, rs


def leave_one_ad(mat: pd.DataFrame, groups: pd.Series, hyps: pd.DataFrame, min_n: int, min_cov: float):
    """Omit each AD participant in turn (from roster and denominator); recompute CN_AD and MCI_AD."""
    g = groups.reindex(mat.columns).to_numpy()
    ad_idx = np.flatnonzero(g == "AD")
    keys = list(zip(hyps["scope_or_region_key"], hyps["feature_id"]))
    if not keys:
        return pd.DataFrame()
    A = mat.loc[keys].to_numpy(float)
    res = {c: {"p": [], "d": [], "ok": []} for c in ("CN_AD", "MCI_AD")}
    for j in ad_idx:
        keep_ad = ad_idx[ad_idx != j]
        for c, g1 in (("CN_AD", "CN"), ("MCI_AD", "MCI")):
            i1 = np.flatnonzero(g == g1)
            X1, X2 = A[:, i1], A[:, keep_ad]
            r = mwu_matrix(X1, X2)
            n1v, n2v = np.isfinite(X1).sum(1), np.isfinite(X2).sum(1)
            ok = (n1v >= min_n) & (n2v >= min_n) & (n1v / len(i1) >= min_cov) & (n2v / len(keep_ad) >= min_cov)
            res[c]["p"].append(r["p"])
            res[c]["d"].append(r["delta"])
            res[c]["ok"].append(ok)
    out = hyps[["scope_or_region_key", "feature_id"]].copy().reset_index(drop=True)
    out["n_omissions"] = len(ad_idx)
    p_mci = hyps["p_CN_MCI"].to_numpy(float)
    for c in ("CN_AD", "MCI_AD"):
        P = np.array(res[c]["p"])
        D = np.array(res[c]["d"])
        OK = np.array(res[c]["ok"])
        d0 = np.sign(hyps[f"delta_{c}"].to_numpy(float))
        out[f"delta_{c}_min"] = np.nanmin(D, 0)
        out[f"delta_{c}_max"] = np.nanmax(D, 0)
        out[f"p_{c}_min"] = np.nanmin(P, 0)
        out[f"p_{c}_max"] = np.nanmax(P, 0)
        out[f"sign_flips_{c}"] = (np.sign(D) != d0[None, :]).sum(0)
        out[f"ineligible_omissions_{c}"] = (~OK).sum(0)
        if c == "CN_AD":
            keep = OK & (P < SCREEN_P) & (p_mci[None, :] < SCREEN_P)
            out["prop_main_screen_preserved"] = keep.mean(0)
    return out


def permutation_p(x1, x2, B: int, rng) -> tuple[float, str]:
    """Two-sided label permutation p for the MWU U statistic (|U - mu| >= observed)."""
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    z = np.concatenate([x1, x2])
    n1, N = len(x1), len(z)
    from scipy.stats import rankdata
    rk = rankdata(z)
    mu = n1 * (N - n1) / 2.0
    u_obs = rk[:n1].sum() - n1 * (n1 + 1) / 2.0
    dev_obs = abs(u_obs - mu)
    if comb(N, n1) <= B:
        cnt = tot = 0
        for c in itertools.combinations(range(N), n1):
            u = rk[list(c)].sum() - n1 * (n1 + 1) / 2.0
            cnt += abs(u - mu) >= dev_obs - 1e-9
            tot += 1
        return cnt / tot, "exact_enumeration"
    perms = np.argsort(rng.random((B, N)), axis=1)[:, :n1]
    u = rk[perms].sum(1) - n1 * (n1 + 1) / 2.0
    cnt = int((np.abs(u - mu) >= dev_obs - 1e-9).sum())
    return (cnt + 1) / (B + 1), "monte_carlo_plus_one"


def permutation_validation(mat, groups, hyps, B, seed, name):
    g = groups.reindex(mat.columns).to_numpy()
    rng = rng_for(seed, "perm:" + name)
    recs = []
    for _, h in hyps.iterrows():
        x = mat.loc[(h["scope_or_region_key"], h["feature_id"])].to_numpy(float)
        for c, (g1, g2) in CONTRASTS.items():
            if not h[f"eligible_{c}"]:
                continue
            x1 = x[g == g1]
            x2 = x[g == g2]
            x1, x2 = x1[np.isfinite(x1)], x2[np.isfinite(x2)]
            p, method = permutation_p(x1, x2, B, rng)
            pp = h[f"p_{c}"]
            recs.append(dict(scope_or_region_key=h["scope_or_region_key"], feature_id=h["feature_id"], contrast=c,
                             p_primary=pp, p_permutation=p, permutation_method=method, n_permutations=B,
                             disagree_at_0_20=(pp < SCREEN_P) != (p < SCREEN_P)))
    return pd.DataFrame(recs)
