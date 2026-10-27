"""Primary pairwise tests, effect sizes, eligibility and descriptive summaries.

Orientation: for contrast (g1, g2), delta = P(g2 > g1) - P(g2 < g1) =
2*U2/(n1*n2) - 1 where U2 is the Mann-Whitney U statistic with g2 supplied first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

CONTRASTS = {"CN_AD": ("CN", "AD"), "CN_MCI": ("CN", "MCI"), "MCI_AD": ("MCI", "AD")}
GROUP_CODE_TO_LABEL = {0: "CN", 1: "MCI", 2: "AD"}   # repository code 0 = CO/HC, reported as CN


def map_group(code) -> str:
    try:
        return GROUP_CODE_TO_LABEL[int(code)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(f"unknown diagnostic code {code!r}; refusing to map it to CN/MCI/AD")


def mwu_primary(x1, x2):
    """scipy MWU (two-sided, asymptotic, continuity). Returns (U2, p, delta)."""
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    res = stats.mannwhitneyu(x2, x1, alternative="two-sided", method="asymptotic", use_continuity=True)
    u2 = float(res.statistic)
    delta = 2.0 * u2 / (len(x1) * len(x2)) - 1.0
    return u2, float(res.pvalue), delta


def mwu_matrix(X1: np.ndarray, X2: np.ndarray):
    """Row-wise MWU with NaN = missing. X1 (C, n1), X2 (C, n2).

    Returns dict of arrays: n1, n2, u2, p, delta. Reproduces scipy's asymptotic
    two-sided p with tie and continuity correction (tested).
    """
    X1 = np.asarray(X1, float)
    X2 = np.asarray(X2, float)
    v1 = np.isfinite(X1)
    v2 = np.isfinite(X2)
    n1 = v1.sum(1).astype(float)
    n2 = v2.sum(1).astype(float)
    gt = (X2[:, None, :] > X1[:, :, None]).sum((1, 2))
    eq = (X2[:, None, :] == X1[:, :, None]).sum((1, 2))
    u2 = gt + 0.5 * eq
    Z = np.concatenate([X1, X2], axis=1)
    vz = np.isfinite(Z)
    t = (Z[:, :, None] == Z[:, None, :]).sum(2).astype(float)
    tie = np.where(vz, t ** 2 - 1.0, 0.0).sum(1)
    N = n1 + n2
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = n1 * n2 / 2.0
        s = np.sqrt(n1 * n2 / 12.0 * ((N + 1.0) - tie / (N * (N - 1.0))))
        U = np.maximum(u2, n1 * n2 - u2)
        z = (U - mu - 0.5) / s
        p = np.clip(2.0 * stats.norm.sf(z), 0.0, 1.0)
        delta = 2.0 * u2 / (n1 * n2) - 1.0
    bad = (n1 < 1) | (n2 < 1) | ~(s > 0)
    p = np.where(bad, np.nan, p)
    delta = np.where((n1 < 1) | (n2 < 1), np.nan, delta)
    return {"n1": n1, "n2": n2, "u2": u2, "p": p, "delta": delta}


def describe(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(median=np.nan, q1=np.nan, q3=np.nan, mean=np.nan, sd=np.nan)
    return dict(median=float(np.median(x)), q1=float(np.quantile(x, .25)), q3=float(np.quantile(x, .75)),
                mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else np.nan)


def build_matrix(long: pd.DataFrame, roster: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot a long table into hypotheses x subjects (valid values only, NaN otherwise).

    long: subject_id, <key_col>, feature_id, value, measurement_status.
    Returns (matrix, n_events matrix). Raises on duplicate keys.
    """
    if long.duplicated(["subject_id", key_col, "feature_id"]).any():
        raise ValueError("duplicate subject/context/feature keys: refusing to aggregate silently")
    ok = long["measurement_status"].isin(["observed", "observed_zero"]) & np.isfinite(long["value"].astype(float))
    v = long.assign(value=long["value"].astype(float).where(ok))
    mat = v.pivot_table(index=[key_col, "feature_id"], columns="subject_id", values="value",
                        aggfunc="first", dropna=False)
    mat = mat.reindex(columns=list(roster["subject_id"]))
    return mat


def contrast_rows(mat: pd.DataFrame, groups: pd.Series, min_n: int, min_cov: float, key_col: str) -> pd.DataFrame:
    """Eligibility, descriptives, primary test and effect for every hypothesis x contrast."""
    g = groups.reindex(mat.columns)
    if g.isna().any():
        raise ValueError("subjects without diagnosis in the matrix")
    expected = g.value_counts()
    rows = []
    arr = mat.to_numpy(float)
    for i, (ctx, fid) in enumerate(mat.index):
        x = arr[i]
        for cname, (g1, g2) in CONTRASTS.items():
            x1 = x[(g == g1).to_numpy()]
            x2 = x[(g == g2).to_numpy()]
            x1v = x1[np.isfinite(x1)]
            x2v = x2[np.isfinite(x2)]
            e1, e2 = int(expected.get(g1, 0)), int(expected.get(g2, 0))
            cov1 = len(x1v) / e1 if e1 else np.nan
            cov2 = len(x2v) / e2 if e2 else np.nan
            d1, d2 = describe(x1v), describe(x2v)
            row = dict(scope_or_region_key=ctx, feature_id=fid, contrast=cname,
                       n1=len(x1v), n2=len(x2v), expected_n1=e1, expected_n2=e2,
                       coverage1=cov1, coverage2=cov2,
                       **{f"{k}1": v for k, v in d1.items()}, **{f"{k}2": v for k, v in d2.items()},
                       u_statistic=np.nan, p_primary=np.nan, cliffs_delta=np.nan,
                       test_method="mannwhitneyu_two_sided_asymptotic_tie_corrected_continuity")
            row["q1_1"], row["q3_1"] = row.pop("q11"), row.pop("q31")
            row["q1_2"], row["q3_2"] = row.pop("q12"), row.pop("q32")
            reasons = []
            if len(x1v) < min_n or len(x2v) < min_n:
                reasons.append(f"fewer than {min_n} valid participants in a group")
            if (cov1 < min_cov) or (cov2 < min_cov):
                reasons.append(f"group coverage below {min_cov:.2f}")
            pooled = np.concatenate([x1v, x2v])
            if not reasons and np.ptp(pooled) == 0:
                row.update(eligible=False, status="constant_not_informative", p_primary=1.0, cliffs_delta=0.0,
                           exclusion_reason="pooled values constant")
            elif reasons:
                row.update(eligible=False, status="ineligible", exclusion_reason="; ".join(reasons))
            else:
                u2, p, delta = mwu_primary(x1v, x2v)
                row.update(eligible=True, status="tested", exclusion_reason="", u_statistic=u2,
                           p_primary=p, cliffs_delta=delta)
            rows.append(row)
    return pd.DataFrame(rows)


def detect_aliases(mat: pd.DataFrame, priority) -> pd.DataFrame:
    """Proven same-support rank-equivalent hypotheses within each context.

    Two features alias when their valid masks are identical and Spearman |rho| == 1.
    The representative is the lowest-priority key (base before dataset, then ID).
    """
    recs = []
    for ctx, sub in mat.groupby(level=0):
        feats = list(sub.index.get_level_values(1))
        a = sub.to_numpy(float)
        masks = np.isfinite(a)
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                if not np.array_equal(masks[i], masks[j]) or masks[i].sum() < 3:
                    continue
                xi, xj = a[i][masks[i]], a[j][masks[j]]
                if np.ptp(xi) == 0 or np.ptp(xj) == 0:
                    continue
                r = stats.spearmanr(xi, xj).statistic
                if abs(r) >= 1 - 1e-12:
                    rep, ali = sorted([feats[i], feats[j]], key=priority)
                    recs.append(dict(context=ctx, representative=rep, alias=ali, spearman=r,
                                     n=int(masks[i].sum()), kind="proven_rank_equivalent_same_support"))
    return pd.DataFrame(recs, columns=["context", "representative", "alias", "spearman", "n", "kind"])
