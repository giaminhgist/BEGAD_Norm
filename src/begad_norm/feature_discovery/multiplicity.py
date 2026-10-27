"""Multiplicity families (BH primary, BY sensitivity) and selection flags.

Families are defined per cohort and level BEFORE selection and contain every
eligible hypothesis from both origins (base160 and dataset100).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

SCREEN_P = 0.20


def _adjust(p: np.ndarray, method: str) -> np.ndarray:
    if len(p) == 0:
        return np.array([])
    return multipletests(p, method=method)[1]


def apply_pairwise(st: pd.DataFrame, family_id: str) -> pd.DataFrame:
    st = st.copy()
    el = st["eligible"].fillna(False).astype(bool) & st["p_primary"].notna()
    st["pairwise_family_id"] = np.where(el, family_id, "")
    st["pairwise_family_size"] = np.where(el, int(el.sum()), 0)
    st["q_bh"] = np.nan
    st["q_by"] = np.nan
    st.loc[el, "q_bh"] = _adjust(st.loc[el, "p_primary"].to_numpy(), "fdr_bh")
    st.loc[el, "q_by"] = _adjust(st.loc[el, "p_primary"].to_numpy(), "fdr_by")
    return st


def wide_table(st: pd.DataFrame, joint_family_id: str, strict_family_id: str) -> pd.DataFrame:
    """One row per hypothesis (context, feature) with all three contrasts side by side."""
    keys = ["scope_or_region_key", "feature_id"]
    cols = ["eligible", "status", "exclusion_reason", "p_primary", "cliffs_delta", "q_bh", "q_by",
            "n1", "n2", "coverage1", "coverage2"]
    w = st.pivot_table(index=keys, columns="contrast", values=cols, aggfunc="first", dropna=False)
    w.columns = [f"{c}_{k}" for c, k in w.columns]
    w = w[w.index.isin(pd.MultiIndex.from_frame(st[keys].drop_duplicates()))]  # no Cartesian artefacts
    w = w.reset_index()
    for c in ("CN_AD", "CN_MCI", "MCI_AD"):
        w[f"eligible_{c}"] = w[f"eligible_{c}"].fillna(False).astype(bool)
        w[f"p_{c}"] = w.pop(f"p_primary_{c}").astype(float)
        w[f"delta_{c}"] = w.pop(f"cliffs_delta_{c}").astype(float)
        w[f"pass_{c}"] = w[f"eligible_{c}"] & (w[f"p_{c}"] < SCREEN_P)
    w["eligible_main"] = w["eligible_CN_AD"] & w["eligible_CN_MCI"]
    w["selected_main"] = w["eligible_main"] & w["pass_CN_AD"] & w["pass_CN_MCI"]
    w["selected_three_pairs"] = w["selected_main"] & w["eligible_MCI_AD"] & w["pass_MCI_AD"]
    w["p_joint"] = np.where(w["eligible_main"], np.maximum(w["p_CN_AD"], w["p_CN_MCI"]), np.nan)
    j = w["eligible_main"] & w["p_joint"].notna()
    w["joint_family_id"] = np.where(j, joint_family_id, "")
    w["joint_family_size"] = np.where(j, int(j.sum()), 0)
    w["q_joint_bh"] = np.nan
    w["q_joint_by"] = np.nan
    w.loc[j, "q_joint_bh"] = _adjust(w.loc[j, "p_joint"].to_numpy(), "fdr_bh")
    w.loc[j, "q_joint_by"] = _adjust(w.loc[j, "p_joint"].to_numpy(), "fdr_by")
    s3 = j & w["eligible_MCI_AD"]
    w["p_joint3"] = np.where(s3, np.maximum(w["p_joint"], w["p_MCI_AD"]), np.nan)
    w["strict3_family_id"] = np.where(s3, strict_family_id, "")
    w["strict3_family_size"] = np.where(s3, int(s3.sum()), 0)
    w["q_joint3_bh"] = np.nan
    w.loc[s3, "q_joint3_bh"] = _adjust(w.loc[s3, "p_joint3"].to_numpy(), "fdr_bh")
    w["pairwise_fdr_supported"] = (w["q_bh_CN_AD"] <= 0.10) & (w["q_bh_CN_MCI"] <= 0.10) & w["eligible_main"]
    w["pairwise_fdr_supported_by"] = (w["q_by_CN_AD"] <= 0.10) & (w["q_by_CN_MCI"] <= 0.10) & w["eligible_main"]
    w["joint_fdr_supported"] = w["q_joint_bh"] <= 0.10
    w["joint_fdr_supported_by"] = w["q_joint_by"] <= 0.10
    w["effect_preferred"] = (w["delta_CN_AD"].abs() >= 0.20) & (w["delta_CN_MCI"].abs() >= 0.20) & w["eligible_main"]
    w["min_abs_control_delta"] = np.minimum(w["delta_CN_AD"].abs(), w["delta_CN_MCI"].abs())
    covs = [w[f"coverage{i}_{c}"] for c in ("CN_AD", "CN_MCI") for i in (1, 2)]
    w["min_control_coverage"] = np.nanmin(np.column_stack(covs).astype(float), axis=1)
    # per-group coverage (CN from CN_AD, MCI and AD from MCI_AD/CN_* rows)
    w["coverage_CN"] = w["coverage1_CN_AD"].astype(float)
    w["coverage_MCI"] = w["coverage2_CN_MCI"].astype(float)
    w["coverage_AD"] = w["coverage2_CN_AD"].astype(float)
    w["min_group_coverage"] = np.nanmin(w[["coverage_CN", "coverage_MCI", "coverage_AD"]].to_numpy(float), axis=1)
    w["direction_pattern"] = [_pattern(a, b, c) for a, b, c in zip(w["delta_CN_AD"], w["delta_CN_MCI"],
                                                                     w["delta_MCI_AD"])]
    return w


def _pattern(d_ad, d_mci, d_mci_ad):
    def sgn(v):
        if not np.isfinite(v) or abs(v) < 1e-12:
            return 0
        return 1 if v > 0 else -1
    a, m = sgn(d_ad), sgn(d_mci)
    if a == 1 and m == 1:
        base = "both patient groups higher than CN"
    elif a == -1 and m == -1:
        base = "both patient groups lower than CN"
    elif a and m and a != m:
        base = "opposite directions (MCI vs AD relative to CN)"
    else:
        base = "indeterminate"
    t = sgn(d_mci_ad)
    return base + ("; AD>MCI" if t == 1 else "; AD<MCI" if t == -1 else "")
