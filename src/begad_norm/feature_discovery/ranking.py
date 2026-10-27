"""Region (task-ROI) evidence, deterministic top-k ranking, reserves and task summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

TASK_ORDER = {"KD1": 0, "KD2": 1, "KD3": 2, "READING": 3}


def split_region(key: str):
    task, stim, roi = key.split("|")
    return task, stim, int(roi.removeprefix("ROI_"))


def region_evidence(w: pd.DataFrame, regions: list[str], p_joint_col="p_joint", pass_col="selected_main",
                    delta_col="min_abs_control_delta", cov_col="min_group_coverage", m_r: dict | None = None):
    """Per-region m_r, p_region, anchor and main-candidate flag.

    w: wide micro table (one row per region x feature) restricted to joint-eligible rows for m_r.
    m_r: optional fixed search sizes (bootstrap uses the original sizes).
    """
    recs = []
    gb = {k: g for k, g in w.groupby("scope_or_region_key")}
    for r in regions:
        g = gb.get(r)
        el = g[g["eligible_main"]] if g is not None else None
        mr = (m_r or {}).get(r, 0 if el is None else len(el))
        if el is None or len(el) == 0 or mr == 0:
            recs.append(dict(region_key=r, m_r=mr, p_region=np.nan, anchor_feature=None,
                             main_candidate=False, testable=mr > 0))
            continue
        pj = el[p_joint_col].fillna(1.0).to_numpy(float)
        order = np.lexsort((el["feature_id"].to_numpy(), -el[cov_col].fillna(-1).to_numpy(float),
                            -el[delta_col].fillna(-1).to_numpy(float), pj))
        a = el.iloc[order[0]]
        recs.append(dict(region_key=r, m_r=mr, p_region=min(1.0, mr * float(pj.min())),
                         anchor_feature=a["feature_id"], main_candidate=bool(el[pass_col].any()),
                         anchor_p_joint=float(pj.min()), anchor_min_abs_delta=float(a[delta_col]),
                         anchor_min_coverage=float(a[cov_col]), testable=True))
    ev = pd.DataFrame(recs)
    t = ev["testable"] & ev["p_region"].notna()
    ev["q_region_bh"] = np.nan
    ev["q_region_by"] = np.nan
    if t.any():
        ev.loc[t, "q_region_bh"] = multipletests(ev.loc[t, "p_region"], method="fdr_bh")[1]
        ev.loc[t, "q_region_by"] = multipletests(ev.loc[t, "p_region"], method="fdr_by")[1]
    return ev


def rank_regions(ev: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Deterministic base ordering of main-candidate regions; returns all with base_rank (NA if not candidate)."""
    ev = ev.copy()
    parts = ev["region_key"].map(split_region)
    ev["canonical_task"] = [p[0] for p in parts]
    ev["stimulus_version"] = [p[1] for p in parts]
    ev["roi_id"] = [p[2] for p in parts]
    c = ev[ev["main_candidate"]].copy()
    c["_tier"] = np.where(c["q_region_bh"] <= 0.10, 0, 1)
    c["_task"] = c["canonical_task"].map(TASK_ORDER)
    c = c.sort_values(["_tier", "p_region", "anchor_min_abs_delta", "anchor_min_coverage", "_task",
                       "stimulus_version", "roi_id"],
                      ascending=[True, True, False, False, True, True, True], kind="mergesort")
    c["base_rank"] = np.arange(1, len(c) + 1)
    ev = ev.merge(c[["region_key", "base_rank"]], on="region_key", how="left")
    ev["in_top_k"] = ev["base_rank"].notna() & (ev["base_rank"] <= k)
    return ev


def pair_specific_reserves(w: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r, g in w.groupby("scope_or_region_key"):
        if g["selected_main"].any():
            continue
        ad = sorted(g.loc[g["pass_CN_AD"], "feature_id"])
        mci = sorted(g.loc[g["pass_CN_MCI"], "feature_id"])
        mad = sorted(g.loc[g["pass_MCI_AD"], "feature_id"])
        if not (ad or mci or mad):
            continue
        if ad and mci:
            kind = "complementary_CN_AD_and_CN_MCI_by_different_features"
        elif ad:
            kind = "CN_AD_only"
        elif mci:
            kind = "CN_MCI_only"
        else:
            kind = "MCI_AD_only"
        if mad and kind != "MCI_AD_only":
            kind += "+MCI_AD"
        recs.append(dict(region_key=r, reserve_type=kind, features_pass_CN_AD=";".join(ad),
                         features_pass_CN_MCI=";".join(mci), features_pass_MCI_AD=";".join(mad),
                         min_p_CN_AD=g["p_CN_AD"].min(), min_p_CN_MCI=g["p_CN_MCI"].min(),
                         min_p_MCI_AD=g["p_MCI_AD"].min()))
    cols = ["region_key", "reserve_type", "features_pass_CN_AD", "features_pass_CN_MCI", "features_pass_MCI_AD",
            "min_p_CN_AD", "min_p_CN_MCI", "min_p_MCI_AD"]
    out = pd.DataFrame(recs, columns=cols)
    return out.merge(ev[["region_key", "m_r", "p_region", "q_region_bh"]], on="region_key", how="left")


def task_ranking(ev: pd.DataFrame, w: pd.DataFrame, expected_rois: dict) -> pd.DataFrame:
    recs = []
    for task in sorted(set(ev["canonical_task"]), key=TASK_ORDER.get):
        e = ev[ev["canonical_task"] == task]
        t = e[e["testable"] & e["p_region"].notna()]
        sel = e[e["main_candidate"]]
        ww = w[w["scope_or_region_key"].str.startswith(task + "|")]
        R = len(t)
        p_task = min(1.0, R * t["p_region"].min()) if R else np.nan
        prof = {c: int(ww.groupby("scope_or_region_key")[f"pass_{c}"].any().sum()) for c in
                ("CN_AD", "CN_MCI", "MCI_AD")}
        recs.append(dict(canonical_task=task, expected_rois=expected_rois.get(task), regions_with_data=len(e),
                         testable_regions=R, main_selected_regions=len(sel),
                         selected_fraction_of_testable=len(sel) / R if R else np.nan,
                         feature_families_tested=int(ww.loc[ww["eligible_main"], "feature_id"].nunique()),
                         median_anchor_min_coverage=float(sel["anchor_min_coverage"].median()) if len(sel) else np.nan,
                         p_task=p_task, regions_any_CN_AD=prof["CN_AD"], regions_any_CN_MCI=prof["CN_MCI"],
                         regions_any_MCI_AD=prof["MCI_AD"]))
    tr = pd.DataFrame(recs)
    t = tr["p_task"].notna()
    tr["q_task_bh"] = np.nan
    tr["q_task_by"] = np.nan
    if t.any():
        tr.loc[t, "q_task_bh"] = multipletests(tr.loc[t, "p_task"], method="fdr_bh")[1]
        tr.loc[t, "q_task_by"] = multipletests(tr.loc[t, "p_task"], method="fdr_by")[1]
    tr["_o"] = tr["canonical_task"].map(TASK_ORDER)
    tr["_c"] = tr["median_anchor_min_coverage"].fillna(-np.inf)
    tr = tr.sort_values(["p_task", "_c", "_o"], ascending=[True, False, True], na_position="last")
    tr["task_rank"] = np.arange(1, len(tr) + 1)
    return tr.drop(columns=["_o", "_c"])
