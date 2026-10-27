"""User-requested task-stratified region lists (post hoc, after the primary run).

Lists (ROI64_T1235 unless --cohort is given):
  control: up to 20 KD and 15 READING regions passing the main rule (p_CN_AD<0.20 AND p_CN_MCI<0.20),
           ordered by the frozen primary base ranking (q_region tier, p_region, effect, coverage).
  mci_ad:  up to 10 KD and 10 READING regions with >=1 micro feature p_MCI_AD<0.20, ordered by
           p_region_MCI_AD = min(1, m_r_MCI_AD * min p_MCI_AD) with BH across all MCI_AD-testable regions.

These lists re-use the frozen statistics; they add no hypotheses to the primary families. The task quotas are
the researcher's request made after the results were seen, so the lists are labelled post hoc.

    uv run python -m begad_norm.feature_discovery.posthoc_lists --run-dir outputs/feature_discovery/<run>
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from begad_norm.feature_discovery.ranking import TASK_ORDER, split_region

KD = ("KD1", "KD2", "KD3")


def mci_ad_regions(w: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for r, g in w.groupby("scope_or_region_key"):
        el = g[g["eligible_MCI_AD"]]
        if not len(el):
            continue
        el = el.assign(_ad=el["delta_MCI_AD"].abs(), _cov=el[["coverage_MCI", "coverage_AD"]].min(axis=1))
        el = el.sort_values(["p_MCI_AD", "_ad", "_cov", "feature_id"], ascending=[True, False, False, True])
        a = el.iloc[0]
        task, stim, roi = split_region(r)
        passing = el[el["pass_MCI_AD"]]
        recs.append(dict(region_key=r, canonical_task=task, stimulus_version=stim, roi_id=roi,
                         m_r_MCI_AD=len(el), p_region_MCI_AD=min(1.0, len(el) * a["p_MCI_AD"]),
                         anchor_feature=a["feature_id"], anchor_p_MCI_AD=a["p_MCI_AD"],
                         anchor_delta_MCI_AD=a["delta_MCI_AD"], anchor_q_bh_MCI_AD=a["q_bh_MCI_AD"],
                         anchor_p_CN_AD=a["p_CN_AD"], anchor_p_CN_MCI=a["p_CN_MCI"],
                         anchor_coverage_MCI=a["coverage_MCI"], anchor_coverage_AD=a["coverage_AD"],
                         anchor_min_coverage=a["_cov"], n_features_pass_MCI_AD=len(passing),
                         features_pass_MCI_AD=";".join(passing.feature_id),
                         also_main_selected=bool(g["selected_main"].any())))
    ev = pd.DataFrame(recs)
    ev["q_region_MCI_AD_bh"] = multipletests(ev["p_region_MCI_AD"], method="fdr_bh")[1]
    ev["q_region_MCI_AD_by"] = multipletests(ev["p_region_MCI_AD"], method="fdr_by")[1]
    ev["candidate"] = ev["anchor_p_MCI_AD"] < 0.20
    ev["_t"] = ev["canonical_task"].map(TASK_ORDER)
    ev = ev.sort_values(["p_region_MCI_AD", "anchor_delta_MCI_AD", "anchor_min_coverage", "_t", "roi_id"],
                        key=lambda s: -s.abs() if s.name == "anchor_delta_MCI_AD" else
                        (-s if s.name == "anchor_min_coverage" else s))
    return ev.drop(columns="_t")


def take(df, tasks, k, rank_col):
    d = df[df["canonical_task"].isin(tasks)].copy()
    d = d.sort_values(rank_col) if rank_col else d
    d = d.head(k).copy()
    d["list_rank"] = np.arange(1, len(d) + 1)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cohort", default="ROI64_T1235")
    ap.add_argument("--kd-control", type=int, default=20)
    ap.add_argument("--reading-control", type=int, default=15)
    ap.add_argument("--kd-mci-ad", type=int, default=10)
    ap.add_argument("--reading-mci-ad", type=int, default=10)
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    out = run / "posthoc_task_stratified"
    out.mkdir(exist_ok=True)
    ra = pd.read_csv(run / "roi_ranking_all.csv")
    ra = ra[(ra.cohort_id == a.cohort) & (ra.status == "main_candidate")]
    ctrl_kd = take(ra, KD, a.kd_control, "base_rank")
    ctrl_rd = take(ra, ("READING",), a.reading_control, "base_rank")
    res = pickle.loads((run / "cache" / "report.pkl").read_bytes())
    w = res["mic"][res["mic"].cohort_id == a.cohort]
    ev = mci_ad_regions(w)
    cand = ev[ev["candidate"]].reset_index(drop=True)
    cand["mci_ad_rank_all_tasks"] = np.arange(1, len(cand) + 1)
    mad_kd = take(cand, KD, a.kd_mci_ad, None)
    mad_rd = take(cand, ("READING",), a.reading_mci_ad, None)
    ctrl = pd.concat([ctrl_kd.assign(list="KD_control_top20"), ctrl_rd.assign(list="READING_control_top15")])
    mad = pd.concat([mad_kd.assign(list="KD_MCI_AD_top10"), mad_rd.assign(list="READING_MCI_AD_top10")])
    ctrl.insert(0, "posthoc_label", "post_hoc_task_quota_requested_after_results")
    mad.insert(0, "posthoc_label", "post_hoc_task_quota_requested_after_results")
    ctrl.to_csv(out / "control_regions_KD20_READING15.csv", index=False, na_rep="NA")
    mad.to_csv(out / "mci_ad_regions_KD10_READING10.csv", index=False, na_rep="NA")
    ev.to_csv(out / "mci_ad_region_evidence_all.csv", index=False, na_rep="NA")
    avail = dict(kd_control=int(ra.canonical_task.isin(KD).sum()), rd_control=int((ra.canonical_task == "READING").sum()),
                 kd_mad=int(cand.canonical_task.isin(KD).sum()), rd_mad=int((cand.canonical_task == "READING").sum()),
                 mad_testable=len(ev), mad_q10=int((ev.q_region_MCI_AD_bh <= .10).sum()))
    print(avail)
    print(ctrl[["list", "list_rank", "canonical_task", "roi_id", "base_rank", "anchor_feature", "anchor_p_CN_AD",
                "anchor_p_CN_MCI", "anchor_p_MCI_AD", "anchor_delta_CN_AD", "anchor_delta_CN_MCI", "p_region",
                "q_region_bh", "top20_bootstrap_frequency", "main_selected_features"]].to_string(index=False))
    print(mad[["list", "list_rank", "canonical_task", "roi_id", "anchor_feature", "anchor_p_MCI_AD",
               "anchor_delta_MCI_AD", "m_r_MCI_AD", "p_region_MCI_AD", "q_region_MCI_AD_bh", "anchor_p_CN_AD",
               "anchor_p_CN_MCI", "anchor_min_coverage", "features_pass_MCI_AD", "also_main_selected"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
