"""ROI64 workbook features: micro (task-ROI) and workbook-derived macro (task scope).

Uses the ORIGINAL T1235.xlsx / T5.xlsx exports, never the *_clean.xlsx files
(those impute totalFixTime and delete valid negative baseline-relative pupil
values). Missing ROI rows are 'unavailable', never zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from begad_norm.feature_discovery.extract_macro import INSUF, INVAL, OBS, UNAV, ZERO

WB_TASK_TO_CANONICAL = {1: "KD1", 2: "KD2", 3: "KD3", 5: "READING"}
EXPECTED_ROIS = {"KD1": 40, "KD2": 40, "KD3": 40, "READING": 159}
STIMULUS_VERSION = {"KD1": "KD_1_highres", "KD2": "KD_2_highres", "KD3": "KD_3_highres",
                    "READING": "lukeminen_1"}
SPEECH_CAP_S = 10.0
ART_GUARD_S = 0.05

MICRO_IDS = {
    "ROI64_T1235": ["M001", "M051", "M061", "D001", "D002", "D003", "D004", "D005"],
    "ROI64_T5": ["M001", "M051", "M061", "D001", "D003", "D004", "D005"],
}
MACRO_IDS = {
    "ROI64_T1235": ["A061", "A062", "A063", "D007", "D008", "D009", "D010", "D011", "D012", "D013", "D014", "D015"],
    "ROI64_T5": ["A061", "A062", "A063", "D007", "D008", "D009", "D010", "D012", "D013", "D014", "D015"],
}
CROSS_IDS = {"ROI64_T1235": ["A079", "D016"], "ROI64_T5": []}


def region_key(task, roi):
    return f"{task}|{STIMULUS_VERSION[task]}|ROI_{int(roi):03d}"


def micro_values(wb: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Observed micro values for every existing workbook row (one row per key)."""
    w = wb.copy()
    w["canonical_task"] = w["task"].map(WB_TASK_TO_CANONICAL)
    if w["canonical_task"].isna().any():
        raise ValueError(f"unknown workbook task codes: {sorted(w.loc[w.canonical_task.isna(), 'task'].unique())}")
    recs = []

    def add(fid, val, status, n=np.nan):
        recs.append(pd.DataFrame({"subject_id": w["pat"], "canonical_task": w["canonical_task"],
                                  "roi_id": w["ROI"], "feature_id": fid, "value": val,
                                  "measurement_status": status, "n_events": n}))

    nfix = w["nFixations"].astype(float)
    add("M001", nfix, np.where(nfix > 0, OBS, ZERO), nfix)
    add("M051", w["totalSpokenTime"], np.where(w["totalSpokenTime"] > 0, OBS, INVAL))
    sd = w["speechDelay"].astype(float)
    cens = (sd <= 0) | (sd >= SPEECH_CAP_S)
    add("M061", sd.where(~cens), np.where(cens, "censored", OBS))
    if cohort == "ROI64_T1235":
        both = w["diffPupL"].notna() & w["diffPupR"].notna()
        add("D001", ((w["diffPupL"] + w["diffPupR"]) / 2).where(both), np.where(both, OBS, INVAL))
        cons = (w["maxPupLB"] <= w["startPupLB"] + 1e-9) & (w["startPupLB"] <= w["minPupLB"] + 1e-9)
        ok = w["diffXpupL"].notna() & cons
        add("D002", w["diffXpupL"].where(ok),
            np.where(w["diffXpupL"].isna(), INVAL, np.where(ok, OBS, INVAL)))
    else:
        ok = (w[["startPupL", "startPupR", "endPupL", "endPupR"]] > 0).all(axis=1)
        add("D001", ((w["diffPupL"] + w["diffPupR"]) / 2).where(ok), np.where(ok, OBS, INVAL))
    reg = w["regSaccades"].astype(float)
    add("D003", reg, np.where(reg > 0, OBS, ZERO))
    add("D004", w["saccadeTotLength"], np.where(w["saccadeTotLength"] > 0, OBS, INVAL))
    okd = w["totalSpokenTime"] >= ART_GUARD_S
    add("D005", (nfix / w["totalSpokenTime"]).where(okd), np.where(okd, OBS, INSUF))
    out = pd.concat(recs, ignore_index=True)
    out["region_key"] = [region_key(t, r) for t, r in zip(out.canonical_task, out.roi_id)]
    if out.duplicated(["subject_id", "region_key", "feature_id"]).any():
        raise ValueError("duplicate subject/task/ROI/feature keys in micro extraction")
    return out


def _valid(s):
    return s["measurement_status"].isin([OBS, ZERO])


def macro_values(micro: pd.DataFrame, cohort: str, roster: list[str], roi_coverage: float) -> pd.DataFrame:
    """Workbook-derived task-scope macro features; equal ROI weighting."""
    recs = []
    piv = {fid: micro[(micro.feature_id == fid) & _valid(micro)]
           .set_index(["subject_id", "canonical_task", "roi_id"])["value"]
           for fid in micro.feature_id.unique()}
    present = micro[micro.feature_id == "M001"].groupby(["subject_id", "canonical_task"]).size()
    tasks = sorted(micro.canonical_task.unique(), key=list(EXPECTED_ROIS).index)
    for sid in roster:
        for task in tasks:
            exp = EXPECTED_ROIS[task]
            need = int(np.ceil(roi_coverage * exp - 1e-9))
            n_obs = int(present.get((sid, task), 0))

            def vals(fid):
                s = piv.get(fid)
                if s is None:
                    return pd.Series(dtype=float)
                try:
                    return s.loc[(sid, task)]
                except KeyError:
                    return pd.Series(dtype=float)

            def put(fid, v, st, n, nroi=None):
                recs.append(dict(subject_id=sid, scope_id=task, feature_id=fid, value=v, measurement_status=st,
                                 n_events=n, n_rois=n if nroi is None else nroi, expected_rois=exp))

            if n_obs == 0:
                for fid in MACRO_IDS[cohort]:
                    put(fid, np.nan, UNAV, 0)
                continue
            sd = vals("M061").to_numpy(float)
            gate = len(sd) >= need
            put("A061", np.median(sd) if gate and len(sd) else np.nan, OBS if gate and len(sd) else INSUF, len(sd))
            ok = gate and len(sd) >= 3
            put("A062", np.subtract(*np.quantile(sd, [.75, .25])) if ok else np.nan, OBS if ok else INSUF, len(sd))
            ok = gate and len(sd) >= 10
            put("A063", np.quantile(sd, .9) if ok else np.nan, OBS if ok else INSUF, len(sd))
            nf = vals("M001").to_numpy(float)
            gate = len(nf) >= need
            put("D007", nf.mean() if gate else np.nan, OBS if gate else INSUF, len(nf))
            put("D008", (nf >= 2).mean() if gate else np.nan, OBS if gate else INSUF, len(nf))
            art = vals("M051").to_numpy(float)
            gate = len(art) >= need
            put("D009", np.median(art) if gate else np.nan, OBS if gate else INSUF, len(art))
            pc = vals("D001").to_numpy(float)
            gate = len(pc) >= need
            put("D010", np.median(pc) if gate else np.nan, OBS if gate else INSUF, len(pc))
            if "D011" in MACRO_IDS[cohort]:
                pr = vals("D002").to_numpy(float)
                gate = len(pr) >= need
                put("D011", np.median(pr) if gate else np.nan, OBS if gate else INSUF, len(pr))
            rg = vals("D003").to_numpy(float)
            gate = len(rg) >= need
            put("D012", (rg >= 1).mean() if gate else np.nan, OBS if gate else INSUF, len(rg))
            for fid, a, b in (("D013", "M001", "M061"), ("D014", "M051", "M061"), ("D015", "M051", "M001")):
                j = pd.concat([vals(a).rename("a"), vals(b).rename("b")], axis=1).dropna()
                if len(j) >= 10 and j.a.nunique() > 1 and j.b.nunique() > 1:
                    put(fid, float(stats.spearmanr(j.a, j.b).statistic), OBS, len(j))
                else:
                    put(fid, np.nan, INSUF, len(j))
    out = pd.DataFrame(recs)
    # cross-task features
    for fid, src in (("A079", "A061"), ("D016", "D007")):
        if fid not in CROSS_IDS[cohort]:
            continue
        v = out[(out.feature_id == src) & out.measurement_status.isin([OBS, ZERO])].pivot(
            index="subject_id", columns="scope_id", values="value").reindex(roster)
        for sid in roster:
            row = v.loc[sid] if sid in v.index else None
            vals4 = [row.get(t, np.nan) if row is not None else np.nan for t in ("READING", "KD1", "KD2", "KD3")]
            if np.all(np.isfinite(vals4)):
                recs.append(dict(subject_id=sid, scope_id="CROSS_BOTH", feature_id=fid,
                                 value=vals4[0] - np.mean(vals4[1:]), measurement_status=OBS, n_events=4,
                                 n_rois=np.nan, expected_rois=np.nan))
            else:
                recs.append(dict(subject_id=sid, scope_id="CROSS_BOTH", feature_id=fid, value=np.nan,
                                 measurement_status=UNAV, n_events=int(np.isfinite(vals4).sum()),
                                 n_rois=np.nan, expected_rois=np.nan))
    return pd.DataFrame(recs)
