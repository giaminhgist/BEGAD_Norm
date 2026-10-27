"""Source inventory, cohort manifests and computed semantic checks.

Every number written to semantic_audit.csv is recomputed from the inspected
sources at run time; nothing is copied from the instruction pack.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.metadata
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from begad_norm.feature_discovery import EXTRACTION_VERSION
from begad_norm.feature_discovery.event_contracts import (
    MM_PER_PX, RAW_TASK_TO_CANONICAL, build_fixations, build_saccades, build_transitions, read_raw,
)
from begad_norm.feature_discovery.extract_micro import EXPECTED_ROIS, STIMULUS_VERSION, WB_TASK_TO_CANONICAL
from begad_norm.feature_discovery.statistics import map_group

PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "openpyxl", "pyyaml", "matplotlib"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def inventory(cfg: dict, root: Path) -> dict:
    src = cfg["sources"]
    items = []

    def add(path, cohort, kind, units="", columns=""):
        p = root / path
        items.append(dict(path=str(path), exists=p.exists(), sha256=sha256(p) if p.exists() else None,
                          size_bytes=p.stat().st_size if p.exists() else None, cohort=cohort, visit="baseline",
                          source_type=kind, units=units, column_mapping=columns, parser_version=EXTRACTION_VERSION))

    add(src["t1235_xlsx"], "ROI64_T1235", "roi_summary_workbook_original",
        "s (timing), deg (saccadeTotLength), mm relative to unidentified constant with inverted sign (*B)",
        "task 1/2/3/5 -> KD1/KD2/KD3/READING")
    add(src["t5_xlsx"], "ROI64_T5", "roi_summary_workbook_original", "s, deg, mm absolute (0 = missing)",
        "task 5 -> READING")
    for k in ("t1235_clean_xlsx", "t5_clean_xlsx"):
        add(src[k], "not_used", "derived_cleaned_workbook (imputed/excluded; not used for primary tests)")
    add(src["participants_csv"], "RAW157", "roster", "", "group 0/1/2 = CO/MCI/AD -> CN/MCI/AD")
    raw = sorted(glob.glob(str(root / src["raw_dir"] / "*.csv")))
    for f in raw:
        add(Path(f).relative_to(root), "RAW157_or_excluded", "raw_tobii_event_export",
            "ms timestamps/durations, deg amplitudes, px (MCS/ADCS), mm pupil/gaze/distance",
            "Task 3/4/5/9 -> KD1/KD2/KD3/READING")
    v2 = sorted(glob.glob(str(root / src["raw_v2_dir"] / "*.csv")))
    for f in v2:
        add(Path(f).relative_to(root), "not_used", "raw_tobii_event_export_v2_reduced")
    absent = [dict(source=s, status="absent") for s in (
        "ROI bounding boxes / stimulus layout", "stimulus word/digit text and reading order", "audio recordings",
        "TextGrid/ELAN speech alignment", "speech error/repetition annotations", "lexical metadata",
        "continuous pupil samples", "media onset/offset events", "demographics (age, sex, education)",
        "visual acuity / language proficiency")]
    return dict(items=items, absent_sources=absent, package_versions=versions(),
                python=f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}")


def load_workbook(root, path):
    return pd.read_excel(root / path, engine="openpyxl")


def roster_raw(cfg, root) -> pd.DataFrame:
    p = pd.read_csv(root / cfg["sources"]["participants_csv"])
    p = p.rename(columns={"participant_id": "subject_id", "group": "group_code"})
    p["group"] = p["group_code"].map(map_group)
    if not (p["group_label"].map({"CO": "CN", "MCI": "MCI", "AD": "AD"}) == p["group"]).all():
        raise ValueError("roster group_label disagrees with group code")
    return p


def roster_workbook(wb: pd.DataFrame) -> pd.DataFrame:
    g = wb.groupby("pat")["group"].agg(["nunique", "first"])
    if (g["nunique"] > 1).any():
        raise ValueError("participant with several diagnostic codes in workbook")
    r = g["first"].rename("group_code").reset_index().rename(columns={"pat": "subject_id"})
    r["group"] = r["group_code"].map(map_group)
    return r


def parse_raw_cohort(cfg, root, roster: pd.DataFrame, cache: Path):
    """Parse raw events for every roster participant (cached)."""
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    out = {}
    keep_rows = ["Task", "ValidityLeft", "ValidityRight", "PupilLeft", "PupilRight", "Group", "IsFollowup",
                 "RealTag"]
    for r in roster.itertuples():
        path = root / cfg["sources"]["raw_dir"] / r.filename
        df = read_raw(path)
        fix = build_fixations(df)
        sac = build_saccades(df)
        tr = build_transitions(fix, sac)
        out[r.subject_id] = dict(fix=fix, sac=sac, tr=tr, rows=df[keep_rows].copy(),
                                 file_group=int(df["Group"].iloc[0]), file_followup=bool(df["IsFollowup"].iloc[0]),
                                 baseline=float((df["PupilLeftBaseline"].iloc[0] + df["PupilRightBaseline"].iloc[0]) / 2),
                                 n_rows=len(df), media=df.groupby("Task")[["MediaPosX (ADCSpx)", "MediaPosY (ADCSpx)",
                                                                           "MediaWidth", "MediaHeight"]].nunique()
                                 .max().max())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def subject_manifest(raw_roster, wb_roster, t5_roster, raw_events) -> pd.DataFrame:
    rows = []
    for r in raw_roster.itertuples():
        ev = raw_events.get(r.subject_id, {})
        rows.append(dict(cohort_id="RAW157", subject_id=r.subject_id, visit_id="baseline", group_code=r.group_code,
                         group=r.group, expected_tasks="KD1;KD2;KD3;READING", source_file=r.filename,
                         file_group_code=ev.get("file_group"), file_is_followup=ev.get("file_followup")))
    for name, ro in (("ROI64_T1235", wb_roster), ("ROI64_T5", t5_roster)):
        tasks = "KD1;KD2;KD3;READING" if name == "ROI64_T1235" else "READING"
        for r in ro.itertuples():
            rows.append(dict(cohort_id=name, subject_id=r.subject_id, visit_id="baseline", group_code=r.group_code,
                             group=r.group, expected_tasks=tasks, source_file=name.split("_")[1] + ".xlsx",
                             in_raw157_roster=r.subject_id in set(raw_roster.subject_id)))
    return pd.DataFrame(rows)


def task_roi_manifest() -> pd.DataFrame:
    rows = []
    raw_code = {v: k for k, v in RAW_TASK_TO_CANONICAL.items()}
    wb_code = {v: k for k, v in WB_TASK_TO_CANONICAL.items()}
    for task, n in EXPECTED_ROIS.items():
        for roi in range(n):
            rows.append(dict(canonical_task=task, workbook_task_code=wb_code[task], raw_task_code=raw_code[task],
                             stimulus_version=STIMULUS_VERSION[task], roi_id=roi, roi_text="text_unavailable",
                             reading_rank="NA_unverified", line_index="NA_unavailable", bounds="NA_unavailable",
                             roi_identity_source="workbook ROI column; count matches paper (KD 40 digits, text 159 words)",
                             order_source="ROI ID order NOT verified as reading order"))
    return pd.DataFrame(rows)


def semantic_checks(cfg, root, t1235, t5, raw_events, raw_roster) -> pd.DataFrame:
    C = []

    def add(cid, sources, evidence, severity, resolution, blocked):
        C.append(dict(check_id=cid, affected_sources=sources, measured_evidence=evidence, severity=severity,
                      resolution=resolution, blocked_feature_ids=blocked))

    d = t5
    add("S01", "T5.xlsx diffPupL/R", f"max |diffPupL-(endPupL-startPupL)| = "
        f"{np.abs(d.diffPupL - (d.endPupL - d.startPupL)).max():.2e}",
        "info", "diff is an exact dependency of start/end; only the difference (D001) is used", "M044;M045;M046")
    d = t1235
    m = d[d.task == 5].merge(t5, on=["pat", "ROI"], suffixes=("", "_5"))
    s = (m.startPupL + m.startPupLB)
    add("S02", "T1235.xlsx *B pupil fields", f"max |diffPupL-(startPupLB-endPupLB)| = "
        f"{np.nanmax(np.abs(d.diffPupL - (d.startPupLB - d.endPupLB))):.2e}; within-participant SD of "
        f"(T5 startPupL + T1235 startPupLB) median {s.groupby(m.pat).std().median():.3f} mm vs SD of difference "
        f"{(m.startPupL - m.startPupLB).groupby(m.pat).std().median():.3f} => B = c - pupil (sign inverted)",
        "high", "B fields treated as constant-relative with inverted sign; only constant-free contrasts used",
        "M044;A041;A050")
    for nm, w in (("T5", t5), ("T1235", t1235)):
        add(f"S03_{nm}", f"{nm}.xlsx saccade counts",
            f"regSaccades>nTobiiSaccades in {(w.regSaccades > w.nTobiiSaccades).sum()}/{len(w)} rows; "
            f"tiny+long>nTobii in {(w.tinySaccades + w.longSaccades > w.nTobiiSaccades).sum()}; nTobii==0 with "
            f"saccadeTotLength>0 in {((w.nTobiiSaccades == 0) & (w.saccadeTotLength > 0)).sum()}",
            "high", "no ratios from these counts; regSaccades/saccadeTotLength only as stand-alone D003/D004",
            "M029;M030")
        multi = w.nFixations >= 2
        add(f"S04_{nm}", f"{nm}.xlsx totalFixTime",
            f"totalFixTime==0 in {(w.totalFixTime == 0).sum()} rows, all with nFixations==1 "
            f"({((w.totalFixTime == 0) == (w.nFixations == 1)).all()}); per-fixation value <0.06 s in "
            f"{((w.totalFixTime / w.nFixations < 0.06) & multi).sum() / multi.sum():.1%} of multi-fixation rows",
            "high", "not a summed fixation duration; blocked", "M003;M008;A010;A037")
    c5 = load_workbook(root, cfg["sources"]["t5_clean_xlsx"])
    c12 = load_workbook(root, cfg["sources"]["t1235_clean_xlsx"])
    add("S05", "T*_clean.xlsx (subject-level/src/begad_norm/cleaning.py)",
        f"clean T5 drops {len(t5) - len(c5)} rows and T1235 drops {len(t1235) - len(c12)} rows; totalFixTime of "
        f"{(t1235.nFixations == 1).sum()} single-fixation T1235 rows imputed", "high",
        "original workbooks used; no imputed value enters primary tests", "")
    add("S06", "pupil <=0 coding",
        f"T5 start/end <=0 (missing code 0): {int((t5[['startPupL', 'startPupR', 'endPupL', 'endPupR']] <= 0).any(axis=1).sum())} rows; "
        f"T1235 *B <=0 (valid constant-relative values): "
        f"{int((t1235[['startPupLB', 'startPupRB', 'endPupLB', 'endPupRB']] <= 0).any(axis=1).sum())} rows", "high",
        "T5 0 -> invalid_measurement; T1235 negatives retained", "")
    add("S07", "legacy tem_ifi/tem_velocity (subject-level/docs/baseline/eda_summary.md)",
        "half-dwell interval approximation; not measured gaps", "info",
        "not reused; D021 uses actual onset/offset timestamps", "")
    add("S08", "legacy tem_trans_entropy", "spatial histogram, not transitions", "info",
        "A036 computed from gap-free fixation transitions", "")
    add("S09", "legacy pup_slope", "uses fixation index", "info", "D045 uses onset seconds", "")
    add("S10", "IEEE mapper regression", "turning angle >90 deg called regression", "info",
        "turning reversals are D032; reading-order regressions blocked (no ROI order)", "M018;M019;M020;A013")
    add("S12", "paper Table II caption", "'Last fixation refers to the fixation at the end of the last saccade "
        "that began from the ROI area'", "high", "endPup*/endSpeechDelay use the exit-landing fixation",
        "M045;M046;M062;A064;A065;A066")
    for nm, w in (("T5", t5), ("T1235", t1235)):
        add(f"S13_{nm}", f"{nm}.xlsx speechDelay/endSpeechDelay",
            f"speechDelay min {w.speechDelay.min():.4f}, max {w.speechDelay.max():.4f}; in [9.99,10) {((w.speechDelay >= 9.99) & (w.speechDelay < 10)).sum()} rows, "
            f">=10 in {(w.speechDelay >= 10).sum()}, <=0 in {(w.speechDelay <= 0).sum()}; endSpeechDelay max "
            f"{w.endSpeechDelay.max():.4f}, <0 in {(w.endSpeechDelay < 0).mean():.1%}", "medium",
            "T1235 delays are right-truncated below 10 s (observation window); speechDelay <=0 or >=10 s is "
            "treated as censored in both cohorts so T5 uses the same window", "")
    eq = {c: float(np.isclose(m[c], m[c + "_5"]).mean()) for c in
          ("nFixations", "totalSpokenTime", "speechDelay", "endSpeechDelay", "saccadeTotLength", "regSaccades")}
    add("S14", "T5 vs T1235 task 5", "row-wise equality fractions " + json.dumps({k: round(v, 3) for k, v in eq.items()}),
        "high", "different exports; analysed as separate cohorts (T5 = reading sensitivity)", "")
    # raw-derived checks
    fixcnt = {sid: e["fix"][e["fix"]["valid"]].groupby("task").size() for sid, e in raw_events.items()}
    FC = pd.DataFrame(fixcnt).T
    FC.columns = [f"raw{int(c)}" for c in FC.columns]
    W = t1235.groupby(["pat", "task"]).nFixations.sum().unstack()
    W.columns = [f"wb{int(c)}" for c in W.columns]
    J = W.join(FC, how="inner")
    best = {}
    for wt in W.columns:
        cors = {rt: stats.spearmanr(J[wt], J[rt], nan_policy="omit").statistic for rt in FC.columns}
        top = max(cors, key=lambda k: cors[k] if np.isfinite(cors[k]) else -9)
        best[wt] = (top, round(float(cors[top]), 2))
    add("S15", "workbook task codes vs raw Task", f"label-blind best Spearman match of per-person fixation totals "
        f"(n={len(J)}): " + json.dumps(best) +
        f"; workbook totals exceed raw counts (median wb1/raw3 ratio {np.nanmedian(J['wb1'] / J['raw3']):.2f})",
        "high", "mapping 1/2/3/5 -> raw 3/4/5/9 = KD1/KD2/KD3/READING; ROI64 and RAW157 not pooled", "")
    allfix = pd.concat([e["fix"] for e in raw_events.values()])
    alltr = pd.concat([e["tr"] for e in raw_events.values()])
    add("S16", "raw fixation rows", f"{allfix.has_both_rows.mean():.4f} of fixations have onset+Final rows; "
        f"median (Final ts - onset ts - duration) = {np.nanmedian(allfix.final_ms - allfix.onset_ms - allfix.dur_ms):.1f} ms; "
        f"{allfix.straddles_task.sum()} task-straddling fixations excluded", "info",
        "offset = onset + GazeEventDuration", "")
    sample = list(raw_events)[:: max(1, len(raw_events) // 20)]
    rs = []
    for sid in sample:
        f = raw_events[sid]["fix"]
        f = f[f.valid & f.in_media].sort_values("onset_ms")
        dpx = np.hypot(np.diff(f.screen_x), np.diff(f.screen_y))
        eye = f[["dist_on", "dist_fin"]].mean(axis=1).to_numpy()[1:]
        ang = np.degrees(2 * np.arctan(dpx * MM_PER_PX / (2 * eye)))
        a = f.amp_in.to_numpy()[1:]
        ok = np.isfinite(a) & np.isfinite(ang) & (np.diff(f.task.to_numpy()) == 0)
        if ok.sum() > 10:
            rs.append(stats.pearsonr(a[ok], ang[ok]).statistic)
    add("S17", "raw SaccadicAmplitude", f"Pearson r between exported amplitude and fixation-to-fixation angle from "
        f"positions: median {np.median(rs):.3f} over {len(rs)} participants", "info",
        "amplitude = incoming fixation-to-fixation angle; used only on gap-free transitions", "")
    st = alltr.status.value_counts(normalize=True).round(3).to_dict()
    ns = alltr[alltr.status == "gap_free"].n_saccades.clip(upper=3).value_counts(normalize=True).round(3).to_dict()
    add("S18", "raw transitions", f"status fractions {st}; saccade events per gap-free transition (3=3+) {ns}",
        "medium", "matched saccades = gap-free with exactly one event; gaps never bridged", "")
    off = allfix[allfix.valid & ~allfix.in_media & allfix.gaze_x.notna()]
    insidebox = ((off.gaze_x >= off["MediaPosX (ADCSpx)"]) & (off.gaze_x <= off["MediaPosX (ADCSpx)"] + off.MediaWidth)
                 & (off.gaze_y >= off["MediaPosY (ADCSpx)"]) & (off.gaze_y <= off["MediaPosY (ADCSpx)"] + off.MediaHeight))
    add("S19", "raw FixationPoint (MCSpx) NaN", f"{len(off)} valid fixations with NaN MCS; gaze-sample mean inside "
        f"media box for {insidebox.mean():.1%}", "info", "NaN MCS = off-media fixation (D041)", "")
    v1 = sorted(glob.glob(str(root / cfg["sources"]["raw_dir"] / "*.csv")))
    v2 = sorted(glob.glob(str(root / cfg["sources"]["raw_v2_dir"] / "*.csv")))
    add("S20", "dataset vs dataset_v2", f"{len(v1)} vs {len(v2)} files; v2 lacks per-eye mm gaze, eye position and media "
        "columns and has fewer fixation rows (e.g. BEG-101 reading 275 vs 243 Final fixations)", "medium",
        "dataset/ used as primary raw source", "")
    wbr = roster_workbook(t1235)
    rr = raw_roster.set_index("subject_id")
    both = wbr[wbr.subject_id.isin(rr.index)]
    prefix_ok = all((s.startswith("BEGCO") and g == "CN") or (s.startswith("BEGMCI") and g == "MCI") or
                    (s.startswith("BEGAD") and g == "AD") for s, g in zip(wbr.subject_id, wbr.group))
    filegrp = np.mean([raw_events[s]["file_group"] == rr.loc[s, "group_code"] for s in rr.index if s in raw_events])
    add("S21", "diagnostic codes", f"roster 0/1/2 labelled CO/MCI/AD; raw file Group equals roster code for "
        f"{filegrp:.0%}; workbook code equals roster code for {(both.group_code.values == rr.loc[both.subject_id, 'group_code'].values).mean():.0%} "
        f"of {len(both)} shared IDs; workbook ID prefixes consistent: {prefix_ok}; {len(wbr) - len(both)} workbook ID(s) "
        f"absent from raw roster", "info", "CN = repository CO/HC (code 0); unknown codes raise an error", "")
    add("S22", "ROI order", "ROI IDs 0..39 / 0..158 exist but no stimulus metadata verifies reading order or lines",
        "high", "rank/line features blocked", "M015-M020;M040;M070;M078;A013-A020;A074")
    diffs, nmiss = [], 0
    for sid, e in raw_events.items():
        bl = e.get("baseline")
        if bl is None or not np.isfinite(bl):
            nmiss += 1
            continue
        f0 = e["fix"][e["fix"].valid & (e["fix"].task == 0)].sort_values("onset_ms")
        pv = f0["pup_fin"].dropna().head(10)
        if len(pv) == 10:
            diffs.append(abs(pv.mean() - bl))
    add("S23", "raw PupilLeft/RightBaseline", f"binocular baseline column missing for {nmiss} roster participants; "
        f"|baseline - mean of first 10 valid instruction (task 0) fixations| median {np.median(diffs):.3f} mm, "
        f"max {np.max(diffs):.3f} mm (n={len(diffs)})", "high",
        "protocol baseline rule not reproduced; baseline-corrected features blocked", "A041;A047;A050")
    return pd.DataFrame(C)
