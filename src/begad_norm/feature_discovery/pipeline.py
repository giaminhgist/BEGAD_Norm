"""Stage implementations. Each stage reads/writes files in the run directory."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from begad_norm.feature_discovery import EXTRACTION_VERSION
from begad_norm.feature_discovery import candidate_freeze as cf
from begad_norm.feature_discovery import source_audit as sa
from begad_norm.feature_discovery.dataset_features import d_table
from begad_norm.feature_discovery.extract_macro import (
    INSUF, OBS, RAW_CROSS_IDS, RAW_MACRO_IDS, UNAV, ZERO, participant_macro,
)
from begad_norm.feature_discovery.extract_micro import (
    CROSS_IDS, EXPECTED_ROIS, MACRO_IDS, MICRO_IDS, STIMULUS_VERSION, macro_values, micro_values, region_key,
)
from begad_norm.feature_discovery.multiplicity import apply_pairwise, wide_table
from begad_norm.feature_discovery.proposal_audit import proposal_audit
from begad_norm.feature_discovery.ranking import pair_specific_reserves, rank_regions, region_evidence, task_ranking
from begad_norm.feature_discovery.registry import combined_registry
from begad_norm.feature_discovery.statistics import build_matrix, contrast_rows, detect_aliases

MICRO_COHORTS = ("ROI64_T1235", "ROI64_T5")
MACRO_COHORTS = ("RAW157", "ROI64_T1235", "ROI64_T5")


class Ctx:
    def __init__(self, root: Path, cfg_path: Path, run_dir: Path):
        self.root = root
        self.cfg_path = cfg_path
        self.cfg = yaml.safe_load(cfg_path.read_text())
        self.run = run_dir
        self.cache = run_dir / "cache"
        self.fig = run_dir / "figures"
        for d in (self.run, self.cache, self.fig):
            d.mkdir(parents=True, exist_ok=True)

    def save(self, name, obj):
        (self.cache / name).write_bytes(pickle.dumps(obj))

    def load(self, name):
        return pickle.loads((self.cache / name).read_bytes())

    def csv(self, df: pd.DataFrame, name: str):
        df.to_csv(self.run / name, index=False, na_rep="NA")


# ----------------------------------------------------------------------------- audit
def stage_audit(c: Ctx):
    inv = sa.inventory(c.cfg, c.root)
    (c.run / "source_manifest.json").write_text(json.dumps(inv, indent=2))
    t1235 = sa.load_workbook(c.root, c.cfg["sources"]["t1235_xlsx"])
    t5 = sa.load_workbook(c.root, c.cfg["sources"]["t5_xlsx"])
    raw_roster = sa.roster_raw(c.cfg, c.root)
    wb_roster = sa.roster_workbook(t1235)
    t5_roster = sa.roster_workbook(t5)
    if set(wb_roster.subject_id) != set(t5_roster.subject_id):
        raise ValueError("T5 and T1235 participant sets differ")
    events = sa.parse_raw_cohort(c.cfg, c.root, raw_roster, c.cache / "raw_events.pkl")
    bad = [s for s, e in events.items() if e["file_followup"]]
    if bad:
        raise ValueError(f"follow-up recordings in baseline roster: {bad}")
    sm = sa.subject_manifest(raw_roster, wb_roster, t5_roster, events)
    c.csv(sm, "subject_manifest.csv")
    c.csv(sa.task_roi_manifest(), "task_roi_manifest.csv")
    sem = sa.semantic_checks(c.cfg, c.root, t1235, t5, events, raw_roster)
    c.csv(sem, "semantic_audit.csv")
    _semantic_md(c, sem, sm)
    c.save("workbooks.pkl", dict(t1235=t1235, t5=t5, raw_roster=raw_roster, wb_roster=wb_roster,
                                 t5_roster=t5_roster))


def _semantic_md(c, sem, sm):
    counts = sm.groupby(["cohort_id", "group"]).size().unstack().reindex(columns=["CN", "MCI", "AD"])
    lines = ["# Semantic audit", "",
             "Every check below was recomputed from the inspected sources in this run (`semantic_audit.csv`).", "",
             "## Cohorts (CN = repository CO/HC, code 0)", "", "```", counts.to_string(), "```", "",
             "RAW157 (raw Tobii event exports) and ROI64 (ROI summary workbooks) are analysed separately: workbook "
             "fixation totals exceed raw event counts, the workbooks were produced by a different recalibrated "
             "pipeline, and T5 differs from T1235 task 5 row by row (S14).", "", "## Checks", ""]
    for r in sem.itertuples():
        lines += [f"### {r.check_id}: {r.affected_sources}", f"- Evidence: {r.measured_evidence}",
                  f"- Severity: {r.severity}", f"- Resolution: {r.resolution}",
                  f"- Blocked IDs: {r.blocked_feature_ids or 'none'}", ""]
    (c.run / "semantic_audit.md").write_text("\n".join(lines))


# ----------------------------------------------------------------------------- propose
def stage_propose(c: Ctx):
    wb = c.load("workbooks.pkl")
    events = pickle.loads((c.cache / "raw_events.pkl").read_bytes())
    prof_dir = c.run / "diagnosis_hidden_profiles"
    prof_dir.mkdir(exist_ok=True)
    # workbook column profiles with diagnosis columns removed
    for name in ("t1235", "t5"):
        d = wb[name].drop(columns=["group", "pat"])
        p = d.describe(percentiles=[.05, .25, .5, .75, .95]).T
        p["n_missing"] = d.isna().sum()
        p["n_le_zero"] = (d <= 0).sum()
        p.to_csv(prof_dir / f"{name}_column_profile.csv")
    allfix = pd.concat([e["fix"] for e in events.values()])
    alltr = pd.concat([e["tr"] for e in events.values()])
    rows = []
    for task, f in allfix[allfix.valid].groupby("task"):
        t = alltr[alltr.task == task]
        rows.append(dict(raw_task=int(task), n_fixations=len(f),
                         dur_ms_q=json.dumps(f.dur_ms.quantile([.1, .25, .5, .75, .9]).round(0).tolist()),
                         in_media_frac=f.in_media.mean(), pupil_both_rows_valid_frac=(
                             f.pup_on.notna() & f.pup_fin.notna()).mean(),
                         transitions=len(t), gap_free_frac=(t.status == "gap_free").mean() if len(t) else np.nan,
                         matched_frac=t.matched.mean() if len(t) else np.nan,
                         multi_saccade_frac=((t.status == "gap_free") & (t.n_saccades >= 2)).mean() if len(t) else np.nan,
                         amp_deg_q=json.dumps(t.amp_deg.quantile([.1, .5, .9]).round(2).tolist()) if len(t) else ""))
    pd.DataFrame(rows).to_csv(prof_dir / "raw_event_profile.csv", index=False)
    d = d_table()
    c.csv(d, "dataset_proposed_features.csv")
    c.csv(proposal_audit(), "proposal_audit.csv")


# ----------------------------------------------------------------------------- freeze
def stage_freeze(c: Ctx):
    inv = json.loads((c.run / "source_manifest.json").read_text())
    rec = cf.write_freeze(c.run, c.cfg_path, inv)
    reg = cf.registry_with_freeze(rec["freeze_id"])
    c.csv(reg, "feature_registry.csv")
    d = d_table(rec["freeze_id"])
    c.csv(d, "dataset_proposed_features.csv")
    shutil.copy(c.cfg_path, c.run / "run_config.yaml")
    from begad_norm.feature_discovery.reporting import write_dataset_catalogue_md
    write_dataset_catalogue_md(c, d, rec)


# ----------------------------------------------------------------------------- extract
def _source_hash(c, key):
    inv = json.loads((c.run / "source_manifest.json").read_text())
    p = c.cfg["sources"][key]
    return next(i["sha256"] for i in inv["items"] if i["path"] == p)[:16]


def raw_macro_long(events, roster) -> pd.DataFrame:
    recs = []
    for r in roster.itertuples():
        e = events[r.subject_id]
        per_task, cross = participant_macro(e["fix"], e["sac"], e["tr"], e["rows"])
        for task, feats in per_task.items():
            for fid, (v, st, n) in feats.items():
                recs.append((r.subject_id, task, fid, v, st, n))
        for fid, (v, st, n) in cross.items():
            recs.append((r.subject_id, RAW_CROSS_IDS[fid], fid, v, st, n))
    return pd.DataFrame(recs, columns=["subject_id", "scope_id", "feature_id", "value", "measurement_status",
                                       "n_events"])


def roi_long(c, cohort, wbdf, roster, roi_cov):
    micro = micro_values(wbdf, cohort)
    macro = macro_values(micro, cohort, list(roster.subject_id), roi_cov)
    return micro, macro


def _expand_micro(micro, roster, cohort):
    tasks = sorted(micro.canonical_task.unique(), key=list(EXPECTED_ROIS).index)
    grid = pd.MultiIndex.from_product(
        [roster.subject_id, [region_key(t, r) for t in tasks for r in range(EXPECTED_ROIS[t])], MICRO_IDS[cohort]],
        names=["subject_id", "region_key", "feature_id"]).to_frame(index=False)
    m = grid.merge(micro, on=["subject_id", "region_key", "feature_id"], how="left")
    parts = m.region_key.str.split("|", expand=True)
    m["canonical_task"] = parts[0]
    m["stimulus_version"] = parts[1]
    m["roi_id"] = parts[2].str.removeprefix("ROI_").astype(int)
    m["measurement_status"] = m["measurement_status"].fillna(UNAV)
    return m


def stage_extract(c: Ctx):
    wb = c.load("workbooks.pkl")
    events = pickle.loads((c.cache / "raw_events.pkl").read_bytes())
    reg = pd.read_csv(c.run / "feature_registry.csv", keep_default_na=False)
    meta = reg.set_index("feature_id")[["origin", "name", "unit"]]
    groups = {}
    for cohort, ro in (("RAW157", wb["raw_roster"]), ("ROI64_T1235", wb["wb_roster"]), ("ROI64_T5", wb["t5_roster"])):
        groups[cohort] = ro.set_index("subject_id")["group"]
    raw = raw_macro_long(events, wb["raw_roster"])
    raw["cohort_id"] = "RAW157"
    raw["dataset_id"] = "raw_tobii_dataset_20250305"
    raw["n_rois"] = np.nan
    raw["expected_rois"] = np.nan
    raw["source_hash"] = "per-file sha256 in source_manifest.json"
    macro_parts, micro_parts = [raw], []
    sens = {}
    for cohort, key, df in (("ROI64_T1235", "t1235_xlsx", wb["t1235"]), ("ROI64_T5", "t5_xlsx", wb["t5"])):
        ro = wb["wb_roster"] if cohort == "ROI64_T1235" else wb["t5_roster"]
        micro, macro = roi_long(c, cohort, df, ro, c.cfg["minimum_roi_coverage_for_task_roi_summaries"])
        for cov in c.cfg["coverage_sensitivity"]:
            sens[(cohort, cov)] = macro_values(micro, cohort, list(ro.subject_id), cov)
        h = _source_hash(c, key)
        mic = _expand_micro(micro, ro, cohort)
        mic["cohort_id"] = cohort
        mic["dataset_id"] = key.replace("_xlsx", "").upper() + ".xlsx"
        mic["source_hash"] = h
        micro_parts.append(mic)
        macro["cohort_id"] = cohort
        macro["dataset_id"] = key.replace("_xlsx", "").upper() + ".xlsx"
        macro["source_hash"] = h
        macro_parts.append(macro)
    macro = pd.concat(macro_parts, ignore_index=True)
    micro = pd.concat(micro_parts, ignore_index=True)
    for df in (macro, micro):
        df["visit_id"] = "baseline"
        df["group"] = [groups[co].get(s) for co, s in zip(df.cohort_id, df.subject_id)]
        df["origin"] = df.feature_id.map(meta["origin"])
        df["feature_name"] = df.feature_id.map(meta["name"])
        df["unit"] = df.feature_id.map(meta["unit"])
        df["extraction_version"] = EXTRACTION_VERSION
    micro["roi_text"] = "text_unavailable"
    micro["n_valid_samples"] = np.nan
    micro["valid_span_seconds"] = np.nan
    mcols = ["dataset_id", "cohort_id", "subject_id", "visit_id", "group", "canonical_task", "stimulus_version",
             "roi_id", "roi_text", "feature_id", "origin", "feature_name", "value", "unit", "measurement_status",
             "n_events", "n_valid_samples", "valid_span_seconds", "source_hash", "extraction_version"]
    acols = ["dataset_id", "cohort_id", "subject_id", "visit_id", "group", "scope_id", "feature_id", "origin",
             "feature_name", "value", "unit", "measurement_status", "n_events", "n_rois", "expected_rois",
             "source_hash", "extraction_version"]
    if micro.duplicated(["cohort_id", "subject_id", "canonical_task", "roi_id", "feature_id"]).any():
        raise ValueError("nonunique micro primary keys")
    if macro.duplicated(["cohort_id", "subject_id", "scope_id", "feature_id"]).any():
        raise ValueError("nonunique macro primary keys")
    c.csv(micro[mcols], "micro_features_long.csv")
    c.csv(macro[acols], "macro_features_long.csv")
    c.save("long.pkl", dict(micro=micro, macro=macro, groups=groups, sens=sens))
    c.csv(_quality(c, wb, events), "quality_features.csv")


def _quality(c, wb, events):
    recs = []
    from begad_norm.feature_discovery.event_contracts import RAW_TASK_TO_CANONICAL
    for r in wb["raw_roster"].itertuples():
        e = events[r.subject_id]
        for code, task in RAW_TASK_TO_CANONICAL.items():
            f = e["fix"][e["fix"]["task_on"] == code]
            fv = f[f.valid]
            tr = e["tr"][e["tr"]["task"] == code] if len(e["tr"]) else e["tr"]
            rows = e["rows"][e["rows"]["Task"] == code]
            span = (fv.offset_ms.max() - fv.onset_ms.min()) if len(fv) else np.nan
            gap_ms = tr.loc[tr.status == "gap", "interval_ms"].clip(lower=0).sum() if len(tr) else 0.0
            recs.append(dict(cohort_id="RAW157", subject_id=r.subject_id, group=r.group, scope_id=task,
                             task_present=len(fv) > 0, n_valid_fixations=len(fv),
                             n_excluded_straddling=int(f.straddles_task.sum()),
                             tracking_valid_fraction=float(((rows.ValidityLeft == 0) & (rows.ValidityRight == 0)).mean())
                             if len(rows) else np.nan,
                             pupil_valid_fraction=float((fv.pup_on.notna() & fv.pup_fin.notna()).mean()) if len(fv) else np.nan,
                             gap_time_fraction=float(gap_ms / span) if span and span > 0 else np.nan,
                             gap_transition_fraction=float((tr.status != "gap_free").mean()) if len(tr) else np.nan,
                             off_media_fixation_fraction=float((~fv.in_media).mean()) if len(fv) else np.nan,
                             baseline_present=bool(np.isfinite(e.get("baseline", np.nan))),
                             missing_roi_fraction=np.nan, imputation_fraction=0.0, n_modalities=2,
                             calibration_quality="not_exported"))
    clean = {"ROI64_T1235": sa.load_workbook(c.root, c.cfg["sources"]["t1235_clean_xlsx"]),
             "ROI64_T5": sa.load_workbook(c.root, c.cfg["sources"]["t5_clean_xlsx"])}
    for cohort, df, ro in (("ROI64_T1235", wb["t1235"], wb["wb_roster"]), ("ROI64_T5", wb["t5"], wb["t5_roster"])):
        from begad_norm.feature_discovery.extract_micro import WB_TASK_TO_CANONICAL
        pcols = [x for x in df.columns if x.startswith(("startPup", "endPup"))]
        cl = clean[cohort]
        for r in ro.itertuples():
            for code in sorted(df.task.unique()):
                task = WB_TASK_TO_CANONICAL[code]
                d = df[(df.pat == r.subject_id) & (df.task == code)]
                dc = cl[(cl.pat == r.subject_id) & (cl.task == code)]
                pv = d[pcols]
                bad = pv.isna() | ((pv <= 0) if cohort == "ROI64_T5" else False)
                recs.append(dict(cohort_id=cohort, subject_id=r.subject_id, group=r.group, scope_id=task,
                                 task_present=len(d) > 0, n_roi_rows=len(d),
                                 missing_roi_fraction=1 - len(d) / EXPECTED_ROIS[task],
                                 pupil_valid_fraction=float((~bad.any(axis=1)).mean()) if len(d) else np.nan,
                                 speech_censored_fraction=float(((d.speechDelay <= 0) | (d.speechDelay >= 10)).mean())
                                 if len(d) else np.nan,
                                 imputation_fraction=0.0,
                                 clean_workbook_rows_dropped=len(d) - len(dc),
                                 clean_workbook_rows_imputed=int((dc.nFixations == 1).sum()),
                                 n_modalities=int(len(d) > 0) + int(d.totalSpokenTime.notna().any())
                                 + int((~bad.any(axis=1)).any()) if len(d) else 0,
                                 calibration_quality="not_exported"))
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------- stats
def _priority(reg_origin):
    def key(fid):
        return (0 if reg_origin.get(fid) == "base160" else 1, fid)
    return key


def run_cohort_level(long_df, key_col, roster_groups, cfg, cohort, level, min_cov, reg_origin):
    roster = roster_groups.reset_index().rename(columns={"index": "subject_id"})
    mat = build_matrix(long_df, roster, key_col)
    # amendment A1 (post-freeze bookkeeping): pivot_table(dropna=False) creates every scope x feature
    # combination; keep only (context, feature) pairs that the extraction actually defines.
    defined = pd.MultiIndex.from_frame(long_df[[key_col, "feature_id"]].drop_duplicates())
    mat = mat.loc[mat.index.isin(defined)]
    aliases = detect_aliases(mat, _priority(reg_origin))
    keep = mat.index.difference(pd.MultiIndex.from_frame(aliases[["context", "alias"]])) if len(aliases) else mat.index
    mat_t = mat.loc[mat.index.isin(keep)]
    st = contrast_rows(mat_t, roster_groups, cfg["minimum_subjects_per_group"], min_cov, key_col)
    fam = f"{cohort}|{level}|pairwise|cov{min_cov:.2f}"
    st = apply_pairwise(st, fam)
    w = wide_table(st, f"{cohort}|{level}|joint|cov{min_cov:.2f}", f"{cohort}|{level}|strict3|cov{min_cov:.2f}")
    if len(aliases):
        al = []
        for a in aliases.itertuples():
            for cn in ("CN_AD", "CN_MCI", "MCI_AD"):
                al.append(dict(scope_or_region_key=a.context, feature_id=a.alias, contrast=cn, eligible=False,
                               status="exact_alias", exclusion_reason=f"alias of {a.representative}"))
        st = pd.concat([st, pd.DataFrame(al)], ignore_index=True)
    st.insert(0, "cohort_id", cohort)
    w.insert(0, "cohort_id", cohort)
    aliases.insert(0, "cohort_id", cohort)
    return dict(mat=mat_t, st=st, w=w, aliases=aliases)


def stage_stats(c: Ctx):
    rec = cf.verify_freeze(c.run, c.cfg_path)
    L = c.load("long.pkl")
    reg = pd.read_csv(c.run / "feature_registry.csv", keep_default_na=False)
    origin = reg.set_index("feature_id")["origin"].to_dict()
    cfg = c.cfg
    res = {}
    for cohort in MACRO_COHORTS:
        m = L["macro"][L["macro"].cohort_id == cohort]
        res[(cohort, "macro")] = run_cohort_level(m, "scope_id", L["groups"][cohort], cfg, cohort, "macro",
                                                   cfg["minimum_group_coverage"], origin)
    for cohort in MICRO_COHORTS:
        m = L["micro"][L["micro"].cohort_id == cohort]
        res[(cohort, "micro")] = run_cohort_level(m, "region_key", L["groups"][cohort], cfg, cohort, "micro",
                                                   cfg["minimum_group_coverage"], origin)
    # region ranking (base, no stability inputs)
    for cohort in MICRO_COHORTS:
        r = res[(cohort, "micro")]
        regions = sorted(r["mat"].index.get_level_values(0).unique())
        w = r["w"]
        ev = region_evidence(w, regions)
        ranked = rank_regions(ev, cfg["roi_top_k"])
        r["ranked"] = ranked
        r["reserves"] = pair_specific_reserves(w, ranked)
        r["tasks"] = task_ranking(ranked, w, EXPECTED_ROIS)
    # coverage sensitivity runs (separately labelled families)
    sens_rows, sens_sel = [], []
    for cov in cfg["coverage_sensitivity"]:
        for cohort in MACRO_COHORTS:
            m = L["macro"][L["macro"].cohort_id == cohort]
            if cohort != "RAW157":
                m = L["sens"][(cohort, cov)].assign(cohort_id=cohort)
            rr = run_cohort_level(m, "scope_id", L["groups"][cohort], cfg, cohort, "macro", cov, origin)
            sens_rows.append(_sens_summary(rr["w"], cohort, "macro", cov))
            sens_sel.append(rr["w"].loc[rr["w"].selected_main, ["cohort_id", "scope_or_region_key", "feature_id"]]
                            .assign(level="macro", coverage=cov))
        for cohort in MICRO_COHORTS:
            m = L["micro"][L["micro"].cohort_id == cohort]
            rr = run_cohort_level(m, "region_key", L["groups"][cohort], cfg, cohort, "micro", cov, origin)
            ev = rank_regions(region_evidence(rr["w"], sorted(rr["mat"].index.get_level_values(0).unique())))
            s = _sens_summary(rr["w"], cohort, "micro", cov)
            s["main_candidate_regions"] = int(ev.main_candidate.sum())
            sens_rows.append(s)
            sens_sel.append(rr["w"].loc[rr["w"].selected_main, ["cohort_id", "scope_or_region_key", "feature_id"]]
                            .assign(level="micro", coverage=cov))
    for k, r in res.items():
        s = _sens_summary(r["w"], k[0], k[1], cfg["minimum_group_coverage"])
        if k[1] == "micro":
            s["main_candidate_regions"] = int(r["ranked"].main_candidate.sum())
        sens_rows.append(s)
    c.csv(pd.DataFrame(sens_rows), "sensitivity_coverage.csv")
    c.csv(pd.concat(sens_sel, ignore_index=True), "sensitivity_coverage_selected.csv")
    c.save("stats.pkl", res)
    _write_multiplicity(c, res, rec)


def _sens_summary(w, cohort, level, cov):
    return dict(cohort_id=cohort, level=level, coverage_threshold=cov,
                label="primary" if abs(cov - 0.60) < 1e-9 else "sensitivity",
                eligible_main_hypotheses=int(w.eligible_main.sum()), selected_main=int(w.selected_main.sum()),
                selected_three_pairs=int(w.selected_three_pairs.sum()),
                joint_fdr_supported=int(w.joint_fdr_supported.sum()))


def _write_multiplicity(c, res, rec):
    fams = []
    for (cohort, level), r in res.items():
        st, w = r["st"], r["w"]
        el = st[st.eligible == True]  # noqa: E712
        for fid, g in el.groupby("pairwise_family_id"):
            fams.append(dict(family_id=fid, kind="pairwise", cohort=cohort, level=level, size=len(g),
                             correction="BH primary; BY sensitivity",
                             origins=g.feature_id.map(lambda x: "dataset100" if x.startswith("D") else "base160")
                             .value_counts().to_dict(),
                             members=[f"{a}@{b}@{cn}" for a, b, cn in zip(g.feature_id, g.scope_or_region_key,
                                                                         g.contrast)]))
        for col, kind in (("joint_family_id", "joint_CN_AD_AND_CN_MCI_maxp"), ("strict3_family_id", "strict3_maxp")):
            g = w[w[col] != ""]
            if len(g):
                fams.append(dict(family_id=g[col].iloc[0], kind=kind, cohort=cohort, level=level, size=len(g),
                                 correction="BH primary; BY sensitivity" if kind.startswith("joint") else "BH",
                                 origins=g.feature_id.map(lambda x: "dataset100" if x.startswith("D") else "base160")
                                 .value_counts().to_dict(),
                                 members=[f"{a}@{b}" for a, b in zip(g.feature_id, g.scope_or_region_key)]))
        if level == "micro":
            rk = r["ranked"]
            t = rk[rk.testable & rk.p_region.notna()]
            fams.append(dict(family_id=f"{cohort}|region|bonferroni_minp_BH", kind="region", cohort=cohort,
                             level=level, size=len(t), correction="BH across testable regions; BY sensitivity",
                             members=list(t.region_key)))
    out = dict(freeze_id=rec["freeze_id"], note="Families are per cohort and level; base160 and dataset100 share "
               "every family. Correlation pruning never changes these families.", families=fams,
               aliases=pd.concat([r["aliases"] for r in res.values()]).to_dict(orient="records"))
    (c.run / "multiplicity_manifest.json").write_text(json.dumps(out, indent=1, default=str))


# ----------------------------------------------------------------------------- stability
def stage_stability(c: Ctx):
    from begad_norm.feature_discovery import stability as stb

    cf.verify_freeze(c.run, c.cfg_path)
    res = c.load("stats.pkl")
    L = c.load("long.pkl")
    cfg = c.cfg
    seed = cfg["seed"]
    cov = cfg["minimum_group_coverage"]
    out = {}
    for (cohort, level), r in res.items():
        name = f"{cohort}|{level}"
        w, mat, groups = r["w"], r["mat"], L["groups"][cohort]
        top = set()
        if level == "micro":
            top = set(r["ranked"].loc[r["ranked"].in_top_k, "region_key"])
        any_pass = w["pass_CN_AD"] | w["pass_CN_MCI"] | w["pass_MCI_AD"]
        show = w["selected_main"] | any_pass | (w["scope_or_region_key"].isin(top) & w["eligible_main"])
        ci = stb.bootstrap_ci(mat, groups, w[show], cfg["bootstrap_effect_ci_resamples"], seed, cov, name)
        regions = sorted(mat.index.get_level_values(0).unique()) if level == "micro" else None
        fs, rs = stb.discovery_bootstrap(mat, groups, w, cfg["bootstrap_discovery_resamples"], seed, cov, name,
                                         regions, cfg["roi_top_k"])
        anchors = set()
        if level == "micro":
            rk = r["ranked"]
            anchors = set(zip(rk.loc[rk.in_top_k, "region_key"], rk.loc[rk.in_top_k, "anchor_feature"]))
        keyset = list(zip(w.scope_or_region_key, w.feature_id))
        loo_mask = w["selected_main"] | pd.Series([k in anchors for k in keyset], index=w.index)
        loo = stb.leave_one_ad(mat, groups, w[loo_mask], cfg["minimum_subjects_per_group"], cov)
        perm_mask = w["selected_main"] & (w["scope_or_region_key"].isin(top) if level == "micro" else True)
        perm = stb.permutation_validation(mat, groups, w[perm_mask], cfg["permutation_validation_resamples"], seed,
                                          name)
        for df in (ci, fs, loo, perm):
            if len(df):
                df.insert(0, "cohort_id", cohort)
        rs.insert(0, "cohort_id", cohort)
        out[(cohort, level)] = dict(ci=ci, fs=fs, rs=rs, loo=loo, perm=perm)
    c.save("stability.pkl", out)
    bs = pd.concat([v["fs"].assign(level=k[1], record="feature") for k, v in out.items()]
                   + [v["rs"].assign(level=k[1], record="region") for k, v in out.items() if len(v["rs"])],
                   ignore_index=True)
    c.csv(bs, "bootstrap_stability.csv")
    c.csv(pd.concat([v["loo"].assign(level=k[1]) for k, v in out.items() if len(v["loo"])], ignore_index=True),
          "ad_influence.csv")
    perm = [v["perm"].assign(level=k[1]) for k, v in out.items() if len(v["perm"])]
    c.csv(pd.concat(perm, ignore_index=True) if perm else pd.DataFrame(
        columns=["cohort_id", "scope_or_region_key", "feature_id", "contrast", "p_primary", "p_permutation",
                 "permutation_method", "n_permutations", "disagree_at_0_20", "level"]), "permutation_validation.csv")


# ----------------------------------------------------------------------------- report
def stage_report(c: Ctx):
    from begad_norm.feature_discovery.reporting import run_report
    run_report(c)
