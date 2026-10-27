"""Selection tables, rankings, figures and the Vietnamese/English results memos."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from begad_norm.feature_discovery.dataset_features import SHORTFALL_NOTE


def write_dataset_catalogue_md(c, d: pd.DataFrame, rec: dict):
    split = d.level.value_counts().to_dict()
    sup = int((d.feasibility_status == "supported").sum())
    intro_en = f"""# Dataset-proposed additional features (D001-D{len(d):03d})

Freeze ID: `{rec['freeze_id']}` ({rec['freeze_time_utc']}). Extension status: **{rec['extension_status']}**
({len(d)} of the 100-definition target; combined catalogue {rec['total_definition_count']} of 260).

## How the inspected data led to these proposals

The repository contains (i) two original ROI summary workbooks (T1235: KD1-3 and reading, 64 participants;
T5: reading, same 64 participants, a different export), with 19 per-ROI columns whose meaning is given by
Table II of the TNSRE 2025 paper, and (ii) 205 raw Tobii Studio event exports (157 baseline CN/MCI/AD
participants) holding fixation and saccade events as onset/offset row pairs with timestamps, fixation
positions, fixation-to-fixation amplitudes, per-eye gaze, eye distance and pupil at event boundaries.
There are no ROI bounding boxes, stimulus text, reading order, audio, TextGrids, error annotations, lexical
metadata, continuous pupil samples or media onset events.

The D definitions therefore use only what these sources measure, after the label-blind checks in
`semantic_audit.md`: constant-free workbook pupil contrasts (D001, D002), the export's direction-based and
amplitude-sum saccade fields as stand-alone counts (D003, D004), a joint gaze-speech ratio (D005),
target-set summaries and within-person couplings of the workbook fields (D007-D016), and raw event-sequence,
saccade dynamics, direction, spatial layout, event-sampled pupil, binocular and postural measures
(D017-D049). Profiles used for invention exclude diagnosis (`diagnosis_hidden_profiles/`).

Level split: {split}. Supported definitions: {sup}; defined but blocked: {len(d) - sup}
(D006, exit-landing speech offset delay, because the fixation time point is undocumented).

**Shortfall.** {SHORTFALL_NOTE}

Rejected, aliased and deferred candidates are kept in `proposal_audit.csv`. Feasibility here is a source and
semantics statement, not a statistical result.

## Definitions
"""
    intro_vi = f"""# Các feature bổ sung đề xuất từ dataset (D001-D{len(d):03d})

Freeze ID: `{rec['freeze_id']}` ({rec['freeze_time_utc']}). Trạng thái mở rộng: **{rec['extension_status']}**
({len(d)} trên mục tiêu 100; catalogue kết hợp {rec['total_definition_count']} trên 260).

## Dữ liệu đã kiểm tra dẫn tới các đề xuất này

Repository có (i) hai workbook tóm tắt ROI gốc (T1235: KD1-3 và đọc, 64 người; T5: đọc, cùng 64 người,
một bản export khác) với 19 cột theo ROI, ý nghĩa lấy từ Bảng II của bài báo TNSRE 2025, và (ii) 205 file
export sự kiện thô Tobii Studio (157 người baseline CN/MCI/AD) chứa fixation và saccade dưới dạng cặp dòng
onset/offset có timestamp, vị trí fixation, biên độ fixation-tới-fixation, gaze từng mắt, khoảng cách mắt và
pupil tại biên sự kiện. Không có bounding box ROI, văn bản kích thích, thứ tự đọc, audio, TextGrid, chú thích
lỗi, metadata từ vựng, chuỗi pupil liên tục hay sự kiện onset của media.

Vì vậy các định nghĩa D chỉ dùng những gì nguồn này đo được, sau các kiểm tra không nhìn nhãn trong
`semantic_audit.md`: tương phản pupil không phụ thuộc hằng số của workbook (D001, D002), các trường saccade theo
hướng và tổng biên độ dùng như số đếm độc lập (D003, D004), một tỉ số gaze-speech (D005), tóm tắt trên tập
target và tương quan trong cùng người của các trường workbook (D007-D016), và các đo lường chuỗi sự kiện thô,
động học saccade, hướng, bố cục không gian, pupil lấy mẫu theo sự kiện, hai mắt và tư thế (D017-D049).
Hồ sơ dùng để đề xuất đã loại chẩn đoán (`diagnosis_hidden_profiles/`).

Phân bố mức: {split}. Định nghĩa đủ nguồn: {sup}; đã định nghĩa nhưng bị chặn: {len(d) - sup}
(D006, độ trễ kết thúc nói theo fixation hạ cánh sau khi rời ROI, vì thời điểm fixation không được ghi rõ).

**Thiếu hụt.** Chỉ viết được {len(d)} định nghĩa có cơ sở. Repository không có hình học ROI, văn bản kích thích,
thứ tự đọc, audio/TextGrid hay chú thích lỗi, và export thô chỉ giữ pupil tại biên sự kiện; workbook ROI có 19
cột tóm tắt, trong đó một số cột chưa thống nhất về ngữ nghĩa. Các ứng viên cần những nguồn đó, hoặc chỉ là biến
thể đổi tên/tương đương thứ hạng, đã bị loại (xem `proposal_audit.csv`).

## Định nghĩa (giữ nguyên tiếng Anh kỹ thuật)
"""
    body = []
    for r in d.itertuples():
        body += [f"### {r.feature_id} `{r.name}` ({r.level}, {r.family})",
                 f"- Formula: {r.formula}", f"- Unit: {r.unit}; scope: {r.scope}",
                 f"- Sources: {r.source_paths} [{r.source_columns}]",
                 f"- Semantics evidence: {r.source_semantics_evidence}",
                 f"- Dataset motivation: {r.dataset_motivation}",
                 f"- Closest existing: {r.closest_existing_ids or 'none'}; incremental: {r.incremental_information}",
                 f"- Support: {r.within_subject_support}" + (f"; conditioning: {r.conditioning}" if r.conditioning else ""),
                 f"- Zero/missing/censoring: {r.zero_rule or 'n/a'} / {r.missing_rule} / {r.censoring_rule}",
                 f"- Parameters: {r.parameters or 'none'} ({r.parameter_origin})",
                 f"- Feasibility: {r.feasibility_status}" + (f" ({r.blocked_reason})" if r.blocked_reason else "")
                 + f"; datasets: {r.supported_datasets or 'none'}", ""]
    (c.run / "DATASET_PROPOSED_100_FEATURES.md").write_text(intro_en + "\n" + "\n".join(body))
    (c.run / "DATASET_PROPOSED_100_FEATURES.vi.md").write_text(intro_vi + "\n" + "\n".join(body))


# =============================================================================
# Report stage
# =============================================================================
GROUP_COLORS = {"CN": "#2a78d6", "MCI": "#eb6834", "AD": "#1baf7a"}   # reference palette slots 1-3
GROUP_MARKERS = {"CN": "o", "MCI": "s", "AD": "^"}
CONTRAST_STYLE = {"CN_AD": ("#2a78d6", "o"), "CN_MCI": ("#eb6834", "s"), "MCI_AD": ("#1baf7a", "^")}
LOW_CONF = {"M051", "M061", "A061", "A062", "A063", "A079", "D001", "D002", "D003", "D004", "D005", "D010",
            "D011", "D012"}


def _tier(r):
    t = []
    if r.get("joint_fdr_supported"):
        t.append("joint_FDR_supported")
    if r.get("region_fdr_supported"):
        t.append("region_FDR_supported")
    if r.get("stable_candidate"):
        t.append("stable_candidate")
    return ";".join(t) or "exploratory_only"


def enrich(w, stab, reg, cohort, level):
    """Join CI, bootstrap, AD influence and permutation results onto the wide table."""
    w = w.copy()
    meta = reg.set_index("feature_id")
    w["origin"] = w.feature_id.map(meta["origin"])
    w["level"] = level
    w["name"] = w.feature_id.map(meta["name"])
    w["unit"] = w.feature_id.map(meta["unit"])
    w["family"] = w.feature_id.map(meta["family"])
    w["semantic_flags"] = w.feature_id.map(meta["semantic_flags"]).fillna("")
    w["conditioning"] = w.feature_id.map(meta["conditioning"]).fillna("")
    w["semantic_confidence"] = np.where(w.feature_id.isin(LOW_CONF), "export_convention", "verified_events")
    k = ["scope_or_region_key", "feature_id"]
    ci = stab["ci"]
    if len(ci):
        for c in ("CN_AD", "CN_MCI", "MCI_AD"):
            x = ci[ci.contrast == c][k + ["ci_low", "ci_high", "ci_status", "ci_valid_reps"]]
            x.columns = k + [f"ci_low_{c}", f"ci_high_{c}", f"ci_status_{c}", f"ci_valid_reps_{c}"]
            w = w.merge(x, on=k, how="left")
    fs = stab["fs"].drop(columns=["cohort_id"], errors="ignore")
    w = w.merge(fs, on=k, how="left")
    loo = stab["loo"]
    if len(loo):
        w = w.merge(loo.drop(columns=["cohort_id"], errors="ignore"), on=k, how="left")
    pm = stab["perm"]
    if len(pm):
        for c in ("CN_AD", "CN_MCI", "MCI_AD"):
            x = pm[pm.contrast == c][k + ["p_permutation", "disagree_at_0_20"]]
            x.columns = k + [f"p_perm_{c}", f"perm_disagree_{c}"]
            w = w.merge(x, on=k, how="left")
    for col in ("direction_stable", "selection_stable"):
        w[col] = w[col].fillna(False).astype(bool)
    w["stable_candidate"] = w["selected_main"] & w["effect_preferred"] & w["direction_stable"] & w["selection_stable"]
    w["region_fdr_supported"] = False
    return w


SEL_COLS = ["cohort_id", "scope_or_region_key", "feature_id", "origin", "level", "name", "unit", "family",
            "p_CN_AD", "p_CN_MCI", "p_MCI_AD", "delta_CN_AD", "delta_CN_MCI", "delta_MCI_AD",
            "ci_low_CN_AD", "ci_high_CN_AD", "ci_low_CN_MCI", "ci_high_CN_MCI", "ci_low_MCI_AD", "ci_high_MCI_AD",
            "ci_status_CN_AD", "ci_status_CN_MCI", "q_bh_CN_AD", "q_bh_CN_MCI", "q_bh_MCI_AD", "q_by_CN_AD",
            "q_by_CN_MCI", "p_joint", "q_joint_bh", "q_joint_by", "joint_family_size", "q_joint3_bh",
            "n1_CN_AD", "n2_CN_AD", "n2_CN_MCI", "coverage_CN", "coverage_MCI", "coverage_AD",
            "selected_main", "selected_three_pairs", "pass_CN_AD", "pass_CN_MCI", "pass_MCI_AD",
            "pairwise_fdr_supported", "joint_fdr_supported", "joint_fdr_supported_by", "effect_preferred",
            "direction_agreement_both_controls", "direction_defined_reps", "direction_stable",
            "selection_freq_main", "selection_freq_three_pairs", "selection_stable",
            "prop_main_screen_preserved", "sign_flips_CN_AD", "ineligible_omissions_CN_AD",
            "p_perm_CN_AD", "p_perm_CN_MCI", "p_perm_MCI_AD", "perm_disagree_CN_AD", "perm_disagree_CN_MCI",
            "direction_pattern", "semantic_confidence", "semantic_flags", "conditioning", "evidence_tier"]


def _cols(df):
    for c in SEL_COLS:
        if c not in df:
            df[c] = np.nan
    return df[SEL_COLS]


def _spearman_pairs(mat, groups, keys, min_n=8):
    from scipy import stats as ss
    g = groups.reindex(mat.columns)
    recs = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = mat.loc[keys[i]].to_numpy(float)
            b = mat.loc[keys[j]].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            n = int(ok.sum())
            rec = dict(a=f"{keys[i][1]}@{keys[i][0]}", b=f"{keys[j][1]}@{keys[j][0]}", n_paired=n,
                       n_CN=int((ok & (g == "CN")).sum()), n_MCI=int((ok & (g == "MCI")).sum()),
                       n_AD=int((ok & (g == "AD")).sum()), spearman=np.nan, spearman_within_group_centered=np.nan)
            if n >= min_n and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0:
                rec["spearman"] = ss.spearmanr(a[ok], b[ok]).statistic
                ac = pd.Series(a).groupby(g.values).transform(lambda s: s - s.median()).to_numpy()
                bc = pd.Series(b).groupby(g.values).transform(lambda s: s - s.median()).to_numpy()
                rec["spearman_within_group_centered"] = ss.spearmanr(ac[ok], bc[ok]).statistic
            recs.append(rec)
    return pd.DataFrame(recs, columns=["a", "b", "n_paired", "n_CN", "n_MCI", "n_AD", "spearman",
                                       "spearman_within_group_centered"])


def compact_panel(sel, corr, thr=0.90):
    cand = sel[sel.effect_preferred & sel.direction_stable & sel.selection_stable].copy()
    cand["_conf"] = np.where(cand.semantic_confidence == "verified_events", 0, 1)
    cand["_cov"] = cand[["coverage_CN", "coverage_MCI", "coverage_AD"]].min(axis=1)
    cand = cand.sort_values(["_conf", "_cov", "q_joint_bh", "min_abs_control_delta", "feature_id"],
                            ascending=[True, False, True, False, True])
    lookup = {}
    for r in corr.itertuples():
        lookup[(r.a, r.b)] = lookup[(r.b, r.a)] = (r.spearman, r.n_paired)
    kept, notes = [], []
    for r in cand.itertuples():
        me = f"{r.feature_id}@{r.scope_or_region_key}"
        clash = [(k, lookup.get((me, k))) for k in kept if lookup.get((me, k)) and
                 np.isfinite(lookup[(me, k)][0]) and abs(lookup[(me, k)][0]) >= thr]
        undefined = [k for k in kept if not lookup.get((me, k)) or not np.isfinite(lookup[(me, k)][0])]
        if clash:
            notes.append(dict(item=me, decision="excluded_correlated", with_=clash[0][0], spearman=clash[0][1][0],
                              n_paired=clash[0][1][1], undefined_correlations=";".join(undefined)))
        else:
            kept.append(me)
            notes.append(dict(item=me, decision="kept", with_="", spearman=np.nan, n_paired=np.nan,
                              undefined_correlations=";".join(undefined)))
    keep_keys = set(kept)
    panel = cand[[f"{a}@{b}" in keep_keys for a, b in zip(cand.feature_id, cand.scope_or_region_key)]]
    return panel, pd.DataFrame(notes, columns=["item", "decision", "with_", "spearman", "n_paired",
                                               "undefined_correlations"])


def run_report(c):
    from begad_norm.feature_discovery.extract_micro import EXPECTED_ROIS
    res = c.load("stats.pkl")
    stab = c.load("stability.pkl")
    L = c.load("long.pkl")
    reg = pd.read_csv(c.run / "feature_registry.csv", keep_default_na=False)
    rec = json.loads((c.run / "candidate_freeze.json").read_text())
    W = {k: enrich(r["w"], stab[k], reg, k[0], k[1]) for k, r in res.items()}
    # region info for micro
    for k in [k for k in res if k[1] == "micro"]:
        rk = res[k]["ranked"].merge(stab[k]["rs"].drop(columns=["cohort_id"]), on="region_key", how="left")
        rk["top_k_frequency"] = rk["top_k_frequency"].fillna(0.0)
        res[k]["ranked"] = rk
        rq = rk.set_index("region_key")["q_region_bh"]
        w = W[k]
        w["q_region_bh"] = w.scope_or_region_key.map(rq)
        w["region_fdr_supported"] = w["q_region_bh"] <= 0.10
    for w in W.values():
        w["evidence_tier"] = [_tier(r) for r in w.to_dict("records")]
    # ----------------------------------------------------------- stats_all
    for level in ("macro", "micro"):
        parts = []
        for k, r in res.items():
            if k[1] != level:
                continue
            st = r["st"].copy()
            w = W[k]
            keys = ["scope_or_region_key", "feature_id"]
            st = st.merge(w[keys + ["p_joint", "joint_family_id", "joint_family_size", "q_joint_bh", "q_joint_by",
                                    "selected_main", "selected_three_pairs", "origin", "conditioning",
                                    "semantic_confidence"]], on=keys, how="left")
            ci = stab[k]["ci"]
            if len(ci):
                st = st.merge(ci.drop(columns=["cohort_id"]), on=keys + ["contrast"], how="left")
            st["ci_status"] = st.get("ci_status", pd.Series(index=st.index, dtype=object)).fillna("ci_not_computed")
            st["selected_this_pair"] = st["eligible"].fillna(False).astype(bool) & (st["p_primary"] < 0.20)
            st["semantic_status"] = st["semantic_confidence"].fillna("exact_alias")
            parts.append(st)
        blocked = reg[(reg.level == level) & (reg.status != "supported")]
        parts.append(pd.DataFrame(dict(cohort_id="ALL", scope_or_region_key="ALL", feature_id=blocked.feature_id,
                                       origin=blocked.origin, contrast="ALL", eligible=False, status=blocked.status,
                                       exclusion_reason=blocked.status_reason, ci_status="ci_not_computed",
                                       selected_main=False, selected_three_pairs=False, selected_this_pair=False,
                                       semantic_status=blocked.status)))
        cols = ["cohort_id", "scope_or_region_key", "feature_id", "origin", "contrast", "eligible", "status",
                "exclusion_reason", "n1", "n2", "expected_n1", "expected_n2", "coverage1", "coverage2", "median1",
                "q1_1", "q3_1", "mean1", "sd1", "median2", "q1_2", "q3_2", "mean2", "sd2", "u_statistic",
                "p_primary", "test_method", "cliffs_delta", "ci_low", "ci_high", "ci_status", "pairwise_family_id",
                "pairwise_family_size", "q_bh", "q_by", "p_joint", "joint_family_id", "joint_family_size",
                "q_joint_bh", "q_joint_by", "selected_main", "selected_three_pairs", "selected_this_pair",
                "conditioning", "semantic_status"]
        out = pd.concat(parts, ignore_index=True)
        for col in cols:
            if col not in out:
                out[col] = np.nan
        c.csv(out[cols], f"{level}_stats_all.csv")
    # ----------------------------------------------------------- macro tables
    mac = pd.concat([W[k] for k in W if k[1] == "macro"], ignore_index=True)
    mac_main = mac[mac.selected_main].sort_values(["cohort_id", "p_joint"])
    c.csv(_cols(mac_main.copy()), "macro_selected_main.csv")
    c.csv(_cols(mac[mac.selected_three_pairs].sort_values(["cohort_id", "p_joint"]).copy()),
          "macro_selected_three_pairs.csv")
    ps = []
    for cn in ("CN_AD", "CN_MCI", "MCI_AD"):
        x = mac[mac[f"pass_{cn}"]].copy()
        x["passing_contrast"] = cn
        ps.append(x)
    mps = pd.concat(ps, ignore_index=True)
    c.csv(_cols(mps.copy()).assign(passing_contrast=mps.passing_contrast.values), "macro_pair_specific.csv")
    red, panels = [], []
    for k in [k for k in res if k[1] == "macro"]:
        w = W[k]
        sel = w[w.selected_main]
        keys = list(zip(sel.scope_or_region_key, sel.feature_id))
        corr = _spearman_pairs(res[k]["mat"], L["groups"][k[0]], keys)
        corr.insert(0, "cohort_id", k[0])
        corr.insert(1, "level", "macro")
        red.append(corr.assign(record="correlation"))
        panel, notes = compact_panel(sel, corr)
        panels.append(panel)
        red.append(notes.assign(cohort_id=k[0], level="macro", record="compact_panel_pruning"))
    panel = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    c.csv(_cols(panel.copy()) if len(panel) else _cols(pd.DataFrame()), "macro_compact_panel.csv")
    # ----------------------------------------------------------- micro tables
    mic = pd.concat([W[k] for k in W if k[1] == "micro"], ignore_index=True)
    rk_all = pd.concat([res[k]["ranked"].assign(cohort_id=k[0]) for k in res if k[1] == "micro"], ignore_index=True)
    top = rk_all[rk_all.in_top_k]
    topkeys = set(zip(top.cohort_id, top.region_key))
    mic["in_top20_region"] = [(a, b) in topkeys for a, b in zip(mic.cohort_id, mic.scope_or_region_key)]
    mic["base_rank"] = [rk_all.set_index(["cohort_id", "region_key"]).base_rank.get((a, b), np.nan)
                        for a, b in zip(mic.cohort_id, mic.scope_or_region_key)]
    msel = mic[mic.selected_main].sort_values(["cohort_id", "base_rank", "p_joint"])
    extra = ["in_top20_region", "base_rank", "q_region_bh"]
    c.csv(pd.concat([_cols(msel.copy()), msel[extra].reset_index(drop=True).set_index(_cols(msel.copy()).index)],
                    axis=1), "micro_selected_main.csv")
    mt = msel[msel.in_top20_region]
    c.csv(pd.concat([_cols(mt.copy()), mt[extra]], axis=1), "micro_selected_top20.csv")
    ps = []
    for cn in ("CN_AD", "CN_MCI", "MCI_AD"):
        x = mic[mic[f"pass_{cn}"]].copy()
        x["passing_contrast"] = cn
        ps.append(x)
    mps2 = pd.concat(ps, ignore_index=True)
    c.csv(_cols(mps2.copy()).assign(passing_contrast=mps2.passing_contrast.values), "micro_pair_specific.csv")
    # micro redundancy within top regions
    for k in [k for k in res if k[1] == "micro"]:
        tk = mt[mt.cohort_id == k[0]]
        keys = list(zip(tk.scope_or_region_key, tk.feature_id))
        corr = _spearman_pairs(res[k]["mat"], L["groups"][k[0]], keys)
        corr.insert(0, "cohort_id", k[0])
        corr.insert(1, "level", "micro_top20")
        red.append(corr.assign(record="correlation"))
    al = pd.concat([r["aliases"] for r in res.values()], ignore_index=True)
    red.append(al.rename(columns={"representative": "a", "alias": "b", "context": "item"})
               .assign(record="proven_alias_collapsed_before_testing"))
    c.csv(pd.concat(red, ignore_index=True), "redundancy_audit.csv")
    # ----------------------------------------------------------- ROI ranking
    anchors = mic.set_index(["cohort_id", "scope_or_region_key", "feature_id"])
    rows = []
    for r in rk_all.itertuples():
        a = anchors.loc[(r.cohort_id, r.region_key, r.anchor_feature)] if isinstance(r.anchor_feature, str) else None
        sel_f = mic[(mic.cohort_id == r.cohort_id) & (mic.scope_or_region_key == r.region_key) & mic.selected_main]
        status = ("main_candidate" if r.main_candidate else "testable_not_selected" if r.testable
                  else "untestable_no_feature_eligible_for_both_control_contrasts")
        rows.append(dict(cohort_id=r.cohort_id, region_key=r.region_key, canonical_task=r.canonical_task,
                         stimulus_version=r.stimulus_version, roi_id=r.roi_id, roi_text="text_unavailable",
                         status=status, base_rank=r.base_rank, in_top20=r.in_top_k, m_r=r.m_r, p_region=r.p_region,
                         q_region_bh=r.q_region_bh, q_region_by=r.q_region_by, anchor_feature=r.anchor_feature,
                         anchor_origin=None if a is None else a["origin"],
                         anchor_p_CN_AD=None if a is None else a["p_CN_AD"],
                         anchor_p_CN_MCI=None if a is None else a["p_CN_MCI"],
                         anchor_p_MCI_AD=None if a is None else a["p_MCI_AD"],
                         anchor_delta_CN_AD=None if a is None else a["delta_CN_AD"],
                         anchor_delta_CN_MCI=None if a is None else a["delta_CN_MCI"],
                         anchor_delta_MCI_AD=None if a is None else a["delta_MCI_AD"],
                         anchor_min_coverage=getattr(r, "anchor_min_coverage", np.nan),
                         anchor_effect_preferred=None if a is None else a["effect_preferred"],
                         anchor_direction_stable=None if a is None else a["direction_stable"],
                         anchor_selection_freq=None if a is None else a["selection_freq_main"],
                         anchor_loo_prop_preserved=None if a is None else a.get("prop_main_screen_preserved"),
                         top20_bootstrap_frequency=r.top_k_frequency,
                         stable_candidate=False if a is None else bool(a["stable_candidate"]),
                         n_main_selected_features=len(sel_f),
                         main_selected_features=";".join(sel_f.sort_values("p_joint").feature_id),
                         origins_selected=";".join(sorted(set(sel_f.origin)))))
    ra = pd.DataFrame(rows)
    ra["primary_or_sensitivity"] = np.where(ra.cohort_id == "ROI64_T1235", "primary_KD_plus_reading",
                                            "reading_only_sensitivity_T5")
    c.csv(ra.sort_values(["cohort_id", "base_rank", "region_key"], na_position="last"), "roi_ranking_all.csv")
    c.csv(ra[ra.in_top20].sort_values(["cohort_id", "base_rank"]), "roi_top20.csv")
    resv = pd.concat([res[k]["reserves"].assign(cohort_id=k[0]) for k in res if k[1] == "micro"], ignore_index=True)
    c.csv(resv, "roi_pair_specific_reserves.csv")
    tasks = pd.concat([res[k]["tasks"].assign(cohort_id=k[0]) for k in res if k[1] == "micro"], ignore_index=True)
    c.csv(tasks, "task_ranking.csv")
    # ----------------------------------------------------------- origin summary
    osum = []
    for (cohort, level), w in W.items():
        for origin in ("base160", "dataset100"):
            o = w[w.origin == origin]
            rg = reg[(reg.origin == origin) & (reg.level == level)]
            osum.append(dict(cohort_id=cohort, level=level, origin=origin, defined=len(rg),
                             available_definitions=int(rg.supported_datasets.str.contains(cohort).sum()),
                             hypotheses_extracted=len(o), testable_main=int(o.eligible_main.sum()),
                             selected_main=int(o.selected_main.sum()),
                             selected_three_pairs=int(o.selected_three_pairs.sum()),
                             pass_MCI_AD=int(o.pass_MCI_AD.sum()), joint_fdr_supported=int(o.joint_fdr_supported.sum()),
                             stable_candidate=int(o.stable_candidate.sum()),
                             in_top20_selected=int(o.get("in_top20_region", pd.Series(False, index=o.index))[o.selected_main].sum())
                             if level == "micro" else np.nan))
    c.csv(pd.DataFrame(osum), "feature_origin_summary.csv")
    W_all = dict(mac=mac, mic=mic, ra=ra, tasks=tasks, resv=resv, panel=panel)
    c.save("report.pkl", W_all)
    make_figures(c, mac, mic, ra, res, L)
    write_memos(c, rec, reg, mac, mic, ra, tasks, resv, panel, res)
    meta_f = c.run / "run_metadata.json"
    meta = json.loads(meta_f.read_text())
    meta["run_status"] = "partial_sources"
    meta["run_status_reason"] = ("ROI geometry/reading order, audio/speech annotations, lexical metadata, continuous "
                                 "pupil and media onset events are absent; supported analyses completed")
    meta["seed"] = c.cfg["seed"]
    meta_f.write_text(json.dumps(meta, indent=1))


def _save(fig, c, name):
    fig.savefig(c.fig / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(c.fig / f"{name}.pdf", bbox_inches="tight")


def make_figures(c, mac, mic, ra, res, L):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})
    # 1. availability heatmap: testable (main-eligible) hypotheses by family x cohort/task
    allw = pd.concat([mac, mic], ignore_index=True)
    allw["task"] = allw.scope_or_region_key.str.split("|").str[0]
    allw["col"] = allw.cohort_id + "\n" + allw.task
    tab = allw.groupby(["family", "col"]).eligible_main.mean().unstack()
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * tab.shape[1], 0.3 * tab.shape[0] + 1.5))
    im = ax.imshow(tab.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns, fontsize=7)
    ax.set_yticks(range(tab.shape[0]), tab.index, fontsize=7)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="#0b0b0b" if (np.isnan(v) or v < .6) else "white")
    g = {k: L["groups"][k].value_counts().reindex(["CN", "MCI", "AD"]).tolist() for k in L["groups"]}
    ax.set_title("Fraction of extracted hypotheses eligible for both control contrasts\n"
                 + "; ".join(f"{k}: CN/MCI/AD={v[0]}/{v[1]}/{v[2]}" for k, v in g.items())
                 + "\n'n/a' = family not defined for that cohort/task (blocked or not applicable)", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.6, label="fraction eligible")
    _save(fig, c, "fig1_availability_heatmap")
    plt.close(fig)
    # 2. macro forest plot
    sel = mac[mac.selected_main].sort_values(["cohort_id", "p_joint"]).reset_index(drop=True)
    if len(sel):
        fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(sel) + 1.4))
        y = np.arange(len(sel))[::-1]
        for k, (cn, (col, mk)) in enumerate(CONTRAST_STYLE.items()):
            off = (k - 1) * 0.22
            d = sel[f"delta_{cn}"].to_numpy(float)
            lo = sel.get(f"ci_low_{cn}", pd.Series(np.nan, index=sel.index)).to_numpy(float)
            hi = sel.get(f"ci_high_{cn}", pd.Series(np.nan, index=sel.index)).to_numpy(float)
            ax.errorbar(d, y + off, xerr=[d - lo, hi - d], fmt=mk, color=col, ms=4, lw=1.2, capsize=0,
                        label=cn.replace("_", " vs "))
        ax.axvline(0, color="#52514e", lw=.8)
        ax.set_yticks(y, [f"{a} {b}@{s}" for a, b, s in zip(sel.cohort_id, sel.feature_id, sel.scope_or_region_key)],
                      fontsize=6.5)
        ax.set_xlabel("Cliff's delta (positive = second-named group higher), 95% participant-bootstrap interval")
        ax.set_title("Main-selected macro contexts (exploratory raw p<0.20 in CN-AD and CN-MCI)\n"
                     "marginal intervals, not post-selection corrected", fontsize=8)
        ax.legend(fontsize=7, loc="lower right", frameon=False)
        ax.grid(axis="x", color="#e6e5e0", lw=.5)
        _save(fig, c, "fig2_macro_forest")
        plt.close(fig)
    # 3. task-ROI evidence map
    for cohort in ra.cohort_id.unique():
        r = ra[ra.cohort_id == cohort]
        tasks = [t for t in ("KD1", "KD2", "KD3", "READING") if t in set(r.canonical_task)]
        fig, axes = plt.subplots(len(tasks), 1, figsize=(10, 1.1 * len(tasks) + .8), squeeze=False,
                                 gridspec_kw={"hspace": 0.6})
        for ax, t in zip(axes[:, 0], tasks):
            rt = r[r.canonical_task == t].sort_values("roi_id")
            n = {"KD1": 40, "KD2": 40, "KD3": 40, "READING": 159}[t]
            v = np.full(n, np.nan)
            v[rt.roi_id.to_numpy()] = -np.log10(rt.p_region.astype(float).to_numpy())
            ax.imshow(v[None, :], cmap="Blues", aspect="auto", vmin=0, vmax=3,
                      extent=(-.5, n - .5, 0, 1))
            un = np.setdiff1d(np.arange(n), rt.loc[rt.p_region.notna(), "roi_id"].to_numpy())
            ax.scatter(un, np.full(len(un), .5), marker="x", s=8, color="#9a9890", lw=.7)
            tt = rt[rt.in_top20]
            ax.scatter(tt.roi_id, np.full(len(tt), .5), marker="o", s=26, facecolors="none", edgecolors="#eb6834",
                       lw=1.2)
            ax.set_yticks([])
            ax.set_ylabel(t, rotation=0, ha="right", va="center")
            ax.set_xlim(-.5, n - .5)
        axes[-1, 0].set_xlabel("ROI ID (order of IDs; NOT verified reading order; text unavailable)")
        fig.suptitle(f"{cohort}: region evidence -log10(p_region) (Bonferroni-min within region); "
                     "x = untestable, orange ring = top-20", fontsize=8)
        _save(fig, c, f"fig3_task_roi_evidence_{cohort}")
        plt.close(fig)
    # 4. top-20 bootstrap frequency and AD influence
    t = ra[ra.in_top20].sort_values(["cohort_id", "base_rank"])
    if len(t):
        fig, ax = plt.subplots(figsize=(7, 0.26 * len(t) + 1.2))
        y = np.arange(len(t))[::-1]
        ax.barh(y, t.top20_bootstrap_frequency, color="#2a78d6", height=.6)
        flag = t.anchor_loo_prop_preserved.astype(float) < 1
        ax.scatter(np.full(flag.sum(), 1.02), y[flag.to_numpy()], marker="^", color="#eb6834", s=18,
                   label="anchor loses main screen in >=1 leave-one-AD run", clip_on=False)
        ax.set_yticks(y, [f"#{int(b)} {c_.split('_')[-1]} {k.split('|')[0]} ROI {k.split('_')[-1]} ({a})" for b, c_, k, a in
                          zip(t.base_rank, t.cohort_id, t.region_key, t.anchor_feature)], fontsize=6.5)
        ax.set_xlim(0, 1)
        ax.axvline(.6, color="#52514e", lw=.6, ls="--")
        ax.set_xlabel("Top-20 membership frequency over 500 participant bootstrap replicates")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        _save(fig, c, "fig4_top20_bootstrap")
        plt.close(fig)
    # 5. participant distributions for representative features
    reps = []
    for cohort in ("RAW157", "ROI64_T1235"):
        s = mac[(mac.cohort_id == cohort) & mac.selected_main].sort_values("p_joint").head(2)
        reps += [(cohort, "macro", r.scope_or_region_key, r.feature_id) for r in s.itertuples()]
    s = ra[(ra.cohort_id == "ROI64_T1235") & ra.in_top20].sort_values("base_rank").head(2)
    reps += [("ROI64_T1235", "micro", r.region_key, r.anchor_feature) for r in s.itertuples()]
    if reps:
        fig, axes = plt.subplots(1, len(reps), figsize=(2.3 * len(reps), 2.8))
        axes = np.atleast_1d(axes)
        rng = np.random.default_rng(0)
        for ax, (cohort, level, ctx, fid) in zip(axes, reps):
            x = res[(cohort, level)]["mat"].loc[(ctx, fid)]
            g = L["groups"][cohort].reindex(x.index)
            for i, grp in enumerate(("CN", "MCI", "AD")):
                v = x[g == grp].to_numpy(float)
                nv = np.isfinite(v).sum()
                v = v[np.isfinite(v)]
                ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=9, alpha=.75, color=GROUP_COLORS[grp],
                           marker=GROUP_MARKERS[grp], lw=0)
                if len(v):
                    ax.hlines(np.median(v), i - .3, i + .3, color="#0b0b0b", lw=1.5)
                ax.text(i, 1.02, f"n={nv}/{(g == grp).sum()}", transform=ax.get_xaxis_transform(), ha="center",
                        fontsize=6.5, color="#52514e")
            ax.set_xticks([0, 1, 2], ["CN", "MCI", "AD"])
            short = ctx if level == "macro" else ctx.split("|")[0] + " ROI " + ctx.split("_")[-1]
            ax.set_title(f"{fid}@{short}\n{cohort}", fontsize=7, pad=12)
        fig.suptitle("Participant-level values (one point per person; bar = median; n = valid/enrolled)",
                     fontsize=8, y=1.08)
        fig.tight_layout()
        _save(fig, c, "fig5_participant_distributions")
        plt.close(fig)


def _f(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:.{d}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{d}f}"


def _ci(r, cn):
    lo, hi = r.get(f"ci_low_{cn}"), r.get(f"ci_high_{cn}")
    return f"[{_f(lo, 2)}, {_f(hi, 2)}]"


def write_memos(c, rec, reg, mac, mic, ra, tasks, resv, panel, res):
    n_new = rec["dataset_definition_count"]
    n_tot = rec["total_definition_count"]
    X = int((reg.status == "supported").sum())
    Xb = int(((reg.status == "supported") & (reg.origin == "base160")).sum())
    Xd = X - Xb
    # Y / Z: contexts eligible for the main screen
    Y = {k: int(g.eligible_main.sum()) for k, g in mac.groupby("cohort_id")}
    Z = {k: int(g.eligible_main.sum()) for k, g in mic.groupby("cohort_id")}
    A = {k: int(g.selected_main.sum()) for k, g in mac.groupby("cohort_id")}
    B = {k: int(g[g.selected_main].joint_fdr_supported.sum()) for k, g in mac.groupby("cohort_id")}
    Cn = {k: int((panel.cohort_id == k).sum()) if len(panel) else 0 for k in mac.cohort_id.unique()}
    D = {k: int((g.status == "main_candidate").sum()) for k, g in ra.groupby("cohort_id")}
    sel_mac = mac[mac.selected_main].sort_values(["cohort_id", "p_joint"])
    sel_mic = mic[mic.selected_main]
    by_origin_mac = sel_mac.groupby(["cohort_id", "origin"]).size().unstack(fill_value=0)
    by_origin_mic = sel_mic.groupby(["cohort_id", "origin"]).size().unstack(fill_value=0)
    top = ra[ra.in_top20].sort_values(["cohort_id", "base_rank"])
    reserves_mad = {k: int(g.pass_MCI_AD.sum()) for k, g in pd.concat([mac, mic]).groupby("cohort_id")}

    def dict_str(d, order=("RAW157", "ROI64_T1235", "ROI64_T5")):
        return ", ".join(f"{k}: {d.get(k, 0)}" for k in order if k in d)

    def macro_table(lang):
        hdr = ("| Cohort | ID@scope | Tên | Nguồn gốc | p CN-AD / CN-MCI / MCI-AD | δ CN-AD [CI] | δ CN-MCI [CI] | δ MCI-AD | q joint (BH) | N CN/MCI/AD | Lựa chọn bootstrap / hướng ổn định | LOO-AD giữ | Ghi chú |"
               if lang == "vi" else
               "| Cohort | ID@scope | Name | Origin | p CN-AD / CN-MCI / MCI-AD | δ CN-AD [CI] | δ CN-MCI [CI] | δ MCI-AD | joint q (BH) | N CN/MCI/AD | bootstrap selection / direction | LOO-AD kept | Notes |")
        lines = [hdr, "|" + "---|" * 13]
        for r in sel_mac.to_dict("records"):
            note = r["evidence_tier"] + ("; " + r["semantic_confidence"] if r["semantic_confidence"] != "verified_events" else "")
            lines.append(
                f"| {r['cohort_id']} | {r['feature_id']}@{r['scope_or_region_key']} | {r['name']} | {r['origin']} | "
                f"{_f(r['p_CN_AD'])} / {_f(r['p_CN_MCI'])} / {_f(r['p_MCI_AD'])} | {_f(r['delta_CN_AD'], 2)} {_ci(r, 'CN_AD')} | "
                f"{_f(r['delta_CN_MCI'], 2)} {_ci(r, 'CN_MCI')} | {_f(r['delta_MCI_AD'], 2)} | {_f(r['q_joint_bh'])} | "
                f"{int(r['n1_CN_AD'])}/{int(r['n2_CN_MCI'])}/{int(r['n2_CN_AD'])} | {_f(r['selection_freq_main'], 2)} / "
                f"{_f(r['direction_agreement_both_controls'], 2)} | {_f(r.get('prop_main_screen_preserved'), 2)} | {note} |")
        return "\n".join(lines)

    def roi_table(cohort, lang):
        t = top[top.cohort_id == cohort]
        if not len(t):
            return "(không có vùng nào qua tiêu chí chính)" if lang == "vi" else "(no region passed the main rule)"
        hdr = ("| Hạng | Task | ROI (text) | Micro features được chọn (≤5 hiển thị) | Anchor: p CN-AD / CN-MCI / MCI-AD; δ | m_r, p_region, q_region | Coverage tối thiểu | Tần suất top-20 | LOO-AD giữ | Mức bằng chứng |"
               if lang == "vi" else
               "| Rank | Task | ROI (text) | Selected micro features (≤5 shown) | Anchor: p CN-AD / CN-MCI / MCI-AD; δ | m_r, p_region, q_region | Min coverage | Top-20 freq | LOO-AD kept | Evidence |")
        lines = [hdr, "|" + "---|" * 10]
        for r in t.to_dict("records"):
            feats = r["main_selected_features"].split(";")
            shown = ", ".join(feats[:5]) + (f" (+{len(feats) - 5})" if len(feats) > 5 else "")
            tier = "stable_candidate" if r["stable_candidate"] else ("region_FDR_supported" if r["q_region_bh"] <= .1
                                                                    else "exploratory_only")
            lines.append(
                f"| {int(r['base_rank'])} | {r['canonical_task']} | ROI {r['roi_id']} (text_unavailable) | {shown} | "
                f"{r['anchor_feature']}: {_f(r['anchor_p_CN_AD'])} / {_f(r['anchor_p_CN_MCI'])} / {_f(r['anchor_p_MCI_AD'])}; "
                f"{_f(r['anchor_delta_CN_AD'], 2)}, {_f(r['anchor_delta_CN_MCI'], 2)}, {_f(r['anchor_delta_MCI_AD'], 2)} | "
                f"{int(r['m_r'])}, {_f(r['p_region'])}, {_f(r['q_region_bh'])} | {_f(r['anchor_min_coverage'], 2)} | "
                f"{_f(r['top20_bootstrap_frequency'], 2)} | {_f(r['anchor_loo_prop_preserved'], 2)} | {tier} |")
        return "\n".join(lines)

    def task_table(lang):
        lines = ["| Cohort | Task | ROI expected | testable | main-selected | p_task | q_task BH | regions passing CN-AD / CN-MCI / MCI-AD |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in tasks.to_dict("records"):
            lines.append(f"| {r['cohort_id']} | {r['canonical_task']} | {r['expected_rois']} | {r['testable_regions']} | "
                         f"{r['main_selected_regions']} | {_f(r['p_task'])} | {_f(r['q_task_bh'])} | "
                         f"{r['regions_any_CN_AD']} / {r['regions_any_CN_MCI']} / {r['regions_any_MCI_AD']} |")
        return "\n".join(lines)

    mad_only = pd.concat([mac, mic])
    mad_only = mad_only[mad_only.pass_MCI_AD & ~mad_only.selected_main].sort_values("p_MCI_AD").head(10)
    mad_lines = "\n".join(f"- {r.cohort_id} {r.feature_id}@{r.scope_or_region_key} ({r.origin}): p_MCI_AD={_f(r.p_MCI_AD)}, "
                          f"δ={_f(r.delta_MCI_AD, 2)}" for r in mad_only.itertuples())
    fdr_mac = sel_mac[sel_mac.joint_fdr_supported]
    fdr_lines = "\n".join(f"- {r.cohort_id} {r.feature_id}@{r.scope_or_region_key} `{r.name}` ({r.origin}): q_joint={_f(r.q_joint_bh)}, "
                          f"BY {'còn' if r.joint_fdr_supported_by else 'không còn'} hỗ trợ" for r in fdr_mac.itertuples())
    panel_lines = "\n".join(f"- {r.cohort_id} {r.feature_id}@{r.scope_or_region_key} `{r.name}`" for r in panel.itertuples()) \
        if len(panel) else "- (rỗng)"
    sens = pd.read_csv(c.run / "sensitivity_coverage.csv")
    sens_txt = "; ".join(f"{r.cohort_id}/{r.level} cov {r.coverage_threshold}: {r.selected_main}"
                         for r in sens.itertuples())
    perm = pd.read_csv(c.run / "permutation_validation.csv")
    n_dis = int(perm.disagree_at_0_20.astype(str).str.lower().eq("true").sum()) if len(perm) else 0

    vi = f"""# BEGAD feature discovery: kết quả (bản tiếng Việt)

Run: `{c.run.name}` · freeze `{rec['freeze_id']}` · seed {c.cfg['seed']} · không train classifier.
Nhóm chứng được báo cáo là **CN** (= mã 0 CO/HC của repository, đã kiểm chứng với roster và tiền tố ID).

**Dữ liệu thực sự đã phân tích.** RAW157 (export sự kiện Tobii thô, 157 người: CN 100 / MCI 41 / AD 16; chỉ
macro, KD1/KD2/KD3/READING) và ROI64_T1235 (workbook ROI gốc, 64 người: CN 32 / MCI 17 / AD 15; KD1-3 + đọc,
micro và macro, là cohort xếp hạng task–ROI chính). ROI64_T5 (cùng 64 người, chỉ đọc, export khác) là phân tích
độ nhạy riêng, không phải tái lặp độc lập. Các cohort không được gộp.

## Câu trả lời

Catalogue gồm 160 định nghĩa gốc + {n_new} đề xuất từ dataset = {n_tot} (mục tiêu 260; **mở rộng chưa đủ: {n_new}/100**).
Có {X}/{n_tot} định nghĩa đủ nguồn ({Xb} gốc, {Xd} mới); macro-context đủ điều kiện tiêu chí chính: {dict_str(Y)};
micro-context đủ điều kiện: {dict_str(Z)}.
Có macro-context qua tiêu chí chính: {dict_str(A)}; trong đó có hỗ trợ FDR (joint BH q ≤ 0.10): {dict_str(B)};
thuộc panel gọn: {dict_str(Cn)}.
Có {D.get('ROI64_T1235', 0)} cặp task–ROI qua tiêu chí chính trong ROI64_T1235; hiển thị {min(20, D.get('ROI64_T1235', 0))} cặp.
(Độ nhạy chỉ đọc T5: {D.get('ROI64_T5', 0)} vùng, hiển thị {min(20, D.get('ROI64_T5', 0))}.)

Một định nghĩa sinh ra nhiều giả thuyết (mỗi task, mỗi ROI, mỗi cặp so sánh), nên {n_tot} định nghĩa không phải số
phép kiểm định. Tiêu chí chính là sàng lọc thăm dò: feasible AND p_CN_AD < 0.20 AND p_CN_MCI < 0.20 (p thô,
Mann–Whitney hai phía). Lựa chọn trên toàn bộ dữ liệu này là thăm dò; mọi đánh giá dự đoán sau này phải lặp lại
việc chọn feature/ROI bên trong dữ liệu train hoặc dùng cohort xác nhận độc lập.

**Đóng góp theo nguồn gốc (cùng một họ đa kiểm định).** Macro được chọn theo origin:
{by_origin_mac.to_string()}
Micro được chọn theo origin:
{by_origin_mic.to_string()}
Không có quota cho nguồn gốc nào; các D-feature không phải cohort xác nhận độc lập.

**Hỗ trợ sau hiệu chỉnh đa kiểm định.** Macro có joint BH q ≤ 0.10 (họ joint của từng cohort):
{fdr_lines or '- không có'}
Không micro-context nào có joint BH q ≤ 0.10 và không vùng nào có q_region ≤ 0.10 (xem bảng ROI). Kết quả micro
vì vậy chỉ là `exploratory_only` hoặc ổn định theo bootstrap, không được hỗ trợ FDR.

**Panel macro gọn** (qua tiêu chí chính, |δ| ≥ 0.20 cả hai so sánh với CN, hướng ổn định ≥ 80%, tần suất chọn ≥ 60%,
loại tương quan |ρ| ≥ 0.90):
{panel_lines}

## Các macro features được chọn

δ là Cliff's delta, dương khi nhóm thứ hai lớn hơn (ví dụ δ CN-AD > 0: AD cao hơn CN). CI là khoảng bootstrap
phần trăm 2000 lần theo người, mang tính mô tả, không hiệu chỉnh sau chọn lọc. Cột bootstrap là tần suất qua
tiêu chí chính và tỉ lệ cùng hướng trong 500 lần lặp lại toàn bộ tìm kiếm.

{macro_table('vi')}

Ý nghĩa sinh lý và giới hạn của từng ID nằm trong `feature_registry.csv` và `DATASET_PROPOSED_100_FEATURES.md`.
Feature `export_convention` dựa trên quy ước export workbook (Bảng II của bài báo) và có ghi chú ngữ nghĩa còn lại.

## Top task–ROI và micro features được chọn (ROI64_T1235, KD + đọc)

Bằng chứng vùng: p_region = min(1, m_r · min p_joint) (Bonferroni trong vùng), BH trên tất cả vùng kiểm định được.
Văn bản ROI không có trong repository nên hiển thị `text_unavailable`; thứ tự ROI ID chưa được xác minh là thứ tự đọc.

{roi_table('ROI64_T1235', 'vi')}

Toàn bộ micro features qua tiêu chí chính trong các vùng này: `micro_selected_top20.csv`; ngoài top-20:
`micro_selected_main.csv`.

### Độ nhạy chỉ đọc (ROI64_T5, không phải tái lặp độc lập)

{roi_table('ROI64_T5', 'vi')}

## Tóm tắt bốn task

{task_table('vi')}

p_task = min(1, R_t · min p_region) là kiểm định bảo thủ về sự tồn tại bằng chứng trong task; không có nghĩa task
nào tốt hơn về lâm sàng. Đọc có nhiều ROI hơn KD nên số vùng được chọn không so sánh trực tiếp.

## Dự phòng theo cặp và MCI_AD

Vùng dự phòng (không qua tiêu chí chính nhưng có feature qua một cặp): {len(resv)} hàng trong
`roi_pair_specific_reserves.csv`. Số giả thuyết qua MCI_AD (p < 0.20) theo cohort: {dict_str(reserves_mad)}.
Mười kết quả MCI_AD mạnh nhất nằm ngoài danh sách chính:
{mad_lines or '- không có'}

## Ổn định, ảnh hưởng và kiểm định hoán vị

- Bootstrap khám phá 500 lần (lặp lại toàn bộ họ đủ điều kiện, không chỉ người thắng): `bootstrap_stability.csv`.
- Bỏ từng người AD: `ad_influence.csv`. Kiểm định hoán vị 9999 lần cho macro được chọn và micro được chọn trong
  top-20: {len(perm)} phép; {n_dis} trường hợp p chính và p hoán vị nằm khác phía của 0.20 (`permutation_validation.csv`).
- Độ nhạy coverage (50%/75%, họ riêng, không thay thế phân tích chính), số được chọn: {sens_txt}.

## Giới hạn ngữ nghĩa/QC chính

- Không có hình học ROI, văn bản, thứ tự đọc, dòng; mọi đặc trưng visit/regression theo thứ tự đọc/landing bị chặn.
- Không có audio/TextGrid/chú thích lỗi; đặc trưng nói chỉ dùng tóm tắt workbook (totalSpokenTime, speechDelay).
  speechDelay của T1235 bị cắt phải dưới 10 s; giá trị ≤0 hoặc ≥10 s xem là censored.
- totalFixTime không phải tổng thời gian fixation (S04); số đếm saccade trong workbook mâu thuẫn (S03).
- Trường pupil *B của T1235 = hằng số − pupil (đảo dấu, hằng số không xác định); baseline trong file thô không tái
  tạo được quy tắc 10 fixation đầu (S23); chỉ dùng tương phản không phụ thuộc hằng số. Pupil thô chỉ có tại biên
  sự kiện; chiếu sáng không được kiểm soát.
- Không có dữ liệu nhân khẩu học (tuổi, giới, học vấn, thị lực, ngôn ngữ) để mô tả hay hiệu chỉnh.
- Cỡ mẫu AD nhỏ (15–16); khoảng tin cậy bootstrap chỉ mang tính mô tả.

Nguồn còn thiếu mở khóa nhiều feature nhất: (1) bounding box ROI + văn bản/thứ tự đọc (khoảng 40 định nghĩa micro
và macro về visit, regression, landing, chuẩn hóa theo từ), (2) TextGrid/ELAN và chú thích lỗi (khoảng 40 định nghĩa
nói và phối hợp gaze–speech), (3) chuỗi pupil liên tục và baseline đã kiểm chứng (khoảng 15), (4) sự kiện onset/offset
của media (A001, A004, A022, A075, A077).

Kết quả trên chỉ thuộc cohort này. Giả thuyết từ tài liệu (ví dụ thay đổi đọc ở MCI/AD) không được xác nhận hay bác
bỏ bởi một sàng lọc p < 0.20.
"""
    (c.run / "RESULTS_VI.md").write_text(vi)
    en = f"""# BEGAD feature discovery: results (English companion)

Run: `{c.run.name}` · freeze `{rec['freeze_id']}` · seed {c.cfg['seed']} · no classifier training.
The control group is reported as **CN** (= repository code 0 CO/HC, verified against roster and ID prefixes).

**Data actually analysed.** RAW157 (raw Tobii event exports, 157 people: CN 100 / MCI 41 / AD 16; macro only,
KD1/KD2/KD3/READING) and ROI64_T1235 (original ROI workbook, 64 people: CN 32 / MCI 17 / AD 15; KD1-3 + reading,
micro and macro; primary task-ROI ranking cohort). ROI64_T5 (same 64 people, reading only, a different export) is
a separate sensitivity analysis, not independent replication. Cohorts are never pooled.

## Answer

The catalogue has 160 base definitions + {n_new} dataset-derived = {n_tot} (target 260; **extension partial: {n_new}/100**).
{X}/{n_tot} definitions have sources ({Xb} base, {Xd} new); macro contexts eligible for the main screen: {dict_str(Y)};
micro contexts eligible: {dict_str(Z)}.
Macro contexts passing the main screen: {dict_str(A)}; with FDR support (joint BH q ≤ 0.10): {dict_str(B)};
in the compact panel: {dict_str(Cn)}.
{D.get('ROI64_T1235', 0)} task–ROI pairs pass the main rule in ROI64_T1235; {min(20, D.get('ROI64_T1235', 0))} shown.
(Reading-only T5 sensitivity: {D.get('ROI64_T5', 0)} regions, {min(20, D.get('ROI64_T5', 0))} shown.)

One definition instantiates many hypotheses (per task, ROI and contrast), so {n_tot} definitions is not the number
of tests. The main screen is exploratory: feasible AND p_CN_AD < 0.20 AND p_CN_MCI < 0.20 (raw two-sided
Mann–Whitney p). Selection on this full dataset is exploratory; any predictive evaluation must repeat feature/ROI
selection inside training data or use an independent confirmation cohort.

**Origin contribution (shared multiplicity families).** Selected macro by origin:
{by_origin_mac.to_string()}
Selected micro by origin:
{by_origin_mic.to_string()}

**Support after multiplicity correction.** Macro with joint BH q ≤ 0.10:
{fdr_lines.replace('còn', 'retains').replace('không retains', 'does not retain') or '- none'}
No micro context has joint BH q ≤ 0.10 and no region has q_region ≤ 0.10, so micro results are exploratory or
bootstrap-stable only.

**Compact macro panel:**
{panel_lines.replace('(rỗng)', '(empty)')}

## Selected macro features

{macro_table('en')}

## Top task-ROI pairs and selected micro features (ROI64_T1235)

{roi_table('ROI64_T1235', 'en')}

### Reading-only sensitivity (ROI64_T5)

{roi_table('ROI64_T5', 'en')}

## Four-task summary

{task_table('en')}

## Pair-specific and MCI_AD reserves

{len(resv)} reserve regions in `roi_pair_specific_reserves.csv`. MCI_AD passes by cohort: {dict_str(reserves_mad)}.
{mad_lines or '- none'}

## Stability, influence and permutation

Permutation checks: {len(perm)}; primary/permutation disagreements at 0.20: {n_dis}. Coverage sensitivity
(selected counts): {sens_txt}.

## Main limitations

No ROI geometry/text/reading order; no audio/TextGrid/error annotations; totalFixTime is not summed dwell; workbook
saccade counts are inconsistent; T1235 pupil fields are sign-inverted relative to an unidentified constant; raw pupil
only at event boundaries; no demographics; AD n = 15-16. Missing sources that would unlock the most definitions:
ROI boxes + text/order, speech alignment + error annotations, continuous pupil with verified baseline, media onset events.
These are results of this cohort only, not confirmation or refutation of literature hypotheses.
"""
    (c.run / "RESULTS_EN.md").write_text(en)
