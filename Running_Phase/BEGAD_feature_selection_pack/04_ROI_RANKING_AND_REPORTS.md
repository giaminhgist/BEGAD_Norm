# Selecting macro features and ranking task–ROI pairs

The requested conclusions are actual computed results: selected macro features,
and up to 20 task–ROI pairs with their selected micro features. This instruction
pack contains no preselected clinical feature or preselected ROI.

Selection uses the combined catalogue: base160 plus the dataset-grounded D
features authored under file 06, targeting 260 definitions. Both origins compete
under the same criteria; neither receives a quota of selected features or regions.
Display `origin` in every selection table and summarize its contribution.

## 1. Which population supplies the primary ranking?

Use the verified T1235 combined ROI cohort when available and semantically usable.
Canonical task names are `KD1`, `KD2`, `KD3`, `READING`, after source verification.
T5-only data provide a separate reading analysis; if T1235 cannot run, label the
primary result `READING_ONLY_FALLBACK`. Do not imply KD was analysed.

Never add duplicate T5 reading records to T1235 to improve N. Matching T5/T1235
participants are a sensitivity/consistency check, not independent replication.
Raw157 macro results and ROI64 macro results have separate cohort labels. Only
combine sources when identities, visits, stimuli and measurement semantics are
verified, preserving the participant denominator and source provenance.

A region key is `(canonical_task, stimulus_version, roi_id)`. In a single-version
cohort this is the user's task–ROI pair. If multiple stimulus versions are present,
report version explicitly; do not merge equal integer IDs across different texts.

## 2. Macro selection outputs

Produce the following distinct tables per cohort:

1. `macro_selected_main.csv`: all feasible macro contexts passing both CN_AD
   and CN_MCI at raw p < 0.20, irrespective of FDR flags.
2. `macro_selected_three_pairs.csv`: main-selected contexts also passing MCI_AD.
3. `macro_pair_specific.csv`: each passing contrast, including features outside
   the main set. Preserve contrast labels and `selected_main`.
4. `macro_compact_panel.csv`: the optional effect/stability-preferred,
   correlation-pruned panel defined in file 03. Keep all alternatives and reasons
   in `redundancy_audit.csv`; an empty compact panel is an honest result.

Each row must name the feature's scope: a specific task, a predefined cross-task
contrast, or the participant-global baseline. `A005@KD1` and `A005@READING` are
different hypotheses. Do not call an unspecified pooled average a subject feature.
For every selected feature show all three comparisons, even if the third does
not pass, and state effect direction rather than only significance stars.

## 3. Region-level evidence without rewarding a larger search

For every testable region r, take **all** its technically eligible micro-feature
hypotheses with both control comparisons, after proven-alias handling. This set
is fixed before the p < 0.20 screen. Let its size be `m_r`.

This includes eligible micro definitions from both `base160` and `dataset100`.
Count unique tested hypotheses after alias handling, not just the original 80
micro IDs. Recompute `m_r` and all downstream region/task evidence after extension.

```text
p_joint(r,f) = max(p_CN_AD(r,f), p_CN_MCI(r,f))
p_region(r)  = min(1, m_r * min_f p_joint(r,f))
```

This Bonferroni minimum-p construction is a conservative test of whether at
least one feature in the region differs in both control comparisons. It accounts
for the number of tested features in that region without assuming their
independence. It is not a probability that the ROI is a biomarker, and it does
not remove bias from differential missingness.

Calculate BH `q_region` across **all** testable regions from all four tasks in
the cohort, before choosing 20. Also export BY q-values. Keep feature-level
pairwise/joint q-values from file 03; region q-values answer a different question
and must not be substituted into the feature columns.

If no feature has both comparisons eligible, `p_region=NA` and the region is
untestable. Record `m_r`, supported feature families, coverage and reason. For
bootstrap ranking use the fixed original `m_r`; temporarily noncomputable joint
hypotheses receive p=1 for multiplicity/ranking in that replicate only.

## 4. Deterministic primary top-20 ranking

A region is a **main candidate** only if it has at least one micro feature
passing the main raw-p AND rule. Define its anchor feature as the feature with
smallest `p_joint`, breaking ties by larger minimum absolute control effect,
then greater minimum group coverage, then ascending feature ID. This anchor
must pass the main screen for a main-candidate region.

Sort all main-candidate regions by this exact base ordering:

1. Region FDR support tier: `q_region_BH <= 0.10` first.
2. `p_region`, ascending.
3. `min(abs(delta_CN_AD_anchor), abs(delta_CN_MCI_anchor))`, descending.
4. Minimum CN/MCI/AD anchor-feature coverage, descending.
5. Canonical task order `KD1, KD2, KD3, READING`, then stimulus version and
   deterministic ROI ID order.

Take the first `min(20, number_of_main_candidate_regions)`. This is the primary
`roi_top20.csv`. Save the complete ordering in `roi_ranking_all.csv`, retaining
untestable and nonselected regions with status/reason and no invented rank.

Attach participant-bootstrap top-20 membership frequency, feature selection
frequency and leave-one-AD influence from file 03. These measure stability but
are **not inputs** to the base ranking used in those bootstrap replicates.
Do not promote an unstable region into a stable tier merely because its raw
p-value is small. Use `stable_candidate` only when the anchor meets both effect
and stability preferences; preserve the original rank and disclose instability.

Do not add bonus points for having more selected features or more ROI rows.
Several highly correlated pupil endpoints are not independent confirmations.
The within-region multiplicity penalty is deliberately conservative; report
this tradeoff and the raw feature evidence together.

If a region has one feature passing CN_AD and a **different** feature passing
CN_MCI, but no single feature passes both, it is a complementary pair-specific
reserve. Put it in `roi_pair_specific_reserves.csv`, not the main top 20.

## 5. Micro features shown for each selected region

For every region in the primary top 20:

- Export **all** its main-selected micro features in `micro_selected_top20.csv`.
- Show a readable set of up to five representative features in the memo, using
  the file-03 redundancy audit and retaining distinct measurement families where
  available. This display cap does not erase other passing features.
- State the actual word/digit only when supplied by validated stimulus metadata.
  Otherwise show the ROI ID and `text_unavailable`; do not infer lexical content.
- Show all three raw p-values, relevant pairwise/joint q-values, two-control
  effect sizes and intervals, MCI_AD effect, valid N/coverage by group, region
  q-value, top-20 frequency and source/semantic flags.
- Distinguish `exploratory_only`, `joint_FDR_supported`, `region_FDR_supported`
  and `stable_candidate`. None means clinically validated.

Also export main-selected micro features from regions outside the top 20 in
`micro_selected_main.csv`; the ranking cutoff is not a reason to hide results.

## 6. Task-level summary

There are four intended tasks, not twenty tasks. Export `task_ranking.csv` for
the actual available tasks with expected/testable/main-selected ROI counts,
selected fraction, feature-family coverage and participant coverage. Do not rank
tasks solely by the number of significant ROIs: reading has more opportunities
than a KD card and ROI definitions may differ in granularity.

For a conservative task-level evidence summary:

```text
R_t       = number of all testable regions in task t
p_task(t) = min(1, R_t * min_r p_region(r))
```

Apply BH across all testable tasks, with BY as sensitivity. Sort task summary by
`p_task`, then median anchor minimum coverage among its main-selected regions,
then canonical task order. An undefined coverage tie-breaker sorts last. A task
with no main-selected regions still appears with its measured p-value/status.

This tests existence of evidence in a task and penalizes searching many regions;
it does not establish that the task is clinically superior or easier to deploy.
Report CN_AD, CN_MCI and MCI_AD profiles separately. Cross-task macro contrasts
are macro hypotheses and are not additional tasks or ROIs in this ranking.

## 7. Required machine-readable files

Write UTF-8 CSV with stable column names. Use NA with explicit status, not blank
strings interpreted as zero. CSV tables with no qualifying rows still need
headers plus a reason in the Markdown report.

| Output | Minimum content |
|---|---|
| `source_manifest.json` | Input hashes, versions, paths, units, availability |
| `subject_manifest.csv` | Person, baseline visit, diagnosis, expected tasks, cohort |
| `task_roi_manifest.csv` | Canonical task, stimulus version, ROI identity/order/geometry/text |
| `feature_registry.csv` | All 160 base IDs plus defined D IDs, origin/level, definitions, dependencies and availability; target 260 |
| `DATASET_PROPOSED_100_FEATURES.md` | Claude-authored additional definitions with dataset evidence and rationale |
| `dataset_proposed_features.csv` | Structured D registry extension, not 100 placeholders |
| `proposal_audit.csv`, `candidate_freeze.json` | Proposal decisions, duplicate checks, source support and frozen-universe hashes/counts |
| `feature_origin_summary.csv` | Available/testable/selected/FDR/stable counts by origin and level within the combined analysis |
| `semantic_audit.csv` and `semantic_audit.md` | Failed checks, evidence, fixes and blocked IDs |
| `quality_features.csv` | Tracking/pupil validity, completeness, imputation, QC sources |
| `micro_features_long.csv`, `macro_features_long.csv` | File-02 schemas, observed values and support/status |
| `micro_stats_all.csv`, `macro_stats_all.csv` | Every candidate context/contrast, including exclusions with reasons |
| `multiplicity_manifest.json` | Exact family membership/size, correction, aliases, sensitivity label |
| `macro_selected_main.csv` | Main-selected macro contexts with all comparison evidence |
| `macro_selected_three_pairs.csv` | All-three-pairs subset |
| `macro_pair_specific.csv` | Passing contrast-specific macro candidates |
| `macro_compact_panel.csv` | Optional stable/effect-preferred/pruned panel |
| `micro_selected_main.csv` | All main-selected micro contexts |
| `micro_selected_top20.csv` | Complete selected feature list within primary top 20 |
| `micro_pair_specific.csv` | All passing contrast-specific micro candidates |
| `roi_ranking_all.csv` | All ROI evidence and status, base rank where eligible |
| `roi_top20.csv` | At most 20 unique eligible region keys and representative features |
| `roi_pair_specific_reserves.csv` | Pair-specific/complementary regions outside main rule |
| `task_ranking.csv` | Coverage-adjusted reporting and conservative task evidence |
| `bootstrap_stability.csv` | Feature selection/direction and ROI top-20 frequency with denominators |
| `ad_influence.csv` | Leave-one-AD effect/p-value/screen changes without identifying subjects in figures |
| `permutation_validation.csv` | Secondary permutation evidence and disagreement flags |
| `redundancy_audit.csv` | Proven aliases, correlations, overlap and pruning reasons |
| `run_config.yaml`, `run_metadata.json` | Frozen parameters, seed, versions, commands, run status |
| `RESULTS_VI.md` | Vietnamese conclusions with no fabricated selection |

`*_stats_all.csv` rows should include:

```text
cohort_id,scope_or_region_key,feature_id,origin,contrast,eligible,status,exclusion_reason,
n1,n2,expected_n1,expected_n2,coverage1,coverage2,
median1,q1_1,q3_1,mean1,sd1,median2,q1_2,q3_2,mean2,sd2,
u_statistic,p_primary,test_method,cliffs_delta,ci_low,ci_high,ci_status,
pairwise_family_id,pairwise_family_size,q_bh,q_by,
p_joint,joint_family_id,joint_family_size,q_joint_bh,q_joint_by,
selected_main,selected_three_pairs,selected_this_pair,conditioning,semantic_status
```

Wide selection tables additionally include all three p-values/deltas together.
Join on explicit keys, never row position. Keep subject identifiers in local
analysis artifacts only; the narrative memo and plots use aggregate evidence.

## 8. Plots and Vietnamese conclusions

Generate compact, readable PNG/PDF figures where the data permit:

1. Availability/coverage heatmap by feature family and task, with group counts.
2. Selected macro effect-size forest plot, all three contrasts and CIs.
3. Task–ROI evidence map with the true ROI order; distinguish unavailable from
   nonsignificant. Overlay on a stimulus only if the real image/geometry exists.
4. Top-20 bootstrap membership frequencies and AD influence flags.
5. Participant-level distributions for representative features, including
   missingness and valid N. Avoid duplicating each person's points as independent
   observations across ROIs in an inferential plot.

`RESULTS_VI.md` must lead with the actual answer:

```text
Catalogue gồm 160 định nghĩa gốc + N_new đề xuất từ dataset = N_total (mục tiêu 260).
Có X/N_total định nghĩa đủ nguồn; Y macro-context và Z micro-context đủ điều kiện.
Có A macro-context qua tiêu chí chính; B có hỗ trợ FDR; C thuộc panel gọn.
Có D cặp task–ROI qua tiêu chí chính; hiển thị min(20,D) cặp.
```

Replace N_new/N_total/X/Y/Z/A/B/C/D with computed values, including whether the
100-definition extension is complete. Explain that one definition can instantiate
many task/ROI hypotheses, so 260 definitions is not the number of tests.
If sources are missing, state the supported cohort/tasks before the counts.

Report selection counts by `base160`/`dataset100` from the same combined-family
analysis. Do not present separate origin-specific q-values as the combined result
or call the additional candidates an independent validation cohort.

Then provide the two requested conclusion tables:

- **Các macro features được chọn**: ID/name, scope, physiological meaning,
  p-values/deltas by contrast, q/CI, valid N, stability and caveats.
- **Top task–ROI và micro features được chọn**: rank, task, ROI/text, selected
  features, region evidence, contrast profile, coverage and stability.

Add the four-task summary, MCI_AD-specific reserve findings, main semantic/QC
limitations and which missing sources would unlock the most blocked features.
Explicitly separate results of this cohort from literature hypotheses.

If no feature passes, state that outcome. If only 7 regions pass, report a top-7
list and a separate reserve table; never relabel 13 failures as selected to fill
20. If no source is testable, deliver the audited registry, implementation and
blocked-analysis report without invented p-values or rankings.
