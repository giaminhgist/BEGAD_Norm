# Statistical protocol: exploratory feature discovery

This file defines the primary numerical protocol. Freeze it before looking at
group p-values. These are operational research choices, not clinical cutoffs.
The purpose is to select interpretable candidates for further study, not to
establish diagnosis or estimate classification accuracy.

## 1. Analysis unit and frozen configuration

One independent participant contributes at most one baseline value to a fixed
micro hypothesis `(cohort, stimulus_version, task, ROI, feature, contrast)` or
macro hypothesis `(cohort, scope, feature, contrast)`. Multiple fixations, ROI
rows, KD cards or repeated visits do not increase the independent sample size.
Different people, rather than rows, are resampled or permuted.

Write this configuration to the run directory before statistical analysis:

```yaml
protocol_version: begad_feature_discovery_v2_expanded260
seed: 20261027
base_definition_count: 160
additional_definition_target: 100
candidate_origins: [base160, dataset100]
require_candidate_freeze: true
contrasts: [CN_AD, CN_MCI, MCI_AD]
primary_test: mannwhitneyu_two_sided_asymptotic_tie_corrected
continuity_correction: true
screen_p: 0.20
main_screen: CN_AD_AND_CN_MCI
strict_screen: ALL_THREE_PAIRS
minimum_subjects_per_group: 8
minimum_group_coverage: 0.60
minimum_roi_coverage_for_task_roi_summaries: 0.60
coverage_sensitivity: [0.50, 0.75]
fdr_method: fdr_bh
fdr_support_q: 0.10
fdr_dependence_sensitivity: fdr_by
preferred_abs_cliffs_delta: 0.20
preferred_direction_stability: 0.80
preferred_selection_frequency: 0.60
redundancy_abs_spearman: 0.90
bootstrap_effect_ci_resamples: 2000
bootstrap_discovery_resamples: 500
permutation_validation_resamples: 9999
roi_top_k: 20
```

Keep the exact extraction-support rules in file 02 in the configuration too.
Record software versions, source hashes, population counts and all exclusions.
Do not optimize any threshold against the number of selected AD features.

Complete file 06's dataset-informed proposal step before diagnostic screening.
Freeze the full base-plus-D catalogue and record the actual count: 260 when the
100-definition extension is complete, or 160 plus the documented smaller N when
source limitations prevent completion. Both origins use this same protocol;
origin is a provenance field, not a separate statistical family or easier gate.

## 2. Eligibility before inference

For each context and contrast, require all of the following:

1. Source meaning, units, timing endpoints and ROI/task identity are resolved.
2. Values are observed or genuine observed zeros. Exclude imputed, censored,
   invalid, unavailable and unsupported values from the primary test.
3. Each of the two groups has at least 8 valid unique participants.
4. Each group has at least 60% valid measurement coverage among its eligible
   enrolled participants expected to perform that stimulus. Missing recordings
   remain in this denominator. For cross-task features use participants expected
   to perform all required tasks, not only successful paired recordings.
5. The pooled observations are not constant. One group may be constant while
   another differs; that alone does not invalidate a test.

Use full-cohort diagnostic denominators from `subject_manifest.csv`, not the
number of rows in a filtered table. A conditional measure such as return latency
is tested among observed returners and may fail coverage. Report its event rate
and its conditional interpretation; do not pretend missing return latency is 0.

For a completely constant pooled feature, record `p=1`, `delta=0`,
`status=constant_not_informative`; exclude it from the testable family rather
than treating it as selected. Missing/invalid p-values remain NA with a reason.
Never replace NA with an apparently favorable value. A count or binary feature
with adequate support is allowed; record ties and event counts.

Eligibility is contrast-specific. Main-screen eligibility requires eligibility
for both CN_AD and CN_MCI. MCI_AD must always be attempted and its status reported.
The three groups normally satisfy the same coverage rules in this joint set.

The secondary 50% and 75% coverage analyses are clearly labeled sensitivity runs,
with their own fixed eligible families. They do not replace the primary 60% run.

## 3. Primary test, descriptive summaries and direction

For every eligible contrast use two-sided Mann–Whitney U with asymptotic
tie correction and continuity correction, consistently across features:

```python
scipy.stats.mannwhitneyu(x_group1, x_group2,
                        alternative="two-sided", method="asymptotic",
                        use_continuity=True)
```

This is a distribution/rank comparison. It is not generally a test of medians
alone when distribution shapes differ. Do not first run Shapiro–Wilk and then
choose whichever t-test/rank test gives the smaller p-value. The primary method
does not change after inspecting significance. A Kruskal–Wallis test across all
three groups may be descriptive, but must not gate the pairwise tests.

For each group export eligible N, observed N, missing N, coverage, median, Q1,
Q3, mean, SD and contributing event/ROI support. Plot participant points when
feasible, not just bars. Binary/count features also need their frequency table.

Define the reported Cliff's delta to be positive when the **second named group
has larger values**:

```text
delta_CN_AD  = P(AD > CN)  - P(AD < CN)
delta_CN_MCI = P(MCI > CN) - P(MCI < CN)
delta_MCI_AD = P(AD > MCI) - P(AD < MCI)
```

If `U2` is the U statistic with group 2 supplied first, use
`delta = 2 * U2 / (n1 * n2) - 1`. Ties contribute zero. Unit tests must verify
the sign, including reversed groups and tied observations.

Do not require an ordered CN < MCI < AD trend. Reading behavior can be nonlinear;
MCI compensation and heterogeneous AD performance are hypotheses, not reasons
to alter group labels. Report a direction pattern separately, e.g. both patient
groups higher, opposite directions, or indeterminate.

## 4. Exactly what counts as selected

The primary exploratory feature list is:

```python
selected_main = eligible_CN_AD and eligible_CN_MCI \
                and p_CN_AD < 0.20 and p_CN_MCI < 0.20

selected_strict_three = selected_main and eligible_MCI_AD \
                        and p_MCI_AD < 0.20
```

Also export three pair-specific lists, each requiring eligibility and p < 0.20
for that contrast, with `selected_main` indicated. Features separating MCI from
AD but not CN from both groups remain useful **pair-specific reserves**. They
must not silently enter the main list. Values exactly 0.20 fail a strict `<`.

No additional effect-size or FDR gate is imposed on the user's main exploratory
list. Instead attach these separate flags:

- `pairwise_fdr_supported`: both control contrasts have their pairwise-family
  BH q <= 0.10;
- `joint_fdr_supported`: the two-contrast joint hypothesis has BH q <= 0.10;
- `effect_preferred`: both control contrasts have absolute delta >= 0.20;
- `direction_stable`: both control effects agree with the original nonzero
  direction in at least 80% of valid bootstrap replicates;
- `selection_stable`: the main screen passes in at least 60% of discovery
  bootstrap replicates.

All these flags are reported, not collapsed into an unexplained score. The
suggested compact panel is the nonredundant subset of main-selected features
meeting `effect_preferred`, `direction_stable` and `selection_stable`; annotate
FDR support independently. It may be empty. Do not loosen thresholds until it
contains a desired number of features.

## 5. Multiple testing: define the family before selecting

Analyse independent or incompatible cohorts separately. In each primary cohort:

| Family | Include before correction |
|---|---|
| Macro pairwise | Every eligible `(scope, macro feature, contrast)` across all macro contexts and all three contrasts |
| Micro pairwise | Every eligible `(task, stimulus version, ROI, micro feature, contrast)` across all tasks/ROIs and all three contrasts |
| Macro joint | Every eligible `(scope, macro feature)` with both control comparisons |
| Micro joint | Every eligible `(task, stimulus version, ROI, micro feature)` with both control comparisons |

For pairwise families apply BH to the complete vector of valid primary p-values,
including nonselected features. Report the family ID and family size for every
row. Never correct only the top 20 regions, only p < 0.20 features, or each ROI
independently while calling the result study-wide FDR.

All relevant M/A and D hypotheses enter these families together. Do not correct
base160 and dataset100 separately and then pool the selected lists. If a base-only
analysis already exists, recompute joint/pairwise correction, region search sizes,
rankings and discovery stability using the expanded universe. Unchanged raw tests
may be cached; their old family q-values and rankings are not current results.

For the two-control joint requirement use the intersection-union p-value:

```text
p_joint = max(p_CN_AD, p_CN_MCI)
```

It tests the claim that the **same feature** differs in both control contrasts;
no independence between these two contrasts is required for this max-p rule.
Apply BH to all eligible joint p-values in the corresponding joint family.
Optional strict-three joint analysis uses the maximum of all three p-values
and a separate, explicitly labeled family. This is evidence about associations,
not a test of three-class predictive performance.

BH is the conventional primary correction; its usual dependence assumptions
are not established merely because features are correlated. Also calculate BY
q-values as a conservative dependence-robust sensitivity analysis. State whether
support survives it. Record failed/unavailable hypotheses outside the family
with reasons; never quietly drop valid large p-values.

Correction applies to its named family. Separate macro, micro, joint, region,
task and cohort families do not provide one simultaneous FDR guarantee for the
union of every reported list. State the family whenever claiming FDR support.

Before testing, collapse only **proven identical or strictly rank-equivalent
measurements with the same observed subject support**, using a fixed canonical
ID. Examples include the same IQR before/after subtraction of one constant.
Record every alias and its representative. A start value, end value and their
difference are algebraically related but are not identical hypotheses; do not
automatically discard the difference. Correlation-based pruning happens later
and must not shrink or recompute the original FDR family.

## 6. Uncertainty and participant influence

For effect confidence intervals use 2,000 stratified participant bootstrap
replicates: resample participants with replacement within each diagnostic group,
carrying all of a participant's task/ROI/feature records together. At a fixed
contrast, calculate the empirical delta distribution and its 2.5th/97.5th
percentiles. Record effective valid replicate count. If fewer than 90% of
replicates have a defined effect, flag `bootstrap_unreliable` and do not present
the interval as well supported. Percentile intervals with 15–16 AD participants
are exploratory; resampling does not create independent patients.

These are marginal descriptive intervals, not simultaneous or post-selection
confidence intervals. They also do not correct informative missingness. Disclose
this when displaying intervals for features selected from the same cohort.

Calculate intervals at least for all main/pair-specific selected features and
all features displayed in the top-20 report; include `ci_not_computed` for other
rows if computing all is too expensive. Do not fabricate an interval from SD.

Perform leave-one-AD-participant-out influence analysis for each main-selected
feature and top-region representative. Recalculate effects and raw p-values
after omitting each AD participant. Export ranges, sign-flip count and proportion
of runs preserving the screen. With an omission reducing support below the
eligibility rule, flag that replicate rather than dropping the rule silently.
This diagnoses dependence on one patient; it is not cross-validation accuracy.

## 7. Discovery stability: rerun the entire eligible search

Use 500 stratified participant bootstrap replicates with a deterministic seed
stream. Freeze the technically eligible candidate universe from the primary
run, including both base160 and dataset100; do not keep only original winners.
Retain all records for each sampled
participant; assign a bootstrap draw ID so repeated draws remain bootstrap
observations rather than being deduplicated back to the original sample.

For each replicate:

1. Resample the complete participant manifest within each diagnostic group.
2. Recompute available group values, effects and primary p-values for **all**
   candidates in the fixed eligible universe.
3. Reapply the raw main/strict/pair-specific screens and the file-04 base ROI
   ranking across **all** originally testable regions.
4. Record feature selection, effect direction, region eligibility and membership
   in that replicate's up-to-20 list.

The original minimum unique-person criterion is a cohort feasibility gate; do
not require 8 distinct original IDs again inside each with-replacement draw.
Otherwise a bootstrap of 8 eligible people would almost always fail by design.
Require at least 2 distinct source participants and at least 2 observed draws
per compared group to compute a bootstrap statistic; record the original-unique
counts and drawn coverage. Use the same 60% drawn-coverage rule. Noncomputable
replicates count as not selected in selection/top-20 frequency; effect-direction
frequency uses only defined replicates and reports that denominator. A zero
effect does not count as agreement with a nonzero original direction.

Bootstrap top-20 frequency uses the deterministic **base ranking**, which does
not contain bootstrap frequency itself. Do not recursively rank by a quantity
being estimated. Keep primary families fixed for bootstrap correction, assigning
p=1 to temporarily noncomputable members for that replicate only. Original
observational tables still retain NA and explicit reasons for invalid tests.

Report stability as sensitivity to this dataset's sampling. It is not independent
replication, a new effective N, or a guarantee of external reproducibility.

## 8. Permutation and other sensitivity analyses

For every main-selected macro feature and main-selected micro feature in the
primary top 20, validate the U result with a two-sided participant-label
permutation test using 9,999 random permutations, or exact enumeration if the
number of distinct label assignments is smaller. Use the same U statistic,
preserve group sizes and use a documented two-sided tail convention. For random
permutations use a nonzero Monte Carlo p-value convention with the +1 correction.

Store `p_permutation` beside `p_primary`; do not replace the primary value by
`min(p_primary, p_permutation)`. These selectively computed validation p-values
do not form a study-wide FDR family and cannot be reported as such. Mark features
whose primary and permutation results fall on opposite sides of 0.20.

Optional sensitivity: Brunner–Munzel for eligible continuous features if unequal
shapes/variances make the U interpretation problematic. Label it secondary and
do not use whichever test passes as the selection rule. Binary features can
have Fisher's exact test as secondary, with an explicit 2x2 table.

Describe age, sex, education, visual acuity, language proficiency, recording
device/session and task completion by group when available. If demographics
are missing, state the limitation. Any later adjusted association analysis needs
a separately frozen low-dimensional specification; it is not silently substituted
for this unadjusted screen. Do not adjust away MMSE/MoCA scores used to define
the diagnosis, or treat linguistic differences as automatically neurological.

## 9. Redundancy and interpretation

Retain the complete main-selected results. Additionally compute Spearman
correlations between selected features on paired participants, separately by
cohort and compatible context. Require at least 8 paired participants and report
overlap and group composition. Flag absolute correlation >=0.90.

Build a deterministic compact panel using a greedy procedure: sort by semantic
confidence, then minimum control-contrast coverage descending, then joint q
ascending, then the weaker absolute control effect descending, then feature ID;
keep a candidate unless it is above the correlation threshold with an already
kept feature. Undefined correlations do not prove independence; mark them.
Do not merge different physiological meanings solely because pooled group
separation induces correlation. Provide a within-group-centered sensitivity
correlation where supported and preserve the full list for interpretation.

No new FDR values are computed on this pruned panel. Report the retained feature,
excluded correlated alternatives, overlap, correlation and physiological reason.
The number of final features is a result, not a quota.

## 10. Primary implementation references

- SciPy Mann–Whitney U (ties, exact/asymptotic methods and interpretation):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
- Statsmodels multiple testing (BH and BY):
  https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html
- SciPy permutation tests (randomization and Monte Carlo p-values):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
- SciPy bootstrap (resampling and intervals):
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html

Pin and record installed versions; these pages are implementation references,
not evidence that the proposed features have diagnostic validity.
