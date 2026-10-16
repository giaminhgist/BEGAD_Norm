# Figure text

Registry of captions and descriptions for every figure generated in this
project. **Update this file each time a new figure is created.**

- `Caption` — the figure caption as it would appear in a manuscript
  (English, starts with the supported finding when one exists).
- `Description` — what the figure shows, the data source, and any notes
  needed to interpret it.

## Registry

| # | Figure | Output | Caption | Description |
|---|---|---|---|---|
| 1 | BEGAD dataset summary | `outputs/fig1_begad_dataset_summary.svg` | Cohort composition (HC/MCI/AD) and SOB score across groups | Panel (a): participant counts; panel (b): SOB by group with Kruskal–Wallis + pairwise tests |
| 2 | Top-6 features by p-value | `outputs/fig2_top6_features.svg` | The 6 processed_dataset features with the smallest Kruskal–Wallis p (100 feature–task tests over KD cards 3–5 + reading 9, no repeated feature) | Panels (a)–(f): each feature at its smallest-p task, HC/MCI/AD distributions |

## Detailed entries

### Figure 1 — BEGAD dataset summary

- **Output**: `outputs/fig1_begad_dataset_summary.svg` (+ `.png` preview)
- **Code**: `code/fig1_begad_dataset_summary.py`
- **Data**: `dataset_v2/participants_20250305.csv`; `SOB` column of the 157 raw CSVs in `dataset_v2/20250305/`; derived table `outputs/fig1_sob_by_participant.csv`
- **Caption**: "The BEGAD dataset. **(a)** Participant counts by diagnostic group: healthy controls (HC, n = 100), mild cognitive impairment (MCI, n = 41), and Alzheimer's disease (AD, n = 16). **(b)** SOB scores (raw export field) by group (HC n = 100, MCI n = 41, AD n = 16). Boxes show the median and interquartile range (IQR); whiskers extend to the most extreme values within 1.5× the IQR; points show individual participants. Group differences: Kruskal–Wallis H = 104.78, p = 1.8×10⁻²³; pairwise Mann–Whitney U tests with Benjamini–Hochberg correction, all p < 0.001 (HC–MCI p = 1.4×10⁻¹¹; HC–AD p = 3.8×10⁻²⁶; MCI–AD p = 4.4×10⁻⁸). Medians (IQR): HC 0 (0–0), MCI 0 (0–1), AD 2.75 (1.88–5.12)."
- **Description**: Cohort summary figure. Panel (a) is the group composition of the 157-participant BEGAD_Norm cohort (the CO label of the roster is displayed as HC). Panel (b) shows the per-participant SOB value, which is constant within each raw export and rises across groups (HC all 0; MCI 0–3; AD 1–8). Notes: the SOB field is not documented in this repository; its 0.5-step distribution is consistent with a CDR-SOB-like scale (unverified), so the axis uses the neutral export-field label. The unit of replication is the participant (n = 100/41/16).

### Figure 2 — Top-6 features by p-value

- **Output**: `outputs/fig2_top6_features.svg` (+ `.png` preview)
- **Code**: `code/fig2_top6_features.py`
- **Data**: `processed_dataset/features.csv` + `processed_dataset/metadata.csv`; test table `outputs/fig2_feature_task_pvalues.csv` (100 tests over tasks 3, 4, 5, 9)
- **Caption**: "The six processed_dataset features with the smallest Kruskal–Wallis p among the 100 feature–task tests (25 features × 4 tasks: King–Devick cards 3–5 and the reading task 9), each shown at the task with its smallest p and with no feature repeated: **(a)** `sac_dur_total` at task 4 (p = 5.5×10⁻⁵), **(b)** `spa_fix_count` at task 5 (p = 1.9×10⁻⁴), **(c)** `tem_dur_total` at task 5 (p = 2.7×10⁻⁴), **(d)** `sac_count` at task 5 (p = 3.5×10⁻⁴), **(e)** `sac_amp_mean` at task 3 (p = 1.3×10⁻³), **(f)** `geo_sacc_amp_mean` at task 5 (p = 1.4×10⁻³). Boxes show the median and interquartile range; whiskers extend to the most extreme values within 1.5× the IQR; points show individual participants (HC n = 100, MCI n = 41, AD n = 16). Units: ms (`sac_dur_total`, `tem_dur_total`), count (`spa_fix_count`, `sac_count`), deg (`sac_amp_mean`), px (`geo_sacc_amp_mean`). P-values are uncorrected; the screening is exploratory."
- **Description**: Each panel shows one feature's distribution by group at its smallest-p task among the four test tasks (KD cards 3, 4, 5 and reading 9), selected greedily over the ascending p list so the 6 features are distinct. Direction of group medians: count and duration features rise with severity in (b), (c), (d) (HC < MCI < AD); saccade-amplitude features (e), (f) are highest in HC; (a) is non-monotonic (MCI highest). Each panel carries a "task n" annotation; the p-table file records all 100 tests for auditability.
