# BEGAD_Norm — Repository Summary

**Scope note.** This describes the repository as it exists on `main` at git HEAD `0d8742b` ("Delete Subject-level folder"). An earlier, larger iteration lived in `subject-level/` and gave the project its name ("Norm" = normative); that directory was deleted at HEAD and is recoverable only from git history. The summary distinguishes what is **in the working tree now** (`[SOURCED:...]`) from what is **in git history only** (cited by commit).

## 1. Project overview and purpose

Pipeline for the **BEGAD battery**: extract gaze features from raw Tobii eye-tracker exports and compare three clinical groups — **CN (cognitively normal/control), MCI, AD**.

Key facts:
- Package `begad-norm`, described as "BEGAD normative / feature-discovery research repo" `[SOURCED:pyproject.toml]`.
- Labels CN/MCI/AD from numeric 0/1/2 in the roster `[SOURCED:dataset/participants_20250305.csv]`. Roster labels code 0 as "CO" (100 rows), but all processed datasets/results rename it "CN" ("CN = repository CO/HC, code 0") `[SOURCED:feature_engineer/semantic_audit.md]`.
- Primary cohort: **157 participants — CN 100 / MCI 41 / AD 16** `[SOURCED:processed_dataset/README.md]`.

Three current phases: (1) feature discovery — a frozen 209-definition catalogue (160 base + 49 dataset-derived) from which 37 task-level features were statistically screened; (2) processed datasets (24-feature complete-case and 33-feature imputed variants); (3) classification experiments (baseline, fusion, contrasts) under `output/`. An earlier normative deep-learning model lived only in git history (deleted with `subject-level/`).

## 2. Repository layout

`[SOURCED: filesystem, verified 2026-11-02]`
- `dataset/`, `dataset_v2/` — 205 raw Tobii CSVs each (identical file lists) + roster `dataset/participants_20250305.csv`.
- `T5.xlsx`, `T5_clean.xlsx`, `T1235.xlsx`, `T1235_clean.xlsx` — ROI summary workbooks (feature-definition sources only; T1235 = KD1-3+reading 64 subjects; T5 = reading only).
- `feature_engineer/` — feature catalogue/proposal, 37-feature summary, registry, semantic audit, candidate freeze, run metadata.
- `processed_dataset/` — feature tables (V1 wide / V2 long 24-feat; 33-feat wide/long), feature-column lists, build scripts, fixed CV split files.
- `src/` — experiment code: `expcommon/` (shared data/model/metrics), `baseline/`, `fusion/`, `contrasts/`.
- `output/` — results artefacts: `baseline/`, `baseline_33feat/`, `fusion/`, `contrasts/`, `features/` (figures).
- `pyproject.toml`, `uv.lock` — uv-managed deps: lightgbm>=4.6.0, numpy>=2.2.6, pandas>=2.3.3, pyyaml>=6.0.3, scikit-learn>=1.7.0, scipy>=1.15.3; no torch in current tree `[SOURCED:pyproject.toml]`.
- No root README in the working tree; only `processed_dataset/README.md` and `README.vi.md`. Overall project docs lived in deleted `subject-level/README.md` `[SOURCED:git history, commit 0d8742b^]`.

## 3. Data sources

**Raw exports.** 205 CSVs per dataset dir (raw Tobii Studio columns: media boxes, fixation/saccade indices, gaze coords, per-eye distance/pupil, validity flags, `Task`, `MediaName`, `Group`, `IsFollowup`, pupil baselines). Roster maps 157 participants (100 code-0 / 41 code-1 / 16 code-2).

**Protocol tasks.** `MediaName` values: `KD_1/2/3_highres.png` (King–Devick digit cards), `lukeminen_1.png` (reading; "lukeminen" = Finnish), `KD_demo`, instruction screens. The deleted README documented 11 protocol tasks (0–10) `[SOURCED:git history]`. Processed datasets use only the four test stimuli KD1/KD2/KD3/READING, one value per participant per task `[SOURCED:processed_dataset/README.md]`.

**ROI workbooks.** `T1235.xlsx` (KD1-3+reading, 64 subjects) and `T5.xlsx` (reading, 64 subjects) define the ROI64_T1235 (CN 32 / MCI 17 / AD 15) and ROI64_T5 cohorts; 19 per-ROI columns defined by "Table II" of a TNSRE 2025 paper (p.1452) `[SOURCED:feature_engineer/semantic_audit.md]`. These are analysed separately and never pooled with RAW157; the 4 ROI64 task features (D007/D010/D011/D015) appear in the 37-feature summary but are excluded from the RAW157 processed datasets by design.

## 4. Feature engineering (the many "N-feature" numbers)

- **209 total definitions** = 160 base160 + 49 dataset-derived (target 260, achieved 209, extension "partial") `[SOURCED:feature_engineer/candidate_freeze.json]`.
- **33 RAW157 task features** = 14 base160 (A002, A003, A021, A023, A024, A026, A027, A028, A029, A030, A031, A033, A035, A046: fixation time/count, saccade count/amplitude/duration/speed/fractions, scanpath length, dispersion, spatial entropy, pupil median) + 19 dataset-derived (D017, D018, D021, D022, D023, D024, D025, D026, D028, D029, D030, D034, D035, D036, D037, D039, D040, D041, D047: serial correlations, coupling, gaze-event span, transition fractions, anisotropy, progression monotonicity, off-media fraction, viewing-distance IQR) `[SOURCED:processed_dataset/README.md; feature_columns_33feat.csv]`.
- **37-feature summary** = 33 RAW157 + 4 ROI64_T1235 (D007/D010/D011/D015); 37 of 65 testable task features passed a post-hoc screen (exploratory, not multiplicity-corrected) `[SOURCED:FEATURES_37_SUMMARY.md]`.
- **24-feature set** = 33 minus 9 dropped for subject completeness (D017, D022, D025, D026, D030, D034, D035, D036, D037), leaving 24 features valid for all 157 subjects × 4 tasks.
- **33-feature set** = all 33, keeping all 157 subjects, with 51 NaN cells (0.25% of 20 724) imputed per fold at model time with training median.
- **Earlier 25-feature scheme (deleted)** = different 25 features (spa_/geo_/tem_/pup_/pupbl_/sac_ prefixes) × 11 tasks = 275 columns `[SOURCED:git history, commit 0d8742b^:subject-level/README.md]`. Not part of current tree.

**Selection/screening.** Post-hoc selection rule (≥2 tasks p<0.10 for both CN-vs-MCI and CN-vs-AD; or ≥1 task MCI-vs-AD p<0.10; or ≥1 task any pairwise p<0.05; Mann–Whitney U two-sided asymptotic, tie/continuity-corrected, one value per subject per task, uncorrected p). Only 4 cells survive BH FDR q≤0.10 (D023@KD3, A002@KD3, A003@KD3, D023@KD2); none survive BY; correlated D023/A002/A003/A021 flagged as not independent `[SOURCED:FEATURES_37_SUMMARY.md §4; OTHER_FEATURES_SUMMARY.md §2]`.

**Processed tables.** V1 wide 157×98; V2 long 628×27 (24-feat); 33feat wide 157×134; 33feat long 628×36; `raw157_feature_values_long_full.csv` 33 284 rows with value/measurement_status/n_events `[SOURCED:processed_dataset/README.md]`. Extraction module (`src/begad_norm/feature_discovery/extract_macro.py`) removed from current tree, recoverable at commit `ffa36c7`.

**Fixed split.** 4 held-out AD subjects (BEGAD-001/005/006/016) appended to every validation fold, never in training; remaining 153 in 4 stratified folds (seed 42); ~115 train (75 CN / 30-31 MCI / 9 AD) per fold, val ~42-43 incl. 7 AD. Flagged non-independent across folds (same 4 AD recur), so pooled CV metrics overweight those 4 subjects; per-fold reporting recommended `[SOURCED:processed_dataset/splits/split_manifest.json; output/baseline/RESULTS.md §5]`.

## 5. Models and methods

Shared leakage-safe preprocessing: per-fold median imputation + standardization fitted on training only `[SOURCED:src/expcommon/data.py]`.

**Baselines (24 & 33 feat).** 6 classifiers (LogisticRegression, SVC-RBF, RandomForest, LightGBM, GaussianNB, MLP); no tuning; threshold 0.5; balanced weights/priors/sample weights; 10 seeds × 4 folds = 40 evals per (problem, model). Problems: 3-class CN/MCI/AD, CN-vs-MCI (pos MCI), CN-vs-AD (pos AD).

**Fusion.** One model per task (KD1-4) on its own 33 features; base families elastic-net LR + calibrated SVM-RBF; matched 12-/9-combo nested tuning by mean CV log-loss; variants: task_kd1..task_reading, fusion_equal4 (mean), fusion_learned (shared weight vector, log-loss + KL-to-uniform λ=0.1, SLSQP), fusion_equal3kd, concat132 (reference). Models unweighted intentionally (balanced weights shift probability scale under fusion) `[SOURCED:src/fusion/config.yaml; output/fusion/RESULTS.md]`.

**Contrasts.** 18 cross-task contrasts before imputation: ratio-log for D023/A002/A003, difference for signed D025/D026/D035/D036/D017/D022. Representations abs132 / abs150 (132+18) / compact84 (33 KD1 + 33 READING + 18), same two model families `[SOURCED:src/contrasts/config.yaml]`.

**Subject-level work (history only).** 25 features × 11 tasks = 275 cols; 5-fold × 20 seeds; classical baselines; an "IEEE feature-set" trial (16/20 features mapped, underperformed in 17/18 cells); EDA; splits; EN/VI reports `[SOURCED:git history — commits 3669a92, 5e57bef, b05cc5b]`.

**Normative model (history only).** Commits b7d1788→8caf4a6: `src/begad_norm/normative/` — learned normative-deviation model (11×25 matrices, encoder 25→128→128, per-epoch CO latent bank, comparator `[z, μ, z−μ, z·μ]`, 64-d deviations, masked deepset mean‖max, MLP head); PyTorch CPU, AdamW, early stop on val AUC. Results 3 problems × 5 folds × 20 seeds = 300 evals committed under `outputs/normative/` `[SOURCED:git history commit 8caf4a6]`. Branch `origin/normative-model` also exists.

## 6. Outputs / results

**Baseline 24 (mean±std over 40 evals):** CN-vs-AD most learnable — LR AUC 0.889 / F1 0.649 / AD recall 0.61; RF best AUC 0.899 but 0.5-threshold sensitivity 0.23. CN-vs-MCI hardest (best AUC 0.65, F1≤0.45). SVM-RBF ranks on AUC but predicts almost nothing at 0.5 (sigmoid calibration compresses; 3-class AD recall 0.04). GNB most AD-sensitive (recall 0.32) at weak acc (0.46); 3-class LightGBM best acc 0.599 `[SOURCED:output/baseline/RESULTS.md]`.

**Baseline 33 vs 24 (paired):** 9 added features help 3-class and CN-vs-MCI (all 6 models gain CN-vs-MCI acc/AUC; AUC gain 0.03–0.09); CN-vs-AD flat (LR 0.891 / F1 0.671) `[SOURCED:output/baseline_33feat/RESULTS.md §3]`.

**Fusion:** splitting by task does NOT generally reduce overfitting — concat132 best under elastic-net (OVR AUC 0.737 / AUC 0.698 / 0.881). Exception: SVM-RBF on CN-vs-AD, every fusion variant beats concat132 in all 4 folds (+0.116 AUC) — fusion acts as a regularizer. Tasks carry partially complementary info (fused inner OOF CE lower than every single task in all 6 problem/model cells), but validation advantage small vs fold variance. Learned weights beat equal only marginally on training (ΔCE 0.002–0.004) and are equal-or-worse on validation in 10/12 comparisons — recommend equal weights `[SOURCED:output/fusion/RESULTS.md]`.

**Contrasts:** help only CN-vs-AD (EN AUC 0.881→0.906; SVM 0.717→0.787, 4/4 folds positive); CN-vs-MCI and 3-class slightly negative (3/4 folds — extra features fit training better but generalize worse). compact84 loses everywhere (−0.036 to −0.082 AUC); KD2/KD3 absolutes not recovered `[SOURCED:output/contrasts/RESULTS.md]`.

**Figures.** `output/features/figure/` one 4-panel boxplot per RAW157 feature (33 × PNG+PDF), cross-checked against FEATURES_37_SUMMARY (379/396 p-cells match; 17 differ due to tie structure) `[SOURCED:output/features/figure/FIGURE_INDEX.md]`.

## 7. Key observations and open questions

Observations:
1. Three regimes: (a) subject-level 25×11 features, 5-fold×20 seeds, IEEE trial, normative ML; (b) frozen 209-def catalogue → 37-feature screen → 24/33-feat datasets; (c) baseline/fusion/contrasts suite. Regimes used different features and splits `[INFERRED]`.
2. Strongest reproducible signal is CN-vs-AD (AUC ≈0.88–0.91 across families); CN-vs-MCI consistently hard.
3. Careful leakage control (per-fold impute/scale, nested CV, frozen splits) and honest reporting (per-fold deltas, caveats on held-out AD reuse and SVM threshold artefact).
4. Feature selection was post-hoc on the same 157 subjects; only 4 p-cells survive FDR — headline numbers are not unbiased effect sizes.

Open questions:
- The TNSRE 2025 reference paper is cited by venue/year/page only — no full title/DOI anywhere.
- `dataset/` vs `dataset_v2/` identical file lists; no surviving explanation of what differs.
- No demographic composition (age/sex/education) documented; "no demographic adjustment was possible".
- Normative-vs-baseline comparison reports deleted (history only).
- 4-fold held-out design (current) vs 5-fold×20 seeds (earlier) are not directly comparable.

## 8. Reproducibility notes

Each family records provenance (git commit, config/data sha256, versions) and re-run commands, e.g. baseline `PYTHONPATH=src .venv/bin/python -m baseline.run_baselines --config src/baseline/config.yaml` (refuses overwrite); 33-feat rebuild via `processed_dataset/make_33feat_dataset.py`; contrasts via `-m contrasts.run_contrasts` (+ `--smoke`); figures via `.venv-figs/bin/python output/features/figure/plot_selected_features.py`. Two post-run artifact completions were applied via `src/expcommon/postprocess_first_full_runs.py` (proba row merging; 3-class balanced-accuracy backfill), with runner code fixed afterward `[SOURCED:output/fusion/RESULTS.md §6; output/contrasts/RESULTS.md §5]`.