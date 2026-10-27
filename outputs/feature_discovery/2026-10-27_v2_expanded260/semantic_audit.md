# Semantic audit

Every check below was recomputed from the inspected sources in this run (`semantic_audit.csv`).

## Cohorts (CN = repository CO/HC, code 0)

```
group         CN  MCI  AD
cohort_id                
RAW157       100   41  16
ROI64_T1235   32   17  15
ROI64_T5      32   17  15
```

RAW157 (raw Tobii event exports) and ROI64 (ROI summary workbooks) are analysed separately: workbook fixation totals exceed raw event counts, the workbooks were produced by a different recalibrated pipeline, and T5 differs from T1235 task 5 row by row (S14).

## Checks

### S01: T5.xlsx diffPupL/R
- Evidence: max |diffPupL-(endPupL-startPupL)| = 5.00e-16
- Severity: info
- Resolution: diff is an exact dependency of start/end; only the difference (D001) is used
- Blocked IDs: M044;M045;M046

### S02: T1235.xlsx *B pupil fields
- Evidence: max |diffPupL-(startPupLB-endPupLB)| = 5.01e-14; within-participant SD of (T5 startPupL + T1235 startPupLB) median 0.028 mm vs SD of difference 0.395 => B = c - pupil (sign inverted)
- Severity: high
- Resolution: B fields treated as constant-relative with inverted sign; only constant-free contrasts used
- Blocked IDs: M044;A041;A050

### S03_T5: T5.xlsx saccade counts
- Evidence: regSaccades>nTobiiSaccades in 2850/6691 rows; tiny+long>nTobii in 2745; nTobii==0 with saccadeTotLength>0 in 3307
- Severity: high
- Resolution: no ratios from these counts; regSaccades/saccadeTotLength only as stand-alone D003/D004
- Blocked IDs: M029;M030

### S04_T5: T5.xlsx totalFixTime
- Evidence: totalFixTime==0 in 389 rows, all with nFixations==1 (True); per-fixation value <0.06 s in 48.8% of multi-fixation rows
- Severity: high
- Resolution: not a summed fixation duration; blocked
- Blocked IDs: M003;M008;A010;A037

### S03_T1235: T1235.xlsx saccade counts
- Evidence: regSaccades>nTobiiSaccades in 5265/13139 rows; tiny+long>nTobii in 6065; nTobii==0 with saccadeTotLength>0 in 6178
- Severity: high
- Resolution: no ratios from these counts; regSaccades/saccadeTotLength only as stand-alone D003/D004
- Blocked IDs: M029;M030

### S04_T1235: T1235.xlsx totalFixTime
- Evidence: totalFixTime==0 in 643 rows, all with nFixations==1 (True); per-fixation value <0.06 s in 47.3% of multi-fixation rows
- Severity: high
- Resolution: not a summed fixation duration; blocked
- Blocked IDs: M003;M008;A010;A037

### S05: T*_clean.xlsx (subject-level/src/begad_norm/cleaning.py)
- Evidence: clean T5 drops 474 rows and T1235 drops 653 rows; totalFixTime of 643 single-fixation T1235 rows imputed
- Severity: high
- Resolution: original workbooks used; no imputed value enters primary tests
- Blocked IDs: none

### S06: pupil <=0 coding
- Evidence: T5 start/end <=0 (missing code 0): 474 rows; T1235 *B <=0 (valid constant-relative values): 653 rows
- Severity: high
- Resolution: T5 0 -> invalid_measurement; T1235 negatives retained
- Blocked IDs: none

### S07: legacy tem_ifi/tem_velocity (subject-level/docs/baseline/eda_summary.md)
- Evidence: half-dwell interval approximation; not measured gaps
- Severity: info
- Resolution: not reused; D021 uses actual onset/offset timestamps
- Blocked IDs: none

### S08: legacy tem_trans_entropy
- Evidence: spatial histogram, not transitions
- Severity: info
- Resolution: A036 computed from gap-free fixation transitions
- Blocked IDs: none

### S09: legacy pup_slope
- Evidence: uses fixation index
- Severity: info
- Resolution: D045 uses onset seconds
- Blocked IDs: none

### S10: IEEE mapper regression
- Evidence: turning angle >90 deg called regression
- Severity: info
- Resolution: turning reversals are D032; reading-order regressions blocked (no ROI order)
- Blocked IDs: M018;M019;M020;A013

### S12: paper Table II caption
- Evidence: 'Last fixation refers to the fixation at the end of the last saccade that began from the ROI area'
- Severity: high
- Resolution: endPup*/endSpeechDelay use the exit-landing fixation
- Blocked IDs: M045;M046;M062;A064;A065;A066

### S13_T5: T5.xlsx speechDelay/endSpeechDelay
- Evidence: speechDelay min 0.0035, max 65.2293; in [9.99,10) 2 rows, >=10 in 18, <=0 in 0; endSpeechDelay max 9.9996, <0 in 7.0%
- Severity: medium
- Resolution: T1235 delays are right-truncated below 10 s (observation window); speechDelay <=0 or >=10 s is treated as censored in both cohorts so T5 uses the same window
- Blocked IDs: none

### S13_T1235: T1235.xlsx speechDelay/endSpeechDelay
- Evidence: speechDelay min 0.0004, max 9.9999; in [9.99,10) 33 rows, >=10 in 0, <=0 in 0; endSpeechDelay max 9.9996, <0 in 16.8%
- Severity: medium
- Resolution: T1235 delays are right-truncated below 10 s (observation window); speechDelay <=0 or >=10 s is treated as censored in both cohorts so T5 uses the same window
- Blocked IDs: none

### S14: T5 vs T1235 task 5
- Evidence: row-wise equality fractions {"nFixations": 0.763, "totalSpokenTime": 1.0, "speechDelay": 0.895, "endSpeechDelay": 0.812, "saccadeTotLength": 0.74, "regSaccades": 0.801}
- Severity: high
- Resolution: different exports; analysed as separate cohorts (T5 = reading sensitivity)
- Blocked IDs: none

### S15: workbook task codes vs raw Task
- Evidence: label-blind best Spearman match of per-person fixation totals (n=63): {"wb1": ["raw3", 0.64], "wb2": ["raw4", 0.64], "wb3": ["raw5", 0.57], "wb5": ["raw9", 0.66]}; workbook totals exceed raw counts (median wb1/raw3 ratio 1.47)
- Severity: high
- Resolution: mapping 1/2/3/5 -> raw 3/4/5/9 = KD1/KD2/KD3/READING; ROI64 and RAW157 not pooled
- Blocked IDs: none

### S16: raw fixation rows
- Evidence: 0.9993 of fixations have onset+Final rows; median (Final ts - onset ts - duration) = -3.0 ms; 2095 task-straddling fixations excluded
- Severity: info
- Resolution: offset = onset + GazeEventDuration
- Blocked IDs: none

### S17: raw SaccadicAmplitude
- Evidence: Pearson r between exported amplitude and fixation-to-fixation angle from positions: median 0.990 over 23 participants
- Severity: info
- Resolution: amplitude = incoming fixation-to-fixation angle; used only on gap-free transitions
- Blocked IDs: none

### S18: raw transitions
- Evidence: status fractions {'gap_free': 0.853, 'gap': 0.145, 'overlap_anomaly': 0.002}; saccade events per gap-free transition (3=3+) {1: 0.8, 2: 0.181, 3: 0.016, 0: 0.003}
- Severity: medium
- Resolution: matched saccades = gap-free with exactly one event; gaps never bridged
- Blocked IDs: none

### S19: raw FixationPoint (MCSpx) NaN
- Evidence: 8285 valid fixations with NaN MCS; gaze-sample mean inside media box for 3.6%
- Severity: info
- Resolution: NaN MCS = off-media fixation (D041)
- Blocked IDs: none

### S20: dataset vs dataset_v2
- Evidence: 205 vs 205 files; v2 lacks per-eye mm gaze, eye position and media columns and has fewer fixation rows (e.g. BEG-101 reading 275 vs 243 Final fixations)
- Severity: medium
- Resolution: dataset/ used as primary raw source
- Blocked IDs: none

### S21: diagnostic codes
- Evidence: roster 0/1/2 labelled CO/MCI/AD; raw file Group equals roster code for 100%; workbook code equals roster code for 100% of 63 shared IDs; workbook ID prefixes consistent: True; 1 workbook ID(s) absent from raw roster
- Severity: info
- Resolution: CN = repository CO/HC (code 0); unknown codes raise an error
- Blocked IDs: none

### S22: ROI order
- Evidence: ROI IDs 0..39 / 0..158 exist but no stimulus metadata verifies reading order or lines
- Severity: high
- Resolution: rank/line features blocked
- Blocked IDs: M015-M020;M040;M070;M078;A013-A020;A074

### S23: raw PupilLeft/RightBaseline
- Evidence: binocular baseline column missing for 16 roster participants; |baseline - mean of first 10 valid instruction (task 0) fixations| median 0.450 mm, max 1.083 mm (n=141)
- Severity: high
- Resolution: protocol baseline rule not reproduced; baseline-corrected features blocked
- Blocked IDs: A041;A047;A050
