# Kết quả các features khác: bản tổng hợp lưu trữ (các file đã xóa ngày 2026-10-28)

Run gốc: `2026-10-27_v2_expanded260` (freeze `freeze_35a60cb67109`, seed 20261027, không huấn luyện classifier).
Cohorts: **RAW157** = export sự kiện thô của Tobii, 157 người tham gia baseline (CN 100 / MCI 41 / AD 16),
chỉ macro, các task KD1/KD2/KD3/READING; **ROI64_T1235** = workbook ROI gốc, 64 người (CN 32 / MCI 17 / AD 15),
micro và macro, KD1-3 + reading; **ROI64_T5** = cùng 64 người, chỉ reading, phân tích sensitivity.
Các cohorts không bao giờ được gộp chung.

File này lưu giữ các kết quả chính của mọi họ feature **không** nằm trong
`FEATURES_37_SUMMARY.md`: màn lọc macro chính, màn lọc micro, xếp hạng task-ROI,
bằng chứng vùng posthoc, các audit và hình vẽ. Các file gốc đã bị xóa ngày 2026-10-28 trong đợt dọn dẹp dự án;
mọi con số dưới đây được trích từ các file đó. Dữ liệu thô đầy đủ vẫn khôi phục được từ lịch sử git.

## 1. Danh mục feature và màn lọc

- 160 định nghĩa base + 49 định nghĩa từ dataset = **209** (mục tiêu 260; mở rộng một phần: 49/100).
  78/209 có nguồn gốc xác định (30 base, 48 mới).
- Macro contexts đủ điều kiện cho màn lọc chính: RAW157 **214**, ROI64_T1235 **39**, ROI64_T5 **10**.
  Micro contexts đủ điều kiện: ROI64_T1235 **1338**, ROI64_T5 **526**.
- Màn lọc chính = khả thi VÀ p_CN_AD < 0.20 VÀ p_CN_MCI < 0.20 (Mann-Whitney U hai phía, p thô).
- Qua màn lọc chính: macro RAW157 **40**, ROI64_T1235 **7**, ROI64_T5 **0**; với hỗ trợ FDR
  (joint BH q <= 0.10): RAW157 **4**, còn lại **0**; trong compact panel: RAW157 **18**, ROI64_T1235 **4**,
  ROI64_T5 **0**.
- Một định nghĩa tạo ra nhiều giả thuyết (theo task, ROI, contrast), nên 209 định nghĩa không phải là số
  phép kiểm định. Việc chọn lọc mang tính thăm dò; mọi mục đích dự đoán phải lặp lại việc chọn feature trong
  dữ liệu huấn luyện hoặc dùng cohort độc lập.

## 2. Kết quả macro

### Được FDR hỗ trợ (joint BH q <= 0.10), đều thuộc RAW157

| ID@scope | Tên | p CN-AD / CN-MCI / MCI-AD | delta CN-AD [CI] | delta CN-MCI [CI] | delta MCI-AD | q joint (BH) | N CN/MCI/AD | bootstrap chọn / hướng | LOO-AD giữ |
|---|---|---|---|---|---|---|---|---|---|
| D023@KD3 | gaze_event_span | 0.000136 / 0.000246 / 0.040 | 0.60 [0.29, 0.85] | 0.39 [0.21, 0.58] | 0.35 | 0.053 | 100/41/16 | 0.99 / 1.00 | 1.00 |
| A002@KD3 | total_fixation_time | 9.5e-05 / 0.000868 / 0.077 | 0.61 [0.31, 0.85] | 0.36 [0.16, 0.56] | 0.30 | 0.093 | 100/41/16 | 0.97 / 1.00 | 1.00 |
| A003@KD3 | task_fixation_count | 0.000181 / 0.002 / 0.033 | 0.58 [0.32, 0.80] | 0.34 [0.15, 0.51] | 0.37 | 0.100 | 100/41/16 | 0.98 / 1.00 | 1.00 |
| D023@KD2 | gaze_event_span | 0.002 / 0.00082 / 0.419 | 0.49 [0.16, 0.77] | 0.36 [0.16, 0.55] | 0.14 | 0.100 | 100/41/16 | 0.95 / 0.99 | 1.00 |

Hiệu chỉnh BY không giữ lại feature nào trong số này. A002, A003 và D023 là thành viên của bộ 37 features
(xem FEATURES_37_SUMMARY.md); D023/A002/A003/A021 tương quan mạnh (đọc thẻ lâu hơn = nhiều fixation và
saccade hơn), nên chúng không phải là các xác nhận độc lập.

### Compact macro panel (22 dòng, `macro_compact_panel.csv`)

RAW157 (18): D023@KD3, D023@KD2, A003@KD3, D036@KD3, D023@READING, A003@KD2, A003@KD1, A021@KD2, A027@KD3,
A046@READING, D029@KD3, A021@KD3, D023@KD1, A002@KD2, A030@KD1, D030@KD3, D034@READING, D025@KD1.
ROI64_T1235 (4): D007@KD1, D007@KD2, D011@KD1, D010@KD3.

### Macro features được chọn (47 dòng, `macro_selected_main.csv`; chỉ nêu các dòng tiêu biểu)

RAW157 stable candidates ngoài bộ FDR gồm D036@KD3 `incoming_amplitude_fixation_duration_coupling`
(0.008 / 0.000121 / 0.729, delta -0.42/-0.41), D023@READING (0.008 / 0.016 / 0.245), A003@KD2
(0.027 / 4.5e-05 / 0.972), A046@READING (0.036 / 0.028 / 0.467, delta -0.33/-0.24), D029@KD3
(0.002 / 0.039 / 0.169, delta -0.48), A027@KD3 (0.040 / 0.027 / 0.003), A021@KD2/KD3, D030@KD3, D023@KD1,
A003@KD1, A002@KD2, A030@KD1, D034@READING, D025@KD1; các dòng exploratory: D040@KD2 (0.004 / 0.063),
D031@KD2/KD3, A046@KD2/KD3, A023@KD2/KD3/READING, A030@READING, A035@READING/KD2, D022@KD3, D038@KD3,
D039@READING, D026@KD1, A027@READING, A021@KD1, A028@READING, A029@KD1, A046@KD1, D029@KD1, A023@KD1.

ROI64_T1235 macro (7, không feature nào được FDR hỗ trợ): D007@KD1 (0.015 / 0.005 / 0.806, delta 0.45/0.49,
stable), D010@KD3 (0.021 / 0.037 / 1.000, delta -0.45, stable), D007@KD2 (0.062 / 0.011 / 0.566, stable),
D011@KD1 (0.020 / 0.062 / 0.509, delta 0.42, stable), D009@KD3 (0.084 / 0.151 / 0.549, exploratory),
D012@KD2 (0.185 / 0.094 / 0.890, exploratory), D013@KD1 (0.189 / 0.123 / 1.000, exploratory).

### Số lượng chọn theo nguồn gốc

Macro: RAW157 base160 23 + dataset100 17; ROI64_T1235 0 + 7. Micro: ROI64_T1235 base160 50 + dataset100 57;
ROI64_T5 14 + 27.

## 3. Kết quả micro (ROI64_T1235; ROI64_T5 = sensitivity chỉ reading)

- 1338 micro contexts đủ điều kiện ở ROI64_T1235 (526 ở ROI64_T5). Không micro context nào đạt joint BH
  q <= 0.10 và không vùng nào đạt q_region <= 0.10, nên kết quả micro chỉ mang tính thăm dò hoặc ổn định
  theo bootstrap.
- 65 cặp task-ROI qua quy tắc chính ở ROI64_T1235 (hiển thị 20; `roi_ranking_all.csv`, `roi_top20.csv`).
  Top 5 (feature neo, p CN-AD / CN-MCI / MCI-AD; delta; p vùng; tần suất top-20 bootstrap):

| Hạng | Task | ROI | Feature neo | p neo (CN-AD / CN-MCI / MCI-AD) | p_region | tần suất top-20 | Bằng chứng |
|---|---|---|---|---|---|---|---|
| 1 | KD1 | ROI 16 | D002 | 0.011 / 0.016 / 0.982 | 0.131 | 0.58 | stable_candidate |
| 2 | READING | ROI 138 | D001 | 0.019 / 0.024 / 1.000 | 0.196 | 0.41 | exploratory_only |
| 3 | KD1 | ROI 23 | M001 | 0.001 / 0.026 / 0.276 | 0.207 | 0.45 | stable_candidate |
| 4 | KD1 | ROI 15 | M001 | 0.006 / 0.028 / 0.610 | 0.221 | 0.45 | stable_candidate |
| 5 | KD3 | ROI 16 | M051 | 0.011 / 0.028 / 0.296 | 0.226 | 0.34 | stable_candidate |

- Sensitivity chỉ reading ở T5 (25 vùng, hiển thị 20). Đầu bảng:
  READING ROI 134 (M001: 0.013 / 0.008 / 0.948, p_region 0.090, freq 0.54, stable),
  READING ROI 88 (D005: 0.012 / 0.024 / 0.815, p_region 0.166, freq 0.73, stable),
  READING ROI 50 (D003: 0.037 / 0.000929 / 0.706, freq 0.48, exploratory).
- Không có hình học/nội dung/thứ tự đọc của ROI cho mọi vùng (`text_unavailable`).

## 4. Xếp hạng task-ROI và reserves theo từng cặp so sánh

Tổng kết bốn task (`task_ranking.csv`): | Cohort | Task | ROI kỳ vọng | testable | được chọn chính | vùng qua CN-AD / CN-MCI / MCI-AD |
| ROI64_T1235 | KD1 | 40 | 40 | 18 | 36 / 34 / 32 |
| ROI64_T1235 | KD2 | 40 | 40 | 16 | 33 / 29 / 26 |
| ROI64_T1235 | KD3 | 40 | 38 | 15 | 33 / 29 / 24 |
| ROI64_T1235 | READING | 159 | 57 | 16 | 60 / 33 / 51 |
| ROI64_T5 | READING | 159 | 80 | 25 | 73 / 67 / 58 |

- 211 vùng dự phòng trong `roi_pair_specific_reserves.csv`. Số lần qua MCI-AD theo cohort: RAW157 40,
  ROI64_T1235 252, ROI64_T5 109. Các kết quả MCI-AD hàng đầu: RAW157 D047@KD3 (p=0.000717, delta 0.58);
  ROI64_T1235 D004@READING ROI 068 (p=0.002, delta 0.79); RAW157 A024@KD3 (p=0.003, delta -0.52);
  RAW157 D047@KD2 (p=0.003, delta 0.50); ROI64_T1235 D004@READING ROI 015 (p=0.004, delta 0.78);
  ROI64_T5 D005@READING ROI 056 (p=0.004, delta 0.67); ROI64_T1235 D004@KD1 ROI 034 (p=0.006, delta 0.64);
  ROI64_T1235 D003@READING ROI 041 (p=0.007, delta -0.71); ROI64_T5 D004@READING ROI 138 (p=0.013, delta 0.59);
  ROI64_T1235 D004@KD2 ROI 027 (p=0.014, delta -0.54).

## 5. Độ ổn định, mức ảnh hưởng và hoán vị

- Kiểm định hoán vị (Monte Carlo +1, 9999 hoán vị): **381**; số lần bất đồng với kiểm định chính tại
  p < 0.20: **8** (`permutation_validation.csv`).
- Bootstrap (500 lần lặp, `bootstrap_stability.csv`): tần suất được chọn ở màn chính D023@KD3 = 0.99;
  A002@KD3 0.97; A003@KD3 0.98; D023@KD2 0.95; mức đồng thuận hướng là 1.00 cho mọi dòng được FDR hỗ trợ.
- Sensitivity theo coverage (số được chọn, `sensitivity_coverage.csv`): RAW157 macro 40 tại cov 0.5 / 0.6 / 0.75;
  ROI64_T1235 macro 8 / 7 / 6; ROI64_T5 macro 0; ROI64_T1235 micro 129 / 107 / 64; ROI64_T5 micro 50 / 41 / 6.
- Ảnh hưởng của AD (leave-one-AD-out, `ad_influence.csv`): hầu hết stable candidates có
  prop_main_screen_preserved = 1.0 và không đảo dấu; ví dụ D023@KD3 giữ 1.0, sign_flips CN-AD = 0.

## 6. Audit chất lượng và dư thừa

- `quality_features.csv` (948 dòng): chỉ số chất lượng dữ liệu theo người tham gia-task (tỷ lệ hợp lệ của
  tracking/pupil, tỷ lệ gap và off-media, tỷ lệ ROI thiếu, trạng thái calibration).
- `redundancy_audit.csv` (2411 dòng): tương quan Spearman từng cặp giữa các macro feature theo task, gồm cả
  tương quan sau khi trung tâm hóa theo nhóm (ví dụ A003@KD1 vs A021@KD1 rho = 0.68, trung tâm hóa 0.65).
- `proposal_audit.csv` (88 dòng): quyết định chấp nhận cho 49 định nghĩa từ dataset (dòng D), với bằng chứng
  từ Bảng II của bài báo (TNSRE 2025, tr. 1452) và hồ sơ workbook không nhãn.
- `dataset_proposed_features.csv`: đầy đủ quy tắc hỗ trợ của 49 định nghĩa dòng D (được giữ lại).

## 7. Bằng chứng vùng posthoc (các file đã xóa; danh sách chính dưới đây)

`control_regions_KD20_READING15.csv` (35 dòng: top-20 vùng KD + top-15 vùng reading đối chứng),
`mci_ad_regions_KD10_READING10.csv` (20 dòng: top-10 vùng KD + top-10 vùng reading theo MCI-AD),
`mci_ad_region_evidence_all.csv` (193 dòng: bảng bằng chứng vùng đầy đủ).

Vùng đối chứng KD (top 10 trong 20; feature neo, p CN-AD của neo): KD1 ROI 016 (D002, 0.011),
KD1 ROI 023 (M001, 0.001), KD1 ROI 015 (M001, 0.006), KD3 ROI 016 (M051, 0.011), KD2 ROI 006 (D003, 0.028),
KD1 ROI 005 (M051, 0.030), KD1 ROI 027 (M001, 0.035), KD1 ROI 028 (D004, 0.002), KD3 ROI 002 (D002, 0.041),
KD2 ROI 038 (D005, 0.003).
Vùng đối chứng reading (top 10 trong 15): ROI 138 (D001, 0.019), ROI 088 (M051, 0.012), ROI 005 (M001, 0.036),
ROI 131 (D003, 0.043), ROI 134 (D004, 0.057), ROI 156 (D003, 0.081), ROI 079 (M001, 0.081),
ROI 105 (D003, 0.023), ROI 007 (D005, 0.089), ROI 000 (D002, 0.109).

Vùng MCI-AD của KD (top 10; neo, p MCI-AD, delta): KD3 ROI 029 (D001, 0.010, 0.75),
KD1 ROI 034 (D004, 0.047, 0.64), KD3 ROI 032 (M051, 0.092, 0.62), KD2 ROI 027 (D004, 0.112, -0.54),
KD1 ROI 026 (D003, 0.147, 0.49), KD1 ROI 008 (D002, 0.163, -0.47), KD2 ROI 010 (D003, 0.175, 0.41),
KD3 ROI 022 (D004, 0.197, 0.50), KD2 ROI 012 (D003, 0.236, 0.46), KD2 ROI 034 (D003, 0.247, -0.48).
Vùng MCI-AD của reading (top 10): ROI 068 (D004, 0.013, 0.79), ROI 015 (D004, 0.027, 0.78),
ROI 041 (D003, 0.052, -0.71), ROI 045 (M001, 0.091, 0.62), ROI 105 (D001, 0.120, -0.61),
ROI 010 (D001, 0.181, 0.59), ROI 057 (D003, 0.202, -0.57), ROI 079 (M001, 0.209, 0.52),
ROI 096 (D005, 0.221, -0.54), ROI 003 (D001, 0.260, 0.44).

Không p vùng nào qua hiệu chỉnh BH/BY (mọi q = 1.0); bằng chứng vùng chỉ mang tính thăm dò.

## 8. Các feature task-level suýt đạt (nằm trong danh sách union 45, ngoài bộ 37)

8 feature này chỉ đạt tối đa một trong ba điều kiện posthoc nên bị loại khỏi FEATURES_37_SUMMARY.md.
Giá trị p: CN-AD / CN-MCI / MCI-AD theo task (KD1/KD2/KD3/READING).

| Cohort | ID | Tên | Điều kiện đạt | Bằng chứng tốt nhất |
|---|---|---|---|---|
| RAW157 | A036 | task_transition_entropy | chỉ cond2 | MCI-AD KD2 = 0.139 |
| RAW157 | D020 | fixation_duration_successive_change_median | chỉ cond2 | MCI-AD READING = 0.143 |
| RAW157 | D031 | transition_direction_entropy | chỉ cond1 | CN-AD KD2 = 0.077; CN-MCI KD2 = 0.057 |
| RAW157 | D033 | post_large_saccade_correction_fraction | chỉ cond2 | MCI-AD KD2 = 0.143 |
| RAW157 | D038 | immediate_return_fraction | chỉ cond2 | CN-AD KD3 = 0.063; MCI-AD KD2 = 0.200 |
| RAW157 | D046 | binocular_horizontal_disparity_iqr | chỉ cond2 | MCI-AD READING = 0.169 |
| ROI64_T1235 | D008 | refixated_target_fraction | chỉ cond2 | MCI-AD KD3 = 0.183 |
| ROI64_T1235 | D012 | leftward_saccade_target_fraction | chỉ cond2 | MCI-AD READING = 0.138 |

## 9. Hình vẽ (đã xóa)

- `fig1_availability_heatmap` (png/pdf): mức khả dụng của các feature contexts theo cohort và cấp độ.
- `fig2_macro_forest` (png/pdf): kích thước hiệu ứng (Cliff's delta kèm CI) của các macro feature được chọn.
- `fig3_task_roi_evidence_ROI64_T1235` và `fig3_task_roi_evidence_ROI64_T5` (png/pdf): xếp hạng bằng chứng
  task-ROI của màn micro.
- `fig4_top20_bootstrap` (png/pdf): tần suất được chọn qua bootstrap của top-20 vùng.
- `fig5_participant_distributions` (png/pdf): phân bố giá trị theo từng người tham gia.

## 10. Danh mục các file đã xóa

- Kết quả: RESULTS_EN.md, RESULTS_VI.md.
- Macro: macro_stats_all.csv (881 dòng), macro_selected_main.csv (47), macro_selected_three_pairs.csv (47),
  macro_compact_panel.csv (22), macro_pair_specific.csv (237), macro_features_long.csv (37659).
- Micro: micro_stats_all.csv (10113), micro_selected_main.csv (148), micro_selected_top20.csv (80),
  micro_pair_specific.csv (1319), micro_features_long.csv (214080).
- ROI/task: roi_ranking_all.csv (438), roi_top20.csv (40), roi_pair_specific_reserves.csv (211),
  task_ranking.csv (5), feature_origin_summary.csv.
- Audit/QA: quality_features.csv (948), redundancy_audit.csv (2411), proposal_audit.csv (88),
  sensitivity_coverage.csv, sensitivity_coverage_selected.csv (343), permutation_validation.csv (381),
  bootstrap_stability.csv (3030), ad_influence.csv (195).
- Manifests: multiplicity_manifest.json, stage_state.json.
- Posthoc: top45_features.csv (45), task_level_union_conditions.csv (45),
  control_regions_KD20_READING15.csv (35), mci_ad_regions_KD10_READING10.csv (20),
  mci_ad_region_evidence_all.csv (193).
- Cache (pickle trung gian, ~140 MB): long.pkl, raw_events.pkl, report.pkl, stability.pkl, stats.pkl,
  workbooks.pkl.
- Hồ sơ chẩn đoán: raw_event_profile.csv, t1235_column_profile.csv, t5_column_profile.csv.
- Hình vẽ: 5 hình, mỗi hình có cả png lẫn pdf.

## 11. Những gì được giữ lại (bộ tài liệu 37 features)

`FEATURES_37_SUMMARY.md` và `.vi.md`,
`task_level_union_conditions_strict.csv`, `feature_registry.csv`,
`DATASET_PROPOSED_100_FEATURES.md` và `.vi.md`, `dataset_proposed_features.csv`, `semantic_audit.md` và
`semantic_audit.csv`, `candidate_freeze.json`, và hồ sơ run (`run_config.yaml`, `run_metadata.json`,
`source_manifest.json`, `subject_manifest.csv`, `task_roi_manifest.csv`).

## 12. Các hạn chế chính (từ RESULTS_EN.md)

Không có hình học/nội dung/thứ tự đọc của ROI; không có chú giải audio/TextGrid/lỗi; totalFixTime không phải
dwell cộng dồn; số saccade trong workbook không nhất quán; các trường pupil của T1235 đảo dấu so với một hằng
số chưa xác định; pupil thô chỉ có ở các mốc sự kiện; không có dữ liệu nhân khẩu; AD n = 15-16. Đây là kết quả
chỉ của cohort này, không phải sự xác nhận hay bác bỏ các giả thuyết trong tài liệu.
