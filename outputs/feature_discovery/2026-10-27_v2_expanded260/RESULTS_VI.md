# BEGAD feature discovery: kết quả (bản tiếng Việt)

Run: `2026-10-27_v2_expanded260` · freeze `freeze_35a60cb67109` · seed 20261027 · không train classifier.
Nhóm chứng được báo cáo là **CN** (= mã 0 CO/HC của repository, đã kiểm chứng với roster và tiền tố ID).

**Dữ liệu thực sự đã phân tích.** RAW157 (export sự kiện Tobii thô, 157 người: CN 100 / MCI 41 / AD 16; chỉ
macro, KD1/KD2/KD3/READING) và ROI64_T1235 (workbook ROI gốc, 64 người: CN 32 / MCI 17 / AD 15; KD1-3 + đọc,
micro và macro, là cohort xếp hạng task–ROI chính). ROI64_T5 (cùng 64 người, chỉ đọc, export khác) là phân tích
độ nhạy riêng, không phải tái lặp độc lập. Các cohort không được gộp.

## Câu trả lời

Catalogue gồm 160 định nghĩa gốc + 49 đề xuất từ dataset = 209 (mục tiêu 260; **mở rộng chưa đủ: 49/100**).
Có 78/209 định nghĩa đủ nguồn (30 gốc, 48 mới); macro-context đủ điều kiện tiêu chí chính: RAW157: 214, ROI64_T1235: 39, ROI64_T5: 10;
micro-context đủ điều kiện: ROI64_T1235: 1338, ROI64_T5: 526.
Có macro-context qua tiêu chí chính: RAW157: 40, ROI64_T1235: 7, ROI64_T5: 0; trong đó có hỗ trợ FDR (joint BH q ≤ 0.10): RAW157: 4, ROI64_T1235: 0, ROI64_T5: 0;
thuộc panel gọn: RAW157: 18, ROI64_T1235: 4, ROI64_T5: 0.
Có 65 cặp task–ROI qua tiêu chí chính trong ROI64_T1235; hiển thị 20 cặp.
(Độ nhạy chỉ đọc T5: 25 vùng, hiển thị 20.)

Một định nghĩa sinh ra nhiều giả thuyết (mỗi task, mỗi ROI, mỗi cặp so sánh), nên 209 định nghĩa không phải số
phép kiểm định. Tiêu chí chính là sàng lọc thăm dò: feasible AND p_CN_AD < 0.20 AND p_CN_MCI < 0.20 (p thô,
Mann–Whitney hai phía). Lựa chọn trên toàn bộ dữ liệu này là thăm dò; mọi đánh giá dự đoán sau này phải lặp lại
việc chọn feature/ROI bên trong dữ liệu train hoặc dùng cohort xác nhận độc lập.

**Đóng góp theo nguồn gốc (cùng một họ đa kiểm định).** Macro được chọn theo origin:
origin       base160  dataset100
cohort_id                       
RAW157            23          17
ROI64_T1235        0           7
Micro được chọn theo origin:
origin       base160  dataset100
cohort_id                       
ROI64_T1235       50          57
ROI64_T5          14          27
Không có quota cho nguồn gốc nào; các D-feature không phải cohort xác nhận độc lập.

**Hỗ trợ sau hiệu chỉnh đa kiểm định.** Macro có joint BH q ≤ 0.10 (họ joint của từng cohort):
- RAW157 D023@KD3 `gaze_event_span` (dataset100): q_joint=0.053, BY không còn hỗ trợ
- RAW157 A002@KD3 `total_fixation_time` (base160): q_joint=0.093, BY không còn hỗ trợ
- RAW157 A003@KD3 `task_fixation_count` (base160): q_joint=0.100, BY không còn hỗ trợ
- RAW157 D023@KD2 `gaze_event_span` (dataset100): q_joint=0.100, BY không còn hỗ trợ
Không micro-context nào có joint BH q ≤ 0.10 và không vùng nào có q_region ≤ 0.10 (xem bảng ROI). Kết quả micro
vì vậy chỉ là `exploratory_only` hoặc ổn định theo bootstrap, không được hỗ trợ FDR.

**Panel macro gọn** (qua tiêu chí chính, |δ| ≥ 0.20 cả hai so sánh với CN, hướng ổn định ≥ 80%, tần suất chọn ≥ 60%,
loại tương quan |ρ| ≥ 0.90):
- RAW157 D023@KD3 `gaze_event_span`
- RAW157 D023@KD2 `gaze_event_span`
- RAW157 A003@KD3 `task_fixation_count`
- RAW157 D036@KD3 `incoming_amplitude_fixation_duration_coupling`
- RAW157 D023@READING `gaze_event_span`
- RAW157 A003@KD2 `task_fixation_count`
- RAW157 A003@KD1 `task_fixation_count`
- RAW157 A021@KD2 `task_saccade_count`
- RAW157 A027@KD3 `saccade_duration_iqr`
- RAW157 A046@READING `task_absolute_pupil_median`
- RAW157 D029@KD3 `rightward_saccade_amplitude_median`
- RAW157 A021@KD3 `task_saccade_count`
- RAW157 D023@KD1 `gaze_event_span`
- RAW157 A002@KD2 `total_fixation_time`
- RAW157 A030@KD1 `task_long_saccade_fraction`
- RAW157 D030@KD3 `leftward_saccade_amplitude_median`
- RAW157 D034@READING `saccade_amplitude_serial_correlation`
- RAW157 D025@KD1 `main_sequence_duration_slope`
- ROI64_T1235 D007@KD1 `target_mean_fixation_count`
- ROI64_T1235 D007@KD2 `target_mean_fixation_count`
- ROI64_T1235 D011@KD1 `target_left_pupil_range_median`
- ROI64_T1235 D010@KD3 `target_pupil_change_median`

## Các macro features được chọn

δ là Cliff's delta, dương khi nhóm thứ hai lớn hơn (ví dụ δ CN-AD > 0: AD cao hơn CN). CI là khoảng bootstrap
phần trăm 2000 lần theo người, mang tính mô tả, không hiệu chỉnh sau chọn lọc. Cột bootstrap là tần suất qua
tiêu chí chính và tỉ lệ cùng hướng trong 500 lần lặp lại toàn bộ tìm kiếm.

| Cohort | ID@scope | Tên | Nguồn gốc | p CN-AD / CN-MCI / MCI-AD | δ CN-AD [CI] | δ CN-MCI [CI] | δ MCI-AD | q joint (BH) | N CN/MCI/AD | Lựa chọn bootstrap / hướng ổn định | LOO-AD giữ | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RAW157 | D023@KD3 | gaze_event_span | dataset100 | 0.000136 / 0.000246 / 0.040 | 0.60 [0.29, 0.85] | 0.39 [0.21, 0.58] | 0.35 | 0.053 | 100/41/16 | 0.99 / 1.00 | 1.00 | joint_FDR_supported;stable_candidate |
| RAW157 | A002@KD3 | total_fixation_time | base160 | 9.5e-05 / 0.000868 / 0.077 | 0.61 [0.31, 0.85] | 0.36 [0.16, 0.56] | 0.30 | 0.093 | 100/41/16 | 0.97 / 1.00 | 1.00 | joint_FDR_supported;stable_candidate |
| RAW157 | A003@KD3 | task_fixation_count | base160 | 0.000181 / 0.002 / 0.033 | 0.58 [0.32, 0.80] | 0.34 [0.15, 0.51] | 0.37 | 0.100 | 100/41/16 | 0.98 / 1.00 | 1.00 | joint_FDR_supported;stable_candidate |
| RAW157 | D023@KD2 | gaze_event_span | dataset100 | 0.002 / 0.00082 / 0.419 | 0.49 [0.16, 0.77] | 0.36 [0.16, 0.55] | 0.14 | 0.100 | 100/41/16 | 0.95 / 0.99 | 1.00 | joint_FDR_supported;stable_candidate |
| RAW157 | D036@KD3 | incoming_amplitude_fixation_duration_coupling | dataset100 | 0.008 / 0.000121 / 0.729 | -0.42 [-0.62, -0.20] | -0.41 [-0.61, -0.21] | 0.06 | 0.332 | 100/41/16 | 0.97 / 1.00 | 1.00 | stable_candidate |
| RAW157 | D023@READING | gaze_event_span | dataset100 | 0.008 / 0.016 / 0.245 | 0.42 [0.10, 0.70] | 0.26 [0.05, 0.45] | 0.20 | 0.571 | 100/41/16 | 0.83 / 0.99 | 1.00 | stable_candidate |
| RAW157 | A003@KD2 | task_fixation_count | base160 | 0.027 / 4.47e-05 / 0.972 | 0.35 [0.01, 0.67] | 0.44 [0.23, 0.61] | 0.01 | 0.818 | 100/41/16 | 0.80 / 0.98 | 1.00 | stable_candidate |
| RAW157 | A046@READING | task_absolute_pupil_median | base160 | 0.036 / 0.028 / 0.467 | -0.33 [-0.62, 3.1e-05] | -0.24 [-0.44, -0.03] | -0.13 | 0.843 | 100/41/16 | 0.67 / 0.97 | 1.00 | stable_candidate |
| RAW157 | D029@KD3 | rightward_saccade_amplitude_median | dataset100 | 0.002 / 0.039 / 0.169 | -0.48 [-0.72, -0.22] | -0.22 [-0.44, -0.00] | -0.24 | 0.843 | 100/41/16 | 0.79 / 0.97 | 1.00 | stable_candidate |
| RAW157 | A027@KD3 | saccade_duration_iqr | base160 | 0.040 / 0.027 / 0.003 | -0.32 [-0.62, 0.02] | 0.24 [0.03, 0.44] | -0.52 | 0.843 | 100/41/16 | 0.63 / 0.95 | 1.00 | stable_candidate |
| RAW157 | A021@KD2 | task_saccade_count | base160 | 0.056 / 0.015 / 0.824 | 0.30 [0.00, 0.57] | 0.26 [0.05, 0.44] | 0.04 | 0.843 | 100/41/16 | 0.71 / 0.96 | 1.00 | stable_candidate |
| RAW157 | A021@KD3 | task_saccade_count | base160 | 0.004 / 0.058 / 0.155 | 0.45 [0.19, 0.67] | 0.20 [0.00, 0.39] | 0.25 | 0.843 | 100/41/16 | 0.74 / 0.98 | 1.00 | stable_candidate |
| RAW157 | D030@KD3 | leftward_saccade_amplitude_median | dataset100 | 0.019 / 0.059 / 0.347 | -0.37 [-0.60, -0.09] | -0.20 [-0.39, 0.01] | -0.16 | 0.843 | 99/41/16 | 0.66 / 0.97 | 1.00 | stable_candidate |
| RAW157 | D023@KD1 | gaze_event_span | dataset100 | 0.020 / 0.060 / 0.316 | 0.36 [0.08, 0.63] | 0.20 [-0.01, 0.40] | 0.17 | 0.843 | 100/41/16 | 0.65 / 0.97 | 1.00 | stable_candidate |
| RAW157 | A003@KD1 | task_fixation_count | base160 | 0.061 / 0.010 / 0.817 | 0.29 [0.02, 0.55] | 0.28 [0.09, 0.46] | 0.04 | 0.843 | 100/41/16 | 0.72 / 0.98 | 1.00 | stable_candidate |
| RAW157 | D040@KD2 | vertical_progression_monotonicity | dataset100 | 0.004 / 0.063 / 0.259 | -0.45 [-0.69, -0.17] | -0.20 [-0.42, 0.02] | -0.20 | 0.843 | 100/41/16 | 0.68 / 0.96 | 1.00 | exploratory_only |
| RAW157 | A002@KD2 | total_fixation_time | base160 | 0.069 / 0.002 / 0.866 | 0.28 [-0.05, 0.61] | 0.33 [0.13, 0.53] | -0.03 | 0.863 | 100/41/16 | 0.70 / 0.94 | 1.00 | stable_candidate |
| RAW157 | D031@KD2 | transition_direction_entropy | dataset100 | 0.077 / 0.057 / 0.760 | 0.28 [-0.06, 0.61] | 0.21 [-0.02, 0.42] | 0.05 | 0.918 | 95/39/16 | 0.56 / 0.91 | 1.00 | exploratory_only |
| RAW157 | A046@KD3 | task_absolute_pupil_median | base160 | 0.084 / 0.033 / 0.689 | -0.27 [-0.59, 0.06] | -0.23 [-0.44, -0.02] | -0.07 | 0.928 | 100/41/16 | 0.58 / 0.94 | 1.00 | exploratory_only |
| RAW157 | A046@KD2 | task_absolute_pupil_median | base160 | 0.101 / 0.055 / 0.600 | -0.26 [-0.58, 0.08] | -0.21 [-0.41, 0.01] | -0.09 | 0.928 | 100/41/16 | 0.51 / 0.93 | 1.00 | exploratory_only |
| RAW157 | A023@KD3 | amplitude_median | base160 | 0.046 / 0.101 / 0.360 | -0.31 [-0.57, -0.03] | -0.18 [-0.37, 0.03] | -0.16 | 0.928 | 100/41/16 | 0.58 / 0.95 | 1.00 | exploratory_only |
| RAW157 | A030@KD1 | task_long_saccade_fraction | base160 | 0.102 / 0.014 / 0.922 | -0.26 [-0.49, 0.01] | -0.26 [-0.44, -0.08] | 0.02 | 0.928 | 100/41/16 | 0.62 / 0.97 | 1.00 | stable_candidate |
| RAW157 | D034@READING | saccade_amplitude_serial_correlation | dataset100 | 0.102 / 0.003 / 0.613 | 0.26 [-0.03, 0.53] | 0.32 [0.11, 0.51] | -0.09 | 0.928 | 99/41/16 | 0.62 / 0.96 | 1.00 | stable_candidate |
| RAW157 | A030@READING | task_long_saccade_fraction | base160 | 0.099 / 0.108 / 0.887 | -0.26 [-0.48, -0.02] | -0.17 [-0.38, 0.04] | -0.03 | 0.928 | 100/41/16 | 0.46 / 0.94 | 1.00 | exploratory_only |
| RAW157 | A035@READING | task_spatial_entropy | base160 | 0.108 / 0.024 / 0.908 | -0.25 [-0.53, 0.03] | -0.24 [-0.43, -0.03] | 0.02 | 0.928 | 100/41/16 | 0.57 / 0.95 | 0.88 | exploratory_only |
| RAW157 | A023@READING | amplitude_median | base160 | 0.074 / 0.116 / 0.763 | -0.28 [-0.50, -0.06] | -0.17 [-0.37, 0.04] | -0.05 | 0.946 | 100/41/16 | 0.50 / 0.94 | 1.00 | exploratory_only |
| RAW157 | D025@KD1 | main_sequence_duration_slope | dataset100 | 0.119 / 0.000656 / 0.494 | 0.25 [-0.08, 0.55] | 0.37 [0.17, 0.55] | -0.12 | 0.946 | 100/41/15 | 0.62 / 0.94 | 0.81 | stable_candidate |
| RAW157 | A023@KD2 | amplitude_median | base160 | 0.124 / 0.022 / 0.986 | -0.24 [-0.52, 0.08] | -0.25 [-0.43, -0.04] | -0.00 | 0.946 | 100/41/16 | 0.54 / 0.94 | 0.69 | exploratory_only |
| RAW157 | D029@KD1 | rightward_saccade_amplitude_median | dataset100 | 0.128 / 0.075 / 0.894 | -0.24 [-0.49, 0.04] | -0.19 [-0.38, 0.02] | 0.02 | 0.946 | 100/41/16 | 0.47 / 0.90 | 0.94 | exploratory_only |
| RAW157 | A046@KD1 | task_absolute_pupil_median | base160 | 0.135 / 0.053 / 0.763 | -0.23 [-0.56, 0.10] | -0.21 [-0.41, -0.00] | -0.05 | 0.966 | 100/41/16 | 0.49 / 0.91 | 0.62 | exploratory_only |
| RAW157 | A029@KD1 | task_tiny_saccade_fraction | base160 | 0.145 / 0.003 / 0.683 | 0.23 [-0.06, 0.48] | 0.32 [0.13, 0.49] | -0.07 | 0.977 | 100/41/16 | 0.56 / 0.95 | 0.56 | exploratory_only |
| RAW157 | A028@READING | saccade_mean_speed_median | base160 | 0.147 / 0.072 / 0.922 | -0.23 [-0.47, 0.03] | -0.19 [-0.38, 0.01] | 0.02 | 0.977 | 100/41/16 | 0.45 / 0.93 | 0.69 | exploratory_only |
| RAW157 | D031@KD3 | transition_direction_entropy | dataset100 | 0.151 / 0.063 / 0.934 | 0.23 [-0.07, 0.49] | 0.21 [-0.03, 0.42] | -0.02 | 0.977 | 94/39/16 | 0.45 / 0.89 | 0.56 | exploratory_only |
| RAW157 | D022@KD3 | fixation_onset_burstiness | dataset100 | 0.158 / 0.005 / 0.676 | 0.22 [-0.07, 0.50] | 0.30 [0.12, 0.49] | -0.07 | 0.992 | 100/41/16 | 0.55 / 0.95 | 0.56 | exploratory_only |
| RAW157 | A027@READING | saccade_duration_iqr | base160 | 0.173 / 0.146 / 0.599 | 0.20 [-0.06, 0.46] | 0.15 [-0.03, 0.33] | 0.09 | 1.000 | 100/41/16 | 0.39 / 0.90 | 0.44 | exploratory_only |
| RAW157 | A021@KD1 | task_saccade_count | base160 | 0.182 / 0.107 / 0.852 | 0.21 [-0.06, 0.45] | 0.17 [-0.03, 0.36] | 0.03 | 1.000 | 100/41/16 | 0.41 / 0.92 | 0.44 | exploratory_only |
| RAW157 | D038@KD3 | immediate_return_fraction | dataset100 | 0.063 / 0.190 / 0.432 | 0.30 [0.06, 0.52] | 0.15 [-0.07, 0.36] | 0.14 | 1.000 | 89/36/15 | 0.44 / 0.91 | 1.00 | exploratory_only |
| RAW157 | D039@READING | fixation_xy_spread_log_anisotropy | dataset100 | 0.199 / 0.006 / 0.351 | 0.20 [-0.08, 0.48] | 0.29 [0.08, 0.49] | -0.16 | 1.000 | 100/41/16 | 0.50 / 0.92 | 0.44 | exploratory_only |
| RAW157 | D026@KD1 | amplitude_duration_coupling | dataset100 | 0.199 / 0.046 / 0.941 | 0.21 [-0.08, 0.49] | 0.21 [0.02, 0.41] | -0.01 | 1.000 | 100/41/15 | 0.43 / 0.90 | 0.44 | exploratory_only |
| RAW157 | A035@KD2 | task_spatial_entropy | base160 | 0.112 / 0.200 / 0.093 | 0.25 [-0.04, 0.53] | -0.14 [-0.37, 0.09] | 0.29 | 1.000 | 100/41/16 | 0.30 / 0.83 | 0.94 | exploratory_only |
| ROI64_T1235 | D007@KD1 | target_mean_fixation_count | dataset100 | 0.015 / 0.005 / 0.806 | 0.45 [0.11, 0.74] | 0.49 [0.20, 0.75] | 0.05 | 0.577 | 31/17/15 | 0.90 / 1.00 | 1.00 | stable_candidate |
| ROI64_T1235 | D010@KD3 | target_pupil_change_median | dataset100 | 0.021 / 0.037 / 1.000 | -0.45 [-0.77, -0.09] | -0.38 [-0.68, -0.05] | -0.01 | 0.606 | 28/15/13 | 0.71 / 0.97 | 1.00 | stable_candidate; export_convention |
| ROI64_T1235 | D007@KD2 | target_mean_fixation_count | dataset100 | 0.062 / 0.011 / 0.566 | 0.35 [-0.07, 0.71] | 0.46 [0.14, 0.74] | 0.12 | 0.606 | 30/16/15 | 0.71 / 0.96 | 1.00 | stable_candidate |
| ROI64_T1235 | D011@KD1 | target_left_pupil_range_median | dataset100 | 0.020 / 0.062 / 0.509 | 0.42 [0.07, 0.73] | 0.32 [0.00, 0.62] | 0.14 | 0.606 | 31/17/15 | 0.69 / 0.98 | 1.00 | stable_candidate; export_convention |
| ROI64_T1235 | D009@KD3 | target_articulation_time_median | dataset100 | 0.084 / 0.151 / 0.549 | 0.34 [-0.02, 0.67] | 0.27 [-0.07, 0.59] | 0.14 | 0.848 | 29/15/13 | 0.44 / 0.93 | 1.00 | exploratory_only |
| ROI64_T1235 | D012@KD2 | leftward_saccade_target_fraction | dataset100 | 0.185 / 0.094 / 0.890 | 0.25 [-0.15, 0.61] | 0.30 [-0.02, 0.62] | 0.03 | 0.848 | 30/16/15 | 0.40 / 0.89 | 0.47 | exploratory_only; export_convention |
| ROI64_T1235 | D013@KD1 | fixation_count_onset_delay_coupling | dataset100 | 0.189 / 0.123 / 1.000 | 0.24 [-0.14, 0.56] | 0.27 [-0.05, 0.56] | 0.00 | 0.848 | 32/17/15 | 0.40 / 0.86 | 0.53 | exploratory_only |

Ý nghĩa sinh lý và giới hạn của từng ID nằm trong `feature_registry.csv` và `DATASET_PROPOSED_100_FEATURES.md`.
Feature `export_convention` dựa trên quy ước export workbook (Bảng II của bài báo) và có ghi chú ngữ nghĩa còn lại.

## Top task–ROI và micro features được chọn (ROI64_T1235, KD + đọc)

Bằng chứng vùng: p_region = min(1, m_r · min p_joint) (Bonferroni trong vùng), BH trên tất cả vùng kiểm định được.
Văn bản ROI không có trong repository nên hiển thị `text_unavailable`; thứ tự ROI ID chưa được xác minh là thứ tự đọc.

| Hạng | Task | ROI (text) | Micro features được chọn (≤5 hiển thị) | Anchor: p CN-AD / CN-MCI / MCI-AD; δ | m_r, p_region, q_region | Coverage tối thiểu | Tần suất top-20 | LOO-AD giữ | Mức bằng chứng |
|---|---|---|---|---|---|---|---|---|---|
| 1 | KD1 | ROI 16 (text_unavailable) | D002, D005, M001, D001 | D002: 0.011 / 0.016 / 0.982; 0.49, 0.43, 0.01 | 8, 0.131, 1.000 | 0.87 | 0.58 | 1.00 | stable_candidate |
| 2 | READING | ROI 138 (text_unavailable) | D001, D003 | D001: 0.019 / 0.024 / 1.000; 0.55, 0.49, 0.00 | 8, 0.196, 1.000 | 0.60 | 0.41 | 0.40 | exploratory_only |
| 3 | KD1 | ROI 23 (text_unavailable) | M001, D005, D002, D003 | M001: 0.001 / 0.026 / 0.276; 0.58, 0.35, 0.23 | 8, 0.207, 1.000 | 0.80 | 0.45 | 1.00 | stable_candidate |
| 4 | KD1 | ROI 15 (text_unavailable) | M001, D003, D002 | M001: 0.006 / 0.028 / 0.610; 0.42, 0.32, 0.10 | 8, 0.221, 1.000 | 0.91 | 0.45 | 1.00 | stable_candidate |
| 5 | KD3 | ROI 16 (text_unavailable) | M051 | M051: 0.011 / 0.028 / 0.296; 0.50, 0.42, 0.24 | 8, 0.226, 1.000 | 0.82 | 0.34 | 1.00 | stable_candidate |
| 6 | KD2 | ROI 6 (text_unavailable) | D003 | D003: 0.028 / 0.004 / 0.984; 0.40, 0.50, -0.01 | 8, 0.227, 1.000 | 0.91 | 0.49 | 1.00 | stable_candidate |
| 7 | KD1 | ROI 5 (text_unavailable) | M051 | M051: 0.030 / 0.029 / 0.416; 0.40, 0.39, -0.17 | 8, 0.240, 1.000 | 0.97 | 0.25 | 1.00 | stable_candidate |
| 8 | READING | ROI 88 (text_unavailable) | M051, D005 | M051: 0.012 / 0.033 / 0.703; 0.53, 0.43, 0.10 | 8, 0.268, 1.000 | 0.72 | 0.29 | 1.00 | stable_candidate |
| 9 | KD1 | ROI 27 (text_unavailable) | M001, D002 | M001: 0.035 / 0.000595 / 0.430; 0.33, 0.56, -0.16 | 8, 0.278, 1.000 | 0.81 | 0.49 | 1.00 | stable_candidate |
| 10 | READING | ROI 5 (text_unavailable) | M001, D002, M061, D005 | M001: 0.036 / 0.034 / 0.654; -0.42, -0.40, -0.10 | 8, 0.287, 1.000 | 0.73 | 0.43 | 1.00 | exploratory_only |
| 11 | READING | ROI 131 (text_unavailable) | D003, M051 | D003: 0.043 / 0.028 / 0.796; 0.43, 0.45, -0.07 | 7, 0.298, 1.000 | 0.65 | 0.10 | 1.00 | exploratory_only |
| 12 | KD1 | ROI 28 (text_unavailable) | D004 | D004: 0.002 / 0.043 / 0.092; -0.63, -0.38, -0.39 | 8, 0.342, 1.000 | 0.80 | 0.40 | 1.00 | stable_candidate |
| 13 | READING | ROI 134 (text_unavailable) | D004, M001, D003 | D004: 0.057 / 0.027 / 0.860; 0.43, 0.48, 0.05 | 6, 0.345, 1.000 | 0.65 | 0.18 | 1.00 | exploratory_only |
| 14 | KD3 | ROI 2 (text_unavailable) | D002, M001 | D002: 0.041 / 0.045 / 0.917; 0.38, 0.37, 0.03 | 8, 0.357, 1.000 | 0.88 | 0.31 | 1.00 | stable_candidate |
| 15 | KD2 | ROI 38 (text_unavailable) | D005, D004, D003, M061, D002 (+1) | D005: 0.003 / 0.046 / 0.281; -0.61, -0.41, -0.25 | 8, 0.366, 1.000 | 0.69 | 0.41 | 1.00 | stable_candidate |
| 16 | KD2 | ROI 31 (text_unavailable) | M051 | M051: 0.032 / 0.046 / 0.950; 0.41, 0.36, 0.02 | 8, 0.367, 1.000 | 0.93 | 0.25 | 1.00 | stable_candidate |
| 17 | KD2 | ROI 29 (text_unavailable) | M001, M061 | M001: 0.003 / 0.047 / 0.284; 0.47, 0.28, 0.21 | 8, 0.374, 1.000 | 0.81 | 0.47 | 1.00 | stable_candidate |
| 18 | KD3 | ROI 29 (text_unavailable) | D001 | D001: 0.056 / 0.037 / 0.001; 0.41, -0.43, 0.75 | 8, 0.448, 1.000 | 0.66 | 0.11 | 1.00 | exploratory_only |
| 19 | READING | ROI 156 (text_unavailable) | D003, M061 | D003: 0.081 / 0.085 / 1.000; 0.35, 0.36, 0.00 | 6, 0.510, 1.000 | 0.65 | 0.12 | 1.00 | exploratory_only |
| 20 | READING | ROI 79 (text_unavailable) | M001 | M001: 0.081 / 0.089 / 0.035; 0.37, -0.36, 0.52 | 6, 0.534, 1.000 | 0.62 | 0.05 | 1.00 | exploratory_only |

Toàn bộ micro features qua tiêu chí chính trong các vùng này: `micro_selected_top20.csv`; ngoài top-20:
`micro_selected_main.csv`.

### Độ nhạy chỉ đọc (ROI64_T5, không phải tái lặp độc lập)

| Hạng | Task | ROI (text) | Micro features được chọn (≤5 hiển thị) | Anchor: p CN-AD / CN-MCI / MCI-AD; δ | m_r, p_region, q_region | Coverage tối thiểu | Tần suất top-20 | LOO-AD giữ | Mức bằng chứng |
|---|---|---|---|---|---|---|---|---|---|
| 1 | READING | ROI 134 (text_unavailable) | M001, D004, D005 | M001: 0.013 / 0.008 / 0.948; 0.42, 0.44, 0.02 | 7, 0.090, 1.000 | 0.71 | 0.54 | 1.00 | stable_candidate |
| 2 | READING | ROI 88 (text_unavailable) | D005, M051 | D005: 0.012 / 0.024 / 0.815; -0.53, -0.43, 0.06 | 7, 0.166, 1.000 | 0.73 | 0.73 | 1.00 | stable_candidate |
| 3 | READING | ROI 50 (text_unavailable) | D003, M001 | D003: 0.037 / 0.000929 / 0.706; 0.46, 0.63, -0.09 | 7, 0.258, 1.000 | 0.66 | 0.48 | 1.00 | exploratory_only |
| 4 | READING | ROI 138 (text_unavailable) | D001, D003 | D001: 0.045 / 0.046 / 0.819; 0.46, 0.43, -0.07 | 7, 0.324, 1.000 | 0.60 | 0.56 | 0.40 | exploratory_only |
| 5 | READING | ROI 5 (text_unavailable) | M001, D005, M061 | M001: 0.068 / 0.076 / 0.571; -0.36, -0.32, -0.12 | 7, 0.530, 1.000 | 0.73 | 0.61 | 1.00 | exploratory_only |
| 6 | READING | ROI 111 (text_unavailable) | D004, M061 | D004: 0.063 / 0.082 / 0.975; 0.43, 0.35, 0.02 | 7, 0.573, 1.000 | 0.60 | 0.36 | 0.40 | exploratory_only |
| 7 | READING | ROI 15 (text_unavailable) | D005, M051 | D005: 0.097 / 0.027 / 0.970; 0.38, 0.47, 0.02 | 6, 0.584, 1.000 | 0.60 | 0.23 | 0.40 | exploratory_only |
| 8 | READING | ROI 41 (text_unavailable) | D003 | D003: 0.103 / 0.020 / 0.001; -0.38, 0.49, -0.84 | 6, 0.620, 1.000 | 0.60 | 0.18 | 0.40 | exploratory_only |
| 9 | READING | ROI 57 (text_unavailable) | M061 | M061: 0.013 / 0.094 / 0.255; 0.56, 0.33, 0.28 | 7, 0.661, 1.000 | 0.67 | 0.44 | 1.00 | exploratory_only |
| 10 | READING | ROI 132 (text_unavailable) | M051 | M051: 0.121 / 0.083 / 0.788; 0.36, 0.36, 0.08 | 6, 0.727, 1.000 | 0.60 | 0.23 | 0.40 | exploratory_only |
| 11 | READING | ROI 78 (text_unavailable) | D003, D004 | D003: 0.136 / 0.069 / 0.790; -0.31, -0.37, 0.07 | 6, 0.817, 1.000 | 0.65 | 0.24 | 0.87 | exploratory_only |
| 12 | READING | ROI 140 (text_unavailable) | M051, D001 | M051: 0.037 / 0.118 / 0.491; 0.40, 0.30, 0.16 | 7, 0.829, 1.000 | 0.82 | 0.67 | 1.00 | exploratory_only |
| 13 | READING | ROI 2 (text_unavailable) | M001, D003 | M001: 0.088 / 0.127 / 0.813; 0.24, 0.21, 0.05 | 7, 0.888, 1.000 | 0.75 | 0.54 | 1.00 | exploratory_only |
| 14 | READING | ROI 7 (text_unavailable) | D003, D005 | D003: 0.045 / 0.127 / 0.811; -0.41, -0.27, -0.06 | 7, 0.892, 1.000 | 0.67 | 0.48 | 1.00 | exploratory_only |
| 15 | READING | ROI 156 (text_unavailable) | D003 | D003: 0.030 / 0.149 / 0.610; 0.43, 0.30, 0.13 | 6, 0.896, 1.000 | 0.65 | 0.29 | 1.00 | exploratory_only |
| 16 | READING | ROI 105 (text_unavailable) | D001, D003 | D001: 0.107 / 0.129 / 0.016; -0.34, 0.30, -0.59 | 7, 0.900, 1.000 | 0.73 | 0.52 | 1.00 | exploratory_only |
| 17 | READING | ROI 93 (text_unavailable) | M001 | M001: 0.132 / 0.028 / 0.301; -0.31, -0.42, 0.23 | 7, 0.923, 1.000 | 0.73 | 0.47 | 0.73 | exploratory_only |
| 18 | READING | ROI 91 (text_unavailable) | M061 | M061: 0.154 / 0.065 / 0.033; 0.33, -0.38, 0.56 | 6, 0.925, 1.000 | 0.60 | 0.25 | 0.40 | exploratory_only |
| 19 | READING | ROI 6 (text_unavailable) | D004, D001 | D004: 0.106 / 0.142 / 0.132; 0.35, 0.29, 0.36 | 7, 0.995, 1.000 | 0.73 | 0.48 | 0.93 | exploratory_only |
| 20 | READING | ROI 80 (text_unavailable) | D004 | D004: 0.064 / 0.147 / 0.350; -0.43, -0.30, -0.25 | 7, 1.000, 1.000 | 0.60 | 0.26 | 0.40 | exploratory_only |

## Tóm tắt bốn task

| Cohort | Task | ROI expected | testable | main-selected | p_task | q_task BH | regions passing CN-AD / CN-MCI / MCI-AD |
|---|---|---|---|---|---|---|---|
| ROI64_T1235 | KD1 | 40 | 40 | 18 | 1.000 | 1.000 | 36 / 34 / 32 |
| ROI64_T1235 | KD2 | 40 | 40 | 16 | 1.000 | 1.000 | 33 / 29 / 26 |
| ROI64_T1235 | KD3 | 40 | 38 | 15 | 1.000 | 1.000 | 33 / 29 / 24 |
| ROI64_T1235 | READING | 159 | 57 | 16 | 1.000 | 1.000 | 60 / 33 / 51 |
| ROI64_T5 | READING | 159 | 80 | 25 | 1.000 | 1.000 | 73 / 67 / 58 |

p_task = min(1, R_t · min p_region) là kiểm định bảo thủ về sự tồn tại bằng chứng trong task; không có nghĩa task
nào tốt hơn về lâm sàng. Đọc có nhiều ROI hơn KD nên số vùng được chọn không so sánh trực tiếp.

## Dự phòng theo cặp và MCI_AD

Vùng dự phòng (không qua tiêu chí chính nhưng có feature qua một cặp): 211 hàng trong
`roi_pair_specific_reserves.csv`. Số giả thuyết qua MCI_AD (p < 0.20) theo cohort: RAW157: 40, ROI64_T1235: 252, ROI64_T5: 109.
Mười kết quả MCI_AD mạnh nhất nằm ngoài danh sách chính:
- RAW157 D047@KD3 (dataset100): p_MCI_AD=0.000717, δ=0.58
- ROI64_T1235 D004@READING|lukeminen_1|ROI_068 (dataset100): p_MCI_AD=0.002, δ=0.79
- RAW157 A024@KD3 (base160): p_MCI_AD=0.003, δ=-0.52
- RAW157 D047@KD2 (dataset100): p_MCI_AD=0.003, δ=0.50
- ROI64_T1235 D004@READING|lukeminen_1|ROI_015 (dataset100): p_MCI_AD=0.004, δ=0.78
- ROI64_T5 D005@READING|lukeminen_1|ROI_056 (dataset100): p_MCI_AD=0.004, δ=0.67
- ROI64_T1235 D004@KD1|KD_1_highres|ROI_034 (dataset100): p_MCI_AD=0.006, δ=0.64
- ROI64_T1235 D003@READING|lukeminen_1|ROI_041 (dataset100): p_MCI_AD=0.007, δ=-0.71
- ROI64_T5 D004@READING|lukeminen_1|ROI_138 (dataset100): p_MCI_AD=0.013, δ=0.59
- ROI64_T1235 D004@KD2|KD_2_highres|ROI_027 (dataset100): p_MCI_AD=0.014, δ=-0.54

## Ổn định, ảnh hưởng và kiểm định hoán vị

- Bootstrap khám phá 500 lần (lặp lại toàn bộ họ đủ điều kiện, không chỉ người thắng): `bootstrap_stability.csv`.
- Bỏ từng người AD: `ad_influence.csv`. Kiểm định hoán vị 9999 lần cho macro được chọn và micro được chọn trong
  top-20: 381 phép; 8 trường hợp p chính và p hoán vị nằm khác phía của 0.20 (`permutation_validation.csv`).
- Độ nhạy coverage (50%/75%, họ riêng, không thay thế phân tích chính), số được chọn: RAW157/macro cov 0.5: 40; ROI64_T1235/macro cov 0.5: 8; ROI64_T5/macro cov 0.5: 0; ROI64_T1235/micro cov 0.5: 129; ROI64_T5/micro cov 0.5: 50; RAW157/macro cov 0.75: 40; ROI64_T1235/macro cov 0.75: 6; ROI64_T5/macro cov 0.75: 0; ROI64_T1235/micro cov 0.75: 64; ROI64_T5/micro cov 0.75: 6; RAW157/macro cov 0.6: 40; ROI64_T1235/macro cov 0.6: 7; ROI64_T5/macro cov 0.6: 0; ROI64_T1235/micro cov 0.6: 107; ROI64_T5/micro cov 0.6: 41.

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
