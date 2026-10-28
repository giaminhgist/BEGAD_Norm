# Dataset đã xử lý: giá trị task-level của bộ 37 features (RAW157)

Giá trị feature cho bộ 37 features (xem `../feature_engineer/FEATURES_37_SUMMARY.md`), giới hạn ở cohort
**RAW157** (export sự kiện thô của Tobii). Bốn features ROI64_T1235 (D007, D010, D011, D015) bị loại theo
thiết kế: chỉ dùng các features của RAW157. Trong 33 features RAW157, 24 features có giá trị hợp lệ cho
**cả 157 subjects ở cả 4 task** và được giữ lại; 9 features bị loại vì quy tắc hỗ trợ trong-subject khiến
1-6 subjects không có giá trị ở ít nhất một task.

- Subjects: **157** người tham gia baseline (CN 100 / MCI 41 / AD 16).
- Tasks: KD1, KD2, KD3, READING (một giá trị cho mỗi subject và mỗi task).
- Trích xuất lại ngày 2026-10-28 bằng chính code pipeline gốc (git commit `ffa36c7`, freeze
  `freeze_35a60cb67109`, config `configs/feature_discovery.yaml`) và đúng phiên bản dependencies gốc
  (Python 3.10, pandas 2.3.3, numpy 2.2.6, scipy 1.15.3). Kiểm tra chéo với bảng đã công bố:
  6/6 p-value tái lập chính xác (ví dụ A002@KD3 CN-AD p = 9.5e-05; D023@KD3 CN-AD p = 0.000136).

## Các file

| File | Cấu trúc |
|---|---|
| `processed_dataset_v1_wide.csv` | **V1 wide.** 1 hàng mỗi subject (157 hàng). Cột: `subject_id`, `group`, sau đó 24 cột feature cho mỗi task theo thứ tự KD1, KD2, KD3, READING, đặt tên `{feature_id}__{task}` (96 cột feature, tổng 98 cột). Không có giá trị khuyết. |
| `processed_dataset_v2_long.csv` | **V2 long.** 4 hàng mỗi subject, mỗi hàng một task (628 hàng). Cột: `subject_id`, `group`, `task`, sau đó 24 cột feature (`A002`...`D047`). Không có giá trị khuyết. |
| `feature_columns.csv` | Danh sách features được giữ với `feature_id`, `feature_name`, `unit`. |
| `raw157_feature_values_long_full.csv` | Bảng dài đầy đủ của RAW157: cả 33 features x 4 task với `value`, `measurement_status` và `n_events` (33.284 hàng). Các features bị loại có mặt ở đây kèm trạng thái theo task (`observed` / `observed_zero` / `insufficient_within_subject_support`), nên bản theo task có thể dựng lại mà không cần chạy lại extraction. |

Cả hai version đều có cột `subject_id` và `group` (nhãn: CN / MCI / AD) để phục vụ cross-validation sau này;
dùng `subject_id` làm nhóm chia để không subject nào xuất hiện ở nhiều hơn một fold.

## Features được giữ (24, đủ 157/157 ở mọi task)

A002 `total_fixation_time`, A003 `task_fixation_count`, A021 `task_saccade_count`, A023 `amplitude_median`,
A024 `amplitude_iqr`, A026 `saccade_duration_median`, A027 `saccade_duration_iqr`, A028
`saccade_mean_speed_median`, A029 `task_tiny_saccade_fraction`, A030 `task_long_saccade_fraction`, A031
`angular_scanpath_length`, A033 `normalized_screen_dispersion`, A035 `task_spatial_entropy`, A046
`task_absolute_pupil_median`, D018 `fixation_duration_quartile_dispersion`, D021 `interfixation_interval_median`,
D023 `gaze_event_span`, D024 `saccadic_time_fraction`, D028 `leftward_transition_fraction`, D029
`rightward_saccade_amplitude_median`, D039 `fixation_xy_spread_log_anisotropy`, D040
`vertical_progression_monotonicity`, D041 `off_media_fixation_time_fraction`, D047 `viewing_distance_iqr`.

Định nghĩa, công thức và đơn vị: `../feature_engineer/FEATURES_37_SUMMARY.md` và
`../feature_engineer/feature_registry.csv`.

## Features bị loại (9) và số subjects hợp lệ theo task

Một feature chỉ được giữ nếu có giá trị hợp lệ cho cả 157 subjects ở cả 4 task; các feature sau không đạt
(số subjects hợp lệ theo task, KD1/KD2/KD3/READING):

| Feature | KD1 | KD2 | KD3 | READING | Lý do |
|---|---|---|---|---|---|
| D017 fixation_duration_serial_correlation | 155 | 156 | 157 | 156 | cần >= 30 gap-free transitions |
| D022 fixation_onset_burstiness | 155 | 156 | 157 | 156 | cần >= 30 inter-onset intervals |
| D025 main_sequence_duration_slope | 156 | 157 | 157 | 157 | cần >= 10 matched saccades |
| D026 amplitude_duration_coupling | 156 | 157 | 157 | 157 | cần >= 10 matched saccades |
| D030 leftward_saccade_amplitude_median | 154 | 155 | 156 | 157 | cần >= 1 leftward saccade |
| D034 saccade_amplitude_serial_correlation | 153 | 154 | 152 | 156 | cần >= 30 transitions liên tiếp |
| D035 fixation_duration_outgoing_amplitude_coupling | 155 | 156 | 157 | 156 | cần >= 30 transitions |
| D036 incoming_amplitude_fixation_duration_coupling | 155 | 156 | 157 | 156 | cần >= 30 transitions |
| D037 post_leftward_fixation_duration_log_ratio | 151 | 155 | 152 | 156 | cần >= 5 saccades trái và phải |

Nếu cần bản theo task của feature bị loại nào đó (ví dụ D025 cho KD2/KD3/READING, nơi có đủ 157 subjects),
có thể khôi phục từ bảng trích xuất dài bằng cách áp dụng quy tắc đủ theo từng task thay vì quy tắc tất cả
các task.

## Biến thể 33 features (157 subjects, impute ô khuyết tại thời điểm huấn luyện)

Có thể dùng 9 features bị loại nếu impute các ô không hỗ trợ được thay vì yêu cầu complete-case.
`processed_dataset_33feat_wide.csv` giữ **đủ 157 subjects**; 51 ô mà subject không hỗ trợ được
(insufficient within-subject support, đều nằm trong 9 features mới thêm) được để NaN. 20 subjects có ít
nhất một ô như vậy (CN 10 / MCI 6 / AD 4: BEG-202, BEG-220, BEG-231, BEG-234, BEG-247, BEG-260, BEG-275,
BEG-277, BEG-282, BEG-283, BEGAD-002, BEGAD-005, BEGAD-007, BEGAD-019, BEGCO-004, BEGCO-020, BEGCO-029,
BEGCO-032, BEGCO-035, BEGMCI-006).

Imputation thực hiện lúc huấn luyện, không ghi sẵn vào bảng: pipeline baseline điền mỗi ô khuyết bằng
**median theo cột, fit trên riêng split training của từng fold** (`SimpleImputer`, `strategy="median"`
trong `src/baseline/run_baselines.py`), nên không giá trị validation hay held-out nào lọt vào training.
Với 51/20724 ô khuyết (0.25%), lựa chọn imputer dự kiến ít ảnh hưởng; các phương án khác (điền 0 theo
đặc thù feature cho D030, cột chỉ báo missingness) là khả thi và cần thay đổi thiết kế nếu dùng.

| File | Cấu trúc |
|---|---|
| `processed_dataset_33feat_wide.csv` | 1 hàng mỗi subject (157 hàng). Cột: `subject_id`, `group`, sau đó 33 cột feature cho mỗi task theo thứ tự KD1, KD2, KD3, READING, đặt tên `{feature_id}__{task}` (132 cột feature, tổng 134 cột). 51 ô NaN, đều trong 9 features mới thêm. |
| `processed_dataset_33feat_long.csv` | 4 hàng mỗi subject, mỗi hàng một task (628 hàng). Cột: `subject_id`, `group`, `task`, sau đó 33 cột feature (`A002`...`D047`). 51 ô NaN. |
| `feature_columns_33feat.csv` | Danh sách 33 features với `feature_id`, `feature_name`, `unit`. |
| `make_33feat_dataset.py` | Script dựng lại. Đọc `raw157_feature_values_long_full.csv`, giữ mọi subject, ghi NaN cho các ô `insufficient_within_subject_support` (assert 51 ô, đều trong 9 features mới thêm), và kiểm tra chéo 24 cột feature dùng chung với `processed_dataset_v1_wide.csv` (sai số 1e-9; hai file serialize float với độ chính xác khác nhau, chênh lệch <= 7e-15). |

Vì không subject nào bị bỏ, baseline 33 features dùng lại **đúng splits của baseline 24 features**:
`splits/cv_fold_assignments.csv`, `splits/heldout_ad_subjects.csv` (4 held-out AD: BEGAD-001, BEGAD-005,
BEGAD-006, BEGAD-016). Config: `src/baseline/config_33feat.yaml` (`output/baseline_33feat/`).

## Nguồn gốc

- Đầu vào trích xuất: `dataset/20250305/*.csv` (export sự kiện thô của Tobii) và
  `dataset/participants_20250305.csv` (roster với mã nhóm 0/1/2 -> CN/MCI/AD).
- Code trích xuất: `src/begad_norm/feature_discovery/extract_macro.py` (git commit `ffa36c7`, khôi phục được từ
  lịch sử git); các giá trị chỉ gồm measurement `observed`/`observed_zero`.
- Thư mục run trích xuất lại: `/tmp/begad_run_260` (scratch; `macro_features_long.csv` chứa bảng dài).
- Hai CSV được xây dựng theo kiểu complete-case: không có NaN ở bất kỳ cột feature nào.
