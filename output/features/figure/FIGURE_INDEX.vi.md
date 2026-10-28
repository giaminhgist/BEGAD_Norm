# Mục lục hình: 33 feature RAW157 được chọn (CN vs MCI vs AD, 4 task)

Sinh bởi `plot_selected_features.py` từ `processed_dataset/raw157_feature_values_long_full.csv` (trích xuất lại 2026-10-28, commit pipeline `ffa36c7`, freeze `freeze_35a60cb67109`). Cohort RAW157: 157 người tham gia baseline (CN 100 / MCI 41 / AD 16). Mỗi feature một hình, bốn panel (KD1, KD2, KD3, READING).

Chín feature (D017, D022, D025, D026, D030, D034, D035, D036, D037) có quy tắc hỗ trợ trong-subject theo task, nên một số người tham gia thiếu giá trị hợp lệ ở một số task; các panel chỉ hiển thị người tham gia hợp lệ, với n hợp lệ theo task được in bên dưới (CN/MCI/AD).

| ID | Feature | Đơn vị | n hợp lệ mỗi task (CN/MCI/AD) | Trạng thái | Hình |
|---|---|---|---|---|---|
| A002 | total_fixation_time | s | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A002_total_fixation_time.png) / [PDF](pdf/A002_total_fixation_time.pdf) |
| A003 | task_fixation_count | count | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A003_task_fixation_count.png) / [PDF](pdf/A003_task_fixation_count.pdf) |
| A021 | task_saccade_count | count | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A021_task_saccade_count.png) / [PDF](pdf/A021_task_saccade_count.pdf) |
| A023 | amplitude_median | deg | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A023_amplitude_median.png) / [PDF](pdf/A023_amplitude_median.pdf) |
| A024 | amplitude_iqr | deg | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A024_amplitude_iqr.png) / [PDF](pdf/A024_amplitude_iqr.pdf) |
| A026 | saccade_duration_median | s | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A026_saccade_duration_median.png) / [PDF](pdf/A026_saccade_duration_median.pdf) |
| A027 | saccade_duration_iqr | s | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A027_saccade_duration_iqr.png) / [PDF](pdf/A027_saccade_duration_iqr.pdf) |
| A028 | saccade_mean_speed_median | deg/s | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A028_saccade_mean_speed_median.png) / [PDF](pdf/A028_saccade_mean_speed_median.pdf) |
| A029 | task_tiny_saccade_fraction | fraction | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A029_task_tiny_saccade_fraction.png) / [PDF](pdf/A029_task_tiny_saccade_fraction.pdf) |
| A030 | task_long_saccade_fraction | fraction | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A030_task_long_saccade_fraction.png) / [PDF](pdf/A030_task_long_saccade_fraction.pdf) |
| A031 | angular_scanpath_length | deg | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A031_angular_scanpath_length.png) / [PDF](pdf/A031_angular_scanpath_length.pdf) |
| A033 | normalized_screen_dispersion | normalized distance | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A033_normalized_screen_dispersion.png) / [PDF](pdf/A033_normalized_screen_dispersion.pdf) |
| A035 | task_spatial_entropy | unitless | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A035_task_spatial_entropy.png) / [PDF](pdf/A035_task_spatial_entropy.pdf) |
| A046 | task_absolute_pupil_median | pupil unit | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](A046_task_absolute_pupil_median.png) / [PDF](pdf/A046_task_absolute_pupil_median.pdf) |
| D017 | fixation_duration_serial_correlation | correlation | KD1 99/41/15; KD2 99/41/16; KD3 100/41/16; READING 99/41/16 | per-task missingness (kept values shown; see index note) | [PNG](D017_fixation_duration_serial_correlation.png) / [PDF](pdf/D017_fixation_duration_serial_correlation.pdf) |
| D018 | fixation_duration_quartile_dispersion | unitless | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D018_fixation_duration_quartile_dispersion.png) / [PDF](pdf/D018_fixation_duration_quartile_dispersion.pdf) |
| D021 | interfixation_interval_median | s | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D021_interfixation_interval_median.png) / [PDF](pdf/D021_interfixation_interval_median.pdf) |
| D022 | fixation_onset_burstiness | unitless | KD1 99/41/15; KD2 99/41/16; KD3 100/41/16; READING 99/41/16 | per-task missingness (kept values shown; see index note) | [PNG](D022_fixation_onset_burstiness.png) / [PDF](pdf/D022_fixation_onset_burstiness.pdf) |
| D023 | gaze_event_span | s | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D023_gaze_event_span.png) / [PDF](pdf/D023_gaze_event_span.pdf) |
| D024 | saccadic_time_fraction | fraction | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D024_saccadic_time_fraction.png) / [PDF](pdf/D024_saccadic_time_fraction.pdf) |
| D025 | main_sequence_duration_slope | s/deg | KD1 100/41/15; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | per-task missingness (kept values shown; see index note) | [PNG](D025_main_sequence_duration_slope.png) / [PDF](pdf/D025_main_sequence_duration_slope.pdf) |
| D026 | amplitude_duration_coupling | correlation | KD1 100/41/15; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | per-task missingness (kept values shown; see index note) | [PNG](D026_amplitude_duration_coupling.png) / [PDF](pdf/D026_amplitude_duration_coupling.pdf) |
| D028 | leftward_transition_fraction | fraction | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D028_leftward_transition_fraction.png) / [PDF](pdf/D028_leftward_transition_fraction.pdf) |
| D029 | rightward_saccade_amplitude_median | deg | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D029_rightward_saccade_amplitude_median.png) / [PDF](pdf/D029_rightward_saccade_amplitude_median.pdf) |
| D030 | leftward_saccade_amplitude_median | deg | KD1 98/41/15; KD2 99/40/16; KD3 99/41/16; READING 100/41/16 | per-task missingness (kept values shown; see index note) | [PNG](D030_leftward_saccade_amplitude_median.png) / [PDF](pdf/D030_leftward_saccade_amplitude_median.pdf) |
| D034 | saccade_amplitude_serial_correlation | correlation | KD1 98/40/15; KD2 99/40/15; KD3 97/39/16; READING 99/41/16 | per-task missingness (kept values shown; see index note) | [PNG](D034_saccade_amplitude_serial_correlation.png) / [PDF](pdf/D034_saccade_amplitude_serial_correlation.pdf) |
| D035 | fixation_duration_outgoing_amplitude_coupling | correlation | KD1 99/41/15; KD2 99/41/16; KD3 100/41/16; READING 99/41/16 | per-task missingness (kept values shown; see index note) | [PNG](D035_fixation_duration_outgoing_amplitude_coupling.png) / [PDF](pdf/D035_fixation_duration_outgoing_amplitude_coupling.pdf) |
| D036 | incoming_amplitude_fixation_duration_coupling | correlation | KD1 99/41/15; KD2 99/41/16; KD3 100/41/16; READING 99/41/16 | per-task missingness (kept values shown; see index note) | [PNG](D036_incoming_amplitude_fixation_duration_coupling.png) / [PDF](pdf/D036_incoming_amplitude_fixation_duration_coupling.pdf) |
| D037 | post_leftward_fixation_duration_log_ratio | log ratio | KD1 99/38/14; KD2 99/40/16; KD3 96/40/16; READING 99/41/16 | per-task missingness (kept values shown; see index note) | [PNG](D037_post_leftward_fixation_duration_log_ratio.png) / [PDF](pdf/D037_post_leftward_fixation_duration_log_ratio.pdf) |
| D039 | fixation_xy_spread_log_anisotropy | log ratio | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D039_fixation_xy_spread_log_anisotropy.png) / [PDF](pdf/D039_fixation_xy_spread_log_anisotropy.pdf) |
| D040 | vertical_progression_monotonicity | correlation | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D040_vertical_progression_monotonicity.png) / [PDF](pdf/D040_vertical_progression_monotonicity.pdf) |
| D041 | off_media_fixation_time_fraction | fraction | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D041_off_media_fixation_time_fraction.png) / [PDF](pdf/D041_off_media_fixation_time_fraction.pdf) |
| D047 | viewing_distance_iqr | mm | KD1 100/41/16; KD2 100/41/16; KD3 100/41/16; READING 100/41/16 | complete 157/157 in all tasks | [PNG](D047_viewing_distance_iqr.png) / [PDF](pdf/D047_viewing_distance_iqr.pdf) |

## Chú thích

Chú thích mỗi hình: Phân bố của {feature} ({unit}) theo nhóm chẩn đoán qua bốn task (KD1, KD2, KD3, READING). Hộp: trung vị, IQR, râu ở 1.5x IQR; chấm: từng người tham gia (CN xanh dương, MCI cam, AD xanh lá). Giá trị p ghi chú: Mann-Whitney U hai phía, tiệm cận có hiệu chỉnh tie và liên tục, tính trên các giá trị hợp lệ hiển thị trong panel; * p < 0.05. Mọi p-value và cỡ nhóm hợp lệ đã được đối chiếu với bảng đã công bố trong FEATURES_37_SUMMARY.md. Lưu ý n hợp lệ áp dụng cho chín feature kể trên.

## Đối chiếu với bảng đã công bố

Mọi p-value ghi chú và n theo nhóm đều được tính lại trên chính giá trị vẽ và so với FEATURES_37_SUMMARY.md (396 ô p, 132 ô n; xem `pvalue_crosscheck.csv`). n hợp lệ: khớp cả 132 ô. P-value: 379/396 ô khớp trong phạm vi làm tròn đã in; 17 ô lệch nhẹ (|chênh| tối đa = 0.014) dù giá trị, tập hợp lệ và đặc tả Mann-Whitney giống hệt. Nguyên nhân: run đã công bố kiểm định trên float thô ở độ chính xác máy với cấu trúc tie khác với giá trị CSV được vẽ ở đây (các feature dày tie như A027/A046/D025/D030/A024, nơi các giá trị gần bằng nhau ở mức 1e-12 làm thay đổi hiệu chỉnh tie). Hình ghi chú p-value của đúng dữ liệu được hiển thị.

## Tái lập

Hình được vẽ bằng matplotlib 3.11.2, pandas 2.3.3, scipy 1.15.3 (.venv-figs). Chạy lại: `.venv-figs/bin/python output/features/figure/plot_selected_features.py`. Đối chiếu p-value / n hợp lệ: `pvalue_crosscheck.csv`.
