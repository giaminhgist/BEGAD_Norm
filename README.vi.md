# BEGAD_Norm

Trích xuất đặc trưng (feature) chuyển động mắt và so sánh nhóm (CO / MCI / AD)
cho bộ test BEGAD.

## 1. Dữ liệu

- **Phiên ghi**: 2025-03-05, mã xuất `BEGAD_ETRA25`.
- **File raw**: 205 file CSV trong `dataset/20250305/` và
  `dataset_v2/20250305/` (schema xuất gồm tọa độ gaze ADCSpx/MCSpx, chỉ số sự
  kiện fixation/saccade, kích thước đồng tử, và các cột `Task`, `MediaName`,
  `Group`, `IsFollowup` theo từng dòng).
- **Lọc nhóm** cho dữ liệu đã xử lý: chỉ giữ người tham gia lần khám đầu thuộc
  nhóm CO / MCI / AD. Bị loại: 36 lần tái khám (follow-up), 10 người NPH
  (Group 3), 2 bản ghi Group −11.
- **Người tham gia**: 157 — CO 100, MCI 41, AD 16
  (`processed_dataset/metadata.csv`).
- **Tasks**: 11 task mỗi người (0–10) → `processed_dataset/features.csv` có
  1727 dòng (157 × 11; 2 dòng all-NaN: `BEGCO-029` và `BEGMCI-019`, task 10).

### Các task trong protocol

Tên media được lấy từ cột `MediaName` của file raw; kích thước hiển thị từ
`TASK_GEOMETRY` trong `src/begad_norm/config.py`.

| Task | Media (raw `MediaName` column) | Geometry on 1920×1080 | Description |
|---|---|---|---|
| 0 | `Instruction Element` | 1920×1080 (full screen) | Instruction screen |
| 1 | `KD_demo_highres.png` | 748×564 | King–Devick demo card (practice) |
| 2 | `Instruction Element 2` | 1920×1080 (full screen) | Instruction screen |
| 3 | `KD_1_highres.png` | 746×566 | King–Devick card 1 |
| 4 | `KD_2_highres.png` | 744×525 | King–Devick card 2 |
| 5 | `KD_3_highres.png` | 737×508 | King–Devick card 3 |
| 6 | `Instruction Element` | 1920×1080 (full screen) | Instruction screen |
| 7 | `lukeminen_harj.png` | 950×417 | Reading practice |
| 8 | `Instruction Element` | 1920×1080 (full screen) | Instruction screen |
| 9 | `lukeminen_1.png` | 918×917 | Reading task |
| 10 | `Instruction Element 2` | 1920×1080 (full screen) | Closing instruction screen |

Cách giải thích King–Devick / bài đọc dựa trên notebook protocol
(`notebooks/Analysisbase_for_Ali.ipynb` gọi tasks 3–5 là "KD tasks" và task 9
là "reading task") và tên file stimulus ("lukeminen" = đọc trong tiếng Phần
Lan). Tasks 0, 2, 6, 8, 10 là màn hình hướng dẫn toàn màn hình, không phải
stimulus kiểm tra.

## 2. Features (72)

72 feature được tính cho mỗi (người tham gia, task) từ các fixation sạch (xem
`docs/baseline/eda_summary_vi.md` để biết quy trình làm sạch, công thức và đơn vị; bản
tiếng Anh: `docs/baseline/eda_summary.md`). Phân nhóm theo tiền tố:

| Tiền tố | n | Ý nghĩa |
|---|---|---|
| `spa_` | 17 | phân bố không gian của fixation |
| `geo_` | 10 | hình học đường quét (scanpath) |
| `tem_` | 11 | động học theo thời gian |
| `pup_` | 7 | kích thước đồng tử |
| `pupbl_` | 7 | đường nền đồng tử |
| `pupdiff_` | 7 | chênh lệch đồng tử |
| `sac_` | 13 | saccade đo được |

Danh sách đầy đủ:

| Nhóm | Feature |
|---|---|
| spa — phân bố không gian của fixation | `spa_fix_count` |
| spa — phân bố không gian của fixation | `spa_mean_x` |
| spa — phân bố không gian của fixation | `spa_mean_y` |
| spa — phân bố không gian của fixation | `spa_std_x` |
| spa — phân bố không gian của fixation | `spa_std_y` |
| spa — phân bố không gian của fixation | `spa_dispersion` |
| spa — phân bố không gian của fixation | `spa_bbox_area` |
| spa — phân bố không gian của fixation | `spa_center_dist_mean` |
| spa — phân bố không gian của fixation | `spa_center_dist_std` |
| spa — phân bố không gian của fixation | `spa_center_frac` |
| spa — phân bố không gian của fixation | `spa_q1` |
| spa — phân bố không gian của fixation | `spa_q2` |
| spa — phân bố không gian của fixation | `spa_q3` |
| spa — phân bố không gian của fixation | `spa_q4` |
| spa — phân bố không gian của fixation | `spa_entropy` |
| spa — phân bố không gian của fixation | `spa_max_grid_frac` |
| spa — phân bố không gian của fixation | `spa_skew_x` |
| geo — hình học đường quét (scanpath) | `geo_scanpath_len` |
| geo — hình học đường quét (scanpath) | `geo_sacc_amp_mean` |
| geo — hình học đường quét (scanpath) | `geo_sacc_amp_std` |
| geo — hình học đường quét (scanpath) | `geo_sacc_amp_max` |
| geo — hình học đường quét (scanpath) | `geo_dx_mean` |
| geo — hình học đường quét (scanpath) | `geo_dy_mean` |
| geo — hình học đường quét (scanpath) | `geo_angle_var` |
| geo — hình học đường quét (scanpath) | `geo_revisit_rate` |
| geo — hình học đường quét (scanpath) | `geo_nn_dist_mean` |
| geo — hình học đường quét (scanpath) | `geo_hull_area` |
| tem — động học theo thời gian | `tem_dur_mean` |
| tem — động học theo thời gian | `tem_dur_std` |
| tem — động học theo thời gian | `tem_dur_total` |
| tem — động học theo thời gian | `tem_dur_max` |
| tem — động học theo thời gian | `tem_first_dur` |
| tem — động học theo thời gian | `tem_last_dur` |
| tem — động học theo thời gian | `tem_ifi_mean` |
| tem — động học theo thời gian | `tem_ifi_std` |
| tem — động học theo thời gian | `tem_velocity_mean` |
| tem — động học theo thời gian | `tem_fix_rate` |
| tem — động học theo thời gian | `tem_trans_entropy` |
| pup — kích thước đồng tử | `pup_mean` |
| pup — kích thước đồng tử | `pup_std` |
| pup — kích thước đồng tử | `pup_min` |
| pup — kích thước đồng tử | `pup_max` |
| pup — kích thước đồng tử | `pup_median` |
| pup — kích thước đồng tử | `pup_slope` |
| pup — kích thước đồng tử | `pup_first_last_diff` |
| pupbl — đường nền đồng tử | `pupbl_mean` |
| pupbl — đường nền đồng tử | `pupbl_std` |
| pupbl — đường nền đồng tử | `pupbl_min` |
| pupbl — đường nền đồng tử | `pupbl_max` |
| pupbl — đường nền đồng tử | `pupbl_median` |
| pupbl — đường nền đồng tử | `pupbl_slope` |
| pupbl — đường nền đồng tử | `pupbl_first_last_diff` |
| pupdiff — chênh lệch đồng tử | `pupdiff_mean` |
| pupdiff — chênh lệch đồng tử | `pupdiff_std` |
| pupdiff — chênh lệch đồng tử | `pupdiff_min` |
| pupdiff — chênh lệch đồng tử | `pupdiff_max` |
| pupdiff — chênh lệch đồng tử | `pupdiff_median` |
| pupdiff — chênh lệch đồng tử | `pupdiff_slope` |
| pupdiff — chênh lệch đồng tử | `pupdiff_first_last_diff` |
| sac — saccade đo được | `sac_count` |
| sac — saccade đo được | `sac_rate` |
| sac — saccade đo được | `sac_dur_mean` |
| sac — saccade đo được | `sac_dur_std` |
| sac — saccade đo được | `sac_dur_total` |
| sac — saccade đo được | `sac_dur_max` |
| sac — saccade đo được | `sac_amp_mean` |
| sac — saccade đo được | `sac_amp_std` |
| sac — saccade đo được | `sac_amp_max` |
| sac — saccade đo được | `sac_absdir_mean` |
| sac — saccade đo được | `sac_absdir_var` |
| sac — saccade đo được | `sac_turn_mean` |
| sac — saccade đo được | `sac_turn_std` |

## 3. Các feature được chọn (25)

Sàng lọc từ `processed_dataset/group_comparison_stats_per_task.csv`: các
feature có p-value Kruskal–Wallis < 0.1 (chưa hiệu chỉnh) ở **ít nhất 3 trong
11 tasks**. Chỉ mang
tính thăm dò — tổng cộng 792 kiểm định đã chạy, vì vậy đa kiểm định là điều
được dự kiến.

| Feature | n_tasks (p<0.1) | Tasks |
|---|---|---|
| Feature | n_tasks (p<0.1) | Tasks |
|---|---|---|
| `spa_fix_count` | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `spa_dispersion` | 3 | 3, 8, 9 |
| `spa_center_dist_std` | 5 | 1, 2, 4, 6, 9 |
| `spa_q2` | 3 | 0, 3, 10 |
| `geo_scanpath_len` | 4 | 0, 2, 4, 5 |
| `geo_sacc_amp_mean` | 5 | 3, 4, 5, 7, 8 |
| `geo_revisit_rate` | 6 | 0, 1, 2, 3, 4, 5 |
| `geo_nn_dist_mean` | 6 | 0, 2, 3, 4, 5, 9 |
| `tem_dur_total` | 4 | 0, 2, 4, 5 |
| `tem_velocity_mean` | 3 | 3, 5, 8 |
| `pup_mean` | 9 | 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_std` | 4 | 1, 2, 9, 10 |
| `pup_min` | 7 | 2, 5, 6, 7, 8, 9, 10 |
| `pup_max` | 7 | 0, 2, 5, 7, 8, 9, 10 |
| `pup_median` | 9 | 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_slope` | 3 | 1, 2, 10 |
| `pupbl_std` | 4 | 1, 2, 9, 10 |
| `pupbl_slope` | 3 | 1, 2, 10 |
| `sac_count` | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `sac_dur_mean` | 4 | 1, 3, 4, 9 |
| `sac_dur_total` | 9 | 0, 1, 2, 3, 4, 5, 6, 7, 10 |
| `sac_dur_max` | 4 | 0, 2, 7, 10 |
| `sac_amp_mean` | 5 | 3, 5, 7, 8, 10 |
| `sac_amp_std` | 3 | 3, 5, 10 |
| `sac_turn_std` | 3 | 0, 8, 9 |

Trong số đó, **8 features** có p < 0.1 ở cả ba thẻ King–Devick (tasks 3, 4, 5):
`spa_fix_count`, `geo_sacc_amp_mean`, `geo_revisit_rate`, `geo_nn_dist_mean`, `pup_mean`, `pup_median`, `sac_count`, `sac_dur_total`.

- Thống kê đầy đủ (H, p, q_bh, trung bình theo nhóm) cho 72 features × 11
  tasks: `docs/baseline/eda_summary_vi.md` (tiếng Anh: `docs/baseline/eda_summary.md`).
- Bộ feature chính thức (25 feature): `finalized_dataset/` — dùng cho mọi thí
  nghiệm từ nay về sau.

## 4. Danh sách mở rộng (39 features, p < 0.2)

Ngưỡng sàng lọc lỏng hơn: các feature có p-value Kruskal–Wallis < 0.2 (chưa
hiệu chỉnh) ở **ít nhất 3 trong 11 tasks**. 25 feature ở Mục 3 là tập con của
39 feature này. Lưu ý thống kê: với 792 kiểm định, nếu không có khác biệt thật
nào thì ngẫu nhiên cũng cho ~28/72 features đạt tiêu chí này, vì vậy danh sách
chỉ nên dùng làm tập ứng viên sàng lọc rộng.

| Feature | Số tasks (p<0.2) | Tasks |
|---|---|---|
| `pup_max` | 11 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_mean` | 11 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_median` | 11 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `pup_min` | 10 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| `sac_dur_total` | 9 | 0, 1, 2, 3, 4, 5, 6, 7, 10 |
| `geo_nn_dist_mean` | 8 | 0, 1, 2, 3, 4, 5, 6, 9 |
| `sac_amp_mean` | 8 | 3, 4, 5, 6, 7, 8, 9, 10 |
| `sac_count` | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `sac_dur_std` | 7 | 0, 1, 2, 6, 7, 9, 10 |
| `sac_turn_std` | 7 | 0, 3, 4, 6, 7, 8, 9 |
| `spa_fix_count` | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `geo_revisit_rate` | 6 | 0, 1, 2, 3, 4, 5 |
| `geo_sacc_amp_mean` | 6 | 3, 4, 5, 6, 7, 8 |
| `geo_scanpath_len` | 6 | 0, 2, 4, 5, 6, 9 |
| `sac_dur_max` | 6 | 0, 2, 6, 7, 8, 10 |
| `sac_dur_mean` | 6 | 1, 2, 3, 4, 9, 10 |
| `spa_center_dist_std` | 6 | 0, 1, 2, 4, 6, 9 |
| `tem_dur_total` | 6 | 0, 2, 4, 5, 6, 10 |
| `pup_std` | 5 | 1, 2, 7, 9, 10 |
| `pupdiff_std` | 5 | 1, 7, 8, 9, 10 |
| `sac_amp_std` | 5 | 3, 5, 7, 9, 10 |
| `spa_q2` | 5 | 0, 3, 4, 5, 10 |
| `spa_skew_x` | 5 | 0, 6, 7, 8, 9 |
| `geo_sacc_amp_max` | 4 | 0, 5, 6, 9 |
| `pupbl_slope` | 4 | 1, 2, 7, 10 |
| `pupbl_std` | 4 | 1, 2, 9, 10 |
| `pupdiff_slope` | 4 | 2, 4, 7, 9 |
| `sac_absdir_var` | 4 | 1, 2, 3, 6 |
| `spa_bbox_area` | 4 | 0, 3, 4, 6 |
| `spa_dispersion` | 4 | 3, 6, 8, 9 |
| `geo_angle_var` | 3 | 1, 3, 5 |
| `pup_first_last_diff` | 3 | 1, 2, 10 |
| `pup_slope` | 3 | 1, 2, 10 |
| `pupbl_first_last_diff` | 3 | 1, 2, 10 |
| `pupbl_max` | 3 | 1, 7, 8 |
| `spa_center_frac` | 3 | 1, 3, 9 |
| `spa_std_x` | 3 | 3, 6, 8 |
| `tem_dur_max` | 3 | 5, 8, 10 |
| `tem_velocity_mean` | 3 | 3, 5, 8 |

- Dữ liệu chỉ gồm 39 feature này: `processed_dataset_v3/`.

## Cấu trúc repository

| Đường dẫn | Nội dung |
|---|---|
| `dataset/`, `dataset_v2/` | file raw xuất từ eye-tracker (2025-03-05) |
| `processed_dataset/` | features (72) + metadata của 157 người tham gia + thống kê sàng lọc |
| `finalized_dataset/` | bộ feature chính thức (25 feature được chọn) |
| `processed_dataset_v3/` | tập con 39 feature (p < 0.2 ở ≥ 3 tasks) |
| `fold/` | phân fold 4-fold mức người tham gia (CSV) |
| `docs/baseline/` | tài liệu dự án (EN + VI): tổng hợp EDA, chia fold, so sánh baseline, baseline bộ dataset chính thức |
| `outputs/` | kết quả thí nghiệm (baseline, provenance, nhật ký chạy) |
| `src/begad_norm/` | mã trích xuất feature, cấu hình và thí nghiệm baseline |
| `tests/` | unit test (config, raw data, features) |

## Ghi chú

- p-value dùng để sàng lọc là p-value Kruskal–Wallis chưa hiệu chỉnh; `q_bh`
  (Benjamini–Hochberg trên toàn bộ 792 kiểm định) có sẵn trong
  `docs/baseline/eda_summary_vi.md`.
- Giá trị feature trên các task màn hình hướng dẫn (0, 2, 6, 8, 10) mô tả việc
  đọc hướng dẫn, không phải stimulus kiểm tra.
