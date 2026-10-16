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

## 2. Features (25)

25 feature được tính cho mỗi (người tham gia, task) từ các fixation sạch (xem
`docs/baseline/eda_summary_vi.md` để biết quy trình làm sạch, công thức và đơn vị; bản
tiếng Anh: `docs/baseline/eda_summary.md`). Phân nhóm theo tiền tố:

| Tiền tố | n | Ý nghĩa |
|---|---|---|
| `spa_` | 4 | phân bố không gian của fixation |
| `geo_` | 4 | hình học đường quét (scanpath) |
| `tem_` | 2 | động học theo thời gian |
| `pup_` | 6 | kích thước đồng tử |
| `pupbl_` | 2 | đường nền đồng tử |
| `sac_` | 7 | saccade đo được |

Danh sách đầy đủ:

| Nhóm | Feature |
|---|---|
| spa — phân bố không gian của fixation | `spa_fix_count` |
| spa — phân bố không gian của fixation | `spa_dispersion` |
| spa — phân bố không gian của fixation | `spa_center_dist_std` |
| spa — phân bố không gian của fixation | `spa_q2` |
| geo — hình học đường quét (scanpath) | `geo_scanpath_len` |
| geo — hình học đường quét (scanpath) | `geo_sacc_amp_mean` |
| geo — hình học đường quét (scanpath) | `geo_revisit_rate` |
| geo — hình học đường quét (scanpath) | `geo_nn_dist_mean` |
| tem — động học theo thời gian | `tem_dur_total` |
| tem — động học theo thời gian | `tem_velocity_mean` |
| pup — kích thước đồng tử | `pup_mean` |
| pup — kích thước đồng tử | `pup_std` |
| pup — kích thước đồng tử | `pup_min` |
| pup — kích thước đồng tử | `pup_max` |
| pup — kích thước đồng tử | `pup_median` |
| pup — kích thước đồng tử | `pup_slope` |
| pupbl — đường nền đồng tử | `pupbl_std` |
| pupbl — đường nền đồng tử | `pupbl_slope` |
| sac — saccade đo được | `sac_count` |
| sac — saccade đo được | `sac_dur_mean` |
| sac — saccade đo được | `sac_dur_total` |
| sac — saccade đo được | `sac_dur_max` |
| sac — saccade đo được | `sac_amp_mean` |
| sac — saccade đo được | `sac_amp_std` |
| sac — saccade đo được | `sac_turn_std` |

## 3. Chọn lọc feature

25 feature được sàng lọc từ tập feature đã trích xuất bằng kiểm định
Kruskal–Wallis cho từng (feature, task): giữ lại các feature có p < 0.1 (chưa
hiệu chỉnh) ở **ít nhất 3 trong 11 tasks**. Việc sàng lọc chỉ mang tính thăm dò
và không hiệu chỉnh đa kiểm định ở bước chọn lọc. Bằng chứng:
`processed_dataset/selection.csv`.

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

## 4. Thí nghiệm baseline

Các mô hình baseline cổ điển (logistic regression, SVM RBF/linear, FNN, QDA,
GNB) trên một mẫu mỗi người tham gia (25 feature × 11 tasks = 275 cột), CV
5-fold mức người tham gia × 20 seeds, có xử lý mất cân bằng lớp (class
weight cân bằng / prior nghịch đảo tần suất, ngưỡng quyết định cố định
0.5):

- Protocol và kết quả: `docs/baseline/baseline_balanced_vi.md`
  (tiếng Anh: `docs/baseline/baseline_balanced.md`).
- Đầu ra dạng máy đọc: `outputs/baseline_balanced/` (kết quả, missingness,
  provenance, nhật ký chạy).
- Chạy lại: `uv run python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml`.

## Cấu trúc repository

| Đường dẫn | Nội dung |
|---|---|
| `dataset/`, `dataset_v2/` | file raw xuất từ eye-tracker (2025-03-05) |
| `processed_dataset/` | dữ liệu đã xử lý (25 feature) + metadata của 157 người tham gia + bằng chứng chọn lọc |
| `fold1/` | phân fold 5-fold mức người tham gia (CSV) |
| `fold/` | split tham chiếu 4-fold cũ (CSV) |
| `docs/baseline/` | tài liệu dự án (EN + VI): tổng hợp EDA, chia fold, baseline cân bằng |
| `outputs/` | kết quả thí nghiệm (baseline, provenance, nhật ký chạy) |
| `src/begad_norm/` | mã trích xuất feature, cấu hình và thí nghiệm baseline |
| `tests/` | unit test (config, raw data, features) |

## Ghi chú

- p-value dùng để sàng lọc là p-value Kruskal–Wallis chưa hiệu chỉnh; 25
  feature được sàng lọc trên chính cohort này, nên hiệu năng CV về sau mang
  thiên lệch chọn lọc trong mẫu.
- Giá trị feature trên các task màn hình hướng dẫn (0, 2, 6, 8, 10) mô tả việc
  đọc hướng dẫn, không phải stimulus kiểm tra.
