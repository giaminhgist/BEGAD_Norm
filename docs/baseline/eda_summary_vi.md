# Tổng hợp EDA BEGAD — dữ liệu, đặc trưng và so sánh nhóm

> Tổng hợp ngày 2026-10-12 từ thư mục `EDA/` cũ (từ điển đặc trưng, báo cáo so
> sánh nhóm, thống kê từng task) trong đợt dọn dẹp repository. Các hình EDA và
> notebook so sánh đã bị xóa; có thể khôi phục từ lịch sử git. Bảng thống kê
> đầy đủ từng task được giữ tại
> `processed_dataset/group_comparison_stats_per_task.csv`.

## 1. Dữ liệu và nhóm người tham gia

- **Phiên ghi**: 2025-03-05, mã xuất `BEGAD_ETRA25`; 205 file CSV eye-tracking
  thô (`dataset/20250305/`, `dataset_v2/20250305/`).
- **Lọc nhóm**: chỉ giữ người tham gia lần khám đầu thuộc nhóm CO / MCI / AD.
  Bị loại: 36 lần tái khám, 10 người NPH (Group 3), 2 bản ghi Group −11.
- **Người tham gia**: 157 — CO 100, MCI 41, AD 16.
- **Protocol**: 11 task mỗi người (0–10): màn hình hướng dẫn (0, 2, 6, 8, 10),
  King–Devick demo/thẻ (1, 3, 4, 5), đọc thử/bài đọc (7, 9). Kích thước media
  trong `TASK_GEOMETRY` (`src/begad_norm/config.py`).

## 2. Quy trình làm sạch

Tính cho từng cặp (người tham gia, task) từ dữ liệu thô `dataset_v2`
(`src/begad_norm/rawdata.py`):

1. Giữ các dòng: `Final == True`, `HasNulls == False`,
   `GazeEventType == "Fixation"`, `FixationIndex > 0` (đã áp lọc nhóm).
2. Cửa sổ thời lượng EMS: `40 ms < GazeEventDuration ≤ 2000 ms` (loại 2 dòng).
3. Giới hạn media từng task trên `FixationPointX/Y (MCSpx)` (loại 0 dòng).
4. Ngoại lệ đồng tử: theo từng cặp (người, task), loại dòng có đồng tử trung
   bình hai mắt lệch trung vị nhóm quá `4 × max(std, 1e-6)` (loại 381 dòng).

Tổng fixation sạch: **116,274**. Cột dấu phẩy thập phân được chuyển sang float.
Fixation được sắp theo `RecordingTimestamp` trong từng task.

## 3. Từ điển đặc trưng (72 đặc trưng)

72 đặc trưng cho mỗi cặp (người tham gia, task) trong
`processed_dataset/features.csv` (1727 dòng = 157 × 11; 2 dòng all-NaN:
`BEGCO-029`, `BEGMCI-019` × task 10). Ký hiệu: `n` = số fixation sạch;
`x, y` = tọa độ tâm fixation (px, tương đối với media); `dur` = thời lượng
fixation (ms); `pup` = đồng tử trung bình hai mắt (mm); `W, H` = kích thước
media của task. Độ lệch chuẩn mẫu (`ddof=1`); cặp có `n < 2` nhận dòng
all-NaN (`MIN_FIX = 2`).

### `spa_` — vị trí không gian & độ phân tán (17)

| Đặc trưng | Công thức | Ghi chú |
|---|---|---|
| `spa_fix_count` | `n` | không bao giờ NaN |
| `spa_mean_x`, `spa_mean_y` | `mean(x)`, `mean(y)` | px |
| `spa_std_x`, `spa_std_y` | `std(x)`, `std(y)` | NaN nếu `n < 2` |
| `spa_dispersion` | `mean(sqrt((x−mean(x))² + (y−mean(y))²))` | px, khoảng cách tới trọng tâm |
| `spa_bbox_area` | `(max(x)−min(x)) × (max(y)−min(y))` | px² |
| `spa_center_dist_mean` / `_std` | khoảng cách fixation tới tâm media `(W/2, H/2)` | px |
| `spa_center_frac` | tỉ lệ nằm trong phần tư trung tâm (`|x−cx|<W/4`, `|y−cy|<H/4`) | |
| `spa_q1…spa_q4` | tỉ lệ theo góc phần tư quanh tâm media | tổng = 1 |
| `spa_entropy` | entropy Shannon của lược đồ lưới 8×6 / `log(48)` | 0 = một ô, 1 = đều |
| `spa_max_grid_frac` | mật độ ô lớn nhất của lược đồ 8×6 | |
| `spa_skew_x` | `mean((x−mean(x))³) / std(x)³` | NaN nếu `n ≤ 2` hoặc `std = 0` |

### `geo_` — hình học đường quét (10)

Saccade suy diễn giữa các fixation liên tiếp: `dx = diff(x)`,
`amp = sqrt(dx² + dy²)`.

| Đặc trưng | Công thức | Ghi chú |
|---|---|---|
| `geo_scanpath_len` | `sum(amp)` | px; NaN nếu `n < 2` |
| `geo_sacc_amp_mean` / `_std` / `_max` | mean / std / max của `amp` | px |
| `geo_dx_mean`, `geo_dy_mean` | `mean(dx)`, `mean(dy)` | độ trôi ròng |
| `geo_angle_var` | `1 − R`, `R = sqrt(mean(cos θ)² + mean(sin θ)²)`, `θ = atan2(dy, dx)` | 0 = một hướng, 1 = đều |
| `geo_revisit_rate` | tỉ lệ fixation có fixation sau đó trong bán kính 60 px | chỉ nhìn về phía trước (`triu`, k=1) |
| `geo_nn_dist_mean` | khoảng cách trung bình tới fixation khác gần nhất | px |
| `geo_hull_area` | diện tích bao lồi của `(x, y)` | px²; NaN nếu `n < 4` hoặc suy biến |

### `tem_` — thời gian (11)

Xấp xỉ IFI nửa dwell (không có timestamp bắt đầu): `half_gap = (dur[:-1] + dur[1:]) / 2`.

| Đặc trưng | Công thức | Ghi chú |
|---|---|---|
| `tem_dur_mean` / `_std` / `_max` | mean / std / max của `dur` | ms |
| `tem_dur_total` | `sum(dur)` | ms |
| `tem_first_dur`, `tem_last_dur` | thời lượng fixation đầu / cuối theo thứ tự thời gian | |
| `tem_ifi_mean` / `_std` | mean / std của `half_gap` | ms; NaN nếu `n < 2` |
| `tem_velocity_mean` | `mean(amp / half_gap)` chỉ trên các số hạng hữu hạn | px/ms |
| `tem_fix_rate` | `n / total_time × 1000`, `total_time = sum(dur) + tem_ifi_mean × (n−1)` | 1/s |
| `tem_trans_entropy` | entropy điểm bắt đầu chuyển tiếp trên lưới 4×3 / `log(12)` | NaN nếu `n < 2` |

### `pup_`, `pupbl_`, `pupdiff_` — đồng tử (7 + 7 + 7)

Cùng 7 công thức trên ba chuỗi: `pup_` = trung bình hai mắt
`(PupilLeft + PupilRight) / 2`; `pupbl_` = trung bình hai mắt đã hiệu chỉnh
baseline (baseline cố định theo bản ghi); `pupdiff_` = chênh lệch trái − phải.

| Đặc trưng | Công thức |
|---|---|
| `<pre>_mean` | `mean(series)` |
| `<pre>_std` | `std(series)` — NaN nếu `n < 2` |
| `<pre>_min` / `<pre>_max` | min / max |
| `<pre>_median` | median |
| `<pre>_slope` | hồi quy tuyến tính (`np.polyfit`) theo thứ tự fixation 0…n−1 |
| `<pre>_first_last_diff` | `series[-1] − series[0]` |

`pupbl_*` là NaN với 16 người thiếu baseline đồng tử trong dữ liệu thô.

### `sac_` — saccade đo được (13)

BEGAD ghi sự kiện saccade trực tiếp (EMS chỉ suy diễn). Thuộc tính theo
fixation: `SaccadicAmplitude` (deg), `AbsoluteSaccadicDirection` (deg),
`RelativeSaccadicDirection` (deg). Các dòng sự kiện saccade `Final` (luôn có
`HasNulls == True`, nên không áp bộ lọc HasNulls) được gộp thành các *run*
cùng fixation hạ cánh; `m` = số run.

| Đặc trưng | Công thức | Ghi chú |
|---|---|---|
| `sac_count` | `m` | 0 hợp lệ |
| `sac_rate` | `m / total_time × 1000` | sự kiện/s |
| `sac_dur_mean` / `_std` / `_total` / `_max` | thời lượng các run | ms; NaN nếu `m = 0` (std: `m < 2`) |
| `sac_amp_mean` / `_std` / `_max` | biên độ saccade theo fixation | deg |
| `sac_absdir_mean` | trung bình vòng của hướng tuyệt đối | deg, 0–360 |
| `sac_absdir_var` | `1 − R` của trung bình vòng | 0 = cùng hướng |
| `sac_turn_mean` / `_std` | mean / std của `min(reldir, 360 − reldir)` | deg, gập về [0, 180] |

## 4. So sánh nhóm và sàng lọc đặc trưng

**Phương pháp**: một kiểm định Kruskal–Wallis cho mỗi (task, feature):
11 task × 72 feature = **792 kiểm định**. Đơn vị: một giá trị cho mỗi (người
tham gia, task); trong một task, ba nhóm là các mẫu độc lập. `p` = p-value
chưa hiệu chỉnh; `q_bh` = Benjamini–Hochberg trên cả 792 kiểm định. Sàng lọc
mang tính thăm dò, cần lưu ý kiểm định bội.

### 25 đặc trưng được chọn (p < 0.1 ở ≥ 3 task)

`finalized_dataset/`; bằng chứng chọn trong `finalized_dataset/selection.csv`.

| Đặc trưng | n_tasks | Tasks (p<0.1) |
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

**8 đặc trưng** có ý nghĩa (p < 0.1) ở cả ba thẻ King–Devick (task 3, 4, 5):
`spa_fix_count`, `geo_sacc_amp_mean`, `geo_revisit_rate`, `geo_nn_dist_mean`,
`pup_mean`, `pup_median`, `sac_count`, `sac_dur_total`.

### 39 đặc trưng mở rộng (p < 0.2 ở ≥ 3 task)

`processed_dataset_v3/`; bằng chứng chọn trong `processed_dataset_v3/selection.csv`.
25 đặc trưng ở trên là tập con. Với giả thuyết không toàn cục, khoảng 28
trong 72 đặc trưng sẽ đạt tiêu chí này chỉ do ngẫu nhiên (792 kiểm định), nên
danh sách này chỉ là tập ứng viên rộng. Đáng chú ý (đủ 11 task ở p < 0.2):
`pup_max`, `pup_mean`, `pup_median`; `pup_min` (10); `sac_dur_total` (9);
`geo_nn_dist_mean`, `sac_amp_mean` (8); `sac_count`, `sac_dur_std`,
`sac_turn_std`, `spa_fix_count` (7); cộng 20 đặc trưng khác với 3–6 task —
danh sách đầy đủ trong `processed_dataset_v3/selection.csv`.

### Missingness

- 2 cặp all-NaN (`BEGCO-029`, `BEGMCI-019` × task 10): dưới 2 fixation sạch.
- `pupbl_*`: NaN với 16 người thiếu baseline đồng tử trong CSV thô; `pup_`
  và `pupdiff_` không bị ảnh hưởng.
- Các đặc trưng còn lại: < 1% NaN trên 1725 dòng không rỗng.

## 5. File liên quan

| File | Nội dung |
|---|---|
| `processed_dataset/group_comparison_stats_per_task.csv` | thống kê đầy đủ từng (task, feature): H, p, q_bh, trung bình nhóm, n |
| `processed_dataset/features.csv`, `metadata.csv` | 72 đặc trưng + metadata của 157 người |
| `finalized_dataset/`, `processed_dataset_v3/` | bộ feature chính thức (25) và tập mở rộng (39) |
| `src/begad_norm/features/stimulus.py`, `features/saccade.py` | cài đặt tham chiếu |
| `docs/baseline/fold_split_vi.md` | chia fold 4-fold mức người tham gia |

Tạo lại đặc trưng: `uv run python -m begad_norm.scripts.build_features`.
