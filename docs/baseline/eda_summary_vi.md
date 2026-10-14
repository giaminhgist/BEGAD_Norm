# Tổng hợp EDA BEGAD — dữ liệu, đặc trưng và so sánh nhóm

> Tổng hợp ngày 2026-10-12 từ thư mục `EDA/` cũ (từ điển đặc trưng, báo cáo so
> sánh nhóm, thống kê từng task) trong đợt dọn dẹp repository. Các hình EDA và
> notebook so sánh đã bị xóa; có thể khôi phục từ lịch sử git. Dữ liệu được
> rút gọn còn 25 đặc trưng được chọn vào ngày 2026-10-13; bằng chứng sàng lọc
> được giữ tại `processed_dataset/selection.csv`.

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

## 3. Từ điển đặc trưng (25 đặc trưng)

25 đặc trưng cho mỗi cặp (người tham gia, task) trong
`processed_dataset/features.csv` (1727 dòng = 157 × 11; 2 dòng all-NaN:
`BEGCO-029`, `BEGMCI-019` × task 10). Ký hiệu: `n` = số fixation sạch;
`x, y` = tọa độ tâm fixation (px, tương đối với media); `dur` = thời lượng
fixation (ms); `pup` = đồng tử trung bình hai mắt (mm); `W, H` = kích thước
media của task; `cx, cy` = tâm media `(W/2, H/2)`. Độ lệch chuẩn mẫu
(`ddof=1`); cặp có `n < 2` nhận dòng all-NaN (`MIN_FIX = 2`).

### `spa_` — vị trí không gian & độ phân tán (4)

| Đặc trưng | Công thức | Ghi chú |
|---|---|---|
| `spa_fix_count` | `n` | không bao giờ NaN |
| `spa_dispersion` | `mean(sqrt((x−mean(x))² + (y−mean(y))²))` | px, khoảng cách tới trọng tâm |
| `spa_center_dist_std` | std của khoảng cách fixation tới `(cx, cy)` | px |
| `spa_q2` | tỉ lệ fixation có `x ≥ cx` và `y < cy` | tỉ lệ theo góc phần tư quanh tâm media |

### `geo_` — hình học đường quét (4)

Saccade suy diễn giữa các fixation liên tiếp: `dx = diff(x)`,
`amp = sqrt(dx² + dy²)`.

| Đặc trưng | Công thức | Ghi chú |
|---|---|---|
| `geo_scanpath_len` | `sum(amp)` | px; NaN nếu `n < 2` |
| `geo_sacc_amp_mean` | `mean(amp)` | px |
| `geo_revisit_rate` | tỉ lệ fixation có fixation sau đó trong bán kính 60 px | chỉ nhìn về phía trước (`triu`, k=1) |
| `geo_nn_dist_mean` | khoảng cách trung bình tới fixation khác gần nhất | px |

### `tem_` — thời gian (2)

Xấp xỉ IFI nửa dwell (không có timestamp bắt đầu): `half_gap = (dur[:-1] + dur[1:]) / 2`.

| Đặc trưng | Công thức | Ghi chú |
|---|---|---|
| `tem_dur_total` | `sum(dur)` | ms |
| `tem_velocity_mean` | `mean(amp / half_gap)` chỉ trên các số hạng hữu hạn | px/ms |

### `pup_`, `pupbl_` — đồng tử (6 + 2)

`pup_` = trung bình hai mắt `(PupilLeft + PupilRight) / 2`; `pupbl_` = trung
bình hai mắt đã hiệu chỉnh baseline (baseline cố định theo bản ghi).

| Đặc trưng | Công thức |
|---|---|
| `pup_mean` | `mean(series)` |
| `pup_std` | `std(series)` — NaN nếu `n < 2` |
| `pup_min` / `pup_max` | min / max |
| `pup_median` | median |
| `pup_slope` | hồi quy tuyến tính (`np.polyfit`) theo thứ tự fixation 0…n−1 |
| `pupbl_std` | `std` của chuỗi đã hiệu chỉnh baseline — NaN nếu `n < 2` |
| `pupbl_slope` | hồi quy tuyến tính (`np.polyfit`) theo thứ tự fixation 0…n−1 |

`pupbl_*` là NaN với 16 người thiếu baseline đồng tử trong dữ liệu thô.

### `sac_` — saccade đo được (7)

BEGAD ghi sự kiện saccade trực tiếp (EMS chỉ suy diễn). Thuộc tính theo
fixation: `SaccadicAmplitude` (deg), `AbsoluteSaccadicDirection` (deg),
`RelativeSaccadicDirection` (deg). Các dòng sự kiện saccade `Final` (luôn có
`HasNulls == True`, nên không áp bộ lọc HasNulls) được gộp thành các *run*
cùng fixation hạ cánh; `m` = số run.

| Đặc trưng | Công thức | Ghi chú |
|---|---|---|
| `sac_count` | `m` | 0 hợp lệ |
| `sac_dur_mean` / `_total` / `_max` | thời lượng các run | ms; NaN nếu `m = 0` (mean: `m < 2`) |
| `sac_amp_mean` / `_std` | biên độ saccade theo fixation | deg |
| `sac_turn_std` | std của `min(reldir, 360 − reldir)` | deg, gập về [0, 180] |

## 4. So sánh nhóm và sàng lọc đặc trưng

**Phương pháp**: một kiểm định Kruskal–Wallis cho mỗi (task, feature) trên tập
đặc trưng đã trích xuất. Đơn vị: một giá trị cho mỗi (người tham gia, task);
trong một task, ba nhóm là các mẫu độc lập. `p` = p-value chưa hiệu chỉnh.
Sàng lọc mang tính thăm dò; không hiệu chỉnh kiểm định bội ở bước chọn lọc.

### 25 đặc trưng được chọn (p < 0.1 ở ≥ 3 task)

`processed_dataset/`; bằng chứng chọn trong `processed_dataset/selection.csv`.

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

### Missingness

- 2 cặp all-NaN (`BEGCO-029`, `BEGMCI-019` × task 10): dưới 2 fixation sạch.
- `pupbl_std`, `pupbl_slope`: NaN với 16 người thiếu baseline đồng tử trong
  CSV thô (10,3% số dòng); các feature `pup_` không bị ảnh hưởng.
- Các đặc trưng còn lại: 0,116% NaN (chỉ 2 dòng all-NaN).

## 5. File liên quan

| File | Nội dung |
|---|---|
| `processed_dataset/features.csv`, `metadata.csv` | 25 đặc trưng + metadata của 157 người |
| `processed_dataset/selection.csv` | bằng chứng sàng lọc (n_tasks, tasks của từng đặc trưng được chọn) |
| `processed_dataset/quality_report.txt` | báo cáo chất lượng khi build (số dòng, cặp all-NaN, số dòng bị loại từng bước, tỉ lệ NaN) |
| `src/begad_norm/features/stimulus.py`, `features/saccade.py` | cài đặt tham chiếu |
| `docs/baseline/fold_split_vi.md` | chia fold 5-fold mức người tham gia |

Tạo lại đặc trưng: `uv run python -m begad_norm.scripts.build_features`.
