# Từ điển Features BEGAD — dataset_v2, theo (participant, task)

Tài liệu này mô tả 72 features trong `processed_dataset/features.csv`
(1727 dòng = 157 người × 11 task; 2 dòng all-NaN: `BEGCO-029` và
`BEGMCI-019`, task 10). Mỗi dòng được định danh bởi `participant_id`
(RealTag) và `task` (0–10). Features chia thành các nhóm: `spa_` (17),
`geo_` (10), `tem_` (11), `pup_` (7), `pupbl_` (7), `pupdiff_` (7), `sac_` (13).

## Nguồn dữ liệu và các bước làm sạch

Mọi feature được tính từ **tập fixation sạch** của một cặp (participant, task),
trích từ `dataset_v2/20250305/BEGAD_ETRA25_<RealTag>.csv` bằng
`src/begad_norm/rawdata.py`:

1. Giữ các dòng: `Final == True`, `HasNulls == False`,
   `GazeEventType == "Fixation"`, `FixationIndex > 0` (cohort:
   `Group ∈ {0,1,2}`, `IsFollowup == 0`, theo roster 157 người trong
   `dataset_v2/participants_20250305.csv`).
2. Cửa sổ thời lượng EMS: `40 ms < GazeEventDuration ≤ 2000 ms`
   (loại 2 dòng trên toàn bộ).
3. Giới hạn media theo task trên `FixationPointX/Y (MCSpx)`
   (kể cả biên; loại 0 dòng).
4. Ngoại lai pupil: theo (participant, task), loại dòng có pupil trung bình
   2 mắt lệch khỏi median nhóm quá `4 × max(std, 1e-6)`
   (giống hệt EMS; loại 381 dòng trên toàn bộ).

Tổng fixation sạch: 116.274 (116.657 sau bước 1, trừ 2 duration + 381 pupil).
Các cột thập phân kiểu châu Âu (`"3,98"`) được chuyển sang float
(`str.replace(',', '.')`). Fixation được sắp theo `RecordingTimestamp`
trong mỗi task.

Ký hiệu: `n` = số fixation sạch của cặp; `x, y` = tọa độ fixation
(`FixationPointX/Y (MCSpx)`, px, tương đối với media); `dur` = thời lượng
fixation (`GazeEventDuration`, ms); `pup` = pupil trung bình 2 mắt
`(PupilLeft + PupilRight) / 2` (mm). `W, H` = kích thước media theo task từ
`TASK_GEOMETRY` (config.py); tâm `(cx, cy) = (W/2, H/2)`. Mọi std dùng
`ddof=1` (SD mẫu). Cặp có `n < 2` nhận dòng all-NaN (EMS `MIN_FIX = 2`).
`NaN` lan truyền khi đầu vào không xác định (xem cột NaN).

## `spa_` — vị trí & độ phân tán không gian (17)

Cài đặt tham chiếu: phần A–B của `compute_stimulus_features` trong
`src/begad_norm/features/stimulus.py` (port nguyên văn từ EMS-Projects).

| Feature | Công thức | Cột nguồn (dataset_v2) | NaN / ghi chú |
|---|---|---|---|
| `spa_fix_count` | `n` | số dòng sau khi làm sạch | không bao giờ NaN |
| `spa_mean_x` | `mean(x)` | `FixationPointX (MCSpx)` | px |
| `spa_mean_y` | `mean(y)` | `FixationPointY (MCSpx)` | px |
| `spa_std_x` | `std(x, ddof=1)` | như trên | NaN nếu `n < 2` |
| `spa_std_y` | `std(y, ddof=1)` | như trên | NaN nếu `n < 2` |
| `spa_dispersion` | `mean(sqrt((x−mean(x))² + (y−mean(y))²))` | cả 2 cột fixation point | khoảng cách trung bình tới trọng tâm, px |
| `spa_bbox_area` | `(max(x)−min(x)) × (max(y)−min(y))` | cả 2 cột | px²; bằng 0 với 1 điểm |
| `spa_center_dist_mean` | `mean(sqrt((x−cx)² + (y−cy)²))` | fixation points + `TASK_GEOMETRY` | px tới tâm media |
| `spa_center_dist_std` | `std(khoảng cách tới tâm, ddof=1)` | như trên | NaN nếu `n < 2` |
| `spa_center_frac` | `mean(\|x−cx\| < W/4 VÀ \|y−cy\| < H/4)` | như trên | tỷ lệ nằm trong 1/4 trung tâm media |
| `spa_q1…spa_q4` | tỷ lệ theo góc phần tư: q1 `x<cx & y<cy`; q2 `x≥cx & y<cy`; q3 `x<cx & y≥cy`; q4 `x≥cx & y≥cy` | như trên | tổng = 1 |
| `spa_entropy` | entropy Shannon của histogram lưới 8×6 trên `(0,W)×(0,H)`, chia `log(48)` | như trên | 0 = 1 ô duy nhất, 1 = phân bố đều |
| `spa_max_grid_frac` | `max(histogram 8×6) / n` | như trên | mật độ ô đỉnh |
| `spa_skew_x` | `mean((x−mean(x))³) / std(x)³` (không hiệu chỉnh chệch) | `FixationPointX (MCSpx)` | NaN nếu `n ≤ 2` hoặc `std == 0` |

## `geo_` — hình học scanpath (10)

Chuyển tiếp giữa các fixation liên tiếp: `dx = diff(x)`, `dy = diff(y)`,
`amp = sqrt(dx² + dy²)` — các saccade *suy diễn* giữa các fixation liên tiếp
(quy ước EMS, giữ nguyên dù BEGAD cũng ghi saccade đo thực — saccade đo thực
nằm ở nhóm `sac_`). Tham chiếu: phần C của `compute_stimulus_features`.

| Feature | Công thức | Nguồn | NaN / ghi chú |
|---|---|---|---|
| `geo_scanpath_len` | `sum(amp)` | fixation points | px; NaN nếu `n < 2` |
| `geo_sacc_amp_mean` | `mean(amp)` | như trên | biên độ saccade suy diễn, px |
| `geo_sacc_amp_std` | `std(amp, ddof=1)` | như trên | NaN nếu `n < 2` |
| `geo_sacc_amp_max` | `max(amp)` | như trên | |
| `geo_dx_mean` | `mean(dx)` | như trên | độ trôi ngang ròng, px |
| `geo_dy_mean` | `mean(dy)` | như trên | độ trôi dọc ròng, px |
| `geo_angle_var` | `1 − R`, `R = sqrt(mean(cos θ)² + mean(sin θ)²)`, `θ = atan2(dy, dx)` | như trên | 0 = cùng một hướng, 1 = đều; bước độ dài 0 tính là `θ=0` (thẳng) |
| `geo_revisit_rate` | tỷ lệ fixation có fixation *về sau* cách trong 60 px (chỉ nhìn về trước, `triu(dist<60, k=1)`) | như trên, `REVISIT_RADIUS = 60` px | fixation cuối không bao giờ được đếm nhưng vẫn ở mẫu số |
| `geo_nn_dist_mean` | trung bình khoảng cách tới fixation *khác* gần nhất | như trên | px |
| `geo_hull_area` | diện tích `scipy.spatial.ConvexHull(x, y)` | như trên | px²; NaN nếu `n < 4` hoặc suy biến (thẳng hàng) |

## `tem_` — thời gian (11)

Xấp xỉ IFI nửa dwell (không ghi onset timestamp):
`half_gap = (dur[:-1] + dur[1:]) / 2`. Tham chiếu: phần D của
`compute_stimulus_features`.

| Feature | Công thức | Nguồn | NaN / ghi chú |
|---|---|---|---|
| `tem_dur_mean` | `mean(dur)` | `GazeEventDuration` (ms) | |
| `tem_dur_std` | `std(dur, ddof=1)` | như trên | NaN nếu `n < 2` |
| `tem_dur_total` | `sum(dur)` | như trên | ms |
| `tem_dur_max` | `max(dur)` | như trên | |
| `tem_first_dur` | `dur[0]` (fixation đầu tiên theo thời gian) | như trên, thứ tự theo `RecordingTimestamp` | |
| `tem_last_dur` | `dur[-1]` | như trên | |
| `tem_ifi_mean` | `mean(half_gap)` | các cặp `GazeEventDuration` | ms; NaN nếu `n < 2` |
| `tem_ifi_std` | `std(half_gap, ddof=1)` | như trên | NaN nếu `n < 2` |
| `tem_velocity_mean` | `mean(amp / half_gap)` chỉ trên các giá trị hữu hạn | fixation points + durations | px/ms của saccade suy diễn; NaN nếu không có giá trị hữu hạn |
| `tem_fix_rate` | `n / total_time × 1000`, `total_time = sum(dur) + tem_ifi_mean × (n−1)` | durations | 1/s; NaN nếu `total_time ≤ 0` |
| `tem_trans_entropy` | entropy Shannon của điểm *bắt đầu* chuyển tiếp `(x[:-1], y[:-1])` trên lưới 4×3, chia `log(12)` | fixation points | NaN nếu `n < 2`; 0 nếu mọi điểm bắt đầu cùng 1 ô |

## `pup_`, `pupbl_`, `pupdiff_` — pupil (7 + 7 + 7)

Cùng 7 công thức áp dụng lên ba chuỗi pupil của mỗi cặp:

- `pup_`: trung bình 2 mắt thô `pup_i = (PupilLeft_i + PupilRight_i) / 2` (mm).
- `pupbl_`: trung bình 2 mắt đã trừ baseline
  `((PupilLeft_i − PupilLeftBaseline) + (PupilRight_i − PupilRightBaseline)) / 2`
  (baseline là hằng số theo từng recording).
- `pupdiff_`: hiệu trái − phải thô `PupilLeft_i − PupilRight_i` (mm).

| Feature | Công thức | Nguồn | NaN / ghi chú |
|---|---|---|---|
| `<pre>_mean` | `mean(chuỗi)` | `PupilLeft`, `PupilRight` (+ `PupilLeftBaseline`, `PupilRightBaseline` với `pupbl_`) | mm; `pupbl_*` NaN với 16 người có baseline bị thiếu trong dữ liệu gốc |
| `<pre>_std` | `std(chuỗi, ddof=1)` | như trên | NaN nếu `n < 2` |
| `<pre>_min` / `<pre>_max` | `min` / `max` của chuỗi | như trên | |
| `<pre>_median` | `median(chuỗi)` | như trên | |
| `<pre>_slope` | hồi quy tuyến tính (`np.polyfit`) của chuỗi theo thứ tự fixation 0…n−1 | như trên, thứ tự theo `RecordingTimestamp` | theo chỉ số fixation; NaN nếu `n < 2` |
| `<pre>_first_last_diff` | `chuỗi[-1] − chuỗi[0]` | như trên | NaN nếu `n < 2` |

`<pre>` ∈ {`pup`, `pupbl`, `pupdiff`} — ví dụ `pup_mean`, `pupbl_std`,
`pupdiff_first_last_diff`. Tham chiếu: phần E của `compute_stimulus_features`.

## `sac_` — saccade đo thực (13)

Nhóm mới (BEGAD có saccade đo thực; EMS chỉ suy diễn). Nguồn:
(a) thuộc tính saccade gắn trên mỗi fixation sạch — `SaccadicAmplitude`
(độ), `AbsoluteSaccadicDirection` (độ, 0–360), `RelativeSaccadicDirection`
(độ, góc quay so với hướng saccade trước); (b) các dòng saccade `Final`
(chúng luôn có `HasNulls == True`, nên KHÔNG áp filter HasNulls cho chúng)
được gom thành *runs*: các dòng saccade liên tiếp theo thời gian có chung
fixation hạ cánh (fixation sạch đầu tiên có timestamp lớn hơn timestamp của
saccade). `run_dur` = max `GazeEventDuration` trong run. `m` = số runs;
`total_time` = tổng thời gian theo EMS (xem `tem_fix_rate`).

Tham chiếu: `src/begad_norm/features/saccade.py` và `extract_saccade_runs`
trong `rawdata.py`.

| Feature | Công thức | Nguồn | NaN / ghi chú |
|---|---|---|---|
| `sac_count` | `m` | saccade runs | 0 hợp lệ |
| `sac_rate` | `m / total_time × 1000` | runs + durations | event/s; NaN nếu `total_time ≤ 0` |
| `sac_dur_mean` | `mean(run_durs)` | `GazeEventDuration` của các dòng saccade Final | ms; NaN nếu `m = 0` |
| `sac_dur_std` | `std(run_durs, ddof=1)` | như trên | NaN nếu `m < 2` |
| `sac_dur_total` | `sum(run_durs)` | như trên | NaN nếu `m = 0` |
| `sac_dur_max` | `max(run_durs)` | như trên | NaN nếu `m = 0` |
| `sac_amp_mean` | `mean(amp)` trên biên độ gắn theo fixation | `SaccadicAmplitude` (độ, thập phân châu Âu) | biên độ saccade hạ cánh vào từng fixation |
| `sac_amp_std` | `std(amp, ddof=1)` | như trên | NaN nếu `n < 2` |
| `sac_amp_max` | `max(amp)` | như trên | |
| `sac_absdir_mean` | trung bình vòng của hướng tuyệt đối: `deg(atan2(mean(sin θ), mean(cos θ))) mod 360` | `AbsoluteSaccadicDirection` (độ) | giá trị cách 360 trong 1e-12 được đưa về 0 |
| `sac_absdir_var` | `1 − R`, `R = sqrt(mean(cos θ)² + mean(sin θ)²)` | như trên | 0 = cùng hướng; NaN nếu `n < 2` |
| `sac_turn_mean` | `mean(min(reldir, 360 − reldir))` | `RelativeSaccadicDirection` (độ, gập về [0, 180]) | độ lớn góc quay trung bình giữa các saccade liên tiếp |
| `sac_turn_std` | `std(turn, ddof=1)` | như trên | NaN nếu `n < 2` |

## File đầu ra

- `processed_dataset/features.csv` — 1727 dòng × 74 cột (`participant_id`,
  `task`, 72 features; thứ tự = `spa_` 17, `geo_` 10, `tem_` 11, `pup_` 7,
  `pupbl_` 7, `pupdiff_` 7, `sac_` 13).
- `processed_dataset/metadata.csv` — nhóm, file, số fixation, số task,
  cờ thiếu baseline theo từng người.
- `processed_dataset/feature_names.txt`, `quality_report.txt`.
- Chạy lại: `uv run python -m begad_norm.scripts.build_features`.
- Test: `uv run python -m unittest discover -s tests` (34 test, gồm probe
  chứng minh port 45 features khớp nguyên văn EMS và golden test end-to-end).

## Dữ liệu khuyết

- 2 cặp all-NaN (`BEGCO-029`, `BEGMCI-019` × task 10): dưới 2 fixation sạch.
- `pupbl_*`: NaN với 16 người có baseline pupil bị thiếu trong CSV gốc
  (ghi trong `quality_report.txt`); `pup_` và `pupdiff_` không bị ảnh hưởng.
- Các feature còn lại: NaN < 1% trên 1725 dòng không rỗng.
