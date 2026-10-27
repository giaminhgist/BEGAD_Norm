# 37 feature mức task: định nghĩa và giá trị p

Run nguồn: `2026-10-27_v2_expanded260` (freeze trong `candidate_freeze.json`). Cohort: **RAW157** = export sự kiện Tobii thô, 157 người baseline (CN 100 / MCI 41 / AD 16); **ROI64_T1235** = workbook ROI gốc, 64 người (CN 32 / MCI 17 / AD 15). Hai cohort không bao giờ được gộp.

Luật chọn (post hoc, đặt ra sau khi đã xem kết quả): feature được liệt kê nếu thoả ít nhất một trong (1) ít nhất 2 task có p < 0.10 ở cả CN vs MCI và CN vs AD; (2) ít nhất 1 task có p MCI vs AD < 0.10; (3) ít nhất 1 task có p < 0.05 ở bất kỳ cặp nào. 37 trên 65 feature mức task kiểm định được đạt; cả 37 đều thoả điều kiện 3, mà riêng điều kiện này đã có thể đúng với khoảng một nửa số feature khi không có khác biệt thật (12 phép test mỗi feature). Danh sách mang tính thăm dò, chưa được xác nhận.

Kiểm định: Mann-Whitney U hai phía (xấp xỉ tiệm cận, hiệu chỉnh ties và continuity), mỗi người một giá trị cho mỗi task. p là p thô (chưa hiệu chỉnh đa kiểm định). Thời lượng tính bằng giây, biên độ bằng độ, đồng tử bằng mm.

## 1. Định nghĩa

| ID | Feature | Cohort | Nguồn gốc | Ý nghĩa | Công thức | Đơn vị | Điều kiện thoả |
|---|---|---|---|---|---|---|---|
| A002 | `total_fixation_time` | RAW157 | base160 | Tổng thời gian nhìn cố định (fixation) trong task. | Sum of durations of unique valid within-task fixation events | s | 1, 2, 3 |
| A003 | `task_fixation_count` | RAW157 | base160 | Số lần fixation trong task. | Number of unique valid within-task fixation events | count | 1, 2, 3 |
| A021 | `task_saccade_count` | RAW157 | base160 | Số saccade (cú chuyển mắt nhanh) trong task. | Count of unique validated within-task saccade events | count | 1, 3 |
| A023 | `amplitude_median` | RAW157 | base160 | Độ lớn điển hình của saccade (trung vị biên độ). | Median measured saccade amplitude | deg | 3 |
| A024 | `amplitude_iqr` | RAW157 | base160 | Độ phân tán độ lớn saccade (IQR biên độ). | Q75−Q25 of measured saccade amplitudes | deg | 2, 3 |
| A026 | `saccade_duration_median` | RAW157 | base160 | Thời lượng điển hình của một saccade. | Median actual saccade duration | s | 2, 3 |
| A027 | `saccade_duration_iqr` | RAW157 | base160 | Độ phân tán thời lượng saccade (IQR). | Q75−Q25 of actual saccade durations | s | 2, 3 |
| A028 | `saccade_mean_speed_median` | RAW157 | base160 | Tốc độ trung bình điển hình của saccade (không phải vận tốc đỉnh). | Median of event amplitude divided by its actual matched duration | deg/s | 3 |
| A029 | `task_tiny_saccade_fraction` | RAW157 | base160 | Tỉ lệ saccade rất nhỏ (<1 độ). | Saccades with amplitude<1 degree divided by all valid task saccades | fraction | 3 |
| A030 | `task_long_saccade_fraction` | RAW157 | base160 | Tỉ lệ saccade dài (>2 độ). | Saccades with amplitude>2 degrees divided by all valid task saccades | fraction | 3 |
| A031 | `angular_scanpath_length` | RAW157 | base160 | Tổng quãng đường mắt di chuyển (độ). | Sum of unique valid within-task measured saccade amplitudes | deg | 3 |
| A033 | `normalized_screen_dispersion` | RAW157 | base160 | Mức độ phân tán của các fixation trên màn hình. | RMS distance from mean normalized screen fixation position | normalized distance | 3 |
| A035 | `task_spatial_entropy` | RAW157 | base160 | Mức độ phủ đều của fixation trên lưới 8x6 của màn hình. | −sum(p_b log(p_b))/log(48) over a fixed 8x6 screen grid | unitless | 2, 3 |
| A046 | `task_absolute_pupil_median` | RAW157 | base160 | Kích thước đồng tử tuyệt đối điển hình (nhạy với ánh sáng). | Median valid uncorrected binocular pupil | pupil unit | 1, 3 |
| D017 | `fixation_duration_serial_correlation` | RAW157 | dataset100 | Mức độ thời lượng của các fixation liên tiếp giống nhau (phụ thuộc chuỗi). | Spearman rho between durations of consecutive fixations joined by a gap-free transition | correlation | 3 |
| D018 | `fixation_duration_quartile_dispersion` | RAW157 | dataset100 | Độ phân tán tương đối (không phụ thuộc thang đo) của thời lượng fixation. | (Q3 - Q1) / (Q3 + Q1) of fixation durations | unitless | 3 |
| D021 | `interfixation_interval_median` | RAW157 | dataset100 | Khoảng thời gian điển hình giữa hai fixation liên tiếp (thời gian chuyển mắt). | Median of onset(k+1) - offset(k) over gap-free transitions | s | 2, 3 |
| D022 | `fixation_onset_burstiness` | RAW157 | dataset100 | Nhịp fixation đều hay dồn cụm (burstiness). | (sd - mean)/(sd + mean) of inter-onset intervals over gap-free transitions | unitless | 3 |
| D023 | `gaze_event_span` | RAW157 | dataset100 | Thời gian từ fixation đầu tới fixation cuối của task (thời gian mắt tham gia làm task). | Offset of last valid fixation - onset of first valid fixation within the task label | s | 1, 2, 3 |
| D024 | `saccadic_time_fraction` | RAW157 | dataset100 | Tỉ lệ thời gian dành cho chuyển mắt so với nhìn cố định. | Sum of gap-free interfixation intervals / (that sum + durations of the fixations they join), within continuous segments | fraction | 2, 3 |
| D025 | `main_sequence_duration_slope` | RAW157 | dataset100 | Thời lượng saccade tăng thêm cho mỗi độ biên độ (độ dốc main sequence). | Theil-Sen slope of matched saccade duration (s) on amplitude (deg) | s/deg | 2, 3 |
| D026 | `amplitude_duration_coupling` | RAW157 | dataset100 | Mức độ chặt chẽ giữa thời lượng và biên độ saccade. | Spearman rho(amplitude, duration) over matched saccades | correlation | 2, 3 |
| D028 | `leftward_transition_fraction` | RAW157 | dataset100 | Tỉ lệ chuyển mắt sang trái (gồm cả quay lại và xuống dòng). | Gap-free transitions with horizontal-dominant leftward displacement (dx<0, /dx/>/dy/) / gap-free transitions with displacement >= 1 deg | fraction | 2, 3 |
| D029 | `rightward_saccade_amplitude_median` | RAW157 | dataset100 | Độ lớn điển hình của saccade tiến (sang phải). | Median amplitude of matched saccades with horizontal-dominant rightward displacement | deg | 3 |
| D030 | `leftward_saccade_amplitude_median` | RAW157 | dataset100 | Độ lớn điển hình của saccade lùi (sang trái). | Median amplitude of matched saccades with horizontal-dominant leftward displacement | deg | 3 |
| D034 | `saccade_amplitude_serial_correlation` | RAW157 | dataset100 | Mức độ độ lớn các saccade liên tiếp giống nhau. | Spearman rho between amplitudes of consecutive gap-free transitions | correlation | 3 |
| D035 | `fixation_duration_outgoing_amplitude_coupling` | RAW157 | dataset100 | Tương quan giữa thời lượng fixation và độ lớn saccade tiếp theo (lập kế hoạch). | Spearman rho(fixation duration, amplitude of the following gap-free transition) | correlation | 3 |
| D036 | `incoming_amplitude_fixation_duration_coupling` | RAW157 | dataset100 | Tương quan giữa độ lớn saccade vừa tới và thời lượng fixation sau đó. | Spearman rho(amplitude of the preceding gap-free transition, fixation duration) | correlation | 3 |
| D037 | `post_leftward_fixation_duration_log_ratio` | RAW157 | dataset100 | Thời lượng fixation sau chuyển mắt sang trái so với sang phải (chi phí quay lại). | log(median duration of fixations entered by leftward transitions / median entered by rightward) | log ratio | 3 |
| D039 | `fixation_xy_spread_log_anisotropy` | RAW157 | dataset100 | Độ trải ngang so với độ trải dọc của fixation. | log(SD of screen x / SD of screen y) over in-media fixation positions (px) | log ratio | 3 |
| D040 | `vertical_progression_monotonicity` | RAW157 | dataset100 | Mức độ mắt di chuyển đều từ trên xuống dưới theo thời gian. | Spearman rho(fixation onset time, screen y) over in-media fixations | correlation | 3 |
| D041 | `off_media_fixation_time_fraction` | RAW157 | dataset100 | Tỉ lệ thời gian fixation trên màn hình nhưng nằm ngoài hình kích thích. | Duration of fixations whose gaze lies outside the stimulus media box / duration of all located fixations | fraction | 3 |
| D047 | `viewing_distance_iqr` | RAW157 | dataset100 | Độ dao động khoảng cách mắt–màn hình (độ ổn định tư thế đầu). | IQR of mean(DistanceLeft, DistanceRight) over fixation samples with both eyes valid | mm | 2, 3 |
| D007 | `target_mean_fixation_count` | ROI64_T1235 | dataset100 | Số fixation trung bình trên mỗi target (chữ số/từ). | Equal-ROI mean of M001 over observed target ROIs | count/target | 1, 2, 3 |
| D010 | `target_pupil_change_median` | ROI64_T1235 | dataset100 | Thay đổi đồng tử điển hình từ fixation đầu trên target tới fixation hạ cánh sau khi rời target. | Equal-ROI median of D001 | mm | 2, 3 |
| D011 | `target_left_pupil_range_median` | ROI64_T1235 | dataset100 | Biên độ dao động đồng tử trái điển hình trong một target. | Equal-ROI median of D002 | mm | 2, 3 |
| D015 | `articulation_fixation_count_coupling` | ROI64_T1235 | dataset100 | Tương quan trong cùng người giữa thời gian phát âm một target và số fixation trên target đó. | Within-person Spearman rho between M051 and M001 across target ROIs | correlation | 3 |

Ghi chú về nguồn và định nghĩa:
- Sự kiện fixation và saccade của RAW157 được dựng lại từ dòng onset và dòng final của export; offset = onset + thời lượng. 'Chuyển tiếp không gián đoạn' nối hai fixation liên tiếp trong cùng task mà khoảng giữa được các saccade phủ kín (sai lệch tối đa 10 ms); không bao giờ nối qua chỗ mất tín hiệu. 'Saccade khớp' là chuyển tiếp không gián đoạn có đúng một saccade, nên biên độ và thời lượng thuộc cùng một cú chuyển mắt.
- Biên độ là góc fixation-tới-fixation do Tobii tính (đã kiểm tra r = 0.99 với hình học). Hướng trái/phải lấy từ vị trí fixation trên màn hình; 'chủ yếu theo chiều ngang' nghĩa là |dx| > |dy|.
- Đồng tử trong RAW157 chỉ có tại các mẫu ở biên sự kiện, không phải chuỗi liên tục; ánh sáng không được kiểm soát.
- Các feature ROI64 (D007, D010, D011, D015) tóm tắt giá trị theo ROI của workbook trên các target của một task, yêu cầu có giá trị hợp lệ ở ít nhất 60% target dự kiến (với tương quan: ít nhất 10 target). D010 và D011 theo quy ước đồng tử của Bảng II (giá trị 'ra' là fixation hạ cánh sau khi rời target).
- Luật hỗ trợ, luật zero/missing và bằng chứng đầy đủ: `feature_registry.csv`, `DATASET_PROPOSED_100_FEATURES.md`, `semantic_audit.md`.

## 2. Giá trị p: 4 task x 3 phép so sánh

**In đậm** = p < 0.05; *in nghiêng* = 0.05 <= p < 0.10; NA = không kiểm định được (ít hơn 8 người hợp lệ hoặc dưới 60% coverage trong một nhóm). p thô; cỡ mẫu xem mục 3.

| ID | Feature | KD1 CN vs MCI | KD1 CN vs AD | KD1 MCI vs AD | KD2 CN vs MCI | KD2 CN vs AD | KD2 MCI vs AD | KD3 CN vs MCI | KD3 CN vs AD | KD3 MCI vs AD | READING CN vs MCI | READING CN vs AD | READING MCI vs AD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A002 | total_fixation_time | *0.090* | 0.376 | 0.811 | **0.002** | *0.069* | 0.866 | **8.7e-04** | **9.5e-05** | *0.077* | 0.908 | 0.412 | 0.483 |
| A003 | task_fixation_count | **0.010** | *0.061* | 0.817 | **4.5e-05** | **0.027** | 0.972 | **0.002** | **1.8e-04** | **0.033** | 0.661 | 0.261 | 0.419 |
| A021 | task_saccade_count | 0.107 | 0.182 | 0.852 | **0.015** | *0.056* | 0.824 | *0.058* | **0.004** | 0.155 | **0.023** | 0.300 | 0.534 |
| A023 | amplitude_median | **0.019** | 0.262 | 0.600 | **0.022** | 0.124 | 0.986 | 0.101 | **0.046** | 0.360 | 0.116 | *0.074* | 0.763 |
| A024 | amplitude_iqr | 0.497 | 0.347 | 0.880 | 0.459 | 0.813 | 0.894 | 0.583 | **0.008** | **0.003** | 0.280 | 0.870 | 0.644 |
| A026 | saccade_duration_median | 0.557 | 0.374 | 0.669 | 0.797 | 0.445 | 0.443 | 0.279 | *0.092* | **0.048** | 0.857 | 0.913 | 0.828 |
| A027 | saccade_duration_iqr | 0.201 | 0.506 | 0.769 | *0.098* | 0.319 | *0.061* | **0.027** | **0.040** | **0.003** | 0.146 | 0.173 | 0.599 |
| A028 | saccade_mean_speed_median | **0.009** | 0.511 | 0.245 | **0.002** | 0.611 | 0.324 | **0.045** | 0.216 | 0.811 | *0.072* | 0.147 | 0.922 |
| A029 | task_tiny_saccade_fraction | **0.003** | 0.145 | 0.683 | 0.181 | 0.792 | 0.564 | 0.780 | 0.816 | 0.670 | 0.260 | 0.200 | 0.638 |
| A030 | task_long_saccade_fraction | **0.014** | 0.102 | 0.922 | *0.100* | 0.246 | 0.965 | 0.290 | 0.740 | 0.494 | 0.108 | *0.099* | 0.887 |
| A031 | angular_scanpath_length | 0.829 | 0.484 | 0.450 | **0.009** | 0.257 | 0.742 | 0.532 | **0.038** | 0.158 | *0.096* | 0.782 | 0.419 |
| A033 | normalized_screen_dispersion | 0.524 | 0.459 | 0.908 | 0.372 | 0.335 | 0.252 | 0.955 | 0.908 | 0.950 | **0.010** | 0.202 | 0.517 |
| A035 | task_spatial_entropy | 0.819 | 0.583 | 0.663 | 0.200 | 0.112 | *0.093* | 0.356 | 0.746 | 0.450 | **0.024** | 0.108 | 0.908 |
| A046 | task_absolute_pupil_median | *0.053* | 0.135 | 0.763 | *0.055* | 0.101 | 0.600 | **0.033** | *0.084* | 0.689 | **0.028** | **0.036** | 0.467 |
| D017 | fixation_duration_serial_correlation | **0.018** | 0.272 | 0.839 | **0.042** | 0.671 | 0.409 | *0.078* | 0.372 | 0.663 | *0.085* | 0.843 | 0.333 |
| D018 | fixation_duration_quartile_dispersion | 0.259 | 0.844 | 0.419 | 0.840 | 0.407 | 0.316 | 0.327 | 0.254 | 0.908 | **0.041** | 0.331 | 0.880 |
| D021 | interfixation_interval_median | 0.637 | **0.026** | **0.019** | 0.350 | 0.183 | *0.080* | 0.453 | **0.026** | **0.022** | 0.826 | 0.464 | 0.521 |
| D022 | fixation_onset_burstiness | 0.346 | 0.841 | 0.395 | 0.186 | 0.381 | 0.124 | **0.005** | 0.158 | 0.676 | 0.213 | 0.318 | 0.651 |
| D023 | gaze_event_span | *0.060* | **0.020** | 0.316 | **8.2e-04** | **0.002** | 0.419 | **2.5e-04** | **1.4e-04** | **0.040** | **0.016** | **0.008** | 0.245 |
| D024 | saccadic_time_fraction | 0.858 | *0.059* | *0.080* | 0.529 | 0.153 | 0.138 | 0.613 | **0.016** | *0.090* | 0.532 | 0.231 | 0.129 |
| D025 | main_sequence_duration_slope | **6.6e-04** | 0.119 | 0.494 | **0.008** | 0.600 | *0.069* | **0.005** | 0.484 | 0.307 | **0.003** | 0.335 | 0.252 |
| D026 | amplitude_duration_coupling | **0.046** | 0.199 | 0.941 | 0.489 | 0.241 | 0.540 | 0.390 | 0.204 | 0.540 | 0.275 | 0.562 | *0.090* |
| D028 | leftward_transition_fraction | 0.876 | 0.639 | 0.638 | **0.047** | 0.540 | 0.163 | 0.915 | 0.313 | 0.389 | 0.584 | 0.133 | *0.060* |
| D029 | rightward_saccade_amplitude_median | *0.075* | 0.128 | 0.894 | 0.505 | 0.645 | 0.929 | **0.039** | **0.002** | 0.169 | 0.127 | 0.329 | 0.908 |
| D030 | leftward_saccade_amplitude_median | 0.364 | 0.149 | 0.698 | 0.159 | 0.255 | 0.724 | *0.059* | **0.019** | 0.347 | 0.693 | 0.904 | 0.670 |
| D034 | saccade_amplitude_serial_correlation | 0.644 | 0.300 | 0.177 | 0.455 | 0.416 | 0.741 | 0.560 | 0.427 | 0.167 | **0.003** | 0.102 | 0.613 |
| D035 | fixation_duration_outgoing_amplitude_coupling | *0.093* | 0.712 | 0.229 | 0.690 | 0.660 | 0.894 | 0.597 | **0.028** | 0.108 | 0.365 | 0.756 | 0.440 |
| D036 | incoming_amplitude_fixation_duration_coupling | **0.037** | 0.603 | 0.579 | 0.125 | 0.881 | 0.389 | **1.2e-04** | **0.008** | 0.729 | 0.819 | 0.875 | 0.894 |
| D037 | post_leftward_fixation_duration_log_ratio | 0.987 | 0.616 | 0.877 | 0.506 | 0.774 | 0.644 | 0.631 | 0.363 | 0.257 | 0.530 | **0.022** | 0.133 |
| D039 | fixation_xy_spread_log_anisotropy | 0.515 | 0.180 | 0.588 | 0.937 | 0.251 | 0.360 | 0.911 | *0.098* | 0.169 | **0.006** | 0.199 | 0.351 |
| D040 | vertical_progression_monotonicity | 0.739 | 0.430 | 0.186 | *0.063* | **0.004** | 0.259 | 0.246 | 0.244 | 0.703 | 0.119 | 0.421 | 0.564 |
| D041 | off_media_fixation_time_fraction | 0.280 | 0.424 | 0.167 | 0.277 | 0.756 | 0.398 | **0.017** | 0.365 | 0.493 | 0.178 | 0.343 | 0.943 |
| D047 | viewing_distance_iqr | 0.998 | 0.153 | 0.153 | 0.401 | **2.7e-04** | **0.003** | 0.521 | **0.001** | **7.2e-04** | 0.214 | **0.003** | *0.087* |
| D007 | target_mean_fixation_count | **0.005** | **0.015** | 0.806 | **0.011** | *0.062* | 0.566 | 0.520 | **0.003** | **0.017** | NA | NA | 0.277 |
| D010 | target_pupil_change_median | 0.462 | *0.055* | 0.260 | 0.100 | 0.403 | **0.049** | **0.037** | **0.021** | 1.000 | NA | NA | NA |
| D011 | target_left_pupil_range_median | *0.062* | **0.020** | 0.509 | 0.323 | 0.101 | 0.409 | 0.969 | **0.011** | **0.026** | NA | NA | *0.073* |
| D015 | articulation_fixation_count_coupling | 0.958 | **0.035** | 0.104 | 0.387 | 0.303 | 0.859 | 0.494 | **0.046** | 0.199 | 0.636 | 0.262 | 0.442 |

## 3. Số người hợp lệ theo task (CN/MCI/AD)

| ID | KD1 | KD2 | KD3 | READING |
|---|---|---|---|---|
| A002 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A003 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A021 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A023 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A024 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A026 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A027 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A028 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A029 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A030 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A031 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A033 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A035 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| A046 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D017 | 99/41/15 | 99/41/16 | 100/41/16 | 99/41/16 |
| D018 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D021 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D022 | 99/41/15 | 99/41/16 | 100/41/16 | 99/41/16 |
| D023 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D024 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D025 | 100/41/15 | 100/41/16 | 100/41/16 | 100/41/16 |
| D026 | 100/41/15 | 100/41/16 | 100/41/16 | 100/41/16 |
| D028 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D029 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D030 | 98/41/15 | 99/40/16 | 99/41/16 | 100/41/16 |
| D034 | 98/40/15 | 99/40/15 | 97/39/16 | 99/41/16 |
| D035 | 99/41/15 | 99/41/16 | 100/41/16 | 99/41/16 |
| D036 | 99/41/15 | 99/41/16 | 100/41/16 | 99/41/16 |
| D037 | 99/38/14 | 99/40/16 | 96/40/16 | 99/41/16 |
| D039 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D040 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D041 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D047 | 100/41/16 | 100/41/16 | 100/41/16 | 100/41/16 |
| D007 | 31/17/15 | 30/16/15 | 29/15/13 | 19/12/10 |
| D010 | 31/17/13 | 29/16/13 | 28/15/13 | 18/12/7 |
| D011 | 31/17/15 | 29/16/13 | 29/15/13 | 19/12/9 |
| D015 | 32/17/15 | 31/16/15 | 31/16/15 | 30/16/14 |

## 4. Lưu ý

- p chưa hiệu chỉnh. Trong các họ đa kiểm định đã đăng ký trước, chỉ D023@KD3, A002@KD3, A003@KD3 và D023@KD2 đạt joint BH q <= 0.10 (không cái nào còn lại với BY).
- D023, A002, A003 và A021 tương quan mạnh (đọc thẻ lâu hơn = nhiều fixation và saccade hơn); chúng không phải các bằng chứng độc lập.
- Ngưỡng chọn được đặt sau khi đã xem kết quả; mọi ứng dụng dự đoán phải lặp lại việc chọn feature bên trong dữ liệu train hoặc dùng cohort độc lập.
- Nhóm AD nhỏ (15-16 người) và không thể hiệu chỉnh nhân khẩu học (không có dữ liệu tuổi/giới/học vấn).
