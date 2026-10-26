# Baseline bộ feature IEEE — phân loại nhóm BEGAD

Chạy thử bộ 20 feature của bài IEEE về AD/MCI đa phương thức (Table II) ánh
xạ lên BEGAD, theo đúng protocol của balanced baseline (`baseline_balanced.md`):
layout rộng theo participant (feature x 11 task, feature-major), 3 bài toán,
6 model, 20 seed cố định, CV 5-fold theo participant phân tầng theo nhóm,
median impute + `StandardScaler` theo fold trên riêng split train, class
weights/priors cân bằng, ngưỡng cố định 0.5. Cùng lịch seed với
`baseline_balanced` -> fold khớp nhau theo từng seed.

## 1. Ánh xạ feature: 16/20 feature khả dụng

| # | Feature IEEE | Ánh xạ lên BEGAD | Trạng thái |
|---|---|---|---|
| 1 | ROI | AOI từng từ/số của task đọc | **không có** (BEGAD không có chú giải AOI) |
| 2 | Number of fixations | `vis_n_fix` = số fixation sạch của task | có |
| 3 | Number of saccades | `vis_n_sac` = số fixation có saccade gắn kèm | có |
| 4 | Number of short saccades (< 1 deg) | `vis_n_sac_short` | có |
| 5 | Number of long saccades (> 2 deg) | `vis_n_sac_long` | có |
| 6 | Number of regressive saccades | `vis_n_sac_reg` (góc quay > 90 deg) | có |
| 7 | Saccade amplitude sum | `vis_sac_amp_sum` (deg) | có |
| 8 | Total fixation time | `vis_fix_time_total` (ms) | có |
| 9 | Left pupil at first fixation | `pup_l_first` (trừ baseline) | có |
| 10 | Left pupil at last fixation | `pup_l_last` | có |
| 11 | Left pupil difference | `pup_l_diff` = cuối - đầu | có |
| 12 | Right pupil at first fixation | `pup_r_first` | có |
| 13 | Right pupil at last fixation | `pup_r_last` | có |
| 14 | Right pupil difference | `pup_r_diff` | có |
| 15 | Minimum left pupil size | `pup_l_min` | có |
| 16 | Maximum left pupil size | `pup_l_max` | có |
| 17 | Min-max difference | `pup_l_minmax_diff` | có |
| 18 | Total spoken time | - | **không có** (BEGAD không có audio) |
| 19 | Delay first fixation -> utterance onset | - | **không có** |
| 20 | Delay last fixation -> utterance end | - | **không có** |

Quy ước theo pipeline của dự án (`rawdata.clean_fixations`): amplitude/hướng
saccade gắn với fixation sạch mà nó đáp xuống, amplitude tính bằng độ,
regressive = góc quay `min(reldir, 360 - reldir) > 90`. Feature pupil được trừ
baseline theo đúng bài báo (giá trị `PupilLeftBaseline` / `PupilRightBaseline`
theo participant); 16/157 participant thiếu baseline một hoặc cả hai mắt nên
feature pupil tuyệt đối của mắt đó là NaN và được median-impute theo fold
(xử lý giống các feature `pupbl` của baseline chính). Các hiệu số
(`pup_l_diff`, `pup_r_diff`, `pup_l_minmax_diff`) bất biến với baseline nên
luôn tính được.

Bảng feature: `processed_dataset/ieee_features.csv` (1.727 dòng = 157
participant x 11 task, 1.020 ô thiếu trên 27.632). Code:
`src/begad_norm/features/ieee.py` + `scripts/build_ieee_features.py`. Kết quả:
`outputs/baseline_ieee/` (1.800 lượt đánh giá).

## 2. Kết quả — mean ± std trên 100 lượt (20 seed x 5 fold)

### co_vs_mci (positive = MCI)

| Model | IEEE AUC | balanced AUC | delta | IEEE acc | bal acc | IEEE sen/spec | bal sen/spec |
|---|---|---|---|---|---|---|---|
| fnn | 0.564±0.112 | 0.695±0.101 | -0.132 | 0.650 | 0.718 | 0.305/0.791 | 0.457/0.824 |
| gnb | 0.548±0.108 | 0.643±0.103 | -0.095 | 0.555 | 0.630 | 0.480/0.586 | 0.591/0.645 |
| logistic_regression | 0.602±0.114 | 0.636±0.087 | -0.034 | 0.633 | 0.657 | 0.389/0.733 | 0.447/0.743 |
| qda | 0.431±0.097 | 0.553±0.094 | -0.123 | 0.613 | 0.696 | 0.093/0.826 | 0.093/0.944 |
| svm_linear | 0.581±0.127 | 0.598±0.097 | -0.017 | 0.632 | 0.645 | 0.371/0.740 | 0.407/0.742 |
| svm_rbf | 0.500±0.115 | 0.738±0.094 | -0.238 | 0.631 | 0.728 | 0.286/0.773 | 0.458/0.839 |

### co_vs_ad (positive = AD)

| Model | IEEE AUC | balanced AUC | delta | IEEE acc | bal acc | IEEE sen/spec | bal sen/spec |
|---|---|---|---|---|---|---|---|
| fnn | 0.726±0.156 | 0.714±0.156 | +0.012 | 0.850 | 0.857 | 0.321/0.935 | 0.307/0.945 |
| gnb | 0.695±0.170 | 0.797±0.159 | -0.101 | 0.854 | 0.847 | 0.418/0.924 | 0.598/0.887 |
| logistic_regression | 0.754±0.145 | 0.857±0.100 | -0.103 | 0.839 | 0.859 | 0.393/0.911 | 0.376/0.936 |
| qda | 0.495±0.196 | 0.594±0.223 | -0.099 | 0.861 | 0.861 | 0.000/0.999 | 0.000/0.999 |
| svm_linear | 0.736±0.152 | 0.864±0.099 | -0.128 | 0.836 | 0.869 | 0.343/0.916 | 0.348/0.952 |
| svm_rbf | 0.746±0.148 | 0.843±0.090 | -0.096 | 0.866 | 0.862 | 0.101/0.989 | 0.056/0.991 |

### co_mci_ad (3 class, metric macro)

| Model | IEEE AUC | balanced AUC | delta | IEEE acc | bal acc | IEEE sen/spec | bal sen/spec |
|---|---|---|---|---|---|---|---|
| fnn | 0.604±0.086 | 0.701±0.086 | -0.097 | 0.576 | 0.647 | 0.397/0.715 | 0.475/0.765 |
| gnb | 0.566±0.095 | 0.657±0.089 | -0.092 | 0.475 | 0.568 | 0.387/0.693 | 0.470/0.756 |
| logistic_regression | 0.631±0.081 | 0.698±0.072 | -0.067 | 0.574 | 0.591 | 0.438/0.725 | 0.452/0.741 |
| qda | 0.455±0.106 | 0.564±0.107 | -0.109 | 0.552 | 0.620 | 0.309/0.650 | 0.341/0.674 |
| svm_linear | 0.629±0.082 | 0.691±0.076 | -0.062 | 0.568 | 0.574 | 0.424/0.711 | 0.427/0.721 |
| svm_rbf | 0.604±0.091 | 0.758±0.071 | -0.155 | 0.562 | 0.648 | 0.374/0.707 | 0.459/0.776 |

## 3. Kết quả chính

1. **Bộ IEEE thua balanced baseline ở 17/18 ô.** Ô duy nhất có delta AUC
   dương là fnn trên co_vs_ad (+0.012, nằm trong nhiễu seed, std 0.156). Tụt
   nhiều nhất: svm_rbf trên co_vs_mci (-0.238, 0.500 so với 0.738) và svm_rbf
   trên co_mci_ad (-0.155). Trung vị delta AUC trên 18 ô: -0.099.
2. **So sánh này không công bằng chỉ theo một chiều.** 25 feature của
   balanced baseline được chọn theo nhãn (Kruskal-Wallis p < 0.1 ở >= 3 task,
   screening trên toàn bộ subject), làm ước lượng CV của nó lạc quan. Bộ IEEE
   được quy định trước (không selection bias) và ít feature hơn (16 so với
   25, 176 so với 275 cột wide). Cần nhớ cả hai chiều: một phần delta phản
   ánh việc chọn feature theo nhãn, và bộ IEEE thiếu modal mạnh nhất của nó.
3. **Bốn feature IEEE không ánh xạ được.** ROI (AOI từng từ của task đọc) và
   cả 3 feature speech không có dữ liệu tương ứng trong BEGAD (không có chú
   giải AOI, không có audio). Fusion network của bài báo đặt speech ngang
   hàng với eye movements; trên BEGAD bộ IEEE thực chất chỉ còn eye movements,
   điều này giải thích hợp lý một phần khoảng cách.
4. **Tính dị biệt giữa các task bị bỏ qua ở đây, giống baseline chính.** So
   sánh mức patch cho thấy gộp 11 task làm loãng tín hiệu riêng task; caveat
   đó áp dụng cho cả hai run trong so sánh này nên không làm đổi kết luận
   tương đối.

## 4. Lưu ý (caveat)

- So sánh khớp cặp: cùng bài toán, model, seed, fold (phân chia round-robin
  seed 42), imputation và ngưỡng như `baseline_balanced`.
- 16/157 participant thiếu baseline pupil một hoặc hai mắt: feature pupil
  tuyệt đối của họ là NaN và được median-impute theo fold (tổng 1.020/27.632
  ô thiếu).
- Bộ feature của run đối chứng được chọn theo nhãn (CV lạc quan); bộ IEEE
  được quy định trước (không bias nhưng thiếu ROI + speech).
- QDA trên co_vs_ad suy biến ở cả hai run (sensitivity 0.0), nên cột AUC của
  nó ít ý nghĩa.

## 5. Bước tiếp theo

- Chạy bộ IEEE theo từng task (kết quả mức patch cho thấy task đơn vượt run
  gộp; bộ IEEE có thể cũng vậy).
- So sánh công bằng: shortlist 16 feature chọn theo nhãn vs bộ IEEE quy định
  trước, cùng CV.
- Chỉ thêm proxy cho ROI (độ phủ vùng stimulus) sau khi có định nghĩa AOI có
  nguyên tắc cho stimulus của BEGAD.
