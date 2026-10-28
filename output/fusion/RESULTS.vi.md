# Học riêng từng task + kết hợp xác suất muộn (bộ dữ liệu 33 feature x 4 task)

Chạy `2026-10-28T21:29Z`. Chi tiết: `results/results_detail.csv` (192 lượt đánh giá),
tổng hợp: `results/summary.csv`, trọng số fusion: `results/fusion_weights.csv`, delta
ghép cặp: `results/paired_deltas.csv`, xác suất từng người:
`results/proba_<problem>_<model_type>.csv`, provenance: `provenance.json`.

## 1. Thiết lập

- **Dữ liệu**: `processed_dataset/processed_dataset_33feat_wide.csv` — 157 người
  (CN 100 / MCI 41 / AD 16), 132 feature (33 feature RAW157 x 4 task:
  KD1/KD2/KD3/READING). 51 ô khuyết được impute theo fold bằng trung vị cột, chỉ fit
  trên tập train (SimpleImputer trước StandardScaler, cả hai đều fit trên dữ liệu
  train ở mọi tầng CV).
- **Split (cố định)**: giống hệt các baseline, từ `processed_dataset/splits/` — 4
  người AD held-out (BEGAD-001, BEGAD-005, BEGAD-006, BEGAD-016) được thêm vào MỌI
  fold validation và không bao giờ vào train; 153 người còn lại tạo 4 fold phân tầng
  (seed 42). Train mỗi fold: 114-115 người (bài toán 3 lớp) với 9 AD; validation:
  42-43 người với 7 AD (3 từ fold + 4 held-out).
- **Model từng task**: mỗi task KD1, KD2, KD3, READING có model riêng trên 33
  feature của nó, với hai họ model nền cùng ngân sách tuning (chọn theo log-loss CV
  trung bình — một proper scoring rule phù hợp cho fusion xác suất):
  - `elasticnet`: LogisticRegression(penalty=elasticnet, saga, max_iter=2000,
    tol=1e-4, random_state=0 được pin — saga dùng random_state để trộn thứ tự dữ
    liệu, không pin thì mỗi lần fit ra kết quả khác nhau). Grid: C x l1_ratio =
    4 x 3 = 12 tổ hợp.
  - `svm_rbf`: SVC(rbf) với calibration sigmoid qua CalibratedClassifierCV với số
    fold calibration phân tầng thích ứng `min(5, số mẫu lớp nhỏ nhất)` (cv=5 cố
    định sẽ lỗi khi inner-inner split có ít hơn 5 mẫu của lớp nhỏ nhất). Grid:
    C x gamma = 3 x 3 = 9 tổ hợp với gamma = k / n_features (feature đã chuẩn hóa
    nên 'scale' của sklearn bằng 1/n_features — giúp ngân sách so sánh được giữa
    các số lượng feature khác nhau).
- **Model không dùng class weight** (class_weight=None) một cách có chủ đích: fusion
  kết hợp xác suất thô của các model và balanced weight sẽ làm lệch thang xác suất
  theo từng bài toán. Baseline (output/baseline_33feat) dùng balanced weight với
  C=1.0 cố định, không tuning; so sánh với baseline chỉ mang tính mô tả.
- **Các biến thể** cho mỗi (bài toán, model, fold):
  - `task_kd1`..`task_reading`: model từng task riêng lẻ (A).
  - `fusion_equal4`: trung bình xác suất của 4 task (B).
  - `fusion_learned`: một vector trọng số task dùng chung cho mọi lớp, học bằng cách
    cực tiểu hóa log-loss trung bình + lambda * KL(w || phân phối đều) trên simplex
    (SLSQP, 13 điểm khởi tạo), lambda = 0.1 (C). Trọng số chỉ học từ dự đoán inner
    OOF của fold train; dữ liệu validation không bao giờ được dùng.
  - `fusion_equal3kd`: trung bình xác suất KD1/KD2/KD3, bỏ READING (D).
  - `concat132`: nối toàn bộ 132 feature vào một model với đúng protocol tuning
    lồng nhau (tham chiếu cho giả thuyết overfitting).
- **Protocol lồng nhau**: xác suất validation đến từ model có hyperparameter được
  chọn bằng inner CV (4 fold, seed 42) trên toàn fold train, rồi refit trên fold đó.
  Dự đoán inner OOF dùng cho trọng số fusion và ngưỡng Youden đến từ model của từng
  inner fold mà việc fit, tuning (inner-inner CV, 4 fold, seed 7) và calibration
  không bao giờ dùng người được dự đoán.
- **Ngưỡng**: metric binary chính ở ngưỡng cố định 0.5; các cột phụ báo ngưỡng
  Youden-J suy ra từ xác suất inner OOF ở mức train và metric validation dưới ngưỡng
  đó. Bài toán 3 lớp dùng argmax.
- **Tính tất định**: mọi thành phần (splitter, saga, SVC, calibration sigmoid,
  SLSQP) đều tất định với một thứ tự dữ liệu cố định, nên mỗi (bài toán, model,
  fold) được đánh giá đúng một lần; 4 fold là đơn vị đánh giá.

## 2. Kết quả tổng hợp (mean ± std trên 4 fold)

### 2.1 3 lớp CN/MCI/AD

| Variant | EN acc | EN macroF1 | EN bacc | EN OVR AUC | EN rec AD | SVM acc | SVM macroF1 | SVM bacc | SVM OVR AUC | SVM rec AD |
|---|---|---|---|---|---|---|---|---|---|---|
| task_kd1 | 0.574 ± 0.030 | 0.245 ± 0.005 | 0.323 ± 0.017 | 0.569 ± 0.070 | 0.000 ± 0.000 | 0.580 ± 0.010 | 0.246 ± 0.003 | 0.327 ± 0.007 | 0.556 ± 0.038 | 0.000 ± 0.000 |
| task_kd2 | 0.604 ± 0.041 | 0.328 ± 0.021 | 0.373 ± 0.016 | 0.711 ± 0.041 | 0.000 ± 0.000 | 0.598 ± 0.025 | 0.324 ± 0.011 | 0.369 ± 0.003 | 0.649 ± 0.068 | 0.000 ± 0.000 |
| task_kd3 | 0.592 ± 0.040 | 0.332 ± 0.043 | 0.371 ± 0.037 | 0.717 ± 0.038 | 0.036 ± 0.062 | 0.574 ± 0.020 | 0.307 ± 0.047 | 0.351 ± 0.029 | 0.686 ± 0.053 | 0.036 ± 0.062 |
| task_reading | 0.592 ± 0.006 | 0.283 ± 0.038 | 0.348 ± 0.017 | 0.636 ± 0.013 | 0.000 ± 0.000 | 0.598 ± 0.004 | 0.289 ± 0.023 | 0.351 ± 0.011 | 0.600 ± 0.046 | 0.000 ± 0.000 |
| fusion_equal4 | 0.592 ± 0.029 | 0.275 ± 0.047 | 0.342 ± 0.032 | 0.727 ± 0.046 | 0.000 ± 0.000 | 0.598 ± 0.004 | 0.289 ± 0.023 | 0.351 ± 0.011 | 0.683 ± 0.049 | 0.000 ± 0.000 |
| fusion_learned | 0.592 ± 0.029 | 0.274 ± 0.045 | 0.342 ± 0.032 | 0.724 ± 0.047 | 0.000 ± 0.000 | 0.598 ± 0.004 | 0.289 ± 0.023 | 0.351 ± 0.011 | 0.682 ± 0.048 | 0.000 ± 0.000 |
| fusion_equal3kd | 0.580 ± 0.033 | 0.272 ± 0.043 | 0.335 ± 0.031 | 0.726 ± 0.051 | 0.000 ± 0.000 | 0.592 ± 0.012 | 0.287 ± 0.022 | 0.348 ± 0.012 | 0.677 ± 0.043 | 0.000 ± 0.000 |
| concat132 | 0.633 ± 0.014 | 0.390 ± 0.058 | 0.422 ± 0.039 | 0.737 ± 0.037 | 0.036 ± 0.062 | 0.615 ± 0.036 | 0.349 ± 0.061 | 0.394 ± 0.048 | 0.707 ± 0.071 | 0.000 ± 0.000 |

### 2.2 CN vs MCI (lớp dương = MCI)

| Variant | EN AUC | EN AP | EN F1 | EN bacc | EN sen | EN spec | SVM AUC | SVM AP | SVM F1 | SVM bacc | SVM sen | SVM spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| task_kd1 | 0.630 ± 0.105 | 0.451 ± 0.119 | 0.000 ± 0.000 | 0.485 ± 0.026 | 0.000 ± 0.000 | 0.970 ± 0.052 | 0.546 ± 0.095 | 0.352 ± 0.069 | 0.000 ± 0.000 | 0.490 ± 0.010 | 0.000 ± 0.000 | 0.980 ± 0.020 |
| task_kd2 | 0.702 ± 0.083 | 0.582 ± 0.050 | 0.186 ± 0.135 | 0.540 ± 0.024 | 0.141 ± 0.135 | 0.940 ± 0.104 | 0.696 ± 0.056 | 0.558 ± 0.035 | 0.247 ± 0.074 | 0.554 ± 0.018 | 0.168 ± 0.073 | 0.940 ± 0.060 |
| task_kd3 | 0.681 ± 0.094 | 0.496 ± 0.113 | 0.117 ± 0.123 | 0.508 ± 0.055 | 0.075 ± 0.083 | 0.940 ± 0.066 | 0.684 ± 0.093 | 0.474 ± 0.104 | 0.234 ± 0.146 | 0.549 ± 0.045 | 0.168 ± 0.102 | 0.930 ± 0.059 |
| task_reading | 0.654 ± 0.063 | 0.435 ± 0.033 | 0.167 ± 0.173 | 0.538 ± 0.045 | 0.125 ± 0.130 | 0.950 ± 0.052 | 0.676 ± 0.051 | 0.519 ± 0.018 | 0.119 ± 0.129 | 0.531 ± 0.034 | 0.073 ± 0.082 | 0.990 ± 0.017 |
| fusion_equal4 | 0.688 ± 0.086 | 0.584 ± 0.109 | 0.083 ± 0.083 | 0.519 ± 0.020 | 0.048 ± 0.048 | 0.990 ± 0.017 | 0.719 ± 0.061 | 0.572 ± 0.040 | 0.087 ± 0.087 | 0.524 ± 0.024 | 0.048 ± 0.048 | 1.000 ± 0.000 |
| fusion_learned | 0.681 ± 0.086 | 0.569 ± 0.125 | 0.080 ± 0.080 | 0.514 ± 0.019 | 0.048 ± 0.048 | 0.980 ± 0.035 | 0.715 ± 0.058 | 0.563 ± 0.023 | 0.083 ± 0.083 | 0.514 ± 0.026 | 0.048 ± 0.048 | 0.980 ± 0.020 |
| fusion_equal3kd | 0.673 ± 0.069 | 0.560 ± 0.116 | 0.107 ± 0.186 | 0.534 ± 0.059 | 0.068 ± 0.118 | 1.000 ± 0.000 | 0.712 ± 0.067 | 0.548 ± 0.053 | 0.083 ± 0.083 | 0.519 ± 0.020 | 0.048 ± 0.048 | 0.990 ± 0.017 |
| concat132 | 0.698 ± 0.065 | 0.555 ± 0.093 | 0.407 ± 0.139 | 0.620 ± 0.051 | 0.341 ± 0.148 | 0.900 ± 0.060 | 0.751 ± 0.038 | 0.570 ± 0.048 | 0.369 ± 0.118 | 0.601 ± 0.041 | 0.291 ± 0.116 | 0.910 ± 0.059 |

### 2.3 CN vs AD (lớp dương = AD)

| Variant | EN AUC | EN AP | EN F1 | EN bacc | EN sen | EN spec | SVM AUC | SVM AP | SVM F1 | SVM bacc | SVM sen | SVM spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| task_kd1 | 0.519 ± 0.089 | 0.244 ± 0.036 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.590 ± 0.105 | 0.353 ± 0.119 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| task_kd2 | 0.806 ± 0.078 | 0.661 ± 0.058 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 | 0.793 ± 0.045 | 0.620 ± 0.078 | 0.211 ± 0.212 | 0.567 ± 0.067 | 0.143 ± 0.143 | 0.990 ± 0.017 |
| task_kd3 | 0.851 ± 0.079 | 0.657 ± 0.143 | 0.118 ± 0.119 | 0.531 ± 0.032 | 0.071 ± 0.071 | 0.990 ± 0.017 | 0.801 ± 0.079 | 0.604 ± 0.130 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 |
| task_reading | 0.736 ± 0.063 | 0.536 ± 0.061 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.700 ± 0.088 | 0.433 ± 0.080 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| fusion_equal4 | 0.851 ± 0.061 | 0.721 ± 0.091 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.833 ± 0.058 | 0.642 ± 0.103 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| fusion_learned | 0.843 ± 0.061 | 0.700 ± 0.090 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 | 0.827 ± 0.057 | 0.635 ± 0.106 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| fusion_equal3kd | 0.839 ± 0.067 | 0.681 ± 0.083 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 | 0.826 ± 0.044 | 0.616 ± 0.105 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| concat132 | 0.881 ± 0.051 | 0.741 ± 0.089 | 0.417 ± 0.177 | 0.654 ± 0.093 | 0.357 ± 0.237 | 0.950 ± 0.052 | 0.717 ± 0.071 | 0.506 ± 0.078 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |

Các metric phụ ở ngưỡng Youden nằm trong detail CSV; điểm đáng chú ý nhất: ở CN vs
MCI, các biến thể fusion tại ngưỡng Youden đạt sensitivity 0.58-0.73 (F1 0.48-0.55)
so với sensitivity 0.05 (F1 0.08) tại ngưỡng cố định 0.5 — xác suất fusion xếp hạng
đúng người MCI nhưng bị lệch về lớp đa số, nên ngưỡng suy ra từ xác suất mức train
khôi phục lại phần lớn giá trị phân biệt.

## 3. Trọng số fusion (học, lambda = 0.1)

Mean ± std trên 4 fold; inner OOF cross-entropy của fusion học được so với fusion
trọng số đều cho thấy việc học được gì trên dữ liệu train.

| Bài toán / model | w_kd1 | w_kd2 | w_kd3 | w_reading | inner CE learned | inner CE equal |
|---|---|---|---|---|---|---|
| 3 lớp elasticnet | 0.215 ± 0.031 | 0.248 ± 0.047 | 0.312 ± 0.017 | 0.225 ± 0.013 | 0.806 | 0.810 |
| 3 lớp svm_rbf | 0.242 ± 0.028 | 0.281 ± 0.032 | 0.253 ± 0.018 | 0.223 ± 0.022 | 0.816 | 0.818 |
| CN vs MCI elasticnet | 0.229 ± 0.019 | 0.261 ± 0.059 | 0.282 ± 0.037 | 0.227 ± 0.016 | 0.584 | 0.587 |
| CN vs MCI svm_rbf | 0.255 ± 0.047 | 0.251 ± 0.042 | 0.284 ± 0.046 | 0.211 ± 0.028 | 0.576 | 0.580 |
| CN vs AD elasticnet | 0.215 ± 0.022 | 0.249 ± 0.021 | 0.288 ± 0.049 | 0.248 ± 0.040 | 0.334 | 0.337 |
| CN vs AD svm_rbf | 0.248 ± 0.015 | 0.290 ± 0.033 | 0.253 ± 0.025 | 0.208 ± 0.024 | 0.339 | 0.342 |

Đọc cùng inner log-loss từng task (trong `results/fusion_weights.csv`): cross-entropy
inner OOF của fusion thấp hơn từng task đơn lẻ trong cả 6 tổ hợp bài toán/model, là
bằng chứng trực tiếp rằng các task mang thông tin bổ trợ cho nhau một phần.

## 4. Delta ghép cặp từng fold (a trừ b, mean ± std trên 4 fold)

| Bài toán | Model | So sánh | Delta AUC | Dấu giữa các fold |
|---|---|---|---|---|
| 3 lớp | elasticnet | fusion_equal4 vs concat132 | -0.010 ± 0.018 | 3/4 âm |
| 3 lớp | elasticnet | fusion_learned vs fusion_equal4 | -0.003 ± 0.006 | 3/4 âm |
| 3 lớp | svm_rbf | fusion_equal4 vs concat132 | -0.024 ± 0.036 | 2/4 âm |
| CN vs MCI | elasticnet | fusion_equal4 vs concat132 | -0.010 ± 0.044 | 3/4 âm |
| CN vs MCI | elasticnet | task_kd2 vs concat132 | +0.004 ± 0.082 | trộn lẫn |
| CN vs MCI | svm_rbf | fusion_equal4 vs concat132 | -0.033 ± 0.041 | 3/4 âm |
| CN vs AD | elasticnet | fusion_equal4 vs concat132 | -0.030 ± 0.036 | 4/4 âm |
| CN vs AD | elasticnet | task_kd3 vs concat132 | -0.030 ± 0.029 | 3/4 âm |
| CN vs AD | svm_rbf | fusion_equal4 vs concat132 | +0.116 ± 0.039 | **4/4 dương** |
| CN vs AD | svm_rbf | task_kd1 vs concat132 | -0.127 ± 0.045 | 4/4 âm |
| CN vs AD | svm_rbf | fusion_learned vs fusion_equal4 | -0.006 ± 0.007 | 2/4 âm, 2/4 hòa |

## 5. Các giả thuyết

**Tách theo task có giảm overfitting so với nối 132 biến không?**
Không có lợi thế chung. Với elastic-net, concat132 là biến thể tốt nhất trên cả ba
bài toán (OVR AUC 0.737 / AUC 0.698 / 0.881); fusion đạt ngang task đơn tốt nhất
nhưng không vượt concat132 (mean delta ghép cặp âm ở 5/6 tổ hợp bài toán/model).
Ngoại lệ có ý nghĩa: với SVM-RBF trên CN vs AD, concat132 là biến thể yếu nhất trừ
KD1 (AUC 0.717) và mọi fusion đều vượt nó, ở cả 4 fold (+0.116 ± 0.039 với
fusion_equal4). Nghĩa là model riêng từng task không tự thân giảm overfitting, mà
fusion đóng vai trò regularizer cứu một model nền fit kém trên không gian 132
feature (SVM không weight vật lộn với lớp AD quá nhỏ). Lưu ý so sánh với baseline chỉ
mang tính mô tả: baseline_33feat LR (balanced, không tuning) đạt OVR AUC 0.705 ở 3
lớp và AUC 0.891 ở CN vs AD; concat132 có tuning không weight của chúng tôi đạt
0.737 và 0.881.

**Các task có cung cấp thông tin bổ trợ cho nhau không?** Có, trên dữ liệu train.
Cross-entropy inner OOF của fusion thấp hơn mọi task đơn lẻ trong cả 6 tổ hợp
(VD: 3 lớp EN: fusion 0.806 so với task tốt nhất 0.807; CN vs AD EN: fusion 0.334
so với task tốt nhất 0.353), và trọng số học được lệch hệ thống khỏi phân phối đều
(KD3 cao nhất trong 4/6 trường hợp, READING thấp nhất trong 4/6). Trên validation
fusion ngang hoặc kém nhẹ task đơn tốt nhất và concat132, nên tín hiệu bổ trợ là có
thật nhưng nhỏ so với phương sai giữa các fold.

**Trọng số học được có tốt hơn trọng số đều và có ổn định giữa các fold không?**
Trọng số học được chỉ thắng trọng số đều 0.002-0.004 CE trên inner OOF (regularization
lambda = 0.1 kéo mạnh về phân phối đều), và trên validation bằng hoặc kém nhẹ trọng
số đều ở 10/12 so sánh (VD: CN vs AD EN -0.009 AUC). Với 105-115 người train và 3
bậc tự do của vector trọng số, vector này hơi overfit inner OOF. Trọng số khá ổn
định giữa các fold (std 0.013-0.059, thứ tự KD3 cao / READING thấp phần lớn được
giữ), nhưng khuyến nghị thực tế từ dữ liệu này là dùng trọng số đều và coi trọng số
học được như công cụ mô tả mức quan trọng của từng task.

## 6. Hạn chế

- Các fold validation trùng nhau ở 4 người AD held-out (theo thiết kế của split cố
  định), nên metric CV gộp đè nặng lên 4 người này; metric từng fold được báo riêng
  trong detail CSV.
- Chỉ có 16 người AD (12 trong CV, 9 mỗi fold train) khiến các ước lượng CN-vs-AD
  và 3 lớp nhiễu; std giữa các fold là thước đo bất định trung thực ở đây.
- lambda = 0.1 là cường độ prior cố định, không tuning; trọng số được regularize về
  phân phối đều theo thiết kế nên mức cải thiện của trọng số học là thận trọng.
- Model không dùng class weight, khác các baseline balanced; balanced weight tương
  tác xấu với fusion xác suất nên được bỏ có chủ đích.
- Một số góc grid regularize yếu (C=10) chạm trần vòng lặp của saga ở vài fold; điểm
  số CV của chúng vẫn dùng được cho chọn lựa (thua xa) và hành vi này giống nhau ở
  mọi biến thể.
- Hai bước hoàn thiện sau chạy đã được áp dụng lên artifacts (xem
  `src/expcommon/postprocess_first_full_runs.py`): gộp các dòng proba của contrasts
  về layout rộng, và tính lại các ô balanced accuracy 3 lớp bị thiếu ở lần chạy đầu
  từ xác suất từng người đã lưu (quy tắc argmax), mỗi ô đều được đối chiếu với
  recall từng lớp đã lưu. Mã runner đã được sửa để ghi trực tiếp cả hai trong các
  lần chạy sau.
