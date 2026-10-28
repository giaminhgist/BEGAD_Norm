# Feature tuyệt đối + contrast xuyên task (bộ dữ liệu 33 feature x 4 task)

Chạy `2026-10-28T21:57Z`. Chi tiết: `results/results_detail.csv` (72 lượt đánh giá),
tổng hợp: `results/summary.csv`, delta ghép cặp: `results/paired_deltas.csv`,
manifest contrast: `results/contrast_manifest.csv`, xác suất từng người:
`results/proba_<problem>_<model_type>.csv`, provenance: `provenance.json`.

## 1. Thiết lập

- **Dữ liệu và split**: giống hệt experiment fusion và các baseline —
  `processed_dataset/processed_dataset_33feat_wide.csv` (157 người, 132 feature),
  split 4 fold cố định với 4 người AD held-out thêm vào mọi fold validation và không
  bao giờ vào train (xem output/fusion/RESULTS.md mục 1).
- **Contrast features (18)**, tính từ ma trận thô TRƯỚC khi impute để missingness
  lan truyền vào imputer theo fold:
  - ratio-log cho các feature thời gian/count D023, A002, A003 (đều dương chặt trong
    dữ liệu này; thành phần không dương sẽ tạo contrast missing):
    `contrast_21 = log(x_KD2/x_KD1)`, `contrast_31 = log(x_KD3/x_KD1)`.
  - difference cho các feature saccade/coupling/temporal có dấu D025, D026, D035,
    D036, D017, D022 (không log-transform feature có dấu):
    `contrast_21 = x_KD2 - x_KD1`, `contrast_31 = x_KD3 - x_KD1`.
  - Manifest (`results/contrast_manifest.csv`) ghi số ô hữu hạn, số NaN và khoảng giá
    trị từng contrast: cả 6 contrast ratio hữu hạn với toàn bộ 157 người; 12 contrast
    difference có 1-2 ô khuyết mỗi cái (từ các ô không hỗ trợ của thành phần), được
    impute theo fold trong pipeline train.
  - READING được giữ là một task riêng, không coi là mức khó thứ tư của KD; các
    contrast không được diễn giải là phản hồi tải nhận thức (chúng có thể phản ánh
    cả khác biệt bố cục và yêu cầu nhiệm vụ).
- **Các representation** (cùng outer split, inner seed và ngân sách tuning):
  - `abs132`: toàn bộ 132 feature tuyệt đối (33 x 4 task).
  - `abs150`: abs132 + 18 contrasts = 150.
  - `compact84`: 33 KD1 + 33 READING + 18 contrasts = 84.
- **Model và protocol**: elastic-net logistic regression (chính) và SVM-RBF có
  calibration (đối chứng), với đúng các grid, tuning theo log-loss, impute/scale theo
  fold, inner CV (4 fold, seed 42) và inner-inner OOF (seed 7) như experiment fusion;
  `abs132` giống hệt từng bit với `concat132` của experiment fusion. Model không
  dùng class weight; ngưỡng 0.5 cho metric binary chính, Youden-J từ inner OOF là cột
  phụ, argmax cho 3 lớp.

## 2. Kết quả tổng hợp (mean ± std trên 4 fold)

### 2.1 3 lớp CN/MCI/AD

| Repr | EN acc | EN macroF1 | EN bacc | EN OVR AUC | EN rec AD | SVM acc | SVM macroF1 | SVM bacc | SVM OVR AUC | SVM rec AD |
|---|---|---|---|---|---|---|---|---|---|---|
| abs132 | 0.633 ± 0.014 | 0.390 ± 0.058 | 0.422 ± 0.039 | 0.737 ± 0.037 | 0.036 ± 0.062 | 0.615 ± 0.036 | 0.349 ± 0.061 | 0.394 ± 0.048 | 0.707 ± 0.071 | 0.000 ± 0.000 |
| abs150 | 0.621 ± 0.022 | 0.390 ± 0.087 | 0.419 ± 0.053 | 0.729 ± 0.041 | 0.071 ± 0.071 | 0.621 ± 0.017 | 0.384 ± 0.040 | 0.411 ± 0.023 | 0.693 ± 0.076 | 0.036 ± 0.062 |
| compact84 | 0.592 ± 0.027 | 0.328 ± 0.069 | 0.370 ± 0.043 | 0.684 ± 0.054 | 0.036 ± 0.062 | 0.597 ± 0.033 | 0.282 ± 0.059 | 0.349 ± 0.044 | 0.629 ± 0.076 | 0.000 ± 0.000 |

### 2.2 CN vs MCI (lớp dương = MCI)

| Repr | EN AUC | EN AP | EN F1 | EN bacc | EN sen | EN spec | SVM AUC | SVM AP | SVM F1 | SVM bacc | SVM sen | SVM spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs132 | 0.698 ± 0.065 | 0.555 ± 0.093 | 0.407 ± 0.139 | 0.620 ± 0.051 | 0.341 ± 0.148 | 0.900 ± 0.060 | 0.751 ± 0.038 | 0.570 ± 0.048 | 0.369 ± 0.118 | 0.601 ± 0.041 | 0.291 ± 0.116 | 0.910 ± 0.059 |
| abs150 | 0.677 ± 0.082 | 0.569 ± 0.102 | 0.348 ± 0.129 | 0.593 ± 0.050 | 0.266 ± 0.122 | 0.920 ± 0.049 | 0.731 ± 0.091 | 0.577 ± 0.081 | 0.377 ± 0.060 | 0.597 ± 0.026 | 0.293 ± 0.072 | 0.900 ± 0.035 |
| compact84 | 0.640 ± 0.096 | 0.517 ± 0.091 | 0.202 ± 0.140 | 0.537 ± 0.045 | 0.143 ± 0.103 | 0.930 ± 0.071 | 0.670 ± 0.093 | 0.498 ± 0.089 | 0.240 ± 0.172 | 0.563 ± 0.058 | 0.166 ± 0.134 | 0.960 ± 0.028 |

### 2.3 CN vs AD (lớp dương = AD)

| Repr | EN AUC | EN AP | EN F1 | EN bacc | EN sen | EN spec | SVM AUC | SVM AP | SVM F1 | SVM bacc | SVM sen | SVM spec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs132 | 0.881 ± 0.051 | 0.741 ± 0.089 | 0.417 ± 0.177 | 0.654 ± 0.093 | 0.357 ± 0.237 | 0.950 ± 0.052 | 0.717 ± 0.071 | 0.506 ± 0.078 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| abs150 | 0.906 ± 0.076 | 0.767 ± 0.125 | 0.466 ± 0.222 | 0.689 ± 0.126 | 0.429 ± 0.303 | 0.950 ± 0.052 | 0.787 ± 0.100 | 0.595 ± 0.126 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 |
| compact84 | 0.821 ± 0.066 | 0.674 ± 0.062 | 0.063 ± 0.108 | 0.518 ± 0.031 | 0.036 ± 0.062 | 1.000 ± 0.000 | 0.681 ± 0.064 | 0.466 ± 0.115 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |

Log-loss inner CV (độ khớp ở mức train) cải thiện với contrasts ở mọi bài toán
(3 lớp EN 0.836 → 0.809; CN vs MCI EN 0.649 → 0.624; CN vs AD EN 0.371 → 0.345), nên
contrasts mang tín hiệu train thật ở khắp nơi; các cột validation ở trên cho thấy
tín hiệu đó khái quát hóa ở đâu.

## 3. Delta ghép cặp từng fold (a trừ b, cùng fold, cùng model)

| Bài toán | Model | So sánh | Delta AUC | Dấu giữa các fold |
|---|---|---|---|---|
| 3 lớp | elasticnet | abs150 vs abs132 | -0.008 ± 0.011 | 3/4 âm |
| 3 lớp | svm_rbf | abs150 vs abs132 | -0.015 ± 0.026 | 3/4 âm |
| 3 lớp | elasticnet | compact84 vs abs132 | -0.054 ± 0.018 | 4/4 âm |
| 3 lớp | svm_rbf | compact84 vs abs132 | -0.079 ± 0.033 | 4/4 âm |
| CN vs MCI | elasticnet | abs150 vs abs132 | -0.020 ± 0.020 | 3/4 âm |
| CN vs MCI | svm_rbf | abs150 vs abs132 | -0.021 ± 0.073 | 3/4 âm |
| CN vs MCI | elasticnet | compact84 vs abs132 | -0.058 ± 0.055 | 3/4 âm |
| CN vs MCI | svm_rbf | compact84 vs abs132 | -0.082 ± 0.084 | 3/4 âm |
| CN vs AD | elasticnet | abs150 vs abs132 | +0.024 ± 0.039 | 3/4 dương |
| CN vs AD | svm_rbf | abs150 vs abs132 | +0.070 ± 0.080 | **4/4 dương** |
| CN vs AD | elasticnet | compact84 vs abs132 | -0.060 ± 0.057 | 3/4 âm |
| CN vs AD | svm_rbf | compact84 vs abs132 | -0.036 ± 0.023 | 4/4 âm |

AUC từng fold cho so sánh chính (abs150 vs abs132, CN vs AD): elastic-net 0.777 vs
0.800, 0.966 vs 0.880, 0.954 vs 0.937, 0.926 vs 0.909; svm_rbf 0.777 vs 0.766, 0.891
vs 0.686, 0.851 vs 0.800, 0.629 vs 0.617.

## 4. Các giả thuyết

**Contrasts có bổ sung thông tin ngoài giá trị tuyệt đối không?** Tùy bài toán.
CN vs AD: có — elastic-net AUC 0.881 → 0.906 (AP 0.741 → 0.767, F1 0.417 → 0.466) và
SVM-RBF AUC 0.717 → 0.787, với mức tăng của SVM dương ở cả 4 fold và của elastic-net
dương ở 3/4 fold. CN vs MCI và 3 lớp: không — delta âm nhỏ cho cả hai model (3/4 fold
âm). Contrasts cải thiện bài toán mà tín hiệu bệnh mạnh nhất và riêng task tham chiếu
(KD1) yếu, nhưng không giúp phân biệt MCI khó hơn hay bài toán 3 lớp, nơi chúng khớp
train tốt hơn (inner log-loss giảm ở mọi nơi) nhưng khái quát hóa kém hơn đôi chút.

**Biểu diễn compact có giữ hoặc cải thiện hiệu năng với ít biến hơn không?**
Không. compact84 (84 feature) thua abs132 ở cả 6 so sánh bài toán/model, từ 0.036 đến
0.082 AUC (3/4 hoặc 4/4 fold âm ở mọi trường hợp), và thua abs150 từ 0.037 đến 0.106.
Các feature tuyệt đối của KD2 và KD3 bị bỏ đi mang thông tin phân biệt mà 18
contrasts không bù lại được; ở CN vs AD, compact thậm chí mất gần hết lợi ích của
contrast (0.821 so với 0.906 của abs150 EN).

**Lợi ích có nhất quán giữa các fold và giữa các bài toán không?** Trong cùng một
bài toán, dấu nhất quán giữa các fold: lợi ích contrast ở CN-vs-AD giữ ở 4/4 fold
SVM và 3/4 fold elastic-net, còn mức thua ở CN-vs-MCI / 3 lớp giữ ở 3/4 fold. Giữa
các bài toán, dấu đảo chiều, nên một phát biểu chung duy nhất về giá trị của
contrasts sẽ gây hiểu nhầm; lợi ích đặc trưng cho CN vs AD.

## 5. Hạn chế

- Cùng các lưu ý về split cố định như experiment fusion (held-out AD trùng lặp trong
  mọi fold validation; lớp AD nhỏ).
- Contrast ratio dùng KD1 làm task tham chiếu; giá trị KD1 nằm trong cả tử và mẫu
  của thang đo, nên hệ số contrast không nên diễn giải nhân quả.
- Contrasts là hiệu xuyên task thô; chưa tách các hiệu ứng độ khó, bố cục hay luyện
  tập.
- Một số góc grid regularize yếu (C=10) chạm trần vòng lặp của saga ở vài fold; việc
  chọn lựa không bị ảnh hưởng (chúng thua xa) và giống nhau giữa các representation.
- Một fold (co_vs_ad, elasticnet, compact84, fold 3) có ngưỡng Youden là "inf" trong
  detail CSV: trên inner OOF của fold đó không ngưỡng nào thắng luật dự đoán toàn âm
  tính, nên các metric phụ theo Youden của dòng này mô tả dự đoán toàn âm tính.
- Hai bước hoàn thiện sau chạy đã được áp dụng lên artifacts (xem
  `src/expcommon/postprocess_first_full_runs.py`): gộp các dòng proba layout dài về
  layout rộng, và tính lại các ô balanced accuracy 3 lớp bị thiếu ở lần chạy đầu từ
  xác suất từng người đã lưu (quy tắc argmax), mỗi ô đều được đối chiếu với recall
  từng lớp đã lưu. Mã runner đã được sửa để ghi trực tiếp cả hai trong các lần chạy
  sau.
