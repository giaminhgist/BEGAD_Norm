# Kết quả baseline: 6 bộ phân loại trên dataset 33 features RAW157 (157 subjects, imputed)

Run `2026-10-28T19:09Z`. Chi tiết: `results/results_detail.csv` (720 lượt đánh giá),
tổng hợp: `results/summary.csv`, confusion: `results/confusion_summary.csv`,
nguồn gốc: `provenance.json`.

## 1. Thiết lập

- **Dữ liệu**: `processed_dataset/processed_dataset_33feat_wide.csv` — 157 subjects (CN 100 / MCI 41 /
  AD 16), 132 features (33 features RAW157 x 4 task: KD1/KD2/KD3/READING). 9 features bị loại ở bản 24
  features (D017, D022, D025, D026, D030, D034, D035, D036, D037) được đưa vào; các ô không hỗ trợ
  được (51 trên 20724 ô, 0.25%, rải trên 20 subjects) để NaN và **được impute theo từng fold bằng
  median từng cột, fit trên riêng split training** (SimpleImputer trước StandardScaler; không giá trị
  nào từ validation hay held-out AD lọt vào training).
- **Split (cố định)**: giống hệt baseline 24 features, từ `processed_dataset/splits/` — 4 subjects
  held-out AD (BEGAD-001, BEGAD-005, BEGAD-006, BEGAD-016) được gắn vào mọi fold validation và không
  bao giờ vào training; 153 subjects còn lại chia 4 fold phân tầng (seed 42). Training mỗi fold:
  114-115 subjects cho bài toán 3 lớp (75 CN / 30-31 MCI / 9 AD); validation: 42-43 subjects với
  7 AD (3 từ fold + 4 held-out).
- **Mô hình**: LogisticRegression, SVC-RBF, RandomForest, LightGBM, GaussianNB, MLP.
- **Xử lý mất cân bằng lớp**: trọng số lớp cân bằng (LR/SVM/RF/LGBM), prior nghịch đảo tần suất (GNB),
  trọng số mẫu (MLP) — đều tính từ split training của từng fold.
- **Giao thức**: impute median theo fold rồi StandardScaler, đều fit trên riêng split training; không
  tuning hyperparameter, ngưỡng quyết định cố định 0.5; 10 model seeds
  `[42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337]` x 4 folds = 40 lượt đánh giá cho mỗi
  (problem, model).
- **Metrics**: 3 lớp = accuracy, macro sensitivity/precision/F1, mean per-class specificity,
  one-vs-rest macro AUC, kèm recall/precision/F1 từng lớp; nhị phân = metrics của lớp positive
  (MCI hoặc AD).

Vì tập subjects, splits và seeds giống hệt baseline 24 features, hai run **ghép cặp theo từng lượt
đánh giá**; mục 3 báo cáo delta ghép cặp (33 features trừ 24).

## 2. Tóm tắt kết quả (mean ± std trên 40 lượt đánh giá)

### 3 lớp CN/MCI/AD

| Model | Acc | Sen (macro) | Prec (macro) | Spec | AUC (OVR) | F1 (macro) | Recall AD | F1 AD |
|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.562 ± 0.023 | 0.467 ± 0.021 | 0.505 ± 0.011 | 0.756 ± 0.025 | 0.705 ± 0.025 | 0.473 ± 0.016 | 0.286 ± 0.000 | 0.356 ± 0.013 |
| svm_rbf | 0.598 ± 0.027 | 0.372 ± 0.039 | 0.272 ± 0.076 | 0.706 ± 0.040 | 0.705 ± 0.092 | 0.310 ± 0.063 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| random_forest | 0.607 ± 0.047 | 0.469 ± 0.064 | 0.584 ± 0.110 | 0.761 ± 0.036 | 0.741 ± 0.050 | 0.469 ± 0.071 | 0.182 ± 0.096 | 0.276 ± 0.127 |
| lightgbm | 0.632 ± 0.045 | 0.473 ± 0.050 | 0.538 ± 0.135 | 0.773 ± 0.030 | 0.732 ± 0.043 | 0.462 ± 0.061 | 0.125 ± 0.112 | 0.186 ± 0.153 |
| gaussian_nb | 0.544 ± 0.013 | 0.479 ± 0.031 | 0.495 ± 0.037 | 0.758 ± 0.015 | 0.618 ± 0.033 | 0.471 ± 0.027 | 0.286 ± 0.101 | 0.325 ± 0.108 |
| mlp | 0.603 ± 0.051 | 0.465 ± 0.056 | 0.550 ± 0.119 | 0.762 ± 0.032 | 0.698 ± 0.052 | 0.462 ± 0.061 | 0.171 ± 0.091 | 0.253 ± 0.120 |

Recall từng lớp (mean ± std): LGBM CN 0.852 ± 0.080 / MCI 0.442 ± 0.080 / AD 0.125 ± 0.112;
MLP CN 0.795 / MCI 0.427 / AD 0.171; RF CN 0.799 / MCI 0.425 / AD 0.182; LR CN 0.700 /
MCI 0.414 / AD 0.286; GNB CN 0.620 / MCI 0.532 / AD 0.286; SVM CN 0.940 / MCI 0.175 /
AD 0.000 (dự đoán gần như tất cả là CN).

### Nhị phân CN vs MCI (positive = MCI)

| Model | Acc | Sen (MCI) | Prec | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.688 ± 0.021 | 0.486 ± 0.053 | 0.468 ± 0.035 | 0.770 ± 0.044 | 0.670 ± 0.034 | 0.474 ± 0.030 |
| svm_rbf | 0.752 ± 0.012 | 0.289 ± 0.144 | 0.740 ± 0.151 | 0.940 ± 0.045 | 0.719 ± 0.085 | 0.379 ± 0.142 |
| random_forest | 0.712 ± 0.053 | 0.465 ± 0.192 | 0.496 ± 0.082 | 0.812 ± 0.057 | 0.697 ± 0.121 | 0.468 ± 0.130 |
| lightgbm | 0.744 ± 0.063 | 0.486 ± 0.101 | 0.592 ± 0.118 | 0.849 ± 0.078 | 0.687 ± 0.099 | 0.525 ± 0.085 |
| gaussian_nb | 0.653 ± 0.061 | 0.607 ± 0.107 | 0.433 ± 0.073 | 0.670 ± 0.071 | 0.625 ± 0.099 | 0.504 ± 0.081 |
| mlp | 0.705 ± 0.043 | 0.437 ± 0.112 | 0.502 ± 0.088 | 0.814 ± 0.067 | 0.668 ± 0.065 | 0.458 ± 0.074 |

### Nhị phân CN vs AD (positive = AD)

| Model | Acc | Sen (AD) | Prec | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.875 ± 0.044 | 0.607 ± 0.186 | 0.817 ± 0.185 | 0.950 ± 0.052 | 0.891 ± 0.066 | 0.671 ± 0.129 |
| svm_rbf | 0.781 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.874 ± 0.037 | 0.000 ± 0.000 |
| random_forest | 0.817 ± 0.039 | 0.207 ± 0.139 | 0.787 ± 0.365 | 0.988 ± 0.018 | 0.914 ± 0.061 | 0.316 ± 0.188 |
| lightgbm | 0.806 ± 0.050 | 0.200 ± 0.152 | 0.654 ± 0.428 | 0.976 ± 0.031 | 0.868 ± 0.035 | 0.296 ± 0.212 |
| gaussian_nb | 0.727 ± 0.041 | 0.393 ± 0.062 | 0.387 ± 0.083 | 0.820 ± 0.045 | 0.639 ± 0.102 | 0.388 ± 0.067 |
| mlp | 0.817 ± 0.030 | 0.279 ± 0.096 | 0.784 ± 0.224 | 0.968 ± 0.037 | 0.813 ± 0.085 | 0.393 ± 0.107 |

## 3. So sánh ghép cặp với baseline 24 features (delta = 33f trừ 24f)

Cùng 157 subjects, cùng splits và seeds; chỉ khác bộ features (cộng 51 ô được impute).
Mean ± std trên 40 cặp (seed, fold); mang tính mô tả, chưa kiểm định ý nghĩa.

| Problem | Model | Acc delta | AUC delta |
|---|---|---|---|
| 3 lớp | logistic_regression | +0.029 ± 0.020 | +0.032 ± 0.015 |
| 3 lớp | svm_rbf | -0.012 ± 0.012 | +0.004 ± 0.034 |
| 3 lớp | random_forest | +0.045 ± 0.041 | +0.031 ± 0.023 |
| 3 lớp | lightgbm | +0.033 ± 0.054 | +0.051 ± 0.029 |
| 3 lớp | gaussian_nb | +0.083 ± 0.040 | +0.006 ± 0.025 |
| 3 lớp | mlp | +0.035 ± 0.059 | +0.047 ± 0.051 |
| CN vs MCI | logistic_regression | +0.028 ± 0.035 | +0.062 ± 0.044 |
| CN vs MCI | svm_rbf | +0.036 ± 0.032 | +0.072 ± 0.036 |
| CN vs MCI | random_forest | +0.037 ± 0.034 | +0.054 ± 0.038 |
| CN vs MCI | lightgbm | +0.060 ± 0.049 | +0.087 ± 0.073 |
| CN vs MCI | gaussian_nb | +0.078 ± 0.037 | +0.029 ± 0.012 |
| CN vs MCI | mlp | +0.049 ± 0.057 | +0.083 ± 0.051 |
| CN vs AD | logistic_regression | +0.016 ± 0.016 | +0.003 ± 0.017 |
| CN vs AD | svm_rbf | -0.008 ± 0.014 | +0.010 ± 0.021 |
| CN vs AD | random_forest | -0.002 ± 0.024 | +0.015 ± 0.024 |
| CN vs AD | lightgbm | -0.026 ± 0.025 | +0.008 ± 0.028 |
| CN vs AD | gaussian_nb | +0.008 ± 0.079 | -0.036 ± 0.026 |
| CN vs AD | mlp | +0.001 ± 0.046 | -0.012 ± 0.079 |

9 features mới thêm giúp nhiều nhất ở **3 lớp** và **CN vs MCI** (cả sáu mô hình đều tăng accuracy
ở CN vs MCI; AUC tăng 0.03-0.09). CN vs AD gần như không đổi.

## 4. Ma trận confusion gộp (tổng trên 10 seeds x 4 folds)

CN vs AD, LogisticRegression (1.280 mẫu validation): CN→CN 950, CN→AD 50, AD→CN 110, AD→AD 170
(recall AD 61%, precision AD 77%).
3 lớp, LightGBM (1.690 mẫu): CN→CN 852 / CN→MCI 127 / CN→AD 21; MCI→CN 220 / MCI→MCI 181 /
MCI→AD 9; AD→CN 78 / AD→MCI 167 / AD→AD 35 (AD phần lớn nhầm với MCI).
3 lớp, GaussianNB: AD→AD 80 / AD→MCI 160 / AD→CN 40 — recall MCI tốt nhất (0.53) và recall AD đồng
hạng nhất (0.29, cùng LR) trong sáu mô hình, đổi lại CN bị phân loại sai nhiều.
3 lớp, SVM-RBF: AD→AD 0 — mô hình không bao giờ dự đoán AD ở ngưỡng 0.5.

## 5. Nhận xét chính

- **CN vs AD vẫn là bài toán học được tốt nhất**: LR đạt AUC 0.891 và F1 tốt nhất (0.671) với
  sensitivity AD 0.61 ở ngưỡng 0.5; RF có AUC tốt nhất (0.914) nhưng sensitivity ở ngưỡng 0.5 chỉ 0.21.
- **9 features mới thêm cải thiện 3 lớp và CN vs MCI trên các mô hình** (delta ghép cặp, mục 3): mọi
  mô hình đều tăng accuracy và AUC ở CN vs MCI, 5/6 mô hình tăng accuracy 3 lớp, và cả sáu tăng AUC
  3 lớp. CN vs AD không đổi.
- **SVM-RBF không dự đoán lớp bệnh ở ngưỡng 0.5** (3 lớp recall AD 0.000, co_vs_ad sensitivity 0.000,
  co_vs_mci 0.29) dù AUC 0.71 / 0.87 / 0.72: hiệu chuẩn sigmoid nén xác suất của các lớp positive
  xuống dưới 0.5. Accuracy cao của nó gần như toàn bộ từ recall CN. Không nên diễn giải accuracy này
  thành khả năng phát hiện; cần tuning ngưỡng.
- **LR và GNB đồng hạng về recall AD 3 lớp tốt nhất (0.286)**; GNB còn có recall MCI tốt nhất (0.53)
  nhờ prior nghịch đảo tần suất. AUC CN vs AD của GNB giảm nhẹ khi thêm features (ghép cặp -0.036).
- LR/SVM/GNB là deterministic: 10 hàng seed của chúng giống hệt nhau; std của các mô hình này chỉ là
  biến thiên giữa các fold. RF/LGBM/MLP biến thiên theo seed.

## 6. Lưu ý

- Cùng 4 subjects held-out AD xuất hiện trong **mọi** fold validation, nên 40 lượt đánh giá của mỗi
  (problem, model) **không độc lập** và mean gộp làm trội 4 subjects này; 4 held-out AD cũng
  **không bao giờ xuất hiện trong training**. Hãy báo cáo metrics theo từng fold khi so sánh mô hình.
- Số AD trong validation nhỏ (7 mỗi fold, 3 trong số đó mới với mô hình) và training AD là 9; ước
  lượng cho AD có phương sai cao (xem cột std).
- 51 ô (0.25%) được impute bằng median training theo cột; với tỷ lệ nhỏ như vậy, lựa chọn imputer dự
  kiến ít ảnh hưởng, nhưng imputation là một phần của pipeline và chưa được ablation ở đây.
- Delta ghép cặp ở mục 3 mang tính mô tả; chưa chạy kiểm định thống kê.
- Không tuning hyperparameter và không tuning ngưỡng (cố định 0.5); kết quả là cận dưới cho các mô
  hình đã tuning.

## 7. Tái lập

- Code: `src/baseline/` (`run_baselines.py`, `config_33feat.yaml`, `aggregate_results.py`); chạy lại với
  `PYTHONPATH=src .venv/bin/python -m baseline.run_baselines --config src/baseline/config_33feat.yaml`
  (từ chối ghi đè kết quả chi tiết đã có).
- Script dựng dataset: `processed_dataset/make_33feat_dataset.py` (157 subjects, 51 ô NaN).
- Ghi trong `provenance.json`: git commit `56834bd` (working tree dirty: các file dataset 33 features
  và config chưa được commit tại thời điểm chạy), config sha256 `3679074a...`, data sha256 (wide CSV
  `cbcecbc7...`, fold assignments, danh sách held-out AD), phiên bản packages (Python 3.12.3,
  numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, scipy 1.18.1, lightgbm 4.7.0).
