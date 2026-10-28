# Kết quả baseline: 6 classifiers trên processed 37-feature dataset (RAW157)

Run `2026-10-28T09:27Z`. Chi tiết: `results/results_detail.csv` (720 đánh giá),
tổng hợp: `results/summary.csv`, ma trận nhầm lẫn: `results/confusion_summary.csv`,
provenance: `provenance.json`.

## 1. Thiết lập

- **Dữ liệu**: `processed_dataset/processed_dataset_v1_wide.csv` — 157 subjects (CN 100 / MCI 41 /
  AD 16), 96 features (24 features RAW157 x 4 tasks: KD1/KD2/KD3/READING), complete-case (không có
  giá trị khuyết).
- **Split (cố định)**: từ `processed_dataset/splits/` — 4 AD subjects giữ riêng được ghép vào mọi
  validation fold và không bao giờ vào training; 153 subjects còn lại chia 4 fold phân tầng.
  Training mỗi fold (bài toán 3-class): 115 subjects (75 CN / 31-30 MCI / 9 AD); validation:
  42-43 subjects với 7 AD (3 từ fold + 4 holdout).
- **Models**: LogisticRegression, SVC-RBF, RandomForest, LightGBM, GaussianNB, MLP.
- **Xử lý imbalance**: balanced class weights (LR/SVM/RF/LGBM), inverse-frequency priors (GNB),
  sample weights (MLP) — đều tính từ training split của từng fold.
- **Protocol**: StandardScaler fit trên training split; không impute, không tuning hyperparameter,
  ngưỡng quyết định cố định 0.5; 10 model seeds `[42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337]`
  x 4 folds = 40 đánh giá cho mỗi (problem, model).
- **Metrics**: 3-class = accuracy, macro sensitivity/precision/F1, specificity trung bình theo lớp,
  AUC one-vs-rest macro, kèm recall/precision/F1 theo từng lớp; binary = metric của lớp dương tính
  (MCI hoặc AD là positive).

## 2. Tổng hợp kết quả (mean ± std trên 40 đánh giá)

### 3-class CN/MCI/AD

| Model | Acc | Sen (macro) | Prec (macro) | Spec | AUC (OVR) | F1 (macro) | Recall AD | F1 AD |
|---|---|---|---|---|---|---|---|---|
| lightgbm | 0.599 ± 0.037 | 0.464 ± 0.044 | 0.615 ± 0.103 | 0.755 ± 0.029 | 0.682 ± 0.026 | 0.469 ± 0.052 | 0.189 ± 0.087 | 0.292 ± 0.113 |
| svm_rbf | 0.610 ± 0.033 | 0.367 ± 0.058 | 0.324 ± 0.218 | 0.695 ± 0.044 | 0.700 ± 0.071 | 0.304 ± 0.096 | 0.036 ± 0.062 | 0.062 ± 0.108 |
| mlp | 0.567 ± 0.050 | 0.436 ± 0.041 | 0.558 ± 0.106 | 0.735 ± 0.025 | 0.652 ± 0.048 | 0.446 ± 0.044 | 0.214 ± 0.085 | 0.311 ± 0.096 |
| random_forest | 0.562 ± 0.046 | 0.434 ± 0.044 | 0.549 ± 0.108 | 0.740 ± 0.029 | 0.710 ± 0.043 | 0.436 ± 0.051 | 0.182 ± 0.084 | 0.264 ± 0.097 |
| logistic_regression | 0.533 ± 0.039 | 0.424 ± 0.038 | 0.452 ± 0.043 | 0.737 ± 0.019 | 0.673 ± 0.036 | 0.425 ± 0.039 | 0.214 ± 0.071 | 0.268 ± 0.071 |
| gaussian_nb | 0.462 ± 0.035 | 0.418 ± 0.032 | 0.423 ± 0.040 | 0.719 ± 0.021 | 0.611 ± 0.040 | 0.407 ± 0.030 | 0.321 ± 0.118 | 0.307 ± 0.113 |

Recall theo lớp (mean ± std): LGBM CN 0.788 ± 0.054 / MCI 0.415 ± 0.070 / AD 0.189 ± 0.087;
RF CN 0.744 / MCI 0.375 / AD 0.182; LR CN 0.690 / MCI 0.368 / AD 0.214; GNB CN 0.520 /
MCI 0.411 / AD 0.321; SVM CN 0.990 / MCI 0.075 / AD 0.036 (dự đoán gần như mọi thứ là CN).

### Binary CN vs MCI (positive = MCI)

| Model | Acc | Sen (MCI) | Prec | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| svm_rbf | 0.716 ± 0.035 | 0.096 ± 0.064 | 0.667 ± 0.408 | 0.970 ± 0.033 | 0.647 ± 0.062 | 0.162 ± 0.103 |
| lightgbm | 0.684 ± 0.044 | 0.439 ± 0.091 | 0.460 ± 0.056 | 0.784 ± 0.064 | 0.600 ± 0.058 | 0.445 ± 0.060 |
| random_forest | 0.675 ± 0.047 | 0.403 ± 0.118 | 0.431 ± 0.087 | 0.785 ± 0.044 | 0.643 ± 0.090 | 0.414 ± 0.101 |
| logistic_regression | 0.660 ± 0.026 | 0.461 ± 0.063 | 0.423 ± 0.035 | 0.740 ± 0.045 | 0.608 ± 0.034 | 0.440 ± 0.041 |
| mlp | 0.656 ± 0.061 | 0.343 ± 0.088 | 0.404 ± 0.111 | 0.783 ± 0.076 | 0.586 ± 0.065 | 0.366 ± 0.090 |
| gaussian_nb | 0.575 ± 0.064 | 0.534 ± 0.092 | 0.352 ± 0.067 | 0.590 ± 0.087 | 0.596 ± 0.096 | 0.423 ± 0.071 |

### Binary CN vs AD (positive = AD)

| Model | Acc | Sen (AD) | Prec | Spec | AUC | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0.859 ± 0.035 | 0.607 ± 0.156 | 0.747 ± 0.159 | 0.930 ± 0.052 | 0.889 ± 0.061 | 0.649 ± 0.091 |
| lightgbm | 0.832 ± 0.047 | 0.307 ± 0.177 | 0.810 ± 0.260 | 0.979 ± 0.027 | 0.860 ± 0.025 | 0.423 ± 0.205 |
| random_forest | 0.820 ± 0.026 | 0.232 ± 0.094 | 0.835 ± 0.212 | 0.984 ± 0.020 | 0.899 ± 0.056 | 0.353 ± 0.117 |
| mlp | 0.816 ± 0.031 | 0.307 ± 0.104 | 0.740 ± 0.220 | 0.959 ± 0.037 | 0.825 ± 0.082 | 0.415 ± 0.110 |
| svm_rbf | 0.789 ± 0.014 | 0.036 ± 0.062 | 0.250 ± 0.433 | 1.000 ± 0.000 | 0.864 ± 0.038 | 0.063 ± 0.108 |
| gaussian_nb | 0.719 ± 0.054 | 0.536 ± 0.062 | 0.402 ± 0.074 | 0.770 ± 0.052 | 0.676 ± 0.121 | 0.458 ± 0.072 |

## 3. Ma trận nhầm lẫn gộp (tổng qua 10 seeds x 4 folds)

CN vs AD, LogisticRegression (1.280 mẫu validation): CN→CN 930, CN→AD 70, AD→CN 110, AD→AD 170
(recall AD 61%, precision AD 71%).
3-class, LightGBM (1.690 mẫu): CN→CN 788 / CN→MCI 206 / CN→AD 6; MCI→CN 228 / MCI→MCI 171 /
MCI→AD 11; AD→CN 83 / AD→MCI 144 / AD→AD 53 (AD phần lớn bị nhầm thành MCI).
3-class, GaussianNB: AD→AD 90 / AD→MCI 160 / AD→CN 30 — recall AD cao nhất (0.32) trong sáu model,
đổi lại tỷ lệ nhầm CN rất lớn.

## 4. Nhận xét chính

- **CN vs AD là bài toán dễ học nhất**: LR đạt AUC 0.889 và F1 0.649 với recall AD cao nhất (0.61);
  RF có AUC tốt nhất (0.899) nhưng sensitivity ở ngưỡng 0.5 chỉ 0.23.
- **SVM-RBF xếp hạng tốt nhưng không dự đoán được lớp bệnh ở ngưỡng 0.5** (recall AD 3-class 0.04,
  sensitivity co_vs_ad 0.04, co_vs_mci 0.10) dù AUC 0.70 / 0.86 / 0.65: sigmoid calibration nén xác
  suất nên gần như không mẫu nào vượt 0.5. Accuracy cao (0.61 ở 3-class) gần như hoàn toàn nhờ
  recall CN (0.99). Không nên diễn giải accuracy này là khả năng phát hiện; cần threshold tuning.
- **GNB nhạy nhất với AD** (recall AD 3-class 0.32; sensitivity co_vs_ad 0.54) nhờ prior
  inverse-frequency, nhưng accuracy tổng thể yếu (3-class acc 0.46).
- CN vs MCI là bài toán khó nhất (AUC tốt nhất 0.65, mọi model macro F1 <= 0.45).
- LR/SVM/GNB là deterministic: 10 dòng seed giống hệt nhau; std báo cáo của các model này chỉ là
  dao động giữa các fold. RF/LGBM/MLP thay đổi theo seed.

## 5. Lưu ý

- Cùng 4 AD holdout xuất hiện ở **mọi** validation fold, nên 40 đánh giá cho mỗi (problem, model)
  **không độc lập** và giá trị gộp thiên về 4 subjects đó; 4 AD holdout cũng **không bao giờ được
  train**. Khi so sánh model nên báo cáo metric theo từng fold.
- Số AD ở validation nhỏ (7 mỗi fold, trong đó 3 là mới với model) và training chỉ có 9 AD; ước
  lượng cho AD có phương sai lớn (xem cột std).
- Không tuning hyperparameter và không tuning ngưỡng (cố định 0.5); kết quả là cận dưới so với
  model đã tuning.
- Các model này dùng 24 features hoàn chỉnh; 9 feature còn lại của bộ 37 bị loại vì không đủ 157
  subjects (xem `processed_dataset/README.md`).

## 6. Tái lập

- Code: `src/baseline/` (`run_baselines.py`, `config.yaml`, `aggregate_results.py`); chạy lại bằng
  `PYTHONPATH=src .venv/bin/python -m baseline.run_baselines --config src/baseline/config.yaml`
  (từ chối ghi đè kết quả detail đã có).
- Ghi trong `provenance.json`: git commit `ffa36c7`, config sha256, data sha256 (wide CSV, fold
  assignments, danh sách AD holdout), phiên bản packages (Python 3.12.3, numpy 2.5.4, pandas 3.1.0,
  scikit-learn 1.9.1, scipy 1.18.1, lightgbm 4.7.0).
