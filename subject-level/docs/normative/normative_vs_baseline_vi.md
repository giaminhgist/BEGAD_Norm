# Mô hình Normative Học Được vs Baseline — Phân loại CO/MCI/AD trên BEGAD

So sánh mô hình normative-deviation học được (`normative_mlp_deepset`, bản port
từ LNGModel EXP-PROP-001) với các baseline cổ điển trong `outputs/baseline_balanced`,
theo CÙNG một giao thức: 3 bài toán × 5-fold CV cấp người tham gia × 20 seed cố
định = 100 lượt đánh giá cho mỗi ô.
Bản tiếng Anh: [normative_vs_baseline.md](normative_vs_baseline.md).

## 1. Phương pháp và giao thức (giống hệt nhau cho cả hai phía)

- **Mô hình (normative)**: encoder 25→128→128 cho từng task → CO latent normative
  bank (μ/σ theo task tính trên encoding CO của train-fold, làm mới mỗi epoch) →
  comparator trên `[z, μ, z−μ, z·μ]` → vector deviation 64-d cho từng task →
  masked deepset pooling (mean‖max) → embedding chủ thể 128-d → MLP head
  (sigmoid cho nhị phân / softmax cho 3 lớp). AdamW lr 1e-3, wd 1e-4, batch 16,
  dropout 0.3, tối đa 150 epoch, patience 30 theo val AUC; checkpoint tốt nhất =
  val AUC cao nhất (epoch sớm nhất thắng khi hòa).
- **Dữ liệu**: một mẫu cho mỗi người tham gia, 25 đặc trưng × 11 task; ô thiếu
  được điền bằng trung bình theo đặc trưng của train-fold, chuẩn hóa theo fold
  trên tập huấn luyện; task toàn-NaN bị mask. Mất cân bằng lớp được xử lý bằng
  trọng số loss nghịch đảo tần suất của train-fold (khớp với
  `class_weight: balanced` của baseline).
- **Bài toán**: `co_vs_mci` (MCI là lớp dương), `co_vs_ad` (AD là lớp dương),
  `co_mci_ad` (3 lớp; Sen/Precision/Spec/F1 macro, AUC macro OVR).
- **Ngưỡng**: cố định 0.5 (nhị phân), argmax (3 lớp). Không tinh chỉnh ngưỡng
  hay siêu tham số ở bất kỳ phía nào.
- **Fold/seed**: giống hệt baseline — fold sinh bởi cùng hàm `stratified_folds`,
  cùng 20 seed `[42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337, 2, 3, 5, 10, 21,
  777, 2020, 2021, 2022, 2023]`.

Mean ± std trên 100 lượt đánh giá cho mỗi (bài toán, mô hình). ΔAUC = hiệu các
trung bình so với mô hình normative (không phải kiểm định bắt cặp).

## 2. Kết quả

### 2.1 `co_vs_mci` (CO 100 vs MCI 41)

| Model | AUC (mean ± std) | ΔAUC (pp) | ACC | SEN | SPEC | PREC | F1 |
|---|---|---|---|---|---|---|---|
| **normative_mlp_deepset** | **0.7674 ± 0.0960** | — | 0.6732 | **0.5838** | 0.7095 | 0.4740 | 0.4809 |
| svm_rbf | 0.7384 ± 0.0945 | −2.90 | **0.7282** | 0.4585 | **0.8390** | 0.5534 | 0.4829 |
| fnn | 0.6954 ± 0.1013 | −7.20 | 0.7178 | 0.4572 | 0.8245 | 0.5349 | 0.4796 |
| gnb | 0.6429 ± 0.1028 | −12.44 | 0.6301 | 0.5910 | 0.6455 | 0.4141 | 0.4808 |
| logistic_regression | 0.6361 ± 0.0873 | −13.13 | 0.6572 | 0.4472 | 0.7430 | 0.4222 | 0.4244 |
| svm_linear | 0.5975 ± 0.0968 | −16.98 | 0.6447 | 0.4068 | 0.7420 | 0.3954 | 0.3906 |
| qda | 0.5535 ± 0.0943 | −21.39 | 0.6962 | 0.0932 | 0.9435 | 0.3493 | 0.1401 |

### 2.2 `co_vs_ad` (CO 100 vs AD 16)

| Model | AUC (mean ± std) | ΔAUC (pp) | ACC | SEN | SPEC | PREC | F1 |
|---|---|---|---|---|---|---|---|
| svm_linear | **0.8638 ± 0.0993** | +5.18 | **0.8687** | 0.3483 | 0.9520 | 0.4977 | 0.3861 |
| logistic_regression | 0.8571 ± 0.0996 | +4.50 | 0.8588 | 0.3758 | 0.9365 | 0.4455 | 0.3839 |
| svm_rbf | 0.8428 ± 0.0903 | +3.07 | 0.8619 | 0.0558 | **0.9910** | 0.1300 | 0.0750 |
| normative_mlp_deepset | 0.8121 ± 0.1145 | — | 0.7716 | 0.4208 | 0.8270 | 0.2965 | 0.2973 |
| gnb | 0.7965 ± 0.1592 | −1.55 | 0.8465 | **0.5975** | 0.8870 | 0.4780 | 0.5120 |
| fnn | 0.7141 ± 0.1562 | −9.80 | 0.8569 | 0.3067 | 0.9450 | 0.4257 | 0.3374 |
| qda | 0.5937 ± 0.2230 | −21.84 | 0.8610 | 0.0000 | 0.9985 | 0.0000 | 0.0000 |

### 2.3 `co_mci_ad` (3 lớp, CO 100 / MCI 41 / AD 16)

| Model | AUC (mean ± std) | ΔAUC (pp) | ACC | SEN | SPEC | PREC | F1 |
|---|---|---|---|---|---|---|---|
| svm_rbf | **0.7583 ± 0.0706** | +1.91 | **0.6478** | 0.4588 | 0.7755 | 0.4359 | 0.4364 |
| **normative_mlp_deepset** | 0.7392 ± 0.0732 | — | 0.5884 | **0.4922** | 0.7574 | 0.4644 | 0.4503 |
| fnn | 0.7008 ± 0.0865 | −3.84 | 0.6471 | 0.4751 | 0.7650 | 0.4815 | 0.4681 |
| logistic_regression | 0.6982 ± 0.0715 | −4.10 | 0.5915 | 0.4519 | 0.7408 | 0.4567 | 0.4440 |
| svm_linear | 0.6913 ± 0.0757 | −4.79 | 0.5736 | 0.4265 | 0.7208 | 0.4374 | 0.4212 |
| gnb | 0.6575 ± 0.0889 | −8.17 | 0.5685 | 0.4697 | 0.7564 | 0.4763 | 0.4574 |
| qda | 0.5645 ± 0.1068 | −17.47 | 0.6203 | 0.3411 | 0.6735 | 0.3105 | 0.2955 |

## 3. Nhận định

- **co_vs_mci**: mô hình normative có AUC cao nhất (0.7674, +2.90 pp so với
  svm_rbf) và sensitivity cao nhất (0.5838); accuracy và specificity nằm giữa
  bảng. Trên bài toán này, tín hiệu lệch-khỏi-chuẩn-CO (deviation-from-CO-norms)
  mạnh nhất trong ba bài toán.
- **co_vs_ad**: mô hình normative xếp thứ 4 về AUC (0.8121), sau svm_linear,
  logistic regression và svm_rbf; sensitivity (0.4208) chỉ đứng sau GNB. Chỉ có
  16 người tham gia AD (3–4 người mỗi validation fold), mọi ước lượng trên bài
  toán này đều nhiễu — std của AUC theo mô hình lên tới 0.09–0.22 — và ngưỡng
  cố định 0.5 đẩy hầu hết mô hình về phía dự đoán toàn âm tính (qda: SEN 0.0000,
  SPEC 0.9985).
- **co_mci_ad**: mô hình normative đứng thứ hai về AUC (0.7392, −1.91 pp so với
  svm_rbf) với sensitivity tốt nhất (0.4922); svm_rbf dẫn đầu về ACC/SPEC.
- Không chạy kiểm định ý nghĩa thống kê nào; ΔAUC là hiệu các trung bình trên
  100 lượt đánh giá dùng chung, và mô hình chỉ dùng MỘT biến thể cố định với MỘT
  ngưỡng cố định. Đây là các so sánh phát triển, không phải tuyên bố về tính
  vượt trội.

## 4. Tái lập và tạo tác (artifacts)

- Kết quả: `outputs/normative/results/results_detail.csv` (300 dòng, một dòng cho
  mỗi lượt đánh giá, theo schema của baseline), `results/summary.csv` (18 dòng),
  `provenance.json` (hash config/data/code, môi trường), `run_log.csv`.
- Artifacts từng lượt chạy (metrics.jsonl, best.pt/last.pt, predictions.csv,
  summary.json, run_info.json): `outputs/normative/runs/`.
- Config: `src/begad_norm/normative/config.yaml`; cài đặt:
  `src/begad_norm/normative/`; kế hoạch: `docs/plans/2026-10-14-normative-model.md`.
- Chạy lại: `uv run python -m begad_norm.normative.run_normative --config
  src/begad_norm/normative/config.yaml --workers 8` (an toàn khi resume; tất định
  theo từng seed).

## 5. Hạn chế

- 25 đặc trưng được sàng lọc trên chính cohort này (Kruskal–Wallis chưa hiệu
  chỉnh p < 0.1 ở ≥ 3 task), nên mọi con số báo cáo đều mang thiên lệch chọn lọc
  trong mẫu (in-sample selection bias); không có test set.
- co_vs_ad chỉ dựa trên 16 người tham gia AD; các validation set theo fold chứa
  3–4 chủ thể AD, nên AUC/SEN/Spec cực kỳ nhiễu và ngưỡng cố định 0.5 tương tác
  mạnh với sự mất cân bằng.
- Chỉ chạy biến thể chính (mlp_deepset); các ablation về pooling/deviation của
  LNGModel (mean/attention/max, hard z-deviation) chưa được chạy.
- Chỉ có chỉ số validation; checkpoint tốt nhất chọn theo validation AUC của từng
  fold (quy tắc phát triển, không phải test độc lập).
- Normative bank chỉ dùng các chủ thể CO của train-fold; các màn hình hướng dẫn
  (task 0, 2, 6, 8, 10) được xử lý như stimulus giống mọi task khác, khớp với
  layout 275 cột của baseline.
