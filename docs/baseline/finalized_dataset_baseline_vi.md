# Baseline Bộ Dataset Chính Thức — 25 Feature × 11 Tasks, 20 Seeds

Thí nghiệm baseline riêng trên bộ feature chính thức `finalized_dataset/`
(25 feature được chọn, Kruskal–Wallis p < 0,1 ở ≥ 3 tasks). Ngày chạy:
2026-10-12. Chạy CPU cục bộ (không GPU / chi phí tính toán).

Chọn bộ feature: xem `docs/baseline/baseline_dataset_comparison_vi.md`
(all72 vs selected25 vs extended39). Lần chạy so sánh 3 dataset được lưu trữ
trong `docs/baseline/baseline_experiment_vi.md` và
`outputs/baseline/comparison_3datasets/`.

## 1. Protocol

- **Mẫu**: một dòng mỗi người tham gia; 25 feature được gộp trên toàn bộ 11
  tasks theo thứ tự feature-major (cột `{feature}_t{task}`) → **275 cột**.
- **Bài toán (3)** — cỡ mẫu (người tham gia):
  - `co_vs_mci` — nhị phân, lớp dương = MCI (141: CO 100 / MCI 41)
  - `co_vs_ad` — nhị phân, lớp dương = AD (116: CO 100 / AD 16)
  - `co_mci_ad` — 3 lớp CO/MCI/AD (157: 100 / 41 / 16)
- **CV**: mức người tham gia, phân tầng theo nhóm, 4 folds × **20 seeds** →
  **80 lượt train/test cho mỗi (bài toán, mô hình)**; tổng 3 × 6 × 80 =
  **1440 lượt**. Cách chia: trong từng nhóm, hoán vị bằng
  `numpy.random.default_rng(seed)` và gán vòng tròn vào các fold.
- **Seeds** (cố định trước khi xem kết quả): 10 seeds của lần chạy so sánh —
  `[42, 2026, 1, 0, 7, 123, 1234, 2024, 2025, 1337]` — cộng 10 seeds mở
  rộng — `[2, 3, 5, 10, 21, 777, 2020, 2021, 2022, 2023]`.
- **Dữ liệu thiếu**: 402 / 43.175 ô (0,93%), 45 / 275 cột bị ảnh hưởng
  (`pupbl_*` của 16 người thiếu baseline đồng tử, và 2 dòng all-NaN task
  10). **Nội suy median theo từng fold chỉ trên tập train**, sau đó
  `StandardScaler` fit chỉ trên tập train.
- **Mô hình** (siêu tham số cố định, không tuning): Logistic Regression
  (`max_iter=2000`), SVM RBF (`C=1, gamma='scale', probability=True`),
  SVM Linear (`C=1, probability=True`), FNN (MLP 64-32, relu, adam,
  lr=1e-3, alpha=1e-4, batch=32, max_iter=300), QDA (`solver='eigen',
  shrinkage=0,5`), GNB (`var_smoothing=1e-9`). `random_state = seed` cho
  LR / SVM / FNN.
- **Chỉ số** (mỗi lượt, trên fold giữ lại): Acc; Sen (nhị phân: recall lớp
  dương, 3 lớp: macro recall); Spec (nhị phân: TN/(TN+FP), 3 lớp: macro
  specificity); AUC (nhị phân: ROC-AUC, 3 lớp: OVR macro); F1 (nhị phân: lớp
  dương, 3 lớp: macro).
- **Chỉ số chính**: AUC. **Tham chiếu ngẫu nhiên** (độ chính xác lớp đa số):
  CO vs MCI 0,709; CO vs AD 0,862; 3 lớp 0,637; AUC ngẫu nhiên = 0,5.

## 2. Kết quả — trung bình ± std trên 80 lượt (20 seeds × 4 folds)

### co_vs_mci (lớp dương = MCI)

| Mô hình | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0,6822±0,0628 | 0,4276±0,1459 | 0,7870±0,0824 | 0,6659±0,0897 | 0,4318±0,1183 |
| svm_rbf | 0,7154±0,0212 | 0,0359±0,0638 | 0,9940±0,0202 | 0,7482±0,0856 | 0,0607±0,1035 |
| svm_linear | 0,6566±0,0724 | 0,4336±0,1426 | 0,7485±0,0912 | 0,6147±0,1081 | 0,4193±0,1131 |
| fnn | 0,7433±0,0631 | 0,4578±0,1545 | 0,8610±0,0779 | 0,7173±0,0801 | 0,5011±0,1318 |
| qda | 0,7058±0,0373 | 0,0858±0,0689 | 0,9600±0,0455 | 0,5880±0,0999 | 0,1392±0,1071 |
| gnb | 0,6575±0,0815 | 0,5780±0,1249 | 0,6905±0,0981 | 0,6534±0,0928 | 0,4964±0,1025 |

### co_vs_ad (lớp dương = AD)

| Mô hình | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0,8612±0,0459 | 0,2656±0,2227 | 0,9565±0,0455 | 0,8510±0,0871 | 0,3145±0,2399 |
| svm_rbf | 0,8621±0,0000 | 0,0000±0,0000 | 1,0000±0,0000 | 0,8492±0,1003 | 0,0000±0,0000 |
| svm_linear | 0,8672±0,0494 | 0,3594±0,2173 | 0,9485±0,0486 | 0,8561±0,0862 | 0,4092±0,2212 |
| fnn | 0,8642±0,0404 | 0,2437±0,2352 | 0,9635±0,0398 | 0,7146±0,1758 | 0,2898±0,2530 |
| qda | 0,8608±0,0066 | 0,0000±0,0000 | 0,9985±0,0076 | 0,5989±0,1982 | 0,0000±0,0000 |
| gnb | 0,8556±0,0685 | 0,5563±0,2352 | 0,9035±0,0682 | 0,7897±0,1510 | 0,5156±0,1982 |

### co_mci_ad (3 lớp)

| Mô hình | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| logistic_regression | 0,6137±0,0630 | 0,4427±0,0837 | 0,7399±0,0427 | 0,6956±0,0665 | 0,4410±0,0965 |
| svm_rbf | 0,6453±0,0196 | 0,3458±0,0203 | 0,6770±0,0152 | 0,7530±0,0679 | 0,2851±0,0367 |
| svm_linear | 0,5888±0,0666 | 0,4338±0,0867 | 0,7301±0,0434 | 0,6979±0,0649 | 0,4318±0,0947 |
| fnn | 0,6629±0,0647 | 0,4653±0,0786 | 0,7623±0,0458 | 0,6891±0,0819 | 0,4634±0,0935 |
| qda | 0,6322±0,0353 | 0,3480±0,0300 | 0,6780±0,0229 | 0,5794±0,0785 | 0,3017±0,0455 |
| gnb | 0,5804±0,0753 | 0,4645±0,0971 | 0,7554±0,0475 | 0,6628±0,0887 | 0,4576±0,0988 |

### AUC trung bình tốt nhất theo bài toán

| Bài toán | Mô hình tốt nhất | AUC |
|---|---|---|
| co_vs_mci | SVM RBF | 0,7482±0,0856 |
| co_vs_ad | SVM Linear | 0,8561±0,0862 |
| co_mci_ad | SVM RBF | 0,7530±0,0679 |

## 3. Ghi chú

- Ước lượng 20 seeds nhất quán với lần chạy so sánh 10 seeds trên cùng bộ
  feature (ví dụ co_vs_ad SVM Linear 0,8518±0,0885 ở 10 seeds so với
  0,8561±0,0862 ở 20 seeds); mở rộng seeds không làm thay đổi thứ hạng mô
  hình hay bài toán.
- SVM RBF dẫn đầu AUC ở co_vs_mci và bài toán 3 lớp, nhưng nhãn dự đoán nhị
  phân của nó sụp về lớp đa số (Sen ≈ 0); thứ hạng đến từ điểm xác suất,
  không phải từ nhãn cứng.
- Độ chính xác ngang hoặc dưới mức ngẫu nhiên lớp đa số ở mọi bài toán; AUC
  là chỉ số có ý nghĩa. Độ lệch chuẩn lớn phản ánh fold kiểm tra nhỏ (29–41
  người) và lớp AD nhỏ (16 người).
- 25 feature được sàng lọc trên chính cohort này (Kruskal–Wallis, p < 0,1 ở
  ≥ 3 tasks), nên hiệu năng CV mang thiên lệch chọn lọc trong mẫu; cần chạy
  chọn lọc lồng để có ước lượng không thiên lệch so với bộ đủ 72 feature
  (xem `docs/baseline/baseline_dataset_comparison_vi.md`).

## 4. Khả năng tái lập

- Mã nguồn: `src/begad_norm/baseline/run_baselines.py`
  (+ `aggregate_results.py`), cấu hình:
  `src/begad_norm/baseline/config.yaml`.
- `outputs/baseline/provenance.json`: git commit, sha256 cấu hình, sha256
  file dữ liệu, phiên bản gói (numpy 2.5.4, pandas 3.0.6,
  scikit-learn 1.9.1, scipy 1.18.1), máy chạy, lệnh, thời lượng.
- Chạy lại:
  `uv run python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml`
  (từ chối ghi đè kết quả cũ), sau đó
  `uv run python -m begad_norm.baseline.aggregate_results`.
- Đã chạy smoke (6 lượt) trước và xóa sau khi dùng (`run_log.csv` có ghi).

## 5. File

| File | Nội dung |
|---|---|
| `outputs/baseline/results/results_detail.csv` | 1440 dòng, một dòng mỗi (bài toán, mô hình, seed, fold) với đủ 5 chỉ số |
| `outputs/baseline/results/summary.csv` | mean / std / n_runs theo (bài toán, mô hình, chỉ số) |
| `outputs/baseline/results/missingness.csv` | số ô thiếu theo cột của ma trận 275 cột |
| `outputs/baseline/run_log.csv` | nhật ký chạy (smoke + full) |
| `outputs/baseline/full_run.log` | stdout của full run |
| `outputs/baseline/provenance.json` | hồ sơ tái lập |
