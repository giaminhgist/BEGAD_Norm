# Thí nghiệm Baseline — Phân loại nhóm BEGAD (participant-wide 275 features)

Các baseline cổ điển trên một mẫu mỗi người tham gia: 25 features chuyển động
mắt đã chọn × 11 tasks = **275 features đầu vào**. Chạy trên CPU local (không
GPU, không chi phí thuê máy).

## 1. Mô tả thí nghiệm

- **Mục tiêu**: đánh giá khả năng phân tách CO / MCI / AD của các baseline
  đơn giản từ features participant-wide. Baseline thăm dò — không có claim cơ
  chế, không diễn giải lâm sàng.
- **Mẫu**: một dòng mỗi người tham gia; 25 features đã chọn (Kruskal–Wallis
  p < 0.1 ở ≥ 3 tasks) được trải phẳng theo 11 tasks theo thứ tự
  feature-major → 275 cột tên `{feature}_t{task}`.
- **Bài toán (3)** — số mẫu (người tham gia):
  - `co_vs_mci` — nhị phân, positive = MCI (141: CO 100 / MCI 41)
  - `co_vs_ad` — nhị phân, positive = AD (116: CO 100 / AD 16)
  - `co_mci_ad` — 3 lớp CO/MCI/AD (157: 100 / 41 / 16)
- **CV**: cấp người tham gia, phân tầng theo nhóm, 4 folds × 10 seeds (1–10)
  → **40 đánh giá train/test cho mỗi (bài toán, mô hình)**; tổng 720 đánh giá.
  Cách split: trong từng nhóm, hoán vị bằng `numpy.random.default_rng(seed)`
  và gán lần lượt vào các fold (cùng phương pháp với `fold/`, dùng seed 42).
- **Dữ liệu thiếu**: 402 / 43175 ô (0.93%) bị thiếu — 18/157 người tham gia
  có ≥ 1 ô thiếu, 45/275 cột bị ảnh hưởng (chủ yếu do `pupbl_*` của người
  thiếu baseline đồng tử, và 2 dòng task-10 all-NaN). **Impute median, fit
  theo từng fold trên tập train.** Chi tiết: `missingness.csv`.
- **Tiền xử lý**: `StandardScaler` trên 275 features, fit theo từng fold trên
  tập train (sau khi impute).
- **Mô hình** (hyperparameter cố định, không tuning):

| Mô hình | Hyperparameters |
|---|---|
| Logistic Regression | `max_iter=2000` |
| SVM RBF | `SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)` |
| SVM Linear | `SVC(C=1.0, kernel='linear', probability=True)` |
| FNN (MLP 64-32) | `MLPClassifier(hidden=(64,32), relu, adam, lr=1e-3, alpha=1e-4, batch=32, max_iter=300)` |
| QDA | `solver='eigen', shrinkage=0.5` (p=275 > n mỗi lớp; solver svd không khả thi) |
| GNB | `var_smoothing=1e-9` |

`random_state = seed` cho LR / SVM / FNN. QDA và GNB là tất định.

- **Metrics** (mỗi đánh giá, trên fold held-out):
  - `Acc` — độ chính xác.
  - `Sen` — nhị phân: recall của lớp positive; 3 lớp: recall macro.
  - `Spec` — nhị phân: TN/(TN+FP) của lớp âm; 3 lớp: specificity macro.
  - `AUC` — nhị phân: ROC-AUC của lớp positive; 3 lớp: OVR macro AUC.
  - `F1` — nhị phân: F1 của lớp positive; 3 lớp: F1 macro.
- **Metric chính**: AUC. **Mức tham chiếu ngẫu nhiên** (độ chính xác lớp đa
  số): CO vs MCI 0.709, CO vs AD 0.862, 3 lớp 0.637.

## 2. Kết quả — mean ± std trên 40 đánh giá (10 seeds × 4 folds)

### CO vs MCI (positive = MCI)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.6680±0.0631 | 0.4020±0.1537 | 0.7780±0.0915 | 0.6632±0.0902 | 0.4030±0.1275 |
| SVM RBF | 0.7143±0.0170 | 0.0250±0.0494 | 0.9970±0.0140 | 0.7334±0.0818 | 0.0435±0.0833 |
| SVM Linear | 0.6383±0.0713 | 0.4023±0.1566 | 0.7360±0.1000 | 0.6248±0.1028 | 0.3843±0.1249 |
| FNN (MLP 64-32) | 0.7384±0.0593 | 0.4286±0.1634 | 0.8660±0.0750 | 0.7052±0.0825 | 0.4747±0.1483 |
| QDA | 0.6981±0.0417 | 0.0857±0.0804 | 0.9490±0.0462 | 0.5678±0.0815 | 0.1345±0.1212 |
| GNB | 0.6534±0.0911 | 0.5748±0.1540 | 0.6860±0.1074 | 0.6462±0.1016 | 0.4909±0.1194 |

### CO vs AD (positive = AD)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.8698±0.0394 | 0.2687±0.2147 | 0.9660±0.0346 | 0.8518±0.0868 | 0.3309±0.2490 |
| SVM RBF | 0.8621±0.0000 | 0.0000±0.0000 | 1.0000±0.0000 | 0.8588±0.1026 | 0.0000±0.0000 |
| SVM Linear | 0.8733±0.0458 | 0.3563±0.2184 | 0.9560±0.0403 | 0.8578±0.0919 | 0.4166±0.2338 |
| FNN (MLP 64-32) | 0.8741±0.0439 | 0.2812±0.2414 | 0.9690±0.0430 | 0.7305±0.1617 | 0.3374±0.2677 |
| QDA | 0.8612±0.0055 | 0.0000±0.0000 | 0.9990±0.0063 | 0.6045±0.1742 | 0.0000±0.0000 |
| GNB | 0.8586±0.0702 | 0.5625±0.2640 | 0.9060±0.0699 | 0.7925±0.1447 | 0.5189±0.2214 |

### CO / MCI / AD (3 lớp)

| Model | Acc | Sen | Spec | AUC | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.6120±0.0772 | 0.4502±0.0941 | 0.7394±0.0523 | 0.6976±0.0688 | 0.4495±0.1027 |
| SVM RBF | 0.6415±0.0188 | 0.3429±0.0190 | 0.6746±0.0147 | 0.7560±0.0729 | 0.2810±0.0346 |
| SVM Linear | 0.5840±0.0682 | 0.4245±0.0899 | 0.7261±0.0477 | 0.6982±0.0807 | 0.4216±0.0952 |
| FNN (MLP 64-32) | 0.6561±0.0631 | 0.4558±0.0686 | 0.7568±0.0415 | 0.6807±0.0785 | 0.4525±0.0811 |
| QDA | 0.6300±0.0330 | 0.3448±0.0270 | 0.6754±0.0212 | 0.5677±0.0907 | 0.2971±0.0410 |
| GNB | 0.5593±0.0781 | 0.4485±0.1009 | 0.7450±0.0477 | 0.6560±0.0865 | 0.4397±0.1009 |

**AUC trung bình tốt nhất theo bài toán**: CO vs MCI — SVM RBF
(0.7334±0.0818); CO vs AD —
SVM RBF (0.8588±0.1026);
3 lớp — SVM RBF
(0.7560±0.0729).

## 3. Tái lập (reproducibility)

- Code: `src/baseline/run_baselines.py` (+ `aggregate_results.py`),
  config: `src/baseline/config.yaml`.
- `provenance.json`: git commit, sha256 của config, sha256 của file dữ liệu,
  phiên bản thư viện, máy chạy, câu lệnh.
- Chạy lại: `python src/baseline/run_baselines.py --config src/baseline/config.yaml`
  (từ chối ghi đè kết quả cũ), sau đó
  `python src/baseline/aggregate_results.py`.
- Ghi chú phát triển (cũng có trong `run_log.csv`): smoke run đầu tiên lỗi ở
  QDA với `solver='svd'` (p > n); đã khắc phục bằng `solver='eigen',
  shrinkage=0.5`.

## 4. Files

| File | Nội dung |
|---|---|
| `results/results_detail.csv` | 720 dòng, mỗi (bài toán, mô hình, seed, fold) kèm đủ 5 metrics |
| `results/summary.csv` | mean / std / n_runs theo (bài toán, mô hình, metric) |
| `results/smoke_detail.csv` | smoke run 6 dòng |
| `missingness.csv` | số ô thiếu theo cột của ma trận 275 features |
| `run_log.csv` | nhật ký chạy, gồm cả lần smoke QDA lỗi |
| `full_run.log` | stdout của full run |
| `provenance.json` | hồ sơ tái lập |

## 5. Hạn chế

- AD ít người tham gia (16 người) và các lớp mất cân bằng (CO 63.7–86.2%) →
  macro metrics và AUC là các chỉ số đáng tin cậy hơn.
- p (275) > n mỗi lớp → các mô hình bị chính quy hóa mạnh (QDA shrinkage);
  mọi mô hình đều có nguy cơ overfit tập train.
- Ô thiếu được impute median (18/157 người bị ảnh hưởng, 0.93% số ô).
- Không tuning hyperparameter — config cố định cho mọi mô hình.
- Một địa điểm, một phiên ghi (2025-03-05); không có kiểm định ngoài.
- 25 features được chọn trên cùng bộ dữ liệu (sàng lọc p < 0.1) — có thể có
  selection bias; kết quả mang tính thăm dò.
