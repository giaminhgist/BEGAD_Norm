# Baseline Cân Bằng — processed_dataset, Class Weights, Ngưỡng Cố Định 0.5

Thí nghiệm baseline trên processed_dataset (25 features) với xử lý mất cân
bằng lớp: class weight cân bằng / prior nghịch đảo tần suất. Không chỉnh
ngưỡng: dự đoán nhị phân dùng ngưỡng quyết định mặc định cố định 0.5
(`model.predict`). Ngày chạy: 2026-10-13. Chạy CPU cục bộ (không GPU / chi
phí tính toán).

## 1. Protocol

- **Mẫu**: một dòng mỗi người tham gia; 25 feature được gộp trên toàn bộ 11
  tasks theo thứ tự feature-major (cột `{feature}_t{task}`) → **275 cột**.
- **Bài toán (3)** — cỡ mẫu (người tham gia):
  - `co_vs_mci` — nhị phân, lớp dương = MCI (141: CO 100 / MCI 41)
  - `co_vs_ad` — nhị phân, lớp dương = AD (116: CO 100 / AD 16)
  - `co_mci_ad` — 3 lớp CO/MCI/AD (157: 100 / 41 / 16)
- **CV**: mức người tham gia, phân tầng theo nhóm, 5 folds × **20 seeds**
  (cố định trước khi xem kết quả: `[42, 2026, 1, 0, 7, 123, 1234, 2024,
  2025, 1337, 2, 3, 5, 10, 21, 777, 2020, 2021, 2022, 2023]`) →
  **100 lượt train/test cho mỗi (bài toán, mô hình)**; tổng 3 × 6 × 100 =
  **1800 lượt**.
- **Dữ liệu thiếu**: 402 / 43.175 ô (0,93%), 45 / 275 cột bị ảnh hưởng
  (`pupbl_*` của 16 người thiếu baseline đồng tử, và 2 dòng all-NaN task
  10). **Nội suy median theo từng fold chỉ trên tập train**, sau đó
  `StandardScaler` fit chỉ trên tập train.
- **Class weights** (`class_weight: balanced`): Logistic Regression, SVM
  RBF và SVM Linear dùng class weight cân bằng của scikit-learn; FNN dùng
  `sample_weight` cân bằng lúc fit (MLPClassifier không có tham số
  `class_weight`); QDA và GNB dùng **prior nghịch đảo tần suất** tính từ tập
  train của từng fold (dạng tương đương generative của weight cân bằng).
- **Ngưỡng quyết định** (`threshold: 0.5`): dự đoán nhị phân dùng ngưỡng
  mặc định cố định 0.5 (`model.predict`); không chỉnh ngưỡng. Bài toán 3
  lớp dùng argmax (không ngưỡng). AUC không bị ảnh hưởng bởi ngưỡng.
- **Mô hình** (siêu tham số cố định, không tuning): Logistic Regression
  (`max_iter=2000`), SVM RBF (`C=1, gamma='scale', probability=True`),
  SVM Linear (`C=1, probability=True`), FNN (MLP 64-32, relu, adam,
  lr=1e-3, alpha=1e-4, batch=32, max_iter=300), QDA (`solver='eigen',
  shrinkage=0,5`), GNB (`var_smoothing=1e-9`). `random_state = seed` cho
  LR / SVM / FNN.
- **Chỉ số** (mỗi lượt, trên fold giữ lại): Acc; Sen (độ nhạy của lớp
  bệnh: nhị phân = lớp dương, 3 lớp = macro recall); Spec (nhị phân:
  TN/(TN+FP); 3 lớp: trung bình TN/(TN+FP) từng lớp); Precision (nhị phân:
  lớp bệnh; 3 lớp: macro); AUC (nhị phân: ROC-AUC; 3 lớp: OVR macro); F1
  (nhị phân: lớp bệnh; 3 lớp: macro). Acc là độ chính xác tổng 3 lớp.
- **Chỉ số chính**: AUC. **Tham chiếu ngẫu nhiên** (độ chính xác lớp đa số):
  CO vs MCI 0,709; CO vs AD 0,862; 3 lớp 0,637; AUC ngẫu nhiên = 0,5.

## 2. Kết quả — trung bình ± std trên 100 lượt (20 seeds × 5 folds)

### co_vs_mci (lớp dương = MCI)

| Mô hình | Acc | AUC | Sen | Spec | Precision | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0,6572±0,0720 | 0,6361±0,0873 | 0,4472±0,1622 | 0,7430±0,0935 | 0,4222±0,1332 | 0,4244±0,1267 |
| svm_rbf | 0,7282±0,0706 | 0,7384±0,0945 | 0,4585±0,1819 | 0,8390±0,0863 | 0,5534±0,1645 | 0,4829±0,1492 |
| svm_linear | 0,6447±0,0736 | 0,5975±0,0968 | 0,4068±0,1721 | 0,7420±0,0937 | 0,3954±0,1389 | 0,3906±0,1363 |
| fnn | 0,7178±0,0763 | 0,6954±0,1013 | 0,4572±0,1590 | 0,8245±0,0886 | 0,5349±0,1787 | 0,4796±0,1429 |
| qda | 0,6962±0,0460 | 0,5535±0,0943 | 0,0932±0,0957 | 0,9435±0,0563 | 0,3493±0,3688 | 0,1401±0,1378 |
| gnb | 0,6301±0,0916 | 0,6429±0,1028 | 0,5910±0,1596 | 0,6455±0,1148 | 0,4141±0,1099 | 0,4808±0,1161 |

### co_vs_ad (lớp dương = AD)

| Mô hình | Acc | AUC | Sen | Spec | Precision | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0,8588±0,0539 | 0,8571±0,0996 | 0,3758±0,2886 | 0,9365±0,0517 | 0,4455±0,3286 | 0,3839±0,2636 |
| svm_rbf | 0,8619±0,0266 | 0,8428±0,0903 | 0,0558±0,1330 | 0,9910±0,0218 | 0,1300±0,3157 | 0,0750±0,1753 |
| svm_linear | 0,8687±0,0474 | 0,8638±0,0993 | 0,3483±0,2544 | 0,9520±0,0443 | 0,4977±0,3617 | 0,3861±0,2575 |
| fnn | 0,8569±0,0532 | 0,7141±0,1562 | 0,3067±0,2515 | 0,9450±0,0505 | 0,4257±0,3569 | 0,3374±0,2618 |
| qda | 0,8610±0,0158 | 0,5937±0,2230 | 0,0000±0,0000 | 0,9985±0,0086 | 0,0000±0,0000 | 0,0000±0,0000 |
| gnb | 0,8465±0,0819 | 0,7965±0,1592 | 0,5975±0,3004 | 0,8870±0,0751 | 0,4780±0,2719 | 0,5120±0,2552 |

### co_mci_ad (3 lớp, chỉ số macro)

| Mô hình | Acc | AUC | Sen | Spec | Precision | F1 |
|---|---|---|---|---|---|---|
| logistic_regression | 0,5915±0,0753 | 0,6982±0,0715 | 0,4519±0,1000 | 0,7408±0,0486 | 0,4567±0,1258 | 0,4440±0,1025 |
| svm_rbf | 0,6478±0,0735 | 0,7583±0,0706 | 0,4588±0,0764 | 0,7755±0,0495 | 0,4359±0,1201 | 0,4364±0,0828 |
| svm_linear | 0,5736±0,0751 | 0,6913±0,0757 | 0,4265±0,0937 | 0,7208±0,0492 | 0,4374±0,1316 | 0,4212±0,1010 |
| fnn | 0,6471±0,0709 | 0,7008±0,0865 | 0,4751±0,0916 | 0,7650±0,0507 | 0,4815±0,1193 | 0,4681±0,0970 |
| qda | 0,6203±0,0447 | 0,5645±0,1068 | 0,3411±0,0347 | 0,6735±0,0278 | 0,3105±0,1129 | 0,2955±0,0479 |
| gnb | 0,5685±0,0912 | 0,6575±0,0889 | 0,4697±0,1081 | 0,7564±0,0557 | 0,4763±0,1223 | 0,4574±0,1053 |

### AUC trung bình tốt nhất theo bài toán

| Bài toán | Mô hình tốt nhất | AUC |
|---|---|---|
| co_vs_mci | SVM RBF | 0,7384±0,0945 |
| co_vs_ad | SVM Linear | 0,8638±0,0993 |
| co_mci_ad | SVM RBF | 0,7583±0,0706 |

## 3. Ghi chú

- **Với ngưỡng cố định 0.5, các mô hình có xác suất hiệu chỉnh tốt vẫn cho
  nhãn cứng có ý nghĩa**: SVM RBF đạt Sen 0,459 và F1 0,483 ở co_vs_mci;
  GNB có Sen nhị phân cao nhất (0,591 ở co_vs_mci, 0,598 ở co_vs_ad) đổi
  lấy Spec thấp hơn (0,646 / 0,887). Dự đoán nhị phân của QDA vẫn sụp đổ ở
  co_vs_ad (Sen = 0).
- **Chỉ số 3 lớp là trung bình macro**: Sen/Precision/Spec/F1 được tính
  trung bình trên ba lớp và AUC là OVR macro; không lớp nào được chỉ định
  riêng làm lớp bệnh (khác với bài toán nhị phân). Acc là độ chính xác
  tổng 3 lớp.
- **AUC không phụ thuộc ngưỡng**; class weight và prior hầu như không dịch
  thứ hạng. Nghiệm của SVM Linear bất biến đúng bằng 0 với class weight ở
  đây (các fold train p > n đã chuẩn hóa phân tách tuyến tính, nơi nghiệm
  max-margin không phụ thuộc class weight); AUC nhị phân của QDA/GNB cũng
  không đổi vì prior chỉ dịch điểm số một hằng số, bảo toàn thứ hạng ROC.
- Độ chính xác nằm trong khoảng một độ lệch chuẩn quanh mức ngẫu nhiên lớp
  đa số ở mọi bài toán; AUC là chỉ số có ý nghĩa. Độ lệch chuẩn lớn phản
  ánh fold kiểm tra nhỏ (23–33 người) và lớp AD nhỏ (16 người).
- 25 feature được sàng lọc trên chính cohort này (Kruskal–Wallis, p < 0,1 ở
  ≥ 3 tasks), nên hiệu năng CV mang thiên lệch chọn lọc trong mẫu.
- **Run Youden đã lưu trữ** (`outputs/baseline_balanced_youden/`) dùng chỉnh
  ngưỡng Youden theo từng fold. AUC và toàn bộ chỉ số 3 lớp giống hệt giữa
  hai run; chỉ các chỉ số nhãn cứng nhị phân (Acc/Sen/Spec/Precision/F1)
  thay đổi.

## 4. Khả năng tái lập

- Mã nguồn: `src/begad_norm/baseline/run_baselines.py`
  (+ `aggregate_results.py`), cấu hình:
  `src/begad_norm/baseline/config.yaml`.
- `outputs/baseline_balanced/provenance.json`: git commit, sha256 cấu hình,
  sha256 file dữ liệu, phiên bản gói (numpy 2.5.4, pandas 3.0.6,
  scikit-learn 1.9.1, scipy 1.18.1), máy chạy, lệnh, thời lượng. Sha256 cấu
  hình được ghi là của `config.yaml` tại thời điểm chạy lại. Sha256 file dữ
  liệu không đổi so với lần chạy đầu của thí nghiệm; protocol đã đổi sau
  lần chạy đầu (`recall` nay là `sen`; chỉ số 3 lớp quay lại macro sau một
  protocol trung gian AD-vs-rest; chỉnh ngưỡng Youden được thay bằng ngưỡng
  cố định 0.5), nên thí nghiệm đã được chạy lại. Các run đã lưu trữ:
  `outputs/baseline_balanced_macro/` (run đầu, chỉ số giống hệt, cột
  `recall`), `outputs/baseline_balanced_adpos/` (run trung gian đã thay thế
  với chỉ số 3 lớp dạng AD-vs-rest), `outputs/baseline_balanced_youden/`
  (run đã thay thế với chỉnh ngưỡng Youden theo fold; AUC và chỉ số 3 lớp
  giống hệt, chỉ số nhãn cứng nhị phân khác). AUC và toàn bộ chỉ số 3 lớp
  giống hệt giữa cả bốn run; chỉ số nhãn cứng nhị phân chỉ khác giữa run
  Youden và các run ngưỡng cố định 0.5.
- Chạy lại:
  `uv run python -m begad_norm.baseline.run_baselines --config src/begad_norm/baseline/config.yaml`
  (từ chối ghi đè kết quả cũ), sau đó
  `uv run python -m begad_norm.baseline.aggregate_results --detail outputs/baseline_balanced/results/results_detail.csv --summary outputs/baseline_balanced/results/summary.csv`.
- Đã chạy smoke (6 lượt mỗi lần) và hai mini-run kiểm chứng (6 lượt mỗi
  lần, bài toán 3 lớp) trước; các CSV chi tiết của chúng đã xóa sau khi
  dùng (`run_log.csv` có ghi).

## 5. File

| File | Nội dung |
|---|---|
| `outputs/baseline_balanced/results/results_detail.csv` | 1800 dòng, một dòng mỗi (bài toán, mô hình, seed, fold) với đủ 6 chỉ số |
| `outputs/baseline_balanced/results/summary.csv` | mean / std / n_runs theo (bài toán, mô hình, chỉ số) |
| `outputs/baseline_balanced/results/missingness.csv` | số ô thiếu theo cột |
| `outputs/baseline_balanced/run_log.csv` | nhật ký chạy (smoke + mini-run kiểm chứng + full) |
| `outputs/baseline_balanced/full_run.log` | stdout của full run |
| `outputs/baseline_balanced/provenance.json` | hồ sơ tái lập |
| `outputs/baseline_balanced_macro/` | run đầu đã lưu trữ (chỉ số giống hệt, cột `recall`) |
| `outputs/baseline_balanced_adpos/` | run trung gian đã thay thế với chỉ số 3 lớp dạng AD-vs-rest |
| `outputs/baseline_balanced_youden/` | run đã thay thế với chỉnh ngưỡng Youden theo fold |
