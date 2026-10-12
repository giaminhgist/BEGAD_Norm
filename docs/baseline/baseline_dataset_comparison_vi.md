# So sánh bộ feature — all72 vs selected25 vs extended39

Ngày phân tích: 2026-10-12. Nguồn: `outputs/baseline/comparison_3datasets/results_detail.csv`
(2160 lượt; xem `docs/baseline/baseline_experiment_vi.md`). So sánh dạng máy đọc:
`outputs/baseline/comparison_3datasets/dataset_comparison.csv`.

> Dataset `selected25` sau đó được đổi tên thành `finalized_dataset` và chọn
> làm bộ feature chính thức cho mọi thí nghiệm.

## 1. Phương pháp

- **Chỉ số**: AUC (chỉ số chính của thí nghiệm).
- **Thiết kế ghép cặp**: với một bài toán và một (seed, fold) cho trước, cả ba
  bộ feature được đánh giá trên đúng cùng một split train/test, nên so sánh
  theo cặp ở đơn vị (seed, fold) (40 đơn vị mỗi bài toán).
- **Gộp**: trên mỗi đơn vị, AUC được trung bình trên 6 mô hình, cho một quan
  sát ghép cặp cho mỗi bộ feature trên mỗi (seed, fold) — 40 quan sát mỗi bài
  toán.
- **Kiểm định**: Friedman trên 3 bộ feature (40 khối), tiếp theo là Wilcoxon
  signed-rank theo cặp trên 40 hiệu ghép cặp. Mang tính thăm dò; các đơn vị
  (seed, fold) chia sẻ dữ liệu huấn luyện giữa các seed, nên p-value hơi lạc
  quan.

## 2. Kết quả

### AUC gộp (trung bình trên 6 mô hình, 40 đơn vị seed-fold)

| Bài toán | all72 (792 cột) | selected25 (275 cột) | extended39 (429 cột) | Friedman p |
|---|---|---|---|---|
| co_vs_mci | 0,6307±0,0903 | 0,6668±0,0775 | 0,6718±0,0834 | < 0,0001 |
| co_vs_ad | 0,6613±0,1052 | 0,7715±0,1071 | 0,7504±0,1112 | < 0,0001 |
| co_mci_ad | 0,6316±0,0554 | 0,6750±0,0621 | 0,6773±0,0620 | < 0,0001 |

### AUC trung bình tốt nhất của từng bộ (mô hình bất kỳ, trung bình trên 40 lượt)

| Bài toán | all72 | selected25 | extended39 |
|---|---|---|---|
| co_vs_mci | 0,6845 (SVM RBF) | 0,7525 (SVM RBF) | **0,7609 (SVM RBF)** |
| co_vs_ad | 0,7602 (LR) | **0,8518 (SVM Linear)** | 0,8313 (SVM RBF) |
| co_mci_ad | 0,6977 (LR) | 0,7468 (SVM RBF) | **0,7484 (SVM RBF)** |

### Kiểm định Wilcoxon theo cặp (40 đơn vị ghép cặp mỗi bài toán)

| Bài toán | selected25 vs all72 | extended39 vs all72 | extended39 vs selected25 |
|---|---|---|---|
| co_vs_mci | +0,0360, p < 0,0001 | +0,0410, p < 0,0001 | +0,0050, p = 0,204 |
| co_vs_ad | +0,1101, p < 0,0001 | +0,0891, p < 0,0001 | −0,0210, p = 0,0007 |
| co_mci_ad | +0,0434, p < 0,0001 | +0,0457, p < 0,0001 | −0,0023, p = 0,618 |

Hiệu AUC ghép cặp trung bình; giá trị dương nghiêng về bộ đứng sau.

## 3. Diễn giải

- **all72 bị vượt trội rõ ràng.** Cả hai bộ đã sàng lọc đều thắng đáng kể ở cả
  ba bài toán (Friedman p < 0,0001; từng cặp p < 0,0001). QDA của nó suy giảm
  dưới ngưỡng ngẫu nhiên (AUC 0,45–0,49) và AUC tốt nhất không vượt 0,76.
- **selected25 so với extended39**:
  - co_vs_ad: selected25 tốt hơn đáng kể (p = 0,0007); cả 6 mô hình riêng lẻ
    đều nghiêng về selected25 ở bài toán này.
  - co_vs_mci và co_mci_ad: không khác biệt đáng kể (p = 0,204 và p = 0,618);
    extended39 nhỉnh hơn chút về trung bình gộp, chủ yếu nhờ SVM RBF và FNN.
- **Lưu ý vòng lặp chọn feature**: selected25 và extended39 được sàng lọc trên
  chính dataset này (Kruskal–Wallis trên toàn cohort, gồm cả người trong fold
  kiểm tra), nên hiệu năng CV của chúng bị thổi phồng so với all72 — bộ duy
  nhất không có chọn lọc trong mẫu. Khoảng cách giữa all72 và hai bộ sàng lọc
  vì thế bị phóng đại; so sánh công bằng cần chọn feature lồng (sàng lọc trong
  từng fold huấn luyện). So sánh 25-vs-39 ít bị ảnh hưởng hơn (cả hai đều chọn
  trong mẫu, 25 là tập con của 39).

## 4. Khuyến nghị

**Dùng bộ 25 feature (275 cột), nay mang tên `finalized_dataset`, làm bộ
feature chính.** Lý do:

- Ngang bằng thống kê với extended39 ở 2/3 bài toán và tốt hơn đáng kể ở
  co_vs_ad.
- Số chiều thấp hơn (275 so với 429 cột): ít áp lực overfit trong chế độ
  p > n, tập gọn và dễ diễn giải hơn.
- Được sàng lọc ở ngưỡng chặt hơn (p < 0,1 so với p < 0,2), nên chứa các
  feature phân biệt vững hơn; extended39 bổ sung chủ yếu các ứng viên yếu.
- extended39 vẫn là lựa chọn thay thế hợp lý nếu trọng tâm riêng là co_vs_mci
  hoặc bài toán 3 lớp, nơi nó có lợi thế danh nghĩa nhỏ, không đáng kể. Không
  nên dùng all72 trực tiếp cho phân loại nếu không giảm chiều hoặc chọn
  feature bên trong vòng CV.
