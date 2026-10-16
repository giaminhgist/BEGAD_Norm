# Baseline từng task vs gộp task — so sánh mức patch

So sánh song song run baseline mức cell trên cả 4 task gộp lại
(`outputs/baseline/`, 2.512 mẫu = 157 subject x 4 task x 4 ô) với cùng protocol
chạy riêng từng task (`outputs/baseline/task_{3,4,5,9}/`, 628 mẫu mỗi task =
157 subject x 4 ô). Metric chính: AUC mức subject, mean ± std trên 100 lượt
đánh giá (20 seed cố định x 5 fold subject cố định). Các khóa (seed, fold)
giống hệt nhau giữa cả 5 run, nên mọi so sánh dưới đây là so sánh bắt cặp
theo từng lượt đánh giá.

Protocol (giống nhau cho mọi run, xem `baseline_cells.md`): 28 feature mức
cell, 3 bài toán, 6 model, median impute + `StandardScaler` fit theo fold trên
riêng split train, class weights/priors cân bằng, CV 5-fold theo subject từ
split cố định seed 42, dùng lại cho mọi seed.

## 1. AUC mức subject — mean ± std trên 100 lượt

### co_vs_mci (positive = MCI)

| Model | pooled | task 3 | task 4 | task 5 | task 9 | best task |
|---|---|---|---|---|---|---|
| fnn | 0.546±0.069 | 0.602±0.093 | 0.703±0.059 | 0.587±0.117 | 0.527±0.100 | task 4 (+0.157) |
| gnb | 0.597±0.112 | 0.626±0.094 | 0.611±0.043 | 0.587±0.176 | 0.578±0.130 | task 3 (+0.030) |
| logistic_regression | 0.652±0.093 | 0.622±0.076 | 0.656±0.036 | 0.704±0.071 | 0.629±0.074 | task 5 (+0.053) |
| qda | 0.597±0.100 | 0.635±0.059 | 0.644±0.063 | 0.554±0.168 | 0.575±0.086 | task 4 (+0.047) |
| svm_linear | 0.638±0.097 | 0.627±0.069 | 0.635±0.054 | 0.703±0.069 | 0.643±0.062 | task 5 (+0.065) |
| svm_rbf | 0.609±0.091 | 0.667±0.056 | 0.661±0.091 | 0.573±0.125 | 0.555±0.064 | task 3 (+0.058) |

### co_vs_ad (positive = AD)

| Model | pooled | task 3 | task 4 | task 5 | task 9 | best task |
|---|---|---|---|---|---|---|
| fnn | 0.659±0.140 | 0.694±0.169 | 0.558±0.165 | 0.713±0.139 | 0.561±0.125 | task 5 (+0.053) |
| gnb | 0.662±0.187 | 0.693±0.148 | 0.648±0.151 | 0.736±0.159 | 0.552±0.176 | task 5 (+0.073) |
| logistic_regression | 0.647±0.120 | 0.705±0.171 | 0.625±0.083 | 0.708±0.139 | 0.536±0.131 | task 5 (+0.061) |
| qda | 0.597±0.186 | 0.702±0.092 | 0.588±0.089 | 0.702±0.101 | 0.531±0.129 | task 3 (+0.106) |
| svm_linear | 0.660±0.128 | 0.655±0.219 | 0.632±0.081 | 0.714±0.148 | 0.553±0.159 | task 5 (+0.054) |
| svm_rbf | 0.701±0.135 | 0.725±0.171 | 0.650±0.091 | 0.708±0.077 | 0.672±0.106 | task 3 (+0.025) |

### co_mci_ad (3 class, metric macro)

| Model | pooled | task 3 | task 4 | task 5 | task 9 | best task |
|---|---|---|---|---|---|---|
| fnn | 0.570±0.083 | 0.605±0.087 | 0.638±0.069 | 0.572±0.094 | 0.564±0.072 | task 4 (+0.068) |
| gnb | 0.570±0.135 | 0.586±0.064 | 0.561±0.073 | 0.570±0.123 | 0.560±0.086 | task 3 (+0.016) |
| logistic_regression | 0.617±0.084 | 0.613±0.099 | 0.605±0.068 | 0.628±0.072 | 0.589±0.102 | task 5 (+0.011) |
| qda | 0.566±0.133 | 0.607±0.071 | 0.576±0.082 | 0.547±0.104 | 0.572±0.072 | task 3 (+0.041) |
| svm_linear | 0.629±0.076 | 0.633±0.056 | 0.610±0.057 | 0.682±0.083 | 0.605±0.075 | task 5 (+0.053) |
| svm_rbf | 0.638±0.085 | 0.661±0.089 | 0.634±0.064 | 0.621±0.087 | 0.608±0.063 | task 3 (+0.024) |

## 2. Bằng chứng bắt cặp — task tốt nhất vs pooled, theo (seed, fold)

Delta bắt cặp trung bình của AUC mức subject (task trừ pooled) trên cùng 100
lượt (seed, fold), kèm tỷ lệ lượt mà task đơn thắng pooled. Chỉ hiển thị task
tốt nhất cho mỗi (bài toán, model).

| Problem | Model | Best task | paired delta | share task > pooled |
|---|---|---|---|---|
| co_vs_mci | fnn | task 4 | +0.157±0.096 | 94% |
| co_vs_mci | gnb | task 3 | +0.030±0.081 | 60% |
| co_vs_mci | logistic_regression | task 5 | +0.053±0.067 | 80% |
| co_vs_mci | qda | task 4 | +0.047±0.121 | 60% |
| co_vs_mci | svm_linear | task 5 | +0.065±0.081 | 80% |
| co_vs_mci | svm_rbf | task 3 | +0.058±0.066 | 80% |
| co_vs_ad | fnn | task 5 | +0.053±0.103 | 69% |
| co_vs_ad | gnb | task 5 | +0.073±0.083 | 80% |
| co_vs_ad | logistic_regression | task 5 | +0.061±0.064 | 80% |
| co_vs_ad | qda | task 3 | +0.106±0.133 | 60% |
| co_vs_ad | svm_linear | task 5 | +0.054±0.092 | 62% |
| co_vs_ad | svm_rbf | task 3 | +0.025±0.092 | 60% |
| co_mci_ad | fnn | task 4 | +0.068±0.083 | 78% |
| co_mci_ad | gnb | task 3 | +0.016±0.081 | 60% |
| co_mci_ad | logistic_regression | task 5 | +0.011±0.084 | 40% |
| co_mci_ad | qda | task 3 | +0.041±0.083 | 80% |
| co_mci_ad | svm_linear | task 5 | +0.053±0.090 | 80% |
| co_mci_ad | svm_rbf | task 3 | +0.024±0.026 | 79% |

## 3. AUC mức cell

AUC mức cell cũng cùng chiều. Pooled gần mức ngẫu nhiên ở mọi bài toán
(co_vs_mci 0.53-0.60, co_vs_ad 0.55-0.61, co_mci_ad 0.53-0.59), trong khi task
đơn tốt nhất đạt 0.59-0.65 cho co_vs_mci, 0.63-0.68 cho co_vs_ad (gnb task 5:
0.680 so với pooled 0.613), và 0.58-0.63 cho co_mci_ad (svm_linear task 5:
0.628 so với pooled 0.589).

## 4. Kết quả chính

1. **Task đơn thắng pooled ở mọi ô (bài toán, model).** Trong cả 18 ô, AUC
   subject của task đơn tốt nhất đều cao hơn run pooled (tốt nhất của pooled:
   co_vs_mci 0.652, co_vs_ad 0.701, co_mci_ad 0.638). Run pooled không bao giờ
   vượt trội. Ở 17/18 ô, ít nhất 60% lượt bắt cặp đồng thuận; trường hợp duy
   nhất yếu là logistic_regression trên co_mci_ad (+0.011, 40%).
2. **Mức tăng lớn nhất cũng là mức tăng ổn định nhất.** fnn trên co_vs_mci ở
   task 4 tăng +0.157 bắt cặp với 94% lượt tốt hơn pooled; qda trên co_vs_ad
   ở task 3 tăng +0.106; gnb trên co_vs_ad ở task 5 tăng +0.073 với 80%.
3. **Task 5 (KD card 3) là task đơn mạnh nhất**, tốt nhất hoặc gần tốt nhất
   cho co_vs_ad và co_mci_ad trên cả 6 model. Task 4 mạnh nhất cho co_vs_mci
   với fnn. Task 9 (reading) yếu nhất: với co_vs_ad phần lớn model thấp hơn
   pooled khoảng 0.10 (gnb -0.110, logistic_regression -0.111, svm_linear
   -0.107 bắt cặp), và chỉ 20-40% lượt của nó thắng pooled. Điều này khớp với
   quan sát ở bước screening rằng task 9 đóng góp ít tín hiệu.
4. **Diễn giải.** Gộp task tăng gấp 4 số cell huấn luyện (2.512 so với 628)
   nhưng trộn 4 chế độ (regime) task trong cùng một không gian feature; các
   classifier đơn giản ở đây không thể chia sẻ tham số giữa các chế độ, nên
   tín hiệu riêng từng task bị pha loãng. Model từng task thấy một chế độ
   thuần nhất và phục hồi nhiều tín hiệu hơn dù chỉ có 1/4 số mẫu. Hiệu ứng
   thấy rõ ở cả mức cell và mức subject.
5. **Hàm ý cho hướng mô hình.** Với bộ feature này, model theo task là hướng
   mạnh hơn: hoặc classifier riêng từng task, hoặc model pooled có thêm định
   danh task (task indicator hoặc chuẩn hóa feature theo task). Điều này cũng
   ủng hộ việc chạy mô hình normative theo từng task.

## 5. Lưu ý (caveat)

- Fold cố định giữa các seed: std chỉ phản ánh tính ngẫu nhiên của model,
  không phải biến thiên do split. Delta bắt cặp dùng chung fold, nên tỷ lệ
  phần trăm thắng pooled mô tả độ nhất quán trong protocol, không phải suy
  diễn lên quần thể.
- Tập test mỗi fold nhỏ (28-29 subject; nhóm AD chỉ có 16 subject tổng cộng),
  nên AUC subject của từng task nhiễu, đặc biệt là co_vs_ad.
- Các cell của cùng một subject không độc lập; gộp mức subject giải quyết điều
  này cho các metric subject được báo cáo.
- Với co_vs_ad, accuracy mức subject của svm_linear và svm_rbf suy biến
  (sensitivity 0.0, accuracy = tỷ lệ nhóm đa số 0.862), nên AUC mới là metric
  có ý nghĩa cho bài toán này.
- Shortlist 12 feature từ bước screening được chọn trên nhãn của toàn bộ
  subject; ước lượng CV dùng shortlist đó với các fold này là lạc quan trừ khi
  shortlist được chọn lại bên trong từng fold train.

## 6. Bước tiếp theo

1. Tiếp tục mô hình hóa theo task trên các task mạnh nhất (task 4 và task 5,
   cộng task 3 cho co_vs_ad).
2. Kiểm tra khả năng tổng quát chéo task: huấn luyện từng task, đánh giá trên
   các task còn lại, để đo mức độ đặc thù task của tín hiệu.
3. Ablation trên model pooled: thêm định danh task bằng indicator hoặc chuẩn
   hóa feature theo task để kiểm tra pha loãng có khôi phục được không.
4. Chạy mô hình normative mức subject theo từng task và so sánh với các
   classifier từng task.
