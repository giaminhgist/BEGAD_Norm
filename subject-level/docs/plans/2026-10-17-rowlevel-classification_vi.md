# Kế hoạch: Phân loại row-level trên T1235_clean và T5_clean (5 model ML + torch multimodal)

Ngày: 2026-10-17. Trạng thái: đã duyệt và đang thực hiện.

## Mục tiêu

Phân loại CO / MCI / AD từ hai dataset eye-tracking row-level đã làm sạch
`T1235_clean.xlsx` (12.486 row, 23 cột, task 1/2/3/5) và `T5_clean.xlsx`
(6.217 row, 20 cột, task 5) với 6 model: logistic regression, SVM-RBF,
random forest, LightGBM, FNN (sklearn), và bản port PyTorch của model
multimodal trong `subject-level/Multimodal.ipynb`.

## Protocol

- Bài toán: co_vs_mci (positive MCI), co_vs_ad (positive AD), co_mci_ad
  (3 lớp).
- Seed split cố định 42; 10 folds. `grouped`: chia stratified round-robin
  theo participant (reuse `stratified_folds` của baseline), đánh giá ở mức
  row và subject (subject = trung bình xác suất dự đoán theo participant,
  ngưỡng 0.5 / argmax). `random`: `StratifiedKFold(10, shuffle,
  random_state=42)` ở mức row theo y, chỉ đánh giá row (arm cố ý leaky để
  định lượng mức rò rỉ).
- 5 model seeds [42, 2026, 1, 0, 7] cho khởi tạo RF / LGBM / FNN / torch.
  LR và SVM-RBF tất định nên 5 lần lặp giống hệt nhau (std = 0 trong báo
  cáo).
- Metrics: acc, sen, precision, spec, auc, f1 (ngữ nghĩa như baseline).
  Impute median và chuẩn hóa theo từng fold, chỉ fit trên train; class
  weight balanced theo row counts cho mọi model ML; model torch không dùng
  class weight (trung thành với notebook).
- Model torch: ba nhánh (visual 7d, speech 5d, pupil 7d cho T5; pupil 10d
  cho T1235 khi thêm maxPupLB/minPupLB/diffXpupL; ROI là cột cuối mỗi
  nhánh), Dense 128-64 với L2 1e-3 và dropout 0.2, fusion 192-128-64,
  softmax 2 unit cho bài toán nhị phân và 3 cho 3 lớp. Adam lr 1e-3,
  batch 32, tối đa 300 epochs, early stopping patience 15 trên tập val
  tách từ train (disjoint theo participant với split grouped).
- Khối lượng mỗi dataset: ML 3 bài toán x 5 model x 5 seeds x 10 folds x
  3 cấu hình đánh giá = 2.250 detail rows; torch 3 x 5 x 10 x 3 = 450;
  tổng 2.700. Số lần huấn luyện torch mỗi dataset = 300.

## Outputs (folder từng dataset ở repo root)

`T1235/` và `T5/`: `code/` (runner, torch model, config, aggregate và
report scripts), `results/results_detail.csv` (2.700 row),
`results/summary.csv` (324 row), `results/fit_times.csv`,
`results/missingness.csv`, `folds/fold_info.csv` (seed 42; grouped: 640
dòng participant, problem = all; random: các row test theo problem), cùng
`provenance.json`, `run_log.csv`, và `report.md` / `report_vi.md`.

## Thứ tự thực hiện

Smoke T1235 -> full T1235 -> aggregate -> báo cáo T1235 (giao ngay) ->
smoke T5 -> full T5 -> aggregate -> báo cáo T5.

## Xác minh

Chạy toàn bộ pytest; invariant sau mỗi run (2.700 detail rows; 50 mỗi
problem, model, split, eval level; 324 summary rows không NaN; schema
fold-info; sha256 provenance khớp config và data; spot-check tất định một
task). Báo cáo luôn có bảng AUC / Sen / Spec, song ngữ EN + VI.
