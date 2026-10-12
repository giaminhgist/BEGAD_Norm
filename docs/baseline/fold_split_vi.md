# Chia fold — 4-fold cross-validation (cấp người tham gia)

## Cấu hình

| Thiết lập | Giá trị |
|---|---|
| Số fold | 4 |
| Đơn vị split | người tham gia (157 người) |
| Phân tầng (stratify) | theo nhóm (CO / MCI / AD) |
| Seed | 42 (NumPy `default_rng`) |
| Cách gán | trong từng nhóm: hoán vị ngẫu nhiên, gán lần lượt vào fold 1–4 |

## Files

- `fold_assignments.csv` — mỗi dòng một người tham gia:
  `participant_id, group, group_label, fold` (fold ∈ 1–4).
- `fold_summary.csv` — số người tham gia theo fold và nhóm.

| fold | n_total | n_CO | n_MCI | n_AD |
|---|---|---|---|---|
| 1 | 40 | 25 | 11 | 4 |
| 2 | 39 | 25 | 10 | 4 |
| 3 | 39 | 25 | 10 | 4 |
| 4 | 39 | 25 | 10 | 4 |

## Lưu ý khi dùng

- **Người tham gia là đơn vị độc lập**: mọi dòng của một người (mọi task, mọi
  bộ feature — `processed_dataset/`, `finalized_dataset/`,
  `processed_dataset_v3/`) đều thuộc cùng một fold.
  Không được tách các dòng (participant, task) sang các fold khác nhau.
- Ghép theo `participant_id` để gắn fold vào bất kỳ bảng feature nào.
- Tỷ lệ nhóm được giữ nguyên ở mọi fold (split phân tầng).
- Split không xét bất kỳ cấu trúc nào khác ngoài nhóm (không dùng metadata khác).
