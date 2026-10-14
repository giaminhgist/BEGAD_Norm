# Chia fold — 5-fold cross-validation (cấp người tham gia)

## Cấu hình

| Thiết lập | Giá trị |
|---|---|
| Số fold | 5 |
| Đơn vị split | người tham gia (157 người) |
| Phân tầng (stratify) | theo nhóm (CO / MCI / AD) |
| Seed | 42 (NumPy `default_rng`) |
| Cách gán | trong từng nhóm: hoán vị ngẫu nhiên, gán lần lượt vào fold 1–5 |

## Files (trong `fold1/`)

- `fold_assignments.csv` — mỗi dòng một người tham gia:
  `participant_id, group, group_label, fold` (fold ∈ 1–5).
- `fold_summary.csv` — số người tham gia theo fold và nhóm.

| fold | n_total | n_CO | n_MCI | n_AD |
|---|---|---|---|---|
| 1 | 33 | 20 | 9 | 4 |
| 2 | 31 | 20 | 8 | 3 |
| 3 | 31 | 20 | 8 | 3 |
| 4 | 31 | 20 | 8 | 3 |
| 5 | 31 | 20 | 8 | 3 |

## Lưu ý khi dùng

- **Người tham gia là đơn vị độc lập**: mọi dòng của một người (mọi task của
  `processed_dataset/`) đều thuộc cùng một fold. Không được tách các dòng
  (participant, task) sang các fold khác nhau.
- Ghép theo `participant_id` để gắn fold vào bất kỳ bảng feature nào.
- Tỷ lệ nhóm được giữ nguyên ở mọi fold (split phân tầng).
- Split không xét bất kỳ cấu trúc nào khác ngoài nhóm (không dùng metadata
  khác).
- Các thí nghiệm baseline tạo split 5-fold theo từng seed bằng cùng phương
  pháp (`numpy.random.default_rng(seed)`, gán vòng tròn trong nhóm); CSV ở
  đây là split tham chiếu với seed 42. Split tham chiếu 4-fold cũ (seed 42)
  vẫn nằm trong `fold/`.
