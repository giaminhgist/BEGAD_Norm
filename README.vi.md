# BEGAD_Norm

Phân tích eye-tracking bộ test BEGAD (CN / MCI / AD). Pipeline subject-level cũ, baseline và mô hình normative
nằm trong `subject-level/` (có README riêng). Package ở cấp gốc chứa phân tích feature discovery.

## Feature discovery (sàng lọc thống kê, không train classifier)

Triển khai `Running_Phase/BEGAD_feature_selection_pack/`: registry kết hợp 160 định nghĩa gốc và các định nghĩa D
lấy từ dataset, trích xuất có audit, sàng lọc Mann–Whitney theo người cho CN_AD, CN_MCI và MCI_AD, hiệu chỉnh
BH/BY, bootstrap theo người, bỏ từng người AD, kiểm định hoán vị, chọn macro và xếp hạng task–ROI.

Một lệnh tái lập (Python 3.10, `uv`):

```bash
uv sync
uv run python -m begad_norm.feature_discovery.cli all \
  --config configs/feature_discovery.yaml \
  --run-dir outputs/feature_discovery/2026-10-27_v2_expanded260
```

Có thể chạy từng stage (`audit`, `propose`, `freeze`, `extract`, `stats`, `stability`, `report`); `stats` từ chối
chạy nếu catalogue, cấu hình hoặc code trích xuất đã đổi sau `freeze`. Cấu hình khác thì dùng `--run-dir` mới;
`--resume` bỏ qua các stage đã xong của cùng một run đã freeze.

Đầu vào (chỉ đọc): `T1235.xlsx`, `T5.xlsx` (workbook ROI gốc; các file `*_clean.xlsx` chỉ được kiểm kê, không dùng
vì có giá trị được impute), `dataset/participants_20250305.csv` và `dataset/20250305/*.csv`.

Kết quả trong thư mục run: `RESULTS_VI.md` (memo tiếng Việt) và `RESULTS_EN.md`, `macro_selected_main.csv`,
`roi_top20.csv`, mọi bảng trong file 04 của pack, `figures/`, `semantic_audit.md`, `candidate_freeze.json`.

Test: `uv run pytest tests/feature_discovery -q`.
