# Figure

Toàn bộ mã tạo figure và output figure được yêu cầu cho dự án này đều nằm
trong thư mục này.

| Đường dẫn | Nội dung |
|---|---|
| `code/` | script tạo figure có thể tái lập (một script mỗi figure; ghi rõ đường dẫn dữ liệu nguồn, định dạng xuất và kích thước trong phần đầu script) |
| `outputs/` | figure đã xuất (ưu tiên SVG/PDF, dùng PNG khi bắt buộc phải raster) |
| `Figure_text.md` | caption + mô tả của mọi figure; cập nhật mỗi khi có figure mới |

Quy ước:

- Mã figure chỉ đọc các file dữ liệu/kết quả thật (không bao giờ ghi cứng số
  liệu vào plot).
- File xuất đặt tên `fig_<chủ đề>.<ext>` và khớp với tên script
  (`code/fig_<chủ đề>.py` → `outputs/fig_<chủ đề>.svg`).
- Một figure được coi là xong khi bản xuất đã được kiểm tra trực quan và các
  chuỗi dữ liệu đã vẽ được đối chiếu với dữ liệu nguồn.
