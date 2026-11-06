# Bộ tài liệu dự án NEU-CLS

Bộ tài liệu này giải thích **dự án hoạt động thế nào**, **dữ liệu chạy qua những bước nào**, và **cách vận hành** dự án hằng ngày. `README.md` ở thư mục gốc là bản hướng dẫn nhanh và kết quả. Bộ tài liệu này đi sâu hơn: mỗi file giải thích một phần, có trích dòng code cụ thể.

## Đọc theo thứ tự nào?

| # | File | Nội dung | Đọc khi |
|---|---|---|---|
| 1 | [01_tong_quan.md](01_tong_quan.md) | Bài toán, kiến trúc, sơ đồ luồng tổng thể, các file phụ thuộc nhau ra sao | Đọc đầu tiên |
| 2 | [02_du_lieu.md](02_du_lieu.md) | `prepare_data.py`, `dataset.py`: chia dữ liệu, bỏ ảnh trùng, biến đổi ảnh thành tensor | Muốn hiểu ảnh đi vào mô hình thế nào |
| 3 | [03_mo_hinh.md](03_mo_hinh.md) | `model.py`: transfer learning, 5 mô hình, thay phần đầu, định dạng checkpoint | Muốn hiểu mô hình hoặc thêm mô hình mới |
| 4 | [04_huan_luyen.md](04_huan_luyen.md) | `train.py`: vòng lặp train, loss, optimizer, chọn `best.pt`, early stopping, resume | Trước khi train |
| 4a | [04a_khuon_mau_train.md](04a_khuon_mau_train.md) | `train.py` nhìn như một khuôn mẫu: 4 phần chuẩn, 5 bước cốt lõi, phần nào đổi khi làm bài khác | Muốn viết script train cho bài mới |
| 5 | [05_danh_gia_va_so_sanh.md](05_danh_gia_va_so_sanh.md) | `evaluate.py`, `compare_models.py`: chỉ số, confusion matrix, ảnh sai, bảng so sánh | Sau khi train |
| 6 | [06_gradcam.md](06_gradcam.md) | `gradcam.py`: mô hình "nhìn" vào đâu, cách tính từng bước | Muốn kiểm tra mô hình có học đúng lỗi không |
| 7 | [07_du_doan_va_onnx.md](07_du_doan_va_onnx.md) | `predict.py`, `export_onnx/`: dự đoán ảnh mới, xuất ONNX, mang sang máy khác | Khi đưa mô hình ra dùng |
| 8 | [08_van_hanh.md](08_van_hanh.md) | Quy trình chuẩn, các kịch bản thường gặp, checklist, xử lý sự cố | Khi vận hành, train lại, triển khai |
| 9 | [09_thuat_ngu.md](09_thuat_ngu.md) | Giải thích thuật ngữ (epoch, logits, F1, ...) | Tra cứu khi gặp từ lạ |
| 10 | [10_danh_gia_cau_truc.md](10_danh_gia_cau_truc.md) | Đánh giá cấu trúc dự án: điểm đã chuẩn, điểm nên cải thiện | Trước khi mở rộng hoặc sửa dự án |
| 11 | [11_so_do_quy_trinh.md](11_so_do_quy_trinh.md) | Sơ đồ toàn bộ quy trình A → H, từng bước nhỏ kèm số dòng code; checklist dùng làm khuôn mẫu cho dự án mới | Muốn nhìn tổng thể, hoặc bắt đầu một dự án mới |

## Quy ước trong tài liệu

- Lệnh viết cho **PowerShell**, chạy từ thư mục gốc `D:\Python_Practice\neu_cls`.
- `python` trong tài liệu nghĩa là `.venv\Scripts\python.exe` của dự án.
- `train.py:129` nghĩa là dòng 129 của file `train.py`. Số dòng đúng với code tại thời điểm viết tài liệu (03/11/2026). Code sửa sau này có thể lệch vài dòng.
- Các câu in ra màn hình trích **nguyên văn** (tiếng Anh) để bạn tìm được bằng Ctrl+F.

## Khi nào phải cập nhật bộ tài liệu này?

Khi đổi một trong các thứ sau, hãy sửa file tương ứng:

| Thay đổi | File cần sửa |
|---|---|
| Thêm/bớt thiết lập trong `config.py` | 01, và file của phần dùng thiết lập đó |
| Đổi cách chia dữ liệu hoặc augmentation | 02 |
| Thêm mô hình mới | 03, 08 |
| Đổi quy tắc chọn `best.pt` hoặc early stopping | 04, 04a |
| Thêm file kết quả mới | 01 (bảng file sinh ra), 05 hoặc 07 |
| Thêm tham số dòng lệnh | 08 (bảng lệnh) |
| Thêm/bớt/đổi thứ tự bước trong bất kỳ script nào, hoặc code lệch số dòng | 11 (sơ đồ và số dòng) |
