# 10. Đánh giá cấu trúc dự án

Đánh giá ngày 06/11/2026. Câu hỏi đặt ra: cấu trúc huấn luyện một mô hình và các file thành phần của dự án đã chuẩn chưa?

**Kết luận:** Cấu trúc này đúng chuẩn. Nó gần như giống hệt bố cục mà các dự án PyTorch phân loại ảnh hay dùng (config / dataset / model / train / evaluate / predict). Chưa thấy lỗi logic nào làm sai kết quả.

## Từng file

| File | Vai trò | Đánh giá |
|---|---|---|
| [config.py](../config.py) | Toàn bộ thiết lập | Chuẩn. Thiết lập nằm một chỗ, chỉ những thứ đổi theo từng lần chạy mới là tham số dòng lệnh. |
| [prepare_data.py](../prepare_data.py) | Chia train/val/test | **Tốt hơn mức thường gặp**: lọc ảnh trùng bằng MD5, chặn ảnh test lọt vào train, seed cố định, không cho ghi đè dữ liệu cũ. |
| [dataset.py](../dataset.py) | Dataset, transform, DataLoader | Chuẩn. Augmentation chỉ áp cho train, chuẩn hóa theo ImageNet, thứ tự lớp cố định. |
| [model.py](../model.py) | Tạo mô hình, nạp checkpoint | Chuẩn. Thay đúng phần đầu phân loại cho từng họ mô hình. Khi nạp lại thì dựng mô hình theo tên và danh sách lớp lưu trong checkpoint. |
| [train.py](../train.py) | Vòng lặp huấn luyện | Chuẩn: `train()`/`eval()`, `zero_grad → backward → step`, scheduler, `best.pt` + `last.pt`, resume, early stopping. |
| [evaluate.py](../evaluate.py) | Đánh giá, hàm dùng chung | Chuẩn: `no_grad`, classification report, confusion matrix, danh sách ảnh sai sắp theo độ tự tin. |
| [predict.py](../predict.py) | Dự đoán ảnh mới | Chuẩn. Dùng lại đúng `build_transforms(train=False)` và `decide_predictions`, nên không bị lệch so với lúc đánh giá. |

Những điểm "chuẩn" quan trọng nhất đều đã có:

- Tập test tách riêng, không dùng để chọn epoch.
- Checkpoint lưu kèm `class_names` và `image_size`.
- Tiền xử lý lúc train, đánh giá và dự đoán dùng chung một hàm.

## Điểm nên cải thiện (xếp theo mức quan trọng)

### 1. Tập test đã quá dễ, không còn so sánh được mô hình

Trong `outputs/compare_test.csv`, cả 4 mô hình đều đạt accuracy 1.0 và sai 0 ảnh:

| model | best_epoch | accuracy | macro_f1 | wrong | params (triệu) | file (MB) | ms/ảnh |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_large | 15 | 1.0 | 1.0 | 0 | 4.21 | 16.2 | 15.5 |
| resnet18 | 14 | 1.0 | 1.0 | 0 | 11.18 | 42.7 | 18.1 |
| efficientnet_b0 | 14 | 1.0 | 1.0 | 0 | 4.02 | 15.6 | 23.3 |
| resnet50 | 18 | 1.0 | 1.0 | 0 | 23.52 | 90.0 | 45.3 |

Vì vậy câu hỏi "mô hình nào tốt hơn" hiện chỉ trả lời được bằng tốc độ và dung lượng, chưa trả lời được bằng độ chính xác. Hơn nữa mỗi mô hình mới train một lần với một seed. Nếu muốn so sánh có ý nghĩa, có thể:

- train mỗi mô hình với 3 seed khác nhau rồi xem độ dao động;
- làm bài khó hơn, ví dụ chỉ cho train 10–20% số ảnh, hoặc thêm nhiễu, mờ, đổi sáng vào ảnh test;
- tốt nhất là thử trên ảnh thật ở nhà máy.

### 2. Checkpoint chưa lưu siêu tham số (✔ đã sửa ngày 06/11/2026)

`best.pt` không ghi lại LR, số epoch hay augmentation lúc train. Sau này sửa [config.py](../config.py) thì sẽ không biết `outputs/resnet18` đã được train với thiết lập nào. Cách làm phổ biến là lưu một bản sao config vào thư mục output hoặc vào checkpoint.

**Đã sửa:** `train.py` giờ lưu mọi thiết lập vào khóa `config` của checkpoint và ghi ra `outputs/<model>/config.json`. Khi `--resume` mà `config.py` đã bị sửa, code in cảnh báo cho từng thiết lập bị đổi. Xem [03_mo_hinh.md](03_mo_hinh.md#36-checkpoint-file-pt-chứa-gì) và [04_huan_luyen.md](04_huan_luyen.md). Lưu ý: 4 mô hình hiện có trong `outputs/` được train trước khi sửa nên chưa có `config`; phải train lại thì mới có.

### 3. Val F1 bão hòa nên tiêu chí chọn epoch thực chất là val loss

Tập val chỉ có khoảng 216 ảnh và F1 = 1.0 ở gần như mọi epoch, nên quy tắc "F1 trước, val loss sau" hiện vận hành như "chọn theo val loss". Cách này không sai, và là lựa chọn hợp lý cho NEU. Chỉ cần hiểu rằng với dữ liệu khó hơn thì F1 mới thật sự quyết định.

### 4. Những điểm nhỏ, không bắt buộc

- **Góc đen khi xoay:** `RandomRotation(10)` tô đen bốn góc ảnh, mà chỉ ảnh train mới có góc đen. Với NEU thì vô hại. Nếu muốn sạch hơn thì có thể dùng tham số `fill` hoặc xoay 90°/180°.
- **LR chung:** backbone và phần đầu phân loại dùng cùng một LR. Nhiều dự án đặt LR của phần đầu lớn hơn, nhưng với NEU thì không cần.
- **Chưa có test tự động** kiểm tra các phần phải khớp nhau (ví dụ `preprocess` của ONNX với `build_transforms`). Hiện [export_onnx/export.py](../export_onnx/export.py) đã tự kiểm chứng phần ONNX, nên mức này là đủ cho một bài luyện tập.
