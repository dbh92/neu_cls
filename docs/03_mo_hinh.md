# 3. Mô hình (`model.py`)

## 3.1. Các mô hình có sẵn

`MODEL_NAMES` ([model.py:5](../model.py#L5)) là danh sách tên hợp lệ. `train.py --model` chỉ chấp nhận các tên này.

| Tên | Họ | Tham số (6 lớp) | Pretrained | Ghi chú |
|---|---|---|---|---|
| `resnet18` | ResNet | 11,2 triệu | ImageNet | Mặc định, cân bằng tốc độ và độ chính xác |
| `resnet50` | ResNet | 23,5 triệu | ImageNet | Sâu hơn, chậm hơn khoảng 3 lần |
| `efficientnet_b0` | EfficientNet | 4,0 triệu | ImageNet | Ít tham số nhưng trên CPU không nhanh |
| `mobilenet_v3_large` | MobileNet | 4,2 triệu | ImageNet | Thiết kế cho máy yếu, nhanh nhất |
| `simple_cnn` | tự viết | ~0,4 triệu | Không | Đối chứng: học từ đầu |

## 3.2. Transfer learning: giữ thân, thay đầu

Một mô hình phân loại pretrained có 2 phần:

```
ảnh (3,224,224) ─► [ THÂN: các lớp tích chập ] ─► vector đặc trưng ─► [ ĐẦU: Linear ] ─► 1000 điểm số ImageNet
                     đã học trên ImageNet          (512, 1280, ...)
```

`build_model` ([model.py:39-62](../model.py#L39-L62)) tạo mô hình với trọng số ImageNet, rồi **thay lớp Linear cuối** bằng một lớp Linear mới có 6 đầu ra. Lớp mới khởi tạo ngẫu nhiên; phần thân giữ nguyên những gì đã học. Khi train, **toàn bộ** mô hình đều được cập nhật (fine-tune toàn bộ, không đóng băng thân), nhưng với learning rate nhỏ (`1e-4`) để không phá hỏng những gì thân đã học.

Mỗi họ mô hình đặt lớp cuối ở một chỗ khác nhau:

| Mô hình | Phần đầu gốc | Dòng code thay |
|---|---|---|
| ResNet18/50 | `model.fc = Linear(512 hoặc 2048 → 1000)` | `model.fc = nn.Linear(model.fc.in_features, 6)` |
| EfficientNet-B0 | `model.classifier = Sequential(Dropout, Linear(1280 → 1000))` | thay `classifier[1]`, giữ Dropout |
| MobileNetV3-Large | `model.classifier = Sequential(Linear(960→1280), Hardswish, Dropout, Linear(1280 → 1000))` | thay `classifier[3]` |

Gặp một mô hình lạ, hãy `print(model)` rồi xem những dòng cuối để biết lớp Linear cuối tên là gì.

Lần đầu tạo một mô hình pretrained, torchvision tải trọng số về `C:\Users\Wisol\.cache\torch\hub\checkpoints\`. Các lần sau đọc từ đó.

## 3.3. `SimpleCNN`: mô hình đối chứng

[model.py:8-36](../model.py#L8-L36). 4 khối `Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2`, số kênh 32 → 64 → 128 → 256, ảnh nhỏ dần 224 → 112 → 56 → 28 → 14. Sau đó `AdaptiveAvgPool2d(1)` lấy trung bình mỗi kênh, `Dropout(0.3)`, và `Linear(256 → 6)`.

Mô hình này không có trọng số pretrained. Mục đích là để thấy rõ transfer learning hơn được bao nhiêu (bài tập 1 trong README). Khi train nó nên tăng `LEARNING_RATE` lên `1e-3`.

## 3.4. Đầu ra của mô hình: logits

Mọi mô hình trả về **logits**: tensor `(batch, 6)`, mỗi số là "điểm số" chưa chuẩn hóa của một lớp, có thể âm hoặc rất lớn. Logits **chưa phải** xác suất:

- Lúc train, `nn.CrossEntropyLoss` nhận thẳng logits (bên trong nó đã tự làm softmax).
- Lúc đánh giá/dự đoán, code gọi `torch.softmax(outputs, dim=1)` để đổi thành xác suất có tổng bằng 1.
- File ONNX cũng xuất ra logits; `predict_onnx.py` tự tính softmax bằng numpy.

## 3.5. `gradcam_layer`: lớp mà Grad-CAM đọc

[model.py:65-68](../model.py#L65-L68) trả về lớp tích chập **cuối cùng** của mô hình:

| Mô hình | Lớp | Bản đồ đặc trưng (ảnh 224) |
|---|---|---|
| ResNet18 | `model.layer4` | `(512, 7, 7)` |
| ResNet50 | `model.layer4` | `(2048, 7, 7)` |
| EfficientNet-B0 | `model.features[-1]` | `(1280, 7, 7)` |
| MobileNetV3-Large | `model.features[-1]` | `(960, 7, 7)` |
| SimpleCNN | `model.features[-1]` | `(256, 14, 14)` |

Chi tiết cách dùng xem [06_gradcam.md](06_gradcam.md).

## 3.6. Checkpoint: file `.pt` chứa gì?

Checkpoint là một **dict** lưu bằng `torch.save`, không phải nguyên đối tượng mô hình:

| Khóa | `best.pt` | `last.pt` | Ý nghĩa |
|---|---|---|---|
| `model_name` | ✔ | ✔ | Ví dụ `"resnet18"`: để biết dựng kiến trúc nào |
| `class_names` | ✔ | ✔ | Danh sách lớp **đúng thứ tự nhãn** lúc train |
| `image_size` | ✔ | ✔ | Kích thước ảnh lúc train (224) |
| `epoch` | ✔ | ✔ | Epoch đã hoàn thành |
| `config` | ✔ | ✔ | Mọi thiết lập của `config.py` lúc train, vd `{"EPOCHS": 20, "LEARNING_RATE": 0.0001, ...}`. Checkpoint train trước 06/11/2026 không có khóa này |
| `model` | ✔ | ✔ | `state_dict`: toàn bộ trọng số |
| `optimizer` | | ✔ | Trạng thái AdamW (để resume) |
| `scheduler` | | ✔ | Trạng thái lịch learning rate |
| `best_score`, `best_val_loss` | | ✔ | Best hiện tại (để resume tiếp tục so sánh đúng) |
| `history` | | ✔ | Lịch sử các epoch |
| `epochs_without_improvement` | | ✔ | Bộ đếm early stopping |

Nhờ lưu `model_name`, `class_names`, `image_size` trong checkpoint, các script sau **không cần** biết bạn đã train thế nào: chỉ cần đưa đường dẫn `best.pt`.

Khóa `config` thì để **người** đọc: sau này bạn sửa `config.py`, vẫn biết được `best.pt` cũ đã train với thiết lập nào. Cách xem nhanh nhất là mở `outputs/<model>/config.json` bằng Notepad (cùng nội dung). Hoặc đọc từ checkpoint:

```
.venv\Scripts\python.exe -c "import torch; print(torch.load('outputs/resnet18/best.pt')['config'])"
```

### Nạp checkpoint

[`load_model_from_checkpoint(path, device)`](../model.py#L71-L77):

1. `torch.load` đọc dict.
2. `build_model(model_name, len(class_names), pretrained=False)` dựng **khung rỗng** cùng kiến trúc. Không cần tải trọng số ImageNet vì sắp nạp đè ngay.
3. `load_state_dict` nạp trọng số đã train.
4. `model.eval()`: tắt Dropout, BatchNorm dùng thống kê đã học. **Bắt buộc** khi dự đoán; quên bước này thì cùng một ảnh có thể cho kết quả khác nhau.

## 3.7. Thêm một mô hình mới

Ví dụ thêm `resnet34`:

1. Thêm tên vào `MODEL_NAMES`.
2. Trong `build_model`, thêm nhánh tạo mô hình và thay lớp cuối (với ResNet34 có thể thêm vào dict cùng `resnet18`/`resnet50`).
3. Trong `gradcam_layer`, trả về lớp tích chập cuối (ResNet34: `layer4`).
4. Chạy `python model.py` để kiểm tra output `(2, 6)`.
5. Train: `python train.py --model resnet34`.

Bỏ sót bước 3 thì Grad-CAM sẽ chọn sai lớp hoặc báo lỗi.

## 3.8. Kiểm tra nhanh

```powershell
.venv\Scripts\python.exe model.py
```

Với mỗi mô hình in: shape đầu ra (phải là `(2, 6)`), số tham số, và kiểu của lớp Grad-CAM.
