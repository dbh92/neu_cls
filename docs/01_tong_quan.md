# 1. Tổng quan dự án

## 1.1. Bài toán

Cho một ảnh xám 200×200 chụp bề mặt thép cán nóng, xác định ảnh đó thuộc **1 trong 6 loại lỗi**:

| Chỉ số | Lớp | Nghĩa |
|---|---|---|
| 0 | `crazing` | Nứt chân chim |
| 1 | `inclusion` | Tạp chất lẫn vào |
| 2 | `patches` | Mảng loang |
| 3 | `pitted_surface` | Rỗ bề mặt |
| 4 | `rolled-in_scale` | Vảy cán ép vào |
| 5 | `scratches` | Vết xước |

Chỉ số là **thứ tự alphabet của tên thư mục**. Mô hình chỉ trả ra con số 0–5; danh sách tên ở trên được lưu kèm trong checkpoint để đổi số thành tên.

Đây là bài **phân loại nhiều lớp** (multi-class): mỗi ảnh có đúng một lớp, và luôn chọn lớp có xác suất cao nhất. Khác bài Magnetic Tile trước, ở đây **không có lớp "OK"** và **không có ngưỡng NG**.

## 1.2. Ý tưởng chính: transfer learning

Thay vì cho mô hình học từ con số 0 (như bản CNN NumPy cũ, đạt 0.942), dự án lấy các mô hình **đã học trên 1,2 triệu ảnh ImageNet**. Các mô hình này đã biết nhận ra cạnh, vân, đốm, vệt. Ta chỉ:

1. giữ nguyên phần "thân" đã học (backbone);
2. thay lớp cuối 1000 lớp ImageNet bằng lớp mới 6 lớp;
3. train tiếp toàn bộ mô hình với learning rate nhỏ (fine-tune).

Kết quả: chỉ sau 1–2 epoch, độ chính xác trên val đã đạt 96–100%.

Dự án còn **so sánh nhiều mô hình** (ResNet18, ResNet50, EfficientNet-B0, MobileNetV3-Large, và một CNN tự viết) trên cùng một tập test, để chọn mô hình hợp lý nhất cho triển khai.

## 1.3. Sơ đồ luồng tổng thể

```
  D:\Huan\...\NEU-DET (ảnh gốc, CHỈ ĐỌC)
        │  train/images/<lớp>/      validation/images/<lớp>/
        ▼
 ┌──────────────────┐
 │ prepare_data.py  │  bỏ ảnh trùng (MD5), chia train 85% / val 15%, validation gốc -> test
 └────────┬─────────┘
          ▼
  data/split/{train,val,test}/<lớp>/*.jpg
          │
          │  (dataset.py: đọc ảnh -> tensor, augmentation cho train)
          ▼
 ┌──────────────────┐     model.py: build_model()
 │    train.py      │◄─── tạo mô hình pretrained, thay phần đầu 6 lớp
 └────────┬─────────┘
          ▼
  outputs/<model>/best.pt, last.pt, config.json, history.csv, training_curves.png
          │
          ├──────────────► evaluate.py      -> confusion_matrix_test.png, misclassified_test.csv/.png
          │                     │
          │                     ▼
          ├──────────────► gradcam.py       -> outputs/<model>/gradcam/gradcam_01.png
          │
          ├──────────────► compare_models.py (đọc MỌI outputs/*/best.pt) -> outputs/compare_test.csv
          │
          ├──────────────► predict.py       -> dự đoán ảnh bất kỳ (cần PyTorch)
          │
          └──────────────► export_onnx/export.py -> export_onnx/model.onnx + model.json
                                                          │
                                                          ▼
                                       export_onnx/predict_onnx.py (KHÔNG cần PyTorch)
                                       -> mang 3 file sang máy chuyền
```

## 1.4. Các file và vai trò

| File | Loại | Vai trò | Chạy trực tiếp để |
|---|---|---|---|
| [config.py](../config.py) | Thiết lập | **Nơi duy nhất** chứa thiết lập | (không chạy) |
| [prepare_data.py](../prepare_data.py) | Script | Sao chép và chia dữ liệu | Tạo `data/split` |
| [dataset.py](../dataset.py) | Module | Dataset, transforms, DataLoader | Kiểm tra số ảnh và shape một batch |
| [model.py](../model.py) | Module | Tạo mô hình, nạp checkpoint, chọn lớp cho Grad-CAM | Kiểm tra 5 mô hình chạy được |
| [train.py](../train.py) | Script | Huấn luyện một mô hình | Train |
| [evaluate.py](../evaluate.py) | Script **và** module | Hàm dùng chung `run_inference`, `decide_predictions`, `compute_metrics` | Đánh giá chi tiết một mô hình |
| [compare_models.py](../compare_models.py) | Script | So sánh mọi mô hình đã train | Bảng so sánh |
| [gradcam.py](../gradcam.py) | Script | Bản đồ nhiệt "mô hình nhìn vào đâu" | Giải thích dự đoán |
| [predict.py](../predict.py) | Script **và** module | Dự đoán ảnh bất kỳ | Dùng mô hình |
| [export_onnx/export.py](../export_onnx/export.py) | Script | Xuất ONNX và kiểm chứng | Chuẩn bị triển khai |
| [export_onnx/predict_onnx.py](../export_onnx/predict_onnx.py) | Script **và** module | Dự đoán bằng ONNX, không cần PyTorch | Chạy trên máy chuyền |

### Ai import ai?

```
config.py  ◄── hầu hết mọi file (trừ model.py và predict_onnx.py)

dataset.py ◄── train, evaluate, predict, compare_models, gradcam, export
model.py   ◄── train, evaluate, predict, compare_models, gradcam, export
evaluate.py ◄── train, predict, compare_models, export      (dùng lại 3 hàm chung)
predict.py  ◄── gradcam                                      (collect_image_paths, true_label_from_path)
predict_onnx.py ◄── export                                   (OnnxClassifier)
```

Hệ quả quan trọng:

- **Quy tắc ra kết luận** (`decide_predictions` trong `evaluate.py`) chỉ viết một lần, và `train.py`, `predict.py`, `compare_models.py` cùng dùng. Như vậy lúc train chấm điểm thế nào thì lúc dùng cũng kết luận y như thế.
- **Tiền xử lý ảnh** (`build_transforms` trong `dataset.py`) chỉ viết một lần cho PyTorch. Riêng `predict_onnx.py` viết lại bằng numpy (vì không được import torch), và `export.py` kiểm chứng hai bản cho kết quả giống nhau.
- `evaluate.py` chỉ gọi `matplotlib.use("Agg")` bên trong `if __name__ == '__main__':`. Nếu đặt ở đầu file, `predict.py --show` (import `evaluate`) sẽ không mở được cửa sổ hình.

## 1.5. Thiết lập (`config.py`)

Mọi con số có thể chỉnh đều nằm ở [config.py](../config.py). Tham số dòng lệnh chỉ dùng cho những thứ **thay đổi theo từng lần chạy** (`--model`, `--checkpoint`, `--split`, ...).

| Nhóm | Thiết lập | Giá trị | Ảnh hưởng tới |
|---|---|---|---|
| Đường dẫn | `NEU_DIR` | `D:/Huan/2.Python/learning/neu_det/datasets/NEU-DET` | `prepare_data.py` |
| | `SPLIT_DIR` | `data/split` | mọi script đọc ảnh |
| | `OUTPUT_DIR` | `outputs` | nơi lưu kết quả |
| Chia dữ liệu | `VAL_RATIO` | `0.15` | tỉ lệ val |
| | `SEED` | `42` | cách chia, khởi tạo ngẫu nhiên khi train |
| Ảnh | `IMAGE_SIZE` | `224` | kích thước đưa vào mô hình |
| | `MEAN`, `STD` | của ImageNet | chuẩn hóa ảnh |
| Mô hình | `MODEL_NAME` | `resnet18` | mô hình mặc định khi không có `--model`/`--checkpoint` |
| | `PRETRAINED` | `True` | dùng trọng số ImageNet |
| Train | `EPOCHS` | `20` | số epoch tối đa |
| | `BATCH_SIZE` | `16` | số ảnh mỗi lần cập nhật |
| | `LEARNING_RATE` | `1e-4` | tốc độ học |
| | `WEIGHT_DECAY` | `1e-4` | chống overfit |
| | `NUM_WORKERS` | `4` | số tiến trình đọc ảnh song song |
| | `USE_CLASS_WEIGHTS` | `True` | cân bằng loss giữa các lớp |
| | `EARLY_STOPPING_PATIENCE` | `7` | tự dừng sau 7 epoch không có best mới |
| Augmentation | `HORIZONTAL_FLIP`, `VERTICAL_FLIP` | `True` | lật ngẫu nhiên |
| | `ROTATION_DEGREES` | `10` | xoay ±10° |
| | `BRIGHTNESS`, `CONTRAST` | `0.2` | đổi sáng/tương phản ±20% |

> Lưu ý: `MODEL_NAME` là mặc định **cho cả** `evaluate.py`, `predict.py`, `gradcam.py`, `export.py` (đường dẫn mặc định `outputs/<MODEL_NAME>/best.pt`). Nếu bạn train `mobilenet_v3_large` mà không đổi `MODEL_NAME`, nhớ thêm `--checkpoint outputs/mobilenet_v3_large/best.pt`.

## 1.6. Những file được sinh ra

Không sửa tay các file này. Tất cả đều nằm trong `.gitignore` và tạo lại được bằng script.

| File | Do script nào tạo | Nội dung |
|---|---|---|
| `data/split/{train,val,test}/<lớp>/*.jpg` | `prepare_data.py` | Bản sao ảnh đã chia |
| `outputs/<model>/best.pt` | `train.py` | Mô hình tốt nhất theo val (dùng cho mọi bước sau) |
| `outputs/<model>/last.pt` | `train.py` | Epoch gần nhất + trạng thái optimizer, để `--resume` |
| `outputs/<model>/config.json` | `train.py` | Mọi thiết lập của `config.py` lúc train (LR, số epoch, augmentation...), ghi một lần khi bắt đầu train |
| `outputs/<model>/history.csv` | `train.py` | Mỗi epoch một dòng: loss, accuracy, F1, thời gian |
| `outputs/<model>/training_curves.png` | `train.py` | Biểu đồ loss và chỉ số theo epoch |
| `outputs/<model>/train_log.txt` | không script nào ghi; bạn chuyển hướng output của `train.py` vào đây (xem [04](04_huan_luyen.md#41-lệnh)) | Bản ghi màn hình lúc train |
| `outputs/<model>/confusion_matrix_<split>.png` | `evaluate.py` | Ma trận nhầm lẫn |
| `outputs/<model>/misclassified_<split>.csv` | `evaluate.py` | Danh sách ảnh sai + xác suất 6 lớp |
| `outputs/<model>/misclassified_<split>.png` | `evaluate.py` | Lưới tối đa 20 ảnh sai |
| `outputs/<model>/gradcam/gradcam_NN.png` | `gradcam.py` | Các trang bản đồ nhiệt |
| `outputs/compare_<split>.csv` | `compare_models.py` | Bảng so sánh mọi mô hình |
| `export_onnx/model.onnx`, `model.json` | `export_onnx/export.py` | Mô hình ONNX + thông số tiền xử lý |

`outputs/_old_selection/` là kết quả của lần train theo quy tắc chọn `best.pt` **cũ** (chỉ so val F1). Thư mục này được giữ lại để đối chiếu, và `compare_models.py` không đọc nó (xem [04_huan_luyen.md](04_huan_luyen.md#45-chọn-bestpt-vì-sao-có-tiêu-chí-phụ-val-loss)).

## 1.7. Môi trường

- Python 3.10 trong `.venv` ở thư mục dự án (ổ D, vì ổ C gần đầy).
- Máy **không có GPU**: mọi thứ chạy trên CPU 8 nhân. Code tự dùng GPU nếu có (`torch.cuda.is_available()`).
- Trọng số pretrained tải một lần về `C:\Users\Wisol\.cache\torch\hub\checkpoints\`.
- Phiên bản thư viện cố định trong [requirements.txt](../requirements.txt). Máy chỉ chạy ONNX cần [export_onnx/requirements.txt](../export_onnx/requirements.txt) (onnxruntime, numpy, pillow).
