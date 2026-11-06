# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Tổng quan

Bài luyện tập thứ hai của người dùng (QC lead, mới học PyTorch): phân loại **6 loại lỗi bề mặt thép** (bộ NEU) bằng transfer learning, rồi so sánh nhiều mô hình cùng họ ResNet. Code được dựng từ dự án `D:\Python_Practice\pythonProject` (bài OK/NG Magnetic Tile) và bỏ phần OK/NG (`NG_CLASS`, ngưỡng NG, QC summary). Cách học người dùng đã chọn: Claude viết code, người dùng đọc rồi chạy. `README.md` là hướng dẫn bằng tiếng Việt; khi đổi cách dùng, thiết lập hoặc file kết quả thì phải cập nhật README. `docs/` là bộ tài liệu chi tiết (tiếng Việt, có trích số dòng code); khi đổi code hãy sửa file docs tương ứng theo bảng ở cuối `docs/README.md`. Bảng "Lỗi thường gặp" trong README trích **nguyên văn** các câu `print`. Git có remote `origin` = https://github.com/dbh92/neu_cls.git (nhánh `main`). `data/`, `outputs/`, `outputs_*/` (vd `outputs_v1/`, bản lưu kết quả cũ), `.idea/`, `model.onnx` và `model.json` nằm trong `.gitignore`; GitHub từ chối file > 100 MB nên không bao giờ commit file `.pt`.

Mỗi dòng code có **comment tiếng Việt cuối dòng**. Câu `print` viết bằng tiếng Anh, vì stdout có thể dùng cp1252. Khi chạy qua pipe, hãy đặt `PYTHONIOENCODING=utf-8`.

**Không sửa** thư mục dữ liệu gốc `D:\Huan\2.Python\learning\neu_det\` (bài NumPy cũ của người dùng). `prepare_data.py` chỉ **sao chép** từ đó.

## Lệnh

Python 3.10 trong `.venv` của thư mục này. Máy không có GPU. Chạy mọi lệnh từ thư mục gốc.

```
.venv/Scripts/python.exe prepare_data.py [--overwrite]
.venv/Scripts/python.exe dataset.py
.venv/Scripts/python.exe model.py
.venv/Scripts/python.exe train.py [--model NAME] [--output-dir DIR] [--resume]
.venv/Scripts/python.exe evaluate.py [--checkpoint outputs/NAME/best.pt] [--split test|val|train]
.venv/Scripts/python.exe predict.py <ảnh|thư mục>... [--checkpoint P] [--show] [--save DIR] [--only-wrong]
.venv/Scripts/python.exe compare_models.py [--split test]
.venv/Scripts/python.exe gradcam.py [ảnh|thư mục...] [--checkpoint P] [--misclassified] [--per-class 3] [--save DIR] [--show]
.venv/Scripts/python.exe export_onnx/export.py [--checkpoint P] [--split test]
.venv/Scripts/python.exe export_onnx/predict_onnx.py <ảnh|thư mục>... [--model-dir DIR]
```

Thời gian train trên CPU 8 nhân (1.223 ảnh train, ảnh 224 pixel) xem ở cột `seconds` trong `outputs/<model>/history.csv`. Muốn chạy thử nhanh mà không ghi đè `outputs/`, hãy làm như sau:
1. Chép `*.py` và `export_onnx/` sang thư mục tạm.
2. Tạo `data/split` nhỏ, khoảng 8 ảnh mỗi lớp.
3. Dùng `sed` đặt `EPOCHS = 2`.
4. Chạy bằng `.venv` của dự án.

## Kiến trúc

- `config.py` là **nơi duy nhất** chứa thiết lập. Chỉ những lựa chọn theo từng lần chạy mới là tham số dòng lệnh: `--model`, `--checkpoint`, `--split`.
- **Chia dữ liệu** (`prepare_data.py`):
  - test = toàn bộ `validation` chính thức của NEU (60 ảnh/lớp). Đây đúng là tập mà bản CNN NumPy đã đạt 0.942, nên hai kết quả so sánh được trực tiếp.
  - train/val = tập `train` chính thức của NEU, chia 85/15 cho mỗi lớp bằng `random.Random(SEED)`.
  - Script bỏ ảnh trùng theo MD5, và ghi nhận ảnh test trước để không ảnh nào trong test lọt vào train. Hiện có `patches_105` trùng với `patches_101`.
  - Script từ chối chạy nếu `data/split` đã tồn tại mà không có `--overwrite`.
- **Tên lớp = tên thư mục**, sắp theo alphabet. Danh sách lớp lưu trong checkpoint. Checkpoint là dict gồm `model_name`, `class_names`, `image_size`, `epoch`, `config`, `model`; riêng `last.pt` có thêm trạng thái để resume. `config` là ảnh chụp mọi hằng số VIẾT HOA của `config.py` (hàm `config_snapshot` trong `train.py`, `MODEL_NAME` được thay bằng `--model`); cùng nội dung được ghi ra `outputs/<model>/config.json`. Checkpoint train trước 06/11/2026 không có khóa `config`, nên đọc bằng `.get("config")`. Khi `--resume` mà config khác lần trước thì chỉ in cảnh báo, không dừng.
- **Mỗi mô hình một thư mục** `outputs/<model>/`, chứa:
  - `best.pt`, `last.pt`, `config.json`, `history.csv`, `training_curves.png`, `train_log.txt`;
  - các file do `evaluate.py` ghi cạnh checkpoint: `confusion_matrix_<split>.png`, `misclassified_<split>.csv/.png`;
  - `gradcam/`.

  Riêng `compare_models.py` ghi `outputs/compare_<split>.csv`.
- **`model.py`**:
  - `build_model` thay phần đầu phân loại theo từng họ mô hình:
    - ResNet: `fc`;
    - EfficientNet: `classifier[1]`;
    - MobileNetV3: `classifier[3]`.
  - `gradcam_layer` trả về `layer4` với ResNet, và `features[-1]` với các mô hình còn lại. Thêm mô hình mới thì phải bổ sung cả hai hàm.
- **`evaluate.py`** là module dùng chung: `run_inference`, `decide_predictions` (argmax) và `compute_metrics` (accuracy, macro F1). `matplotlib.use("Agg")` chỉ được gọi trong `__main__`. Mô hình tốt nhất được chọn theo **macro F1 trên val**; nếu F1 bằng nhau thì chọn epoch có **val loss thấp hơn** (`best_val_loss` được lưu trong `last.pt`). Early stopping đếm số epoch liên tiếp không có best mới theo cùng quy tắc này.
- **Lý do có tiêu chí phụ val loss:** với bộ NEU, val F1 chạm 1.0 ngay từ epoch 1–5. Nếu chỉ dùng `>` trên val F1, `best.pt` sẽ bị giữ ở epoch *sớm nhất* đạt 1.0, khi mô hình còn non. Kết quả của lần train theo cách cũ nằm trong `outputs/_old_selection/` (thư mục này `compare_models.py` không đọc), và so sánh ghi ở mục 6 của README.
- **`gradcam.py`**:
  - Dùng forward hook trên `gradcam_layer`, kèm `register_hook` trên output để lấy gradient. Hook được gỡ trong `finally`.
  - `--misclassified` đọc `misclassified_test.csv`, nên phải chạy `evaluate.py` trên tập test trước.
- **`export_onnx/predict_onnx.py`**:
  - Không import torch hay `config`.
  - Hàm `preprocess` phải khớp với `build_transforms(train=False)`, và `decide` phải khớp với `decide_predictions`.
  - `export.py` kiểm chứng sai lệch xác suất nhỏ hơn 1e-4 và kết luận trùng 100%; nếu không đạt thì thoát với mã 1.

## Ràng buộc

- Trên Windows, khi `NUM_WORKERS > 0`, code chạy ở cấp module phải nằm trong `if __name__ == '__main__':`.
- Trọng số pretrained nằm trong `C:\Users\Wisol\.cache\torch\hub\checkpoints\`. Ổ C gần đầy, nên dữ liệu và `.venv` đặt ở ổ D.
