# 7. Dự đoán ảnh mới và xuất ONNX

Có hai cách dùng mô hình đã train:

| | `predict.py` | `export_onnx/predict_onnx.py` |
|---|---|---|
| Cần | PyTorch, torchvision, toàn bộ dự án | Chỉ `onnxruntime`, `numpy`, `pillow` + 3 file |
| Mô hình | `outputs/<model>/best.pt` | `export_onnx/model.onnx` + `model.json` |
| Tốc độ (MobileNetV3, CPU) | ~16 ms/ảnh (chỉ mô hình) | ~4 ms/ảnh (chỉ mô hình) |
| Hiện hình | `--show`, `--save`, `--only-wrong` | Chỉ in chữ |
| Dùng khi | Thử nghiệm trên máy dev | Mang sang máy chuyền, tích hợp chương trình khác |

## 7.1. `predict.py`

### Lệnh

```powershell
.venv\Scripts\python.exe predict.py anh_moi.jpg --checkpoint outputs/mobilenet_v3_large/best.pt
.venv\Scripts\python.exe predict.py D:\anh_chuyen\ca1 --checkpoint outputs/mobilenet_v3_large/best.pt
.venv\Scripts\python.exe predict.py data/split/test/scratches data/split/test/inclusion --show --only-wrong
.venv\Scripts\python.exe predict.py data/split/test/patches --save outputs/predict_patches
```

> Chú ý: khi đưa một **thư mục**, script chỉ lấy file ảnh nằm **trực tiếp** trong thư mục đó, không đi vào thư mục con. `data/split/test` chỉ chứa 6 thư mục con nên sẽ không có ảnh nào. Hãy đưa từng thư mục lớp.

| Tham số | Ý nghĩa |
|---|---|
| `paths` | Một hoặc nhiều file ảnh / thư mục (bắt buộc) |
| `--checkpoint` | Mặc định `outputs/<MODEL_NAME>/best.pt` |
| `--show` | Hiện lưới ảnh kèm kết luận, 20 ảnh mỗi cửa sổ |
| `--save DIR` | Lưu lưới ra `DIR/predictions_01.png`, ... |
| `--only-wrong` | Chỉ vẽ ảnh đoán sai (cần ảnh nằm trong thư mục mang tên lớp) |

### Luồng

```
nạp checkpoint → build_transforms(train=False) với image_size của checkpoint
→ chạy thử 1 ảnh đen (để đo giờ ảnh đầu không bị chậm giả)
→ với từng ảnh:
     mở + convert("RGB") → transform → model → softmax → decide_predictions
     nhãn thật = tên thư mục cha nếu trùng tên lớp, không thì "chưa biết"
     in 1 dòng
→ in tổng kết: số ảnh, số sai, ms trung bình/chậm nhất
→ nếu --show/--save: vẽ lưới
```

### Dòng kết quả

Ví dụ minh họa (con số không lấy từ một lần chạy thật):

```
scratches       | scratches 99.80%  inclusion  0.15%  crazing  0.03% |   18.2 ms | data/split/test/scratches/scratches_241.jpg
scratches       | scratches 91.23%  inclusion  8.70%  patches  0.04% |   17.9 ms | data/split/test/inclusion/inclusion_283.jpg  <- WRONG (true inclusion)
```

Cột 1 là kết luận, rồi 3 lớp có xác suất cao nhất, thời gian (tính cả mở ảnh và tiền xử lý), đường dẫn, và `<- WRONG` nếu biết nhãn thật mà đoán sai.

**Xem lớp thứ 2:** nếu lớp thứ 2 có xác suất đáng kể (vd 30–50%), ảnh đó là ca "phân vân" giữa hai loại lỗi. Trong vận hành có thể chuyển những ảnh này cho người kiểm tra lại.

**Nhận dạng nhãn thật từ tên thư mục:** ảnh nằm trong `.../scratches/` thì coi nhãn thật là `scratches`. Muốn kiểm tra hàng loạt ảnh mới đã phân loại tay, chỉ cần xếp chúng vào các thư mục mang đúng tên 6 lớp.

Ảnh không mở được (file hỏng, không phải ảnh) thì in `Skipping ...` và bỏ qua, không dừng chương trình.

## 7.2. Xuất ONNX (`export_onnx/export.py`)

ONNX là định dạng mô hình chung, chạy được bằng `onnxruntime` trên nhiều ngôn ngữ (Python, C#, C++...) mà không cần PyTorch.

### Lệnh

```powershell
.venv\Scripts\python.exe export_onnx\export.py --checkpoint outputs/mobilenet_v3_large/best.pt
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--checkpoint` | `outputs/<MODEL_NAME>/best.pt` | Mô hình cần xuất |
| `--output-dir` | `export_onnx/` | Nơi ghi `model.onnx` và `model.json` (**ghi đè** bản cũ) |
| `--split` | `test` | Tập ảnh dùng để kiểm chứng |

### Ba việc script làm

**1. Xuất** ([export.py:38-67](../export_onnx/export.py#L38-L67))

- `torch.onnx.export` với ảnh giả batch 2 và `dynamic_shapes` cho chiều batch: file ONNX nhận 1 hay nhiều ảnh một lần đều được.
- `external_data=False`: trọng số nằm chung trong một file `model.onnx`.
- Đầu vào tên `image`, shape `(batch, 3, 224, 224)`, float32. Đầu ra tên `logits`, shape `(batch, 6)`, **chưa** softmax.
- Ghi `model.json`: mọi thông số mà chương trình dùng ONNX cần biết.

```json
{
  "model_file": "model.onnx",
  "model_name": "mobilenet_v3_large",
  "class_names": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
  "image_size": 224,
  "color": "RGB",
  "resize": "bilinear",
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "input_name": "image",
  "output_name": "logits",
  "checkpoint": "outputs/mobilenet_v3_large/best.pt",
  "epoch": 5,
  "exported_at": "2026-11-03 13:30:17"
}
```

`checkpoint`, `epoch` và `exported_at` cho biết file ONNX được xuất **từ checkpoint nào, lúc nào**: dùng như số phiên bản mô hình.

**2. Kiểm chứng** ([export.py:70-92](../export_onnx/export.py#L70-L92))

Chạy toàn bộ tập `--split` bằng **cả hai** đường:

- PyTorch: DataLoader + torchvision transforms (chuẩn để so);
- ONNX: `OnnxClassifier` của `predict_onnx.py`, tiền xử lý bằng numpy.

Rồi so: sai lệch xác suất lớn nhất, số kết luận trùng nhau, độ chính xác của ONNX, thời gian trọn quy trình. **Đạt** khi sai lệch ≤ `1e-4` **và** kết luận trùng 100%. Không đạt thì in `Verification FAILED: do not use this model.onnx.` và thoát mã 1.

Bước này bắt được lỗi hay gặp nhất khi triển khai: tiền xử lý ở chương trình dùng mô hình **khác** lúc train (sai thứ tự màu, sai kiểu resize, quên chuẩn hóa).

**3. Đo tốc độ** ([export.py:95-109](../export_onnx/export.py#L95-L109)): chỉ phần mô hình, 1 ảnh/lần, 50 lần, PyTorch so với ONNX.

## 7.3. `predict_onnx.py`: chạy không cần PyTorch

### Lệnh

```powershell
python export_onnx\predict_onnx.py anh_moi.jpg
python export_onnx\predict_onnx.py D:\anh_chuyen\ca1 --model-dir D:\model_neu
```

Dòng in ra cùng định dạng với `predict.py`.

### Thiết kế

File này **cố ý không import** `torch`, `config.py` hay bất kỳ file nào khác của dự án. Mọi thông số (lớp, kích thước, mean/std) đọc từ `model.json`. Vì vậy chỉ cần chép 3 file sang máy khác là chạy được.

`OnnxClassifier` có 3 phương thức chính, và mỗi cái phải **khớp** với một phần của bản PyTorch:

| Phương thức | Phải khớp với | Làm gì |
|---|---|---|
| `preprocess(image)` | `build_transforms(train=False)` | RGB → resize bilinear → `/255` → `(x − mean)/std` → `(H,W,3)` thành `(3,H,W)` |
| `predict_probabilities(images)` | `torch.softmax` trong `run_inference` | Gộp batch → chạy ONNX → softmax bằng numpy (trừ max trước để không tràn số) |
| `decide(probs)` | `decide_predictions` | argmax |

Sửa một bên mà quên bên kia thì `export.py` sẽ báo FAILED.

### Gói mang sang máy chuyền

```
<thư mục bất kỳ>/
├── predict_onnx.py
├── model.onnx
├── model.json
└── requirements.txt      (onnxruntime, numpy, pillow)
```

Trên máy đích:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe predict_onnx.py <ảnh hoặc thư mục>
```

### Tích hợp vào chương trình khác (C#, ...)

Đọc `model.json` và làm đúng các bước:

1. Mở ảnh, đổi về **RGB** (OpenCV và nhiều thư viện C# mặc định là **BGR**: phải đổi).
2. Resize về `image_size × image_size`, kiểu bilinear.
3. Chia 255 → giá trị 0..1.
4. Trừ `mean`, chia `std`, theo từng kênh R, G, B.
5. Sắp thành `(1, 3, H, W)` float32 (kênh màu đứng trước).
6. Chạy với đầu vào tên `input_name`, lấy đầu ra `output_name`.
7. Softmax, rồi argmax, rồi tra `class_names`.

Sau khi tích hợp, hãy chạy thử trên vài chục ảnh test và so với kết quả `predict_onnx.py`. Kết luận phải trùng hoàn toàn.
