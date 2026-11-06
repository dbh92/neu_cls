# NEU-CLS: phân loại 6 loại lỗi bề mặt thép bằng ResNet và các mô hình cùng họ

Đây là bài luyện tập thứ hai, sau bài OK/NG Magnetic Tile (`D:\Python_Practice\pythonProject`). Hai bài dùng cùng bộ khung code. Những điểm mới của bài này:

| | Bài Magnetic Tile | Bài này |
|---|---|---|
| Bài toán | 2 lớp OK/NG | **6 loại lỗi** (nhiều lớp) |
| Dữ liệu | Lệch: NG ít hơn OK nhiều | Cân bằng: mỗi lớp khoảng 240 ảnh |
| Quy tắc ra kết luận | argmax hoặc ngưỡng NG | Chỉ dùng argmax (không có "ngưỡng NG") |
| Mô hình | ResNet18 | ResNet18, ResNet50, EfficientNet-B0, MobileNetV3, **so sánh với nhau** |
| Công cụ mới | | `compare_models.py` (bảng so sánh), `gradcam.py` (mô hình nhìn vào đâu) |
| Đối chứng | | Bản CNN bạn tự viết bằng NumPy (`neu_det`): **0.942** trên cùng 360 ảnh test |

## 1. Dữ liệu

**NEU Surface Defect Database**, Đại học Northeastern (Trung Quốc). Ảnh xám 200×200 chụp bề mặt thép cán nóng.

| Lớp | Nghĩa |
|---|---|
| `crazing` | Nứt chân chim |
| `inclusion` | Tạp chất lẫn vào |
| `patches` | Mảng loang |
| `pitted_surface` | Rỗ bề mặt |
| `rolled-in_scale` | Vảy cán ép vào |
| `scratches` | Vết xước |

Dataset chỉ dùng cho học tập và nghiên cứu. Trích dẫn:

- K. Song, Y. Yan, *A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects*, Applied Surface Science, 2013.
- Y. He, K. Song, Q. Meng, Y. Yan, *An end-to-end steel surface defect detection approach via fusing multiple hierarchical features*, IEEE Transactions on Instrumentation and Measurement, 2020 (bản NEU-DET).

Ảnh được **sao chép** từ `D:\Huan\2.Python\learning\neu_det\datasets\NEU-DET` (đường dẫn đặt trong `NEU_DIR` của `config.py`). Thư mục gốc không bị sửa. Cách chia như sau:

| Tập | Lấy từ | Mỗi lớp | Dùng để |
|---|---|---|---|
| train | 85% của `train` gốc | 204 ảnh (patches 203) | Học |
| val | 15% của `train` gốc | 36 ảnh | Chọn epoch tốt nhất, early stopping |
| test | **toàn bộ** `validation` gốc | 60 ảnh | Chấm điểm cuối cùng. Là đúng 360 ảnh mà bản NumPy đã dùng |

`prepare_data.py` còn **bỏ ảnh trùng** bằng cách so mã MD5 của file. Bộ gốc có `patches_105.jpg` giống hệt `patches_101.jpg`. Nếu giữ cả hai, một bản có thể rơi vào train và bản kia rơi vào val, khiến mô hình được "chấm điểm trên ảnh đã học". Đây chính là lỗi đã gặp ở bài Magnetic Tile với ảnh `exp5_num_5874`.

## 2. Cài đặt (đã làm sẵn)

```powershell
cd D:\Python_Practice\neu_cls
C:\Users\Wisol\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. Các bước chạy

Chạy mọi lệnh từ thư mục `D:\Python_Practice\neu_cls`.

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1. Chuẩn bị dữ liệu | `.venv\Scripts\python.exe prepare_data.py` | `data/split/{train,val,test}/<lớp>/` |
| 2. Kiểm tra dữ liệu | `.venv\Scripts\python.exe dataset.py` | Số ảnh mỗi tập/lớp, shape một batch |
| 3. Kiểm tra mô hình | `.venv\Scripts\python.exe model.py` | Output (2, 6) và số tham số của 5 mô hình |
| 4. Train | `.venv\Scripts\python.exe train.py --model resnet18` | `outputs/resnet18/best.pt`, `history.csv`, `training_curves.png`, `config.json` |
| 5. Đánh giá | `.venv\Scripts\python.exe evaluate.py --checkpoint outputs/resnet18/best.pt` | Bảng từng lớp, confusion matrix, danh sách và lưới ảnh sai |
| 6. So sánh mô hình | `.venv\Scripts\python.exe compare_models.py` | Bảng so sánh, `outputs/compare_test.csv` |
| 7. Grad-CAM | `.venv\Scripts\python.exe gradcam.py --checkpoint outputs/resnet18/best.pt` | `outputs/resnet18/gradcam/gradcam_01.png` |
| 8. Dự đoán ảnh bất kỳ | `.venv\Scripts\python.exe predict.py <ảnh hoặc thư mục> --show` | Lớp đoán và 3 lớp có xác suất cao nhất |
| 9. Xuất ONNX | `.venv\Scripts\python.exe export_onnx\export.py --checkpoint outputs/efficientnet_b0/best.pt` | `export_onnx/model.onnx` + `model.json`, có kiểm chứng khớp PyTorch |

Ghi chú:

- `train.py` không có `--model` thì dùng `MODEL_NAME` trong `config.py`. Không có `--checkpoint` thì các script dùng `outputs/<MODEL_NAME>/best.pt`.
- Train lại cùng một mô hình **sẽ ghi đè** thư mục của mô hình đó. Muốn giữ bản cũ, thêm `--output-dir outputs/resnet18_v2`.
- Bị ngắt giữa chừng thì chạy lại với `--resume`, giữ nguyên `--model`.
- Tên mô hình hợp lệ: `resnet18`, `resnet50`, `efficientnet_b0`, `mobilenet_v3_large`, `simple_cnn`.

## 4. Thứ tự đọc code

1. `config.py`: mọi thiết lập.
2. `prepare_data.py`: chia dữ liệu và bỏ ảnh trùng.
3. `dataset.py`: giữ nguyên như bài trước.
4. `model.py`: **đọc kỹ `build_model`**. Mỗi họ mô hình đặt phần đầu phân loại ở một chỗ khác nhau (bảng bên dưới).
5. `train.py`: giống bài trước, thêm `--model` và `--output-dir`.
6. `evaluate.py`: bỏ phần NG; thêm "Most confused" và lưới ảnh sai.
7. `compare_models.py`: script mới.
8. `gradcam.py`: script mới, quan trọng nhất về mặt kiến thức.
9. `predict.py` và `export_onnx/`: giống bài trước, in thêm 3 lớp có xác suất cao nhất.

### Thay phần đầu phân loại (transfer learning)

Mô hình pretrained gồm 2 phần:

- **thân** (backbone): các lớp tích chập, đã học trích đặc trưng từ 1,2 triệu ảnh ImageNet;
- **đầu** (head): lớp cuối, phân loại 1000 lớp ImageNet.

Transfer learning là **giữ thân, thay đầu** bằng một lớp mới có số đầu ra bằng số lớp của bạn, ở đây là 6.

| Mô hình | Đầu nằm ở | Cấu trúc đầu gốc | Dòng code thay |
|---|---|---|---|
| ResNet18 / ResNet50 | `model.fc` | `Linear(512 hoặc 2048 → 1000)` | `model.fc = nn.Linear(in_features, 6)` |
| EfficientNet-B0 | `model.classifier[1]` | `Sequential(Dropout, Linear(1280 → 1000))` | thay phần tử `[1]`, giữ Dropout |
| MobileNetV3-Large | `model.classifier[3]` | `Sequential(Linear(960 → 1280), Hardswish, Dropout, Linear(1280 → 1000))` | thay phần tử `[3]` |

Muốn biết đầu của một mô hình lạ nằm ở đâu, hãy `print(model)` rồi xem các dòng cuối.

## 5. Đọc kết quả

### Confusion matrix nhiều lớp

- Hàng là lớp thật, cột là lớp đoán. Đường chéo là số ảnh đoán đúng.
- Ô ngoài đường chéo cho biết **cặp lớp nào hay bị nhầm**. `evaluate.py` liệt kê sẵn các cặp này ở mục `Most confused`.
- Trong bảng từng lớp:
  - **precision** cao nghĩa là khi mô hình nói "scratches" thì thường đúng;
  - **recall** cao nghĩa là có bao nhiêu ảnh scratches thật thì bắt được gần hết.

### Grad-CAM

- Mỗi ảnh có 2 ô: ảnh gốc ở bên trái, ảnh phủ bản đồ nhiệt ở bên phải.
- **Màu đỏ** là vùng làm mô hình tin vào lớp được đoán nhiều nhất; màu xanh là vùng hầu như không ảnh hưởng.
- Mô hình **tốt** sẽ có vùng đỏ nằm trên vết lỗi: vết xước, đốm rỗ, mảng loang.
- Mô hình **đáng ngờ** có vùng đỏ nằm ở viền ảnh, ở một góc cố định, hoặc trên nền trơn. Khi đó mô hình đang dùng "mẹo" của bộ dữ liệu (độ sáng, viền...) chứ không học lỗi thật, và sẽ sai khi sang chuyền khác.
- Bản đồ nhiệt thô và nhòe là chuyện bình thường. ResNet chỉ có lưới 7×7 ở lớp cuối, rồi được phóng lên 224×224.
- Dùng `--misclassified` để soi các ảnh bị đoán sai: mô hình đã nhìn nhầm vào đâu?

## 6. Kết quả

Kết quả train ngày 03/11/2026 trên CPU, ảnh 224 pixel, `EPOCHS = 20`, `EARLY_STOPPING_PATIENCE = 7`. Đánh giá trên tập test gồm 360 ảnh, đúng tập bản NumPy đã dùng. Bảng dưới lấy từ `compare_models.py` (`outputs/compare_test.csv`).

| Mô hình | Epoch best | Accuracy test | Sai | Tham số | File | ms/ảnh (PyTorch) | Thời gian train |
|---|---|---|---|---|---|---|---|
| mobilenet_v3_large | 5 | **1.0000** | 0 | 4,2 triệu | 16 MB | **16,5** | 12 phút (12 epoch) |
| resnet50 | 1 | **1.0000** | 0 | 23,5 triệu | 90 MB | 44,8 | 26 phút (8 epoch) |
| efficientnet_b0 | 3 | 0.9972 | 1 | 4,0 triệu | 16 MB | 27,7 | 17 phút (10 epoch) |
| resnet18 | 2 | 0.9944 | 2 | 11,2 triệu | 43 MB | 20,1 | 10 phút (9 epoch) |
| CNN NumPy (bài cũ) | – | 0.9420 | ≈21 | – | – | – | – |

### Nhận xét

1. **Transfer learning vượt xa bản tự viết.** Mô hình kém nhất cũng chỉ sai 2/360 ảnh, trong khi bản NumPy sai khoảng 21 ảnh. Mỗi mô hình train chỉ mất 10–26 phút, và val accuracy đã đạt 96–100% ngay sau epoch đầu tiên. Lợi thế đến từ trọng số pretrained: mô hình đã biết sẵn cạnh, vân và đốm từ ImageNet.
2. **Bẫy chọn epoch "best".** Cả 4 mô hình chạm val F1 = 1.0000 rất sớm. Sau đó không epoch nào *lớn hơn* 1.0 được nữa, nên `best.pt` bị giữ ở epoch sớm nhất: epoch 1 với ResNet50, epoch 2 với ResNet18. Khi đánh giá thử `last.pt` (epoch cuối) trên tập test, **cả 4 mô hình đều đúng 360/360**. Nói cách khác, 3 ảnh sai ở bảng trên là do chọn epoch quá non, không phải do kiến trúc. Bài học: khi tập val quá dễ hoặc quá nhỏ (216 ảnh), chỉ số trên val bị "bão hòa" và không còn phân biệt được các epoch. Lúc đó cần thêm tiêu chí phụ, ví dụ val loss thấp hơn, hoặc một tập val khó hơn.
3. **Các ảnh bị sai đều là cặp dễ nhầm thật.**
   - ResNet18 đoán 2 ảnh `inclusion` (283, 284) thành `scratches`. Hai ảnh này có nền tối, và vết lẫn tạp chất dài, mảnh, chạy dọc, nhìn giống vết xước.
   - EfficientNet đoán ảnh `scratches_248` thành `rolled-in_scale`, với độ tự tin chỉ 48%. Vết xước trong ảnh rất mờ, còn nền có vân giống vảy cán.
4. **Grad-CAM** (`outputs/<model>/gradcam/` và `gradcam_wrong/`):
   - Với ảnh đúng, vùng đỏ nằm đúng trên vết lỗi: vết nứt của crazing, vệt của inclusion, mảng tối của patches, đốm rỗ, vết xước sáng. Mô hình học lỗi thật chứ không học mẹo.
   - Với ảnh sai, vùng đỏ nằm lệch khỏi vết lỗi, rơi vào vùng nền tối. Có thể mô hình đã dựa phần nào vào **độ sáng nền**: ảnh scratches và rolled-in_scale trong bộ NEU thường tối, ảnh inclusion thường sáng. Ở nhà máy, mỗi chuyền có ánh sáng khác nhau, nên đây là điểm cần kiểm tra trước khi đưa mô hình sang chuyền khác.
5. **Lớn hơn không có nghĩa là tốt hơn.**
   - ResNet50 có số tham số gấp 5,6 lần MobileNetV3, file nặng gấp 5,6 lần và chạy chậm gấp 2,7 lần, nhưng độ chính xác như nhau.
   - EfficientNet-B0 nhỏ nhưng chạy trên CPU lại chậm hơn ResNet18. Số tham số ít không có nghĩa là chạy nhanh.
6. **Đề xuất triển khai: MobileNetV3-Large**, vì vừa đúng 100% vừa nhanh nhất. Bản ONNX (`export_onnx/model.onnx`, 16,3 MB) đã kiểm chứng trên tập test:
   - sai lệch xác suất so với PyTorch là 8e-6;
   - kết luận trùng 360/360 ảnh;
   - chỉ riêng mô hình chạy 4,0 ms/ảnh, nhanh gấp 4,3 lần PyTorch;
   - tính cả mở ảnh và tiền xử lý thì khoảng 6–7 ms/ảnh.

## 7. Bài tập tự làm

Mỗi bài chỉ cần sửa `config.py` hoặc thêm tham số dòng lệnh. Dùng `--output-dir` riêng để không ghi đè kết quả cũ. Sau đó chạy `compare_models.py` để so sánh.

1. **Tự viết so với pretrained:** `train.py --model simple_cnn --output-dir outputs/simple_cnn`. Nhớ tăng `LEARNING_RATE` lên `1e-3`. So sánh với bản NumPy 0.942 và với ResNet18.
2. **Bỏ pretrained:** đặt `PRETRAINED = False`, rồi train ResNet18 vào `outputs/resnet18_scratch`. Cùng kiến trúc, chỉ khác trọng số ban đầu. Chênh lệch bao nhiêu?
3. **Bỏ augmentation:** tắt các phép lật, xoay, đổi độ sáng. Xem `training_curves.png`: train loss và val loss có tách xa nhau hơn (overfit) không?
4. **Ít dữ liệu:** xóa bớt ảnh trong `data/split/train` (ví dụ chỉ giữ 30 ảnh mỗi lớp), rồi train lại. Đây là tình huống giống nhà máy nhất. Xong thì chạy `prepare_data.py --overwrite` để khôi phục dữ liệu.
5. **Thêm một mô hình mới**, ví dụ `convnext_tiny` hoặc `resnet34`: sửa `MODEL_NAMES`, `build_model` và `gradcam_layer`.

## 8. Lỗi thường gặp

| Thông báo | Nguyên nhân và cách xử lý |
|---|---|
| `Folder not found: ... Check NEU_DIR in config.py.` | Thư mục NEU gốc bị di chuyển. Sửa `NEU_DIR` |
| `data/split already exists. Use --overwrite to create it again.` | Đã chia rồi. Chỉ dùng `--overwrite` khi muốn chia lại |
| `data/split/train not found. Run prepare_data.py first.` | Chưa chạy bước 1 |
| `Checkpoint not found: ... Run train.py first.` | Chưa train mô hình đó, hoặc gõ sai tên thư mục trong `--checkpoint` |
| `Class names or model changed since last run, cannot resume. Train from scratch.` | `--resume` với `--model` khác lần trước. Bỏ `--resume` hoặc sửa `--model` |
| `Warning: ... changed since last run: ... -> ...` | `--resume` sau khi đã sửa `config.py`. Vẫn chạy tiếp, nhưng optimizer và lịch learning rate lấy từ `last.pt` nên `LEARNING_RATE` mới không có tác dụng. Muốn dùng thiết lập mới thì train lại không có `--resume` |
| `No checkpoints found in outputs/*/best.pt. Run train.py first.` | `compare_models.py` chưa có mô hình nào để so |
| `... misclassified_test.csv not found. Run evaluate.py --checkpoint ... first.` | `gradcam.py --misclassified` cần chạy `evaluate.py` trên tập test trước |
| `Verification FAILED: do not use this model.onnx.` | ONNX không khớp PyTorch. Kiểm tra lại `preprocess` trong `predict_onnx.py` |
| `RuntimeError: DataLoader worker ...` hoặc treo khi bắt đầu train | Đặt `NUM_WORKERS = 0` trong `config.py` |

## 9. Cấu trúc thư mục

```
neu_cls/
├── config.py            # mọi thiết lập
├── prepare_data.py      # NEU gốc -> data/split (bỏ ảnh trùng)
├── dataset.py           # Dataset, transforms, DataLoader
├── model.py             # 5 mô hình, thay phần đầu, lớp dùng cho Grad-CAM
├── train.py             # train một mô hình -> outputs/<model>/
├── evaluate.py          # đánh giá + module dùng chung
├── compare_models.py    # so sánh mọi mô hình trên cùng tập test
├── gradcam.py           # bản đồ nhiệt: mô hình nhìn vào đâu
├── predict.py           # dự đoán ảnh bất kỳ
├── export_onnx/         # xuất ONNX + chạy không cần PyTorch
├── docs/                # tài liệu chi tiết: cách hoạt động, luồng, vận hành (bắt đầu từ docs/README.md)
├── data/split/          # sinh ra bởi prepare_data.py (không sửa tay)
└── outputs/<model>/     # sinh ra bởi train/evaluate/gradcam (không sửa tay)
```
