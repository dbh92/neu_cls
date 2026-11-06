# 5. Đánh giá và so sánh mô hình

## 5.1. Ba hàm dùng chung trong `evaluate.py`

`evaluate.py` vừa là script, vừa là **module** mà `train.py`, `predict.py`, `compare_models.py` và `export_onnx/export.py` cùng import. Nhờ vậy cả dự án chấm điểm theo đúng một cách.

| Hàm | Vào | Ra | Ai dùng |
|---|---|---|---|
| [`run_inference(model, loader, device, criterion=None)`](../evaluate.py#L17-L30) | mô hình, DataLoader | `(loss trung bình hoặc None, nhãn thật, xác suất)` | train (val), evaluate, compare, export |
| [`decide_predictions(probabilities)`](../evaluate.py#L33-L34) | danh sách xác suất | danh sách chỉ số lớp (argmax) | mọi nơi ra kết luận |
| [`compute_metrics(labels, predictions, class_names)`](../evaluate.py#L37-L40) | nhãn thật, nhãn đoán | `{"accuracy", "macro_f1"}` | train, evaluate, compare |

`run_inference` luôn gọi `model.eval()` và chạy trong `torch.no_grad()`: không augmentation, không Dropout, không tính gradient, nên nhanh và cho kết quả ổn định.

## 5.2. Các chỉ số

Ví dụ với lớp `scratches`:

| Chỉ số | Câu hỏi nó trả lời | Công thức |
|---|---|---|
| **Precision** | Khi mô hình nói "scratches", bao nhiêu lần đúng? | đúng là scratches / số lần đoán scratches |
| **Recall** | Có bao nhiêu ảnh scratches thật, mô hình bắt được bao nhiêu? | đúng là scratches / số ảnh scratches thật |
| **F1** | Cân bằng hai cái trên | `2 × P × R / (P + R)` |
| **Macro F1** | Trung bình F1 của 6 lớp, mỗi lớp nặng như nhau | `(F1₁ + ... + F1₆) / 6` |
| **Accuracy** | Tỉ lệ đoán đúng trên tổng số ảnh | đúng / tổng |

Ở góc nhìn QC: recall thấp ở một lớp nghĩa là **lọt lỗi** loại đó (bị gán sang loại khác); precision thấp nghĩa là **báo nhầm** sang loại đó.

Với NEU (các lớp gần cân bằng), accuracy và macro F1 gần như bằng nhau. Macro F1 được chọn làm tiêu chí chính vì khi dữ liệu lệch (vd một lớp chỉ có 20 ảnh), accuracy có thể vẫn cao trong khi lớp nhỏ bị đoán sai hết.

## 5.3. `evaluate.py`: đánh giá chi tiết một mô hình

### Lệnh

```powershell
.venv\Scripts\python.exe evaluate.py --checkpoint outputs/resnet18/best.pt              # tập test (mặc định)
.venv\Scripts\python.exe evaluate.py --checkpoint outputs/resnet18/best.pt --split val  # tập val
.venv\Scripts\python.exe evaluate.py --checkpoint outputs/resnet18/last.pt              # epoch cuối
```

### Luồng

```
nạp checkpoint → lấy class_names, image_size từ checkpoint
→ DataLoader của --split (không augmentation, không xáo trộn)
→ run_inference → decide_predictions → compute_metrics
→ in: dòng tổng kết, bảng từng lớp, confusion matrix dạng chữ, "Most confused"
→ lưu: confusion_matrix_<split>.png
→ lọc ảnh sai, sắp theo độ tự tin giảm dần
→ lưu: misclassified_<split>.csv (+ misclassified_<split>.png nếu có ảnh sai)
```

Mọi file được lưu **cạnh checkpoint**: đánh giá `outputs/resnet18/best.pt` thì file ra nằm trong `outputs/resnet18/`.

### Đọc kết quả in ra màn hình

1. **Dòng tổng kết:** `Model: resnet18 (epoch 14) | split: test (360 images) | accuracy ... | macro F1 ...`. Kiểm tra số epoch có đúng là checkpoint bạn muốn không.
2. **Bảng từng lớp** (`classification_report` của scikit-learn): precision, recall, F1, `support` (số ảnh thật của lớp).
3. **Confusion matrix dạng chữ:** hàng = lớp thật, cột = lớp đoán (tên cắt còn 8 ký tự). Đường chéo là số ảnh đúng.
4. **Most confused:** tối đa 5 cặp `lớp thật -> lớp đoán` bị nhầm nhiều nhất, đọc ngay được mà không cần dò ma trận.

### Danh sách ảnh sai

`misclassified_<split>.csv` (mở được bằng Excel, mã hóa `utf-8-sig`). Ví dụ minh họa:

| image | true | predicted | prob_crazing | ... | prob_scratches |
|---|---|---|---|---|---|
| data/split/test/inclusion/inclusion_283.jpg | inclusion | scratches | 0.0001 | ... | 0.9123 |

Các dòng được **sắp theo độ tự tin của lớp đoán, giảm dần**. Ảnh sai mà mô hình rất tự tin (vd 95%) đáng xem nhất: có thể nhãn gốc bị gán sai, hoặc hai loại lỗi thật sự giống nhau. Ảnh sai với độ tự tin thấp (vd 48%) là mô hình đang "phân vân", thường là ca khó.

`misclassified_<split>.png` vẽ tối đa 20 ảnh sai đầu tiên, mỗi ô ghi `pred <lớp> <xác suất>`, `true <lớp>` và tên file.

Cột `image` lưu đường dẫn **tương đối** (`data/split/test/...`). `gradcam.py --misclassified` đọc lại cột này, nên phải chạy cả hai script từ thư mục gốc dự án.

## 5.4. `compare_models.py`: so sánh mọi mô hình

### Lệnh

```powershell
.venv\Scripts\python.exe compare_models.py               # tập test
.venv\Scripts\python.exe compare_models.py --split val
```

### Hoạt động

1. Tìm mọi file khớp `outputs/*/best.pt`. Chỉ đúng **một cấp** thư mục, nên `outputs/_old_selection/<model>/best.pt` **không** bị tính. Ngược lại, thư mục bạn tạo bằng `--output-dir outputs/resnet18_v2` **sẽ** được tính.
2. Với từng checkpoint: chấm trên `--split` bằng đúng 3 hàm dùng chung, rồi đo tốc độ.
3. Sắp xếp: macro F1 cao trước; F1 bằng nhau thì mô hình nhanh hơn trước.
4. In bảng. Với tập test, in thêm dòng `numpy_cnn` 0.942 để đối chiếu.
5. Lưu `outputs/compare_<split>.csv`.

### Các cột

| Cột | Ý nghĩa | Ghi chú |
|---|---|---|
| `model` | Tên kiến trúc | Lấy từ checkpoint, không phải tên thư mục |
| `best_epoch` | Epoch của `best.pt` | |
| `accuracy`, `macro_f1` | Chỉ số trên `--split` | |
| `wrong` | Số ảnh sai | |
| `params_m` | Số tham số (triệu) | |
| `file_mb` | Dung lượng `best.pt` (MB) | |
| `ms_per_image` | Thời gian **chỉ phần mô hình**, 1 ảnh/lần, CPU | Trung bình 30 lần sau 5 lần khởi động; không tính mở ảnh và tiền xử lý |

Tốc độ đo trên CPU (`device = cpu`) vì máy chuyền thường không có GPU. Con số phụ thuộc vào máy đang làm gì khác: **đừng chạy so sánh trong lúc đang train**, nếu không cột `ms_per_image` sẽ bị chậm sai lệch.

### Đọc bảng để chọn mô hình triển khai

1. Lọc các mô hình có độ chính xác chấp nhận được.
2. Trong số đó, ưu tiên mô hình **nhanh** và **nhẹ** hơn.
3. Không chọn theo số tham số: EfficientNet-B0 ít tham số hơn ResNet18 nhưng chạy chậm hơn trên CPU.
4. Khi nhiều mô hình cùng đúng 100% trên test, kết quả test không còn phân biệt được chúng. Xem thêm val loss trong `history.csv` và Grad-CAM để đánh giá độ "chắc chắn".
