# 4. Huấn luyện (`train.py`)

## 4.1. Lệnh

```powershell
.venv\Scripts\python.exe train.py                                   # mô hình MODEL_NAME trong config.py
.venv\Scripts\python.exe train.py --model mobilenet_v3_large        # chọn mô hình
.venv\Scripts\python.exe train.py --model resnet18 --output-dir outputs/resnet18_v2   # giữ bản cũ
.venv\Scripts\python.exe train.py --model resnet18 --resume         # chạy tiếp sau khi bị ngắt
```

Muốn lưu lại toàn bộ màn hình vào `train_log.txt` (code không tự ghi file này):

```powershell
New-Item -ItemType Directory -Force outputs\resnet18 | Out-Null
cmd /c ".venv\Scripts\python.exe train.py --model resnet18 > outputs\resnet18\train_log.txt 2>&1"
```

Dùng `cmd /c ... >` thay cho `>` của PowerShell 5.1, vì PowerShell 5.1 ghi file dạng UTF-16. Các dòng epoch có `flush=True`, nên bạn mở file trong lúc train vẫn thấy tiến độ.

## 4.2. Luồng tổng thể

```
đọc tham số (--model, --output-dir, --resume)
cố định seed, chọn thiết bị (CPU/GPU)
│
├─ 1. DỮ LIỆU     kiểm tra data/split/train, tạo train_loader (có augmentation) + val_loader
├─ 2. MÔ HÌNH     build_model, class weights, CrossEntropyLoss, AdamW, CosineAnnealingLR
├─ 3. CHECKPOINT  nếu --resume: nạp lại mọi trạng thái từ last.pt
│
└─ 4. VÒNG LẶP EPOCH  (epoch = start_epoch .. EPOCHS-1)
     ├─ early stopping?  → dừng
     ├─ TRAIN   duyệt 77 batch: forward → loss → backward → cập nhật trọng số
     ├─ scheduler.step()   giảm learning rate
     ├─ VAL     run_inference trên 216 ảnh val → val loss, accuracy, macro F1
     ├─ có best mới?  (F1 cao hơn, hoặc F1 bằng mà val loss thấp hơn)
     ├─ in một dòng kết quả
     ├─ lưu best.pt (nếu best mới) và last.pt (luôn luôn)
     └─ cập nhật history.csv + training_curves.png
```

## 4.3. Các bước chuẩn bị

### Bước 1: Dữ liệu ([train.py:58-70](../train.py#L58-L70))

- Không có `data/split/train` → in `data/split/train not found. Run prepare_data.py first.` và dừng.
- Lớp nào không có ảnh train, hoặc val trống → dừng, yêu cầu chạy lại `prepare_data.py --overwrite`.

### Bước 2: Mô hình, loss, optimizer ([train.py:72-80](../train.py#L72-L80))

| Thành phần | Chọn | Vì sao |
|---|---|---|
| Hàm mất mát | `CrossEntropyLoss(weight=class_weights)` | Chuẩn cho phân loại nhiều lớp |
| Trọng số lớp | `tổng ảnh / (số lớp × ảnh của lớp)` | Lớp ít ảnh được tính loss nặng hơn. NEU gần cân bằng nên mọi trọng số ≈ 1: các lớp 204 ảnh → 0,999, patches 203 ảnh → 1,004 |
| Optimizer | `AdamW(lr=1e-4, weight_decay=1e-4)` | Tự điều chỉnh bước học từng tham số, hợp với fine-tune |
| Lịch learning rate | `CosineAnnealingLR(T_max=EPOCHS)` | LR giảm dần theo hình cosine, từ `1e-4` về gần 0 ở epoch 20 |

Lưu ý: lịch cosine tính theo `EPOCHS = 20`. Nếu early stopping dừng ở epoch 9, learning rate chưa kịp xuống thấp. Điều đó không sao, vì `best.pt` đã được lưu.

### Bước 3: Resume ([train.py:92-112](../train.py#L92-L112))

Khi có `--resume` **và** `last.pt` tồn tại, code nạp lại: trọng số, optimizer, scheduler, epoch đã xong, `best_score`, `best_val_loss`, `history`, bộ đếm early stopping. Vòng lặp chạy tiếp từ epoch kế tiếp như chưa từng bị ngắt.

Nếu danh sách lớp hoặc `--model` khác lần trước → in `Class names or model changed since last run, cannot resume. Train from scratch.` và dừng.

Có `--resume` mà **không có** `last.pt` → không báo lỗi, train từ đầu.

Nếu `config.py` đã bị sửa so với lần train trước, mỗi thiết lập bị đổi in một dòng `Warning: EPOCHS changed since last run: 20 -> 30`, rồi thêm dòng `Note: --resume keeps the optimizer and learning rate schedule from last.pt. To use new settings, train without --resume.` Code **vẫn chạy tiếp**, vì có những thay đổi vô hại như `NUM_WORKERS`. Nhưng hãy nhớ: optimizer và scheduler được nạp lại từ `last.pt`, nên đổi `LEARNING_RATE` khi resume **không có tác dụng**. Còn `EPOCHS` mới thì vòng lặp có dùng, trong khi lịch cosine vẫn tính theo `EPOCHS` cũ. Resume chỉ nên dùng để chạy tiếp đúng lần train cũ.

Trước khi vào vòng lặp, `train.py` ghi `config.json` (mọi thiết lập của lần chạy này) vào thư mục output. Mỗi checkpoint cũng mang theo cùng nội dung đó trong khóa `config`.

## 4.4. Một epoch diễn ra thế nào

### Phần train ([train.py:123-135](../train.py#L123-L135))

```python
model.train()                         # bật Dropout, BatchNorm cập nhật thống kê theo batch
for images, labels in train_loader:   # 77 batch, mỗi batch 16 ảnh đã augmentation ngẫu nhiên
    outputs = model(images)           # forward: logits (16, 6)
    loss_value = criterion(outputs, labels)
    optimizer.zero_grad()             # xóa gradient của batch trước
    loss_value.backward()             # tính gradient cho mọi trọng số
    optimizer.step()                  # cập nhật trọng số
scheduler.step()                      # giảm learning rate, mỗi epoch một lần
```

`train_loss` được tính là trung bình loss **trên mỗi ảnh** (nhân loss batch với số ảnh trong batch rồi chia tổng số ảnh), nên batch cuối 7 ảnh không làm lệch con số.

### Phần val ([train.py:137-138](../train.py#L137-L138))

`run_inference` (trong `evaluate.py`) chạy mô hình ở chế độ `eval()` và `no_grad()` trên 216 ảnh val, trả về val loss, nhãn thật, xác suất. `decide_predictions` lấy argmax. `compute_metrics` tính accuracy và macro F1. Đây **chính là** các hàm mà `evaluate.py` dùng để chấm tập test, nên cách tính điểm lúc train và lúc đánh giá giống hệt nhau.

### Dòng in ra mỗi epoch

```
Epoch 14/20 | train loss 0.0051 | val loss 0.0003 | val acc 1.0000 | val F1 1.0000 | 84s *
Epoch 15/20 | train loss 0.0080 | val loss 0.0018 | val acc 1.0000 | val F1 1.0000 | 90s (no improvement 1/7)
```

- `*`: epoch này là best mới, `best.pt` vừa được ghi đè.
- `(no improvement 1/7)`: đã 1 epoch liên tiếp không có best mới; tới 7/7 thì tự dừng.

## 4.5. Chọn `best.pt`: vì sao có tiêu chí phụ val loss

### Quy tắc

[train.py:147-148](../train.py#L147-L148):

```python
is_best = (metrics["macro_f1"] > best_score
           or (metrics["macro_f1"] == best_score and val_loss < best_val_loss))
```

1. **Tiêu chí chính: macro F1 trên val cao hơn.** Macro F1 là trung bình F1 của 6 lớp, nên lớp nào cũng quan trọng như nhau.
2. **Tiêu chí phụ: F1 bằng nhau thì chọn epoch có val loss thấp hơn.** Val loss thấp hơn nghĩa là mô hình đoán đúng **một cách tự tin hơn** (xác suất của lớp đúng gần 1 hơn).

### Vì sao cần tiêu chí phụ?

Với bộ NEU, val F1 chạm **1.0000** ngay từ epoch 1–5. Sau đó không epoch nào có thể *lớn hơn* 1.0. Nếu chỉ dùng `>`, `best.pt` bị giữ mãi ở epoch **sớm nhất** đạt 1.0, khi mô hình còn non.

Lần train đầu tiên dùng quy tắc cũ (kết quả còn trong `outputs/_old_selection/`): ResNet18 bị chọn epoch 2, ResNet50 epoch 1. Trên tập test, `best.pt` của ResNet18 sai 2 ảnh, EfficientNet sai 1 ảnh, trong khi `last.pt` (epoch cuối) của cả 4 mô hình đều đúng 360/360. Như vậy lỗi nằm ở **cách chọn epoch**, không phải ở kiến trúc.

### Ví dụ thật: ResNet18 với quy tắc mới

Trích `outputs/resnet18/history.csv`:

| Epoch | val F1 | val loss | Best mới? | Lý do |
|---|---|---|---|---|
| 1 | 0.9907 | 0.03089 | ✔ | lần đầu |
| 2 | 1.0000 | 0.00699 | ✔ | F1 cao hơn |
| 3 | 0.9954 | 0.01037 | | F1 thấp hơn |
| 6 | 1.0000 | 0.00236 | ✔ | F1 bằng, loss thấp hơn 0.00699 |
| 7 | 1.0000 | 0.00124 | ✔ | loss thấp hơn |
| 12 | 1.0000 | 0.00086 | ✔ | loss thấp hơn |
| 13 | 1.0000 | 0.00047 | ✔ | loss thấp hơn |
| 14 | 1.0000 | 0.00032 | ✔ | loss thấp hơn → **best cuối cùng** |
| 15–20 | | | | không epoch nào có loss < 0.00032 |

Theo quy tắc cũ, best dừng ở epoch 2. Theo quy tắc mới, best là epoch 14.

## 4.6. Early stopping

`epochs_without_improvement` tăng 1 sau mỗi epoch **không** có best mới, và về 0 khi có best mới. Đầu mỗi vòng lặp, nếu bộ đếm ≥ `EARLY_STOPPING_PATIENCE` (7) thì in `Early stopping: no new best (val F1, then val loss) for 7 epochs.` và dừng.

Kiểm tra được đặt ở **đầu** vòng lặp (không phải cuối), để khi chạy `--resume` trên một lần train đã tự dừng thì cũng dừng ngay, không chạy thêm epoch nào.

Ví dụ ResNet18 ở trên: sau epoch 14 có 6 epoch không cải thiện (15–20), chưa tới 7, nên chạy đủ 20 epoch.

Đặt `EARLY_STOPPING_PATIENCE = None` để tắt.

## 4.7. Lưu file sau mỗi epoch

| File | Khi nào ghi | Dùng để |
|---|---|---|
| `best.pt` | Chỉ khi có best mới | Mọi bước sau (evaluate, predict, export...) |
| `last.pt` | **Mọi** epoch | `--resume` |
| `config.json` | Một lần, trước epoch đầu tiên | Biết mô hình này được train với thiết lập nào |
| `history.csv` | Mọi epoch (ghi lại cả file) | Xem số liệu, đo thời gian (cột `seconds`) |
| `training_curves.png` | Mọi epoch | Xem bằng mắt |

Vì ghi sau **mỗi** epoch, bạn có thể mở `training_curves.png` trong lúc train vẫn đang chạy. Nếu máy tắt đột ngột, mất tối đa một epoch.

### Đọc `training_curves.png`

- **Hình trái:** train loss và val loss. Cả hai giảm là tốt. Train loss giảm mà val loss **tăng dần** là dấu hiệu overfit (mô hình học thuộc ảnh train).
- **Hình phải:** val accuracy và val macro F1, trục 0–1,05. Với NEU, hai đường gần như chạm 1.0 ngay sau vài epoch.

## 4.8. Thời gian train

Trên CPU 8 nhân (xem cột `seconds` trong `history.csv`):

| Mô hình | Giây/epoch (khoảng) |
|---|---|
| resnet18 | 80 |
| mobilenet_v3_large | 95 |
| efficientnet_b0 | 150 |
| resnet50 | ~200 |

Epoch đầu luôn chậm hơn (khởi động tiến trình đọc ảnh, cấp phát bộ nhớ). Nếu chạy hai lần train cùng lúc, mỗi lần sẽ chậm đi rõ rệt vì tranh CPU.

## 4.9. Tính lặp lại

`torch.manual_seed(SEED)` cố định khởi tạo lớp cuối, thứ tự xáo trộn và augmentation. Train lại cùng thiết lập sẽ cho kết quả **rất gần** lần trước, nhưng không bảo đảm giống từng chữ số, vì phép tính song song trên CPU có thể cộng theo thứ tự khác nhau.
