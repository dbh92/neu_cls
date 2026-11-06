# 4a. `train.py` là khuôn mẫu train một mô hình

File này trả lời câu hỏi: *"`train.py` có phải là cấu trúc khuôn mẫu để train một mô hình không?"*. Phần giải thích chi tiết từng bước (dòng in ra, ví dụ chọn `best.pt`, thời gian train) nằm ở [04_huan_luyen.md](04_huan_luyen.md).

## Trả lời ngắn

**Đúng.** Đây là khuôn mẫu chuẩn để train một mô hình học có giám sát bằng PyTorch. Gần như mọi script train, từ phân loại ảnh, phát hiện vật thể đến văn bản, đều có 4 phần theo đúng thứ tự này. Trong file, 4 phần được đánh dấu bằng các comment `# ===== 1.` đến `# ===== 4.`:

```
1. Dữ liệu          → DataLoader cho train và val
2. Mô hình + loss + optimizer (+ scheduler)
3. Checkpoint       → chuẩn bị nơi lưu, resume nếu cần
4. Vòng lặp epoch:
     ├─ train:    mỗi batch: forward → loss → zero_grad → backward → step
     ├─ validate: đo loss/F1 trên val (không học)
     ├─ chọn best, early stopping
     └─ lưu checkpoint + lịch sử
```

## Giải thích từng phần

### Mở đầu ([train.py:47-56](../train.py#L47-L56))

- `--model` chọn mô hình cho lần chạy. Kết quả được ghi vào `outputs/<model>/`, nên train nhiều mô hình không ghi đè nhau.
- `torch.manual_seed(SEED)` cố định thứ tự xáo batch, augmentation và khởi tạo lớp `fc` mới. Nhờ đó, chạy lại sẽ ra kết quả gần như giống hệt.
- `device` sẽ là `cpu` trên máy này. Code viết sẵn để tự dùng GPU nếu có.

### 1. Dữ liệu ([train.py:59-70](../train.py#L59-L70))

- `make_dataloader("train", train=True)` xáo ảnh và **bật augmentation** (lật, xoay, đổi sáng).
- `make_dataloader("val", train=False)` không xáo và không augmentation. Val phải "sạch" thì điểm của các epoch mới so sánh được với nhau.
- Script kiểm tra không có lớp nào trống. Lý do: nếu một lớp không có ảnh, [train.py:76](../train.py#L76) sẽ chia cho 0.

### 2. Mô hình, loss, optimizer ([train.py:73-80](../train.py#L73-L80))

Đây là 4 thành phần bài train nào cũng phải chọn:

| Thành phần | Ở đây dùng | Vai trò |
|---|---|---|
| `model` | ResNet18 pretrained, thay `fc` thành 6 đầu ra | Hàm biến ảnh thành 6 điểm số (logits) |
| `criterion` | `CrossEntropyLoss` | Đo mức sai: điểm của lớp đúng càng thấp thì loss càng lớn |
| `optimizer` | `AdamW` | Dựa vào gradient để chỉnh trọng số |
| `scheduler` | `CosineAnnealingLR` | Bước học lớn lúc đầu, giảm dần về gần 0 ở cuối để tinh chỉnh |

`class_weights` là phần tùy chọn. NEU có khoảng 204 ảnh mỗi lớp nên các trọng số đều xấp xỉ 1, gần như không có tác dụng. Phần này được giữ lại cho bộ dữ liệu lệch lớp sau này.

### 3. Checkpoint và resume ([train.py:83-115](../train.py#L83-L115))

Có hai file, mỗi file một mục đích:

- **`best.pt`**: chỉ lưu khi có epoch tốt nhất mới. `evaluate.py` và `predict.py` dùng file này.
- **`last.pt`**: lưu **sau mọi epoch**, gồm trọng số, optimizer, scheduler, lịch sử và bộ đếm early stopping. Khi train bị ngắt, `--resume` sẽ chạy tiếp đúng từ chỗ dừng.

Vì sao phải lưu cả optimizer? AdamW có "bộ nhớ" riêng, là trung bình động của gradient. Nếu chỉ lưu trọng số thì khi resume, optimizer bắt đầu lại từ đầu và quá trình train bị xáo trộn.

### 4. Vòng lặp huấn luyện ([train.py:119-175](../train.py#L119-L175))

**a) Phần cốt lõi, có trong mọi script PyTorch** ([train.py:124-135](../train.py#L124-L135)):

```python
model.train()                          # bật chế độ train (Dropout, BatchNorm cập nhật)
for images, labels in train_loader:
    outputs = model(images)            # ① forward: ảnh → logits
    loss = criterion(outputs, labels)  # ② so với nhãn thật → loss
    optimizer.zero_grad()              # ③ xóa gradient của batch trước
    loss.backward()                    # ④ tính gradient cho mọi trọng số
    optimizer.step()                   # ⑤ chỉnh trọng số theo gradient
scheduler.step()                       # sau mỗi epoch: giảm learning rate
```

Đây là 5 bước nên thuộc lòng. Lỗi hay gặp nhất là **quên `zero_grad()`**. PyTorch *cộng dồn* gradient, nên nếu quên thì gradient của các batch trước dồn vào batch sau và trọng số bị chỉnh sai.

**b) Validate** ([train.py:137-138](../train.py#L137-L138)): `run_inference` ([evaluate.py:17](../evaluate.py#L17)) gọi `model.eval()` và `torch.no_grad()`. Khi đánh giá thì bắt buộc phải có cả hai:

- `eval()` tắt Dropout và cho BatchNorm dùng thống kê đã học.
- `no_grad()` không tính gradient, nên chạy nhanh hơn và tốn ít bộ nhớ hơn.

`model.train()` ([train.py:124](../train.py#L124)) nằm **trong** vòng epoch. Lý do: sau mỗi lần validate, mô hình đang ở chế độ `eval` và phải bật lại chế độ train.

**c) Chọn best và early stopping** ([train.py:147-152](../train.py#L147-L152)):

- Tiêu chí chính là val macro F1 cao hơn. Nếu F1 bằng nhau thì chọn epoch có val loss thấp hơn. Với NEU, val F1 lên 1.0 rất sớm; nếu không có tiêu chí phụ, `best.pt` sẽ bị giữ ở epoch non đầu tiên đạt 1.0.
- Sau 7 epoch liên tiếp không có best mới thì dừng sớm. Điều kiện dừng được kiểm tra ở **đầu** vòng lặp ([train.py:120](../train.py#L120)), để khi `--resume` một lần chạy đã tự dừng thì nó dừng ngay.

**d) Lưu kết quả** ([train.py:162-175](../train.py#L162-L175)): checkpoint lưu cả `class_names`, `model_name` và `config` (mọi thiết lập lúc train). Nhờ vậy, chỉ cần file `.pt` là `predict.py` biết phải dựng mô hình nào và nhãn số 3 là lớp gì. `history.csv` và `training_curves.png` được cập nhật sau mỗi epoch, nên mở xem được ngay khi đang train.

## Dùng khuôn mẫu này cho bài khác

| Phần | Thường phải đổi | Thường giữ nguyên |
|---|---|---|
| Dữ liệu | Dataset/transform, ví dụ ảnh kèm bounding box | Cách dùng DataLoader |
| Mô hình | Kiến trúc, số đầu ra | `.to(device)` |
| Loss | Hồi quy dùng `MSELoss`; OK/NG nhị phân có thể dùng `BCEWithLogitsLoss` | — |
| Chỉ số chọn best | mAP, recall của lớp NG, ... | Logic so sánh, lưu best |
| Vòng lặp 5 bước | — | **Giữ nguyên** |
| Checkpoint, resume, early stopping | — | Giữ nguyên |

Các khuôn mẫu "công nghiệp" thường có thêm vài thứ mà file này chưa có, vì bài nhỏ chạy trên CPU chưa cần:

- mixed precision (AMP) khi có GPU;
- gradient clipping;
- ghi log ra TensorBoard;
- learning rate warmup.

Mỗi thứ chỉ là vài dòng cắm thêm vào đúng 4 phần trên; khung chính không đổi.
