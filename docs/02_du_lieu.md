# 2. Dữ liệu: từ ảnh gốc đến tensor

Phần này đi theo một tấm ảnh, từ thư mục NEU gốc cho tới lúc thành con số đưa vào mô hình.

## 2.1. Nguồn dữ liệu

Bộ **NEU Surface Defect Database** (bản NEU-DET) nằm ở `NEU_DIR` trong `config.py`:

```
NEU-DET/
├── train/images/<lớp>/*.jpg        240 ảnh mỗi lớp
└── validation/images/<lớp>/*.jpg    60 ảnh mỗi lớp
```

Thư mục này là dữ liệu của bài NumPy cũ. **Không bao giờ sửa** nó. `prepare_data.py` chỉ **sao chép** ra `data/split`.

## 2.2. `prepare_data.py`: chia dữ liệu

### Kết quả chia

| Tập | Lấy từ | Mỗi lớp | Tổng | Dùng để |
|---|---|---|---|---|
| train | 85% `train` gốc | 204 (patches 203) | 1.223 | Mô hình học |
| val | 15% `train` gốc | 36 | 216 | Chọn epoch tốt nhất, early stopping |
| test | **toàn bộ** `validation` gốc | 60 | 360 | Chấm điểm cuối cùng |

Tập test được chọn **đúng bằng** tập mà bản CNN NumPy cũ đã dùng (đạt 0.942), nên kết quả hai bên so sánh trực tiếp được.

### Các bước trong code

1. **Kiểm tra nguồn** ([prepare_data.py:33-43](../prepare_data.py#L33-L43)): hai thư mục `train/images` và `validation/images` phải tồn tại và có **cùng danh sách lớp**. Sai thì in lỗi và thoát mã 1.
2. **Không ghi đè** ([prepare_data.py:45-49](../prepare_data.py#L45-L49)): nếu `data/split` đã có, script dừng, trừ khi có `--overwrite`. Lý do: chia lại sẽ đổi tập val, khiến các mô hình đã train trước đó không còn so sánh công bằng.
3. Với **từng lớp** ([prepare_data.py:54-73](../prepare_data.py#L54-L73)):
   1. Tính MD5 của mọi ảnh test và ghi vào từ điển `seen` **trước**.
   2. Duyệt ảnh train. Ảnh nào có MD5 đã nằm trong `seen` thì **bỏ** (in `duplicate skipped: ...`).
   3. Xáo trộn ảnh train còn lại bằng `random.Random(SEED)`.
   4. `num_val = round(số ảnh × 0.15)` ảnh đầu → val, phần còn lại → train.
   5. Sao chép bằng `shutil.copy2`.

### Vì sao phải bỏ ảnh trùng?

Bộ gốc có `patches_105.jpg` giống hệt từng byte `patches_101.jpg`. Nếu một bản rơi vào train và bản kia rơi vào val, mô hình sẽ được "chấm điểm trên ảnh đã học", và điểm val cao giả tạo. Bài Magnetic Tile trước đã gặp đúng lỗi này.

Ảnh test được ghi vào `seen` **trước** ảnh train, nên nếu một ảnh train trùng với ảnh test thì ảnh train bị bỏ, còn tập test giữ nguyên. Tập test không bao giờ bị "rò" vào train.

`seen` dùng chung cho mọi lớp, nên script cũng bắt được ảnh trùng **khác lớp** (một ảnh bị gắn hai nhãn).

### Vì sao chia lần nào cũng giống nhau?

`random.Random(config.SEED)` tạo một bộ sinh ngẫu nhiên riêng với hạt giống cố định, và danh sách ảnh được sắp theo tên trước khi xáo. Cùng `SEED`, cùng ảnh gốc thì `--overwrite` lần nào cũng ra đúng cùng một cách chia.

## 2.3. `dataset.py`: từ file ảnh thành tensor

### Tên lớp

[`get_class_names()`](../dataset.py#L9-L11) đọc tên thư mục con của `data/split/train` và **sắp theo alphabet**. Lớp thứ `i` trong danh sách có nhãn số `i`. Danh sách này được lưu vào checkpoint, nên về sau mọi script đọc thứ tự lớp từ checkpoint chứ không đọc lại thư mục.

### `ImageClassificationDataset`

[dataset.py:30-50](../dataset.py#L30-L50). Khi khởi tạo, lớp này chỉ ghi lại danh sách `(đường dẫn, nhãn số)` theo thứ tự lớp rồi theo tên file. Ảnh **chưa** được mở; ảnh chỉ được mở khi DataLoader cần (`__getitem__`), để không tốn RAM.

Trong `__getitem__`:

```python
image = Image.open(path).convert("RGB")   # ảnh xám 1 kênh -> 3 kênh giống nhau
image = self.transform(image)            # resize, augmentation, ToTensor, Normalize
return image, label
```

Ảnh NEU là ảnh xám. Mô hình pretrained cần 3 kênh R, G, B, nên `convert("RGB")` chép kênh xám thành 3 kênh giống nhau.

### Chuỗi biến đổi (`build_transforms`)

[dataset.py:14-27](../dataset.py#L14-L27):

| Bước | train | val/test | Kết quả |
|---|---|---|---|
| `Resize((224, 224))` | ✔ | ✔ | Ảnh PIL 224×224 (phóng từ 200×200) |
| `RandomHorizontalFlip()` | ✔ (50%) | | Lật ngang |
| `RandomVerticalFlip()` | ✔ (50%) | | Lật dọc |
| `RandomRotation(10)` | ✔ | | Xoay ngẫu nhiên −10°..+10°, góc trống tô đen |
| `ColorJitter(0.2, 0.2)` | ✔ | | Độ sáng và tương phản ×(0,8..1,2) |
| `ToTensor()` | ✔ | ✔ | Tensor `(3, 224, 224)`, giá trị 0..1 |
| `Normalize(MEAN, STD)` | ✔ | ✔ | `(x − mean) / std` theo từng kênh, giá trị khoảng −2..+2,6 |

**Nguyên tắc augmentation:** chỉ bật phép biến đổi nào mà ảnh thật cũng có thể như thế. Bề mặt thép không có chiều trên/dưới, trái/phải nên lật được; ánh sáng ở chuyền có thể thay đổi nên đổi sáng được. Tập val/test **không bao giờ** augmentation, để mỗi lần đánh giá ra cùng một con số.

**Vì sao chuẩn hóa bằng mean/std của ImageNet?** Các mô hình pretrained đã học trên ảnh được chuẩn hóa đúng như vậy. Đưa ảnh có thang giá trị khác vào thì phần thân đã học sẽ "nhìn" sai.

### DataLoader

[`make_dataloader(split, class_names, train)`](../dataset.py#L53-L66):

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `batch_size` | 16 | Gom 16 ảnh thành tensor `(16, 3, 224, 224)` |
| `shuffle` | chỉ train | Train xáo mỗi epoch; val/test giữ thứ tự để khớp với `dataset.samples` |
| `num_workers` | 4 | 4 tiến trình phụ đọc ảnh song song với lúc mô hình tính |
| `persistent_workers` | `True` khi workers > 0 | Không tạo lại tiến trình mỗi epoch (trên Windows tạo tiến trình rất chậm) |

`shuffle=False` cho val/test là điều kiện để `evaluate.py` ghép được dự đoán thứ `i` với file ảnh `dataset.samples[i]`.

Với 1.223 ảnh train và batch 16, mỗi epoch có 77 batch (batch cuối 7 ảnh).

## 2.4. Hình dạng dữ liệu qua từng bước

```
file .jpg 200×200 xám
  → PIL "RGB" 200×200                      (convert)
  → PIL 224×224                            (Resize)
  → [train] lật / xoay / đổi sáng          (augmentation)
  → tensor (3, 224, 224), 0..1             (ToTensor)
  → tensor (3, 224, 224), đã chuẩn hóa     (Normalize)
  → batch  (16, 3, 224, 224)               (DataLoader)
  → mô hình → logits (16, 6)
  → softmax → xác suất (16, 6), mỗi hàng cộng lại = 1
  → argmax → nhãn đoán (16,), mỗi số 0..5
```

## 2.5. Kiểm tra nhanh

```powershell
.venv\Scripts\python.exe dataset.py
```

In ra danh sách lớp, số ảnh từng tập/lớp và shape của một batch. Kết quả đúng: train 1.223, val 216, test 360 ảnh, batch `(16, 3, 224, 224)`.
