# 6. Grad-CAM: mô hình nhìn vào đâu? (`gradcam.py`)

## 6.1. Để làm gì?

Độ chính xác cao chưa chắc mô hình đã học **đúng thứ**. Mô hình có thể dựa vào "mẹo" của bộ dữ liệu: độ sáng nền, viền ảnh, một góc cố định... Những mẹo này đúng trên bộ NEU nhưng sẽ sai khi sang chuyền khác có ánh sáng khác.

Grad-CAM vẽ một **bản đồ nhiệt** phủ lên ảnh:

- **đỏ**: vùng làm mô hình tin vào lớp được đoán nhiều nhất;
- **xanh**: vùng hầu như không ảnh hưởng.

Mô hình tốt có vùng đỏ nằm trên vết lỗi (vết xước, đốm rỗ, mảng loang). Mô hình đáng ngờ có vùng đỏ ở viền, ở góc, hoặc trên nền trơn.

## 6.2. Lệnh

```powershell
# Mặc định: 3 ảnh test đầu tiên của mỗi lớp (18 ảnh = 1 trang)
.venv\Scripts\python.exe gradcam.py --checkpoint outputs/resnet18/best.pt

# Các ảnh bị đoán sai (phải chạy evaluate.py trên test trước)
.venv\Scripts\python.exe evaluate.py --checkpoint outputs/resnet18/best.pt
.venv\Scripts\python.exe gradcam.py --checkpoint outputs/resnet18/best.pt --misclassified --save outputs/resnet18/gradcam_wrong

# Ảnh hoặc thư mục bất kỳ, hiện cửa sổ
.venv\Scripts\python.exe gradcam.py data/split/test/scratches --checkpoint outputs/resnet18/best.pt --show

# 6 ảnh mỗi lớp
.venv\Scripts\python.exe gradcam.py --checkpoint outputs/resnet18/best.pt --per-class 6
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `paths` | (trống) | Ảnh/thư mục cần giải thích; trống thì lấy `--per-class` ảnh test mỗi lớp |
| `--checkpoint` | `outputs/<MODEL_NAME>/best.pt` | Mô hình |
| `--misclassified` | tắt | Lấy ảnh trong `misclassified_test.csv` cạnh checkpoint |
| `--per-class` | 3 | Số ảnh mỗi lớp khi không đưa `paths` |
| `--save` | `<thư mục checkpoint>/gradcam` | Nơi lưu `gradcam_01.png`, `gradcam_02.png`, ... |
| `--show` | tắt | Hiện cửa sổ; đóng cửa sổ để sang trang sau |

Thứ tự ưu tiên chọn ảnh: `--misclassified` → `paths` → mặc định.

## 6.3. Cách tính, từng bước

Hàm [`compute_gradcam`](../gradcam.py#L23-L44). Ví dụ với ResNet18:

```
ảnh (1, 3, 224, 224)
   │ forward
   ▼
layer4  ──► A = bản đồ đặc trưng (512, 7, 7)       ← forward hook cất A
   │
   ▼
... → logits (1, 6) → chọn lớp c có xác suất cao nhất
   │ backward từ logits[c]
   ▼
G = ∂logits[c] / ∂A   (512, 7, 7)                  ← hook trên tensor A cất G
```

1. **Bắt bản đồ đặc trưng.** `layer.register_forward_hook` gắn một hàm được gọi ngay sau khi `layer4` tính xong. Hàm này cất output `A`, và gắn tiếp `A.register_hook(...)` để lúc backward chạy qua `A` thì cất luôn gradient `G`.
2. **Forward** trong `torch.enable_grad()`. Bình thường lúc dự đoán ta tắt gradient; ở đây phải bật để backward được. Mô hình vẫn ở chế độ `eval()`.
3. **Chọn lớp** `c` = argmax của xác suất, tức Grad-CAM luôn giải thích cho **lớp được đoán**, không phải lớp thật.
4. **Backward** từ `logits[0, c]` (logit, không phải xác suất).
5. **Trọng số từng kênh:** `w_k = trung bình của G_k trên 7×7`. Kênh nào làm tăng điểm của lớp `c` nhiều thì `w_k` lớn.
6. **Cộng các kênh:** `cam = ReLU(Σ_k w_k · A_k)`, ra lưới `(7, 7)`. ReLU chỉ giữ phần **làm tăng** điểm của lớp `c`.
7. **Chuẩn hóa** về 0..1 (chia cho giá trị lớn nhất).
8. **Phóng to** 7×7 → 224×224 bằng nội suy bilinear để phủ lên ảnh.

Hook luôn được gỡ trong khối `finally`, kể cả khi có lỗi. Nếu không gỡ, mỗi ảnh sẽ gắn thêm một hook chồng lên nhau.

### Vì sao bản đồ nhiệt thô và nhòe?

Lớp tích chập cuối của ResNet, EfficientNet, MobileNet chỉ có lưới **7×7** với ảnh 224. Mỗi ô 7×7 ứng với khoảng 32×32 pixel ảnh. Phóng lên 224×224 nên vùng đỏ luôn là các mảng tròn mờ, không bám sát từng nét vết lỗi. Đó là chuyện bình thường. `simple_cnn` có lưới 14×14 nên chi tiết hơn một chút.

## 6.4. Đọc hình

Mỗi trang tối đa 18 ảnh (6 hàng × 3 ảnh). Mỗi ảnh có 2 ô:

| Ô trái | Ô phải |
|---|---|
| Ảnh gốc, tiêu đề `true <lớp>` và tên file | Ảnh gốc phủ bản đồ nhiệt (độ trong 45%), tiêu đề `pred <lớp>` và độ tự tin |

Màu tiêu đề ô phải: **xanh lá** = đúng, **đỏ** = sai, **đen** = không biết nhãn thật (ảnh không nằm trong thư mục mang tên lớp).

### Câu hỏi nên đặt ra khi xem

1. Với ảnh **đúng**: vùng đỏ có nằm trên vết lỗi không? Nếu đúng mà vùng đỏ lại ở nền, mô hình đúng "nhờ may".
2. Với ảnh **sai**: mô hình đã nhìn vào đâu? Vùng đỏ lệch khỏi vết lỗi gợi ý nó dựa vào thứ khác (độ sáng nền, vân bề mặt).
3. Vùng đỏ có hay **lặp lại ở cùng một vị trí** (góc, viền) trên nhiều ảnh không? Đó là dấu hiệu mô hình học mẹo.

Kết quả đã quan sát được trên NEU ghi ở mục 6 của `README.md`.
