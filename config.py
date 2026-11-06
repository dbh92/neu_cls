# Mọi thiết lập của dự án nằm ở đây. Muốn đổi gì thì sửa file này, không cần sửa các file khác.

# ===== Đường dẫn =====
NEU_DIR = "D:/Huan/2.Python/learning/neu_det/datasets/NEU-DET"  # Bộ NEU-DET gốc (chỉ đọc, không sửa): có train/images/<lớp>/ và validation/images/<lớp>/
SPLIT_DIR = "data/split"  # prepare_data.py chép ảnh vào đây thành train/ val/ test/
OUTPUT_DIR = "outputs"  # Mỗi mô hình có thư mục riêng: outputs/<tên mô hình>/ chứa checkpoint, biểu đồ, báo cáo

# ===== Chia dữ liệu =====
VAL_RATIO = 0.15  # Lấy 15% ảnh train của NEU (mỗi lớp) làm validation để chọn epoch tốt nhất; tập test là toàn bộ validation của NEU
SEED = 42  # Hạt giống ngẫu nhiên: giữ nguyên thì lần nào chia dữ liệu/train cũng ra kết quả giống nhau

# ===== Ảnh =====
IMAGE_SIZE = 224  # Ảnh NEU 200x200 được phóng lên 224x224 (kích thước chuẩn của các mô hình ImageNet)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")  # Các đuôi file được coi là ảnh
MEAN = (0.485, 0.456, 0.406)  # Mean theo kênh R, G, B của ImageNet (bộ ảnh mà các mô hình pretrained đã học), dùng để chuẩn hóa
STD = (0.229, 0.224, 0.225)  # Độ lệch chuẩn theo kênh R, G, B của ImageNet

# ===== Mô hình =====
MODEL_NAME = "efficientnet_b0"  # Mô hình mặc định: "resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large" hoặc "simple_cnn"; train.py --model đổi được cho từng lần chạy
PRETRAINED = True  # True: dùng trọng số đã học trên ImageNet (transfer learning); False: khởi tạo ngẫu nhiên, học từ đầu

# ===== Huấn luyện =====
EPOCHS = 2  # Số lần duyệt hết tập train; NEU dễ với transfer learning nên 20 epoch là đủ
BATCH_SIZE = 16  # Số ảnh mỗi lần học; giảm xuống 8 nếu báo hết bộ nhớ
LEARNING_RATE = 1e-4  # Tốc độ học; mọi mô hình dùng chung để so sánh công bằng (simple_cnn học từ đầu nên tăng lên 1e-3)
WEIGHT_DECAY = 1e-4  # Phạt trọng số lớn để chống học thuộc lòng (overfit)
NUM_WORKERS = 4  # Số tiến trình phụ đọc và xử lý ảnh song song (máy này có 8 nhân CPU); gặp lỗi thì đặt về 0
USE_CLASS_WEIGHTS = True  # True: lớp ít ảnh được tính loss nặng hơn; NEU cân bằng (mỗi lớp ~204 ảnh) nên mọi trọng số gần bằng 1
EARLY_STOPPING_PATIENCE = 7  # Tự dừng khi 7 epoch liên tiếp không có best mới (val F1 cao hơn, hoặc F1 bằng mà val loss thấp hơn); None = tắt, luôn chạy đủ EPOCHS

# ===== Augmentation (biến đổi ngẫu nhiên ảnh train để mô hình học tổng quát hơn) =====
# Chỉ bật phép biến đổi nào mà ảnh THẬT cũng có thể như vậy: bề mặt thép không có chiều trên/dưới, trái/phải nên lật được
HORIZONTAL_FLIP = True  # Lật ngang
VERTICAL_FLIP = True  # Lật dọc
ROTATION_DEGREES = 10  # Xoay ngẫu nhiên trong khoảng ±10 độ; 0 = không xoay
BRIGHTNESS = 0.2  # Thay đổi độ sáng ngẫu nhiên ±20% (ánh sáng chiếu khác nhau); 0 = tắt
CONTRAST = 0.2  # Thay đổi độ tương phản ngẫu nhiên ±20%; 0 = tắt
