import torch  # Thư viện PyTorch chính
import torch.nn as nn  # Module chứa các lớp mạng nơ-ron
from torchvision import models  # Các mô hình CNN có sẵn (ResNet, EfficientNet, MobileNet, ...)

MODEL_NAMES = ("resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large", "simple_cnn")  # Các tên mô hình hợp lệ


def conv_block(in_channels, out_channels):  # Một khối tích chập: Conv -> BatchNorm -> ReLU -> MaxPool
    return nn.Sequential(  # Xếp các lớp nối tiếp
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Bộ lọc 3x3 quét ảnh tìm đặc trưng (cạnh, vết, hoa văn)
        nn.BatchNorm2d(out_channels),  # Chuẩn hóa theo batch: học ổn định và nhanh hơn
        nn.ReLU(),  # Hàm kích hoạt: số âm thành 0
        nn.MaxPool2d(kernel_size=2),  # Giữ giá trị lớn nhất trong mỗi ô 2x2 -> ảnh nhỏ đi một nửa
    )


class SimpleCNN(nn.Module):  # CNN tự viết, học từ đầu (giống bản NumPy về ý tưởng); dùng để thấy transfer learning hơn bao nhiêu
    def __init__(self, num_classes):  # num_classes: số lớp (6 với NEU)
        super().__init__()  # Gọi hàm khởi tạo của nn.Module (bắt buộc)
        self.features = nn.Sequential(  # Phần trích xuất đặc trưng: 4 khối, ảnh 224 -> 112 -> 56 -> 28 -> 14
            conv_block(3, 32),  # 3 kênh màu -> 32 kênh đặc trưng
            conv_block(32, 64),  # 32 -> 64
            conv_block(64, 128),  # 64 -> 128
            conv_block(128, 256),  # 128 -> 256
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # Lấy trung bình mỗi kênh về 1 số -> chạy được với mọi kích thước ảnh
        self.classifier = nn.Sequential(  # Phần phân loại
            nn.Flatten(),  # (batch, 256, 1, 1) -> (batch, 256)
            nn.Dropout(0.3),  # Khi train, tắt ngẫu nhiên 30% nơ-ron để chống học thuộc lòng
            nn.Linear(256, num_classes),  # 256 -> số lớp: điểm số (logits) của từng lớp
        )

    def forward(self, x):  # Ảnh đi qua mô hình theo thứ tự nào
        x = self.features(x)  # Trích xuất đặc trưng
        x = self.pool(x)  # Gom về 1 số mỗi kênh
        return self.classifier(x)  # Trả về logits (batch, num_classes)


def build_model(model_name, num_classes, pretrained=True):  # Tạo mô hình theo tên; transfer learning = giữ phần thân đã học, thay phần đầu phân loại
    if model_name == "simple_cnn":  # CNN tự viết
        return SimpleCNN(num_classes)  # Không có trọng số pretrained, tham số pretrained bị bỏ qua

    if model_name in ("resnet18", "resnet50"):  # Họ ResNet: phần đầu phân loại là một lớp Linear tên "fc"
        constructor, weights = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),  # 11,2 triệu tham số (với 6 lớp)
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),  # 23,5 triệu tham số (với 6 lớp), sâu hơn nhiều, chậm hơn khoảng 3 lần
        }[model_name]  # Chọn hàm tạo và trọng số theo tên
        model = constructor(weights=weights if pretrained else None)  # Tạo mô hình (lần đầu sẽ tải trọng số về ~/.cache/torch)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Thay lớp 1000 lớp ImageNet bằng lớp mới; in_features = 512 (resnet18) hoặc 2048 (resnet50)
        return model  # Trả về mô hình

    if model_name == "efficientnet_b0":  # EfficientNet: classifier = Sequential(Dropout, Linear) -> lớp Linear nằm ở vị trí [1]
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)  # 4,0 triệu tham số (với 6 lớp)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Chỉ thay lớp Linear, giữ nguyên Dropout đứng trước
        return model  # Trả về mô hình

    if model_name == "mobilenet_v3_large":  # MobileNetV3: classifier = Sequential(Linear, Hardswish, Dropout, Linear) -> lớp cuối ở vị trí [3]
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)  # 4,2 triệu tham số (với 6 lớp), thiết kế cho máy yếu
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)  # Thay lớp Linear cuối cùng (1280 -> số lớp)
        return model  # Trả về mô hình

    raise ValueError("Unknown model '{}', choose one of {}".format(model_name, MODEL_NAMES))  # Tên sai thì báo lỗi


def gradcam_layer(model, model_name):  # Lớp tích chập cuối cùng của mô hình: Grad-CAM đọc "bản đồ đặc trưng" ở đây (gradcam.py)
    if model_name in ("resnet18", "resnet50"):  # ResNet
        return model.layer4  # Khối cuối, ảnh 224 -> bản đồ 7x7
    return model.features[-1]  # EfficientNet, MobileNet và SimpleCNN đều đặt phần tích chập trong model.features


def load_model_from_checkpoint(checkpoint_path, device):  # Nạp mô hình đã train từ file checkpoint (dùng cho evaluate.py, predict.py, ...)
    checkpoint = torch.load(checkpoint_path, map_location=device)  # Đọc file; map_location đưa tensor về đúng thiết bị (CPU/GPU)
    model = build_model(checkpoint["model_name"], len(checkpoint["class_names"]), pretrained=False)  # Tạo mô hình rỗng cùng kiến trúc (không cần tải pretrained vì sẽ nạp trọng số ngay sau)
    model.load_state_dict(checkpoint["model"])  # Nạp trọng số đã học
    model.to(device)  # Chuyển mô hình sang thiết bị
    model.eval()  # Chế độ đánh giá: tắt Dropout, BatchNorm dùng thống kê đã học
    return model, checkpoint  # Trả về mô hình và thông tin đi kèm (class_names, image_size, ...)


if __name__ == '__main__':  # Chạy trực tiếp file để kiểm tra các mô hình
    input_data = torch.rand(2, 3, 224, 224)  # 2 ảnh giả ngẫu nhiên kích thước 3x224x224
    for name in MODEL_NAMES:  # Với từng mô hình
        model = build_model(name, num_classes=6)  # Tạo mô hình 6 lớp (NEU)
        model.eval()  # Chế độ đánh giá để BatchNorm không cần batch lớn
        num_params = sum(p.numel() for p in model.parameters())  # Đếm tổng số tham số
        with torch.no_grad():  # Không cần gradient khi chỉ kiểm tra
            output_shape = tuple(model(input_data).shape)  # Phải là (2, 6)
        print("{:<19} output {} | {:>10,} params | Grad-CAM layer: {}".format(
            name, output_shape, num_params, type(gradcam_layer(model, name)).__name__))  # In kết quả
