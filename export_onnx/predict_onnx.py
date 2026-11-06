# Dự đoán loại lỗi bằng model.onnx, KHÔNG cần PyTorch: chỉ dùng onnxruntime, numpy, pillow.
# File này cố ý không import config.py hay các file khác của dự án, để chép riêng 3 file (predict_onnx.py, model.onnx, model.json) sang máy khác là chạy được.
# Mọi thông số (lớp, kích thước ảnh, mean/std) đọc từ model.json do export.py tạo ra.

import argparse  # Thư viện đọc tham số dòng lệnh
import json  # Thư viện đọc file JSON (model.json)
import os  # Thư viện làm việc với file/thư mục
import time  # Thư viện đo thời gian xử lý mỗi ảnh
import numpy as np  # Thư viện tính toán mảng số, thay cho tensor của PyTorch
import onnxruntime as ort  # Thư viện chạy mô hình ONNX (nhẹ hơn PyTorch rất nhiều)
from PIL import Image  # Thư viện mở file ảnh

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")  # Các đuôi file được coi là ảnh (giống config.IMAGE_EXTENSIONS)
DEFAULT_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))  # Mặc định model.onnx và model.json nằm cùng thư mục với file này
TOP_K = 3  # Số lớp có xác suất cao nhất được in ra cho mỗi ảnh


class OnnxClassifier:  # Nạp model.onnx + model.json một lần, sau đó dự đoán bao nhiêu ảnh cũng được
    def __init__(self, model_dir=DEFAULT_MODEL_DIR):  # model_dir: thư mục chứa model.onnx và model.json
        with open(os.path.join(model_dir, "model.json"), encoding="utf-8") as file:  # Mở file thông số
            self.info = json.load(file)  # Đọc thành dict
        self.class_names = self.info["class_names"]  # Danh sách lớp đúng thứ tự lúc train
        self.image_size = self.info["image_size"]  # Kích thước ảnh đầu vào, vd 224
        self.mean = np.array(self.info["mean"], dtype=np.float32)  # Mean theo kênh R, G, B dùng để chuẩn hóa
        self.std = np.array(self.info["std"], dtype=np.float32)  # Độ lệch chuẩn theo kênh R, G, B
        self.session = ort.InferenceSession(os.path.join(model_dir, self.info["model_file"]),
                                            providers=["CPUExecutionProvider"])  # Nạp mô hình ONNX, chạy bằng CPU
        self.input_name = self.session.get_inputs()[0].name  # Tên đầu vào của mô hình ("image")

    def preprocess(self, image):  # Ảnh PIL -> mảng (3, H, W); PHẢI giống hệt build_transforms(train=False) trong dataset.py
        image = image.convert("RGB")  # Ép về 3 kênh màu theo thứ tự R, G, B (ảnh xám cũng thành 3 kênh)
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)  # Resize kiểu bilinear, giống transforms.Resize
        array = np.asarray(image, dtype=np.float32) / 255.0  # Mảng (H, W, 3) giá trị 0..1, giống transforms.ToTensor
        array = (array - self.mean) / self.std  # Chuẩn hóa theo từng kênh, giống transforms.Normalize
        return array.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W): mô hình cần kênh màu đứng trước

    def predict_probabilities(self, images):  # Danh sách ảnh PIL -> danh sách xác suất, vd [[0.97, 0.01, ...], ...]
        batch = np.stack([self.preprocess(image) for image in images])  # Gộp thành batch (số ảnh, 3, H, W)
        logits = self.session.run(None, {self.input_name: batch})[0]  # Chạy mô hình, lấy logits (số ảnh, số lớp)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))  # Softmax: trừ max trước để exp không bị tràn số
        return (exp / exp.sum(axis=1, keepdims=True)).tolist()  # Chia cho tổng -> xác suất có tổng = 1

    def decide(self, probs):  # Xác suất của 1 ảnh -> chỉ số lớp kết luận; cùng quy tắc với decide_predictions trong evaluate.py
        return probs.index(max(probs))  # Chọn lớp có xác suất cao nhất (argmax)


def collect_image_paths(paths):  # Gom danh sách file ảnh: file thì giữ nguyên, thư mục thì lấy mọi ảnh bên trong
    image_paths = []  # Danh sách kết quả
    for path in paths:  # Duyệt từng đường dẫn người dùng đưa vào
        if os.path.isdir(path):  # Nếu là thư mục
            for name in sorted(os.listdir(path)):  # Duyệt các file theo thứ tự tên
                if name.lower().endswith(IMAGE_EXTENSIONS):  # Chỉ lấy file ảnh
                    image_paths.append(os.path.join(path, name))  # Thêm đường dẫn đầy đủ
        else:  # Nếu là file
            image_paths.append(path)  # Thêm luôn
    return image_paths  # Trả về danh sách


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file
    parser = argparse.ArgumentParser(description="Predict the defect class with model.onnx (no PyTorch needed)")  # Tạo bộ phân tích tham số
    parser.add_argument("paths", nargs="+", help="image files or folders")  # Một hoặc nhiều ảnh/thư mục (bắt buộc)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="folder with model.onnx and model.json")  # Thư mục chứa mô hình
    args = parser.parse_args()  # Đọc tham số

    if not os.path.isfile(os.path.join(args.model_dir, "model.json")):  # Chưa xuất mô hình
        print("model.json not found in {}. Run export_onnx/export.py first.".format(args.model_dir))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình
    classifier = OnnxClassifier(args.model_dir)  # Nạp mô hình
    classifier.predict_probabilities([Image.new("RGB", (classifier.image_size, classifier.image_size))])  # Chạy thử 1 ảnh đen (lần đầu luôn chậm) để thời gian đo ở ảnh đầu tiên không bị sai lệch

    timings = []  # Thời gian xử lý (ms) của từng ảnh, để tính trung bình
    for image_path in collect_image_paths(args.paths):  # Duyệt từng ảnh
        start = time.perf_counter()  # Bắt đầu bấm giờ: tính cả mở ảnh, xử lý và dự đoán
        try:  # Thử mở ảnh
            image = Image.open(image_path).convert("RGB")  # Mở ảnh và ép về 3 kênh màu
        except OSError as error:  # File không tồn tại hoặc không phải ảnh
            print("Skipping {}: {}".format(image_path, error))  # Báo lỗi rồi chuyển sang ảnh tiếp theo
            continue  # Sang ảnh kế tiếp
        probs = classifier.predict_probabilities([image])[0]  # Xác suất từng lớp của ảnh này
        prediction = classifier.decide(probs)  # Kết luận lớp (chỉ số)
        elapsed_ms = (time.perf_counter() - start) * 1000  # Dừng bấm giờ, đổi giây sang mili-giây
        timings.append(elapsed_ms)  # Lưu lại để tính trung bình
        folder = os.path.basename(os.path.dirname(os.path.abspath(image_path)))  # Tên thư mục cha, vd "scratches" nếu ảnh nằm trong .../scratches/
        true_label = folder if folder in classifier.class_names else None  # Nhãn thật nếu thư mục cha là tên lớp
        wrong = true_label is not None and true_label != classifier.class_names[prediction]  # Có biết nhãn thật và đoán sai không
        top = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:TOP_K]  # Chỉ số TOP_K lớp có xác suất cao nhất
        top_text = "  ".join("{} {:6.2%}".format(classifier.class_names[i], probs[i]) for i in top)  # vd "scratches 98.10%  crazing  1.20%  ..."
        print("{:<15} | {} | {:6.1f} ms | {}{}".format(classifier.class_names[prediction], top_text, elapsed_ms, image_path,
                                                       "  <- WRONG (true {})".format(true_label) if wrong else ""))  # In giống predict.py

    if timings:  # Có ít nhất 1 ảnh được dự đoán
        print("{} images | average {:.1f} ms/image | slowest {:.1f} ms (open + preprocess + predict)".format(
            len(timings), sum(timings) / len(timings), max(timings)))  # Tổng kết tốc độ
