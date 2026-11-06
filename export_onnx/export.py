# Xuất một checkpoint (mặc định outputs/<MODEL_NAME>/best.pt) sang ONNX (model.onnx + model.json), rồi tự kiểm chứng ONNX cho kết quả giống PyTorch.
# Chạy từ thư mục gốc dự án: python export_onnx/export.py [--checkpoint outputs/efficientnet_b0/best.pt]

import argparse  # Thư viện đọc tham số dòng lệnh
import datetime  # Thư viện ngày giờ, ghi thời điểm xuất vào model.json
import json  # Thư viện ghi file JSON
import os  # Thư viện làm việc với file/thư mục
import sys  # Thư viện hệ thống, dùng để thêm thư mục gốc vào đường dẫn import
import time  # Thư viện đo thời gian
import numpy as np  # Thư viện tính toán mảng số
import torch  # Thư viện PyTorch chính
from PIL import Image  # Thư viện mở file ảnh

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Thư mục gốc dự án (thư mục cha của export_onnx/)
sys.path.insert(0, PROJECT_DIR)  # Để import được config.py, model.py... ở thư mục gốc

import config  # Các thiết lập của dự án (config.py)
from dataset import make_dataloader  # Tạo DataLoader cho một tập
from evaluate import run_inference, decide_predictions  # Dự đoán bằng PyTorch và quy tắc ra kết luận, dùng làm chuẩn để so sánh
from model import load_model_from_checkpoint  # Nạp mô hình đã train
from predict_onnx import OnnxClassifier  # Bộ dự đoán bằng ONNX (cùng thư mục)

EXPORT_DIR = os.path.dirname(os.path.abspath(__file__))  # Thư mục export_onnx/, nơi lưu model.onnx và model.json
MAX_PROB_DIFF = 1e-4  # Sai lệch xác suất tối đa cho phép giữa PyTorch và ONNX (0.01%)


def checkpoint_label(checkpoint_path):  # Đường dẫn checkpoint để ghi vào model.json: tương đối nếu nằm trong dự án, đầy đủ nếu ở ngoài
    full_path = os.path.abspath(checkpoint_path)  # Đường dẫn đầy đủ
    try:  # Thử tính đường dẫn tương đối so với thư mục dự án
        relative_path = os.path.relpath(full_path, PROJECT_DIR)  # vd outputs\resnet18\best.pt
    except ValueError:  # Khác ổ đĩa: Windows không tính được đường dẫn tương đối
        return full_path.replace("\\", "/")  # Dùng đường dẫn đầy đủ
    if relative_path.startswith(".."):  # Cùng ổ nhưng nằm ngoài thư mục dự án: khó đọc
        return full_path.replace("\\", "/")  # Dùng đường dẫn đầy đủ
    return relative_path.replace("\\", "/")  # Trong dự án: đường dẫn tương đối, vd outputs/resnet18/best.pt


def export_model(model, checkpoint, checkpoint_path, output_dir):  # Ghi model.onnx và model.json vào output_dir
    image_size = checkpoint["image_size"]  # Kích thước ảnh lúc train, vd 224
    dummy_input = torch.zeros(2, 3, image_size, image_size)  # Ảnh giả: exporter chạy thử mô hình một lần để ghi lại các phép tính; batch 2 để không bị cố định batch = 1
    torch.onnx.export(
        model,  # Mô hình đã ở chế độ eval (load_model_from_checkpoint đã gọi model.eval())
        (dummy_input,),  # Đầu vào mẫu
        os.path.join(output_dir, "model.onnx"),  # File ONNX đầu ra
        input_names=["image"],  # Tên đầu vào trong file ONNX
        output_names=["logits"],  # Tên đầu ra: logits, chưa qua softmax
        dynamic_shapes=({0: torch.export.Dim("batch")},),  # Chiều 0 (số ảnh) không cố định -> gửi 1 hay nhiều ảnh một lần đều được
        external_data=False,  # Gộp trọng số vào chung 1 file model.onnx (mặc định tách ra file .data riêng)
        verbose=False,  # Tắt thông báo của exporter (có emoji, in ra console cp1252 sẽ gây UnicodeEncodeError)
    )
    info = {  # Thông số đi kèm mô hình: chương trình nào dùng model.onnx cũng phải đọc từ đây, không tự gõ lại
        "model_file": "model.onnx",  # Tên file mô hình
        "model_name": checkpoint["model_name"],  # Kiến trúc, vd "resnet18"
        "class_names": checkpoint["class_names"],  # Thứ tự lớp lúc train (6 lớp NEU)
        "image_size": image_size,  # Kích thước ảnh đầu vào
        "color": "RGB",  # Thứ tự kênh màu (C#/OpenCV mặc định BGR, phải đổi)
        "resize": "bilinear",  # Kiểu resize
        "mean": list(config.MEAN),  # Mean để chuẩn hóa
        "std": list(config.STD),  # Std để chuẩn hóa
        "input_name": "image",  # Tên đầu vào, shape (batch, 3, image_size, image_size), float32
        "output_name": "logits",  # Tên đầu ra, shape (batch, số lớp); cần softmax để ra xác suất
        "checkpoint": checkpoint_label(checkpoint_path),  # Checkpoint gốc
        "epoch": checkpoint["epoch"],  # Epoch của checkpoint
        "exported_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Thời điểm xuất, dùng như phiên bản mô hình
    }
    with open(os.path.join(output_dir, "model.json"), "w", encoding="utf-8") as file:  # Mở file để ghi
        json.dump(info, file, indent=2, ensure_ascii=False)  # Ghi dict thành JSON dễ đọc


def verify(model, classifier, split):  # So kết quả PyTorch và ONNX trên toàn bộ một tập; trả về True nếu đạt
    loader = make_dataloader(split, classifier.class_names, train=False, image_size=classifier.image_size)  # Ảnh của tập, xử lý bằng torchvision
    _, labels, torch_probs = run_inference(model, loader, torch.device("cpu"))  # Xác suất từ PyTorch (chuẩn để so sánh)

    onnx_probs = []  # Xác suất từ ONNX của từng ảnh
    start = time.perf_counter()  # Bắt đầu bấm giờ
    for path, _ in loader.dataset.samples:  # Cùng thứ tự ảnh với PyTorch (shuffle=False)
        image = Image.open(path).convert("RGB")  # Mở ảnh
        onnx_probs.extend(classifier.predict_probabilities([image]))  # Xử lý ảnh bằng numpy + chạy ONNX
    pipeline_ms = (time.perf_counter() - start) * 1000 / len(onnx_probs)  # Thời gian trung bình mỗi ảnh: mở ảnh + xử lý + dự đoán

    max_diff = float(np.abs(np.array(torch_probs) - np.array(onnx_probs)).max())  # Sai lệch xác suất lớn nhất
    torch_predictions = decide_predictions(torch_probs)  # Kết luận của PyTorch
    onnx_predictions = [classifier.decide(probs) for probs in onnx_probs]  # Kết luận của ONNX
    same = sum(1 for a, b in zip(torch_predictions, onnx_predictions) if a == b)  # Số ảnh có kết luận trùng nhau
    correct = sum(1 for p, label in zip(onnx_predictions, labels) if p == label)  # Số ảnh ONNX đoán đúng

    print("Verify on {} ({} images):".format(split, len(labels)))  # Tiêu đề
    print("  max probability difference PyTorch vs ONNX: {:.2e} (limit {:.0e})".format(max_diff, MAX_PROB_DIFF))  # Sai lệch xác suất
    print("  same decision: {}/{}".format(same, len(labels)))  # Số kết luận trùng
    print("  ONNX accuracy: {}/{} = {:.2%}".format(correct, len(labels), correct / len(labels)))  # Độ chính xác của ONNX
    print("  ONNX full pipeline (open + preprocess + predict): {:.1f} ms/image".format(pipeline_ms))  # Tốc độ thực tế
    return max_diff <= MAX_PROB_DIFF and same == len(labels)  # Đạt khi sai lệch nhỏ và mọi kết luận trùng nhau


def benchmark(model, classifier, runs=50):  # Đo tốc độ riêng phần mô hình (1 ảnh mỗi lần) của PyTorch và ONNX
    batch = np.random.rand(1, 3, classifier.image_size, classifier.image_size).astype(np.float32)  # 1 ảnh giả đã xử lý sẵn
    tensor = torch.from_numpy(batch)  # Cùng dữ liệu dưới dạng tensor cho PyTorch
    timings = {}  # Thời gian trung bình (ms) của từng bên
    with torch.no_grad():  # Không tính gradient
        for name, run in (("PyTorch", lambda: model(tensor)),
                          ("ONNX", lambda: classifier.session.run(None, {classifier.input_name: batch}))):  # Hai cách chạy cần so
            for _ in range(5):  # Chạy khởi động vài lần (lần đầu thường chậm)
                run()  # Chạy thử
            start = time.perf_counter()  # Bắt đầu bấm giờ
            for _ in range(runs):  # Chạy nhiều lần lấy trung bình
                run()  # Dự đoán 1 ảnh
            timings[name] = (time.perf_counter() - start) * 1000 / runs  # ms mỗi ảnh
    print("Speed, model only, 1 image: PyTorch {:.1f} ms | ONNX {:.1f} ms ({:.1f}x faster)".format(
        timings["PyTorch"], timings["ONNX"], timings["PyTorch"] / timings["ONNX"]))  # In kết quả


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file (bắt buộc trên Windows khi NUM_WORKERS > 0)
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX and verify it against PyTorch")  # Tạo bộ phân tích tham số
    parser.add_argument("--checkpoint", default=os.path.join(config.OUTPUT_DIR, config.MODEL_NAME, "best.pt"))  # File mô hình cần xuất
    parser.add_argument("--output-dir", default=EXPORT_DIR, help="where to write model.onnx and model.json")  # Thư mục đầu ra
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="images used for verification")  # Tập ảnh để kiểm chứng
    args = parser.parse_args()  # Đọc tham số

    if not os.path.isfile(args.checkpoint):  # Chưa có mô hình
        print("Checkpoint not found: {}. Run train.py first.".format(args.checkpoint))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, torch.device("cpu"))  # Nạp mô hình lên CPU, đã ở chế độ eval
    os.makedirs(args.output_dir, exist_ok=True)  # Tạo thư mục đầu ra nếu chưa có

    export_model(model, checkpoint, args.checkpoint, args.output_dir)  # Xuất ONNX + model.json
    onnx_size_mb = os.path.getsize(os.path.join(args.output_dir, "model.onnx")) / 1024 / 1024  # Dung lượng file ONNX (MB)
    print("Saved model.onnx ({:.1f} MB) and model.json to {} ({} epoch {})".format(
        onnx_size_mb, args.output_dir, checkpoint["model_name"], checkpoint["epoch"]))  # Báo đã lưu

    classifier = OnnxClassifier(args.output_dir)  # Nạp lại file vừa xuất bằng onnxruntime
    passed = verify(model, classifier, args.split)  # Kiểm chứng
    benchmark(model, classifier)  # Đo tốc độ
    if not passed:  # Không đạt
        print("Verification FAILED: do not use this model.onnx.")  # Cảnh báo
        raise SystemExit(1)  # Dừng với mã lỗi
    print("Verification passed: model.onnx gives the same results as PyTorch.")  # Đạt
