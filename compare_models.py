# So sánh mọi mô hình đã train (outputs/<tên mô hình>/best.pt) trên CÙNG một tập ảnh: độ chính xác, kích thước, tốc độ.
# Kết quả in ra bảng và lưu vào outputs/compare_<tập>.csv.

import argparse  # Thư viện đọc tham số dòng lệnh
import csv  # Thư viện ghi file CSV
import glob  # Thư viện tìm file theo mẫu, vd outputs/*/best.pt
import os  # Thư viện làm việc với file/thư mục
import time  # Thư viện đo thời gian
import torch  # Thư viện PyTorch chính
import config  # Các thiết lập của dự án (config.py)
from dataset import make_dataloader  # Tạo DataLoader cho một tập
from evaluate import run_inference, decide_predictions, compute_metrics  # Các hàm đánh giá dùng chung
from model import load_model_from_checkpoint  # Nạp mô hình đã train

NUMPY_CNN_TEST_ACCURACY = 0.942  # Kết quả bản CNN viết tay bằng NumPy (dự án neu_det, Bước 11) trên đúng 360 ảnh này, để so sánh
SPEED_RUNS = 30  # Số lần chạy 1 ảnh để lấy thời gian trung bình


def measure_speed(model, image_size):  # Thời gian (ms) để mô hình xử lý 1 ảnh trên CPU, chỉ tính phần mô hình
    image = torch.rand(1, 3, image_size, image_size)  # 1 ảnh giả đã xử lý sẵn (nội dung ảnh không ảnh hưởng tốc độ)
    with torch.no_grad():  # Không tính gradient
        for _ in range(5):  # Chạy khởi động vài lần (lần đầu thường chậm)
            model(image)  # Chạy thử
        start = time.perf_counter()  # Bắt đầu bấm giờ
        for _ in range(SPEED_RUNS):  # Chạy nhiều lần lấy trung bình
            model(image)  # Dự đoán 1 ảnh
    return (time.perf_counter() - start) * 1000 / SPEED_RUNS  # ms mỗi ảnh


if __name__ == '__main__':  # Bắt buộc trên Windows khi NUM_WORKERS > 0
    parser = argparse.ArgumentParser(description="Compare all trained models on the same split")  # Tạo bộ phân tích tham số
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")  # So sánh trên tập nào
    args = parser.parse_args()  # Đọc tham số

    checkpoint_paths = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*", "best.pt")))  # Mọi mô hình đã train, mỗi mô hình một thư mục
    if not checkpoint_paths:  # Chưa có mô hình nào
        print("No checkpoints found in {}/*/best.pt. Run train.py first.".format(config.OUTPUT_DIR))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình
    device = torch.device("cpu")  # Đo trên CPU vì máy chuyền thường không có GPU

    rows = []  # Kết quả của từng mô hình
    for checkpoint_path in checkpoint_paths:  # Với từng mô hình
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)  # Nạp mô hình
        class_names = checkpoint["class_names"]  # Danh sách lớp lúc train
        loader = make_dataloader(args.split, class_names, train=False, image_size=checkpoint["image_size"])  # Ảnh của tập cần so sánh
        _, labels, probabilities = run_inference(model, loader, device)  # Dự đoán toàn bộ tập
        predictions = decide_predictions(probabilities)  # Ra kết luận
        metrics = compute_metrics(labels, predictions, class_names)  # Accuracy, macro F1
        rows.append({  # Lưu một dòng kết quả
            "model": checkpoint["model_name"],  # Tên mô hình
            "best_epoch": checkpoint["epoch"],  # Epoch được chọn (val F1 cao nhất)
            "accuracy": round(metrics["accuracy"], 4),  # Tỉ lệ đúng
            "macro_f1": round(metrics["macro_f1"], 4),  # F1 trung bình các lớp
            "wrong": sum(p != l for p, l in zip(predictions, labels)),  # Số ảnh sai
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),  # Số tham số (triệu)
            "file_mb": round(os.path.getsize(checkpoint_path) / 1024 / 1024, 1),  # Dung lượng file best.pt (MB)
            "ms_per_image": round(measure_speed(model, checkpoint["image_size"]), 1),  # Tốc độ trên CPU, 1 ảnh mỗi lần
        })
        print("Evaluated {}".format(checkpoint_path), flush=True)  # Báo tiến độ

    rows.sort(key=lambda row: (-row["macro_f1"], row["ms_per_image"]))  # Sắp: F1 cao trước; bằng nhau thì mô hình nhanh hơn trước
    print("\nSplit: {} ({} images)".format(args.split, len(labels)))  # Tiêu đề bảng
    print("{:<19} {:>5} {:>9} {:>9} {:>6} {:>9} {:>8} {:>9}".format(
        "model", "epoch", "accuracy", "macro F1", "wrong", "params M", "file MB", "ms/image"))  # Dòng tiêu đề
    for row in rows:  # Mỗi mô hình một dòng
        print("{model:<19} {best_epoch:>5} {accuracy:>9.4f} {macro_f1:>9.4f} {wrong:>6} {params_m:>9.2f} {file_mb:>8.1f} {ms_per_image:>9.1f}".format(**row))  # In dòng kết quả
    if args.split == "test":  # Kết quả NumPy chỉ có trên tập test
        print("{:<19} {:>5} {:>9.4f}   (handwritten NumPy CNN from the neu_det project)".format("numpy_cnn", "-", NUMPY_CNN_TEST_ACCURACY))  # Dòng so sánh

    csv_path = os.path.join(config.OUTPUT_DIR, "compare_{}.csv".format(args.split))  # File kết quả
    with open(csv_path, "w", newline="", encoding="utf-8") as f:  # Mở file để ghi
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))  # Bộ ghi CSV theo tên cột
        writer.writeheader()  # Ghi dòng tiêu đề
        writer.writerows(rows)  # Ghi mỗi mô hình một dòng
    print("\nSaved {}".format(csv_path))  # Báo đã lưu
