import argparse  # Thư viện đọc tham số dòng lệnh
import csv  # Thư viện ghi file CSV (mở được bằng Excel)
import math  # Thư viện toán học, dùng hàm ceil (làm tròn lên) để tính số hàng của lưới ảnh
import os  # Thư viện làm việc với file/thư mục
import torch  # Thư viện PyTorch chính
import matplotlib  # Thư viện vẽ hình
import matplotlib.pyplot as plt  # Hàm vẽ hình
from PIL import Image  # Thư viện mở file ảnh
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score  # Các chỉ số đánh giá
import config  # Các thiết lập của dự án (config.py)
from dataset import make_dataloader  # Tạo DataLoader cho một tập
from model import load_model_from_checkpoint  # Nạp mô hình đã train

MAX_WRONG_IMAGES = 20  # Số ảnh sai tối đa vẽ ra lưới misclassified_<tập>.png


def run_inference(model, dataloader, device, criterion=None):  # Cho mô hình dự đoán toàn bộ một tập; trả về (loss trung bình, nhãn thật, xác suất)
    model.eval()  # Chế độ đánh giá: tắt Dropout, BatchNorm dùng thống kê đã học
    total_loss = 0.  # Cộng dồn loss (chỉ tính khi có criterion)
    all_labels = []  # Nhãn thật của mọi ảnh
    all_probabilities = []  # Xác suất từng lớp của mọi ảnh, vd [[0.9, 0.02, ...], ...] (6 số mỗi ảnh)
    with torch.no_grad():  # Không tính gradient -> nhanh và tốn ít bộ nhớ
        for images, labels in dataloader:  # Duyệt từng batch
            outputs = model(images.to(device))  # Logits (batch, số lớp)
            if criterion is not None:  # Nếu cần tính loss (lúc train, để theo dõi val loss)
                total_loss += criterion(outputs, labels.to(device)).item() * images.size(0)  # Loss trung bình của batch x số ảnh = tổng loss của batch
            all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().tolist())  # softmax đổi logits thành xác suất có tổng = 1
            all_labels.extend(labels.tolist())  # Thêm nhãn thật
    average_loss = total_loss / len(dataloader.dataset) if criterion is not None else None  # Loss trung bình trên mỗi ảnh
    return average_loss, all_labels, all_probabilities  # Trả về kết quả


def decide_predictions(probabilities):  # Từ xác suất ra kết luận lớp cho từng ảnh: chọn lớp có xác suất cao nhất (argmax)
    return [probs.index(max(probs)) for probs in probabilities]  # Bài nhiều lớp không có "ngưỡng NG" như bài OK/NG nên chỉ cần argmax


def compute_metrics(labels, predictions, class_names):  # Tính các chỉ số chính: accuracy và macro F1
    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)  # Tỉ lệ đoán đúng trên tổng số ảnh
    macro_f1 = f1_score(labels, predictions, labels=list(range(len(class_names))), average="macro", zero_division=0)  # F1 trung bình 6 lớp, lớp nào cũng quan trọng như nhau
    return {"accuracy": accuracy, "macro_f1": macro_f1}  # Trả về dạng dict


def top_confusions(matrix, class_names, limit=5):  # Các cặp (lớp thật -> lớp đoán) bị nhầm nhiều nhất, bỏ qua đường chéo (đoán đúng)
    pairs = [(int(matrix[i][j]), class_names[i], class_names[j])
             for i in range(len(class_names)) for j in range(len(class_names)) if i != j and matrix[i][j] > 0]  # Mọi ô ngoài đường chéo có ảnh
    return sorted(pairs, reverse=True)[:limit]  # Sắp giảm dần theo số ảnh, lấy tối đa limit cặp


def save_wrong_grid(wrong, class_names, figure_path):  # Vẽ các ảnh bị đoán sai thành lưới, mỗi ô ghi lớp đoán / lớp thật
    cols = min(5, len(wrong))  # Tối đa 5 cột
    rows = math.ceil(len(wrong) / cols)  # Số hàng cần thiết
    fig, axes = plt.subplots(rows, cols, figsize=(max(3 * cols, 7), 3.4 * rows + 0.4), squeeze=False)  # Tạo lưới ô hình; squeeze=False để axes luôn là bảng 2 chiều
    for ax in axes.flat:  # Với mọi ô
        ax.axis("off")  # Ẩn trục tọa độ (ô thừa ở cuối sẽ để trống)
    for ax, (path, label, prediction, probs) in zip(axes.flat, wrong):  # Mỗi ảnh sai một ô
        ax.imshow(Image.open(path).convert("RGB"))  # Hiện ảnh gốc
        ax.set_title("pred {} {:.0%}\ntrue {}\n{}".format(class_names[prediction], probs[prediction], class_names[label],
                                                       os.path.basename(path)), color="red", fontsize=8)  # Lớp đoán + độ tự tin, lớp thật, tên file
    fig.suptitle("Misclassified images (first {})".format(len(wrong)))  # Tiêu đề
    fig.tight_layout(rect=(0, 0, 1, 0.96))  # Tự căn chỉnh khoảng cách cho khỏi chồng chữ, chừa phía trên cho tiêu đề
    fig.savefig(figure_path, dpi=100)  # Lưu hình
    plt.close(fig)  # Đóng hình để giải phóng bộ nhớ


if __name__ == '__main__':  # Chạy trực tiếp file để đánh giá mô hình đã train
    matplotlib.use("Agg")  # Chỉ lưu hình ra file, không mở cửa sổ (đặt ở đây để predict.py import file này vẫn mở được cửa sổ --show)
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test, val or train split")  # Tạo bộ phân tích tham số
    parser.add_argument("--checkpoint", default=os.path.join(config.OUTPUT_DIR, config.MODEL_NAME, "best.pt"))  # File mô hình cần đánh giá
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")  # Đánh giá trên tập nào (mặc định test)
    args = parser.parse_args()  # Đọc tham số

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dùng GPU nếu có
    if not os.path.isfile(args.checkpoint):  # Chưa có mô hình
        print("Checkpoint not found: {}. Run train.py first.".format(args.checkpoint))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)  # Nạp mô hình
    class_names = checkpoint["class_names"]  # Danh sách lớp lúc train (thứ tự phải giữ nguyên)
    dataloader = make_dataloader(args.split, class_names, train=False, image_size=checkpoint["image_size"])  # DataLoader của tập cần đánh giá, cùng kích thước ảnh lúc train
    _, labels, probabilities = run_inference(model, dataloader, device)  # Dự đoán toàn bộ tập
    predictions = decide_predictions(probabilities)  # Ra kết luận từng ảnh
    metrics = compute_metrics(labels, predictions, class_names)  # Accuracy và macro F1
    output_dir = os.path.dirname(os.path.abspath(args.checkpoint))  # Lưu báo cáo cạnh checkpoint, vd outputs/resnet18/

    print("Model: {} (epoch {}) | split: {} ({} images) | accuracy {:.4f} | macro F1 {:.4f}".format(
        checkpoint["model_name"], checkpoint["epoch"], args.split, len(labels), metrics["accuracy"], metrics["macro_f1"]))  # In thông tin đánh giá
    print(classification_report(labels, predictions, labels=list(range(len(class_names))), target_names=class_names,
                                digits=4, zero_division=0))  # Bảng precision/recall/F1 từng lớp

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))  # Ma trận nhầm lẫn: hàng = lớp thật, cột = lớp đoán
    short_names = [name[:8] for name in class_names]  # Tên lớp cắt ngắn để bảng in vừa màn hình, vd "rolled-i"
    print("Confusion matrix (rows = true, columns = predicted):")  # Tiêu đề
    print(" " * 17 + "".join("{:>9}".format(name) for name in short_names))  # Dòng tên cột
    for name, row in zip(class_names, matrix):  # Mỗi hàng là một lớp thật
        print("{:<17}".format(name) + "".join("{:>9}".format(value) for value in row))  # In số ảnh từng ô
    confusions = top_confusions(matrix, class_names)  # Các cặp hay bị nhầm nhất
    if confusions:  # Có ít nhất một ảnh sai
        print("\nMost confused (true -> predicted):")  # Tiêu đề
        for count, true_name, predicted_name in confusions:  # Mỗi cặp một dòng
            print("  {:>3} x  {} -> {}".format(count, true_name, predicted_name))  # vd "  3 x  crazing -> rolled-in_scale"

    figure_path = os.path.join(output_dir, "confusion_matrix_{}.png".format(args.split))  # Đường dẫn file hình
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(cmap="Blues", values_format="d", xticks_rotation=45)  # Vẽ ma trận nhầm lẫn; xoay tên cột 45 độ cho khỏi chồng chữ
    plt.title("{} - {} split".format(checkpoint["model_name"], args.split))  # Tiêu đề hình
    plt.savefig(figure_path, dpi=120, bbox_inches="tight")  # Lưu hình
    plt.close()  # Đóng hình để giải phóng bộ nhớ

    wrong = [(path, label, prediction, probs) for (path, label), prediction, probs
             in zip(dataloader.dataset.samples, predictions, probabilities) if prediction != label]  # Lọc các ảnh đoán sai (thứ tự samples khớp vì DataLoader không xáo trộn)
    wrong.sort(key=lambda item: item[3][item[2]], reverse=True)  # Sắp theo độ tự tin giảm dần: sai mà rất tự tin là đáng xem nhất (có thể do nhãn sai)
    csv_path = os.path.join(output_dir, "misclassified_{}.csv".format(args.split))  # File liệt kê ảnh bị đoán sai
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig để Excel đọc đúng
        writer = csv.writer(f)  # Bộ ghi CSV
        writer.writerow(["image", "true", "predicted"] + ["prob_" + name for name in class_names])  # Dòng tiêu đề
        for path, label, prediction, probs in wrong:  # Mỗi ảnh đoán sai một dòng
            writer.writerow([path, class_names[label], class_names[prediction]] + ["{:.4f}".format(p) for p in probs])  # Ghi đường dẫn, nhãn thật, nhãn đoán, xác suất
    print("\n{} misclassified images -> {}".format(len(wrong), csv_path))  # Báo số ảnh sai
    if wrong:  # Có ảnh sai thì vẽ lưới để xem bằng mắt
        grid_path = os.path.join(output_dir, "misclassified_{}.png".format(args.split))  # Đường dẫn file hình
        save_wrong_grid(wrong[:MAX_WRONG_IMAGES], class_names, grid_path)  # Vẽ tối đa MAX_WRONG_IMAGES ảnh
        print("Misclassified images figure -> {}".format(grid_path))  # Báo nơi lưu hình
    print("Confusion matrix figure -> {}".format(figure_path))  # Báo nơi lưu hình
