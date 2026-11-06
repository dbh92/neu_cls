import argparse  # Thư viện đọc tham số dòng lệnh
import csv  # Thư viện ghi file CSV
import json  # Thư viện ghi file JSON (config.json đọc được bằng Notepad)
import os  # Thư viện làm việc với file/thư mục
import time  # Thư viện đo thời gian mỗi epoch
import torch  # Thư viện PyTorch chính
import torch.nn as nn  # Module chứa hàm mất mát
import matplotlib  # Thư viện vẽ hình
matplotlib.use("Agg")  # Chỉ lưu hình ra file, không mở cửa sổ
import matplotlib.pyplot as plt  # Hàm vẽ hình
from matplotlib.ticker import MaxNLocator  # Để trục epoch chỉ hiện số nguyên
import config  # Các thiết lập của dự án (config.py)
from dataset import get_class_names, make_dataloader  # Đọc tên lớp, tạo DataLoader
from model import MODEL_NAMES, build_model  # Danh sách mô hình hợp lệ, hàm tạo mô hình
from evaluate import run_inference, decide_predictions, compute_metrics  # Các hàm đánh giá dùng chung


def config_snapshot(model_name):  # Chụp lại mọi thiết lập trong config.py lúc train, để sau này biết mô hình đã được train thế nào
    snapshot = {key: value for key, value in vars(config).items() if key.isupper()}  # Chỉ lấy các thiết lập viết HOA (bỏ những thứ Python tự thêm như __name__)
    snapshot["MODEL_NAME"] = model_name  # Ghi tên mô hình thật sự được train (--model có thể khác MODEL_NAME trong config.py)
    return snapshot  # Trả về dict, vd {"EPOCHS": 20, "LEARNING_RATE": 0.0001, ...}


def save_history(history, output_dir):  # Lưu lịch sử train ra CSV và vẽ biểu đồ
    with open(os.path.join(output_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:  # Mở file CSV để ghi
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))  # Bộ ghi CSV theo tên cột
        writer.writeheader()  # Ghi dòng tiêu đề
        writer.writerows(history)  # Ghi mỗi epoch một dòng

    epochs = [row["epoch"] for row in history]  # Trục hoành: số epoch
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 4))  # 2 biểu đồ cạnh nhau
    ax_loss.plot(epochs, [row["train_loss"] for row in history], label="train loss")  # Loss trên tập train
    ax_loss.plot(epochs, [row["val_loss"] for row in history], label="val loss")  # Loss trên tập validation; tăng trong khi train loss giảm = overfit
    ax_loss.set_xlabel("epoch")  # Nhãn trục hoành
    ax_loss.legend()  # Hiện chú thích
    for key in ("val_accuracy", "val_macro_f1"):  # Các chỉ số trên validation
        ax_metric.plot(epochs, [row[key] for row in history], label=key)  # Vẽ đường của chỉ số
    ax_metric.set_xlabel("epoch")  # Nhãn trục hoành
    ax_metric.set_ylim(0, 1.05)  # Các chỉ số nằm trong [0, 1]
    ax_metric.legend()  # Hiện chú thích
    for ax in (ax_loss, ax_metric):  # Với cả 2 biểu đồ
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Trục epoch chỉ hiện 1, 2, 3, ... thay vì 1.25, 1.5
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=120, bbox_inches="tight")  # Lưu hình
    plt.close(fig)  # Đóng hình


if __name__ == '__main__':  # Bắt buộc trên Windows khi NUM_WORKERS > 0
    parser = argparse.ArgumentParser(description="Train a steel surface defect classifier (settings are in config.py)")  # Tạo bộ phân tích tham số
    parser.add_argument("--model", default=config.MODEL_NAME, choices=MODEL_NAMES, help="model to train (default: MODEL_NAME in config.py)")  # Chọn mô hình cho lần chạy này
    parser.add_argument("--output-dir", help="where to save checkpoints (default: outputs/<model>)")  # Thư mục lưu kết quả
    parser.add_argument("--resume", action="store_true", help="continue from <output-dir>/last.pt")  # Chạy tiếp từ checkpoint cuối nếu bị ngắt
    args = parser.parse_args()  # Đọc tham số
    output_dir = args.output_dir or os.path.join(config.OUTPUT_DIR, args.model)  # Mỗi mô hình một thư mục -> train nhiều mô hình không ghi đè nhau

    torch.manual_seed(config.SEED)  # Cố định ngẫu nhiên để kết quả lặp lại được
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Dùng GPU nếu có

    # ===== 1. Dữ liệu =====
    if not os.path.isdir(os.path.join(config.SPLIT_DIR, "train")):  # Chưa chuẩn bị dữ liệu
        print("{}/train not found. Run prepare_data.py first.".format(config.SPLIT_DIR))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình
    class_names = get_class_names()  # Tên lớp lấy từ data/split/train (6 lớp, theo alphabet)
    train_loader = make_dataloader("train", class_names, train=True)  # Tập train: có xáo trộn và augmentation
    val_loader = make_dataloader("val", class_names, train=False)  # Tập validation: không augmentation
    counts = [sum(1 for _, label in train_loader.dataset.samples if label == i) for i in range(len(class_names))]  # Số ảnh train của từng lớp
    print("Classes: {} | train images per class: {} | val images: {}".format(
        class_names, counts, len(val_loader.dataset)))  # In thống kê dữ liệu
    if min(counts) == 0 or len(val_loader.dataset) == 0:  # Có lớp không có ảnh train, hoặc tập val trống
        print("Every class needs train images and the val split must not be empty. Run prepare_data.py --overwrite.")  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình

    # ===== 2. Mô hình, hàm mất mát, bộ tối ưu =====
    model = build_model(args.model, len(class_names), config.PRETRAINED).to(device)  # Tạo mô hình và chuyển sang thiết bị
    class_weights = None  # Mặc định mọi lớp có trọng số như nhau
    if config.USE_CLASS_WEIGHTS:  # Cân bằng khi số ảnh các lớp chênh lệch
        class_weights = torch.tensor([sum(counts) / (len(counts) * c) for c in counts], dtype=torch.float32, device=device)  # Lớp càng ít ảnh trọng số càng lớn; các lớp bằng nhau thì đều = 1
        print("Class weights: {}".format(dict(zip(class_names, [round(w, 3) for w in class_weights.tolist()]))))  # In trọng số từng lớp
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # Hàm mất mát cho phân loại nhiều lớp
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)  # AdamW: bộ tối ưu tự điều chỉnh bước học, hợp với fine-tune
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)  # Giảm learning rate dần theo hình cosine, về gần 0 ở epoch cuối

    # ===== 3. Checkpoint =====
    os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục kết quả
    best_path = os.path.join(output_dir, "best.pt")  # Mô hình tốt nhất theo validation -> dùng cho evaluate.py, predict.py
    last_path = os.path.join(output_dir, "last.pt")  # Trạng thái epoch gần nhất -> dùng cho --resume
    start_epoch = 0  # Epoch bắt đầu
    best_score = -1.  # Macro F1 tốt nhất trên validation đến hiện tại
    best_val_loss = float("inf")  # Val loss của epoch best (tiêu chí phụ khi val F1 bằng nhau)
    history = []  # Lịch sử chỉ số từng epoch
    epochs_without_improvement = 0  # Số epoch liên tiếp không có best mới (dùng cho early stopping)
    train_config = config_snapshot(args.model)  # Thiết lập của lần train này, được lưu vào checkpoint và config.json
    if args.resume and os.path.isfile(last_path):  # Chạy tiếp nếu có cờ --resume và file tồn tại
        checkpoint = torch.load(last_path, map_location=device)  # Đọc checkpoint cuối
        if checkpoint["class_names"] != class_names or checkpoint["model_name"] != args.model:  # Danh sách lớp hoặc mô hình đã thay đổi
            print("Class names or model changed since last run, cannot resume. Train from scratch.")  # Báo lỗi
            raise SystemExit(1)  # Dừng chương trình
        model.load_state_dict(checkpoint["model"])  # Khôi phục trọng số
        optimizer.load_state_dict(checkpoint["optimizer"])  # Khôi phục bộ tối ưu
        scheduler.load_state_dict(checkpoint["scheduler"])  # Khôi phục lịch learning rate (giữ nguyên EPOCHS như lần đầu)
        start_epoch = checkpoint["epoch"]  # Epoch đã hoàn thành
        best_score = checkpoint["best_score"]  # Điểm tốt nhất
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))  # Val loss của best (last.pt cũ chưa có thì coi như chưa có)
        history = checkpoint["history"]  # Lịch sử
        epochs_without_improvement = checkpoint["epochs_without_improvement"]  # Bộ đếm early stopping
        print("Resumed from epoch {}".format(start_epoch))  # Thông báo
        saved_config = checkpoint.get("config")  # Thiết lập của lần train trước (last.pt cũ chưa có thì là None)
        if saved_config is not None:  # Có thiết lập cũ để so sánh
            changed = [key for key in sorted(set(saved_config) | set(train_config)) if saved_config.get(key) != train_config.get(key)]  # Các thiết lập đã bị sửa trong config.py
            for key in changed:  # Mỗi thiết lập bị sửa một dòng
                print("Warning: {} changed since last run: {} -> {}".format(key, saved_config.get(key), train_config.get(key)))  # Cảnh báo, vẫn chạy tiếp (vd đổi NUM_WORKERS là vô hại)
            if changed:  # Có ít nhất một thiết lập bị sửa
                print("Note: --resume keeps the optimizer and learning rate schedule from last.pt. To use new settings, train without --resume.")  # LEARNING_RATE mới không có tác dụng vì optimizer được nạp lại

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:  # Bản thiết lập dễ đọc, nằm cạnh best.pt
        json.dump(train_config, f, indent=2)  # Mỗi thiết lập một dòng

    # ===== 4. Vòng lặp huấn luyện =====
    print("Model: {} | device: {} | epochs: {} | output: {}".format(args.model, device, config.EPOCHS, output_dir))  # In cấu hình ** step 1: Bắt đầu Epoch
    for epoch in range(start_epoch, config.EPOCHS):  # Mỗi vòng là một epoch
        if config.EARLY_STOPPING_PATIENCE and epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:  # Kiểm tra ở đầu vòng để --resume sau khi đã tự dừng cũng dừng ngay
            print("Early stopping: no new best (val F1, then val loss) for {} epochs.".format(epochs_without_improvement))  # Báo lý do dừng
            break  # Thoát vòng lặp huấn luyện
        epoch_start = time.perf_counter()  # Bắt đầu bấm giờ epoch
        model.train()  # Chế độ train: bật Dropout, BatchNorm cập nhật thống kê
        total_loss = 0.  # Cộng dồn loss của epoch
        for images, labels in train_loader:  # Duyệt từng batch  ** step 2: Training
            images = images.to(device)  # Chuyển ảnh sang thiết bị
            labels = labels.to(device)  # Chuyển nhãn sang thiết bị
            outputs = model(images)  # forward: tính logits
            loss_value = criterion(outputs, labels)  # Tính loss
            optimizer.zero_grad()  # Xóa gradient cũ
            loss_value.backward()  # backward: tính gradient
            optimizer.step()  # Cập nhật trọng số
            total_loss += loss_value.item() * images.size(0)  # Cộng tổng loss của batch
        scheduler.step()  # Cập nhật learning rate sau mỗi epoch    ** step 3: Update Learning Rate

        val_loss, val_labels, val_probabilities = run_inference(model, val_loader, device, criterion)  # Đánh giá trên validation   ** step 4: Validation
        metrics = compute_metrics(val_labels, decide_predictions(val_probabilities), class_names)  # Accuracy, macro F1     ** step 5: Calculate Metrics
        history.append({  # Lưu chỉ số của epoch này
            "epoch": epoch + 1,  # Số thứ tự epoch
            "train_loss": total_loss / len(train_loader.dataset),  # Loss trung bình trên tập train
            "val_loss": val_loss,  # Loss trung bình trên tập validation
            "val_accuracy": metrics["accuracy"],  # Tỉ lệ đúng
            "val_macro_f1": metrics["macro_f1"],  # F1 trung bình các lớp
            "seconds": round(time.perf_counter() - epoch_start, 1),  # Thời gian của epoch (giây)
        })
        is_best = (metrics["macro_f1"] > best_score  # Tiêu chí chính: macro F1 cao hơn (công bằng giữa các lớp)
                   or (metrics["macro_f1"] == best_score and val_loss < best_val_loss))  # F1 bằng nhau (vd cùng 1.0) thì chọn epoch có val loss thấp hơn = mô hình tự tin hơn
        if is_best:  # Có best mới                          ** step 6: Check Best Model
            best_score = metrics["macro_f1"]  # Cập nhật điểm tốt nhất
            best_val_loss = val_loss  # Ghi nhớ val loss của best
        epochs_without_improvement = 0 if is_best else epochs_without_improvement + 1  # Tốt hơn thì đếm lại từ 0, không thì cộng thêm 1
        status_text = ""  # Ghi chú cuối dòng
        if is_best:  # Epoch tốt nhất mới
            status_text = " *"  # Đánh dấu *
        elif config.EARLY_STOPPING_PATIENCE:  # Đang bật early stopping
            status_text = " (no improvement {}/{})".format(epochs_without_improvement, config.EARLY_STOPPING_PATIENCE)  # Cho biết còn bao nhiêu epoch nữa thì tự dừng
        print("Epoch {}/{} | train loss {:.4f} | val loss {:.4f} | val acc {:.4f} | val F1 {:.4f} | {:.0f}s{}".format(
            epoch + 1, config.EPOCHS, history[-1]["train_loss"], val_loss, metrics["accuracy"], metrics["macro_f1"],
            history[-1]["seconds"], status_text), flush=True)  # In kết quả epoch; flush để thấy ngay khi chạy nền

        checkpoint = {  # Thông tin cần để dùng lại mô hình sau này
            "model_name": args.model,  # Tên kiến trúc
            "class_names": class_names,  # Danh sách lớp theo đúng thứ tự nhãn
            "image_size": config.IMAGE_SIZE,  # Kích thước ảnh lúc train
            "epoch": epoch + 1,  # Epoch đã hoàn thành
            "config": train_config,  # Mọi thiết lập lúc train (LR, số epoch, augmentation, ...)
            "model": model.state_dict(),  # Trọng số mô hình
        }
        if is_best:  # Nếu là kết quả tốt nhất                  * step 7: Save Checkpoint
            torch.save(checkpoint, best_path)  # Lưu best.pt
        torch.save(dict(checkpoint, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        best_score=best_score, best_val_loss=best_val_loss, history=history,
                        epochs_without_improvement=epochs_without_improvement), last_path)  # Luôn lưu last.pt kèm trạng thái để --resume
        save_history(history, output_dir)  # Cập nhật history.csv và training_curves.png sau mỗi epoch

    print("Best val macro F1: {:.4f} (val loss {:.4f}) -> {}".format(best_score, best_val_loss, best_path))  # Kết quả cuối
    print("Next step: python evaluate.py --checkpoint {}".format(best_path))  # Gợi ý bước tiếp theo
