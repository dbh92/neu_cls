# Grad-CAM: tô màu vùng ảnh mà mô hình "nhìn vào" khi ra kết luận (đỏ = ảnh hưởng nhiều, xanh = ít).
# Dùng để kiểm tra mô hình có nhìn đúng vết lỗi hay chỉ dựa vào nền, viền ảnh, độ sáng... (học "mẹo" thay vì học lỗi).
# Cách tính: lấy bản đồ đặc trưng ở lớp tích chập cuối, nhân với gradient của điểm số lớp được chọn, cộng lại theo kênh.

import argparse  # Thư viện đọc tham số dòng lệnh
import csv  # Thư viện đọc file CSV (danh sách ảnh sai do evaluate.py tạo)
import math  # Thư viện toán học, dùng hàm ceil (làm tròn lên) để tính số trang, số hàng
import os  # Thư viện làm việc với file/thư mục
import torch  # Thư viện PyTorch chính
import torch.nn.functional as F  # Các hàm tiện ích: softmax, interpolate (phóng to bản đồ nhiệt)
import matplotlib  # Thư viện vẽ hình
import matplotlib.pyplot as plt  # Hàm vẽ hình
from PIL import Image  # Thư viện mở file ảnh
import config  # Các thiết lập của dự án (config.py)
from dataset import build_transforms  # Dùng lại đúng chuỗi xử lý ảnh của tập val/test
from model import gradcam_layer, load_model_from_checkpoint  # Lớp tích chập cuối của từng mô hình, hàm nạp mô hình
from predict import collect_image_paths, true_label_from_path  # Gom danh sách ảnh, đoán nhãn thật từ tên thư mục

IMAGES_PER_ROW = 3  # Mỗi ảnh chiếm 2 ô (ảnh gốc + bản đồ nhiệt) -> 6 cột mỗi hàng
IMAGES_PER_PAGE = 18  # 6 hàng x 3 ảnh mỗi trang (vừa 6 lớp x 3 ảnh)


def compute_gradcam(model, layer, input_tensor):  # Trả về (bản đồ nhiệt 0..1 cùng kích thước ảnh, chỉ số lớp đoán, xác suất các lớp)
    saved = {}  # Nơi cất bản đồ đặc trưng và gradient bắt được từ hook
    def forward_hook(module, inputs, output):  # Hàm được PyTorch gọi ngay sau khi lớp `layer` tính xong
        saved["activation"] = output  # Cất bản đồ đặc trưng (1, C, h, w), vd (1, 512, 7, 7) với resnet18
        output.register_hook(lambda grad: saved.update(gradient=grad))  # Khi backward chạy qua đây, cất gradient (cùng shape)
    handle = layer.register_forward_hook(forward_hook)  # Gắn hook vào lớp
    try:  # Dù có lỗi cũng phải gỡ hook ở finally
        with torch.enable_grad():  # Cần gradient (bình thường lúc dự đoán thì tắt)
            logits = model(input_tensor)  # forward: hook cất bản đồ đặc trưng
            probs = F.softmax(logits, dim=1)[0]  # Xác suất các lớp của ảnh
            class_index = int(probs.argmax())  # Giải thích cho lớp được đoán
            model.zero_grad()  # Xóa gradient cũ (nếu có)
            logits[0, class_index].backward()  # backward từ điểm số của lớp đó: hook cất gradient
    finally:  # Luôn chạy
        handle.remove()  # Gỡ hook để lần sau không bị gắn chồng

    activation = saved["activation"][0].detach()  # (C, h, w): mỗi kênh là một "bộ dò" đặc trưng
    weights = saved["gradient"][0].mean(dim=(1, 2))  # (C,): độ quan trọng của từng kênh với lớp được chọn = trung bình gradient
    cam = F.relu((weights[:, None, None] * activation).sum(dim=0))  # (h, w): cộng các kênh theo trọng số; ReLU giữ phần làm TĂNG điểm của lớp
    cam = cam / cam.max() if cam.max() > 0 else cam  # Đưa về 0..1 để tô màu
    cam = F.interpolate(cam[None, None], size=input_tensor.shape[2:], mode="bilinear", align_corners=False)[0, 0]  # Phóng 7x7 lên 224x224 cho khớp ảnh
    return cam.numpy(), class_index, probs.tolist()  # Trả về kết quả


def read_misclassified(csv_path):  # Đọc danh sách ảnh bị đoán sai từ file evaluate.py đã ghi
    with open(csv_path, encoding="utf-8-sig") as f:  # utf-8-sig khớp với lúc evaluate.py ghi
        return [row["image"] for row in csv.DictReader(f)]  # Lấy cột đường dẫn ảnh


def default_images(class_names, per_class):  # Mặc định: per_class ảnh đầu tiên của mỗi lớp trong tập test
    paths = []  # Danh sách kết quả
    for class_name in class_names:  # Với từng lớp
        paths.extend(collect_image_paths([os.path.join(config.SPLIT_DIR, "test", class_name)])[:per_class])  # Lấy per_class ảnh đầu
    return paths  # Trả về danh sách


def draw_pages(results, image_size, show, save_dir):  # Vẽ từng cặp (ảnh gốc, bản đồ nhiệt) thành lưới
    num_pages = math.ceil(len(results) / IMAGES_PER_PAGE)  # Số trang cần vẽ
    for page in range(num_pages):  # Vẽ từng trang
        chunk = results[page * IMAGES_PER_PAGE:(page + 1) * IMAGES_PER_PAGE]  # Các ảnh thuộc trang này
        cols = 2 * min(IMAGES_PER_ROW, len(chunk))  # Số cột (mỗi ảnh 2 ô)
        rows = math.ceil(len(chunk) / IMAGES_PER_ROW)  # Số hàng
        fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.7 * rows + 0.4), squeeze=False)  # Tạo lưới ô hình
        for ax in axes.flat:  # Với mọi ô
            ax.axis("off")  # Ẩn trục tọa độ
        for index, result in enumerate(chunk):  # Mỗi ảnh một cặp ô
            ax_image, ax_cam = axes[index // IMAGES_PER_ROW][2 * (index % IMAGES_PER_ROW):2 * (index % IMAGES_PER_ROW) + 2]  # Ô ảnh gốc và ô bản đồ nhiệt
            image = result["image"].resize((image_size, image_size))  # Ảnh gốc cùng kích thước với đầu vào mô hình
            ax_image.imshow(image)  # Ô trái: ảnh gốc
            ax_image.set_title("true {}\n{}".format(result["true"] or "?", result["name"][-24:]), fontsize=8)  # Nhãn thật + tên file
            ax_cam.imshow(image)  # Ô phải: ảnh gốc làm nền
            ax_cam.imshow(result["cam"], cmap="jet", alpha=0.45, vmin=0, vmax=1)  # Phủ bản đồ nhiệt bán trong suốt lên trên
            color = "black" if result["true"] is None else ("green" if result["true"] == result["predicted"] else "red")  # Xanh = đúng, đỏ = sai
            ax_cam.set_title("pred {}\n{:.1%}".format(result["predicted"], result["confidence"]), color=color, fontsize=8)  # Lớp đoán + độ tự tin
        fig.suptitle("Grad-CAM page {}/{} - red areas drive the prediction most".format(page + 1, num_pages))  # Tiêu đề trang
        fig.tight_layout(rect=(0, 0, 1, 0.97))  # Tự căn chỉnh khoảng cách, chừa 3% phía trên cho tiêu đề trang
        
        # Step 9. Xuất kết quả: gradcam_01.png, gradcam_02.png...
        if save_dir:  # Lưu ra file
            os.makedirs(save_dir, exist_ok=True)  # Tạo thư mục nếu chưa có
            figure_path = os.path.join(save_dir, "gradcam_{:02d}.png".format(page + 1))  # vd gradcam_01.png
            fig.savefig(figure_path, dpi=100)  # Lưu hình
            print("Saved {}".format(figure_path))  # Báo đã lưu
        if show:  # Hiện cửa sổ
            plt.show()  # Đóng cửa sổ thì sang trang tiếp theo
        plt.close(fig)  # Đóng hình để giải phóng bộ nhớ


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file
    
    # step 1. Đọc tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Grad-CAM heatmaps: where does the model look?")  # Tạo bộ phân tích tham số       
    parser.add_argument("paths", nargs="*", help="image files or folders (default: a few test images per class)")  # Ảnh cần giải thích (không bắt buộc)
    parser.add_argument("--checkpoint", default=os.path.join(config.OUTPUT_DIR, config.MODEL_NAME, "best.pt"))  # File mô hình
    parser.add_argument("--misclassified", action="store_true", help="use the wrong images listed by evaluate.py (misclassified_test.csv)")  # Giải thích các ảnh bị đoán sai
    parser.add_argument("--per-class", type=int, default=3, help="images per class when no paths are given")  # Số ảnh mỗi lớp khi không đưa ảnh
    parser.add_argument("--save", metavar="FOLDER", help="where to save PNG pages (default: <checkpoint folder>/gradcam)")  # Nơi lưu hình
    parser.add_argument("--show", action="store_true", help="also show the pages in a window")  # Hiện cửa sổ
    args = parser.parse_args()  # Đọc tham số
    if not args.show:  # Không cần cửa sổ
        matplotlib.use("Agg")  # Chỉ lưu ra file (chạy được cả khi không có màn hình)

    # step 2. Load checkpoint: Model + class_names + image_size
    if not os.path.isfile(args.checkpoint):  # Chưa có mô hình
        print("Checkpoint not found: {}. Run train.py first.".format(args.checkpoint))  # Hướng dẫn
        raise SystemExit(1)  # Dừng chương trình
    device = torch.device("cpu")  # Grad-CAM chỉ chạy vài chục ảnh, CPU là đủ
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)  # Nạp mô hình (đã ở chế độ eval)
    class_names = checkpoint["class_names"]  # Danh sách lớp lúc train

    # step 3. Xác định Grad-CAM layer: Lớp convolution cuối
    layer = gradcam_layer(model, checkpoint["model_name"])  # Lớp tích chập cuối của mô hình này
    transform = build_transforms(train=False, image_size=checkpoint["image_size"])  # Xử lý ảnh giống hệt lúc đánh giá
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))  # Thư mục của mô hình, vd outputs/resnet18

    # step 4. Chọn ảnh đầu vào: Ảnh sai / ảnh chỉ định / ảnh test mặc định
    if args.misclassified:  # Lấy danh sách ảnh sai
        csv_path = os.path.join(checkpoint_dir, "misclassified_test.csv")  # File do evaluate.py tạo
        if not os.path.isfile(csv_path):  # Chưa chạy evaluate.py
            print("{} not found. Run evaluate.py --checkpoint {} first.".format(csv_path, args.checkpoint))  # Hướng dẫn
            raise SystemExit(1)  # Dừng chương trình
        image_paths = read_misclassified(csv_path)  # Đường dẫn các ảnh sai
    elif args.paths:  # Người dùng đưa ảnh/thư mục
        image_paths = collect_image_paths(args.paths)  # Gom danh sách ảnh
    else:  # Không đưa gì
        image_paths = default_images(class_names, args.per_class)  # Lấy vài ảnh test mỗi lớp
    if not image_paths:  # Không có ảnh nào
        print("No images to explain.")  # Thông báo (vd không có ảnh sai nào)
        raise SystemExit(0)  # Kết thúc bình thường

    # Step 5. Tiền xử lý ảnh, xử lý từng ảnh: Resize → Tensor → Normalize → Batch
    results = []  # Kết quả từng ảnh
    for image_path in image_paths:  # Với từng ảnh
        image = Image.open(image_path).convert("RGB")  # Mở ảnh và ép về 3 kênh màu

        # Step 6. compute_gradcam(): Tính bản đồ nhiệt Grad-CAM, lớp đoán, xác suất các lớp
        cam, prediction, probs = compute_gradcam(model, layer, transform(image).unsqueeze(0).to(device))  # Tính bản đồ nhiệt
       
        # Step 7. Thu thập kết quả: Ảnh gốc + CAM + nhãn thật + nhãn dự đoán + confidence
        true_label = true_label_from_path(image_path, class_names)  # Nhãn thật từ tên thư mục cha (nếu có)
        results.append({"image": image, "cam": cam, "name": os.path.basename(image_path), "true": true_label,
                        "predicted": class_names[prediction], "confidence": probs[prediction]})  # Lưu để vẽ
        print("{:<15} {:6.1%} | true {:<15} | {}".format(class_names[prediction], probs[prediction], true_label or "?", image_path))  # In kết quả từng ảnh
    
    # Step 8. Vẽ và lưu từng trang lưới ảnh gốc + bản đồ nhiệt (heatmap)
    draw_pages(results, checkpoint["image_size"], args.show, args.save or os.path.join(checkpoint_dir, "gradcam"))  # Vẽ và lưu
