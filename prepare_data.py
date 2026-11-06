# Chép ảnh từ bộ NEU-DET gốc vào data/split/{train,val,test}/<lớp>/.
# test = toàn bộ tập validation chính thức của NEU (360 ảnh): đúng tập mà bản CNN NumPy (neu_det) đã dùng, nên so sánh được trực tiếp.
# train/val = tập train chính thức của NEU, chia mỗi lớp 85% / 15%.

import argparse  # Thư viện đọc tham số dòng lệnh
import hashlib  # Thư viện tính mã băm (MD5) để phát hiện ảnh trùng nhau từng byte
import os  # Thư viện làm việc với file/thư mục
import random  # Thư viện ngẫu nhiên, dùng để xáo trộn ảnh trước khi chia
import shutil  # Thư viện sao chép/xóa file và thư mục
import config  # Các thiết lập của dự án (config.py)


def list_images(folder):  # Danh sách đường dẫn ảnh trong một thư mục, sắp theo tên
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder)) if name.lower().endswith(config.IMAGE_EXTENSIONS)]  # Chỉ lấy file ảnh


def file_hash(path):  # Mã MD5 của file: 2 file có cùng mã = giống hệt nhau
    with open(path, "rb") as file:  # Mở file dạng nhị phân
        return hashlib.md5(file.read()).hexdigest()  # Đọc toàn bộ (ảnh NEU chỉ ~15 KB) rồi tính mã


def copy_files(paths, destination):  # Chép danh sách file vào một thư mục
    os.makedirs(destination, exist_ok=True)  # Tạo thư mục đích nếu chưa có
    for path in paths:  # Với từng file
        shutil.copy2(path, destination)  # Sao chép (giữ nguyên file gốc ở NEU)


if __name__ == '__main__':  # Chỉ chạy khi chạy trực tiếp file
    parser = argparse.ArgumentParser(description="Copy NEU-DET images into data/split/{train,val,test}")  # Tạo bộ phân tích tham số
    parser.add_argument("--overwrite", action="store_true", help="delete data/split and create it again")  # Cho phép chia lại
    args = parser.parse_args()  # Đọc tham số

    train_source = os.path.join(config.NEU_DIR, "train", "images")  # Ảnh train gốc của NEU, mỗi lớp một thư mục
    test_source = os.path.join(config.NEU_DIR, "validation", "images")  # Ảnh validation gốc của NEU -> dùng làm test
    for folder in (train_source, test_source):  # Kiểm tra cả 2 thư mục nguồn
        if not os.path.isdir(folder):  # Không tìm thấy
            print("Folder not found: {}. Check NEU_DIR in config.py.".format(folder))  # Hướng dẫn
            raise SystemExit(1)  # Dừng chương trình
    class_names = sorted(name for name in os.listdir(train_source) if os.path.isdir(os.path.join(train_source, name)))  # Tên lớp = tên thư mục, sắp theo alphabet
    test_classes = sorted(name for name in os.listdir(test_source) if os.path.isdir(os.path.join(test_source, name)))  # Tên lớp trong tập test
    if class_names != test_classes:  # Hai tập phải có cùng danh sách lớp
        print("Train classes {} differ from validation classes {}.".format(class_names, test_classes))  # Báo lỗi
        raise SystemExit(1)  # Dừng chương trình

    if os.path.exists(config.SPLIT_DIR):  # Đã chia trước đó
        if not args.overwrite:  # Không cho ghi đè
            print("{} already exists. Use --overwrite to create it again.".format(config.SPLIT_DIR))  # Hướng dẫn
            raise SystemExit(1)  # Dừng chương trình
        shutil.rmtree(config.SPLIT_DIR)  # Xóa thư mục cũ (chỉ là bản sao, ảnh gốc ở NEU không bị ảnh hưởng)

    seen = {}  # Mã MD5 -> ảnh đầu tiên có mã đó, để bỏ ảnh trùng
    rng = random.Random(config.SEED)  # Bộ sinh số ngẫu nhiên riêng với hạt giống cố định -> chia lần nào cũng giống nhau
    print("{:<16} {:>6} {:>6} {:>6}".format("class", "train", "val", "test"))  # Tiêu đề bảng thống kê
    for class_name in class_names:  # Với từng lớp
        test_images = list_images(os.path.join(test_source, class_name))  # Ảnh test của lớp
        for path in test_images:  # Ghi nhận mã của ảnh test trước
            seen.setdefault(file_hash(path), path)  # Ảnh train nào trùng ảnh test sẽ bị bỏ ở bước dưới (tránh rò rỉ test vào train)

        train_images = []  # Ảnh train của lớp, sau khi bỏ ảnh trùng
        for path in list_images(os.path.join(train_source, class_name)):  # Duyệt ảnh train gốc
            digest = file_hash(path)  # Mã của ảnh
            if digest in seen:  # Trùng một ảnh đã có (cùng lớp hoặc khác lớp, ở train hoặc test)
                print("  duplicate skipped: {} (same as {})".format(os.path.basename(path), os.path.basename(seen[digest])))  # Báo ảnh bị bỏ
                continue  # Không chép ảnh này
            seen[digest] = path  # Ghi nhận ảnh mới
            train_images.append(path)  # Giữ lại

        rng.shuffle(train_images)  # Xáo trộn để val không chỉ gồm các ảnh có số thứ tự lớn nhất
        num_val = round(len(train_images) * config.VAL_RATIO)  # Số ảnh val của lớp, vd 239 * 0.15 = 36
        copy_files(train_images[num_val:], os.path.join(config.SPLIT_DIR, "train", class_name))  # Phần còn lại -> train
        copy_files(train_images[:num_val], os.path.join(config.SPLIT_DIR, "val", class_name))  # num_val ảnh đầu -> val
        copy_files(test_images, os.path.join(config.SPLIT_DIR, "test", class_name))  # Toàn bộ validation của NEU -> test
        print("{:<16} {:>6} {:>6} {:>6}".format(class_name, len(train_images) - num_val, num_val, len(test_images)))  # In số ảnh mỗi tập
    print("Done -> {}. Next step: python train.py".format(config.SPLIT_DIR))  # Gợi ý bước tiếp theo
