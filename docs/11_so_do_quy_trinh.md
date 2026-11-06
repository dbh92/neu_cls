# 11. Sơ đồ quy trình huấn luyện (khuôn mẫu chung)

File này vẽ **toàn bộ** quy trình của dự án thành sơ đồ, từ lúc cài môi trường đến lúc đưa mô hình ra dùng, chi tiết đến từng bước nhỏ. Mỗi bước ghi số dòng code tương ứng để bạn mở ra đối chiếu.

Bố cục 8 giai đoạn (A → H) là **khuôn mẫu chuẩn** cho mọi dự án phân loại ảnh bằng PyTorch. Làm dự án mới thì đi lần lượt từ A đến H, thay phần "trong dự án này" bằng dữ liệu và mô hình của bạn. Xem [mục 11.4](#114-checklist-áp-dụng-cho-dự-án-mới).

Số dòng đúng với code ngày 06/11/2026.

## Cách đọc sơ đồ

| Ký hiệu | Nghĩa |
|---|---|
| `┌─ X ───` … `└────` | Một giai đoạn hoặc một khối bước |
| `[D3.2]` | Mã bước: giai đoạn D, khối 3, bước 2. Dùng để tra trong bảng giải thích |
| `◇ ... ?` | Điểm quyết định, rẽ nhánh `có` / `không` |
| `╔═ VÒNG LẶP` … `╚═` | Các bước bên trong được lặp lại nhiều lần |
| `⇒ file` | File được ghi ra ở bước này |
| `✖ DỪNG` | Chương trình in thông báo lỗi và thoát |
| `:129` | Dòng 129 của file đang nói tới, vd `train.py:129` |

---

## 11.1. Sơ đồ tổng thể: 8 giai đoạn

```
┌─ A. MÔI TRƯỜNG ─────────────────────────────── 1 lần cho cả dự án
│  .venv, requirements.txt, config.py
└──────────────┬─────────────────────────────────
               ▼
┌─ B. CHUẨN BỊ DỮ LIỆU ───────────────────────── prepare_data.py
│  chia train / val / test, bỏ ảnh trùng           ⇒ data/split/
└──────────────┬─────────────────────────────────
               ▼
┌─ C. KIỂM TRA TRƯỚC KHI TRAIN ───────────────── dataset.py, model.py
│  dữ liệu đọc đúng chưa? mô hình chạy được chưa?
└──────────────┬─────────────────────────────────
               ▼
      ╔═ LẶP CHO TỪNG MÔ HÌNH (resnet18, resnet50, efficientnet_b0, mobilenet_v3_large)
      ║
      ║   ┌─ D. HUẤN LUYỆN ─────────────────────── train.py --model X
      ║   │  ⇒ outputs/X/best.pt, last.pt, config.json, history.csv, training_curves.png
      ║   └──────────┬─────────────────────────────
      ║              ▼
      ║   ┌─ E. ĐÁNH GIÁ TRÊN TEST ─────────────── evaluate.py --checkpoint outputs/X/best.pt
      ║   │  ⇒ confusion_matrix_test.png, misclassified_test.csv/.png
      ║   └──────────┬─────────────────────────────
      ║              ▼
      ║   ┌─ F. PHÂN TÍCH LỖI ──────────────────── gradcam.py, predict.py
      ║   │  mô hình nhìn vào đâu? sai ở ảnh nào, vì sao?   ⇒ outputs/X/gradcam/
      ║   └──────────┬─────────────────────────────
      ║              ▼
      ║   ◇ Kết quả đạt yêu cầu?
      ║     ├─ không ──► chẩn đoán (mục 11.3) → sửa config.py hoặc dữ liệu
      ║     │            → quay lại B (nếu sửa dữ liệu) hoặc D (nếu sửa thiết lập)
      ║     └─ có ─────► sang mô hình tiếp theo
      ║
      ╚═ hết danh sách mô hình
               ▼
┌─ G. SO SÁNH CÁC MÔ HÌNH ────────────────────── compare_models.py
│  cùng tập test: accuracy, F1, số ảnh sai, kích thước, tốc độ   ⇒ outputs/compare_test.csv
└──────────────┬─────────────────────────────────
               ▼
      chọn 1 mô hình (độ chính xác trước, rồi tốc độ / dung lượng)
               ▼
┌─ H. XUẤT VÀ TRIỂN KHAI ─────────────────────── export_onnx/export.py, predict_onnx.py
│  PyTorch → ONNX, tự kiểm chứng, chạy không cần PyTorch   ⇒ model.onnx, model.json
└────────────────────────────────────────────────
```

### Dữ liệu biến đổi thế nào qua các bước

Một ảnh đi từ file đến kết luận như sau:

```
file .jpg 200×200 ảnh xám (data/split/train/scratches/xxx.jpg)
   │  Image.open().convert("RGB")                          dataset.py:47
   ▼
ảnh PIL 200×200, 3 kênh giống nhau
   │  Resize(224, 224)                                     dataset.py:15
   │  [chỉ train] lật ngang, lật dọc, xoay ±10°, đổi sáng/tương phản   dataset.py:16-24
   │  ToTensor()       giá trị 0..255 → 0..1                dataset.py:25
   │  Normalize(MEAN, STD) của ImageNet                    dataset.py:26
   ▼
tensor (3, 224, 224)
   │  DataLoader gom 16 ảnh                                dataset.py:59-66
   ▼
batch (16, 3, 224, 224)  +  nhãn (16,) vd [3, 0, 5, ...]
   │  model(images)                                        train.py:129
   ▼
logits (16, 6)  điểm số thô của 6 lớp
   ├─ lúc train:   CrossEntropyLoss(logits, nhãn) → loss → backward
   └─ lúc đánh giá: softmax → xác suất (16, 6), tổng mỗi hàng = 1   evaluate.py:27
                     │  argmax                              evaluate.py:34
                     ▼
                  lớp dự đoán, vd 5 = class_names[5] = "scratches"
```

---

## 11.2. Chi tiết từng giai đoạn

### A. Môi trường

```
┌─ A. MÔI TRƯỜNG ─────────────────────────────────────────────
│
│  [A1] Tạo môi trường ảo Python 3.10 ở ổ D        .venv/
│  [A2] Cài thư viện cố định phiên bản             requirements.txt
│        torch, torchvision, scikit-learn, matplotlib, pillow, onnx, onnxruntime
│  [A3] Kiểm tra: python -c "import torch; print(torch.__version__)"
│  [A4] Đặt thiết lập trong config.py (nơi DUY NHẤT chứa thiết lập)
│        ├─ đường dẫn:     NEU_DIR, SPLIT_DIR, OUTPUT_DIR
│        ├─ chia dữ liệu:  VAL_RATIO = 0.15, SEED = 42
│        ├─ ảnh:           IMAGE_SIZE = 224, MEAN, STD
│        ├─ mô hình:       MODEL_NAME, PRETRAINED
│        ├─ huấn luyện:    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
│        │                 NUM_WORKERS, USE_CLASS_WEIGHTS, EARLY_STOPPING_PATIENCE
│        └─ augmentation:  HORIZONTAL_FLIP, VERTICAL_FLIP, ROTATION_DEGREES, BRIGHTNESS, CONTRAST
│  [A5] Lần train đầu tự tải trọng số pretrained về
│        C:\Users\Wisol\.cache\torch\hub\checkpoints\
└─────────────────────────────────────────────────────────────
```

### B. Chuẩn bị dữ liệu: `prepare_data.py`

```
┌─ B. CHUẨN BỊ DỮ LIỆU ───────────────────────── python prepare_data.py [--overwrite]
│
│  [B1] Đọc tham số --overwrite                                       :29-31
│  [B2] ◇ Có đủ 2 thư mục nguồn NEU (train/images, validation/images)?   :33-36
│        └─ không → "Folder not found: ... Check NEU_DIR in config.py."  ✖ DỪNG
│  [B3] Tên lớp = tên thư mục con, sắp theo alphabet (cả 2 tập)          :39-40
│  [B4] ◇ Hai tập có cùng danh sách lớp?                                  :41
│        └─ không → "Train classes ... differ from validation classes ..."  ✖ DỪNG
│  [B5] ◇ data/split đã tồn tại?                                          :45
│        ├─ có, không --overwrite → "data/split already exists. ..."      ✖ DỪNG
│        └─ có, có --overwrite   → xóa data/split cũ (chỉ là bản sao)     :49
│  [B6] seen = {} (mã MD5 → ảnh đầu tiên),  rng = Random(SEED)            :51-52
│
│  ╔═ LẶP CHO TỪNG LỚP (6 lớp)                                            :54
│  ║  [B7]  Tính MD5 mọi ảnh TEST của lớp, ghi vào seen TRƯỚC              :55-57
│  ║         (để ảnh train trùng ảnh test bị loại → không rò rỉ test vào train)
│  ║  ╔═ LẶP CHO TỪNG ẢNH TRAIN GỐC                                        :60
│  ║  ║  [B8]  tính MD5                                                    :61
│  ║  ║  ◇ MD5 đã có trong seen?                                           :62
│  ║  ║    ├─ có    → "duplicate skipped: ..." bỏ ảnh                       :63-64
│  ║  ║    └─ không → ghi vào seen, giữ ảnh                                :65-66
│  ║  ╚═
│  ║  [B9]  Xáo trộn ảnh train bằng rng (cố định theo SEED)                :68
│  ║  [B10] num_val = round(số ảnh × 0.15)                                 :69
│  ║  [B11] Chép: phần còn lại → train/,  num_val ảnh đầu → val/,
│  ║        toàn bộ validation NEU → test/                                 :70-72
│  ║        ⇒ data/split/{train,val,test}/<lớp>/
│  ║  [B12] In 1 dòng: lớp, số ảnh train, val, test                        :73
│  ╚═
│  [B13] "Done -> data/split. Next step: python train.py"                 :74
└─────────────────────────────────────────────────────────────
```

Kết quả: train 1.223 ảnh, val 216 ảnh, test 360 ảnh (60 mỗi lớp).

### C. Kiểm tra trước khi train

```
┌─ C. KIỂM TRA ───────────────────────────────────────────────
│
│  [C1] python dataset.py                                    dataset.py:69-77
│        ├─ in danh sách 6 lớp
│        ├─ in số ảnh mỗi tập, mỗi lớp  → khớp với bảng ở B12?
│        └─ lấy thử 1 batch → shape phải là (16, 3, 224, 224)
│
│  [C2] python model.py                                      model.py:80-89
│        ╔═ LẶP CHO TỪNG MÔ HÌNH
│        ║  tạo mô hình 6 lớp → chạy 2 ảnh giả
│        ║  → output phải là (2, 6); in số tham số và lớp dùng cho Grad-CAM
│        ╚═
│
│  ◇ Cả 2 đều đúng?  không → sửa trước khi train (train lỗi giữa chừng mất nhiều thời gian hơn)
└─────────────────────────────────────────────────────────────
```

### D. Huấn luyện: `train.py`

Đây là giai đoạn chính, gồm 6 khối D0 → D5. Khối D4 lặp lại mỗi epoch.

```
┌─ D0. KHỞI ĐỘNG ─────────────────────────────── python train.py --model X [--resume]
│  [D0.1] Đọc tham số: --model, --output-dir, --resume                :48-52
│  [D0.2] output_dir = outputs/<model>  (mỗi mô hình một thư mục)     :53
│  [D0.3] torch.manual_seed(SEED)  → chạy lại ra kết quả giống nhau   :55
│  [D0.4] device = cuda nếu có GPU, không thì cpu                     :56
└──────────────┬──────────────────────────────────────────────
               ▼
┌─ D1. DỮ LIỆU ──────────────────────────────────────────────
│  [D1.1] ◇ Có data/split/train?                                      :59-61
│          └─ không → "data/split/train not found. Run prepare_data.py first."  ✖ DỪNG
│  [D1.2] class_names = tên thư mục trong train/, theo alphabet       :62
│          ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
│  [D1.3] train_loader: xáo trộn + augmentation, batch 16             :63
│  [D1.4] val_loader:   không xáo, không augmentation                 :64
│  [D1.5] Đếm số ảnh train mỗi lớp, in thống kê                       :65-67
│  [D1.6] ◇ Có lớp 0 ảnh, hoặc val trống?                             :68-70
│          └─ có → "Every class needs train images ..."             ✖ DỪNG
└──────────────┬──────────────────────────────────────────────
               ▼
┌─ D2. MÔ HÌNH, LOSS, OPTIMIZER ─────────────────────────────
│  [D2.1] build_model(X, 6 lớp, pretrained)                           :73
│          ├─ tải kiến trúc + trọng số ImageNet                   model.py:44-48
│          ├─ thay lớp cuối 1000 lớp → 6 lớp                      model.py:49
│          └─ .to(device)
│  [D2.2] ◇ USE_CLASS_WEIGHTS?                                        :74-77
│          └─ có → trọng số lớp = tổng ảnh / (6 × ảnh của lớp)   (NEU ≈ 1)
│  [D2.3] criterion = CrossEntropyLoss(weight)                        :78
│  [D2.4] optimizer = AdamW(lr = 1e-4, weight_decay = 1e-4)           :79
│  [D2.5] scheduler = CosineAnnealingLR(T_max = EPOCHS)               :80
└──────────────┬──────────────────────────────────────────────
               ▼
┌─ D3. CHECKPOINT VÀ RESUME ─────────────────────────────────
│  [D3.1] Tạo output_dir; đặt đường dẫn best.pt, last.pt              :83-85
│  [D3.2] Khởi tạo trạng thái: start_epoch = 0, best_score = -1,
│         best_val_loss = ∞, history = [], epochs_without_improvement = 0   :86-90
│  [D3.3] train_config = ảnh chụp mọi thiết lập của config.py         :91
│  [D3.4] ◇ Có --resume và có last.pt?                                :92
│   ├─ không → train từ đầu
│   └─ có:
│        [D3.5] đọc last.pt                                           :93
│        [D3.6] ◇ class_names hoặc model khác lần trước?              :94-96
│                └─ có → "Class names or model changed ..."         ✖ DỪNG
│        [D3.7] khôi phục 8 thứ: trọng số, optimizer, scheduler, epoch,
│               best_score, best_val_loss, history, bộ đếm early stopping   :97-104
│        [D3.8] "Resumed from epoch N"                                :105
│        [D3.9] ◇ config.py đã bị sửa so với lần trước?               :106-112
│                └─ có → "Warning: ... changed since last run: ..." + "Note: ..."  (vẫn chạy tiếp)
│  [D3.10] Ghi thiết lập ra file                                      :114-115
│          ⇒ outputs/X/config.json
└──────────────┬──────────────────────────────────────────────
               ▼
         (sang D4, xem sơ đồ riêng bên dưới)
               ▼
┌─ D5. KẾT THÚC ─────────────────────────────────────────────
│  [D5.1] "Best val macro F1: ... (val loss ...) -> outputs/X/best.pt"    :177
│  [D5.2] "Next step: python evaluate.py --checkpoint outputs/X/best.pt"  :178
└─────────────────────────────────────────────────────────────
```

#### D4. Vòng lặp epoch (trái tim của quá trình train)

```
[D4.0] In cấu hình: "Model: X | device: cpu | epochs: 20 | output: ..."     :118

╔═ VÒNG LẶP EPOCH: epoch = start_epoch … EPOCHS-1   (tối đa 20 lần)          :119
║
║  [D4.1] ◇ Early stopping: đã 7 epoch liên tiếp không có best mới?           :120-122
║          └─ có → "Early stopping: no new best ..." → THOÁT VÒNG LẶP → D5
║  [D4.2] Bấm giờ epoch                                                        :123
║
║  ┌─ PHẦN TRAIN: mô hình HỌC ───────────────────────────────────
║  │ [D4.3] model.train()   bật Dropout, BatchNorm cập nhật thống kê            :124
║  │ [D4.4] total_loss = 0                                                      :125
║  │
║  │ ╔═ VÒNG LẶP BATCH: 77 batch × 16 ảnh   (DataLoader đọc ảnh, xem 11.1)     :126
║  │ ║  [D4.5]  ảnh, nhãn → device                                             :127-128
║  │ ║  [D4.6]  ① FORWARD    outputs = model(images)   → logits (16, 6)        :129
║  │ ║  [D4.7]  ② LOSS       loss = criterion(outputs, labels)                 :130
║  │ ║  [D4.8]  ③ ZERO_GRAD  xóa gradient của batch trước                      :131
║  │ ║  [D4.9]  ④ BACKWARD   tính gradient cho mọi trọng số                    :132
║  │ ║  [D4.10] ⑤ STEP       AdamW chỉnh trọng số theo gradient                :133
║  │ ║  [D4.11] cộng dồn: total_loss += loss × số ảnh trong batch              :134
║  │ ╚═ hết 77 batch
║  │
║  │ [D4.12] scheduler.step()   giảm learning rate theo đường cosine           :135
║  └─────────────────────────────────────────────────────────────
║
║  ┌─ PHẦN VALIDATION: chỉ ĐO, không học ──────────────────────
║  │ [D4.13] run_inference(model, val_loader)                                   :137
║  │          ├─ model.eval()      tắt Dropout, BatchNorm dùng thống kê đã học  evaluate.py:18
║  │          ├─ torch.no_grad()   không tính gradient → nhanh, ít RAM         evaluate.py:22
║  │          ╔═ LẶP 14 batch val (216 ảnh)
║  │          ║  forward → cộng loss → softmax → xác suất                     evaluate.py:23-28
║  │          ╚═
║  │          └─ trả về: val_loss, nhãn thật, xác suất
║  │ [D4.14] decide_predictions: argmax → lớp dự đoán                           :138
║  │ [D4.15] compute_metrics: accuracy, macro F1                                :138
║  └─────────────────────────────────────────────────────────────
║
║  [D4.16] Ghi 1 dòng lịch sử: epoch, train_loss, val_loss,
║          val_accuracy, val_macro_f1, seconds                                 :139-146
║
║  [D4.17] ◇ CÓ BEST MỚI?                                                      :147-148
║           val F1 > best_score
║           HOẶC (val F1 = best_score VÀ val loss < best_val_loss)
║     ├─ có    → best_score = F1, best_val_loss = val loss                     :149-151
║     │          epochs_without_improvement = 0                                 :152
║     └─ không → epochs_without_improvement += 1                              :152
║
║  [D4.18] In: "Epoch 5/20 | train loss ... | val F1 ... | 80s *"             :153-160
║          (* = best mới; "(no improvement 3/7)" = đang đếm early stopping)
║
║  ┌─ LƯU KẾT QUẢ (sau MỌI epoch) ──────────────────────────────
║  │ [D4.19] Tạo dict checkpoint: model_name, class_names, image_size,
║  │         epoch, config, model (trọng số)                                   :162-169
║  │ [D4.20] ◇ best mới?  có → ⇒ best.pt                                       :170-171
║  │ [D4.21] luôn luôn → ⇒ last.pt  (= checkpoint + optimizer, scheduler,
║  │         best_score, best_val_loss, history, bộ đếm)                       :172-174
║  │ [D4.22] save_history → ⇒ history.csv, training_curves.png                 :175
║  └─────────────────────────────────────────────────────────────
║
╚═ sang epoch tiếp theo  (hết 20 epoch → D5)
```

Năm bước ① → ⑤ trong vòng lặp batch là phần **giống hệt nhau ở mọi dự án PyTorch**. Lỗi hay gặp nhất là quên ③ `zero_grad`: PyTorch cộng dồn gradient, nên gradient của batch trước sẽ dồn vào batch sau.

### E. Đánh giá trên test: `evaluate.py`

```
┌─ E. ĐÁNH GIÁ ───────────────────────────────── python evaluate.py --checkpoint outputs/X/best.pt [--split test]
│
│  [E1] Đọc tham số --checkpoint, --split (mặc định test)               :67-70
│  [E2] ◇ Có file checkpoint?                                           :73-75
│        └─ không → "Checkpoint not found: ... Run train.py first."   ✖ DỪNG
│  [E3] load_model_from_checkpoint                                 model.py:71-77
│        ├─ đọc file .pt
│        ├─ dựng kiến trúc theo model_name (pretrained=False, không cần tải)
│        ├─ nạp trọng số đã học
│        └─ model.eval()
│  [E4] DataLoader tập test: không augmentation, image_size lấy từ checkpoint   :78
│  [E5] run_inference → xác suất 360 ảnh                                :79
│  [E6] argmax → dự đoán;  accuracy, macro F1                           :80-81
│  [E7] In: model, epoch, split, accuracy, macro F1                     :84-85
│  [E8] In bảng precision / recall / F1 từng lớp                        :86-87
│  [E9] Confusion matrix (hàng = lớp thật, cột = lớp đoán): in ra màn hình   :89-94
│  [E10] In các cặp nhầm nhiều nhất "true -> predicted"                 :95-99
│  [E11] ⇒ confusion_matrix_test.png                                    :101-105
│  [E12] Lọc ảnh sai, sắp theo độ tự tin giảm dần
│        (sai mà rất tự tin = đáng xem nhất, có thể do nhãn sai)        :107-109
│  [E13] ⇒ misclassified_test.csv  (đường dẫn, nhãn thật, nhãn đoán, 6 xác suất)   :110-116
│  [E14] ◇ Có ảnh sai?  có → ⇒ misclassified_test.png (lưới tối đa 20 ảnh)   :117-120
└─────────────────────────────────────────────────────────────
```

Quy tắc: tập **test chỉ dùng ở bước này**, sau khi đã chọn xong `best.pt` bằng val. Không dùng test để chọn epoch hay chỉnh thiết lập.

### F. Phân tích lỗi: `gradcam.py`, `predict.py`

```
┌─ F1. GRAD-CAM: mô hình nhìn vào đâu? ─────────── python gradcam.py --checkpoint ... [--misclassified]
│
│  [F1.1] Nạp mô hình từ checkpoint
│  [F1.2] Chọn ảnh:
│          ◇ --misclassified?
│           ├─ có → đọc misclassified_test.csv (phải chạy E trước)  gradcam.py:119-124
│           │       chưa có file → "... not found. Run evaluate.py ..."  ✖ DỪNG
│           ├─ có đường dẫn ảnh → dùng ảnh đó
│           └─ không có gì → 3 ảnh test đầu tiên mỗi lớp            gradcam.py:52-56
│  ╔═ LẶP CHO TỪNG ẢNH                                               gradcam.py:135
│  ║  [F1.3] gắn forward hook vào lớp tích chập cuối (gradcam_layer)  gradcam.py:28
│  ║  [F1.4] forward → hook cất bản đồ đặc trưng (vd 512 × 7 × 7)     gradcam.py:31
│  ║  [F1.5] backward từ điểm số của lớp dự đoán → hook cất gradient  gradcam.py:35
│  ║  [F1.6] gỡ hook (trong finally)                                  gradcam.py:37
│  ║  [F1.7] trọng số kênh = trung bình gradient;                       gradcam.py:39-40
│  ║         bản đồ nhiệt = tổng có trọng số các kênh → ReLU             gradcam.py:41
│  ║         → chia cho max để về 0..1 → phóng 7×7 lên 224×224          gradcam.py:42-43
│  ╚═
│  [F1.8] ⇒ outputs/X/gradcam/gradcam_01.png ...                     gradcam.py:148
│
│  Đọc kết quả: vùng đỏ phải nằm TRÊN vết lỗi. Nếu đỏ ở nền hay ở góc ảnh
│  → mô hình học "đường tắt" sai, dù accuracy có cao.
└─────────────────────────────────────────────────────────────

┌─ F2. XEM ẢNH SAI BẰNG MẮT ──────────────────── python predict.py data/split/test --only-wrong --show
│  [F2.1] Duyệt từng ảnh: mở → tiền xử lý giống E → dự đoán → top-3 xác suất
│  [F2.2] Nhãn thật lấy từ tên thư mục cha (nếu là tên lớp)
│  [F2.3] In mỗi ảnh 1 dòng, đánh dấu "<- WRONG"; vẽ lưới xanh/đỏ nếu có --show/--save
└─────────────────────────────────────────────────────────────
```

### G. So sánh các mô hình: `compare_models.py`

```
┌─ G. SO SÁNH ────────────────────────────────── python compare_models.py [--split test]
│
│  [G1] Tìm mọi outputs/*/best.pt   (không có → "No checkpoints found ..."  ✖ DỪNG)   :34-38
│  [G2] Đo trên CPU (giống máy chuyền thực tế)                        :39
│  ╔═ LẶP CHO TỪNG CHECKPOINT                                          :42
│  ║  [G3] nạp mô hình → dự đoán CÙNG tập test → accuracy, F1, số ảnh sai   :43-47
│  ║  [G4] số tham số, dung lượng file best.pt                         :54-55
│  ║  [G5] tốc độ: chạy 5 lần khởi động, rồi đo trung bình 30 lần 1 ảnh   :19-27
│  ╚═
│  [G6] Sắp xếp: F1 cao trước, bằng nhau thì nhanh hơn trước           :60
│  [G7] In bảng + dòng CNN NumPy (0.942) để so sánh                    :61-67
│  [G8] ⇒ outputs/compare_test.csv                                     :69-73
└─────────────────────────────────────────────────────────────
```

### H. Xuất và triển khai: `export_onnx/`

```
┌─ H. XUẤT ONNX ──────────────────────────────── python export_onnx/export.py --checkpoint outputs/X/best.pt
│
│  [H1] ◇ Có checkpoint?  không → "Checkpoint not found ..."      ✖ DỪNG   export.py:119-121
│  [H2] Nạp mô hình PyTorch lên CPU (đã ở chế độ eval)                    export.py:122
│  [H3] torch.onnx.export → ⇒ model.onnx                                   export.py:41
│  [H4] ⇒ model.json: class_names, image_size, mean, std, nguồn checkpoint   export.py:66-67
│  [H5] Nạp lại model.onnx bằng onnxruntime                                export.py:130
│  [H6] KIỂM CHỨNG trên cả tập test                                        export.py:70-92
│        ├─ chạy PyTorch và ONNX trên cùng ảnh
│        ├─ sai lệch xác suất lớn nhất ≤ 1e-4 ?
│        └─ kết luận trùng 360/360 ?
│  [H7] Đo tốc độ PyTorch vs ONNX, 1 ảnh mỗi lần (luôn chạy)               export.py:95-110, 132
│  [H8] ◇ H6 đạt cả hai điều kiện?                                          export.py:133-136
│         ├─ không → "Verification FAILED: do not use this model.onnx."  ✖ DỪNG (mã 1)
│         └─ có    → "Verification passed: model.onnx gives the same results as PyTorch."
└──────────────┬──────────────────────────────────────────────
               ▼
┌─ H9. DÙNG Ở MÁY KHÁC ───────────────────────── python export_onnx/predict_onnx.py <ảnh|thư mục>
│  chỉ cần model.onnx + model.json + onnxruntime, KHÔNG cần PyTorch
│  preprocess phải khớp build_transforms(train=False); decide phải khớp argmax
└─────────────────────────────────────────────────────────────
```

---

## 11.3. Vòng chẩn đoán khi kết quả chưa đạt

Ở cuối giai đoạn F, nếu kết quả chưa đạt thì đọc `training_curves.png` và `confusion_matrix_test.png` theo sơ đồ này:

```
◇ Train loss có giảm không?
 ├─ không giảm / giảm rất chậm  → HỌC CHƯA ĐƯỢC (underfit)
 │     → kiểm tra lại dữ liệu và nhãn (C1), tăng LEARNING_RATE, tăng EPOCHS,
 │       dùng PRETRAINED = True, thử mô hình lớn hơn
 │
 └─ có giảm
      ◇ Val loss có tăng dần trong khi train loss vẫn giảm?
       ├─ có  → HỌC THUỘC LÒNG (overfit)
       │     → thêm augmentation, tăng WEIGHT_DECAY, bật EARLY_STOPPING_PATIENCE,
       │       thêm dữ liệu, thử mô hình nhỏ hơn
       │
       └─ không
            ◇ Val tốt nhưng test kém hơn nhiều?
             ├─ có  → val và test KHÁC NHAU về bản chất (ánh sáng, máy chụp, thời điểm)
             │     → chia lại dữ liệu cho đại diện hơn (B), thêm ảnh giống test vào train
             │
             └─ không
                  ◇ Sai tập trung ở 1–2 cặp lớp (confusion matrix)?
                   ├─ có  → xem ảnh sai (F2) và Grad-CAM (F1):
                   │       nhãn sai? → sửa nhãn rồi quay lại B
                   │       lỗi thật sự giống nhau? → thêm ảnh cho cặp đó, tăng IMAGE_SIZE
                   └─ không → mô hình đã ổn, sang G
```

Mỗi lần chỉ đổi **một** thiết lập rồi train lại. Đổi nhiều thứ cùng lúc thì không biết thay đổi nào có tác dụng. Train lại thì bỏ `--resume`, và dùng `--output-dir` mới nếu muốn giữ kết quả cũ để so sánh.

---

## 11.4. Checklist áp dụng cho dự án mới

Đi lần lượt từ trên xuống. Cột "Giữ nguyên" là phần chép sang dự án mới gần như không phải sửa.

| Bước | Việc cần làm | Phải đổi | Giữ nguyên |
|---|---|---|---|
| A | Tạo `.venv`, `requirements.txt`, `config.py` | Đường dẫn dữ liệu, số lớp, kích thước ảnh | Cách đặt mọi thiết lập ở một file |
| B | Script chia dữ liệu train/val/test | Nguồn dữ liệu, tỉ lệ chia | Seed cố định, bỏ ảnh trùng, ghi test trước để chống rò rỉ, không ghi đè khi chưa cho phép |
| C | Chạy thử dataset và mô hình | — | Kiểm tra shape trước khi train |
| D1 | Dataset + DataLoader | Cách đọc ảnh, augmentation hợp với ảnh **thật** (vd ảnh có chiều trên/dưới thì không lật dọc) | Augmentation chỉ cho train; val/test giữ nguyên |
| D2 | Mô hình, loss, optimizer, scheduler | Kiến trúc, số đầu ra; loss theo bài toán (OK/NG có thể dùng `BCEWithLogitsLoss`) | AdamW + cosine là điểm khởi đầu tốt cho fine-tune |
| D3 | Checkpoint, resume, lưu config | — | `best.pt` + `last.pt` + `config.json`; lưu `class_names` trong checkpoint |
| D4 | Vòng lặp epoch | Chỉ số chọn best (vd recall của lớp NG thay cho macro F1) | 5 bước forward → loss → zero_grad → backward → step; `train()` / `eval()` + `no_grad()`; early stopping |
| E | Đánh giá trên test | Chỉ số báo cáo theo yêu cầu QC | Test chỉ dùng 1 lần, sau khi chọn xong mô hình |
| F | Phân tích lỗi | Lớp dùng cho Grad-CAM nếu đổi kiến trúc | Xem ảnh sai + Grad-CAM trước khi tin con số |
| G | So sánh mô hình | Danh sách mô hình | Cùng tập test, cùng thiết lập, đo tốc độ trên đúng máy sẽ chạy |
| H | Xuất ONNX và kiểm chứng | — | Luôn so kết quả ONNX với PyTorch trước khi dùng |

### Bố cục file chuẩn cho dự án mới

```
<du_an_moi>/
├── config.py          # [A]  mọi thiết lập
├── prepare_data.py    # [B]  chia train / val / test
├── dataset.py         # [D1] Dataset, transforms, DataLoader   (chạy trực tiếp = bước C1)
├── model.py           # [D2] tạo mô hình, nạp checkpoint       (chạy trực tiếp = bước C2)
├── train.py           # [D]  khung D0 → D5
├── evaluate.py        # [E]  đánh giá + các hàm dùng chung (run_inference, decide, metrics)
├── predict.py         # [F2] dự đoán ảnh bất kỳ
├── gradcam.py         # [F1] giải thích mô hình
├── compare_models.py  # [G]  so sánh nhiều mô hình
├── export_onnx/       # [H]  xuất + chạy không cần PyTorch
├── requirements.txt   # [A]
├── data/split/        # sinh ra ở B (không sửa tay)
└── outputs/<model>/   # sinh ra ở D, E, F (không sửa tay)
```

Hai nguyên tắc giữ cho mọi giai đoạn khớp nhau:

1. **Một nơi cho mỗi thứ.** Thiết lập chỉ ở `config.py`. Tiền xử lý ảnh chỉ ở `build_transforms`. Quy tắc ra kết luận chỉ ở `decide_predictions`. Các script khác gọi lại, không viết lại.
2. **Checkpoint tự mô tả.** File `.pt` mang theo tên mô hình, danh sách lớp, kích thước ảnh và thiết lập lúc train. Chỉ cần đưa file này là mọi bước E → H chạy được, không cần nhớ đã train thế nào.
