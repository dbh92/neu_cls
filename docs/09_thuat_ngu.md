# 9. Thuật ngữ

Cột "Ở đâu" chỉ chỗ thuật ngữ xuất hiện trong dự án.

| Thuật ngữ | Nghĩa | Ở đâu |
|---|---|---|
| **Accuracy** | Tỉ lệ ảnh đoán đúng trên tổng số ảnh | `compute_metrics` |
| **AdamW** | Bộ tối ưu tự điều chỉnh bước học cho từng trọng số, kèm weight decay | `train.py` |
| **argmax** | Chỉ số của phần tử lớn nhất. Ở đây: lớp có xác suất cao nhất | `decide_predictions` |
| **Augmentation** | Biến đổi ngẫu nhiên ảnh train (lật, xoay, đổi sáng) để mô hình học tổng quát hơn | `build_transforms(train=True)` |
| **Backbone (thân)** | Phần tích chập của mô hình, trích đặc trưng từ ảnh | `model.py` |
| **Backward** | Bước tính gradient của loss theo mọi trọng số | `loss_value.backward()` |
| **Batch** | Nhóm ảnh được xử lý cùng lúc (16 ảnh) | `BATCH_SIZE` |
| **BatchNorm** | Lớp chuẩn hóa theo batch; hành xử khác nhau giữa `train()` và `eval()` | trong mọi mô hình |
| **Checkpoint** | File `.pt` lưu trọng số và thông tin đi kèm | `best.pt`, `last.pt` |
| **Class weights** | Trọng số loss theo lớp; lớp ít ảnh nặng hơn | `USE_CLASS_WEIGHTS` |
| **Confusion matrix** | Bảng đếm: hàng = lớp thật, cột = lớp đoán | `evaluate.py` |
| **CosineAnnealingLR** | Lịch giảm learning rate theo đường cosine | `train.py` |
| **CrossEntropyLoss** | Hàm mất mát cho phân loại nhiều lớp; nhận logits | `train.py` |
| **DataLoader** | Đối tượng gom ảnh thành batch, đọc song song | `make_dataloader` |
| **Dataset** | Đối tượng biết "ảnh thứ i là gì, nhãn gì" | `ImageClassificationDataset` |
| **Dropout** | Tắt ngẫu nhiên một phần nơ-ron khi train để chống overfit; tắt khi `eval()` | `SimpleCNN`, EfficientNet, MobileNet |
| **Early stopping** | Tự dừng train khi nhiều epoch liên tiếp không tiến bộ | `EARLY_STOPPING_PATIENCE` |
| **Epoch** | Một lần duyệt hết tập train | `EPOCHS` |
| **F1** | Trung bình điều hòa của precision và recall | `classification_report` |
| **Fine-tune** | Train tiếp một mô hình pretrained trên dữ liệu mới, learning rate nhỏ | `train.py` |
| **Forward** | Bước đưa ảnh qua mô hình để ra logits | `model(images)` |
| **Forward hook** | Hàm được PyTorch gọi ngay sau khi một lớp tính xong; dùng để "nghe lén" output | `gradcam.py` |
| **Gradient** | Đạo hàm: trọng số thay đổi một chút thì loss (hoặc điểm số) thay đổi bao nhiêu | backward |
| **Grad-CAM** | Kỹ thuật vẽ bản đồ nhiệt vùng ảnh ảnh hưởng tới kết luận | `gradcam.py` |
| **Head (đầu)** | Lớp cuối của mô hình, biến đặc trưng thành điểm số từng lớp | `fc`, `classifier[...]` |
| **ImageNet** | Bộ 1,2 triệu ảnh, 1000 lớp, dùng để train trước các mô hình | trọng số pretrained |
| **Learning rate** | Độ lớn mỗi bước cập nhật trọng số | `LEARNING_RATE` |
| **Logits** | Điểm số thô của từng lớp, chưa qua softmax | đầu ra mô hình, ONNX |
| **Loss** | Con số đo mô hình sai bao nhiêu; train là làm loss nhỏ đi | `train_loss`, `val_loss` |
| **Macro F1** | Trung bình F1 của các lớp, mỗi lớp nặng như nhau | tiêu chí chọn `best.pt` |
| **MD5** | Mã băm của file; hai file cùng mã = giống hệt nhau | `prepare_data.py` |
| **Normalize** | Chuẩn hóa `(x − mean) / std` theo kênh màu | `build_transforms` |
| **ONNX** | Định dạng mô hình chung, chạy bằng onnxruntime không cần PyTorch | `export_onnx/` |
| **Overfit** | Mô hình học thuộc ảnh train, kém trên ảnh mới. Dấu hiệu: train loss giảm, val loss tăng | `training_curves.png` |
| **Precision** | Khi mô hình nói lớp X, bao nhiêu phần trăm đúng | `classification_report` |
| **Pretrained** | Mô hình đã được train sẵn (trên ImageNet) | `PRETRAINED` |
| **Recall** | Trong các ảnh thật sự là lớp X, mô hình bắt được bao nhiêu phần trăm | `classification_report` |
| **Resume** | Chạy tiếp lần train bị ngắt từ `last.pt` | `--resume` |
| **Seed** | Hạt giống ngẫu nhiên; cùng seed thì cùng kết quả ngẫu nhiên | `SEED` |
| **Softmax** | Đổi logits thành xác suất dương, tổng bằng 1 | `run_inference`, `predict_onnx.py` |
| **`state_dict`** | Dict chứa toàn bộ trọng số của mô hình (hoặc trạng thái optimizer) | checkpoint |
| **Support** | Số ảnh thật của một lớp trong tập đang đánh giá | `classification_report` |
| **Tensor** | Mảng nhiều chiều của PyTorch, vd ảnh `(3, 224, 224)` | khắp nơi |
| **Test** | Tập ảnh chỉ dùng để chấm điểm cuối, không dùng để chọn gì | `data/split/test` |
| **Transfer learning** | Dùng lại mô hình đã học ở bài khác, chỉ thay đầu và train tiếp | `build_model` |
| **Validation (val)** | Tập ảnh dùng để chọn epoch tốt nhất và quyết định early stopping | `data/split/val` |
| **Weight decay** | Phạt trọng số lớn để chống overfit | `WEIGHT_DECAY` |
| **`eval()` / `train()`** | Chế độ của mô hình: `eval()` tắt Dropout và cố định BatchNorm | `load_model_from_checkpoint`, `train.py` |
| **`no_grad()`** | Tắt tính gradient khi chỉ cần dự đoán, nhanh và tiết kiệm bộ nhớ | `run_inference`, `predict.py` |
