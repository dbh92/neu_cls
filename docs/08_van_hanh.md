# 8. Vận hành dự án

Phần này là "sổ tay vận hành": làm gì, theo thứ tự nào, trong từng tình huống.

## 8.1. Bảng lệnh

Chạy từ `D:\Python_Practice\neu_cls`. `python` = `.venv\Scripts\python.exe`.

| Lệnh | Tham số | Ghi ra |
|---|---|---|
| `python prepare_data.py` | `--overwrite` | `data/split/` |
| `python dataset.py` | | (chỉ in) |
| `python model.py` | | (chỉ in) |
| `python train.py` | `--model NAME`, `--output-dir DIR`, `--resume` | `outputs/<model>/best.pt, last.pt, history.csv, training_curves.png` |
| `python evaluate.py` | `--checkpoint P`, `--split test\|val\|train` | `confusion_matrix_<split>.png`, `misclassified_<split>.csv/.png` cạnh checkpoint |
| `python compare_models.py` | `--split test\|val\|train` | `outputs/compare_<split>.csv` |
| `python gradcam.py [ảnh\|thư mục...]` | `--checkpoint P`, `--misclassified`, `--per-class N`, `--save DIR`, `--show` | `<thư mục checkpoint>/gradcam/gradcam_NN.png` |
| `python predict.py <ảnh\|thư mục>...` | `--checkpoint P`, `--show`, `--save DIR`, `--only-wrong` | `DIR/predictions_NN.png` nếu có `--save` |
| `python export_onnx\export.py` | `--checkpoint P`, `--output-dir DIR`, `--split` | `export_onnx/model.onnx`, `model.json` |
| `python export_onnx\predict_onnx.py <ảnh\|thư mục>...` | `--model-dir DIR` | (chỉ in) |

Mặc định của `--checkpoint` ở mọi script là `outputs/<MODEL_NAME>/best.pt`, với `MODEL_NAME` trong `config.py` (hiện là `resnet18`).

## 8.2. Quy trình chuẩn từ đầu đến cuối

```
 1. prepare_data.py              (một lần duy nhất)
 2. dataset.py, model.py         (kiểm tra nhanh, vài giây)
 3. train.py --model X           (10–30 phút mỗi mô hình)      ┐
 4. evaluate.py --checkpoint outputs/X/best.pt                  │ lặp cho
 5. gradcam.py  --checkpoint outputs/X/best.pt                  │ từng mô hình
    gradcam.py  --checkpoint ... --misclassified --save outputs/X/gradcam_wrong   ┘
 6. compare_models.py            (sau khi đã train mọi mô hình)
 7. chọn mô hình triển khai
 8. export_onnx\export.py --checkpoint outputs/<chọn>/best.pt  (phải thấy "Verification passed")
 9. chép predict_onnx.py + model.onnx + model.json + requirements.txt sang máy đích
```

Train **lần lượt** từng mô hình, không chạy song song: hai lần train cùng lúc tranh nhau 8 nhân CPU, cả hai đều chậm, và thời gian trong `history.csv` không còn so sánh được.

## 8.3. Các kịch bản thường gặp

### A. Train lại một mô hình đã có

```powershell
.venv\Scripts\python.exe train.py --model resnet18
```

Lệnh này **ghi đè** `best.pt`, `last.pt`, `history.csv`, `training_curves.png` trong `outputs/resnet18/`. Nhưng các file do `evaluate.py` và `gradcam.py` tạo trước đó (`confusion_matrix_test.png`, `misclassified_test.csv`, `gradcam/`...) **vẫn còn** và giờ đã **cũ**, không khớp với `best.pt` mới. Sau khi train xong, hãy chạy lại bước 4–5.

Muốn giữ bản cũ để so sánh: dùng `--output-dir outputs/resnet18_v2`, hoặc đổi tên thư mục cũ trước khi train.

### B. Train bị ngắt giữa chừng (tắt máy, Ctrl+C)

```powershell
.venv\Scripts\python.exe train.py --model resnet18 --resume
```

- Giữ nguyên `--model` (và `--output-dir` nếu lần trước có dùng).
- **Không đổi** `EPOCHS` hay các thiết lập train trong `config.py` giữa hai lần chạy. Lịch learning rate được khôi phục theo `EPOCHS` của lần đầu.
- Mất tối đa epoch đang chạy dở.

### C. Thử nghiệm một thay đổi (bài tập trong README)

1. Sửa `config.py` (vd `PRETRAINED = False`).
2. Train vào thư mục riêng: `train.py --model resnet18 --output-dir outputs/resnet18_scratch`.
3. `compare_models.py` sẽ tự đưa thư mục mới vào bảng.
4. **Trả `config.py` về như cũ** sau khi xong, nếu không lần train sau sẽ dùng nhầm thiết lập thử nghiệm.

Ghi lại thay đổi đã thử vào tên thư mục (vd `resnet18_noaug`, `resnet18_lr1e-3`), vì checkpoint không lưu các thiết lập train như learning rate hay augmentation.

### D. Chạy thử nhanh mà không đụng tới `outputs/`

Khi sửa code và muốn biết code có chạy được không:

1. Chép mọi file `*.py` và thư mục `export_onnx/` sang một thư mục tạm.
2. Trong thư mục tạm, tạo `data/split` nhỏ (khoảng 8 ảnh mỗi lớp cho mỗi tập).
3. Đặt `EPOCHS = 2` trong bản `config.py` ở thư mục tạm.
4. Chạy bằng `.venv` của dự án, từ thư mục tạm.

### E. Muốn chia lại dữ liệu

```powershell
.venv\Scripts\python.exe prepare_data.py --overwrite
```

Chỉ làm khi thực sự cần (vd đã xóa bớt ảnh train cho bài tập 4 và muốn khôi phục). Cùng `SEED` thì cách chia giống hệt lần trước. Đổi `SEED` hoặc `VAL_RATIO` thì tập val thay đổi, và các mô hình cũ phải train lại mới so sánh công bằng được (tập test không đổi).

### F. Có thêm ảnh mới, hoặc thêm một loại lỗi mới

`prepare_data.py` chỉ đọc từ `NEU_DIR` với cấu trúc `train/images/<lớp>/` và `validation/images/<lớp>/`. Cách làm:

1. Tạo một thư mục dữ liệu mới **ngoài** thư mục NEU gốc (không sửa NEU gốc), cùng cấu trúc, gồm ảnh cũ + ảnh mới.
2. Trỏ `NEU_DIR` tới thư mục đó.
3. `prepare_data.py --overwrite`.
4. Train lại **mọi** mô hình.

Thêm một loại lỗi mới (thư mục lớp mới) thì danh sách lớp thay đổi: mô hình cũ chỉ có 6 đầu ra, không dùng tiếp được. `--resume` sẽ báo `Class names or model changed since last run...`. Phải train lại từ đầu.

### G. Đưa một mô hình mới ra dùng

Checklist trước khi chép sang máy đích:

- [ ] `evaluate.py` trên test: accuracy và macro F1 đạt yêu cầu; đã xem `misclassified_test.png`.
- [ ] `gradcam.py`: vùng đỏ nằm trên vết lỗi với ảnh đúng; đã xem ảnh sai bằng `--misclassified`.
- [ ] `compare_models.py`: mô hình được chọn có tốc độ chấp nhận được.
- [ ] `export.py` in `Verification passed: model.onnx gives the same results as PyTorch.`
- [ ] Mở `model.json`: `checkpoint` và `epoch` đúng là mô hình bạn chọn.
- [ ] Trên máy đích, `predict_onnx.py` chạy được trên vài ảnh test và cho cùng kết luận.
- [ ] Giữ lại bản `model.onnx` + `model.json` cũ để quay lại nếu cần (`exported_at` cho biết phiên bản).

## 8.4. Trạng thái dự án lúc viết tài liệu (03/11/2026, 15:20)

Đây là ảnh chụp tại một thời điểm, sẽ cũ đi khi bạn chạy tiếp.

- Các mô hình đang được **train lại theo quy tắc chọn `best.pt` mới** (có tiêu chí phụ val loss):
  - `resnet18`: xong, `best.pt` = epoch 14.
  - `mobilenet_v3_large`: xong, `best.pt` = epoch 15.
  - `efficientnet_b0`: đang train.
  - `resnet50`: chưa có trong `outputs/` (chỉ có bản cũ trong `_old_selection/`).
- Thư mục của `resnet18` và `mobilenet_v3_large` **chưa có** file của `evaluate.py` và `gradcam.py` theo `best.pt` mới.
- `export_onnx/model.json` ghi `epoch: 5`, tức `model.onnx` được xuất từ `best.pt` **cũ** của MobileNetV3 (quy tắc cũ). `best.pt` hiện tại là epoch 15. Nếu vẫn chọn MobileNetV3 để triển khai, cần chạy lại `export.py`.
- Bảng kết quả ở mục 6 của `README.md` là của lần train theo quy tắc cũ. Sau khi train lại xong, chạy `compare_models.py` và cập nhật bảng đó.

## 8.5. Dung lượng ổ đĩa

| Thứ | Dung lượng |
|---|---|
| `.venv` | ~1 GB |
| `data/split` | ~30 MB |
| `outputs/<model>/` | `best.pt` cỡ 16 MB (MobileNet, EfficientNet) đến 90 MB (ResNet50); `last.pt` lớn gấp ~3 lần vì chứa thêm trạng thái AdamW |
| `outputs/_old_selection/` | ~670 MB |
| Trọng số pretrained (ổ **C**) | vài chục MB mỗi mô hình |

Khi thiếu chỗ: xóa `last.pt` của những mô hình đã train xong và không cần resume (giữ `best.pt`). `outputs/_old_selection/` cũng có thể xóa khi không cần đối chiếu nữa.

## 8.6. Xử lý sự cố

| Hiện tượng / thông báo | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Folder not found: ... Check NEU_DIR in config.py.` | Thư mục NEU gốc bị di chuyển | Sửa `NEU_DIR` |
| `data/split already exists. Use --overwrite to create it again.` | Đã chia rồi | Không cần làm gì; chỉ dùng `--overwrite` khi muốn chia lại |
| `data/split/train not found. Run prepare_data.py first.` | Chưa chia dữ liệu, hoặc chạy lệnh ở sai thư mục | `cd D:\Python_Practice\neu_cls` rồi chạy lại |
| `Checkpoint not found: ... Run train.py first.` | Sai đường dẫn, hoặc quên `--checkpoint` nên script tìm `outputs/<MODEL_NAME>/best.pt` | Kiểm tra `--checkpoint` |
| `Class names or model changed since last run, cannot resume. Train from scratch.` | `--resume` với `--model` khác, hoặc dữ liệu đã đổi lớp | Bỏ `--resume` hoặc sửa `--model` |
| `No checkpoints found in outputs/*/best.pt. Run train.py first.` | Chưa train mô hình nào | Train ít nhất một mô hình |
| `... misclassified_test.csv not found. Run evaluate.py --checkpoint ... first.` | Chưa đánh giá test cho checkpoint đó | Chạy `evaluate.py` trước |
| `No images to explain.` | `--misclassified` nhưng mô hình không sai ảnh nào | Bình thường |
| `No images to show.` | `--only-wrong` nhưng không có ảnh sai | Bình thường |
| `predict.py` không in gì | Thư mục đưa vào chỉ chứa thư mục con (vd `data/split/test`) | Đưa thư mục lớp, vd `data/split/test/scratches` |
| `Verification FAILED: do not use this model.onnx.` | ONNX không khớp PyTorch | Kiểm tra `preprocess` trong `predict_onnx.py` có khớp `build_transforms` không |
| `RuntimeError: DataLoader worker ...` hoặc treo khi bắt đầu train | Lỗi tiến trình phụ trên Windows | Đặt `NUM_WORKERS = 0` |
| Hết bộ nhớ (`out of memory`, máy đơ) | Batch quá lớn với mô hình to | Giảm `BATCH_SIZE` xuống 8 |
| `UnicodeEncodeError` khi chạy qua pipe | Console dùng cp1252 | `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy |
| Train chậm bất thường | Đang chạy một lần train/so sánh khác | Đợi xong, chạy lần lượt |
| `gradcam.py --misclassified` không mở được ảnh | Chạy từ thư mục khác (CSV lưu đường dẫn tương đối) | Chạy từ thư mục gốc dự án |
| Kết quả `evaluate` không khớp `best.pt` vừa train | Đang xem file ảnh/CSV của lần đánh giá trước | Chạy lại `evaluate.py` |

### Khi có lỗi viết thêm code

Mọi file Python chạy được ở cấp module (không nằm trong hàm) phải đặt trong `if __name__ == '__main__':`. Trên Windows, mỗi tiến trình phụ của DataLoader import lại file chính; thiếu dòng này thì tiến trình phụ sẽ chạy lại cả script và gây lỗi hoặc treo.
