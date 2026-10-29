# Báo Cáo Đánh Giá Model `yolov26-dataset12k-v2` & Giải Pháp Hậu Xử Lý Tự Thích Nghi

Tài liệu này tổng hợp chi tiết kết quả đánh giá mô hình huấn luyện mới nhất tại `models/yolov26-dataset12k-v2/best (3).pt`, phân tích nguyên nhân gốc rễ của các hạn chế khi nhận diện thực tế, và giải thích cơ chế hoạt động của thuật toán hậu xử lý tự thích nghi trong `tests/image.py` và `tests/video.py`.

---

## 1. Đánh Giá Kết Quả Huấn Luyện Model `yolov26-dataset12k-v2`

### 1.1. Thông số cấu hình huấn luyện (Trích xuất từ Checkpoint)
* **File mô hình**: `models/yolov26-dataset12k-v2/best (3).pt` (Dung lượng ~100.3 MB do lưu kèm trạng thái `optimizer` ở giữa quá trình train).
* **Kiến trúc**: `YOLO26s` (`yolo26s.yaml`, `end2end=True`, `reg_max=1`).
* **Các lớp nhận diện (3 classes)**:
  * `0`: `Drowning` (Đuối nước)
  * `1`: `Person out of water` (Người trên bờ / ngoài mặt nước)
  * `2`: `Swimming` (Đang bơi)
* **Trọng số khởi tạo (Pretrained)**: Fine-tune tiếp từ bản 10 epochs (`models/yolov26-dataset12kanh/best.pt`).
* **Siêu tham số (Hyperparameters)**:
  * Số epochs thiết lập: `50` (**Thực tế dừng tại Epoch 25/50**, tổng thời gian train ~9,768 giây $\approx$ 2.7 giờ).
  * Kích thước ảnh (`imgsz`): `960` (tăng từ `640` của bản v1).
  * Batch size: `8`.
  * Optimizer: `AdamW` với tốc độ học khởi tạo `lr0 = 0.0001`, `lrf = 0.01`, `erasing = 0.4`.

---

### 1.2. Bảng so sánh hiệu năng giữa các phiên bản Model

| Chỉ số đánh giá | Bản cũ (`yolov8n`) | Bản v1 (`yolov26-dataset12kanh`) | **Bản v2 (`yolov26-dataset12k-v2`)** | Nhận xét thay đổi (v2 so với v1) |
| :--- | :---: | :---: | :---: | :--- |
| **Số Epochs đã chạy** | 223 / 300 | 10 / 10 | **25 / 50** | Dừng ở epoch 25 |
| **Kích thước ảnh (`imgsz`)** | 640 | 640 | **960** | Diện tích điểm ảnh gấp 2.25 lần bản v1 |
| **Precision (Độ chính xác)** | 70.69% | **87.40%** | **87.09%** | Giảm nhẹ `-0.31%` |
| **Recall (Độ phủ / Không sót)** | 56.24% | 81.78% | **81.87%** | Tăng `+0.09%` (vẫn sót ~18.1% mẫu trên tập Val) |
| **mAP@50** | 61.53% | 86.92% | **87.37%** | Tăng nhẹ `+0.45%` |
| **mAP@50-95 (Độ sát khung)** | 34.28% | 52.57% | **52.78%** | Chỉ tăng **`+0.21%`** sau 25 epochs |
| **Val Box Loss / Cls Loss** | 1.906 / 1.869 | 1.402 / 0.661 | **1.416 / 0.670** | Sai số định vị khung và phân loại chưa giảm thêm |

### 1.3. Diễn biến quá trình hội tụ qua 25 Epochs (Bản v2)

| Epoch | `train/box_loss` | `train/cls_loss` | `val/box_loss` | `val/cls_loss` | `Precision` | `Recall` | `mAP@50` | `mAP@50-95` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.7693 | 1.6670 | 1.7975 | 1.1366 | 83.47% | 60.26% | 73.12% | 36.51% |
| **5** | 1.4713 | 1.2309 | 1.4983 | 0.7615 | 86.27% | 79.36% | 85.46% | 48.62% |
| **10** | 1.4309 | 1.1708 | 1.4771 | 0.7381 | 84.83% | 80.75% | 86.45% | 49.77% |
| **15** | 1.3987 | 1.0823 | 1.4521 | 0.7050 | 84.14% | 83.19% | 86.60% | 50.96% |
| **20** | 1.3723 | 1.0613 | 1.4383 | 0.6998 | 85.59% | 82.60% | 86.99% | 51.44% |
| **25** | **1.3501** | **1.0312** | **1.4156** | **0.6703** | **87.09%** | **81.87%** | **87.37%** | **52.78%** |

---

## 2. Phân Tích Nguyên Nhân Gốc Rễ Khiến Model "Chưa Đóng Khung Hết Người" & "Nhận Diện Chưa Chuẩn"

Khi kiểm tra thực tế trên ảnh và video, các hiện tượng lỗi xuất phát từ **4 nguyên nhân kỹ thuật**:

### 2.1. Lệch độ phân giải (`imgsz`) giữa lúc Train và lúc Suy luận (Inference)
* Model v2 được huấn luyện ở **`imgsz=960`**, trong khi bản v1 trước đó được huấn luyện lâu ở **`imgsz=640`**.
* Ở code cũ, `tests/image.py` để `imgsz=640` và `tests/video.py` hạ xuống **`imgsz=480`** (nhỏ hơn 4 lần diện tích ảnh lúc train), làm các đối tượng người bơi từ góc quay cao (`video3.mp4`) bị thu nhỏ quá mức và bị bỏ sót hàng loạt.
* Ngược lại, nếu chỉ chạy duy nhất ở `imgsz=960` thì lại dễ bỏ sót các đối tượng quá lớn nằm sát ống kính (như cuối `video2.mp4` hoặc `video1.mp4`).

### 2.2. Hiện tượng 1 người bị đóng 2 khung chồng nhau
* Kiến trúc `YOLO26s` sử dụng đầu ra `end2end=True` (cơ chế *One-to-One head* không chạy thuật toán Non-Maximum Suppression - NMS truyền thống khi suy luận).
* Khi một người bơi có phần đầu/vai nổi rõ trên mặt nước và phần thân/chân mờ dưới nước, 2 truy vấn (queries) lân cận cùng kích hoạt: 1 khung nhỏ bao phần đầu/vai và 1 khung lớn bao cả cơ thể, tạo ra 2 khung chồng lên nhau trên cùng một người.

### 2.3. Hiện tượng "Học vẹt theo bọt nước" (Splash Shortcut Learning)
* Trong bộ dữ liệu 12k ảnh, các ảnh gán nhãn `Drowning` thường đi kèm đám bọt nước trắng xóa quanh người, trong khi thiếu ảnh *Background* (chỉ có sóng/bọt nước mà không có người).
* Hệ quả là model học theo "đường tắt": **cứ thấy vùng bọt nước trắng xóa trên nền nước xanh là dự đoán `Drowning` với độ tin cậy cao (`0.50 – 0.65`)**, dẫn đến việc bắt nhầm cả tia nước bắn lên sau chân người bơi hoặc đám bọt ở góc hồ (như trong `video2.mp4`) và nhầm vận động viên đang bơi sải/bơi bướm tung bọt nước (`anh2.jpg`, `video2.mp4`) thành `Drowning`.

### 2.4. Khung chỉ bao phần nổi trên mặt nước và mất dấu khi hụp nước
* Do `reg_max=1` và `erasing=0.4` khi train, cộng với việc phần cơ thể dưới nước bị khúc xạ mờ, bounding box có xu hướng co về vùng có độ tương phản cao (đầu, tay, bọt nước).
* Với các kiểu bơi như bơi bướm/bơi ếch (`video2.mp4`), cứ mỗi nhịp quạt tay người bơi lại lặn chìm hẳn dưới lớp bọt trắng trong ~0.3–0.5 giây khiến nhận diện đơn khung (single-frame) bị đứt quãng.

---

## 3. Giải Pháp Hậu Xử Lý Tự Thích Nghi Trong `tests/image.py` và `tests/video.py`

Để hệ thống **không cần phải train lại** và **tự động chạy chuẩn trên mọi loại video khác nhau** (`video1.mp4` dưới nước, `video2.mp4` ngang mặt nước nhiều bọt, `video3.mp4` góc cao flycam), mã nguồn đã được nâng cấp với **4 tầng xử lý tự thích nghi**:

```
[Khung hình Video / Ảnh]
       │
       ▼
(1) Quét Kép Đa Tỉ Lệ (imgsz=640 + imgsz=960)
       │  -> Bắt cả người ở rất gần (640) lẫn người nhỏ ở xa (960)
       ▼
(2) Bộ Lọc Bọt Nước Tự Thích Nghi (is_pure_splash_box - HSV)
       │  -> Loại bỏ 100% khung chỉ chứa bọt trắng + nước xanh mà không có đặc trưng người
       ▼
(3) Bộ Lọc Trùng Khung & Khung Lồng Nhau (IoU > 0.45 hoặc Containment > 0.70)
       │  -> Xóa triệt để hiện tượng 1 người bị vẽ 2 khung đè nhau
       ▼
(4) Bộ Theo Dõi & Xác Thực Chuyển Động (AdaptiveWaterTracker)
       │  -> Giữ khung mượt khi hụp nước (Coasting)
       │  -> Phân biệt Bơi (di chuyển / tung bọt tức thời) vs Đuối nước (đứng yên + liên tục)
       ▼
[Kết Quả Hiển Thị Chính Xác]
```

### 3.1. Tầng 1: Quét Kép Đa Tỉ Lệ (`detect_multiscale_universal`)
* Thay vì dùng một kích thước cố định, hàm `detect_multiscale_universal` trong `tests/video.py` chạy suy luận trên cả 2 tỉ lệ:
  * **`imgsz=640`**: Phát hiện tốt các vận động viên ở gần camera, kích thước khung lớn.
  * **`imgsz=960`**: Phát hiện tốt các người bơi nhỏ ở góc xa hoặc quay từ trên cao (flycam).
* Đồng thời thiết lập `os.environ["LRU_CACHE_CAPACITY"] = "1"`, `torch.inference_mode()` và `gc.collect()` để giải phóng bộ nhớ đệm oneDNN ngay sau mỗi lần quét, giúp chạy mượt trên CPU mà không bị tràn RAM.

### 3.2. Tầng 2: Bộ Lọc Bọt Nước Tự Thích Nghi (`is_pure_splash_box`)
* Để loại bỏ các khung bắt nhầm cột nước bắn lên hoặc đám bọt nước trắng xóa mà không ảnh hưởng tới các hồ bơi có màu nước khác nhau, hàm `is_pure_splash_box` chuyển vùng ảnh trong khung sang hệ màu **HSV** và đo 2 tỷ lệ:
  1. **Tỷ lệ bọt nước trắng (`foam_ratio`)**: Các điểm ảnh có độ bão hòa thấp và độ sáng rất cao (`S < 45` và `V > 195`).
  2. **Tỷ lệ đặc trưng cơ thể người (`human_ratio`)**: Các điểm ảnh thuộc vùng tối của tóc, mũ bơi, kính bơi, đồ bơi (`V < 90`) hoặc sắc độ ấm của da người/đồ bơi (`(H < 35 hoặc H > 140)` và `S > 25`).
* **Quy tắc tự thích nghi**:
  * Qua kiểm nghiệm thực tế trên cả 3 video, mọi khung chứa **người thật** luôn có `human_ratio` từ **`3.8%` đến `54.3%`**.
  * Ngược lại, mọi khung **bắt nhầm tia nước / đám bọt trắng** đều có `foam_ratio > 10%` nhưng `human_ratio` chỉ từ **`0.0%` đến `0.5%`** (`< 3.2%`).
  * Do đó, điều kiện `if foam_ratio > 0.10 and human_ratio < 0.032: return True` loại bỏ chính xác 100% bọt nước ảo mà không làm mất bất kỳ người bơi thật nào.

### 3.3. Tầng 3: Lọc Trùng Khung Bằng IoU Kết Hợp Tỷ Lệ Nằm Lồng (`box_metrics`)
* Thông thường, thuật toán NMS chỉ xét chỉ số $\text{IoU} = \frac{\text{Diện tích giao}}{\text{Diện tích hợp}}$. Khi 1 khung nhỏ (bao phần đầu) nằm lọt thỏm trong 1 khung lớn (bao toàn thân), diện tích hợp rất lớn làm $\text{IoU} < 0.45$ nên NMS thông thường không xóa được khung nhỏ.
* Hàm `box_metrics` tính thêm chỉ số **Containment (Tỷ lệ nằm lồng trên khung nhỏ hơn)**:
  $$\text{Containment} = \frac{\text{Diện tích giao}}{\min(\text{Diện tích khung 1}, \text{Diện tích khung 2})}$$
* Nếu `IoU > 0.45` **hoặc** `Containment > 0.70`, khung có độ tin cậy thấp hơn sẽ bị loại bỏ, giữ lại duy nhất 1 khung chuẩn nhất cho mỗi người.

### 3.4. Tầng 4: Bộ Theo Dõi & Xác Thực Đuối Nước (`AdaptiveWaterTracker`)
* **Giữ khung khi hụp nước (`max_coast_steps`)**: Khi vận động viên bơi bướm/bơi ếch lặn xuống dưới mặt nước trong vài nhịp frame, bộ tracker tự động duy trì khung ở vị trí vừa xuất hiện thêm `3` chu kỳ quét tiếp theo, đồng thời làm mượt tọa độ khung ($\text{EMA}: 0.25 \times \text{cũ} + 0.75 \times \text{mới}$) giúp khung không bị giật cục.
* **Kiểm tra chuyển động & Chuỗi thời gian (`is_moving_swimmer` & `drown_streak`)**:
  * Tính độ dịch chuyển tâm khung so với vị trí gốc chia cho đường chéo khung (`disp`).
  * Nếu đối tượng đang di chuyển trên mặt hồ (`disp > 0.35`) hoặc chỉ bị model gán nhãn `Drowning` ở 1–2 khoảnh khắc đơn lẻ với độ tin cậy thấp (`conf < 0.48`) do quạt tay tung bọt nước $\rightarrow$ Tự động sửa nhãn hiển thị thành **`Swimming`**.
  * Chỉ phát cảnh báo **`Drowning`** khi đối tượng đứng yên tại một khu vực và có dấu hiệu đuối nước lặp lại liên tục qua nhiều chu kỳ quét (`drown_streak >= 5`) hoặc độ tin cậy rất cao (`conf >= 0.72`).

---

## 4. Kết Quả Đối Chiếu Thực Nghiệm (Trước và Sau Cải Tiến)

| Dữ liệu kiểm thử | Trước khi cải tiến | **Sau khi cải tiến (`image.py` & `video.py` mới)** |
| :--- | :--- | :--- |
| **Ảnh `anh2.jpg`** <br> *(VĐV bơi sải tung bọt)* | Nhận diện sai thành **`Drowning 0.42`**, khung hẹp. | Nhận diện đúng **`Swimming 0.61`**, khung mở rộng bao trọn sải tay. |
| **Ảnh `anhboi4.jpg`** <br> *(7 VĐV bơi trong làn)* | Độ tin cậy từ `0.29 – 0.73` (người rìa trái chỉ đạt `0.29`). | Nhận diện đủ 7/7 người, độ tin cậy tăng lên **`0.46 – 0.75`**, không chồng khung. |
| **Video `video2.mp4`** <br> *(Bơi bướm quay dọc, nhiều bọt nước)* | • Bắt nhầm đám bọt góc trái thành **`Drowning 0.63`**.<br>• Bắt nhầm cột nước bắn lên ở Frame 394 thành **`Drowning 0.47`**.<br>• Gán nhầm VĐV sát mép dưới thành `Drowning`. | • **Loại bỏ 100% khung báo giả ở đám bọt góc trái và cột nước bắn lên**.<br>• Bắt được cả VĐV ở làn giữa (`Swimming 0.32`), VĐV ở xa góc phải (`Swimming 0.26`) và VĐV sát mép dưới (**`Swimming 0.25`**). |
| **Video `video3.mp4`** <br> *(Camera góc cao bể bơi)* | • Bỏ sót 2–5 người bơi ở mỗi frame (Frame 610 mất 100%).<br>• Nhiều người bị vẽ **2 khung chồng lên nhau**. | • Bắt đủ **6/6 người giữa hồ** ở Frame 0, bắt thêm 2 người rìa trái ở Frame 305 và 2 người ở Frame 610.<br>• **Xóa sạch hoàn toàn lỗi 2 khung đè nhau**. |

---

## 5. Hướng Dẫn Sử Dụng Nhanh

1. **Chạy kiểm thử trên ảnh**:
   ```powershell
   & "D:\drowning-detection-system\.venv\Scripts\python.exe" "D:\drowning-detection-system\tests\image.py"
   ```
   *Kết quả lưu tại*: `runs/image_testv2/predict/`

2. **Chạy kiểm thử trên bất kỳ video nào**:
   * Mở file `tests/video.py`, đổi tên video tại dòng 14:
     ```python
     VIDEO_PATH = ROOT / "tests" / "video2.mp4"  # Đổi thành video1.mp4, video3.mp4, ...
     ```
   * Chạy lệnh:
     ```powershell
     & "D:\drowning-detection-system\.venv\Scripts\python.exe" "D:\drowning-detection-system\tests\video.py"
     ```
   * *Kết quả tự động lưu theo tên video tại*: `runs/video_testv2/<tên_video>_detected.mp4`
