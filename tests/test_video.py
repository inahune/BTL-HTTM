
import cv2
from pathlib import Path
from ultralytics import YOLO

# Model mới train 10 epochs
MODEL_PATH = Path("models/data12kanhtrain10epochs/best.pt")

# Video đầu vào
VIDEO_PATH = Path("data/video5.mp4")

# Video kết quả
OUTPUT_PATH = Path("runs/video5_result.mp4")

# Cấu hình
FRAME_STEP = 10
CONF = 0.25
IMGSZ = 640

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH.resolve()}")

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy video: {VIDEO_PATH.resolve()}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load model đã train, không train lại
model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Không mở được video.")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Không tạo được video đầu ra.")

frame_idx = 0
processed = 0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Chỉ chạy nhận diện mỗi 10 frame
        if frame_idx % FRAME_STEP == 0:
            result = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                verbose=False
            )[0]

            frame = result.plot()
            processed += 1

        out.write(frame)
        frame_idx += 1

finally:
    cap.release()
    out.release()

print(f"Tổng số frame: {total_frames}")
print(f"Số frame đã nhận diện: {processed}")
print(f"Video kết quả: {OUTPUT_PATH.resolve()}")
