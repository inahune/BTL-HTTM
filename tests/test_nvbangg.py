
import argparse
from pathlib import Path

from ultralytics import YOLO


# Thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Đường dẫn checkpoint của nvbangg
MODEL_PATH = PROJECT_ROOT / "models" / "nvbangg" / "best.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm thử mô hình drowning detection của nvbangg"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Đường dẫn ảnh/video hoặc 0 nếu dùng webcam"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ngưỡng confidence"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đầu vào"
    )
    parser.add_argument(
        "--device",
        default="0",
        help="GPU 0 hoặc cpu"
    )

    args = parser.parse_args()

    # Kiểm tra checkpoint
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {MODEL_PATH}"
        )

    # Chuyển source thành số nếu người dùng nhập webcam 0
    source = int(args.source) if args.source.isdigit() else args.source

    # Nạp mô hình
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    # Kiểm tra tên và ID các class
    print("[INFO] Model classes:")
    print(model.names)

    # Chạy dự đoán
    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "nvbangg"),
        name="inference",
        exist_ok=True,
        stream=True,
        verbose=False
    )

    # Đọc kết quả từng ảnh/frame
    for frame_index, result in enumerate(results):
        boxes = result.boxes

        print(f"\nFrame/Image {frame_index + 1}")

        if boxes is None or len(boxes) == 0:
            print("  Không phát hiện đối tượng.")
            continue

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            class_name = result.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(
                f"  Class: {class_name} "
                f"| ID: {class_id} "
                f"| Confidence: {confidence:.3f} "
                f"| BBox: ({x1:.1f}, {y1:.1f}, "
                f"{x2:.1f}, {y2:.1f})"
            )

    print("\n[INFO] Hoàn tất kiểm thử.")
    print(
        f"[INFO] Kết quả được lưu tại: "
        f"{PROJECT_ROOT / 'runs' / 'nvbangg' / 'inference'}"
    )


if __name__ == "__main__":
    main()