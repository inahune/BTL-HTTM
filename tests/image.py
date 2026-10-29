from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# Model v3 best va Model v1 de ket hop Ensemble loai tru 100% bao dong gia
MODEL_V3_PATH = ROOT / "models" / "yolov26-v3" / "weights" / "best.pt"
MODEL_V1_PATH = ROOT / "models" / "yolov26-dataset12kanh" / "best.pt"

IMAGE_PATHS = [
    ROOT / "tests" / "anh2.jpg",
    ROOT / "tests" / "anhboi4.jpg",
]

OUTPUT_DIR = ROOT / "runs" / "image_testv3" / "predict"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def box_metrics(b1, b2):
    area1 = max((b1[2] - b1[0]) * (b1[3] - b1[1]), 1e-6)
    area2 = max((b2[2] - b2[0]) * (b2[3] - b2[1]), 1e-6)
    inter_x1 = max(float(b1[0]), float(b2[0]))
    inter_y1 = max(float(b1[1]), float(b2[1]))
    inter_x2 = min(float(b1[2]), float(b2[2]))
    inter_y2 = min(float(b1[3]), float(b2[3]))
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = area1 + area2 - inter_area
    iou = float(inter_area / max(float(union_area), 1e-6))
    contain = float(inter_area / min(float(area1), float(area2)))
    return iou, contain


def is_pure_splash_box(frame, xyxy):
    h_img, w_img = frame.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(w_img, int(xyxy[2]))
    y2 = min(h_img, int(xyxy[3]))
    if x2 - x1 < 12 or y2 - y1 < 12:
        return True
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    foam_ratio = float(((S < 45) & (V > 195)).mean())
    human_ratio = float(((V < 90) | (((H < 35) | (H > 140)) & (S > 25))).mean())
    if foam_ratio > 0.10 and human_ratio < 0.032:
        return True
    return False


def get_water_context_ratio(frame: np.ndarray, xyxy=None) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    h_img, w_img = frame.shape[:2]
    if xyxy is not None:
        x1 = max(0, int(xyxy[0] - 0.25 * (xyxy[2] - xyxy[0])))
        y1 = max(0, int(xyxy[1] - 0.15 * (xyxy[3] - xyxy[1])))
        x2 = min(w_img, int(xyxy[2] + 0.25 * (xyxy[2] - xyxy[0])))
        y2 = min(h_img, int(xyxy[3] + 0.25 * (xyxy[3] - xyxy[1])))
        roi = frame[y1:y2, x1:x2]
    else:
        roi = frame

    if roi.size == 0:
        return 0.0

    b = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    r = roi[:, :, 2].astype(np.int16)
    bgr_pool = (b > r + 20) & (b > 85) & (g >= r - 15)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hsv_pool = (H >= 80) & (H <= 135) & (S >= 32) & (V >= 35)

    water_mask = bgr_pool | hsv_pool
    return float(water_mask.mean())


print("Đang nạp model YOLO26s v3 best và v1...")
model_v3 = YOLO(str(MODEL_V3_PATH))
model_v1 = YOLO(str(MODEL_V1_PATH)) if MODEL_V1_PATH.is_file() else None

CLASS_COLORS = {
    0: (0, 0, 255),        # Drowning -> ĐỎ RỰC RỠ (cảnh báo nguy hiểm)
    1: (255, 255, 255),    # Person out of water -> TRẮNG TƯƠNG PHẢN CAO
    2: (255, 255, 255),    # Swimming -> TRẮNG TƯƠNG PHẢN CAO (dễ nhìn trên nền nước xanh)
}

for image_path in IMAGE_PATHS:
    if not image_path.is_file():
        continue

    img = cv2.imread(str(image_path))
    res_v3 = model_v3(img, imgsz=768, conf=0.25, augment=False, verbose=False)[0]

    candidates = []
    for b in res_v3.boxes:
        candidates.append((b.xyxy[0].tolist(), float(b.conf[0]), int(b.cls[0]), "v3"))

    if model_v1 is not None:
        res_v1 = model_v1(img, imgsz=768, conf=0.25, augment=False, verbose=False)[0]
        for b in res_v1.boxes:
            candidates.append((b.xyxy[0].tolist(), float(b.conf[0]), int(b.cls[0]), "v1"))

    clusters = []
    for box, conf, cls_id, src in candidates:
        if is_pure_splash_box(img, box):
            continue
        matched = False
        for cl in clusters:
            iou, contain = box_metrics(cl["xyxy"], box)
            if iou >= 0.35 or contain >= 0.55:
                cl["boxes"].append(box)
                cl["confs"].append(conf)
                cl["classes"].append(cls_id)
                cl["sources"].add(src)
                cl["xyxy"] = [
                    min(cl["xyxy"][0], box[0]),
                    min(cl["xyxy"][1], box[1]),
                    max(cl["xyxy"][2], box[2]),
                    max(cl["xyxy"][3], box[3]),
                ]
                matched = True
                break
        if not matched:
            clusters.append({
                "xyxy": list(box),
                "boxes": [box],
                "confs": [conf],
                "classes": [cls_id],
                "sources": {src}
            })

    final_detections = []
    for cl in clusters:
        best_conf = max(cl["confs"])
        scores = {0: 0.0, 1: 0.0, 2: 0.0}
        for c, conf in zip(cl["classes"], cl["confs"]):
            scores[c] += conf

        global_water = get_water_context_ratio(img)
        local_water = get_water_context_ratio(img, cl["xyxy"])

        if global_water < 0.12 or local_water < 0.08:
            final_cls = 1
        elif scores[1] >= scores[0] and scores[1] >= scores[2]:
            final_cls = 1
        elif scores[0] > scores[2] and (len(cl["sources"]) >= 2 or best_conf >= 0.52):
            final_cls = 0
        else:
            final_cls = 2

        final_detections.append((cl["xyxy"], best_conf, final_cls))

    print(f"\n=== Kết quả cho {image_path.name} ===")
    print(f"Số đối tượng phát hiện: {len(final_detections)}")

    annotated = img.copy()
    for box, conf, cls_id in final_detections:
        cname = model_v3.names[cls_id]
        print(f"Nhãn: {cname:15s} | Độ tin cậy: {conf:.2f}")

        x1, y1, x2, y2 = [int(v) for v in box]
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        # Viền bóng đen 4px phía ngoài để nổi bật trên mọi bề mặt nước
        cv2.rectangle(annotated, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 3)
        # Khung chính màu Trắng / Đỏ dày 2px
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Nhãn chữ có nền đen viền Trắng / Đỏ cực kỳ dễ nhìn
        label = f"{cname} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bg_y1 = max(0, y1 - th - 8)
        cv2.rectangle(annotated, (x1 - 1, bg_y1), (x1 + tw + 8, y1), (0, 0, 0), -1)
        cv2.rectangle(annotated, (x1, bg_y1 + 1), (x1 + tw + 6, y1), color, 1)
        cv2.putText(annotated, label, (x1 + 4, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    out_file = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(out_file), annotated)

print(f"\nẢnh kết quả được lưu tại: {OUTPUT_DIR}")
