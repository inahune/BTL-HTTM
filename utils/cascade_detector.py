from ultralytics import YOLO

# Quy ước class thống nhất của BTL-HTTM
OUTPUT_NAMES = {
    0: "Drowning",
    1: "Person out of water",
    2: "Swimming",
}


def normalize_name(name):
    """Chuẩn hóa tên class để tránh khác biệt chữ hoa, dấu cách."""
    return "".join(
        char.lower()
        for char in str(name)
        if char.isalnum()
    )


def resolve_class_id(names, candidates):
    """
    Tìm class ID theo tên thực tế trong checkpoint.
    Không mặc định class ID giữa hai model giống nhau.
    """
    if isinstance(names, list):
        names = dict(enumerate(names))

    normalized = {
        normalize_name(name): int(class_id)
        for class_id, name in names.items()
    }

    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]

    raise ValueError(
        f"Không tìm thấy class {candidates}. "
        f"Checkpoint đang có names={names}"
    )


def box_metrics(box_a, box_b):
    """Tính IoU và containment của hai bounding box."""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    area_b = max((bx2 - bx1) * (by2 - by1), 1e-6)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0.0, ix2 - ix1)
        * max(0.0, iy2 - iy1)
    )

    union = area_a + area_b - intersection

    iou = intersection / max(union, 1e-6)
    containment = intersection / min(area_a, area_b)

    return iou, containment


def predict_cascade(
    frame,
    drowning_model,
    detail_model,
    conf_thres=0.22,
    imgsz=640,
):
    """
    Kết hợp:
      Model 1: Drowning / Not Drowning
      Model 2: Drowning / Person out of water / Swimming

    Output:
      [(bbox, confidence, final_class_id), ...]

    final_class_id:
      0 = Drowning
      1 = Person out of water
      2 = Swimming
    """

    # --------------------------------------------------
    # 1. Xác định class ID theo tên thực tế của checkpoint
    # --------------------------------------------------
    drown_id = resolve_class_id(
        drowning_model.names,
        ["Drowning"],
    )

    not_drown_id = resolve_class_id(
        drowning_model.names,
        ["Not Drowning"],
    )

    out_id = resolve_class_id(
        detail_model.names,
        ["Person out of water"],
    )

    swim_id = resolve_class_id(
        detail_model.names,
        ["Swimming"],
    )

    # --------------------------------------------------
    # 2. Chạy Model 1
    # --------------------------------------------------
    result_1 = drowning_model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf_thres,
        iou=0.45,
        verbose=False,
    )[0]

    # --------------------------------------------------
    # 3. Chạy Model 2
    # --------------------------------------------------
    result_2 = detail_model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf_thres,
        iou=0.45,
        verbose=False,
    )[0]

    # Lấy box của Model 2 thuộc hai class cần phân biệt
    detail_boxes = []

    if result_2.boxes is not None:
        for box in result_2.boxes:
            cls_id = int(box.cls.item())

            if cls_id not in (out_id, swim_id):
                continue

            detail_boxes.append({
                "bbox": box.xyxy[0].tolist(),
                "conf": float(box.conf.item()),
                "class_id": cls_id,
            })

    final_boxes = []
    used_detail = set()

    # --------------------------------------------------
    # 4. Xử lý kết quả Model 1
    # --------------------------------------------------
    if result_1.boxes is not None:
        for box in result_1.boxes:

            cls_id = int(box.cls.item())
            conf_1 = float(box.conf.item())
            bbox_1 = box.xyxy[0].tolist()

            # Trường hợp 1: Drowning
            if cls_id == drown_id:
                final_boxes.append(
                    (bbox_1, conf_1, 0)
                )
                continue

            # Trường hợp 2: Not Drowning
            if cls_id != not_drown_id:
                continue

            # Tìm box tương ứng trong Model 2
            best_index = -1
            best_score = 0.0

            for index, detail in enumerate(detail_boxes):

                if index in used_detail:
                    continue

                iou, containment = box_metrics(
                    bbox_1,
                    detail["bbox"]
                )

                score = max(iou, containment * 0.8)

                if score > best_score:
                    best_score = score
                    best_index = index

            # Không tìm được box tương ứng:
            # Không tự đoán Swimming hoặc Person out of water.
            if best_index < 0 or best_score < 0.40:
                continue

            matched = detail_boxes[best_index]
            used_detail.add(best_index)

            if matched["class_id"] == out_id:
                final_class = 1
            else:
                final_class = 2

            final_boxes.append(
                (
                    matched["bbox"],
                    matched["conf"],
                    final_class,
                )
            )

    return final_boxes