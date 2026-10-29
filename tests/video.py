import os
os.environ["LRU_CACHE_CAPACITY"] = "1"

from pathlib import Path
import gc
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path(__file__).resolve().parent.parent

V3_PATH = ROOT / "models" / "yolov26-v3" / "weights" / "best.pt"
V3_RECALL_PATH = ROOT / "models" / "yolov26-v3" / "weights" / "best_recall.pt"
MODEL_PATH = V3_PATH if V3_PATH.is_file() else (ROOT / "models" / "yolov26-dataset12k-v2" / "best (3).pt")
MODEL_V1_PATH = ROOT / "models" / "yolov26-dataset12kanh" / "best.pt"
VIDEO_PATH = ROOT / "tests" / "video7.mp4"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Khong tim thay video: {VIDEO_PATH}")


def box_metrics(b1, b2):
    """Tinh IoU va ty le nam long (containment) giua 2 khung [x1, y1, x2, y2]."""
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
    """Loai bo khung chi chua bot nuoc trang + nuoc xanh ma khong co dac trung co the nguoi."""
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


def detect_keyframe(models_list, frame, conf_thres: float = 0.15):
    """
    Ket hop Ensemble Da Mo Hinh (v3 Recall + v3 Best + v1):
    - m3_recall @ 1088: Bat tron nguoi o xa va nhung nguoi nho bi bo sot.
    - m3_best @ 1088: Chuan xac bien khung va phan loai.
    - m1 @ 960: Bat nguoi boi loi goc kho va thanh be.
    """
    raw_preds = []
    for src, mdl in models_list:
        if mdl is None:
            continue
        sz = 1088 if "v3" in src else 960
        res = mdl.predict(
            source=frame,
            imgsz=sz,
            conf=max(0.12, conf_thres - 0.05),
            iou=0.45,
            agnostic_nms=True,
            augment=False,
            save=False,
            verbose=False,
        )[0]
        if res.boxes is not None:
            for box in res.boxes:
                b = box.xyxy[0].tolist()
                if not is_pure_splash_box(frame, b):
                    raw_preds.append((b, float(box.conf[0]), int(box.cls[0]), src))
        del res

    raw_preds.sort(key=lambda x: x[1], reverse=True)
    clusters = []
    for xyxy, conf, cls_id, src in raw_preds:
        matched = False
        for cl in clusters:
            iou, contain = box_metrics(xyxy, cl["xyxy"])
            if iou > 0.25 or contain > 0.40:
                if conf >= 0.18:
                    cl["xyxy"] = [
                        min(cl["xyxy"][0], xyxy[0]),
                        min(cl["xyxy"][1], xyxy[1]),
                        max(cl["xyxy"][2], xyxy[2]),
                        max(cl["xyxy"][3], xyxy[3]),
                    ]
                cl["votes"].append((cls_id, conf, src))
                matched = True
                break
        if not matched:
            clusters.append({"xyxy": list(xyxy), "votes": [(cls_id, conf, src)]})

    kept = []
    for cl in clusters:
        votes = cl["votes"]
        best_conf = max(v[1] for v in votes)
        if best_conf < conf_thres and len(votes) < 2:
            continue

        source_max = {}
        for c, cf, src in votes:
            source_max[(c, src)] = max(source_max.get((c, src), 0.0), cf)

        scores = {0: 0.0, 1: 0.0, 2: 0.0}
        drown_sources = set()
        for (c, src), cf in source_max.items():
            scores[c] += cf
            if c == 0 and cf >= 0.35:
                drown_sources.add(src)

        if scores[1] > scores[0] and scores[1] > scores[2]:
            final_cls = 1
        elif scores[0] > scores[2]:
            if scores[2] < 0.20 and scores[0] >= 0.30:
                final_cls = 0
            elif len(drown_sources) >= 2 or (scores[0] >= 0.75 and scores[0] > scores[2] * 1.50):
                final_cls = 0
            else:
                final_cls = 2
        else:
            final_cls = 2

        kept.append((cl["xyxy"], min(0.98, best_conf * 1.05), final_cls))
    return kept


class SmoothInterpolationTracker:
    """
    Ky thuat Chia Frame + Noi suy Tuyen tinh (Keyframe + Linear Box Interpolation):
    - Chi chay YOLO tren cac Keyframe cach nhau KEYFRAME_STEP frame (vd: moi 6 frame).
    - Giua Keyframe A va Keyframe B, noi suy tuyen tinh toa do [x1, y1, x2, y2] cho cac frame trung gian:
          box(k) = (1 - alpha) * box_A + alpha * box_B   (voi alpha = k / KEYFRAME_STEP)
    - Nho do video xuat ra chay dung 30 FPS goc va khung nhan dien truot muot tren tung frame,
      khong bi giat cuc hay nhay coc.
    """

    def __init__(self, max_coast_steps: int = 2, drowning_confirm_steps: int = 4):
        self.tracks = {}  # track_id -> dict(id, xyxy, conf, cls_id, drown_streak, origin_cx, origin_cy, missed)
        self.next_id = 1
        self.max_coast_steps = max_coast_steps
        self.drowning_confirm_steps = drowning_confirm_steps

    @staticmethod
    def _center_and_size(xyxy):
        cx = 0.5 * (xyxy[0] + xyxy[2])
        cy = 0.5 * (xyxy[1] + xyxy[3])
        diag = max(np.hypot(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]), 1.0)
        return cx, cy, diag

    def step_keyframe(self, detections):
        """Cap nhat danh sach track tai Keyframe moi va tra ve snapshot {track_id: track_dict}."""
        new_tracks = {}
        used_det = set()

        for tid, trk in self.tracks.items():
            best_score = 0.0
            best_idx = -1
            for i, (xyxy, conf, cls_id) in enumerate(detections):
                if i in used_det:
                    continue
                iou, contain = box_metrics(trk["xyxy"], xyxy)
                score = max(iou, contain * 0.8)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0 and best_score > 0.15:
                used_det.add(best_idx)
                xyxy, conf, raw_cls = detections[best_idx]
                # Gan lien 1:1 voi toa do phat hien moi nhat, khong dung he so keo lui gay tre khung
                smooth_xyxy = list(xyxy)

                cx, cy, diag = self._center_and_size(smooth_xyxy)
                w_box = max(smooth_xyxy[2] - smooth_xyxy[0], 1.0)
                h_box = max(smooth_xyxy[3] - smooth_xyxy[1], 1.0)
                aspect_ratio = w_box / h_box

                total_disp = np.hypot(cx - trk.get("origin_cx", cx), cy - trk.get("origin_cy", cy)) / diag
                step_disp = np.hypot(cx - trk.get("last_cx", cx), cy - trk.get("last_cy", cy)) / diag
                is_moving_swimmer = total_disp > 0.25 or step_disp > 0.06 or (aspect_ratio > 1.30 and trk.get("swim_streak", 0) > 3)

                if raw_cls == 0 and conf >= 0.48 and not is_moving_swimmer:
                    drown_streak = trk["drown_streak"] + 1
                    swim_streak = max(0, trk.get("swim_streak", 0) - 1)
                else:
                    drown_streak = max(0, trk["drown_streak"] - 1)
                    swim_streak = trk.get("swim_streak", 0) + 1

                if raw_cls == 0 and not is_moving_swimmer and (drown_streak >= self.drowning_confirm_steps or conf >= 0.85):
                    final_cls = 0
                elif raw_cls == 0:
                    final_cls = 2
                else:
                    final_cls = raw_cls

                new_tracks[tid] = {
                    "id": tid,
                    "xyxy": smooth_xyxy,
                    "conf": conf,
                    "cls_id": final_cls,
                    "drown_streak": drown_streak,
                    "swim_streak": swim_streak,
                    "origin_cx": trk["origin_cx"],
                    "origin_cy": trk["origin_cy"],
                    "last_cx": cx,
                    "last_cy": cy,
                    "missed": 0,
                }
            else:
                missed = trk["missed"] + 1
                if missed <= self.max_coast_steps and trk["conf"] >= 0.22:
                    coasted = dict(trk)
                    coasted["missed"] = missed
                    new_tracks[tid] = coasted

        for i, (xyxy, conf, raw_cls) in enumerate(detections):
            if i in used_det:
                continue
            tid = self.next_id
            self.next_id += 1
            cx, cy, _ = self._center_and_size(xyxy)
            drown_streak = 1 if (raw_cls == 0 and conf >= 0.48) else 0
            swim_streak = 1 if raw_cls == 2 else 0
            final_cls = 0 if (raw_cls == 0 and conf >= 0.85) else (2 if raw_cls == 0 else raw_cls)
            new_tracks[tid] = {
                "id": tid,
                "xyxy": list(xyxy),
                "conf": conf,
                "cls_id": final_cls,
                "drown_streak": drown_streak,
                "swim_streak": swim_streak,
                "origin_cx": cx,
                "origin_cy": cy,
                "last_cx": cx,
                "last_cy": cy,
                "missed": 0,
            }

        self.tracks = new_tracks
        return {tid: dict(t) for tid, t in self.tracks.items()}

    @staticmethod
    def interpolate(tracks_start: dict, tracks_end: dict, alpha: float):
        """
        Noi suy tuyen tinh toa do khung giua 2 Keyframe theo he so alpha in [0.0, 1.0].
        Neu 1 doi tuong co mat o ca 2 Keyframe -> khung truot muot tu vi tri A sang vi tri B.
        """
        result_boxes = []
        all_ids = set(tracks_start.keys()) | set(tracks_end.keys())

        for tid in all_ids:
            t1 = tracks_start.get(tid)
            t2 = tracks_end.get(tid)
            if t1 is not None and t2 is not None:
                xyxy = [(1.0 - alpha) * t1["xyxy"][k] + alpha * t2["xyxy"][k] for k in range(4)]
                conf = (1.0 - alpha) * t1["conf"] + alpha * t2["conf"]
                cls_id = t2["cls_id"] if alpha >= 0.5 else t1["cls_id"]
                result_boxes.append((xyxy, conf, cls_id))
            elif t1 is not None and alpha < 0.6:
                result_boxes.append((t1["xyxy"], t1["conf"], t1["cls_id"]))
            elif t2 is not None and alpha >= 0.4:
                result_boxes.append((t2["xyxy"], t2["conf"], t2["cls_id"]))

        return result_boxes


def draw_detections(frame, parsed_boxes, names):
    """Ve khung nhan dien len frame: Drowning -> ĐỎ, Swimming / Out of water -> TRẮNG."""
    annotator = Annotator(frame.copy(), line_width=2)
    for xyxy, conf, cls_id in parsed_boxes:
        color = (0, 0, 255) if cls_id == 0 else (255, 255, 255)
        txt_color = (255, 255, 255) if cls_id == 0 else (0, 0, 0)
        label = f"{names[cls_id]} {conf:.2f}"
        annotator.box_label(xyxy, label, color=color, txt_color=txt_color)
    return annotator.result()


if __name__ == "__main__":
    m3_best = YOLO(str(V3_PATH))
    m3_rec = YOLO(str(V3_RECALL_PATH)) if V3_RECALL_PATH.is_file() else None
    model_v1 = YOLO(str(MODEL_V1_PATH)) if MODEL_V1_PATH.is_file() else None
    models_list = [("v3_rec", m3_rec), ("v3_best", m3_best), ("v1", model_v1)]

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = fps if fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Giu nguyen do phan giai goc (Full HD 1920x1080) de khong lam mat nguoi nho o xa
    out_width = width
    out_height = height

    output_dir = ROOT / "runs" / "video_testv3"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{VIDEO_PATH.stem}_detected.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (out_width, out_height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Khong tao duoc video dau ra")

    # Chia theo buoc 2 frame: giup khung gan lien 1:1 voi nguoi boi, hoan toan khong bi tre
    KEYFRAME_STEP = 2
    CONF = 0.15

    frame_count = 0
    tracker = SmoothInterpolationTracker(
        max_coast_steps=3,
        drowning_confirm_steps=max(7, int(out_fps * 0.6 / KEYFRAME_STEP))
    )
    names = m3_best.names

    try:
        with torch.inference_mode():
            ret, first_frame = cap.read()
            if ret:
                dets_start = detect_keyframe(models_list, first_frame, conf_thres=CONF)
                tracks_start = tracker.step_keyframe(dets_start)

                # Viet frame dau tien
                boxes_0 = SmoothInterpolationTracker.interpolate(tracks_start, tracks_start, 0.0)
                writer.write(draw_detections(first_frame, boxes_0, names))
                frame_count += 1

                while True:
                    chunk_frames = []
                    for _ in range(KEYFRAME_STEP):
                        ok, frm = cap.read()
                        if not ok:
                            break
                        chunk_frames.append(frm)

                    if not chunk_frames:
                        break

                    # Chay Ensemble tren frame cuoi cua cum (Keyframe tiep theo)
                    last_frm = chunk_frames[-1]
                    dets_end = detect_keyframe(models_list, last_frm, conf_thres=CONF)
                    tracks_end = tracker.step_keyframe(dets_end)

                    # Noi suy tuyen tinh toa do khung 1:1 cho frame trung gian
                    n_chunk = len(chunk_frames)
                    for idx_in_chunk, frm in enumerate(chunk_frames, start=1):
                        alpha = idx_in_chunk / float(n_chunk)
                        interp_boxes = SmoothInterpolationTracker.interpolate(tracks_start, tracks_end, alpha)
                        writer.write(draw_detections(frm, interp_boxes, names))
                        frame_count += 1

                    tracks_start = tracks_end
                    gc.collect()

                    if frame_count % 60 < KEYFRAME_STEP:
                        print(f"Da xu ly muot {frame_count} frame (1:1 Tracking)")

    finally:
        cap.release()
        writer.release()

    print(f"Da xu ly xong {frame_count} frame (30 FPS + Noi suy khung muot)")
    print(f"Video ket qua: {output_path}")
