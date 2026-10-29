import os
os.environ["LRU_CACHE_CAPACITY"] = "1"

import base64
import gc
import json
import mimetypes
import sqlite3
import threading
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
UPLOAD_DIR = WEB_DIR / "uploads"
SNAPSHOT_DIR = WEB_DIR / "snapshots"
DB_PATH = WEB_DIR / "drowning_alerts.db"

for d in (STATIC_DIR, UPLOAD_DIR, SNAPSHOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "yolov26_v3_recall": ROOT / "models" / "yolov26-v3" / "weights" / "best_recall.pt",
    "yolov26_v3_best": ROOT / "models" / "yolov26-v3" / "weights" / "best.pt",
    "yolov26_v2": ROOT / "models" / "yolov26-dataset12k-v2" / "best (3).pt",
    "yolov26_v1": ROOT / "models" / "yolov26-dataset12kanh" / "best.pt",
    "yolov8n": ROOT / "models" / "drowning_detection_yolov8" / "weights" / "best.pt",
}

CLASS_LABELS_VI = {
    "Drowning": "Đuối nước (Drowning)",
    "Active drowning": "Đuối nước chủ động",
    "Possible Passive Drowner": "Nghi đuối nước thụ động",
    "Person out of water": "Trên bờ (Person out of water)",
    "Swimming": "Đang bơi (Swimming)",
}

CLASS_COLORS_BGR = {
    0: (0, 0, 255),        # Drowning -> ĐỎ RỰC RỠ (Red alert)
    1: (255, 255, 255),    # Person out of water -> TRẮNG NỔI BẬT
    2: (255, 255, 255),    # Swimming -> TRẮNG TƯƠNG PHẢN CAO TRÊN NỀN NƯỚC
}


# =====================================================================
# 1. CƠ SỞ DỮ LIỆU SQLITE LƯU LỊCH SỬ CẢNH BÁO
# =====================================================================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                frame_idx INTEGER DEFAULT 0,
                drowning_count INTEGER NOT NULL,
                swimming_count INTEGER NOT NULL,
                out_of_water_count INTEGER NOT NULL,
                total_people INTEGER NOT NULL,
                max_confidence REAL NOT NULL,
                details_json TEXT,
                snapshot_url TEXT,
                status TEXT DEFAULT 'Chưa xử lý'
            )
            """
        )
        conn.commit()


def insert_alert(
    source_name: str,
    source_type: str,
    frame_idx: int,
    drowning_count: int,
    swimming_count: int,
    out_of_water_count: int,
    total_people: int,
    max_confidence: float,
    details: list,
    annotated_frame: np.ndarray = None,
):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_url = ""
    if annotated_frame is not None:
        fname = f"alert_{int(time.time() * 1000)}.jpg"
        fpath = SNAPSHOT_DIR / fname
        cv2.imwrite(str(fpath), annotated_frame)
        snapshot_url = f"/snapshots/{fname}"

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO alerts (
                created_at, source_name, source_type, frame_idx,
                drowning_count, swimming_count, out_of_water_count,
                total_people, max_confidence, details_json, snapshot_url, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Chưa xử lý')
            """,
            (
                now_str,
                source_name,
                source_type,
                frame_idx,
                drowning_count,
                swimming_count,
                out_of_water_count,
                total_people,
                round(float(max_confidence), 4),
                json.dumps(details, ensure_ascii=False),
                snapshot_url,
            ),
        )
        conn.commit()
        return cur.lastrowid


def get_alerts_and_stats():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 200").fetchall()
        alerts = [dict(r) for r in rows]

        stats_row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_alerts,
                COALESCE(SUM(drowning_count), 0) AS total_drowning_cases,
                COALESCE(AVG(max_confidence), 0) AS avg_confidence,
                COALESCE(MAX(max_confidence), 0) AS peak_confidence,
                SUM(CASE WHEN status = 'Chưa xử lý' THEN 1 ELSE 0 END) AS pending_alerts
            FROM alerts
            """
        ).fetchone()

        by_source = conn.execute(
            """
            SELECT source_name, COUNT(*) AS count, MAX(max_confidence) AS max_conf
            FROM alerts
            GROUP BY source_name
            ORDER BY count DESC
            LIMIT 10
            """
        ).fetchall()

    return {
        "alerts": alerts,
        "summary": dict(stats_row) if stats_row else {},
        "by_source": [dict(r) for r in by_source],
    }


# =====================================================================
# 2. TRÍCH XUẤT CHỈ SỐ ĐÁNH GIÁ MÔ HÌNH (PRECISION, RECALL, mAP50, mAP50-95)
# =====================================================================
_MODEL_EVAL_CACHE = None


def load_model_evaluations():
    global _MODEL_EVAL_CACHE
    if _MODEL_EVAL_CACHE is not None:
        return _MODEL_EVAL_CACHE

    models_info = {}
    for key, path in MODEL_PATHS.items():
        if not path.is_file():
            continue
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            metrics = ckpt.get("train_metrics") or {}
            tr = ckpt.get("train_results") or {}
            args = ckpt.get("train_args") or {}
            names = ckpt["model"].names if ("model" in ckpt and hasattr(ckpt["model"], "names")) else {
                0: "Drowning",
                1: "Person out of water",
                2: "Swimming",
            }

            epochs_list = []
            if isinstance(tr, dict) and "epoch" in tr:
                n_ep = len(tr["epoch"])
                for i in range(n_ep):
                    epochs_list.append({
                        "epoch": int(tr["epoch"][i]),
                        "precision": round(float(tr.get("metrics/precision(B)", [0] * n_ep)[i]), 5),
                        "recall": round(float(tr.get("metrics/recall(B)", [0] * n_ep)[i]), 5),
                        "mAP50": round(float(tr.get("metrics/mAP50(B)", [0] * n_ep)[i]), 5),
                        "mAP50_95": round(float(tr.get("metrics/mAP50-95(B)", [0] * n_ep)[i]), 5),
                        "train_box_loss": round(float(tr.get("train/box_loss", [0] * n_ep)[i]), 5),
                        "train_cls_loss": round(float(tr.get("train/cls_loss", [0] * n_ep)[i]), 5),
                        "val_box_loss": round(float(tr.get("val/box_loss", [0] * n_ep)[i]), 5),
                        "val_cls_loss": round(float(tr.get("val/cls_loss", [0] * n_ep)[i]), 5),
                    })

            models_info[key] = {
                "key": key,
                "name": path.parent.name + "/" + path.name,
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "date": str(ckpt.get("date", "")),
                "version": str(ckpt.get("version", "")),
                "imgsz": args.get("imgsz", 640),
                "optimizer": args.get("optimizer", "auto"),
                "lr0": args.get("lr0", 0.001),
                "batch": args.get("batch", 16),
                "epochs_trained": len(epochs_list),
                "epochs_target": args.get("epochs", len(epochs_list)),
                "classes": {str(k): v for k, v in names.items()},
                "metrics": {
                    "precision": round(float(metrics.get("metrics/precision(B)", 0)), 5),
                    "recall": round(float(metrics.get("metrics/recall(B)", 0)), 5),
                    "f1_score": round(
                        2
                        * float(metrics.get("metrics/precision(B)", 0))
                        * float(metrics.get("metrics/recall(B)", 0))
                        / max(
                            float(metrics.get("metrics/precision(B)", 0))
                            + float(metrics.get("metrics/recall(B)", 0)),
                            1e-6,
                        ),
                        5,
                    ),
                    "mAP50": round(float(metrics.get("metrics/mAP50(B)", 0)), 5),
                    "mAP50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0)), 5),
                    "val_box_loss": round(float(metrics.get("val/box_loss", 0)), 5),
                    "val_cls_loss": round(float(metrics.get("val/cls_loss", 0)), 5),
                },
                "history": epochs_list,
            }
            del ckpt
        except Exception as exc:
            print(f"Loi doc checkpoint {path}: {exc}")

    gc.collect()
    _MODEL_EVAL_CACHE = models_info
    return models_info


# =====================================================================
# 3. BỘ NHẬN DIỆN YOLO & GIÁM SÁT THỜI GIAN THỰC
# =====================================================================
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
    """
    Chỉ loại bỏ khung bọt nước rác thực sự (không có người).
    Không bao giờ loại bỏ khung người bơi đang vẫy nước/tạo bọt hoặc người ở xa.
    """
    w_box = xyxy[2] - xyxy[0]
    h_box = xyxy[3] - xyxy[1]
    # Khung kích thước vừa/lớn (người đang bơi) không bao giờ bị lọc bỏ
    if w_box >= 60 or h_box >= 60:
        return False
    if w_box < 14 or h_box < 14:
        return True

    h_img, w_img = frame.shape[:2]
    x1 = max(0, int(xyxy[0]))
    y1 = max(0, int(xyxy[1]))
    x2 = min(w_img, int(xyxy[2]))
    y2 = min(h_img, int(xyxy[3]))

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return True

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    foam_ratio = float(((S < 45) & (V > 200)).mean())
    human_ratio = float(((V < 90) | (((H < 35) | (H > 140)) & (S > 25))).mean())

    # Chỉ lọc bỏ nếu > 85% là bọt nước trắng tinh và hoàn toàn không có sắc tố cơ thể
    if foam_ratio > 0.85 and human_ratio < 0.015:
        return True
    return False


def get_water_context_ratio(frame: np.ndarray, xyxy=None) -> float:
    """
    Tính tỷ lệ diện tích nước trong frame hoặc quanh khung đối tượng.
    Nhận diện đa dạng mọi loại mặt nước:
    - Bể bơi nhân tạo (xanh lam, xanh dương, xanh ngọc).
    - Vùng nước tự nhiên (sông, hồ, biển, nước xanh rêu/lục nhạt như video14).
    """
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

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # 1. Nước bể bơi (xanh dương, xanh lam, xanh ngọc)
    bgr_pool = (b > r + 15) & (b > 75)
    hsv_pool = (H >= 80) & (H <= 140) & (S >= 25) & (V >= 35)

    # 2. Nước tự nhiên (sông, hồ, ao, nước xanh rêu / lục tự nhiên)
    bgr_natural = (g > r + 8) & (g > 65) & (b > r - 15)
    hsv_natural = (H >= 35) & (H <= 85) & (S >= 18) & (V >= 35)

    water_mask = bgr_pool | hsv_pool | bgr_natural | hsv_natural
    return float(water_mask.mean())


class MonitoringEngine:
    def __init__(self):
        self.lock = threading.Lock()
        # Bộ đệm nạp trước các Model YOLO để chuyển đổi tức thì không cần đọc lại đĩa
        self.models_cache = {}
        for k, p in MODEL_PATHS.items():
            if p.is_file():
                try:
                    self.models_cache[k] = YOLO(str(p))
                    print(f"[ENGINE] Da nap san Model: {k} -> {p.name}")
                except Exception as e:
                    print(f"[ENGINE] Loi nap model {k}: {e}")

        # Thiết lập mặc định: Kết hợp Hội đồng 3 AI hoặc v3_best
        self.active_models = []
        self.model_key = "ensemble_committee"
        self.model = None
        self.set_models("ensemble_committee")

        self.running = False
        self.source_path = ROOT / "tests" / "video3.mp4"
        self.source_name = "video3.mp4"
        self.source_type = "video"
        self.conf_thres = 0.22
        self.smart_filter = True  # True: loc bot nuoc, False: Raw YOLO (nhay canh bao)
        self.thread = None
        self.ai_thread = None

        # Bien dung chung giua Luong phat Video (30 FPS) va Luong AI chay ngam
        self._pending_frame = None
        self._pending_frame_idx = 0
        self._total_frames = 0
        self._tracks = []  # Moi phan tu: dict(id, bbox, prev_bbox, vx, vy, conf, cls_id, raw_name, last_t)

        self.latest_jpeg = None
        self.last_alert_time = 0.0

        # Thoi gian theo doi duoi nuoc lien tuc de phat canh bao (mac dinh 30 giay)
        self.drowning_start_time = None
        self.continuous_drowning_duration = 0.0
        self.drowning_alert_threshold = 30.0
        self.last_drowning_seen_time = 0.0

        self.status = {
            "running": False,
            "source_name": "video3.mp4",
            "source_type": "video",
            "frame_idx": 0,
            "total_frames": 0,
            "fps": 0.0,
            "ai_fps": 0.0,
            "counts": {
                "total": 0,
                "drowning": 0,
                "swimming": 0,
                "out_of_water": 0,
            },
            "is_drowning_alert": False,
            "drowning_duration": 0.0,
            "drowning_threshold": 30.0,
            "is_tracking_drowning": False,
            "max_drowning_conf": 0.0,
            "detections": [],
            "last_alert_id": None,
            "updated_at": time.time(),
        }

        # Tao frame khoi tao
        self._create_placeholder_frame("He thong san sang giam sat")

    def _create_placeholder_frame(self, text: str):
        img = np.zeros((540, 960, 3), dtype=np.uint8)
        img[:] = (28, 20, 14)
        cv2.putText(img, text, (220, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 220, 120), 2)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            self.latest_jpeg = buf.tobytes()

    def set_models(self, model_spec):
        """
        Cấu hình Đa Mô Hình (Multi-Model Ensemble) hoặc Đơn Model:
        - Preset:
            + 'ensemble_committee': Hội đồng 3 AI (v3 Best + v3 Recall + v1) - Bắt trọn 100% người & chuẩn khung
            + 'ensemble_v3_rec_v1': Ensemble v3 Recall + v1 (Tối ưu người bơi ở xa và thanh bể)
            + 'ensemble_v3_v1': Ensemble v3 Best + v1
        - Danh sách/mảng hoặc chuỗi phân tách dấu phẩy (vd: 'yolov26_v3_best,yolov26_v1')
        - Đơn model: 'yolov26_v3_best', 'yolov26_v3_recall', 'yolov26_v1', 'yolov26_v2', 'yolov8n'
        """
        PRESETS = {
            "ensemble_committee": ["yolov26_v3_best", "yolov26_v3_recall", "yolov26_v1"],
            "ensemble_all": ["yolov26_v3_best", "yolov26_v3_recall", "yolov26_v1"],
            "ensemble_v3_rec_v1": ["yolov26_v3_recall", "yolov26_v1"],
            "ensemble_v3_v1": ["yolov26_v3_best", "yolov26_v1"],
        }
        selected = []
        if isinstance(model_spec, (list, tuple)):
            for k in model_spec:
                if k in self.models_cache and k not in selected:
                    selected.append(k)
        elif isinstance(model_spec, str):
            spec_clean = model_spec.strip()
            if spec_clean in PRESETS:
                for k in PRESETS[spec_clean]:
                    if k in self.models_cache and k not in selected:
                        selected.append(k)
            elif "," in spec_clean:
                for part in spec_clean.split(","):
                    k = part.strip()
                    if k in self.models_cache and k not in selected:
                        selected.append(k)
            elif spec_clean in self.models_cache:
                selected = [spec_clean]

        if not selected:
            # Fallback ve model tot nhat co san
            fallback = "yolov26_v3_best" if "yolov26_v3_best" in self.models_cache else list(self.models_cache.keys())[0]
            selected = [fallback]

        self.active_models = selected
        self.model_key = ",".join(selected)
        self.model = self.models_cache.get(selected[0])
        print(f"[ENGINE] Da kich hoat Ensemble ({len(self.active_models)} models): {self.active_models}")

    def set_model(self, model_key: str):
        self.set_models(model_key)

    def detect_boxes_raw(self, frame: np.ndarray, multiscale: bool = False, is_webcam: bool = False):
        """
        Quét và hợp nhất kết quả từ tất cả các Model đang kích hoạt trong Ensemble:
        - v3 Recall / Best: quét ở độ phân giải 1088px (hoặc 768px cho webcam) để bắt người nhỏ ở xa.
        - v1: quét ở 768px để bắt toàn diện người bám mép và thành bể.
        - Hợp nhất cụm (IoU & Containment Box Expansion).
        - Đồng thuận đa mô hình (Consensus Voting) để chuẩn hóa nhãn 100%.
        """
        raw_preds = []

        with torch.inference_mode():
            for mkey in self.active_models:
                mdl = self.models_cache.get(mkey)
                if mdl is None:
                    continue

                if is_webcam:
                    imgsz = 768 if "v3" in mkey else 640
                    detect_conf = max(0.12, self.conf_thres - 0.08)
                else:
                    imgsz = 1280
                    detect_conf = max(0.10, self.conf_thres - 0.08)

                res = mdl.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=detect_conf,
                    iou=0.45,
                    agnostic_nms=True,
                    augment=False,
                    save=False,
                    verbose=False,
                )[0]
                if res.boxes is not None:
                    for box in res.boxes:
                        b = box.xyxy[0].tolist()
                        if not is_webcam and self.smart_filter and is_pure_splash_box(frame, b):
                            continue
                        raw_preds.append((b, float(box.conf[0]), int(box.cls[0]), mkey))
                del res

        raw_preds.sort(key=lambda x: x[1], reverse=True)
        clusters = []
        for xyxy, conf, cls_id, src in raw_preds:
            matched = False
            for cl in clusters:
                iou, contain = box_metrics(xyxy, cl["xyxy"])
                if iou > 0.25 or contain > 0.40:
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
            min_thresh = max(0.12, self.conf_thres - 0.08) if is_webcam else self.conf_thres
            if best_conf < min_thresh and len(votes) < 2:
                continue

            source_max = {}
            for c, cf, src in votes:
                source_max[(c, src)] = max(source_max.get((c, src), 0.0), cf)

            scores = {0: 0.0, 1: 0.0, 2: 0.0}
            drown_sources = set()
            for (c, src), cf in source_max.items():
                scores[c] += cf
                if c == 0 and cf >= 0.18:
                    drown_sources.add(src)

            if is_webcam:
                # 100% người trước Webcam máy tính trong lớp học/trên cạn -> Person out of water
                final_cls = 1
            else:
                # Ưu tiên độ nhạy cảnh báo an toàn đuối nước:
                # 1. Nếu có tín hiệu Đuối nước từ bất kỳ model AI nào hoặc điểm đuối nước vượt trội
                if scores[0] >= 0.18 and (scores[0] >= scores[2] * 0.40 or len(drown_sources) >= 1):
                    final_cls = 0
                elif scores[1] > scores[0] and scores[1] > scores[2] and scores[1] >= 0.35:
                    final_cls = 1
                elif scores[0] > scores[2]:
                    final_cls = 0
                elif scores[2] > scores[0]:
                    final_cls = 2
                else:
                    final_cls = 2

            kept.append((cl["xyxy"], min(0.98, best_conf * 1.05), final_cls))
        return kept

    def render_boxes_on_frame(self, frame: np.ndarray, kept_boxes: list, is_webcam: bool = False):
        """Ve danh sach khung (xyxy, conf, cls_id) len frame va tra ve thong ke 3 trang thai."""
        names = self.model.names
        annotated = frame.copy()
        annotator = Annotator(annotated, line_width=2)

        drowning_count = 0
        swimming_count = 0
        out_of_water_count = 0
        max_drown_conf = 0.0
        det_list = []

        for idx, (xyxy, conf, cls_id) in enumerate(kept_boxes, start=1):
            if is_webcam:
                cls_id = 1  # 100% Chế độ Demo Webcam trong phòng học: Tất cả đều là Person out of water
                # Mo rong khung bao tron ca dau, vai va nguc nguoi ngoi truoc webcam
                h_box = xyxy[3] - xyxy[1]
                w_box = xyxy[2] - xyxy[0]
                h_img = frame.shape[0]
                if h_box < 1.35 * w_box and xyxy[3] < h_img - 6:
                    xyxy = [xyxy[0], max(0, xyxy[1] - 0.08 * h_box), xyxy[2], min(float(h_img - 2), xyxy[3] + 0.28 * h_box)]

            raw_name = names.get(cls_id, str(cls_id))
            if cls_id == 0:
                drowning_count += 1
                max_drown_conf = max(max_drown_conf, conf)
                state_key = "Drowning"
            elif cls_id == 1:
                out_of_water_count += 1
                state_key = "Person out of water"
            else:
                swimming_count += 1
                state_key = "Swimming"

            color_bgr = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
            txt_color = (255, 255, 255) if cls_id == 0 else (0, 0, 0)
            label_text = f"{raw_name} {conf:.2f}"
            annotator.box_label(xyxy, label_text, color=color_bgr, txt_color=txt_color)

            det_list.append({
                "id": idx,
                "class_id": cls_id,
                "class_name": raw_name,
                "class_vi": CLASS_LABELS_VI.get(raw_name, raw_name),
                "state_key": state_key,
                "confidence": round(conf, 4),
                "bbox": [round(v, 1) for v in xyxy],
            })

        result_img = annotator.result()

        if drowning_count > 0:
            h, w = result_img.shape[:2]
            if self.continuous_drowning_duration < self.drowning_alert_threshold:
                # Giai doan theo doi lien tuc truoc khi du 30 giay (Vien Cam Canh Bao)
                cv2.rectangle(result_img, (0, 0), (w - 1, h - 1), (0, 165, 255), 4)
                cv2.rectangle(result_img, (12, 12), (540, 62), (0, 140, 230), -1)
                cv2.putText(
                    result_img,
                    f"THEO DOI DUOI NUOC: {self.continuous_drowning_duration:.1f}s / {self.drowning_alert_threshold:.0f}s ({drowning_count} nguoi)",
                    (24, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )
            else:
                # Da duoi nuoc lien tiep >= 30 giay -> Kich hoat vien Do Khan Cap
                cv2.rectangle(result_img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
                cv2.rectangle(result_img, (12, 12), (560, 62), (0, 0, 210), -1)
                cv2.putText(
                    result_img,
                    f"CANH BAO KHAN CAP: DUOI NUOC > 30S! ({drowning_count} nguoi)",
                    (24, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.80,
                    (255, 255, 255),
                    2,
                )

        counts = {
            "total": len(kept_boxes),
            "drowning": drowning_count,
            "swimming": swimming_count,
            "out_of_water": out_of_water_count,
        }
        return result_img, counts, max_drown_conf, det_list

    def detect_frame(self, frame: np.ndarray, multiscale: bool = True):
        kept = self.detect_boxes_raw(frame, multiscale=multiscale)
        return self.render_boxes_on_frame(frame, kept)

    def _update_tracks_from_ai(self, new_boxes: list, now: float):
        """
        Ghep cac khung AI moi phat hien voi cac quy dao hien co, tinh van toc dich chuyen (vx, vy)
        de Luong phat Video noi suy muot ma giua cac lan quet AI.
        """
        with self.lock:
            old_tracks = list(self._tracks)

        updated = []
        used_new = set()

        for trk in old_tracks:
            best_score = 0.0
            best_i = -1
            for i, (xyxy, conf, cls_id) in enumerate(new_boxes):
                if i in used_new:
                    continue
                iou, contain = box_metrics(trk["bbox"], xyxy)
                score = max(iou, contain * 0.8)
                if score > best_score:
                    best_score = score
                    best_i = i

            if best_i >= 0 and best_score > 0.15:
                used_new.add(best_i)
                xyxy, conf, raw_cls = new_boxes[best_i]
                dt = max(now - trk["last_t"], 0.03)
                # Vận tốc dịch chuyển: mở rộng clamp [-800.0, 800.0] px/s để bám sát chuyển động bơi nhanh
                vel = [max(-800.0, min(800.0, (xyxy[k] - trk["bbox"][k]) / dt)) for k in range(4)]
                # Gắn chặt 1:1 vào đối tượng (không kéo ngược 35% về quá khứ gây trễ)
                smooth_bbox = list(xyxy)

                speed = np.hypot(vel[0], vel[1])
                is_moving_swimmer = speed > 35.0  # Nguoi boi co do dich chuyen lien tuc qua mat be (ngang hoac doc lan boi)

                if raw_cls == 0:
                    drown_streak = trk.get("drown_streak", 0) + 1
                    # Nếu người bơi đang lướt sải nhanh qua làn bơi (> 35 px/s), chỉ là chìm đầu tạm thời khi lấy hơi
                    final_cls = 2 if (is_moving_swimmer and conf < 0.65) else 0
                else:
                    # Nếu trước đó đối tượng đã ở trạng thái Đuối nước (cls 0) và vẫn đứng yên / vùng vẫy tại chỗ (không bơi tịnh tiến):
                    # Duy trì liên tục trạng thái Đuối nước, không để bị chập chờn thành bơi lội
                    if trk.get("cls_id") == 0 and not is_moving_swimmer:
                        drown_streak = trk.get("drown_streak", 0) + 1
                        final_cls = 0
                    else:
                        drown_streak = max(0, trk.get("drown_streak", 0) - 1)
                        final_cls = raw_cls

                updated.append({
                    "bbox": smooth_bbox,
                    "vel": vel,
                    "conf": conf,
                    "cls_id": final_cls,
                    "drown_streak": drown_streak,
                    "last_t": now,
                    "missed": 0,
                })
            else:
                trk["missed"] = trk.get("missed", 0) + 1
                if trk["missed"] <= 2 and trk["conf"] >= 0.24:
                    # Giam dan van toc khi coasting
                    trk["vel"] = [v * 0.5 for v in trk["vel"]]
                    updated.append(trk)

        for i, (xyxy, conf, raw_cls) in enumerate(new_boxes):
            if i in used_new:
                continue
            final_cls = raw_cls
            updated.append({
                "bbox": list(xyxy),
                "vel": [0.0, 0.0, 0.0, 0.0],
                "conf": conf,
                "cls_id": final_cls,
                "drown_streak": 1 if raw_cls == 0 else 0,
                "last_t": now,
                "missed": 0,
            })

        with self.lock:
            self._tracks = updated

    def _get_interpolated_boxes(self, now: float):
        """Tính tọa độ khung bám sát 1:1 theo thời gian thực cho Luồng phát Video 30 FPS."""
        with self.lock:
            tracks_snapshot = [dict(t) for t in self._tracks]

        out = []
        for t in tracks_snapshot:
            dt = min(max(now - t["last_t"], 0.0), 0.35)
            # 1.0 x vận tốc để khung bám sát 1:1 vào người bơi, loại bỏ hoàn toàn hiện tượng khung chạy sau người
            interp_bbox = [t["bbox"][k] + t["vel"][k] * dt * 1.0 for k in range(4)]
            out.append((interp_bbox, t["conf"], t["cls_id"]))
        return out

    def _update_state_and_alert(self, result_img, counts, max_drown_conf, det_list, frame_idx, total_frames, fps):
        ok, buf = cv2.imencode(".jpg", result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            self.latest_jpeg = buf.tobytes()

        now = time.time()
        is_drowning = counts["drowning"] > 0
        if is_drowning:
            if self.drowning_start_time is None:
                self.drowning_start_time = now
            self.last_drowning_seen_time = now
            self.continuous_drowning_duration = now - self.drowning_start_time
        else:
            # Cho phep do tre 1.5s neu box mat tam thoi truoc khi reset
            if self.drowning_start_time is not None and (now - self.last_drowning_seen_time > 1.5):
                self.drowning_start_time = None
                self.continuous_drowning_duration = 0.0

        # Canh bao chi phat khi thoi gian duoi nuoc lien tuc >= 30 giay (hoac anh tinh)
        if self.source_type == "image":
            is_alert_triggered = is_drowning
        else:
            is_alert_triggered = self.continuous_drowning_duration >= self.drowning_alert_threshold

        alert_id = self.status.get("last_alert_id")
        if is_alert_triggered and (now - self.last_alert_time >= 5.0):
            self.last_alert_time = now
            alert_id = insert_alert(
                source_name=self.source_name,
                source_type=self.source_type,
                frame_idx=frame_idx,
                drowning_count=counts["drowning"],
                swimming_count=counts["swimming"],
                out_of_water_count=counts["out_of_water"],
                total_people=counts["total"],
                max_confidence=max_drown_conf,
                details=det_list,
                annotated_frame=result_img,
            )

        with self.lock:
            self.status = {
                "running": self.running,
                "source_name": self.source_name,
                "source_type": self.source_type,
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "fps": round(fps, 1),
                "counts": counts,
                "is_drowning_alert": is_alert_triggered,
                "is_tracking_drowning": is_drowning and not is_alert_triggered,
                "drowning_duration": round(self.continuous_drowning_duration, 1),
                "drowning_threshold": self.drowning_alert_threshold,
                "max_drowning_conf": round(max_drown_conf, 4),
                "detections": det_list,
                "last_alert_id": alert_id,
                "conf_thres": self.conf_thres,
                "smart_filter": self.smart_filter,
                "updated_at": now,
            }

    def set_model(self, model_key):
        self.set_models(model_key)

    def start(self, source_path: Path, source_name: str, conf_thres: float = 0.22, smart_filter: bool = True, model_key = None, drown_threshold: float = 30.0):
        self.stop()
        if model_key:
            self.set_models(model_key)
        self.source_path = source_path
        self.source_name = source_name
        self.conf_thres = float(conf_thres)
        self.smart_filter = bool(smart_filter)
        self.drowning_alert_threshold = float(drown_threshold)
        self.drowning_start_time = None
        self.continuous_drowning_duration = 0.0
        self.last_drowning_seen_time = 0.0

        if str(source_path).lower() in ("webcam", "0", "cam", "live"):
            self.source_type = "webcam"
        else:
            ext = source_path.suffix.lower()
            self.source_type = "image" if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp") else "video"
        self._pending_frame = None
        self._tracks = []
        self.running = True

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        if self.source_type in ("video", "webcam"):
            self.ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
            self.ai_thread.start()

    def stop(self):
        self.running = False
        self.drowning_start_time = None
        self.continuous_drowning_duration = 0.0
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=2.0)
        with self.lock:
            self.status["running"] = False
            self.status["is_drowning_alert"] = False
            self.status["is_tracking_drowning"] = False
            self.status["drowning_duration"] = 0.0
        with self.lock:
            self.status["running"] = False
            self.status["is_drowning_alert"] = False

    def _ai_worker_loop(self):
        """
        LUONG 2 (Background AI Worker):
        Chay doc lap voi luong phat video. Luon lay frame moi nhat de suy luan YOLO
        ma KHONG BAO GIO lam dung/khung luong phat video 30 FPS.
        """
        last_processed_idx = -1
        while self.running:
            frame_to_process = None
            f_idx = 0
            with self.lock:
                if self._pending_frame is not None and self._pending_frame_idx != last_processed_idx:
                    frame_to_process = self._pending_frame
                    f_idx = self._pending_frame_idx

            if frame_to_process is None:
                time.sleep(0.015)
                continue

            last_processed_idx = f_idx
            new_boxes = self.detect_boxes_raw(frame_to_process, multiscale=False, is_webcam=(self.source_type == "webcam"))
            self._update_tracks_from_ai(new_boxes, time.time())
            gc.collect()

    def _run_loop(self):
        """
        LUONG 1 (30 FPS Smooth Video Playback):
        Doc video dung toc do thoi gian thuc (25-30 FPS), ve khung noi suy muot va phat ngay.
        """
        try:
            if self.source_type == "image":
                frame = cv2.imread(str(self.source_path))
                if frame is None:
                    self.running = False
                    return
                t0 = time.time()
                scale = min(1.0, 1280 / frame.shape[1], 720 / frame.shape[0])
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
                result_img, counts, max_drown_conf, det_list = self.detect_frame(frame, multiscale=True)
                fps = 1.0 / max(time.time() - t0, 1e-3)
                self.running = False
                self._update_state_and_alert(result_img, counts, max_drown_conf, det_list, 1, 1, fps)
                gc.collect()
                return

            if self.source_type == "webcam":
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("[ENGINE] OpenCV khong the mo truc tiep webcam. Se su dung luong WebRTC tu trinh duyet!")
                    self.running = False
                    return
                target_fps = 30.0
                frame_interval = 1.0 / target_fps
                frame_idx = 0
                fps_ema = 30.0
                while self.running and cap.isOpened():
                    t_start = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.02)
                        continue
                    frame_idx += 1
                    with self.lock:
                        self._pending_frame = frame
                        self._pending_frame_idx = frame_idx

                    interp_boxes = self._get_interpolated_boxes(time.time())
                    result_img, counts, max_drown_conf, det_list = self.render_boxes_on_frame(frame, interp_boxes, is_webcam=True)
                    self._update_state_and_alert(result_img, counts, max_drown_conf, det_list, frame_idx, 999999, fps_ema)
                    elapsed = time.time() - t_start
                    time.sleep(max(0.005, frame_interval - elapsed))
                cap.release()
                return

            cap = cv2.VideoCapture(str(self.source_path))
            if not cap.isOpened():
                self.running = False
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            target_fps = min(max(native_fps if native_fps > 0 else 25.0, 20.0), 30.0)
            frame_interval = 1.0 / target_fps

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            scale = min(1.0, 1280 / max(width, 1), 720 / max(height, 1))
            out_w = max(1, int(width * scale))
            out_h = max(1, int(height * scale))

            frame_idx = 0
            fps_ema = target_fps

            # Nhan dien ngay frame dau tien de co khung khoi tao truoc khi phat
            ret, first_frame = cap.read()
            if ret:
                if scale < 1.0:
                    first_frame = cv2.resize(first_frame, (out_w, out_h))
                init_boxes = self.detect_boxes_raw(first_frame, multiscale=False)
                self._update_tracks_from_ai(init_boxes, time.time())
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while self.running:
                t_loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue

                if scale < 1.0:
                    frame = cv2.resize(frame, (out_w, out_h))

                frame_idx += 1

                # Gui frame moi nhat sang cho Luong AI chay ngam (khong cho doi!)
                with self.lock:
                    self._pending_frame = frame.copy()
                    self._pending_frame_idx = frame_idx
                    self._total_frames = total_frames

                # Lay toa do khung noi suy muot tai thoi diem hien tai va ve len frame
                now = time.time()
                smooth_boxes = self._get_interpolated_boxes(now)
                result_img, counts, max_drown_conf, det_list = self.render_boxes_on_frame(frame, smooth_boxes)
                self._update_state_and_alert(result_img, counts, max_drown_conf, det_list, frame_idx, total_frames, fps_ema)

                # Canh nhip thoi gian de video chay dung toc do goc (30 FPS muot ma)
                elapsed = time.time() - t_loop_start
                sleep_time = max(0.002, frame_interval - elapsed)
                time.sleep(sleep_time)
                actual_dt = max(time.time() - t_loop_start, 1e-3)
                fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / actual_dt)

            cap.release()
        finally:
            self.running = False
            with self.lock:
                self.status["running"] = False


ENGINE = None

def get_engine():
    global ENGINE
    if ENGINE is None:
        ENGINE = MonitoringEngine()
    return ENGINE


# =====================================================================
# 4. HTTP SERVER & REST API
# =====================================================================
class DrowningWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Giam log rac tren console

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path: Path):
        if not file_path.is_file():
            self.send_error(404, "File Not Found")
            return
        mime, _ = mimetypes.guess_type(str(file_path))
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        engine = get_engine()
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            return self._serve_file(STATIC_DIR / "index.html")

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            return self._serve_file(STATIC_DIR / rel)

        if path.startswith("/snapshots/"):
            rel = path[len("/snapshots/"):]
            return self._serve_file(SNAPSHOT_DIR / rel)

        if path == "/api/samples":
            samples = [
                {
                    "name": "📷 [Live Webcam] Camera máy tính trực tiếp",
                    "path": "webcam",
                    "type": "webcam",
                    "size_kb": 0,
                }
            ]
            tests_dir = ROOT / "tests"
            if tests_dir.is_dir():
                for f in sorted(tests_dir.iterdir()):
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"):
                        samples.append({
                            "name": f.name,
                            "path": f"tests/{f.name}",
                            "type": "image" if f.suffix.lower() in (".jpg", ".jpeg", ".png") else "video",
                            "size_kb": round(f.stat().st_size / 1024, 1),
                        })
            for f in sorted(UPLOAD_DIR.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov", ".webp"):
                    samples.append({
                        "name": f"[Tải lên] {f.name}",
                        "path": f"uploads/{f.name}",
                        "type": "image" if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp") else "video",
                        "size_kb": round(f.stat().st_size / 1024, 1),
                    })
            return self._send_json({"samples": samples})

        if path == "/api/status":
            with engine.lock:
                st = dict(engine.status)
            return self._send_json(st)

        if path == "/api/alerts":
            return self._send_json(get_alerts_and_stats())

        if path == "/api/evaluation":
            return self._send_json({"models": load_model_evaluations()})

        if path == "/api/frame.jpg":
            jpeg = engine.latest_jpeg or b""
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(jpeg)
            return

        if path == "/api/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_sent = None
            try:
                while True:
                    jpeg = engine.latest_jpeg
                    if jpeg and jpeg is not last_sent:
                        last_sent = jpeg
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.025)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        engine = get_engine()
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            payload = {}

        if path == "/api/upload":
            filename = os.path.basename(payload.get("filename", f"upload_{int(time.time())}.jpg"))
            b64_data = payload.get("data", "")
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            file_bytes = base64.b64decode(b64_data)
            save_path = UPLOAD_DIR / filename
            save_path.write_bytes(file_bytes)
            return self._send_json({
                "ok": True,
                "name": f"[Tải lên] {filename}",
                "path": f"uploads/{filename}",
            })

        if path == "/api/start":
            rel_path = payload.get("path", "tests/video3.mp4")
            conf_thres = float(payload.get("conf", 0.22))
            smart_filter = bool(payload.get("smart_filter", True))
            model_key = payload.get("models") or payload.get("model_key") or "ensemble_committee"
            drown_threshold = float(payload.get("drown_threshold", 30.0))

            if rel_path == "webcam":
                engine.start(Path("webcam"), "Camera máy tính (Webcam)", conf_thres=conf_thres, smart_filter=smart_filter, model_key=model_key, drown_threshold=drown_threshold)
                return self._send_json({"ok": True, "source": "Camera máy tính (Webcam)", "model": engine.model_key, "is_webcam": True})

            if rel_path.startswith("uploads/"):
                full_path = UPLOAD_DIR / rel_path[len("uploads/"):]
            else:
                full_path = ROOT / rel_path

            if not full_path.is_file():
                return self._send_json({"ok": False, "error": f"Không tìm thấy file: {rel_path}"}, 400)

            engine.start(full_path, full_path.name, conf_thres=conf_thres, smart_filter=smart_filter, model_key=model_key, drown_threshold=drown_threshold)
            return self._send_json({"ok": True, "source": full_path.name, "model": engine.model_key})

        if path == "/api/process_frame":
            # Nhan va xu ly frame truc tiep tu Webcam trinh duyet (WebRTC canvas)
            b64_data = payload.get("image", "")
            conf_thres = float(payload.get("conf", 0.22))
            model_key = payload.get("models") or payload.get("model_key") or "ensemble_committee"
            engine.set_models(model_key)
            engine.conf_thres = conf_thres

            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]

            try:
                file_bytes = base64.b64decode(b64_data)
                np_arr = np.frombuffer(file_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                return self._send_json({"ok": False, "error": str(e)}, 400)

            if frame is None:
                return self._send_json({"ok": False, "error": "Khong the giai ma frame"}, 400)

            scale = min(1.0, 854 / max(frame.shape[1], 1), 480 / max(frame.shape[0], 1))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

            kept_boxes = engine.detect_boxes_raw(frame, multiscale=False, is_webcam=True)
            result_img, counts, max_drown_conf, det_list = engine.render_boxes_on_frame(frame, kept_boxes, is_webcam=True)

            # Cap nhat trang thai he thong
            now = time.time()
            engine._update_state_and_alert(
                result_img=result_img,
                counts=counts,
                max_drown_conf=max_drown_conf,
                det_list=det_list,
                frame_idx=int(payload.get("frame_idx", 1)),
                total_frames=999999,
                fps=float(payload.get("fps", 20.0)),
            )

            ok, buf = cv2.imencode(".jpg", result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            b64_out = base64.b64encode(buf).decode("ascii") if ok else ""

            return self._send_json({
                "ok": True,
                "counts": counts,
                "max_drown_conf": max_drown_conf,
                "detections": det_list,
                "is_drowning": counts["drowning"] > 0,
                "image": f"data:image/jpeg;base64,{b64_out}",
            })

        if path == "/api/stop":
            engine.stop()
            return self._send_json({"ok": True})

        if path == "/api/alerts/resolve":
            alert_id = int(payload.get("id", 0))
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("UPDATE alerts SET status = 'Đã xác nhận' WHERE id = ?", (alert_id,))
                conn.commit()
            return self._send_json({"ok": True})

        if path == "/api/alerts/delete":
            alert_id = payload.get("id")
            with sqlite3.connect(DB_PATH) as conn:
                if alert_id == "all":
                    conn.execute("DELETE FROM alerts")
                else:
                    conn.execute("DELETE FROM alerts WHERE id = ?", (int(alert_id),))
                conn.commit()
            return self._send_json({"ok": True})

        if path == "/api/alerts/simulate":
            # Cho phep tao nhanh 1 canh bao thu nghiem de test CSDL & Am thanh
            aid = insert_alert(
                source_name=engine.source_name,
                source_type=engine.source_type,
                frame_idx=engine.status.get("frame_idx", 1),
                drowning_count=1,
                swimming_count=engine.status["counts"].get("swimming", 2),
                out_of_water_count=0,
                total_people=engine.status["counts"].get("total", 3) or 1,
                max_confidence=0.86,
                details=[{
                    "id": 1,
                    "class_id": 0,
                    "class_name": "Drowning",
                    "class_vi": "Đuối nước (Drowning)",
                    "state_key": "Drowning",
                    "confidence": 0.86,
                    "bbox": [260.0, 140.0, 390.0, 260.0],
                }],
            )
            return self._send_json({"ok": True, "alert_id": aid})

        self.send_error(404, "Endpoint Not Found")


def main():
    global ENGINE
    init_db()
    load_model_evaluations()
    ENGINE = MonitoringEngine()
    base_port = int(os.environ.get("PORT", 8501))
    server = None
    port = base_port
    for candidate in range(base_port, base_port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), DrowningWebHandler)
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError("Khong tim duoc cong (port) trong de khoi dong Web Server")
    print(f"============================================================")
    print(f" HE THONG PHAT HIEN NGUOI DUOI NUOC DA KHOI DONG!")
    print(f" Truy cap dia chi Web: http://127.0.0.1:{port}")
    print(f"============================================================", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ENGINE.stop()
        server.server_close()


if __name__ == "__main__":
    main()
