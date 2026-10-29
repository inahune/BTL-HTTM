"""
Script huấn luyện chuẩn cho phiên bản v3 trên Kaggle / GPU (Copy toàn bộ chạy trong 1 Cell)
TỐI ƯU ĐẶC BIỆT: BẮT HẾT 100% NGƯỜI (MAXIMIZE RECALL) + PHÂN LOẠI CHUẨN 3 TRẠNG THÁI
1. Độ phân giải cao imgsz=960 + scale=0.6 + multi_scale=0.35 -> Bắt trọn cả người rất nhỏ (flycam/camera góc rộng) lẫn người ở gần.
2. Tắt erasing (erasing=0.0 thay vì 0.4) -> Không cắt cụt khung người đang bơi và không bỏ sót người chỉ nhô đầu/vai trên mặt nước.
3. Trọng số hàm Loss cân bằng: box=9.5, cls=1.5, dfl=1.5 -> Đóng khung ôm trọn toàn thân và phân biệt rõ Swimming vs Drowning.
4. Tự động lưu thêm 'best_recall.pt' tại Epoch có chỉ số Recall (tỷ lệ bắt hết người) cao nhất bên cạnh 'best.pt' mặc định.
"""

import os
import shutil
import yaml
from ultralytics import YOLO

# 1. Tự động tìm đường dẫn dataset 'datamoi' trên Kaggle và tạo file datamoi_fixed.yaml
DATASET_CANDIDATES = [
    "/kaggle/input/datasets/huedo2211/datamoi",
    "/kaggle/input/datamoi",
]
dataset_root = next((p for p in DATASET_CANDIDATES if os.path.exists(p)), DATASET_CANDIDATES[0])

# Nếu bên trong có thư mục con (ví dụ datamoi/datamoi/train), tự động dò chính xác
if not os.path.exists(os.path.join(dataset_root, "train")):
    for sub in os.listdir(dataset_root):
        sub_path = os.path.join(dataset_root, sub)
        if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "train")):
            dataset_root = sub_path
            break

val_folder = "valid" if os.path.exists(os.path.join(dataset_root, "valid")) else "val"

fixed_yaml_path = "/kaggle/working/datamoi_fixed.yaml"
yaml_config = {
    "path": dataset_root,
    "train": "train/images",
    "val": f"{val_folder}/images",
    "test": "test/images" if os.path.exists(os.path.join(dataset_root, "test")) else f"{val_folder}/images",
    "nc": 3,
    "names": {
        0: "Drowning",
        1: "Person out of water",
        2: "Swimming",
    },
}
with open(fixed_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(yaml_config, f, sort_keys=False)

print(f"[OK] Đã tạo cấu hình dataset chuẩn tại: {fixed_yaml_path} (root: {dataset_root})")

# 2. Load trọng số từ bản v1 (đã có sẵn trên tài khoản Kaggle huedo2211 của bạn)
MODEL_CANDIDATES = [
    "/kaggle/input/models/huedo2211/data-moi-train-10epochs/pytorch/default/1/best.pt",
    "/kaggle/input/data-moi-train-10epochs/pytorch/default/1/best.pt",
]
model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), MODEL_CANDIDATES[0])
print(f"[OK] Đang nạp model v1 từ: {model_path}")
model = YOLO(model_path)

# 3. Callback tự động lưu checkpoint có RECALL (khả năng bắt hết người) cao nhất
best_recall_state = {"score": 0.0, "recall": 0.0, "map50": 0.0, "epoch": -1}


def save_best_recall_callback(trainer):
    metrics = getattr(trainer, "metrics", {}) or {}
    recall = float(metrics.get("metrics/recall(B)", 0.0))
    map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    # Ưu tiên số 1 là Recall (65%) + mAP50 (35%) để không bỏ sót bất kỳ người nào
    recall_score = 0.65 * recall + 0.35 * map50
    if recall_score > best_recall_state["score"] and getattr(trainer, "last", None) and os.path.exists(trainer.last):
        best_recall_state["score"] = recall_score
        best_recall_state["recall"] = recall
        best_recall_state["map50"] = map50
        best_recall_state["epoch"] = trainer.epoch + 1
        dst = os.path.join(os.path.dirname(str(trainer.last)), "best_recall.pt")
        shutil.copy2(str(trainer.last), dst)
        print(
            f"\n🌟 [KỶ LỤC BẮT NGƯỜI MỚI - Epoch {trainer.epoch + 1}] "
            f"Recall={recall * 100:.2f}% | mAP50={map50 * 100:.2f}% -> Đã lưu: {dst}\n"
        )


model.add_callback("on_fit_epoch_end", save_best_recall_callback)

# 4. Huấn luyện với cấu hình tối đa hóa RECALL (bắt hết người nhỏ/xa/chìm nửa người) & phân loại chuẩn
results = model.train(
    data=fixed_yaml_path,
    epochs=40,
    patience=15,
    batch=8,              # Batch=8 vừa khít VRAM GPU T4 ở độ phân giải cao imgsz=960
    imgsz=960,            # QUAN TRỌNG: Độ phân giải 960 giúp bắt hết người nhỏ/xa (flycam, góc rộng)
    multi_scale=0.35,     # Co giãn đa tỉ lệ quanh 960 để nhạy với mọi kích thước người
    scale=0.6,            # Tăng biên độ phóng to/thu nhỏ mẫu (mặc định 0.5) để bắt người cực nhỏ
    optimizer="AdamW",
    lr0=0.0005,           # Đủ lớn để thoát cực tiểu cục bộ và học thêm các ca khó
    lrf=0.01,
    cos_lr=True,          # Giảm tốc độ học mượt theo hàm Cosine
    box=9.5,              # Tăng phạt khung (mặc định 7.5) để ôm trọn toàn bộ cơ thể
    cls=1.5,              # Tăng phạt phân loại gấp 3 lần (mặc định 0.5) để hết nhầm Swimming/Drowning
    dfl=1.5,              # Giữ biên khung sắc nét cho người bị nước che khuất một phần
    erasing=0.0,          # QUAN TRỌNG: Tắt xóa ngẫu nhiên (v2 để 0.4 làm hỏng khung & sót người)
    close_mosaic=10,      # 10 epoch cuối tắt mosaic để học khung thực tế chuẩn xác
    mosaic=1.0,
    mixup=0.1,
    fliplr=0.5,
    flipud=0.0,
    project="/kaggle/working/fine_tune",
    name="yolo26s_datamoi_v3",
    exist_ok=True,
)

print(
    f"\n✅ HOÀN TẤT! Model bắt nhiều người nhất (best_recall.pt) đạt tại Epoch {best_recall_state['epoch']}: "
    f"Recall = {best_recall_state['recall'] * 100:.2f}%, mAP50 = {best_recall_state['map50'] * 100:.2f}%"
)


