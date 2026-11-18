from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

MODEL_1 = ROOT / "models" / "nvbangg" / "best.pt"
MODEL_2 = ROOT / "models" / "yolov26-dataset12kanh" / "best.pt"

for name, path in [
    ("nvbangg", MODEL_1),
    ("YOLO26 v1", MODEL_2),
]:
    if not path.is_file():
        raise FileNotFoundError(path)

    model = YOLO(str(path))
    print(f"\n{name}")
    print("Path:", path)
    print("Classes:", model.names)
