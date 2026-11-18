Tôi đã đối chiếu lại cấu trúc trong `BTL-HTTM.zip`, đặc biệt là `web_app/server.py`, `web_app/static/app.js`, `web_app/static/index.html` và `tests/video.py`.

Với lựa chọn của bạn, ta sẽ triển khai theo mô hình **cascade hai giai đoạn**, không phải ensemble cộng điểm như cơ chế hiện tại:

| Vai trò      | Checkpoint                             | Nhiệm vụ                                                      |
| ------------ | -------------------------------------- | ------------------------------------------------------------- |
| Model 1      | `models/nvbangg/best.pt`               | Phát hiện `Drowning` và `Not Drowning`                        |
| Model 2      | `models/yolov26-dataset12kanh/best.pt` | Phân biệt `Person out of water` và `Swimming`                 |
| Kết quả cuối | Quy ước class của BTL-HTTM             | Chỉ có 3 class: `Drowning`, `Person out of water`, `Swimming` |

**Quy tắc quan trọng:** Nếu Model 1 nhận diện `Drowning`, giữ kết quả đó. Nếu Model 1 nhận diện `Not Drowning`, chỉ nhận kết quả `Swimming` hoặc `Person out of water` khi Model 2 tìm được bounding box tương ứng. Không được lấy class ID của hai mô hình rồi cộng trực tiếp, vì cùng một ID có thể mang ý nghĩa khác nhau.

Tôi đề xuất tạo một module dùng chung để `tests/video.py` và `web_app/server.py` có cùng logic, tránh hai nơi xử lý khác nhau.

## **1. Chuẩn bị cấu trúc thư mục**

Đặt checkpoint nvbangg tại:

```
BTL-HTTM/
├── models/
│   ├── nvbangg/
│   │   └── best.pt
│   └── yolov26-dataset12kanh/
│       └── best.pt
├── utils/
│   ├── __init__.py
│   └── cascade_detector.py
├── tests/
│   └── video.py
└── web_app/
    ├── server.py
    └── static/
        ├── app.js
        └── index.html
```

Nếu chưa có thư mục `utils`, hãy tạo mới.

## **2. Tạo module&#x20;**`utils/cascade_detector.py`

Đây là phần xử lý chung của hai mô hình. Module sẽ:

1. Chạy Model 1.
2. Giữ nguyên các box `Drowning`.
3. Với box `Not Drowning`, tìm box trùng khớp từ Model 2.
4. Chuyển class của Model 2 về đúng ID của dự án.
5. Bỏ qua box `Not Drowning` không được Model 2 phân loại rõ ràng.

```
from ultralytics import YOLO# Quy ước class thống nhất của BTL-HTTMOUTPUT_NAMES = {    0: "Drowning",    1: "Person out of water",    2: "Swimming",}def normalize_name(name):    """Chuẩn hóa tên class để tránh khác biệt chữ hoa, dấu cách."""    return "".join(        char.lower()        for char in str(name)        if char.isalnum()    )def resolve_class_id(names, candidates):    """    Tìm class ID theo tên thực tế trong checkpoint.    Không mặc định class ID giữa hai model giống nhau.    """    if isinstance(names, list):        names = dict(enumerate(names))    normalized = {        normalize_name(name): int(class_id)        for class_id, name in names.items()    }    for candidate in candidates:        key = normalize_name(candidate)
```

Lưu ý: `best_score < 0.40` là ngưỡng ghép box ban đầu để thử nghiệm, không phải ngưỡng đã được chứng minh tối ưu. Bạn sẽ cần điều chỉnh bằng video kiểm thử.

## **3. Sửa&#x20;**`web_app/server.py`

Đây là nơi quan trọng nhất vì server đang xử lý cả video, webcam và cập nhật kết quả cho giao diện.

### **3.1. Khai báo đường dẫn Model 1**

Trong `MODEL_PATHS`, thêm:

```
"nvbangg_drowning": (    ROOT / "models" / "nvbangg" / "best.pt"),
```

Giữ nguyên mục:

```
"yolov26_v1": (    ROOT / "models" / "yolov26-dataset12kanh" / "best.pt"),
```

Sau phần khai báo `ROOT`, bổ sung import module dùng chung:

```
import sysif str(ROOT) not in sys.path:    sys.path.insert(0, str(ROOT))from utils.cascade_detector import (    predict_cascade,    OUTPUT_NAMES,)
```

### **3.2. Thêm preset cascade trong&#x20;**`set_models()`

Tìm hàm `set_models()` và thêm vào `PRESETS`:

```
"cascade_nvbangg_v1": [    "nvbangg_drowning",    "yolov26_v1",],
```

Sau khi xử lý `selected`, trước phần gán `self.active_models`, thêm:

```
# Nếu chọn cascade, bắt buộc sử dụng đúng hai modelif "nvbangg_drowning" in selected:    selected = [        "nvbangg_drowning",        "yolov26_v1",    ]    missing = [        key for key in selected        if key not in self.models_cache    ]    if missing:        raise RuntimeError(            f"Không thể chạy Cascade. Model chưa được nạp: {missing}"        )
```

Đồng thời, thay:

```
self.model = self.models_cache.get(selected[0])
```

bằng:

```
# Model 2 có class names theo quy ước 3 class của dự ánself.model = self.models_cache.get(    "yolov26_v1") if "nvbangg_drowning" in selected else self.models_cache.get(selected[0])
```

### **3.3. Chỉnh&#x20;**`detect_boxes_raw()`

Tìm đầu hàm:

```
def detect_boxes_raw(self, frame: np.ndarray, multiscale: bool = False):
```

Ngay sau phần khai báo hàm và trước `raw_preds = []`, thêm:

```
# ==================================================# CASCADE: nvbangg Drowning + YOLO26 v1# ==================================================if (    "nvbangg_drowning" in self.active_models    and "yolov26_v1" in self.active_models):    return predict_cascade(        frame=frame,        drowning_model=self.models_cache["nvbangg_drowning"],        detail_model=self.models_cache["yolov26_v1"],        conf_thres=self.conf_thres,        imgsz=640,    )
```

Phần ensemble cũ phía dưới giữ nguyên. Như vậy, những cấu hình ensemble cũ vẫn có thể hoạt động độc lập; cấu hình cascade sẽ đi qua nhánh xử lý riêng.

### **3.4. Sửa tên class khi hiển thị**

Trong `render_boxes_on_frame()`, hiện tại bạn có:

```
names = self.model.names
```

Thay bằng:

```
names = OUTPUT_NAMES
```

Tiếp theo, thay:

```
raw_name = names.get(cls_id, str(cls_id))
```

bằng:

```
raw_name = OUTPUT_NAMES.get(    cls_id,    f"Unknown_{cls_id}")
```

Như vậy, giao diện không còn lấy tên class từ checkpoint nvbangg, mà luôn hiển thị theo quy ước ba class của dự án.

Các phần thống kê hiện tại đã sử dụng:

- `cls_id == 0`: `Drowning`
- `cls_id == 1`: `Person out of water`
- `cls_id == 2`: `Swimming`

Vì vậy, bạn có thể giữ nguyên phần đếm số lượng và tạo `det_list`.

### **3.5. Sửa tracker để không tự đổi Drowning thành Swimming**

Đây là một điểm rất quan trọng trong mã hiện tại.

Trong `_update_tracks_from_ai()`, bạn đang có logic:

```
if raw_cls == 0:    drown_streak = trk.get("drown_streak", 0) + 1    final_cls = 2 if (is_moving_swimmer and conf < 0.65) else 0else:    drown_streak = max(0, trk.get("drown_streak", 0) - 1)    final_cls = raw_cls
```

Logic này có thể tự chuyển một box `Drowning` thành `Swimming` chỉ vì đối tượng di chuyển nhanh.

Với thiết kế mới, tôi đề nghị thay bằng:

```
if raw_cls == 0:    drown_streak = trk.get("drown_streak", 0) + 1else:    drown_streak = max(        0,        trk.get("drown_streak", 0) - 1    )# Không tự thay đổi class do chuyển độngfinal_cls = raw_cls
```

Ở phần tạo track mới, tìm:

```
final_cls = raw_cls
```

Nếu bản mã của bạn đang có logic khác để tự gán class cho track mới, hãy đảm bảo kết quả cuối cùng vẫn là:

```
final_cls = raw_cls
```

**Lý do:** Phân loại Drowning đã được giao cho Model 1. Tracker chỉ nên hỗ trợ ghép đối tượng và làm mượt bounding box, không tự suy đoán lại class bằng quy tắc vận tốc.

### **3.6. Đổi cấu hình mặc định của server**

Trong `MonitoringEngine.__init__()`, thay:

```
self.model_key = "ensemble_committee"self.model = Noneself.set_models("ensemble_committee")
```

bằng:

```
self.model_key = "cascade_nvbangg_v1"self.model = Noneself.set_models("cascade_nvbangg_v1")
```

Trong hai endpoint `/api/start` và `/api/process_frame`, thay giá trị mặc định:

```
"ensemble_committee"
```

bằng:

```
"cascade_nvbangg_v1"
```

Việc này giúp server chọn cascade nếu client không gửi cấu hình model.

## **4. Sửa&#x20;**`tests/video.py`

File này đang có hàm `detect_keyframe()` kết hợp ba model bằng cơ chế voting. Ta sẽ thay phần nhận diện bằng cùng logic cascade, đồng thời giữ lại `SmoothInterpolationTracker`.

### **4.1. Bổ sung import**

Sau phần khai báo `ROOT`, thêm:

```
import sysif str(ROOT) not in sys.path:    sys.path.insert(0, str(ROOT))from utils.cascade_detector import (    predict_cascade,    OUTPUT_NAMES,)
```

Thêm đường dẫn Model 1:

```
NVBANGG_PATH = (    ROOT / "models" / "nvbangg" / "best.pt")
```

Giữ nguyên:

```
MODEL_V1_PATH = (    ROOT / "models" / "yolov26-dataset12kanh" / "best.pt")
```

### **4.2. Thay hàm&#x20;**`detect_keyframe()`

Thay toàn bộ hàm cũ bằng:

```
def detect_keyframe(    model_drowning,    model_detail,    frame,    conf_thres=0.22,):    """    Cascade:    Model 1: Drowning / Not Drowning    Model 2: Person out of water / Swimming    """    return predict_cascade(        frame=frame,        drowning_model=model_drowning,        detail_model=model_detail,        conf_thres=conf_thres,        imgsz=640,    )
```

### **4.3. Sửa phần&#x20;**`if __name__ == "__main__":`

Thay đoạn nạp model hiện tại:

```
m3_best = YOLO(str(V3_PATH))m3_rec = YOLO(str(V3_RECALL_PATH)) if V3_RECALL_PATH.is_file() else Nonemodel_v1 = YOLO(str(MODEL_V1_PATH)) if MODEL_V1_PATH.is_file() else Nonemodels_list = [    ("v3_rec", m3_rec),    ("v3_best", m3_best),    ("v1", model_v1)]
```

bằng:

```
if not NVBANGG_PATH.is_file():    raise FileNotFoundError(        f"Không tìm thấy Model nvbangg: {NVBANGG_PATH}"    )if not MODEL_V1_PATH.is_file():    raise FileNotFoundError(        f"Không tìm thấy Model YOLO26 v1: {MODEL_V1_PATH}"    )model_drowning = YOLO(str(NVBANGG_PATH))model_detail = YOLO(str(MODEL_V1_PATH))print("[INFO] Model 1:", model_drowning.names)print("[INFO] Model 2:", model_detail.names)names = OUTPUT_NAMES
```

Ở hai vị trí gọi `detect_keyframe()` (frame đầu và keyframe tiếp theo), thay:

```
dets_start = detect_keyframe(    models_list,    first_frame,    conf_thres=CONF)
```

bằng:

```
dets_start = detect_keyframe(    model_drowning,    model_detail,    first_frame,    conf_thres=CONF)
```

Và thay:

```
dets_end = detect_keyframe(    models_list,    last_frm,    conf_thres=CONF)
```

bằng:

```
dets_end = detect_keyframe(    model_drowning,    model_detail,    last_frm,    conf_thres=CONF)
```

### **4.4. Sửa tracker trong&#x20;**`tests/video.py`

Trong `SmoothInterpolationTracker.step_keyframe()`, hiện tại có logic tự chuyển `Drowning` thành `Swimming` nếu đối tượng di chuyển.

Ở nhánh cập nhật track hiện có, thay phần quyết định `final_cls` bằng:

```
# Giữ nguyên class từ cascade detectorfinal_cls = raw_cls
```

Ở nhánh tạo track mới, thay phần:

```
final_cls = 0 if (raw_cls == 0 and conf >= 0.85) else (2 if raw_cls == 0 else raw_cls)
```

bằng:

```
final_cls = raw_cls
```

Như vậy, `SmoothInterpolationTracker` vẫn thực hiện ghép track và nội suy, nhưng không tự đổi kết quả phân loại.

`draw_detections()` hiện nhận `names` từ `m3_best.names`. Sau khi thay phần khởi tạo ở trên, biến `names` đã được gán bằng `OUTPUT_NAMES`, nên phần vẽ có thể giữ nguyên.

## **5. Sửa giao diện web**

Nếu chỉ sửa backend mà không sửa giao diện, frontend vẫn có thể gửi lựa chọn ensemble cũ. Vì vậy cần cập nhật hai file sau.

### **5.1.&#x20;**`web_app/static/index.html`

Trong danh sách cấu hình model, thêm một option mới:

```
<optgroup label="🔥 KẾT HỢP NHIỀU MODEL">
    <option value="cascade_nvbangg_v1" selected>
        Cascade: nvbangg Drowning + YOLO26 v1
    </option>

    <option value="ensemble_committee">
        Hội Đồng 3 AI (v3 Best + v3 Recall + v1)
    </option>

    <option value="ensemble_v3_rec_v1">
        Ensemble v3 Recall + v1
    </option>

    <option value="ensemble_v3_v1">
        Ensemble v3 Best + v1
    </option>
</optgroup>
```

Trong phần checkbox model, thêm:

```
<label class="ensemble-cb-item">
    <input
        type="checkbox"
        class="chk-model"
        value="nvbangg_drowning"
        checked
    />
    <span>
        <strong>nvbangg Drowning</strong>
        (Drowning / Not Drowning)
    </span>
</label>
```

Và sửa checkbox `yolov26_v1` thành:

```
<label class="ensemble-cb-item">
    <input
        type="checkbox"
        class="chk-model"
        value="yolov26_v1"
        checked
    />
    <span>
        <strong>YOLO26 v1</strong>
        (Person out of water / Swimming)
    </span>
</label>
```

Các checkbox `yolov26_v3_best`, `yolov26_v3_recall` và `yolov26_v2` nên bỏ `checked` mặc định để tránh chọn nhầm nhiều mô hình.

### **5.2.&#x20;**`web_app/static/app.js`

Tìm:

```
const ENSEMBLE_PRESETS = {
```

Thêm preset:

```
const ENSEMBLE_PRESETS = {
  cascade_nvbangg_v1: [
    "nvbangg_drowning",
    "yolov26_v1"
  ],

  ensemble_committee: [
    "yolov26_v3_best",
    "yolov26_v3_recall",
    "yolov26_v1"
  ],

  ensemble_v3_rec_v1: [
    "yolov26_v3_recall",
    "yolov26_v1"
  ],

  ensemble_v3_v1: [
    "yolov26_v3_best",
    "yolov26_v1"
  ],

  yolov26_v3_best: ["yolov26_v3_best"],
  yolov26_v3_recall: ["yolov26_v3_recall"],
  yolov26_v1: ["yolov26_v1"],
  yolov26_v2: ["yolov26_v2"],
  yolov8n: ["yolov8n"],
};
```

Trong `getSelectedModels()`, sửa giá trị mặc định:

```
function getSelectedModels() {
  const checked = Array.from(
    document.querySelectorAll(".chk-model:checked")
  ).map((cb) => cb.value);

  return checked.length > 0
    ? checked
    : ["nvbangg_drowning", "yolov26_v1"];
}
```

Khi người dùng chọn `Cascade: nvbangg Drowning + YOLO26 v1`, frontend sẽ tự tích đúng hai checkbox và gửi hai model này lên server.

## **6. Kiểm thử theo thứ tự**

Không nên sửa tất cả rồi chạy ngay web. Hãy kiểm thử từng tầng để dễ tìm lỗi.

### **Bước 1 — Kiểm tra hai checkpoint**

Tạo file `tests/check_cascade_models.py`:

```
from pathlib import Pathfrom ultralytics import YOLOROOT = Path(__file__).resolve().parents[1]MODEL_1 = ROOT / "models" / "nvbangg" / "best.pt"MODEL_2 = ROOT / "models" / "yolov26-dataset12kanh" / "best.pt"for name, path in [    ("nvbangg", MODEL_1),    ("YOLO26 v1", MODEL_2),]:    if not path.is_file():        raise FileNotFoundError(path)    model = YOLO(str(path))    print(f"\n{name}")    print("Path:", path)    print("Classes:", model.names)
```

Chạy:

```
python tests/check_cascade_models.py
```

Kết quả cần xác nhận:

```
nvbangg
Classes: {0: 'Drowning', 1: 'Not Drowning'}

YOLO26 v1
Classes: {
    0: 'Drowning',
    1: 'Person out of water',
    2: 'Swimming'
}
```

Đây là kết quả kỳ vọng theo nhãn mà ta đang sử dụng. Nếu tên class thực tế khác, hãy kiểm tra trước khi chạy cascade; module sẽ báo lỗi thay vì âm thầm gán nhầm ID.

### **Bước 2 — Chạy thử video độc lập**

```
python tests/video.py
```

Kiểm tra:

- Video đầu ra vẫn có đầy đủ frame.
- `Drowning` được lấy từ Model 1.
- `Swimming` và `Person out of water` được lấy từ Model 2.
- Không còn tình trạng tracker tự đổi `Drowning` thành `Swimming`.

### **Bước 3 — Chạy web**

Mở Terminal tại thư mục gốc:

```
python web_app/server.py
```

Truy cập địa chỉ:

```
http://localhost:8000
```

Chọn cấu hình:

```
Cascade: nvbangg Drowning + YOLO26 v1
```

Sau đó thử lần lượt:

1. Video có người đang bơi.
2. Video có người ở trên bờ.
3. Video có tình huống Drowning.
4. Video có nhiều người cùng xuất hiện.

Kiểm tra cả nhãn trên bounding box và các bộ đếm `drowning`, `swimming`, `out_of_water`.

### **Bổ sung: sửa kiểm tra đường dẫn đầu&#x20;**`tests/video.py`

Trong file hiện tại, phần đầu có:

```
if not MODEL_PATH.is_file():    raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")if not VIDEO_PATH.is_file():    raise FileNotFoundError(f"Khong tim thay video: {VIDEO_PATH}")
```

Vì sau khi chuyển sang cascade, `MODEL_PATH` không còn được sử dụng để dự đoán, hãy thay bằng:

```
for path in (NVBANGG_PATH, MODEL_V1_PATH, VIDEO_PATH):    if not path.is_file():        raise FileNotFoundError(            f"Không tìm thấy file: {path}"        )
```

Ngoài ra, trong `detect_frame()` của server, nếu file thực tế vẫn có hai dòng gọi liên tiếp `self.detect_boxes_raw(...)`, hãy xóa một dòng để tránh chạy cascade hai lần trên cùng frame.

## **7. Các điểm cần đánh giá sau khi tích hợp**

Có ba rủi ro kỹ thuật bạn cần đặc biệt chú ý:

| Rủi ro                   | Nguyên nhân                                                   | Cách kiểm tra                             |
| ------------------------ | ------------------------------------------------------------- | ----------------------------------------- |
| Bỏ sót người             | Model 1 không phát hiện thì Model 2 không có box để phân loại | Kiểm tra Recall của Model 1               |
| Không ghép được box      | Hai mô hình tạo bounding box khác nhau                        | Kiểm tra IoU/containment trên video       |
| Phân loại sai trạng thái | Model 2 nhầm Swimming với Person out of water                 | Đánh giá confusion matrix theo từng class |

**Một điểm cần cân nhắc:** Cascade hiện tại ưu tiên kết quả Drowning từ nvbangg. Đây là lựa chọn phù hợp với mục tiêu thiết kế của bạn, nhưng không đồng nghĩa mô hình đó luôn chính xác hơn YOLO26. Nếu nvbangg nhận nhầm một người đang bơi là Drowning, Model 2 sẽ không có quyền sửa kết quả đó.

Ngoài ra, cả hai mô hình đều chạy trên mỗi frame. FPS của hệ thống có thể giảm đáng kể so với chỉ chạy một model. Cần đo FPS thực tế thay vì mặc định rằng cascade sẽ nhanh hơn ensemble cũ.

## **8. Checklist xác nhận hoàn thành**

**Tiến độ kiểm tra**

0/9

Đã đặt nvbangg/best.pt đúng thư mục

Hai checkpoint nạp thành công và class names đúng

Module cascade_detector.py chạy độc lập

tests/video.py xuất video đúng 3 class

Tracker không tự thay đổi class Drowning

Server khởi động và nạp được cả hai model

Giao diện chọn được cấu hình Cascade

Webcam và video đều hiển thị đúng 3 class

Đã đo FPS và kiểm tra các trường hợp phân loại sai

**Khuyến nghị của tôi:** Trước mắt, hãy chạy thành công `tests/video.py` rồi mới chuyển sang web. Nếu cả hai nơi cùng dùng `predict_cascade()`, bạn sẽ tránh được tình trạng video kiểm thử cho kết quả khác với giao diện giám sát.

Một giới hạn cần nhớ: bản sửa trên là hướng dẫn tích hợp dựa trên mã nguồn trong ZIP, chưa phải bản patch đã được chạy kiểm thử trên máy của bạn.