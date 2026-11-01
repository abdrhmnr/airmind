"""
INSTANT mode (key I): a small open-vocabulary detector (YOLO-World, through `ultralytics`) puts a
Chinese label on everyday objects in real time. Point at one with your index finger for a moment to hear
it and save it to your notebook.

The Chinese names come from the fixed list below (checked against CC-CEDICT by the tests), never from a
model, so they can't be made up. Anything not on the list is still the vision model's job (draw a circle).

Optional and heavy (PyTorch, ~1 GB):   pip install ultralytics
First use downloads yolov8s-worldv2.pt (~25 MB) and a CLIP text model.
"""
import threading
import time

import cv2

OBJECTS = {
    "cup": "杯子", "mug": "杯子", "bottle": "瓶子", "water bottle": "保温杯", "glass": "玻璃杯", "bowl": "碗",
    "plate": "盘子", "chopsticks": "筷子", "spoon": "勺子", "fork": "叉子", "knife": "刀",
    "cell phone": "手机", "laptop": "笔记本电脑", "computer monitor": "显示器", "keyboard": "键盘",
    "computer mouse": "鼠标", "headphones": "耳机", "earphones": "耳机", "remote control": "遥控器",
    "charger": "充电器", "cable": "电线", "power bank": "充电宝", "speaker": "音箱", "camera": "相机",
    "watch": "手表", "clock": "钟", "glasses": "眼镜", "sunglasses": "墨镜", "book": "书", "notebook": "本子",
    "pen": "笔", "pencil": "铅笔", "scissors": "剪刀", "paper": "纸", "backpack": "背包", "handbag": "手提包",
    "wallet": "钱包", "key": "钥匙", "umbrella": "雨伞", "hat": "帽子", "shoe": "鞋", "sock": "袜子",
    "shirt": "衬衫", "jacket": "外套", "tie": "领带", "chair": "椅子", "table": "桌子", "desk": "书桌",
    "bed": "床", "pillow": "枕头", "blanket": "毯子", "sofa": "沙发", "lamp": "台灯", "door": "门",
    "window": "窗户", "curtain": "窗帘", "mirror": "镜子", "plant": "植物", "flower": "花", "vase": "花瓶",
    "picture frame": "相框", "box": "盒子", "bag": "袋子", "trash can": "垃圾桶", "towel": "毛巾",
    "toothbrush": "牙刷", "toothpaste": "牙膏", "soap": "肥皂", "tissue": "纸巾", "apple": "苹果",
    "banana": "香蕉", "orange": "橙子", "bread": "面包", "egg": "鸡蛋", "tea": "茶", "fan": "风扇",
    "air conditioner": "空调", "television": "电视", "refrigerator": "冰箱", "microwave": "微波炉",
    "kettle": "水壶", "teddy bear": "泰迪熊", "ball": "球", "guitar": "吉他", "person": "人",
    "hand": "手", "bicycle": "自行车", "car": "汽车",
}


class InstantDetector:
    def __init__(self, model_name="yolov8s-worldv2.pt", conf=0.25):
        self.model_name, self.conf = model_name, conf
        self.model, self.error, self.loading = None, None, False
        self.enabled = False
        self.boxes, self.boxes_t = [], 0.0          # [(x0, y0, x1, y1, en, zh, conf)]
        self._frame, self._lock = None, threading.Lock()
        self._thread = None

    @staticmethod
    def installed():
        try:
            import importlib.util
            return importlib.util.find_spec("ultralytics") is not None
        except Exception:
            return False

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled and self.model is None and not self.loading:
            self.loading = True
            threading.Thread(target=self._load, daemon=True).start()
        if self.enabled and (self._thread is None or not self._thread.is_alive()):
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        if not self.enabled:
            self.boxes = []
        return self.enabled

    def _load(self):
        try:
            from ultralytics import YOLOWorld
            m = YOLOWorld(self.model_name)
            m.set_classes(list(OBJECTS))
            self.device = "mps"
            try:
                import torch
                if not torch.backends.mps.is_available():
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"
            self.model = m
        except ImportError:
            self.error = "pip install ultralytics"
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"[:120]
        finally:
            self.loading = False

    def feed(self, frame):
        with self._lock:
            self._frame = frame

    def _loop(self):
        names = list(OBJECTS)
        while self.enabled:
            if self.model is None:
                time.sleep(0.2)
                if self.error:
                    return
                continue
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is None:
                time.sleep(0.02)
                continue
            k = 640 / max(frame.shape[:2])
            small = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            try:
                res = self.model.predict(small, conf=self.conf, verbose=False, device=self.device, imgsz=640)[0]
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"[:120]
                return
            out = []
            for b, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                en = names[int(c)]
                if en in ("person", "hand"):
                    continue
                x0, y0, x1, y1 = (v / k for v in b)
                out.append((x0, y0, x1, y1, en, OBJECTS[en], s))
            self.boxes, self.boxes_t = out, time.perf_counter()

    def pointed(self, tip):
        """The smallest detected box that contains the fingertip."""
        if tip is None:
            return None
        x, y = tip
        hits = [b for b in self.boxes if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        return min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if hits else None
