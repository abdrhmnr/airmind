"""
Stickers that stay on the object: each new sticker gets an OpenCV CSRT tracker (opencv-contrib),
so when you move the cup, its 杯子 label moves with it.

To keep the camera smooth: tracking runs on a half-size picture, only the newest few stickers are
tracked, and a sticker whose object is lost fades and stays where it was last seen.
"""
import time

import cv2
import numpy as np

SCALE = 0.5
MAX_TRACKED = 4


def _make_tracker():
    for make in (getattr(cv2, "TrackerCSRT_create", None),
                 getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None),
                 getattr(cv2, "TrackerKCF_create", None),
                 getattr(getattr(cv2, "legacy", None), "TrackerKCF_create", None)):
        if make:
            try:
                return make()
            except Exception:
                continue
    return None


class Sticker:
    def __init__(self, img, box, frame):
        """img: BGRA sticker picture. box: (x, y, w, h) of the object in the frame."""
        self.img, self.box = img, [float(v) for v in box]
        self.tracker, self.lost_at, self.alive = None, None, True
        x, y, w, h = self.box
        if w >= 24 and h >= 24:
            t = _make_tracker()
            if t is not None:
                small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
                try:
                    t.init(small, tuple(int(v * SCALE) for v in (x, y, w, h)))
                    self.tracker = t
                except Exception:
                    self.tracker = None

    @property
    def anchor(self):
        x, y, w, h = self.box
        return np.array([x + w / 2, y - 4])

    @property
    def alpha(self):
        if self.lost_at is None:
            return 1.0
        return max(0.35, 1.0 - (time.perf_counter() - self.lost_at))


class StickerBoard:
    def __init__(self):
        self.items, self.n = [], 0

    def add(self, img, box, frame):
        new = Sticker(img, box, frame)
        # asked about the same spot again: the new label replaces the old one
        self.items = [s for s in self.items if np.hypot(*(s.anchor - new.anchor)) > 0.5 * max(box[2], box[3], 120)]
        self.items.append(new)
        self.items = self.items[-12:]
        for s in self.items[:-MAX_TRACKED]:                     # older ones stop moving (saves time)
            s.tracker = None

    def clear(self):
        self.items = []

    def update(self, frame):
        """Call once per camera frame (before overlays are drawn)."""
        self.n += 1
        live = [s for s in self.items if s.tracker is not None]
        if not live or self.n % 2:                              # every other frame is plenty
            return
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        for s in live:
            try:
                ok, b = s.tracker.update(small)
            except Exception:
                ok = False
            if ok:
                nb = [v / SCALE for v in b]
                # ignore sudden jumps / size explosions: that's the tracker grabbing your hand instead
                jump = np.hypot(nb[0] - s.box[0], nb[1] - s.box[1])
                if jump < 0.6 * max(s.box[2], s.box[3]) + 40 and 0.5 < nb[2] / max(s.box[2], 1) < 2:
                    s.box, s.lost_at = nb, None
                    continue
            if s.lost_at is None:
                s.lost_at = time.perf_counter()

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)
