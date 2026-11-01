"""
Taking a better picture before asking.

FrameBuffer   keeps the last ~half second of clean camera frames. When you ask, the SHARPEST one is used
              (variance of the Laplacian inside the area the model will see), not just the latest one,
              which is often blurred because your hand was moving to make the thumbs-up.

AutoCapture   "hold it still and it asks by itself": when a hand holds something inside the red box and
              keeps it still for a moment, it fires once. It fires again only when what's in the box
              changes (a new object), so holding the same cup doesn't repeat the answer.
"""
import time
from collections import deque

import cv2
import numpy as np


def sharpness(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if max(g.shape) > 320:
        k = 320 / max(g.shape)
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class FrameBuffer:
    def __init__(self, seconds=0.6, size=14):
        self.seconds, self.frames = seconds, deque(maxlen=size)

    def push(self, frame, t=None):
        self.frames.append((t or time.perf_counter(), frame))

    def best(self, rect, now=None):
        """The sharpest recent frame for this (x, y, w, h) area, and its sharpness."""
        if not self.frames:
            return None, 0.0
        now = now or time.perf_counter()
        x, y, w, h = rect
        recent = [f for t, f in self.frames if now - t <= self.seconds] or [self.frames[-1][1]]
        scored = [(sharpness(f[y:y + h, x:x + w]), f) for f in recent]
        s, f = max(scored, key=lambda sf: sf[0])
        return f, s


class AutoCapture:
    HOLD = 0.9          # seconds of stillness before it fires
    COOLDOWN = 2.5

    def __init__(self):
        self.enabled = False
        self.prev, self.ref = None, None
        self.still_since, self.last_fire = None, 0.0
        self.progress = 0.0

    def _thumb(self, frame, rect):
        x, y, w, h = rect
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None
        g = cv2.cvtColor(cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)

    def update(self, frame, rect, hand_centers, busy):
        """Returns True once when it's time to ask. `hand_centers`: palm centers of hands in view."""
        now = time.perf_counter()
        self.progress = 0.0
        if not self.enabled or rect is None:
            self.prev = None
            return False
        cur = self._thumb(frame, rect)
        if cur is None:
            return False
        moving = self.prev is None or float(np.mean(np.abs(cur - self.prev))) > 4.0
        self.prev = cur
        x, y, w, h = rect
        held = any(x <= cx <= x + w and y <= cy <= y + h for cx, cy in hand_centers)
        if busy or not held or moving or now - self.last_fire < self.COOLDOWN:
            self.still_since = None
            return False
        if self.ref is not None and float(np.mean(np.abs(cur - self.ref))) < 14.0:
            self.still_since = None                               # same thing as last time
            return False
        self.still_since = self.still_since or now
        self.progress = min(1.0, (now - self.still_since) / self.HOLD)
        if self.progress >= 1.0:
            self.ref, self.last_fire, self.still_since = cur, now, None
            return True
        return False

    def forget(self):
        """Allow the same object to be asked about again (e.g. after clearing)."""
        self.ref = None
