"""Hand tracking helpers: jitter filter, gesture reading, shape snapping, choosing the drawing hand."""
import cv2
import numpy as np


class OneEuro:
    """Jitter filter that stays responsive on fast strokes."""

    def __init__(self, min_cutoff=1.2, beta=0.012, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    @staticmethod
    def _a(cutoff, dt):
        tau = 1 / (2 * np.pi * cutoff)
        return 1 / (1 + tau / dt)

    def __call__(self, x, dt):
        x = np.asarray(x, np.float32)
        if self.x is None:
            self.x, self.dx = x, np.zeros_like(x)
            return x
        dx = (x - self.x) / dt
        self.dx = self.dx + self._a(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        self.x = self.x + self._a(cutoff, dt) * (x - self.x)
        return self.x

    def reset(self):
        self.x = self.dx = None


def read_hand(L):
    """Gesture facts from 21 landmarks in pixels."""
    palm_w = np.linalg.norm(L[5] - L[17]) + 1e-6
    palm_c = L[[0, 5, 9, 13, 17]].mean(axis=0)
    ext = [np.linalg.norm(L[t] - L[0]) > np.linalg.norm(L[p] - L[0]) * 1.12
           for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
    folded = [np.linalg.norm(L[t] - palm_c) < palm_w * 0.95 for t in (8, 12, 16, 20)]
    size = max(palm_w, 0.8 * np.linalg.norm(L[0] - L[9]))   # stays stable when the hand turns sideways
    pinch = np.linalg.norm(L[4] - L[8]) / size
    thumb_up = (L[4][1] < L[3][1] < L[2][1]) and (L[2][1] - L[4][1] > palm_w * 0.55) and L[4][1] < L[5][1] - palm_w * 0.25
    return dict(
        pinch=pinch,
        pointing=ext[0] and not any(ext[1:]),   # index finger out, the rest curled: a "pen"
        tip=L[8].copy(),
        mid=(L[4] + L[8]) / 2,
        thumbs_up=thumb_up and all(folded),
        fist=all(folded) and not thumb_up and pinch > 0.36,
        open_palm=all(ext) and pinch > 0.9,
        center=palm_c,
    )


class Shape(list):
    """A stroke that was snapped to a clean shape: drawn as-is (no smoothing, keeps sharp corners)."""
    closed = True


def snap_shape(stroke):
    """Rough circle / square / rectangle / triangle / line  ->  the clean geometric version (or None)."""
    pts = np.asarray(stroke, np.float32)
    if len(pts) < 8:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    size = max(w, h)
    if size < 40:
        return None
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    gap = float(np.linalg.norm(pts[0] - pts[-1]))

    if gap > 0.35 * size:                                   # open stroke: only a straight line gets snapped
        a, b = pts[0], pts[-1]
        d = b - a
        n = np.linalg.norm(d)
        if n < 40:
            return None
        rel = pts - a
        dist = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / n
        if dist.max() < 0.08 * n and length < 1.25 * n:
            line = Shape([a, b])
            line.closed = False
            return line
        return None

    poly = pts.reshape(-1, 1, 2)
    hull = cv2.convexHull(poly)
    hull_area = cv2.contourArea(hull)
    if hull_area < 300:
        return None
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(hull)
    fill = hull_area / max(rw * rh, 1)        # how much of its bounding box the shape fills:
    #                                           rectangle ~1.0, circle/ellipse ~0.79, triangle ~0.5
    corners = cv2.approxPolyDP(hull, 0.07 * cv2.arcLength(hull, True), True).reshape(-1, 2)

    if fill > 0.88:
        if abs(rw - rh) < 0.15 * max(rw, rh):              # nearly equal sides -> square
            rw = rh = (rw + rh) / 2
        level = 90 * round(ang / 90)                        # nearly level -> make it level
        if abs(ang - level) < 10:
            ang = level
        return Shape(list(cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.float32)))
    if 0.68 < fill <= 0.88:
        (ex, ey), (ew, eh), eang = cv2.fitEllipse(pts)
        if min(ew, eh) > 0.8 * max(ew, eh):                 # nearly round -> perfect circle
            ew = eh = (ew + eh) / 2
        ring = cv2.ellipse2Poly((int(ex), int(ey)), (int(ew / 2), int(eh / 2)), int(eang), 0, 360, 6)
        return Shape(list(ring.astype(np.float32)))
    if fill <= 0.68 and len(corners) == 3:
        return Shape(list(corners.astype(np.float32)))
    return None


def stroke_points(s):
    """Points to draw for a stroke: snapped shapes as-is (closed), freehand smoothed."""
    if isinstance(s, Shape):
        pts = np.array(s, np.int32)
        return pts, s.closed
    return chaikin(s), False


def chaikin(pts, rounds=2):
    p = np.asarray(pts, np.float32)
    for _ in range(rounds):
        if len(p) < 3:
            break
        q, r = 0.75 * p[:-1] + 0.25 * p[1:], 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], np.stack([q, r], 1).reshape(-1, 2), p[-1:]])
    return p.astype(np.int32)


def pick_hands(res, band, prefer, last):
    """Returns (drawing_hand, command_hand). The drawing hand is the one that was just drawing,
    otherwise your chosen hand (right/left); the other hand, if any, gives commands."""
    if not res.hand_landmarks:
        return None, None
    bx, by, bw, bh = band
    hands = []
    for i, lms in enumerate(res.hand_landmarks):
        pts = np.array([[bx + p.x * bw, by + p.y * bh] for p in lms], np.float32)
        cat = res.handedness[i][0] if res.handedness and res.handedness[i] else None
        # On the live (mirrored) camera MediaPipe's label comes out swapped: its "Left" is your right hand.
        side = {"left": "right", "right": "left"}.get(cat.category_name.lower(), "") if cat else ""
        hands.append((pts, side, cat.score if cat else 0.0))
    if len(hands) == 1:
        pts, side, _ = hands[0]
        if prefer == "any" or side in ("", prefer) or last is not None:
            return pts, None
        return None, pts                                    # only your other hand is up: it can still command
    if last is not None:                                    # keep following the hand we were on
        draw = min(hands, key=lambda h: np.linalg.norm(h[0][[4, 8]].mean(axis=0) - last))
    else:
        mine = [h for h in hands if h[1] == prefer]
        draw = max(mine or hands, key=lambda h: h[2])
    other = next(h for h in hands if h is not draw)
    return draw[0], other[0]
