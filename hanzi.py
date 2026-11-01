"""
Air calligraphy helpers.

- the model doesn't see the camera: it gets your strokes alone, black on white, centered (much easier to read)
- your strokes are compared with the real character drawn in a font -> a similarity score
"""
import cv2
import numpy as np

SIZE = 256


def _norm_points(strokes, size=SIZE, margin=0.14):
    pts = np.concatenate([np.asarray(s, np.float32) for s in strokes if len(s) > 1])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max(float((hi - lo).max()), 1.0)
    k = size * (1 - 2 * margin) / span
    off = (size - (hi - lo) * k) / 2
    return [(np.asarray(s, np.float32) - lo) * k + off for s in strokes if len(s) > 1]


def strokes_image(strokes, size=SIZE, thickness=None):
    """Black-on-white picture of the strokes, centered and scaled to fill the square."""
    img = np.full((size, size, 3), 255, np.uint8)
    th = thickness or max(4, size // 28)
    for s in _norm_points(strokes, size):
        cv2.polylines(img, [s.astype(np.int32)], False, (0, 0, 0), th, cv2.LINE_AA)
    return img


def _mask_from_strokes(strokes, size):
    m = np.zeros((size, size), np.uint8)
    for s in _norm_points(strokes, size, margin=0.08):
        cv2.polylines(m, [s.astype(np.int32)], False, 255, max(3, size // 40), cv2.LINE_AA)
    return m


def _mask_from_char(char, font_path, size):
    from PIL import Image, ImageDraw, ImageFont
    big = Image.new("L", (size * 2, size * 2), 0)
    f = ImageFont.truetype(font_path, int(size * 1.4))
    ImageDraw.Draw(big).text((size // 4, size // 8), char, font=f, fill=255)
    a = np.asarray(big)
    ys, xs = np.nonzero(a > 60)
    if len(xs) == 0:
        return None
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    k = size * 0.84 / max(h, w)
    crop = cv2.resize(crop, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    m = np.zeros((size, size), np.uint8)
    y0, x0 = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    m[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    # font strokes are thick and ours are thin: compare skeleton-ish versions
    m = (m > 100).astype(np.uint8) * 255
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, size // 64))
    return m


def similarity(strokes, char, font_path, size=128):
    """0..1: how much your air-writing overlaps the real character (both blurred a little, so wobbly is OK)."""
    if not font_path or not strokes:
        return None
    a = _mask_from_strokes(strokes, size)
    b = _mask_from_char(char, font_path, size)
    if b is None:
        return None
    k = np.ones((size // 10, size // 10), np.uint8)
    a, b = cv2.dilate(a, k) > 0, cv2.dilate(b, k) > 0
    inter = np.logical_and(a, b).sum()
    return float(2 * inter / max(1, a.sum() + b.sum()))


def char_image(char, font_path, size=160, color=(255, 255, 255)):
    """The real character as a small BGR picture on a dark background."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (size, size), (28, 28, 30))
    f = ImageFont.truetype(font_path, int(size * 0.78 / max(1, len(char)) ** 0.5))
    d = ImageDraw.Draw(img)
    w = d.textlength(char, font=f)
    d.text(((size - w) / 2, size * 0.06), char, font=f, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def strokes_thumb(strokes, size=160, ink=(80, 200, 255)):
    """Your strokes as a small BGR picture on a dark background (for the result card)."""
    img = np.full((size, size, 3), 28, np.uint8)
    for s in _norm_points(strokes, size):
        cv2.polylines(img, [s.astype(np.int32)], False, ink, max(3, size // 26), cv2.LINE_AA)
    return img


def brush(out, stroke, color, shadow=(25, 25, 25)):
    """Calligraphy look: thick when you move slowly, thin when you move fast."""
    p = np.asarray(stroke, np.float32)
    if len(p) < 2:
        return
    for a, b in zip(p[:-1], p[1:]):
        seg = float(np.linalg.norm(b - a))
        th = int(np.clip(18 - seg * 0.55, 5, 18))
        cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), shadow, th + 4, cv2.LINE_AA)
    for a, b in zip(p[:-1], p[1:]):
        seg = float(np.linalg.norm(b - a))
        th = int(np.clip(18 - seg * 0.55, 5, 18))
        cv2.line(out, tuple(a.astype(int)), tuple(b.astype(int)), color, th, cv2.LINE_AA)
