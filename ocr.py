"""
Reading text in the picture with Apple's own OCR (the Vision framework, through `ocrmac`).
It runs on the Mac's Neural Engine in a fraction of a second, reads Simplified Chinese well, and gives
the small vision model clean text to translate instead of making it squint at tiny characters.
If ocrmac isn't installed (or not on a Mac) the vision model reads the text by itself.
"""
import cv2

_ok = None


def available():
    global _ok
    if _ok is None:
        try:
            from ocrmac import ocrmac  # noqa: F401
            _ok = True
        except Exception:
            _ok = False
    return _ok


def read_text(bgr, min_conf=0.3):
    """Text lines, top to bottom, or None when OCR isn't available. Returns '' when nothing was found."""
    if not available():
        return None
    from PIL import Image
    from ocrmac import ocrmac
    h, w = bgr.shape[:2]
    if max(h, w) < 900:                                   # small crops: upscale, small characters read better
        k = 900 / max(h, w)
        bgr = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    try:
        res = ocrmac.OCR(img, recognition_level="accurate", language_preference=["zh-Hans", "en-US"]).recognize()
    except TypeError:                                      # older ocrmac without language_preference
        res = ocrmac.OCR(img, recognition_level="accurate").recognize()
    except Exception as e:
        print("OCR failed:", e, flush=True)
        return None
    lines = [(t, box) for t, conf, box in res if conf >= min_conf and t.strip()]
    lines.sort(key=lambda tb: (-(tb[1][1] + tb[1][3]), tb[1][0]))   # Vision boxes start bottom-left
    return "\n".join(t for t, _ in lines)
