"""Drawing on the camera picture: Arabic/Chinese aware text, cards, stickers."""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import cv2

import chinese

SIZE = [1280, 720]                 # canvas W, H (set by the app)
ACCENT = (255, 196, 92)            # soft blue (BGR)
GREEN, RED, ORANGE, GREY = (110, 210, 110), (80, 80, 240), (40, 160, 255), (165, 165, 172)


def rgba(c, a=255):
    return (c[2], c[1], c[0], a)


class Text:
    AR_FONTS = ["/System/Library/Fonts/GeezaPro.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    CJK_FONTS = ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
                 "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    MIXED_FONTS = ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"]
    LATIN_FONTS = ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc",
                   "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

    def __init__(self):
        from PIL import ImageFont
        self.ImageFont = ImageFont
        first = lambda fs: next((f for f in fs if Path(f).exists()), None)
        self.paths = {"ar": first(self.AR_FONTS), "cjk": first(self.CJK_FONTS), "latin": first(self.LATIN_FONTS),
                      "mixed": first(self.MIXED_FONTS)}   # Arabic + Chinese in one line
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            self._shape = lambda s: get_display(arabic_reshaper.reshape(s))
        except ImportError:
            self._shape = lambda s: s

    def font(self, size, text=""):
        cjk = re.search(r"[\u3000-\u30ff\u3400-\u9fff\uac00-\ud7af\uff00-\uffef]", text)
        rtl = self.is_rtl(text)
        script = "mixed" if (rtl and cjk and self.paths["mixed"]) else "ar" if rtl else "cjk" if cjk else "latin"
        return self._font(size, script)

    @lru_cache(maxsize=48)
    def _font(self, size, script):
        path = self.paths[script] or self.paths["latin"] or self.paths["ar"]
        if not path:
            return self.ImageFont.load_default()
        return self.ImageFont.truetype(path, size, layout_engine=self.ImageFont.Layout.BASIC)

    def shape(self, s):
        return self._shape(s) if re.search(r"[؀-ۿ]", s) else s

    def wrap(self, text, size, max_w):
        f, lines, cur = self.font(size, text), [], ""
        if re.search(r"[\u3400-\u9fff]", text) and len(text.split()) <= 2:   # Chinese: no spaces, wrap by character
            for line in text.split("\n"):
                cur = ""
                for ch in line:
                    if f.getlength(cur + ch) > max_w and cur:
                        lines.append(cur)
                        cur = ch
                    else:
                        cur += ch
                if cur:
                    lines.append(cur)
            return lines
        for word in text.split():
            trial = (cur + " " + word).strip()
            if f.getlength(self.shape(trial)) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines

    def is_rtl(self, s):
        return bool(re.search(r"[؀-ۿ]", s))

    @lru_cache(maxsize=64)
    def label(self, text, size, color):
        """Small single-line label as a BGRA array (cached)."""
        from PIL import Image, ImageDraw
        f = self.font(size, text)
        s = self.shape(text)
        w = int(f.getlength(s)) + 8
        img = Image.new("RGBA", (max(w, 1), int(size * 1.5)), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((4, 0), s, font=f, fill=(color[2], color[1], color[0], 255))
        return np.asarray(img)[:, :, [2, 1, 0, 3]].copy()


def blit(dst, bgra, x, y, alpha=1.0):
    h, w = bgra.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    src = bgra[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    a = src[..., 3:4] / 255 * alpha
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (roi * (1 - a) + src[..., :3] * a).astype(np.uint8)


def round_rect(img, x0, y0, x1, y1, r, color, alpha):
    over = img[y0:y1, x0:x1].copy()
    mask = np.zeros(over.shape[:2], np.uint8)
    cv2.rectangle(mask, (r, 0), (x1 - x0 - r, y1 - y0), 255, -1)
    cv2.rectangle(mask, (0, r), (x1 - x0, y1 - y0 - r), 255, -1)
    for cx, cy in ((r, r), (x1 - x0 - r, r), (r, y1 - y0 - r), (x1 - x0 - r, y1 - y0 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    a = (mask.astype(np.float32) / 255 * alpha)[..., None]
    img[y0:y1, x0:x1] = (over * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def render_card(text_r, kind_label, say, math_line=None, question=None):
    """Answer card as a BGRA image, built once per answer."""
    from PIL import Image, ImageDraw
    cw, pad = min(SIZE[0] - 48, 900), 26
    inner = cw - pad * 2
    say_lines = text_r.wrap(say, 34, inner)
    math_lines = text_r.wrap(math_line, 36, inner) if math_line else []
    q_lines = text_r.wrap(f"“{question}”", 26, inner) if question else []
    h = (pad + 34 + len(q_lines) * 36 + (8 if q_lines else 0) + sum(48 for _ in math_lines)
         + (10 if math_lines else 0) + len(say_lines) * 48 + pad)
    img = Image.new("RGBA", (cw, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, cw - 1, h - 1), radius=28, fill=(16, 16, 18, 205))
    rtl = text_r.is_rtl(say)
    y = pad
    lbl = text_r.shape(kind_label)
    lf = text_r.font(22, kind_label)
    lw = lf.getlength(lbl)
    d.text((cw - pad - lw if rtl else pad, y), lbl, font=lf, fill=(ACCENT[2], ACCENT[1], ACCENT[0], 255))
    y += 38
    for ln in q_lines:
        s, f = text_r.shape(ln), text_r.font(26, ln)
        x = cw - pad - f.getlength(s) if text_r.is_rtl(ln) else pad
        d.text((x, y), s, font=f, fill=(170, 170, 178, 255))
        y += 36
    if q_lines:
        y += 8
    for ln in math_lines:
        d.text((pad, y), ln, font=text_r.font(36, ln), fill=(255, 255, 255, 255))
        y += 48
    if math_lines:
        y += 10
    for ln in say_lines:
        s = text_r.shape(ln)
        f = text_r.font(34, ln)
        x = cw - pad - f.getlength(s) if rtl else pad
        d.text((x, y), s, font=f, fill=(240, 240, 240, 255))
        y += 48
    return np.asarray(img)[:, :, [2, 1, 0, 3]].copy()


def render_sticker(text_r, word, pron, ink):
    """Small label that stays stuck where you circled the object."""
    from PIL import Image, ImageDraw
    fw, fp = text_r.font(40, word), text_r.font(20, pron)
    w = int(max(fw.getlength(word), fp.getlength(pron))) + 36
    h = 92 if pron else 64
    img = Image.new("RGBA", (w, h + 14), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=16, fill=(16, 16, 18, 215), outline=(ink[2], ink[1], ink[0], 255), width=3)
    d.polygon([(w / 2 - 9, h - 2), (w / 2 + 9, h - 2), (w / 2, h + 12)], fill=(ink[2], ink[1], ink[0], 255))
    d.text(((w - fw.getlength(word)) / 2, 8), word, font=fw, fill=(255, 255, 255, 255))
    if pron:
        d.text(((w - fp.getlength(pron)) / 2, 60), pron, font=fp, fill=(ACCENT[2], ACCENT[1], ACCENT[0], 255))
    return np.asarray(img)[:, :, [2, 1, 0, 3]].copy()


# --------------------------------------------------------------------------- generic card builder
def build_card(text_r, rows, width, pad=22, bg=(16, 16, 18), alpha=218, border=None):
    """rows: list of dicts {segs: [(text, size, bgr_color)], align: center|left|right, h: line height,
    badge: bgr (draws the row as a pill)}. Returns a BGRA image."""
    from PIL import Image, ImageDraw
    heights = [r.get("h") or int(max(s[1] for s in r["segs"]) * 1.32) + r.get("space", 0) for r in rows]
    h = pad * 2 + sum(heights)
    img = Image.new("RGBA", (width, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, width - 1, h - 1), radius=26, fill=(bg[2], bg[1], bg[0], alpha),
                        outline=rgba(border) if border else None, width=3 if border else 0)
    y = pad
    for r, rh in zip(rows, heights):
        segs = [(text_r.shape(t), text_r.font(sz, t), sz, col) for t, sz, col in r["segs"] if t]
        total = sum(f.getlength(s) for s, f, _, _ in segs)
        align = r.get("align", "center")
        x = pad if align == "left" else width - pad - total if align == "right" else (width - total) / 2
        if r.get("badge") is not None and segs:
            sz = max(s[2] for s in segs)
            d.rounded_rectangle((x - 12, y - 2, x + total + 12, y + sz * 1.25), radius=int(sz * 0.6),
                                fill=rgba(r["badge"], 235))
        for s, f, sz, col in segs:
            d.text((x, y), s, font=f, fill=rgba(col))
            x += f.getlength(s)
        y += rh
    return np.asarray(img)[:, :, [2, 1, 0, 3]].copy()


def tone_segs(word, size, tones=None):
    """Each Chinese character in its tone color."""
    tones = tones or chinese.tones(word)
    out, i = [], 0
    for ch in word:
        if chinese.has_han(ch) and i < len(tones):
            out.append((ch, size, chinese.TONE_COLORS.get(tones[i], (255, 255, 255))))
            i += 1
        else:
            out.append((ch, size, (255, 255, 255)))
    return out


def pinyin_segs(word, size):
    syl = [chinese.numbered_to_marks(s) for s in chinese.numbered(word)]
    tones = chinese.tones(word)
    out = []
    for k, (s, t) in enumerate(zip(syl, tones)):
        out.append((s + (" " if k < len(syl) - 1 else ""), size, chinese.TONE_COLORS.get(t, ACCENT)))
    return out


def _wrapped(text_r, text, size, color, inner, align="center"):
    return [{"segs": [(ln, size, color)], "align": align} for ln in text_r.wrap(text, size, inner)] if text else []


# --------------------------------------------------------------------------- the cards
def render_learn_card(text_r, c):
    """c: dict from App._make_word (word, target, verified, hsk, en, ar, measure, example...)."""
    cw, pad = min(SIZE[0] - 48, 540), 22
    inner = cw - 2 * pad
    zh = c["target"] == "zh"
    tags = [(f"{'中文' if zh else 'EN'}", 20, ACCENT)]
    if c.get("hsk"):
        tags.append((f"   HSK {c['hsk']}", 20, (230, 230, 230)))
    rows = [{"segs": tags}]
    if zh:
        rows.append({"segs": [("DICTIONARY OK" if c["verified"] else "NOT IN DICTIONARY, CHECK IT", 15, (255, 255, 255))],
                     "badge": (60, 140, 60) if c["verified"] else (30, 110, 220), "h": 30})
        rows.append({"segs": tone_segs(c["word"], 76, c.get("tones")), "h": 96})
        rows.append({"segs": pinyin_segs(c["word"], 32), "h": 46})
        if c.get("measure_phrase"):
            rows.append({"segs": [(c["measure_phrase"], 24, (230, 200, 150)), ("   " + c["measure_py"], 20, (200, 180, 140))]})
    else:
        rows.append({"segs": [(c["word"], 64, (255, 255, 255))], "h": 80})
        if c.get("pron"):
            rows.append({"segs": [(c["pron"], 28, ACCENT)]})
    if c.get("en") and zh:
        rows += _wrapped(text_r, c["en"], 24, (225, 225, 230), inner)
    if c.get("ar"):
        rows += _wrapped(text_r, c["ar"], 26, (225, 225, 230), inner)
    if c.get("example"):
        rows.append({"segs": [(" ", 8, GREY)], "h": 10})
        rows += _wrapped(text_r, c["example"], 24, (235, 235, 235), inner)
        rows += _wrapped(text_r, c.get("example_py", ""), 18, (200, 160, 220), inner)
        rows += _wrapped(text_r, c.get("example_tr", ""), 18, GREY, inner)
    alts = c.get("alts") or []
    if len(alts) > 1:
        segs = []
        for i, a in enumerate(alts):
            mark = "> " if a == c["word"] else ""
            segs.append((f"{mark}{i + 1} {a}   ", 22, ACCENT if a == c["word"] else (200, 200, 200)))
        rows.append({"segs": [(" ", 8, GREY)], "h": 10})
        rows.append({"segs": segs})
        rows.append({"segs": [("not sure? press 1 / 2 / 3", 16, GREY)]})
    border = None if not zh or c["verified"] else ORANGE
    return build_card(text_r, rows, cw, pad, border=border)


def render_tutor_card(text_r, question, zh, py, en, ar, tip):
    cw, pad = min(SIZE[0] - 48, 740), 24
    inner = cw - 2 * pad
    rows = [{"segs": [("中文老师 · TUTOR", 20, ACCENT)]}]
    rows += _wrapped(text_r, f"“{question}”", 22, GREY, inner)
    rows += [{"segs": tone_segs(ln, 44), "h": 58} for ln in text_r.wrap(zh, 44, inner)]
    rows += _wrapped(text_r, py, 24, (200, 160, 220), inner)
    rows += _wrapped(text_r, en, 22, (225, 225, 230), inner)
    rows += _wrapped(text_r, ar, 24, (225, 225, 230), inner)
    if tip:
        rows.append({"segs": [(" ", 8, GREY)], "h": 8})
        rows += _wrapped(text_r, tip, 20, (150, 210, 240), inner)
    rows.append({"segs": [("Y = practice saying it", 16, GREY)]})
    return build_card(text_r, rows, cw, pad)


def render_read_card(text_r, what, text, py, en, ar, verdict, hits, source):
    cw, pad = min(SIZE[0] - 48, 760), 24
    inner = cw - 2 * pad
    rows = [{"segs": [(f"读 READ · {what or 'text'}", 20, ACCENT), (f"   ({source})", 16, GREY)]}]
    if verdict == "haram":
        bad = "  ".join(f"{t} {m}" for t, m, lv in hits if lv == "haram")
        rows.append({"segs": [("NOT HALAL: " + bad, 22, (255, 255, 255))], "badge": (40, 40, 200), "h": 40})
    elif verdict == "doubtful":
        dbt = "  ".join(f"{t} {m}" for t, m, lv in hits)
        rows.append({"segs": [("CHECK: " + dbt, 20, (20, 20, 20))], "badge": (40, 190, 255), "h": 38})
    elif verdict == "halal":
        rows.append({"segs": [("清真 HALAL mark found", 22, (255, 255, 255))], "badge": (60, 150, 60), "h": 40})
    lines = [ln for ln in (text or "").split("\n") if ln.strip()][:6]
    for ln in lines:
        rows += _wrapped(text_r, ln, 30, (255, 255, 255), inner)
    rows += _wrapped(text_r, py, 18, (200, 160, 220), inner)
    rows.append({"segs": [(" ", 8, GREY)], "h": 8})
    rows += _wrapped(text_r, en, 22, (225, 225, 230), inner)
    rows += _wrapped(text_r, ar, 24, (225, 225, 230), inner)
    border = {"haram": RED, "doubtful": ORANGE, "halal": GREEN}.get(verdict)
    return build_card(text_r, rows, cw, pad, border=border)


def render_pron_card(text_r, word, score, rows_in, heard):
    """Pronunciation result: each target syllable over what you said, green = right, red = wrong."""
    cw, pad = min(SIZE[0] - 48, 560), 22
    rows = [{"segs": [("说一说 SAY IT", 20, ACCENT), (f"   {int(score * 100)}%", 22, GREEN if score >= 0.99 else ORANGE)]},
            {"segs": tone_segs(word, 64), "h": 80}]
    top, bottom = [], []
    for tgt, got, sound, tone in rows_in:
        col = GREEN if tone else ORANGE if sound else RED
        top.append((tgt + "  ", 30, (235, 235, 235)))
        bottom.append(((got or "—") + "  ", 30, col))
    rows.append({"segs": top})
    rows.append({"segs": bottom})
    rows.append({"segs": [(f"heard: {heard}" if heard else "heard nothing", 18, GREY)]})
    msg = ("Perfect!" if score >= 0.99 else "Sound right, check the tone (orange)" if all(r[2] for r in rows_in)
           else "Listen again (P) and try once more (Y)")
    rows.append({"segs": [(msg, 20, (230, 230, 230))]})
    return build_card(text_r, rows, cw, pad, border=GREEN if score >= 0.99 else None)


def render_challenge_card(text_r, word, score_txt, state="ask", seen=None, en=None):
    cw, pad = min(SIZE[0] - 48, 520), 22
    rows = [{"segs": [("找一找 FIND IT", 20, ACCENT), (f"   {score_txt}", 20, (220, 220, 220))]},
            {"segs": tone_segs(word, 72), "h": 90},
            {"segs": pinyin_segs(word, 28)}]
    if state == "ask":
        rows.append({"segs": [("show it in the red box", 20, GREY)]})
    elif state == "ok":
        rows.append({"segs": [(f"RIGHT! {en or ''}", 26, (255, 255, 255))], "badge": (60, 150, 60), "h": 44})
    else:
        rows.append({"segs": [(f"That's {seen or 'something else'}", 22, (255, 255, 255))], "badge": (40, 40, 200), "h": 40})
        if en:
            rows.append({"segs": [(f"{word} = {en}", 20, (225, 225, 230))]})
    border = {"ok": GREEN, "miss": RED}.get(state)
    return build_card(text_r, rows, cw, pad, border=border)


def render_box_label(text_r, zh, py, en, color):
    """Small live label for INSTANT mode boxes."""
    return build_card(text_r, [{"segs": tone_segs(zh, 30), "h": 38}, {"segs": [(py, 16, ACCENT)], "h": 20}],
                      max(110, int(text_r.font(30, zh).getlength(zh)) + 40), pad=8, alpha=200, border=color)
