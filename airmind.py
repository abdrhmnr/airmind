#!/usr/bin/env python3
"""
AirMind: draw in the air with your finger, a vision model running on your Mac
looks at the camera frame + your drawing and answers out loud.

  pinch (thumb + index)  -> draw
  thumbs up (hold)       -> ask the AI      (or SPACE)
  fist (hold)            -> clear           (or C)
  V key                  -> talk to it: ask any question out loud

  Circle a real object  -> it tells you what it is
  Write math            -> it solves it (verified with SymPy)
  Sketch something      -> it guesses
  Write a word          -> it reads + translates

Fully local + free: MediaPipe for hands, Ollama or LM Studio for the vision model,
macOS `say` for the voice.

Keys: P say it again | Z ask zone right/left/off | G shape snapping on/off | T learn mode (中文 / EN / off) | K ink color | X remove stickers
      V talk | SPACE ask | C clear | L arabic/english | F full screen | R record | S photo | M mute | Q quit
"""
import argparse
import base64
import queue
import json
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

VERSION = "2026-10-31 draw-first"
W, H = 720, 1280
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
OUT_DIR = HERE / "captures"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

INKS = [("blue", (255, 120, 10)), ("red", (40, 40, 235)), ("yellow", (0, 210, 255)), ("white", (250, 250, 250))]
INK_SHADOW = (25, 25, 25)
ACCENT = (255, 196, 92)          # soft blue (BGR)
SEND_INK = (255, 0, 255)         # strokes are magenta in the image the model sees

VISION_HINTS = ("vl", "vision", "llava", "gemma3", "gemma-3", "minicpm-v", "moondream", "pixtral",
                "bakllava", "llama3.2-vision", "granite3.2-vision", "mistral-small3", "internvl", "phi-3.5-vision",
                "phi-4-multimodal", "smolvlm", "gemma4", "gemma-4")
PREFERRED = ("qwen3-vl", "qwen2.5vl", "qwen2.5-vl", "gemma3", "gemma-3", "minicpm-v", "llama3.2-vision", "llava")

SYSTEM = {
    "ar": """You are AirMind, a sharp and funny assistant that sees through a webcam.
The user draws in the air with a finger; the strokes are overlaid on the image in bright MAGENTA.
Decide what the magenta strokes are and reply:
- "object": the strokes circle, underline or point at a real thing in the scene. Name the thing precisely (brand/type if visible) and add one short interesting fact.
- "math": handwritten math. Copy it exactly into "math_expr" using Python/SymPy syntax (use * and **, use = for equations, x as the variable).
- "drawing": a sketch. Guess what it depicts, playfully.
- "text": a handwritten word or short phrase. Read it and translate it (Arabic<->English).
- "scene": there are no strokes. Say something fun about what you see.
- "answer": the user asked you a question out loud. Answer it directly using what you see (up to 3 short sentences).
Reply with JSON only: {"kind": "...", "say": "...", "math_expr": ""}
"say" MUST be in Egyptian Arabic dialect, at most 2 short sentences, natural, witty, no emojis.""",
    "en": """You are AirMind, a sharp and funny assistant that sees through a webcam.
The user draws in the air with a finger; the strokes are overlaid on the image in bright MAGENTA.
Decide what the magenta strokes are and reply:
- "object": the strokes circle, underline or point at a real thing in the scene. Name it precisely and add one short interesting fact.
- "math": handwritten math. Copy it exactly into "math_expr" using Python/SymPy syntax (use * and **, use = for equations, x as the variable).
- "drawing": a sketch. Guess what it depicts, playfully.
- "text": a handwritten word or short phrase. Read it and translate it to Arabic.
- "scene": there are no strokes. Say something fun about what you see.
- "answer": the user asked you a question out loud. Answer it directly using what you see (up to 3 short sentences).
Reply with JSON only: {"kind": "...", "say": "...", "math_expr": ""}
"say" must be in English, at most 2 short sentences, witty, no emojis.""",
}

LEARN_LANGS = {"zh": "Simplified Chinese (Mandarin)", "en": "English"}
LEARN_TAG = {"zh": "中文", "en": "EN"}
LEARN_SYSTEM = """You are a friendly vocabulary tutor that sees through a webcam.
The user circled something with bright MAGENTA strokes. If there are no strokes, pick the most prominent
object in front of the camera (not the person, unless they clearly point at a body part or clothing).
Identify the object precisely and teach its name in {lang}.
Reply with JSON only:
{{"word": "<the name in {lang}{zh_note}>", "pron": "<{pron_note}>", "en": "<English name>", "ar": "<Arabic name>",
"example": "<a very short, natural example sentence in {lang} using the word>"}}"""


def learn_prompt(target):
    if target == "zh":
        return LEARN_SYSTEM.format(lang=LEARN_LANGS[target], zh_note=", simplified characters",
                                   pron_note="pinyin with tone marks, e.g. bēizi")
    return LEARN_SYSTEM.format(lang=LEARN_LANGS[target], zh_note="",
                               pron_note="simple pronunciation guide split into syllables, e.g. KUP or LAP-top")


KIND_LABEL = {
    "ar": {"object": "شايفها", "math": "مسألة", "drawing": "رسمة", "text": "كلمة", "scene": "الكادر", "answer": "سؤالك", "error": "مشكلة"},
    "en": {"object": "I SEE IT", "math": "MATH", "drawing": "DRAWING", "text": "WORD", "scene": "SCENE", "answer": "YOU ASKED", "error": "ERROR"},
}
UI = {
    "ar": {"thinking": "بيفكر", "hint": "لِم الإبهام والسبابة عشان ترسم   •   لايك = اسأل   •   قبضة = امسح",
           "ask": "اسأل", "clear": "امسح", "listen": "اتكلم", "answer": "الإجابة", "no_server": "شغّل Ollama أو LM Studio الأول",
           "listening": "سامعك...", "no_stt": "ثبّت faster-whisper الأول:  pip install faster-whisper",
           "didnt_hear": "ما سمعتش حاجة، جرّب تاني وقرّب من المايك"},
    "en": {"thinking": "thinking", "hint": "pinch to draw   •   thumbs up = ask   •   fist = clear",
           "ask": "ASK", "clear": "CLEAR", "listen": "TALK", "answer": "The answer is", "no_server": "Start Ollama or LM Studio first",
           "listening": "Listening...", "no_stt": "Install faster-whisper first:  pip install faster-whisper",
           "didnt_hear": "I didn't catch that. Try again a bit closer to the mic"},
}


# --------------------------------------------------------------------------- running helper programs
def run_tool(cmd, timeout=None):
    """subprocess.run that starts the program with posix_spawn instead of fork().

    Forking a busy macOS process (camera + OpenCV window + MediaPipe threads) can freeze the
    child before it even starts, which is exactly how `say` was "hanging" inside the app while
    it worked fine from the Terminal. posix_spawn needs: a full path, close_fds=False."""
    exe = shutil.which(cmd[0])
    if not exe:
        raise FileNotFoundError(cmd[0])
    return subprocess.run([exe] + cmd[1:], capture_output=True, text=True, timeout=timeout,
                          close_fds=False, stdin=subprocess.DEVNULL)


# --------------------------------------------------------------------------- local vision model
def http_json(url, payload=None, timeout=5):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def is_vision(name):
    n = name.lower()
    return any(h in n for h in VISION_HINTS)


def discover_models():
    """Every model the local servers know about: [(backend, url, name)]."""
    found = []
    try:
        for m in http_json("http://127.0.0.1:11434/api/tags", timeout=2).get("models", []):
            found.append(("ollama", "http://127.0.0.1:11434", m["name"]))
    except Exception:
        pass
    for port in (1234, 8080):   # LM Studio, llama.cpp llama-server (both OpenAI-compatible)
        try:
            for m in http_json(f"http://127.0.0.1:{port}/v1/models", timeout=2).get("data", []):
                found.append(("lmstudio" if port == 1234 else "llama.cpp", f"http://127.0.0.1:{port}", m["id"]))
        except Exception:
            pass
    return found


def pick_model(found, wanted=None):
    if wanted:
        for f in found:
            if wanted.lower() in f[2].lower():
                return f
        return None
    vis = [f for f in found if is_vision(f[2])]
    for pref in PREFERRED:
        for f in vis:
            if pref in f[2].lower():
                return f
    return vis[0] if vis else None


class LocalVLM:
    def __init__(self, backend, url, model, lang):
        self.backend, self.url, self.model, self.lang = backend, url, model, lang

    def warm_up(self):
        try:
            if self.backend == "ollama":
                http_json(f"{self.url}/api/generate", {"model": self.model, "keep_alive": "30m"}, timeout=180)
        except Exception:
            pass

    def ask(self, jpeg, has_strokes, question=None, focus=False):
        what = ("This is a close-up of the object I marked with a magenta line or circle"
                if has_strokes else "This is a close-up of the object I'm holding up")
        if focus and question:
            return self.chat(SYSTEM[self.lang], f'{what}. I asked out loud: "{question}". '
                             'Answer it about this object. Use kind "answer".', jpeg)
        if focus:
            return self.chat(SYSTEM[self.lang], f"{what}. What is it? Name it precisely and add one short fun fact. "
                             'Use kind "object" (or "math"/"text" if I marked writing).', jpeg)
        if question:
            prompt = (f'The user asked out loud: "{question}". Answer it based on what you see'
                      f'{" (the magenta strokes show what they mean)" if has_strokes else ""}. Use kind "answer".')
        else:
            prompt = "What do my magenta strokes mean?" if has_strokes else "No strokes this time. What do you see?"
        return self.chat(SYSTEM[self.lang], prompt, jpeg)

    def ask_learn(self, jpeg, has_strokes, target, focus=False):
        if focus:
            prompt = ("This is a close-up of the object I marked with a magenta line or circle. Teach me its word."
                      if has_strokes else
                      "This is a close-up of the object I'm holding up. Teach me its word (ignore hands and fingers).")
        else:
            prompt = "Teach me the word for what I circled." if has_strokes else "Teach me the word for the main object you see."
        return self.chat(learn_prompt(target), prompt, jpeg)

    def chat(self, system, prompt, jpeg):
        b64 = base64.b64encode(jpeg).decode()
        if self.backend == "ollama":
            r = http_json(f"{self.url}/api/chat", {
                "model": self.model, "stream": False, "format": "json", "keep_alive": "30m",
                "options": {"temperature": 0.4, "num_predict": 220},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": prompt, "images": [b64]}],
            }, timeout=120)
            return r["message"]["content"]
        r = http_json(f"{self.url}/v1/chat/completions", {
            "model": self.model, "temperature": 0.4, "max_tokens": 220,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": [
                             {"type": "text", "text": prompt},
                             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
        }, timeout=300)
        return r["choices"][0]["message"]["content"]


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def parse_reply(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    say = str(d.get("say", "") if d else raw).strip().strip('"')
    return str(d.get("kind") or "scene").lower(), say, str(d.get("math_expr") or "").strip()


def solve_math(expr):
    """Returns (pretty_line, answer_text) or None. The model only transcribes; SymPy does the math."""
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                                standard_transformations, convert_xor)
        tr = standard_transformations + (implicit_multiplication_application, convert_xor)
        clean = expr.replace("×", "*").replace("÷", "/").replace("−", "-").replace("==", "=").rstrip("=").strip()
        if not clean:
            return None
        if "=" in clean:
            lhs, rhs = clean.split("=", 1)
            eq = sp.Eq(parse_expr(lhs, transformations=tr), parse_expr(rhs, transformations=tr))
            syms = sorted(eq.free_symbols, key=str)
            if not syms:
                return f"{clean}", ("True" if bool(eq) else "False")
            sol = sp.solve(eq, syms[0])
            ans = ", ".join(str(sp.nsimplify(s)) for s in sol) or "no solution"
            return f"{clean}   ->   {syms[0]} = {ans}", ans
        val = sp.nsimplify(parse_expr(clean, transformations=tr))
        if val.is_number and not val.is_Integer:
            approx = sp.N(val, 6)
            ans = f"{val} ≈ {approx}" if str(val) != str(approx) else str(approx)
        else:
            ans = str(val)
        return f"{clean} = {ans}", ans
    except Exception:
        return None


# --------------------------------------------------------------------------- listening (speech to text, local)
class Listener:
    """Records from the mic until you stop talking (or press V again)."""
    RATE = 16000

    def __init__(self):
        self.active, self.chunks, self.level = False, [], 0.0

    def start(self):
        import sounddevice as sd
        self.chunks, self.active, self.level = [], True, 0.0
        self.t0, self.last_voice, self.floor = time.perf_counter(), None, None
        self.stream = sd.InputStream(samplerate=self.RATE, channels=1, dtype="float32",
                                     blocksize=self.RATE // 10, callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        a = indata[:, 0].copy()
        self.chunks.append(a)
        rms = float(np.sqrt(np.mean(a ** 2)))
        self.level = rms
        if self.floor is None or len(self.chunks) <= 3:           # first 0.3 s: learn the room noise
            self.floor = rms if self.floor is None else max(self.floor, rms)
        elif rms > max(0.012, self.floor * 3):
            self.last_voice = time.perf_counter()

    def should_stop(self):
        now = time.perf_counter()
        if now - self.t0 > 15:
            return True
        if self.last_voice:
            return now - self.last_voice > 1.3                    # you paused: done
        return now - self.t0 > 6                                   # nothing said

    def stop(self):
        self.active = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        heard = self.last_voice is not None
        return (np.concatenate(self.chunks) if self.chunks else np.zeros(0, np.float32)), heard


class SpeechToText:
    """faster-whisper on the CPU. `base` is ~145 MB (downloaded once), `small` ~480 MB and better at Arabic."""

    def __init__(self, size):
        self.size, self.model, self.error, self.started = size, None, None, False

    def ensure_loaded(self):
        """Loaded on first use only: faster-whisper pulls in PyAV, whose copy of the macOS
        audio/video libraries clashes with OpenCV's, so we keep it out unless you talk."""
        if not self.started:
            self.started = True
            print("Loading the speech model (first time you press V)...", flush=True)
            threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.size, device="cpu", compute_type="int8")
        except ImportError:
            self.error = "missing"
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print("Speech model failed to load:", self.error)

    def transcribe(self, audio):
        for _ in range(600):                                       # first run may still be downloading
            if self.model or self.error:
                break
            time.sleep(0.1)
        if not self.model:
            return None
        segments, _ = self.model.transcribe(audio, beam_size=1, vad_filter=False,
                                            condition_on_previous_text=False)
        return " ".join(seg.text.strip() for seg in segments).strip()


# --------------------------------------------------------------------------- text rendering (Arabic aware)
class Text:
    AR_FONTS = ["/System/Library/Fonts/GeezaPro.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    CJK_FONTS = ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
                 "/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]
    LATIN_FONTS = ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc",
                   "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]

    def __init__(self):
        from PIL import ImageFont
        self.ImageFont = ImageFont
        first = lambda fs: next((f for f in fs if Path(f).exists()), None)
        self.paths = {"ar": first(self.AR_FONTS), "cjk": first(self.CJK_FONTS), "latin": first(self.LATIN_FONTS)}
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display
            self._shape = lambda s: get_display(arabic_reshaper.reshape(s))
        except ImportError:
            self._shape = lambda s: s

    def font(self, size, text=""):
        script = "ar" if self.is_rtl(text) else "cjk" if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text) else "latin"
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
    cw, pad = min(W - 48, 900), 26
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


def render_word_card(text_r, tag, word, pron, en, ar, example):
    """Vocabulary card: big word, pronunciation, meanings, example."""
    from PIL import Image, ImageDraw
    cw, pad = min(W - 48, 470), 22
    inner = cw - pad * 2
    rows = [(tag, 20, ACCENT, 30), (word, 64, (255, 255, 255), 78), (pron, 30, ACCENT, 42)]
    meaning = " · ".join(x for x in (en, ar) if x)
    if meaning:
        rows.append((meaning, 26, (215, 215, 220), 38))
    for ln in text_r.wrap(example, 22, inner) if example else []:
        rows.append((ln, 22, (165, 165, 172), 30))
    h = pad * 2 + sum(r[3] for r in rows)
    img = Image.new("RGBA", (cw, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, cw - 1, h - 1), radius=26, fill=(16, 16, 18, 215))
    y = pad
    for text, size, col, step in rows:
        if " · " in text and text_r.is_rtl(text):            # "cup · كوباية": draw the two halves separately
            left, right = text.split(" · ", 1)
            fl, fr = text_r.font(size, left), text_r.font(size, right)
            sr = text_r.shape(right)
            total = fl.getlength(left + "  ·  ") + fr.getlength(sr)
            x = (cw - total) / 2
            d.text((x, y), left + "  ·  ", font=fl, fill=(col[2], col[1], col[0], 255))
            d.text((x + fl.getlength(left + "  ·  "), y), sr, font=fr, fill=(col[2], col[1], col[0], 255))
        else:
            s, f = text_r.shape(text), text_r.font(size, text)
            d.text(((cw - f.getlength(s)) / 2, y), s, font=f, fill=(col[2], col[1], col[0], 255))
        y += step
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


# --------------------------------------------------------------------------- hands
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


# --------------------------------------------------------------------------- app
class AirMind:
    def __init__(self, vlm, lang, voices, muted=False):
        self.vlm, self.voices, self.muted = vlm, voices, muted
        self.set_lang(lang)
        self.text = Text()
        self.strokes, self.cur = [], None
        self.drawing, self.pinch_frames, self.off_frames = False, 0, 0
        self.ended_at = -1.0
        self.band = (0, 0, W, H)
        self.listener, self.stt, self.last_frame = Listener(), None, None
        self.frozen, self.frozen_t = None, 0.0                   # (still picture of the red box, its x/y) while thinking
        self.speech_q, self.speech_status, self.last_spoken = queue.Queue(), None, []
        threading.Thread(target=self._speech_loop, daemon=True).start()
        self.ink_i, self.learn, self.stickers = 0, None, []     # learn: None | "zh" | "en"
        self.snap = True                                         # tidy rough shapes into clean ones
        self.zone = "right"                                      # fixed red ask zone: "right" | "left" | None
        self.filter = OneEuro()
        self.hold = {"ask": 0.0, "clear": 0.0, "listen": 0.0}
        self.thinking, self.card, self.card_t = False, None, 0.0
        self.cursor = None
        self.rec = None
        self.started = time.perf_counter()
        self.tmp = Path(tempfile.mkdtemp(prefix="airmind_"))
        self.say_n = 0

    @property
    def ink(self):
        return INKS[self.ink_i][1]

    def zone_rect(self):
        """The red box, on your right side of the mirrored picture (or left)."""
        bx, by, bw, bh = self.band
        w, h = int(bw * 0.30), int(bh * 0.55)
        x = bx + bw - w - int(bw * 0.04) if self.zone == "right" else bx + int(bw * 0.04)
        return x, by + int(bh * 0.18), w, h

    def cycle_zone(self):
        order = ["right", "left", None]
        self.zone = order[(order.index(self.zone) + 1) % len(order)]

    def unfreeze(self):
        self.frozen = None

    def cycle_learn(self):
        order = [None, "zh", "en"]
        self.learn = order[(order.index(self.learn) + 1) % len(order)]
        self.card = None

    def set_lang(self, lang):
        self.lang, self.ui, self.voice = lang, UI[lang], self.voices.get(lang)
        if self.vlm:
            self.vlm.lang = lang

    # ---- gestures -----------------------------------------------------------
    def update_hand(self, hand, dt, other=None):
        """`hand` draws; `other` (your second hand, if in view) can give the thumbs-up / fist commands."""
        cmd = self._commands(read_hand(other), dt) if other is not None else None
        if hand is None:
            self.end_stroke()
            self.cursor = None
            if other is None:
                self.hold = {"ask": 0.0, "clear": 0.0, "listen": 0.0}
            return cmd
        g = read_hand(hand)
        p = self.filter(g["mid"], dt)
        self.cursor = p
        if not self.drawing and g["pinch"] < 0.20:          # fingertips really touching
            self.pinch_frames += 1
            if self.pinch_frames >= 4:                        # held for ~0.13 s, not a passing brush
                self.drawing, self.cur = True, []
                self.strokes.append(self.cur)
        elif self.drawing and g["pinch"] > 0.32:
            self.end_stroke()
        elif not self.drawing:
            self.pinch_frames = 0
        if self.drawing:
            if not self.cur or np.linalg.norm(p - self.cur[-1]) > 2:
                self.cur.append(p.copy())
            return cmd
        return cmd if other is not None else self._commands(g, dt)

    def _commands(self, g, dt):
        fired = None
        for name, active in (("ask", g["thumbs_up"]), ("clear", g["fist"])):
            self.hold[name] = self.hold[name] + dt if active else 0.0
            if self.hold[name] > 0.7:
                fired, self.hold[name] = name, -1.5   # cooldown
        return fired, g["center"]

    def end_stroke(self):
        if self.drawing and self.cur is not None and len(self.cur) < 3 and self.strokes:
            self.strokes.pop()
        elif self.drawing and self.snap and self.cur is not None and self.strokes and self.strokes[-1] is self.cur:
            shape = snap_shape(self.cur)
            if shape is not None:
                self.strokes[-1] = shape
        if self.drawing:
            self.ended_at = time.perf_counter()
        self.drawing, self.cur, self.pinch_frames, self.off_frames = False, None, 0, 0
        self.filter.reset()

    def clear(self):
        self.strokes, self.card, self.stickers = [], None, []
        self.unfreeze()

    # ---- asking -------------------------------------------------------------
    # ---- listening ----------------------------------------------------------
    def toggle_listen(self):
        if self.listener.active:
            self._finish_listening()
            return
        if self.thinking:
            return
        if self.stt is None or self.stt.error == "missing":
            self.show(KIND_LABEL[self.lang]["error"], self.ui["no_stt"])
            return
        self.stt.ensure_loaded()
        try:
            self.listener.start()
            self.card = None
        except Exception as e:
            self.show(KIND_LABEL[self.lang]["error"], f"Mic: {e}"[:160])

    def tick_listening(self):
        if self.listener.active and self.listener.should_stop():
            self._finish_listening()

    def _finish_listening(self):
        audio, heard = self.listener.stop()
        if not heard or len(audio) < Listener.RATE // 2:
            self.show(KIND_LABEL[self.lang]["error"], self.ui["didnt_hear"])
            return
        self.thinking = True
        threading.Thread(target=self._transcribe_then_ask, args=(audio,), daemon=True).start()

    def _transcribe_then_ask(self, audio):
        try:
            text = self.stt.transcribe(audio)
        except Exception as e:
            text, self.stt.error = None, str(e)
        self.thinking = False
        if not text:
            self.show(KIND_LABEL[self.lang]["error"], self.ui["didnt_hear"] if self.stt.model else f"Speech: {self.stt.error}")
            return
        print("You said:", text)
        self.ask(self.last_frame, question=text)

    def ask(self, frame, question=None):
        if self.thinking or frame is None:
            return
        if self.vlm is None:
            self.show(KIND_LABEL[self.lang]["error"], self.ui["no_server"])
            return
        has_strokes = any(len(s) > 1 for s in self.strokes)
        # What the model looks at, in order of priority:
        #   1) you drew something  -> a close-up around your drawing (the object you marked)
        #   2) the red box is on   -> a close-up of what's inside the box
        #   3) otherwise           -> the whole camera view
        anchor = None
        if has_strokes:
            pts = np.array([p for s in self.strokes for p in s], np.float32)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            side = max(x1 - x0, y1 - y0, 160) * 1.35                    # square-ish, with a margin
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            rw, rh = int(min(W - rx, side)), int(min(H - ry, side))
            anchor = np.array([cx, y0 - 4])                              # sticker sits on top of the drawing
        elif self.zone:
            rx, ry, rw, rh = self.zone_rect()
        else:
            rx, ry, rw, rh = self.band
        clean = frame[ry:ry + rh, rx:rx + rw].copy()
        marked = frame.copy()
        for s in self.strokes:
            if len(s) > 1:
                pts_s, closed = stroke_points(s)
                cv2.polylines(marked, [pts_s], closed, SEND_INK, 4, cv2.LINE_AA)
        crop = marked[ry:ry + rh, rx:rx + rw]
        k = 512 / max(rw, rh) if (has_strokes or self.zone) else 768 / max(rw, rh)
        crop = cv2.resize(crop, (max(1, int(rw * k)), max(1, int(rh * k))),
                          interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        focus = has_strokes or bool(self.zone)
        if focus:                                  # that area holds still until the answer goes away
            self.frozen, self.frozen_t = (clean, (rx, ry)), time.perf_counter()
        self.thinking, self.card = True, None
        if self.learn and not question:
            threading.Thread(target=self._learn_worker, args=(jpeg.tobytes(), has_strokes, anchor, self.learn, focus),
                             daemon=True).start()
        else:
            threading.Thread(target=self._ask_worker, args=(jpeg.tobytes(), has_strokes, question, focus),
                             daemon=True).start()
        if has_strokes and not self.learn:
            self.strokes = []

    def _learn_worker(self, jpeg, has_strokes, anchor, target, focus=False):
        try:
            d = parse_json(self.vlm.ask_learn(jpeg, has_strokes, target, focus))
            word = str(d.get("word") or "").strip()
            if not word:
                raise ValueError("no word in the reply")
            pron, en, ar, ex = (str(d.get(k) or "").strip() for k in ("pron", "en", "ar", "example"))
            self.card = render_word_card(self.text, f"{LEARN_TAG[target]}", word, pron,
                                         en if target != "en" else "", ar, ex)
            self.card_t, self.card_corner = time.perf_counter(), True
            if anchor is not None:                          # drawn circle -> sticker on the object
                self.stickers.append((anchor, render_sticker(self.text, word, pron, self.ink)))
                self.stickers = self.stickers[-12:]
                self.strokes = []
            self.thinking = False
            self.speak(word, voice=self.voices.get(target), rate=140)          # 杯子
            if target == "zh" and en:
                self.speak(en, voice=self.voices.get("en"))                   # ... "cup"
            elif target == "en" and ar:
                self.speak(ar, voice=self.voices.get("ar"))
        except Exception as e:
            self.unfreeze()
            self.show(KIND_LABEL[self.lang]["error"], f"{type(e).__name__}: {e}"[:160])
        finally:
            self.thinking = False

    def _ask_worker(self, jpeg, has_strokes, question=None, focus=False):
        try:
            kind, say, expr = parse_reply(self.vlm.ask(jpeg, has_strokes, question, focus))
            math_line = None
            if kind == "math" or expr:
                solved = solve_math(expr) if expr else None
                if solved:
                    math_line, ans = solved
                    kind = "math"
                    say = f"{self.ui['answer']} {ans}" if not say or len(say) > 90 else say
            if question:
                kind = "answer" if kind not in ("math",) else kind
            label = KIND_LABEL[self.lang].get(kind, KIND_LABEL[self.lang]["scene"])
            self.show(label, say, math_line, question)
            if focus:
                self.card_corner = True                    # red-box answers sit beside the box and fade out
            self.speak(say)
        except Exception as e:
            self.unfreeze()
            self.show(KIND_LABEL[self.lang]["error"], f"{type(e).__name__}: {e}"[:160])
        finally:
            self.thinking = False

    def show(self, label, say, math_line=None, question=None):
        self.card_corner = False
        self.card = render_card(self.text, label, say, math_line, question)
        self.card_t = time.perf_counter()

    def speak(self, text, voice=None, rate=None):
        """Queue something to say. A single speech thread works through the queue, so nothing waits on it."""
        if not text:
            return
        if self.muted:
            self.speech_status = ("muted (press M)", time.perf_counter())
            print("(muted, press M to unmute)", flush=True)
            return
        self.last_spoken.append((text, voice or self.voice, rate))
        self.last_spoken = self.last_spoken[-2:]
        self.speech_q.put((text, voice or self.voice, rate))

    def _speech_loop(self):
        while True:
            text, voice, rate = self.speech_q.get()
            try:
                self.speech_status = (f"speaking: {text[:40]}", time.perf_counter())
                ok = self._say(text, voice, rate)
                self.speech_status = (f"spoke: {text[:40]}" if ok else f"SPEECH FAILED: {text[:30]} (see Terminal)",
                                      time.perf_counter())
            except Exception as e:
                print("speech error:", e, flush=True)
                self.speech_status = (f"SPEECH ERROR: {e}"[:60], time.perf_counter())

    def replay(self):
        for text, voice, rate in list(self.last_spoken):
            self.speech_q.put((text, voice, rate))

    def _say(self, text, voice, rate):
        """Say it out loud. Preferred: hand it to speaker.py (started by start.sh from the shell),
        because `say` launched from inside this camera app can freeze. Fallback: run `say` here."""
        path = None
        if self.rec:
            self.say_n += 1
            path = str(self.tmp / f"say_{self.say_n}.aiff")
        try:
            r = http_json("http://127.0.0.1:8765/say", {"text": text, "voice": voice, "rate": rate, "file": path},
                          timeout=95)
            print(f"speak via helper [{voice or 'default'}] {text[:60]!r}: {'ok' if r.get('ok') else 'FAILED'}", flush=True)
            if r.get("ok") and path:
                self.rec["audio"].append((time.perf_counter() - self.rec["start"], Path(path)))
            return bool(r.get("ok"))
        except (urllib.error.URLError, ConnectionError, OSError):
            if not getattr(self, "_helper_warned", False):
                self._helper_warned = True
                print("speech helper not running (start with ./start.sh, or run: python speaker.py &)", flush=True)
        if not shutil.which("say"):
            print("`say` not found: speech only works on macOS", flush=True)
            return False
        tries = [voice, voice.split(" (")[0]] if voice and " (" in voice else [voice]
        extra = ["-r", str(rate)] if rate else []
        for v in tries + ([None] if voice else []):
            args = ["say"] + (["-v", v] if v else []) + extra
            if self.rec:
                self.say_n += 1
                path = self.tmp / f"say_{self.say_n}.aiff"
                res = run_tool(args + ["-o", str(path), text], timeout=30)
                if res.returncode == 0:
                    self.rec["audio"].append((time.perf_counter() - self.rec["start"], path))
                    run_tool(["afplay", str(path)], timeout=60)
            else:
                print(f"speak [{v or 'default voice'}] {text[:60]!r} ...", flush=True)
                try:
                    res = run_tool(args + [text], timeout=30)
                except subprocess.TimeoutExpired:
                    print("speak: `say` hung for 30 s. macOS audio is stuck: run  sudo killall coreaudiod", flush=True)
                    return False
            ok = res.returncode == 0 and "not found" not in (res.stderr + res.stdout).lower()
            print(f"speak [{v or 'default voice'}] {text[:60]!r}: {'ok' if ok else 'FAILED ' + (res.stderr or res.stdout).strip()}",
                  flush=True)
            if ok:
                return True
        return False

    # ---- drawing ------------------------------------------------------------
    def render(self, frame, now, hold_info):
        out = frame.copy()
        if self.frozen is not None:                        # the red box holds still until the answer
            still, (fx, fy) = self.frozen
            h, w = still.shape[:2]
            out[fy:fy + h, fx:fx + w] = still
            if time.perf_counter() - self.frozen_t < 0.18:       # little camera-shutter flash
                out[fy:fy + h, fx:fx + w] = cv2.addWeighted(still, 0.5, np.full_like(still, 255), 0.5, 0)
        if self.zone:
            zx, zy, zw, zh = self.zone_rect()
            red = (40, 40, 235)
            cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), (20, 20, 20), 7, cv2.LINE_AA)
            cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), red, 4, cv2.LINE_AA)
            tag = self.text.label("ASK ZONE" if self.lang == "en" else "حط الحاجة هنا", 22, (255, 255, 255))
            round_rect(out, zx, zy - 38, zx + tag.shape[1] + 24, zy - 4, 12, red, 0.95)
            blit(out, tag, zx + 12, zy - 36)
        for (ax, ay), st in self.stickers:
            blit(out, st, int(ax - st.shape[1] / 2), int(ay - st.shape[0]))
        if self.learn:
            chip = self.text.label(f"LEARN  {LEARN_TAG[self.learn]}", 26, (255, 255, 255))
            round_rect(out, 20, 20, 20 + chip.shape[1] + 30, 20 + 50, 25, self.ink, 0.9)
            blit(out, chip, 35, 28)
        for s in self.strokes:
            if len(s) > 1:
                pts, closed = stroke_points(s)
                cv2.polylines(out, [pts], closed, INK_SHADOW, 8, cv2.LINE_AA)
                cv2.polylines(out, [pts], closed, self.ink, 5, cv2.LINE_AA)
        if self.cursor is not None:
            c = tuple(int(v) for v in self.cursor)
            if self.drawing:
                cv2.circle(out, c, 7, self.ink, -1, cv2.LINE_AA)
            else:
                cv2.circle(out, c, 14, INK_SHADOW, 4, cv2.LINE_AA)
                cv2.circle(out, c, 14, self.ink, 2, cv2.LINE_AA)
        if hold_info is not None:
            for name in ("ask", "clear", "listen"):
                frac = self.hold[name] / 0.7
                if frac > 0.05:
                    cx, cy = (int(v) for v in hold_info)
                    cv2.ellipse(out, (cx, cy - 90), (38, 38), -90, 0, 360, (60, 60, 60), 5, cv2.LINE_AA)
                    cv2.ellipse(out, (cx, cy - 90), (38, 38), -90, 0, 360 * min(frac, 1), ACCENT, 5, cv2.LINE_AA)
                    lbl = self.text.label(self.ui[name], 24, (255, 255, 255))
                    blit(out, lbl, cx - lbl.shape[1] // 2, cy - 160)

        if self.speech_status and now - self.speech_status[1] < 6:
            msg = self.speech_status[0]
            col = (80, 80, 255) if "FAIL" in msg or "ERROR" in msg else (230, 230, 230)
            lbl = self.text.label(msg, 22, col)
            round_rect(out, 16, H - 92, 16 + lbl.shape[1] + 24, H - 52, 14, (16, 16, 18), 0.8)
            blit(out, lbl, 28, H - 88)
        if self.listener.active:
            pill = self.text.label(self.ui["listening"], 28, (255, 255, 255))
            pw = pill.shape[1] + 150
            x0, y0 = (W - pw) // 2, H - 150
            round_rect(out, x0, y0, x0 + pw, y0 + 64, 32, (16, 16, 18), 0.85)
            cv2.circle(out, (x0 + 34, y0 + 32), 10, (60, 60, 240), -1, cv2.LINE_AA)
            blit(out, pill, x0 + 56, y0 + 12)
            lvl = min(1.0, self.listener.level * 12)
            for k in range(5):
                hh = int(8 + 30 * lvl * (0.5 + 0.5 * np.sin(now * 14 + k)))
                cx = x0 + pw - 70 + k * 12
                cv2.line(out, (cx, y0 + 32 - hh // 2), (cx, y0 + 32 + hh // 2), ACCENT, 5, cv2.LINE_AA)
        elif self.thinking:
            pill = self.text.label(self.ui["thinking"], 28, (255, 255, 255))
            pw = pill.shape[1] + 110
            x0, y0 = (W - pw) // 2, H - 150
            round_rect(out, x0, y0, x0 + pw, y0 + 64, 32, (16, 16, 18), 0.8)
            blit(out, pill, x0 + 90 if self.text.is_rtl(self.ui["thinking"]) else x0 + 24, y0 + 12)
            dx = x0 + 28 if self.text.is_rtl(self.ui["thinking"]) else x0 + pw - 74
            for k in range(3):
                a = 0.35 + 0.65 * max(0, np.sin(now * 6 - k * 0.8))
                cv2.circle(out, (dx + k * 20, y0 + 32), 6, tuple(int(c * a) for c in ACCENT), -1, cv2.LINE_AA)
        elif self.card is not None:
            fade = min(1.0, (now - self.card_t) / 0.25)
            ch = self.card.shape[0]
            bx, by, bw, bh = self.band
            y = by + bh + 40 if H > W else H - ch - 40   # story: right under the camera band
            x = (W - self.card.shape[1]) // 2
            if getattr(self, "card_corner", False):
                age = now - self.card_t                   # word cards sit in the corner and fade after 7 s,
                fade = min(fade, max(0.0, min(1.0, 8 - age)))   # leaving the sticker on the object
                if W > H:                                 # keep clear of the red box
                    x = 28 if self.zone == "right" else W - self.card.shape[1] - 28
            if self.frozen is not None and getattr(self, "card_corner", False) and now - self.card_t > 8:
                self.unfreeze()                           # answer gone -> the red box goes live again
            if fade > 0:
                blit(out, self.card, x, y + int((1 - min(fade, 1)) * 30), fade)
        return out


# --------------------------------------------------------------------------- recording
def open_writer(path):
    for code in ("avc1", "mp4v"):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), 30, (W, H))
        if wr.isOpened():
            return wr
    return None


def finish_recording(rec):
    rec["writer"].release()
    raw, final = rec["raw"], rec["final"]
    ff = shutil.which("ffmpeg")
    if not ff:
        raw.rename(final)
        print(f"Saved {final}  (install ffmpeg with `brew install ffmpeg` to get the voice inside the video)")
        return
    dur = rec["frames"] / 30
    cmd = [ff, "-y", "-loglevel", "error", "-i", str(raw)]
    clips = [(o, p) for o, p in rec["audio"] if o < dur]
    for _, p in clips:
        cmd += ["-i", str(p)]
    if clips:
        chains = [f"[{i + 1}:a]adelay={int(o * 1000)}|{int(o * 1000)}[a{i}]" for i, (o, _) in enumerate(clips)]
        mix = "".join(f"[a{i}]" for i in range(len(clips))) + f"amix=inputs={len(clips)}:normalize=0[aout]"
        cmd += ["-filter_complex", ";".join(chains + [mix]), "-map", "0:v", "-map", "[aout]", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-t", f"{dur:.2f}",
            "-movflags", "+faststart", str(final)]
    res = run_tool(cmd)
    if res.returncode == 0:
        raw.unlink(missing_ok=True)
        print(f"Saved {final}")
    else:
        raw.rename(final)
        print(f"Saved {final} without voice (ffmpeg: {res.stderr.strip()[:200]})")


# --------------------------------------------------------------------------- setup + loop
def ensure_hand_model():
    MODEL_DIR.mkdir(exist_ok=True)
    p = MODEL_DIR / "hand_landmarker.task"
    if not p.exists():
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        print("Downloading hand model ...")
        with urllib.request.urlopen(HAND_URL, context=ctx) as r, open(p, "wb") as f:
            f.write(r.read())
    return str(p)


def find_voice(locale_re, prefer=()):
    if not shutil.which("say"):
        return None
    try:
        out = run_tool(["say", "-v", "?"]).stdout
    except Exception:
        return None
    # lines look like "Samantha            en_US    # ..." or "Tingting (Chinese (China mainland)) zh_CN    # ..."
    names = []
    for ln in out.splitlines():
        m = re.match(r"^(.+?)\s+([a-z]{2,3}_[A-Za-z0-9]+)\s+#", ln.strip())
        if m and re.search(locale_re, m.group(2) + " "):
            names.append(m.group(1).strip())
    for p in prefer:
        for n in names:
            if n.startswith(p):
                return n
    return names[0] if names else None


def compose(cam):
    """Lay the camera frame onto the W x H canvas. Returns (canvas, camera_view, band).

    Wide mode: the camera fills the canvas. Story mode (9:16): the WHOLE wide camera view sits
    in a band across the canvas, over a blurred copy of itself, so nothing is cropped away and
    there's room under it for the answer card. `band` = (x, y, w, h) of the live camera area.
    """
    h, w = cam.shape[:2]
    if W > H:
        view = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
        return view, view, (0, 0, W, H)
    bh = int(W * h / w)
    by = int(H * 0.36) - bh // 2
    view = cv2.resize(cam, (W, bh), interpolation=cv2.INTER_AREA)
    k = H / h
    cover = cv2.resize(cam, (int(w * k / 8), H // 8), interpolation=cv2.INTER_AREA)
    cover = cv2.GaussianBlur(cover, (0, 0), 3)
    cover = cv2.resize(cover, (int(w * k), H), interpolation=cv2.INTER_LINEAR)
    x0 = (cover.shape[1] - W) // 2
    canvas = (cover[:, x0:x0 + W].astype(np.float32) * 0.4).astype(np.uint8)
    canvas[by:by + bh] = view
    return canvas, view, (0, by, W, bh)


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


def run(app, cam_index, prefer_hand="right"):
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=ensure_hand_model()),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.6, min_tracking_confidence=0.5))
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        sys.exit("Could not open the camera. System Settings > Privacy & Security > Camera > allow your Terminal.")
    OUT_DIR.mkdir(exist_ok=True)
    win = "AirMind"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, *((1280, 720) if W > H else (560, 996)))
    full = False

    t0 = prev = time.perf_counter()
    last_ts, fps = -1, 30.0
    while True:
        ok, raw = cap.read()
        if not ok:
            continue
        now = time.perf_counter()
        dt, prev = max(1e-3, min(0.1, now - prev)), now
        fps = fps * 0.9 + 0.1 / dt
        ts = max(last_ts + 1, int((now - t0) * 1000))
        last_ts = ts
        frame, view, (bx, by, bw, bh) = compose(cv2.flip(raw, 1))
        app.band = (bx, by, bw, bh)
        app.last_frame = frame
        app.tick_listening()

        res = hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                              data=cv2.cvtColor(view, cv2.COLOR_BGR2RGB)), ts)
        hand, other = pick_hands(res, app.band, prefer_hand, app.cursor)
        ev = app.update_hand(hand, dt, other)
        hold_at = None
        if ev:
            fired, hold_at = ev
            if fired == "ask":
                app.ask(frame)
            elif fired == "clear":
                app.clear()
            elif fired == "listen":
                app.toggle_listen()
        out = app.render(frame, now, hold_at)

        if app.rec:
            due = int((now - app.rec["start"]) * 30) - app.rec["frames"]
            for _ in range(max(0, due)):
                app.rec["writer"].write(out)
                app.rec["frames"] += 1

        view = out.copy()
        if now - t0 < 25 and not app.strokes and app.card is None:
            hint = app.text.label(app.ui["hint"], 22, (235, 235, 235))
            blit(view, hint, (W - hint.shape[1]) // 2, 28)
        model = app.vlm.model if app.vlm else "no model"
        cv2.circle(view, (24, H - 26), 6, (90, 220, 120) if app.vlm else (60, 60, 230), -1, cv2.LINE_AA)
        cv2.putText(view, f"{'SNAP  ' if app.snap else ''}{app.lang.upper()}  [L]  {Path(model).name[:34]}  {fps:4.1f} fps", (38, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        if app.rec:
            cv2.circle(view, (W - 110, 34), 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(view, f"REC {int(now - app.rec['start'])}s", (W - 92, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, view)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            app.ask(frame)
        elif k == ord("c"):
            app.clear()
        elif k == ord("v"):
            app.toggle_listen()
        elif k == ord("t"):
            app.cycle_learn()
        elif k == ord("k"):
            app.ink_i = (app.ink_i + 1) % len(INKS)
        elif k == ord("x"):
            app.stickers = []
        elif k == ord("g"):
            app.snap = not app.snap
        elif k == ord("z"):
            app.cycle_zone()
        elif k == ord("p"):
            app.replay()
        elif k == ord("m"):
            app.muted = not app.muted
        elif k == ord("f"):
            full = not full
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        elif k == ord("l"):
            app.set_lang("en" if app.lang == "ar" else "ar")
            app.card = None
            t0 = now - 20   # flash the hint in the new language for a few seconds
        elif k == ord("s"):
            p = OUT_DIR / f"airmind_{time.strftime('%H%M%S')}.png"
            cv2.imwrite(str(p), out)
            print("Saved", p)
        elif k == ord("r"):
            if app.rec is None:
                stamp = time.strftime("%H%M%S")
                raw_path = app.tmp / f"raw_{stamp}.mp4"
                wr = open_writer(raw_path)
                if wr:
                    app.rec = dict(writer=wr, start=now, frames=0, audio=[], raw=raw_path,
                                   final=OUT_DIR / f"airmind_{stamp}.mp4")
                    print("Recording ...")
            else:
                rec, app.rec = app.rec, None
                threading.Thread(target=finish_recording, args=(rec,)).start()
    if app.rec:
        finish_recording(app.rec)
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="AirMind: draw in the air, a local AI answers")
    ap.add_argument("--model", help="part of the model name to use (default: best vision model found)")
    ap.add_argument("--lang", choices=["ar", "en"], default="ar", help="answer language (press L while running to switch)")
    ap.add_argument("--voice", help="macOS voice name (default: an Arabic voice if installed)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--hand", choices=["right", "left", "any"], default="right",
                    help="the hand you draw with; the other hand is ignored (default right)")
    ap.add_argument("--story", action="store_true", help="vertical 9:16 output for Instagram stories (default: wide 16:9)")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--stt", default="base", help="speech model: tiny | base | small | off  (default base, ~145 MB)")
    ap.add_argument("--list", action="store_true", help="list the models found and exit")
    ap.add_argument("--ask-image", help="send a photo to the model once and print the answer (test without camera)")
    a = ap.parse_args()
    global W, H
    W, H = (720, 1280) if a.story else (1280, 720)

    found = discover_models()
    if a.list:
        if not found:
            print("No Ollama (11434), LM Studio (1234) or llama-server (8080) is running.")
        for backend, _, name in found:
            print(f"  [{backend}] {name}{'   <- vision' if is_vision(name) else ''}")
        return

    choice = pick_model(found, a.model)
    vlm = LocalVLM(*choice, a.lang) if choice else None
    if vlm:
        print(f"Using {vlm.model} via {vlm.backend}")
        threading.Thread(target=vlm.warm_up, daemon=True).start()
    elif found:
        print("Found models, but none looks like a vision model. Get one with:  ollama pull qwen2.5vl:7b")
        print("Or force one with --model NAME. Found:", ", ".join(f[2] for f in found))
    else:
        print("No local AI server found. Start Ollama (or LM Studio's server) and run again.")

    if a.ask_image:
        if not vlm:
            sys.exit(1)
        img = cv2.imread(a.ask_image)
        k = 1024 / max(img.shape[:2])
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        t = time.perf_counter()
        raw = vlm.ask(jpeg.tobytes(), False)
        print(f"answered in {time.perf_counter() - t:.1f}s")
        print(raw, "\n->", parse_reply(raw))
        return

    voices = {"ar": find_voice(r"\bar[_-]"), "en": find_voice(r"\ben[_-]US", ("Samantha", "Alex", "Ava", "Zoe")),
              "zh": find_voice(r"\bzh[_-]CN", ("Tingting", "Ting-Ting", "Lili", "Yu-shu"))}
    if not voices["zh"] and shutil.which("say"):
        print("No Chinese voice: System Settings > Accessibility > Spoken Content > System Voice > Manage Voices > Chinese (China)")
    if a.voice:
        voices[a.lang] = a.voice
    if not voices["ar"] and shutil.which("say"):
        print("No Arabic voice installed: System Settings > Accessibility > Spoken Content > System Voice > Manage Voices > Arabic")
    print("AirMind version:", VERSION, flush=True)
    print("Voices:", voices, flush=True)
    app = AirMind(vlm, a.lang, voices, a.mute)
    app.stt = SpeechToText(a.stt) if a.stt != "off" else None
    if not a.mute:   # voice check: you should hear this right away
        app.speak("Ready" if a.lang == "en" else "جاهز")
    run(app, a.cam, a.hand)


if __name__ == "__main__":
    main()
