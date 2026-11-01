#!/usr/bin/env python3
"""
AirMind: draw in the air with your finger, a vision model running on your Mac looks at the camera
and answers out loud. Built around learning Chinese from the real objects around you.

  pinch (thumb + index)  -> draw              thumbs up (hold) -> ask        fist (hold) -> clear
  hold an object still in the red box (AUTO on) -> it asks by itself
  point at an object in INSTANT mode -> hear its Chinese name

Modes
  T  learn words: 中文 -> EN -> off      word card with dictionary-checked pinyin, tones, 量词, HSK
  O  read text (menus, labels, signs) + halal check of ingredients
  H  "find it" review game from your notebook
  I  instant labels on everything (needs: pip install ultralytics)
  W  write a Chinese character in the air (calligraphy brush) -> recognized, spoken, scored
  B  interpreter: speak Arabic/English -> Chinese subtitles + voice, and back
  U  a short story with the last words you learned, read aloud karaoke-style
  O  (READ) now paints the translation right on top of each line of the menu / label

Keys: 1/2/3 pick another candidate | Y say it (pronunciation check) | N notebook -> Anki | A auto-ask
      P say again | Z red box right/left/off | G shape snapping | K ink | X remove stickers
      V talk | SPACE ask | C clear | L arabic/english | F full screen | R record | S photo | M mute | Q quit

Everything runs locally: MediaPipe (hands), Ollama / LM Studio / llama-server (vision model),
CC-CEDICT + pypinyin (Chinese facts), SymPy (math), faster-whisper (speech), macOS `say` + Vision OCR.
"""
import argparse
import queue
import shutil
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import chinese
import hanzi
import ocr
import ui
from audio import Listener, SpeechToText
from capture import AutoCapture, FrameBuffer
from detector import InstantDetector
from hands import OneEuro, pick_hands, read_hand, snap_shape, stroke_points
from tracker import StickerBoard
from ui import ACCENT, Text, blit, round_rect, render_card, render_sticker
from vlm import (LocalVLM, discover_models, http_json, is_vision, parse_json, parse_learn, parse_reply,
                 pick_model)
from vocab import Vocab

VERSION = "2026-11-01 wow"
W, H = 1280, 720
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
OUT_DIR = HERE / "captures"
HAND_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

INKS = [("blue", (255, 120, 10)), ("red", (40, 40, 235)), ("yellow", (0, 210, 255)), ("white", (250, 250, 250))]
INK_SHADOW = (25, 25, 25)
SEND_INK = (255, 0, 255)         # strokes are magenta in the image the model sees
MIN_STROKE = 30                  # px: smaller "strokes" are just fingers holding something
CONTEXT_SECONDS = 90             # a question within this time is a follow-up about the same picture

KIND_LABEL = {
    "ar": {"object": "شايفها", "math": "مسألة", "drawing": "رسمة", "text": "كلمة", "scene": "الكادر", "answer": "سؤالك",
           "error": "مشكلة", "info": "تمام"},
    "en": {"object": "I SEE IT", "math": "MATH", "drawing": "DRAWING", "text": "WORD", "scene": "SCENE",
           "answer": "YOU ASKED", "error": "ERROR", "info": "OK"},
}
UI = {
    "ar": {"thinking": "بيفكر", "hint": "لِم الإبهام والسبابة عشان ترسم   •   لايك = اسأل   •   قبضة = امسح",
           "ask": "اسأل", "clear": "امسح", "listen": "اتكلم", "answer": "الإجابة",
           "no_server": "شغّل Ollama أو LM Studio الأول",
           "listening": "سامعك...", "say_it": "قول الكلمة بالصيني...", "no_stt": "ثبّت faster-whisper الأول:  pip install faster-whisper",
           "didnt_hear": "ما سمعتش حاجة، جرّب تاني وقرّب من المايك",
           "no_words": "لسه ما اتعلمتش كلمات. اضغط T واعمل دايرة حوالين حاجة الأول",
           "no_word": "اتعلم كلمة الأول (T)، وبعدين اضغط Y عشان تتمرن على نطقها",
           "haram": "خلي بالك: المكونات فيها خنزير أو كحول", "doubtful": "انتبه: فيه مكونات مش واضح إنها حلال",
           "halal": "عليها علامة حلال", "right": "صح", "wrong": "لا، دي",
           "write_first": "اكتب حرف صيني في الهوا الأول (لِم صوابعك وارسم)، وبعدين لايك",
           "need_words": "اتعلم كلمتين على الأقل الأول (T وبعدين اعمل دايرة حوالين حاجتين)",
           "no_text": "مش لاقي كلام مكتوب. قرّب الورقة وخليها مستوية"},
    "en": {"thinking": "thinking", "hint": "pinch to draw   •   thumbs up = ask   •   fist = clear",
           "ask": "ASK", "clear": "CLEAR", "listen": "TALK", "answer": "The answer is",
           "no_server": "Start Ollama or LM Studio first",
           "listening": "Listening...", "say_it": "Say the word in Chinese...", "no_stt": "Install faster-whisper first:  pip install faster-whisper",
           "didnt_hear": "I didn't catch that. Try again a bit closer to the mic",
           "no_words": "No words yet. Press T and circle something first",
           "no_word": "Learn a word first (T), then press Y to practice saying it",
           "haram": "Careful: this contains pork or alcohol", "doubtful": "Check this: some ingredients may not be halal",
           "halal": "It has a halal mark", "right": "Right", "wrong": "No, that's",
           "write_first": "Write a Chinese character in the air first (pinch and draw), then thumbs up",
           "need_words": "Learn at least two words first (T, then circle two things)",
           "no_text": "I can't find any text. Hold it closer and flat"},
}


# --------------------------------------------------------------------------- running helper programs
def run_tool(cmd, timeout=None):
    """subprocess.run with posix_spawn instead of fork(): forking this busy camera process can freeze
    the child on macOS (that's how `say` used to hang). posix_spawn needs a full path + close_fds=False."""
    exe = shutil.which(cmd[0])
    if not exe:
        raise FileNotFoundError(cmd[0])
    return subprocess.run([exe] + cmd[1:], capture_output=True, text=True, timeout=timeout,
                          close_fds=False, stdin=subprocess.DEVNULL)


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


def stroke_size(s):
    p = np.asarray(s, np.float32)
    return float(max(p.max(axis=0) - p.min(axis=0))) if len(p) > 1 else 0.0


# --------------------------------------------------------------------------- app
class AirMind:
    def __init__(self, vlm, lang, voices, muted=False, vocab=None):
        self.vlm, self.voices, self.muted = vlm, voices, muted
        self.set_lang(lang)
        self.text = Text()
        self.strokes, self.cur = [], None
        self.drawing, self.pinch_frames, self.off_frames = False, 0, 0
        self.ended_at = -1.0
        self.band = (0, 0, W, H)
        self.listener, self.stt, self.last_frame = Listener(), None, None
        self.listen_purpose = "ask"                              # "ask" | "practice"
        self.frozen, self.frozen_t = None, 0.0                   # still picture of the asked area while thinking
        self.speech_q, self.speech_status, self.last_spoken = queue.Queue(), None, []
        threading.Thread(target=self._speech_loop, daemon=True).start()
        self.ink_i, self.learn = 0, None                         # learn: None | "zh" | "en"
        self.mode = None                                         # None | read | find | write | interpret
        self.board, self.pending_stickers = StickerBoard(), queue.Queue()
        self.frames, self.auto = FrameBuffer(), AutoCapture()
        self.detector = InstantDetector()
        self.vocab = vocab
        self.word = None                                         # the word on screen (for 1/2/3, Y, follow-ups)
        self.context = None                                      # last picture + conversation, for follow-ups
        self.find = None                                         # the "find it" game state
        self.point = {"box": None, "t": 0.0, "done": None}       # instant-mode pointing
        self.label_cache = {}
        self.snap = True
        self.zone = "right"                                      # fixed red ask zone: "right" | "left" | None
        self.filter = OneEuro()
        self.hold = {"ask": 0.0, "clear": 0.0, "listen": 0.0}
        self.thinking, self.card, self.card_t, self.card_corner, self.card_life = False, None, 0.0, False, 8.0
        self.cursor, self.g, self.hand_centers = None, None, []
        self.card_left = True
        self.lens = None                                         # translations painted over the text (READ)
        self.session_words = []                                  # Chinese words learned since start (for stories)
        self.story = None
        self.interp_auto = True                                  # interpreter: listen again after each line
        self.rec = None
        self.tmp = Path(tempfile.mkdtemp(prefix="airmind_"))
        self.say_n = 0

    @property
    def ink(self):
        return INKS[self.ink_i][1]

    def zone_rect(self):
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
        if self.learn and self.mode == "read":
            self.mode = None
        self.card = None

    def toggle_mode(self, mode):
        self.mode = None if self.mode == mode else mode
        self.card, self.lens = None, None
        self.unfreeze()
        if self.mode == "find":
            self.next_find()
        else:
            self.find = None
        if self.mode == "write":
            self.strokes = []
        if self.mode == "interpret" and not self.listener.active:
            self.toggle_listen("interpret")

    def set_lang(self, lang):
        self.lang, self.ui, self.voice = lang, UI[lang], self.voices.get(lang)
        if self.vlm:
            self.vlm.lang = lang

    def info(self, msg, kind="info"):
        self.show(KIND_LABEL[self.lang][kind], msg)

    # ---- gestures -----------------------------------------------------------
    def update_hand(self, hand, dt, other=None):
        """`hand` draws; `other` (your second hand, if in view) can give the thumbs-up / fist commands."""
        cmd = self._commands(read_hand(other), dt) if other is not None else None
        if hand is None:
            self.end_stroke()
            self.cursor, self.g = None, None
            if other is None:
                self.hold = {"ask": 0.0, "clear": 0.0, "listen": 0.0}
            return cmd
        g = read_hand(hand)
        self.g = g
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
        if self.drawing and self.cur is not None and self.strokes and self.strokes[-1] is self.cur:
            if len(self.cur) < 2 or stroke_size(self.cur) < (6 if self.mode == "write" else MIN_STROKE):
                self.strokes.pop()                     # fingers holding an object, not a drawing
            elif self.snap and self.mode != "write":   # no shape snapping while writing characters
                shape = snap_shape(self.cur)
                if shape is not None:
                    self.strokes[-1] = shape
        if self.drawing:
            self.ended_at = time.perf_counter()
        self.drawing, self.cur, self.pinch_frames, self.off_frames = False, None, 0, 0
        self.filter.reset()

    def clear(self):
        self.strokes, self.card, self.lens, self.story = [], None, None, None
        self.board.clear()
        self.auto.forget()
        self.unfreeze()

    # ---- listening ----------------------------------------------------------
    def toggle_listen(self, purpose="ask"):
        if self.listener.active:
            self._finish_listening()
            return
        if self.thinking:
            return
        if self.stt is None or self.stt.error == "missing":
            self.info(self.ui["no_stt"], "error")
            return
        if purpose == "practice" and not self.practice_target():
            self.info(self.ui["no_word"], "error")
            return
        self.listen_purpose = purpose
        self.stt.ensure_loaded()
        try:
            self.listener.start()
            if purpose == "ask":
                self.card = None
        except Exception as e:
            self.info(f"Mic: {e}"[:160], "error")

    def practice_target(self):
        if self.word and self.word.get("target", "zh") == "zh":
            return self.word.get("say") or self.word["word"]
        if self.find:
            return self.find["word"]
        return None

    def tick_listening(self):
        if self.listener.active and self.listener.should_stop():
            self._finish_listening()

    def _finish_listening(self):
        audio, heard = self.listener.stop()
        if not heard or len(audio) < Listener.RATE // 2:
            self.info(self.ui["didnt_hear"], "error")
            return
        self.thinking = True
        threading.Thread(target=self._transcribe_then_act, args=(audio, self.listen_purpose), daemon=True).start()

    def _transcribe_then_act(self, audio, purpose):
        try:
            lang = None
            if purpose == "practice":
                text, _ = self.stt.transcribe(audio, force="zh")
            elif purpose == "interpret":
                text, lang = self.stt.transcribe(audio, allowed=("ar", "en", "zh"))
            else:
                allowed = ("zh", "ar", "en") if self.learn == "zh" else ("ar", "en")
                text, lang = self.stt.transcribe(audio, allowed=allowed)
        except Exception as e:
            text, self.stt.error = None, str(e)
        self.thinking = False
        if purpose == "practice":
            self.check_pronunciation(text or "")
            return
        if purpose == "interpret":
            if text:
                self.thinking = True
                self._guard(self._interpret, {"text": text, "lang": lang})
            else:
                self.info(self.ui["didnt_hear"], "error")
            return
        if not text:
            self.info(self.ui["didnt_hear"] if self.stt.model else f"Speech: {self.stt.error}", "error")
            return
        print("You said:", text, flush=True)
        self.ask(self.last_frame, question=text)

    def check_pronunciation(self, heard):
        target = self.practice_target()
        if not target:
            return
        score, rows = chinese.compare_pronunciation(target, heard)
        print(f"practice {target}: heard {heard!r} -> {score:.0%}", flush=True)
        self.card = ui.render_pron_card(self.text, target, score, rows, heard)
        self.card_t, self.card_corner, self.card_life = time.perf_counter(), True, 10.0
        if self.vocab and self.vocab.get(target):
            self.vocab.result(target, score >= 0.99, "say")
        if score < 0.99:
            self.speak(target, voice=self.voices.get("zh"), rate=110)       # hear it again, slowly

    # ---- choosing what the model sees ------------------------------------------
    def _region(self):
        """(rect, strokes_used, anchor_box). Priority: your drawing > the red box > the whole view."""
        real = [s for s in self.strokes if len(s) > 1 and stroke_size(s) >= MIN_STROKE]
        if real:
            pts = np.array([p for s in real for p in s], np.float32)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            side = max(x1 - x0, y1 - y0, 160) * 1.35
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            rw, rh = int(min(W - rx, side)), int(min(H - ry, side))
            return (rx, ry, rw, rh), real, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        if self.zone:
            x, y, w, h = r = self.zone_rect()
            return r, [], (x + w // 5, y + h // 5, w * 3 // 5, h * 3 // 5)   # sticker/tracker: the middle of the box
        return self.band, [], None

    def _encode(self, img, longest):
        h, w = img.shape[:2]
        k = longest / max(w, h)
        img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                         interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    def ask(self, frame, question=None, auto=False):
        if self.thinking or frame is None:
            return
        if self.vlm is None:
            self.info(self.ui["no_server"], "error")
            return
        if self.mode == "write" and not question:
            if not any(len(s) > 1 for s in self.strokes):
                self.info(self.ui["write_first"], "error")
                return
            self.thinking, self.card = True, None
            job = {"strokes": [np.asarray(s, np.float32) for s in self.strokes if len(s) > 1]}
            threading.Thread(target=self._guard, args=(self._hanzi_worker, job), daemon=True).start()
            return
        self.lens = None
        (rx, ry, rw, rh), strokes, box = self._region()
        has_strokes = bool(strokes)
        self.card_left = rx + rw / 2 > W / 2                    # put the answer on the other side of the object
        focus = has_strokes or bool(self.zone)
        # the sharpest of the last ~0.5 s of frames: the latest one is often blurred by the thumbs-up
        sharp, _ = self.frames.best((rx, ry, rw, rh))
        base = sharp if sharp is not None else frame
        clean = base[ry:ry + rh, rx:rx + rw].copy()
        marked = base.copy()
        for s in strokes:
            pts_s, closed = stroke_points(s)
            cv2.polylines(marked, [pts_s], closed, SEND_INK, 4, cv2.LINE_AA)
        crop = marked[ry:ry + rh, rx:rx + rw]
        jpeg = self._encode(crop, 512 if focus else 768)
        if focus:                                  # that area holds still until the answer goes away
            self.frozen, self.frozen_t = (clean, (rx, ry)), time.perf_counter()
        follow = (question and self.context and time.perf_counter() - self.context["t"] < CONTEXT_SECONDS
                  and not has_strokes and self.mode is None)
        self.thinking, self.card = True, None
        job = dict(jpeg=jpeg, has_strokes=has_strokes, focus=focus, question=question, clean=clean,
                   box=box, frame=base, origin=(rx, ry))
        if follow:
            job["jpeg"] = self.context["jpeg"]             # same picture as before
            target = self._tutor_worker if self.context["kind"] in ("learn", "tutor") else self._ask_worker
            job["history"] = self.context["history"]
        elif self.mode == "read" and not question:
            job["jpeg"] = self._encode(base[ry:ry + rh, rx:rx + rw], 1280)
            target = self._read_worker
        elif self.mode == "find" and self.find and not question:
            target = self._find_worker
        elif self.learn and question:
            target = self._tutor_worker
        elif self.learn:
            target = self._learn_worker
        else:
            target = self._ask_worker
        threading.Thread(target=self._guard, args=(target, job), daemon=True).start()
        if has_strokes and not self.learn:
            self.strokes = []

    def _guard(self, fn, job):
        try:
            fn(job)
        except Exception as e:
            self.unfreeze()
            msg = f"{type(e).__name__}: {e}"
            if isinstance(e, urllib.error.URLError):
                msg = f"{self.ui['no_server']} ({e.reason})"
            self.info(msg[:160], "error")
            print("error:", msg, flush=True)
        finally:
            self.thinking = False

    # ---- 1) normal questions --------------------------------------------------
    def _ask_worker(self, job):
        history = job.get("history")
        raw = self.vlm.ask(job["jpeg"], job["has_strokes"], job["question"], job["focus"], history=history)
        kind, say, expr = parse_reply(raw)
        math_line = None
        if kind == "math" or expr:
            solved = solve_math(expr) if expr else None
            if solved:
                math_line, ans = solved
                kind = "math"
                say = f"{self.ui['answer']} {ans}" if not say or len(say) > 90 else say
        if job["question"]:
            kind = "answer" if kind != "math" else kind
        label = KIND_LABEL[self.lang].get(kind, KIND_LABEL[self.lang]["scene"])
        self.show(label, say, math_line, job["question"])
        if job["focus"]:
            self.card_corner, self.card_life = True, 10.0
        self.speak(say)
        prompt = job["question"] or "What is it?"
        hist = (history or []) + [(prompt, raw)]
        self.context = {"kind": "ask", "jpeg": job["jpeg"], "history": hist[-4:], "t": time.perf_counter()}

    # ---- 2) learn a word ------------------------------------------------------
    def _learn_worker(self, job):
        target = self.learn
        raw = self.vlm.ask_learn(job["jpeg"], job["has_strokes"], target, job["focus"])
        cands, ex, ex_tr = parse_learn(raw)
        if not cands:
            raise ValueError("the model didn't name anything. Hold it closer / more light, then try again")
        first = cands[0]["word"]                                  # the example sentence was written for this one
        if target == "zh":
            best = chinese.best_candidate(cands) or cands[0]      # first candidate that is a real dictionary word
            cands = [best] + [c for c in cands if c is not best]
        info = self._make_word(cands[0], target, ex, ex_tr, cands)
        info["first_example"] = (ex, ex_tr, first)
        self._show_word(info, job, new=True)

    def _make_word(self, cand, target, ex, ex_tr, cands):
        w = cand["word"]
        info = {"target": target, "word": w, "conf": cand["confidence"], "en": cand["en"], "ar": cand["ar"],
                "alts": [c["word"] for c in cands], "cands": cands, "verified": True}
        if target == "zh":
            v = chinese.verify(w)
            w = info["word"] = v["word"]
            info.update(verified=v["found"], tones=v["tones"], pinyin=v["pinyin"], hsk=v["hsk"],
                        en=v["en_dict"] or cand["en"])
            mp = chinese.measure_phrase(w, v["measure"])
            if mp:
                info["measure_phrase"], info["measure_py"] = mp
            if ex and w in ex:                                    # the example must use THIS word
                info.update(example=ex, example_py=chinese.sentence_pinyin(ex), example_tr=ex_tr)
        else:
            if ex and w.lower() in ex.lower():
                info.update(example=ex, example_tr=ex_tr)
        return info

    def _show_word(self, info, job, new):
        if not job.get("focus"):
            self.unfreeze()
        self.word = info
        self.card = ui.render_learn_card(self.text, info)
        self.card_t, self.card_corner = time.perf_counter(), True
        self.card_life = 14.0 if len(info["alts"]) > 1 else 8.0
        w, target = info["word"], info["target"]
        if target == "zh":
            self._remember_word(w, job.get("replaced"))
        sticker = render_sticker(self.text, w, info.get("pinyin", ""), self.ink)
        if new and job.get("box") is not None:
            self.pending_stickers.put((sticker, job["box"], job["frame"]))
            self.strokes = []
        elif not new and len(self.board):
            self.board.items[-1].img = sticker                     # picked another candidate: relabel
        audio = None
        if self.vocab:
            if not new and job.get("replaced") and self.vocab.get(job["replaced"]) and self.vocab.get(job["replaced"])["seen"] <= 1:
                self.vocab.remove(job["replaced"])
            entry = dict(word=w, lang=target, pinyin=info.get("pinyin", ""), en=info.get("en", ""), ar=info.get("ar", ""),
                         measure=(f"{info['measure_phrase']}  {info['measure_py']}" if info.get("measure_phrase") else ""),
                         hsk=info.get("hsk"), example=info.get("example", ""), example_pinyin=info.get("example_py", ""),
                         example_en=info.get("example_tr", ""), verified=int(info["verified"]))
            jpeg = cv2.imencode(".jpg", job["clean"], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes() \
                if job.get("clean") is not None and job["clean"].size else None
            self.vocab.add(entry, jpeg)
            if not (self.vocab.get(w) or {}).get("audio"):
                audio = self.vocab.audio_path(w)
        self.context = {"kind": "learn", "jpeg": job["jpeg"], "history": [], "t": time.perf_counter(), "job": job}
        self.thinking = False
        self.speak(w, voice=self.voices.get(target), rate=140,
                   save_to=audio, on_done=(lambda ok, w=w, a=audio: ok and self.vocab.set_audio(w, a)) if audio else None)
        if target == "zh" and info.get("en"):
            self.speak(info["en"].split(";")[0], voice=self.voices.get("en"))
        elif target == "en" and info.get("ar"):
            self.speak(info["ar"], voice=self.voices.get("ar"))

    def choose(self, i):
        """Keys 1/2/3: the model wasn't sure, take its other candidate instead."""
        if not self.word or i >= len(self.word.get("cands", [])) or self.thinking:
            return
        cands = self.word["cands"]
        pick = cands[i]
        if pick["word"] == self.word["word"]:
            return
        ex, ex_tr, ex_word = self.word.get("first_example", ("", "", ""))
        info = self._make_word(pick, self.word["target"], ex if ex_word == pick["word"] else "", ex_tr, cands)
        info["first_example"] = self.word.get("first_example")
        job = dict(self.context.get("job", {}) if self.context else {})
        job["replaced"] = self.word["word"]
        self._show_word(info, job, new=False)

    # ---- 3) questions to the Chinese tutor ----------------------------------------
    def _tutor_worker(self, job):
        ctx_word = self.word["word"] if self.word else None
        history = job.get("history") or []
        raw = self.vlm.ask_tutor(job["jpeg"], job["question"], ctx_word, history=history)
        d = parse_json(raw)
        zh = str(d.get("zh") or "").strip()
        if not zh:
            raise ValueError("no answer in the reply")
        py = chinese.sentence_pinyin(zh)
        self.card = ui.render_tutor_card(self.text, job["question"], zh, py, d.get("en", ""), d.get("ar", ""),
                                         d.get("tip", ""))
        self.card_t, self.card_corner, self.card_life = time.perf_counter(), False, 0
        if self.word is None or self.word.get("target") != "zh":
            self.word = {"target": "zh", "word": zh, "say": zh, "cands": [], "alts": []}
        else:
            self.word = dict(self.word, say=zh)                   # Y now practices this sentence
        self.thinking = False
        self.speak(zh, voice=self.voices.get("zh"), rate=150)
        self.speak(d.get("ar") if self.lang == "ar" else d.get("en"))
        hist = history + [(f'My question: "{job["question"]}"', raw)]
        self.context = {"kind": "tutor", "jpeg": job["jpeg"], "history": hist[-4:], "t": time.perf_counter()}

    # ---- 4) read text + halal check ------------------------------------------------
    def _read_worker(self, job):
        lines = ocr.read_lines(job["clean"]) if job.get("clean") is not None else None
        if lines:
            self._lens(job, lines[:18])
            return
        if lines is not None and not lines:
            self.unfreeze()
            self.info(self.ui["no_text"], "error")
            return
        ocr_text = ocr.read_text(job["clean"]) if job.get("clean") is not None else None
        source = "Apple OCR + model" if ocr_text else "model"
        raw = self.vlm.ask_read(job["jpeg"], ocr_text or None)
        d = parse_json(raw)
        text = (ocr_text or str(d.get("text") or "")).strip()
        verdict, hits = chinese.halal_check(f"{ocr_text or ''}\n{d.get('text') or ''}")
        concerns = str(d.get("concerns") or "").strip()
        if verdict is None and concerns and concerns.lower() not in ("none", "no", "n/a", "-"):
            verdict, hits = "doubtful", [(concerns[:60], "", "doubtful")]
        han = chinese.han_only(text)
        py = chinese.sentence_pinyin(text) if 0 < len(han) <= 40 else ""
        self.card = ui.render_read_card(self.text, d.get("what", ""), text, py, d.get("en", ""), d.get("ar", ""),
                                        verdict, hits, source)
        self.card_t, self.card_corner, self.card_life = time.perf_counter(), False, 0
        self.unfreeze()
        self.thinking = False
        if verdict:
            self.speak(self.ui[verdict])
        tr = str(d.get("ar" if self.lang == "ar" else "en") or "")
        self.speak(tr[:160])
        self.context = {"kind": "ask", "jpeg": job["jpeg"], "history": [("Read this text.", raw)], "t": time.perf_counter()}

    def _lens(self, job, lines):
        """Paint the translation of every line right on top of it (like Google Lens)."""
        d = parse_json(self.vlm.translate_lines([t for t, _ in lines]))
        got = {}
        for it in d.get("items") or []:
            if isinstance(it, dict):
                try:
                    got[int(it.get("i", -1))] = it
                except (TypeError, ValueError):
                    pass
        key = "ar" if self.lang == "ar" else "en"
        rx, ry = job["origin"]
        items = []
        for i, (t, (x, y, w, h)) in enumerate(lines):
            lv, _ = chinese.halal_check(t)
            items.append((rx + x, ry + y, w, h, str(got.get(i, {}).get(key) or ""), lv if lv in ("haram", "doubtful") else None))
        verdict, hits = chinese.halal_check("\n".join(t for t, _ in lines))
        self.frozen, self.frozen_t = (job["clean"], (rx, ry)), time.perf_counter() - 1   # keep the text still under it
        self.lens = {"items": items, "verdict": verdict, "hits": hits, "origin": (rx, ry)}
        self.card = None
        self.thinking = False
        print("lens:", [(t, it[4]) for (t, _), it in zip(lines, items)], flush=True)
        if verdict:
            self.speak(self.ui[verdict])

    # ---- 6) air calligraphy -----------------------------------------------------------
    def _hanzi_worker(self, job):
        strokes = job["strokes"]
        img = hanzi.strokes_image(strokes)
        d = parse_json(self.vlm.read_hanzi(cv2.imencode(".jpg", img)[1].tobytes()))
        cands = [c for c in d.get("candidates") or [] if isinstance(c, dict) and chinese.has_han(str(c.get("text", "")))]
        if not cands:
            raise ValueError("couldn't read it: write bigger and slower")
        best = next((c for c in cands if chinese.lookup(chinese.han_only(c["text"]))), cands[0])
        char = chinese.han_only(best["text"])
        alts = [char] + [chinese.han_only(c["text"]) for c in cands if chinese.han_only(c["text"]) != char]
        font = self.text.paths.get("cjk")
        score = hanzi.similarity(strokes, char, font) if font else None
        mine = hanzi.strokes_thumb(strokes)
        ref = hanzi.char_image(char, font) if font else mine
        en = chinese.english(char) or str(best.get("en") or "")
        ar = str(best.get("ar") or "")
        self.card = ui.render_hanzi_card(self.text, char, alts, score, mine, ref, ar, en)
        self.card_t, self.card_corner, self.card_life = time.perf_counter(), True, 14.0
        self.card_left = True
        self.strokes = []
        self.word = {"target": "zh", "word": char, "cands": [], "alts": []}
        self.thinking = False
        if self.vocab and chinese.lookup(char):
            self.vocab.add(dict(word=char, lang="zh", pinyin=chinese.pinyin(char), en=en, ar=ar, verified=1,
                                hsk=chinese.hsk_level(char)), cv2.imencode(".jpg", mine)[1].tobytes())
            self._remember_word(char)
        self.speak(char, voice=self.voices.get("zh"), rate=130)
        self.speak(ar if self.lang == "ar" and ar else en.split(";")[0])

    def _remember_word(self, w, replaced=None):
        if replaced in self.session_words:
            self.session_words.remove(replaced)
        if w in self.session_words:
            self.session_words.remove(w)
        self.session_words.append(w)

    # ---- 7) interpreter ------------------------------------------------------------------
    def _interpret(self, job):
        text, lang = job["text"], job["lang"] or "ar"
        target = ("ar" if self.lang == "ar" else "en") if lang == "zh" else "zh"
        print(f"interpret [{lang} -> {target}]: {text}", flush=True)
        d = parse_json(self.vlm.interpret(text, target))
        out = str(d.get("translation") or "").strip()
        if not out:
            raise ValueError("no translation in the reply")
        py = chinese.sentence_pinyin(out) if target == "zh" else chinese.sentence_pinyin(text)
        tag = {"ar": "AR", "en": "EN", "zh": "中文"}
        self.card = ui.render_subtitle(self.text, text, out, py, f"{tag[lang]}  →  {tag[target]}")
        self.card_t, self.card_corner, self.card_life = time.perf_counter(), False, 0
        if target == "zh":
            self.word = {"target": "zh", "word": out, "say": out, "cands": [], "alts": []}   # Y: practice saying it
        self.thinking = False
        self.speak(out, voice=self.voices.get(target), rate=150 if target == "zh" else None,
                   on_done=lambda ok: self._relisten())

    def _relisten(self):
        if self.mode == "interpret" and self.interp_auto and not self.listener.active and not self.thinking:
            time.sleep(0.3)
            self.toggle_listen("interpret")

    # ---- 8) a story with your words (karaoke) ------------------------------------------------
    def make_story(self):
        words = self.session_words[-3:]
        if len(words) < 2 and self.vocab:
            words = [r["word"] for r in self.vocab.all() if r["lang"] == "zh"][-3:]
        if len(words) < 2:
            self.info(self.ui["need_words"], "error")
            return
        if self.thinking or self.vlm is None:
            return
        self.thinking, self.card = True, None
        threading.Thread(target=self._guard, args=(self._story_worker, {"words": words}), daemon=True).start()

    def _story_worker(self, job):
        d = parse_json(self.vlm.story(job["words"]))
        sents = [x for x in d.get("sentences") or [] if isinstance(x, dict) and chinese.has_han(str(x.get("zh", "")))]
        if not sents:
            raise ValueError("no story in the reply")
        self.story = {"title": str(d.get("title") or ""), "sentences": sents, "words": job["words"], "cur": 0}
        self._story_card()
        self.thinking = False
        if self.muted:
            self.story["cur"] = -1
            self._story_card()
            return
        for k, sn in enumerate(sents):
            self.speak(sn["zh"], voice=self.voices.get("zh"), rate=140, on_done=lambda ok, k=k: self._story_next(k))

    def _story_card(self):
        st = self.story
        st["card"] = self.card = ui.render_story_card(self.text, st, st["cur"], self.lang)
        self.card_t, self.card_corner, self.card_life = time.perf_counter(), False, 0

    def _story_next(self, k):
        st = self.story
        if not st or self.card is not st.get("card"):
            return                                  # something else is on screen now
        st["cur"] = k + 1 if k + 1 < len(st["sentences"]) else -1
        self._story_card()

    # ---- 5) the "find it" game ---------------------------------------------------
    def next_find(self):
        if not self.vocab or self.vocab.count() == 0:
            self.mode, self.find = None, None
            self.info(self.ui["no_words"], "error")
            return
        prev = self.find or {"ok": 0, "n": 0}
        r = self.vocab.next_challenge(avoid=prev.get("word"))
        if not r:
            return
        self.find = {"word": r["word"], "en": r["en"] or "", "ok": prev["ok"], "n": prev["n"], "state": "ask", "t": 0}
        self.auto.forget()
        self._find_card()
        self.speak(r["word"], voice=self.voices.get("zh"), rate=130)

    def _find_card(self, seen=None):
        f = self.find
        self.card = ui.render_challenge_card(self.text, f["word"], f"{f['ok']}/{f['n']}", f["state"], seen,
                                             f["en"] if f["state"] != "ask" else None)
        self.card_t, self.card_corner, self.card_life = time.perf_counter(), False, 0

    def _find_worker(self, job):
        f = self.find
        d = parse_json(self.vlm.ask_check(job["jpeg"], f["word"], f["en"] or f["word"]))
        ok = bool(d.get("match"))
        f["n"] += 1
        f["ok"] += ok
        f["state"], f["t"] = ("ok" if ok else "miss"), time.perf_counter()
        self.vocab.result(f["word"], ok, "found")
        self._find_card(str(d.get("seen") or ""))
        self.unfreeze()
        self.thinking = False
        if ok:
            self.speak("对！", voice=self.voices.get("zh"))
        else:
            self.speak(f"{self.ui['wrong']} {d.get('seen') or ''}")
            self.speak(f["word"], voice=self.voices.get("zh"), rate=120)

    def tick_find(self, now):
        f = self.find
        if self.mode == "find" and f and f["state"] != "ask" and now - f["t"] > 3.0 and not self.thinking:
            self.next_find()

    # ---- 6) instant labels (detector) -----------------------------------------------
    def tick_instant(self, frame, now):
        det = self.detector
        if not det.enabled:
            return
        det.feed(frame)
        if self.g is None or not self.g.get("pointing") or self.drawing:
            self.point["box"] = None
            return
        b = det.pointed(self.g["tip"])
        key = b[5] if b else None
        if key != self.point["box"]:
            self.point.update(box=key, t=now)
            return
        if key and now - self.point["t"] > 0.9 and self.point["done"] != (key, int(self.point["t"])):
            self.point["done"] = (key, int(self.point["t"]))
            x0, y0, x1, y1, en, zh, _ = b
            v = chinese.verify(zh)
            info = {"target": "zh", "word": zh, "conf": 1.0, "en": v["en_dict"] or en, "ar": "", "alts": [zh],
                    "cands": [], "verified": v["found"], "tones": v["tones"], "pinyin": v["pinyin"], "hsk": v["hsk"]}
            mp = chinese.measure_phrase(zh, v["measure"])
            if mp:
                info["measure_phrase"], info["measure_py"] = mp
            x0, y0, x1, y1 = (int(max(0, v_)) for v_ in (x0, y0, x1, y1))
            clean = frame[y0:y1, x0:x1].copy()
            jpeg = self._encode(clean, 512) if clean.size else b""
            self._show_word(info, dict(jpeg=jpeg, clean=clean, box=(x0, y0, x1 - x0, y1 - y0), frame=frame, focus=True),
                            new=True)

    # ---- cards + speech --------------------------------------------------------------
    def show(self, label, say, math_line=None, question=None):
        self.card_corner, self.card_life = False, 0
        self.card = render_card(self.text, label, say, math_line, question)
        self.card_t = time.perf_counter()

    def speak(self, text, voice=None, rate=None, save_to=None, on_done=None):
        """Queue something to say. One speech thread works through the queue, so nothing waits on it."""
        if not text:
            return
        if self.muted:
            self.speech_status = ("muted (press M)", time.perf_counter())
            if save_to and on_done:                            # still make the notebook audio, silently
                self.speech_q.put((text, voice or self.voice, rate, save_to, on_done, False))
            return
        self.last_spoken.append((text, voice or self.voice, rate))
        self.last_spoken = self.last_spoken[-2:]
        self.speech_q.put((text, voice or self.voice, rate, save_to, on_done, True))

    def _speech_loop(self):
        while True:
            text, voice, rate, save_to, on_done, play = self.speech_q.get()
            try:
                self.speech_status = (f"speaking: {text[:40]}", time.perf_counter())
                ok = self._say(text, voice, rate, save_to, play)
                self.speech_status = (f"spoke: {text[:40]}" if ok else f"SPEECH FAILED: {text[:30]} (see Terminal)",
                                      time.perf_counter())
                if on_done:
                    threading.Thread(target=on_done, args=(ok,), daemon=True).start()
            except Exception as e:
                print("speech error:", e, flush=True)
                self.speech_status = (f"SPEECH ERROR: {e}"[:60], time.perf_counter())

    def replay(self):
        for text, voice, rate in list(self.last_spoken):
            self.speech_q.put((text, voice, rate, None, None, True))

    def _say(self, text, voice, rate, save_to=None, play=True):
        """Preferred: hand it to speaker.py (started by start.sh from the shell), because `say` launched
        from inside this camera app can freeze. Fallback: run `say` here."""
        path = str(save_to) if save_to else None
        if self.rec and not path:
            self.say_n += 1
            path = str(self.tmp / f"say_{self.say_n}.aiff")
        try:
            r = http_json("http://127.0.0.1:8765/say", {"text": text, "voice": voice, "rate": rate, "file": path,
                                                       "play": play}, timeout=95)
            print(f"speak via helper [{voice or 'default'}] {text[:60]!r}: {'ok' if r.get('ok') else 'FAILED'}", flush=True)
            if r.get("ok") and path and self.rec and play:
                self.rec["audio"].append((time.perf_counter() - self.rec["start"], Path(path)))
            return bool(r.get("ok"))
        except (urllib.error.URLError, ConnectionError, OSError):
            if not getattr(self, "_helper_warned", False):
                self._helper_warned = True
                print("speech helper not running (start with ./start.sh, or run: python speaker.py &)", flush=True)
        if not shutil.which("say"):
            return False
        tries = [voice, voice.split(" (")[0]] if voice and " (" in voice else [voice]
        extra = ["-r", str(rate)] if rate else []
        for v in tries + ([None] if voice else []):
            args = ["say"] + (["-v", v] if v else []) + extra
            try:
                if path:
                    res = run_tool(args + ["-o", path, text], timeout=30)
                    if res.returncode == 0 and play:
                        if self.rec:
                            self.rec["audio"].append((time.perf_counter() - self.rec["start"], Path(path)))
                        run_tool(["afplay", path], timeout=60)
                elif play:
                    res = run_tool(args + [text], timeout=30)
                else:
                    return False
            except subprocess.TimeoutExpired:
                print("speak: `say` hung for 30 s. macOS audio is stuck: run  sudo killall coreaudiod", flush=True)
                return False
            ok = res.returncode == 0 and "not found" not in (res.stderr + res.stdout).lower()
            if ok:
                return True
        return False

    def export_notebook(self):
        if not self.vocab or self.vocab.count() == 0:
            self.info(self.ui["no_words"], "error")
            return

        def work():
            try:
                p = self.vocab.export_anki()
                self.vocab.export_csv()
                self.info(f"{p.name}: {self.vocab.count()} words  ->  {p.parent}")
                if shutil.which("open"):
                    run_tool(["open", "-R", str(p)])
            except ImportError:
                self.info("pip install genanki", "error")
            except Exception as e:
                self.info(f"{type(e).__name__}: {e}"[:160], "error")
        threading.Thread(target=work, daemon=True).start()

    # ---- per-frame work ---------------------------------------------------------------
    def tick(self, frame, now):
        self.frames.push(frame, now)
        while not self.pending_stickers.empty():
            img, box, f = self.pending_stickers.get()
            self.board.add(img, box, f)
        self.board.update(frame)
        self.tick_find(now)
        self.tick_instant(frame, now)
        if self.auto.enabled and self.zone:
            busy = self.thinking or self.listener.active or (self.card is not None and now - self.card_t < 2.0
                                                             and not self.card_corner and self.mode != "find")
            if self.auto.update(frame, self.zone_rect(), self.hand_centers, busy):
                self.ask(frame, auto=True)

    # ---- drawing ------------------------------------------------------------
    def _label_img(self, zh, en, color):
        key = (zh, color)
        if key not in self.label_cache:
            self.label_cache[key] = ui.render_box_label(self.text, zh, chinese.pinyin(zh), en, color)
        return self.label_cache[key]

    def render(self, frame, now, hold_info):
        out = frame.copy()
        if self.frozen is not None:
            still, (fx, fy) = self.frozen
            h, w = still.shape[:2]
            out[fy:fy + h, fx:fx + w] = still
            if time.perf_counter() - self.frozen_t < 0.18:       # little camera-shutter flash
                out[fy:fy + h, fx:fx + w] = cv2.addWeighted(still, 0.5, np.full_like(still, 255), 0.5, 0)
        if self.detector.enabled:
            for x0, y0, x1, y1, en, zh, conf in self.detector.boxes:
                hot = self.point["box"] == zh
                col = (0, 220, 255) if hot else (230, 230, 230)
                cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), col, 3 if hot else 1, cv2.LINE_AA)
                lab = self._label_img(zh, en, col)
                blit(out, lab, int(x0), int(y0) - lab.shape[0] - 2)
            if hot_t := (self.point["box"] and now - self.point["t"]):
                c = tuple(int(v) for v in self.g["tip"]) if self.g else None
                if c and hot_t < 0.9:
                    cv2.ellipse(out, c, (24, 24), -90, 0, 360 * hot_t / 0.9, ACCENT, 4, cv2.LINE_AA)
        if self.zone:
            zx, zy, zw, zh = self.zone_rect()
            red = (40, 40, 235)
            cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), (20, 20, 20), 7, cv2.LINE_AA)
            cv2.rectangle(out, (zx, zy), (zx + zw, zy + zh), red, 4, cv2.LINE_AA)
            tag_txt = "ASK ZONE" + ("  ·  AUTO" if self.auto.enabled else "")
            tag = self.text.label(tag_txt, 22, (255, 255, 255))
            round_rect(out, zx, max(0, zy - 38), zx + tag.shape[1] + 24, max(34, zy - 4), 12, red, 0.95)
            blit(out, tag, zx + 12, max(2, zy - 36))
            if self.auto.progress > 0:                          # "hold still..." ring
                c = (zx + zw // 2, zy + zh // 2)
                cv2.ellipse(out, c, (46, 46), -90, 0, 360, (60, 60, 60), 6, cv2.LINE_AA)
                cv2.ellipse(out, c, (46, 46), -90, 0, 360 * self.auto.progress, ACCENT, 6, cv2.LINE_AA)
        for st in self.board:
            ax, ay = st.anchor
            blit(out, st.img, int(ax - st.img.shape[1] / 2), int(ay - st.img.shape[0]), st.alpha)
        chips = []
        if self.learn:
            chips.append(f"LEARN  {'中文' if self.learn == 'zh' else 'EN'}")
        if self.mode == "read":
            chips.append("READ  读")
        if self.mode == "find":
            chips.append("FIND  找")
        if self.mode == "write":
            chips.append("WRITE  写")
        if self.mode == "interpret":
            chips.append("INTERPRET  翻译")
        if self.detector.enabled:
            chips.append("INSTANT" if self.detector.model else
                         f"INSTANT: {self.detector.error}" if self.detector.error else "INSTANT loading...")
        if self.vocab:
            chips.append(f"{self.vocab.count()} words")
        x = 20
        for ch in chips:
            lab = self.text.label(ch, 22, (255, 255, 255))
            round_rect(out, x, 20, x + lab.shape[1] + 24, 62, 21, self.ink if ch.startswith(("LEARN", "READ", "FIND", "WRITE", "INTERPRET"))
                       else (60, 60, 60), 0.9)
            blit(out, lab, x + 12, 26)
            x += lab.shape[1] + 34
        if self.lens:
            ui.draw_lens(out, self.text, self.lens["items"])
            v = self.lens["verdict"]
            if v:
                msg = {"haram": "NOT HALAL: ", "doubtful": "CHECK: ", "halal": "清真 HALAL "}[v] + \
                      "  ".join(f"{t} {m}" for t, m, lv in self.lens["hits"] if v != "haram" or lv == "haram")[:60]
                lab = self.text.label(msg, 24, (255, 255, 255))
                ox, oy = self.lens["origin"]
                col = {"haram": (40, 40, 200), "doubtful": (20, 140, 230), "halal": (60, 150, 60)}[v]
                round_rect(out, ox, max(0, oy - 44), ox + lab.shape[1] + 24, max(40, oy - 4), 14, col, 0.95)
                blit(out, lab, ox + 12, max(2, oy - 42))
        for s in self.strokes:
            if len(s) > 1 and self.mode == "write":
                hanzi.brush(out, s, (60, 200, 255))
                continue
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
            lbl = self.text.label(msg[:60], 22, col)
            sx = W - lbl.shape[1] - 40                          # bottom-right: out of the cards' way
            round_rect(out, sx - 12, H - 92, sx + lbl.shape[1] + 12, H - 52, 14, (16, 16, 18), 0.8)
            blit(out, lbl, sx, H - 88)
        if self.listener.active:
            pill = self.text.label(self.ui["say_it" if self.listen_purpose == "practice" else "listening"], 28, (255, 255, 255))
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
        if self.card is not None and not (self.thinking and self.mode != "find"):
            fade = min(1.0, (now - self.card_t) / 0.25)
            ch, cw = self.card.shape[:2]
            bx, by, bw, bh = self.band
            y = by + bh + 40 if H > W else H - ch - 40
            x = (W - cw) // 2
            if self.card_corner:
                age = now - self.card_t                   # word cards fade, leaving the sticker on the object
                fade = min(fade, max(0.0, min(1.0, self.card_life - age)))
                if W > H:
                    x = 28 if self.card_left else W - cw - 28
                    y = H - ch - 110
            elif self.mode == "interpret":
                y = H - ch - 30                                     # subtitles: bottom center
            elif W > H and (self.zone or self.mode):
                x = 28 if self.card_left else W - cw - 28             # keep cards clear of the object / red box
            y = max(70, min(y, H - ch - 10))
            if self.frozen is not None and self.card_corner and now - self.card_t > self.card_life:
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
    clips = [(o, p) for o, p in rec["audio"] if o < dur and Path(p).exists()]
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
    import re
    if not shutil.which("say"):
        return None
    try:
        out = run_tool(["say", "-v", "?"]).stdout
    except Exception:
        return None
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
    """Lay the camera frame onto the W x H canvas. Returns (canvas, camera_view, band)."""
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


def run(app, cam_index, prefer_hand="right"):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
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
        app.hand_centers = [h[[0, 5, 9, 13, 17]].mean(axis=0) for h in (hand, other) if h is not None]
        ev = app.update_hand(hand, dt, other)
        app.tick(frame, now)
        hold_at = None
        if ev:
            fired, hold_at = ev
            if fired == "ask":
                app.ask(frame)
            elif fired == "clear":
                app.clear()
        out = app.render(frame, now, hold_at)

        if app.rec:
            due = int((now - app.rec["start"]) * 30) - app.rec["frames"]
            for _ in range(max(0, due)):
                app.rec["writer"].write(out)
                app.rec["frames"] += 1

        view = out.copy()
        if now - t0 < 25 and not app.strokes and app.card is None:
            hint = app.text.label(app.ui["hint"], 22, (235, 235, 235))
            blit(view, hint, (W - hint.shape[1]) // 2, 72)
        model = app.vlm.model if app.vlm else "no model"
        cv2.circle(view, (24, H - 26), 6, (90, 220, 120) if app.vlm else (60, 60, 230), -1, cv2.LINE_AA)
        cv2.putText(view, f"{'SNAP  ' if app.snap else ''}{app.lang.upper()}  [L]  {Path(model).name[:34]}  {fps:4.1f} fps",
                    (38, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        if app.rec:
            cv2.circle(view, (W - 110, 34), 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(view, f"REC {int(now - app.rec['start'])}s", (W - 92, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, view)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if not handle_key(app, k, frame, out, now):
            continue
        if k == ord("f"):
            full = not full
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        elif k == ord("l"):
            t0 = now - 20   # flash the hint in the new language for a few seconds
    if app.rec:
        finish_recording(app.rec)
    cap.release()
    cv2.destroyAllWindows()


def handle_key(app, k, frame, out, now):
    """All keyboard actions except window ones. Returns True when the key was something."""
    if k == 255:
        return False
    c = chr(k) if k < 128 else ""
    if c == " ":
        app.ask(frame)
    elif c == "c":
        app.clear()
    elif c == "v":
        app.toggle_listen("interpret" if app.mode == "interpret" else "ask")
    elif c == "w":
        app.toggle_mode("write")
    elif c == "b":
        app.toggle_mode("interpret")
    elif c == "u":
        app.make_story()
    elif c == "y":
        app.toggle_listen("practice")
    elif c == "t":
        app.cycle_learn()
    elif c == "o":
        app.toggle_mode("read")
        if app.mode == "read" and not ocr.available():
            print("Tip: pip install ocrmac  (Apple's OCR reads Chinese labels much better)", flush=True)
    elif c == "h":
        app.toggle_mode("find")
    elif c == "i":
        on = app.detector.toggle()
        if on and not app.detector.installed():
            app.info("pip install ultralytics", "error")
    elif c == "a":
        app.auto.enabled = not app.auto.enabled
        app.auto.forget()
    elif c in "123":
        app.choose(int(c) - 1)
    elif c == "n":
        app.export_notebook()
    elif c == "k":
        app.ink_i = (app.ink_i + 1) % len(INKS)
    elif c == "x":
        app.board.clear()
    elif c == "g":
        app.snap = not app.snap
    elif c == "z":
        app.cycle_zone()
    elif c == "p":
        app.replay()
    elif c == "m":
        app.muted = not app.muted
    elif c == "l":
        app.set_lang("en" if app.lang == "ar" else "ar")
        app.card = None
    elif c == "s":
        p = OUT_DIR / f"airmind_{time.strftime('%H%M%S')}.png"
        cv2.imwrite(str(p), out)
        print("Saved", p)
    elif c == "r":
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
    return True


def learn_from_image(vlm, path, target="zh"):
    """--learn-image: the whole word pipeline on one photo, no camera (handy for testing a model)."""
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Can't read {path}")
    k = 512 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    t = time.perf_counter()
    raw = vlm.ask_learn(jpeg, False, target, focus=True)
    cands, ex, ex_tr = parse_learn(raw)
    print(f"answered in {time.perf_counter() - t:.1f}s\nraw: {raw}")
    for c in cands:
        v = chinese.verify(c["word"]) if target == "zh" else {}
        print(f"  {c['word']:<8} {v.get('pinyin', ''):<14} conf {c['confidence']:.2f}  "
              f"dict: {'yes' if v.get('found') else 'NO'}  {v.get('en_dict') or c['en']}  HSK {v.get('hsk') or '-'}")
    best = chinese.best_candidate(cands) if target == "zh" else (cands[0] if cands else None)
    print("chosen:", best and best["word"], "| example:", ex, "|", chinese.sentence_pinyin(ex), "|", ex_tr)


def main():
    global W, H
    ap = argparse.ArgumentParser(description="AirMind: draw in the air, a local AI answers")
    ap.add_argument("--model", help="part of the model name to use (default: best vision model found)")
    ap.add_argument("--lang", choices=["ar", "en"], default="ar", help="answer language (press L while running to switch)")
    ap.add_argument("--voice", help="macOS voice name (default: an Arabic voice if installed)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--hand", choices=["right", "left", "any"], default="right",
                    help="the hand you draw with; the other hand gives commands (default right)")
    ap.add_argument("--story", action="store_true", help="vertical 9:16 output for Instagram stories (default: wide 16:9)")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--stt", default="base", help="speech model: tiny | base | small | off  (default base, ~145 MB)")
    ap.add_argument("--learn", choices=["zh", "en"], help="start in learn mode")
    ap.add_argument("--auto", action="store_true", help="start with auto-ask on (hold an object still in the red box)")
    ap.add_argument("--list", action="store_true", help="list the models found and exit")
    ap.add_argument("--ask-image", help="send a photo to the model once and print the answer (test without camera)")
    ap.add_argument("--learn-image", help="run the Chinese word pipeline on a photo and print it (test without camera)")
    a = ap.parse_args()
    W, H = (720, 1280) if a.story else (1280, 720)
    ui.SIZE[:] = [W, H]
    chinese.preload()

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
        print("Found models, but none looks like a vision model. Get one with:  ollama pull qwen3-vl:4b")
        print("Or force one with --model NAME. Found:", ", ".join(f[2] for f in found))
    else:
        print("No local AI server found. Start Ollama (or LM Studio's server) and run again.")

    if a.ask_image or a.learn_image:
        if not vlm:
            sys.exit(1)
        if a.learn_image:
            learn_from_image(vlm, a.learn_image, a.learn or "zh")
            return
        img = cv2.imread(a.ask_image)
        k = 1024 / max(img.shape[:2])
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        t = time.perf_counter()
        raw = vlm.ask(jpeg, False)
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
    print("Apple OCR:", "yes" if ocr.available() else "no (pip install ocrmac)",
          "| Instant mode:", "yes" if InstantDetector.installed() else "no (pip install ultralytics)", flush=True)
    vocab = Vocab()
    print(f"Notebook: {vocab.count()} words in {vocab.root}", flush=True)
    app = AirMind(vlm, a.lang, voices, a.mute, vocab)
    app.learn = a.learn
    app.auto.enabled = a.auto
    app.stt = SpeechToText(a.stt) if a.stt != "off" else None
    if not a.mute:
        app.speak("Ready" if a.lang == "en" else "جاهز")
    run(app, a.cam, a.hand)


if __name__ == "__main__":
    main()
