"""
Listening: record from the mic until you stop talking, then turn it into text with faster-whisper (local, CPU).
"""
import threading
import time

import numpy as np


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

    def transcribe(self, audio, allowed=("ar", "en"), force=None):
        """Speech -> text. The language is chosen only among `allowed` (e.g. Arabic/English, or Chinese in
        learn mode): short clips are easy to mis-detect as some other language, which gives garbage text.
        `force` skips detection (pronunciation practice is always Chinese).
        Returns (text, language) or (None, None)."""
        for _ in range(600):                                       # first run may still be downloading
            if self.model or self.error:
                break
            time.sleep(0.1)
        if not self.model:
            return None, None
        lang = force or self.pick_language(audio, allowed)
        kw = dict(beam_size=1 if lang != "zh" else 3, vad_filter=False, condition_on_previous_text=False)
        if lang == "zh":
            kw["initial_prompt"] = "以下是普通话的句子。"           # nudges Whisper to simplified characters
        segments, info = self.model.transcribe(audio, language=lang, **kw)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, lang or getattr(info, "language", None)

    def pick_language(self, audio, allowed):
        if not allowed:
            return None
        if len(allowed) == 1:
            return allowed[0]
        try:
            probs = dict(self.model.detect_language(audio)[2])        # faster-whisper >= 1.1
        except Exception:
            try:
                _, info = self.model.transcribe(audio[: self.RATE * 8], beam_size=1, without_timestamps=True)
                probs = dict(info.all_language_probs or [])
            except Exception:
                return allowed[0]
        return max(allowed, key=lambda l: probs.get(l, 0.0))

    RATE = 16000


