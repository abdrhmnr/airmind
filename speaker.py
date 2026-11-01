#!/usr/bin/env python3
"""
Tiny speech helper for AirMind.

`say` freezes when launched from inside the busy camera app, so start.sh runs this small program
from the shell and AirMind sends it the text over http://127.0.0.1:8765.

Some installed voices hang forever (e.g. a Tingting variant whose files never finished
downloading). So each voice gets a short try; one that hangs is skipped from then on and the
next voice for the same language is used.

    python speaker.py          (start.sh does this for you)
"""
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8765
SAY = os.environ.get("AIRMIND_SAY", "/usr/bin/say")
AFPLAY = os.environ.get("AIRMIND_AFPLAY", "/usr/bin/afplay")
TRY_SECONDS = 8                       # a voice that can't render a word in 8 s is treated as broken

FALLBACKS = {
    "zh": ["Tingting", "Meijia", "Eddy (Chinese (China mainland))", "Flo (Chinese (China mainland))",
           "Reed (Chinese (China mainland))", "Sandy (Chinese (China mainland))"],
    "en": ["Samantha", "Alex", "Eddy (English (US))", "Flo (English (US))", "Reed (English (US))"],
    "ar": ["Majed", "Maged"],
}
broken = set()
jobs = queue.Queue()
tmpdir = tempfile.mkdtemp(prefix="airmind_speaker_")


def language_of(text, voice):
    if re.search(r"[㐀-鿿]", text) or (voice and "Chinese" in voice):
        return "zh"
    if re.search(r"[؀-ۿ]", text) or (voice and "Arabic" in voice):
        return "ar"
    return "en"


def candidates(text, voice):
    out = []
    if voice:
        short = voice.split(" (")[0]
        # "Tingting (Chinese (China mainland))" and "Tingting" are the same voice: if one hung, skip both
        if short not in broken and voice not in broken:
            out += [voice, short]
    out += FALLBACKS[language_of(text, voice)]
    seen, order = set(), []
    for v in out:
        if v not in seen and v not in broken:
            seen.add(v)
            order.append(v)
    return order + [None]             # None = the Mac's default voice, last resort


def say(text, voice=None, rate=None, file=None, play=True):
    for v in candidates(text, voice):
        out = file or os.path.join(tmpdir, "speech.aiff")
        cmd = [SAY] + (["-v", v] if v else []) + (["-r", str(rate)] if rate else []) + ["-o", out, text]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=TRY_SECONDS)
        except subprocess.TimeoutExpired:
            print(f"[speaker] voice {v!r} hung, skipping it from now on", flush=True)
            broken.add(v)
            if v and voice and v == voice:
                broken.add(voice.split(" (")[0])
            continue
        except OSError as e:
            print(f"[speaker] can't run say: {e}", flush=True)
            return False
        if res.returncode != 0 or "not found" in (res.stderr + res.stdout).lower():
            print(f"[speaker] voice {v!r} unavailable ({res.stderr.strip()})", flush=True)
            broken.add(v)
            continue
        if not play:                  # warm-up: just proving this voice works
            print(f"[speaker] voice {v!r} works", flush=True)
            return True
        try:
            played = subprocess.run([AFPLAY, out], capture_output=True, text=True, timeout=60)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"[speaker] couldn't play the sound: {e}", flush=True)
            return False
        if played.returncode != 0 or "error" in (played.stderr + played.stdout).lower():
            # the voice is fine, the Mac's audio output itself refused (e.g. AudioQueueStart failed)
            print(f"[speaker] Mac audio output FAILED: {(played.stderr or played.stdout).strip()}", flush=True)
            return False
        print(f"[speaker] [{v or 'default'}] {text[:50]!r}: ok", flush=True)
        return True
    return False


def worker():
    while True:
        job, done = jobs.get()
        try:
            done["ok"] = say(**job)
        except Exception as e:
            print(f"[speaker] error: {e}", flush=True)
        done["event"].set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._reply({"status": "ok", "skipped_voices": sorted(v for v in broken if v)})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        job = {k: data.get(k) for k in ("text", "voice", "rate", "file")}
        done = {"event": threading.Event(), "ok": False}
        jobs.put((job, done))
        done["event"].wait(90)
        self._reply({"ok": done["ok"]})


def warm_up():
    """Find the stuck voices before the first question, so answers aren't delayed by them."""
    for text, voice in (("你好", "Tingting"), ("hi", "Samantha")):
        done = {"event": threading.Event(), "ok": False}
        jobs.put(({"text": text, "voice": voice, "play": False}, done))


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    warm_up()
    print(f"[speaker] ready on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
