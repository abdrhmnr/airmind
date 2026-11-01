#!/usr/bin/env python3
"""
Vocabulary notebook: every word you learn in front of the camera is kept here, with the photo of
YOUR object and the Mac's pronunciation, so you can review it later.

  vocab/vocab.db        SQLite: words, pinyin, meanings, review stats
  vocab/img/*.jpg       the close-up the model looked at
  vocab/audio/*.m4a     the spoken word

Command line:
  python vocab.py list              show your words
  python vocab.py anki              make vocab/AirMind.apkg  (double-click it to import into Anki)
  python vocab.py csv               make vocab/vocab.csv
"""
import csv
import hashlib
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "vocab"

SCHEMA = """
CREATE TABLE IF NOT EXISTS words (
    word TEXT PRIMARY KEY, lang TEXT, pinyin TEXT, en TEXT, ar TEXT, measure TEXT, hsk INTEGER,
    example TEXT, example_pinyin TEXT, example_en TEXT, verified INTEGER,
    image TEXT, audio TEXT, created REAL, seen INTEGER DEFAULT 1,
    found_ok INTEGER DEFAULT 0, found_miss INTEGER DEFAULT 0,
    say_ok INTEGER DEFAULT 0, say_miss INTEGER DEFAULT 0,
    due REAL DEFAULT 0, interval REAL DEFAULT 0
);
"""


class Vocab:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        (self.root / "img").mkdir(parents=True, exist_ok=True)
        (self.root / "audio").mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.root / "vocab.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    @staticmethod
    def slug(word):
        return hashlib.md5(word.encode()).hexdigest()[:10]

    def image_path(self, word):
        return self.root / "img" / f"{self.slug(word)}.jpg"

    def audio_path(self, word, ext="aiff"):
        return self.root / "audio" / f"{self.slug(word)}.{ext}"

    def add(self, entry, jpeg=None):
        """entry: dict with word, lang, pinyin, en, ar, measure, hsk, example, example_pinyin, example_en, verified.
        Adding a word you already have just counts it again (and keeps the newer photo)."""
        w = entry["word"]
        img = None
        if jpeg is not None:
            img = self.image_path(w)
            img.write_bytes(jpeg)
        with self.lock:
            old = self.db.execute("SELECT word FROM words WHERE word=?", (w,)).fetchone()
            cols = ("lang", "pinyin", "en", "ar", "measure", "hsk", "example", "example_pinyin", "example_en", "verified")
            vals = [entry.get(c) for c in cols]
            if old:
                sets = ", ".join(f"{c}=COALESCE(NULLIF(?, ''), {c})" for c in cols)
                self.db.execute(f"UPDATE words SET {sets}, seen=seen+1, image=COALESCE(?, image) WHERE word=?",
                                (*vals, str(img) if img else None, w))
            else:
                self.db.execute(f"INSERT INTO words (word, {', '.join(cols)}, image, created, due) VALUES (?{', ?' * len(cols)}, ?, ?, ?)",
                                (w, *vals, str(img) if img else None, time.time(), time.time()))
            self.db.commit()
        return not old

    def remove(self, word):
        with self.lock:
            self.db.execute("DELETE FROM words WHERE word=?", (word,))
            self.db.commit()

    def set_audio(self, word, path):
        """Store the spoken word. On macOS the .aiff is turned into a small .m4a that Anki plays everywhere."""
        path = Path(path)
        if not path.exists():
            return
        final = path
        if shutil.which("afconvert"):
            m4a = self.audio_path(word, "m4a")
            r = subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(path), str(m4a)], capture_output=True)
            if r.returncode == 0:
                path.unlink(missing_ok=True)
                final = m4a
        with self.lock:
            self.db.execute("UPDATE words SET audio=? WHERE word=?", (str(final), word))
            self.db.commit()

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM words").fetchone()[0]

    def all(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM words ORDER BY created")]

    def get(self, word):
        r = self.db.execute("SELECT * FROM words WHERE word=?", (word,)).fetchone()
        return dict(r) if r else None

    # ---- review (spaced repetition, Leitner-style) -------------------------------------------------
    def next_challenge(self, avoid=None):
        """The word that's most due; ties go to the one you get wrong most."""
        rows = self.db.execute(
            "SELECT * FROM words WHERE word != ? ORDER BY (due > ?), due, (found_miss + say_miss) DESC, RANDOM() LIMIT 1",
            (avoid or "", time.time())).fetchone()
        if rows is None and avoid:
            rows = self.db.execute("SELECT * FROM words LIMIT 1").fetchone()
        return dict(rows) if rows else None

    def result(self, word, ok, kind="found"):
        """kind 'found' = you showed the right object; 'say' = you pronounced it right."""
        r = self.get(word)
        if not r:
            return
        iv = r["interval"] or 0
        iv = max(600.0, iv * 2.5) if ok else 0.0         # right: wait longer (10 min, 25 min, 1 h ...); wrong: again soon
        col = f"{kind}_{'ok' if ok else 'miss'}"
        with self.lock:
            self.db.execute(f"UPDATE words SET {col}={col}+1, interval=?, due=? WHERE word=?",
                            (iv, time.time() + (iv if ok else 30), word))
            self.db.commit()

    # ---- export ----------------------------------------------------------------------------------
    def export_csv(self, path=None):
        path = Path(path or self.root / "vocab.csv")
        rows = self.all()
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["word", "pinyin", "english", "arabic", "measure", "hsk", "example", "example_pinyin",
                        "example_english", "times_seen", "found_ok", "found_miss", "say_ok", "say_miss"])
            for r in rows:
                w.writerow([r["word"], r["pinyin"], r["en"], r["ar"], r["measure"], r["hsk"], r["example"],
                            r["example_pinyin"], r["example_en"], r["seen"], r["found_ok"], r["found_miss"],
                            r["say_ok"], r["say_miss"]])
        return path

    def export_anki(self, path=None):
        """Two cards per word: photo -> Chinese (recognise), and sound -> meaning (listen)."""
        import genanki
        path = Path(path or self.root / "AirMind.apkg")
        model = genanki.Model(
            1607392319, "AirMind word",
            fields=[{"name": n} for n in ("Word", "Pinyin", "English", "Arabic", "Measure", "Example",
                                           "ExamplePinyin", "ExampleEnglish", "Picture", "Sound", "HSK")],
            templates=[
                {"name": "Picture -> Chinese",
                 "qfmt": '<div class="pic">{{Picture}}</div><div class="hint">这是什么？ What is this?</div>',
                 "afmt": '{{FrontSide}}<hr id="answer"><div class="word">{{Word}}</div><div class="py">{{Pinyin}}</div>'
                         '{{Sound}}<div class="mean">{{English}} · {{Arabic}}</div><div class="cl">{{Measure}}</div>'
                         '<div class="ex">{{Example}}<br><span class="py2">{{ExamplePinyin}}</span><br>{{ExampleEnglish}}</div>'},
                {"name": "Listen -> meaning",
                 "qfmt": '<div class="hint">听 Listen</div>{{Sound}}',
                 "afmt": '{{FrontSide}}<hr id="answer"><div class="word">{{Word}}</div><div class="py">{{Pinyin}}</div>'
                         '<div class="pic">{{Picture}}</div><div class="mean">{{English}} · {{Arabic}}</div>'},
            ],
            css=""".card{font-family:-apple-system,"PingFang SC",sans-serif;text-align:center;background:#111;color:#eee}
.pic img{max-width:90%;max-height:320px;border-radius:14px}.word{font-size:64px;margin-top:8px}
.py{font-size:28px;color:#f5c45c}.mean{font-size:22px;margin:8px}.cl{color:#9ad;font-size:20px}
.ex{color:#aaa;font-size:18px;margin-top:10px}.py2{color:#c9a}.hint{color:#888;font-size:18px;margin:10px}""")
        deck = genanki.Deck(2059400110, "AirMind 中文")
        media = []
        for r in self.all():
            pic = snd = ""
            if r["image"] and Path(r["image"]).exists():
                media.append(r["image"])
                pic = f'<img src="{Path(r["image"]).name}">'
            if r["audio"] and Path(r["audio"]).exists():
                media.append(r["audio"])
                snd = f"[sound:{Path(r['audio']).name}]"
            deck.add_note(genanki.Note(model=model, guid=genanki.guid_for(r["word"]), fields=[
                r["word"], r["pinyin"] or "", r["en"] or "", r["ar"] or "", r["measure"] or "", r["example"] or "",
                r["example_pinyin"] or "", r["example_en"] or "", pic, snd, str(r["hsk"] or "")]))
        pkg = genanki.Package(deck)
        pkg.media_files = media
        pkg.write_to_file(str(path))
        return path


if __name__ == "__main__":
    v = Vocab()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "anki":
        print("Saved", v.export_anki(), f"({v.count()} words). Double-click it to add the cards to Anki.")
    elif cmd == "csv":
        print("Saved", v.export_csv())
    else:
        for r in v.all():
            print(f"{r['word']:<8} {r['pinyin'] or '':<16} {r['en'] or '':<28} {r['ar'] or '':<14} "
                  f"HSK {r['hsk'] or '-'}  seen {r['seen']}  found {r['found_ok']}/{r['found_ok'] + r['found_miss']}  "
                  f"said {r['say_ok']}/{r['say_ok'] + r['say_miss']}")
        print(f"{v.count()} words")
