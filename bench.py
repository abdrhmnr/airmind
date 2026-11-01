#!/usr/bin/env python3
"""
Which vision model is best for YOUR room? Measure it instead of guessing.

1) Make a test set (photos + the right Chinese word):
     python bench.py make
   copies the photos from your notebook into bench/ and writes bench/labels.csv.
   OPEN labels.csv and fix any wrong word (the notebook's words came from a model, so they can be wrong).
   You can also drop your own photos into bench/ and add a line "file,word" for each.

2) Run the models you have (Ollama names, or any part of a name from `python airmind.py --list`):
     python bench.py run qwen2.5vl:3b qwen3-vl:2b qwen3-vl:4b

   For each model: top-1 accuracy (the word the app would show), top-3 (the right word is among the
   1/2/3 choices), how often the answer is a real dictionary word, and seconds per photo.
   Results are also saved to bench/results.csv.
"""
import csv
import shutil
import statistics
import sys
import time
from pathlib import Path

import cv2

import chinese
from vlm import LocalVLM, discover_models, parse_learn, pick_model
from vocab import Vocab

HERE = Path(__file__).resolve().parent
BENCH = HERE / "bench"


def make():
    BENCH.mkdir(exist_ok=True)
    labels = BENCH / "labels.csv"
    have = {}
    if labels.exists():
        with open(labels, encoding="utf-8-sig") as f:
            have = {r["file"]: r["word"] for r in csv.DictReader(f)}
    v = Vocab()
    n = 0
    for r in v.all():
        if r["lang"] != "zh" or not r["image"] or not Path(r["image"]).exists():
            continue
        name = Path(r["image"]).name
        shutil.copy(r["image"], BENCH / name)
        if name not in have:
            have[name] = r["word"]
            n += 1
    with open(labels, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "word"])
        for k, word in sorted(have.items()):
            w.writerow([k, word])
    print(f"{len(have)} photos in {BENCH} ({n} new). Check the words in {labels} before running.")


def load_set():
    labels = BENCH / "labels.csv"
    if not labels.exists():
        sys.exit("No bench/labels.csv yet: run  python bench.py make  first.")
    with open(labels, encoding="utf-8-sig") as f:
        rows = [(BENCH / r["file"], r["word"].strip()) for r in csv.DictReader(f) if r.get("word")]
    return [(p, w) for p, w in rows if p.exists()]


def run(names):
    data = load_set()
    if not data:
        sys.exit("bench/labels.csv has no photos that exist.")
    found = discover_models()
    results = []
    for name in names:
        choice = pick_model(found, name)
        if not choice:
            print(f"!! {name}: not found (is it downloaded? `ollama pull {name}`)")
            continue
        vlm = LocalVLM(*choice, "en")
        vlm.warm_up()
        top1 = top3 = real = 0
        times, misses = [], []
        for path, word in data:
            img = cv2.imread(str(path))
            k = 512 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            t = time.perf_counter()
            try:
                cands, _, _ = parse_learn(vlm.ask_learn(jpeg, False, "zh", focus=True))
            except Exception as e:
                cands = []
                print(f"   {path.name}: {type(e).__name__}: {e}")
            times.append(time.perf_counter() - t)
            best = chinese.best_candidate(cands)
            words = [c["word"] for c in cands]
            hit1 = bool(best) and best["word"] == word
            top1 += hit1
            top3 += word in words
            real += bool(best) and chinese.lookup(best["word"]) is not None
            if not hit1:
                misses.append(f"{path.name}: wanted {word}, got {' / '.join(words) or '-'}")
            print(f"  {choice[2][:28]:<28} {path.name:<20} {word:<6} -> {' / '.join(words)}  ({times[-1]:.1f}s)")
        n = len(data)
        results.append({"model": choice[2], "photos": n, "top1": top1 / n, "top3": top3 / n, "real_word": real / n,
                        "sec_median": statistics.median(times), "sec_max": max(times)})
        if misses:
            print("  misses:\n    " + "\n    ".join(misses[:15]))
    if not results:
        return
    print(f"\n{'model':<30}{'top-1':>8}{'top-3':>8}{'real word':>11}{'sec/photo':>11}")
    for r in sorted(results, key=lambda r: (-r["top1"], r["sec_median"])):
        print(f"{r['model'][:29]:<30}{r['top1']:>8.0%}{r['top3']:>8.0%}{r['real_word']:>11.0%}{r['sec_median']:>11.1f}")
    out = BENCH / "results.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time"] + list(results[0]))
        if new:
            w.writeheader()
        for r in results:
            w.writerow({"time": time.strftime("%Y-%m-%d %H:%M"), **r})
    print("saved", out)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("make", "run"):
        print(__doc__)
    elif sys.argv[1] == "make":
        make()
    else:
        run(sys.argv[2:] or ["qwen2.5vl:3b", "qwen3-vl:2b", "qwen3-vl:4b"])
