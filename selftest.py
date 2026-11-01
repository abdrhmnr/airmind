#!/usr/bin/env python3
"""Checks every part of AirMind without the camera.   python selftest.py"""
import importlib
import time

import numpy as np

OK, BAD, WARN = "  ok  ", " FAIL ", " note "


def line(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def main():
    fails = 0
    for mod, why in (("cv2", "camera"), ("mediapipe", "hands"), ("PIL", "text"), ("sympy", "math"),
                     ("pypinyin", "pinyin"), ("genanki", "Anki export"), ("faster_whisper", "speech"),
                     ("sounddevice", "microphone"), ("arabic_reshaper", "Arabic text"), ("bidi", "Arabic text")):
        try:
            importlib.import_module(mod)
            line(OK, f"{mod:<16} ({why})")
        except Exception as e:
            fails += 1
            line(BAD, f"{mod:<16} ({why}): {e}   ->  pip install -r requirements.txt")
    import cv2
    line(OK if hasattr(cv2, "TrackerCSRT_create") or hasattr(getattr(cv2, "legacy", None), "TrackerCSRT_create") else WARN,
         "CSRT tracker (stickers follow objects)" + ("" if hasattr(cv2, "TrackerCSRT_create") else
                                                      ": needs opencv-contrib-python"))

    import chinese
    t = time.perf_counter()
    n = chinese.dictionary_size()
    v = chinese.verify("杯子")
    good = n > 100000 and v["pinyin"] == "bēizi" and v["measure"] == ("个", "gè") and v["hsk"] == 1
    fails += not good
    line(OK if good else BAD, f"CC-CEDICT {n} entries in {time.perf_counter() - t:.1f}s; 杯子 = {v['pinyin']}, "
                              f"{v['en_dict']}, 量词 {v['measure']}, HSK {v['hsk']}")
    line(OK, "sentence pinyin: 我有一个杯子 -> " + chinese.sentence_pinyin("我有一个杯子"))
    s, _ = chinese.compare_pronunciation("杯子", "被子")
    line(OK if 0.5 < s < 1 else BAD, f"pronunciation check: 杯子 vs 被子 = {s:.0%} (tone mistake caught)")
    verdict, _ = chinese.halal_check("配料：小麦粉，猪油，白砂糖")
    line(OK if verdict == "haram" else BAD, f"halal check on '配料：小麦粉，猪油' -> {verdict}")

    import ocr
    if ocr.available():
        img = np.full((160, 560, 3), 255, np.uint8)
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        try:
            f = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 64)
            ImageDraw.Draw(pil).text((20, 40), "配料：猪油", font=f, fill=(0, 0, 0))
            got = ocr.read_text(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR))
            line(OK if got and "猪" in got else WARN, f"Apple OCR read: {got!r}")
        except OSError:
            line(OK, "Apple OCR available")
    else:
        line(WARN, "Apple OCR not available (pip install ocrmac). READ mode will use the vision model only.")

    from vocab import Vocab
    try:
        voc = Vocab()
        line(OK, f"notebook: {voc.count()} words in {voc.root}")
    except Exception as e:
        fails += 1
        line(BAD, f"notebook: {e}")

    from detector import InstantDetector
    line(OK if InstantDetector.installed() else WARN,
         "INSTANT mode " + ("ready" if InstantDetector.installed() else "not installed (optional: ./setup.sh instant)"))

    from vlm import LocalVLM, discover_models, parse_learn, pick_model
    found = discover_models()
    choice = pick_model(found)
    if not choice:
        line(WARN, "no vision model running right now (start.sh starts one); skipped the model test")
    else:
        vlm = LocalVLM(*choice, "en")
        img = np.full((384, 384, 3), 235, np.uint8)
        cv2.circle(img, (192, 192), 120, (30, 30, 200), -1)          # a big red ball
        jpeg = cv2.imencode(".jpg", img)[1].tobytes()
        t = time.perf_counter()
        try:
            cands, ex, tr = parse_learn(vlm.ask_learn(jpeg, False, "zh", focus=True))
            best = chinese.best_candidate(cands)
            line(OK if cands else BAD, f"{choice[2]} ({choice[0]}): {[c['word'] for c in cands]} -> {best and best['word']} "
                                       f"in {time.perf_counter() - t:.1f}s; JSON schema "
                                       f"{'accepted' if vlm.schema_ok else 'refused (plain JSON fallback)'}")
        except Exception as e:
            fails += 1
            line(BAD, f"{choice[2]}: {type(e).__name__}: {e}")
    print("\nAll good." if not fails else f"\n{fails} problem(s) above.")


if __name__ == "__main__":
    main()
