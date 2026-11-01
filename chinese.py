"""
Chinese helpers that do NOT trust the vision model for facts.

The model only proposes a word ("杯子"). Everything a learner relies on is computed or checked here:
  - pinyin + tones        -> CC-CEDICT, else pypinyin (never the model)
  - is it a real word?    -> CC-CEDICT (about 120k entries, offline, data/cedict.txt.gz)
  - measure word (量词)   -> the CL: field of CC-CEDICT
  - HSK level             -> official HSK 3.0 word lists (data/hsk3.json)
  - pinyin of a sentence  -> pypinyin
  - halal check of a label -> fixed keyword list below
  - pronunciation check   -> compare the tones of what Whisper heard with the target word

Same idea as SymPy for the math: the model reads, the tools are the source of truth.
"""
import difflib
import gzip
import json
import re
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

HAN = re.compile(r"[㐀-鿿豈-﫿]")
TONE_MARKS = {"a": "āáǎàa", "e": "ēéěèe", "i": "īíǐìi", "o": "ōóǒòo", "u": "ūúǔùu", "ü": "ǖǘǚǜü"}
# Pleco-style tone colors (BGR): 1 red, 2 orange, 3 green, 4 blue, neutral grey
TONE_COLORS = {1: (60, 60, 235), 2: (40, 150, 255), 3: (80, 200, 90), 4: (235, 140, 50), 5: (170, 170, 170)}


def has_han(s):
    return bool(s and HAN.search(s))


def han_only(s):
    return "".join(HAN.findall(s or ""))


# --------------------------------------------------------------------------- pinyin
def numbered_to_marks(syl):
    """'bei1' -> 'bēi', 'lu:4' / 'lv4' -> 'lǜ', 'zi5' -> 'zi'. Anything else is returned unchanged."""
    m = re.fullmatch(r"([A-Za-züÜ:v]+?)([1-5])", syl)
    if not m:
        return syl
    base, tone = m.group(1), int(m.group(2))
    base = base.replace("u:", "ü").replace("U:", "Ü").replace("v", "ü")
    if tone == 5:
        return base
    low = base.lower()
    if "a" in low:
        i = low.index("a")
    elif "e" in low:
        i = low.index("e")
    elif "ou" in low:
        i = low.index("o")
    else:
        i = max(low.rfind(v) for v in "iouü")
        if i < 0:
            return base
    ch = low[i]
    marked = TONE_MARKS[ch][tone - 1]
    if base[i].isupper():
        marked = marked.upper()
    return base[:i] + marked + base[i + 1:]


def _pypinyin(word, style):
    from pypinyin import Style, lazy_pinyin
    st = {"tone3": Style.TONE3, "tone": Style.TONE}[style]
    kw = dict(style=st, errors="ignore")
    if style == "tone3":
        kw["neutral_tone_with_five"] = True
    return lazy_pinyin(word, **kw)                      # 一 / 不 tone changes: see sandhi() below


def numbered(word):
    """Pinyin with tone numbers per Han character, e.g. 杯子 -> ['bei1', 'zi5']."""
    e = lookup(word)
    if e and len(e["numbered"]) == len(han_only(word)):
        return e["numbered"]
    return [s.replace("ü", "v") for s in _pypinyin(han_only(word), "tone3")]


def tones(word):
    out = []
    for s in numbered(word):
        m = re.search(r"([1-5])$", s)
        out.append(int(m.group(1)) if m else 5)
    return out


def pinyin(word, spaced=False):
    """Tone-marked pinyin for a word or a sentence. Words: dictionary pronunciation. Sentences: pypinyin."""
    if not has_han(word):
        return ""
    e = lookup(word)
    if e:
        syl = [numbered_to_marks(s) for s in e["numbered"]]
    else:
        if len(han_only(word)) > 6:                     # a sentence: keep punctuation-free, spaced
            return " ".join(_pypinyin(word, "tone"))
        syl = _pypinyin(word, "tone")
    return " ".join(syl) if spaced else join_syllables(syl)


def join_syllables(syl):
    """['nǚ', 'ér'] -> "nǚ'ér": the apostrophe goes before a syllable that starts with a, e or o."""
    out = ""
    for i, s in enumerate(syl):
        if i and s[:1].lower() in "aāáǎàeēéěèoōóǒò":
            out += "'"
        out += s
    return out


def sentence_pinyin(text):
    """Pinyin for a whole sentence, word by word (pypinyin segments it), punctuation kept out."""
    if not has_han(text):
        return ""
    parts = re.findall(r"[\u3400-\u9fff]+|[A-Za-z0-9]+", text)
    out = []
    for p in parts:
        if not has_han(p):
            out.append(p)
            continue
        syl = sandhi(p, _pypinyin(p, "tone"))            # tones in context (一 / 不 changes, 多音字)
        i = 0
        for w in segment(p):                             # ... grouped into words: "wǒ yǒu yí gè bēizi"
            out.append(join_syllables(syl[i:i + len(w)]))
            i += len(w)
    return " ".join(out)


NUMERALS = set("零一二三四五六七八九十百千万亿两第")


def sandhi(chars, syl):
    """一 and 不 change tone before another syllable: 一本 yì běn, 一个 yí gè, 不是 bú shì."""
    if len(syl) != len(chars):
        return syl
    syl = list(syl)
    nxt_tone = lambda j: tones(chars[j])[0] if j < len(chars) and has_han(chars[j]) else None
    for i, ch in enumerate(chars):
        t = nxt_tone(i + 1)
        if ch == "一":
            counting = (i > 0 and chars[i - 1] in NUMERALS) or (i + 1 < len(chars) and chars[i + 1] in NUMERALS)
            syl[i] = "yī" if (counting or not t) else "yí" if t == 4 else "yì"
        elif ch == "不" and t == 4:
            syl[i] = "bú"
    return syl


def segment(text, longest=6):
    """Split a run of Chinese characters into dictionary words (greedy longest match)."""
    _load()
    out, i = [], 0
    while i < len(text):
        for n in range(min(longest, len(text) - i), 0, -1):
            w = text[i:i + n]
            if n == 1 or w in _dict:
                out.append(w)
                i += n
                break
    return out


# --------------------------------------------------------------------------- CC-CEDICT
_dict, _lock, _ready = {}, threading.Lock(), threading.Event()
LINE = re.compile(r"^(\S+) (\S+) \[([^\]]*)\] /(.*)/\s*$")
BAD_DEF = re.compile(r"^(surname |variant of|old variant|archaic variant|used in |see |abbr\. for|erhua variant|"
                     r"also written|japanese variant|Taiwan pr\.)", re.I)


def _load():
    with _lock:
        if _ready.is_set():
            return
        path = DATA / "cedict.txt.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for ln in f:
                    if ln.startswith("#"):
                        continue
                    m = LINE.match(ln)
                    if not m:
                        continue
                    trad, simp, py, defs = m.groups()
                    entry = (trad, py.split(), defs.split("/"))
                    _dict.setdefault(simp, []).append(entry)
                    if trad != simp:
                        _dict.setdefault(trad, []).append(entry)
        _ready.set()


def preload():
    """Start loading the dictionary in the background (about 1 s)."""
    threading.Thread(target=_load, daemon=True).start()


def dictionary_size():
    _load()
    return len(_dict)


def _score(entry):
    trad, py, defs = entry
    good = [d for d in defs if d and not d.startswith("CL:") and not BAD_DEF.match(d)]
    proper = py and py[0][:1].isupper()
    return (len(good) > 0, not proper, len(good))


def lookup(word):
    """Best CC-CEDICT entry for a word: {word, numbered, defs, measure} or None."""
    if not word:
        return None
    _load()
    entries = _dict.get(word.strip())
    if not entries:
        return None
    trad, py, defs = max(entries, key=_score)
    measure = []
    for d in defs:
        if d.startswith("CL:"):
            for item in d[3:].split(","):
                m = re.match(r"(?:[^|\[]+\|)?([^\[]+)\[([^\]]+)\]", item.strip())
                if m:
                    measure.append((m.group(1), numbered_to_marks(m.group(2))))
    clean = [re.sub(r"\s*\(?\bCL:.*$", "", d).strip() for d in defs if d and not d.startswith("CL:")]
    clean = [d for d in clean if d and not BAD_DEF.match(d)] or [d for d in clean if d]
    return {"word": word, "trad": trad, "numbered": [s.lower() if s[:1].isupper() and len(word) > 1 else s for s in py],
            "defs": clean, "measure": measure}


def english(word, n=3):
    e = lookup(word)
    if not e:
        return ""
    out = []
    for d in e["defs"]:
        d = re.sub(r"\s*\[[^\]]*\]", "", d)              # drop "[pin1 yin1]" refs inside definitions
        if re.search(r"\((slang|derived from|dialect|old|literary|Tw|coll)", d) or len(d) > 45:
            continue
        if d and d not in out:
            out.append(d)
        if len(out) >= n:
            break
    return "; ".join(out)


# --------------------------------------------------------------------------- HSK
_hsk = None


def hsk_level(word):
    global _hsk
    if _hsk is None:
        try:
            _hsk = json.loads((DATA / "hsk3.json").read_text(encoding="utf-8"))
        except OSError:
            _hsk = {}
    return _hsk.get(word)


# --------------------------------------------------------------------------- the one call the app uses
def verify(word):
    """Everything about a word that we can know for sure.

    found=False means the model proposed something that is not in the dictionary: show it, but flag it."""
    word = (word or "").strip()
    word = re.sub(r"[\s　，。、！？；：“”‘’()（）]", "", word)
    e = lookup(word)
    return {
        "word": word,
        "found": e is not None,
        "pinyin": pinyin(word),
        "tones": tones(word) if has_han(word) else [],
        "en_dict": english(word),
        "measure": e["measure"][0] if e and e["measure"] else None,
        "hsk": hsk_level(word),
    }


def best_candidate(cands):
    """cands: [{"word": ..., "confidence": ...}]. The first one that is a real dictionary word wins;
    a big confidence gap still lets the model's first choice stand."""
    cands = [c for c in cands if has_han(str(c.get("word", "")))]
    if not cands:
        return None
    for c in cands:
        if lookup(re.sub(r"\s", "", c["word"])):
            return c
    return cands[0]


def measure_phrase(word, measure):
    """一个杯子 -> ('一个杯子', 'yí gè bēizi'). Uses tone sandhi for 一."""
    if not measure:
        return None
    cl, cl_py = measure
    t = tones(cl)[0] if tones(cl) else 4
    yi = "yí" if t == 4 else "yì"                      # tone change of 一 before the measure word
    return f"一{cl}{word}", f"{yi} {cl_py} {pinyin(word)}"


# --------------------------------------------------------------------------- pronunciation check
def compare_pronunciation(target, heard):
    """Compare what Whisper heard with the target word, syllable by syllable.

    Returns (score 0..1, [(target_syllable, heard_syllable_or_None, sound_ok, tone_ok)]).
    Whisper writes characters, so a wrong tone usually shows up as a different character with that tone.
    Neutral tones (5) in the target accept any tone."""
    tgt = numbered(target)
    hrd = [s.replace("ü", "v") for s in _pypinyin(han_only(heard), "tone3")] if has_han(heard) else []
    base = lambda s: re.sub(r"[1-5]$", "", s)
    tone = lambda s: int(s[-1]) if s and s[-1].isdigit() else 5
    sm = difflib.SequenceMatcher(a=[base(s) for s in tgt], b=[base(s) for s in hrd], autojunk=False)
    pairs = [None] * len(tgt)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            pairs[a + k] = b + k
    # unmatched target syllables: pair them with the unmatched heard syllable in the same slot, if any
    used = {p for p in pairs if p is not None}
    free = [j for j in range(len(hrd)) if j not in used]
    for i in range(len(tgt)):
        if pairs[i] is None and free:
            j = min(free, key=lambda j: abs(j - i))
            if abs(j - i) <= 1:
                pairs[i] = j
                free.remove(j)
    rows, pts = [], 0.0
    for i, t in enumerate(tgt):
        h = hrd[pairs[i]] if pairs[i] is not None else None
        sound = h is not None and base(h) == base(t)
        tone_ok = sound and (tone(t) == 5 or tone(h) == tone(t))
        pts += (0.5 if sound else 0) + (0.5 if tone_ok else 0)
        rows.append((numbered_to_marks(t), numbered_to_marks(h) if h else None, sound, tone_ok))
    return (pts / len(tgt) if tgt else 0.0), rows


# --------------------------------------------------------------------------- halal check (food labels, menus)
# level "haram": pork / alcohol. "doubtful": meat or gelatin of unknown slaughter / source.
HALAL_TERMS = [
    ("猪", "pork / pig", "haram"), ("豬", "pork / pig", "haram"), ("培根", "bacon", "haram"),
    ("火腿", "ham", "haram"), ("叉烧", "char siu (pork)", "haram"), ("腊肉", "cured meat (usually pork)", "haram"),
    ("五花肉", "pork belly", "haram"), ("排骨", "spare ribs (usually pork)", "doubtful"),
    ("午餐肉", "luncheon meat (usually pork)", "haram"), ("豚", "pork (tonkotsu)", "haram"),
    ("猪油", "lard", "haram"), ("荤油", "animal fat / lard", "haram"),
    ("酒精", "alcohol", "haram"), ("料酒", "cooking wine", "haram"), ("黄酒", "rice wine", "haram"),
    ("米酒", "rice wine", "haram"), ("啤酒", "beer", "haram"), ("葡萄酒", "wine", "haram"),
    ("白酒", "baijiu (liquor)", "haram"), ("红酒", "red wine", "haram"), ("朗姆", "rum", "haram"),
    ("酒糟", "distiller's grains", "haram"), ("醪糟", "fermented rice wine", "haram"),
    ("明胶", "gelatin", "doubtful"), ("吉利丁", "gelatin", "doubtful"), ("骨胶", "bone gelatin", "doubtful"),
    ("动物油", "animal fat", "doubtful"), ("牛油", "beef tallow", "doubtful"), ("牛肉", "beef", "doubtful"),
    ("羊肉", "lamb / mutton", "doubtful"), ("鸡肉", "chicken", "doubtful"), ("鸭", "duck", "doubtful"),
    ("肉松", "meat floss", "doubtful"), ("香肠", "sausage", "doubtful"), ("肠", "sausage / intestine", "doubtful"),
    ("骨汤", "bone broth", "doubtful"), ("高汤", "stock", "doubtful"), ("乳化剂", "emulsifier (source?)", "doubtful"),
]
HALAL_OK = ("清真", "HALAL", "Halal", "حلال")


def halal_check(text):
    """Returns ('halal' | 'haram' | 'doubtful' | None, [(term, meaning, level)]).

    'halal' only when a 清真 / HALAL mark is seen and nothing haram is listed."""
    text = text or ""
    hits, seen = [], set()
    for term, meaning, level in sorted(HALAL_TERMS, key=lambda t: -len(t[0])):
        if term in text and not any(term in s for s in seen):
            hits.append((term, meaning, level))
            seen.add(term)
    certified = any(k in text for k in HALAL_OK)
    if any(h[2] == "haram" for h in hits):
        return "haram", hits
    if certified:
        return "halal", hits
    if hits:
        return "doubtful", hits
    return None, hits
