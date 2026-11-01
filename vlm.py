"""
The local vision model: finding it (Ollama / LM Studio / llama-server), the prompts, and
structured replies. Every request sends a JSON schema, so even a 2-3B model answers in the exact
shape the app expects (Ollama `format`, OpenAI-style `response_format` for llama.cpp / LM Studio).
"""
import base64
import json
import re
import urllib.error
import urllib.request

VISION_HINTS = ("vl", "vision", "llava", "gemma3", "gemma-3", "minicpm-v", "moondream", "pixtral",
                "bakllava", "llama3.2-vision", "granite3.2-vision", "mistral-small3", "internvl", "phi-3.5-vision",
                "phi-4-multimodal", "smolvlm", "gemma4", "gemma-4")
PREFERRED = ("qwen3-vl", "qwen2.5vl", "qwen2.5-vl", "gemma3", "gemma-3", "minicpm-v", "llama3.2-vision", "llava")


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


# --------------------------------------------------------------------------- prompts + schemas
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
"say" must be in English, at most 2 short sentences, witty, no emojis.""",
}
ASK_SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["object", "math", "drawing", "text", "scene", "answer"]},
    "say": {"type": "string"}, "math_expr": {"type": "string"}}, "required": ["kind", "say", "math_expr"]}

LEARN_LANGS = {"zh": "Simplified Chinese (Mandarin)", "en": "English"}
LEARN_SYSTEM = """You are a careful vocabulary tutor that sees through a webcam.
The user marked an object with bright MAGENTA strokes, or holds it up. If there are no strokes, use the most
prominent object (ignore the hand and fingers holding it, and ignore the person unless they point at a body part
or clothing).
Identify the object and give its everyday name in {lang} ({rule}).
Give up to 3 candidates, best first, each with a confidence from 0 to 1 that is honest: if the object is unclear,
the confidences must be low. Use the most common everyday word a native speaker would say, not a rare or
technical term.
"example" is a very short, simple sentence in {lang} using the FIRST candidate (beginner level).
"example_tr" is its translation into {tr}."""
LEARN_RULE = {"zh": "simplified characters only, no pinyin", "en": "a single common English noun"}
LEARN_SCHEMA = {"type": "object", "properties": {
    "candidates": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "object", "properties": {
        "word": {"type": "string"}, "en": {"type": "string"}, "ar": {"type": "string"},
        "confidence": {"type": "number"}}, "required": ["word", "en", "ar", "confidence"]}},
    "example": {"type": "string"}, "example_tr": {"type": "string"}},
    "required": ["candidates", "example", "example_tr"]}

TUTOR_SYSTEM = """You are a patient Mandarin Chinese tutor that sees through a webcam. The learner is a beginner
(HSK 1-3) living in China. They ask you something out loud (in Arabic, English or Chinese) about what the camera
shows, e.g. "how do I say I want to buy this?".
Answer with ONE short, natural sentence a native speaker would really say, in simplified Chinese ("zh"),
its English ("en") and Egyptian Arabic ("ar") meaning, and a one-line usage tip ("tip", in {tip_lang}, may be empty)."""
TUTOR_SCHEMA = {"type": "object", "properties": {
    "zh": {"type": "string"}, "en": {"type": "string"}, "ar": {"type": "string"}, "tip": {"type": "string"}},
    "required": ["zh", "en", "ar", "tip"]}

READ_SYSTEM = """You read text in photos taken by a webcam: menus, food labels, signs, packages, screens.
Copy the main text exactly as written ("text", keep the original language and line breaks, max ~12 lines),
translate it to English ("en") and Egyptian Arabic ("ar"), and say in a few words what it is ("what", e.g.
"ingredients list", "restaurant menu", "street sign").
If the photo shows food ingredients or a menu, also list in "concerns" anything containing pork, lard,
alcohol / wine / cooking wine, gelatin, or meat of unknown origin (empty string if none)."""
READ_SCHEMA = {"type": "object", "properties": {
    "text": {"type": "string"}, "en": {"type": "string"}, "ar": {"type": "string"},
    "what": {"type": "string"}, "concerns": {"type": "string"}}, "required": ["text", "en", "ar", "what", "concerns"]}

CHECK_SYSTEM = """You check a vocabulary game. The learner was asked to show a specific object to the webcam.
Look at the photo (ignore the hand holding the object) and decide honestly whether the object shown is that thing
(or a normal variant of it). "seen" = what the object in the photo really is, in English, 1-4 words."""
CHECK_SCHEMA = {"type": "object", "properties": {"match": {"type": "boolean"}, "seen": {"type": "string"}},
                "required": ["match", "seen"]}


def learn_system(target):
    return LEARN_SYSTEM.format(lang=LEARN_LANGS[target], rule=LEARN_RULE[target],
                               tr="English" if target == "zh" else "Arabic")


class LocalVLM:
    def __init__(self, backend, url, model, lang):
        self.backend, self.url, self.model, self.lang = backend, url, model, lang
        self.schema_ok = True             # the server accepted a JSON schema (turned off if it refuses)

    def warm_up(self):
        try:
            if self.backend == "ollama":
                http_json(f"{self.url}/api/generate", {"model": self.model, "keep_alive": "30m"}, timeout=180)
        except Exception:
            pass

    # ---- the different questions -------------------------------------------------------------
    def ask(self, jpeg, has_strokes, question=None, focus=False, history=None):
        what = ("This is a close-up of the object I marked with a magenta line or circle"
                if has_strokes else "This is a close-up of the object I'm holding up")
        if history and question:
            prompt = f'Follow-up question, about the same picture: "{question}". Use kind "answer".'
        elif focus and question:
            prompt = f'{what}. I asked out loud: "{question}". Answer it about this object. Use kind "answer".'
        elif focus:
            prompt = (f"{what}. What is it? Name it precisely and add one short fun fact. "
                      'Use kind "object" (or "math"/"text" if I marked writing).')
        elif question:
            prompt = (f'The user asked out loud: "{question}". Answer it based on what you see'
                      f'{" (the magenta strokes show what they mean)" if has_strokes else ""}. Use kind "answer".')
        else:
            prompt = "What do my magenta strokes mean?" if has_strokes else "No strokes this time. What do you see?"
        return self.chat(SYSTEM[self.lang], prompt, jpeg, ASK_SCHEMA, history=history)

    def ask_learn(self, jpeg, has_strokes, target, focus=False):
        if focus:
            prompt = ("This is a close-up of the object I marked with a magenta line or circle. Teach me its word."
                      if has_strokes else
                      "This is a close-up of the object I'm holding up. Teach me its word (ignore hands and fingers).")
        else:
            prompt = "Teach me the word for what I circled." if has_strokes else "Teach me the word for the main object you see."
        return self.chat(learn_system(target), prompt, jpeg, LEARN_SCHEMA, temperature=0.2, max_tokens=320)

    def ask_tutor(self, jpeg, question, context_word=None, history=None):
        ctx = f' (we were just talking about "{context_word}")' if context_word else ""
        prompt = f'My question{ctx}: "{question}"'
        system = TUTOR_SYSTEM.format(tip_lang="Egyptian Arabic" if self.lang == "ar" else "English")
        return self.chat(system, prompt, jpeg, TUTOR_SCHEMA, temperature=0.3, history=history)

    def ask_read(self, jpeg, ocr_text=None):
        prompt = "Read and translate the text in this photo."
        if ocr_text:
            prompt += ("\nAn OCR engine already read this (it can contain small mistakes, fix them using the photo):\n"
                       + ocr_text[:1500])
        return self.chat(READ_SYSTEM, prompt, jpeg, READ_SCHEMA, temperature=0.1, max_tokens=700)

    def ask_check(self, jpeg, word, en):
        return self.chat(CHECK_SYSTEM, f'The learner was asked to show: "{en}" ({word}). Is that what the photo shows?',
                         jpeg, CHECK_SCHEMA, temperature=0.0, max_tokens=60)

    # ---- transport ---------------------------------------------------------------------------
    def chat(self, system, prompt, jpeg, schema=None, temperature=0.4, max_tokens=260, history=None):
        """history: [(user_prompt, assistant_reply)] of earlier turns about the same picture (follow-ups)."""
        b64 = base64.b64encode(jpeg).decode()
        if self.backend == "ollama":
            msgs = [{"role": "system", "content": system}]
            turns = list(history or []) + [(prompt, None)]
            for i, (u, a) in enumerate(turns):
                m = {"role": "user", "content": u}
                if i == 0:
                    m["images"] = [b64]
                msgs.append(m)
                if a is not None:
                    msgs.append({"role": "assistant", "content": a})
            body = {"model": self.model, "stream": False, "keep_alive": "30m",
                    "options": {"temperature": temperature, "num_predict": max_tokens}, "messages": msgs,
                    "format": schema if (schema and self.schema_ok) else "json"}
            try:
                r = http_json(f"{self.url}/api/chat", body, timeout=120)
            except urllib.error.HTTPError as e:
                if not (schema and self.schema_ok and e.code in (400, 500)):
                    raise
                self.schema_ok = False                        # old Ollama: plain JSON mode
                body["format"] = "json"
                r = http_json(f"{self.url}/api/chat", body, timeout=120)
            return r["message"]["content"]

        msgs = [{"role": "system", "content": system}]
        turns = list(history or []) + [(prompt, None)]
        for i, (u, a) in enumerate(turns):
            content = [{"type": "text", "text": u}]
            if i == 0:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            msgs.append({"role": "user", "content": content})
            if a is not None:
                msgs.append({"role": "assistant", "content": a})
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}, "messages": msgs}
        if schema and self.schema_ok:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "strict": True, "schema": schema}}
        try:
            r = http_json(f"{self.url}/v1/chat/completions", body, timeout=300)
        except urllib.error.HTTPError as e:
            if "response_format" not in body or e.code not in (400, 422, 500):
                raise
            self.schema_ok = False
            body.pop("response_format")
            body["messages"][0]["content"] += "\nReply with JSON only, matching: " + json.dumps(schema)
            r = http_json(f"{self.url}/v1/chat/completions", body, timeout=300)
        return r["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- reply parsing
def parse_json(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def parse_reply(raw):
    d = parse_json(raw)
    say = str(d.get("say", "") if d else raw).strip().strip('"')
    return str(d.get("kind") or "scene").lower(), say, str(d.get("math_expr") or "").strip()


def parse_learn(raw):
    """-> (candidates [{word,en,ar,confidence}], example, example_tr). Also accepts the old one-word shape."""
    d = parse_json(raw)
    cands = d.get("candidates")
    if not isinstance(cands, list):
        cands = [{"word": d.get("word", ""), "en": d.get("en", ""), "ar": d.get("ar", ""), "confidence": 0.5}] \
            if d.get("word") else []
    out = []
    for c in cands[:3]:
        if not isinstance(c, dict) or not str(c.get("word", "")).strip():
            continue
        try:
            conf = float(c.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = conf / 100 if conf > 1 else conf
        out.append({"word": str(c["word"]).strip(), "en": str(c.get("en") or "").strip(),
                    "ar": str(c.get("ar") or "").strip(), "confidence": max(0.0, min(1.0, conf))})
    seen, uniq = set(), []
    for c in out:
        if c["word"] not in seen:
            seen.add(c["word"])
            uniq.append(c)
    return uniq, str(d.get("example") or "").strip(), str(d.get("example_tr") or d.get("example_en") or "").strip()
