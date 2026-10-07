"""Step 3: answer generation, with refusal and citation checks done IN CODE.

Pipeline for one question:
    retrieve (validity filter inside)  ->  refusal check (code)  ->  LLM (only the chunks)
    ->  verify that the quoted sentence really exists in the cited chunk  ->  result dict

Only `call_llm()` talks to an API. In an air-gapped deployment this is the one function that
is replaced by a call to a local model (vLLM / Ollama); everything else stays as is.

Run:  python src/answer.py "כמה שעות בחודש עובד סטודנט?"
      python src/answer.py --models        (list the Gemini models your key can use)
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import Retriever, label, BIDI  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PROVIDER = os.getenv("LLM_PROVIDER", "gemini")                 # gemini | groq
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
# tried in order when the main model is overloaded (503) or rate-limited (429)
CALL_TIMEOUT_S = float(os.getenv("CALL_TIMEOUT_S", "20"))      # per model call
TOTAL_BUDGET_S = float(os.getenv("TOTAL_BUDGET_S", "45"))      # per question
GEMINI_FALLBACKS = [m.strip() for m in os.getenv("GEMINI_FALLBACKS", "gemini-flash-lite-latest,gemini-3.5-flash-lite").split(",") if m.strip()]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---- refusal in code: only a safety net for clearly unrelated questions --------------------
# Measured on eval/questions.csv (44 questions, multilingual-e5-small, with glossary), top-1 chunk:
#   answerable questions        cos 0.845-0.913, query-word coverage 0.29-1.00
#   off-topic questions         cos 0.733-0.816, coverage 0.00-0.40
#   in-domain, not in corpus    cos 0.809-0.860 (they overlap with answerable ones, so for those
#                               the LLM decides: answerable=false)
# Rule with embeddings: refuse in code when cos < COS_MIN AND coverage < COV_SEM_MIN.
# COS_MIN sits between the highest off-topic score (0.816) and the lowest answerable one (0.845).
# Coverage alone cannot separate the groups (0.40 off-topic vs 0.29 answerable), so without
# embeddings the rule is weaker: refuse only when almost no query word is found (COV_MIN; catches 3 of the 7 off-topic questions).
# All thresholds are PROVISIONAL: a 44-question sample, to be re-measured on a larger gold set.
COS_MIN = float(os.getenv("COS_MIN", "0.83"))
COV_SEM_MIN = float(os.getenv("COV_SEM_MIN", "0.50"))
COV_MIN = float(os.getenv("COV_MIN", "0.25"))
TOP_K = int(os.getenv("TOP_K", "5"))

CONTACT = "takam@mof.gov.il"

SYSTEM = """אתה עוזר שעונה על שאלות לגבי הוראות תכ"ם (משרד האוצר, אגף השכר).
כללים מחייבים:
1. ענה אך ורק על סמך הקטעים שמופיעים למטה ("מקורות"). אל תשתמש בידע כללי ואל תנחש.
2. אם המקורות אינם עונים על השאלה במפורש, החזר answerable=false. עדיף לסרב מאשר לנחש.
3. אל תמציא מספרים, תאריכים או סעיפים. מספר שמופיע בתשובה חייב להופיע במקור.
4. אם המקור מפנה למסמך אחר שאינו במקורות (תקשי"ר, חוזר, הסכם), ציין זאת ב-external_ref ואל תענה במקומו.
5. אם הסכום או הכלל תלוי בתאריך (מהדורה, "החל מיום"), ציין את התאריך בתשובה.
6. התשובה קצרה, בעברית פשוטה, עד 3 משפטים.
7. quote חייב להיות העתקה מילה במילה של משפט (או שורת טבלה) אחד מהמקור שעליו התשובה מבוססת, בלי שינוי.
החזר JSON בלבד בצורה:
{"answerable": true/false, "answer": "...", "source": <מספר המקור>, "quote": "...", "external_ref": "..." או null}"""


# ------------------------------------------------------------------ LLM call (swap here)
# structured output: the model must return valid JSON (an unescaped quote in a quote/answer used to break parsing)
REPLY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answerable": {"type": "BOOLEAN"}, "answer": {"type": "STRING"}, "source": {"type": "INTEGER"},
        "quote": {"type": "STRING"}, "external_ref": {"type": "STRING", "nullable": True}},
    "required": ["answerable", "answer", "source", "quote"],
}

LAST_MODEL = {"name": None}


def call_llm(prompt: str, system: str = SYSTEM) -> str:
    """Return the raw text of the model's reply (expected to be JSON)."""
    if PROVIDER == "gemini":
        from google import genai
        from google.genai import types
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is missing in .env")
        import time
        # a hard time limit per call (ms) and for the whole question, so the screen never hangs
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(CALL_TIMEOUT_S * 1000)))
        deadline = time.time() + TOTAL_BUDGET_S
        last = None
        for model in [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]:
            if time.time() > deadline:
                break
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system, temperature=0,
                        response_mime_type="application/json", response_schema=REPLY_SCHEMA))
                LAST_MODEL["name"] = model
                return resp.text
            except Exception as e:           # 503 / 429 / timeout / 404: go on to the next model
                last = e
        raise last or RuntimeError("LLM time budget exhausted")
    if PROVIDER == "groq":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model=GROQ_MODEL, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    raise RuntimeError(f"unknown LLM_PROVIDER: {PROVIDER}")


def list_models():
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    for m in client.models.list():
        if "generateContent" in (getattr(m, "supported_actions", None) or []):
            print(m.name.replace("models/", ""))


# ------------------------------------------------------------------ helpers
def build_prompt(question: str, hits) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        c = h["chunk"]
        parts.append(f"[{i}] {label(c)} | תוקף מ-{c['valid_from'] or 'לא ידוע'}\n{c['text']}")
    return "מקורות:\n\n" + "\n\n".join(parts) + f"\n\nשאלה: {question}"


def _norm_map(text: str):
    """Keep only letters/digits; return (normalized string, index map back to `text`)."""
    chars, idx = [], []
    for i, ch in enumerate(BIDI.sub("", text)):
        if ch.isalnum():
            chars.append(ch)
            idx.append(i)
    return "".join(chars), idx


def locate(quote: str, text: str):
    """Find `quote` inside `text` ignoring spaces/punctuation/table bars.
    Returns (start, end) in `text`, or None."""
    if not quote:
        return None
    t, tmap = _norm_map(text)
    q, _ = _norm_map(quote)
    if len(q) < 6:
        return None
    pos = t.find(q)
    if pos < 0:
        return None
    return tmap[pos], tmap[pos + len(q) - 1] + 1


def best_line(question: str, text: str):
    """Fallback highlight: the line sharing most words with the question."""
    from retrieve import tokenize
    qs = set(tokenize(question, with_variants=False))
    best, best_score, off = None, 0, 0
    for line in text.split("\n"):
        score = len(qs & set(tokenize(line, with_variants=False)))
        if score > best_score:
            best, best_score = (off, off + len(line)), score
        off += len(line) + 1
    return best


def _parse_json(raw: str):
    """Parse the model reply. Models sometimes emit unescaped quotes (תקשי"ר) inside strings,
    which breaks json.loads; in that case pull the known fields out with regexes."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    braces = re.search(r"\{.*\}", raw, re.S)
    for cand in (raw, braces.group(0) if braces else None):
        if cand:
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                pass
    if not re.search(r'"answerable"\s*:', raw):
        return None
    ans = re.search(r'"answerable"\s*:\s*(true|false)', raw)
    # a string value runs up to the next known key (or the closing brace)
    def field(name, nxt):
        m = re.search(r'"%s"\s*:\s*"(.*?)"\s*,\s*"(?:%s)"\s*:' % (name, "|".join(nxt)), raw, re.S)
        return m.group(1).replace('\\"', '"') if m else ""
    src = re.search(r'"source"\s*:\s*"?(\d+)', raw)
    ext = re.search(r'"external_ref"\s*:\s*(null|"(.*?)"\s*\}?\s*$)', raw, re.S)
    return {"answerable": bool(ans and ans.group(1) == "true"),
            "answer": field("answer", ["source", "quote", "external_ref"]),
            "source": int(src.group(1)) if src else 1,
            "quote": field("quote", ["external_ref"]),
            "external_ref": (ext.group(2) if ext and ext.group(2) else None)}


def nums(s: str) -> set:
    return set(re.findall(r"\d+(?:[.,]\d+)*", BIDI.sub("", s or "")))


def meta_nums(c) -> set:
    """Numbers the prompt itself gives the model (edition, valid-from date) and that the system
    prompt tells it to mention. They are legitimate in an answer even though they are not in the quote."""
    out = nums(str(c.get("edition") or ""))
    vf = str(c.get("valid_from") or "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", vf):
        y, m, d = vf.split("-")
        out |= {y, m, d, str(int(m)), str(int(d)), f"{d}.{m}.{y}", f"{int(d)}.{int(m)}.{y}"}
    return out


# a Takshir reference written with its chapter title:  תקשי"ר, "תוספת מעונות", פרק 25.61
_TITLED_REF = re.compile(r"""תקשי["״]ר,?\s*["״'’“”]+([^"״'’“”]{2,60}?)["״'’“”]+,?\s*פרק\s*(\d+\.\d+(?:\.\d+)?)""")


def number_spans(missing: set, text: str, main_span):
    """For every number missing from the quote: (start, end) of the FIRST line of `text` that holds it,
    so the screen can highlight the second place that supports the answer."""
    lines, off = [], 0
    for line in text.split("\n"):
        lines.append((off, off + len(line), nums(line)))
        off += len(line) + 1
    spans = []
    for n in sorted(missing):
        for a, b, ln in lines:
            if main_span and a < main_span[1] and b > main_span[0]:
                continue
            if n in ln:
                if (a, b) not in spans:
                    spans.append((a, b))
                break
    return spans


def _pointers(question: str, hits, limit: int = 3):
    """Takshir chapters worth pointing to when we refuse. A chunk often cites several chapters
    (e.g. 24.15, 35.11, 25.61 in different sentences), so a chapter is listed only if the title
    written next to it shares at least half of its words with the question."""
    from retrieve import tokenize
    qset = set(tokenize(question))
    out = []
    for h in hits[:3]:
        text = BIDI.sub("", h["chunk"]["text"])
        for m in _TITLED_REF.finditer(text):
            words = tokenize(m.group(1), with_variants=False)
            covered = [w for w in words if set(tokenize(w)) & qset]
            if words and len(covered) / len(words) >= 0.5 and m.group(2) not in out:
                out.append(m.group(2))
    return out[:limit]


# ------------------------------------------------------------------ main entry point
_RETRIEVER = None


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER


def answer_question(question: str, as_of: date | None = None, retriever: Retriever | None = None) -> dict:
    """Returns a dict:
        refused (bool), reason (str), answer (str), hits (list),
        source (hit dict or None), quote (str), span ((start, end) in source text or None),
        verified (bool: quote found verbatim AND every number of the answer is in the quote or in the
        edition/date metadata), numbers_in_source (bool: quote verbatim, numbers found elsewhere in the
        source text only), external_ref (str|None), pointers (list)
    """
    import time
    t0 = time.time()
    r = retriever or get_retriever()
    as_of = as_of or date.today()
    hits = r.search(question, k=TOP_K, as_of=as_of)
    t_search = time.time() - t0
    out = dict(refused=True, reason="", answer="", hits=hits, source=None, quote="", span=None,
               verified=False, numbers_in_source=False, external_ref=None, pointers=_pointers(question, hits), as_of=as_of, extra_spans=[],
               timing={"search_s": round(t_search, 2), "llm_s": 0.0, "model": None}, not_yet_valid=None)

    def _weak(h):
        if h["cos"] is not None:
            return h["cos"] < COS_MIN and h["coverage"] < COV_SEM_MIN
        return h["coverage"] < COV_MIN

    def _check_future(out):
        # Past date chosen: does the topic exist in a version that only became valid later?
        if as_of < date.today():
            now = r.search(question, k=1, as_of=date.today())
            if now and not _weak(now[0]) and now[0]["chunk"]["valid_from"]:
                out["not_yet_valid"] = now[0]["chunk"]

    # ---- refusal in code, before any LLM call -------------------------------------------
    if not hits:
        out["reason"] = "no_valid_sources"
        _check_future(out)
        return out
    top = hits[0]
    if _weak(top):
        out["reason"] = "low_similarity"
        _check_future(out)
        return out

    # ---- LLM -----------------------------------------------------------------------------
    t1 = time.time()
    try:
        data = _parse_json(call_llm(build_prompt(question, hits)))
        out["timing"].update(llm_s=round(time.time() - t1, 2), model=LAST_MODEL["name"])
    except Exception as e:
        out["timing"].update(llm_s=round(time.time() - t1, 2))                                   # network, key, quota, bad JSON
        out["reason"] = f"llm_error: {e}"
        return out
    if not data or not data.get("answerable"):
        out["reason"] = "model_says_not_in_sources"
        out["external_ref"] = (data or {}).get("external_ref")
        return out

    try:
        idx = int(data.get("source", 1)) - 1
    except (ValueError, TypeError):
        idx = 0
    src = hits[idx] if 0 <= idx < len(hits) else hits[0]    # out-of-range (e.g. 0) -> first hit
    quote = (data.get("quote") or "").strip()
    answer_text = (data.get("answer") or "").strip()
    span = locate(quote, src["chunk"]["text"])
    # every number in the answer must also appear in the quote, otherwise it is "not verified"
    ans_nums = nums(answer_text)
    verified = span is not None and ans_nums <= (nums(quote) | meta_nums(src["chunk"]))
    # weaker level: the quote is verbatim, and every number of the answer exists somewhere in the same
    # source text (e.g. the answer lists two table rows but the quote is only one of them)
    numbers_in_source = span is not None and ans_nums <= (nums(src["chunk"]["text"]) | meta_nums(src["chunk"]))
    if span is None:                                         # show the closest line, flagged
        span = best_line(question, src["chunk"]["text"])
    missing = ans_nums - nums(quote) - meta_nums(src["chunk"])
    extra = number_spans(missing, src["chunk"]["text"], span) if (missing and numbers_in_source) else []
    out["extra_spans"] = extra
    out.update(refused=False, reason="ok", answer=answer_text,
               source=src, quote=quote, span=span, verified=verified, numbers_in_source=numbers_in_source,
               external_ref=data.get("external_ref"))

    # ---- past date + source no longer valid today: point to the current text (code only) ----
    out["superseded_by"] = None
    vt = str(src["chunk"].get("valid_to") or "")
    if as_of < date.today() and vt and vt < date.today().isoformat():
        now = r.search(question, k=1, as_of=date.today())
        if now and not _weak(now[0]):
            out["superseded_by"] = now[0]["chunk"]
    return out


REFUSAL_TEXT = ("לא נמצאה במאגר המסמכים תשובה מפורשת לשאלה זו. "
                f"אפשר לפנות אל {CONTACT} או לחפש באתר תכ\"ם.")


if __name__ == "__main__":
    if "--models" in sys.argv:
        list_models()
        sys.exit()
    q = " ".join(sys.argv[1:]) or "כמה שעות בחודש עובד סטודנט?"
    res = answer_question(q)
    if res["refused"]:
        print("REFUSED:", res["reason"])
        print(REFUSAL_TEXT)
        if res["pointers"]:
            print("מפנה אל (מחוץ למאגר): תקשי\"ר", ", ".join(res["pointers"]))
    else:
        c = res["source"]["chunk"]
        print(res["answer"])
        print("\nמקור:", label(c), "| תוקף מ-", c["valid_from"], "| verified:", res["verified"])
        print("ציטוט:", res["quote"])
        if res["external_ref"]:
            print("הפניה חיצונית:", res["external_ref"])
