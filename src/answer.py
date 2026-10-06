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
GEMINI_FALLBACKS = [m.strip() for m in os.getenv("GEMINI_FALLBACKS", "gemini-flash-lite-latest,gemini-2.5-flash").split(",") if m.strip()]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---- refusal thresholds (PROVISIONAL: tuned on a few questions, see eval/ and README) ----
# With embeddings: refuse when the best chunk's cosine is below COS_MIN.
# Lexical only:    refuse when the share of query words found in the best chunk is below COV_MIN.
COS_MIN = float(os.getenv("COS_MIN", "0.85"))
COV_MIN = float(os.getenv("COV_MIN", "0.40"))
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
                        response_mime_type="application/json"))
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
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else None


def _pointers(hits, limit: int = 3):
    """External references (Takshir etc.) cited by the two best chunks."""
    seen, out = set(), []
    for h in hits[:2]:
        for r in h["chunk"].get("refs_takshir", []):
            if r not in seen:
                seen.add(r)
                out.append(r)
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
        verified (bool: quote found verbatim), external_ref (str|None), pointers (list)
    """
    import time
    t0 = time.time()
    r = retriever or get_retriever()
    as_of = as_of or date.today()
    hits = r.search(question, k=TOP_K, as_of=as_of)
    t_search = time.time() - t0
    out = dict(refused=True, reason="", answer="", hits=hits, source=None, quote="", span=None,
               verified=False, external_ref=None, pointers=_pointers(hits), as_of=as_of,
               timing={"search_s": round(t_search, 2), "llm_s": 0.0, "model": None}, not_yet_valid=None)

    def _weak(h):
        return (h["cos"] < COS_MIN) if h["cos"] is not None else (h["coverage"] < COV_MIN)

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
    nums = lambda s: set(re.findall(r"\d+(?:[.,]\d+)*", BIDI.sub("", s)))
    verified = span is not None and nums(answer_text) <= nums(quote)
    if span is None:                                         # show the closest line, flagged
        span = best_line(question, src["chunk"]["text"])
    out.update(refused=False, reason="ok", answer=answer_text,
               source=src, quote=quote, span=span, verified=verified,
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
