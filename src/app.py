"""Step 4: Streamlit UI.   Run:  streamlit run src/app.py

Two ways to use it:
  * a full question  -> short answer + the exact source section with the supporting sentence
                        highlighted + link to the official page
  * one or two words -> a few ready-made questions to click, then the same flow
Refusals are decided in code (answer.py); the screen only displays them.
"""
from __future__ import annotations

import csv
import html
import sys
from datetime import date, datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer import REFUSAL_TEXT, answer_question, CONTACT  # noqa: E402
from retrieve import Retriever, label  # noqa: E402
from suggest import is_keyword_query, suggest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK = ROOT / "data" / "feedback.csv"

EXAMPLES = [
    "כמה שבועות חופשת לידה מגיעים להורה?",
    "האם מגיעה קצובת ביגוד בתקופת לידה ללא תשלום?",
    "כמה עולה אגרת רישוי לעובד בעל רכב שירות?",
    "מה תקרת ימי החופשה שאפשר לצבור ולפדות בתום שירות?",
]

st.set_page_config(page_title="עוזר הוראות תכ״ם (POC)", page_icon="📄", layout="centered")
st.markdown(
    """
    <style>
      .block-container {direction: rtl; text-align: right; max-width: 820px;}
      .stTextInput input, .stDateInput input {direction: rtl; text-align: right;}
      .src {background:#f6f7f9; border:1px solid #dde1e6; border-radius:8px; padding:12px 14px;
            line-height:1.7; direction: rtl; text-align: right; color:#1b1f24;}
      .src mark {background:#ffe58a; padding:1px 2px; border-radius:3px;}
      .src mark.m2 {background:#b9dcff; padding:1px 2px; border-radius:3px;}
      .meta {color:#5b6470; font-size:0.9rem;}
      @media (prefers-color-scheme: dark) {
        .src {background:#1f242b; border-color:#39414b; color:#e8ecf1;}
        .src mark {background:#7a6200; color:#fff;}
        .src mark.m2 {background:#14507f; color:#fff;}
        .meta {color:#a9b3bf;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="טוען את המאגר...")
def load_retriever():
    return Retriever()


def render_source(text: str, span, extra=()):
    """HTML of the chunk with the supporting span highlighted (yellow), and optionally the
    lines that hold the remaining numbers of the answer (blue)."""
    marks = ([(span[0], span[1], "mark")] if span else []) + [(a, b, "m2") for a, b in extra]
    marks.sort()
    body, pos = "", 0
    for a, b, cls in marks:
        if a < pos:
            continue
        tag = "<mark>" if cls == "mark" else '<mark class="m2">'
        body += html.escape(text[pos:a]) + tag + html.escape(text[a:b]) + "</mark>"
        pos = b
    body += html.escape(text[pos:])
    return f'<div class="src">{body.replace(chr(10), "<br>")}</div>'


def save_feedback(question, answer, note):
    FEEDBACK.parent.mkdir(exist_ok=True)
    new = not FEEDBACK.exists()
    with open(FEEDBACK, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "question", "answer", "note"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), question, answer, note])


def _d(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").strftime("%d/%m/%Y")
    except ValueError:
        return str(s)


def fmt_range(c) -> str:
    vf, vt = c.get("valid_from"), c.get("valid_to")
    if not vf and not vt:
        return "תאריך לא ידוע"
    start = _d(vf) if vf else "לא ידוע"
    return f"{start} עד {_d(vt)}" if vt else f"מ-{start} ועד היום"


def show_answer(question: str, as_of: date, retriever):
    with st.spinner("מחפש ומנסח תשובה..."):
        res = answer_question(question, as_of=as_of, retriever=retriever)
    t = res.get("timing", {})
    st.caption(f"⏱ חיפוש {t.get('search_s')} שנ׳ · מודל {t.get('llm_s')} שנ׳ ({t.get('model') or 'לא הופעל'})")

    if res["refused"]:
        if res["reason"].startswith("llm_error"):
            st.error("השירות של המודל אינו זמין כרגע (עומס או תקלה זמנית). אפשר לנסות שוב בעוד רגע.")
            with st.expander("פרטי השגיאה"):
                st.code(res["reason"][11:])
        elif res.get("not_yet_valid"):
            c = res["not_yet_valid"]
            st.warning(f"הנושא קיים בהוראה {c['doc_id']} ({c['title']}), אך המהדורה שבמאגר בתוקף רק מ-"
                       f"{c['valid_from']}, ולא הייתה בתוקף בתאריך שנבחר ({res['as_of']}). "
                       "המאגר אינו כולל מהדורות קודמות.")
        else:
            st.warning(REFUSAL_TEXT)
        if res["pointers"] or res["external_ref"]:
            refs = ", ".join(res["pointers"]) or res["external_ref"]
            st.info(f"ייתכן שהמידע נמצא בתקשי״ר ({refs}), שההוראות הקרובות מפנות אליו. המקור הזה אינו במאגר.")
        with st.expander("המקורות הקרובים ביותר שנמצאו (לא בהכרח עונים)"):
            for h in res["hits"][:3]:
                st.markdown(f"**{label(h['chunk'])}**")
                st.markdown(f'<div class="src">{html.escape(h["chunk"]["text"][:400])}…</div>',
                            unsafe_allow_html=True)
        return

    c = res["source"]["chunk"]
    st.success(res["answer"])
    if res["external_ref"]:
        st.info(f"שימו לב: {res['external_ref']} (מקור חיצוני למאגר)")
    st.markdown(
        f'<div class="meta">מקור: {label(c)} · בתוקף: {fmt_range(c)} · '
        f'<a href="{html.escape(c["source_url"] or "", quote=True)}" target="_blank">פתיחה במקור</a></div>',
        unsafe_allow_html=True)
    nxt = res.get("superseded_by")
    if nxt:
        st.warning(f"⚠️ התשובה לפי המצב ב-{res['as_of'].strftime('%d/%m/%Y')}. מקור זה אינו בתוקף היום; "
                   f"הנושא מוסדר כעת ב{nxt['doc_type'] if nxt.get('doc_type') else 'הוראה'} {nxt['doc_id']} "
                   f"(בתוקף: {fmt_range(nxt)}).")
        with st.expander("הנוסח הנוכחי"):
            st.markdown(f"**{label(nxt)}**")
            st.markdown(f'<div class="src">{html.escape(nxt["text"]).replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True)
    if not res["verified"]:
        if res.get("numbers_in_source"):
            st.caption("⚠️ אומת חלקית: הציטוט נמצא מילה במילה במקור, ומספרי התשובה מופיעים בקטע, "
                       "אך לא כולם בציטוט עצמו. הציטוט מודגש בצהוב, והשורה עם שאר המספרים בכחול.")
        else:
            st.caption("⚠️ לא אומת: הציטוט לא נמצא מילה במילה במקור, או שמספר בתשובה אינו מופיע בקטע. "
                       "יש לבדוק מול המקור המודגש.")
    with st.expander("פתח מקור", expanded=True):
        st.markdown(render_source(c["text"], res["span"], res.get("extra_spans", ())), unsafe_allow_html=True)
    with st.expander("מקורות נוספים שנבדקו"):
        for h in res["hits"]:
            if h is not res["source"]:
                st.markdown(f"- {label(h['chunk'])}")

    with st.popover("דווח על תשובה שגויה"):
        note = st.text_input("מה לא נכון?", key="fb_note")
        if st.button("שלח דיווח"):
            save_feedback(question, res["answer"], note)
            st.toast("תודה, הדיווח נשמר")


# ------------------------------------------------------------------ page
st.title("עוזר הוראות תכ״ם")
st.caption("מערכת ניסיונית (POC) המבוססת על מדגם קטן מפרק 13 (שכר). היא מסייעת באיתור מידע, "
           "ואינה מחליפה את ההוראה הרשמית. יש לאמת מול המסמך המקורי.")

retriever = load_retriever()

c1, c2 = st.columns([3, 1])
with c2:
    as_of = st.date_input("תקף לתאריך", value=date.today(), format="DD/MM/YYYY")
with c1:
    typed = st.text_input("שאלה, או מילה אחת", key="typed",
                          placeholder="הקלד כאן שאלה")

if "chosen" not in st.session_state:
    st.session_state.chosen = None


def choose(q):
    st.session_state.chosen = q


# a new text in the box cancels the previously chosen question (otherwise its old answer stays on screen)
if st.session_state.get("last_typed") != typed:
    st.session_state.chosen = None
    st.session_state.last_typed = typed

question = st.session_state.chosen
if typed.strip():
    if is_keyword_query(typed):
        options, mode = suggest(typed, retriever)
        if options:
            st.markdown("**כמה שאלות אפשריות בנושא:**" if mode == "direct"
                        else "**המילה מופיעה במסמכים שעוסקים בנושאים הבאים (לחצו על שאלה):**")
            for i, q in enumerate(options):
                st.button(q, key=f"sg{i}", on_click=choose, args=(q,))
        else:
            st.info("לא נמצאו שאלות מוכנות למילה הזו. אפשר לנסח שאלה שלמה.")
        question = st.session_state.chosen
    else:
        question = typed.strip()
        st.session_state.chosen = None
else:
    st.markdown("**דוגמאות:**")
    for i, q in enumerate(EXAMPLES):
        st.button(q, key=f"ex{i}", on_click=choose, args=(q,))
    question = st.session_state.chosen

if question:
    st.markdown(f"##### {question}")
    show_answer(question, as_of, retriever)
