import streamlit as st
import subprocess
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="תכם", layout="wide")

st.title("🏛️ תכם - עזר מידע לעובדי המדינה")
st.markdown("*מערכת חיפוש בהוראות שכר וזכויות עובדים*")

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"

# Check if data is ingested
if not CHUNKS_JSONL.exists():
    st.warning("⚙️ First run: indexing documents...", icon="⏳")
    try:
        subprocess.run(
            [".venv/Scripts/python", "src/ingest.py"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            check=True
        )
        st.success("✅ Indexing complete!")
        st.rerun()
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Ingest failed: {e.stderr.decode()}")
        st.stop()

# Load answer generator
try:
    from answer import generate_answer
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

with st.sidebar:
    st.header("⚙️ הגדרות")
    as_of_date = st.date_input(
        "תאריך יעילות (אופציונלי)",
        value=datetime.now().date()
    )

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.subheader("💬 שאלתך")
    question = st.text_area(
        "הקלד את שאלתך בנושא שכר וזכויות עובדים:",
        placeholder="לדוגמה: מה הוא הסכום של תוספת יוקר?",
        height=100,
        label_visibility="collapsed"
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        submit = st.button("🔍 חפש", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ נקה", use_container_width=True):
            st.rerun()

with col2:
    st.subheader("📄 תשובה")

    if submit and question:
        with st.spinner("🔎 חוזר בחיפוש..."):
            result = generate_answer(question, as_of_date=as_of_date)

        # Display answer
        st.markdown("---")
        if result.get("refusal_reason"):
            st.error(f"❌ {result['answer']}\n\n📧 צור קשר: takam@mof.gov.il")
        else:
            st.success(result['answer'])

        # Display sources
        st.markdown("---")
        st.subheader("📚 מקורות")

        for src in result.get("sources", []):
            doc_id = src['doc_id']
            section = src['section']
            confidence = f"{src['score']*100:.0f}%"
            full_text = src['text']

            # Highlight question keywords in source text
            highlighted_text = full_text
            for word in question.split():
                if len(word) > 2:  # Skip short words
                    highlighted_text = highlighted_text.replace(
                        word,
                        f"🟡 **{word}**"
                    )

            with st.expander(
                f"📄 הוראה {doc_id} - סעיף {section} ({confidence})",
                expanded=True
            ):
                st.markdown(f"""
**מקור:** H.{doc_id} מהדורה {src['edition']}, בתוקף מ-{src['valid_from']}
                """)

                # Scrollable text container
                st.markdown(f"""
<div style="height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9; font-size: 14px; line-height: 1.6;">
{highlighted_text}
</div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.warning("""
⚠️ **זו מערכת ניסיונית** - ביקור בתכם הממשלתי לאימות

אם יש שגיאה או שאלה: [takam@mof.gov.il](mailto:takam@mof.gov.il)
        """)

    elif submit and not question:
        st.error("⚠️ אנא הקלד שאלה")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<small>© 2026 משרד האוצר - תכם | <a href='mailto:takam@mof.gov.il'>צור קשר</a></small>
</div>
""", unsafe_allow_html=True)
