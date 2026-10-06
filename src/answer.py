#!/usr/bin/env python3
"""Step 3: answer generation via Gemini - using REST API (not gRPC)."""

import os
import json
import requests
from dotenv import load_dotenv
from retrieve import retrieve

load_dotenv()
GEMINI_AVAILABLE = True

def generate_answer(question, as_of_date=None, threshold=0.3):
    """
    Retrieve chunks and generate answer via Gemini.

    If top score < threshold: refuse + point to takam@mof.gov.il
    Otherwise: send chunks to Gemini with citation instructions.
    """
    chunks = retrieve(question, top_k=5, as_of_date=as_of_date)

    if not chunks or chunks[0]['score'] < threshold:
        return {
            "answer": "לא מצאתי מידע רלוונטי בבסיס הנתונים שלי.",
            "sources": [],
            "confidence": 0.0,
            "refusal_reason": "Low confidence - refer to official channel"
        }

    if not GEMINI_AVAILABLE:
        return {
            "answer": "❌ Gemini API not available. Install: pip install google-genai",
            "sources": chunks[:3],
            "confidence": 0.0,
            "refusal_reason": "API not configured"
        }

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "answer": "❌ GEMINI_API_KEY not set in .env",
            "sources": chunks[:3],
            "confidence": chunks[0]['score'],
            "refusal_reason": "Missing API key"
        }

    # Format context
    context = "\n\n".join([
        f"📄 {c['doc_id']} (מהדורה {c['edition']}) - סעיף {c['section']}\n{c['text'][:500]}"
        for c in chunks
    ])

    prompt = f"""אתה עוזר מידע למערכת תכם (מערכת מידע לעובדי הממשלה בישראל).

שאלה: {question}

מידע רלוונטי מהקורפוס:
{context}

הנחיות:
1. ענה רק מהמידע בקורפוס שלמעלה
2. ציין את מקור המידע: "הוראה [מספר], מהדורה [מספר]"
3. אם המידע לא מובן או לא רלוונטי: אמור "לא נמצא מידע מדויק" והפנה ל-takam@mof.gov.il

תשובה:"""

    # Use REST API instead of gRPC (works with Netfree)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, json=payload, timeout=5, verify=True)
        response.raise_for_status()
        data = response.json()
        if "candidates" in data and data["candidates"]:
            answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            answer_text = "❌ No response"
    except requests.exceptions.Timeout:
        answer_text = f"**תשובה (ממקורות)**\n\n{chunks[0]['text'][:400]}..."
    except Exception as e:
        error_str = str(e).lower()
        if any(x in error_str for x in ["ssl", "certificate", "connection"]):
            answer_text = f"**תשובה (ממקורות)**\n\n{chunks[0]['text'][:400]}..."
        else:
            answer_text = f"❌ {str(e)[:100]}"

    return {
        "answer": answer_text,
        "sources": chunks[:3],
        "confidence": chunks[0]['score'],
        "refusal_reason": None
    }

if __name__ == "__main__":
    result = generate_answer("מה היא תוספת יוקר?")
    print(result["answer"])
    print("\n--- Sources ---")
    for s in result["sources"]:
        print(f"- {s['doc_id']}: {s['text'][:80]}...")
