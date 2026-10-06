"""Keyword -> suggested questions.

When the user types only a word or two ("לידה", "סטודנט"), we do not guess an answer.
We offer ready-made questions (written by hand in eval/questions.csv, type "answer"):
  1. questions whose own wording contains the keyword, then
  2. questions about documents that the keyword retrieves.
Production note: the question list would be curated by the content owners (or generated
offline and reviewed), never generated live without review.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import tokenize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.csv"


def prepared_questions():
    rows = list(csv.DictReader(open(QUESTIONS, encoding="utf8")))
    return [r for r in rows if r["type"] == "answer"]


def is_keyword_query(text: str) -> bool:
    t = text.strip()
    return bool(t) and "?" not in t and len(t.split()) <= 2


def suggest(keyword: str, retriever, limit: int = 4):
    """Returns (questions, mode): mode is 'direct' (the word is in the question) or
    'context' (only the documents mention it)."""
    kw = set(tokenize(keyword, with_variants=False))
    qs = prepared_questions()
    # documents in which a chunk really contains the keyword (not just a vague similarity)
    hit_docs = []
    for h in retriever.search(keyword, k=8):
        if kw & set(tokenize(h["chunk"]["text"], with_variants=False)) and h["chunk"]["doc_id"] not in hit_docs:
            hit_docs.append(h["chunk"]["doc_id"])
    scored = []
    for q in qs:
        words = set(tokenize(q["question"]))
        text_match = 1 if kw & words else 0
        doc_rank = hit_docs.index(q["expected_doc"]) if q["expected_doc"] in hit_docs else None
        if not text_match and doc_rank is None:
            continue
        scored.append((-text_match, doc_rank if doc_rank is not None else 99, q["question"]))
    direct = sorted(x for x in scored if x[0] == -1)
    if direct:
        return [x[2] for x in direct[:limit]], "direct"
    return [x[2] for x in sorted(scored)[:3]], "context"
