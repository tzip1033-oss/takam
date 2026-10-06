"""Step 2: retrieval.

- Lexical: BM25 over "context + text" (context = doc type, id, title, heading), with simple
  Hebrew prefix handling (ה/ו/ב/ל/מ/כ/ש are also indexed stripped).
- Semantic (optional): local multilingual embeddings via sentence-transformers. Needed for
  vocabulary gaps such as "הלוואה" vs "מקדמה". Enabled automatically if the package is
  installed; set USE_EMBEDDINGS=0 to force lexical-only.
- Validity filter runs in CODE before ranking (default: as of today).
- Returns refusal signals (coverage, cosine) so the caller can refuse in code.

Try it:  python src/retrieve.py "כמה שעות עובד סטודנט בחודש"
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
EMB_CACHE = ROOT / "data" / "processed" / "embeddings.npy"
EMB_META = ROOT / "data" / "processed" / "embeddings.json"
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

STOP = set(
    "של את על עם מה כמה איך האם לי אני זה אם או גם כל יש מגיע מגיעה מגיעים מגיעות הוא היא "
    "הם הן לא כן אבל כי אז רק עוד בין כדי לפי מן אל עד ללא מי מתי איפה למה אפשר צריך צריכה "
    "קורה קורים בשירות המדינה".split()
)
PREFIXES = "ובכלמהש"
BIDI = re.compile(r"[\u200E\u200F\u202A-\u202E]")


def tokenize(text: str, with_variants: bool = True):
    text = BIDI.sub("", text)
    words = re.findall(r"[א-ת]{2,}|\d+(?:[.,]\d+)*|[A-Za-z]{2,}", text)
    out = []
    for w in words:
        if w in STOP:
            continue
        out.append(w)
        if with_variants:
            s = w
            for _ in range(2):
                if len(s) >= 4 and s[0] in PREFIXES:
                    s = s[1:]
                    out.append(s)
                else:
                    break
    return out


def _parse_date(s):
    try:
        return date.fromisoformat(s) if s else None
    except ValueError:
        return None


def is_valid(chunk, as_of: date) -> bool:
    vf, vt = _parse_date(chunk.get("valid_from")), _parse_date(chunk.get("valid_to"))
    if vf and as_of < vf:
        return False
    if vt and as_of > vt:
        return False
    return True


class Retriever:
    def __init__(self, use_embeddings: bool | None = None):
        self.chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf8")]
        self.texts = [c["context"] + "\n" + c["text"] for c in self.chunks]
        self.tok = [tokenize(t) for t in self.texts]
        self.tokset = [set(t) for t in self.tok]
        self.bm25 = BM25Okapi(self.tok)
        self.emb = None
        self.model = None
        if use_embeddings is None:
            use_embeddings = os.getenv("USE_EMBEDDINGS", "1") != "0"
        if use_embeddings:
            self._init_embeddings()

    def _init_embeddings(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[retrieve] sentence-transformers not installed -> lexical only", file=sys.stderr)
            return
        self.model = SentenceTransformer(EMB_MODEL)
        key = {"model": EMB_MODEL, "n": len(self.chunks),
               "ids": [c["chunk_id"] for c in self.chunks][:5]}
        if EMB_CACHE.exists() and EMB_META.exists() and json.load(open(EMB_META)) == key:
            self.emb = np.load(EMB_CACHE)
            return
        passages = ["passage: " + t for t in self.texts]
        self.emb = self.model.encode(passages, normalize_embeddings=True,
                                     show_progress_bar=True, batch_size=16)
        np.save(EMB_CACHE, self.emb)
        json.dump(key, open(EMB_META, "w"))

    def search(self, query: str, k: int = 5, as_of: date | None = None):
        as_of = as_of or date.today()
        qtok = tokenize(query)
        base = set(tokenize(query, with_variants=False))
        lex = np.array(self.bm25.get_scores(qtok)) if qtok else np.zeros(len(self.chunks))
        lex_norm = lex / lex.max() if lex.max() > 0 else lex
        cos = None
        if self.emb is not None:
            q = self.model.encode(["query: " + query], normalize_embeddings=True)[0]
            cos = self.emb @ q

        valid = np.array([is_valid(c, as_of) for c in self.chunks])
        def ranks(scores):
            order = np.argsort(-np.where(valid, scores, -1e9))
            r = np.empty(len(scores), dtype=int)
            r[order] = np.arange(len(scores))
            return r
        fused = 1.0 / (60 + ranks(lex))
        if cos is not None:
            fused = fused + 1.0 / (60 + ranks(cos))
        fused = np.where(valid, fused, -1)

        results = []
        for i in np.argsort(-fused)[:k]:
            if fused[i] < 0:
                continue
            cov = len(base & self.tokset[i]) / len(base) if base else 0.0
            results.append(dict(
                chunk=self.chunks[i], fused=float(fused[i]), lex=float(lex[i]),
                lex_norm=float(lex_norm[i]), cos=float(cos[i]) if cos is not None else None,
                coverage=cov))
        return results


def label(c):
    return f"{c['doc_type']} {c['doc_id']} | {c['section']} | מהדורה {c['edition'] or '?'}"


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "כמה שעות עובד סטודנט בחודש"
    r = Retriever()
    for i, h in enumerate(r.search(q, k=5), 1):
        c = h["chunk"]
        cos = f" cos={h['cos']:.2f}" if h["cos"] is not None else ""
        print(f"{i}. {label(c)}  cov={h['coverage']:.2f} lex={h['lex']:.1f}{cos}")
        print("   ", c["text"][:110].replace("\n", " "))
