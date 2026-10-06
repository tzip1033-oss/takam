"""Step 5: evaluation.

python src/evaluate.py            retrieval only (no API key needed): is the right document
                                  in the top-k, and what do the refusal signals look like?
python src/evaluate.py --llm      also generate answers (needs a key in .env) and print them
                                  for manual checking.

Types in eval/questions.csv: answer | answer_stretch | refusal_pointer |
temporal_stretch. 'stretch' rows are expected to fail until we add annex extraction
or older editions.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import Retriever, label

ROOT = Path(__file__).resolve().parent.parent
K = 5


def main(use_llm=False):
    rows = list(csv.DictReader(open(ROOT / "eval" / "questions.csv", encoding="utf8")))
    r = Retriever()
    print(f"embeddings: {'yes' if r.emb is not None else 'NO (lexical only)'}  k={K}\n")
    print(f"{'id':>2} {'type':16} {'doc_id':9} {'rank':>4} {'cov':>6} {'lex':>6} {'cos':>6}  question")
    hits = total = 0
    for q in rows:
        res = r.search(q["question"], k=K)
        docs = [h["chunk"]["doc_id"] for h in res]
        exp = q["expected_source"].split()[0] if q.get("expected_source") else None
        rank = (docs.index(exp) + 1) if exp and exp in docs else None
        top = res[0] if res else None
        cos = f"{top['cos']:.2f}" if top and top["cos"] is not None else " -"
        cov = f"{top['coverage']:.2f}" if top else " -"
        lex = f"{top['lex']:.1f}" if top else " -"
        print(f"{q['id']:>2} {q['type']:16} {exp or '-':9} {str(rank or '-'):>4} "
              f"{cov:>6} {lex:>6} {cos:>6}  {q['question'][:40]}")
        if q["type"] == "answer":
            total += 1
            hits += 1 if rank else 0
        if use_llm:
            from answer import generate_answer
            out = generate_answer(q["question"])
            print("   expected:", q["expected_answer"][:120])
            print("   got     :", out["answer"][:300].replace("\n", " "))
    print(f"\nrecall@{K} on 'answer' questions: {hits}/{total}")
    print("Look at the refusal rows (types refusal*): their top coverage/lex/cos are the values")
    print("we must stay BELOW to refuse. Compare with the 'answer' rows to pick a threshold.")


if __name__ == "__main__":
    main(use_llm="--llm" in sys.argv)
