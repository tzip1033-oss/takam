"""Step 5: evaluation.

python src/evaluate.py            retrieval only (no API key needed): is the right document
                                  in the top-k, and what do the refusal signals look like?
python src/evaluate.py --llm      also generate answers (needs a key in .env) and print them
                                  for manual checking.

Types in eval/questions.csv: answer | answer_stretch | refusal | refusal_pointer |
temporal_stretch. 'stretch' rows are expected to fail until we add annex extraction
or older editions.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import Retriever, label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
K = 5


def main(use_llm=False):
    rows = list(csv.DictReader(open(ROOT / "eval" / "questions.csv", encoding="utf8")))
    r = Retriever()
    print(f"embeddings: {'yes' if r.emb is not None else 'NO (lexical only)'}  k={K}\n")
    print(f"{'id':>2} {'type':16} {'expected':9} {'rank':>4} {'top cov':>7} {'top lex':>7} {'top cos':>7}  question")
    hits = total = 0
    for q in rows:
        res = r.search(q["question"], k=K)
        docs = [h["chunk"]["doc_id"] for h in res]
        exp = q["expected_doc"]
        rank = (docs.index(exp) + 1) if exp and exp in docs else None
        top = res[0] if res else None
        cos = f"{top['cos']:.2f}" if top and top["cos"] is not None else "  -"
        print(f"{q['id']:>2} {q['type']:16} {exp or '-':9} {str(rank or '-'):>4} "
              f"{top['coverage'] if top else 0:7.2f} {top['lex'] if top else 0:7.1f} {cos:>7}  {q['question'][:46]}")
        if q["type"] == "answer":
            total += 1
            hits += 1 if rank else 0
        if use_llm:
            from answer import answer_question
            out = answer_question(q["question"])
            print("   expected:", q["expected_answer"][:120])
            print("   got     :", out["text"][:300].replace("\n", " "))
    print(f"\nrecall@{K} on 'answer' questions: {hits}/{total}")
    print("Look at the refusal rows (types refusal*): their top cov/lex/cos are the values "
          "we must stay BELOW to refuse. Compare with the 'answer' rows to pick a threshold.")


if __name__ == "__main__":
    main(use_llm="--llm" in sys.argv)
