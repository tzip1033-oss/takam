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
import time
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
    stats = dict(ans=0, ans_ok=0, ans_verified=0, ans_partly=0, ans_refused=0, ans_error=0, ref=0, ref_ok=0, ref_code=0, ref_error=0)
    wrong_refusals, missed_refusals, unverified = [], [], []
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
            for _ in range(2):              # free-tier quota / overload: wait and retry, so an API error is not read as a refusal
                if not out["reason"].startswith("llm_error"):
                    break
                time.sleep(20)
                out = answer_question(q["question"])
            print("   expected:", q["expected_answer"][:120])
            got = out["answer"] if not out["refused"] else "REFUSED (" + out["reason"][:60] + ")"
            if out["timing"].get("model"):
                print("   model   :", out["timing"]["model"])
            print("   got     :", got[:300].replace("\n", " "))
            if not out["refused"]:
                print("   verified:", out["verified"], "(numbers in source:", out["numbers_in_source"], ") | source:", label(out["source"]["chunk"]))
            if q["type"] == "answer":
                stats["ans"] += 1
                if out["reason"].startswith("llm_error"):
                    stats["ans_error"] += 1
                elif out["refused"]:
                    stats["ans_refused"] += 1
                    wrong_refusals.append(q["id"])
                else:
                    stats["ans_ok"] += out["source"]["chunk"]["doc_id"] == exp     # right document
                    stats["ans_verified"] += out["verified"]
                    stats["ans_partly"] += (not out["verified"]) and out["numbers_in_source"]
                    if not out["verified"]:
                        unverified.append(q["id"])
            elif q["type"] in ("refusal", "refusal_pointer", "temporal_stretch"):
                stats["ref"] += 1
                stats["ref_error"] += out["reason"].startswith("llm_error")
                stats["ref_ok"] += out["refused"] and not out["reason"].startswith("llm_error")
                stats["ref_code"] += out["refused"] and not out["reason"].startswith(("model", "llm"))
                if not out["refused"]:
                    missed_refusals.append(q["id"])
            if out["pointers"] and out["refused"]:
                print("   pointers:", ", ".join(out["pointers"]))
    print(f"\nrecall@{K} on 'answer' questions: {hits}/{total}")
    if use_llm:
        a, f = stats["ans"], stats["ref"]
        print(f"\nanswer questions: {a}  answered from the expected document: {stats['ans_ok']}/{a}  "
              f"wrongly refused: {stats['ans_refused']}/{a} {wrong_refusals}  API errors: {stats['ans_error']}\n"
              f"  answered: {a - stats['ans_refused'] - stats['ans_error']}  verified: {stats['ans_verified']}  "
              f"partly verified (numbers elsewhere in the source): {stats['ans_partly']}  not verified: {unverified}")
        print(f"refusal questions: {f}  correctly refused: {stats['ref_ok']}/{f} "
              f"(of them refused in code before the model: {stats['ref_code']})  answered anyway: {missed_refusals}  API errors: {stats['ref_error']}")
    print("Look at the refusal rows (types refusal*): their top cov/lex/cos are the values "
          "we must stay BELOW to refuse. Compare with the 'answer' rows to pick a threshold.")


if __name__ == "__main__":
    main(use_llm="--llm" in sys.argv)
