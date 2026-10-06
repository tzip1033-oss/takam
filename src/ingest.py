"""Step 1: read data/raw/*.docx -> section-level chunks with metadata.

Run:  python src/ingest.py            (writes data/processed/chunks.jsonl)
      python src/ingest.py --check    (also prints numbering sanity checks)

Design notes
- Word auto-numbering is NOT in the paragraph text. The documents use paragraph styles
  that encode the level ("כותרת רמה 1", "סעיף רמה 2", ...), so we recompute the section
  numbers (2.1.4 etc.) ourselves by counting levels. pandoc breaks this numbering.
- Edition / validity come from data/metadata.csv (taken from the site), not from the file.
- Tables are kept (rows joined with " | "). Annexes are chunked separately.
- Embedded objects (PDF / Excel inside the DOCX) are NOT extracted yet: we only warn.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
META = ROOT / "data" / "metadata.csv"
OUT = ROOT / "data" / "processed" / "chunks.jsonl"
MAX_CHARS = 1500

LEVEL_RE = re.compile(r"רמה\s*(\d)")
BIDI = re.compile(r"[\u200E\u200F\u202A-\u202E]")

ALL_UNITS: dict = {}


def clean(text: str) -> str:
    return " ".join(BIDI.sub("", text).split())


def iter_blocks(doc):
    """Yield paragraphs and tables in document order."""
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def para_level(p: Paragraph):
    name = p.style.name or ""
    if name == "Heading 1":
        return 1
    m = LEVEL_RE.search(name)
    if m:
        return int(m.group(1))
    return None


def numbering_off(p: Paragraph) -> bool:
    """numId == 0 means: this paragraph is explicitly NOT numbered."""
    pPr = p._p.pPr
    if pPr is None or pPr.numPr is None:
        return False
    n = pPr.numPr.numId
    return n is not None and n.val == 0


def table_text(tbl: Table) -> str:
    """One line per row, cells joined with ' | '."""
    rows = []
    for r in tbl.rows:
        seen, cells = set(), []
        for c in r.cells:
            if id(c._tc) in seen:
                continue
            seen.add(id(c._tc))
            cells.append(clean(c.text) or "-")
        if any(x != "-" for x in cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def embedded_objects(path: Path):
    with zipfile.ZipFile(path) as z:
        return [n for n in z.namelist() if n.startswith("word/embeddings/")]


def parse_doc(path: Path):
    """Return a list of units: dict(kind, label, is_new, heading, text)."""
    doc = Document(path)
    counters = [0] * 6
    h1 = ""
    last_label = ""
    annex = None
    wait_annex_title = False
    units = []

    for blk in iter_blocks(doc):
        if isinstance(blk, Table):
            txt = table_text(blk)
            if txt:
                if annex:
                    units.append(dict(kind="annex_table", label=annex["label"], is_new=True,
                                      heading=annex["title"], text=txt))
                else:
                    units.append(dict(kind="table", label=last_label, is_new=True,
                                      heading=h1, text=txt))
            continue

        text = clean(blk.text)
        if not text:
            continue
        style = blk.style.name or ""

        if style == "נספח - כותרת":
            annex = {"label": text, "title": ""}
            wait_annex_title = True
            continue
        if style == "נספח - תיאור" and annex is not None and wait_annex_title:
            annex["title"] = text
            wait_annex_title = False
            continue
        if annex is not None:
            units.append(dict(kind="annex", label=annex["label"], is_new=True,
                              heading=annex["title"], text=text))
            continue

        lvl = para_level(blk)
        if lvl and not numbering_off(blk):
            counters[lvl - 1] += 1
            for i in range(lvl, len(counters)):
                counters[i] = 0
            label = ".".join(str(c) for c in counters[:lvl])
            if lvl == 1:
                h1 = text
            last_label = label
            units.append(dict(kind="body", label=label, is_new=True, heading=h1, text=text))
        else:
            units.append(dict(kind="body", label=last_label or "0", is_new=False,
                              heading=h1, text=text))
    return units


SOFT_MIN = 700


def group_key(u):
    if u["kind"].startswith("annex"):
        return ("annex", u["label"])
    return ("body", u["label"].split(".")[0])


def is_heading(u):
    """Short level-2 line such as 'תשלום מקדמת שכר' -> good place to start a new chunk."""
    return (u["kind"] == "body" and u["is_new"] and u["label"].count(".") == 1
            and len(u["text"]) <= 60)


def build_chunks(units, meta):
    raw_chunks = []

    def flush(k, buf):
        if not buf:
            return
        lines = []
        for u in buf:
            prefix = f"{u['label']} " if (u["kind"] == "body" and u["is_new"]) else ""
            lines.append(prefix + u["text"])
        first, last = buf[0], buf[-1]
        raw_chunks.append(dict(
            section_first=first["label"], section_last=last["label"],
            heading=first["heading"], is_annex=k[0] == "annex",
            has_table=any("table" in u["kind"] for u in buf),
            text="\n".join(lines),
        ))

    groups = []
    for u in units:
        k = group_key(u)
        if not groups or groups[-1][0] != k:
            groups.append((k, []))
        groups[-1][1].append(u)

    for k, us in groups:
        buf, size = [], 0
        for u in us:
            if "table" in u["kind"]:
                flush(k, buf)
                buf, size = [], 0
                cur, cur_n = [], 0
                for r in u["text"].split("\n"):
                    if cur and cur_n + len(r) > MAX_CHARS:
                        flush(k, [dict(u, text="\n".join(cur))])
                        cur, cur_n = [], 0
                    cur.append(r)
                    cur_n += len(r)
                flush(k, [dict(u, text="\n".join(cur))])
                continue
            n = len(u["text"])
            if buf and (size + n > MAX_CHARS or (size >= SOFT_MIN and is_heading(u))):
                flush(k, buf)
                buf, size = [], 0
            buf.append(u)
            size += n
        flush(k, buf)

    out = []
    for i, c in enumerate(raw_chunks):
        sec = c["section_first"] if c["section_first"] == c["section_last"] \
            else f"{c['section_first']}-{c['section_last']}"
        c.update(
            chunk_id=f"{meta['doc_id']}#{i:03d}",
            doc_id=meta["doc_id"], doc_type=meta["doc_type"], title=meta["title"],
            edition=meta["edition"], valid_from=meta["valid_from"], valid_to=meta["valid_to"],
            parent=meta["parent"], source_url=meta["source_url"], section=sec,
        )
        c["context"] = f"{meta['doc_type']} {meta['doc_id']} {meta['title']} | {c['heading']}"
        c.update(extract_refs(c["text"]))
        out.append(c)
    return out


REF_TAKAM = re.compile(r"(?:מס'|מספר|תכ[\"״]ם)[^\d]{0,80}?(\d{1,2}\.\d{1,2}(?:\.\d{1,2}){0,2})")
REF_TAKSHIR = re.compile(r"תקשי[\"״]ר[^\d]{0,60}?(\d{1,2}\.\d{1,3})")
REF_INTERNAL = re.compile(r"סעיפ(?:ים|ף)\s*(\d(?:\.\d+)+)")


def extract_refs(text):
    t = BIDI.sub("", text)
    return dict(
        refs_takam=sorted(set(REF_TAKAM.findall(t))),
        refs_takshir=sorted(set(REF_TAKSHIR.findall(t))),
        refs_internal=sorted(set(REF_INTERNAL.findall(t))),
    )


def load_metadata():
    rows = {}
    with open(META, encoding="utf8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["file"]] = r
    return rows


def main(check=False):
    meta = load_metadata()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_chunks, report = [], []
    for path in sorted(RAW.glob("*.docx")):
        m = meta.get(path.name)
        if m is None:
            print(f"!! no metadata row for {path.name} - skipped")
            continue
        units = parse_doc(path)
        ALL_UNITS[m["doc_id"]] = units
        chunks = build_chunks(units, m)
        all_chunks.extend(chunks)
        report.append((m["doc_id"], len(units), len(chunks),
                       sum(len(c["text"]) for c in chunks), embedded_objects(path)))

    with open(OUT, "w", encoding="utf8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(all_chunks)} chunks -> {OUT}\n")
    print(f"{'doc':10} {'units':>6} {'chunks':>7} {'chars':>8}  warnings")
    for doc_id, nu, nc, nch, emb in report:
        warn = f"{len(emb)} embedded object(s) NOT extracted" if emb else ""
        print(f"{doc_id:10} {nu:6d} {nc:7d} {nch:8d}  {warn}")
    todo = [m["file"] for m in meta.values() if "TODO" in (m["verified_on_site"] + m["doc_id"])]
    if todo:
        print("\nmetadata TODO rows:", ", ".join(todo))
    if check:
        sanity()


def sanity():
    """Numbering checks against references written inside the documents themselves."""
    print("\n--- numbering sanity checks ---")

    def find(doc_id, needle):
        for u in ALL_UNITS.get(doc_id, []):
            if needle in u["text"]:
                return u["label"]
        return None

    checks = [
        ("13.1.5", "נוכח החגים פסח וסוכות", "2.1.4",
         "holiday months (the document itself cites it as 2.1.4)"),
        ("13.1.10", "ברציפות, זכאי לדמי לידה", "1.2", "paid birth period (cited as 1.2)"),
        ("13.1.10", "היעדרות בתום תקופת הלידה וההורות", "2.8", "absence after leave (cited as 2.8)"),
    ]
    for doc_id, needle, expected, why in checks:
        got = find(doc_id, needle)
        status = "OK " if got == expected else "?? "
        print(f"{status}{doc_id}: '{needle[:28]}...' -> {got} (expected {expected}) {why}")


if __name__ == "__main__":
    main(check="--check" in sys.argv)
