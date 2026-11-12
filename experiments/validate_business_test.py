"""Validate data/business_test.jsonl (evaluation-only business-document hallucination test set).

Checks the record schema, span offsets, per-document structure and label / error-type balance,
then prints counts by domain and error type. Exits non-zero if any check fails.

    python experiments/validate_business_test.py [path]
"""
import collections
import json
import re
import sys
from pathlib import Path

ERROR_TYPES = {
    "number_change", "date_or_time_arithmetic", "calculation_error", "entity_swap", "negation_flip",
    "dropped_condition", "unsupported_addition", "conflated_clauses", "wrong_unit_or_currency",
    "overgeneralization",
}
KEYS = {"id", "domain", "question", "documents", "response", "label", "error_type", "spans", "explanation"}
ID_RE = re.compile(r"^biz_(\d+)_([abcd])$")


def main(path):
    errors = []
    recs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = set()
    by_doc = collections.defaultdict(list)
    for r in recs:
        rid = r.get("id")
        if set(r) != KEYS:
            errors.append(f"{rid}: keys {sorted(set(r) ^ KEYS)} missing/extra")
            continue
        m = ID_RE.match(rid or "")
        if not m:
            errors.append(f"{rid}: bad id")
            continue
        if rid in ids:
            errors.append(f"{rid}: duplicate id")
        ids.add(rid)
        by_doc[m.group(1)].append(r)
        if not (isinstance(r["documents"], list) and len(r["documents"]) == 1 and isinstance(r["documents"][0], str)):
            errors.append(f"{rid}: documents must be a list with one string")
            continue
        nw = len(r["documents"][0].split())
        if not 150 <= nw <= 700:
            errors.append(f"{rid}: document has {nw} words (need 150-700)")
        for k in ("domain", "question", "response"):
            if not isinstance(r[k], str) or not r[k].strip():
                errors.append(f"{rid}: empty {k}")
        resp = r["response"]
        ns = len(re.findall(r"[.!?](\s|$)", resp))
        if not 1 <= ns <= 6:
            errors.append(f"{rid}: response has {ns} sentences (need 1-6)")
        if r["label"] == 0:
            if r["error_type"] is not None or r["spans"] != [] or r["explanation"] is not None:
                errors.append(f"{rid}: faithful record must have error_type=None, spans=[], explanation=None")
        elif r["label"] == 1:
            if r["error_type"] not in ERROR_TYPES:
                errors.append(f"{rid}: bad error_type {r['error_type']!r}")
            if not (isinstance(r["explanation"], str) and r["explanation"].strip()):
                errors.append(f"{rid}: missing explanation")
            if not r["spans"]:
                errors.append(f"{rid}: hallucinated record needs at least one span")
            for sp in r["spans"]:
                if not (isinstance(sp, list) and len(sp) == 2 and all(isinstance(x, int) for x in sp)
                        and 0 <= sp[0] < sp[1] <= len(resp)):
                    errors.append(f"{rid}: bad span {sp}")
                    continue
                text = resp[sp[0]:sp[1]]
                if text != text.strip():
                    errors.append(f"{rid}: span {sp} has leading/trailing whitespace: {text!r}")
        else:
            errors.append(f"{rid}: label must be 0 or 1")

    for doc, rs in by_doc.items():
        if sorted(r["id"][-1] for r in rs) != list("abcd"):
            errors.append(f"doc {doc}: needs exactly answers a-d")
        if len({r["documents"][0] for r in rs}) != 1 or len({r["question"] for r in rs}) != 1:
            errors.append(f"doc {doc}: answers must share one document and question")
        labs = [r["label"] for r in rs]
        if labs.count(1) != 2:
            errors.append(f"doc {doc}: needs 2 faithful and 2 hallucinated answers")
        types = [r["error_type"] for r in rs if r["label"] == 1]
        if len(set(types)) != len(types):
            errors.append(f"doc {doc}: hallucinated answers must have different error types")

    n = len(recs)
    pos = sum(r.get("label") == 1 for r in recs)
    etypes = collections.Counter(r["error_type"] for r in recs if r.get("label") == 1)
    domains = collections.Counter()
    for doc, rs in by_doc.items():
        domains[rs[0]["domain"]] += 1
    if len(by_doc) < 60:
        errors.append(f"only {len(by_doc)} documents (need >= 60)")
    if n < 240:
        errors.append(f"only {n} records (need >= 240)")
    if pos * 2 != n:
        errors.append(f"label balance {pos}/{n} is not exactly half")
    for t in ERROR_TYPES:
        if etypes[t] < 8:
            errors.append(f"error type {t} appears {etypes[t]} times (need >= 8)")

    print(f"file: {path}")
    print(f"documents: {len(by_doc)}  records: {n}  hallucinated: {pos}  faithful: {n - pos}")
    print("\ndocuments by domain:")
    for d, c in sorted(domains.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {d:24s} {c:3d}")
    print("\nhallucinated answers by error type:")
    for t in sorted(ERROR_TYPES):
        print(f"  {t:24s} {etypes[t]:3d}")
    wcs = [len(rs[0]["documents"][0].split()) for rs in by_doc.values()]
    if wcs:
        print(f"\ndocument length (words): min {min(wcs)}, max {max(wcs)}, mean {sum(wcs) / len(wcs):.0f}")
    if errors:
        print(f"\nFAILED with {len(errors)} problem(s):")
        for e in errors:
            print("  - " + e)
        return 1
    print("\nOK: all checks passed")
    return 0


if __name__ == "__main__":
    default = Path(__file__).resolve().parent.parent / "data" / "business_test.jsonl"
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default))
