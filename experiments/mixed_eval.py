"""The routed chatbot on a mix of everyday requests, not just trivia: 120 messages across factual questions
(TriviaQA), open questions (Natural Questions), math word problems (GSM8K), how-to, code, creative writing and
advice / small talk (results/mixed_eval_items.json). Sent to a running server through vt.client (streamed,
a silent request sent again once) and scored per kind:

    routing      share sent the way each kind should go, and the confusion
    factual,     answer correctness (reference answers), and whether the verdict tells right from wrong:
    open, math   caught = wrong answers called likely wrong; false alarms = right ones called likely wrong
    the rest     false alarms (any "likely wrong" verdict), [unsure:] markers left in, grounding of how-to
    serving      errors, retried requests, seconds

    python experiments/mixed_eval.py --url https://<...>.modal.run --workers 4 --out mixed.json
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correction_rate import is_right  # noqa: E402


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace("[unsure:]", ""))
    return float(nums[-1].replace(",", "")) if nums else None


def score(rows):
    out = {"n": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
           "retried_and_answered": sum((r.get("attempts") or 1) > 1 for r in rows if not r.get("error"))}
    ok = [r for r in rows if not r.get("error")]
    secs = sorted(r["seconds"] for r in ok)
    out["seconds_median"] = secs[len(secs) // 2] if secs else None
    out["seconds_p95"] = secs[int(0.95 * (len(secs) - 1))] if secs else None
    out["routing"] = {"accuracy": round(sum(r["route"] == r["expect"] for r in ok) / max(1, len(ok)), 3)}
    for kind in sorted({r["kind"] for r in ok}):
        sub = [r for r in ok if r["kind"] == kind]
        k = {"n": len(sub), "routed_as": dict(Counter(r["route"] for r in sub)),
             "routing_accuracy": round(sum(r["route"] == r["expect"] for r in sub) / len(sub), 3),
             "verdicts": dict(Counter((r.get("assessment") or {}).get("verdict", "none") for r in sub)),
             "markers_left": sum("[unsure" in r["text"] for r in sub),
             "grounded": sum(bool(r.get("grounded")) for r in sub)}
        judged = [r for r in sub if r.get("right") is not None]
        if judged:
            right = [r for r in judged if r["right"]]
            wrong = [r for r in judged if not r["right"]]
            lw = lambda r: (r.get("assessment") or {}).get("verdict") == "likely wrong"  # noqa: E731
            k.update(accuracy=round(len(right) / len(judged), 3),
                     wrong_caught=f"{sum(map(lw, wrong))}/{len(wrong)}",
                     right_false_alarm=f"{sum(map(lw, right))}/{len(right)}")
        else:
            k["false_alarms"] = sum((r.get("assessment") or {}).get("verdict") == "likely wrong" for r in sub)
        out[kind] = k
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--items", default="results/mixed_eval_items.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="first n of each kind (smoke test)")
    ap.add_argument("--out", default="mixed.json")
    a = ap.parse_args()
    from vt.client import chat
    items = json.load(open(a.items))
    if a.limit:
        seen = Counter()
        items = [x for x in items if (seen.update([x["kind"]]) or seen[x["kind"]] <= a.limit)]
    todo, rows, lock = list(enumerate(items)), [], threading.Lock()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i, x = todo.pop(0)
            t = time.time()
            try:
                r = chat(a.url, [{"role": "user", "content": x["q"]}], max_new_tokens=384)
                row = {"i": i, "kind": x["kind"], "expect": x["expect"], "q": x["q"][:200], "route": r.get("route"),
                       "grounded": r.get("grounded", False), "assessment": r.get("assessment"),
                       "math_check": r.get("math_check"), "text": r.get("text", "")[:1200],
                       "attempts": r.get("_attempts"), "seconds": round(time.time() - t, 1)}
                if "aliases" in x:
                    row["right"] = is_right(row["text"], x["aliases"])
                if "answer" in x:
                    n = last_number(row["text"])
                    row["right"] = n is not None and abs(n - x["answer"]) < 1e-6 * max(1, abs(x["answer"])) + 1e-6
            except Exception as e:                              # lost after the retry, or a server error
                row = {"i": i, "kind": x["kind"], "expect": x["expect"], "q": x["q"][:200], "route": None,
                       "error": repr(e)[:200], "seconds": round(time.time() - t, 1), "text": ""}
            with lock:
                rows.append(row)
                print(json.dumps({k: row.get(k) for k in ("i", "kind", "route", "seconds", "error")}), flush=True)

    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    res = score(rows)
    json.dump(dict(res, url=a.url, rows=sorted(rows, key=lambda r: r["i"])), open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)
    if res["errors"] > len(rows) / 2:
        raise SystemExit(f"{res['errors']} of {len(rows)} requests failed")


if __name__ == "__main__":
    main()
