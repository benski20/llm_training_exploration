"""How often does the live demo's Wikipedia lookup fix a doubted answer? Sends TriviaQA questions to a
running VeriChat server (as a user would, default settings) and, for every reply that was looked up,
compares the closed-book answer with the one written from Wikipedia.

    python experiments/correction_rate.py --url https://<...>.modal.run --n 100 --workers 4 --out correction.json

Needs only `datasets` (and numpy) on the client. Deploy without the rate limit for the run.
"""
import argparse
import json
import random
import re
import string
import threading
import time
import urllib.error
import urllib.request


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_right(text, aliases):
    t = f" {norm(text)} "
    return any(f" {norm(a)} " in t for a in aliases if norm(a))


def client_chat(url, messages):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from vt.client import chat
    return chat(url, messages)


def verification_summary(ok):
    """How the demo's source checks of each final answer line up with whether it was right."""
    v = [r for r in ok if r.get("verdicts") is not None]
    if not v:
        return None

    def first(r):
        return r["verdicts"][0] if r["verdicts"] else "no claims"
    out = {"replies_checked": len(v), "by_answer": {}}
    for name, grp in (("right", [r for r in v if r["final_right"]]), ("wrong", [r for r in v if not r["final_right"]])):
        counts = {}
        for r in grp:
            counts[first(r)] = counts.get(first(r), 0) + 1
        out["by_answer"][name] = {"n": len(grp), "first_claim_verdicts": counts}
    scored = [(1 - r["p_supported"][0], not r["final_right"]) for r in v if r["p_supported"] and r["p_supported"][0] is not None]
    pos, neg = [s for s, w in scored if w], [s for s, w in scored if not w]
    if pos and neg:
        out["auroc_first_claim"] = round(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 3)
    out["judged"] = len(scored)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="correction.json")
    ap.add_argument("--retry", action="store_true", help="send through vt.client (streamed; a request silent for "
                                                         "45 s is sent again once), as the page does")
    a = ap.parse_args()
    from datasets import load_dataset
    d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    qs = [(x["question"], x["answer"]["normalized_aliases"] + [x["answer"]["value"]])
          for x in d.select(random.Random(a.seed).sample(range(len(d)), a.n))]
    url, rows, lock = a.url.rstrip("/") + "/chat", [], threading.Lock()
    todo = list(enumerate(qs))

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i, (q, al) = todo.pop(0)
            body = json.dumps({"messages": [{"role": "user", "content": q}], "mode": "hedge"}).encode()
            t = time.time()
            try:
                if a.retry:
                    r = client_chat(a.url, [{"role": "user", "content": q}])
                else:
                    r = json.loads(urllib.request.urlopen(urllib.request.Request(
                        url, body, {"Content-Type": "application/json"}), timeout=400).read())
            except Exception as e:                                # lost, timed out, or a server error
                r = {"error": repr(e)[:200]}
            ret = r.get("retrieval") or {}
            row = {"i": i, "q": q, "seconds": round(time.time() - t, 1), "error": r.get("error"),
                   "attempts": r.get("_attempts"),
                   "looked_up": bool(ret.get("retrieved")), "trigger_score": ret.get("trigger_score"),
                   "final_right": is_right(r.get("text", ""), al) if "text" in r else None}
            ver = r.get("verification")
            if ver is not None:                                   # the demo's source checks of this reply
                row["verdicts"] = [v.get("verdict") for v in ver]
                row["p_supported"] = [v.get("p_supported") for v in ver]
            if ret.get("retrieved"):
                row["closed_book_right"] = is_right(ret.get("closed_book_text", ""), al)
                row["passages"] = [p[:80] for p in ret.get("passages", [])]
            with lock:
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("i", "seconds", "looked_up", "final_right")}), flush=True)

    ts = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ok = [r for r in rows if r["final_right"] is not None]
    lk = [r for r in ok if r["looked_up"]]
    closed = [r.get("closed_book_right", r["final_right"]) for r in ok]
    res = {"url": a.url, "n": len(rows), "errors": len(rows) - len(ok), "looked_up": len(lk),
           "retried_and_answered": sum((r.get("attempts") or 1) > 1 for r in ok),
           "looked_up_share": round(len(lk) / max(1, len(ok)), 3),
           "accuracy_closed_book": round(sum(closed) / max(1, len(ok)), 3),
           "accuracy_served": round(sum(r["final_right"] for r in ok) / max(1, len(ok)), 3),
           "among_looked_up": {"wrong_to_right": sum(not r["closed_book_right"] and r["final_right"] for r in lk),
                               "right_to_wrong": sum(r["closed_book_right"] and not r["final_right"] for r in lk),
                               "stayed_right": sum(r["closed_book_right"] and r["final_right"] for r in lk),
                               "stayed_wrong": sum(not r["closed_book_right"] and not r["final_right"] for r in lk)},
           "seconds_median": sorted(r["seconds"] for r in rows)[len(rows) // 2] if rows else None,
           "verification": verification_summary(ok),
           "rows": sorted(rows, key=lambda r: r["i"])}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


if __name__ == "__main__":
    main()
