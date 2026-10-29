"""Does a Wikipedia lookup find the answer? Answer recall of the passages the demo would put in the prompt:
lead sections of the top hits (what the demo uses) against the best-matching passages of the top pages'
full text. No model and no GPU: TriviaQA questions, live Wikipedia, string match on the answer aliases.

    python experiments/retrieval_recall.py --n 200 --out retrieval_recall.json

Runs from any machine Wikipedia does not rate-limit (the GitHub workflow retrieval-eval.yml).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vt.retrieval import Wikipedia  # noqa: E402

from correction_rate import is_right  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=31)               # the correction-rate runs' sample first
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pause", type=float, default=0.3)
    ap.add_argument("--out", default="retrieval_recall.json")
    a = ap.parse_args()
    from datasets import load_dataset
    d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    qs = [(x["question"], x["answer"]["normalized_aliases"] + [x["answer"]["value"]])
          for x in d.select(random.Random(a.seed).sample(range(len(d)), a.n))]
    systems = {"lead": Wikipedia(k=a.k), "full_text": Wikipedia(k=a.k, full_text=True)}
    rows, failed = [], {s: 0 for s in systems}
    for i, (q, al) in enumerate(qs):
        row = {"i": i, "q": q}
        for name, w in systems.items():
            t = time.time()
            ps = w(q)
            failed[name] += not ps
            row[name] = {"found": is_right(" ".join(ps), al), "n": len(ps), "words": sum(len(p.split()) for p in ps),
                         "seconds": round(time.time() - t, 2), "titles": [p.split(" (Wikipedia)")[0] for p in ps]}
        rows.append(row)
        print(json.dumps({"i": i, **{s: row[s]["found"] for s in systems},
                          "errors": {s: dict(w.errors) for s, w in systems.items()}}), flush=True)
        time.sleep(a.pause)
    res = {"n": len(rows), "k": a.k}
    for s in systems:
        r = [row[s] for row in rows]
        res[s] = {"recall": round(sum(x["found"] for x in r) / len(r), 3), "no_passages": failed[s],
                  "words_mean": round(sum(x["words"] for x in r) / len(r), 1),
                  "seconds_median": sorted(x["seconds"] for x in r)[len(r) // 2]}
    res["errors"] = {s: dict(w.errors) for s, w in systems.items()}
    res["only_full_text_found"] = sum(r["full_text"]["found"] and not r["lead"]["found"] for r in rows)
    res["only_lead_found"] = sum(r["lead"]["found"] and not r["full_text"]["found"] for r in rows)
    res["rows"] = rows
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))


if __name__ == "__main__":
    main()
