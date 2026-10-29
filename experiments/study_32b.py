"""Judge vs probe, at a scale that can tell them apart: the served model answers 400 TriviaQA and 400 NQ
questions closed-book (the serving code and settings), its internal probe scores each answer (the weakest
claim, as the demo's lookup trigger), then each answer's check-worthy claims are looked up on Wikipedia
(full text, reranked) and judged by the same model against the evidence (vt.claim_verify, as served).
Reports AUROC (wrong vs right) of each, the paired bootstrap of their difference, and their combination
fitted on one half and tested on the other.

    python experiments/study_32b.py --model unsloth/Qwen3-32B-bnb-4bit --head <risk head> --out study.json
    python experiments/study_32b.py --model Qwen/Qwen2.5-0.5B-Instruct --head <head> --corpus docs/ --limit 6   # smoke
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vt.correctness import is_right  # noqa: E402


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return None
    return float((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())


def paired(a, b, y, n=2000, seed=0):
    """AUROC(a) - AUROC(b) with a 95% bootstrap interval and P(difference > 0)."""
    rng, y = np.random.default_rng(seed), np.asarray(y)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(auroc(a[i], y[i]) - auroc(b[i], y[i]))
    d = np.array(d)
    return {"diff": round(auroc(a, y) - auroc(b, y), 4), "ci95": [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])],
            "p_better": round(float((d > 0).mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--questions", default="results/study_questions.json")
    ap.add_argument("--limit", type=int, help="first n questions of each source (smoke test)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rerank", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--corpus", help="look up in a local folder instead of Wikipedia (smoke test)")
    ap.add_argument("--out", default="study.json")
    a = ap.parse_args()
    from vt.claim_verify import ClaimVerifier
    from vt.retrieval import Corpus, Wikipedia
    from vt.verichat import VeriChat, apply_sampling
    qs = json.load(open(a.questions))
    if a.limit:
        qs = [q for q in qs if q["source"] == "triviaqa"][:a.limit] + [q for q in qs if q["source"] == "nq"][:a.limit]
    bot = VeriChat.load(a.model, a.head)
    apply_sampling(bot, 0.7, 0.8, 20, 1.5)                     # the demo's decoding
    if a.corpus:
        retriever = Corpus.from_dir(a.corpus)
    else:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(a.rerank, max_length=256)
        retriever = Wikipedia(full_text=True, pages=5,
                              rerank=lambda q, ps: ce.predict([(q, p) for p in ps], batch_size=32, show_progress_bar=False))
    verifier = ClaimVerifier(bot.model, bot.tok, retriever, batch=16)
    rows, t0 = [], time.time()
    for k in range(0, len(qs), a.batch):                      # closed-book answers, as served (no lookup)
        chunk = qs[k:k + a.batch]
        outs = bot.reply_batch([{"messages": [{"role": "user", "content": q["q"]}], "mode": "hedge",
                                 "retrieve": False, "max_new_tokens": 96} for q in chunk])
        for q, o in zip(chunk, outs):
            text = o["text"].replace("[unsure:]", "").strip()
            score, claim = VeriChat.doubt_score(o)
            rows.append({"source": q["source"], "q": q["q"], "text": text[:400], "right": is_right(text, q["aliases"]),
                         "probe": score, "reply_confidence": o.get("reply_confidence")})
        print(json.dumps({"answered": len(rows), "seconds": round(time.time() - t0)}), flush=True)
    t1 = time.time()
    for k in range(0, len(rows), 32):                          # look up and judge every answer's claims
        vs = verifier.verify_batch([(r["q"], r["text"]) for r in rows[k:k + 32]])
        for r, v in zip(rows[k:k + 32], vs):
            r["verification"] = [{kk: c[kk] for kk in ("verdict", "p_supported", "sources")} for c in v]
        print(json.dumps({"verified": min(k + 32, len(rows)), "seconds": round(time.time() - t1),
                          "lookup_errors": dict(getattr(retriever, "errors", {}) or {})}), flush=True)
    res = summarize(rows)
    res.update(model=a.model, seconds={"answers": round(t1 - t0), "verification": round(time.time() - t1)})
    json.dump(dict(res, rows=rows), open(a.out, "w"), indent=1)
    print("ROWS " + json.dumps(rows), flush=True)
    print(json.dumps(res, indent=1), flush=True)


def summarize(rows):
    def judge(r):                                              # the first judged claim's doubt; None: no evidence
        v = [c["p_supported"] for c in r.get("verification") or [] if c["p_supported"] is not None]
        return 1 - v[0] if v else None
    out = {"n": len(rows), "accuracy": {}, "auroc": {}, "paired_judge_minus_probe": {}, "combined": {}}
    for src in ("triviaqa", "nq", "all"):
        sub = [r for r in rows if src == "all" or r["source"] == src]
        if not sub:
            continue
        y = [not r["right"] for r in sub]
        probe = [1 - r["probe"] if r["probe"] is not None else 0.5 for r in sub]
        out["accuracy"][src] = round(float(np.mean([r["right"] for r in sub])), 3)
        ev = [i for i, r in enumerate(sub) if judge(r) is not None]
        out["auroc"][src] = {"probe": auroc(probe, y),
                             "judge_where_evidence": auroc([judge(sub[i]) for i in ev], [y[i] for i in ev]),
                             "probe_where_evidence": auroc([probe[i] for i in ev], [y[i] for i in ev]),
                             "evidence_share": round(len(ev) / len(sub), 3)}
        if len(ev) > 20:
            out["paired_judge_minus_probe"][src] = paired([judge(sub[i]) for i in ev], [probe[i] for i in ev],
                                                          [y[i] for i in ev])
        # the served combination, fitted on even rows and tested on odd: judge where evidence, else probe,
        # averaged with the probe with weight w
        best = None
        for w in np.linspace(0, 1, 11):
            s = [w * judge(r) + (1 - w) * p if judge(r) is not None else p for r, p in zip(sub, probe)]
            fit = auroc(s[0::2], y[0::2])
            if fit is not None and (best is None or fit > best[1]):
                best = (w, fit)
        if best:
            w = best[0]
            s = [w * judge(r) + (1 - w) * p if judge(r) is not None else p for r, p in zip(sub, probe)]
            out["combined"][src] = {"judge_weight": round(float(w), 2), "test_auroc": auroc(s[1::2], y[1::2]),
                                    "probe_test_auroc": auroc(probe[1::2], y[1::2])}
    return out


if __name__ == "__main__":
    main()
