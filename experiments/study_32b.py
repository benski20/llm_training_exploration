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
    if len(y) < 2:
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(auroc(a[i], y[i]) - auroc(b[i], y[i]))
    if not d:
        return None
    d = np.array([x for x in d if x is not None])
    if not len(d) or auroc(a, y) is None:
        return None                                            # one class only: nothing to compare
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
    ap.add_argument("--rows", help="the answers of an earlier run (its JSON): skip answering")
    ap.add_argument("--gather-only", action="store_true", help="with --rows: only look the claims up (no model, CPU)")
    ap.add_argument("--judge-gathered", action="store_true", help="with --rows from --gather-only: only judge")
    a = ap.parse_args()
    if a.rows and a.gather_only:
        return gather_rows(a)
    if a.rows and a.judge_gathered:
        return judge_rows(a)
    from vt.claim_verify import ClaimVerifier
    from vt.verichat import VeriChat, apply_sampling
    qs = json.load(open(a.questions))
    if a.limit:
        qs = [q for q in qs if q["source"] == "triviaqa"][:a.limit] + [q for q in qs if q["source"] == "nq"][:a.limit]
    bot = VeriChat.load(a.model, a.head)
    apply_sampling(bot, 0.7, 0.8, 20, 1.5)                     # the demo's decoding
    retriever = make_retriever(a)
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


def make_retriever(a):
    from vt.retrieval import Corpus, Wikipedia
    if a.corpus:
        return Corpus.from_dir(a.corpus)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(a.rerank, max_length=256)
    return Wikipedia(full_text=True, pages=5,
                     rerank=lambda q, ps: ce.predict([(q, p) for p in ps], batch_size=32, show_progress_bar=False))


def gather_rows(a):
    """The lookups for saved answers, on CPU: every check-worthy claim's evidence, saved for judging."""
    from vt.claim_verify import gather
    rows = json.load(open(a.rows))["rows"]
    retriever, t = make_retriever(a), time.time()
    for k in range(0, len(rows), 16):
        for r, g in zip(rows[k:k + 16], gather([(r["q"], r["text"]) for r in rows[k:k + 16]], retriever)):
            r["gathered"] = g
        print(json.dumps({"gathered": min(k + 16, len(rows)), "seconds": round(time.time() - t),
                          "lookup_errors": dict(getattr(retriever, "errors", {}) or {})}), flush=True)
    json.dump({"rows": rows, "lookup_errors": dict(getattr(retriever, "errors", {}) or {})}, open(a.out, "w"))


GRADER = "You grade answers to quiz questions. Answer only Yes or No."


def judge_rows(a):
    """The judging of gathered evidence with the served model (vt.claim_verify, as served), yes/no and
    three-way on the same evidence, and each answer graded against its reference answers by the same model
    (a fairer label than string matching: "red, green and blue" is right for "red blue and green")."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vt.claim_verify import ClaimVerifier
    from vt.verichat import gpu_dtype
    d = json.load(open(a.rows))
    rows, cuda = d["rows"], torch.cuda.is_available()
    aliases = {q["q"]: q["aliases"] for q in json.load(open(a.questions))}
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=gpu_dtype() if cuda else torch.float32,
                                                 attn_implementation="sdpa", device_map="auto" if cuda else None).eval()
    v2, v3 = ClaimVerifier(model, tok, None, batch=16), ClaimVerifier(model, tok, None, batch=16, three_way=True)
    t = time.time()
    for k in range(0, len(rows), 64):
        chunk = rows[k:k + 64]
        g = [r["gathered"] for r in chunk]
        for r, ver2, ver3 in zip(chunk, v2.judge_gathered(g), v3.judge_gathered(g)):
            r["verification"] = [{kk: c[kk] for kk in ("verdict", "p_supported", "sources")} for c in ver2]
            r["verification3"] = [{kk: c.get(kk) for kk in ("verdict", "p_supported", "p_contradicted", "p_not_stated")}
                                  for c in ver3]
        grades = v2.ask_yes_no(GRADER, [f"Question: {r['q']}\nReference answers: {'; '.join(aliases.get(r['q'], [])[:8])}\n"
                                        f"Answer given: {r['text'][:600]}\n\nIs the answer given correct, matching a "
                                        "reference answer in meaning? Answer Yes or No." for r in chunk])
        for r, p in zip(chunk, grades):
            r["p_right_llm"], r["right_llm"] = round(p, 4), p >= 0.5
        print(json.dumps({"judged": min(k + 64, len(rows)), "seconds": round(time.time() - t)}), flush=True)
    res = {"string_labels": summarize(rows), "llm_labels": summarize(rows, label="right_llm"),
           "three_way": {lab: evaluate(rows, lab) for lab in ("right", "right_llm")},
           "label_agreement": round(float(np.mean([r["right"] == r["right_llm"] for r in rows])), 3)}
    res.update(model=a.model, lookup_errors=d.get("lookup_errors"), seconds={"judging": round(time.time() - t)})
    json.dump(dict(res, rows=rows), open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def evaluate(rows, label):
    """Yes/no against three-way judging, each alone and mixed with the probe as the demo does."""
    def first(r, key):
        v = [c for c in r.get(key) or [] if c.get("p_supported") is not None]
        return v[0] if v else None

    def probe(r):
        return 1 - r["probe"] if r["probe"] is not None else 0.5

    def two(r):
        c = first(r, "verification")
        return 0.9 * (1 - c["p_supported"]) + 0.1 * probe(r) if c else probe(r)

    def three(r):
        c = first(r, "verification3")
        if not c:
            return probe(r)
        w = 0.9 * (1 - c["p_not_stated"])
        return w * c["p_contradicted"] / max(c["p_supported"] + c["p_contradicted"], 1e-6) + (1 - w) * probe(r)
    out = {}
    for src in ("triviaqa", "nq", "all"):
        sub = [r for r in rows if src == "all" or r["source"] == src]
        if not sub:
            continue
        y = [not r[label] for r in sub]
        out[src] = {"probe": auroc([probe(r) for r in sub], y), "two_way_mixed": auroc([two(r) for r in sub], y),
                    "three_way_mixed": auroc([three(r) for r in sub], y),
                    "three_minus_two": paired([three(r) for r in sub], [two(r) for r in sub], y),
                    "not_stated_share": round(float(np.mean([(first(r, "verification3") or {}).get("p_not_stated", 1) >= 0.5
                                                             for r in sub])), 3)}
    return out


def summarize(rows, label="right"):
    def judge(r):                                              # the first judged claim's doubt; None: no evidence
        v = [c["p_supported"] for c in r.get("verification") or [] if c["p_supported"] is not None]
        return 1 - v[0] if v else None
    out = {"n": len(rows), "label": label, "accuracy": {}, "auroc": {}, "paired_judge_minus_probe": {}, "combined": {}}
    for src in ("triviaqa", "nq", "all"):
        sub = [r for r in rows if src == "all" or r["source"] == src]
        if not sub:
            continue
        y = [not r[label] for r in sub]
        probe = [1 - r["probe"] if r["probe"] is not None else 0.5 for r in sub]
        out["accuracy"][src] = round(float(np.mean([r[label] for r in sub])), 3)
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
