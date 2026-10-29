"""Evidence-based verification against the internal probe, on the same labeled replies of the 32B
(results/kaggle/chateval-32h/chateval/traces.json: 200 factual replies with right/wrong labels, 56 everyday).

For each check-worthy sentence of a reply (one that names something or gives a number): look it up on
Wikipedia (the question plus the sentence), then let an NLI model judge each passage against the sentence.
A sentence's support is its best entailment, its conflict its strongest contradiction. Compared with the
probe's reply_confidence by AUROC (wrong vs right), on TriviaQA and NQ apart, and the share of everyday
replies that would be marked.

    python experiments/evidence_verify.py --nli cross-encoder/nli-deberta-v3-base --out evidence_verify.json
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vt.retrieval import Wikipedia  # noqa: E402

TRACES = "results/kaggle/chateval-32h/chateval/traces.json"


def sentences(text):
    text = re.sub(r"[*_#`>]+", "", text)                       # markdown
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -•\t") for p in parts if len(p.strip()) > 3]


def check_worthy(s):
    """Names something or gives a number: a word capitalized mid-sentence, or a digit."""
    w = s.split()
    return any(ch.isdigit() for ch in s) or any(t[:1].isupper() for t in w[1:])


class Judge:
    """A chat model as the verifier: given the evidence, the question and a claim, P("Yes") against P("No")
    for "does the evidence support the claim?" (one forward pass, no generation)."""

    def __init__(self, name, dtype="float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(os.cpu_count() or 4)
        self.torch, self.tok = torch, AutoTokenizer.from_pretrained(name)
        # float32 on CPU: bfloat16 matmuls are many times slower on CPUs without native support
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=getattr(torch, dtype)).eval()
        self.yes = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes")]
        self.no = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No")]

    def __call__(self, question, claim, passages):
        ev = "\n\n".join(p[:900] for p in passages)
        msg = [{"role": "system", "content": "You check claims against evidence. Answer only Yes or No."},
               {"role": "user", "content": f"Evidence:\n{ev}\n\nQuestion: {question}\nClaim: {claim}\n\n"
                                           "Does the evidence support the claim? Answer Yes or No."}]
        ids = self.tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
        with self.torch.no_grad():
            z = self.model(ids[:, -1500:]).logits[0, -1].float()
        y, n = self.torch.logsumexp(z[self.yes], 0), self.torch.logsumexp(z[self.no], 0)
        return float(self.torch.sigmoid(y - n))


def auroc(scores, y):
    s, y = np.asarray(scores, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return round(float(gt), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=TRACES)
    ap.add_argument("--nli", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--max-sentences", type=int, default=3)
    ap.add_argument("--out", default="evidence_verify.json")
    ap.add_argument("--limit", type=int, help="only the first n factual and n everyday replies (smoke test)")
    ap.add_argument("--rerank", help="cross-encoder for lookups (5 pages) and for keeping only relevant passages")
    ap.add_argument("--relevant", type=float, default=0.0, help="reranker logit a passage needs to count as evidence")
    ap.add_argument("--pause", type=float, default=0.2, help="seconds between replies (Wikipedia rate limits)")
    ap.add_argument("--judge", help="a chat model that reads the evidence and says whether it supports the claim, "
                                    "e.g. Qwen/Qwen2.5-3B-Instruct (P(Yes) is the support)")
    a = ap.parse_args()
    judge = Judge(a.judge) if a.judge else None
    from sentence_transformers import CrossEncoder
    nli = CrossEncoder(a.nli, max_length=512)
    labels = {v.lower(): int(k) for k, v in nli.model.config.id2label.items()}
    ent, con = labels["entailment"], labels["contradiction"]
    rel = None
    if a.rerank:
        rel = CrossEncoder(a.rerank, max_length=256)
        wiki = Wikipedia(k=3, full_text=True, pages=5,
                         rerank=lambda q, ps: rel.predict([(q, p) for p in ps], batch_size=32, show_progress_bar=False))
    else:
        wiki = Wikipedia(k=3, full_text=True)
    T = json.load(open(a.traces))
    if a.limit:
        T = [x for x in T if x["kind"] == "factual"][:a.limit] + [x for x in T if x["kind"] == "everyday"][:a.limit]
    rows = []
    for n, x in enumerate(T):
        sents = [s for s in sentences(x["text"]) if check_worthy(s)][:a.max_sentences]
        judged = []
        for s in sents:
            ps = wiki(f"{x['prompt']} {s}")
            if ps and rel is not None:                       # evidence only where it is about the claim
                r = rel.predict([(s, p.split(": ", 1)[-1]) for p in ps], show_progress_bar=False)
                ps = [p for p, v in zip(ps, r) if float(v) > a.relevant]
            if not ps:
                judged.append({"s": s, "support": None, "conflict": None, "passages": 0})
                continue
            logits = np.asarray(nli.predict([(p.split(": ", 1)[-1], s) for p in ps]))
            pr = np.exp(logits - logits.max(-1, keepdims=True))
            pr /= pr.sum(-1, keepdims=True)
            j = {"s": s[:160], "support": round(float(pr[:, ent].max()), 4),
                 "conflict": round(float(pr[:, con].max()), 4), "passages": len(ps),
                 "evidence": [p[:700] for p in ps]}
            if judge is not None:
                j["judge"] = round(judge(x["prompt"], s, ps), 4)
            judged.append(j)
        rows.append({"kind": x["kind"], "source": x.get("source", x.get("cat")), "wrong": x.get("wrong"),
                     "reply_confidence": x.get("reply_confidence"), "sentences": judged})
        if n % 20 == 0:
            print(json.dumps({"done": n + 1, "errors": dict(wiki.errors)}), flush=True)
            json.dump({"partial": True, "rows": rows}, open(a.out, "w"))       # survives a timeout
        time.sleep(a.pause)

    def first(r, key, default):
        v = [j[key] for j in r["sentences"] if j[key] is not None]
        return v[0] if v else default

    def worst(r, key, default, f):
        v = [j[key] for j in r["sentences"] if j[key] is not None]
        return f(v) if v else default
    signals = {                                                       # higher = more likely wrong
        "probe (1 - reply_confidence)": lambda r: 1 - (r["reply_confidence"] if r["reply_confidence"] is not None else 1),
        "evidence: 1 - support, first claim": lambda r: 1 - first(r, "support", 0.5),
        "evidence: 1 - support, weakest claim": lambda r: 1 - worst(r, "support", 0.5, min),
        "evidence: conflict, first claim": lambda r: first(r, "conflict", 0.0),
        "evidence: conflict - support, first claim": lambda r: first(r, "conflict", 0.0) - first(r, "support", 0.5),
    }
    if judge is not None:
        signals["judge: 1 - P(supported), first claim"] = lambda r: 1 - first(r, "judge", 0.5)
        signals["judge: 1 - P(supported), weakest claim"] = lambda r: 1 - worst(r, "judge", 0.5, min)
    fa = [r for r in rows if r["kind"] == "factual"]
    res = {"replies": len(rows), "nli": a.nli, "lookup_errors": dict(wiki.errors), "auroc": {}}
    for name, f in signals.items():
        res["auroc"][name] = {src: auroc([f(r) for r in fa if r["source"] == src],
                                         [r["wrong"] for r in fa if r["source"] == src]) for src in ("triviaqa", "nq")}
        res["auroc"][name]["all"] = auroc([f(r) for r in fa], [r["wrong"] for r in fa])
    z = lambda v: (np.asarray(v) - np.mean(v)) / (np.std(v) + 1e-9)  # noqa: E731
    combo = z([signals["probe (1 - reply_confidence)"](r) for r in fa]) + \
        z([signals["evidence: conflict - support, first claim"](r) for r in fa])
    res["auroc"]["probe + evidence (z-sum)"] = {"all": auroc(combo, [r["wrong"] for r in fa])}
    for src in ("triviaqa", "nq"):
        idx = [i for i, r in enumerate(fa) if r["source"] == src]
        res["auroc"]["probe + evidence (z-sum)"][src] = auroc(combo[idx], [fa[i]["wrong"] for i in idx])
    ev = [r for r in rows if r["kind"] == "everyday"]
    res["everyday"] = {"with_check_worthy_claims": round(np.mean([bool(r["sentences"]) for r in ev]), 3),
                       "any_claim_contradicted_gt_0.5": round(np.mean([any((j["conflict"] or 0) > 0.5 for j in r["sentences"]) for r in ev]), 3),
                       "all_claims_supported_gt_0.5": round(np.mean([bool(r["sentences"]) and all((j["support"] or 0) > 0.5 for j in r["sentences"]) for r in ev]), 3)}
    if judge is not None:
        jz = z([signals["probe (1 - reply_confidence)"](r) for r in fa]) + \
            z([signals["judge: 1 - P(supported), first claim"](r) for r in fa])
        res["auroc"]["probe + judge (z-sum)"] = {"all": auroc(jz, [r["wrong"] for r in fa])}
        for src in ("triviaqa", "nq"):
            idx = [i for i, r in enumerate(fa) if r["source"] == src]
            res["auroc"]["probe + judge (z-sum)"][src] = auroc(jz[idx], [fa[i]["wrong"] for i in idx])
        res["everyday"]["any_claim_judged_unsupported_lt_0.2"] = round(np.mean(
            [any(j.get("judge") is not None and j["judge"] < 0.2 for j in r["sentences"]) for r in ev]), 3)
        res["factual_any_judged_unsupported_lt_0.2"] = {w: round(np.mean(
            [any(j.get("judge") is not None and j["judge"] < 0.2 for j in r["sentences"]) for r in fa if r["wrong"] == w]), 3)
            for w in (0, 1)}
        res["judge"] = a.judge
    res["factual_any_contradicted"] = {w: round(np.mean([any((j["conflict"] or 0) > 0.5 for j in r["sentences"]) for r in fa if r["wrong"] == w]), 3) for w in (0, 1)}
    res["rows"] = rows
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
