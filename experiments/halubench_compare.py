"""Head to head on HaluBench (PatronusAI): the grounding check (vt/grounding.py: a probe on a reader model's
neurons plus the evidence shift and disagreement, trained without human labels) against MiniCheck
(lytang/MiniCheck-RoBERTa-Large, a fact-checker trained on synthetic data), on the same answers: 50 per source
(DROP, CovidQA, PubMedQA, FinanceBench, RAGTruth, HaluEval), half hallucinated. Both see the passage as the
evidence and "question answer" as the claim; MiniCheck reads a long passage in chunks and keeps the
best-supported one, as it is meant to be used.

AUROC for "hallucinated" overall and per source, the share of hallucinated answers caught at a 10% false-alarm
rate, and seconds per answer (CPU).

    python experiments/halubench_compare.py --reader Qwen/Qwen3-4B --probe heads/grounding_probe_qwen3_4b.npz \
        --per-source 50 --out results/halubench_compare.json
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def sample(per_source, seed=0):
    from datasets import load_dataset
    rows = list(load_dataset("PatronusAI/HaluBench", split="test"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    out, seen = [], {}
    for r in rows:
        key = (r["source_ds"], r["label"])
        if seen.get(key, 0) < per_source // 2:
            seen[key] = seen.get(key, 0) + 1
            out.append(r)
    return out


def recall_at_fpr(score, y, fpr=0.1):
    neg = np.sort(score[y == 0])
    t = neg[int(np.ceil((1 - fpr) * len(neg))) - 1] if len(neg) else np.inf
    return float((score[y == 1] > t).mean())


class MiniCheck:
    def __init__(self, name="lytang/MiniCheck-RoBERTa-Large", chunk_words=300):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.tok = torch, AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.chunk_words = chunk_words

    def p_unsupported(self, passage, claim):
        w = passage.split()
        chunks = [" ".join(w[i:i + self.chunk_words]) for i in range(0, max(1, len(w)), self.chunk_words)] or [""]
        x = self.tok(chunks, [claim] * len(chunks), return_tensors="pt", padding=True, truncation="only_first",
                     max_length=512)
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(**x).logits, -1)[:, 1]
        return float(1 - p.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3-4B")
    ap.add_argument("--probe", default="heads/grounding_probe_qwen3_4b.npz")
    ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--out", default="halubench_compare.json")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.correctness import auroc
    from vt.grounding import Grounding
    rows = sample(a.per_source)
    tok = AutoTokenizer.from_pretrained(a.reader)
    reader = AutoModelForCausalLM.from_pretrained(a.reader, torch_dtype=torch.bfloat16).eval()
    g = Grounding(reader, tok, a.probe, max_context_chars=12000)
    mc = MiniCheck()
    out, t_g, t_m = [], 0.0, 0.0
    for i, r in enumerate(rows):
        claim = f"{r['question']} {r['answer']}"
        t = time.time()
        res = g.check([{"role": "user", "content": r["question"]}], str(r["answer"]), documents=[r["passage"]])
        t_g += time.time() - t
        zs = [s["z"] for s in res["sentences"]] or [0.0]
        t = time.time()
        pm = mc.p_unsupported(r["passage"], claim)
        t_m += time.time() - t
        out.append({"source": r["source_ds"], "y": int(r["label"] == "FAIL"), "grounding_total": res["total"],
                    "grounding_max": max(zs), "grounding_verdict": res["verdict"], "minicheck": pm})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} grounding {t_g / (i + 1):.1f}s minicheck {t_m / (i + 1):.2f}s", flush=True)
    y = np.array([o["y"] for o in out])
    res = {"n": len(out), "reader": a.reader, "seconds_per_answer": {"grounding": round(t_g / len(out), 2),
                                                                    "minicheck": round(t_m / len(out), 2)}}
    for name in ("grounding_total", "grounding_max", "minicheck"):
        s = np.array([o[name] for o in out], float)
        res[name] = {"auroc": round(float(auroc(s, y.astype(float))), 3),
                     "caught_at_10pct_false_alarms": round(recall_at_fpr(s, y), 3),
                     "per_source": {src: round(float(auroc(s[m], y[m].astype(float))), 3)
                                    for src in sorted({o["source"] for o in out})
                                    for m in [np.array([o["source"] == src for o in out])]}}
    print(json.dumps(res, indent=1))
    json.dump(dict(res, rows=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
