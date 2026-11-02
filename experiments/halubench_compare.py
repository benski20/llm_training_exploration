"""Head to head on HaluBench (PatronusAI): the grounding check (vt/grounding.py: a probe on a reader model's
neurons plus the evidence shift and disagreement, trained without human labels) against MiniCheck
(lytang/MiniCheck-RoBERTa-Large, a fact-checker trained on synthetic data), on the same answers: 50 per source
(DROP, CovidQA, PubMedQA, FinanceBench, RAGTruth, HaluEval), half hallucinated. Both see the passage as the
evidence and "question answer" as the claim; MiniCheck reads a long passage in chunks and keeps the
best-supported one, as it is meant to be used.

--add-hhem adds Vectara's HHEM-2.1-open on the same answers afterwards. AUROC for "hallucinated" overall and per
source, the share of hallucinated answers caught at a 10% false-alarm
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


def summarize(out, names):
    from vt.correctness import auroc
    y = np.array([o["y"] for o in out])
    res = {}
    for name in names:
        s = np.array([o[name] for o in out], float)
        res[name] = {"auroc": round(float(auroc(s, y.astype(float))), 3),
                     "caught_at_10pct_false_alarms": round(recall_at_fpr(s, y), 3),
                     "per_source": {src: round(float(auroc(s[m], y[m].astype(float))), 3)
                                    for src in sorted({o["source"] for o in out})
                                    for m in [np.array([o["source"] == src for o in out])]}}
    return res


def add_hhem(path, per_source):
    """Vectara's HHEM-2.1-open on the answers of an earlier run (same sample), merged into its results."""
    from transformers import AutoModelForSequenceClassification
    m = AutoModelForSequenceClassification.from_pretrained("vectara/hallucination_evaluation_model",
                                                           trust_remote_code=True)    # its own model code
    res = json.load(open(path))
    rows, t = sample(per_source), time.time()
    assert [r["source_ds"] for r in rows] == [o["source"] for o in res["rows"]], "not the same sample"
    for r, o in zip(rows, res["rows"]):
        o["hhem"] = float(1 - m.predict([(r["passage"], f"{r['question']} {r['answer']}")])[0])
    res["seconds_per_answer"]["hhem"] = round((time.time() - t) / len(rows), 2)
    res.update(summarize(res["rows"], ["hhem"]))
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3-4B")
    ap.add_argument("--probe", default="heads/grounding_probe_qwen3_4b.npz")
    ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--out", default="halubench_compare.json")
    ap.add_argument("--claim", choices=["answer", "statement"], default="answer",
                    help="what the grounding check reads: the answer as given, or \"question answer\" (as MiniCheck)")
    ap.add_argument("--sources", nargs="+", help="only these HaluBench sources")
    ap.add_argument("--add-hhem", metavar="RESULTS", help="score HHEM-2.1-open (Vectara; runs its own model code) on "
                    "the answers of an earlier run and merge it into that results file")
    a = ap.parse_args()
    if a.add_hhem:
        return add_hhem(a.add_hhem, a.per_source)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.grounding import Grounding
    rows = [r for r in sample(a.per_source) if not a.sources or r["source_ds"] in a.sources]
    tok = AutoTokenizer.from_pretrained(a.reader)
    reader = AutoModelForCausalLM.from_pretrained(a.reader, torch_dtype=torch.bfloat16).eval()
    g = Grounding(reader, tok, a.probe, max_context_chars=12000)
    mc = MiniCheck()
    out, t_g, t_m = [], 0.0, 0.0
    for i, r in enumerate(rows):
        claim = f"{r['question']} {r['answer']}"
        t = time.time()
        reply = claim if a.claim == "statement" else str(r["answer"])  # a one-word answer is no sentence to read
        res = g.check([{"role": "user", "content": r["question"]}], reply, documents=[r["passage"]])
        t_g += time.time() - t
        zs = [s["z"] for s in res["sentences"]] or [0.0]
        t = time.time()
        pm = mc.p_unsupported(r["passage"], claim)
        t_m += time.time() - t
        out.append({"source": r["source_ds"], "y": int(r["label"] == "FAIL"), "grounding_total": res["total"],
                    "grounding_max": max(zs), "grounding_verdict": res["verdict"], "minicheck": pm})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(rows)} grounding {t_g / (i + 1):.1f}s minicheck {t_m / (i + 1):.2f}s", flush=True)
    res = {"n": len(out), "reader": a.reader, "claim": a.claim, "seconds_per_answer": {"grounding": round(t_g / len(out), 2),
                                                                    "minicheck": round(t_m / len(out), 2)}}
    res.update(summarize(out, ("grounding_total", "grounding_max", "minicheck")))
    print(json.dumps(res, indent=1))
    json.dump(dict(res, rows=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
