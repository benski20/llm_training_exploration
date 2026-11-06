"""Reading a model's OWN answers from documents, with signals that need no training: does one setting hold a
false-alarm rate across domains?

Earlier work covered its neighbours, not this:
- the model reading its own closed-book answers (VISION.md: trained probes, AUROC 0.78-0.86, calibration lost on a
  new dataset);
- a reader judging other models' document-grounded answers (docs/grounding_trace.md, docs/source_flow.md:
  thresholds drift by domain, its own confidence is at chance).

Here Qwen3-4B answers questions from a passage (greedy, its chat template, no thinking), in four domains with
short gold answers:

    drop        numbers computed from a passage (HaluBench's DROP, correct rows as gold)
    finance     figures from financial filings (HaluBench's FinanceBench)
    halueval    multi-hop names from Wikipedia paragraphs (HaluEval QA)
    pubmed      yes / no / maybe judgements on abstracts (HaluBench's PubMedQA)

Each answer is labeled right or wrong against its gold, then read in one pass with the passage and one without:
its own token confidence, entropy and margin, the evidence shift (log p with the passage minus without), the
copying heads' conflict and copy support, and source flow (evidence share; reply minus sink). Reported: AUROC
per signal and domain, and the reliability test: a threshold set for 10% false alarms on one domain's right
answers, applied unchanged to the others.

    python experiments/self_grounded.py run --n 80
    python experiments/self_grounded.py analyze
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = os.environ.get("SELF_GROUNDED_CACHE", "results/self_grounded_qwen3_4b")
DOMAINS = ("drop", "finance", "halueval", "pubmed")
INSTRUCT = {"drop": "Answer the question from the passage. Give only the short answer (a number or a few words).",
            "finance": "Answer the question from the filing excerpt. Give only the figure, with its unit.",
            "halueval": "Answer the question from the passage. Give only the short answer.",
            "pubmed": "Answer the question from the abstract. Start with yes, no or maybe, then give one sentence."}


def items(domain, n, seed=0):
    from datasets import load_dataset
    if domain == "halueval":
        rows = list(load_dataset("pminervini/HaluEval", "qa", split="data"))
        random.Random(seed).shuffle(rows)
        return [{"question": r["question"], "passage": r["knowledge"], "gold": r["right_answer"]} for r in rows[:n]]
    src = {"drop": "DROP", "finance": "FinanceBench", "pubmed": "pubmedQA"}[domain]
    rows = [r for r in load_dataset("PatronusAI/HaluBench", split="test") if r["source_ds"] == src and r["label"] == "PASS"]
    random.Random(seed).shuffle(rows)
    return [{"question": r["question"], "passage": r["passage"], "gold": str(r["answer"])} for r in rows[:n]]


def norm(s):
    s = re.sub(r"\b(a|an|the)\b", " ", s.lower())
    return " ".join(re.sub(r"[^a-z0-9.%-]", " ", s).split())


def numbers(s):
    out = []
    for m in re.finditer(r"-?\d[\d,]*\.?\d*\s*(billion|million|thousand|bn|m|k|%)?", s.lower()):
        try:
            v = float(m.group(0).split()[0].rstrip("%").replace(",", "").rstrip("bnmk") or "nan")
        except ValueError:
            continue
        mult = {"billion": 1e9, "bn": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}.get((m.group(1) or "").strip(), 1)
        out.append(v * mult)
    return out


def right(domain, pred, gold):
    """Lenient, per domain: the gold's number (any unit scale, 1%), its yes/no/maybe, or its words."""
    p, g = norm(pred), norm(gold)
    if domain == "pubmed":
        first = lambda s: next((w.strip(".,;:") for w in s.split() if w.strip(".,;:") in ("yes", "no", "maybe")), None)   # noqa: E731
        return first(g) is not None and first(p) == first(g)
    gn, pn = numbers(gold), numbers(pred)
    if domain in ("drop", "finance") and gn:
        alts = [norm(a) for a in gold.split(",")] if domain == "drop" else []
        if any(a and a in p for a in alts):
            return True
        scales = (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9) if domain == "finance" else (1,)
        return any(abs(abs(x) * sc - abs(y)) <= 0.01 * abs(y) + 1e-9 for x in pn for y in gn for sc in scales)
    if g and g in p:
        return True
    pt, gt = p.split(), g.split()
    common = sum(min(pt.count(w), gt.count(w)) for w in set(gt))
    if not common:
        return False
    prec, rec = common / len(pt), common / len(gt)
    return 2 * prec * rec / (prec + rec) >= 0.5


def run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.copyheads import HeadAttention, find_copy_heads, token_copy
    from vt.grounding import select
    from vt.sourceflow import SourceFlow, regions_for
    tok = AutoTokenizer.from_pretrained(a.reader)
    model = AutoModelForCausalLM.from_pretrained(a.reader, dtype=torch.bfloat16).eval()
    layers = list(range(2, model.config.num_hidden_layers, 4))
    heads = [(l, h) for l, h, _ in find_copy_heads(model)]
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    path = f"{CACHE}.jsonl"
    seen = {(json.loads(l)["domain"], json.loads(l)["i"]) for l in open(path)} if os.path.exists(path) else set()
    t0, k = time.time(), 0
    with open(path, "a") as f:
        for d in a.domains:
            for i, it in enumerate(items(d, a.n)):
                if (d, i) in seen:
                    continue
                passage = "\n\n".join(select([it["passage"]], it["question"], a.budget))
                question = f"{INSTRUCT[d]}\n\nQuestion: {it['question']}"
                pre, _ = regions_for(tok, question, passage, 0)
                with torch.no_grad():
                    g = model.generate(torch.tensor([pre]), max_new_tokens=60, do_sample=False,
                                       pad_token_id=tok.eos_token_id)[0, len(pre):].tolist()
                ans = []
                for t in g:
                    if t in eos:
                        break
                    ans.append(t)
                text = tok.decode(ans).strip()
                if not ans:
                    continue
                R = len(ans)
                pre, region = regions_for(tok, question, passage, R)
                P = len(pre)
                q = range(P - 1, P + R - 1)
                with SourceFlow(model, layers) as sf, HeadAttention(model, heads) as ha:
                    sf.at(q, region), ha.at(q)
                    with torch.no_grad():
                        lw = model(input_ids=torch.tensor([pre + ans]), logits_to_keep=R + 1).logits[0, :-1].float()
                    flow, att = sf.collect(), ha.collect()
                lw = torch.log_softmax(lw, -1)
                pre0, _ = regions_for(tok, question, "", R)                       # the same, without the passage
                with torch.no_grad():
                    lo = torch.log_softmax(model(input_ids=torch.tensor([pre0 + ans]),
                                                 logits_to_keep=R + 1).logits[0, :-1].float(), -1)
                idx, ids = torch.arange(R), torch.tensor(ans)
                lp_w, lp_o = lw[idx, ids].numpy(), lo[idx, ids].numpy()
                ent = -(lw.exp() * lw).sum(-1).numpy()
                top2 = lw.topk(2, -1).values.numpy()
                match, conflict = token_copy(att, pre, ans)
                wordy = np.array([bool(re.search(r"[A-Za-z0-9]", tok.decode([t]))) for t in ans])
                w = np.where(wordy)[0] if wordy.any() else np.arange(R)
                rec = {"domain": d, "i": i, "question": it["question"], "gold": it["gold"], "answer": text,
                       "right": bool(right(d, text, it["gold"])), "tokens": R, "prefix_tokens": P,
                       "logp_mean": float(lp_w.mean()), "logp_min": float(lp_w.min()),
                       "entropy_mean": float(ent.mean()), "entropy_max": float(ent.max()),
                       "margin_min": float((top2[:, 0] - top2[:, 1]).min()),
                       "shift_mean": float((lp_w - lp_o).mean()), "shift_min": float((lp_w - lp_o)[w].min()),
                       "copy_conflict": float(conflict[w].max()), "uncopied": float((match[w] < 0.05).mean()),
                       "evidence_share": float(flow[:, :, 2].mean()),
                       "reply_minus_sink": float(flow[:, :, 3].mean() - flow[:, :, 0].mean())}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                k += 1
                if k % 10 == 0:
                    print(f"{d} {i + 1}/{a.n} {(time.time() - t0) / k:.1f}s an answer", flush=True)


# higher = more likely wrong, fixed before looking at any label
SIGNALS = {"neg_logp_mean": ("logp_mean", -1), "neg_logp_min": ("logp_min", -1), "entropy_mean": ("entropy_mean", 1),
           "entropy_max": ("entropy_max", 1), "neg_margin_min": ("margin_min", -1), "neg_shift_mean": ("shift_mean", -1),
           "neg_shift_min": ("shift_min", -1), "copy_conflict": ("copy_conflict", 1), "uncopied": ("uncopied", 1),
           "neg_evidence_share": ("evidence_share", -1), "reply_minus_sink": ("reply_minus_sink", 1)}
COMBO = ("neg_logp_min", "entropy_max", "neg_margin_min", "neg_shift_mean", "copy_conflict")


def analyze(a):
    from vt.correctness import auroc
    rows = [json.loads(l) for l in open(f"{CACHE}.jsonl")]
    doms = [d for d in DOMAINS if any(r["domain"] == d for r in rows)]
    val = lambda rs, s: np.array([SIGNALS[s][1] * r[SIGNALS[s][0]] for r in rs])     # noqa: E731
    res = {"n": len(rows), "domains": {}, "transfer": {}}
    by = {d: [r for r in rows if r["domain"] == d] for d in doms}
    for d, rs in by.items():
        y = np.array([not r["right"] for r in rs], float)                    # 1 = wrong
        res["domains"][d] = {"answers": len(rs), "accuracy": round(1 - y.mean(), 3),
                             "auroc_wrong": {s: (round(float(auroc(val(rs, s), y)), 3) if 0 < y.sum() < len(y) else None)
                                             for s in SIGNALS}}

    def combo(rs, ref):
        """Sum of the chosen signals, each on the reference domain's scale (its answers, no labels)."""
        return sum((val(rs, s) - val(ref, s).mean()) / (val(ref, s).std() + 1e-9) for s in COMBO)
    for d, rs in by.items():
        y = np.array([not r["right"] for r in rs], float)
        res["domains"][d]["auroc_wrong"]["combo"] = round(float(auroc(combo(rs, rs), y)), 3) if 0 < y.sum() < len(y) else None
    for src in doms:                                         # a 10% false-alarm threshold set on src, used everywhere
        out = {}
        for s in ("neg_logp_min", "entropy_max", "combo"):
            score = (lambda rs: combo(rs, by[src])) if s == "combo" else (lambda rs, s=s: val(rs, s))
            ok = score([r for r in by[src] if r["right"]])
            thr = float(np.quantile(ok, 0.9))
            per = {}
            for d, rs in by.items():
                sc, y = score(rs), np.array([not r["right"] for r in rs])
                per[d] = {"false_alarms": round(float((sc[~y] > thr).mean()), 3) if (~y).any() else None,
                          "caught": round(float((sc[y] > thr).mean()), 3) if y.any() else None}
            out[s] = per
        res["transfer"][f"threshold_from_{src}"] = out
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--reader", default="Qwen/Qwen3-4B")
    r.add_argument("--n", type=int, default=80)
    r.add_argument("--budget", type=int, default=6000)
    r.add_argument("--domains", nargs="+", default=list(DOMAINS))
    z = sub.add_parser("analyze")
    z.add_argument("--out", default="results/self_grounded_qwen3_4b.json")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
