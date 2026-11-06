"""Idea 1: test each claim against its own alternatives, so the score has no scale to drift between domains.

For every checkable claim in a reply (a number, a capitalised name, a leading yes / no / maybe), alternatives are
made by swapping in other spans of the same type from the same evidence (other numbers, other names; yes <-> no).
A frozen reader scores the whole reply under each variant, with the evidence and without it:

    support(variant) = log p(reply variant | question + evidence) - log p(reply variant | question)

The claim's rank among its own variants is its score: p = (1 + #variants supported at least as well) / (1 + #variants).
A written claim the evidence backs ranks first (p small); one it does not back ranks among, or below, its
alternatives. The fixed rule "flag when the written claim is not the best supported" uses no fitted threshold: the
question is whether its false-alarm rate holds across HaluBench's sources without calibration, where every score
so far moved with the domain (docs/self_grounded.md, docs/source_flow.md).

Same 300 HaluBench answers as results/halubench_compare_qwen3_4b_seed1.json (seed 1), Qwen3-4B, CPU.

    python experiments/claim_contrast.py run
    python experiments/claim_contrast.py analyze
"""
import argparse
import copy
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = os.environ.get("CLAIM_CONTRAST_CACHE", "results/claim_contrast_qwen3_4b")
NUM = re.compile(r"(?<![\w.])-?\$?\d(?:[\d,]*\d)?(?:\.\d+)?%?")
NAME = re.compile(r"\b[A-Z][a-zA-Z'\-]+(?:\s+(?:of|the|de|van|von|and|&)?\s*[A-Z][a-zA-Z'\-]+)*")
STOP = {"The", "A", "An", "In", "On", "It", "This", "That", "He", "She", "They", "We", "I", "Yes", "No", "Maybe",
        "There", "These", "Those", "His", "Her", "Its", "Their", "As", "At", "By", "For", "From", "If", "But", "And",
        "Or", "So", "To", "With", "Of", "After", "Before", "During", "When", "While", "Which", "Who", "What"}


def claims(reply):
    """(start, end, kind) of each checkable span: numbers, names, a leading yes/no/maybe."""
    out = [(m.start(), m.end(), "num") for m in NUM.finditer(reply)]
    for m in NAME.finditer(reply):
        if m.group(0) in STOP or any(s < m.end() and m.start() < e for s, e, _ in out):
            continue
        first = m.start() == 0 or re.search(r"[.!?\n]\s*$", reply[max(0, m.start() - 3):m.start()])
        if first and " " not in m.group(0) and len(reply.split()) > 3:        # a sentence's first word
            continue
        out.append((m.start(), m.end(), "name"))
    m = re.match(r"\s*(yes|no|maybe)\b", reply, re.I)
    if m:
        out.append((m.start(1), m.end(1), "yn"))
    return sorted(out)


def alternatives(span, kind, evidence, k):
    if kind == "yn":
        w = span.lower()
        alts = [a for a in ("yes", "no", "maybe") if a != w]
        return [a.capitalize() if span[0].isupper() else a for a in alts]
    norm = lambda s: re.sub(r"[\s,$%]", "", s.lower())                      # noqa: E731
    if kind == "num":
        shape = lambda x: ("money" if "$" in x else "pct" if "%" in x else        # noqa: E731
                           "year" if re.fullmatch(r"(1[89]|20)\d\d", x) else "num")
        seen = Counter(m.group(0) for m in NUM.finditer(evidence))
        cands = [c for c, _ in seen.most_common() if norm(c) != norm(span)]
        same = [c for c in cands if shape(c) == shape(span)]
        return (same if len(same) >= 2 else cands)[:k]
    seen = Counter()
    for m in NAME.finditer(evidence):                                        # not only ever capitalised by
        start = m.start() == 0 or re.search(r"[.!?\n]\s*$", evidence[max(0, m.start() - 3):m.start()])
        if not start or " " in m.group(0):                                   # standing first in a sentence
            seen[m.group(0)] += 1
    cands = [c for c, _ in seen.most_common() if norm(c) != norm(span) and c not in STOP]
    return cands[:k]


class Scorer:
    """log p of many reply variants after one shared prefix: the prefix is read once and its cache reused."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok

    def prefix(self, user):
        msgs = [{"role": "user", "content": user}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(p, add_special_tokens=False).input_ids

    def logps(self, user, replies, batch=4):
        import torch
        pre = self.prefix(user)
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([pre[:-1]]), use_cache=True)
        base, last = out.past_key_values, pre[-1]
        res = []
        for b in range(0, len(replies), batch):
            chunk = [self.tok(r, add_special_tokens=False).input_ids for r in replies[b:b + batch]]
            L = max(map(len, chunk))
            ids = torch.full((len(chunk), L + 1), self.tok.pad_token_id or 0)
            att = torch.zeros((len(chunk), len(pre) - 1 + L + 1), dtype=torch.long)
            att[:, :len(pre) - 1] = 1
            for i, c in enumerate(chunk):
                ids[i, 0] = last
                ids[i, 1:1 + len(c)] = torch.tensor(c)
                att[i, len(pre) - 1:len(pre) + len(c)] = 1
            cache = copy.deepcopy(base)
            cache.batch_repeat_interleave(len(chunk))
            with torch.no_grad():
                lg = self.model(input_ids=ids, attention_mask=att, past_key_values=cache, use_cache=True).logits.float()
            lp = torch.log_softmax(lg[:, :-1], -1)
            for i, c in enumerate(chunk):
                t = torch.tensor(c)
                res.append(float(lp[i, torch.arange(len(c)), t].sum()))
        return res


def run(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from halubench_compare import sample
    from vt.grounding import select
    tok = AutoTokenizer.from_pretrained(a.reader)
    model = AutoModelForCausalLM.from_pretrained(a.reader, dtype=torch.bfloat16).eval()
    sc = Scorer(model, tok)
    path = f"{CACHE}.jsonl"
    seen = {json.loads(l)["i"] for l in open(path)} if os.path.exists(path) else set()
    rows, t0, k = sample(a.per_source, a.seed), time.time(), 0
    with open(path, "a") as f:
        for i, r in enumerate(rows):
            if i in seen:
                continue
            reply, q = str(r["answer"]), r["question"]
            ev = "\n\n".join(select([r["passage"]], f"{q} {reply}", a.budget))
            tests = []
            for s, e, kind in claims(reply)[:a.max_claims]:
                alts = alternatives(reply[s:e], kind, ev, a.k)
                if alts:
                    tests.append((s, e, kind, alts))
            variants = [reply] + [reply[:s] + alt + reply[e:] for s, e, _, alts in tests for alt in alts]
            rec = {"i": i, "source": r["source_ds"], "y": int(r["label"] == "FAIL"), "claims": []}
            if tests:
                with_ev = sc.logps(f"{q}\n\n{ev}", variants)
                without = sc.logps(q, variants)
                sup = np.array(with_ev) - np.array(without)
                lw = np.array(with_ev)
                j = 1
                for s, e, kind, alts in tests:
                    n = len(alts)
                    alt_sup, alt_lw = sup[j:j + n], lw[j:j + n]
                    rec["claims"].append({"span": reply[s:e], "kind": kind, "alternatives": n,
                                          "p_support": float((1 + (alt_sup >= sup[0]).sum()) / (1 + n)),
                                          "p_likelihood": float((1 + (alt_lw >= lw[0]).sum()) / (1 + n)),
                                          "margin_support": float(sup[0] - alt_sup.max()),
                                          "margin_likelihood": float(lw[0] - alt_lw.max())})
                    j += n
            f.write(json.dumps(rec) + "\n")
            f.flush()
            k += 1
            if k % 10 == 0:
                print(f"{i + 1}/{len(rows)} {(time.time() - t0) / k:.1f}s an answer", flush=True)


def analyze(a):
    from vt.correctness import auroc
    rows = [json.loads(l) for l in open(f"{CACHE}.jsonl")]
    base = {b["source"] + str(j): b for j, b in enumerate(json.load(open(a.baseline))["rows"])} if a.baseline else {}
    srcs = sorted({r["source"] for r in rows})
    res = {"n": len(rows), "testable": round(float(np.mean([bool(r["claims"]) for r in rows])), 3), "by_source": {}}
    for kind in ("support", "likelihood"):
        for src in srcs + ["all"]:
            rs = [r for r in rows if r["claims"] and (src == "all" or r["source"] == src)]
            if not rs:
                continue
            y = np.array([r["y"] for r in rs], float)
            p = np.array([max(c[f"p_{kind}"] for c in r["claims"]) for r in rs])              # the worst claim
            m = np.array([-min(c[f"margin_{kind}"] for c in r["claims"]) for r in rs])
            first = p > np.array([min(1 / (1 + c["alternatives"]) for c in r["claims"]) for r in rs]) + 1e-9
            d = res["by_source"].setdefault(src, {"testable_answers": len(rs), "hallucinated": int(y.sum())})
            d[kind] = {"auroc_rank": round(float(auroc(p, y)), 3) if 0 < y.sum() < len(y) else None,
                       "auroc_margin": round(float(auroc(m, y)), 3) if 0 < y.sum() < len(y) else None,
                       # the fixed rule: flag when a written claim is not the best supported of its variants
                       "rule_false_alarms": round(float(first[y == 0].mean()), 3) if (y == 0).any() else None,
                       "rule_caught": round(float(first[y == 1].mean()), 3) if (y == 1).any() else None}
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--reader", default="Qwen/Qwen3-4B")
    r.add_argument("--per-source", type=int, default=50)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--budget", type=int, default=8000)
    r.add_argument("--k", type=int, default=8, help="alternatives per claim")
    r.add_argument("--max-claims", type=int, default=6)
    z = sub.add_parser("analyze")
    z.add_argument("--baseline", default="")
    z.add_argument("--out", default="results/claim_contrast_qwen3_4b.json")
    a = ap.parse_args()
    {"run": run, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
