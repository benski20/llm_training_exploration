"""Does source flow (vt/sourceflow.py) flag errors in every kind of output, not only copied spans?

Three kinds of output, one frozen reader (Qwen3-4B), one forward pass each, no task-specific features:

    halubench     answers grounded in a passage (unit: the answer; 300, seed 1, the sample of
                  results/halubench_compare_qwen3_4b_seed1.json)
    mathshepherd  reasoning: GSM8K / MATH solutions, each step labeled by Math-Shepherd's rollouts (unit: the step;
                  no documents, the sources are the question and the earlier steps)
    wikibio       long form with no documents: GPT-3 biographies, each sentence annotated accurate or not
                  (unit: the sentence)

Per unit: the share of the attention output drawn from sink / question / evidence / reply at every 4th layer, the
written tokens' log-probabilities, and the copying heads' conflict and copy support. Reported: each label-free
scalar's AUROC per dataset; a linear readout fitted within a dataset (5-fold); and the real test of generality,
a readout fitted on two kinds of output and scored on the third.

    python experiments/source_flow.py extract --dataset mathshepherd --n 150
    python experiments/source_flow.py analyze --out results/source_flow_qwen3_4b.json
"""
import argparse
import ast
import json
import os
import random
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = os.environ.get("SOURCE_FLOW_CACHE", "results/source_flow_qwen3_4b")


def items_mathshepherd(n, seed=0):
    from datasets import load_dataset
    d = load_dataset("peiyi9979/Math-Shepherd", split="train")     # stored in order of label: sample it all
    rows = d.select(random.Random(seed).sample(range(len(d)), 20000))
    bad, good = [], []
    for r in rows:
        inp, lab = r["input"], r["label"]
        m = re.search(r"Step 1:", inp)
        if not m:
            continue
        steps = [s.strip() for s in inp[m.start():].split("ки") if s.strip()]
        marks = re.findall(r" ([+-])(?=\n|$)", lab[m.start():])
        if len(steps) != len(marks) or not 2 <= len(steps) <= 8:
            continue
        item = {"question": inp[:m.start()].strip(), "evidence": "", "units": list(zip(steps, [int(x == "-") for x in marks])),
                "task": r["task"]}
        (bad if "-" in marks else good).append(item)
        if len(bad) >= n // 2 and len(good) >= n - n // 2:
            break
    return bad[:n // 2] + good[:n - n // 2]


def items_wikibio(n):
    from datasets import load_dataset
    out = []
    for r in load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation"):
        lit = lambda v: ast.literal_eval(v) if isinstance(v, str) else list(v)   # noqa: E731
        sents, ann = lit(r["gpt3_sentences"]), lit(r["annotation"])
        text = r["gpt3_text"]
        name = text.split(" (")[0] if 0 < text.find(" (") < 60 else " ".join(text.split()[:3])
        out.append({"question": f"This is a Wikipedia passage about {name}:", "evidence": "",
                    "units": [(s, int(a != "accurate")) for s, a in zip(sents, ann)]})
    return out[:n]


def items_halubench(n, seed=1):
    from halubench_compare import sample
    from vt.grounding import select
    out = []
    for r in sample(n // 6, seed):
        ev = "\n\n".join(select([r["passage"]], f"{r['question']} {r['answer']}", 8000))
        out.append({"question": r["question"], "evidence": ev, "units": [(str(r["answer"]), int(r["label"] == "FAIL"))],
                    "source": r["source_ds"]})
    return out


def extract(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.copyheads import HeadAttention, find_copy_heads, token_copy
    from vt.sourceflow import SourceFlow, regions_for
    items = {"mathshepherd": lambda: items_mathshepherd(a.n), "wikibio": lambda: items_wikibio(a.n),
             "halubench": lambda: items_halubench(a.n)}[a.dataset]()
    tok = AutoTokenizer.from_pretrained(a.reader)
    model = AutoModelForCausalLM.from_pretrained(a.reader, dtype=torch.bfloat16).eval()
    layers = list(range(2, model.config.num_hidden_layers, 4))
    heads = [(l, h) for l, h, _ in find_copy_heads(model)]
    path = f"{CACHE}.{a.dataset}.jsonl"
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    seen = {json.loads(l)["item"] for l in open(path)} if done else set()
    t0, k = time.time(), 0
    with open(path, "a") as f:
        for i, it in enumerate(items):
            if i in seen:
                continue
            sep = " " if a.dataset == "wikibio" else "\n"
            reply, spans, pos = "", [], 0
            for text, y in it["units"]:
                reply += (sep if reply else "") + text
                spans.append((len(reply) - len(text), len(reply), y))
            enc = tok(reply, add_special_tokens=False, return_offsets_mapping=True)
            R = len(enc.input_ids)
            pre, region = regions_for(tok, it["question"], it["evidence"], R)
            P = len(pre)
            q = range(P - 1, P + R - 1)
            with SourceFlow(model, layers) as sf, HeadAttention(model, heads) as ha:
                sf.at(q, region), ha.at(q)
                with torch.no_grad():
                    logits = model(input_ids=torch.tensor([pre + enc.input_ids]), logits_to_keep=R + 1).logits[0, :-1]
                flow, att = sf.collect(), ha.collect()                          # layers x R x regions
            lp = torch.log_softmax(logits.float(), -1)[torch.arange(R), torch.tensor(enc.input_ids)].numpy()
            match, conflict = token_copy(att, pre, enc.input_ids)
            wordy = np.array([bool(re.search(r"[A-Za-z0-9]", tok.decode([t]))) for t in enc.input_ids])
            for u, (s, e, y) in enumerate(spans):
                js = [j for j, (ts, te) in enumerate(enc.offset_mapping) if ts >= s and te <= e]
                if not js:
                    continue
                w = [j for j in js if wordy[j] and j > 0] or js
                rec = {"item": i, "unit": u, "y": y, "dataset": a.dataset, "source": it.get("source", it.get("task", "")),
                       "flow": flow[:, js].mean(1).round(4).tolist(),               # layers x regions
                       "logp_mean": float(lp[js].mean()), "logp_min": float(lp[js].min()),
                       "copy_conflict": float(conflict[w].max()), "uncopied": float((match[w] < 0.05).mean()),
                       "tokens": len(js)}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            k += 1
            if k % 10 == 0:
                print(f"{a.dataset} {i + 1}/{len(items)} {(time.time() - t0) / k:.1f}s an item", flush=True)
    json.dump({"layers": layers, "copy_heads": heads}, open(f"{CACHE}.meta.json", "w"))


def load(ds):
    rows = [json.loads(l) for l in open(f"{CACHE}.{ds}.jsonl")]
    F = np.array([r["flow"] for r in rows])                                     # units x layers x regions
    extra = np.array([[r["logp_mean"], r["logp_min"], r["copy_conflict"], r["uncopied"]] for r in rows])
    return rows, F, extra, np.array([r["y"] for r in rows], float), np.array([r["item"] for r in rows])


def analyze(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    from vt.correctness import auroc
    from vt.sourceflow import REGIONS
    data = {ds: load(ds) for ds in ("halubench", "mathshepherd", "wikibio") if os.path.exists(f"{CACHE}.{ds}.jsonl")}
    meta = json.load(open(f"{CACHE}.meta.json"))
    z = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-9)        # noqa: E731  per dataset, no labels
    res = {"layers": meta["layers"], "datasets": {}, "transfer": {}}
    feats = {}
    for ds, (rows, F, extra, y, groups) in data.items():
        scal = {f"share_{r}_mean": F[:, :, i].mean(1) for i, r in enumerate(REGIONS)}
        mid = slice(len(meta["layers"]) // 3, 2 * len(meta["layers"]) // 3 + 1)
        scal.update({f"share_{r}_mid": F[:, mid, i].mean(1) for i, r in enumerate(REGIONS)})
        scal.update({"neg_logp_mean": -extra[:, 0], "neg_logp_min": -extra[:, 1], "copy_conflict": extra[:, 2],
                     "uncopied": extra[:, 3]})
        sets = {"flow": z(F.reshape(len(F), -1)), "logp": z(extra[:, :2]),
                "flow+logp": z(np.hstack([F.reshape(len(F), -1), extra[:, :2]])),
                "flow+logp+copy": z(np.hstack([F.reshape(len(F), -1), extra]))}
        feats[ds] = (sets, y)
        cv = {}
        for name, X in sets.items():
            p = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(X, y, groups):
                p[te] = LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            cv[name] = round(float(auroc(p, y)), 3)
        res["datasets"][ds] = {"units": len(y), "positive": int(y.sum()),
                               "label_free_auroc": {k: round(float(auroc(v, y)), 3) for k, v in scal.items()},
                               "cv5_auroc": cv}
    for target in feats:                                     # fitted on the other kinds of output only
        others = [d for d in feats if d != target]
        if not others:
            continue
        out = {}
        for name in ("flow", "logp", "flow+logp", "flow+logp+copy"):
            X = np.vstack([feats[d][0][name] for d in others])
            yy = np.concatenate([feats[d][1] for d in others])
            m = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(X, yy)
            out[name] = round(float(auroc(m.predict_proba(feats[target][0][name])[:, 1], feats[target][1])), 3)
        res["transfer"][f"to_{target}_from_{'+'.join(others)}"] = out
    print(json.dumps(res, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dataset", choices=["halubench", "mathshepherd", "wikibio"], required=True)
    e.add_argument("--n", type=int, default=150)
    e.add_argument("--reader", default="Qwen/Qwen3-4B")
    z = sub.add_parser("analyze")
    z.add_argument("--out", default="results/source_flow_qwen3_4b.json")
    a = ap.parse_args()
    {"extract": extract, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
