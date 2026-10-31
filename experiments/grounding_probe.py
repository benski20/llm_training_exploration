"""Train on the mechanism, test on nature: a probe for "this claim is grounded in the context" learned from
labels made by construction, then scored on human-labeled hallucinations from a different domain.

    train   Dolly's reference paragraphs (Wikipedia-style text, CC-BY-SA), no human labels: for each paragraph a
            reply that is one of its sentences (grounded), that sentence with a number or a name swapped
            (contradicted), or one of its sentences taken out of the paragraph shown (unsupported: on topic,
            but not in the context)
    read    the model's MLP neurons (down_proj inputs) averaged over the reply's tokens, every 4th layer, with
            the paragraph in the context
    test    RAGTruth's test answers (news summaries, QA from passages, data-to-text from reviews; written by
            other models, hallucinated spans marked by people): every sentence read the same way, in one pass
            over context and answer

A dense L2 probe and sparse probes (the k neuron poles that best separate grounded from not on the synthetic
data), AUROC for "hallucinated" at sentence and response level, next to the training-free evidence shift
(0.69 sentence) and this repo's trained heads on unseen sources (0.58).

    python experiments/grounding_probe.py --model Qwen/Qwen3-4B --train 600 --test 200 --out results/grounding_probe_qwen3_4b.json
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

from evidence_shift import prefix, sentences, spans  # noqa: E402


def sent_split(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 8]


def corrupt(s, rng, names):
    """The sentence with one number or one capitalised name changed, or None if it has neither."""
    nums = list(re.finditer(r"\b\d[\d,]*(?:\.\d+)?\b", s))
    caps = [m for m in re.finditer(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]{2,}\b", s)]
    if nums and (not caps or rng.random() < 0.5):
        m = rng.choice(nums)
        v = m.group(0).replace(",", "")
        try:
            x = float(v)
        except ValueError:
            return None
        new = str(int(x) + rng.choice([-3, -2, -1, 1, 2, 5, 10])) if x == int(x) else f"{x * rng.choice([0.5, 2, 1.5]):.1f}"
        return s[:m.start()] + new + s[m.end():]
    if caps:
        m = rng.choice(caps)
        other = [n for n in names if n != m.group(0)]
        return s[:m.start()] + rng.choice(other) + s[m.end():] if other else None
    return None


LEADS = ("", "", "According to the text, ", "The passage says that ", "It notes that ", "In short, ", "Notably, ")


def phrase(s, rng):
    """The sentence worded like part of an answer: a lead-in, its parentheses dropped, its first letter lowered
    after a lead-in. Applied alike to every kind, so the form says nothing about the label."""
    s = re.sub(r"\s*\([^()]*\)", "", s).strip()
    lead = rng.choice(LEADS)
    return lead + (s[0].lower() + s[1:] if lead and s[:1].isupper() and not s[:2].isupper() else s)


def synthetic(n, seed=0, phrasing=False):
    from datasets import load_dataset
    rng = random.Random(seed)
    ctxs = [r["context"] for r in load_dataset("databricks/databricks-dolly-15k")["train"]
            if r["context"] and 60 <= len(r["context"].split()) <= 400]
    rng.shuffle(ctxs)
    names = sorted({w for c in ctxs[:2000] for w in re.findall(r"\b[A-Z][a-z]{3,}\b", c)})
    out = []
    for i, c in enumerate(ctxs):
        ss = sent_split(c)
        if not ss:
            continue
        s = rng.choice(ss)
        kind = ("grounded", "contradicted", "unsupported")[len(out) % 3]
        if kind == "contradicted":
            s2 = corrupt(s, rng, names)
            if s2 is None or s2 == s:
                continue
            s = s2
        elif kind == "unsupported":                          # on topic but not in the context: taken out of it
            if len(ss) < 3:
                continue
            c = c.replace(s, " ").strip()
        out.append({"context": c, "reply": phrase(s, rng) if phrasing else s, "kind": kind, "y": int(kind != "grounded")})
        if len(out) >= n:
            break
    return out


class Reader:
    """MLP neuron activations (down_proj inputs) at every position, for every `every`-th layer."""

    def __init__(self, model, every=4):
        self.acts, self.layers = {}, list(range(every - 1, len(model.model.layers), every))
        self.model = model
        for i in self.layers:
            model.model.layers[i].mlp.down_proj.register_forward_pre_hook(self._hook(i))

    def _hook(self, i):
        def f(mod, inp):
            self.acts[i] = inp[0][0].float()
        return f

    def run(self, ids):
        import torch
        with torch.no_grad():
            self.model(input_ids=torch.tensor([ids]))

    def mean(self, a, b):
        """Mean activation of positions [a, b) in every read layer, concatenated."""
        return np.concatenate([self.acts[i][a:b].mean(0).cpu().numpy() for i in self.layers]).astype(np.float32)


def ragtruth_features(model, tok, reader, n, max_chars, t0):
    """Each sentence of RAGTruth test answers: its mean MLP activations (context in view), its mean evidence
    shift and its largest disagreement, its label; read before the other passes (no copy of the activations)."""
    from datasets import load_dataset

    from evidence_shift import disagreement, token_logprobs
    ds = load_dataset("wandb/RAGTruth-processed", split="test").shuffle(seed=0)
    bad, good = [r for r in ds if spans(r)], [r for r in ds if not spans(r)]
    rows = bad[:n // 2] + good[:n - n // 2]
    Xt, yt, kt, resp, ev = [], [], [], [], []
    for i, r in enumerate(rows):
        out = r["output"]
        pre = prefix(tok, f"{r['query']}\n\n{r['context'][:max_chars]}")
        enc = tok(out, add_special_tokens=False, return_offsets_mapping=True)
        reader.run(pre + enc.input_ids)
        sp, here = spans(r), []
        for s, e in sentences(out):
            js = [j for j, (ts, te) in enumerate(enc.offset_mapping) if ts >= s and te <= e]
            if not js:
                continue
            kinds = [k for a_, b_, k in sp if s < b_ and a_ < e]
            Xt.append(reader.mean(len(pre) + js[0], len(pre) + js[-1] + 1))
            yt.append(int(bool(kinds)))
            kt.append(kinds[0] if kinds else "")
            resp.append(i)
            here.append(js)
        reader.acts = {}
        lw, dw = token_logprobs(model, tok, pre, enc.input_ids)
        lo, do = token_logprobs(model, tok, prefix(tok, r["query"]), enc.input_ids)
        dis, shift = disagreement(dw, do, enc.input_ids), lw - lo
        del dw, do
        ev += [(float(shift[js].mean()), float(dis[js].max())) for js in here]
        if (i + 1) % 50 == 0:
            print(f"ragtruth {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    return (np.stack(Xt), np.array(yt), kt, np.array(resp), np.array(ev),
            np.array([int(bool(spans(r))) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--max-context-chars", type=int, default=4000)
    ap.add_argument("--out", default="grounding_probe.json")
    ap.add_argument("--export", help="save the probe (C=0.01) for serving with vt/grounding.py (.npz)")
    ap.add_argument("--test-cache", help="save / reuse the RAGTruth side here (.npz; depends only on the model)")
    ap.add_argument("--phrasing", action="store_true", help="give every constructed reply the wording of an "
                    "answer (a lead-in, parentheses dropped), so copied text cannot be told by its form")
    a = ap.parse_args()
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from vt.correctness import auroc
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    reader = Reader(model)
    t0 = time.time()

    syn = synthetic(a.train, phrasing=a.phrasing)
    Xs, ys, ks = [], [], []
    for i, ex in enumerate(syn):
        pre = prefix(tok, "Answer from this text.\n\n" + ex["context"])
        rep = tok(ex["reply"], add_special_tokens=False).input_ids
        reader.run(pre + rep)
        Xs.append(reader.mean(len(pre), len(pre) + len(rep)))
        ys.append(ex["y"])
        ks.append(ex["kind"])
        if (i + 1) % 100 == 0:
            print(f"synthetic {i + 1}/{len(syn)} {time.time() - t0:.0f}s", flush=True)
    Xs, ys = np.stack(Xs), np.array(ys)

    if a.test_cache and os.path.exists(a.test_cache):           # the test side depends only on the model
        z_ = np.load(a.test_cache, allow_pickle=True)
        Xt, yt, kt, resp, ev, resp_y = (z_[k] for k in ("Xt", "yt", "kt", "resp", "ev", "resp_y"))
        Xt, kt = Xt.astype(np.float32), list(kt)
    else:
        Xt, yt, kt, resp, ev, resp_y = ragtruth_features(model, tok, reader, a.test, a.max_context_chars, t0)
        if a.test_cache:
            np.savez(a.test_cache, Xt=Xt.astype(np.float16), yt=yt, kt=np.array(kt), resp=resp, ev=ev, resp_y=resp_y)
    n_resp = len(resp_y)

    def report(score):
        sent = round(float(auroc(score, yt.astype(float))), 3)
        per = np.array([score[resp == i].max() if (resp == i).any() else -1e9 for i in range(n_resp)])
        zs = (score - score.mean()) / (score.std() + 1e-9)          # how much ungroundedness a reply holds in all
        tot = np.array([np.clip(zs[resp == i], 0, None).sum() for i in range(n_resp)])
        conflict = (yt == 0) | np.array(["Conflict" in k for k in kt])
        baseless = (yt == 0) | np.array(["Baseless" in k for k in kt])
        return {"sentence": sent, "response": round(float(auroc(per, resp_y.astype(float))), 3),
                "response_sum": round(float(auroc(tot, resp_y.astype(float))), 3),
                "sentence_conflicts_only": round(float(auroc(score[conflict], yt[conflict].astype(float))), 3),
                "sentence_baseless_only": round(float(auroc(score[baseless], yt[baseless].astype(float))), 3)}

    res = {"model": a.model, "layers_read": reader.layers, "synthetic": {"n": len(syn), "kinds": {k: ks.count(k) for k in set(ks)}},
           "ragtruth": {"responses": n_resp, "sentences": len(yt), "hallucinated_sentences": int(yt.sum())}}
    sc = StandardScaler().fit(Xs)
    cut = int(0.8 * len(ys))
    for C in (0.001, 0.01, 0.1):
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xs[:cut]), ys[:cut])
        held = float(auroc(clf.decision_function(sc.transform(Xs[cut:])), ys[cut:].astype(float)))
        clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xs), ys)
        res.setdefault("dense_probe", {})[f"C={C}"] = dict(report(clf.decision_function(sc.transform(Xt))),
                                                            synthetic_heldout=round(held, 3))
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)                # noqa: E731
    clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xs), ys)
    probe = clf.decision_function(sc.transform(Xt))
    if a.export:                                                # what vt/grounding.py loads to serve the probe
        own = clf.decision_function(sc.transform(Xs))            # its scores on the constructions: the scale
        np.savez(a.export, layers=np.array(reader.layers), mean=sc.mean_.astype(np.float32),
                 scale=sc.scale_.astype(np.float32), coef=clf.coef_[0].astype(np.float32),
                 intercept=np.float32(clf.intercept_[0]), score_mean=np.float32(own.mean()),
                 score_std=np.float32(own.std()), model=np.array(a.model),
                 grounded_mean=np.float32(own[ys == 0].mean()), ungrounded_mean=np.float32(own[ys == 1].mean()))
    res["evidence"] = {"mean_shift": report(-ev[:, 0]), "max_disagreement": report(ev[:, 1])}
    res["combined"] = {"probe+disagreement": report(z(probe) + z(ev[:, 1])),
                       "probe+shift": report(z(probe) - z(ev[:, 0])),
                       "probe+shift+disagreement": report(z(probe) - z(ev[:, 0]) + z(ev[:, 1]))}
    res["per_sentence"] = {"y": yt.tolist(), "kind": kt, "response": resp.tolist(), "probe": probe.tolist(),
                           "mean_shift": ev[:, 0].tolist(), "max_disagreement": ev[:, 1].tolist()}
    from recall_neurons import fast_pole_auroc
    pa = fast_pole_auroc(Xs, ys.astype(float))
    order = np.argsort(-np.abs(pa - 0.5))
    n = Xs.shape[1]

    def poles(X, idx):
        j = idx % n
        v = np.maximum(-X[:, j], 0) if idx >= n else np.maximum(X[:, j], 0)
        return v * np.sign(pa[idx] - 0.5) / (np.maximum(-Xs[:, j], 0) if idx >= n else np.maximum(Xs[:, j], 0)).std().clip(1e-6)
    for k in (1, 10, 30, 100):
        res.setdefault("sparse_poles", {})[f"k={k}"] = report(np.mean([poles(Xt, i) for i in order[:k]], 0))
    res["top_poles"] = [{"layer": reader.layers[(i % n) // (n // len(reader.layers))], "neuron": int((i % n) % (n // len(reader.layers))),
                         "pole": "negative" if i >= n else "positive", "synthetic_auroc": round(float(pa[i]), 3)}
                        for i in order[:10]]
    res["seconds"] = round(time.time() - t0)
    print(json.dumps({k: v for k, v in res.items() if k != "per_sentence"}, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
