"""Causal unit attribution: which sentence / step makes the reader's own verdict say "wrong"?

The reader (Qwen3-1.7B, the shipped 1.7B head's reader) reads [task, reply, judge question] and gives its verdict
j = log P(No) - log P(Yes) (larger = more likely wrong). Two training-free ways to ask the reader's internals which
unit drives that verdict:

    ablation   remove unit k from the reply and re-read: delta_k = j(full) - j(without k). A unit whose removal makes the
               reply look right is the reader's causal reason for "wrong". The task prefix is KV-cached, so each
               variant re-reads only the reply and the judge question.
    gradient   input x gradient of j on the token embeddings, summed per unit (one backward pass).

Compared with the shipped unit probe (heads/omni_qwen3_1p7b.npz; LEAKY here: it was trained on the whole corpus) and
token surprisal, on the corpus domains with unit labels: ProcessBench (label = the first wrong step) and FELM
(label = each wrong segment). Two questions, rules stated before running:

  L  localisation, wrong replies: is the top unit a labelled wrong one (top-1, top-2) vs chance (#wrong / #units)?
  R  reply level: does max_k delta_k (or its share of the total) add to the verdict j alone? Mean per-domain AUROC;
     j + z(max delta) per domain (label-free) must beat j by a paired bootstrap CI above 0 to count.

    python experiments/novel_causal_units.py --per-domain 12 --out results/omni/novel_causal_units.json
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from vt.omni import OmniVerifier, encode, make_item  # noqa: E402

DOMAINS = ["pb_gsm8k", "pb_math", "pb_olympiad", "pb_omnimath", "felm_math", "felm_reasoning", "felm_science",
           "felm_wk", "felm_writing"]


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    p, n = s[y == 1], s[y == 0]
    if not len(p) or not len(n):
        return float("nan")
    return float(((p[:, None] > n[None]) + 0.5 * (p[:, None] == n[None])).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/user/omni/corpus.jsonl")
    ap.add_argument("--head", default="heads/omni_qwen3_1p7b.npz")
    ap.add_argument("--per-domain", type=int, default=12, help="wrong and correct replies each, per domain")
    ap.add_argument("--max-chars", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.manual_seed(0)
    v = OmniVerifier.load(a.head).attach_reader(threads=a.threads)
    rd, model, tok = v.reader, v.reader.model, v.reader.tok
    yes, no = torch.tensor(rd.yes), torch.tensor(rd.no)
    rows = [json.loads(l) for l in open(a.corpus)]
    rng = random.Random(0)
    pick = []
    for d in DOMAINS:
        x = [r for r in rows if r["domain"] == d and len(r["response"]) <= a.max_chars and len(r["units"]) >= 2]
        w = [r for r in x if r["label"] == 1 and 1 in r["unit_labels"]]
        c = [r for r in x if r["label"] == 0]
        pick += rng.sample(w, min(a.per_domain, len(w))) + rng.sample(c, min(a.per_domain, len(c)))
    print("replies", len(pick), flush=True)

    def verdict(head_ids, cache, reply):
        """j for one reply variant, reusing the cached task prefix."""
        rid = tok(reply, add_special_tokens=False)["input_ids"] + rd.suffix
        with torch.no_grad():
            o = model(input_ids=torch.tensor([rid]), past_key_values=copy.deepcopy(cache), use_cache=True,
                      logits_to_keep=1)
        lp = o.logits[0, -1].float().log_softmax(-1)
        return float(torch.logsumexp(lp[no], 0) - torch.logsumexp(lp[yes], 0))

    out, t0 = [], time.time()
    for k, r in enumerate(pick):
        it = make_item(r["response"], messages=r["messages"]) if False else \
            {"messages": r["messages"], "response": r["response"], "units": [tuple(u) for u in r["units"]],
             "doc": r.get("doc") or "", "question": r["messages"][-1]["content"]}
        ids, r0, roff = encode(tok, it, rd.max_tokens, rd.suffix)
        head_ids = ids[:r0]
        with torch.no_grad():
            pre = model(input_ids=torch.tensor([head_ids]), use_cache=True)
        cache = pre.past_key_values
        resp, units = r["response"], it["units"]
        j0 = verdict(head_ids, cache, resp)
        delta = []
        for (s, e) in units:
            delta.append(j0 - verdict(head_ids, cache, (resp[:s] + resp[e:]).strip()))
        # input x gradient of j on the embeddings of the full sequence
        emb = model.get_input_embeddings()(torch.tensor([ids])).detach().float().requires_grad_(True)
        o = model(inputs_embeds=emb.to(model.dtype), logits_to_keep=1)
        lp = o.logits[0, -1].float().log_softmax(-1)
        jj = torch.logsumexp(lp[no], 0) - torch.logsumexp(lp[yes], 0)
        jj.backward()
        tok_attr = (emb.grad[0] * emb[0]).sum(-1).detach().numpy()
        n = len(ids) - r0 - len(rd.suffix)
        starts = np.array([q[0] for q in roff[:n]])
        ends = np.array([q[1] for q in roff[:n]])
        grad_u = [float(tok_attr[r0:r0 + n][(starts < e) & (ends > s)].sum()) for (s, e) in units]
        model.zero_grad(set_to_none=True)
        # the shipped unit probe and surprisal (one more forward pass, features as served)
        f = rd.features(it)
        U = v.unit_matrix(f)
        uidx = [u["index"] for u in f["units"]]
        surpr = {u["index"]: float(-u["sc"][0]) for u in f["units"]}
        out.append({"id": r["id"], "domain": r["domain"], "label": r["label"], "unit_labels": r["unit_labels"],
                    "j0": j0, "j_reader": -f["j_logit"], "delta": delta, "grad": grad_u, "U": U.tolist(), "uidx": uidx,
                    "surprisal": [surpr.get(i, 0.0) for i in range(len(units))]})
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(pick)} {(time.time() - t0) / (k + 1):.1f} s per reply", flush=True)
            json.dump(out, open(a.out + ".partial", "w"))

    # unit probe scores with a label-free per-domain profile (mean / sd of the domain's unit features)
    for d in DOMAINS:
        sel = [o for o in out if o["domain"] == d and o["U"]]
        if not sel:
            continue
        allU = np.concatenate([np.array(o["U"]) for o in sel])
        mu, sd = allU.mean(0), allU.std(0) + 1e-6
        for o in sel:
            sc = ((np.array(o["U"]) - mu) / sd) @ v.art["uw"] + float(v.art["ub"])
            full = np.full(len(o["delta"]), -1e9)
            for i, s in zip(o["uidx"], sc):
                full[i] = s
            o["probe"] = full.tolist()
        for o in sel:
            del o["U"]

    methods = ["delta", "grad", "probe", "surprisal"]
    loc = {m: {"top1": [], "top2": []} for m in methods}
    chance = []
    for o in out:
        if o["label"] != 1 or "probe" not in o:
            continue
        lab = np.array(o["unit_labels"])
        chance.append(float((lab == 1).sum() / len(lab)))
        for m in methods:
            order = np.argsort(-np.array(o[m]))
            loc[m]["top1"].append(float(lab[order[0]] == 1))
            loc[m]["top2"].append(float(any(lab[order[:2]] == 1)))
    res = {"n_replies": len(out), "localisation": {m: {k: float(np.mean(v_)) for k, v_ in d_.items()}
                                                   for m, d_ in loc.items()},
           "chance_top1": float(np.mean(chance)), "per_domain": {}}

    def z(x):
        x = np.asarray(x, float)
        return (x - x.mean()) / (x.std() + 1e-9)
    gains, base = [], []
    for d in DOMAINS:
        sel = [o for o in out if o["domain"] == d]
        if len({o["label"] for o in sel}) < 2:
            continue
        y = np.array([o["label"] for o in sel])
        j0 = np.array([o["j0"] for o in sel])
        md = np.array([max(o["delta"]) for o in sel])
        share = np.array([max(o["delta"]) / (np.abs(o["delta"]).sum() + 1e-6) for o in sel])
        mg = np.array([max(o["grad"]) for o in sel])
        res["per_domain"][d] = {"verdict": auroc(j0, y), "max_delta": auroc(md, y), "delta_share": auroc(share, y),
                                "max_grad": auroc(mg, y), "verdict+max_delta": auroc(z(j0) + z(md), y)}
        base.append((y, j0, md))
    res["mean"] = {k: float(np.nanmean([p[k] for p in res["per_domain"].values()]))
                   for k in ("verdict", "max_delta", "delta_share", "max_grad", "verdict+max_delta")}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(1000):
        dd = []
        for y, j0, md in base:
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) < 2:
                continue
            dd.append(auroc(z(j0[i]) + z(md[i]), y[i]) - auroc(j0[i], y[i]))
        diffs.append(np.mean(dd))
    res["gain_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    json.dump({"summary": res, "replies": out}, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
