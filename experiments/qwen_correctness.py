"""Does exact attribution on an unmodified pretrained model tell when its answer is correct?

Qwen2.5-0.5B-Instruct answers TriviaQA questions (the same 1,000 questions, prompt and fit/test
split as kaggle/qwen_glassbox/run.py). For the first answer token, with the model's decisions
(RMS scales, attention patterns, SiLU gate values) fixed, its logit is decomposed exactly:

  by input token   how much of the score comes from the question's own tokens versus the chat
                   template and biases, how concentrated it is (`vt.qwen_glassbox.explain`)
  by layer         how much each attention and MLP layer wrote (`layer_contributions`)

Detectors (fitted on one half, scored on the other; AUROC and error among the most confident
50%): softmax, NLL, logit margin; the post-hoc hidden-state probe (the bar: 0.759 on Kaggle);
probes on the attribution statistics, on the per-layer contributions, and combinations.

    python experiments/qwen_correctness.py --n 1000
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "kaggle", "qwen_glassbox"))
import run as R  # noqa: E402
from vt.qwen_glassbox import explain, layer_contributions  # noqa: E402


def question_groups(tok, prompt, q):
    """A component name per prompt position: each question token on its own, the rest 'template'."""
    enc = tok(prompt, return_offsets_mapping=True)
    a = prompt.rindex(q)
    b = a + len(q)
    return [f"q@{j}" if s < b and e > a else "template" for j, (s, e) in enumerate(enc["offset_mapping"])]


def attribution_stats(parts):
    pos = {n: v for n, v in parts.items() if v > 0}
    tot = sum(pos.values()) or 1.0
    q = np.array([v for n, v in pos.items() if n.startswith("q@")]) / tot
    p = q / (q.sum() or 1.0)
    signed_q = sum(v for n, v in parts.items() if n.startswith("q@"))
    return np.array([q.sum(), q.max(initial=0.0), pos.get("template", 0.0) / tot,
                     sum(v for n, v in pos.items() if n.startswith("bias@")) / tot,
                     float(-(p * np.log(p + 1e-12)).sum()), signed_q, parts.get("template", 0.0),
                     sum(parts.values())], dtype=np.float32)


ATTR_NAMES = ["question_share", "max_question_token_share", "template_share", "bias_share",
              "question_entropy", "question_signed", "template_signed", "logit"]


@torch.no_grad()
def collect(model, tok, items, log_every=25):
    rows, t0 = [], time.time()
    for i, it in enumerate(items):
        prompt = R.chat(tok, it["q"])
        ids = tok(prompt, return_tensors="pt").input_ids
        gen = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        ans_ids = gen[0, ids.shape[1]:]
        ans = tok.decode(ans_ids, skip_special_tokens=True).strip()
        right = any(R.norm(al) and R.norm(al) in R.norm(ans) for al in it["aliases"])
        out = model(ids, output_hidden_states=True)
        logits = out.logits[0, -1].float()
        lp = F.log_softmax(logits, -1)
        t = ids.shape[1] - 1
        first = int(ans_ids[0]) if len(ans_ids) else int(lp.argmax())
        top2 = logits.topk(2).values
        groups = question_groups(tok, prompt, it["q"])
        parts, logit = explain(model, ids, t, first, groups)
        lparts, _ = layer_contributions(model, ids, t, first)
        rows.append({"q": it["q"], "answer": ans, "right": int(right), "msp": float(lp.max().exp()),
                     "neg_nll": float(lp[first]), "margin": float(top2[0] - top2[1]),
                     "h": [h[0, -1].float().numpy() for h in out.hidden_states[1:]],
                     "attr": attribution_stats(parts), "layers": np.array(list(lparts.values()), np.float32),
                     "exact_err": abs(sum(parts.values()) - logit),
                     "top_question_tokens": [(tok.decode([int(ids[0, int(n[2:])])]), round(v, 3)) for n, v in
                                             sorted(((n, v) for n, v in parts.items() if n.startswith("q@")),
                                                    key=lambda x: -x[1])[:3]]})
        if (i + 1) % log_every == 0:
            R.log(phase="collect", done=i + 1, acc=float(np.mean([r["right"] for r in rows])),
                  s=round(time.time() - t0))
    return rows


def detectors(rows):
    idx = np.random.default_rng(0).permutation(len(rows))
    fit, test = [rows[i] for i in idx[: len(idx) // 2]], [rows[i] for i in idx[len(idx) // 2:]]
    yf, yt = np.array([r["right"] for r in fit]), np.array([r["right"] for r in test])
    out = {"n": len(rows), "accuracy": float(np.mean([r["right"] for r in rows])), "test_n": len(test),
           "attribution_max_abs_error": float(max(r["exact_err"] for r in rows))}

    def score(name, s, **extra):
        s = np.asarray(s, float)
        keep = np.argsort(-s)
        out[name] = {"auroc": R.auroc(s, yt), "error_at_50pct": float(1 - yt[keep[: len(yt) // 2]].mean()),
                     "error_at_25pct": float(1 - yt[keep[: len(yt) // 4]].mean()), **extra}

    def probe(name, f):
        pr, w, l2 = R.fit_probe_cv(np.stack([f(r) for r in fit]), yf, "cpu")
        score(name, pr(np.stack([f(r) for r in test])), l2=l2)
        return w

    score("softmax", [r["msp"] for r in test])
    score("neg_nll", [r["neg_nll"] for r in test])
    score("margin", [r["margin"] for r in test])
    best, h = None, len(fit) // 2
    for L in range(len(fit[0]["h"])):
        Xf = np.stack([r["h"][L] for r in fit])
        pr, _ = R.fit_probe(Xf[:h], yf[:h], "cpu")
        s = R.auroc(pr(Xf[h:]), yf[h:])
        if s is not None and (best is None or s > best[0]):
            best = (s, L)
    L = best[1] if best else len(fit[0]["h"]) - 1
    probe("posthoc_probe", lambda r: r["h"][L])
    out["posthoc_probe"]["layer"] = L
    for j, name in enumerate(ATTR_NAMES):                      # each attribution statistic alone
        out.setdefault("single_attr_auroc", {})[name] = R.auroc([r["attr"][j] for r in test], yt)
    w_attr = probe("attribution_stats", lambda r: r["attr"])
    w_lay = probe("layer_contributions", lambda r: r["layers"])
    probe("output_plus_attribution", lambda r: np.concatenate([[r["msp"], r["neg_nll"], r["margin"]],
                                                              r["attr"], r["layers"]]))
    probe("probe_plus_attribution", lambda r: np.concatenate([r["h"][L], [r["msp"], r["margin"]],
                                                             r["attr"], r["layers"]]))
    out["attribution_weights"] = dict(zip(ATTR_NAMES, np.round(w_attr, 4).tolist()))
    names = ["embed"] + [f"{k}@{i}" for i in range(len(fit[0]["h"])) for k in ("attn", "mlp")]
    order = np.argsort(-np.abs(w_lay))[:8]
    out["layer_weights_top"] = {names[i]: round(float(w_lay[i]), 4) for i in order}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/qwen_correctness.json")
    ap.add_argument("--rows", default=None, help="also pickle the per-question rows here (large)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    rows = collect(model, tok, R.load_trivia(a.n, False))
    if a.rows:
        pickle.dump(rows, open(a.rows, "wb"))
    res = detectors(rows)
    right = [r for r in rows if r["right"]][:6]
    wrong = [r for r in rows if not r["right"]][:6]
    res["examples"] = [{k: r[k] for k in ("q", "answer", "right", "msp", "top_question_tokens")} |
                       {"question_share": round(float(r["attr"][0]), 3)} for r in right + wrong]
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
