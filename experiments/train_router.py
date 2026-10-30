"""Trains the message router's classifier: a sentence embedding (small, CPU) and a softmax layer over four kinds,
from human-written prompts in public datasets, none of them the evaluation sets:

    information  Dolly open_qa + general_qa, and brainstorming that asks how   -> answer from a search, check it
                 (CC-BY-SA)
    math         GSM8K train questions, MATH problems (MIT)                    -> re-compute the arithmetic
    code         MBPP task descriptions (CC-BY), short glaive code questions   -> no fact verdict
                 (Apache-2.0)
    chat         Dolly brainstorming + creative_writing                        -> no fact verdict
    --no-robots  adds HuggingFaceH4/no_robots (CC-BY-NC: research only) to every kind

Writes the layer (vt/router_head.npz: weights, bias, kinds, embedding model name) and its held-out scores on
results/mixed_eval_items.json and results/router_holdout.json, which it never trains on.

    python experiments/train_router.py --embed sentence-transformers/all-MiniLM-L6-v2 --out vt/router_head.npz
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

KINDS = ("information", "math", "code", "chat")      # "text" (pasted text) is decided by its shape: vt.router.pasted
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                 "prealgebra", "precalculus")
HOWTO = re.compile(r"^\s*(how|what are (some )?(good |the best )?ways|what('s| is) the best way)\b", re.I)
DOLLY = {"open_qa": "information", "general_qa": "information", "closed_qa": "text",
         "information_extraction": "text", "summarization": "text", "brainstorming": "chat",
         "creative_writing": "chat"}
NO_ROBOTS = {"Open QA": "information", "Closed QA": "text", "Extract": "text", "Summarize": "text",
             "Coding": "code", "Generation": "chat", "Brainstorm": "chat", "Chat": "chat", "Rewrite": "chat"}


def message(instruction, context=""):
    return f"{instruction}\n\n{context}".strip() if context else instruction


YAHOO_INFO = {0, 2, 3, 5, 6, 9}                    # society, health, reference, sports, finance, politics
YAHOO_CHAT = {8}                                     # family & relationships: personal advice


def stream(name, cfg=None, n=3000, keep=lambda r: True):
    from datasets import load_dataset
    rows = load_dataset(name, cfg, split="train", streaming=True)
    return [r for r, _ in zip((r for r in rows if keep(r)), range(n))]


def extra(sources):
    """Real users' questions in their own words: casual, lowercase, typos."""
    data = []
    if "so" in sources:                              # Stack Overflow titles (CC-BY-SA): programming questions
        data += [(r["title"], "code") for r in stream("pacovaldez/stackoverflow-questions", n=3000)]
    if "eli5" in sources:                            # Reddit ELI5 questions: why / how does
        data += [(r["question"], "information") for r in stream("sentence-transformers/eli5", n=1500)]
    if "yahoo" in sources:                           # Yahoo Answers titles by topic
        rows = stream("community-datasets/yahoo_answers_topics", n=20000)
        data += [(r["question_title"], "information") for r in rows if r["topic"] in YAHOO_INFO][:1500]
        data += [(r["question_title"], "chat") for r in rows if r["topic"] in YAHOO_CHAT][:1500]
    return data


def load(no_robots, per_kind, seed=0, sources=()):
    from datasets import load_dataset
    data = []
    for r in load_dataset("databricks/databricks-dolly-15k")["train"]:
        k = DOLLY.get(r["category"])
        if k == "chat" and r["category"] == "brainstorming" and HOWTO.search(r["instruction"]):
            k = "information"                               # "how can I sleep better?": advice with facts in it
        if k and k != "text":
            data.append((r["instruction"], k))
    data += [(r["question"], "math") for r in load_dataset("openai/gsm8k", "main")["train"]]
    for subject in MATH_SUBJECTS:                           # MATH (MIT): algebra, probability, calculus...
        data += [(r["problem"], "math") for r in load_dataset("EleutherAI/hendrycks_math", subject)["train"]
                 if len(r["problem"]) < 300]
    code = load_dataset("glaiveai/glaive-code-assistant-v3", split="train", streaming=True)  # Apache-2.0
    data += [(r["question"], "code") for r, _ in zip((r for r in code if len(r["question"]) < 400), range(3000))]
    for split in ("train", "validation", "prompt"):
        data += [(r["text"], "code") for r in load_dataset("google-research-datasets/mbpp", "full")[split]]
    data += extra(sources)
    if no_robots:
        for r in load_dataset("HuggingFaceH4/no_robots")["train"]:
            k = NO_ROBOTS.get(r["category"])
            if k and k != "text":
                data.append((r["prompt"], k))
    held = set()
    for path in ("results/mixed_eval_items.json", "results/router_holdout.json"):
        held |= {" ".join(x["q"].lower().split()) for x in json.load(open(path))}
    data = [(q, k) for q, k in data if q.strip() and " ".join(q.lower().split()) not in held]
    rng = random.Random(seed)
    rng.shuffle(data)
    out, seen = [], Counter()
    for q, k in data:                                       # at most per_kind of each: no kind dominates
        if seen[k] < per_kind:
            seen[k] += 1
            out.append((q, k))
    return out


def train(X, y, n_classes, l2=1e-4, steps=1500, lr=0.05):
    """Softmax regression, class-balanced cross-entropy with L2, full-batch Adam (torch)."""
    import torch
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
    counts = torch.bincount(yt, minlength=n_classes).clamp(min=1).float()
    weight = len(y) / (n_classes * counts)
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt, weight=weight) + l2 * (W ** 2).sum()
        loss.backward()
        opt.step()
    return W.detach().numpy(), b.detach().numpy()


def log_loss(X, y, W, b):
    Z = X @ W + b
    Z -= Z.max(1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def no_robots_test():
    """no_robots' 500 test prompts (CC-BY-NC, evaluation only), expected kinds from their human categories."""
    from datasets import load_dataset
    return [{"q": r["prompt"], "expect": NO_ROBOTS[r["category"]], "kind": r["category"]}
            for r in load_dataset("HuggingFaceH4/no_robots")["test"] if r["category"] in NO_ROBOTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--per-kind", type=int, default=2000)
    ap.add_argument("--no-robots", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], choices=["so", "eli5", "yahoo"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--cache", help="save / reuse the training embeddings here (.npz)")
    ap.add_argument("--out", default="vt/router_head.npz")
    ap.add_argument("--report", default="results/router_train.json")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    from vt.router import TextRouter
    data = load(a.no_robots, a.per_kind, sources=a.extra)
    print("training messages:", dict(Counter(k for _, k in data)), flush=True)
    emb = SentenceTransformer(a.embed)
    y = np.array([KINDS.index(k) for _, k in data])
    if a.cache and os.path.exists(a.cache):
        X = np.load(a.cache)["X"]
    else:
        X = emb.encode([q[:2000] for q, _ in data], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        if a.cache:
            np.savez(a.cache, X=X)
    cut = int(0.9 * len(data))
    fits = {}
    for l2 in (1e-5, 1e-4, 1e-3, 1e-2):                     # L2 strength by held-out log loss
        W, b = train(X[:cut], y[:cut], len(KINDS), l2=l2, steps=a.steps)
        fits[l2] = (log_loss(X[cut:], y[cut:], W, b), float(((X[cut:] @ W + b).argmax(1) == y[cut:]).mean()))
        print(f"l2 {l2:g}: validation log loss {fits[l2][0]:.3f}, accuracy {fits[l2][1]:.3f}", flush=True)
    l2 = min(fits, key=lambda k: fits[k][0])
    val = fits[l2][1]
    W, b = train(X, y, len(KINDS), l2=l2, steps=a.steps)    # then on everything
    np.savez(a.out, W=W.astype(np.float32), b=b.astype(np.float32), kinds=np.array(KINDS), embed=np.array(a.embed))
    router = TextRouter(a.out, embedder=emb)
    from experiments.router_eval import summarize
    from vt.router import rule
    report = {"embed": a.embed, "no_robots": a.no_robots, "extra": a.extra, "train": dict(Counter(k for _, k in data)),
              "validation_accuracy": round(val, 3), "l2": l2}
    sets = {p: json.load(open(p)) for p in ("results/mixed_eval_items.json", "results/router_holdout.json")}
    if not a.no_robots:                                     # human-labeled categories, never trained on here
        sets["no_robots_test"] = no_robots_test()
    for path, items in sets.items():
        routed = router.route_batch([[{"role": "user", "content": x["q"]}] for x in items])
        rows = [{"q": x["q"][:200], "expect": x["expect"], "rule": rule(x["q"]), "route": k, "probs": p}
                for x, (k, p) in zip(items, routed)]
        report[os.path.basename(path)] = summarize(rows)
        report[os.path.basename(path)]["wrong"] = [(r["expect"], r["route"], r["q"][:90]) for r in rows
                                                    if r["route"] != r["expect"]]
    print(json.dumps(report, indent=1))
    json.dump(report, open(a.report, "w"), indent=1)


if __name__ == "__main__":
    main()
