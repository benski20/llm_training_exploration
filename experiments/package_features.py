"""Token purity of a served Glassbox package's built-in features (the Kaggle-trained students),
on local text in the package's own tokenizer; same measure as feature_interpretability.py.

    python experiments/package_features.py results/kaggle/glassbox-distill-g24/glassbox_package
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from feature_interpretability import unit_stats  # noqa: E402
from vt.chat_serve import load_package  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--tokens", type=int, default=60000)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/package_features.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    m, tok, man = load_package(a.package)
    al = json.load(open(os.path.join(ROOT, "data", "alpaca", "alpaca_data.json")))
    wiki = [ln.strip() for ln in open(os.path.join(ROOT, "data", "wikitext-2", "valid.txt")) if len(ln.strip()) > 80]
    rng = random.Random(0)
    docs = [f"User: {x['instruction']}\nAssistant: {x['output']}" for x in rng.sample(al, 2000)] + wiki
    rng.shuffle(docs)
    ids = []
    for d in docs:
        ids += tok.encode(d).ids + [0]
        if len(ids) >= a.tokens:
            break
    X = torch.tensor(ids[:a.tokens // a.block * a.block]).view(-1, a.block)
    L = m.config.n_layer
    layers = sorted({L // 2, L - 1})
    acts = {i: [] for i in layers}
    for i in range(0, len(X), 8):
        m(X[i:i + 8])
        for j in layers:
            acts[j].append(m.trace["feat"][j]["acts"].flatten(0, 1).clamp_min(0).float())
    toks = X.flatten().numpy()
    res = {"package": a.package, "format": man.get("format"), "tokens": int(len(toks)),
           "feat_recon": bool(getattr(m.config, "feat_recon", False))}
    for j in layers:
        res[f"layer{j}"] = unit_stats(torch.cat(acts[j]).numpy(), toks)
    print(json.dumps(res, indent=1))
    allres = json.load(open(a.out)) if os.path.exists(a.out) else {}
    allres[a.package] = res
    json.dump(allres, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
