"""Are Glassbox-2.3's built-in features more interpretable than what a standard model offers?

Compares, on the same tokens:
  * Glassbox-2.3's exactly sparse features (32 of 1,024 active per token, no reconstruction error);
  * the matched GPT's MLP neurons (ReLU, 1,536 per layer);
  * a top-k sparse autoencoder trained post hoc on the GPT's residual stream after the same
    block (1,024 latents, k = 32): the standard tool for reading a trained model.
Per unit: how often it fires, how concentrated its top-20 activating tokens are on one token
(token purity), and the share of its activation mass on its most frequent token. Purity is
an automatic, crude proxy for monosemanticity, not a human judgement.

    python experiments/feature_interpretability.py --layer 5
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from chat_eval_large import MODELS  # noqa: E402
from selective_answering import DATA  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def unit_stats(acts, toks, top=20, min_fires=20):
    """acts (N, U) non-negative, toks (N,): per-unit density, top-k token purity, mass share."""
    fires = (acts > 0).sum(0)
    keep = np.nonzero(fires >= min_fires)[0]
    dens, pur, mass = [], [], []
    for u in keep:
        a = acts[:, u]
        idx = np.argpartition(-a, top)[:top]
        c = Counter(toks[idx].tolist())
        tok, cnt = c.most_common(1)[0]
        pur.append(cnt / top)
        on = a > 0
        mass.append(float(a[on][toks[on] == tok].sum() / a[on].sum()))
        dens.append(float(on.mean()))
    pur, mass, dens = map(np.array, (pur, mass, dens))
    return {"units": int(acts.shape[1]), "units_firing_20+": int(len(keep)),
            "median_density": float(np.median(dens)) if len(keep) else None,
            "median_top20_purity": float(np.median(pur)) if len(keep) else None,
            "share_purity_at_least_0.8": float((pur >= 0.8).mean()) if len(keep) else None,
            "median_mass_on_top_token": float(np.median(mass)) if len(keep) else None}


def train_sae(x, n_latents=1024, k=32, steps=3000, bs=4096, seed=0):
    """Top-k sparse autoencoder on activations x (N, d); returns latent codes and the
    fraction of variance it leaves unexplained."""
    g = torch.Generator().manual_seed(seed)
    mu, sd = x.mean(0), x.std() + 1e-6
    xn = (x - mu) / sd
    d = x.shape[1]
    enc = torch.nn.Linear(d, n_latents)
    dec = torch.nn.Linear(n_latents, d, bias=False)
    with torch.no_grad():
        dec.weight.copy_(torch.randn(d, n_latents, generator=g) / d ** 0.5)
        enc.weight.copy_(dec.weight.T)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)

    def code(z):
        pre = F.relu(enc(z))
        top = pre.topk(k, -1)
        return torch.zeros_like(pre).scatter_(-1, top.indices, top.values)
    for _ in range(steps):
        b = xn[torch.randint(len(xn), (bs,), generator=g)]
        loss = F.mse_loss(dec(code(b)), b)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        c = torch.cat([code(xn[i:i + 8192]) for i in range(0, len(xn), 8192)])
        rec = torch.cat([dec(c[i:i + 8192]) for i in range(0, len(c), 8192)])
        fvu = float(((rec - xn) ** 2).sum() / ((xn - xn.mean(0)) ** 2).sum())
    return c.numpy(), fvu


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--windows", type=int, default=400)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/feature_interpretability.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    V = tok.get_vocab_size()
    stream = torch.load(f"{DATA}/stream.pt")
    g = torch.Generator().manual_seed(7)
    ix = torch.randint(len(stream) - 257, (a.windows,), generator=g)
    X = torch.stack([stream[i:i + 256] for i in ix.tolist()])
    toks = X.flatten().numpy()

    name, make, ckpt = MODELS["glassbox2_3"]
    gb = make(V); gb.m.load_state_dict(torch.load(ckpt)["model"]); gb.m.eval()
    feats = []
    for i in range(0, len(X), 16):
        gb.m(X[i:i + 16])
        feats.append(gb.m.trace["feat"][a.layer]["acts"].flatten(0, 1).clamp_min(0))
    feats = torch.cat(feats).numpy()

    _, make_b, ckpt_b = MODELS["baseline"]
    gpt = make_b(V); gpt.m.load_state_dict(torch.load(ckpt_b)["model"]); gpt.m.eval()
    neurons, resid = [], []
    blk = gpt.m.blocks[a.layer]
    for i in range(0, len(X), 16):
        gpt._hs = []
        gpt.m(X[i:i + 16])
        neurons.append(F.relu(blk.mlp.stats[0]).flatten(0, 1))
        resid.append(gpt._hs[a.layer].flatten(0, 1))
    neurons = torch.cat(neurons).numpy()
    resid = torch.cat(resid)
    with torch.enable_grad():
        sae_codes, fvu = train_sae(resid)

    res = {"layer": a.layer, "tokens": int(len(toks)),
           "glassbox_features": {**unit_stats(feats, toks), "reconstruction_error": 0.0},
           "gpt_mlp_neurons": unit_stats(neurons, toks),
           "gpt_sae_latents": {**unit_stats(sae_codes, toks), "fraction_variance_unexplained": fvu}}
    print(json.dumps(res, indent=1))
    allres = json.load(open(a.out)) if os.path.exists(a.out) else {}
    allres[f"layer{a.layer}"] = res
    json.dump(allres, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
