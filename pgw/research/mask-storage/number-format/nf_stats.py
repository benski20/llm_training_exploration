"""Per linear group: CS bound vs realized output magnitude vs activation-rounding noise, in PGW integer units.

For every layer/group we take the group's input activations on wikitext-2 test tokens, quantize them per
token to A=12 bits exactly like pgw.core.ring.quantize_activations, take int8 per-row weights like
quantize_weight_rows, and compute the exact integer product y = xq @ w8^T (float64 is exact here).
Reported per group (log2 units): CS bound max_j||w8_j||*||xq||, realized max_j|y_j|, and the std of the
activation-rounding error  y - w8 @ (x/sx)  (the noise floor that already exists at w8a12).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

torch.set_num_threads(2)
GROUPS = {"qkv": ("q_proj", "k_proj", "v_proj"), "o": ("o_proj",), "gate_up": ("gate_proj", "up_proj"),
          "down": ("down_proj",)}


def w8_rows(w: torch.Tensor) -> torch.Tensor:
    w = w.to(torch.float64)
    s = w.abs().amax(1, keepdim=True) / 127
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.clamp(torch.round(w / s), -127, 127)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--abits", type=int, default=12)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, "/home/user/llm_training_exploration/pgw/benchmarks")
    from quant_quality import load_tokens  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    ids = load_tokens(tok, "test", args.tokens)
    qmax = 2 ** (args.abits - 1) - 1
    acc: dict[str, dict[str, list]] = {g: {"cs": [], "real": [], "noise": [], "real_rowcs": []} for g in GROUPS}
    wcache: dict[str, torch.Tensor] = {}
    wnorm_stats: dict[str, list] = {g: [] for g in GROUPS}
    layers = model.model.layers
    for li, layer in enumerate(layers):
        for g, names in GROUPS.items():
            par = layer.self_attn if g in ("qkv", "o") else layer.mlp
            w8 = torch.cat([w8_rows(getattr(par, n).weight.detach()) for n in names], 0)
            wcache[f"{li}.{g}"] = w8.to(torch.int8)
            nrm = w8.norm(dim=1)
            wnorm_stats[g].append({"layer": li, "max_rownorm": float(nrm.max()), "median_rownorm": float(nrm.median()),
                                   "n": w8.shape[1], "m": w8.shape[0]})
    hooks = []

    def mk(li: int, g: str):
        def hook(_m, inp, _o):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float64)
            amax = x.abs().amax(1, keepdim=True)
            sx = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
            xs = x / sx
            xq = torch.clamp(torch.round(xs), -qmax, qmax)
            w8 = wcache[f"{li}.{g}"].to(torch.float64)
            y = xq @ w8.t()
            nrm = w8.norm(dim=1)
            cs = nrm.max() * xq.norm(dim=1)
            rowcs = (xq.norm(dim=1)[:, None] * nrm[None, :])  # per-row CS bound
            e = y - xs @ w8.t()
            a = acc[g]
            a["cs"].append(torch.log2(cs).numpy())
            a["real"].append(torch.log2(y.abs().amax(1)).numpy())
            a["noise"].append(np.full(x.shape[0], math.log2(float(e.std()))))
            a["real_rowcs"].append(torch.log2((y.abs() / rowcs).amax(1)).numpy())
        return hook

    for li, layer in enumerate(layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(mk(li, "qkv")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(mk(li, "o")))
        hooks.append(layer.mlp.gate_proj.register_forward_hook(mk(li, "gate_up")))
        hooks.append(layer.mlp.down_proj.register_forward_hook(mk(li, "down")))
    for i in range(0, len(ids) - args.seq + 1, args.seq):
        model(ids[i : i + args.seq][None])
    out: dict = {"model": args.model, "tokens": args.tokens, "abits": args.abits, "groups": {}}
    for g, a in acc.items():
        nl = len(layers)
        cs = np.concatenate(a["cs"])
        real = np.concatenate(a["real"])
        noise = np.concatenate(a["noise"])
        gap = cs - real
        q = lambda v, p: float(np.quantile(v, p))  # noqa: E731
        per_layer_cs_max = [float(np.max(c)) for c in a["cs"][:nl]] if False else None
        # per-layer maxima (calls are appended layer by layer per forward; group by layer index)
        csl = np.array([np.max(np.concatenate(a["cs"][li::nl])) for li in range(nl)])
        rel = np.array([np.max(np.concatenate(a["real"][li::nl])) for li in range(nl)])
        out["groups"][g] = {
            "log2_cs": {"max": float(cs.max()), "p99.99": q(cs, .9999), "p99": q(cs, .99), "p50": q(cs, .5)},
            "log2_realized_max": {"max": float(real.max()), "p99.99": q(real, .9999), "p99.9": q(real, .999),
                                   "p99": q(real, .99), "p50": q(real, .5)},
            "log2_cs_minus_realized": {"min": float(gap.min()), "p50": q(gap, .5)},
            "log2_noise_std": {"min": float(noise.min()), "p50": q(noise, .5), "max": float(noise.max())},
            "log2_max_|y|/rowCS": float(np.concatenate(a["real_rowcs"]).max()),
            "per_layer_log2_cs_max": [round(float(v), 2) for v in csl],
            "per_layer_log2_real_max": [round(float(v), 2) for v in rel],
            "weight_rownorm": wnorm_stats[g],
        }
        print(g, json.dumps({k: v for k, v in out["groups"][g].items() if k.startswith("log2")}), flush=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
