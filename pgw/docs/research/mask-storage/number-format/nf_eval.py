"""Exact-integer emulation of PGW linears with (a) a narrower decode width K per group and
(b) truncated stored masks: only bits [t, K) of W8 r are stored, so the gateway decodes
y + delta with delta = 2^(t-1) - (W8 r mod 2^t), uniform in (-2^(t-1), 2^(t-1)].

Config string: "i:K=kq,ko,kg,kd:t=tq,to,tg,td:mode=cs|wrap[:A=12]"
  mode=cs   : per token, largest A <= 12 with max_j||w8_j||*||xq|| + 2^(t-1) < 2^(K-1)  (deterministic, no wrap)
  mode=wrap : always A=12; outputs with |y+delta| >= 2^(K-1) would wrap; counted (the protocol detects them
              with a local integer check and retries); emulation uses the unwrapped value.
Perplexity / top-1 agreement on wikitext-2 test as in pgw benchmarks/quant_quality.py (reused for data).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch
from torch import nn

sys.path.insert(0, "/home/user/llm_training_exploration/pgw/benchmarks")
from quant_quality import evaluate, load_tokens  # type: ignore  # noqa: E402

torch.set_num_threads(2)
GROUPS = {"qkv": ("q_proj", "k_proj", "v_proj"), "o": ("o_proj",), "gate_up": ("gate_proj", "up_proj"),
          "down": ("down_proj",)}
GI = {"qkv": 0, "o": 1, "gate_up": 2, "down": 3}
STATS: dict[str, dict[str, float]] = {}
GEN = torch.Generator().manual_seed(1234)


def parse(s: str) -> dict:
    d = {"K": [64] * 4, "t": [0] * 4, "mode": "cs", "A": 12}
    for part in s.split(":")[1:]:
        k, v = part.split("=")
        if k in ("K", "t"):
            vals = [int(z) for z in v.split(",")]
            d[k] = vals * 4 if len(vals) == 1 else vals
        elif k == "A":
            d["A"] = int(v)
        else:
            d[k] = v
    return d


def w8_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w = w.to(torch.float64)
    s = w.abs().amax(1, keepdim=True) / 127
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.clamp(torch.round(w / s), -127, 127), s[:, 0]


class ILinear(nn.Module):
    def __init__(self, lin: nn.Linear, K: int, t: int, mode: str, A: int, wnorm: float, grp: str) -> None:
        super().__init__()
        w8, self.sw = w8_rows(lin.weight.detach())
        self.w8 = w8.to(torch.int8)  # float64 copies of every layer do not fit RAM for 1.5B
        self.b = None if lin.bias is None else lin.bias.detach().to(torch.float64)
        self.K, self.t, self.mode, self.A, self.wnorm = K, t, mode, A, wnorm
        self.st = STATS.setdefault(grp, {"calls": 0, "reduced": 0, "min_bits": 99, "wrap_calls": 0, "wrap_elems": 0,
                                         "elems": 0})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt, shp = x.dtype, x.shape
        flat = x.reshape(-1, shp[-1]).to(torch.float64)
        limit = 2.0 ** (self.K - 1)
        half_t = 2.0 ** (self.t - 1) if self.t > 0 else 0.0
        T = flat.shape[0]
        xq = torch.empty_like(flat)
        sx = torch.empty(T, 1, dtype=torch.float64)
        chosen = torch.full((T,), -1, dtype=torch.int64)
        amax = flat.abs().amax(-1, keepdim=True)
        bits_range = range(self.A, 1, -1) if self.mode == "cs" else [self.A]
        for bits in bits_range:
            qmax = 2 ** (bits - 1) - 1
            s = amax / qmax
            s = torch.where(s == 0, torch.ones_like(s), s)
            q = torch.clamp(torch.round(flat / s), -qmax, qmax)
            fits = (q.norm(dim=-1) * self.wnorm + half_t < limit) | torch.tensor(self.mode != "cs")
            fits &= chosen < 0
            xq[fits], sx[fits] = q[fits], s[fits]
            chosen[fits] = bits
            if bool((chosen >= 0).all()):
                break
        y = xq @ self.w8.to(torch.float64).t()  # exact integers in float64
        if self.t > 0:
            lo = torch.randint(0, 2 ** self.t, y.shape, generator=GEN, dtype=torch.int64).to(torch.float64)
            y = y + (half_t - lo)
        st = self.st
        st["calls"] += T
        st["elems"] += y.numel()
        st["reduced"] += int((chosen < self.A).sum())
        st["min_bits"] = min(st["min_bits"], int(chosen.min()))
        wr = y.abs() >= limit
        st["wrap_elems"] += int(wr.sum())
        st["wrap_calls"] += int(wr.any(-1).sum())
        out = y * sx * self.sw[None, :]
        if self.b is not None:
            out = out + self.b
        return out.reshape(*shp[:-1], -1).to(dt)


def apply(model: nn.Module, cfg: dict) -> None:
    for layer in model.model.layers:
        for grp, names in GROUPS.items():
            par = layer.self_attn if grp in ("qkv", "o") else layer.mlp
            wn = max(float(w8_rows(getattr(par, n).weight.detach())[0].norm(dim=1).max()) for n in names)
            i = GI[grp]
            for n in names:
                setattr(par, n, ILinear(getattr(par, n), cfg["K"][i], cfg["t"][i], cfg["mode"], cfg["A"], wn, grp))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    load = lambda: AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()  # noqa: E731
    test = load_tokens(tok, "test", args.eval_tokens)
    results = []
    ref_top = None
    for name in args.configs:
        STATS.clear()
        t0 = time.time()
        model = load()
        if name != "fp":
            apply(model, parse(name))
        ppl, agree, tops = evaluate(model, test, args.seq, ref_top)
        if name == "fp":
            ref_top = tops
        row = {"config": name, "ppl": round(ppl, 4), "top1_agreement_vs_fp": None if agree is None else round(agree, 4),
               "seconds": round(time.time() - t0, 1)}
        if name != "fp":
            row["stats"] = {g: {"frac_tokens_fewer_bits": round(s["reduced"] / max(1, s["calls"]), 6),
                                "min_bits": s["min_bits"], "wrap_call_rate": s["wrap_calls"] / max(1, s["calls"]),
                                "wrap_elem_rate": s["wrap_elems"] / max(1, s["elems"])} for g, s in STATS.items()}
        results.append(row)
        print(json.dumps(row), flush=True)
        del model
        with open(args.json, "w") as fh:
            json.dump({"model": args.model, "eval_tokens": args.eval_tokens, "results": results}, fh, indent=1)


if __name__ == "__main__":
    main()
