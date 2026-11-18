"""Quality of PGW number formats: perplexity and top-1 agreement on wikitext-2.

The masked protocol computes exactly the quantized model (ring arithmetic is exact when K is large
enough), so quality depends only on the quantization, which this script fake-quantizes in float64:

* weights: symmetric per output row, ``wbits`` bits (8 = today's int8 packs, 4 = candidate);
* activations: symmetric per token, ``abits`` bits (12 = K40A12, 8 = K32A8);
* optional SmoothQuant (``alpha``): per-input-channel scales ``s`` move activation outliers into
  the weights (x/s, W*s). Only scales that can be folded into the previous op are used: RMSNorm
  weights for qkv and gate_up, v_proj rows for o_proj (shared across heads of a GQA group), up_proj
  rows for down_proj. Calibrated on wikitext-2 *validation*, evaluated on *test*.
* optional GPTQ-free AWQ-style weight search is not included; see --wbits 4 results for RTN.

Usage:
    python benchmarks/quant_quality.py --model Qwen/Qwen2.5-0.5B-Instruct --configs fp w8a12 w8a8 w8a8_sq w4a8 w4a8_sq
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

GROUPS = {"qkv": ("q_proj", "k_proj", "v_proj"), "o": ("o_proj",), "gate_up": ("gate_proj", "up_proj"),
          "down": ("down_proj",)}


@dataclass(frozen=True)
class Cfg:
    name: str
    wbits: int | None  # None = float weights
    abits: int | None  # None = float activations
    alpha: float | None = None  # SmoothQuant migration strength
    ring: int | None = None  # adaptive mode: per token, the largest A <= abits whose exact bound fits K


def parse_cfg(s: str) -> Cfg:
    if s == "fp":
        return Cfg(s, None, None)
    if s.startswith("k") and "ad" in s:  # e.g. k24ad, k32ad_sq: W8, adaptive A<=12 fitting a K-bit ring
        k = int(s[1:s.index("ad")])
        return Cfg(s, 8, 12, 0.5 if "_sq" in s else None, ring=k)
    base, _, sq = s.partition("_sq")
    w, a = base[1:].split("a")
    alpha = float(sq) if sq not in ("",) else 0.5
    return Cfg(s, int(w), int(a), alpha if "_sq" in s else None)


def ring_bits(wbits: int, abits: int, n: int) -> int:
    """Smallest K with |sum_i w_i x_i| < 2^(K-1) for n terms (exact unmasking)."""
    bound = n * (2 ** (wbits - 1) - 1) * (2 ** (abits - 1) - 1)
    return math.ceil(math.log2(bound + 1)) + 1


def quant_rows(t: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    s = t.abs().amax(dim=-1, keepdim=True) / qmax
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.clamp(torch.round(t / s), -qmax, qmax) * s


STATS: dict[int, dict[str, int]] = {}


class QLinear(nn.Module):
    def __init__(self, lin: nn.Linear, cfg: Cfg, smooth: torch.Tensor | None) -> None:
        super().__init__()
        w = lin.weight.detach().to(torch.float64)
        self.inv = None
        if smooth is not None:
            w = w * smooth[None, :]
            self.inv = (1.0 / smooth)
        # Keep integer weights (int8/int16) plus float64 row scales; widen one layer at a time in
        # forward. A float64 copy of every weight does not fit in memory beyond ~1B parameters.
        if cfg.wbits:
            qmax = 2 ** (cfg.wbits - 1) - 1
            sc = w.abs().amax(-1, keepdim=True) / qmax
            sc = torch.where(sc == 0, torch.ones_like(sc), sc)
            self.wq = torch.clamp(torch.round(w / sc), -qmax, qmax).to(torch.int16 if qmax > 127 else torch.int8)
            self.wscale = sc.reshape(1, -1)
        else:
            self.wq, self.wscale = w.to(torch.float32), None
        self.b = None if lin.bias is None else lin.bias.detach().to(torch.float64)
        self.abits = cfg.abits
        self.ring = cfg.ring
        if cfg.ring:
            # integer weight rows: |w8_j . xq| <= ||w8_j||_2 * ||xq||_2 (Cauchy-Schwarz), exact bound
            ws = w.abs().amax(-1, keepdim=True) / 127
            ws = torch.where(ws == 0, torch.ones_like(ws), ws)
            w8 = torch.clamp(torch.round(w / ws), -127, 127)
            self.wnorm = float(w8.norm(dim=-1).max())
            del w8
            self.stats = STATS.setdefault(id(cfg), {"tokens": 0, "reduced": 0, "min_bits": 99})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.to(torch.float64)
        if self.inv is not None:
            x = x * self.inv
        if self.ring:
            x = self.adaptive(x)
        elif self.abits:
            x = quant_rows(x, self.abits)
        y = x @ self.wq.to(torch.float64).t()
        if self.wscale is not None:
            y = y * self.wscale
        if self.b is not None:
            y = y + self.b
        return y.to(dt)


def _adaptive(self: QLinear, x: torch.Tensor) -> torch.Tensor:
    """Quantize each token with the most bits (<= abits) for which the exact bound fits the ring."""
    assert self.ring is not None and self.abits is not None
    limit = 2.0 ** (self.ring - 1)
    flat = x.reshape(-1, x.shape[-1])
    out = torch.empty_like(flat)
    chosen = torch.full((flat.shape[0],), -1, dtype=torch.int64)
    for bits in range(self.abits, 1, -1):
        qmax = 2 ** (bits - 1) - 1
        s = flat.abs().amax(-1, keepdim=True) / qmax
        s = torch.where(s == 0, torch.ones_like(s), s)
        q = torch.clamp(torch.round(flat / s), -qmax, qmax)
        fits = (q.norm(dim=-1) * self.wnorm < limit) & (chosen < 0)
        out[fits] = (q * s)[fits]
        chosen[fits] = bits
        if bool((chosen >= 0).all()):
            break
    st = self.stats
    st["tokens"] += int(flat.shape[0])
    st["reduced"] += int((chosen < self.abits).sum())
    st["min_bits"] = min(st["min_bits"], int(chosen.min()))
    return out.reshape(x.shape)


QLinear.adaptive = _adaptive  # type: ignore[attr-defined]


def load_tokens(tok: Any, split: str, n: int) -> torch.Tensor:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids[:n]


@torch.no_grad()
def calibrate(model: nn.Module, ids: torch.Tensor, seq: int) -> dict[str, torch.Tensor]:
    """Per-input-channel absmax of the input of every linear we quantize."""
    stats: dict[str, torch.Tensor] = {}
    hooks = []
    for li, layer in enumerate(model.model.layers):
        for mod_name, mod in (("q_proj", layer.self_attn.q_proj), ("o_proj", layer.self_attn.o_proj),
                              ("gate_proj", layer.mlp.gate_proj), ("down_proj", layer.mlp.down_proj)):
            key = f"{li}.{mod_name}"

            def hook(_m: nn.Module, inp: tuple[torch.Tensor, ...], _o: Any, key: str = key) -> None:
                a = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).amax(0).to(torch.float64)
                stats[key] = torch.maximum(stats[key], a) if key in stats else a

            hooks.append(mod.register_forward_hook(hook))
    for i in range(0, len(ids) - seq + 1, seq):
        model(ids[i : i + seq][None])
    for h in hooks:
        h.remove()
    return stats


def smooth_scales(model: nn.Module, stats: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    cfg = model.config
    out: dict[str, torch.Tensor] = {}
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_heads
    for li, layer in enumerate(model.model.layers):
        a, m = layer.self_attn, layer.mlp
        for grp, mods, key in (("qkv", (a.q_proj, a.k_proj, a.v_proj), f"{li}.q_proj"),
                               ("gate_up", (m.gate_proj, m.up_proj), f"{li}.gate_proj"),
                               ("o", (a.o_proj,), f"{li}.o_proj"), ("down", (m.down_proj,), f"{li}.down_proj")):
            wmax = torch.stack([mm.weight.detach().abs().amax(0).to(torch.float64) for mm in mods]).amax(0)
            xmax = stats[key]
            s = (xmax.clamp(min=1e-5) ** alpha) / (wmax.clamp(min=1e-5) ** (1 - alpha))
            if grp == "o":  # o_proj input channel (h, d) can only be folded into v channel (kv(h), d)
                s = s.view(n_kv, n_heads // n_kv, head_dim).amax(1, keepdim=True).expand(-1, n_heads // n_kv, -1)
                s = s.reshape(-1)
            out[f"{li}.{grp}"] = s.clamp(min=1e-5)
    return out


def apply(model: nn.Module, cfg: Cfg, scales: dict[str, torch.Tensor] | None) -> None:
    for li, layer in enumerate(model.model.layers):
        for grp, names in GROUPS.items():
            parent = layer.self_attn if grp in ("qkv", "o") else layer.mlp
            s = scales[f"{li}.{grp}"] if scales else None
            for nm in names:
                setattr(parent, nm, QLinear(getattr(parent, nm), cfg, s))


@torch.no_grad()
def evaluate(model: nn.Module, ids: torch.Tensor, seq: int, ref_top: list[torch.Tensor] | None
             ) -> tuple[float, float | None, list[torch.Tensor]]:
    nll, count, agree, tops = 0.0, 0, 0, []
    for k, i in enumerate(range(0, len(ids) - seq + 1, seq)):
        x = ids[i : i + seq][None]
        logits = model(x).logits[0].to(torch.float64)
        lp = torch.log_softmax(logits[:-1], -1)
        nll -= float(lp.gather(1, x[0, 1:, None]).sum())
        count += seq - 1
        top = logits.argmax(-1)
        tops.append(top)
        if ref_top is not None:
            agree += int((top == ref_top[k]).sum())
    agreement = agree / sum(len(t) for t in tops) if ref_top is not None else None
    return math.exp(nll / count), agreement, tops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--configs", nargs="+", default=["fp", "w8a12", "w8a8", "w8a8_sq", "w4a8", "w4a8_sq"])
    ap.add_argument("--eval-tokens", type=int, default=16384)
    ap.add_argument("--calib-tokens", type=int, default=8192)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model)
    load: Callable[[], nn.Module] = lambda: AutoModelForCausalLM.from_pretrained(  # noqa: E731
        args.model, torch_dtype=torch.float32).eval()
    test = load_tokens(tok, "test", args.eval_tokens)
    calib = load_tokens(tok, "validation", args.calib_tokens)
    base = load()
    stats = calibrate(base, calib, args.seq)
    hidden, inter = base.config.hidden_size, base.config.intermediate_size
    del base

    results: list[dict[str, Any]] = []
    ref_top = None
    for name in args.configs:
        cfg = parse_cfg(name)
        t0 = time.time()
        model = load()
        if cfg.wbits or cfg.abits:
            scales = smooth_scales(model, stats, cfg.alpha) if cfg.alpha is not None else None
            apply(model, cfg, scales)
        ppl, agree, tops = evaluate(model, test, args.seq, ref_top)
        if name == "fp":
            ref_top = tops
        row: dict[str, Any] = {"config": name, "ppl": round(ppl, 4),
                               "top1_agreement_vs_fp": None if agree is None else round(agree, 4),
                               "seconds": round(time.time() - t0, 1)}
        if cfg.ring:
            st = STATS.get(id(cfg), {})
            row["adaptive"] = {"ring_bits": cfg.ring, "token_linear_calls": st.get("tokens"),
                               "fraction_with_fewer_bits": round(st.get("reduced", 0) / max(1, st.get("tokens", 1)), 5),
                               "min_activation_bits": st.get("min_bits")}
        elif cfg.wbits and cfg.abits:
            row["ring_bits_needed"] = {"n=hidden": ring_bits(cfg.wbits, cfg.abits, hidden),
                                       "n=intermediate": ring_bits(cfg.wbits, cfg.abits, inter)}
        results.append(row)
        print(json.dumps(row), flush=True)
        del model
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"model": args.model, "eval_tokens": args.eval_tokens, "seq": args.seq, "results": results},
                      fh, indent=1)


if __name__ == "__main__":
    main()
