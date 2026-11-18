"""Same statistics as nf_stats.py, for a model too large for RAM: stream the safetensors layer by layer
over HTTP range requests (nothing written to disk) and run the decoder layer by layer on one sequence."""
from __future__ import annotations

import argparse
import json
import math
import queue
import struct
import threading

import numpy as np
import requests
import torch

torch.set_num_threads(2)
torch.set_grad_enabled(False)
GROUPS = ("qkv", "o", "gate_up", "down")


class Remote:
    def __init__(self, repo: str) -> None:
        self.base = f"https://huggingface.co/{repo}/resolve/main/"
        self.s = requests.Session()
        idx = self.s.get(self.base + "model.safetensors.index.json").json()
        self.wmap = idx["weight_map"]
        self.headers: dict[str, tuple[int, dict]] = {}

    def header(self, f: str) -> tuple[int, dict]:
        if f not in self.headers:
            r = self.s.get(self.base + f, headers={"Range": "bytes=0-7"})
            n = struct.unpack("<Q", r.content)[0]
            h = json.loads(self.s.get(self.base + f, headers={"Range": f"bytes=8-{7 + n}"}).content)
            self.headers[f] = (8 + n, h)
        return self.headers[f]

    def get(self, name: str) -> torch.Tensor:
        f = self.wmap[name]
        off, h = self.header(f)
        meta = h[name]
        a, b = meta["data_offsets"]
        for _ in range(5):
            try:
                r = self.s.get(self.base + f, headers={"Range": f"bytes={off + a}-{off + b - 1}"}, timeout=600)
                if len(r.content) == b - a:
                    break
            except requests.RequestException:
                pass
        assert meta["dtype"] == "BF16", meta["dtype"]
        t = torch.frombuffer(bytearray(r.content), dtype=torch.bfloat16).reshape(meta["shape"])
        return t.to(torch.float32)


def w8_rows(w: torch.Tensor) -> torch.Tensor:
    w = w.to(torch.float64)
    s = w.abs().amax(1, keepdim=True) / 127
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.clamp(torch.round(w / s), -127, 127)


def stats(x: torch.Tensor, w8: torch.Tensor, qmax: int) -> dict[str, np.ndarray]:
    x = x.reshape(-1, x.shape[-1]).to(torch.float64)
    amax = x.abs().amax(1, keepdim=True)
    sx = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    xs = x / sx
    xq = torch.clamp(torch.round(xs), -qmax, qmax)
    y = xq @ w8.t()
    nrm = w8.norm(dim=1)
    e = y - xs @ w8.t()
    return {"cs": torch.log2(nrm.max() * xq.norm(dim=1)).numpy(), "real": torch.log2(y.abs().amax(1)).numpy(),
            "noise": np.full(x.shape[0], math.log2(float(e.std()))),
            "real_rowcs": torch.log2((y.abs() / (xq.norm(dim=1)[:, None] * nrm[None, :])).amax(1)).numpy(),
            "wmax": float(nrm.max()), "wmed": float(nrm.median())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--abits", type=int, default=12)
    ap.add_argument("--layers", type=int, default=0, help="0 = all")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, "/home/user/llm_training_exploration/pgw/benchmarks")
    from quant_quality import load_tokens  # type: ignore
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

    cfg = AutoConfig.from_pretrained(args.model)
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = load_tokens(tok, "test", args.tokens)
    rem = Remote(args.model)
    nl = args.layers or cfg.num_hidden_layers
    qmax = 2 ** (args.abits - 1) - 1

    def layer_names(i: int) -> list[str]:
        p = f"model.layers.{i}."
        return [p + s for s in ("input_layernorm.weight", "post_attention_layernorm.weight",
                                "self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight",
                                "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                                "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                                "mlp.down_proj.weight")]

    q: queue.Queue = queue.Queue(maxsize=1)

    def prefetch() -> None:
        for i in range(nl):
            q.put((i, {n: rem.get(n) for n in layer_names(i)}))

    emb = rem.get("model.embed_tokens.weight")
    h = emb[ids][None].clone()
    del emb
    threading.Thread(target=prefetch, daemon=True).start()
    rot = Qwen2RotaryEmbedding(cfg)
    pos = torch.arange(h.shape[1])[None]
    pe = rot(h, pos)
    acc: dict[str, dict[str, list]] = {g: {"cs": [], "real": [], "noise": [], "real_rowcs": [], "wmax": [], "wmed": []}
                                       for g in GROUPS}
    for _ in range(nl):
        i, sd = q.get()
        layer = Qwen2DecoderLayer(cfg, i).eval()
        p = f"model.layers.{i}."
        layer.load_state_dict({k[len(p):]: v for k, v in sd.items()}, strict=True)
        w8 = {"qkv": w8_rows(torch.cat([sd[p + f"self_attn.{n}_proj.weight"] for n in "qkv"])),
              "o": w8_rows(sd[p + "self_attn.o_proj.weight"]),
              "gate_up": w8_rows(torch.cat([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]])),
              "down": w8_rows(sd[p + "mlp.down_proj.weight"])}
        mods = {"qkv": layer.self_attn.q_proj, "o": layer.self_attn.o_proj, "gate_up": layer.mlp.gate_proj,
                "down": layer.mlp.down_proj}
        hooks = []
        for g, m in mods.items():
            def hk(_m, inp, _o, g=g):
                s = stats(inp[0].detach(), w8[g], qmax)
                for k, v in s.items():
                    acc[g][k].append(v)
            hooks.append(m.register_forward_hook(hk))
        out = layer(h, attention_mask=None, position_ids=pos, position_embeddings=pe)
        h = out[0] if isinstance(out, tuple) else out
        for hk_ in hooks:
            hk_.remove()
        print(i, {g: (round(float(acc[g]["cs"][-1].max()), 2), round(float(acc[g]["real"][-1].max()), 2),
                      round(float(acc[g]["noise"][-1][0]), 2)) for g in GROUPS}, flush=True)
        del layer, sd, w8
    res: dict = {"model": args.model, "tokens": args.tokens, "layers": nl, "abits": args.abits, "groups": {}}
    for g, a in acc.items():
        cs, real, noise = (np.concatenate(a[k]) for k in ("cs", "real", "noise"))
        qf = lambda v, pp: float(np.quantile(v, pp))  # noqa: E731
        res["groups"][g] = {
            "log2_cs": {"max": float(cs.max()), "p99": qf(cs, .99), "p50": qf(cs, .5)},
            "log2_realized_max": {"max": float(real.max()), "p99.9": qf(real, .999), "p99": qf(real, .99),
                                   "p50": qf(real, .5)},
            "log2_cs_minus_realized": {"min": float((cs - real).min()), "p50": qf(cs - real, .5)},
            "log2_noise_std": {"min": float(noise.min()), "p50": qf(noise, .5), "max": float(noise.max())},
            "log2_max_|y|/rowCS": float(np.concatenate(a["real_rowcs"]).max()),
            "per_layer_log2_cs_max": [round(float(c.max()), 2) for c in a["cs"]],
            "per_layer_log2_real_max": [round(float(c.max()), 2) for c in a["real"]],
            "per_layer_log2_noise": [round(float(c[0]), 2) for c in a["noise"]],
            "max_rownorm": max(a["wmax"]), "median_rownorm": float(np.median(a["wmed"])),
        }
    with open(args.json, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
