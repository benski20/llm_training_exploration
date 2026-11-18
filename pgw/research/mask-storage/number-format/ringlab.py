"""Layer-wise PGW number-format lab: per-group ring K, adaptive activation bits, bound statistics.

Runs a Qwen2 model layer by layer (so 7B can be streamed shard by shard from the HF hub with <10 GB
disk), carrying one hidden-state stream per config. Every linear group (qkv, o, gate_up, down) is
replaced by fake-quantized int8-per-row weights x per-token symmetric A-bit activations, matching
PGW arithmetic (benchmarks/quant_quality.py), but with a ring size K chosen *per group*.

Adaptive rule (exact ring arithmetic): per token t, A_t = largest A <= 12 with
    max_j ||w8_j||_2 * ||xq_t(A)||_2 < 2^(K_g - 1)          (Cauchy-Schwarz, guarantees no wrap)
Group max is over all rows of the fused group linear (q,k,v share one call; gate,up share one).

Statistics recorded per (layer, group, token) at A=12: log2 of the CS bound, log2 of the actual
max |yq|, and the activation "flatness" ||xq||_2/qmax.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
torch.set_grad_enabled(False)

GROUPS = {"qkv": ("self_attn", ("q_proj", "k_proj", "v_proj")), "o": ("self_attn", ("o_proj",)),
          "gate_up": ("mlp", ("gate_proj", "up_proj")), "down": ("mlp", ("down_proj",))}


def parse_cfg(s: str) -> dict:
    """fp | w8a12 | w8a8 | ring:qkv=22,o=22,gate_up=22,down=24[,amax=12]"""
    if s == "fp":
        return {"name": s, "mode": "fp"}
    if s.startswith("w8a"):
        return {"name": s, "mode": "fixed", "a": int(s[3:])}
    assert s.startswith("ring:"), s
    d = {"name": s, "mode": "ring", "k": {}, "amax": 12}
    for kv in s[5:].split(","):
        k, v = kv.split("=")
        if k in ("amax", "split"):
            d[k] = int(v)
        else:
            d["k"][k] = int(v)
    return d


class GroupQ:
    """Shared int8 weights for one fused group in one layer."""

    def __init__(self, lins: list[nn.Linear]):
        w = torch.cat([l.weight.detach().float() for l in lins], 0)
        sc = w.abs().amax(-1, keepdim=True) / 127
        sc = torch.where(sc == 0, torch.ones_like(sc), sc)
        w8 = torch.clamp(torch.round(w / sc), -127, 127)
        self.w8t = w8.t().contiguous()  # (n, m) float32 holding ints
        self.sw = sc.reshape(1, -1)
        self.rownorm = w8.norm(dim=-1)
        self.wn = float(self.rownorm.max())
        self.wn_med = float(self.rownorm.median())
        self.splits = [l.weight.shape[0] for l in lins]
        self.bias = [None if l.bias is None else l.bias.detach().float() for l in lins]
        self.n = w.shape[1]


def quant(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True) / qmax
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.clamp(torch.round(x / s), -qmax, qmax), s


class Recorder:
    def __init__(self):
        self.d: dict[str, list] = {}

    def add(self, key, arr):
        self.d.setdefault(key, []).append(arr.astype(np.float32))

    def cat(self):
        return {k: np.concatenate(v) for k, v in self.d.items()}


def group_forward(g: GroupQ, x: torch.Tensor, cfg: dict, gname: str, rec: Recorder | None,
                  li: int) -> list[torch.Tensor]:
    shp = x.shape
    flat = x.reshape(-1, shp[-1]).float()
    ys = None
    if cfg.get("split"):
        # gateway computes the top-k |x| channels of each token itself (plaintext, it holds W);
        # only the remainder is quantized, masked and sent. Costs m*k MACs on the gateway.
        idx = flat.abs().topk(cfg["split"], -1).indices
        xs = flat.gather(1, idx)
        ys = torch.einsum("tk,tkm->tm", xs, g.w8t[idx])
        flat = flat.scatter(1, idx, 0.0)
    if cfg["mode"] == "fixed":
        xq, sx = quant(flat, cfg["a"])
    else:
        K = cfg["k"].get(gname, 99)
        amax = cfg["amax"]
        limit = 2.0 ** (K - 1)
        xq = torch.empty_like(flat)
        sx = torch.empty(flat.shape[0], 1)
        chosen = torch.full((flat.shape[0],), -1, dtype=torch.int64)
        for bits in range(amax, 1, -1):
            q, s = quant(flat, bits)
            fits = (q.norm(dim=-1) * g.wn < limit) & (chosen < 0)
            xq[fits] = q[fits]
            sx[fits] = s[fits]
            chosen[fits] = bits
            if bool((chosen >= 0).all()):
                break
        if rec is not None:
            rec.add(f"abits/{gname}", chosen.numpy())
    yq = xq @ g.w8t
    if rec is not None and cfg.get("stats"):
        q12, _ = quant(flat, 12) if cfg["mode"] == "ring" else (xq, None)
        nrm = q12.norm(dim=-1)
        y12 = yq if cfg["mode"] == "fixed" else q12 @ g.w8t
        rec.add(f"log2cs/{gname}", torch.log2(nrm * g.wn).numpy())
        rec.add(f"log2act/{gname}", torch.log2(y12.abs().amax(-1).clamp(min=1)).numpy())
        rec.add(f"flat/{gname}", (nrm / 2047).numpy())
        rec.add(f"layer/{gname}", np.full(flat.shape[0], li))
    y = yq * sx * g.sw
    if ys is not None:
        y = y + ys * g.sw
    outs = []
    for part, b in zip(torch.split(y, g.splits, dim=-1), g.bias):
        if b is not None:
            part = part + b
        outs.append(part.reshape(*shp[:-1], -1))
    return outs


class Dispatch(nn.Module):
    """Stands in for one linear of a group; computes the whole group once and caches siblings."""

    def __init__(self, holder: dict, gname: str, idx: int):
        super().__init__()
        self.holder, self.gname, self.idx = holder, gname, idx

    def forward(self, x):
        h = self.holder
        key = (self.gname, x.data_ptr())
        if h.get("key") != key:
            h["key"] = key
            h["out"] = group_forward(h["groups"][self.gname], x, h["cfg"], self.gname, h["rec"], h["li"])
        return h["out"][self.idx]


def patch(layer, holder):
    orig = {}
    for gname, (parent, names) in GROUPS.items():
        p = getattr(layer, parent)
        for i, nm in enumerate(names):
            orig[(parent, nm)] = getattr(p, nm)
            setattr(p, nm, Dispatch(holder, gname, i))
    return orig


def unpatch(layer, orig):
    for (parent, nm), mod in orig.items():
        setattr(getattr(layer, parent), nm, mod)


# ---------------------------------------------------------------- weight sources
class Weights:
    def __init__(self, model_id: str, stream: bool):
        from huggingface_hub import hf_hub_download
        self.model_id, self.stream = model_id, stream
        self.dl = hf_hub_download
        try:
            idx = json.load(open(self.dl(model_id, "model.safetensors.index.json")))
            self.map = idx["weight_map"]
        except Exception:
            from safetensors import safe_open
            with safe_open(self.dl(model_id, "model.safetensors"), "pt") as fh:
                self.map = {k: "model.safetensors" for k in fh.keys()}
        self.cur_file, self.cur = None, None

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open
        import collections
        if not hasattr(self, "left"):
            self.left = collections.Counter(self.map.values())
            self.handles = {}
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(self.dl(self.model_id, f), "pt")
        t = self.handles[f].get_tensor(name).float()
        self.left[f] -= 1
        if self.left[f] == 0 and self.stream:
            del self.handles[f]
            gc.collect()
            self._drop(f)
        return t

    def _drop(self, f):
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == self.model_id:
                for rev in repo.revisions:
                    for fi in rev.files:
                        if fi.file_name == f:
                            os.remove(fi.blob_path)
                            print(f"dropped {f}", flush=True)


def load_tokens(tok, split, n):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    return tok(text, return_tensors="pt").input_ids[0][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--stats-config", default=None, help="config whose stream records bound stats")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--stream", action="store_true", help="delete shards after use (7B)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.masking_utils import create_causal_mask

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = load_tokens(tok, "test", args.tokens)
    nseq = len(ids) // args.seq
    ids = ids[: nseq * args.seq].reshape(nseq, args.seq)
    conf = AutoConfig.from_pretrained(args.model)
    conf._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(conf, torch_dtype=torch.float32)
    W = Weights(args.model, args.stream)
    cfgs = [parse_cfg(c) for c in args.configs]
    if args.stats_config:
        for c in cfgs:
            c["stats"] = c["name"] == args.stats_config
    recs = {c["name"]: Recorder() for c in cfgs}

    def materialize(mod, prefix):
        mod.to_empty(device="cpu")
        sd = {k: W.get(prefix + k) for k in mod.state_dict().keys() if prefix + k in W.map}
        missing = mod.load_state_dict(sd, strict=False)
        return missing

    emb = model.model.embed_tokens
    materialize(emb, "model.embed_tokens.")
    rot = model.model.rotary_emb
    rot.to_empty(device="cpu")
    rot.__init__(conf)  # recompute inv_freq buffers
    batches = [ids[i : i + args.batch] for i in range(0, nseq, args.batch)]
    hs = {c["name"]: [emb(b).float() for b in batches] for c in cfgs}
    pos = torch.arange(args.seq)[None]
    pe = rot(hs[cfgs[0]["name"]][0], pos)
    masks = [create_causal_mask(config=conf, inputs_embeds=b_h, attention_mask=None, past_key_values=None,
                                position_ids=pos) for b_h in hs[cfgs[0]["name"]]]
    emb.to("meta")
    gc.collect()
    print(f"setup {time.time()-t0:.0f}s, {nseq} seqs", flush=True)

    meta = {}
    for li, layer in enumerate(model.model.layers):
        materialize(layer, f"model.layers.{li}.")
        groups = {}
        for gname, (parent, names) in GROUPS.items():
            p = getattr(layer, parent)
            groups[gname] = GroupQ([getattr(p, nm) for nm in names])
            meta[f"{li}/{gname}"] = {"wn_max": groups[gname].wn, "wn_med": groups[gname].wn_med,
                                     "n": groups[gname].n, "m": sum(groups[gname].splits)}
        for c in cfgs:
            if c["mode"] == "fp":
                hs[c["name"]] = [layer(h, attention_mask=mk, position_ids=pos, position_embeddings=pe)
                                 for h, mk in zip(hs[c["name"]], masks)]
                continue
            holder = {"groups": groups, "cfg": c, "rec": recs[c["name"]], "li": li}
            orig = patch(layer, holder)
            out = []
            for h, mk in zip(hs[c["name"]], masks):
                holder["key"] = None
                out.append(layer(h, attention_mask=mk, position_ids=pos, position_embeddings=pe))
            hs[c["name"]] = out
            unpatch(layer, orig)
        layer.to("meta")
        del groups
        gc.collect()
        print(f"layer {li} done {time.time()-t0:.0f}s", flush=True)

    norm = model.model.norm
    materialize(norm, "model.norm.")
    head = model.lm_head
    if "lm_head.weight" in W.map:
        head.to_empty(device="cpu")
        head.weight.data = W.get("lm_head.weight")
    else:
        head.to_empty(device="cpu")
        head.weight.data = W.get("model.embed_tokens.weight")
    acc = {c["name"]: {"nll": 0.0, "cnt": 0, "agree": 0, "kl": 0.0} for c in cfgs}
    for bi, bb in enumerate(batches):
        for si in range(bb.shape[0]):
            b = bb[si : si + 1]
            ref = None
            for c in cfgs:
                logits = head(norm(hs[c["name"]][bi][si : si + 1])).double()
                lp = torch.log_softmax(logits[:, :-1], -1)
                del logits
                a = acc[c["name"]]
                a["nll"] -= float(lp.gather(2, b[:, 1:, None]).sum())
                a["cnt"] += b[:, 1:].numel()
                top = lp.argmax(-1)
                if ref is None:
                    ref = (top, lp, lp.exp())
                else:
                    a["agree"] += int((top == ref[0]).sum())
                    a["kl"] += float((ref[2] * (ref[1] - lp)).sum())
                    del lp
            del ref
    results = []
    for ci, c in enumerate(cfgs):
        a = acc[c["name"]]
        row = {"config": c["name"], "ppl": round(math.exp(a["nll"] / a["cnt"]), 4)}
        if ci:
            row["top1_vs_first"] = round(a["agree"] / a["cnt"], 4)
            row["kl_vs_first"] = round(a["kl"] / a["cnt"], 6)
        r = recs[c["name"]].cat()
        for gname in GROUPS:
            if f"abits/{gname}" in r:
                ab = r[f"abits/{gname}"]
                row[f"abits_{gname}"] = {"mean": round(float(ab.mean()), 3),
                                         "frac_lt12": round(float((ab < 12).mean()), 4), "min": int(ab.min())}
        results.append(row)
        print(json.dumps(row), flush=True)
        if c.get("stats"):
            np.savez_compressed(args.out.replace(".json", "_stats.npz"),
                                **{k.replace("/", "__"): v for k, v in r.items()})
    json.dump({"model": args.model, "tokens": int(ids.numel()), "seq": args.seq, "results": results,
               "weights": meta, "seconds": round(time.time() - t0)}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
