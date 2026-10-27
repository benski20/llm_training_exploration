"""Glassbox-2: a transformer whose standard interpretability analyses are exact by construction.

What post-hoc interpretability approximates on a standard LLM, this model gives exactly:

* Features are the neurons. Each MLP is a wide sparse feature layer: exactly k of
  n_features fire per token (top-k of ReLU), the rest are exactly 0. No sparse
  autoencoder is needed after training, and nothing is lost to reconstruction error.
* Exact attribution. Every part is linear once its discrete decisions are fixed:
  RMS norm (scale per position), softmax attention patterns, convex mixing weights,
  feature gates and the active feature set. Given those decisions (all read off the
  forward pass), each output logit is exactly a sum of contributions from each input
  token, each memory read and each bias. `explain` returns that decomposition; the
  contributions add up to the logit to floating-point precision. Attribution graphs
  on standard models approximate this with replacement models and error terms.
* Knowledge in a logged, hard-gated memory layer (vt.glassbox.PKMemory): a gate of
  exactly 0 proves a token used no stored knowledge, and during training each memory
  slot records the training document that wrote it most strongly, so a memory read
  can be traced to training text.
* A self-knowledge variable k per token sets the sharpness of the next-token
  distribution and is trained on the model's own correctness.

It is an ordinary generative model at inference: tokens in, next-token distribution out.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .glassbox import GBConfig, PKMemory, alibi_bias


@dataclass
class G2Config:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    n_features: int = 4096
    k_active: int = 32
    mem_layer: int = 4
    mem_keys: int = 128
    mem_topk: int = 8
    attn_topk: int = 0             # >0: each query attends to at most this many positions per head
                                   # (exact zeros elsewhere): fewer attention edges to inspect
    mem_gate: str = "hard_st"      # hard gate, straight-through gradient ("hard": plain clamp, whose
                                   # gates freeze once saturated at 0 or 1)
    mix: str = "convex"            # how layers update the residual stream: "convex" (keep/replace
                                   # weights) or "add" (standard residual sums). Both are linear given
                                   # the decisions, so attribution is exact either way


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def scale(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return x * self.scale(x) * self.weight


class Attention(nn.Module):
    """Softmax attention heads, mixed into the residual by convex weights (keep, head 1..H)."""

    def __init__(self, c):
        super().__init__()
        self.h, self.dh = c.n_head, c.n_embd // c.n_head
        self.topk = c.attn_topk
        self.add = c.mix == "add"
        self.norm = RMSNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.o = nn.Parameter(torch.randn(c.n_head, self.dh, c.n_embd) * 0.02)
        self.mix = nn.Linear(c.n_embd, c.n_head + 1)

    def decisions(self, x):
        B, T, C = x.shape
        z = self.norm(x)
        q, k, _ = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in self.qkv(z).split(C, -1))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh) + alibi_bias(T, self.h, x.device)
        if self.topk and self.topk < T:                 # keep the top-k scores; exp(-inf) is exactly 0
            kth = att.topk(self.topk, -1).values[..., -1:]
            att = att.masked_fill(att < kth, float("-inf"))
        return {"s": self.norm.scale(x), "A": att.softmax(-1), "pi": self.mix(z).softmax(-1)}

    def apply(self, x, d):
        """The layer as a linear map of x, given the decisions d (exact for the forward pass)."""
        B, T, C = x.shape
        z = x * d["s"] * self.norm.weight
        v = (z @ self.qkv.weight[2 * C:].T).view(B, T, self.h, self.dh).transpose(1, 2)
        u = torch.einsum("bhtk,hkd->bhtd", d["A"] @ v, self.o)
        if self.add:
            return x + u.sum(1)
        pi = d["pi"]
        return pi[..., :1] * x + (pi[..., 1:].transpose(1, 2)[..., None] * u).sum(1)

    def forward(self, x):
        d = self.decisions(x)
        return self.apply(x, d), d


class SparseFeatures(nn.Module):
    """A wide feature layer with exactly k active features per token, mixed in by a gate."""

    def __init__(self, c):
        super().__init__()
        self.k = c.k_active
        self.add = c.mix == "add"
        self.norm = RMSNorm(c.n_embd)
        self.enc = nn.Linear(c.n_embd, c.n_features)
        self.dec = nn.Linear(c.n_features, c.n_embd, bias=False)
        nn.init.normal_(self.dec.weight, 0, 0.02)
        self.gate = nn.Linear(c.n_embd, 1)

    def decisions(self, x):
        z = self.norm(x)
        with torch.no_grad():                       # the active set is a discrete choice
            pre = self.enc(z)
            top = pre.topk(self.k, -1)
            mask = torch.zeros_like(pre, dtype=torch.bool).scatter_(-1, top.indices, True) & (pre > 0)
        return {"s": self.norm.scale(x), "mask": mask, "g": torch.sigmoid(self.gate(z))}

    def features(self, x, d, with_bias=True):
        z = x * d["s"] * self.norm.weight
        pre = z @ self.enc.weight.T + (self.enc.bias if with_bias else 0)
        return pre * d["mask"]                                  # exactly 0 outside the active set

    def apply(self, x, d, with_bias=True, f=None):
        f = self.features(x, d, with_bias) if f is None else f
        if self.add:
            return x + self.dec(f)
        return (1 - d["g"]) * x + d["g"] * self.dec(f)

    def forward(self, x):
        d = self.decisions(x)
        d["acts"] = self.features(x, d)
        return self.apply(x, d, f=d["acts"]), d


class Glassbox2(nn.Module):
    def __init__(self, c: G2Config):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.n_embd)
        nn.init.normal_(self.embed.weight, 0, 0.02)
        self.attn = nn.ModuleList([Attention(c) for _ in range(c.n_layer)])
        self.feat = nn.ModuleList([SparseFeatures(c) for _ in range(c.n_layer)])
        self.mem = PKMemory(GBConfig(vocab_size=c.vocab_size, n_embd=c.n_embd, mem_keys=c.mem_keys,
                                     mem_topk=c.mem_topk, mem_value_init=0.0, mem_gate=c.mem_gate))
        self.norm_f = RMSNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.know = nn.Linear(c.n_embd, 1)
        self.sharp_a = nn.Parameter(torch.tensor(0.0))
        self.sharp_b = nn.Parameter(torch.tensor(1.0))
        n_slots = c.mem_keys ** 2
        self.register_buffer("writer_doc", torch.full((n_slots,), -1, dtype=torch.long))
        self.register_buffer("writer_w", torch.zeros(n_slots))
        self.trace = None

    def temperature(self, k):
        return F.softplus(F.softplus(self.sharp_a) * k + self.sharp_b) + 0.05

    def forward(self, idx):
        """Returns logits (B, T, V), k (B, T). self.trace holds every decision and activation."""
        x = self.embed(idx)
        tr = {"attn": [], "feat": [], "mem": None, "hs": []}
        for i in range(self.config.n_layer):
            x, d = self.attn[i](x)
            tr["attn"].append(d)
            if i == self.config.mem_layer:
                x = self.mem(x, None)[0]
                tr["mem"] = self.mem.trace
            x, d = self.feat[i](x)
            tr["feat"].append(d)
            tr["hs"].append(x)
        tr["s_f"] = self.norm_f.scale(x)
        z = self.norm_f(x)
        tr["z_f"] = z
        logits, k = self.head(z), self.know(z).squeeze(-1)
        tr["raw_logits"] = logits
        self.trace = tr
        return logits * self.temperature(k)[..., None], k

    @torch.no_grad()
    def record_writers(self, doc_ids):
        """After a training forward pass: for each memory slot read, remember the training
        document (doc_ids (B, T)) that read it with the largest gated weight so far."""
        tr = self.trace["mem"]
        w = (tr["weights"] * tr["gate"][..., None]).flatten()
        slots = tr["slots"].flatten()
        docs = doc_ids[..., None].expand_as(tr["slots"]).flatten()
        best = torch.full_like(self.writer_w, -1.0).scatter_reduce(0, slots, w, "amax")
        win = (w >= best[slots]) & (w > self.writer_w[slots])
        self.writer_w[slots[win]] = w[win]
        self.writer_doc[slots[win]] = docs[win]

    @torch.no_grad()
    def explain(self, idx, t, token):
        """Exact decomposition of the raw logit of `token` at position t of idx[0]:
        {"token@j": contribution of input token j, "memory@j": of the memory read at j,
         "bias@layer": of feature-layer biases}. The values sum to the raw logit."""
        self.forward(idx[:1])
        tr = self.trace
        T, C = idx.shape[1], self.config.n_embd
        names = [f"token@{j}" for j in range(T)] + [f"memory@{j}" for j in range(T)] + \
                [f"bias@{i}" for i in range(self.config.n_layer)]
        S = len(names)
        comp = torch.zeros(S, T, C, device=idx.device)
        comp[torch.arange(T), torch.arange(T)] = self.embed(idx[0])     # each token's embedding
        for i in range(self.config.n_layer):
            comp = self.attn[i].apply(comp, {k: v.expand(S, *v.shape[1:]) for k, v in tr["attn"][i].items()})
            if i == self.config.mem_layer:
                m = tr["mem"]
                b = m["gate"][0][..., None]                             # (T, 1)
                comp = (1 - b) * comp
                comp[T + torch.arange(T), torch.arange(T)] += (b * m["read"][0])
            d = {k: v.expand(S, *v.shape[1:]) for k, v in tr["feat"][i].items() if k != "acts"}
            biased = self.feat[i].apply(comp, d, with_bias=False)
            g, mask = tr["feat"][i]["g"][0], tr["feat"][i]["mask"][0]
            scale = 1.0 if self.feat[i].add else g
            biased[T * 2 + i] += scale * self.feat[i].dec(self.feat[i].enc.bias * mask)
            comp = biased
        z = comp[:, t] * tr["s_f"][0, t] * self.norm_f.weight
        contrib = z @ self.head.weight[token]
        return dict(zip(names, contrib.tolist()))

    @torch.no_grad()
    def memory_free(self):
        """From the last forward: True at t when the output at t provably does not depend on any
        memory content. Gates are computed before the read, so they do not depend on memory.
        * Memory after the last attention layer (mem_layer == n_layer - 1): the read at t can only
          reach position t, so t is free exactly when its own gate is 0.
        * Memory earlier: an open read at p changes p's state, hence its keys, hence what later
          queries attend to (with softmax weights or top-k selection alike), so every t >= p may
          depend on it; t is free only when every gate at positions <= t is 0."""
        closed = self.trace["mem"]["gate"] == 0
        if self.config.mem_layer == self.config.n_layer - 1:
            return closed
        return closed.int().cumprod(-1).bool()

    def active_features(self, layer):
        """(B, T, k) indices and values of the active features at a layer, from the last forward."""
        acts = self.trace["feat"][layer]["acts"]
        v, i = acts.topk(self.config.k_active, -1)
        return i, v
