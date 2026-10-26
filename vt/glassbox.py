"""Glass-box LM: a transformer designed to be read mechanistically.

Post-hoc interpretability has to undo superposition, guess where a quantity
lives, and then argue that a readout is causal. This architecture fixes the
answers in advance:

* **Evidence stream.** Every position carries, next to its content vector x,
  a distribution e over sources: each input position and the memory. The
  residual stream is updated only by convex mixing,

      x' = pi_0 x + sum_h pi_h (attention head h output),     e' = pi_0 e + sum_h pi_h A_h e
      x' = (1 - b) x + b memory(x),                             e' = (1 - b) e + b [memory]

  so e is an exact account of how much of each position's content was mixed
  in from each source. It is computed by the architecture, not fitted.
* **Starved core, explicit memory.** Token embeddings are fixed random
  vectors and the core has no MLPs, so it cannot store facts. A product-key
  memory (Lample et al., 2019) is the only trainable fact store; every read
  exposes its slots, weights and match score.
* **Self-knowledge bottleneck.** A scalar k ("will my answer be right?") is
  read at the answer position, trained on the model's *own* correctness
  (labels computed on the fly, so they exist for any checkable task), and the
  answer/abstain decision is a fixed monotone function of k:
  P(abstain) = sigmoid(beta - softplus(alpha) * k).
* **Counterfactual memory training.** `patch` replaces the memory read at a
  position with another example's read; training the output to follow the
  patch (interchange intervention) makes the read causally sufficient.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GBConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    mem_layer: int = 2             # block whose attention is followed by the memory read
    mem_keys: int = 128            # product keys per half: mem_keys**2 slots
    mem_topk: int = 8
    fixed_embed: bool = True       # starve the core: embeddings cannot store facts


def alibi_bias(T, n_head, device=None):
    slopes = torch.tensor([2 ** (-8 * (h + 1) / n_head) for h in range(n_head)], device=device)
    i = torch.arange(T, device=device)
    dist = (i[:, None] - i[None, :]).float()
    return (-slopes[:, None, None] * dist).masked_fill(dist < 0, float("-inf"))


class ConvexAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.dh = c.n_head, c.n_embd // c.n_head
        self.ln = nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.o = nn.Parameter(torch.randn(c.n_head, self.dh, c.n_embd) * 0.02)
        self.mix = nn.Linear(c.n_embd, c.n_head + 1)          # keep, head 1..H

    def forward(self, x, e):
        B, T, C = x.shape
        z = self.ln(x)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in self.qkv(z).split(C, -1))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh) + alibi_bias(T, self.h, x.device)
        a = att.softmax(-1)                                      # (B, H, T, T)
        u = torch.einsum("bhtk,hkd->bhtd", a @ v, self.o)        # each head's output, in model space
        pi = self.mix(z).softmax(-1)                             # (B, T, H + 1)
        x = pi[..., :1] * x + (pi[..., 1:].transpose(1, 2)[..., None] * u).sum(1)
        e = pi[..., :1] * e + (pi[..., 1:].transpose(1, 2)[..., None] * (a @ e[:, None])).sum(1)
        self.last_attn = a
        return x, e


class PKMemory(nn.Module):
    def __init__(self, c, dk=64):
        super().__init__()
        self.n, self.topk = c.mem_keys, c.mem_topk
        self.ln = nn.LayerNorm(c.n_embd)
        self.q = nn.Linear(c.n_embd, 2 * dk, bias=False)
        self.qn1, self.qn2 = nn.LayerNorm(dk), nn.LayerNorm(dk)
        self.k1 = nn.Parameter(torch.randn(self.n, dk) / math.sqrt(dk))
        self.k2 = nn.Parameter(torch.randn(self.n, dk) / math.sqrt(dk))
        self.values = nn.Embedding(self.n * self.n, c.n_embd)
        nn.init.normal_(self.values.weight, 0.0, 1.0)
        self.gate = nn.Linear(c.n_embd, 1)
        self.patch = None              # (mask (B, T) bool, read (B, d)) for interchange interventions
        self.trace = None

    def read(self, z):
        q1, q2 = self.q(z).chunk(2, -1)
        s1, s2 = self.qn1(q1) @ self.k1.T, self.qn2(q2) @ self.k2.T
        t1, i1 = s1.topk(self.topk, -1)
        t2, i2 = s2.topk(self.topk, -1)
        s = (t1[..., :, None] + t2[..., None, :]).flatten(-2)
        idx = (i1[..., :, None] * self.n + i2[..., None, :]).flatten(-2)
        best, j = s.topk(self.topk, -1)
        idx = idx.gather(-1, j)
        w = best.softmax(-1)
        return (w[..., None] * self.values(idx)).sum(-2), dict(slots=idx, weights=w, match=best[..., 0])

    def forward(self, x, e):
        z = self.ln(x)
        r, tr = self.read(z)
        if self.patch is not None:
            mask, val = self.patch
            r = torch.where(mask[..., None], val[:, None, :].expand_as(r), r)
        tr["read"] = r
        self.trace = tr
        b = torch.sigmoid(self.gate(z))
        x = (1 - b) * x + b * r
        e = (1 - b) * e
        e[..., -1:] = e[..., -1:] + b                              # the last source is "memory"
        tr["gate"] = b.squeeze(-1)
        return x, e


class GlassBoxLM(nn.Module):
    def __init__(self, c: GBConfig):
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab_size, c.n_embd)
        nn.init.normal_(self.embed.weight, 0.0, 1.0)
        self.embed.weight.requires_grad_(not c.fixed_embed)
        self.attn = nn.ModuleList([ConvexAttention(c) for _ in range(c.n_layer)])
        self.mem = PKMemory(c)
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.know = nn.Linear(c.n_embd, 1)                         # self-knowledge variable k
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, idx):
        """Returns logits (B, T, V), k (B, T), evidence (B, T, T + 1) and hidden states per block."""
        B, T = idx.shape
        x = self.embed(idx)
        e = torch.cat([torch.eye(T).expand(B, T, T), torch.zeros(B, T, 1)], -1)
        hs = []
        for i, blk in enumerate(self.attn):
            x, e = blk(x, e)
            if i == self.config.mem_layer:
                x, e = self.mem(x, e)
            hs.append(x)
        z = self.ln_f(x)
        return self.head(z), self.know(z).squeeze(-1), e, hs

    def abstain_logit(self, k):
        return self.beta - F.softplus(self.alpha) * k
