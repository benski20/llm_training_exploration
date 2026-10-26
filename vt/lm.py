"""A standard decoder-only GPT with two verification-friendly training knobs
that cost O(params) per step and so scale to any model size:

- `ternary`: every linear layer is a BitLinear with weights in {-a, 0, +a}
  (absmean scaling, straight-through estimator; BitNet b1.58).
- `margin_weight`: the neuron-margin loss. For an MLP neuron z = w.x + b, the
  sign of z cannot change for any input perturbation with ||dx||_inf <= eps
  iff |z| > eps * ||w||_1. We penalise relu(1 - |z| / (eps ||w||_1)), pushing
  each neuron to be provably on or off in an eps-ball around every token's
  activation. Stable neurons are linear pieces, so a block whose neurons are
  stable is exactly (and cheaply) verifiable locally. This is the single-layer,
  closed-form version of the ReLU-stability loss, so it needs no bound
  propagation through attention.

The MLP uses ReLU (not GELU) so "stable" has an exact meaning.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ternarize


@dataclass
class GPTConfig:
    vocab_size: int = 256          # byte-level: works on any text
    block_size: int = 128
    pos_encoding: str = "learned"  # "learned" (absolute) or "alibi" (relative; extrapolates to longer inputs)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    mlp_ratio: int = 4
    dropout: float = 0.0
    ternary: bool = False
    margin_eps: float = 0.01       # eps of the stability ball (on normed MLP input)
    margin_weight: float = 0.0


class BitLinear(nn.Linear):
    def __init__(self, i, o, bias=False, ternary=False):
        super().__init__(i, o, bias=bias)
        self.ternary = ternary

    def qweight(self):
        return ternarize(self.weight) if self.ternary else self.weight

    def forward(self, x):
        return F.linear(x, self.qweight(), self.bias)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head = c.n_head
        self.alibi = c.pos_encoding == "alibi"
        if self.alibi:
            slopes = torch.tensor([2 ** (-8 * (i + 1) / c.n_head) for i in range(c.n_head)])
            self.register_buffer("slopes", slopes, persistent=False)
        self.qkv = BitLinear(c.n_embd, 3 * c.n_embd, ternary=c.ternary)
        self.proj = BitLinear(c.n_embd, c.n_embd, ternary=c.ternary)
        self.dropout = c.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        drop = self.dropout if self.training else 0.0
        if self.alibi:
            i = torch.arange(T, device=x.device)
            dist = (i[:, None] - i[None, :]).float()
            bias = -self.slopes[:, None, None] * dist
            bias = bias.masked_fill(dist < 0, float("-inf"))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), dropout_p=drop)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = BitLinear(c.n_embd, c.mlp_ratio * c.n_embd, bias=True, ternary=c.ternary)
        self.out = BitLinear(c.mlp_ratio * c.n_embd, c.n_embd, ternary=c.ternary)
        self.eps = c.margin_eps
        self.stats = None

    def forward(self, x):
        z = self.fc(x)
        # ||w||_1 per neuron, of the weights actually used in the forward pass
        radius = self.eps * self.fc.qweight().abs().sum(1)
        self.stats = (z, radius)
        return self.out(F.relu(z))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.RMSNorm(c.n_embd), nn.RMSNorm(c.n_embd)
        self.attn, self.mlp = Attention(c), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.wte = nn.Embedding(c.vocab_size, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd) if c.pos_encoding == "learned" else None
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.RMSNorm(c.n_embd)
        self.head = BitLinear(c.n_embd, c.vocab_size, ternary=c.ternary)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("out.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    def margin_loss(self):
        terms = [F.relu(1 - z.abs() / (r + 1e-8)).mean() for z, r in
                 (b.mlp.stats for b in self.blocks)]
        return torch.stack(terms).mean()

    @torch.no_grad()
    def stable_fraction(self):
        """Fraction of (token, MLP neuron) pairs provably stable in the eps-ball,
        from the last forward pass."""
        num = den = 0
        for b in self.blocks:
            z, r = b.mlp.stats
            num += (z.abs() > r).sum().item()
            den += z.numel()
        return num / den

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, top_k=None):
        for _ in range(n):
            logits, _ = self(idx[:, -self.config.block_size:])
            logits = logits[:, -1] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx
