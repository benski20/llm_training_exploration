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
    mlp_ratio: int = 0             # 0: no MLPs (fully starved core); >0: small MLP core for language skill
    track_evidence: bool = True    # the evidence stream is read-only: it can be skipped while training
    sharpness: bool = False        # per-token k sets the next-token distribution's sharpness
    mem_evidence: str = "pooled"   # "pooled": one memory source; "per_position": each read is a source
    mem2_layer: int = -1           # optional second (late) memory read, after this block's attention
    mem_value_init: float = 1.0    # std of initial memory values; 0: unwritten memory reads as empty
    null_slot: bool = False        # a learned "nothing stored" option in every read (value 0)


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
        if e is not None:
            e = pi[..., :1] * e + (pi[..., 1:].transpose(1, 2)[..., None] * (a @ e[:, None])).sum(1)
        self.last_attn = a
        return x, e


class ConvexMLP(nn.Module):
    """Per-position transform: it moves no information between positions, so the
    evidence is unchanged."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c.n_embd)
        self.fc = nn.Linear(c.n_embd, c.mlp_ratio * c.n_embd)
        self.out = nn.Linear(c.mlp_ratio * c.n_embd, c.n_embd)
        self.gate = nn.Linear(c.n_embd, 1)

    def forward(self, x, e):
        z = self.ln(x)
        b = torch.sigmoid(self.gate(z))
        return (1 - b) * x + b * self.out(F.gelu(self.fc(z))), e


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
        nn.init.normal_(self.values.weight, 0.0, c.mem_value_init)
        self.gate = nn.Linear(c.n_embd, 1)
        if c.null_slot:
            self.null = nn.Parameter(torch.tensor(0.0))
        self.lscale = nn.Parameter(torch.tensor(1.0))   # affine map of the match score, for calibration
        self.lbias = nn.Parameter(torch.tensor(0.0))
        self.patch = None              # (mask (B, T), read (B, d) or (B, T, d)): interchange interventions
        self.trace = None
        self.register_buffer("usage", torch.zeros(self.n * self.n))   # read mass per slot while storing

    @torch.no_grad()
    def accumulate(self, mask):
        """Add the last forward's read weights at positions `mask` (B, T) to the slot-usage counts."""
        slots, w = self.trace["slots"][mask], self.trace["weights"][mask]
        self.usage.index_add_(0, slots.flatten(), w.flatten())

    def familiarity(self):
        """Per position of the last forward: usage-weighted read, log(1 + usage). Not trained:
        a read landing on slots that were never used to store anything scores 0."""
        return (self.trace["weights"] * torch.log1p(self.usage[self.trace["slots"]])).sum(-1)

    def read(self, z):
        q1, q2 = self.q(z).chunk(2, -1)
        s1, s2 = self.qn1(q1) @ self.k1.T, self.qn2(q2) @ self.k2.T
        t1, i1 = s1.topk(self.topk, -1)
        t2, i2 = s2.topk(self.topk, -1)
        s = (t1[..., :, None] + t2[..., None, :]).flatten(-2)
        idx = (i1[..., :, None] * self.n + i2[..., None, :]).flatten(-2)
        best, j = s.topk(self.topk, -1)
        idx = idx.gather(-1, j)
        tr = dict(slots=idx, match=best[..., 0])
        if hasattr(self, "null"):
            wn = torch.cat([best, self.null.expand(best.shape[:-1])[..., None]], -1).softmax(-1)
            w, tr["null"] = wn[..., :-1], wn[..., -1]                # weight on "nothing stored"
        else:
            w = best.softmax(-1)
        tr["weights"] = w
        return (w[..., None] * self.values(idx)).sum(-2), tr

    def forward(self, x, e):
        z = self.ln(x)
        r, tr = self.read(z)
        if self.patch is not None:
            mask, val = self.patch
            val = val[:, None, :].expand_as(r) if val.dim() == 2 else val
            r = torch.where(mask[..., None], val, r)
        tr["read"] = r
        self.trace = tr
        b = torch.sigmoid(self.gate(z))
        x = (1 - b) * x + b * r
        if e is not None:
            T = x.shape[1]
            e = (1 - b) * e
            if e.shape[-1] == 2 * T:                               # sources T..2T-1: the read at each position
                e[..., T:] = e[..., T:] + b * torch.eye(T)
            else:
                e[..., -1:] = e[..., -1:] + b                      # the last source is "memory"
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
        if c.mlp_ratio:
            self.mlp = nn.ModuleList([ConvexMLP(c) for _ in range(c.n_layer)])
        self.mem = PKMemory(c)
        if c.mem2_layer >= 0:
            self.mem2 = PKMemory(c)
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.know = nn.Linear(c.n_embd, 1)                         # self-knowledge variable k
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        if c.sharpness:
            self.sharp_a = nn.Parameter(torch.tensor(0.0))
            self.sharp_b = nn.Parameter(torch.tensor(1.0))

    def forward(self, idx, evidence=None):
        """Returns logits (B, T, V), k (B, T), evidence (B, T, T + 1) or None, and hidden
        states per block. With `sharpness`, logits are scaled by a fixed increasing
        function of k, so the model's commitment to its next token flows through k."""
        B, T = idx.shape
        track = self.config.track_evidence if evidence is None else evidence
        x = self.embed(idx)
        n_mem = T if self.config.mem_evidence == "per_position" else 1
        e = torch.cat([torch.eye(T).expand(B, T, T), torch.zeros(B, T, n_mem)], -1) if track else None
        hs = []
        for i, blk in enumerate(self.attn):
            x, e = blk(x, e)
            if i == self.config.mem_layer:
                x, e = self.mem(x, e)
            if i == self.config.mem2_layer:
                x, e = self.mem2(x, e)
            if self.config.mlp_ratio:
                x, e = self.mlp[i](x, e)
            hs.append(x)
        z = self.ln_f(x)
        logits, k = self.head(z), self.know(z).squeeze(-1)
        if self.config.sharpness:
            logits = logits * self.temperature(k)[..., None]
        return logits, k, e, hs

    def temperature(self, k):
        """Inverse temperature, increasing in k."""
        return F.softplus(F.softplus(self.sharp_a) * k + self.sharp_b) + 0.05

    def abstain_logit(self, k):
        return self.beta - F.softplus(self.alpha) * k
