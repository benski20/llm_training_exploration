"""A fully piecewise-linear, transformer-shaped model.

Each block is   s <- (I + M) s            (linear token mixing over positions)
                s <- s + W2 relu(W1 s + b1) + b2   (per-position ReLU MLP)
with no softmax or layer norm, so the whole network is exactly encodable as a
MILP. With `ternary=True` every weight matrix is quantised to {-a, 0, +a}
(BitNet b1.58-style absmean scaling, straight-through estimator).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize(w, bits):
    """Symmetric uniform k-bit grid with absmax scale (straight-through)."""
    n = 2 ** (bits - 1) - 1
    scale = w.abs().max().clamp(min=1e-8) / n
    q = (w / scale).round().clamp(-n, n) * scale
    return w + (q - w).detach()


def ternarize(w):
    scale = w.abs().mean().clamp(min=1e-8)
    q = (w / scale).round().clamp(-1, 1) * scale
    return w + (q - w).detach()


class MixerNet(nn.Module):
    def __init__(self, seq_len, vocab, d_model=8, d_hidden=16, n_blocks=2,
                 n_classes=2, ternary=False, causal=False, bits=None, shared=False,
                 uniform=False):
        super().__init__()
        L, V, d, h = seq_len, vocab, d_model, d_hidden
        self.L, self.V, self.d, self.h, self.C = L, V, d, h, n_classes
        self.n_blocks, self.ternary = n_blocks, ternary
        self.bits = bits  # k-bit uniform weights (used when ternary is False)
        self.W_in = nn.Parameter(torch.randn(d, V) / math.sqrt(V))
        self.b_in = nn.Parameter(torch.zeros(d))
        # shared: weight-tied across positions. Mixing is causal Toeplitz,
        # M[t, j] = m[t - j] (relative position), and the readout is a pooled
        # per-position head plus a last-position head. Same weights for any L.
        # uniform (implies shared): the kernel is m[0] on the diagonal and one
        # value m[1] for every earlier position, i.e. causal sum attention with
        # equal weights. Position t then depends only on its token and the
        # token counts of its prefix, which is what the exact verifier
        # (vt/exact.py) exploits.
        self.uniform = uniform
        shared = shared or uniform
        self.shared = shared
        if shared:
            causal = True
            off = torch.arange(L)[:, None] - torch.arange(L)[None, :]
            self.register_buffer("offset", off.clamp(min=0), persistent=False)
            self.register_buffer("mask", (off >= 0).float(), persistent=False)
            n_k = 2 if uniform else L
            self.mix = nn.ParameterList(
                [nn.Parameter(torch.randn(n_k) * 0.3 / math.sqrt(L)) for _ in range(n_blocks)])
        else:
            self.mix = nn.ParameterList(
                [nn.Parameter(torch.randn(L, L) * 0.3 / math.sqrt(L)) for _ in range(n_blocks)])
        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.randn(h, d) / math.sqrt(d)) for _ in range(n_blocks)])
        self.b1 = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(n_blocks)])
        self.W2 = nn.ParameterList(
            [nn.Parameter(torch.randn(d, h) * 0.5 / math.sqrt(h)) for _ in range(n_blocks)])
        self.b2 = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(n_blocks)])
        if shared:
            self.W_pool = nn.Parameter(torch.randn(n_classes, d) / math.sqrt(L * d))
            self.W_last = nn.Parameter(torch.randn(n_classes, d) / math.sqrt(d))
        else:
            self.W_out = nn.Parameter(torch.randn(n_classes, L * d) / math.sqrt(L * d))
        if not shared:
            # causal: position t mixes only positions <= t, as in a decoder LM
            self.register_buffer("mask", torch.tril(torch.ones(L, L)) if causal else torch.ones(L, L),
                                 persistent=False)
        self.b_out = nn.Parameter(torch.zeros(n_classes))

    def w(self, p):
        if self.ternary:
            return ternarize(p)
        return quantize(p, self.bits) if self.bits else p

    def mixw(self, i, L=None):
        L = L or self.L
        if self.shared:
            k = self.w(self.mix[i])
            if self.uniform:
                k = torch.cat([k[:1], k[1:2].expand(self.L - 1)])
            return k[self.offset[:L, :L]] * self.mask[:L, :L]
        assert L == self.L, "only weight-tied models run at other lengths"
        return self.w(self.mix[i] * self.mask) * self.mask

    def wout(self, L=None):
        """Effective (C, L*d) readout, position-major like the flattened state."""
        L = L or self.L
        if self.shared:
            W = self.w(self.W_pool).repeat(1, L)
            last = torch.zeros_like(W)
            last[:, -self.d:] = self.w(self.W_last)
            return W + last
        return self.w(self.W_out)

    def forward(self, x):
        L = x.shape[1]  # weight-tied models accept any length <= self.L
        s = x @ self.w(self.W_in).T + self.b_in
        for i in range(self.n_blocks):
            s = s + torch.einsum("ij,bjd->bid", self.mixw(i, L), s)
            z = s @ self.w(self.W1[i]).T + self.b1[i]
            s = s + F.relu(z) @ self.w(self.W2[i]).T + self.b2[i]
        return s.flatten(1) @ self.wout(L).T + self.b_out

    def ibp(self, x, eps):
        """Interval bounds over the L-inf ball of radius eps around x.

        Returns (pre-activation bounds [(l, u)] per block, centre, radius of
        the final flattened state)."""
        W = self.w(self.W_in)
        c = x @ W.T + self.b_in
        r = torch.full_like(x, eps) @ W.abs().T
        eye = torch.eye(self.L, device=x.device)
        pre = []
        for i in range(self.n_blocks):
            IM = eye + self.mixw(i)
            c = torch.einsum("ij,bjd->bid", IM, c)
            r = torch.einsum("ij,bjd->bid", IM.abs(), r)
            W1 = self.w(self.W1[i])
            zc, zr = c @ W1.T + self.b1[i], r @ W1.abs().T
            l, u = zc - zr, zc + zr
            pre.append((l, u))
            al, au = F.relu(l), F.relu(u)
            W2 = self.w(self.W2[i])
            c = c + ((al + au) / 2) @ W2.T + self.b2[i]
            r = r + ((au - al) / 2) @ W2.abs().T
        return pre, c.flatten(1), r.flatten(1)

    def margin_lower_bounds(self, x, y, eps):
        """Lower bounds on logit_y - logit_j for every j (0 at j == y)."""
        pre, c, r = self.ibp(x, eps)
        W = self.wout()
        Wd = W[y][:, None, :] - W[None, :, :]
        bd = self.b_out[y][:, None] - self.b_out[None, :]
        lb = torch.einsum("bcd,bd->bc", Wd, c) - torch.einsum("bcd,bd->bc", Wd.abs(), r) + bd
        return lb, pre

    @torch.no_grad()
    def export(self, length=None):
        """Numpy float64 copy of the deployed (quantised, if ternary) weights,
        at `length` (weight-tied models only) or the training length."""
        L = length or self.L
        f = lambda t: self.w(t).detach().double().numpy()
        g = lambda t: t.detach().double().numpy()
        return {
            "L": L, "V": self.V, "d": self.d, "h": self.h, "C": self.C,
            "W_in": f(self.W_in), "b_in": g(self.b_in),
            "mix": [self.mixw(i, L).detach().double().numpy() for i in range(self.n_blocks)],
            "W1": [f(m) for m in self.W1], "b1": [g(m) for m in self.b1],
            "W2": [f(m) for m in self.W2], "b2": [g(m) for m in self.b2],
            "W_out": self.wout(L).detach().double().numpy(), "b_out": g(self.b_out),
        }

    @torch.no_grad()
    def weight_zero_frac(self):
        mats = [self.W_in, *self.W1, *self.W2] + ([self.W_pool, self.W_last] if self.shared else [self.W_out])
        qs = [self.w(m) for m in mats] + [self.w(m) for m in self.mix]
        return float(sum((q == 0).sum() for q in qs) / sum(q.numel() for q in qs))
