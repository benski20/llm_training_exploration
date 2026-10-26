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


def ternarize(w):
    scale = w.abs().mean().clamp(min=1e-8)
    q = (w / scale).round().clamp(-1, 1) * scale
    return w + (q - w).detach()


class MixerNet(nn.Module):
    def __init__(self, seq_len, vocab, d_model=8, d_hidden=16, n_blocks=2,
                 n_classes=2, ternary=False, causal=False):
        super().__init__()
        L, V, d, h = seq_len, vocab, d_model, d_hidden
        self.L, self.V, self.d, self.h, self.C = L, V, d, h, n_classes
        self.n_blocks, self.ternary = n_blocks, ternary
        # causal: position t mixes only positions <= t, as in a decoder LM
        self.register_buffer("mask", torch.tril(torch.ones(L, L)) if causal else torch.ones(L, L), persistent=False)
        self.W_in = nn.Parameter(torch.randn(d, V) / math.sqrt(V))
        self.b_in = nn.Parameter(torch.zeros(d))
        self.mix = nn.ParameterList(
            [nn.Parameter(torch.randn(L, L) * 0.3 / math.sqrt(L)) for _ in range(n_blocks)])
        self.W1 = nn.ParameterList(
            [nn.Parameter(torch.randn(h, d) / math.sqrt(d)) for _ in range(n_blocks)])
        self.b1 = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(n_blocks)])
        self.W2 = nn.ParameterList(
            [nn.Parameter(torch.randn(d, h) * 0.5 / math.sqrt(h)) for _ in range(n_blocks)])
        self.b2 = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(n_blocks)])
        self.W_out = nn.Parameter(torch.randn(n_classes, L * d) / math.sqrt(L * d))
        self.b_out = nn.Parameter(torch.zeros(n_classes))

    def w(self, p):
        return ternarize(p) if self.ternary else p

    def mixw(self, i):
        return self.w(self.mix[i] * self.mask) * self.mask

    def forward(self, x):
        s = x @ self.w(self.W_in).T + self.b_in
        for i in range(self.n_blocks):
            s = s + torch.einsum("ij,bjd->bid", self.mixw(i), s)
            z = s @ self.w(self.W1[i]).T + self.b1[i]
            s = s + F.relu(z) @ self.w(self.W2[i]).T + self.b2[i]
        return s.flatten(1) @ self.w(self.W_out).T + self.b_out

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
        W = self.w(self.W_out)
        Wd = W[y][:, None, :] - W[None, :, :]
        bd = self.b_out[y][:, None] - self.b_out[None, :]
        lb = torch.einsum("bcd,bd->bc", Wd, c) - torch.einsum("bcd,bd->bc", Wd.abs(), r) + bd
        return lb, pre

    @torch.no_grad()
    def export(self):
        """Numpy float64 copy of the deployed (quantised, if ternary) weights."""
        f = lambda t: self.w(t).detach().double().numpy()
        g = lambda t: t.detach().double().numpy()
        return {
            "L": self.L, "V": self.V, "d": self.d, "h": self.h, "C": self.C,
            "W_in": f(self.W_in), "b_in": g(self.b_in),
            "mix": [self.mixw(i).detach().double().numpy() for i in range(self.n_blocks)],
            "W1": [f(m) for m in self.W1], "b1": [g(m) for m in self.b1],
            "W2": [f(m) for m in self.W2], "b2": [g(m) for m in self.b2],
            "W_out": f(self.W_out), "b_out": g(self.b_out),
        }

    @torch.no_grad()
    def weight_zero_frac(self):
        mats = [self.W_in, self.W_out, *self.W1, *self.W2]
        qs = [self.w(m) for m in mats] + [self.mixw(i)[self.mask.bool()] for i in range(self.n_blocks)]
        return float(sum((q == 0).sum() for q in qs) / sum(q.numel() for q in qs))
