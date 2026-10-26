"""Count-state generators: generative models whose whole output distribution is
exactly auditable.

A CountGen reads tokens left to right with windowed-uniform causal mixing (the
same block as an exactly verifiable MixerNet): after reading x_<=t its state is

    s_t = E[x_t] + sum_{i<w} k_i E[x_{t-i}] + k_w * sum_v count_v(x_<=t-w) E[v],

so the next-token distribution depends only on (the last w tokens, the counts
of older tokens). That state has polynomially many values, so a dynamic program
over (generator state x spec automaton) computes exact properties of the
distribution over all V^L outputs (see vt/gen_verify.py).

Next-token probabilities use a rational normalisation instead of softmax:

    p(v) = (relu(z_v) + eps) / sum_u (relu(z_u) + eps)

With eps = 0 at deployment, a token can have probability exactly 0, and every
probability is an exact rational function of the (dyadic) weights.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CountGen(nn.Module):
    def __init__(self, vocab, d_model=8, d_hidden=32, window=1, max_len=512):
        super().__init__()
        V, d, h, w = vocab, d_model, d_hidden, window
        self.V, self.d, self.h, self.w, self.max_len = V, d, h, w, max_len
        self.E = nn.Parameter(torch.randn(V, d) / math.sqrt(d))
        self.kern = nn.Parameter(torch.cat([torch.zeros(w), torch.full((1,), 0.5)]))
        self.W1 = nn.Parameter(torch.randn(h, d) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(h))
        self.Wo = nn.Parameter(torch.randn(V, h) / math.sqrt(h))
        self.bo = nn.Parameter(torch.ones(V))
        self.z0 = nn.Parameter(torch.ones(V))          # logits for the first token
        off = torch.arange(max_len)[:, None] - torch.arange(max_len)[None, :]
        self.register_buffer("offset", off.clamp(min=0), persistent=False)
        self.register_buffer("causal", (off >= 0).float(), persistent=False)

    def mixing(self, L):
        k = torch.cat([self.kern[:self.w], self.kern[self.w:].expand(max(0, L - self.w))])[:L]
        M = k[self.offset[:L, :L]] * self.causal[:L, :L]
        return torch.eye(L, device=M.device) + M

    def scores(self, x):
        """x: (B, L) token ids. Returns nonnegative scores (B, L+1, V): row t is
        the next-token score after reading x_<t (row 0 = first token)."""
        B, L = x.shape
        s = self.mixing(L) @ self.E[x]                   # (B, L, d)
        z = F.relu(s @ self.W1.T + self.b1) @ self.Wo.T + self.bo
        first = self.z0.expand(B, 1, self.V)
        return F.relu(torch.cat([first, z], 1))

    def log_prob(self, x, eps):
        """Exact log-likelihood of each sequence under p = (relu(z)+eps)/sum."""
        sc = self.scores(x)[:, :-1] + eps
        p = sc / sc.sum(-1, keepdim=True)
        return torch.log(p.gather(-1, x[..., None]).squeeze(-1)).sum(-1)

    @torch.no_grad()
    def sample(self, n, L, eps=0.0, generator=None):
        x = torch.zeros(n, 0, dtype=torch.long)
        for t in range(L):
            sc = self.scores(x)[:, -1] + eps
            tot = sc.sum(-1, keepdim=True)
            if (tot <= 0).any():
                raise RuntimeError("dead end: all next-token scores are zero")
            nxt = torch.multinomial(sc / tot, 1, generator=generator)
            x = torch.cat([x, nxt], 1)
        return x

    @torch.no_grad()
    def export(self):
        f = lambda t: t.detach().double().numpy()
        return {"kind": "countgen", "V": self.V, "d": self.d, "h": self.h, "w": self.w,
                "E": f(self.E), "kern": f(self.kern), "W1": f(self.W1), "b1": f(self.b1),
                "Wo": f(self.Wo), "bo": f(self.bo), "z0": f(self.z0)}


def np_scores(G, x):
    """Float64 next-token scores for one sequence (matches CountGen.scores)."""
    L, w = len(x), G["w"]
    k = list(G["kern"][:w]) + [G["kern"][w]] * max(0, L - w)
    out = [np.maximum(G["z0"], 0)]
    for t in range(L):
        s = G["E"][x[t]].copy()
        for j in range(t + 1):
            s = s + k[t - j] * G["E"][x[j]]
        z = np.maximum(G["W1"] @ s + G["b1"], 0) @ G["Wo"].T + G["bo"]
        out.append(np.maximum(z, 0))
    return np.array(out)
