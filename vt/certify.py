"""Certified attribution for glass-box models: prove what an output does not depend on.

Statement certified (sound, for the given input):
    "whatever values the memory returns, at any position, the model's
     decision at position t is the same"
i.e. the decision provably does not depend on stored knowledge: it comes from
the input alone.

Method: run the model exactly up to the memory read; replace every read by
the box enclosing all value vectors (a read is a convex combination of
values, so it lies in their hull, inside that box); propagate interval bounds
through the remaining layers (interval bound propagation). The glass-box
design keeps this tight: the residual stream is only ever a convex mix of its
inputs, so every mixing step is also bounded by the hull of what it mixes,
and the bounds use whichever is tighter.

Sound by construction (every step over-approximates); it can fail to certify
a decision that is in fact independent, never the reverse. Tests check both
that zero-width boxes reproduce the exact forward pass and that sampled
memory contents stay inside the bounds.
"""
import math

import torch
import torch.nn.functional as F

from .glassbox import alibi_bias

GELU_MIN_X, GELU_MIN = -0.7517916, -0.1699712


def lin(lo, hi, W, b=None):
    c, r = (lo + hi) / 2, (hi - lo) / 2
    oc = c @ W.T + (b if b is not None else 0)
    orr = r @ W.abs().T
    return oc - orr, oc + orr


def mul(al, ah, bl, bh):
    p = torch.stack([al * bl, al * bh, ah * bl, ah * bh])
    return p.min(0).values, p.max(0).values


def layernorm(lo, hi, ln):
    """Sound bounds for LayerNorm over a box."""
    n = lo.shape[-1]
    # x_i - mean = (1 - 1/n) x_i - (1/n) sum_{j != i} x_j, a linear function of the box
    sl, sh = lo.sum(-1, keepdim=True), hi.sum(-1, keepdim=True)
    cl = (1 - 1 / n) * lo - (sh - hi) / n
    ch = (1 - 1 / n) * hi - (sl - lo) / n
    near = torch.clamp(torch.maximum(cl, -ch), min=0)                  # distance of [cl, ch] to 0
    far = torch.maximum(cl.abs(), ch.abs())
    var_lo, var_hi = (near ** 2).mean(-1, keepdim=True), (far ** 2).mean(-1, keepdim=True)
    sl_, sh_ = 1 / torch.sqrt(var_hi + ln.eps), 1 / torch.sqrt(var_lo + ln.eps)
    yl, yh = mul(cl, ch, sl_.expand_as(cl), sh_.expand_as(ch))
    g, b = ln.weight, ln.bias
    return torch.where(g >= 0, yl * g, yh * g) + b, torch.where(g >= 0, yh * g, yl * g) + b


def softmax_bounds(lo, hi):
    m = hi.max(-1, keepdim=True).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    el, eh = torch.exp(lo - m), torch.exp(hi - m)
    Sl, Sh = el.sum(-1, keepdim=True), eh.sum(-1, keepdim=True)
    low = el / (el + Sh - eh).clamp_min(1e-30)
    up = eh / (eh + Sl - el).clamp_min(1e-30)
    return low, up.clamp(max=1.0)


def convex_mix(wl, wh, il, ih):
    """sum_j w_j item_j with w on the simplex: weights (..., n), items (..., n, C)."""
    pl, ph = mul(wl[..., None], wh[..., None], il, ih)
    pl, ph = pl.sum(-2), ph.sum(-2)
    return torch.maximum(pl, il.min(-2).values), torch.minimum(ph, ih.max(-2).values)


def gelu_bounds(lo, hi):
    gl, gh = F.gelu(lo), F.gelu(hi)
    low = torch.where(hi <= GELU_MIN_X, gh, torch.where(lo >= GELU_MIN_X, gl, torch.full_like(lo, GELU_MIN)))
    up = torch.where(hi <= GELU_MIN_X, gl, torch.where(lo >= GELU_MIN_X, gh, torch.maximum(gl, gh)))
    return low, up


def attention_box(blk, lo, hi):
    B, T, C = lo.shape
    H, dh = blk.h, blk.dh
    zl, zh = layernorm(lo, hi, blk.ln)
    W = blk.qkv.weight
    ql, qh = lin(zl, zh, W[:C]); kl, kh = lin(zl, zh, W[C:2 * C]); vl, vh = lin(zl, zh, W[2 * C:])
    sh = lambda t: t.view(B, T, H, dh).transpose(1, 2)                    # noqa: E731
    ql, qh, kl, kh, vl, vh = map(sh, (ql, qh, kl, kh, vl, vh))
    sl, su = mul(ql[:, :, :, None], qh[:, :, :, None], kl[:, :, None], kh[:, :, None])
    sl, su = sl.sum(-1) / math.sqrt(dh), su.sum(-1) / math.sqrt(dh)
    bias = alibi_bias(T, H, lo.device)
    sl, su = sl + bias, su + bias
    al, ah = softmax_bounds(sl, su)                                       # (B, H, T, T)
    ol, oh = convex_mix(al, ah, vl[:, :, None].expand(B, H, T, T, dh), vh[:, :, None].expand(B, H, T, T, dh))
    ul, uh = [], []
    for h in range(H):                                                    # each head into model space
        a_, b_ = lin(ol[:, h], oh[:, h], blk.o[h].T)
        ul.append(a_); uh.append(b_)
    ml, mh = lin(zl, zh, blk.mix.weight, blk.mix.bias)
    pl, ph = softmax_bounds(ml, mh)                                       # (B, T, H + 1)
    il = torch.stack([lo] + ul, -2); ih = torch.stack([hi] + uh, -2)      # (B, T, H + 1, C)
    return convex_mix(pl, ph, il, ih)


def mlp_box(m, lo, hi):
    zl, zh = layernorm(lo, hi, m.ln)
    gl, gh = lin(zl, zh, m.gate.weight, m.gate.bias)
    bl, bh = torch.sigmoid(gl), torch.sigmoid(gh)
    fl, fh = gelu_bounds(*lin(zl, zh, m.fc.weight, m.fc.bias))
    yl, yh = lin(fl, fh, m.out.weight, m.out.bias)
    wl = torch.cat([1 - bh, bl], -1); wh = torch.cat([1 - bl, bh], -1)
    return convex_mix(wl, wh, torch.stack([lo, yl], -2), torch.stack([hi, yh], -2))


@torch.no_grad()
def memory_free_bounds(model, idx, positions=None, values=None):
    """Interval bounds on the raw logits and on k at every position when the memory reads at
    `positions` (bool (B, T); default: every position) may be any convex combination of the
    memory values (`values`, default: all of them). Returns
    (logit_lo, logit_hi, k_lo, k_hi, raw_logits, k) where the last two are the exact forward pass."""
    c = model.config
    if c.mem2_layer >= 0 or c.mem_heads > 1:
        raise NotImplementedError("certify supports one plain memory read")
    B, T = idx.shape
    pos = torch.ones(B, T, dtype=torch.bool) if positions is None else positions
    V = model.mem.values.weight if values is None else values
    vl, vh = V.min(0).values, V.max(0).values
    x = model.embed(idx)
    lo = hi = None
    for i, blk in enumerate(model.attn):
        if lo is None:
            x, _ = blk(x, None)
        else:
            lo, hi = attention_box(blk, lo, hi)
        if i == c.mem_layer:
            z = model.mem.ln(x)
            r, _ = model.mem.read(z)
            b = model.mem.gate_value(z)
            rl = torch.where(pos[..., None], vl.expand_as(r), r)
            rh = torch.where(pos[..., None], vh.expand_as(r), r)
            lo, hi = (1 - b) * x + b * rl, (1 - b) * x + b * rh
        if c.mlp_ratio:
            if lo is None:
                x, _ = model.mlp[i](x, None)
            else:
                lo, hi = mlp_box(model.mlp[i], lo, hi)
    zl, zh = layernorm(lo, hi, model.ln_f)
    ll, lh = lin(zl, zh, model.head.weight)
    kl, kh = lin(zl, zh, model.know.weight, model.know.bias)
    logits, k, _, _ = model(idx, evidence=False)
    if c.sharpness:
        logits = logits / model.temperature(k)[..., None]
    return ll, lh, kl.squeeze(-1), kh.squeeze(-1), logits, k


def certified_argmax(lo, hi, choices=None):
    """Index of the option whose lower bound beats every other option's upper bound, or -1.
    lo, hi: (..., V); choices: optional tensor of allowed option indices."""
    if choices is not None:
        lo, hi = lo[..., choices], hi[..., choices]
    best = lo.argmax(-1)
    others = hi.scatter(-1, best[..., None], float("-inf")).max(-1).values
    ok = lo.gather(-1, best[..., None]).squeeze(-1) > others
    out = best if choices is None else choices[best]
    return torch.where(ok, out, torch.full_like(out, -1))
