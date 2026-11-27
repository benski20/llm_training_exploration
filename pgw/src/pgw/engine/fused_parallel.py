"""Parallel attention + MLP blocks with the RMSNorm fusion: ONE provider trip per layer (prototype, default off).

A parallel block (GPT-J / GPT-NeoX / PaLM / Falcon style) is ``h' = h + o(attn(n1(h))) + down(silu(gate) * up)``
with ``[gate | up] = gate_up(n2(h))``: attention and MLP read the same residual. With the fused2 machinery
(:mod:`pgw.engine.fused2`) a whole layer is one pair trip:

* stage 1 (first linear): ``H = R + hmul * (W1_8 x)`` with ``W1 = [W_o | W_down]`` (one int8 matrix, per-row scales over
  both parts) and ``x = quant([a_attn | m])`` (one per-row activation scale): the residual delta of the layer, added
  on the provider in the ring, exactly as fused2 adds ``W_o a``;
* stage 2: ``Z = W2'_8 H`` with ``W2' = [W_qkv diag(g1 * grid) ; W_gate_up diag(g2 * grid)]`` (the next layer's two
  norm gains folded, rows stacked; per-row int8 as two separate folds would be);
* the client computes ``rms(h')`` from the verified ``H``, scales ``Z``, runs RoPE + softmax attention and SiLU,
  quantizes ``[a | m]`` of the next layer and sends the next trip.

Trips per pass: layer 0's ``[qkv | gate_up]`` (one single trip on the quantized embedding), then one pair per layer;
the last layer's trip is ``W1`` alone, or with :attr:`ParallelFusedSchedule.head_fold` the folded LM head as its
second stage: **L + 1** trips with the head on the provider (standard sequential layers: 4L + 1; fused2: 2L + 1).

Exactness, masks, privacy and integrity are fused2's (same trip kinds, the same :class:`Fused2MaskedBackend`): the
provider gets ``(R + v - hmul * W1 r_x, x + r_x)``, uniform for any data, each mask row once; the corrections ``W1 r_x``
and ``W2' v`` are two single-stage products (no product chain), with ``sum(in) = d + I`` input and ``d + (qkv + 2I)``
output elements per layer and slot: the same element counts as one standard slot of the four linears; every returned
value (``H``, ``Z``, the head's logits) passes the range check and the ``F_(2^61-1)`` Freivalds check before use.

Scope of this prototype: the engine reads a Qwen2 / Llama pack **as if** it were a parallel-block model (a different
model: the outputs are not the sequential checkpoint's; only exactness, trips, privacy and integrity are meaningful on
such a pack). A real parallel-block checkpoint (Pythia / GPT-NeoX: LayerNorm, GELU, partial rotary, biases) needs its
architecture in :mod:`pgw.engine.model` first; LayerNorm folds the same way (``LN(x) = g * C x / std + b`` with the
centering matrix ``C``, linear: ``W diag(g) C`` is folded, ``W b`` joins the bias, ``std`` comes from ``H``). See
``research/reviews/consumer-round-trips.md``.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError
from pgw.core.ring import RingParams, quantize_weight_rows
from pgw.core.types import LinearId, LinearKind
from pgw.engine.backends import LinearCall
from pgw.engine.fused2 import (
    FUSED2_META,
    F2Linear,
    Fused2FoldedHead,
    Fused2Schedule,
    Fused2Trip,
    normed_output,
    quantize_pair,
    quantize_single,
    residual_from,
    rinv_rows,
)
from pgw.modelpack.fused2_weights import _fold, _make_linear

__all__ = ["ParallelFusedSchedule", "ParallelWeights", "enable_fused_parallel", "parallel_float_reference"]


def _float_linear(pack: Any, layer: int, kind: LinearKind) -> tuple[torch.Tensor, torch.Tensor | None]:
    lw = pack.load_linear(LinearId(layer, kind), device="cpu")
    return lw.w8.to(torch.float64) * lw.scale.to(torch.float64)[:, None], lw.bias


class ParallelWeights:
    """Public tables of the parallel-block fused reference, derived from a pack's linears (pack time; no secret).

    ``first[l]`` = int8 ``[W_o | W_down]`` of layer ``l`` (with residual-grid multipliers ``hmul`` from ``bits``-bit
    migration over both next gains); ``second[l]`` = the stacked fold ``[W_qkv diag(g_in * grid) ; W_gate_up
    diag(g_post * grid)]`` of layer ``l``'s input (``grid`` = the previous first linear's residual grid; 1 for layer
    0: the quantized embedding); ``head`` (``head=True``) = ``int8(W_head diag(g_final * grid))``. Float sources: the
    pack's dequantized int8 weights (a second rounding; a real deployment folds the checkpoint's float weights).
    """

    def __init__(self, pack: Any, *, bits: int = 8, head: bool = False, sigma: bool = True):
        a = pack.arch
        self.num_layers = nl = int(a.num_layers)
        self.fingerprint = str(getattr(pack, "fingerprint", ""))
        self.split_qkv = None
        load = pack.load_plain
        g_in = [load(f"layers.{i}.input_norm", dtype=torch.float32).to(torch.float64) for i in range(nl)]
        g_post = [load(f"layers.{i}.post_attn_norm", dtype=torch.float32).to(torch.float64) for i in range(nl)]
        g_fin = load("final_norm", dtype=torch.float32).to(torch.float64)
        self.first: list[F2Linear] = []
        self.second: list[F2Linear] = []
        self.head: F2Linear | None = None
        self.qkv_rows = 0
        for i in range(nl):
            wo, bo = _float_linear(pack, i, LinearKind.O)
            wd, bd = _float_linear(pack, i, LinearKind.DOWN)
            if bo is not None or bd is not None:
                raise ConfigError("parallel fused prototype: o / down biases are not supported")
            w8, sw = quantize_weight_rows(torch.cat([wo, wd], dim=1))
            if i + 1 < nl:
                gain = torch.maximum(g_in[i + 1].abs(), g_post[i + 1].abs())
            else:
                gain = g_fin.abs() if head else None
            hmul = None
            if gain is not None and bits > 0:
                c = gain * sw.to(torch.float32).to(torch.float64)
                kappa = float(c.max()) / 2.0 ** bits
                hmul = torch.floor(c / kappa).clamp(min=1.0).to(torch.int64) if kappa > 0 else None
            self.first.append(_make_linear(f"{i}.o+down/p", w8, sw, None, sigma=sigma, hmul=hmul))
        for i in range(nl):
            grid = self.first[i - 1].grid if i else torch.ones(int(a.hidden_size), dtype=torch.float64)
            wq, bq = _float_linear(pack, i, LinearKind.QKV)
            wg, bg = _float_linear(pack, i, LinearKind.GATE_UP)
            fq = _fold(f"{i}.qkv/p", wq, g_in[i] * grid, bq)
            fg = _fold(f"{i}.gate_up/p", wg, g_post[i] * grid, bg)
            self.qkv_rows = fq.m
            bias = None
            if fq.bias is not None or fg.bias is not None:
                bias = torch.cat([fq.bias if fq.bias is not None else torch.zeros(fq.m, dtype=torch.float64),
                                  fg.bias if fg.bias is not None else torch.zeros(fg.m, dtype=torch.float64)])
            w8 = torch.cat([fq.w8, fg.w8], dim=0)
            sc = torch.cat([fq.scale, fg.scale])
            st = _make_linear(f"{i}.qkv+gate_up/p", w8, sc.to(torch.float32), bias)
            self.second.append(F2Linear(st.key, st.w8, sc, bias, st.wnorm, st.sigma))
        if head:
            name = "embed_tokens" if getattr(a, "tie_word_embeddings", False) else "lm_head"
            fh = load(name, dtype=torch.float32)
            self.head = _fold(f"{nl}.lm_head/p", fh, g_fin * self.first[-1].grid, None)

    def all_linears(self) -> list[F2Linear]:
        return self.second + self.first + ([self.head] if self.head is not None else [])

    def linears(self) -> list[F2Linear]:
        return self.all_linears()


class ParallelFusedSchedule(Fused2Schedule):
    """:class:`pgw.engine.model.Schedule` of the parallel-block fused reference: one trip per layer (+ layer 0's)."""

    name = "fused_parallel"

    def __init__(self, weights: ParallelWeights, ring: RingParams, *, head_fold: bool = False):
        if not ring.cap or ring.cap + 1 >= 61:
            raise ConfigError("the parallel fused schedule needs a capped ring with 2^(C+1) < 2^61 - 1")
        if head_fold and weights.head is None:
            raise ConfigError("head_fold needs ParallelWeights(head=True)")
        self.w = weights  # type: ignore[assignment]
        self.ring = ring
        self.head_fold = bool(head_fold)
        self.local_qkv0 = False
        self.head_rows = 0
        self.gateway_macs = 0
        self.h_absmax = None
        self.bits_hist: dict[int, int] = {}
        self.native_decode = None
        self._lock = threading.Lock()

    def linears(self, model: Any, layer: int) -> Sequence[LinearId]:
        lids = [LinearId(layer, LinearKind.O)]
        return [LinearId(0, LinearKind.QKV), *lids] if layer == 0 else lids

    def _split(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.w.qkv_rows  # type: ignore[attr-defined]
        return z[:, :q], z[:, q:]

    def layer(self, model: Any, layer: int, st: Any) -> Any:
        w, ring, eps = self.w, self.ring, float(model.eps)
        ex = st.extra.setdefault(FUSED2_META, {})
        if layer == 0:
            hs = [h.detach().to(torch.float64).numpy() for h in st.hs]
            lin = w.second[0]  # type: ignore[attr-defined]
            trips = []
            for h in hs:
                xq, s = quantize_single(torch.from_numpy(h), lin, ring)
                trips.append((Fused2Trip("single", lin, xq, ring.cap), s))
            ys = yield [LinearCall(LinearId(0, LinearKind.QKV), t.x, sess, {FUSED2_META: t})
                        for (t, _), sess in zip(trips, st.sessions, strict=True)]
            zs = [normed_output(y, lin, s, rinv_rows(h, eps)) for y, (_, s), h in zip(ys, trips, hs, strict=True)]
        else:
            hs, zs = ex["h"], ex["z"]
        qkv, gus = zip(*(self._split(z) for z in zs), strict=True)
        attn = model._fused2_attention(layer, st, list(qkv))
        inter = model.split_gate_up[0]
        acts = [st.ops.silu_mul(g[:, :inter].to(model.dtype), g[:, inter:].to(model.dtype)) for g in gus]
        xs = [torch.cat([a.to(torch.float32), m.to(torch.float32)], dim=1) for a, m in zip(attn, acts, strict=True)]
        l1 = w.first[layer]  # type: ignore[attr-defined]
        last = layer + 1 >= w.num_layers
        if last and self.head_fold:
            yield from self._last_with_head(model, st, xs, hs, l1=l1, hd=w.head, lid=LinearId(layer, LinearKind.O))
            return
        l2 = None if last else w.second[layer + 1]  # type: ignore[attr-defined]
        trips = []
        for x, h in zip(xs, hs, strict=True):
            aq, rq, s, bits = quantize_pair(x, h, l1, l2, ring)
            self._hist(bits)
            kind = "single" if last else "pair"
            trips.append((Fused2Trip(kind, l1, aq, ring.cap, l2, rq), s))
        outs = yield [LinearCall(LinearId(layer, LinearKind.O), t.x, sess, {FUSED2_META: t})
                      for (t, _), sess in zip(trips, st.sessions, strict=True)]
        hs2, nz = [], []
        for y, (t, s) in zip(outs, trips, strict=True):
            if last:
                hs2.append(residual_from(y + t.r, l1, s))  # single: y = hmul * W1 x (no multipliers here)
            else:
                h2 = residual_from(y[:, : l1.m], l1, s)
                hs2.append(h2)
                assert l2 is not None
                nz.append(normed_output(y[:, l1.m :], l2, s, rinv_rows(h2, eps)))
        ex["h"], ex["z"] = hs2, nz
        st.hs = [torch.from_numpy(h.astype(np.float32)).to(model.dtype) for h in hs2]


def enable_fused_parallel(model: Any, weights: ParallelWeights, ring: RingParams, *,
                          head_fold: bool = False) -> ParallelFusedSchedule:
    """Install the parallel-block fused schedule (its backend must take fused2 calls; torch float ops only)."""
    if not getattr(model.backend, "fused2", False):
        raise ConfigError("the model's backend does not take fused2 calls")
    if model.float_ops is not None:
        raise ConfigError("the parallel fused prototype runs on the torch float ops")
    sched = ParallelFusedSchedule(weights, ring, head_fold=head_fold)
    if head_fold:
        model.head_mode = Fused2FoldedHead()
    model.schedule = sched
    return sched


@torch.no_grad()
def parallel_float_reference(pack: Any, ids: torch.Tensor) -> torch.Tensor:
    """Float64 forward of the pack read as a parallel-block model (dequantized int8 linears; all rows' logits).
    The semantic reference the fixed-point schedule approximates (tests: closeness only)."""
    from pgw.engine.rope import RotaryEmbedding

    a = pack.arch
    dt = torch.float64
    load = pack.load_plain
    nh, kvh, hd = a.num_heads, a.num_kv_heads, a.head_dim
    eps = a.rms_norm_eps

    def norm(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return g * x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps)

    def lin(i: int, k: LinearKind, x: torch.Tensor) -> torch.Tensor:
        w, b = _float_linear(pack, i, k)
        y = x @ w.t()
        return y if b is None else y + b.to(dt)

    T = int(ids.numel())
    h = load("embed_tokens", dtype=torch.float32)[ids].to(dt)
    rope = RotaryEmbedding(hd, a.rope_theta, a.rope_scaling, a.max_position_embeddings, "cpu")
    cos, sin = rope.cos_sin(0, T, torch.float64)
    from pgw.engine.rope import apply_rope_reference as apply_rope

    mask = torch.full((T, T), float("-inf"), dtype=dt).triu(1)
    nq = nh * hd
    nk = kvh * hd
    for i in range(a.num_layers):
        g1 = load(f"layers.{i}.input_norm", dtype=torch.float32).to(dt)
        g2 = load(f"layers.{i}.post_attn_norm", dtype=torch.float32).to(dt)
        y = lin(i, LinearKind.QKV, norm(h, g1))
        q = apply_rope(y[:, :nq].reshape(T, nh, hd), cos, sin)
        k = apply_rope(y[:, nq : nq + nk].reshape(T, kvh, hd), cos, sin)
        v = y[:, nq + nk :].reshape(T, kvh, hd)
        rep = nh // kvh
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        att = torch.einsum("thd,shd->hts", q, k) / hd**0.5 + mask
        o = torch.einsum("hts,shd->thd", att.softmax(-1), v).reshape(T, nq)
        gu = lin(i, LinearKind.GATE_UP, norm(h, g2))
        inter = gu.shape[1] // 2
        m = torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:]
        h = h + lin(i, LinearKind.O, o) + lin(i, LinearKind.DOWN, m)
    hn = norm(h, load("final_norm", dtype=torch.float32).to(dt))
    name = "embed_tokens" if getattr(a, "tie_word_embeddings", False) else "lm_head"
    return hn @ load(name, dtype=torch.float32).to(dt).t()
