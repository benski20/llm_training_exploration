"""Mixed sequential / parallel block schedule with the RMSNorm fusion: a per-layer layer map (opt-in, default off).

Each layer is either **sequential** (``"s"``: the checkpoint's own block, two fused2 trips: ``o`` + residual ->
``gate_up``, then ``down`` + residual -> the next layer's input) or **parallel** (``"p"``: ``h' = h + o(attn(n1 h)) +
down(silu(gate) * up)`` with ``[gate | up] = gate_up(n2 h)``, one trip: ``[o | down]`` + residual -> the next layer's
input). A partial parallel conversion keeps the layers that do not tolerate the change sequential
(``research/reviews/parallel-conversion.md``: the per-layer sensitivity and the quality of each map).

Trips per pass with the head folded into the last trip: ``1 + 2 S + P`` (``S`` sequential, ``P`` parallel layers;
the ``1`` is layer 0's input trip, 0 with ``local_qkv0``): ``2L + 1`` for an all-sequential map (fused2), ``L + 1``
for an all-parallel one (:mod:`pgw.engine.fused_parallel`); :func:`trips_per_pass`.

The input stage of layer ``i`` is the second stage of the trip that ends layer ``i - 1`` (layer 0's: a single trip on
the quantized embedding): ``qkv`` with ``n1``'s gain folded (``"i.qkv/f2"``) for a sequential layer, the stacked
``[qkv diag(g1) ; gate_up diag(g2)]`` (``"i.qkv+gate_up/p"``) for a parallel one, both on the residual grid of the
previous trip's first linear. That first linear is ``down`` (``"i.down"``) for a sequential layer and the requantized
``[o | down]`` (``"i.o+down/p"``) for a parallel one; its residual-grid migration multipliers (``bits``) are taken
over the gains of the *next* layer's input stage (``g1``; ``max(|g1|, |g2|)`` when that layer is parallel; the final
norm's with the folded head). Every derivation rule is fused2's (:class:`pgw.modelpack.fused2_weights.Fused2Weights`
with :func:`migration_multipliers`, int8 fold) for sequential layers and the parallel prototype's
(:class:`pgw.engine.fused_parallel.ParallelWeights`) for parallel ones, so an all-``"s"`` map gives fused2's tables
and an all-``"p"`` map the parallel prototype's, bit for bit (tested).

Exactness, masks, privacy and integrity are fused2's: the same trip kinds over the same
:class:`pgw.engine.fused2.Fused2MaskedBackend` (the provider gets ``(R + v - hmul W1 r_x, x + r_x)``, uniform for
any data, each mask row once; every returned ``H``, ``Z`` and logit row passes the range check and the Freivalds
check over ``F_(2^61-1)`` before use). The layer map is public (it fixes the trip count and the item table, which the
provider sees anyway) and pinned: a pack that declares it (``ArchConfig.parallel_layers``, covered by the pack
fingerprint) is refused by every schedule that does not run exactly that map, and the lane fingerprint binds the item
table (item table v4, :func:`pgw.service.fused2_lane.mixed_items`).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
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
    _exact,
    normed_output,
    quantize_pair,
    quantize_single,
    residual_from,
    rinv_rows,
)
from pgw.modelpack.fused2_weights import _fold, _make_linear

__all__ = [
    "MixedFusedSchedule",
    "MixedWeights",
    "enable_fused_mixed",
    "layer_map_from_parallel",
    "mixed_float_reference",
    "normalize_layer_map",
    "pack_layer_map",
    "trips_per_pass",
]

SEQ, PAR = "s", "p"


def layer_map_from_parallel(num_layers: int, parallel_layers: Iterable[int]) -> str:
    """The layer map (``"s"`` / ``"p"`` per layer) with the given layers parallel."""
    par = {int(i) for i in parallel_layers}
    bad = sorted(i for i in par if not 0 <= i < int(num_layers))
    if bad:
        raise ConfigError(f"parallel layers {bad} out of range for {num_layers} layers")
    return "".join(PAR if i in par else SEQ for i in range(int(num_layers)))


def normalize_layer_map(layer_map: str | Sequence[str], num_layers: int) -> str:
    """Canonical layer map: a string of ``num_layers`` characters ``s`` / ``p`` (case-insensitive input)."""
    s = "".join(layer_map).lower() if not isinstance(layer_map, str) else layer_map.lower()
    if len(s) != int(num_layers) or set(s) - {SEQ, PAR}:
        raise ConfigError(f"a layer map is {num_layers} characters 's' (sequential) / 'p' (parallel), got {s!r}")
    return s


def pack_layer_map(pack: Any) -> str:
    """The layer map a pack declares (``arch.parallel_layers``; all sequential when absent)."""
    a = pack.arch
    return layer_map_from_parallel(int(a.num_layers), getattr(a, "parallel_layers", None) or ())


def trips_per_pass(layer_map: str, *, local_qkv0: bool = False, head_trip: bool = False) -> int:
    """Provider round trips per pass: layer 0's input trip (none with ``local_qkv0``) + 2 per sequential layer +
    1 per parallel layer, the head folded into the last trip (``head_trip``: + 1 for a separate head trip)."""
    s = layer_map.count(SEQ)
    p = layer_map.count(PAR)
    return (0 if local_qkv0 else 1) + 2 * s + p + (1 if head_trip else 0)


def _float_linear(pack: Any, layer: int, kind: LinearKind) -> tuple[torch.Tensor, torch.Tensor | None]:
    lw = pack.load_linear(LinearId(layer, kind), device="cpu")
    return lw.w8.to(torch.float64) * lw.scale.to(torch.float64)[:, None], lw.bias


def _multipliers(gain: torch.Tensor | None, sw: torch.Tensor, bits: int) -> torch.Tensor | None:
    """fused2's pack-time migration rule (:func:`pgw.modelpack.fused2_weights.migration_multipliers`)."""
    if gain is None or bits <= 0:
        return None
    c = gain * sw.to(torch.float32).to(torch.float64)
    kappa = float(c.max()) / 2.0 ** bits
    return torch.floor(c / kappa).clamp(min=1.0).to(torch.int64) if kappa > 0 else None


class MixedWeights:
    """Public tables of the mixed fused reference (pack time; no secret).

    ``inp[i]``: layer ``i``'s input stage (folded); ``o[i]`` / ``gu[i]``: a sequential layer's trip A (None for a
    parallel layer); ``end[i]``: the first linear of the trip that ends layer ``i`` (``down`` or ``[o | down]``);
    ``head`` (``head=True``): the folded LM head. ``layer_map`` None takes the pack's declared map
    (:func:`pack_layer_map`); an explicit map must equal a declared non-empty one (a pack without a declared map can
    be *read* with any map: a different model, for measurements and tests; :attr:`pinned` is False then).
    """

    def __init__(self, pack: Any, layer_map: str | Sequence[str] | None = None, *, bits: int = 8,
                 head: bool = False, sigma: bool = True):
        a = pack.arch
        self.num_layers = nl = int(a.num_layers)
        declared = pack_layer_map(pack)
        lm = declared if layer_map is None else normalize_layer_map(layer_map, nl)
        if getattr(a, "parallel_layers", None) and lm != declared:
            raise ConfigError(f"layer map {lm!r} differs from the pack's declared map {declared!r}")
        self.layer_map = lm
        #: True when the map is the pack's declared one (covered by the pack fingerprint).
        self.pinned = lm == declared
        self.fingerprint = str(getattr(pack, "fingerprint", ""))
        self.bits = int(bits)
        self.split_qkv = None
        load = pack.load_plain
        g_in = [load(f"layers.{i}.input_norm", dtype=torch.float32).to(torch.float64) for i in range(nl)]
        g_post = [load(f"layers.{i}.post_attn_norm", dtype=torch.float32).to(torch.float64) for i in range(nl)]
        g_fin = load("final_norm", dtype=torch.float32).to(torch.float64)

        def in_gain(j: int) -> torch.Tensor | None:
            if j >= nl:
                return g_fin.abs() if head else None
            return g_in[j].abs() if lm[j] == SEQ else torch.maximum(g_in[j].abs(), g_post[j].abs())

        self.o: list[F2Linear | None] = []
        self.gu: list[F2Linear | None] = []
        self.end: list[F2Linear] = []
        self.inp: list[F2Linear] = []
        self.head: F2Linear | None = None
        self.qkv_rows = 0
        for i in range(nl):
            if lm[i] == SEQ:
                lo, ld = (pack.load_linear(LinearId(i, k), device="cpu") for k in (LinearKind.O, LinearKind.DOWN))
                self.o.append(_make_linear(f"{i}.o", lo.w8, lo.scale, lo.bias, sigma=sigma,
                                           hmul=_multipliers(g_post[i].abs(), lo.scale, bits)))
                self.end.append(_make_linear(f"{i}.down", ld.w8, ld.scale, ld.bias, sigma=sigma,
                                             hmul=_multipliers(in_gain(i + 1), ld.scale, bits)))
            else:
                wo, bo = _float_linear(pack, i, LinearKind.O)
                wd, bd = _float_linear(pack, i, LinearKind.DOWN)
                if bo is not None or bd is not None:
                    raise ConfigError("mixed fused: a parallel layer's o / down biases are not supported")
                w8, sw = quantize_weight_rows(torch.cat([wo, wd], dim=1))
                self.o.append(None)
                self.end.append(_make_linear(f"{i}.o+down/p", w8, sw, None, sigma=sigma,
                                             hmul=_multipliers(in_gain(i + 1), sw, bits)))
        for i in range(nl):
            grid = self.end[i - 1].grid if i else torch.ones(int(a.hidden_size), dtype=torch.float64)
            wq, bq = _float_linear(pack, i, LinearKind.QKV)
            wg, bg = _float_linear(pack, i, LinearKind.GATE_UP)
            if lm[i] == SEQ:
                o = self.o[i]
                assert o is not None
                fq = _fold(f"{i}.qkv/f2", wq, g_in[i] * grid, bq)
                self.inp.append(fq)
                self.gu.append(_fold(f"{i}.gate_up/f2", wg, g_post[i] * o.grid, bg))
            else:
                fq = _fold(f"{i}.qkv/p", wq, g_in[i] * grid, bq)
                fg = _fold(f"{i}.gate_up/p", wg, g_post[i] * grid, bg)
                bias = None
                if fq.bias is not None or fg.bias is not None:
                    bias = torch.cat([fq.bias if fq.bias is not None else torch.zeros(fq.m, dtype=torch.float64),
                                      fg.bias if fg.bias is not None else torch.zeros(fg.m, dtype=torch.float64)])
                sc = torch.cat([fq.scale, fg.scale])
                st = _make_linear(f"{i}.qkv+gate_up/p", torch.cat([fq.w8, fg.w8], dim=0), sc.to(torch.float32), bias)
                self.inp.append(F2Linear(st.key, st.w8, sc, bias, st.wnorm, st.sigma))
                self.gu.append(None)
            self.qkv_rows = fq.m
        if head:
            name = "embed_tokens" if getattr(a, "tie_word_embeddings", False) else "lm_head"
            fh = load(name, dtype=torch.float32)
            suffix = "f2" if lm[-1] == SEQ else "p"
            self.head = _fold(f"{nl}.lm_head/{suffix}", fh, g_fin * self.end[-1].grid, None)

    @property
    def parallel_layers(self) -> list[int]:
        return [i for i, c in enumerate(self.layer_map) if c == PAR]

    def trips_per_pass(self, *, local_qkv0: bool = False) -> int:
        return trips_per_pass(self.layer_map, local_qkv0=local_qkv0)

    def trip_plan(self) -> list[tuple[F2Linear, F2Linear | None]]:
        """``(l1, l2)`` of every trip of one pass in order (``l2`` None: a single trip; the folded head's pair last),
        the keys :class:`pgw.engine.fused2.Fused2MaskSource` stocks."""
        nl = self.num_layers
        plan: list[tuple[F2Linear, F2Linear | None]] = [(self.inp[0], None)]
        for i in range(nl):
            o, gu = self.o[i], self.gu[i]
            if o is not None and gu is not None:
                plan.append((o, gu))
            plan.append((self.end[i], self.inp[i + 1] if i + 1 < nl else None))
        if self.head is not None:
            plan.append((self.end[-1], self.head))
        return plan

    def all_linears(self) -> list[F2Linear]:
        out: list[F2Linear] = []
        for i in range(self.num_layers):
            out += [lin for lin in (self.inp[i], self.o[i], self.gu[i], self.end[i]) if lin is not None]
        return out + ([self.head] if self.head is not None else [])

    def linears(self) -> list[F2Linear]:
        return self.all_linears()

    def gateway_cost(self) -> dict[str, int]:
        """No split-fold gateway columns in the mixed reference (the interface of ``Fused2Weights.gateway_cost``)."""
        return {"columns": 0, "weight_bytes": 0, "macs_per_token": 0}


class MixedFusedSchedule(Fused2Schedule):
    """:class:`pgw.engine.model.Schedule` of the mixed fused reference: per layer two trips (sequential) or one
    (parallel), plus layer 0's input trip; the layer map is :attr:`layer_map`."""

    name = "fused_mixed"

    def __init__(self, weights: MixedWeights, ring: RingParams, *, head_fold: bool = False,
                 local_qkv0: bool = False):
        if not ring.cap or ring.cap + 1 >= 61:
            raise ConfigError("the mixed fused schedule needs a capped ring with 2^(C+1) < 2^61 - 1")
        if head_fold and weights.head is None:
            raise ConfigError("head_fold needs MixedWeights(head=True)")
        if head_fold and weights.end[-1].sigma <= 0.0:
            raise ConfigError("head_fold needs the last trip's spectral-norm bound (sigma=True)")
        self.w = weights  # type: ignore[assignment]
        self.ring = ring
        self.layer_map = weights.layer_map
        self.head_fold = bool(head_fold)
        self.local_qkv0 = bool(local_qkv0)
        self.head_rows = 0
        self.gateway_macs = 0
        self.h_absmax = None
        self.bits_hist: dict[int, int] = {}
        self.native_decode = None
        self._lock = threading.Lock()

    def runs_layer_map(self) -> str:
        """The layer map this schedule computes (checked against a pack's declared map)."""
        return self.layer_map

    def linears(self, model: Any, layer: int) -> Sequence[LinearId]:
        lids = ([LinearId(layer, LinearKind.O), LinearId(layer, LinearKind.DOWN)] if self.layer_map[layer] == SEQ
                else [LinearId(layer, LinearKind.O)])
        return [LinearId(0, LinearKind.QKV), *lids] if layer == 0 and not self.local_qkv0 else lids

    def layer(self, model: Any, layer: int, st: Any) -> Any:
        w, ring, eps = self.w, self.ring, float(model.eps)
        sessions = st.sessions
        ex = st.extra.setdefault(FUSED2_META, {})
        par = self.layer_map[layer] == PAR
        if layer == 0:
            hs = [h.detach().to(torch.float64).numpy() for h in st.hs]
            lin = w.inp[0]  # type: ignore[attr-defined]
            trips = []
            for h in hs:
                xq, s = quantize_single(torch.from_numpy(h), lin, ring)
                trips.append((Fused2Trip("single", lin, xq, ring.cap), s))
            if self.local_qkv0:  # the same exact integers as the provider's product (no trip, no mask)
                ys = [_exact(t.x, lin, ring) for t, _ in trips]
            else:
                ys = yield [LinearCall(LinearId(0, LinearKind.QKV), t.x, sess, {FUSED2_META: t})
                            for (t, _), sess in zip(trips, sessions, strict=True)]
            zs = [normed_output(y, lin, s, rinv_rows(h, eps), t.x) for y, (t, s), h in zip(ys, trips, hs, strict=True)]
        else:
            hs, zs = ex["h"], ex["z"]
        if par:
            q = w.qkv_rows  # type: ignore[attr-defined]
            attn = model._fused2_attention(layer, st, [z[:, :q] for z in zs])
            inter = model.split_gate_up[0]
            acts = [st.ops.silu_mul(z[:, q:][:, :inter].to(model.dtype), z[:, q:][:, inter:].to(model.dtype))
                    for z in zs]
            xs: list[torch.Tensor] = [torch.cat([a.to(torch.float32), m.to(torch.float32)], dim=1)
                                      for a, m in zip(attn, acts, strict=True)]
            h_in = hs
        else:
            attn = model._fused2_attention(layer, st, zs)
            l1, l2 = w.o[layer], w.gu[layer]  # type: ignore[attr-defined]
            trips = []
            for a, h in zip(attn, hs, strict=True):
                aq, rq, s, bits = quantize_pair(a, h, l1, l2, ring)
                self._hist(bits)
                trips.append((Fused2Trip("pair", l1, aq, ring.cap, l2, rq), s))
            outs = yield [LinearCall(LinearId(layer, LinearKind.O), t.x, sess, {FUSED2_META: t})
                          for (t, _), sess in zip(trips, sessions, strict=True)]
            h_in, xs = [], []
            inter = model.split_gate_up[0]
            for y, (_, s) in zip(outs, trips, strict=True):
                h1 = residual_from(y[:, : l1.m], l1, s)
                self._rec(l1, y[:, : l1.m])
                gu = normed_output(y[:, l1.m :], l2, s, rinv_rows(h1, eps), y[:, : l1.m])
                h_in.append(h1)
                xs.append(st.ops.silu_mul(gu[:, :inter].to(model.dtype), gu[:, inter:].to(model.dtype)))
        # the trip that ends the layer: down (sequential) or [o | down] (parallel) + residual -> next input stage
        l1 = w.end[layer]  # type: ignore[attr-defined]
        lid = LinearId(layer, LinearKind.O if par else LinearKind.DOWN)
        last = layer + 1 >= w.num_layers
        if last and self.head_fold:
            yield from self._last_with_head(model, st, xs, h_in, l1=l1, hd=w.head, lid=lid)
            ex["z"] = []
            return
        l2 = None if last else w.inp[layer + 1]  # type: ignore[attr-defined]
        trips = []
        for x, h in zip(xs, h_in, strict=True):
            aq, rq, s, bits = quantize_pair(x, h, l1, l2, ring)
            self._hist(bits)
            trips.append((Fused2Trip("single", l1, aq, ring.cap, None, rq) if last
                          else Fused2Trip("pair", l1, aq, ring.cap, l2, rq), s))
        outs = yield [LinearCall(lid, t.x, sess, {FUSED2_META: t})
                      for (t, _), sess in zip(trips, sessions, strict=True)]
        hs2, nz = [], []
        for y, (t, s) in zip(outs, trips, strict=True):
            if last:
                hs2.append(residual_from(y + t.r, l1, s))  # the gateway adds the residual itself (exact integers)
            else:
                h2 = residual_from(y[:, : l1.m], l1, s)
                self._rec(l1, y[:, : l1.m])
                hs2.append(h2)
                assert l2 is not None
                nz.append(normed_output(y[:, l1.m :], l2, s, rinv_rows(h2, eps), y[:, : l1.m]))
        ex["h"], ex["z"] = hs2, nz
        st.hs = [torch.from_numpy(h.astype(np.float32)).to(model.dtype) for h in hs2]


def enable_fused_mixed(model: Any, weights: MixedWeights, ring: RingParams, *, head_fold: bool = False,
                       local_qkv0: bool = False) -> MixedFusedSchedule:
    """Install the mixed fused schedule (its backend must take fused2 calls; torch float ops only)."""
    if not getattr(model.backend, "fused2", False):
        raise ConfigError("the model's backend does not take fused2 calls")
    if model.float_ops is not None:
        raise ConfigError("the mixed fused schedule runs on the torch float ops")
    if int(model.num_layers) != weights.num_layers:
        raise ConfigError("mixed fused weights of another depth")
    sched = MixedFusedSchedule(weights, ring, head_fold=head_fold, local_qkv0=local_qkv0)
    if head_fold:
        if getattr(model, "lm_head_fn", None) is not None:
            raise ConfigError("the folded head does not combine with a draft head (lm_head_fn)")
        model.head_mode = Fused2FoldedHead()
    model.schedule = sched
    return sched


@torch.no_grad()
def mixed_float_reference(pack: Any, ids: torch.Tensor, layer_map: str) -> torch.Tensor:
    """Float64 forward of the pack with the given layer map (dequantized int8 linears; all rows' logits): the
    semantic reference the fixed-point schedule approximates (tests: closeness only)."""
    from pgw.engine.rope import RotaryEmbedding, apply_rope_reference

    a = pack.arch
    lm = normalize_layer_map(layer_map, int(a.num_layers))
    dt = torch.float64
    load = pack.load_plain
    nh, kvh, hd = a.num_heads, a.num_kv_heads, a.head_dim
    eps = a.rms_norm_eps

    def norm(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return g * x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps)

    def lin(i: int, k: LinearKind, x: torch.Tensor) -> torch.Tensor:
        wf, b = _float_linear(pack, i, k)
        y = x @ wf.t()
        return y if b is None else y + b.to(dt)

    T = int(ids.numel())
    h = load("embed_tokens", dtype=torch.float32)[ids].to(dt)
    rope = RotaryEmbedding(hd, a.rope_theta, a.rope_scaling, a.max_position_embeddings, "cpu")
    cos, sin = rope.cos_sin(0, T, torch.float64)
    mask = torch.full((T, T), float("-inf"), dtype=dt).triu(1)
    nq, nk = nh * hd, kvh * hd
    qk = getattr(a.arch, "value", a.arch) == "qwen3"
    for i in range(a.num_layers):
        g1 = load(f"layers.{i}.input_norm", dtype=torch.float32).to(dt)
        g2 = load(f"layers.{i}.post_attn_norm", dtype=torch.float32).to(dt)
        y = lin(i, LinearKind.QKV, norm(h, g1))
        q = y[:, :nq].reshape(T, nh, hd)
        k = y[:, nq : nq + nk].reshape(T, kvh, hd)
        if qk:
            q = norm(q, load(f"layers.{i}.q_norm", dtype=torch.float32).to(dt))
            k = norm(k, load(f"layers.{i}.k_norm", dtype=torch.float32).to(dt))
        q, k = apply_rope_reference(q, cos, sin), apply_rope_reference(k, cos, sin)
        v = y[:, nq + nk :].reshape(T, kvh, hd)
        rep = nh // kvh
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        att = torch.einsum("thd,shd->hts", q, k) / hd**0.5 + mask
        o = lin(i, LinearKind.O, torch.einsum("hts,shd->thd", att.softmax(-1), v).reshape(T, nq))
        h1 = h + o
        gu = lin(i, LinearKind.GATE_UP, norm(h if lm[i] == PAR else h1, g2))
        inter = gu.shape[1] // 2
        h = h1 + lin(i, LinearKind.DOWN, torch.nn.functional.silu(gu[:, :inter]) * gu[:, inter:])
    hn = norm(h, load("final_norm", dtype=torch.float32).to(dt))
    name = "embed_tokens" if getattr(a, "tie_word_embeddings", False) else "lm_head"
    return hn @ load(name, dtype=torch.float32).to(dt).t()
