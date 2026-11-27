"""Fused two-trip layer ("fused2", WP-B4 prototype): 2 provider round trips per decoder layer.

Default off. This is a **new pinned reference model** (fixed-point residual stream, norm gains folded into
the following linears); its logits differ from today's int8 reference (``PlainBackend`` + standard
schedule). Design, proofs and quality numbers: ``research/plans/fused2-two-trips-per-layer.md``.

Per layer the standard schedule makes four trips (qkv, o, gate_up, down). fused2 makes two:

* **trip A** ``(R, aq) -> (H1, Z)``: ``H1 = R + Wo8 aq`` (o-proj plus the residual add, exact integers),
  ``Z = Wgu'8 H1`` (gate_up applied to the *un-normalized* residual, RMSNorm gain folded into ``Wgu'``);
* **trip B** ``(R', act_q) -> (H2, Z')``: ``H2 = R' + Wd8 act_q``, ``Z' = Wqkv'8(next layer) H2``.

plus one trip for layer 0's qkv (``2L + 1`` trips per token; the last layer's trip B is the plain down
product, the gateway adds the residual). RMSNorm is linear except for the per-row scalar ``1/rms``, which
the gateway applies *after* the product: ``W diag(g) x / rms(x)``; it computes ``rms`` from the exact
``H`` it gets back in the same trip.

Fixed-point semantics (normative for this reference; every rounding is done by the gateway):

* ``aq, s = quant(a, A_t)``: per-row absmax, ``A_t`` bits (``A_t <= A``, adaptive, see below).
* The residual is requantized by the gateway at the start of each trip to the output grid of the trip's
  first linear: ``R[j] = rint(h[j] / (sw1[j] * s))``, so ``R + W1_8 aq`` is exactly ``(h + W1 a_q) /
  (sw1 * s)`` channel-wise, no rounding on the provider.
* ``W2'`` = int8 per-row quantization of ``W2 diag(g * sw1)`` (``g`` the RMSNorm gain, ``sw1`` the first
  linear's row scales). Layer 0's qkv uses ``W diag(g)`` and the per-row absmax quantized embedding.
  With residual-grid migration (``Fused2Weights(multipliers=...)``, e.g. :func:`migration_multipliers`; the
  quality fix of 2026-11-23) the first linear's output is ``hmul * (W1_8 aq)`` and every ``sw1`` above reads
  ``sw1 / hmul`` (the residual's grid), so ``W2' = int8(W2 diag(g * sw1 / hmul))`` has near-uniform columns.
* After the trip: ``h' = (H * sw1) * s`` (float64), ``rinv = 1/sqrt(mean(h'^2) + eps)``,
  ``z = (Z * sw2) * (s * rinv) + bias`` (float64, then float32).
* ``A_t`` is the most bits ``<= A`` for which the exact a-priori bounds
  ``max|R| + wn1 ||aq||_2 <= 2^C`` and ``wn2 (||R||_2 + sigma1 ||aq||_2) <= 2^C`` hold
  (``wn`` the largest row norm, ``sigma1`` an upper bound on the spectral norm of ``W1_8``), so every
  genuine output decodes (``2^C < 2^(K-1)``) and the range check uses ``2^C``.

Masks (the chain correction costs nothing extra): with ``r_a`` and ``v`` uniform and independent, the
gateway masks ``R`` with ``r_R = v - W1 r_a``. The provider computes ``H_hat = R_hat + W1 a_hat = H + v`` and
``Z_hat = W2' H_hat = Z + W2' v``. So the corrections are ``v`` and ``W2' v``: two *independent*
single-stage products (``W1 r_a`` and ``W2' v``), exactly what a slot holds today for o and gate_up; no
product chain, no extra homomorphic depth. ``(R_hat, a_hat)`` is uniform for any data (``r_R`` is uniform and
independent of ``r_a`` because ``v`` is).

Integrity: both stages are checked on the unmasked integers with Freivalds over ``F_p``, ``p = 2^61 - 1``
(``u1 . (H - R) == (u1 W1) . aq`` and ``u2 . Z == (u2 W2') . H``) plus the range check ``|H|, |Z| <= 2^C``;
``2^(C+1) < p`` makes each check sound to ``log2 p`` bits per vector (the p31 field cannot hold these bounds).

Prototype scope: the plain reference backend and a masked backend whose masks come from an in-process
plain factory (:class:`Fused2MaskSource`) and whose provider is the in-process :class:`Fused2Provider`
(exact ring products on masked integers only). Wire protocol, segment format and the store are follow-ups.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, IntegrityError, RingOverflowError
from pgw.core.prng import SeedExpander
from pgw.core.ring import RingParams
from pgw.core.ringmm import PreparedWeight
from pgw.core.types import LinearId, LinearKind
from pgw.engine.backends import BackendStats, LinearBackend, LinearCall, exact_int_matmul
from pgw.modelpack.fused2_weights import (
    F2Linear,
    Fused2Weights,
    hf_float_linears,
    migration_multipliers,
    select_gateway_columns,
)

__all__ = [
    "FUSED2_META",
    "P61",
    "F2Linear",
    "Fused2FoldedHead",
    "Fused2MaskSource",
    "Fused2MaskedBackend",
    "Fused2PlainBackend",
    "Fused2Provider",
    "Fused2Schedule",
    "Fused2Trip",
    "Fused2Weights",
    "calibrate_fused2",
    "enable_fused2",
    "gateway_product",
    "hf_float_linears",
    "migration_multipliers",
    "select_gateway_columns",
]

#: Mersenne prime of the fused2 Freivalds field.
P61 = (1 << 61) - 1
#: ``LinearCall.meta`` key carrying the :class:`Fused2Trip` of a fused2 call.
FUSED2_META = "fused2"
#: Relative slack on float64 norm bounds (as ``pgw.core.ring._NORM_SLACK``).
_SLACK = 1.0 + 1e-9
# The public weights (F2Linear, Fused2Weights, migration_multipliers, ...) live in pgw.modelpack.fused2_weights
# (no engine imports: the compute service derives its lane tables from them) and are re-exported here.


# ---- gateway arithmetic (shared by the plain and masked backends) ----------------------------------
@dataclass
class Fused2Trip:
    """One fused2 provider call (rows of one sequence).

    ``kind = "pair"``: returns ``[H | Z]`` with ``H = r + l1.w8 x``, ``Z = l2.w8 H``.
    ``kind = "single"``: returns ``Y = l1.w8 x`` (layer 0's qkv; the last layer's down).
    ``x`` int64 ``(T, n1)`` (``|x| <= 2^C``), ``r`` int64 ``(T, m1)`` (pair only). Every genuine output is
    bounded by ``2^cap`` (chosen by the gateway before the trip).
    """

    kind: str
    l1: F2Linear
    x: torch.Tensor
    cap: int
    l2: F2Linear | None = None
    r: torch.Tensor | None = None

    @property
    def out_features(self) -> int:
        return self.l1.m + (self.l2.m if self.l2 is not None else 0)


def _quant_rows(xf: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    qmax = (1 << (bits - 1)) - 1
    amax = np.abs(xf).max(axis=1)
    s = np.where(amax > 0, amax / qmax, 1.0)
    q = np.clip(np.rint(xf / s[:, None]), -qmax, qmax)
    return q, s


def quantize_single(x: torch.Tensor, lin: F2Linear, ring: RingParams) -> tuple[torch.Tensor, np.ndarray]:
    """Per-row absmax quantization with the most bits ``<= A`` such that ``wnorm ||xq||_2 <= 2^C``."""
    xf = x.detach().to(torch.float64).numpy()
    if not np.isfinite(xf).all():
        raise RingOverflowError("non-finite activation values cannot be quantized")
    rows = xf.shape[0]
    xq = np.zeros(xf.shape, dtype=np.int64)
    s = np.ones(rows)
    todo = np.arange(rows)
    limit = float(1 << ring.cap)
    for bits in range(ring.a, 1, -1):
        q, sb = _quant_rows(xf[todo], bits)
        fits = np.sqrt((q * q).sum(axis=1)) * lin.wnorm * _SLACK <= limit
        idx = todo[fits]
        xq[idx] = q[fits].astype(np.int64)
        s[idx] = sb[fits]
        todo = todo[~fits]
        if not todo.size:
            break
    if todo.size:
        raise RingOverflowError(f"fused2: {lin.key} input cannot be bounded by 2^{ring.cap}")
    return torch.from_numpy(xq), s


def quantize_pair(a: torch.Tensor, h: np.ndarray, l1: F2Linear, l2: F2Linear | None,
                  ring: RingParams) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Gateway quantization of a trip: ``(aq, R, s, bits)`` for activations ``a`` and residual ``h`` (float64).

    Per row the most bits ``A_t <= A`` such that ``max|R| + wn1 ||aq|| <= 2^C`` and (with ``l2``)
    ``wn2 (||R|| + sigma1 ||aq||) <= 2^C`` (for a split fold: ``R`` and ``sigma1`` restricted to the
    provider's channels). ``R`` is on ``l1``'s residual grid ``sw1 / hmul``.
    """
    af = a.detach().to(torch.float64).numpy()
    if not (np.isfinite(af).all() and np.isfinite(h).all()):
        raise RingOverflowError("non-finite values cannot be quantized")
    rows = af.shape[0]
    aq = np.zeros(af.shape, dtype=np.int64)
    rq = np.zeros(h.shape, dtype=np.int64)
    s = np.ones(rows)
    bits_used = np.zeros(rows, dtype=np.int64)
    todo = np.arange(rows)
    limit = float(1 << ring.cap)
    sw1 = l1.grid.numpy()
    keep = None if l2 is None or l2.keep is None or not l2.sigma_in else l2.keep.numpy()
    for bits in range(ring.a, 1, -1):
        q, sb = _quant_rows(af[todo], bits)
        rr = np.rint(h[todo] / (sw1[None, :] * sb[:, None]))
        na = np.sqrt((q * q).sum(axis=1))
        b_h = np.abs(rr).max(axis=1) + l1.wnorm * na
        fits = b_h * _SLACK <= limit
        if l2 is not None:
            if keep is None:
                b_z = l2.wnorm * (np.sqrt((rr * rr).sum(axis=1)) + l1.sigma * na)
            else:  # Z = W2'[:, keep] H[keep] (W2' is zero on the gateway columns)
                rk = rr[:, keep]
                b_z = l2.wnorm * (np.sqrt((rk * rk).sum(axis=1)) + l2.sigma_in * na)
            fits &= b_z * _SLACK <= limit
        idx = todo[fits]
        aq[idx] = q[fits].astype(np.int64)
        rq[idx] = rr[fits].astype(np.int64)
        s[idx] = sb[fits]
        bits_used[idx] = bits
        todo = todo[~fits]
        if not todo.size:
            break
    if todo.size:
        raise RingOverflowError(f"fused2: {l1.key} trip cannot be bounded by 2^{ring.cap}")
    return torch.from_numpy(aq), torch.from_numpy(rq), s, bits_used


def residual_from(hq: torch.Tensor, lin: F2Linear, s: np.ndarray) -> np.ndarray:
    """``h = (H * grid) * s`` in float64 (the fixed-point residual as floats; ``grid = sw / hmul``)."""
    return (hq.numpy().astype(np.float64) * lin.grid.numpy()[None, :]) * s[:, None]


def rinv_rows(h: np.ndarray, eps: float) -> np.ndarray:
    """``1 / sqrt(mean(h^2) + eps)`` per row, float64."""
    return 1.0 / np.sqrt((h * h).mean(axis=1) + eps)


def gateway_product(xin: torch.Tensor, lin: F2Linear) -> torch.Tensor:
    """Exact int64 ``xin[:, gcols] @ gw^T`` (the gateway columns of a split fold; ``xin`` is the verified
    ``H`` of the trip, or layer 0's quantized embedding). :class:`Fused2Schedule` checks the bound
    ``g * qmax_g * 2^C < 2^63``, so the int64 sums cannot wrap."""
    assert lin.gcols is not None and lin.gw is not None
    return xin.index_select(1, lin.gcols) @ lin.gw.t()


def normed_output(zq: torch.Tensor, lin: F2Linear, s: np.ndarray, rinv: np.ndarray,
                  xin: torch.Tensor | None = None) -> torch.Tensor:
    """``z = (Z * sw [+ Zg * sg]) * (s * rinv) + bias`` as float32 (``Zg`` the gateway columns' product)."""
    zf = zq.numpy().astype(np.float64) * lin.scale.numpy()[None, :]
    if lin.gcols is not None:
        if xin is None:
            raise ConfigError(f"fused2 {lin.key}: the gateway columns need the trip's input")
        assert lin.gscale is not None
        zf = zf + gateway_product(xin, lin).numpy().astype(np.float64) * lin.gscale.numpy()[None, :]
    z = zf * (s * rinv)[:, None]
    if lin.bias is not None:
        z = z + lin.bias.numpy()[None, :]
    return torch.from_numpy(z.astype(np.float32))


# ---- the schedule ---------------------------------------------------------------------------------
class Fused2Schedule:
    """:class:`pgw.engine.model.Schedule` for fused2 (2 trips per layer, plus layer 0's qkv).

    The pass state's residual ``st.hs`` holds the float32 cast of the fixed-point residual after each layer
    (for drafters); the exact float64 residual and the next layer's qkv travel in ``st.extra``.
    """

    name = "fused2"

    def __init__(self, weights: Fused2Weights, ring: RingParams, *, head_fold: bool = False,
                 local_qkv0: bool = False):
        if not ring.cap:
            raise ConfigError("fused2 needs a capped ring (e.g. K48A12C45)")
        if ring.cap + 1 >= 61:
            raise ConfigError("fused2 cap must satisfy 2^(C+1) < 2^61 - 1 (Freivalds field)")
        for lin in weights.qkv + weights.gu:
            if lin.gw is not None and (int(lin.gw.shape[1]) * ((1 << (lin.gbits - 1)) - 1)
                                       * (1 << ring.cap)) >= 1 << 63:
                raise ConfigError(f"fused2 {lin.key}: gateway columns' product can exceed int64 at this cap")
        if head_fold and weights.head is None:
            raise ConfigError("fused2 head_fold needs weights with the folded head (Fused2Weights(head=True))")
        if head_fold and weights.down[-1].sigma <= 0.0:
            raise ConfigError("fused2 head_fold needs the last down's spectral-norm bound (sigma=True)")
        self.w = weights
        self.ring = ring
        #: Opt-in (default off): the LM head is the second stage of the last trip (last down + folded head) for
        #: the rows that need logits; the model's head mode must be :class:`Fused2FoldedHead`. 2L + 1 trips per
        #: pass with the head on the provider (instead of 2L + 2 with a separate masked head trip).
        self.head_fold = bool(head_fold)
        #: Opt-in (default off): layer 0's qkv product (the folded ``W_qkv0 diag(g)`` times the quantized embedding
        #: row, a public int8 matrix: 1 / 17 / 37 / 84 MB at 0.5B / 7B / 32B / 72B) is computed exactly on the
        #: gateway: the same integers, one trip fewer per pass, the provider sees strictly less.
        self.local_qkv0 = bool(local_qkv0)
        #: Logit rows the folded head computed (diagnostics).
        self.head_rows = 0
        #: Gateway integer MACs of split-fold columns (cost accounting).
        self.gateway_macs = 0
        #: When a dict: per first-linear key, the per-channel max ``|H| / hmul`` seen (calibration only).
        self.h_absmax: dict[str, torch.Tensor] | None = None
        #: Activation bits used per trip row (diagnostics: ``{bits: rows}``).
        self.bits_hist: dict[int, int] = {}
        #: The native decode driver (:mod:`pgw.engine.fused2_native`, opt-in) or None.
        self.native_decode: Any = None
        self._lock = threading.Lock()

    def linears(self, model: Any, layer: int) -> Sequence[LinearId]:
        lids = [LinearId(layer, LinearKind.O), LinearId(layer, LinearKind.DOWN)]
        return [LinearId(0, LinearKind.QKV), *lids] if layer == 0 and not self.local_qkv0 else lids

    def _hist(self, bits: np.ndarray) -> None:
        with self._lock:
            for b in bits.tolist():
                self.bits_hist[int(b)] = self.bits_hist.get(int(b), 0) + 1

    def _rec(self, l1: F2Linear, hq: torch.Tensor) -> None:
        if self.h_absmax is None or not hq.numel():
            return
        a = hq.abs().to(torch.float64)
        if l1.hmul is not None:
            a = a / l1.hmul.to(torch.float64)
        a = a.amax(dim=0)
        with self._lock:
            old = self.h_absmax.get(l1.key)
            self.h_absmax[l1.key] = a if old is None else torch.maximum(old, a)

    def _gw(self, lin: F2Linear, trips: Sequence[tuple[Any, np.ndarray]]) -> None:
        if lin.gw is not None:
            rows = sum(int(s.shape[0]) for _, s in trips)
            with self._lock:
                self.gateway_macs += rows * int(lin.gw.numel())

    def layer(self, model: Any, layer: int, st: Any) -> Any:
        w, ring, eps = self.w, self.ring, float(model.eps)
        sessions = st.sessions
        ex = st.extra.setdefault(FUSED2_META, {})
        if layer == 0:
            hs = [h.detach().to(torch.float64).numpy() for h in st.hs]
            lin = w.qkv[0]
            trips = []
            for h in hs:
                xq, s = quantize_single(torch.from_numpy(h), lin, ring)
                trips.append((Fused2Trip("single", lin, xq, ring.cap), s))
            if self.local_qkv0:  # the same exact integers as the provider's product (no trip, no mask)
                ys = [_exact(t.x, lin, ring) for t, _ in trips]
            else:
                ys = yield [LinearCall(LinearId(0, LinearKind.QKV), t.x, sess, {FUSED2_META: t})
                            for (t, _), sess in zip(trips, sessions, strict=True)]
            qkv = [normed_output(y, lin, s, rinv_rows(h, eps), t.x)
                   for y, (t, s), h in zip(ys, trips, hs, strict=True)]
            self._gw(lin, trips)
        else:
            hs, qkv = ex["h"], ex["qkv"]
        attn = model._fused2_attention(layer, st, qkv)
        # trip A: o + residual add -> gate_up (post-attention norm folded)
        l1, l2 = w.o[layer], w.gu[layer]
        trips = []
        for a, h in zip(attn, hs, strict=True):
            aq, rq, s, bits = quantize_pair(a, h, l1, l2, ring)
            self._hist(bits)
            trips.append((Fused2Trip("pair", l1, aq, ring.cap, l2, rq), s))
        outs = yield [LinearCall(LinearId(layer, LinearKind.O), t.x, sess, {FUSED2_META: t})
                      for (t, _), sess in zip(trips, sessions, strict=True)]
        hs1, acts = [], []
        inter = model.split_gate_up[0]
        for y, (_, s) in zip(outs, trips, strict=True):
            h1 = residual_from(y[:, : l1.m], l1, s)
            self._rec(l1, y[:, : l1.m])
            gu = normed_output(y[:, l1.m :], l2, s, rinv_rows(h1, eps), y[:, : l1.m])
            hs1.append(h1)
            self._gw(l2, [(None, s)])
            acts.append(st.ops.silu_mul(gu[:, :inter].to(model.dtype), gu[:, inter:].to(model.dtype)))
        # trip B: down + residual add -> next layer's qkv (input norm folded); the last layer: down only
        l1 = w.down[layer]
        last = layer + 1 >= w.num_layers
        if last and self.head_fold:
            yield from self._last_with_head(model, st, acts, hs1)
            return
        l2 = None if last else w.qkv[layer + 1]
        trips = []
        for a, h in zip(acts, hs1, strict=True):
            aq, rq, s, bits = quantize_pair(a, h, l1, l2, ring)
            self._hist(bits)
            trip = (Fused2Trip("single", l1, aq, ring.cap, None, rq) if last
                    else Fused2Trip("pair", l1, aq, ring.cap, l2, rq))
            trips.append((trip, s))
        outs = yield [LinearCall(LinearId(layer, LinearKind.DOWN), t.x, sess, {FUSED2_META: t})
                      for (t, _), sess in zip(trips, sessions, strict=True)]
        hs2, nxt = [], []
        for y, (t, s) in zip(outs, trips, strict=True):
            if last:
                hq = y + t.r  # the gateway adds the residual itself (exact integers)
                hs2.append(residual_from(hq, l1, s))
            else:
                h2 = residual_from(y[:, : l1.m], l1, s)
                self._rec(l1, y[:, : l1.m])
                hs2.append(h2)
                assert l2 is not None
                nxt.append(normed_output(y[:, l1.m :], l2, s, rinv_rows(h2, eps), y[:, : l1.m]))
                self._gw(l2, [(None, s)])
        ex["h"], ex["qkv"] = hs2, nxt
        st.hs = [torch.from_numpy(h.astype(np.float32)).to(model.dtype) for h in hs2]


    def _last_with_head(self, model: Any, st: Any, acts: list[torch.Tensor], hs1: list[np.ndarray],
                        l1: F2Linear | None = None, hd: F2Linear | None = None, lid: LinearId | None = None) -> Any:
        """The last layer's trip B with the folded head: per sequence a pair call ``(down, head)`` for the rows
        that need logits (the last row; every row with ``all_logits``; the last ``last_logits`` rows) and a single
        call (down, multipliers applied) for the other rows, all in one trip. ``Z`` of the head rows is the
        int8 head of the final-normed residual: ``logits = (Z * sw_head) * (s / rms(h))``."""
        w, ring, eps = self.w, self.ring, float(model.eps)
        l1 = w.down[-1] if l1 is None else l1
        hd = w.head if hd is None else hd
        assert hd is not None
        ex = st.extra[FUSED2_META]
        lid = LinearId(w.num_layers - 1, LinearKind.DOWN) if lid is None else lid
        calls: list[LinearCall] = []
        # (head rows, (trip, s) of the other rows or None, (trip, s) of the head rows)
        plan: list[tuple[int, Any, Any]] = []
        for seq, a, h, sess in zip(st.seqs, acts, hs1, st.sessions, strict=True):
            rows = int(a.shape[0])
            nh = rows if seq.all_logits else (min(rows, int(seq.last_logits)) if seq.last_logits > 0 else 1)
            aq, rq, s, bits = quantize_pair(a, h, l1, hd, ring)  # every row bounded for both stages
            self._hist(bits)
            cut = rows - nh
            lo = None
            if cut:
                lo = (Fused2Trip("single", l1, aq[:cut], ring.cap, None, rq[:cut]), s[:cut])
                calls.append(LinearCall(lid, lo[0].x, sess, {FUSED2_META: lo[0]}))
            hi = (Fused2Trip("pair", l1, aq[cut:], ring.cap, hd, rq[cut:]), s[cut:])
            calls.append(LinearCall(lid, hi[0].x, sess, {FUSED2_META: hi[0]}))
            plan.append((nh, lo, hi))
        outs = yield calls
        at = 0
        hs2, logits = [], []
        for nh, lo, hi in plan:
            parts = []
            if lo is not None:
                y = outs[at]
                at += 1
                parts.append(residual_from(y + lo[0].r, l1, lo[1]))  # y includes hmul (single with multipliers)
            y = outs[at]
            at += 1
            h2 = residual_from(y[:, : l1.m], l1, hi[1])
            parts.append(h2)
            logits.append(normed_output(y[:, l1.m :], hd, hi[1], rinv_rows(h2, eps)))
            self.head_rows += nh
            hs2.append(np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0])
        ex["h"], ex["qkv"] = hs2, []
        ex["logits"] = logits
        st.hs = [torch.from_numpy(h.astype(np.float32)).to(model.dtype) for h in hs2]


class Fused2FoldedHead:
    """:class:`pgw.engine.model.HeadMode` of :attr:`Fused2Schedule.head_fold`: the logits come from the last trip
    (no head trip of its own; :meth:`linears` is empty). Same output forms as the gateway head: the last row's
    logits, all rows' (``all_logits``) or :class:`pgw.engine.model.RowLogits` (``last_logits``)."""

    name = "fused2_folded"
    #: :meth:`pgw.engine.model.TransformerModel._steps` hands this head the pass state (the last trip's logits).
    from_pass_state = True
    supports_pad_rows = False
    rows_exact = True

    def linears(self, model: Any) -> Sequence[LinearId]:
        return ()

    def logits_from(self, model: Any, seqs: Sequence[Any], st: Any) -> list[Any]:
        from pgw.engine.model import RowLogits

        got = st.extra.get(FUSED2_META, {}).get("logits")
        if got is None or len(got) != len(seqs):
            raise ConfigError("the fused2 folded head needs the fused2 schedule with head_fold=True")
        out: list[Any] = []
        for s, h, lg in zip(seqs, st.hs, got, strict=True):
            if s.last_logits > 0:
                n = int(lg.shape[0])
                out.append(RowLogits(h[-n:], model.final_norm, model.eps, model._head, lg))
            else:
                out.append(lg if s.all_logits else lg[-1])
        return out

    def steps(self, model: Any, seqs: Sequence[Any], hs: list[torch.Tensor]) -> Any:  # pragma: no cover
        raise ConfigError("the fused2 folded head runs from the pass state (Fused2FoldedHead.logits_from)")
        yield


def enable_fused2(model: Any, weights: Fused2Weights, ring: RingParams, *, head_fold: bool = False,
                  local_qkv0: bool = False) -> Fused2Schedule:
    """Install the fused2 schedule on ``model`` (its backend must take fused2 calls)."""
    if not getattr(model.backend, "fused2", False):
        raise ConfigError("the model's backend does not take fused2 calls")
    if model.float_ops is not None:
        from pgw.engine.model import NativeFloatOps

        # the native float ops (attention, SiLU in pgw_fast) are their own pinned reference, the one the native
        # fused2 decode driver (pgw.engine.fused2_native) reproduces bit for bit
        if not isinstance(model.float_ops, NativeFloatOps):
            raise ConfigError("fused2 runs on the torch or the native float ops")
    sched = Fused2Schedule(weights, ring, head_fold=head_fold, local_qkv0=local_qkv0)
    if head_fold:
        if getattr(model, "lm_head_fn", None) is not None:
            raise ConfigError("the fused2 folded head does not combine with a draft head (lm_head_fn)")
        model.head_mode = Fused2FoldedHead()
    model.schedule = sched
    return sched


def calibrate_fused2(pack: Any, float_linears: Mapping[str, torch.Tensor] | None, calib: Sequence[torch.Tensor],
                     ring: RingParams, *, bits: int = 6, columns: int = 4, headroom_bits: int = 0,
                     gateway_bits: int = 16) -> dict[str, Any]:
    """Pack-time choice of the migrated split fold (``Fused2Weights(multipliers=..., gateway_columns=...)``).

    Why: folding ``g * sw1`` into the next linear before per-row int8 rounding makes the folded matrix's
    column scales ``c = g * sw1`` vary 10-30x within a row, so its int8 rows are coarse for most columns
    (measured: the whole top-1 gap of the unsplit fold on Qwen2.5-0.5B). Per trip's first linear and its
    fold, with ``c_j = |g_j| * sw1_j``:

    1. ``columns`` gateway columns: the largest column absmax of ``|W| * c`` (they set the row scales);
    2. integer multipliers ``hmul_j = max(1, floor(c_j / kappa))``, ``kappa = max_{j kept} c_j / 2^bits``,
       so the provider's columns see ``c_j / hmul_j`` within ``[kappa, 2 kappa)`` (the int8 rows of the
       folded matrix then look like today's per-row int8 weights);
    3. channels whose migrated residual would exceed the largest ``|H|`` seen on ``calib`` (the reference
       without migration, all trips, times ``2^headroom_bits``) keep ``hmul = 1`` and go to the gateway
       too (Qwen's massive-activation channels), so ``H`` gains no bits.

    ``calib`` is public calibration text (token ids; not the evaluation data). The result is public and
    deterministic for a pack + calibration set. Layer 0's qkv takes step 1 only; the last down none.
    """
    from pgw.engine.model import TransformerModel

    base = Fused2Weights(pack, float_linears)
    model = TransformerModel(pack, Fused2PlainBackend(ring))
    sched = enable_fused2(model, base, ring)
    sched.h_absmax = {}
    with torch.no_grad():
        for ids in calib:
            model.forward(ids, model.new_cache(len(ids) + 1))
    hmax = sched.h_absmax
    h_glob = max(float(v.max()) for v in hmax.values()) * 2.0 ** headroom_bits
    nl = base.num_layers
    g_in = [pack.load_plain(f"layers.{i}.input_norm", dtype=torch.float32).to(torch.float64) for i in range(nl)]
    g_post = [pack.load_plain(f"layers.{i}.post_attn_norm", dtype=torch.float32).to(torch.float64) for i in range(nl)]

    def float_w(layer: int, kind: LinearKind) -> torch.Tensor:
        if float_linears is not None:
            return float_linears[f"{layer}.{kind.value}"].to(torch.float64)
        lw = pack.load_linear(LinearId(layer, kind), device="cpu")
        return lw.w8.to(torch.float64) * lw.scale.to(torch.float64)[:, None]

    mults: dict[str, torch.Tensor] = {}
    gcols: dict[str, torch.Tensor] = {}
    gcols["0.qkv/f2"] = select_gateway_columns(float_w(0, LinearKind.QKV), g_in[0], columns)
    for i in range(nl):
        pairs = [(base.o[i], g_post[i], f"{i}.gate_up/f2", (i, LinearKind.GATE_UP))]
        if i + 1 < nl:
            pairs.append((base.down[i], g_in[i + 1], f"{i + 1}.qkv/f2", (i + 1, LinearKind.QKV)))
        for l1, g, fkey, (fl, fk) in pairs:
            c = g.abs() * l1.scale
            wf = float_w(fl, fk)
            score = (wf.abs() * c[None, :]).amax(dim=0)
            del wf
            keep = torch.ones_like(c, dtype=torch.bool)
            keep[score.topk(min(columns, int(c.numel()))).indices] = False
            kappa = float(c[keep].max()) / 2.0 ** bits
            m = torch.floor(c / kappa).clamp(min=1.0) if kappa > 0 else torch.ones_like(c)
            seen = hmax.get(l1.key, torch.zeros_like(c))
            keep &= ~(m * seen > h_glob)
            m[~keep] = 1.0
            mults[l1.key] = m.to(torch.int64)
            gcols[fkey] = (~keep).nonzero().flatten()
    cols_per = [int(v.numel()) for v in gcols.values()]
    return {"multipliers": mults, "gateway_columns": gcols, "gateway_bits": gateway_bits,
            "stats": {"h_max_bits": math.log2(max(h_glob, 1.0)) - headroom_bits,
                      "gateway_columns_max": max(cols_per), "gateway_columns_mean": sum(cols_per) / len(cols_per),
                      "hmul_max": max(int(v.max()) for v in mults.values())}}


# ---- exact integer products --------------------------------------------------------------------------
_PW_CACHE: dict[int, PreparedWeight] = {}
_PW_LOCK = threading.Lock()


def _prepared(lin: F2Linear) -> PreparedWeight:
    key = id(lin.w8)
    pw = _PW_CACHE.get(key)
    if pw is None or pw.w8 is not lin.w8:  # the id of a freed tensor can be reused
        with _PW_LOCK:
            pw = _PW_CACHE.get(key)
            if pw is None or pw.w8 is not lin.w8:
                pw = _PW_CACHE[key] = PreparedWeight(lin.w8, cache_float=False)
    return pw


def _exact(x: torch.Tensor, lin: F2Linear, ring: RingParams) -> torch.Tensor:
    """Exact ``x @ W8^T`` for a genuine product bounded by ``2^C < 2^(K-1)``: the ring product, centered."""
    xmax = int(x.abs().max() if x.numel() else 0)
    if xmax <= ring.qmax:
        return exact_int_matmul(x, lin.w8t, ring.a)  # small inputs (activations): the int8 digit path
    if xmax * 128 * lin.n < 1 << 53:  # every partial sum an exact float64 integer: the same integers
        return torch.round(x.to(torch.float64) @ lin.w8t.to(torch.float64)).to(torch.int64)
    return _center(_prepared(lin).matmul(x, ring.k), ring.k)


class Fused2PlainBackend(LinearBackend):
    """The fused2 reference: exact integer products on the gateway (no masks, no provider)."""

    fused2 = True

    def __init__(self, ring: RingParams):
        self.ring = ring
        self.stats = BackendStats()
        #: Largest ``log2 |output|`` seen per trip kind (``"single"``, ``"H"``, ``"Z"``): diagnostics only.
        self.max_bits: dict[str, float] = {}

    def _note(self, name: str, y: torch.Tensor) -> None:
        if y.numel():
            b = math.log2(max(1, int(y.abs().max())))
            if b > self.max_bits.get(name, 0.0):
                self.max_bits[name] = b

    def _one(self, t: Fused2Trip) -> torch.Tensor:
        y1 = _exact(t.x, t.l1, self.ring)
        if t.kind == "single":
            if t.l1.hmul is not None:  # a single trip of a first linear with multipliers (head_fold's other rows)
                y1 = y1 * t.l1.hmul
            self._note("single", y1)
            return y1
        assert t.r is not None and t.l2 is not None
        h = t.r + (y1 if t.l1.hmul is None else y1 * t.l1.hmul)
        z = _exact(h, t.l2, self.ring)
        self._note("H", h)
        self._note("Z", z)
        return torch.cat([h, z], dim=1)

    def forward(self, call: LinearCall) -> torch.Tensor:
        trip = call.meta.get(FUSED2_META) if call.meta else None
        if not isinstance(trip, Fused2Trip):
            raise ConfigError("Fused2PlainBackend takes fused2 calls only")
        self.stats.add(calls=1, rows=int(call.x.shape[0]))
        return self._one(trip)

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        return [self.forward(c) for c in calls]


# ---- arithmetic in F_p, p = 2^61 - 1 -----------------------------------------------------------------
def _limbs16(v: np.ndarray) -> list[np.ndarray]:
    """Values in ``[0, 2^64)`` (int64 holding values < 2^63) as four 16-bit limbs (int64)."""
    return [(v >> (16 * i)) & 0xFFFF for i in range(4)]


def _limbs21(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Values in ``[0, 2^63)`` as three 21-bit limbs (int64; the top one < 2^21 for values < 2^63)."""
    m = (1 << 21) - 1
    return v & m, (v >> 21) & m, v >> 42


def _mod_p61(y: np.ndarray) -> np.ndarray:
    """``y mod (2^61 - 1)`` for int64 ``|y| < 2^62`` (Mersenne fold; ~2.5x faster than ``np.mod``)."""
    r = (y & P61) + (y >> 61)  # y = hi * 2^61 + lo == hi + lo (mod p); hi in [-2, 1]
    r = np.where(r < 0, r + P61, r)
    return np.where(r >= P61, r - P61, r)


def dot_mod_p(y: np.ndarray, u: np.ndarray, p: int = P61, *,
              ul: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """``(y @ u^T) mod p`` exactly for int64 ``y (T, n)`` (any sign, ``|y| < 2^62``) and ``u (c, n)`` in ``[0, p)``.

    Both sides (reduced to ``[0, p)``, 61 bits) are split in three 21-bit limbs; each limb product is < 2^42, so a
    product group ``sum_{i+j=s} y_i . u_j`` over ``n <= 2^19`` terms (at most three products per group) stays
    below 2^63: int64 sums are exact. The five groups are recombined with Python integers on the small
    ``(T, c)`` result. Longer rows fall back to the 16-bit limb path (:func:`_dot_mod_p16`). Returns int64
    ``(T, c)`` in ``[0, p)``. ``ul``: the limbs of ``u`` (:func:`_limbs21`), when the caller caches them.
    """
    if y.shape[-1] > (1 << 19) or p != P61:
        return _dot_mod_p16(y, u, p)
    yl = _limbs21(_mod_p61(y.astype(np.int64)))
    if ul is None:
        ul = _limbs21(u.astype(np.int64))
    out = np.zeros((y.shape[0], u.shape[0]), dtype=object)
    # few rows (decode): elementwise products and sums vectorize far better than numpy's integer matmul
    small = y.shape[0] * u.shape[0] * y.shape[-1] <= 1 << 21
    for s in range(5):
        acc = None
        for i in range(max(0, s - 2), min(2, s) + 1):
            if small:
                t = (yl[i][:, None, :] * ul[s - i][None, :, :]).sum(axis=2)
            else:
                t = yl[i] @ ul[s - i].T
            acc = t if acc is None else acc + t
        assert acc is not None
        out += acc.astype(object) * (1 << (21 * s))
    return np.asarray(out % p, dtype=np.int64)


def _dot_mod_p16(y: np.ndarray, u: np.ndarray, p: int = P61) -> np.ndarray:
    """:func:`dot_mod_p` with 16-bit limbs (each limb product < 2^32; int64 sums of up to 2^31 terms exact)."""
    yp = np.mod(y.astype(np.int64), p)
    yl, ul = _limbs16(yp), _limbs16(u.astype(np.int64))
    out = np.zeros((y.shape[0], u.shape[0]), dtype=object)
    for i, a in enumerate(yl):
        for j, b in enumerate(ul):
            out += (a @ b.T).astype(object) * (1 << (16 * (i + j)))
    return np.asarray(out % p, dtype=np.int64)


def matmul_mod_p(u: np.ndarray, w8: torch.Tensor, p: int = P61) -> np.ndarray:
    """``u @ W8 mod p`` for ``u (c, m)`` in ``[0, p)`` and int8 ``W8 (m, n)`` (factory precompute)."""
    w = w8.to(torch.int64).numpy()
    out = np.zeros((u.shape[0], w.shape[1]), dtype=object)
    for i, a in enumerate(_limbs16(u.astype(np.int64))):
        out += (a @ w).astype(object) * (1 << (16 * i))
    return np.asarray(out % p, dtype=np.int64)


# ---- masks: the plain factory path (prototype) -----------------------------------------------------
@dataclass
class _Checks:
    u: np.ndarray  # (c, m) in [0, p)
    uw: np.ndarray  # (c, n) = u W8 mod p
    _ul: Any = None  # cached 21-bit limbs of u and uw (dot_mod_p)
    _uwl: Any = None

    def limbs(self) -> tuple[Any, Any]:
        if self._ul is None:
            self._ul = _limbs21(self.u.astype(np.int64))
            self._uwl = _limbs21(self.uw.astype(np.int64))
        return self._ul, self._uwl


@dataclass
class TripMasks:
    """Masks of one trip (``rows`` rows). Secret: never leaves the gateway / factory."""

    r_x: torch.Tensor  # (T, n1) uniform
    wr_x: torch.Tensor  # (T, m1) = W1 r_x mod 2^K (factory)
    v: torch.Tensor | None = None  # (T, m1) uniform (pair)
    wv: torch.Tensor | None = None  # (T, m2) = W2 v mod 2^K (factory)
    #: Check vectors of the segment the rows came from (store masks; None: the source's static vectors).
    #: Owned by the mask source (not wiped with the rows).
    c1: Any = None
    c2: Any = None

    def wipe(self) -> None:
        for t in (self.r_x, self.wr_x, self.v, self.wv):
            if t is not None:
                t.zero_()


class Fused2MaskSource:
    """In-process plain mask factory for fused2 (prototype of the segment content).

    Per linear (or pair) key and slot, ``r_x`` and ``v`` are expanded from a fresh 32-byte seed
    (:class:`pgw.core.prng.SeedExpander`); the corrections ``W1 r_x`` and ``W2' v`` are two independent
    single-stage products, as for today's o and gate_up rows. Rows are single use: a cursor per trip key
    only moves forward. Check vectors are uniform in ``F_p`` (secret) with ``u W8 mod p`` precomputed.
    """

    def __init__(self, weights: Fused2Weights, ring: RingParams, *, checks: int = 1, seed: bytes | None = None):
        self.w = weights
        self.ring = ring
        self.checks = int(checks)
        if self.checks < 1:
            raise ConfigError("fused2 needs at least one check vector")
        self._sx = SeedExpander(seed if seed is not None else os.urandom(32))
        self._cursor: dict[str, int] = {}
        self._lock = threading.Lock()
        self._pw: dict[str, PreparedWeight] = {}
        self._checks: dict[str, _Checks] = {}
        #: Factory work done: ring-product MACs per trip kind (cost accounting).
        self.macs = 0
        #: Precomputed rows per trip key (:meth:`prefill`): ``[(first row, TripMasks)]`` in row order.
        self._stock: dict[str, list[tuple[int, TripMasks]]] = {}
        self.prefill_s = 0.0
        self.stock_hits = 0
        self.stock_misses = 0

    def _prepared(self, lin: F2Linear) -> PreparedWeight:
        pw = self._pw.get(lin.key)
        if pw is None:
            pw = self._pw[lin.key] = PreparedWeight(lin.w8, cache_float=False)
        return pw

    def check_vectors(self, lin: F2Linear) -> _Checks:
        hit = self._checks.get(lin.key)
        if hit is None:
            raw = self._sx.ring_uniform((self.checks, lin.m), 64, "u-p61", lin.key).numpy().view(np.uint64)
            u = (raw % np.uint64(P61)).astype(np.int64)
            # u . (H - R) == u . (hmul * W1 x) == ((u * hmul) W1) . x  (mod p)
            uh = u if lin.hmul is None else (u.astype(object) * lin.hmul.numpy().astype(object)[None, :] % P61
                                              ).astype(np.int64)
            hit = self._checks[lin.key] = _Checks(u, matmul_mod_p(uh, lin.w8))
        return hit

    def trip_plan(self) -> list[tuple[F2Linear, F2Linear | None]]:
        """The ``(l1, l2)`` of every trip of one pass, in order (``l2`` None: a single trip)."""
        w = self.w
        plan: list[tuple[F2Linear, F2Linear | None]] = [(w.qkv[0], None)]
        for i in range(w.num_layers):
            plan.append((w.o[i], w.gu[i]))
            plan.append((w.down[i], w.qkv[i + 1] if i + 1 < w.num_layers else None))
        if w.head is not None:  # head_fold: the last trip's logit rows are a (down, head) pair of their own key
            plan.append((w.down[-1], w.head))
        return plan

    def prefill(self, rows: int) -> float:
        """Compute the next ``rows`` rows of every trip key ahead of use (batched products), so a trip's
        :meth:`take` is a slice. The same rows :meth:`take` would compute lazily (same seeds, cursors);
        still single use. Returns the seconds spent. One "slot" = one row of every trip key (one token
        row of one pass): ``r_x``, ``W1 r_x``, ``v``, ``W2' v``, element counts equal to today's slot."""
        import time as _time

        t0 = _time.perf_counter()
        k = self.ring.k
        for l1, l2 in self.trip_plan():
            key = l1.key + ("+" + l2.key if l2 is not None else "")
            with self._lock:
                stock = self._stock.setdefault(key, [])
                start = self._cursor.get(key, 0) + sum(int(s[1].r_x.shape[0]) for s in stock)
            r_x = self._sx.ring_uniform_rows(l1.n, k, start, rows, "f2", "r", key)
            wr_x = self._prepared(l1).matmul(r_x, k)
            self.macs += rows * l1.n * l1.m
            tm = TripMasks(r_x, wr_x)
            if l2 is not None:
                tm.v = self._sx.ring_uniform_rows(l1.m, k, start, rows, "f2", "v", key)
                tm.wv = self._prepared(l2).matmul(tm.v, k)
                self.macs += rows * l2.n * l2.m
            with self._lock:
                stock.append((start, tm))
        self.prefill_s += _time.perf_counter() - t0
        return _time.perf_counter() - t0

    def stocked_rows(self) -> int:
        """Rows ready for every trip key (the minimum over keys; 0 without :meth:`prefill`)."""
        with self._lock:
            if not self._stock:
                return 0
            return min(sum(int(s[1].r_x.shape[0]) for s in st) for st in self._stock.values())

    def _from_stock(self, key: str, start: int, rows: int) -> TripMasks | None:
        st = self._stock.get(key)
        if not st:
            return None
        parts: list[TripMasks] = []
        need, pos = rows, start
        while need:
            if not st or st[0][0] != pos:
                break
            s0, tm = st[0]
            n = int(tm.r_x.shape[0])
            take = min(need, n)
            parts.append(TripMasks(*(None if t is None else t[:take].clone() for t in (tm.r_x, tm.wr_x, tm.v, tm.wv))))
            if take == n:
                st.pop(0)
                tm.wipe()
            else:
                rest = TripMasks(*(None if t is None else t[take:].clone() for t in (tm.r_x, tm.wr_x, tm.v, tm.wv)))
                tm.wipe()
                st[0] = (s0 + take, rest)
            need -= take
            pos += take
        if need:
            for p in parts:
                p.wipe()
            return None
        if len(parts) == 1:
            return parts[0]
        cat = [None if parts[0].v is None and i >= 2 else torch.cat([getattr(p, f) for p in parts])
               for i, f in enumerate(("r_x", "wr_x", "v", "wv"))]
        for p in parts:
            p.wipe()
        return TripMasks(*cat)

    def take(self, trip: Fused2Trip) -> TripMasks:
        rows = int(trip.x.shape[0])
        key = trip.l1.key + ("+" + trip.l2.key if trip.l2 is not None else "")
        with self._lock:
            start = self._cursor.get(key, 0)
            self._cursor[key] = start + rows
            hit = self._from_stock(key, start, rows)
        if hit is not None:
            self.stock_hits += 1
            return hit
        self.stock_misses += 1
        k = self.ring.k
        r_x = self._sx.ring_uniform_rows(trip.l1.n, k, start, rows, "f2", "r", key)
        wr_x = self._prepared(trip.l1).matmul(r_x, k)
        self.macs += rows * trip.l1.n * trip.l1.m
        if trip.l2 is None:
            return TripMasks(r_x, wr_x)
        v = self._sx.ring_uniform_rows(trip.l1.m, k, start, rows, "f2", "v", key)
        wv = self._prepared(trip.l2).matmul(v, k)
        self.macs += rows * trip.l2.n * trip.l2.m
        return TripMasks(r_x, wr_x, v, wv)


# ---- the provider (in-process; sees masked integers and public ids only) ------------------------------
class Fused2Provider:
    """The compute service's fused2 op: exact products mod ``2^K`` of public int8 weights.

    ``pair(k1, k2, r_hat, x_hat) -> (H_hat, Z_hat)`` with ``H_hat = r_hat + hmul1 * (W1 x_hat)`` and
    ``Z_hat = W2 H_hat`` (``hmul1`` the first linear's public multipliers, 1 by default);
    ``single(k1, x_hat) -> W1 x_hat``. Inputs are uniform ring elements; the provider holds no secret.
    """

    def __init__(self, weights: Fused2Weights, k: int):
        self.k = int(k)
        self._w = {lin.key: PreparedWeight(lin.w8, cache_float=False) for lin in weights.all_linears()}
        self._hmul = {lin.key: lin.hmul for lin in weights.all_linears() if lin.hmul is not None}
        #: Every array the provider received (tests inspect what it saw).
        self.seen: list[tuple[str, torch.Tensor]] = []
        self.record = False

    def single(self, k1: str, x_hat: torch.Tensor) -> torch.Tensor:
        if self.record:
            self.seen.append((k1, x_hat.clone()))
        y = self._w[k1].matmul(x_hat, self.k)
        hm = self._hmul.get(k1)
        if hm is not None:  # a first linear's single trip (head_fold's rows without logits): multipliers too
            y = (y * hm) & ((1 << self.k) - 1)
        return y

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.record:
            self.seen.append((k1, x_hat.clone()))
            self.seen.append((k1 + ":r", r_hat.clone()))
        mask = (1 << self.k) - 1
        y = self._w[k1].matmul(x_hat, self.k)
        hm = self._hmul.get(k1)
        if hm is not None:
            y = (y * hm) & mask  # public per-channel multipliers (the residual grid), exact mod 2^K
        h_hat = (r_hat + y) & mask
        return h_hat, self._w[k2].matmul(h_hat, self.k)


def _center(v: torch.Tensor, k: int) -> torch.Tensor:
    v = v & ((1 << k) - 1)
    half = 1 << (k - 1)
    return torch.where(v >= half, v - (1 << k), v)


@dataclass
class _F2Stats:
    trips: int = 0
    rows: int = 0
    bytes_up: int = 0
    bytes_down: int = 0
    checks: int = 0


class Fused2MaskedBackend(LinearBackend):
    """fused2 over a provider: mask, call, unmask, range + Freivalds (``F_p``) checks on every output.

    Bit-identical to :class:`Fused2PlainBackend` (the same integers) for an honest provider; any wrong
    ``H``, ``Z`` or factory correction is caught (probability ``1 - 1/p`` per check vector) before use.
    """

    fused2 = True

    def __init__(self, ring: RingParams, provider: Any, masks: Fused2MaskSource | None = None, *,
                 weights: Fused2Weights | None = None, noise_monitor: Any = None,
                 on_integrity_failure: Any = None):
        if not ring.cap or ring.cap + 1 >= 61:
            raise ConfigError("fused2 needs a capped ring with 2^(C+1) < 2^61 - 1")
        self.ring = ring
        self.provider = provider
        #: Optional :class:`pgw.observability.OutboundNoiseMonitor` (the gateway's): every trip's ``x_hat``
        #: passes its every-call checks (duplicate rows) and sampled statistics before it leaves; the native
        #: driver (:mod:`pgw.engine.fused2_native`) uses the same monitor. Observes only; never changes a value.
        self.noise_monitor = noise_monitor
        #: ``fn(error)`` called on a failed range / Freivalds check (the gateway trips its circuit breaker).
        self.on_integrity_failure = on_integrity_failure
        #: The default mask source (in-process / segment file source; or None: every call's session is a
        #: mask-store reservation of fused2 pair-mask segments, :mod:`pgw.engine.fused2_store`).
        self.masks = masks
        #: The public fused2 tables (needed for store sessions: keys and shapes; never ``w8``).
        self.weights = weights if weights is not None else getattr(masks, "w", None)
        self.stats = BackendStats()
        self.f2 = _F2Stats()
        self._store_lock = threading.Lock()
        #: id(session) -> (session, Fused2StoreMasks) of the store sessions seen (dropped once closed)
        self._store: dict[int, tuple[Any, Any]] = {}

    def masks_for(self, session: Any) -> Any:
        """The mask source of a call: a store session's :class:`pgw.engine.fused2_store.Fused2StoreMasks`, else
        the backend's default source."""
        from pgw.engine.fused2_store import Fused2StoreMasks, is_store_session

        if is_store_session(session):
            with self._store_lock:
                hit = self._store.get(id(session))
                if hit is not None and hit[0] is session:
                    return hit[1]
                for sid, (sess, src) in list(self._store.items()):  # sweep closed sessions: wipe their copies
                    if getattr(sess, "closed", False):
                        src.wipe()
                        del self._store[sid]
                if self.weights is None:
                    raise ConfigError("fused2 store masks need the backend's weights (Fused2MaskedBackend(weights=))")
                src = Fused2StoreMasks(session, self.weights, self.ring)
                self._store[id(session)] = (session, src)
                return src
        if self.masks is None:
            raise ConfigError("fused2 call without a mask source (no store session and no default source)")
        return self.masks

    def release(self, session: Any) -> None:
        """Forget a store session (wipes this backend's copies of its check vectors)."""
        with self._store_lock:
            hit = self._store.pop(id(session), None)
        if hit is not None:
            hit[1].wipe()

    def sync_trip_ok(self, rows: int) -> bool:
        """The engine's ``sync_decode`` probe: a small pass runs wholly on the plaintext thread (where the native
        fused2 driver may take it) when the provider is a synchronous lane channel."""
        return hasattr(self.provider, "ch") and 0 < int(rows) <= int(getattr(self.provider, "max_rows", 0) or 0)

    def release_session_checks(self, session: Any) -> None:
        """The engine's / gateway's end-of-session hook (:class:`pgw.engine.backends.MaskedBackend` name):
        :meth:`release`, and the native driver's copies of a closed session's check vectors."""
        self.release(session)
        nd = getattr(self, "native_decode", None)
        if nd is not None and getattr(session, "closed", False):
            nd.wipe_checks()

    def _verify(self, lin: F2Linear, xq: torch.Tensor, yq: torch.Tensor, offset: torch.Tensor | None = None,
                ch: Any = None, masks: Any = None) -> None:
        """Range check ``|yq| <= 2^C`` and ``u . (yq - offset) == (u W8) . xq (mod p)`` for every row."""
        bound = 1 << self.ring.cap
        if ch is None:
            src = masks if masks is not None else self.masks
            if src is None:
                raise ConfigError(f"fused2 {lin.key}: no check vectors (no mask source)")
            ch = src.check_vectors(lin)
        nat = _native_p61()
        if nat is not None and yq.numel():  # the same predicate in Rust (pgw_fast.p61_check), GIL released
            c = np.ascontiguousarray
            st = nat(c(yq.numpy()), None if offset is None else c(offset.numpy()), c(ch.u), c(xq.numpy()),
                     c(ch.uw), bound)
            self.f2.checks += 1
            if st == 2:
                raise IntegrityError(f"fused2 {lin.key}: output out of range")
            if st != 0:
                raise IntegrityError(f"fused2 {lin.key}: Freivalds check failed")
            return
        if int(yq.abs().max()) > bound if yq.numel() else False:
            raise IntegrityError(f"fused2 {lin.key}: output out of range")
        y = yq if offset is None else yq - offset
        ul, uwl = ch.limbs()
        lhs = dot_mod_p(y.numpy(), ch.u, ul=ul)
        rhs = dot_mod_p(xq.numpy(), ch.uw, ul=uwl)
        self.f2.checks += 1
        if not np.array_equal(lhs, rhs):
            raise IntegrityError(f"fused2 {lin.key}: Freivalds check failed")

    def _one(self, t: Fused2Trip, src: Any = None) -> torch.Tensor:
        src = self.masks if src is None else src
        if int(t.x.abs().max()) > (1 << self.ring.cap) if t.x.numel() else False:
            raise RingOverflowError("fused2 input out of range")
        if not getattr(src, "partial", False):
            return self._masked(t, src.take(t), src)
        # store masks: a take never crosses a segment extent, so the trip may run in row pieces (rows are
        # independent; every piece is masked, checked and its masks wiped like a whole trip)
        rows = int(t.x.shape[0])
        if rows == 0:  # nothing to mask (no row is taken)
            return torch.zeros((0, t.out_features), dtype=torch.int64)
        outs: list[torch.Tensor] = []
        a = 0
        while a < rows:
            sub = t if a == 0 else Fused2Trip(t.kind, t.l1, t.x[a:], t.cap, t.l2, None if t.r is None else t.r[a:])
            m = src.take(sub)
            n = int(m.r_x.shape[0])
            if n < rows - a:
                sub = Fused2Trip(t.kind, t.l1, t.x[a:a + n], t.cap, t.l2, None if t.r is None else t.r[a:a + n])
            outs.append(self._masked(sub, m, src))
            a += n
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

    def _prep(self, t: Fused2Trip, m: TripMasks) -> tuple[str, str | None, torch.Tensor, torch.Tensor | None]:
        """The provider request of one trip piece: ``(k1, k2, x_hat, r_hat)`` (masked; the noise monitor sees
        ``x_hat`` before it leaves)."""
        k = self.ring.k
        mask = (1 << k) - 1
        x_hat = (t.x + m.r_x) & mask
        noise = self.noise_monitor
        if noise is not None:  # duplicate rows (mask reuse) and sampled uniformity, before the payload leaves
            noise.guard(x_hat, t.x)
        bpe = k // 8
        self.f2.trips += 1
        self.f2.rows += int(t.x.shape[0])
        if t.kind == "single":
            self.f2.bytes_up += x_hat.numel() * bpe
            return t.l1.key, None, x_hat, None
        assert t.r is not None and t.l2 is not None and m.v is not None and m.wv is not None
        wr = m.wr_x if t.l1.hmul is None else (m.wr_x * t.l1.hmul) & mask
        r_r = (m.v - wr) & mask  # r_R = v - hmul * W1 r_x: then H_hat - v = R + hmul * W1 x exactly
        r_hat = (t.r + r_r) & mask
        self.f2.bytes_up += (x_hat.numel() + r_hat.numel()) * bpe
        return t.l1.key, t.l2.key, x_hat, r_hat

    def _finish(self, t: Fused2Trip, m: TripMasks, out: Any, src: Any) -> torch.Tensor:
        """Unmask and check the provider's answer to one trip piece (range + ``F_p`` Freivalds, both stages)."""
        k = self.ring.k
        mask = (1 << k) - 1
        bpe = k // 8
        if t.kind == "single":
            y_hat = out
            self.f2.bytes_down += y_hat.numel() * bpe
            # a first linear with multipliers (head_fold's rows without logits): the provider returns
            # hmul * W1 x_hat, so the correction is hmul * W1 r_x; the check vectors are (u * hmul) W1
            wr = m.wr_x if t.l1.hmul is None else (m.wr_x * t.l1.hmul) & mask
            y = _center(y_hat - wr, k)
            self._verify(t.l1, t.x, y, ch=m.c1, masks=src)
            return y
        assert m.v is not None and m.wv is not None and t.l2 is not None
        h_hat, z_hat = out
        self.f2.bytes_down += (h_hat.numel() + z_hat.numel()) * bpe
        h = _center(h_hat - m.v, k)
        self._verify(t.l1, t.x, h, offset=t.r, ch=m.c1, masks=src)  # stage 1: H - R == hmul * W1 x
        z = _center(z_hat - m.wv, k)
        self._verify(t.l2, h, z, ch=m.c2, masks=src)  # stage 2: Z == W2' H (H verified)
        return torch.cat([h, z], dim=1)

    def _ask(self, req: tuple[str, str | None, torch.Tensor, torch.Tensor | None]) -> Any:
        k1, k2, x_hat, r_hat = req
        if k2 is None:
            return self.provider.single(k1, x_hat)
        return self.provider.pair(k1, k2, r_hat, x_hat)

    def _masked(self, t: Fused2Trip, m: TripMasks, src: Any) -> torch.Tensor:
        try:
            return self._finish(t, m, self._ask(self._prep(t, m)), src)
        finally:
            m.wipe()

    def _pieces(self, t: Fused2Trip, src: Any) -> list[tuple[Fused2Trip, TripMasks]]:
        """The trip's row pieces with their masks (store masks: a take never crosses a segment extent)."""
        if int(t.x.abs().max()) > (1 << self.ring.cap) if t.x.numel() else False:
            raise RingOverflowError("fused2 input out of range")
        if not getattr(src, "partial", False):
            return [(t, src.take(t))]
        rows = int(t.x.shape[0])
        out: list[tuple[Fused2Trip, TripMasks]] = []
        a = 0
        try:
            while a < rows:
                sub = t if a == 0 else Fused2Trip(t.kind, t.l1, t.x[a:], t.cap, t.l2,
                                                  None if t.r is None else t.r[a:])
                m = src.take(sub)
                n = int(m.r_x.shape[0])
                if n < rows - a:
                    sub = Fused2Trip(t.kind, t.l1, t.x[a:a + n], t.cap, t.l2, None if t.r is None else t.r[a:a + n])
                out.append((sub, m))
                a += n
        except BaseException:
            for _, m in out:
                m.wipe()
            raise
        return out

    def forward_many_multi(self, calls: Sequence[LinearCall]) -> list[torch.Tensor] | None:
        """Several fused2 calls of one step (the last trip's single and head-fold items; the sequences of a
        batched step) in **one** lane-format-2 multi call when the provider offers it: one round trip instead of
        one per call. Every piece is masked, checked and its masks wiped exactly as one call per piece (the same
        rows, bodies and checks). ``None``: not applicable (the caller runs one call per call)."""
        prov = self.provider
        if len(calls) < 2 or not callable(getattr(prov, "many", None)):
            return None
        trips = []
        for c in calls:
            trip = c.meta.get(FUSED2_META) if c.meta else None
            if not isinstance(trip, Fused2Trip) or int(trip.x.shape[0]) == 0:
                return None
            trips.append(trip)
        pieces: list[tuple[int, Fused2Trip, TripMasks, Any]] = []
        try:
            for i, (c, trip) in enumerate(zip(calls, trips, strict=True)):
                src = self.masks_for(c.session)
                for sub, m in self._pieces(trip, src):
                    pieces.append((i, sub, m, src))
            reqs = [self._prep(t, m) for _, t, m, _ in pieces]
            if prov.multi_ok(reqs):
                answers = prov.many(reqs)
            else:  # more items or rows than one multi call carries: one call per piece (same bodies)
                answers = [self._ask(r) for r in reqs]
            outs: list[list[torch.Tensor]] = [[] for _ in calls]
            for (i, t, m, src), ans in zip(pieces, answers, strict=True):
                outs[i].append(self._finish(t, m, ans, src))
            for c in calls:
                self.stats.add(calls=1, rows=int(c.x.shape[0]))
            return [o[0] if len(o) == 1 else torch.cat(o, dim=0) for o in outs]
        except IntegrityError as exc:
            self.integrity_failed(exc)
            raise
        finally:
            for _, _, m, _ in pieces:
                m.wipe()

    def forward(self, call: LinearCall) -> torch.Tensor:
        trip = call.meta.get(FUSED2_META) if call.meta else None
        if not isinstance(trip, Fused2Trip):
            raise ConfigError("Fused2MaskedBackend takes fused2 calls only")
        self.stats.add(calls=1, rows=int(call.x.shape[0]))
        try:
            return self._one(trip, self.masks_for(call.session))
        except IntegrityError as exc:
            self.integrity_failed(exc)
            raise

    def integrity_failed(self, exc: BaseException) -> None:
        """Report a failed check to :attr:`on_integrity_failure` (never masks the original error)."""
        fn = self.on_integrity_failure
        if fn is not None:
            try:
                fn(exc)
            except Exception:  # pragma: no cover - defensive
                logging.getLogger(__name__).exception("fused2 integrity-failure callback failed")

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        got = self.forward_many_multi(calls)
        if got is not None:
            return got
        return [self.forward(c) for c in calls]


def _native_p61() -> Any:
    """``pgw_fast.p61_check`` when the extension has it and ``PGW_FUSED2_NATIVE_CHECK`` is not 0 (bit-identical
    verdicts to the numpy path; default on when available)."""
    if os.environ.get("PGW_FUSED2_NATIVE_CHECK", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    from pgw.core.native import fast

    return getattr(fast, "p61_check", None) if fast is not None else None


def soundness_bits(checks: int = 1, p: int = P61) -> float:
    """Bits per call of the fused2 check (per stage; both stages must pass)."""
    return checks * (math.log2(p) - math.log2(1 + 2.0**-3))

