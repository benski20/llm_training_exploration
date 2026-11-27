"""In-process narrow-wire masked backend (opt-in prototype; ``pgw.core.narrow``, ``consumer-bandwidth.md``).

Runs every linear the way a narrow-wire trip would, end to end in one process, and counts the wire bytes:

1. client: quantize exactly as :class:`pgw.engine.backends.PlainBackend` (same function, same bits);
   draw a fresh mask ``r`` in ``[0, 2^m)`` from a ChaCha20 stream (never reused: a per-call counter);
   ``x_hat = (xq + r) mod 2^m``; pack at ``m`` bits;
2. provider (:func:`pgw.core.narrow.provider_apply`): sees only the packed ``x_hat``; returns the low ``w_j`` bits of
   ``W8 x_hat`` per output row, packed;
3. client: subtract the mask correction ``(W8 r) mod 2^w_j`` (made on the device here: the factory's product),
   add the carry term (carry mode only), centre, range-check, and verify with the prime-field Freivalds check
   (``p = 2^31 - 1``, secret ``u``) before the dequantization (same function as the plain backend).

Outputs are bit-identical to :class:`PlainBackend` (tested on the tiny pack and Qwen2.5-0.5B). Not wired into the
lane, the service or the mask store: a measurement and exactness prototype, off by default.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from pgw.core.errors import IntegrityError, ShapeError
from pgw.core.freivalds import P31, prime_check, prime_precompute
from pgw.core.narrow import (
    NarrowPlan,
    carry_budget,
    carry_term,
    decode_outputs,
    mask_correction,
    mask_inputs,
    pack_up,
    plan_linear,
    provider_apply,
    unpack_down,
    unpack_up,
)
from pgw.core.prng import SeedExpander, new_seed
from pgw.core.ring import RingParams, dequantize_outputs, quantize_activations
from pgw.core.types import LinearId
from pgw.engine.backends import BackendStats, LinearBackend, LinearCall, _grouped_forward, _to_int_weight, _WeightCache

__all__ = ["NarrowMaskedBackend", "NarrowWireStats"]


@dataclass
class NarrowWireStats:
    """Wire bytes of the narrow path and of today's ``K``-bit wire for the same calls."""

    rows: int = 0
    up: int = 0
    down: int = 0
    k_up: int = 0
    k_down: int = 0
    carries: int = 0  # non-zero carries (client-local, never sent)
    carry_slots: int = 0  # gathered columns including padding (constant per row)
    carry_overflows: int = 0
    per_linear: dict[str, list[int]] = field(default_factory=dict)  # kind -> [rows, up, down, k_up, k_down]


@dataclass
class _Lin:
    w: Any  # _IntWeight
    plan: NarrowPlan
    u: np.ndarray  # (1, m) secret check vector in [0, p)
    uw: np.ndarray  # (1, n) u W8 mod p
    budget: int


class NarrowMaskedBackend(LinearBackend):
    """Exact masked linears over narrow wire rings (see the module docs).

    Args:
        pack: Engine pack (as :class:`PlainBackend`).
        ring: Capped ring (K32A12C27 by default in the engine).
        in_bits: ``None`` = weight-free mode (``m = max_j w_j``); an int = carry mode (``ring.a <= m``).
        provider: Optional replacement of :func:`provider_apply` (tests inject tampering providers).
        fail_log2: Target probability (log2) that a row overflows the padded carry budget.
    """

    rows_exact = True

    def __init__(self, pack: object, ring: RingParams, *, in_bits: int | None = None, seed: bytes | None = None,
                 provider: Callable[[torch.Tensor, np.ndarray, NarrowPlan], bytes] | None = None,
                 fail_log2: float = -40.0):
        if not ring.cap:
            raise ShapeError("the narrow prototype needs a capped ring (outputs bounded per call)")
        self.pack = pack
        self.ring = ring
        self.in_bits = in_bits
        self.stats = BackendStats()
        self.wire = NarrowWireStats()
        self._weights = _WeightCache(pack, torch.device("cpu"), _to_int_weight)
        self._lins: dict[str, _Lin] = {}
        self._lock = threading.Lock()
        self._sx = SeedExpander(seed if seed is not None else new_seed())
        self._ctr = 0
        self._provider = provider or provider_apply
        self._fail = fail_log2
        #: Optional ``fn(linear_id, xq)`` called with every call's quantized input (measurement only).
        self.observe: Callable[[LinearId, np.ndarray], None] | None = None

    def _lin(self, lid: LinearId) -> _Lin:
        hit = self._lins.get(lid.key)
        if hit is not None:
            return hit
        w = self._weights.get(lid)
        w8 = w.w8t.t().contiguous()  # (m, n)
        bits = None if self.in_bits is None else max(self.ring.a, min(self.in_bits, 64))
        plan = plan_linear(w8, self.ring, None)
        if bits is not None and bits < plan.in_bits:
            plan = plan_linear(w8, self.ring, bits)
        u = (self._sx.ring_uniform((1, w.out_features), 64, "narrow-u", lid.key).numpy().view(np.uint64)
             % np.uint64(P31)).astype(np.int64)
        uw = prime_precompute(torch.from_numpy(u), w8).numpy()
        budget = carry_budget(w.in_features, plan.in_bits, self.ring.qmax, self._fail) if plan.carry_rows.size else 0
        lin = _Lin(w, plan, u, uw, budget)
        self._lins[lid.key] = lin
        return lin

    def plan(self, lid: LinearId) -> NarrowPlan:
        return self._lin(lid).plan

    def forward(self, call: LinearCall) -> torch.Tensor:
        t0 = time.perf_counter()
        lin = self._lin(call.id)
        w, plan = lin.w, lin.plan
        x = call.x
        if x.ndim != 2 or x.shape[1] != w.in_features:
            raise ShapeError(f"{call.id}: x must be (T, {w.in_features}), got {tuple(x.shape)}")
        rows = int(x.shape[0])
        xq, sx = quantize_activations(x.to("cpu"), self.ring, w.wnorm)
        if self.observe is not None:
            self.observe(call.id, xq.numpy())
        with self._lock:
            ctr = self._ctr
            self._ctr += 1
        r = self._sx.ring_uniform((rows, w.in_features), plan.in_bits, "narrow-r", call.id.key, ctr).numpy()
        xh, c = mask_inputs(xq.numpy(), r, plan)
        up = pack_up(xh, plan)
        corr = mask_correction(w.w8t, r, plan)
        ct, over = carry_term(w.w8t, c, plan, lin.budget)
        # ---- provider: the packed x_hat only ----
        down = self._provider(w.w8t, unpack_up(up, rows, plan), plan)
        # ---- client ----
        y_hat = unpack_down(down, rows, plan)
        yq = decode_outputs(y_hat, corr, ct if plan.carry_rows.size else None, plan)
        ok = prime_check(xq.numpy(), yq, lin.u, lin.uw)
        if not bool(np.all(ok)):
            self.stats.add(integrity_failures=1)
            raise IntegrityError(f"{call.id}: narrow output failed the prime-field check")
        y = dequantize_outputs(torch.from_numpy(yq), sx, w.scale, w.bias, torch.float32)
        ku, kd = plan.k_bytes(rows, self.ring.k)
        ws = self.wire
        with self._lock:
            ws.rows += rows
            ws.up += len(up)
            ws.down += len(down)
            ws.k_up += ku
            ws.k_down += kd
            ws.carries += int(np.count_nonzero(c))
            ws.carry_slots += rows * lin.budget if plan.carry_rows.size else 0
            ws.carry_overflows += over
            pl = ws.per_linear.setdefault(call.id.kind.value, [0, 0, 0, 0, 0])
            for i, v in enumerate((rows, len(up), len(down), ku, kd)):
                pl[i] += v
        self.stats.add(calls=1, rows=rows, round_trips=1, bytes_up=len(up), bytes_down=len(down),
                       freivalds_checked=1, local_seconds=time.perf_counter() - t0)
        return y.to(x.device)

    def forward_many(self, calls: Sequence[LinearCall]) -> list[torch.Tensor]:
        return _grouped_forward(self, calls)
