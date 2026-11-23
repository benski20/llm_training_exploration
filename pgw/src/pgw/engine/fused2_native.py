"""Native fused2 decode driver (``PGW_FUSED2_DECODE_NATIVE=1`` or :func:`enable_fused2_native`; opt-in, default off).

A fused2 decode pass of one sequence (a few rows, no tree, no padding, no captured hidden states) on the native
float ops (``float_ops = "native"``) with :class:`pgw.engine.fused2.Fused2MaskedBackend` over the lane
(:class:`pgw.engine.fused2_lane.LaneFused2Provider`) runs all its ``2L + 1`` trips and the float work between them
in ``pgw_fast.Fused2Driver`` (``rust/pgw-fast/src/decode_f2.rs``) with the GIL released, Python called once per
driver call. Bit for bit the Python fused2 path (:class:`pgw.engine.fused2.Fused2Schedule` with the same backend):
the same quantization (numpy's float64 roundings and pairwise sums reproduced exactly), the same mask rows (taken
here from the backend's mask source at the start of the pass: one row of every trip key per token row, single
use), the same payload bytes on the same lane items, the same range + ``F_p`` Freivalds checks of both stages
before any output is used, the same native attention / SiLU kernels.

Python's part per pass: take the masks (``Fused2MaskSource`` / ``Fused2SegmentMasks``), draw the noise monitor's
sampling decisions in trip order (a backend without a monitor samples nothing), call the driver, book the
statistics, run a sampled trip's statistics before the next call builds a payload, wipe the masks. Failures raise
what the Python path raises (``IntegrityError`` for a failed check, ``RingOverflowError``, ``ProtocolError`` for
a refused or broken lane call; a pair item is never retried with a new mask).

Not handled here (the ordinary path runs): prompts above the lane's rows, several sequences, trees, padding,
captured hidden states, split folds (gateway columns), the head (it runs after the layers as before).
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Any

import numpy as np
import torch

from pgw.core.errors import ConfigError, IntegrityError, ProtocolError, RingOverflowError

log = logging.getLogger(__name__)

ENV = "PGW_FUSED2_DECODE_NATIVE"
#: Rows per pass the driver takes.
MAX_ROWS = 8

_DONE, _LANE_STATUS, _OVERFLOW, _DUPLICATE, _CHECK_FAILED, _LANE_ERROR, _NONFINITE = 0, 2, 3, 4, 5, 6, 7


def _f64(t: torch.Tensor | None) -> np.ndarray | None:
    return None if t is None else np.ascontiguousarray(t.detach().cpu().to(torch.float64).numpy())


class Fused2NativeDecode:
    """The native fused2 driver of one model, schedule and masked backend."""

    def __init__(self, model: Any, schedule: Any, backend: Any):
        from pgw.core.native import fast
        from pgw.engine.fused2_lane import LaneFused2Provider
        from pgw.engine.model import NativeFloatOps

        if fast is None or not hasattr(fast, "Fused2Driver"):
            raise ConfigError("the native fused2 driver needs a pgw_fast with Fused2Driver (rust/tools/build_fast.py)")
        ops = model.float_ops
        if not isinstance(ops, NativeFloatOps):
            raise ConfigError("the native fused2 driver runs on float_ops='native'")
        if not isinstance(backend.provider, LaneFused2Provider):
            raise ConfigError("the native fused2 driver needs the lane provider (LaneFused2Provider)")
        w, ring = schedule.w, schedule.ring
        for lin in w.linears():
            if lin.gcols is not None:
                raise ConfigError("the native fused2 driver does not take split folds (gateway columns)")
        self.model, self.sched, self.backend = model, schedule, backend
        L = int(model.num_layers)
        self.layers = L
        self.drv = fast.Fused2Driver(int(model.arch.hidden_size), int(model.num_heads), int(model.num_kv_heads),
                                     int(model.head_dim), int(model.split_gate_up[0]), float(model.scale),
                                     float(model.eps), int(ops.attn_threads), int(ops.attn_block), bool(ops.kv_bf16),
                                     L, int(ring.k), int(ring.a), int(ring.cap))
        prov = backend.provider
        #: ``(l1, l2)`` per trip, ``l2`` None for a single (the masks' and the lane items' trip plan).
        self.plan: list[tuple[Any, Any]] = [(w.qkv[0], None)]
        for i in range(L):
            self.plan.append((w.o[i], w.gu[i]))
            self.plan.append((w.down[i], w.qkv[i + 1] if i + 1 < L else None))
        self.names = [l1.key + ("+" + l2.key if l2 is not None else "") for l1, l2 in self.plan]
        for idx, ((l1, l2), name) in enumerate(zip(self.plan, self.names, strict=True)):
            item = int(prov._ix[name])
            self.drv.set_linear(idx, 1, l1.n, l1.m, _f64(l1.scale), _f64(l1.bias), _f64(l1.grid),
                                None if l1.hmul is None else np.ascontiguousarray(l1.hmul.numpy().astype(np.int64)),
                                float(l1.wnorm), float(l1.sigma), item)
            c1 = backend.masks.check_vectors(l1)
            self.drv.set_checks(idx, 1, np.ascontiguousarray(c1.u, dtype=np.int64),
                                np.ascontiguousarray(c1.uw, dtype=np.int64))
            if l2 is not None:
                self.drv.set_linear(idx, 2, l2.n, l2.m, _f64(l2.scale), _f64(l2.bias), None, None, float(l2.wnorm),
                                    float(l2.sigma), item)
                c2 = backend.masks.check_vectors(l2)
                self.drv.set_checks(idx, 2, np.ascontiguousarray(c2.u, dtype=np.int64),
                                    np.ascontiguousarray(c2.uw, dtype=np.int64))
        noise = getattr(backend, "noise_monitor", None)
        self.dup_hist = None
        if noise is not None and hasattr(noise, "native_dup_history") and hasattr(noise, "packed_dup_params"):
            self.dup_hist = noise.native_dup_history()
            for idx, (l1, _) in enumerate(self.plan):
                params = noise.packed_dup_params(l1.n)
                if params is not None:
                    cols, shift, mult, groups, per = params
                    limit = int(noise.config.duplicate_history_rows) * int(groups)
                    self.drv.set_dup(idx, (np.ascontiguousarray(cols, dtype=np.int64), int(shift),
                                           np.ascontiguousarray(mult, dtype=np.uint64), int(groups), int(per), limit))
        #: Test hook: when a list, every call's ``(pos, payload)`` (the provider-visible bodies).
        self.record: list[tuple[int, bytes]] | None = None
        #: [passes, driver calls, trips, local ns, lane ns, provider compute us]
        self.totals = [0, 0, 0, 0, 0, 0]

    # ---- eligibility ----------------------------------------------------------------------------------
    def eligible(self, st: Any) -> bool:
        if len(st.seqs) != 1 or st.trees[0] is not None or st.extra:
            return False
        h = st.hs[0]
        t = int(h.shape[0])
        prov = self.backend.provider
        if not 0 < t <= min(MAX_ROWS, int(prov.max_rows)) or h.dtype != torch.float32 or h.device.type != "cpu":
            return False
        if int(getattr(st.seqs[0], "serial_tail", 0) or 0) > 0:
            return False
        cache = st.caches[0]
        return (cache.dtype == torch.bfloat16) == bool(self.model.float_ops.kv_bf16)

    # ---- one pass -------------------------------------------------------------------------------------
    def _masks(self, t: int) -> list[Any]:
        from pgw.engine.fused2 import Fused2Trip

        cap = int(self.sched.ring.cap)
        dummy = torch.empty((t, 0), dtype=torch.int64)
        out = []
        try:
            for l1, l2 in self.plan:
                kind = "single" if l2 is None else "pair"
                out.append(self.backend.masks.take(Fused2Trip(kind, l1, dummy, cap, l2, None)))
        except BaseException:
            for m in out:
                m.wipe()
            raise
        return out

    def run(self, st: Any) -> None:
        """Every decoder layer of the pass; writes ``st.hs`` (the float32 residual, as the Python schedule)."""
        be = self.backend
        model = self.model
        h = st.hs[0]
        t = int(h.shape[0])
        start = int(st.starts[0])
        cache = st.caches[0]
        kc, vc = cache.numpy_views()
        rope_np = st.rope_np
        if rope_np is None:
            from pgw.engine.model import _np32

            rope_np = st.rope_np = [(_np32(c), _np32(s)) for c, s in st.rope]
        cos, sin = rope_np[0]
        prov = be.provider
        drv = self.drv
        hn = h.detach().numpy() if h.is_contiguous() else np.ascontiguousarray(h.detach().numpy())
        masks = self._masks(t)
        arrs: list[tuple[Any, ...]] = []
        for m in masks:
            arrs.append(tuple(None if x is None else x.numpy() for x in (m.r_x, m.wr_x, m.v, m.wv)))
        noise = getattr(be, "noise_monitor", None)
        plan_call = getattr(noise, "plan_call", None) if noise is not None else None
        total = len(self.plan)
        drv.begin(hn, start)
        ok = False
        try:
            while drv.pos < total:
                pos = int(drv.pos)
                # the per-trip sampling draws in trip order; a sampled trip ends the call, so its statistics run
                # before any later payload is built (the per-trip path's order)
                flags = bytearray()
                end = total
                for idx in range(pos, total):
                    f = 2 if (plan_call is not None and bool(plan_call())) else 0
                    if self.record is not None:
                        f |= 4
                    flags.append(f)
                    if f & 2:
                        end = idx + 1
                        break
                rids = secrets.token_bytes(16 * (end - pos))
                lane = (prov.ch, int(prov.ring_index), int(prov.deadline_ms), float(prov.timeout_s))
                t0 = time.perf_counter()
                code, info, stats, sampled, recorded = drv.run(end, arrs[pos:end], bytes(flags), rids, kc, vc, cos,
                                                               sin, lane, self.dup_hist)
                self._book(stats, t, time.perf_counter() - t0)
                if self.record is not None:
                    self.record.extend((int(p), bytes(b)) for p, b in recorded)
                self._sampled(sampled)
                if code != _DONE:
                    self._raise(code, info)
            h_out = torch.empty((t, int(model.arch.hidden_size)), dtype=torch.float32)
            drv.end(h_out.numpy())
            ok = True
        finally:
            if not ok:
                drv.abort()
            for m in masks:
                m.wipe()
        self.totals[0] += 1
        st.hs = [h_out]

    def _book(self, stats: Any, t: int, wall: float) -> None:
        trips, rows, up, down, checks, remote_ns, local_ns, compute_us, _dup_rows, _dups = (int(v) for v in stats)
        be = self.backend
        f2 = be.f2
        f2.trips += trips
        f2.rows += rows
        f2.bytes_up += up
        f2.bytes_down += down
        f2.checks += checks
        be.stats.add(calls=trips, rows=rows)
        prov = be.provider
        prov.calls += trips
        prov.bytes_up += up
        prov.bytes_down += down
        self.totals[1] += 1
        self.totals[2] += trips
        self.totals[3] += local_ns
        self.totals[4] += remote_ns
        self.totals[5] += compute_us

    def _sampled(self, vals: list[Any]) -> None:
        noise = getattr(self.backend, "noise_monitor", None)
        for _pos, xh, xq in vals:
            try:
                if noise is not None:
                    noise.inspect_planned(torch.from_numpy(xh), torch.from_numpy(xq))
            finally:
                xh.fill(0)
                xq.fill(0)

    def _raise(self, code: int, info: Any) -> None:
        if code == _CHECK_FAILED:
            pos, stage, why = info
            l1, l2 = self.plan[pos]
            key = (l1 if stage == 1 else l2).key
            msg = "output out of range" if why == 2 else "Freivalds check failed"
            raise IntegrityError(f"fused2 {key}: {msg}")
        if code == _OVERFLOW:
            raise RingOverflowError(str(info[1]))
        if code == _NONFINITE:
            raise RingOverflowError("non-finite values cannot be quantized")
        if code == _DUPLICATE:
            raise IntegrityError(f"fused2 trip {info[0]}: duplicate outbound rows ({info[1]})")
        if code == _LANE_STATUS:
            raise ProtocolError(f"fused2 lane call {self.names[info[0]]!r} refused (status {info[1]})")
        if code == _LANE_ERROR:
            raise ProtocolError(f"fused2 lane call {self.names[info[0]]!r} failed ({info[1]})")
        raise ProtocolError(f"fused2 native driver: code {code}")


def enable_fused2_native(model: Any) -> Fused2NativeDecode:
    """Attach the native driver to ``model``'s fused2 schedule (the ordinary path stays the fallback)."""
    sched = model.schedule
    if sched is None or getattr(sched, "name", "") != "fused2":
        raise ConfigError("the native fused2 driver needs the fused2 schedule")
    nd = Fused2NativeDecode(model, sched, model.backend)
    sched.native_decode = nd
    return nd


def maybe_enable(model: Any) -> Fused2NativeDecode | None:
    """:func:`enable_fused2_native` when ``PGW_FUSED2_DECODE_NATIVE`` is set (logs and returns None if refused)."""
    if os.environ.get(ENV, "0").strip().lower() in ("", "0", "false", "no", "off"):
        return None
    try:
        return enable_fused2_native(model)
    except ConfigError as exc:
        log.warning("native fused2 driver unavailable: %s", exc)
        return None
