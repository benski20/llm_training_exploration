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

Python's part per pass: take the masks (``Fused2MaskSource`` / ``Fused2SegmentMasks``, or the session's mask-store
reservation of fused2 pair-mask segments via :class:`pgw.engine.fused2_store.Fused2StoreMasks`: then a pass runs
natively only when every trip's rows fit the current segment extent, and the driver's check vectors are installed
from the extent the rows came from whenever its segment changes), draw the noise monitor's
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
        #: head_fold (item table v2): the last trip is the pair (last down, folded head); passes in which every row
        #: gets logits (decode, and verify passes with ``last_logits`` = rows) run natively, others take the Python
        #: path. local_qkv0: trip 0 computed exactly in the driver from the public matrix (no mask, no trip).
        self.head_fold = bool(getattr(schedule, "head_fold", False))
        self.local_qkv0 = bool(getattr(schedule, "local_qkv0", False))
        if (self.head_fold or self.local_qkv0) and not hasattr(fast.Fused2Driver, "set_head_fold"):
            raise ConfigError("this pgw_fast has no fused2 head_fold / local_qkv0 pass (rebuild rust/pgw-fast)")
        self.keys = [lin.key for lin in w.all_linears()]
        if self.local_qkv0:
            self.keys = [k for k in self.keys if k != w.qkv[0].key]
        self.drv = fast.Fused2Driver(int(model.arch.hidden_size), int(model.num_heads), int(model.num_kv_heads),
                                     int(model.head_dim), int(model.split_gate_up[0]), float(model.scale),
                                     float(model.eps), int(ops.attn_threads), int(ops.attn_block), bool(ops.kv_bf16),
                                     L, int(ring.k), int(ring.a), int(ring.cap))
        prov = backend.provider
        #: Store sessions only: check vectors come with each pass's masks (per segment), not once here.
        static = backend.masks is not None and not getattr(backend.masks, "partial", False)
        #: ``(trip index, stage)`` -> token of the check vectors installed in the driver (store masks).
        self._tokens: dict[tuple[int, int], Any] = {}
        self._store_checks = False
        #: ``(l1, l2)`` per trip, ``l2`` None for a single (the masks' and the lane items' trip plan).
        self.plan: list[tuple[Any, Any]] = [(w.qkv[0], None)]
        for i in range(L):
            self.plan.append((w.o[i], w.gu[i]))
            self.plan.append((w.down[i], w.qkv[i + 1] if i + 1 < L else (w.head if self.head_fold else None)))
        self.names = [l1.key + ("+" + l2.key if l2 is not None else "") for l1, l2 in self.plan]
        if self.head_fold:
            self.drv.set_head_fold(int(w.head.m))
        if self.local_qkv0:
            w0 = w.qkv[0].w8
            if w0.numel() and 0 in tuple(w0.stride()):  # the weight-free tables' placeholder (a broadcast zero)
                raise ConfigError("local_qkv0 needs layer 0's folded qkv matrix (gateway_tables(keep_qkv0=True))")
            self.drv.set_local_qkv0(np.ascontiguousarray(w0.numpy()))
        for idx, ((l1, l2), name) in enumerate(zip(self.plan, self.names, strict=True)):
            item = int(prov._ix[name]) if not (idx == 0 and self.local_qkv0) else 0
            self.drv.set_linear(idx, 1, l1.n, l1.m, _f64(l1.scale), _f64(l1.bias), _f64(l1.grid),
                                None if l1.hmul is None else np.ascontiguousarray(l1.hmul.numpy().astype(np.int64)),
                                float(l1.wnorm), float(l1.sigma), item)
            if static and not (idx == 0 and self.local_qkv0):
                c1 = backend.masks.check_vectors(l1)
                self.drv.set_checks(idx, 1, np.ascontiguousarray(c1.u, dtype=np.int64),
                                    np.ascontiguousarray(c1.uw, dtype=np.int64))
            if l2 is not None:
                self.drv.set_linear(idx, 2, l2.n, l2.m, _f64(l2.scale), _f64(l2.bias), None, None, float(l2.wnorm),
                                    float(l2.sigma), item)
                if static:
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
        if getattr(self.model, "qk_norms", None) is not None:  # no per-head q/k norm in the native driver
            return False
        if len(st.seqs) != 1 or st.trees[0] is not None or st.extra:
            return False
        h = st.hs[0]
        t = int(h.shape[0])
        prov = self.backend.provider
        if not 0 < t <= min(MAX_ROWS, int(prov.max_rows)) or h.dtype != torch.float32 or h.device.type != "cpu":
            return False
        if int(getattr(st.seqs[0], "serial_tail", 0) or 0) > 0:
            return False
        if self.head_fold:  # the driver gives every row logits: decode (1 row) or a verify pass of last_logits rows
            s = st.seqs[0]
            nh = t if s.all_logits else (min(t, int(s.last_logits)) if s.last_logits > 0 else 1)
            if nh != t:
                return False
        cache = st.caches[0]
        if (cache.dtype == torch.bfloat16) != bool(self.model.float_ops.kv_bf16):
            return False
        sessions = getattr(st, "sessions", None)
        src = self.backend.masks_for(sessions[0] if sessions else None)
        fits = getattr(src, "pass_fits", None)
        # store masks: the whole pass must come from one segment extent per linear (one set of check vectors
        # per trip); otherwise the ordinary path runs it in row pieces
        return fits is None or bool(fits(self.keys, t))

    # ---- one pass -------------------------------------------------------------------------------------
    def _masks(self, t: int, src: Any) -> list[Any]:
        from pgw.engine.fused2 import Fused2Trip

        cap = int(self.sched.ring.cap)
        dummy = torch.empty((t, 0), dtype=torch.int64)
        out = []
        try:
            for idx, (l1, l2) in enumerate(self.plan):
                if idx == 0 and self.local_qkv0:
                    out.append(None)  # no masks: computed locally
                    continue
                kind = "single" if l2 is None else "pair"
                m = src.take(Fused2Trip(kind, l1, dummy, cap, l2, None))
                out.append(m)
                if int(m.r_x.shape[0]) != t:  # store masks: eligible() saw the pass fit one extent
                    raise IntegrityError("fused2 native pass: mask rows cross a segment extent")
            if getattr(src, "partial", False):
                self._install_checks(out)
                self._store_checks = True
            elif self._store_checks:  # the driver's vectors are a store segment's (or zeroed): never mix
                raise ConfigError("the native fused2 driver serves store sessions only once one was seen")
        except BaseException:
            for m in out:
                if m is not None:
                    m.wipe()
            raise
        return out

    def _install_checks(self, masks: list[Any]) -> None:
        """Store masks: install each trip's check vectors (of the segment its rows came from) when they changed."""
        for idx, m in enumerate(masks):
            if m is None:
                continue
            for stage, ch in ((1, m.c1), (2, m.c2)):
                if ch is None:
                    if stage == 1 or self.plan[idx][1] is not None:
                        raise IntegrityError("fused2 native pass: store masks without check vectors")
                    continue
                token = getattr(ch, "token", None)  # (segment id, linear, "p61"): same token, same vectors
                if token is None or self._tokens.get((idx, stage), (None,))[0] != token:
                    self.drv.set_checks(idx, stage, np.ascontiguousarray(ch.u, dtype=np.int64),
                                        np.ascontiguousarray(ch.uw, dtype=np.int64))
                    self._tokens[(idx, stage)] = (token, int(ch.u.shape[0]), int(ch.u.shape[1]), int(ch.uw.shape[1]))

    def wipe_checks(self) -> None:
        """Overwrite the driver's copies of store check vectors with zeros (the next pass installs fresh ones;
        a pass never runs on zero vectors: tokens are cleared, so every trip's vectors are installed again)."""
        if not self._tokens:
            return
        for (idx, stage), (_tok, c, m, n) in list(self._tokens.items()):
            self.drv.set_checks(idx, stage, np.zeros((c, m), dtype=np.int64), np.zeros((c, n), dtype=np.int64))
        self._tokens.clear()

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
        sessions = getattr(st, "sessions", None)
        src = be.masks_for(sessions[0] if sessions else None)
        masks = self._masks(t, src)
        arrs: list[tuple[Any, ...]] = []
        for m in masks:
            arrs.append((None, None, None, None) if m is None else
                        tuple(None if x is None else x.numpy() for x in (m.r_x, m.wr_x, m.v, m.wv)))
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
                    try:
                        self._raise(code, info)
                    except IntegrityError as exc:
                        report = getattr(be, "integrity_failed", None)
                        if report is not None:
                            report(exc)
                        raise
            h_out = torch.empty((t, int(model.arch.hidden_size)), dtype=torch.float32)
            logits = None
            if self.head_fold:
                logits = torch.empty((t, int(self.plan[-1][1].m)), dtype=torch.float32)
                drv.end(h_out.numpy(), logits.numpy())
            else:
                drv.end(h_out.numpy())
            ok = True
        finally:
            if not ok:
                drv.abort()
            for m in masks:
                if m is not None:
                    m.wipe()
            if getattr(src, "closed", False):  # the session's reservation closed: drop its check vectors too
                self.wipe_checks()
        self.totals[0] += 1
        st.hs = [h_out]
        if logits is not None:  # Fused2FoldedHead.logits_from reads them (the Python path's ex["logits"])
            from pgw.engine.fused2 import FUSED2_META

            st.extra[FUSED2_META] = {"logits": [logits]}
            self.sched.head_rows += t

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
    model.backend.native_decode = nd  # the backend's release_session_checks also wipes the driver's copies
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
