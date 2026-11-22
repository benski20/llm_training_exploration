"""Native decode driver (``PGW_DECODE_NATIVE=1``, opt-in; default off).

A decode pass of one sequence (a few rows, no tree, no padding, no captured hidden states) on the native
float path (``float_ops = "native"``) with every linear a masked remote trip of the native per-trip path
(``PGW_TRIP_NATIVE``) runs its decoder layers in ``pgw_fast.DecodeDriver``: the float operations between
the trips (the same kernels and bits as :meth:`TransformerModel._layer_native`) and the trips themselves
(``TripLease`` prepare, duplicate-row fingerprints recorded before the payload leaves, the lane trip on the
native ``TripChannel``, ``TripTicket`` finish with every check) with the GIL released. Python runs once per
call (``PGW_DECODE_NATIVE_LAYERS`` layers per call, 0 = all): it takes the leases (the same window
``ensure``, exhaustion pre-check and soundness check as the per-trip path, before any row is taken), draws
the Freivalds and noise-sampling decisions of each trip in trip order, generates the request ids, and
afterwards books the statistics and hands the sampled values to the noise monitor's statistics.

What the driver cannot do itself comes back at a trip boundary: a trip the lane did not serve (no native
channel, a refusal, a broken connection) is sent here with the same request (payload bytes and id) on the
ordinary fallback and finished by :meth:`MaskedBackend._finish_lease`, and the driver continues with its
output. Same rows in the same order (each once), same payload bytes, same checks, same outputs: masked ==
plain bit for bit (``tests/engine/test_decode_native.py``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import numpy as np
import torch

from pgw.core.errors import IntegrityError, MaskReuseError, ProtocolError, RingOverflowError, ShapeError

log = logging.getLogger(__name__)

#: The engine's synchronous pass (``TransformerModel.forward_batch``) on this thread: the driver may block
#: for the whole pass (never on the event loop's inline hops).
sync_pass = threading.local()

#: Rows per pass the driver takes (decode and speculation verify passes; prompts take the ordinary path).
MAX_ROWS = 8


class NativeDecode:
    """The driver of one model and masked backend (built lazily by :func:`driver_for`)."""

    def __init__(self, model: Any, backend: Any):
        from pgw.core.native import fast

        self.model = model
        self.backend = backend
        ops = model.float_ops
        L = int(model.num_layers)
        self.layers = L
        hidden = int(model.arch.hidden_size)
        inter = int(model.split_gate_up[0])
        self.drv = fast.DecodeDriver(hidden, int(model.num_heads), int(model.num_kv_heads), int(model.head_dim),
                                     inter, float(model.scale), float(model.eps), int(ops.attn_threads),
                                     int(ops.attn_block), bool(ops.kv_bf16), L)
        norms = model._norm_np
        if norms is None:
            from pgw.engine.model import _np32

            norms = model._norm_np = ([_np32(w) for w in model.input_norms], [_np32(w) for w in model.post_norms])
        for layer in range(L):
            self.drv.set_layer(layer, norms[0][layer], norms[1][layer])
        self.lids = [lid for layer in range(L) for lid in model._lids[layer]]
        self.metas = [backend._meta.get(lid) for lid in self.lids]
        noise = backend.noise_monitor
        self.dup_hist = None
        if noise is not None and hasattr(noise, "native_dup_history") and hasattr(noise, "packed_dup_params"):
            self.dup_hist = noise.native_dup_history()
        self.rows = 0
        self.per_call = max(0, int(os.environ.get("PGW_DECODE_NATIVE_LAYERS", "0") or 0)) or L
        self._lane_ix: tuple[int, Any] | None = None
        #: Draws of trips a call did not reach (after a hand-back): used by the next call of the pass.
        self._left_flags: dict[int, int] = {}
        #: [driver calls, trips run natively, native local ns (prepare + finish + float ops), lane ns, run CPU s]
        self.totals = [0, 0, 0, 0, 0.0]

    def configure(self, t: int) -> None:
        """Per-trip constants for ``t``-row trips (thread counts and split depend on the rows)."""
        if self.rows == t:
            return
        be = self.backend
        noise = be.noise_monitor
        for idx, (lid, meta) in enumerate(zip(self.lids, self.metas, strict=True)):
            n, m = int(meta.in_features), int(meta.out_features)
            scale, bias, bound = be._consts(lid, meta)
            pool = int(be.trip_pool_min_elems)
            split_p = be.trip_split if be.trip_split >= 2 and t * n >= be.trip_split_min_elems else 0
            split_f = be.trip_split if be.trip_split >= 2 and t * m >= be.trip_split_min_elems else 0
            dup = None
            if noise is not None and self.dup_hist is not None:
                params = noise.packed_dup_params(n)
                if params is not None:
                    cols, shift, mult, groups, per = params
                    limit = int(noise.config.duplicate_history_rows) * int(groups)
                    dup = (np.ascontiguousarray(cols, dtype=np.int64), int(shift),
                           np.ascontiguousarray(mult, dtype=np.uint64), int(groups), int(per), limit)
            self.drv.set_linear(idx, n, m, scale, bias, int(bound), 0 if t * n >= pool else 1,
                                0 if t * m >= pool else 1, split_p, split_f, dup)
        self.rows = t

    # ---- eligibility ----------------------------------------------------------------------------------
    def eligible(self, st: Any) -> bool:
        be = self.backend
        if len(st.seqs) != 1 or st.trees[0] is not None or st.extra:
            return False
        if be.noise_monitor is not None and self.dup_hist is None:
            return False
        h = st.hs[0]
        t = int(h.shape[0])
        if (not 0 < t <= min(int(be.max_rows_per_item), MAX_ROWS) or h.dtype != torch.float32
                or h.device.type != "cpu"):
            return False
        cache = st.caches[0]
        if (cache.dtype == torch.bfloat16) != bool(self.model.float_ops.kv_bf16):
            return False
        noise = be.noise_monitor
        if noise is not None:  # sampled values are inspected whole (no random block draw)
            mx = int(getattr(noise, "_max_values", 0) or 0)
            if any(t * int(m.in_features) > mx for m in self.metas):
                return False
        is_local = getattr(be, "local_linear", None)
        if is_local is not None and any(is_local(lid) for lid in self.lids):
            return False
        sess = st.sessions[0]
        res = sess if sess is not None else be.mask_source
        if res is None or getattr(res, "trip_leases", None) is None:
            return False
        return be._use_trip([_Call(self.lids[0], h, sess)]) and all(be._trip_self_check(m) for m in self.metas)

    # ---- one pass -----------------------------------------------------------------------------------
    def run(self, st: Any) -> None:
        """All decoder layers of the pass; writes ``st.hs`` (and leaves ``st.normed`` empty)."""
        from pgw.core.native import fast  # noqa: F401  (loaded: eligible() ran)

        be = self.backend
        model = self.model
        h = st.hs[0]
        t = int(h.shape[0])
        self.configure(t)
        sess = st.sessions[0]
        res = be._reservation(_Call(self.lids[0], h, sess))
        start = int(st.starts[0])
        cache = st.caches[0]
        kc, vc = cache.numpy_views()
        rope_np = st.rope_np
        if rope_np is None:
            from pgw.engine.model import _np32

            rope_np = st.rope_np = [(_np32(c), _np32(s)) for c, s in st.rope]
        cos, sin = rope_np[0]
        pre = st.normed
        st.normed = None
        normed = None
        if pre is not None and pre[0] == 0:
            normed = pre[1][0].numpy()
        drv = self.drv
        hn = h.detach().numpy() if h.is_contiguous() else np.ascontiguousarray(h.detach().numpy())
        self._left_flags = {}
        drv.begin(hn, 0, start, normed)
        ok = False
        try:
            inject = None
            L4 = 4 * self.layers
            while drv.pos < L4:
                pos = int(drv.pos)
                inject = self._call(res, sess, t, min(L4, (pos // 4 + self.per_call) * 4), kc, vc, cos, sin, inject)
            h_out = torch.empty((t, int(model.arch.hidden_size)), dtype=torch.float32)
            drv.end(h_out.numpy(), None)
            ok = True
        finally:
            if not ok:
                drv.abort()
        st.hs = [h_out]

    def _call(self, res: Any, sess: Any, t: int, end: int, kc: Any, vc: Any, cos: Any, sin: Any,
              inject: Any) -> Any:
        """One driver call from its position to trip ``end``; returns the output to inject (a trip finished
        here after a hand-back, or run on the ordinary path for want of a lease) or ``None``."""
        be = self.backend
        drv = self.drv
        from pgw.engine.backends import _note_activity

        _note_activity()
        # as MaskedBackend._prepare: the session's deferred checks and sampled statistics first
        if be._noise_pending is not None:
            be._join_noise()
        if be._pending or be._failed:
            be._join_pending([_Call(self.lids[0], None, sess)])
        pos = int(drv.pos)
        first = pos + (1 if inject is not None else 0)
        leases_map = res.trip_leases
        leases = []
        noise = be.noise_monitor
        plan_call = getattr(noise, "plan_call", None) if noise is not None else None
        no_lease = None
        for idx in range(first, end):
            lid, meta = self.lids[idx], self.metas[idx]
            lease = be._trip_lease_for(res, leases_map, lid, meta, t)
            if lease is None:  # this trip takes the ordinary path (after the driver ran the ones before it)
                no_lease = idx
                end = idx
                break
            leases.append(lease)
        if no_lease is not None and end == first and inject is None:
            return self._ordinary_trip(sess, no_lease)
        flags = bytearray(end - first)
        left = self._left_flags
        self._left_flags = {}
        for i, idx in enumerate(range(first, end)):  # the per-trip draws, in trip order (each trip's once)
            f = left.pop(idx, None)
            if f is None:
                check = be._should_check()
                sampled = bool(plan_call()) if plan_call is not None else False
                f = (1 if check else 0) | (2 if sampled else 0)
                if check and not leases[i].field and be._defer_ok(self.lids[idx]):
                    f |= 4
            flags[i] = f
        rids = os.urandom(16 * (end - first))
        client = be.client
        lane_got = None
        lane_arg = None
        drv_ch = getattr(client, "driver_channel", None)
        if drv_ch is not None and end > first and getattr(client, "lane_eligible_for_driver", True):
            lane_got = drv_ch(be.fingerprint, be.ring.name, t, be.deadline_ms)
            if lane_got is not None:
                ix = self._lane_index(lane_got[0])
                if ix is None:
                    client.driver_release(lane_got[0], lane_got[1], lane_got[2])
                    lane_got = None
                else:
                    lane_arg = (lane_got[1], lane_got[3], ix,
                                int(be.deadline_ms if be.deadline_ms is not None else 0), lane_got[4])
        c0 = time.thread_time()
        try:
            code, info, stats, sampled_vals = drv.run(end, leases, bytes(flags), rids, kc, vc, cos, sin, lane_arg,
                                                      self.dup_hist, inject)
        except BaseException:
            if lane_got is not None:
                client.driver_release(lane_got[0], lane_got[1], lane_got[2])
            raise
        (trips, rows, up, down, checked, skipped, deferred, deferred_ok, remote_ns, local_ns, lane_trips, lane_sent,
         lane_recv, rtt_ns, dup_rows, dups) = stats
        if lane_got is not None:
            st_ = 0
            err = None
            if code == 1 and info[3] >= 0:
                st_, err = (0, info[4]) if info[4] is not None else (int(info[3]), None)
            if st_:
                lane_trips += 1
                lane_sent += 36 + len(info[0])
            client.driver_release(lane_got[0], lane_got[1], lane_got[2], trips=lane_trips, sent=lane_sent,
                                  received=lane_recv, last_rtt_s=rtt_ns / 1e9, status=st_, error=err)
        # statistics, as _finish_lease books them per trip
        tot = self.totals
        tot[4] += time.thread_time() - c0
        tot[0] += 1
        tot[1] += int(trips)
        tot[2] += int(local_ns)
        tot[3] += int(remote_ns)
        if trips:
            be.stats.add_many(int(trips), int(rows), int(trips), int(up), int(down), int(checked) + int(deferred_ok),
                              int(skipped), remote_ns / 1e9, local_ns / 1e9, 0 < t <= be.decode_max_rows)
        if deferred:
            be.stats.add(deferred_checks=int(deferred))
        if noise is not None and dup_rows and not dups:
            noise.record_native_dups(int(dup_rows), 0)
        self._sampled(sampled_vals)
        if code == 0:
            return self._ordinary_trip(sess, no_lease) if no_lease is not None else None
        if code == 1:
            # trips after the handed-back one keep their draws for the next call (drawn once, in trip order)
            j = int(drv.pos)
            self._left_flags = {idx: flags[idx - first] for idx in range(j + 1, end)}
            return self._handback(info, res, sess, t, end, first)
        lid = self.lids[int(drv.pos)] if int(drv.pos) < len(self.lids) else self.lids[-1]
        if code == 2:
            status, bad = info
            bound = int(be._consts(lid, be._meta.get(lid))[2])
            try:
                be._raise_status(int(status), int(bad), lid, t, bound)
            except IntegrityError as exc:
                self._report(exc)
                raise
            raise IntegrityError("trip finish failed", details={"status": int(status)})
        if code == 3:
            kind, msg = info
            if kind in ("nonfinite", "overflow"):
                raise RingOverflowError(str(msg))
            raise ShapeError(str(msg))
        if code == 4:
            dups_n, at_rows = info
            if noise is not None:
                noise.record_native_dups(int(dup_rows), int(dups_n), int(at_rows))
            raise IntegrityError("duplicate outbound rows (possible mask reuse)", details={"duplicates": int(dups_n)})
        if code == 5:
            status, bad = info
            err: Exception
            if int(status) == 3:
                raise MaskReuseError("mask reservation closed during a compute call")
            if int(status) == 1:
                err = IntegrityError("compute service result failed the Freivalds check",
                                     details={"linear": lid.key, "bad_rows": int(bad), "rows": t, "deferred": True})
            else:
                err = IntegrityError("deferred Freivalds check could not run",
                                     details={"linear": lid.key, "status": int(status), "deferred": True})
            self._report(err)
            raise err
        if code == 6:
            raise MaskReuseError("mask lease could not serve a call it reported ready for")
        raise ProtocolError(f"compute response item is malformed: {info}")

    def _report(self, exc: IntegrityError) -> None:
        be = self.backend
        be.stats.add(integrity_failures=1)
        log.error("integrity check failed: code=%s details=%s", exc.code, exc.details)
        if be.on_integrity_failure is not None:
            try:
                be.on_integrity_failure(exc)
            except Exception:  # pragma: no cover - callback must not mask the error
                log.exception("on_integrity_failure callback raised")

    def _sampled(self, vals: list[Any]) -> None:
        """The noise monitor's statistics of the sampled trips (their every-call checks ran natively)."""
        if not vals:
            return
        be = self.backend
        noise = be.noise_monitor
        for _pos, xh, xq in vals:
            if noise is None:
                xq.fill(0)
                xh.fill(0)
                continue
            if be.noise_async_sampled:
                be._noise_submit_xq(noise, xh, xq)
            else:
                try:
                    noise.inspect_planned(torch.from_numpy(xh), torch.from_numpy(xq))
                finally:
                    xh.fill(0)
                    xq.fill(0)

    def _lane_index(self, lane: Any) -> Any:
        hit = self._lane_ix
        if hit is not None and hit[0] is lane:
            return hit[1]
        ix = np.zeros(len(self.lids), dtype=np.uint16)
        for i, (lid, meta) in enumerate(zip(self.lids, self.metas, strict=True)):
            ent = lane.linears.get(lid.key)
            if ent is None or int(ent[1]) != int(meta.out_features) or int(ent[2]) != int(meta.in_features):
                return None
            ix[i] = int(ent[0])
        self._lane_ix = (lane, ix)
        return ix

    def _ordinary_trip(self, sess: Any, idx: int) -> Any:
        """Trip ``idx`` on the backend's ordinary path (no native lease for it); its output to inject."""
        if int(self.drv.pos) != idx:  # pragma: no cover - the driver stopped right before it
            raise ShapeError("native decode driver out of step")
        x = torch.from_numpy(self.drv.current_input())
        out = self.backend.forward_many([_Call(self.lids[idx], x, sess)])[0]
        return np.ascontiguousarray(out.to(torch.float32).numpy(), dtype=np.float32)

    def _handback(self, info: Any, res: Any, sess: Any, t: int, end: int, first: int) -> Any:
        """Send a trip the driver handed back on the ordinary fallback and finish it (same request)."""
        from pgw.engine.backends import _Prepared
        from pgw.wire.messages import ComputeItem, ComputeRequest

        be = self.backend
        payload, ticket, rid, lane_status, _lane_err = info
        idx = int(self.drv.pos)
        lid, meta = self.lids[idx], self.metas[idx]
        call = _Call(lid, _CPU_ROW, sess)
        request = ComputeRequest(request_id=bytes(rid).hex(), fingerprint=be.fingerprint, ring=be.ring.name,
                                 items=(ComputeItem(lid.key, t, payload),), deadline_ms=be.deadline_ms)
        prep = _Prepared((call,), [], [], [meta], request, 0.0)
        prep.trip = True
        prep.one = True
        prep.ticket = ticket
        prep.defer = bool(ticket.checked) and not _lease_field(res, lid) and be._defer_ok(lid)
        prep.reservations = (res,)
        client = be.client
        t0 = time.perf_counter()
        try:
            if lane_status >= 0 and hasattr(client, "fallback_sync"):
                response = client.fallback_sync(request)  # the lane saw it and did not serve it
            elif hasattr(client, "compute_sync"):
                response = client.compute_sync(request)
            else:
                from pgw.engine.backends import _run_coro_blocking

                response = _run_coro_blocking(client.compute(request))
        except BaseException as exc:
            prep.wipe()
            if isinstance(exc, Exception):
                raise be._wrap_transport(exc) from exc
            raise
        out = be._finish_lease(prep, response, time.perf_counter() - t0)[0]
        if be._pending or be._failed:  # a deferred check of this trip: joined before the next payload
            be._join_pending([call])
        return np.ascontiguousarray(out.numpy(), dtype=np.float32)


def _lease_field(res: Any, lid: Any) -> int:
    lease = res.trip_leases.get(lid.key)
    return int(lease.field) if lease is not None else 0


_CPU_ROW = torch.zeros(1, 1)


def _Call(lid: Any, x: Any, sess: Any) -> Any:  # noqa: N802 - a LinearCall factory
    from pgw.engine.backends import LinearCall

    return LinearCall(lid, x if x is not None else _CPU_ROW, sess)


def driver_for(model: Any) -> NativeDecode | None:
    """The model's driver when its backend enables it (``PGW_DECODE_NATIVE``), else ``None``."""
    be = model.backend
    if not getattr(be, "decode_native", False):
        return None
    drv = getattr(model, "_native_decode", None)
    if drv is None or drv.backend is not be:
        from pgw.engine.model import NativeFloatOps

        if not isinstance(model.float_ops, NativeFloatOps) or model.schedule is not None:
            return None
        try:
            drv = NativeDecode(model, be)
        except Exception as exc:  # e.g. shapes the driver does not take: the ordinary path
            log.warning("native decode driver unavailable: %s", exc)
            be.decode_native = False
            return None
        model._native_decode = drv
    return drv
