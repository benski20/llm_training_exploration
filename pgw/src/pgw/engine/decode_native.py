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

Noise statistics keep the per-trip path's order: a trip the monitor samples ends a driver call, so its
statistics run before any later payload is built and the next trip's sampling draw comes after them (an alarm
then blocks the very next payload, fail closed). Inline statistics (``PGW_NOISE_ASYNC_SAMPLED`` off) go
further, as the per-trip path does: the sampled trip is handed back after its native prepare and its
statistics run before its own payload leaves (the trip is then sent by the ordinary path, same request).
With ``PGW_NOISE_ASYNC_SAMPLED=1`` the sampled trip stays on the native lane and its statistics are joined
before the next payload. Cost: one more driver call per sampled trip (5 % of trips at the default rate).

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
        #: Inline statistics of a sampled trip with the trip held in the driver (``DECODE_HOLD``): the driver
        #: keeps the prepared trip, its statistics run here, and the next call sends it on the native lane.
        self.native_hold = bool(getattr(fast, "DECODE_HOLD", 0)) and os.environ.get(
            "PGW_DECODE_NATIVE_HOLD", "1").strip().lower() not in ("0", "false", "no", "off")
        #: ``(trip index, lease)`` of the trip held in the driver (prepared, statistics run, not sent).
        self._held: tuple[int, Any] | None = None
        #: [driver calls, trips run natively, native local ns (prepare + finish + float ops), lane ns, run CPU s]
        self.totals = [0, 0, 0, 0, 0.0]
        #: Several answers per pass (``PGW_DECODE_NATIVE_MULTI``, :meth:`run_multi`): on when set and built.
        self.multi = os.environ.get("PGW_DECODE_NATIVE_MULTI", "0").strip().lower() in (
            "1", "true", "yes", "on") and hasattr(self.drv, "run_multi")
        #: [several-answer driver calls, several-answer trips run natively]
        self.multi_totals = [0, 0]
        self._ts: list[int] = []
        self._left_multi: dict[int, bytes] = {}
        self._held_multi: tuple[int, list[Any]] | None = None

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
        self._held = None
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
            self._held = None
            if not ok:
                drv.abort()  # zeroes a held trip too (its payload never leaves)
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
        inline_stats = noise is not None and not be.noise_async_sampled
        hold = False  # the call's last trip is sampled and goes back to Python before its payload leaves
        flags = bytearray()
        left = self._left_flags
        self._left_flags = {}
        held = self._held
        self._held = None
        native_hold = inline_stats and self.native_hold
        for idx in range(first, end):
            lid, meta = self.lids[idx], self.metas[idx]
            if held is not None and held[0] == idx:
                # the trip the driver holds (prepared before its statistics ran): its lease and draws as
                # drawn then; the driver sends the held payload (no second prepare, no second draw)
                leases.append(held[1])
                flags.append(left.pop(idx) & ~(2 | 8))
                continue
            lease = be._trip_lease_for(res, leases_map, lid, meta, t)
            if lease is None:  # this trip takes the ordinary path (after the driver ran the ones before it)
                no_lease = idx
                end = idx
                break
            leases.append(lease)
            # the per-trip draws, in trip order (each trip's once), drawn only up to the call's last trip
            f = left.pop(idx, None)
            if f is None:
                check = be._should_check()
                sampled = bool(plan_call()) if plan_call is not None else False
                f = (1 if check else 0) | (2 if sampled else 0)
                if check and not lease.field and be._defer_ok(lid):
                    f |= 4
            if f & 2 and inline_stats and idx > first:
                # inline statistics: the sampled trip starts a call of its own (below), drawn once
                left[idx] = f
                leases.pop()
                end = idx
                break
            flags.append(f)
            if f & 2:
                # a sampled trip ends the call, so its statistics run before any later payload is built
                # and the next trip's sampling draw comes after them (the per-trip path's order). With
                # noise_async_sampled they run beside the trip's own send and are joined before the next
                # payload; inline, the driver hands the trip back after its prepare (no lane for it) and the
                # statistics run before its payload leaves: an alarm stops that payload, as on the per-trip path
                end = idx + 1
                hold = inline_stats
                if native_hold:
                    flags[-1] |= 8  # the driver keeps the prepared trip (code 8); sent by the next call
                break
        self._left_flags = left  # draws of trips past this call's end (after a hand-back), for the next call
        if no_lease is not None and end == first and inject is None:
            return self._ordinary_trip(sess, no_lease)
        rids = os.urandom(16 * (end - first))
        client = be.client
        lane_got = None
        lane_arg = None
        drv_ch = getattr(client, "driver_channel", None)
        if drv_ch is not None and end > first and not hold and getattr(client, "lane_eligible_for_driver", True):
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
        if code == 8:
            # held after its prepare: its statistics ran just above, before anything of it left; an alarm
            # stops it here (run() aborts the pass, which zeroes the held payload), as on the per-trip path
            j = int(drv.pos)
            if noise is not None and getattr(noise, "fail_closed", False) and getattr(noise, "alarmed", False):
                raise IntegrityError("outbound-noise monitor alarm: masked traffic blocked until an admin reset",
                                     details={"reasons": list(noise.report().reasons[-4:])})
            self._held = (j, leases[j - first])
            self._left_flags[j] = flags[j - first] & ~8
            return None
        if code == 0:
            return self._ordinary_trip(sess, no_lease) if no_lease is not None else None
        if code == 1:
            # trips after the handed-back one keep their draws for the next call (drawn once, in trip order)
            j = int(drv.pos)
            self._left_flags.update({idx: flags[idx - first] for idx in range(j + 1, end)})
            return self._handback(info, res, sess, t, end, first)
        lid = self.lids[int(drv.pos)] if int(drv.pos) < len(self.lids) else self.lids[-1]
        self._raise_code(code, info, lid, int(dup_rows), t)
        raise ProtocolError(f"compute response item is malformed: {info}")  # pragma: no cover - raised above

    def _raise_code(self, code: int, info: Any, lid: Any, dup_rows: int, t: int) -> None:
        """Raise the error of a failed driver call (:meth:`run` / :meth:`run_multi` codes 2-7)."""
        be = self.backend
        noise = be.noise_monitor
        if code == 2:
            status, bad = info[0], info[1]
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
            status, bad = info[0], info[1]
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

    # ---- several answers per pass (PGW_DECODE_NATIVE_MULTI) ------------------------------------------
    def eligible_multi(self, st: Any) -> bool:
        """A decode pass of several answers the driver runs (:meth:`run_multi`): each sequence a decode-sized pass
        on its own reservation, the leased several-answer trips on (``PGW_TRIP_NATIVE_MULTI``), and one lane multi
        call able to carry every answer's rows."""
        be = self.backend
        k = len(st.seqs)
        if not self.multi or k < 2 or st.extra or any(tr is not None for tr in st.trees):
            return False
        if not getattr(be, "trip_native_multi", False):
            return False
        noise = be.noise_monitor
        if noise is not None and (self.dup_hist is None or getattr(noise, "plan_call", None) is None):
            return False
        cap_rows = min(int(be.max_rows_per_item), MAX_ROWS)
        total = 0
        kvb = bool(self.model.float_ops.kv_bf16)
        for h, cache in zip(st.hs, st.caches, strict=True):
            t = int(h.shape[0])
            if not 0 < t <= cap_rows or h.dtype != torch.float32 or h.device.type != "cpu":
                return False
            if (cache.dtype == torch.bfloat16) != kvb:
                return False
            total += t
        multi_rows = getattr(be.client, "multi_rows", None)
        if multi_rows is None:
            return False
        cap = int(multi_rows())
        if total > cap or k > cap:
            return False
        if noise is not None:  # sampled values are inspected whole (no random block draw)
            mx = int(getattr(noise, "_max_values", 0) or 0)
            tmax = max(int(h.shape[0]) for h in st.hs)
            if any(tmax * int(m.in_features) > mx for m in self.metas):
                return False
        is_local = getattr(be, "local_linear", None)
        if is_local is not None and any(is_local(lid) for lid in self.lids):
            return False
        seen: set[int] = set()
        for sess in st.sessions:
            res = sess if sess is not None else be.mask_source
            if res is None or getattr(res, "trip_leases", None) is None or id(res) in seen:
                return False
            seen.add(id(res))
        return be._use_trip([_Call(self.lids[0], st.hs[0], st.sessions[0])]) and all(
            be._trip_self_check(m) for m in self.metas)

    def run_multi(self, st: Any) -> None:
        """Every decoder layer of a several-answer pass; writes ``st.hs`` (and leaves ``st.normed`` empty).

        The trips of the per-trip path of several answers (:meth:`MaskedBackend._prepare_trip_leases`): per trip
        one item per answer in answer order and one request id; every answer's lease resolved before any draw,
        then each answer's Freivalds and noise-sampling draws in answer order. A trip with a sampled answer ends
        the driver call and is held after its prepare: the statistics run before its payloads leave and before
        the next trip's draws (an alarm stops it, fail closed). A trip the lane does not serve goes back to
        Python with the same request (:meth:`_handback_multi`); a trip with no lease for some answer runs on the
        ordinary path (:meth:`_ordinary_multi`)."""
        be = self.backend
        model = self.model
        drv = self.drv
        if self.rows == 0:
            self.configure(1)  # the linears' constants (the several-answer items run one thread each)
        sessions = list(st.sessions)
        self._ts = [int(h.shape[0]) for h in st.hs]
        resv = [be._reservation(_Call(self.lids[0], h, s)) for h, s in zip(st.hs, sessions, strict=True)]
        rope_np = st.rope_np
        if rope_np is None:
            from pgw.engine.model import _np32

            rope_np = st.rope_np = [(_np32(c), _np32(s)) for c, s in st.rope]
        kvs = [c.numpy_views() for c in st.caches]
        pre = st.normed
        st.normed = None
        normed = None
        if pre is not None and pre[0] == 0:
            normed = [x.numpy() for x in pre[1]]
        hs = [h.detach().numpy() if h.is_contiguous() else np.ascontiguousarray(h.detach().numpy()) for h in st.hs]
        self._left_multi = {}
        self._held_multi = None
        drv.begin_multi(hs, [int(s) for s in st.starts], normed)
        ok = False
        try:
            inject = None
            L4 = 4 * self.layers
            while drv.mpos < L4:
                pos = int(drv.mpos)
                inject = self._call_multi(resv, sessions, min(L4, (pos // 4 + self.per_call) * 4), kvs, rope_np,
                                          inject)
            outs = [torch.empty((t, int(model.arch.hidden_size)), dtype=torch.float32) for t in self._ts]
            drv.end_multi([o.numpy() for o in outs])
            ok = True
        finally:
            self._left_multi = {}
            self._held_multi = None
            if not ok:
                drv.abort()  # zeroes a held trip too (its payloads never leave)
        st.hs = outs

    def _call_multi(self, resv: list[Any], sessions: list[Any], end: int, kvs: list[Any], ropes: list[Any],
                    inject: Any) -> Any:
        """One driver call of a several-answer pass from its position to trip ``end``; returns the outputs to
        inject (a trip finished here after a hand-back, or run on the ordinary path) or ``None``."""
        be = self.backend
        drv = self.drv
        from pgw.core.factory_yield import note_activity as _note_activity

        _note_activity()
        k = len(resv)
        ts = self._ts
        # as MaskedBackend._prepare: the sessions' deferred checks and sampled statistics first
        if be._noise_pending is not None:
            be._join_noise()
        if be._pending or be._failed:
            be._join_pending([_Call(self.lids[0], None, s) for s in sessions])
        pos = int(drv.mpos)
        first = pos + (1 if inject is not None else 0)
        noise = be.noise_monitor
        plan_call = getattr(noise, "plan_call", None) if noise is not None else None
        leases: list[Any] = []
        flags = bytearray()
        left = self._left_multi
        self._left_multi = {}
        held = self._held_multi
        self._held_multi = None
        no_lease = None
        for idx in range(first, end):
            lid, meta = self.lids[idx], self.metas[idx]
            if held is not None and held[0] == idx:
                # the trip the driver holds (prepared before its statistics ran): its leases and draws as made
                # then; the driver sends the held payloads (no second prepare, no second draw)
                leases.extend(held[1])
                flags.extend(left.pop(idx))
                continue
            got = []
            for res, t in zip(resv, ts, strict=True):
                lease = be._trip_lease_for(res, res.trip_leases, lid, meta, t, ahead=True)
                if lease is None:
                    break
                got.append(lease)
            if len(got) < k:  # this trip takes the ordinary path (after the driver ran the ones before it)
                no_lease = idx
                end = idx
                break
            f = left.pop(idx, None)
            if f is None:  # the per-trip draws, in trip order and answer order, each made once
                fb = bytearray()
                for lease in got:
                    check = be._should_check()
                    sampled = bool(plan_call()) if plan_call is not None else False
                    b = (1 if check else 0) | (2 if sampled else 0)
                    if check and not lease.field and be._defer_ok(lid):
                        b |= 4
                    fb.append(b)
                f = bytes(fb)
            leases.extend(got)
            if any(b & 2 for b in f):
                # a sampled answer ends the call: the driver holds the trip after its prepare (bit 3); its
                # statistics run before its payloads leave and before the next trip's draws
                flags.extend(b | 8 for b in f)
                end = idx + 1
                break
            flags.extend(f)
        if no_lease is not None and end == first and inject is None:
            return self._ordinary_multi(sessions, no_lease)
        rids = os.urandom(16 * (end - first))
        client = be.client
        lane_got = None
        lane_arg = None
        drv_ch = getattr(client, "driver_channel", None)
        only_hold = end - first == 1 and bool(flags) and flags[0] & 8
        if drv_ch is not None and end > first and not only_hold and getattr(client, "lane_eligible_for_driver", True):
            lane_got = drv_ch(be.fingerprint, be.ring.name, 1, be.deadline_ms)
            if lane_got is not None:
                ix = self._lane_index(lane_got[0])
                if ix is None or str(getattr(lane_got[1], "transport", "")) == "shm":
                    client.driver_release(lane_got[0], lane_got[1], lane_got[2])
                    lane_got = None
                else:
                    lane_arg = (lane_got[1], lane_got[3], ix,
                                int(be.deadline_ms if be.deadline_ms is not None else 0), lane_got[4])
        c0 = time.thread_time()
        try:
            code, info, stats, sampled_vals = drv.run_multi(end, leases, bytes(flags), rids, kvs, ropes, lane_arg,
                                                            self.dup_hist, inject, int(be.trip_many_parallel_elems))
        except BaseException:
            if lane_got is not None:
                client.driver_release(lane_got[0], lane_got[1], lane_got[2])
            raise
        (trips, rows, up, down, checked, skipped, deferred, deferred_ok, remote_ns, local_ns, lane_trips, lane_sent,
         lane_recv, rtt_ns, dup_rows, dups, items) = stats
        if lane_got is not None:
            st_ = 0
            err = None
            if code == 1 and info[3] >= 0:
                st_, err = (0, info[4]) if info[4] is not None else (int(info[3]), None)
            if st_:
                lane_trips += 1
                lane_sent += 48 + 8 * k + sum(len(p) for p in info[0])
            client.driver_release(lane_got[0], lane_got[1], lane_got[2], trips=lane_trips, sent=lane_sent,
                                  received=lane_recv, last_rtt_s=rtt_ns / 1e9, status=st_, error=err,
                                  multi_items=int(items))
        tot = self.totals
        tot[4] += time.thread_time() - c0
        tot[0] += 1
        tot[1] += int(trips)
        tot[2] += int(local_ns)
        tot[3] += int(remote_ns)
        self.multi_totals[0] += 1
        self.multi_totals[1] += int(trips)
        if trips:
            # as _finish_leases books each trip: the items as calls, one round trip each
            be.stats.add_many(int(items), int(rows), int(trips), int(up), int(down), int(checked) + int(deferred_ok),
                              int(skipped), remote_ns / 1e9, local_ns / 1e9,
                              0 < sum(ts) <= be.decode_max_rows)
        if deferred:
            be.stats.add(deferred_checks=int(deferred))
        if noise is not None and dup_rows and not dups:
            noise.record_native_dups(int(dup_rows), 0)
        if code == 8:
            # held after its prepare: the statistics of its sampled answers in answer order, before any of its
            # payloads leaves; an alarm stops it here (run_multi aborts the pass, zeroing the held payloads)
            self._sampled_multi(sampled_vals)
            if noise is not None and getattr(noise, "fail_closed", False) and getattr(noise, "alarmed", False):
                raise IntegrityError("outbound-noise monitor alarm: masked traffic blocked until an admin reset",
                                     details={"reasons": list(noise.report().reasons[-4:])})
            j = int(drv.mpos)
            base = (j - first) * k
            self._left_multi[j] = bytes(b & ~8 for b in flags[base:base + k])
            self._held_multi = (j, leases[base:base + k])
            return None
        self._sampled_multi(sampled_vals)
        if code == 0:
            return self._ordinary_multi(sessions, no_lease) if no_lease is not None else None
        if code == 1:
            # trips after the handed-back one keep their draws for the next call (drawn once, in trip order)
            j = int(drv.mpos)
            self._left_multi.update({idx: bytes(flags[(idx - first) * k:(idx - first + 1) * k])
                                     for idx in range(j + 1, end)})
            fl = bytes(flags[(j - first) * k:(j - first + 1) * k])
            return self._handback_multi(info, resv, sessions, fl)
        mpos = int(drv.mpos)
        lid = self.lids[mpos] if mpos < len(self.lids) else self.lids[-1]
        self._raise_code(code, info, lid, int(dup_rows), max(ts))
        raise ProtocolError(f"compute response item is malformed: {info}")  # pragma: no cover - raised above

    def _sampled_multi(self, vals: list[Any]) -> None:
        """The noise monitor's statistics of a held trip's sampled answers, in answer order (inline)."""
        if not vals:
            return
        noise = self.backend.noise_monitor
        for _pos, _a, xh, xq in vals:
            try:
                if noise is not None:
                    noise.inspect_planned(torch.from_numpy(xh), torch.from_numpy(xq))
            finally:
                xh.fill(0)
                xq.fill(0)

    def _ordinary_multi(self, sessions: list[Any], idx: int) -> Any:
        """Trip ``idx`` of a several-answer pass on the backend's ordinary path (no native lease for some
        answer); the outputs to inject."""
        if int(self.drv.mpos) != idx:  # pragma: no cover - the driver stopped right before it
            raise ShapeError("native decode driver out of step")
        xs = self.drv.current_inputs()
        lid = self.lids[idx]
        outs = self.backend.forward_many([_Call(lid, torch.from_numpy(x), s) for x, s in zip(xs, sessions,
                                                                                            strict=True)])
        return [np.ascontiguousarray(o.to(torch.float32).numpy(), dtype=np.float32) for o in outs]

    def _handback_multi(self, info: Any, resv: list[Any], sessions: list[Any], flags: bytes) -> Any:
        """Send a several-answer trip the driver handed back on the ordinary fallback (the same request: items in
        answer order, payloads and id) and finish it (:meth:`MaskedBackend._finish_leases`)."""
        from pgw.engine.backends import _Prepared
        from pgw.wire.messages import ComputeItem, ComputeRequest

        be = self.backend
        payloads, tickets, rid, lane_status, _lane_err = info
        idx = int(self.drv.mpos)
        lid, meta = self.lids[idx], self.metas[idx]
        calls = [_Call(lid, _CPU_ROW, s) for s in sessions]
        request = ComputeRequest(request_id=bytes(rid).hex(), fingerprint=be.fingerprint, ring=be.ring.name,
                                 items=tuple(ComputeItem(lid.key, int(tk.rows), pl)
                                             for tk, pl in zip(tickets, payloads, strict=True)),
                                 deadline_ms=be.deadline_ms)
        prep = _Prepared(tuple(calls), [], [], [meta] * len(calls), request, 0.0)
        prep.trip = True
        prep.tickets = list(tickets)
        prep.defers = [bool(f & 4) for f in flags]
        prep.reservations = tuple(resv)
        client = be.client
        noise = be.noise_monitor
        if noise is not None and getattr(noise, "fail_closed", False) and getattr(noise, "alarmed", False):
            prep.wipe()
            raise IntegrityError("outbound-noise monitor alarm: masked traffic blocked until an admin reset",
                                 details={"reasons": list(noise.report().reasons[-4:])})
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
        outs = be._finish_leases(prep, response, time.perf_counter() - t0)
        if be._pending or be._failed:  # deferred checks of this trip: joined before the next payload
            be._join_pending(calls)
        return [np.ascontiguousarray(o.numpy(), dtype=np.float32) for o in outs]

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
        noise = be.noise_monitor
        if noise is not None and getattr(noise, "fail_closed", False) and getattr(noise, "alarmed", False):
            prep.wipe()  # the trip's statistics (just inspected) alarmed: its payload never leaves
            raise IntegrityError("outbound-noise monitor alarm: masked traffic blocked until an admin reset",
                                 details={"reasons": list(noise.report().reasons[-4:])})
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
