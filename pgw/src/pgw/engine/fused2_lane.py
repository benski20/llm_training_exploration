"""Gateway client of the fused2 pair items on the service trip lane (opt-in, default off).

:class:`LaneFused2Provider` has the :class:`pgw.engine.fused2.Fused2Provider` interface (``single`` / ``pair`` on
masked ring elements), so :class:`pgw.engine.fused2.Fused2MaskedBackend` masks, unmasks, range-checks and
Freivalds-checks both stages exactly as before; only the transport changes: one lane call per trip, over the
native ``pgw_fast.TripChannel`` (Unix socket, optionally the shared-memory segment) with the ticket's tables.

Item table and body layout: :mod:`pgw.service.fused2_lane` (pair body ``pack(x_hat) || pack(R_hat)``, reply
``pack(H_hat) || pack(Z_hat)``). A trip with more rows than the channel serves is sent as consecutive calls of at
most ``max_rows`` rows (rows are independent; the same elements in total). Every refusal raises
:class:`pgw.core.errors.ProtocolError`: the backend never retries a pair item with a new mask.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Sequence
from typing import Any

import torch

from pgw.core.errors import ConfigError, ProtocolError
from pgw.core.packing import pack_ring, unpack_ring

__all__ = ["LaneFused2Provider", "connect_uds", "items_of"]


class LaneFused2Provider:
    """fused2 provider over one lane channel (``trip(ring, linear, rows, rid, deadline_ms, body, out, timeout)``).

    Args:
        channel: A connected, opened lane channel.
        names: The ticket's item table (``linears``), in order.
        items: ``{name: (n1, m1, m2)}`` widths from the gateway's own public tables (``m2 = 0``: a single).
        k: Ring bits.
        ring_index: Index of the ring in the ticket's ring table.
        max_rows: Rows per call (the channel's maximum).
    """

    def __init__(self, channel: Any, names: Sequence[str], items: dict[str, tuple[int, int, int]], k: int, *,
                 ring_index: int = 0, max_rows: int = 8, deadline_ms: int = 0, timeout_s: float = 60.0,
                 hmul: dict[str, torch.Tensor] | None = None, max_items: int = 0, max_multi_rows: int = 0):
        self.ch = channel
        self.k = int(k)
        self.bpe = self.k // 8
        self._ix = {name: i for i, name in enumerate(names)}
        missing = [n for n in items if n not in self._ix]
        if missing:
            raise ConfigError(f"fused2 lane: the service's table lacks items {missing[:3]}")
        self._items = dict(items)
        self.ring_index = int(ring_index)
        self.max_rows = max(1, int(max_rows))
        self.deadline_ms = int(deadline_ms)
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        #: Per call: provider-reported compute (us) and the wall time of the call (ms).
        self.compute_us: list[int] = []
        self.trip_ms: list[float] = []
        self.calls = 0
        self.bytes_up = 0
        self.bytes_down = 0
        #: When a list: every request body sent (tests compare what the provider saw across transports).
        self.record: list[tuple[str, int, bytes]] | None = None
        #: Public multipliers of single items whose first linear has them (head_fold's rows without logits: the
        #: last down). The service's single item returns ``W1 x_hat``; this client multiplies by ``hmul`` mod 2^K
        #: (public, exact), so :meth:`single` returns ``hmul * W1 x_hat`` like the in-process provider.
        self._hmul = {key: torch.as_tensor(h, dtype=torch.int64) for key, h in (hmul or {}).items()}
        #: Lane format 2 (multi frames): the most items / rows one call may carry (0: one item per call).
        self.max_items = int(max_items)
        self.max_multi_rows = int(max_multi_rows)
        self.multi_calls = 0

    @property
    def compute_ms(self) -> list[float]:
        """Provider-reported compute per call, ms (the socket harness's field)."""
        return [u / 1e3 for u in self.compute_us]

    def _call(self, name: str, rows: int, body: bytes, out_len: int) -> bytearray:
        out = bytearray(out_len)
        rid = secrets.token_bytes(16)
        with self._lock:
            t0 = time.perf_counter()
            st, cu = self.ch.trip(self.ring_index, self._ix[name], rows, rid, self.deadline_ms, body, out,
                                  self.timeout_s)
            self.trip_ms.append((time.perf_counter() - t0) * 1e3)
        if st != 0:
            raise ProtocolError(f"fused2 lane call {name!r} refused (status {st})")
        self.calls += 1
        self.compute_us.append(int(cu))
        self.bytes_up += len(body)
        self.bytes_down += out_len
        if self.record is not None:
            self.record.append((name, rows, bytes(body)))
        return out

    def _chunks(self, rows: int) -> list[tuple[int, int]]:
        return [(a, min(rows, a + self.max_rows)) for a in range(0, rows, self.max_rows)]

    def _post_single(self, k1: str, y: torch.Tensor) -> torch.Tensor:
        hm = self._hmul.get(k1)
        if hm is not None:  # public multipliers, exact mod 2^K (what the in-process provider applies)
            y = (y * hm) & ((1 << self.k) - 1)
        return y

    def single(self, k1: str, x_hat: torch.Tensor) -> torch.Tensor:
        n1, m1, _ = self._items[k1]
        rows = int(x_hat.shape[0])
        outs = []
        for a, b in self._chunks(rows):
            r = b - a
            y = self._call(k1, r, pack_ring(x_hat[a:b], self.k), r * m1 * self.bpe)
            outs.append(unpack_ring(y, (r, m1), self.k))
        return self._post_single(k1, torch.cat(outs) if len(outs) != 1 else outs[0])

    def multi_ok(self, reqs: Sequence[tuple[str, str | None, torch.Tensor, torch.Tensor | None]]) -> bool:
        """Whether :meth:`many` can carry ``reqs`` in one lane-format-2 call (items and rows within the offer)."""
        if self.max_items < 2 or len(reqs) < 2 or len(reqs) > self.max_items:
            return False
        rows = [int(x.shape[0]) for _, _, x, _ in reqs]
        return min(rows) > 0 and sum(rows) <= self.max_multi_rows and max(rows) <= self.max_rows

    def many(self, reqs: Sequence[tuple[str, str | None, torch.Tensor, torch.Tensor | None]]
             ) -> list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """Several items of one trip in **one** lane-format-2 multi call (one round trip): ``reqs[i] = (k1, k2,
        x_hat, r_hat)`` (``k2``/``r_hat`` None: a single). Returns per item ``y_hat`` (single, multipliers applied
        as :meth:`single`) or ``(H_hat, Z_hat)``. The same item bodies as one call per item (the provider sees the
        same masked elements); :meth:`multi_ok` must hold."""
        names, rows, bodies, outs_len = [], [], [], 0
        widths = []
        for k1, k2, x_hat, r_hat in reqs:
            name = k1 if k2 is None else f"{k1}+{k2}"
            _, m1, m2 = self._items[name]
            r = int(x_hat.shape[0])
            body = pack_ring(x_hat, self.k) + (b"" if k2 is None else pack_ring(r_hat, self.k))
            names.append(name)
            rows.append(r)
            bodies.append(body)
            widths.append((r, m1, m2))
            outs_len += r * (m1 + m2) * self.bpe
        out = bytearray(outs_len)
        rid = secrets.token_bytes(16)
        with self._lock:
            t0 = time.perf_counter()
            st, cu = self.ch.trip_multi(self.ring_index, [self._ix[n] for n in names], rows, rid, self.deadline_ms,
                                        bodies, out, self.timeout_s)
            self.trip_ms.append((time.perf_counter() - t0) * 1e3)
        if st != 0:
            raise ProtocolError(f"fused2 lane multi call {names!r} refused (status {st})")
        self.calls += 1
        self.multi_calls += 1
        self.compute_us.append(int(cu))
        self.bytes_up += sum(len(b) for b in bodies)
        self.bytes_down += outs_len
        if self.record is not None:
            for n, r, b in zip(names, rows, bodies, strict=True):
                self.record.append((n, r, bytes(b)))
        res: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        mv = memoryview(out)
        at = 0
        for (k1, k2, _, _), (r, m1, m2) in zip(reqs, widths, strict=True):
            a = r * m1 * self.bpe
            if k2 is None:
                res.append(self._post_single(k1, unpack_ring(mv[at : at + a], (r, m1), self.k)))
                at += a
            else:
                b = r * m2 * self.bpe
                res.append((unpack_ring(mv[at : at + a], (r, m1), self.k),
                            unpack_ring(mv[at + a : at + a + b], (r, m2), self.k)))
                at += a + b
        return res

    def pair(self, k1: str, k2: str, r_hat: torch.Tensor, x_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        name = f"{k1}+{k2}"
        n1, m1, m2 = self._items[name]
        rows = int(x_hat.shape[0])
        hs, zs = [], []
        for a, b in self._chunks(rows):
            r = b - a
            body = pack_ring(x_hat[a:b], self.k) + pack_ring(r_hat[a:b], self.k)
            y = self._call(name, r, body, r * (m1 + m2) * self.bpe)
            cut = r * m1 * self.bpe
            hs.append(unpack_ring(memoryview(y)[:cut], (r, m1), self.k))
            zs.append(unpack_ring(memoryview(y)[cut:], (r, m2), self.k))
        if len(hs) == 1:
            return hs[0], zs[0]
        return torch.cat(hs), torch.cat(zs)

    def close(self) -> None:
        close = getattr(self.ch, "close", None)
        if callable(close):
            close()


def single_multipliers(weights: Any) -> dict[str, torch.Tensor]:
    """``{k1: hmul}`` of the single items whose first linear has public multipliers (the last down with the folded
    head: its rows without logits)."""
    from pgw.service.fused2_lane import fused2_items

    by_key = {lin.key: lin for lin in weights.all_linears()}
    return {it.k1: by_key[it.k1].hmul for it in fused2_items(weights)
            if it.k2 is None and by_key[it.k1].hmul is not None}


def items_of(weights: Any) -> dict[str, tuple[int, int, int]]:
    """``{item name: (n1, m1, m2)}`` from the gateway's public fused2 tables (no weights needed)."""
    from pgw.service.fused2_lane import fused2_items

    return {it.name: (it.n1, it.m1, it.m2) for it in fused2_items(weights)}


def connect_uds(path: str, ticket: bytes, names: Sequence[str], weights: Any, k: int, *, shm: bool = False,
                max_rows: int = 8, spin_us: int = 50, timeout_s: float = 60.0) -> LaneFused2Provider:
    """Open a native lane channel on the service's Unix socket (``shm``: move it to the shared-memory segment;
    refused -> :class:`ConfigError`) and wrap it."""
    from pgw.core.native import fast

    if fast is None or not hasattr(fast, "TripChannel"):
        raise ConfigError("fused2 lane: the pgw_fast extension with the lane client is not installed")
    ch = fast.TripChannel.connect_service_uds(path, ticket, float(timeout_s), int(max_rows), int(spin_us))
    if shm and not ch.attach_shm():
        ch.close()
        raise ConfigError("fused2 lane: the service refused the shared-memory transport")
    return LaneFused2Provider(ch, names, items_of(weights), k, max_rows=max_rows, timeout_s=timeout_s,
                              hmul=single_multipliers(weights))
