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
                 ring_index: int = 0, max_rows: int = 8, deadline_ms: int = 0, timeout_s: float = 60.0):
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

    def single(self, k1: str, x_hat: torch.Tensor) -> torch.Tensor:
        n1, m1, _ = self._items[k1]
        rows = int(x_hat.shape[0])
        outs = []
        for a, b in self._chunks(rows):
            r = b - a
            y = self._call(k1, r, pack_ring(x_hat[a:b], self.k), r * m1 * self.bpe)
            outs.append(unpack_ring(y, (r, m1), self.k))
        return torch.cat(outs) if len(outs) != 1 else outs[0]

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
    return LaneFused2Provider(ch, names, items_of(weights), k, max_rows=max_rows, timeout_s=timeout_s)
