"""fused2 pair-mask segments: the factory holds the weights, the gateway does not (opt-in, default off).

The no-chain trick (``research/plans/fused2-two-trips-per-layer.md`` 2.2): per pair trip ``r_x`` and ``v`` are
uniform and independent, the gateway masks the residual with ``r_R = v - hmul * (W1 r_x)``, and unmasks with ``v``
and ``W2' v``. So a slot needs, per trip key, the two uniform streams and two **independent single-stage
products** ``W1 r_x`` and ``W2' v`` (the W1 correction and the W2' correction). The products need the weights;
the gateway does not have them, so a customer-side plain factory (:class:`Fused2MaskFactory`, the same process
that holds the pack for today's mask factory) writes them into **segments** the gateway consumes
(:class:`Fused2SegmentMasks`), and gives the gateway the secret Freivalds check vectors once
(:meth:`Fused2MaskFactory.check_blob`: ``u`` and ``(u * hmul) W1 mod p`` / ``u W2' mod p`` per linear).

Segment format (little endian, version 1)::

    magic  b"PGWF2S01" | u8 K | 3 x u8 0 | u32 rows | u64 first_row | 32 B fused2 lane fingerprint | u32 trips
    per trip of the pass, in trip order (:func:`pgw.service.fused2_lane.fused2_items`):
        u16 len | item name (utf-8) | u8 pair | 32 B stream key of r_x | (pair) 32 B stream key of v
        packed W1 r_x  (rows * m1 elements of K/8 bytes)
        (pair) packed W2' v  (rows * m2 elements)
    32 B SHA-256 of everything before it

``r_x`` and ``v`` are the rows ``[first_row, first_row + rows)`` of the ChaCha20 streams whose keys the segment
carries (:meth:`pgw.core.prng.SeedExpander.ring_uniform_rows_with_key`): the same rows
:class:`pgw.engine.fused2.Fused2MaskSource` draws for the same seed, so the two sources are bit-identical. A
segment is as secret as the masks (keys included); it is used once (cursors only move forward, consumed rows are
wiped). One slot = one row of every trip key = one token row of one pass (the same elements as today's slot, at
``K/8`` bytes each).
"""

from __future__ import annotations

import hashlib
import io
import struct
import threading
from dataclasses import dataclass
from typing import Any

import torch

from pgw.core.errors import ConfigError, IntegrityError, ProtocolError
from pgw.core.packing import pack_ring, unpack_ring
from pgw.core.prng import SeedExpander
from pgw.core.ring import RingParams
from pgw.engine.fused2 import Fused2MaskSource, Fused2Trip, TripMasks, _Checks

__all__ = ["SEGMENT_MAGIC", "Fused2MaskFactory", "Fused2SegmentMasks"]

SEGMENT_MAGIC = b"PGWF2S01"
_HEAD = struct.Struct("<8sB3xIQ32sI")


def _fp_bytes(fp: str) -> bytes:
    raw = bytes.fromhex(fp) if fp else b""
    if len(raw) not in (0, 32):
        raise ConfigError("fused2 lane fingerprint must be 64 hex characters")
    return raw.ljust(32, b"\0")


class Fused2MaskFactory(Fused2MaskSource):
    """Customer-side plain factory for fused2 (holds the public weights; never the gateway)."""

    def __init__(self, weights: Any, ring: RingParams, *, lane_fingerprint: str = "", **kw: Any):
        super().__init__(weights, ring, **kw)
        self.lane_fingerprint = lane_fingerprint
        self._next = 0

    def check_blob(self) -> bytes:
        """The secret check vectors of every linear (``u``, ``u W mod p`` with the stage-1 multipliers folded)."""
        out = {}
        for lin in self.w.linears():
            ch = self.check_vectors(lin)
            out[lin.key] = {"u": torch.from_numpy(ch.u), "uw": torch.from_numpy(ch.uw)}
        buf = io.BytesIO()
        torch.save({"fingerprint": self.lane_fingerprint, "checks": out}, buf)
        return buf.getvalue()

    def segment(self, rows: int) -> bytes:
        """The next ``rows`` slots as one segment (advances the factory's cursor)."""
        from pgw.service.fused2_lane import fused2_items

        if rows < 1:
            raise ConfigError("a segment holds at least one slot")
        k = self.ring.k
        by_key = {lin.key: lin for lin in self.w.linears()}
        start = self._next
        self._next += rows
        items = fused2_items(self.w)
        parts = [_HEAD.pack(SEGMENT_MAGIC, k, rows, start, _fp_bytes(self.lane_fingerprint), len(items))]
        for it in items:
            key = it.name  # the trip key of Fused2MaskSource: "k1" or "k1+k2"
            kr = self._sx.stream_key("f2", "r", key)
            r_x = SeedExpander.ring_uniform_rows_with_key(kr, it.n1, k, start, rows)
            wr = self._prepared(by_key[it.k1]).matmul(r_x, k)
            self.macs += rows * it.n1 * it.m1
            name = key.encode()
            parts += [struct.pack("<H", len(name)), name, bytes([1 if it.k2 else 0]), kr]
            if it.k2 is not None:
                kv = self._sx.stream_key("f2", "v", key)
                v = SeedExpander.ring_uniform_rows_with_key(kv, it.m1, k, start, rows)
                wv = self._prepared(by_key[it.k2]).matmul(v, k)
                self.macs += rows * it.m1 * it.m2
                parts += [kv, pack_ring(wr, k), pack_ring(wv, k)]
                v.zero_()
            else:
                parts.append(pack_ring(wr, k))
            r_x.zero_()
        body = b"".join(parts)
        return body + hashlib.sha256(body).digest()


@dataclass
class _Seg:
    first: int
    rows: int
    kr: bytes
    kv: bytes | None
    wr: torch.Tensor
    wv: torch.Tensor | None


class Fused2SegmentMasks:
    """The gateway's fused2 mask source from factory segments (no weights; the ``take`` / ``check_vectors``
    interface of :class:`pgw.engine.fused2.Fused2MaskSource`).

    Args:
        weights: The gateway's public fused2 tables (shapes, scales, multipliers; ``w8`` is not used).
        ring: The fused2 ring.
        check_blob: :meth:`Fused2MaskFactory.check_blob` (secret).
        lane_fingerprint: The fused2 lane fingerprint segments must carry (empty: not checked).
    """

    def __init__(self, weights: Any, ring: RingParams, check_blob: bytes, *, lane_fingerprint: str = ""):
        from pgw.service.fused2_lane import fused2_items

        self.w = weights
        self.ring = ring
        self.lane_fingerprint = lane_fingerprint
        blob = torch.load(io.BytesIO(check_blob), weights_only=True)
        if lane_fingerprint and blob.get("fingerprint") not in ("", lane_fingerprint):
            raise ConfigError("fused2 check vectors belong to another weight set")
        self._checks = {k: _Checks(v["u"].numpy(), v["uw"].numpy()) for k, v in blob["checks"].items()}
        self._items = {it.name: it for it in fused2_items(weights)}
        self._order = [it.name for it in fused2_items(weights)]
        self._segs: dict[str, list[_Seg]] = {name: [] for name in self._order}
        self._cursor: dict[str, int] = {name: 0 for name in self._order}
        self._end = 0  # first row past the last added segment
        self._lock = threading.Lock()
        self.macs = 0  # the gateway does no factory products
        self.segments = 0
        self.stock_hits = 0
        self.stock_misses = 0

    def check_vectors(self, lin: Any) -> _Checks:
        try:
            return self._checks[lin.key]
        except KeyError as exc:
            raise ConfigError(f"fused2: no check vectors for {lin.key}") from exc

    def add_segment(self, data: bytes | bytearray | memoryview) -> int:
        """Verify and stock one segment; returns its slot count. Segments must be added in row order."""
        mv = memoryview(data)
        if len(mv) < _HEAD.size + 32 or hashlib.sha256(mv[:-32]).digest() != bytes(mv[-32:]):
            raise IntegrityError("fused2 mask segment is corrupt")
        magic, k, rows, first, fp, ntrips = _HEAD.unpack_from(mv, 0)
        if magic != SEGMENT_MAGIC or k != self.ring.k or ntrips != len(self._order):
            raise ProtocolError("fused2 mask segment does not match this weight set or ring")
        if self.lane_fingerprint and fp != _fp_bytes(self.lane_fingerprint):
            raise ProtocolError("fused2 mask segment belongs to another weight set")
        bpe = k // 8
        pos = _HEAD.size
        parsed: list[tuple[str, _Seg]] = []
        for name in self._order:
            (ln,) = struct.unpack_from("<H", mv, pos)
            pos += 2
            if bytes(mv[pos:pos + ln]).decode() != name:
                raise ProtocolError("fused2 mask segment trips are out of order")
            pos += ln
            it = self._items[name]
            pair = mv[pos] == 1
            pos += 1
            if pair != (it.k2 is not None):
                raise ProtocolError("fused2 mask segment trip kind mismatch")
            kr = bytes(mv[pos:pos + 32])
            pos += 32
            kv = None
            if pair:
                kv = bytes(mv[pos:pos + 32])
                pos += 32
            nb = rows * it.m1 * bpe
            wr = unpack_ring(mv[pos:pos + nb], (rows, it.m1), k)
            pos += nb
            wv = None
            if pair:
                nb = rows * it.m2 * bpe
                wv = unpack_ring(mv[pos:pos + nb], (rows, it.m2), k)
                pos += nb
            parsed.append((name, _Seg(first, rows, kr, kv, wr, wv)))
        if pos != len(mv) - 32:
            raise ProtocolError("fused2 mask segment has trailing bytes")
        with self._lock:
            if first != self._end:
                raise ProtocolError(f"fused2 mask segment starts at row {first}, expected {self._end}")
            for name, seg in parsed:
                self._segs[name].append(seg)
            self._end = first + rows
            self.segments += 1
        return int(rows)

    def stocked_rows(self) -> int:
        with self._lock:
            return min(self._end - c for c in self._cursor.values())

    def slots_used(self) -> int:
        """Slots consumed (rows of the most-used trip key)."""
        with self._lock:
            return max(self._cursor.values())

    def take(self, trip: Fused2Trip) -> TripMasks:
        rows = int(trip.x.shape[0])
        key = trip.l1.key + ("+" + trip.l2.key if trip.l2 is not None else "")
        it = self._items.get(key)
        if it is None:
            raise ConfigError(f"fused2: unknown trip {key}")
        k = self.ring.k
        with self._lock:
            start = self._cursor[key]
            if start + rows > self._end:
                self.stock_misses += 1
                raise ConfigError(f"fused2: mask segments exhausted ({key}: need {rows} rows, "
                                  f"{self._end - start} left)")
            self._cursor[key] = start + rows
            segs = self._segs[key]
            parts: list[TripMasks] = []
            pos, need = start, rows
            while need:
                s = segs[0]
                off = pos - s.first
                n = min(need, s.rows - off)
                r_x = SeedExpander.ring_uniform_rows_with_key(s.kr, it.n1, k, pos, n)
                tm = TripMasks(r_x, s.wr[off:off + n].clone())
                if s.kv is not None:
                    assert s.wv is not None
                    tm.v = SeedExpander.ring_uniform_rows_with_key(s.kv, it.m1, k, pos, n)
                    tm.wv = s.wv[off:off + n].clone()
                s.wr[off:off + n].zero_()
                if s.wv is not None:
                    s.wv[off:off + n].zero_()
                parts.append(tm)
                pos += n
                need -= n
                if pos >= s.first + s.rows:
                    segs.pop(0)
            self.stock_hits += 1
        if len(parts) == 1:
            return parts[0]
        cat = TripMasks(torch.cat([p.r_x for p in parts]), torch.cat([p.wr_x for p in parts]))
        if parts[0].v is not None:
            cat.v = torch.cat([p.v for p in parts])  # type: ignore[misc]
            cat.wv = torch.cat([p.wv for p in parts])  # type: ignore[misc]
        for p in parts:
            p.wipe()
        return cat


def gateway_tables(weights: Any) -> Any:
    """A copy of ``weights`` without the int8 matrices (stride-0 zero views of the same shapes): what a weight-free
    gateway holds (scales, biases, norms, bounds, multipliers). Used by tests to prove the gateway path never
    reads ``w8``."""
    import copy
    import dataclasses

    w = copy.copy(weights)
    for part in ("qkv", "o", "gu", "down"):
        setattr(w, part, [dataclasses.replace(lin, w8=torch.zeros((), dtype=torch.int8).expand(lin.m, lin.n))
                          for lin in getattr(weights, part)])
    return w



def main(argv: list[str] | None = None) -> None:  # pragma: no cover - process entry point (e2e harness)
    """``python -m pgw.engine.fused2_masks factory``: the customer-side plain factory as its own process. Writes the
    secret check vectors (``check.bin``) and ``--segments`` segments of ``--rows`` slots (``seg-00000.bin``, ...)
    into ``--out`` (files mode 0600) for a weight-free gateway (:class:`Fused2SegmentMasks`)."""
    import argparse
    import os
    from pathlib import Path

    from pgw.engine.fused2 import Fused2Weights

    ap = argparse.ArgumentParser(description="fused2 plain mask factory (opt-in harness)")
    ap.add_argument("factory", choices=["factory"])
    ap.add_argument("--weights", required=True, help="Fused2Weights.save() file (full public weights)")
    ap.add_argument("--ring", default="K56A12C53")
    ap.add_argument("--lane-fingerprint", default="")
    ap.add_argument("--rows", type=int, required=True, help="slots per segment")
    ap.add_argument("--segments", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ring = RingParams.from_name(args.ring)
    fac = Fused2MaskFactory(Fused2Weights.load(args.weights), ring, lane_fingerprint=args.lane_fingerprint)
    out = Path(args.out)
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    old = os.umask(0o177)
    try:
        (out / "check.bin.tmp").write_bytes(fac.check_blob())
        os.replace(out / "check.bin.tmp", out / "check.bin")
        for i in range(args.segments):
            (out / f"seg-{i:05d}.bin.tmp").write_bytes(fac.segment(args.rows))
            os.replace(out / f"seg-{i:05d}.bin.tmp", out / f"seg-{i:05d}.bin")
    finally:
        os.umask(old)
    print(f"fused2 factory: {args.segments} x {args.rows} slots, {fac.macs / max(1, args.rows * args.segments):.0f} "
          f"MACs per slot", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
