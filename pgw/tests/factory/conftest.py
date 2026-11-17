"""Fixtures for the mask factory tests: a fake pack, a fake (in-memory) mask store and helpers."""

from __future__ import annotations

import json
import struct
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from pgw.core import LINEAR_ORDER, LinearId, LinearKind, LinearSpec, RingParams, SeedExpander, ringmm_reference

# odd, non-multiple-of-8 dims on purpose
FAKE_DIMS = {LinearKind.QKV: (40, 20), LinearKind.O: (20, 24), LinearKind.GATE_UP: (52, 20), LinearKind.DOWN: (20, 26)}


class FakePack:
    """Minimal ModelPack stand-in: random int8 linears (including -128/127 extremes)."""

    def __init__(self, layers: int = 2, seed: int = 0, name: str = "fake", dims: dict[Any, tuple[int, int]] | None = None):
        self.name = name
        gen = torch.Generator().manual_seed(seed)
        self.fingerprint = f"{seed:02x}" * 32
        d = dims or FAKE_DIMS
        self._specs = [LinearSpec(LinearId(i, k), *d[k]) for i in range(layers) for k in LINEAR_ORDER]
        self.weights: dict[str, torch.Tensor] = {}
        for s in self._specs:
            w = torch.randint(-128, 128, (s.out_features, s.in_features), dtype=torch.int8, generator=gen)
            w[0, 0] = -128
            w[-1, -1] = 127
            self.weights[s.id.key] = w
        self.loads: list[str] = []

    def linear_specs(self) -> list[LinearSpec]:
        return list(self._specs)

    def check_ring(self, params: RingParams) -> None:
        for s in self._specs:
            params.check_in_features(s.in_features)

    def load_linear(self, lid: LinearId | str, device: str | torch.device = "cpu") -> Any:
        self.loads.append(str(lid))
        return SimpleNamespace(w8=self.weights[str(lid)].to(device))


@dataclass
class FakeSegment:
    segment_id: str
    fingerprint: str
    ring: RingParams
    specs: list[LinearSpec]
    slots: int
    seed: bytes
    checks: int
    chunk_slots: int
    factory_id: str
    wr: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    uw: dict[str, torch.Tensor] = field(default_factory=dict)
    consumed: int = 0


class FakeWriter:
    """Mimics SegmentWriter: linear-major, in-order chunk writes; copies inputs (caller zeroizes)."""

    def __init__(self, store: FakeMaskStore, seg: FakeSegment, fail_after: int | None = None):
        self.store = store
        self.seg = seg
        self.fail_after = fail_after
        self.calls = 0
        self.committed = False
        self.aborted = False
        self._order = [s.id.key for s in seg.specs]
        self._cur = 0

    def write_linear(self, lid: LinearId, wr: torch.Tensor, uw: torch.Tensor) -> None:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise OSError("disk full (simulated)")
        key = lid.key
        spec = next(s for s in self.seg.specs if s.id.key == key)
        rows_done = sum(t.shape[0] for t in self.seg.wr.get(key, []))
        if rows_done == 0:
            # must be the next linear in header order
            assert self._order[self._cur] == key, f"out of order write {key}"
        else:
            assert self._order[self._cur] == key
        assert wr.dtype == torch.int64 and wr.shape[1] == spec.out_features
        assert wr.shape[0] <= self.seg.chunk_slots
        assert uw.shape == (self.seg.checks, spec.in_features)
        if key in self.seg.uw:
            assert torch.equal(self.seg.uw[key], uw), "uW changed between chunks"
        self.seg.uw[key] = uw.clone()
        self.seg.wr.setdefault(key, []).append(wr.clone())
        if rows_done + wr.shape[0] == self.seg.slots:
            self._cur += 1
        assert rows_done + wr.shape[0] <= self.seg.slots

    def commit(self) -> str:
        assert not self.aborted
        assert self._cur == len(self._order), "segment incomplete"
        self.committed = True
        self.store.segments[self.seg.segment_id] = self.seg
        return self.seg.segment_id

    def abort(self) -> None:
        self.aborted = True
        self.store.aborted += 1


class FakeMaskStore:
    """In-memory MaskStore stand-in with inventory, writer and low-water listeners."""

    def __init__(self, root: Path | None = None, low: int = 100, target: int = 300):
        self.segments: dict[str, FakeSegment] = {}
        self.writers: list[FakeWriter] = []
        self.aborted = 0
        self.listeners: list[Any] = []
        self.fail_after: int | None = None
        self.settings = SimpleNamespace(root=root, low_water_slots=low, target_slots=target)

    def writer(self, fingerprint: str, ring: RingParams, specs: Sequence[LinearSpec], *, slots: int, seed: bytes,
               checks: int = 2, chunk_slots: int = 256, factory_id: str = "local") -> FakeWriter:
        assert isinstance(ring, RingParams)
        seg = FakeSegment(uuid.uuid4().hex, fingerprint, ring, list(specs), slots, bytes(seed), checks, chunk_slots,
                          factory_id)
        w = FakeWriter(self, seg, self.fail_after)
        self.writers.append(w)
        return w

    def inventory(self, fingerprint: str | None = None, ring: RingParams | None = None) -> list[Any]:
        agg: dict[tuple[str, str], list[FakeSegment]] = {}
        for s in self.segments.values():
            if (fingerprint is None or s.fingerprint == fingerprint) and (ring is None or s.ring == ring):
                agg.setdefault((s.fingerprint, s.ring.name), []).append(s)
        return [SimpleNamespace(fingerprint=fp, ring=r, segments=len(v), slots_total=sum(x.slots for x in v),
                                slots_available=sum(x.slots - x.consumed for x in v), bytes_on_disk=0)
                for (fp, r), v in agg.items()]

    def consume(self, fingerprint: str, slots: int) -> None:
        for s in self.segments.values():
            if s.fingerprint != fingerprint:
                continue
            take = min(slots, s.slots - s.consumed)
            s.consumed += take
            slots -= take

    def add_low_water_listener(self, fn: Any) -> None:
        self.listeners.append(fn)


def reference_segment(pack: FakePack | Any, ring: RingParams, seed: bytes, slots: int, checks: int
                      ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Independent re-derivation: ``{key: (W r (slots, m), u W (checks, n))}`` with Python-int math."""
    exp = SeedExpander(seed)
    out = {}
    for s in pack.linear_specs():
        key = s.id.key
        w = pack.load_linear(s.id).w8
        r = exp.ring_uniform((slots, s.in_features), ring.k, "r", key)
        wr = torch.from_numpy(ringmm_reference(r, w, ring.k))
        u = exp.ring_uniform((checks, s.out_features), ring.k, "u", key)
        uw = torch.from_numpy(ringmm_reference(u, w.t().contiguous(), ring.k)) if checks else torch.zeros((0, s.in_features), dtype=torch.int64)
        out[key] = (wr, uw)
    return out


def write_fake_segment_file(path: Path, header: dict[str, Any], payload: bytes = b"\x00" * 64) -> Path:
    """A file with the public segment framing (magic, u32 hlen, header JSON) and dummy chunks."""
    body = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PGWMASK1" + struct.pack("<I", len(body)) + body + payload)
    return path


@pytest.fixture
def fake_pack() -> FakePack:
    return FakePack()


@pytest.fixture
def fake_store(tmp_path: Path) -> FakeMaskStore:
    return FakeMaskStore(root=tmp_path / "store")
