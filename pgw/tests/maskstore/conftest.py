"""Fixtures for mask store tests: random int8 "models", segment builders, stores."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, LinearId, LinearSpec, RingParams, SeedExpander, linear_ids
from pgw.maskstore import MaskStore, StaticKeyProvider

FP = "ab" * 32
FP2 = "cd" * 32


@dataclass
class ToyModel:
    """Random int8 weights for a few linears (tiny shapes, two layers)."""

    fingerprint: str
    specs: list[LinearSpec]
    weights: dict[str, torch.Tensor]  # (m, n) int64 in [-127, 127]

    def wr(self, key: str, r: torch.Tensor, ring: RingParams) -> torch.Tensor:
        return (r @ self.weights[key].t()) & ring.mask

    def uw(self, key: str, u: torch.Tensor, ring: RingParams) -> torch.Tensor:
        return (u @ self.weights[key]) & ring.mask


def make_model(fingerprint: str = FP, layers: int = 2, seed: int = 0) -> ToyModel:
    g = torch.Generator().manual_seed(seed)
    shapes = {"qkv": (24, 16), "o": (16, 16), "gate_up": (32, 16), "down": (16, 16)}
    specs: list[LinearSpec] = []
    weights: dict[str, torch.Tensor] = {}
    for lid in linear_ids(layers):
        m, n = shapes[lid.kind.value]
        specs.append(LinearSpec(lid, m, n))
        weights[lid.key] = torch.randint(-127, 128, (m, n), generator=g, dtype=torch.int64)
    return ToyModel(fingerprint, specs, weights)


def write_segment(
    store: Any,
    model: ToyModel,
    slots: int,
    *,
    ring: RingParams = RING_K40_A12,
    checks: int = 2,
    chunk_slots: int = 4,
    seed: bytes | None = None,
    write_rows: int = 3,
    order: str = "linear",
    destination: str | None = None,
) -> str:
    """Generate a correct segment with the normative derivation and write it into ``store``."""
    seed = seed if seed is not None else os.urandom(32)
    exp = SeedExpander(seed)
    w = store.writer(model.fingerprint, ring, model.specs, slots=slots, seed=seed, checks=checks,
                     chunk_slots=chunk_slots, factory_id="test",
                     **({"destination": destination} if destination is not None else {}))
    uws = {}
    for s in model.specs:
        if checks:
            u = exp.ring_uniform((checks, s.out_features), ring.k, "u", s.id.key)
            uws[s.id.key] = model.uw(s.id.key, u, ring)
        else:
            uws[s.id.key] = torch.zeros((0, s.in_features), dtype=torch.int64)
    jobs: list[tuple[LinearSpec, int, int]] = []
    for s in model.specs:
        for start in range(0, slots, write_rows):
            jobs.append((s, start, min(write_rows, slots - start)))
    if order == "slot":  # interleave linears (writer must accept any linear order)
        jobs.sort(key=lambda j: (j[1], j[0].id.ordinal))
    with w:
        for s, start, n in jobs:
            r = exp.ring_uniform_rows(s.in_features, ring.k, start, n, "r", s.id.key)
            w.write_linear(s.id, model.wr(s.id.key, r, ring), uws[s.id.key])
    return w.segment_id


@pytest.fixture
def kp() -> StaticKeyProvider:
    return StaticKeyProvider(os.urandom(32))


@pytest.fixture
def model() -> ToyModel:
    return make_model()


@pytest.fixture
def open_store(tmp_path: Path, kp: StaticKeyProvider) -> Iterator[Callable[..., MaskStore]]:
    """Factory ``open_store(**overrides)`` for stores at ``tmp_path/store``; all closed at teardown."""
    opened: list[MaskStore] = []

    def _open(root: Path | None = None, provider: Any = None, **overrides: Any) -> MaskStore:
        overrides.setdefault("low_water_slots", 0)
        s = MaskStore.open(root or tmp_path / "store", provider or kp, **overrides)
        opened.append(s)
        return s

    yield _open
    for s in opened:
        with contextlib.suppress(Exception):
            s.close()


@pytest.fixture
def store(open_store: Callable[..., MaskStore]) -> MaskStore:
    return open_store()


def lid(s: str) -> LinearId:
    return LinearId.parse(s)
