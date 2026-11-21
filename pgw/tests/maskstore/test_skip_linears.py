"""Segments without some linears (factory ``skip_linear_kinds``) and ``MaskStore.require_linears``.

A gateway that computes qkv and o locally on every step never takes their masks, so a factory may skip
them. These tests pin the safety rules: a reservation never lacks a linear its caller declared, segments
lacking one are retired (written off, never handed out), reservations may span full and partial segments,
and the masks of the generated linears are the same bits as in a full segment.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, SeedExpander
from pgw.core.errors import ConfigError, MaskExhaustedError
from pgw.maskstore import MaskStore

from .conftest import ToyModel, write_segment

RING = RING_K40_A12


def _partial(model: ToyModel, skip: set[str]) -> ToyModel:
    return replace(model, specs=[s for s in model.specs if s.id.kind.value not in skip])


def _ids(model: ToyModel, skip: set[str] = frozenset()) -> list[str]:  # type: ignore[assignment]
    return [s.id.key for s in model.specs if s.id.kind.value not in skip]


def _expected_r(seed: bytes, model: ToyModel, key: str, start: int, rows: int) -> torch.Tensor:
    n = next(s.in_features for s in model.specs if s.id.key == key)
    return SeedExpander(seed).ring_uniform_rows(n, RING.k, start, rows, "r", key)


def test_partial_segment_masks_equal_full(store: MaskStore, model: ToyModel) -> None:
    seed = os.urandom(32)
    write_segment(store, _partial(model, {"qkv", "o"}), 8, seed=seed)
    store.require_linears(model.fingerprint, RING, _ids(model, {"qkv", "o"}))
    res = store.reserve(model.fingerprint, RING, 8)
    try:
        assert set(res.linear_keys) == set(_ids(model, {"qkv", "o"}))
        r, wr = res.take("0.gate_up", 8)
        assert torch.equal(r, _expected_r(seed, model, "0.gate_up", 0, 8))
        assert torch.equal(wr, model.wr("0.gate_up", r, RING))
        with pytest.raises(ConfigError, match="not covered"):
            res.take("0.qkv", 1)
    finally:
        res.close()


def test_full_requirement_retires_partial_segments(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, _partial(model, {"qkv", "o"}), 8)
    full_id = write_segment(store, model, 8)
    assert store.available_slots(model.fingerprint, RING) == 16
    assert store.require_linears(model.fingerprint, RING, _ids(model)) == 1
    assert store.available_slots(model.fingerprint, RING) == 8  # the partial segment is written off
    res = store.reserve(model.fingerprint, RING, 8)
    try:
        assert [e.segment_id for e in res.extents] == [full_id]
        res.take("1.qkv", 8)
    finally:
        res.close()
    with pytest.raises(MaskExhaustedError):
        store.reserve(model.fingerprint, RING, 1)  # never falls back to the retired segment
    assert store.verify_integrity().consumed_total == 16  # consumption accounting stays consistent


def test_partial_segment_arriving_later_is_retired_at_reserve(store: MaskStore, model: ToyModel) -> None:
    store.require_linears(model.fingerprint, RING, _ids(model))
    write_segment(store, _partial(model, {"qkv", "o"}), 8)  # a misconfigured factory
    with pytest.raises(MaskExhaustedError):
        store.reserve(model.fingerprint, RING, 4)
    assert store.available_slots(model.fingerprint, RING) == 0
    write_segment(store, model, 8)
    res = store.reserve(model.fingerprint, RING, 4)
    res.close()


def test_reservation_spans_full_and_partial_segments(store: MaskStore, model: ToyModel) -> None:
    s1, s2 = os.urandom(32), os.urandom(32)
    write_segment(store, model, 5, seed=s1)
    write_segment(store, _partial(model, {"qkv", "o"}), 6, seed=s2)
    assert store.require_linears(model.fingerprint, RING, _ids(model, {"qkv", "o"})) == 0
    res = store.reserve(model.fingerprint, RING, 9)
    try:
        assert set(res.linear_keys) == set(_ids(model, {"qkv", "o"}))
        r1, _ = res.take("1.down", 5)
        r2, wr2 = res.take("1.down", 4)
        assert torch.equal(r1, _expected_r(s1, model, "1.down", 0, 5))
        assert torch.equal(r2, _expected_r(s2, model, "1.down", 0, 4))
        assert torch.equal(wr2, model.wr("1.down", r2, RING))
        _, uw = res.checks("1.down")
        assert uw.shape[1] == 16
        with pytest.raises(ConfigError):
            res.take("0.o", 1)
    finally:
        res.close()


def test_without_requirement_partial_reservation_fails_loudly(store: MaskStore, model: ToyModel) -> None:
    """Legacy callers (no require_linears): a missing linear is an error, never a short or reused mask."""
    write_segment(store, _partial(model, {"qkv", "o"}), 4)
    res = store.reserve(model.fingerprint, RING, 4)
    try:
        with pytest.raises(ConfigError, match="not covered"):
            res.take("0.qkv", 1)
    finally:
        res.close()


def test_requirement_survives_reopen_only_when_declared_again(open_store: Callable[..., MaskStore],
                                                               model: ToyModel) -> None:
    """The requirement is per process (declared by the gateway at every start-up from its configuration)."""
    s = open_store()
    write_segment(s, _partial(model, {"qkv", "o"}), 4)
    s.require_linears(model.fingerprint, RING, _ids(model, {"qkv", "o"}))
    assert s.available_slots(model.fingerprint, RING) == 4
    s.close()
    s2 = open_store()
    # restarted with local linears off: the partial segment is retired, not used
    assert s2.require_linears(model.fingerprint, RING, _ids(model)) == 1
    assert s2.available_slots(model.fingerprint, RING) == 0


def test_generator_skip_kinds(tiny_pack: Any) -> None:
    from pgw.factory import MaskGenerator

    seed = bytes(range(32))
    full = {(g.id.key, g.slot_start): g.wr.clone()
            for g in MaskGenerator(tiny_pack, "K32A12C27", device="cpu").generate(seed, 40, chunk_slots=40)}
    gen = MaskGenerator(tiny_pack, "K32A12C27", device="cpu", skip_kinds={"qkv", "o"})
    part = {(g.id.key, g.slot_start): g.wr.clone() for g in gen.generate(seed, 40, chunk_slots=40)}
    assert part and all(k[0].split(".")[1] in ("gate_up", "down") for k in part)
    assert all(torch.equal(part[k], full[k]) for k in part)
    assert gen.bytes_per_slot < MaskGenerator(tiny_pack, "K32A12C27", device="cpu").bytes_per_slot
    with pytest.raises(ConfigError):
        MaskGenerator(tiny_pack, "K32A12C27", device="cpu", skip_kinds={"qkv", "o", "gate_up", "down"})
    with pytest.raises(ConfigError):
        MaskGenerator(tiny_pack, "K32A12C27", device="cpu", skip_kinds={"lm_head"})


def test_factory_setting_validation(tmp_path: Any) -> None:
    from pgw.factory.config import FactorySettings

    s = FactorySettings(pack_path=tmp_path, skip_linear_kinds=" o, qkv ,o")
    assert s.skip_linear_kinds == "qkv,o" and s.skip_kinds == frozenset({"qkv", "o"})
    assert FactorySettings(pack_path=tmp_path).skip_kinds == frozenset()
    with pytest.raises(ValueError):
        FactorySettings(pack_path=tmp_path, skip_linear_kinds="qkv,attn")
    with pytest.raises(ValueError):
        FactorySettings(pack_path=tmp_path, skip_linear_kinds="qkv,o,gate_up,down")
