"""Generator output self-check: a faulty ring GEMM must never produce a committed segment."""

from __future__ import annotations

import pytest
import torch

from pgw.core import RING_K40_A12, RING_K48_A20, ConfigError, IntegrityError
from pgw.factory import MaskGenerator
from pgw.factory import kernels as fk

from .conftest import FakeMaskStore, FakePack, reference_segment

SEED = bytes(range(32))


def _corrupt_matmul(monkeypatch: pytest.MonkeyPatch, *, linear_index: int = 0, flip_bit: int = 0) -> None:
    """Make RingKernel.matmul wrong in one bit of every row of one linear (like a saturating kernel)."""
    orig = fk.RingKernel.matmul
    calls = {"n": 0}

    def bad(self: fk.RingKernel, r: torch.Tensor, params: object) -> torch.Tensor:
        out = orig(self, r, params)  # type: ignore[arg-type]
        if calls["n"] == linear_index:
            out = out.clone()
            out[:, 0] ^= 1 << flip_bit
        calls["n"] += 1
        return out

    monkeypatch.setattr(fk.RingKernel, "matmul", bad)


@pytest.mark.parametrize("checks", [0, 2])
@pytest.mark.parametrize("pipeline", [True, False])
def test_faulty_kernel_aborts_segment(monkeypatch: pytest.MonkeyPatch, fake_pack: FakePack,
                                      fake_store: FakeMaskStore, checks: int, pipeline: bool) -> None:
    _corrupt_matmul(monkeypatch, flip_bit=0)  # odd error: every check vector sees it
    gen = MaskGenerator(fake_pack, RING_K40_A12, pipeline=pipeline, self_check_rows=1)
    with pytest.raises(IntegrityError) as ei:
        gen.generate_into_store(fake_store, 6, checks=checks, chunk_slots=3)
    assert ei.value.details["linear"] == fake_pack.linear_specs()[0].id.key
    assert fake_store.aborted == 1 and not fake_store.segments


def test_self_check_off_lets_fault_through(monkeypatch: pytest.MonkeyPatch, fake_pack: FakePack,
                                           fake_store: FakeMaskStore) -> None:
    _corrupt_matmul(monkeypatch)
    gen = MaskGenerator(fake_pack, RING_K40_A12, self_check_rows=0)
    gen.generate_into_store(fake_store, 4, chunk_slots=4)
    assert len(fake_store.segments) == 1  # documents why the check is on by default


@pytest.mark.parametrize("rows", [-1, 3])
def test_self_check_passes_on_exact_output(fake_pack: FakePack, rows: int) -> None:
    gen = MaskGenerator(fake_pack, RING_K48_A20, self_check_rows=rows)
    items = list(gen.generate(SEED, 7, checks=1, chunk_slots=4))
    ref = reference_segment(fake_pack, RING_K48_A20, SEED, 7, 1)
    for spec in fake_pack.linear_specs():
        got = torch.cat([i.wr for i in items if i.id == spec.id])
        assert got.tolist() == ref[spec.id.key][0].tolist()
    per_linear = 7 if rows < 0 else 3 + 3  # chunks of 4 and 3 rows
    assert gen.last_stats.self_checked_rows == per_linear * len(fake_pack.linear_specs())


def test_self_check_rows_validated(fake_pack: FakePack) -> None:
    with pytest.raises(ConfigError):
        MaskGenerator(fake_pack, RING_K40_A12, self_check_rows=-2)


def test_settings_wire_self_check(tmp_path: object) -> None:
    from pgw.factory import FactorySettings
    from pgw.factory.runtime import build_generators

    s = FactorySettings(pack_path=tmp_path, self_check_rows=-1)  # type: ignore[arg-type]
    gens = build_generators(s, [FakePack()])
    assert [g.self_check_rows for g in gens.values()] == [-1]


def test_weak_integrity_margin_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Docs finding 2: K32A8 with few check vectors on wide layers gets a warning (once per checks value)."""
    import logging

    from pgw.core import RING_K32_A8, RING_K48_A20
    from pgw.core.freivalds import soundness_bits
    from pgw.factory import MaskGenerator

    from .conftest import FakePack

    pack = FakePack()
    gen = MaskGenerator(pack, RING_K32_A8)
    worst = min(soundness_bits(RING_K32_A8, s.in_features, 1) for s in gen.specs)
    with caplog.at_level(logging.WARNING, logger="pgw.factory.generator"):
        gen._warn_weak_integrity(1)
        gen._warn_weak_integrity(1)
    msgs = [r.getMessage() for r in caplog.records if "weak integrity" in r.getMessage()]
    assert len(msgs) == (1 if worst < 16 else 0)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pgw.factory.generator"):
        MaskGenerator(pack, RING_K48_A20)._warn_weak_integrity(2)
    assert not [r for r in caplog.records if "weak integrity" in r.getMessage()]
