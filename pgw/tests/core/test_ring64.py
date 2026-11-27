"""Exactness of the fused2 provider op kernels (:mod:`pgw.core.ring64`) against the Python-int reference.

Runs in the Triton interpreter on hosts without CUDA (``tests/conftest.py``), on the device otherwise: both
variants (uint64 CUDA-core ``sum``, digit-plane tensor-core ``dot``), split-K partials, every ring width the op
serves, random and extreme operands, pair (``H = R + hmul * W1 x``, ``Z = W2' H``) and single items.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from pgw.core.packing import pack_ring
from pgw.core.ring64 import (
    PLANES_MAX_SPLIT_N,
    PairConfig,
    PairOp,
    _per_split,
    _rand_ring,
    check_op,
    digit_planes,
    gemm_planes,
    gemv64,
    pair_reference,
    self_check,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _kernels_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests exercise the Triton kernels (in the interpreter on CPU), not the CPU provider's exact path."""
    monkeypatch.setenv("PGW_RING64_CPU", "triton")


def test_cpu_exact_path_equals_the_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.core.ring64 import check_op

    monkeypatch.setenv("PGW_RING64_CPU", "exact")
    gen = torch.Generator().manual_seed(5)
    w1, w2 = _w(gen, 24, 40), _w(gen, 33, 24)
    hm = torch.randint(1, 1 << 12, (24,), generator=gen, dtype=torch.int64)
    for op in (PairOp(56, w1, w2, hm, max_rows=8), PairOp(56, w1, max_rows=8), PairOp(64, w1, w2, None, max_rows=8)):
        for rows in (1, 3, 8):
            for case in range(2):
                assert check_op(op, rows, seed=rows + case, extreme=case == 1) is None


def _w(gen: torch.Generator, m: int, n: int) -> torch.Tensor:
    w = torch.randint(-128, 128, (m, n), generator=gen, dtype=torch.int64).to(torch.int8)
    w[0] = -128
    w[-1] = 127
    return w


@pytest.mark.parametrize("variant", ["sum", "dot", "planes"])
@pytest.mark.parametrize("k", [56, 64, 32, 48])
def test_pair_op_exact(variant: str, k: int) -> None:
    gen = torch.Generator().manual_seed(k)
    w1, w2 = _w(gen, 24, 40), _w(gen, 33, 24)
    hm = torch.randint(1, 1 << 12, (24,), generator=gen, dtype=torch.int64)
    op = PairOp(k, w1, w2, hm, device=DEV, max_rows=5, variant=variant)
    bn = 32 if variant == "planes" else 16  # int8 tl.dot needs K >= 32
    op.set_config(2, PairConfig(16, bn, 3, 4, variant), PairConfig(16, bn, 2, 4, variant))  # several partials
    for rows in (1, 2, 5):
        for extreme in (False, True):
            assert check_op(op, rows, seed=rows, extreme=extreme) is None
    single = PairOp(k, w1, device=DEV, max_rows=3, variant=variant)
    assert check_op(single, 3, seed=9) is None


def test_variants_and_splits_agree_bit_for_bit() -> None:
    gen = torch.Generator().manual_seed(5)
    w = _w(gen, 48, 200).to(DEV)
    x = torch.randint(-(1 << 62), 1 << 62, (3, 200), generator=gen, dtype=torch.int64).to(DEV)
    sums = []
    for cfg, ndig in ((PairConfig(16, 64, 1, 4, "sum"), 8), (PairConfig(8, 32, 5, 4, "sum"), 8),
                      (PairConfig(16, 16, 7, 4, "dot"), 8), (PairConfig(32, 64, 2, 4, "dot"), 8)):
        p = gemv64(x, w, cfg, ndig=ndig)
        sums.append(p.sum(dim=0).cpu())  # torch int64 sums wrap mod 2^64 like the epilogue
    for s in sums[1:]:
        assert torch.equal(s, sums[0])
    want, _ = pair_reference(64, w.cpu(), x.cpu())
    assert torch.equal(sums[0], want)


def test_rejects_bad_shapes() -> None:
    from pgw.core.errors import ShapeError

    w = torch.zeros((4, 8), dtype=torch.int8)
    with pytest.raises(ShapeError):
        PairOp(60, w)
    with pytest.raises(ShapeError):
        PairOp(56, w, torch.zeros((3, 5), dtype=torch.int8))
    op = PairOp(56, w, max_rows=2)
    with pytest.raises(ShapeError):
        op.run(3, b"\0" * op.in_bytes(3))
    with pytest.raises(ShapeError):
        op.run(1, b"\0" * 5)
    assert op.run(1, pack_ring(torch.zeros((1, 8), dtype=torch.int64), 56)) == b"\0" * op.out_bytes(1)


def test_self_check_passes() -> None:
    assert self_check(DEV, rows=(1, 3)) == []


def test_planes_gemv_exact_all_rows_and_splits() -> None:
    """The digit-plane tensor-core GEMV: one int8 dot per tile over 8 balanced digit planes per row, int32 over a
    whole split, recombined in uint64 == the Python-int product mod 2^64, for 1..16 rows, any tile and split."""
    import numpy as np

    gen = torch.Generator().manual_seed(11)
    w = _w(gen, 80, 333).to(DEV)
    rng = np.random.default_rng(1)
    for rows in (1, 2, 3, 5, 8, 16):
        x = _rand_ring(rng, (rows, 333), 64)
        if rows == 1:
            x[:] = -1  # all digits -1 / carries everywhere
        want, _ = pair_reference(64, w.cpu(), x)
        for cfg in (PairConfig(16, 32, 1, 4, "planes"), PairConfig(32, 64, 4, 4, "planes"),
                    PairConfig(64, 256, 2, 4, "planes")):
            p = gemm_planes(digit_planes(x).to(DEV), rows, w, cfg)
            assert torch.equal(p.sum(dim=0).cpu(), want), (rows, cfg)


def test_planes_split_bound_keeps_int32_exact() -> None:
    # 128 * 128 * n_per_split must stay below 2^31 for every split, whatever split the caller asks for
    assert 128 * 128 * PLANES_MAX_SPLIT_N < 1 << 31
    per, splits = _per_split(300_000, PairConfig(64, 256, 1, 4, "planes"))
    assert per <= PLANES_MAX_SPLIT_N and per * splits >= 300_000
    assert _per_split(300_000, PairConfig(64, 256, 1, 4, "sum"))[1] == 1  # the uint64 variants have no such limit


def test_planes_pair_mixed_rows_reuse_buffers() -> None:
    """One op serving 8, then 1, 3 rows (stale digit planes of padded rows must never reach an output)."""
    gen = torch.Generator().manual_seed(3)
    w1, w2 = _w(gen, 40, 72), _w(gen, 90, 40)
    hm = torch.randint(1, 1 << 8, (40,), generator=gen, dtype=torch.int64)
    op = PairOp(56, w1, w2, hm, device=DEV, max_rows=8, variant="planes")
    op.set_config(3, PairConfig(16, 32, 2, 4, "planes"), PairConfig(32, 32, 1, 4, "planes"))
    for rows, extreme in ((8, True), (1, False), (3, False), (8, False), (2, True)):
        assert check_op(op, rows, seed=rows + 100, extreme=extreme) is None
    with pytest.raises(Exception):
        op.set_config(1, PairConfig(16, 32, 1, 4, "planes"), PairConfig(16, 16, 1, 4, "sum"))
