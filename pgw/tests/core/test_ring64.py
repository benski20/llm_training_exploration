"""Exactness of the fused2 provider op kernels (:mod:`pgw.core.ring64`) against the Python-int reference.

Runs in the Triton interpreter on hosts without CUDA (``tests/conftest.py``), on the device otherwise: both
variants (uint64 CUDA-core ``sum``, digit-plane tensor-core ``dot``), split-K partials, every ring width the op
serves, random and extreme operands, pair (``H = R + hmul * W1 x``, ``Z = W2' H``) and single items.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from pgw.core.packing import pack_ring  # noqa: E402
from pgw.core.ring64 import PairConfig, PairOp, check_op, gemv64, pair_reference, self_check  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _w(gen: torch.Generator, m: int, n: int) -> torch.Tensor:
    w = torch.randint(-128, 128, (m, n), generator=gen, dtype=torch.int64).to(torch.int8)
    w[0] = -128
    w[-1] = 127
    return w


@pytest.mark.parametrize("variant", ["sum", "dot"])
@pytest.mark.parametrize("k", [56, 64, 32, 48])
def test_pair_op_exact(variant: str, k: int) -> None:
    gen = torch.Generator().manual_seed(k)
    w1, w2 = _w(gen, 24, 40), _w(gen, 33, 24)
    hm = torch.randint(1, 1 << 12, (24,), generator=gen, dtype=torch.int64)
    op = PairOp(k, w1, w2, hm, device=DEV, max_rows=5, variant=variant)
    op.set_config(2, PairConfig(16, 16, 3, 4, variant), PairConfig(16, 16, 2, 4, variant))  # several partials
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
