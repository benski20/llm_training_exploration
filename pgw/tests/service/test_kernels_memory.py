"""Provider memory fixes in :mod:`pgw.service.kernels` that run on CPU: a large maskgen result copied back per
column block (bit-identical to the whole-result path), the maskgen constant from chunked row sums."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.core.ringmm import cpu_int8_available, pad_weight_cuda
from pgw.core.types import LinearId, LinearKind, LinearSpec
from pgw.service import kernels as K

needs_int8 = pytest.mark.skipif(not cpu_int8_available(), reason="CPU torch._int_mm unavailable or inexact")


@needs_int8
@pytest.mark.parametrize("ylo,yhi", [(0, 8), (2, 7)])
def test_maskgen_blocks_to_host_equals_whole(ylo: int, yhi: int) -> None:
    rng = np.random.default_rng(3)
    m, n, cols, nb = 70, 45, 100, 7
    w8 = torch.from_numpy(rng.integers(-128, 128, (m, n), dtype=np.int8))
    wpad = pad_weight_cuda(w8, "cpu")
    raw = torch.from_numpy(rng.integers(0, 256, (n, cols, nb), dtype=np.uint8))
    const = torch.from_numpy(rng.integers(-(2**62), 2**62, (m, 1), dtype=np.int64))
    whole = K._maskgen_block_tn(wpad, raw, m, n, const, ylo, yhi).contiguous().numpy()
    for step in (32, 64, 96, 128):
        got = K._maskgen_blocks_to_host(wpad, raw, m, n, const, ylo, yhi, step)
        assert got.shape == (m, cols, yhi - ylo) and np.array_equal(got, whole), step


@needs_int8
def test_maskgen_const_uses_chunked_rowsum(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CPU maskgen constant equals 128 * rowsum * sum 2^(8d) + rnd mod 2^64 and comes from weight_rowsum."""
    calls: list[int] = []
    real = K.weight_rowsum

    def spy(w: torch.Tensor, *a, **kw):  # type: ignore[no-untyped-def]
        calls.append(int(w.shape[0]))
        return real(w, *a, **kw)

    monkeypatch.setattr(K, "weight_rowsum", spy)
    w = torch.randint(-128, 128, (40, 33), dtype=torch.int8)
    k = K.ComputeKernel(LinearSpec(LinearId(0, LinearKind.O), 40, 33), w, device="cpu")
    c = k._maskgen_const(7, 5).numpy().view(np.uint64).reshape(-1)
    scale = (128 * sum(1 << (8 * d) for d in range(7))) % (1 << 64)
    want = [(int(r) * scale + 5) % (1 << 64) for r in w.to(torch.int64).sum(1)]
    assert [int(v) for v in c] == want and calls == [40]
