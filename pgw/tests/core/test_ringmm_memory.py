"""Working memory of the exact ring matmul's offset-byte path: measured on CPU, modelled for the GPU.

The CUDA path runs the same torch code as these CPU calls (``torch._int_mm`` and element-wise ops), so the bytes
an operator allocates are the same on both devices for the same shapes. :class:`tests.core.alloc_tracker.AllocPeak`
measures them here; :func:`pgw.core.ringmm.offset_byte_work_bytes` is the model the row chunking and the
provider memory plan (:mod:`pgw.service.memplan`) use.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.core.ringmm import cpu_int8_available, ringmm_reference
from tests.core.alloc_tracker import AllocPeak

pytestmark = pytest.mark.skipif(not cpu_int8_available(), reason="CPU torch._int_mm unavailable or inexact")


def _legacy_offset_byte_core(x, k, w, rowsum, p):  # type: ignore[no-untyped-def]
    """The combine before 2026-11-26 (three stacked ``(K/8, b, mp)`` int64 temporaries), for comparison."""
    nb = k // 8
    b, n = x.shape
    mp, np_ = w.shape
    rows = nb * b
    sh = torch.arange(0, 8 * nb, 8, dtype=torch.int64).view(nb, 1, 1)
    pm = torch.tensor([(1 << (k - 8 * i)) - 1 for i in range(nb)], dtype=torch.int64).view(nb, 1, 1)
    planes = (x.contiguous().view(torch.uint8).view(b, n, 8)[:, :, :nb] ^ 0x80).view(torch.int8)
    d8 = torch.zeros((p, np_), dtype=torch.int8)
    d8[:rows, :n].view(nb, b, n).copy_(planes.permute(2, 0, 1))
    y = torch._int_mm(d8, w.t())[:rows].to(torch.int64).view(nb, b, mp)
    acc = ((y & pm) << sh).sum(0)
    c = (128 * sum(1 << (8 * i) for i in range(nb))) % (1 << k)
    return (acc + c * rowsum) & ((1 << k) - 1)


@pytest.mark.parametrize("k", [8, 32, 40, 56])
@pytest.mark.parametrize("b,m,n", [(1, 64, 96), (9, 160, 64), (40, 96, 288), (130, 32, 32)])
def test_offset_byte_core_memory_is_modelled_and_lean(k: int, b: int, m: int, n: int) -> None:
    """Measured peak <= the model (and not far below it); bit-identical to the old combine and to the reference;
    the old combine's peak is several times larger for multi-plane rings."""
    from pgw.core.ringmm import offset_byte_consts, offset_byte_core, offset_byte_rows, offset_byte_work_bytes

    g = torch.Generator().manual_seed(k + b + m)
    w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
    x = torch.randint(0, 1 << min(k, 62), (b, n), generator=g) & ((1 << k) - 1)
    rs = w.to(torch.int64).sum(1)
    sh, pm = offset_byte_consts(k, "cpu")
    p = offset_byte_rows(k, b)
    with AllocPeak(exclude=[x, w, rs, sh, pm]) as t:
        got = offset_byte_core(x, k, w, rs, sh, pm, p)
    model = offset_byte_work_bytes(k, b, n, n, m, p)
    assert t.peak <= model, (t.peak, model)
    assert t.peak >= 0.8 * model, (t.peak, model)
    with AllocPeak(exclude=[x, w, rs]) as t_old:
        want = _legacy_offset_byte_core(x, k, w, rs, p)
    assert torch.equal(got, want)
    assert np.array_equal(got.numpy(), ringmm_reference(x, w, k))
    if k // 8 >= 4 and b >= 9:
        assert t_old.peak > 2 * t.peak, (t_old.peak, t.peak)


def test_offset_byte_mm_chunks_rows_within_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call over the eager budget runs in row chunks: bit-identical, peak bounded by budget + result."""
    import importlib

    rm = importlib.import_module("pgw.core.ringmm")

    budget = 256 << 10
    monkeypatch.setattr(rm, "eager_work_budget", lambda: budget)
    g = torch.Generator().manual_seed(5)
    m, n, b, k = 512, 96, 300, 32
    w = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
    x = torch.randint(0, 1 << 32, (b, n), generator=g)
    rs = w.to(torch.int64).sum(1)
    assert rm.offset_byte_work_bytes(k, b, n, n, m) > 8 * budget
    with AllocPeak(exclude=[x, w, rs]) as t:
        got = rm._offset_byte_mm(x, k, w, rs, m)
    assert np.array_equal(got.numpy(), ringmm_reference(x, w, k))
    assert t.peak <= budget + 8 * b * m + 4096, t.peak  # the (b, m) result plus one chunk


def test_chunk_rows_for() -> None:
    from pgw.core.ringmm import chunk_rows_for

    assert chunk_rows_for(100, lambda r: 10 * r, 5) == 5
    assert chunk_rows_for(1000, lambda r: 10 * r, 500) == 96  # 100 rows fit, rounded down to 8
    assert chunk_rows_for(35, lambda r: 10 * r, 500) == 3
    assert chunk_rows_for(1, lambda r: 10 * r, 500) == 1  # one row is the floor
    assert chunk_rows_for(1, lambda r: 10 * r, 0) == 1


def test_eager_work_budget_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.core.errors import ConfigError
    from pgw.core.ringmm import EAGER_WORK_BYTES, eager_work_budget

    monkeypatch.delenv("PGW_SERVICE_EAGER_WORK_MB", raising=False)
    assert eager_work_budget() == EAGER_WORK_BYTES
    monkeypatch.setenv("PGW_SERVICE_EAGER_WORK_MB", "64")
    assert eager_work_budget() == 64 << 20
    for bad in ("x", "8"):
        monkeypatch.setenv("PGW_SERVICE_EAGER_WORK_MB", bad)
        with pytest.raises(ConfigError):
            eager_work_budget()


def test_weight_rowsum_never_copies_the_weight_to_int64() -> None:
    """``weight_rowsum`` allocates one bounded int64 chunk; ``sum(1, dtype=torch.int64)`` (the maskgen constant
    before 2026-11-26) allocates an int64 copy of the whole weight: 8x its size."""
    from torch.profiler import ProfilerActivity, profile

    from pgw.core.ringmm import weight_rowsum

    def largest_alloc(fn):  # type: ignore[no-untyped-def]
        # the cast inside ``sum(dtype=)`` happens below the Python dispatcher: the profiler sees it
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            out = fn()
        return out, max((e.cpu_memory_usage for e in prof.events() if e.cpu_memory_usage > 0), default=0)

    w = torch.randint(-128, 128, (2048, 512), dtype=torch.int8)
    want, naive = largest_alloc(lambda: w.sum(1, dtype=torch.int64))
    got, chunked = largest_alloc(lambda: weight_rowsum(w, chunk_bytes=1 << 20))
    assert torch.equal(got, want)
    assert naive >= 8 * w.numel()
    assert chunked <= 1 << 20, chunked


def test_pad_weight_cuda_on_cpu_target_copies() -> None:
    from pgw.core.ringmm import pad_weight_cuda, padded_shape

    w = torch.randint(-128, 128, (64, 96), dtype=torch.int8)
    p = pad_weight_cuda(w, "cpu")
    assert p.data_ptr() != w.data_ptr() and torch.equal(p, w)
    w2 = torch.randint(-128, 128, (37, 70), dtype=torch.int8)
    p2 = pad_weight_cuda(w2, "cpu")
    assert tuple(p2.shape) == padded_shape(37, 70) == (64, 96)
    assert torch.equal(p2[:37, :70], w2) and int(p2[37:].abs().sum()) == 0 and int(p2[:, 70:].abs().sum()) == 0


def test_executor_eager_chunks_prefill_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service executor's eager path (prefill-sized calls) splits rows to its budget: bit-identical
    compact results, measured peak bounded by the budget plus the host result, modelled by ``eager_work_bytes``."""
    import pgw.service.gpu_exec as ge
    from pgw.core.ringmm import pad_weight_cuda
    from pgw.service.gpu_exec import DeviceExecutor, eager_capturer, eager_work_bytes, from_compact

    g = torch.Generator().manual_seed(9)
    m, n = 300, 160
    w8 = torch.randint(-128, 128, (m, n), dtype=torch.int8, generator=g)
    w = pad_weight_cuda(w8, "cpu")
    rs = w.to(torch.int64).sum(1)
    for k in (32, 40):
        x = torch.randint(0, 1 << k, (257, n), generator=g)
        ex = DeviceExecutor("cpu", capturer=eager_capturer)
        full = ex.run(x, k, w, rs, m, graphs=None)
        assert ex.stats.chunked == 0
        budget = 128 << 10
        monkeypatch.setattr(ge, "eager_work_budget", lambda b=budget: b)
        with AllocPeak(exclude=[x, w, rs]) as t:
            got = ex.run(x, k, w, rs, m, graphs=None)
        monkeypatch.undo()
        assert ex.stats.chunked == 1
        assert torch.equal(got, full)
        assert np.array_equal(from_compact(got, k).numpy(), ringmm_reference(x, w8, k))
        out_bytes = got.numel() * got.element_size()
        assert t.peak <= budget + out_bytes + 8192, (k, t.peak)
        # the model of one unchunked call is an upper bound of its measured peak
        x1 = x[:40]
        with AllocPeak(exclude=[x, w, rs]) as t1:
            ex.run(x1, k, w, rs, m, graphs=None)
        assert t1.peak <= eager_work_bytes(k, 40, n, int(w.shape[1]), int(w.shape[0]), m) + 40 * m * 8, (k, t1.peak)
