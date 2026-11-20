"""The deterministic float32 LM head: native kernel == torch fallback, bit for bit; rows independent.

``pgw.engine.lm_head.lm_head_reference`` (torch, any device) and ``pgw_fast.lm_head`` (Rust, SIMD,
multi-threaded) implement the same fixed summation order (16 partial sums over the columns, then a
fixed tree; no FMA). The private gateway path and the plaintext reference both use it, so their
logits agree bit for bit however rows are grouped into passes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.core import native
from pgw.engine.lm_head import LANES, lm_head_logits, lm_head_reference

needs_native = pytest.mark.skipif(not native.native_available("lm_head"),
                                  reason="pgw_fast.lm_head not built (rust/tools/build_fast.py)")

SHAPES = [(1, 1), (7, 3), (15, 20), (16, 16), (17, 33), (31, 64), (64, 130), (100, 257), (515, 70), (896, 300)]


def _data(d: int, v: int, seed: int, rows: int = 9) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(v, d, generator=g) * 0.05
    h = torch.randn(rows, d, generator=g) * 3.0
    # mixed magnitudes and signed zeros make rounding order visible
    h[:, ::5] *= 1e3
    w[::7, ::3] *= 1e-3
    h[0, 0] = -0.0
    return h, w


def _scalar(h: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The documented order, one value at a time (numpy float32 scalars)."""
    out = np.zeros((h.shape[0], w.shape[0]), dtype=np.float32)
    for i in range(h.shape[0]):
        for j in range(w.shape[0]):
            acc = [np.float32(0.0)] * LANES
            for k in range(h.shape[1]):
                acc[k % LANES] = np.float32(acc[k % LANES] + np.float32(h[i, k] * w[j, k]))
            width = LANES
            while width > 1:
                width //= 2
                acc = [np.float32(acc[lane] + acc[lane + width]) for lane in range(width)]
            out[i, j] = acc[0]
    return out


def test_fallback_matches_scalar_order() -> None:
    for d, v in [(1, 2), (17, 5), (40, 3)]:
        h, w = _data(d, v, d)
        assert np.array_equal(lm_head_reference(h[:3], w).numpy().view(np.uint32),
                              _scalar(h[:3].numpy(), w.numpy()).view(np.uint32))


def test_fallback_rows_independent_of_batch_and_block() -> None:
    h, w = _data(100, 257, 1)
    full = lm_head_reference(h, w)
    for r in range(1, 10):
        assert torch.equal(lm_head_reference(h[:r], w, block=64), full[:r])
        assert torch.equal(lm_head_reference(h[r - 1 : r], w, block=1000), full[r - 1 : r])
    assert torch.allclose(full, h @ w.t(), rtol=1e-5, atol=1e-2)


@needs_native
@pytest.mark.parametrize(("d", "v"), SHAPES)
def test_native_equals_fallback(d: int, v: int) -> None:
    h, w = _data(d, v, d * 1000 + v)
    want = lm_head_reference(h, w)
    for r in range(1, 10):
        got = torch.from_numpy(native.fast.lm_head(h[:r].numpy(), w.numpy()))
        assert torch.equal(got, want[:r]), (d, v, r)
    assert torch.equal(lm_head_logits(h, w), want)


@needs_native
def test_native_rows_independent_of_rows_and_threads() -> None:
    h, w = _data(896, 4099, 5, rows=12)
    hn, wn = h.numpy(), w.numpy()
    base = native.fast.lm_head(hn, wn, threads=1)
    for threads in (0, 2, 3, 4, 8):
        assert np.array_equal(native.fast.lm_head(hn, wn, threads=threads), base)
    for i in range(12):
        assert np.array_equal(native.fast.lm_head(np.ascontiguousarray(hn[i : i + 1]), wn)[0], base[i])
    assert torch.equal(lm_head_reference(h, w), torch.from_numpy(base))


@needs_native
def test_native_rejects_bad_inputs() -> None:
    h = np.zeros((2, 8), dtype=np.float32)
    w = np.zeros((3, 8), dtype=np.float32)
    assert native.fast.lm_head(h, w).shape == (2, 3)
    with pytest.raises(ValueError, match="columns"):
        native.fast.lm_head(h, np.zeros((3, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        native.fast.lm_head(h.astype(np.float64), w)
    with pytest.raises(ValueError, match="contiguous"):
        native.fast.lm_head(np.zeros((8, 2), dtype=np.float32).T, w)
    assert native.fast.lm_head(np.zeros((0, 8), dtype=np.float32), w).shape == (0, 3)
    assert native.fast.LM_HEAD_LANES == LANES


def test_logits_entry_point_handles_views() -> None:
    """Non-contiguous inputs (e.g. a transposed head) give the same bits."""
    h, w = _data(33, 20, 9)
    want = lm_head_logits(h, w)
    wt = w.t().contiguous().t()
    assert not wt.is_contiguous()
    assert torch.equal(lm_head_logits(h, wt), want)
    assert torch.equal(lm_head_logits(h[:, :].t().contiguous().t(), w), want)


# ---- bfloat16 heads (lossless compaction) ------------------------------------------------------------
def _bf16_valued(w: torch.Tensor) -> torch.Tensor:
    """float32 values that are exactly bfloat16 (as in a head loaded from a bfloat16 checkpoint)."""
    return w.to(torch.bfloat16).to(torch.float32)


def test_compact_head_is_lossless_or_a_no_op() -> None:
    from pgw.engine.lm_head import compact_head

    _, w = _data(64, 50, seed=3)
    assert compact_head(w) is w  # float32 values with low bits set: kept
    wb = _bf16_valued(w)
    c = compact_head(wb, chunk_rows=7)
    assert c.dtype == torch.bfloat16 and torch.equal(c.to(torch.float32), wb)
    wb[49, 3] = 1.0 + 2.0**-20  # one value needing more than bfloat16 precision, in the last chunk
    assert compact_head(wb, chunk_rows=7) is wb
    assert compact_head(torch.zeros(3, 4, dtype=torch.float64)).dtype == torch.float64


@pytest.mark.parametrize(("d", "v"), [(1, 1), (17, 33), (100, 257), (896, 300)])
def test_bf16_head_gives_the_float32_bits(d: int, v: int) -> None:
    h, w = _data(d, v, seed=d + v)
    wf = _bf16_valued(w)
    wb = wf.to(torch.bfloat16)
    want = lm_head_reference(h, wf)
    assert torch.equal(lm_head_reference(h, wb), want)  # fallback widens exactly
    assert torch.equal(lm_head_logits(h, wb), want)  # native bfloat16 kernel (or the fallback)
    for r in (1, 4, 9):
        assert torch.equal(lm_head_logits(h[:r], wb), want[:r])


@pytest.mark.skipif(not native.native_available("lm_head_bf16"), reason="pgw_fast.lm_head_bf16 not built")
def test_native_bf16_threads_and_bad_inputs() -> None:
    h, w = _data(515, 70, seed=11)
    wb = _bf16_valued(w).to(torch.bfloat16)
    bits = wb.view(torch.int16).numpy().view("uint16")
    want = lm_head_reference(h, wb).numpy()
    for t in (0, 1, 3, 8):
        assert np.array_equal(native.fast.lm_head_bf16(h.numpy(), bits, threads=t), want)
    with pytest.raises(ValueError):
        native.fast.lm_head_bf16(h.numpy(), bits[:, :10])
    with pytest.raises(ValueError):
        native.fast.lm_head_bf16(h.numpy(), np.asfortranarray(bits))


def test_model_bf16_head_same_logits_as_float32_head() -> None:
    """The model's bfloat16 head (compacted at load) gives the float32 head's logits, bit for bit."""
    from pgw.engine.lm_head import compact_head

    g = torch.Generator().manual_seed(5)
    hidden = torch.randn(6, 896, generator=g)
    w = _bf16_valued(torch.randn(1000, 896, generator=g) * 0.02)
    c = compact_head(w)
    assert c.dtype == torch.bfloat16 and c.untyped_storage().nbytes() == w.untyped_storage().nbytes() // 2
    assert torch.equal(lm_head_logits(hidden, c), lm_head_logits(hidden, w))


# ---- instruction sets and register blocking (phase 2, package D) ----------------------------------
needs_isa = pytest.mark.skipif(not native.native_available("lm_head_isa"),
                               reason="pgw_fast.lm_head_isa not built (rust/tools/build_fast.py)")
ISAS = ("auto", "avx512", "avx2", "avx", "sse", "scalar")


@needs_isa
@pytest.mark.parametrize(("d", "v"), [(1, 1), (15, 20), (17, 33), (100, 257), (515, 70), (3584, 67)])
def test_every_isa_and_row_count_gives_the_reference_bits(d: int, v: int) -> None:
    """Every instruction set and every row count (register blocks of 1/2/4 rows plus remainders,
    weight blocks of 1/2/4/8 rows plus remainders) gives the fallback's bits, f32 and bf16 heads."""
    h, w = _data(d, v, seed=7 * d + v, rows=11)
    wf = _bf16_valued(w)
    bits = wf.to(torch.bfloat16).view(torch.int16).numpy().view("uint16")
    want = lm_head_reference(h, wf).numpy()
    want32 = lm_head_reference(h, w).numpy()
    for isa in ISAS:
        for r in range(1, 12):
            hn = np.ascontiguousarray(h[:r].numpy())
            for t in (1, 3):
                got = native.fast.lm_head_bf16(hn, bits, threads=t, isa=isa)
                assert np.array_equal(got.view(np.uint32), want[:r].view(np.uint32)), (isa, r, t)
            got32 = native.fast.lm_head(hn, w.numpy(), isa=isa)
            assert np.array_equal(got32.view(np.uint32), want32[:r].view(np.uint32)), (isa, r)


@needs_isa
def test_isa_resolution_and_errors() -> None:
    best = native.fast.lm_head_isa()
    assert best in ISAS[1:]
    assert native.fast.lm_head_isa("auto") == best
    assert native.fast.lm_head_isa("scalar") == "scalar"
    order = ["avx512", "avx2", "avx", "sse"]
    if best != "scalar":  # x86_64: a request never resolves to something wider than itself
        for isa in order:
            assert order.index(native.fast.lm_head_isa(isa)) >= order.index(isa)
    with pytest.raises(ValueError, match="isa"):
        native.fast.lm_head_isa("avx1024")
    with pytest.raises(ValueError, match="isa"):
        native.fast.lm_head(np.zeros((1, 4), np.float32), np.zeros((2, 4), np.float32), isa="neon")


@needs_isa
def test_logits_entry_point_honours_the_isa_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    import pgw.engine.lm_head as mod

    h, w = _data(100, 257, seed=4)
    wb = _bf16_valued(w).to(torch.bfloat16)
    want = lm_head_reference(h, wb)
    for isa in ISAS:
        monkeypatch.setattr(mod, "ISA", isa)
        assert torch.equal(mod.lm_head_logits(h, wb), want), isa
        assert torch.equal(mod.lm_head_logits(h, _bf16_valued(w)), want), isa
        assert mod.lm_head_isa() == native.fast.lm_head_isa(isa)
    monkeypatch.setenv("PGW_LM_HEAD_ISA", "bogus")
    assert mod._isa_setting() == "auto"
    monkeypatch.setenv("PGW_LM_HEAD_ISA", " AVX2 ")
    assert mod._isa_setting() == "avx2"


@needs_isa
def test_wide_rows_mixed_magnitudes_7b_width() -> None:
    """7B hidden size, 16 rows (four 4-row blocks), extreme magnitudes and signed zeros."""
    g = torch.Generator().manual_seed(77)
    h = torch.randn(16, 3584, generator=g) * 4.0
    h[:, ::11] *= 1e4
    h[3] = -0.0
    w = torch.randn(203, 3584, generator=g) * 0.02
    w[::5, ::7] *= 1e-6
    w = _bf16_valued(w)
    want = lm_head_reference(h, w)
    bits = w.to(torch.bfloat16).view(torch.int16).numpy().view("uint16")
    for isa in ISAS:
        got = native.fast.lm_head_bf16(h.numpy(), bits, isa=isa)
        assert torch.equal(torch.from_numpy(got), want), isa
