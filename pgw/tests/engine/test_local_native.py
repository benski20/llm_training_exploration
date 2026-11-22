"""L2: the native local linear (``pgw_fast.local_linear_into``: quantize + int16 x int8 GEMV + range check +
rescale in one call) is bit-identical to the reference local product, on every ring it accepts, refuses the
same inputs, and the backend falls back to the numpy + torch form with ``PGW_LOCAL_NATIVE=0``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.core import RING_K40_A12, IntegrityError
from pgw.core.errors import RingOverflowError
from pgw.core.ring import DEFAULT_RING, RingParams
from pgw.core.types import LinearId, LinearKind
from pgw.engine import InMemoryPack, LinearCall, LocalComputeClient, MaskedBackend
from pgw.engine import backends as B

pytestmark = pytest.mark.skipif(not B.native_available("local_linear_into", "trip_prepare_into"),
                                reason="needs pgw_fast with local_linear_into")


def _backend(pack: InMemoryPack, ring: RingParams = DEFAULT_RING) -> MaskedBackend:
    return MaskedBackend(pack, None, LocalComputeClient(pack), ring, pack.fingerprint, local_linears="attention")


@pytest.mark.parametrize("ring", [DEFAULT_RING, RING_K40_A12, RingParams(32, 12), RingParams(32, 15, 29)],
                         ids=lambda r: r.name)
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("lid", ["0.qkv", "1.o"])
def test_native_local_equals_reference(qwen_pack: InMemoryPack, ring: RingParams, rows: int, lid: str) -> None:
    be = _backend(qwen_pack, ring)
    assert be._local_native
    lin = LinearId.parse(lid)
    n = qwen_pack.load_linear(lin).w8.shape[1]
    gen = torch.Generator().manual_seed(rows * 31 + len(lid))
    for scale in (1.0, 1e-3, 40.0, 1e-30):
        x = torch.randn(rows, n, generator=gen) * scale
        x[0, 3] = 25.0 * scale
        if rows > 2:
            x[1] = 0.0
        meta, w = be._meta.get(lin), be._local_w.get(lin)
        ref = be._local_reference(x, meta, w, lin)
        got = be._forward_local_one(LinearCall(lin, x))
        assert be._local_fast_checked and all(be._local_fast_checked.values()), "the native path was not taken"
        assert be._local_w8n, "the native kernel was not called"
        assert got.dtype == ref.dtype and got.shape == ref.shape and torch.equal(got, ref)
    be.close()


def test_native_local_wide_shapes_and_threads() -> None:
    """Shapes with column tails, odd n, several flush blocks (A = 15: 512 features) and the pool split."""
    fast = B.native.fast
    rng = np.random.default_rng(3)
    for rows, n, m, ring in [(1, 1031, 517, RingParams(32, 15, 29)), (2, 4097, 2600, DEFAULT_RING),
                             (1, 2048, 4096, DEFAULT_RING), (5, 33, 16, RingParams(32, 12)),
                             (12, 900, 1152, DEFAULT_RING), (41, 640, 300, DEFAULT_RING)]:
        w8 = rng.integers(-128, 128, size=(m, n), dtype=np.int8)
        w8t = np.ascontiguousarray(w8.T)
        wnorm = float(np.sqrt((w8.astype(np.float64) ** 2).sum(axis=1)).max())
        scale = rng.random(m, dtype=np.float32)
        bias = rng.standard_normal(m).astype(np.float32)
        x = rng.standard_normal((rows, n)).astype(np.float32)
        from pgw.core.ring import dequantize_outputs, quantize_activations

        xq, sx = quantize_activations(torch.from_numpy(x), ring, wnorm if ring.cap else None)
        yq = xq.numpy().astype(np.int64) @ w8.T.astype(np.int64)
        ref = dequantize_outputs(torch.from_numpy(yq), sx, torch.from_numpy(scale), torch.from_numpy(bias)).numpy()
        bound = int(ring.output_bound(n))
        for threads in (1, 0):
            out = np.empty((rows, m), dtype=np.float32)
            sxo = np.empty(rows, dtype=np.float64)
            st = fast.local_linear_into(x, w8t, scale, bias, ring.k, ring.a, int(ring.cap or 0),
                                        wnorm if ring.cap else 0.0, bound, sxo, out, threads)
            want = 2 if int(np.abs(yq).max()) > bound else 0
            assert st == want
            if st == 0:
                assert np.array_equal(out.view(np.uint32), ref.view(np.uint32))
                assert np.array_equal(sxo, sx.numpy())
            else:
                assert not out.any()


def test_native_local_refuses_like_the_reference(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    be = _backend(qwen_pack)
    lid = LinearId(0, LinearKind.QKV)
    x = torch.randn(2, qwen_pack.arch.hidden_size)
    be.forward_local([LinearCall(lid, x)])  # the self-check passes first
    assert list(be._local_fast_checked.values()) == [True] and be._local_w8n
    scale, bias, _bound = be._consts(lid, be._meta.get(lid))
    monkeypatch.setattr(be, "_consts", lambda *_: (scale, bias, 0))  # every nonzero output is out of range
    with pytest.raises(IntegrityError) as ei:
        be.forward_local([LinearCall(lid, x)])
    assert ei.value.details["bound"] == 0
    monkeypatch.undo()
    with pytest.raises(RingOverflowError):
        be.forward_local([LinearCall(lid, torch.full((1, qwen_pack.arch.hidden_size), float("nan")))])
    be.close()


def test_native_local_env_off(qwen_pack: InMemoryPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_LOCAL_NATIVE", "0")
    monkeypatch.setattr(B, "cpu_int8_fast", lambda: True)
    be = _backend(qwen_pack)
    assert not be._local_native
    lid = LinearId(0, LinearKind.QKV)
    x = torch.randn(1, qwen_pack.arch.hidden_size)
    got = be._forward_local_one(LinearCall(lid, x))
    assert not be._local_w8n
    assert torch.equal(got, be._local_reference(x, be._meta.get(lid), be._local_w.get(lid), lid))
    be.close()


def test_native_local_view_follows_reloaded_weights(qwen_pack: InMemoryPack) -> None:
    """The cached int8 view is tied to the weight tensor it came from: reloaded weights (same linear key)
    are read from the new tensor, never from a stale view."""
    be = _backend(qwen_pack)
    lid = LinearId(0, LinearKind.QKV)
    x = torch.randn(1, qwen_pack.arch.hidden_size)
    be._forward_local_one(LinearCall(lid, x))
    first = be._local_w8n[lid.key][0]
    assert be._local_w is not None
    be._local_w._cache.clear()  # the next get() converts a fresh tensor
    w = be._local_w.get(lid)
    assert w.w8t is not first
    got = be._forward_local_one(LinearCall(lid, x))
    assert be._local_w8n[lid.key][0] is w.w8t
    assert torch.equal(got, be._local_reference(x, be._meta.get(lid), w, lid))
    be.close()
