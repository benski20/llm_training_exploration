"""Linear backends: exact integer reference, masked protocol, integrity checks and failure modes."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    ConfigError,
    FingerprintMismatchError,
    IntegrityError,
    LinearId,
    MaskExhaustedError,
    ProtocolError,
    RingParams,
    ServiceUnavailableError,
    ShapeError,
    dequantize_outputs,
    quantize_activations,
)
from pgw.engine import (
    FloatBackend,
    InMemoryPack,
    LinearCall,
    LocalComputeClient,
    LocalMaskReservation,
    LocalMaskSource,
    MaskedBackend,
    PlainBackend,
    QuantPlainBackend,
    exact_int_matmul,
)
from tests.engine.conftest import TwoExtentReservation

QKV0 = LinearId.parse("0.qkv")
DOWN1 = LinearId.parse("1.down")


def _x(rows: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, n, generator=g) * 3


@pytest.mark.parametrize("bits", [8, 12, 16, 20, 24])
def test_exact_int_matmul(bits: int) -> None:
    g = torch.Generator().manual_seed(bits)
    qmax = (1 << (bits - 1)) - 1
    xq = torch.randint(-qmax, qmax + 1, (5, 70), generator=g, dtype=torch.int64)
    xq[0, :] = qmax
    xq[1, :] = -qmax
    w = torch.randint(-128, 128, (70, 33), generator=g, dtype=torch.int64)
    got = exact_int_matmul(xq, w.to(torch.int8), bits)
    ref = torch.tensor((xq.numpy().astype(object) @ w.numpy().astype(object)).astype("int64"))
    assert torch.equal(got, ref)
    assert exact_int_matmul(xq[:0], w.to(torch.int8), bits).shape == (0, 33)


@pytest.mark.gpu
@pytest.mark.parametrize("bits", [8, 12, 20])
@pytest.mark.parametrize("t,n,m", [(1, 64, 48), (5, 70, 33), (40, 3584, 4608)])
def test_exact_int_matmul_cuda(bits: int, t: int, n: int, m: int) -> None:  # pragma: no cover - GPU only
    """CUDA path, both weight layouts: row-major (n, m) and the column-major view PlainBackend keeps.

    Regression: T4 + CUDA 13 cuBLASLt rejects a row-major int8 ``mat2`` (CUBLAS_STATUS_NOT_SUPPORTED).
    """
    g = torch.Generator().manual_seed(bits + t)
    qmax = (1 << (bits - 1)) - 1
    xq = torch.randint(-qmax, qmax + 1, (t, n), generator=g, dtype=torch.int64)
    xq[0, :] = qmax
    w = torch.randint(-128, 128, (n, m), generator=g, dtype=torch.int64)
    ref = torch.tensor((xq.numpy().astype(object) @ w.numpy().astype(object)).astype("int64"))
    w8 = w.to(torch.int8)
    for wt in (w8.cuda(), w8.t().contiguous().cuda().t()):
        assert torch.equal(exact_int_matmul(xq.cuda(), wt, bits).cpu(), ref)


@pytest.mark.gpu
def test_plain_backend_cuda_matches_cpu(qwen_pack: InMemoryPack, ring: RingParams) -> None:  # pragma: no cover
    x = _x(7, qwen_pack.load_linear(QKV0).w8.shape[1])
    cpu = PlainBackend(qwen_pack, ring).forward(LinearCall(QKV0, x))
    gpu = PlainBackend(qwen_pack, ring, device="cuda").forward(LinearCall(QKV0, x))
    assert torch.equal(cpu, gpu)


def test_plain_backend_matches_reference(qwen_pack: InMemoryPack, ring: RingParams) -> None:
    be = PlainBackend(qwen_pack, ring)
    lw = qwen_pack.load_linear(QKV0)
    x = _x(6, lw.w8.shape[1])
    y = be.forward(LinearCall(QKV0, x))
    xq, sx = quantize_activations(x, ring)
    yq = xq @ lw.w8.to(torch.int64).t()
    assert torch.equal(y, dequantize_outputs(yq, sx, lw.scale, lw.bias))
    assert QuantPlainBackend is PlainBackend
    with pytest.raises(ShapeError):
        be.forward(LinearCall(QKV0, torch.zeros(2, 3)))


def test_plain_forward_many_concatenates_exactly(qwen_pack: InMemoryPack) -> None:
    be = PlainBackend(qwen_pack, RING_K40_A12, preload=True)
    n = qwen_pack.arch.hidden_size
    calls = [LinearCall(QKV0, _x(3, n, 1)), LinearCall(QKV0, _x(1, n, 2)),
             LinearCall(LinearId.parse("0.gate_up"), _x(2, n, 3))]
    many = be.forward_many(calls)
    for c, y in zip(calls, many, strict=True):
        assert torch.equal(y, be.forward(c))


def test_float_backend(qwen_pack: InMemoryPack) -> None:
    be = FloatBackend(qwen_pack, RING_K40_A12)
    lw = qwen_pack.load_linear(QKV0)
    x = _x(4, lw.w8.shape[1])
    ref = x @ (lw.w8.float() * lw.scale[:, None]).t() + lw.bias
    torch.testing.assert_close(be.forward(LinearCall(QKV0, x)), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("lid", ["0.qkv", "0.o", "1.gate_up", "1.down"])
def test_masked_equals_plain_every_ring(qwen_pack: InMemoryPack, ring: RingParams, lid: str) -> None:
    lin = LinearId.parse(lid)
    n = qwen_pack.load_linear(lin).w8.shape[1]
    client = LocalComputeClient(qwen_pack, record=True)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint)
    res = LocalMaskReservation(qwen_pack, ring, 16)
    x = _x(5, n, 7)
    y = be.forward(LinearCall(lin, x, session=res))
    assert torch.equal(y, PlainBackend(qwen_pack, ring).forward(LinearCall(lin, x)))
    # the service saw only masked values: never the quantized activations
    seen = client.received[0][1]
    xq, _ = quantize_activations(x, ring)
    assert not torch.equal(seen, xq & ring.mask)
    assert int(seen.max()) < ring.modulus and int(seen.min()) >= 0
    assert res.remaining(lin) == 11
    st = be.stats.snapshot()
    assert st["round_trips"] == 1 and st["freivalds_checked"] == 1 and st["rows"] == 5


def test_masked_payload_looks_uniform(qwen_pack: InMemoryPack) -> None:
    """Even an all-zero input produces uniform-looking ring elements on the wire."""
    ring = RING_K40_A12
    client = LocalComputeClient(qwen_pack, record=True)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint)
    res = LocalMaskReservation(qwen_pack, ring, 64)
    be.forward(LinearCall(QKV0, torch.zeros(64, qwen_pack.arch.hidden_size), session=res))
    seen = client.received[0][1].double()
    mean = float(seen.mean()) / ring.modulus
    assert abs(mean - 0.5) < 0.02
    top_bit = ((client.received[0][1] >> (ring.k - 1)) & 1).double().mean()
    assert abs(float(top_bit) - 0.5) < 0.03


def test_masked_multi_call_single_round_trip(qwen_pack: InMemoryPack) -> None:
    ring = RING_K48_A20
    client = LocalComputeClient(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint)
    plain = PlainBackend(qwen_pack, ring)
    src = LocalMaskSource(qwen_pack)
    sessions = [src.reserve(qwen_pack.fingerprint, ring, 8) for _ in range(3)]
    n = qwen_pack.arch.hidden_size
    calls = [LinearCall(QKV0, _x(r, n, r), session=s) for r, s in zip((1, 4, 2), sessions, strict=True)]
    outs = be.forward_many(calls)
    assert client.requests == 1
    for c, y in zip(calls, outs, strict=True):
        assert torch.equal(y, plain.forward(LinearCall(c.id, c.x)))


def test_masked_splits_at_extent_boundaries(qwen_pack: InMemoryPack) -> None:
    ring = RING_K40_A12
    client = LocalComputeClient(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint)
    res = TwoExtentReservation(qwen_pack, ring, 3, 10)
    x = _x(7, qwen_pack.arch.hidden_size, 11)
    y = be.forward(LinearCall(QKV0, x, session=res))
    assert res.boundaries_seen == [3, 4]
    assert torch.equal(y, PlainBackend(qwen_pack, ring).forward(LinearCall(QKV0, x)))
    assert be.stats.snapshot()["freivalds_checked"] == 2


def test_masked_max_rows_per_item(qwen_pack: InMemoryPack) -> None:
    ring = RING_K40_A12
    seen_items: list[int] = []

    class Spy(LocalComputeClient):
        def compute_sync(self, request: Any) -> Any:
            seen_items.extend(it.rows for it in request.items)
            return super().compute_sync(request)

    be = MaskedBackend(qwen_pack, None, Spy(qwen_pack), ring, qwen_pack.fingerprint, max_rows_per_item=4)
    res = LocalMaskReservation(qwen_pack, ring, 16)
    x = _x(10, qwen_pack.arch.hidden_size)
    y = be.forward(LinearCall(QKV0, x, session=res))
    assert seen_items == [4, 4, 2]
    assert torch.equal(y, PlainBackend(qwen_pack, ring).forward(LinearCall(QKV0, x)))


def _flip_low_bit(lid: str, y: torch.Tensor) -> torch.Tensor:
    y = y.clone()
    y[0, 0] += 1
    return y


def _flip_high_bit(lid: str, y: torch.Tensor) -> torch.Tensor:
    y = y.clone()
    y[0, 0] += 1 << 38
    return y


@pytest.mark.parametrize("tamper", [_flip_low_bit, _flip_high_bit])
def test_tampering_detected(qwen_pack: InMemoryPack, tamper: Any) -> None:
    ring = RING_K40_A12
    tripped: list[Exception] = []
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack, tamper=tamper), ring, qwen_pack.fingerprint,
                       on_integrity_failure=tripped.append)
    res = LocalMaskReservation(qwen_pack, ring, 4)
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(QKV0, _x(2, qwen_pack.arch.hidden_size), session=res))
    assert len(tripped) == 1 and be.stats.snapshot()["integrity_failures"] == 1


def test_range_check_without_freivalds(qwen_pack: InMemoryPack) -> None:
    ring = RING_K40_A12
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack, tamper=_flip_high_bit), ring,
                       qwen_pack.fingerprint, freivalds_rate=0.0)
    res = LocalMaskReservation(qwen_pack, ring, 4, checks=0)
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(QKV0, _x(2, qwen_pack.arch.hidden_size), session=res))
    # undetectable by range check alone, but freivalds_rate=0 means no check ran
    be2 = MaskedBackend(qwen_pack, None, LocalComputeClient(qwen_pack), ring, qwen_pack.fingerprint,
                        freivalds_rate=0.0)
    be2.forward(LinearCall(QKV0, _x(2, qwen_pack.arch.hidden_size), session=LocalMaskReservation(qwen_pack, ring, 2)))
    assert be2.stats.snapshot()["freivalds_skipped"] == 1


def test_mask_exhaustion_precheck_consumes_nothing(qwen_pack: InMemoryPack) -> None:
    ring = RING_K40_A12
    client = LocalComputeClient(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint)
    ok = LocalMaskReservation(qwen_pack, ring, 8)
    short = LocalMaskReservation(qwen_pack, ring, 2)
    n = qwen_pack.arch.hidden_size
    with pytest.raises(MaskExhaustedError):
        be.forward_many([LinearCall(QKV0, _x(2, n), session=ok), LinearCall(QKV0, _x(3, n), session=short)])
    assert ok.remaining(QKV0) == 8 and short.remaining(QKV0) == 2 and client.requests == 0
    # same reservation used twice in one batch: needs are summed
    with pytest.raises(MaskExhaustedError):
        be.forward_many([LinearCall(QKV0, _x(5, n), session=ok), LinearCall(QKV0, _x(4, n), session=ok)])


def test_reservation_single_use(qwen_pack: InMemoryPack) -> None:
    ring = RING_K32_A8
    res = LocalMaskReservation(qwen_pack, ring, 4, seed=bytes(32))
    r1, wr1 = res.take(QKV0, 2)
    r2, _ = res.take(QKV0, 2)
    assert not torch.equal(r1, r2)
    with pytest.raises(MaskExhaustedError):
        res.take(QKV0, 1)
    # the same seed regenerates the same masks (factory derivation is deterministic, SPEC 7.3)
    again = LocalMaskReservation(qwen_pack, ring, 4, seed=bytes(32))
    r1b, wr1b = again.take(QKV0, 2)
    assert torch.equal(r1, r1b) and torch.equal(wr1, wr1b)
    from pgw.core import SeedExpander

    exp = SeedExpander(bytes(32)).ring_uniform_rows(qwen_pack.arch.hidden_size, ring.k, 0, 2, "r", "0.qkv")
    assert torch.equal(exp, r1)
    res.close()
    from pgw.core import MaskReuseError

    with pytest.raises(MaskReuseError):
        res.take(LinearId.parse("0.o"), 1)


def test_masked_backend_config_errors(qwen_pack: InMemoryPack) -> None:
    client = LocalComputeClient(qwen_pack)
    with pytest.raises(FingerprintMismatchError):
        MaskedBackend(qwen_pack, None, client, RING_K40_A12, "0" * 64)
    with pytest.raises(ConfigError):
        MaskedBackend(qwen_pack, None, client, RING_K40_A12, qwen_pack.fingerprint, freivalds_rate=2.0)
    be = MaskedBackend(qwen_pack, None, client, RING_K40_A12, qwen_pack.fingerprint)
    with pytest.raises(ConfigError):  # no session and no default mask source
        be.forward(LinearCall(QKV0, _x(1, qwen_pack.arch.hidden_size)))
    with pytest.raises(ConfigError):  # ring mismatch
        be.forward(LinearCall(QKV0, _x(1, qwen_pack.arch.hidden_size),
                              session=LocalMaskReservation(qwen_pack, RING_K48_A20, 2)))
    default = LocalMaskReservation(qwen_pack, RING_K40_A12, 2)
    be2 = MaskedBackend(qwen_pack, default, client, RING_K40_A12, qwen_pack.fingerprint)
    be2.forward(LinearCall(QKV0, _x(1, qwen_pack.arch.hidden_size)))
    assert default.remaining(QKV0) == 1
    with pytest.raises(FingerprintMismatchError):
        LocalMaskSource(qwen_pack).reserve("1" * 64, RING_K40_A12, 1)


def test_bad_responses(qwen_pack: InMemoryPack) -> None:
    from pgw.wire.messages import ComputeResponse, ComputeResult

    ring = RING_K40_A12
    n = qwen_pack.arch.hidden_size

    class Wrong(LocalComputeClient):
        mode = "id"

        def compute_sync(self, request: Any) -> Any:
            resp = super().compute_sync(request)
            if self.mode == "id":
                return ComputeResponse("other", resp.items)
            if self.mode == "count":
                return ComputeResponse(resp.request_id, ())
            if self.mode == "linear":
                it = resp.items[0]
                return ComputeResponse(resp.request_id, (ComputeResult("0.o", it.rows, it.y_hat),))
            if self.mode == "bytes":
                it = resp.items[0]
                return ComputeResponse(resp.request_id, (ComputeResult(it.linear, it.rows, it.y_hat[:-1]),))
            raise ConnectionError("network down")

    client = Wrong(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint)
    for mode, exc in (("id", ProtocolError), ("count", ProtocolError), ("linear", ProtocolError),
                      ("bytes", ProtocolError), ("net", ServiceUnavailableError)):
        client.mode = mode
        with pytest.raises(exc):
            be.forward(LinearCall(QKV0, _x(1, n), session=LocalMaskReservation(qwen_pack, ring, 1)))


def test_service_rejects_wrong_fingerprint(qwen_pack: InMemoryPack, llama_pack: InMemoryPack) -> None:
    ring = RING_K40_A12
    be = MaskedBackend(qwen_pack, None, LocalComputeClient(llama_pack), ring, qwen_pack.fingerprint)
    with pytest.raises(FingerprintMismatchError):
        be.forward(LinearCall(QKV0, _x(1, qwen_pack.arch.hidden_size),
                              session=LocalMaskReservation(qwen_pack, ring, 1)))


def test_noise_monitor_is_consulted(qwen_pack: InMemoryPack) -> None:
    ring = RING_K40_A12

    class Monitor:
        def __init__(self) -> None:
            self.calls = 0

        def guard(self, x_hat: torch.Tensor, xq: torch.Tensor | None = None) -> None:
            self.calls += 1
            if self.calls > 1:
                raise IntegrityError("noise alarm")

    mon = Monitor()
    client = LocalComputeClient(qwen_pack)
    be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint, noise_monitor=mon)
    n = qwen_pack.arch.hidden_size
    be.forward(LinearCall(QKV0, _x(1, n), session=LocalMaskReservation(qwen_pack, ring, 1)))
    with pytest.raises(IntegrityError):
        be.forward(LinearCall(QKV0, _x(1, n), session=LocalMaskReservation(qwen_pack, ring, 1)))
    assert client.requests == 1  # blocked before sending


async def test_masked_async_paths(qwen_pack: InMemoryPack) -> None:
    ring = RING_K40_A12
    n = qwen_pack.arch.hidden_size
    plain = PlainBackend(qwen_pack, ring)

    class SyncOnly:
        def __init__(self) -> None:
            self.inner = LocalComputeClient(qwen_pack)

        def compute_sync(self, request: Any) -> Any:
            return self.inner.compute_sync(request)

    for client in (LocalComputeClient(qwen_pack), SyncOnly()):
        be = MaskedBackend(qwen_pack, None, client, ring, qwen_pack.fingerprint)
        x = _x(3, n)
        ys = await asyncio.gather(
            be.aforward(LinearCall(QKV0, x, session=LocalMaskReservation(qwen_pack, ring, 3))),
            be.aforward(LinearCall(QKV0, x, session=LocalMaskReservation(qwen_pack, ring, 3))),
        )
        ref = plain.forward(LinearCall(QKV0, x))
        assert all(torch.equal(y, ref) for y in ys)
    assert await be.aforward_many([]) == []


def test_masked_sync_with_async_only_client(qwen_pack: InMemoryPack) -> None:
    ring = RING_K40_A12

    class AsyncOnly:
        def __init__(self) -> None:
            self.inner = LocalComputeClient(qwen_pack)

        async def compute(self, request: Any) -> Any:
            return self.inner.compute_sync(request)

    be = MaskedBackend(qwen_pack, None, AsyncOnly(), ring, qwen_pack.fingerprint)
    x = _x(2, qwen_pack.arch.hidden_size)
    y = be.forward(LinearCall(QKV0, x, session=LocalMaskReservation(qwen_pack, ring, 2)))
    assert torch.equal(y, PlainBackend(qwen_pack, ring).forward(LinearCall(QKV0, x)))


async def test_local_client_models(qwen_pack: InMemoryPack) -> None:
    c = LocalComputeClient(qwen_pack)
    m = await c.connect(qwen_pack.fingerprint, "K40A12")
    assert m.models[0].fingerprint == qwen_pack.fingerprint
    assert len(m.models[0].linears) == 4 * qwen_pack.arch.num_layers
    with pytest.raises(FingerprintMismatchError):
        await c.connect("0" * 64)
    with pytest.raises(ConfigError):
        await c.connect(None, "K56A40")
    await c.aclose()


def test_exact_int_matmul_uses_shared_extreme_operand_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU int8 kernel that is exact on small all-ones inputs but saturates on extreme operands must
    not be used by the plain (reference) backend: the shared probe rejects it and the float64 path
    keeps results exact."""
    import sys

    core_ringmm = sys.modules["pgw.core.ringmm"]  # pgw.core.ringmm the attribute is the function

    real = torch._int_mm

    def saturating(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return real(a, b).clamp(-(1 << 20), 1 << 20)

    core_ringmm.cpu_int8_available.cache_clear()
    monkeypatch.setattr(torch, "_int_mm", saturating)
    try:
        assert core_ringmm.cpu_int8_available() is False
        gen = torch.Generator().manual_seed(9)
        xq = torch.full((3, 4096), -2047, dtype=torch.int64)
        w = torch.full((4096, 5), -128, dtype=torch.int8)
        got = exact_int_matmul(xq, w, 12)
        assert torch.equal(got, torch.full((3, 5), 2047 * 128 * 4096, dtype=torch.int64))
        xr = torch.randint(-2047, 2048, (4, 300), dtype=torch.int64, generator=gen)
        wr = torch.randint(-127, 128, (300, 7), dtype=torch.int8, generator=gen)
        assert torch.equal(exact_int_matmul(xr, wr, 12), xr @ wr.to(torch.int64))
    finally:
        monkeypatch.setattr(torch, "_int_mm", real)
        core_ringmm.cpu_int8_available.cache_clear()


def test_one_digit_gemm_implementation_everywhere() -> None:
    import sys

    core_ringmm = sys.modules["pgw.core.ringmm"]  # pgw.core.ringmm the attribute is the function
    from pgw.factory import kernels as fk
    from pgw.service import kernels as sk

    assert sk.int8_ring_matmul is core_ringmm.int8_ring_matmul is fk.int8_ring_matmul
    assert sk.cpu_int8_available is core_ringmm.cpu_int8_available is fk.cpu_int8_available
    assert core_ringmm.int_mm_max_inner("cpu") == 65_536
    for mod in (sk, fk):
        assert not hasattr(mod, "_INT_MM_MAX_INNER") and not hasattr(mod, "INT_MM_MAX_INNER")


def test_masked_backend_never_loads_int8_weights(tiny_pack: Any) -> None:
    """The masked gateway reads only scales/bias/shapes (``load_linear_meta``), never ``W8``."""
    ring = RingParams(40, 12)
    lid = LinearId.parse("0.qkv")
    meta = tiny_pack.load_linear_meta(lid)
    full = tiny_pack.load_linear(lid)
    assert torch.equal(meta.scale, full.scale) and (meta.out_features, meta.in_features) == tuple(full.w8.shape)

    class GatewayView:
        """The gateway's view of the pack: loading a full weight is a test failure."""

        def __init__(self, pack: Any) -> None:
            self._pack = pack
            self.meta_calls: list[str] = []

        def __getattr__(self, name: str) -> Any:
            return getattr(self._pack, name)

        def load_linear(self, *a: Any, **k: Any) -> Any:
            raise AssertionError("gateway loaded a full int8 weight")

        def load_linear_meta(self, lid: Any, **k: Any) -> Any:
            self.meta_calls.append(str(lid))
            return self._pack.load_linear_meta(lid, **k)

    view = GatewayView(tiny_pack)
    x = _x(3, meta.in_features, 2)
    be = MaskedBackend(view, None, LocalComputeClient(tiny_pack), ring, tiny_pack.fingerprint)
    y = be.forward(LinearCall(lid, x, session=LocalMaskReservation(tiny_pack, ring, 8)))
    assert torch.equal(y, PlainBackend(tiny_pack, ring).forward(LinearCall(lid, x)))
    assert view.meta_calls == ["0.qkv"]
