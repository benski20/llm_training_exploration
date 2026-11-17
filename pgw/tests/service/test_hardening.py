"""Regression tests for issues found in the service hardening review."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RING_K40_A12, LinearId, LinearKind, LinearSpec, PGWError, RingParams
from pgw.service import BatchScheduler, ModelRegistry, RegisteredModel, ServiceSettings, build_hypercorn_config
from pgw.service.app import _client_cert_header_middleware
from pgw.service.handler import TokenBucket
from pgw.service.kernels import ComputeKernel, cpu_int8_available, int8_ring_matmul
from tests.service.conftest import expected, rand_x

FP = "ab" * 32


# ---- token bucket ----------------------------------------------------------------------------
class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_token_bucket_charges_oversized_requests_in_full() -> None:
    """A request larger than the burst must not cost only ``burst`` (rate-limit bypass)."""
    clk = _Clock()
    tb = TokenBucket(rate=10.0, burst=20.0, clock=clk)
    assert tb.take(1000) == 0.0  # full bucket: admitted, balance goes to -980
    wait = tb.take(1)
    assert wait == pytest.approx(98.1)  # repay 980 debt + 1 token at 10/s
    clk.t = 50.0
    assert tb.take(1) > 0  # still in debt
    clk.t = 100.0
    assert tb.take(20) == 0.0  # bucket full again (capped at burst)
    assert tb.take(1) > 0


def test_token_bucket_long_run_rate_is_respected() -> None:
    clk = _Clock()
    tb = TokenBucket(rate=100.0, burst=50.0, clock=clk)
    admitted = 0
    for _ in range(10_000):
        clk.t += 0.01
        if tb.take(500) == 0.0:
            admitted += 500
    # 100 s at 100 rows/s, plus at most one burst-sized request of slack
    assert admitted <= 100 * 100 + 500


def test_token_bucket_validation() -> None:
    with pytest.raises(ValueError):
        TokenBucket(0, 1)
    with pytest.raises(ValueError):
        TokenBucket(1, 1).take(-1)


# ---- CPU int8 chunking -----------------------------------------------------------------------
@pytest.mark.skipif(not cpu_int8_available(), reason="CPU torch._int_mm unavailable")
def test_cpu_int8_uses_cpu_chunk_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    kmod = sys.modules["pgw.core.ringmm"]  # the package attribute is the ringmm() function

    calls: list[int] = []
    real = torch._int_mm

    def spy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        calls.append(int(a.shape[1]))
        return real(a, b)

    monkeypatch.setattr(kmod, "INT_MM_MAX_INNER_CPU", 24)
    monkeypatch.setattr(torch, "_int_mm", spy)
    ring = RingParams(48, 20)
    w = torch.full((5, 60), -128, dtype=torch.int8)
    w[1::2] = 127
    x = rand_x(3, 60, ring)
    assert torch.equal(int8_ring_matmul(x, w, ring.k), expected(x, w, ring))
    assert calls and max(calls) <= 24


def test_int8_ring_matmul_rejects_bad_ring_width() -> None:
    from pgw.core import ConfigError

    with pytest.raises(ConfigError):
        int8_ring_matmul(torch.zeros((1, 8), dtype=torch.int64), torch.zeros((2, 8), dtype=torch.int8), 44)


# ---- batcher: kernel shape contract ----------------------------------------------------------
class _BadShapeKernel(ComputeKernel):
    def run(self, x_hat: torch.Tensor, ring: Any) -> torch.Tensor:
        return torch.zeros((int(x_hat.shape[0]) + 1, self.out_features), dtype=torch.int64)


async def test_kernel_wrong_shape_fails_batch_not_other_requests() -> None:
    reg = ModelRegistry("cpu", verify="fingerprint")
    spec = LinearSpec(LinearId(0, LinearKind.O), 8, 16)
    w = torch.randint(-127, 128, (8, 16), dtype=torch.int8)
    reg.register(RegisteredModel("bad", FP, [spec], {"0.o": _BadShapeKernel(spec, w, device="cpu")},
                                 rings=("K40A12",)))
    b = BatchScheduler(reg, batch_window_ms=2)
    await b.start()
    res = await asyncio.gather(
        *(b.submit(FP, RING_K40_A12, "0.o", rand_x(2, 16, RING_K40_A12, seed=i)) for i in range(3)),
        return_exceptions=True,
    )
    assert all(isinstance(r, PGWError) and r.code == "internal" for r in res)
    assert b.queued_rows == 0 and b.queued_items == 0
    await b.stop()


# ---- trusted client-cert header --------------------------------------------------------------
async def _run_mw(headers: list[tuple[bytes, bytes]], tls: dict[str, Any] | None = None) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        seen.update(scope["extensions"]["tls"])

    mw = _client_cert_header_middleware(inner, "X-Client-Cert")
    scope = {"type": "http", "headers": headers, "extensions": {"tls": dict(tls or {})}}

    async def _noop(*_a: Any) -> Any:
        return None

    await mw(scope, _noop, _noop)
    return seen


async def test_trusted_header_single_value_is_used() -> None:
    seen = await _run_mw([(b"x-client-cert", b" CN=gw ")])
    assert seen == {"client_cert_name": "CN=gw"}


async def test_trusted_header_duplicates_and_direct_tls_ignored(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        seen = await _run_mw([(b"x-client-cert", b"CN=gw"), (b"X-Client-Cert", b"CN=evil")],
                             tls={"client_cert_name": "CN=direct"})
    assert "client_cert_name" not in seen and "client_cert_subject" not in seen
    assert any("copies" in r.getMessage() for r in caplog.records)


# ---- plaintext listener warning --------------------------------------------------------------
def test_warns_when_serving_without_tls_on_public_interface(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="pgw.service.server"):
        build_hypercorn_config(ServiceSettings(host="0.0.0.0", port=1234, allow_unauthenticated=True))  # noqa: S104
    assert any("WITHOUT TLS" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pgw.service.server"):
        build_hypercorn_config(ServiceSettings(host="127.0.0.1", port=1234, allow_unauthenticated=True))
    assert not any("WITHOUT TLS" in r.getMessage() for r in caplog.records)


def test_settings_reject_unknown_backend(tmp_path: Path) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ServiceSettings(backend="tpu", allow_unauthenticated=True)  # type: ignore[arg-type]
