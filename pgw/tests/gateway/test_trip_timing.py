"""Round-trip telemetry of the gateway's compute-client wrapper (``_InstrumentedClient``, ``TripTiming``)."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pgw.gateway.metrics import GatewayMetrics
from pgw.gateway.runtime import ModelRuntime, TripTiming, _InstrumentedClient, _row_class
from pgw.gateway.safety import BreakerOpenError, CircuitBreaker
from tests.gateway.conftest import MODEL, plain_settings


def _req(*rows: int, linear: str = "0.qkv") -> Any:
    return SimpleNamespace(items=[SimpleNamespace(linear=linear, rows=r) for r in rows])


def _resp(queue_ms: float = 0.0, compute_ms: float = 0.0) -> Any:
    return SimpleNamespace(timing=SimpleNamespace(queue_ms=queue_ms, compute_ms=compute_ms))


class _AsyncOnly:
    def __init__(self) -> None:
        self.calls = 0

    async def compute(self, request: Any) -> Any:
        self.calls += 1
        return _resp(1.0, 2.0)


class _WithSync(_AsyncOnly):
    def compute_sync(self, request: Any) -> Any:
        self.calls += 1
        return _resp(1.0, 2.0)


def _sample(m: GatewayMetrics, name: str, suffix: str, **labels: str) -> float | None:
    v = m.registry.get_sample_value(name + suffix, labels)
    return None if v is None else float(v)


# ---- compute_sync presence ------------------------------------------------------------------------------
def test_compute_sync_only_when_inner_has_it() -> None:
    m = GatewayMetrics(enabled=False)
    a = _InstrumentedClient(_AsyncOnly(), m, MODEL)
    assert not hasattr(a, "compute_sync")
    s = _InstrumentedClient(_WithSync(), m, MODEL)
    assert callable(s.compute_sync)
    s.compute_sync(_req(1))
    assert s.round_trips == 1 and s.inner.calls == 1


def test_non_callable_compute_sync_is_not_exposed() -> None:
    inner = _AsyncOnly()
    inner.compute_sync = None  # type: ignore[attr-defined]
    assert not callable(getattr(_InstrumentedClient(inner, GatewayMetrics(enabled=False), MODEL), "compute_sync",
                                None))


# ---- fail-closed gate -----------------------------------------------------------------------------------
async def test_breaker_open_refuses_sync_and_async_round_trips() -> None:
    br = CircuitBreaker(MODEL, None)
    inner = _WithSync()
    c = _InstrumentedClient(inner, GatewayMetrics(enabled=False), MODEL, br)
    c.compute_sync(_req(1))
    await c.compute(_req(1))
    assert inner.calls == 2
    br.trip(reason="test")
    with pytest.raises(BreakerOpenError):
        c.compute_sync(_req(1))
    with pytest.raises(BreakerOpenError):
        await c.compute(_req(1))
    assert inner.calls == 2  # nothing reached the service once the breaker opened
    assert c.round_trips == 2


# ---- TripTiming -----------------------------------------------------------------------------------------
@pytest.mark.parametrize(("rows", "cls"), [(0, "1"), (1, "1"), (2, "2-8"), (8, "2-8"), (9, "9+"), (4096, "9+")])
def test_row_class(rows: int, cls: str) -> None:
    assert _row_class(rows) == cls


def test_trip_timing_buckets_and_wire_split() -> None:
    c = _InstrumentedClient(_WithSync(), GatewayMetrics(enabled=False), MODEL)
    c._record(_req(1), _resp(queue_ms=1.0, compute_ms=2.0), 0.010)  # wire = 10 - 3 ms
    c._record(_req(1), _resp(queue_ms=1.0, compute_ms=2.0), 0.002)  # service > rtt: wire clamps at 0
    c._record(_req(3, 4), _resp(queue_ms=0.5, compute_ms=0.5), 0.004)  # 7 rows, multi-item
    c._record(_req(32), None, 0.005)  # no ServerTiming: all of it counts as wire
    snap = c.timing_snapshot()
    assert set(snap) == {"1", "2-8", "9+"}
    one = snap["1"]
    assert one["trips"] == 2 and one["rows"] == 2 and one["multi_item"] == 0
    assert one["rtt_s"] == pytest.approx(0.012)
    assert one["queue_s"] == pytest.approx(0.002) and one["compute_s"] == pytest.approx(0.004)
    assert one["wire_s"] == pytest.approx(0.007)
    mid = snap["2-8"]
    assert mid["trips"] == 1 and mid["rows"] == 7 and mid["multi_item"] == 1
    assert mid["wire_s"] == pytest.approx(0.003)
    big = snap["9+"]
    assert big["rows"] == 32 and big["queue_s"] == 0.0 and big["wire_s"] == pytest.approx(0.005)
    assert c.round_trips == 4


def test_trip_timing_is_thread_safe() -> None:
    t = TripTiming()
    c = _InstrumentedClient(_WithSync(), GatewayMetrics(enabled=False), MODEL)
    n, threads = 2000, 4

    def work() -> None:
        for _ in range(n):
            t.add(1, 1, 0.001, 0.0, 0.0, 0.001)
            c.compute_sync(_req(1))

    ts = [threading.Thread(target=work) for _ in range(threads)]
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    assert t.snapshot()["1"]["trips"] == n * threads
    assert c.round_trips == n * threads
    assert c.timing_snapshot()["1"]["trips"] == n * threads


# ---- histograms -----------------------------------------------------------------------------------------
def test_split_histograms_record() -> None:
    m = GatewayMetrics()
    if not m.enabled:
        pytest.skip("prometheus_client not installed")
    c = _InstrumentedClient(_WithSync(), m, MODEL)
    c._record(_req(1, linear="3.mlp_down"), _resp(queue_ms=1.0, compute_ms=2.0), 0.010)
    lbl = {"model": MODEL, "linear_kind": "mlp_down"}
    assert _sample(m, "pgw_gateway_round_trip_seconds", "_count", **lbl) == 1.0
    assert _sample(m, "pgw_gateway_network_seconds", "_count", **lbl) == 1.0
    assert _sample(m, "pgw_gateway_network_seconds", "_sum", **lbl) == pytest.approx(0.007)
    assert _sample(m, "pgw_gateway_service_seconds", "_count", **lbl) == 1.0
    assert _sample(m, "pgw_gateway_service_seconds", "_sum", **lbl) == pytest.approx(0.003)
    body, _ = m.render()
    assert b"pgw_gateway_network_seconds_bucket" in body and b"pgw_gateway_service_seconds_bucket" in body


def test_cert_expiry_gauge_is_registered() -> None:
    m = GatewayMetrics()
    if not m.enabled:
        pytest.skip("prometheus_client not installed")
    m.get("pgw_gateway_client_cert_expiry_seconds").labels(model=MODEL).set(42.0)
    assert m.registry.get_sample_value("pgw_gateway_client_cert_expiry_seconds", {"model": MODEL}) == 42.0


# ---- Generator wiring (plan interface C3) ---------------------------------------------------------------
async def test_spec_max_batch_reaches_generator(tiny_pack_dir: Path, tmp_path: Path) -> None:
    settings = plain_settings(tiny_pack_dir, tmp_path, model_kw={"spec_max_batch": 3})
    rt = ModelRuntime(settings.all_models()[0], settings)
    await rt.start()
    try:
        assert rt.generator.spec_max_batch == 3
    finally:
        await rt.stop()
