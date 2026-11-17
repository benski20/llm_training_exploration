"""Tests for the metric contract and :class:`PGWMetrics` (SPEC 11.1)."""

from __future__ import annotations

import re
import urllib.request

import httpx
import pytest
from prometheus_client import CollectorRegistry, Counter

from pgw.observability import metrics as m
from pgw.observability.metrics import METRIC_NAMES, PGWMetrics, default_registry, reset_default_registry, safe_label

#: Label names that may appear in the contract: ids only, never prompt data.
ALLOWED_LABELS = {
    "tenant",
    "code",
    "linear_kind",
    "backend",
    "model",
    "endpoint",
    "status",
    "kind",
    "check",
    "fingerprint",
    "ring",
    "device",
}


@pytest.fixture
def reg() -> CollectorRegistry:
    return CollectorRegistry(auto_describe=True)


def test_contract_shape() -> None:
    name_re = re.compile(r"^pgw_(service|gateway|masks|factory)_[a-z0-9_]+$")
    for name, (kind, labels, help_) in METRIC_NAMES.items():
        assert name_re.match(name), name
        assert kind in {"counter", "gauge", "histogram"}
        assert (kind == "counter") == name.endswith("_total"), name
        assert set(labels) <= ALLOWED_LABELS, (name, labels)
        assert help_
    for key in (
        "pgw_gateway_ttft_seconds",
        "pgw_gateway_tpot_seconds",
        "pgw_service_request_seconds",
        "pgw_masks_available_slots",
        "pgw_gateway_integrity_failures_total",
        "pgw_gateway_noise_alarm",
    ):
        assert key in METRIC_NAMES


def test_all_metrics_render(reg: CollectorRegistry) -> None:
    pm = PGWMetrics(reg)
    pm.preregister()
    for name, (kind, labels, _) in METRIC_NAMES.items():
        c = pm.get(name)
        child = c.labels(**{lbl: "x" for lbl in labels}) if labels else c
        if kind == "counter":
            child.inc(2)
        elif kind == "gauge":
            child.set(3)
        else:
            child.observe(0.2)
    body, ctype = pm.render()
    text = body.decode()
    assert ctype.startswith("text/plain")
    for name, (kind, _, _) in METRIC_NAMES.items():
        sample = name + ("_count" if kind == "histogram" else "")
        assert re.search(rf"^{sample}[{{ ]", text, re.M), name
    assert "_created" not in text
    # Every exposed family is a contract metric.
    families = set(re.findall(r"^# TYPE (\S+) ", text, re.M))
    assert families <= {n.removesuffix("_total") for n in METRIC_NAMES} | set(METRIC_NAMES)


def test_shared_collectors_per_registry(reg: CollectorRegistry) -> None:
    a, b = PGWMetrics(reg), PGWMetrics(reg)
    assert a.get("pgw_service_rows_total") is b.get("pgw_service_rows_total")
    other = PGWMetrics(CollectorRegistry())
    assert other.get("pgw_service_rows_total") is not a.get("pgw_service_rows_total")


def test_default_registry_shared_and_resettable() -> None:
    reset_default_registry()
    try:
        r1 = default_registry()
        a, b = PGWMetrics(), PGWMetrics()
        assert a.registry is b.registry is r1
        a.get("pgw_factory_slots_generated_total").labels(fingerprint="f", ring="K40A12", device="cpu").inc(5)
        assert b.value("pgw_factory_slots_generated_total", fingerprint="f", ring="K40A12", device="cpu") == 5
        reset_default_registry()
        assert default_registry() is not r1
        assert (
            PGWMetrics().value("pgw_factory_slots_generated_total", fingerprint="f", ring="K40A12", device="cpu")
            is None
        )
    finally:
        reset_default_registry()


def test_unknown_and_prefix_restricted(reg: CollectorRegistry) -> None:
    pm = PGWMetrics(reg)
    with pytest.raises(KeyError):
        pm.get("pgw_gateway_prompt_text")
    svc = PGWMetrics(reg, prefixes=("pgw_service_",))
    svc.get("pgw_service_inflight")
    with pytest.raises(KeyError):
        svc.get("pgw_gateway_noise_alarm")
    svc.preregister()  # only its own families
    assert reg.get_sample_value("pgw_service_inflight") == 0.0


def test_disabled_is_noop() -> None:
    pm = PGWMetrics(enabled=False)
    c = pm.get("pgw_gateway_tokens_total")
    c.labels(model="m", kind="prompt").inc()
    c.set(1)
    c.observe(1)
    c.dec()
    assert pm.render()[0] == b""
    assert pm.value("pgw_gateway_tokens_total", model="m", kind="prompt") is None
    with pytest.raises(KeyError):
        pm.get("nope")
    with pytest.raises(RuntimeError):
        pm.start_http_server(0)


def test_value_and_buckets(reg: CollectorRegistry) -> None:
    pm = PGWMetrics(reg)
    pm.get("pgw_gateway_tokens_total").labels(model="m", kind="completion").inc(7)
    pm.get("pgw_service_queue_depth").set(4)
    h = pm.get("pgw_service_batch_rows").labels(linear_kind="qkv")
    h.observe(300)
    pm.get("pgw_gateway_ttft_seconds").labels(model="m").observe(0.3)
    assert pm.value("pgw_gateway_tokens_total", model="m", kind="completion") == 7
    assert pm.value("pgw_service_queue_depth") == 4
    assert pm.value("pgw_service_batch_rows", linear_kind="qkv") == 1
    assert pm.value("pgw_gateway_tpot_seconds", model="m") is None
    assert reg.get_sample_value("pgw_service_batch_rows_bucket", {"linear_kind": "qkv", "le": "512.0"}) == 1
    assert reg.get_sample_value("pgw_service_batch_rows_bucket", {"linear_kind": "qkv", "le": "256.0"}) == 0
    assert reg.get_sample_value("pgw_gateway_ttft_seconds_bucket", {"model": "m", "le": "0.5"}) == 1
    assert reg.get_sample_value("pgw_gateway_ttft_seconds_bucket", {"model": "m", "le": "0.25"}) == 0


def test_collect_hooks(reg: CollectorRegistry) -> None:
    pm = PGWMetrics(reg)
    calls = []

    def refresh() -> None:
        calls.append(1)
        pm.get("pgw_masks_available_slots").labels(fingerprint="abc", ring="K40A12").set(1000 - len(calls))

    def broken() -> None:
        raise RuntimeError("boom")

    pm.add_collect_hook(broken)
    pm.add_collect_hook(refresh)
    body, _ = pm.render()
    assert b'pgw_masks_available_slots{fingerprint="abc",ring="K40A12"} 999.0' in body
    pm.render()
    assert len(calls) == 2


def test_reuses_collector_registered_elsewhere(reg: CollectorRegistry) -> None:
    # A component registered its own fallback collector with the contract name first.
    own = Counter("pgw_service_rows", "Masked rows computed", ("tenant", "linear_kind"), registry=reg)
    pm = PGWMetrics(reg)
    assert pm.get("pgw_service_rows_total") is own


async def test_asgi_app(reg: CollectorRegistry) -> None:
    pm = PGWMetrics(reg)
    pm.get("pgw_service_inflight").set(2)
    transport = httpx.ASGITransport(app=pm.asgi_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        r = await client.get("/metrics")
        assert r.status_code == 200
        assert "pgw_service_inflight 2.0" in r.text
        assert r.headers["cache-control"] == "no-store"
        r = await client.post("/metrics")
        assert r.status_code == 405


def test_start_http_server(reg: CollectorRegistry) -> None:
    pm = PGWMetrics(reg)
    pm.get("pgw_factory_slots_per_second").labels(fingerprint="f", ring="K40A12", device="cpu").set(12.5)
    res = pm.start_http_server(0, "127.0.0.1")
    server = res[0] if isinstance(res, tuple) else None
    if server is None:  # pragma: no cover - very old prometheus_client
        pytest.skip("start_http_server does not return the server")
    try:
        port = server.server_address[1]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
            body = resp.read().decode()
        assert "pgw_factory_slots_per_second" in body and "12.5" in body
    finally:
        server.shutdown()
        server.server_close()


def test_safe_label() -> None:
    assert safe_label("qwen2.5-0.5b") == "qwen2.5-0.5b"
    assert safe_label('a b"c\n<d>') == "a_b_c__d_"
    assert len(safe_label("x" * 500)) == 64
    assert safe_label(42) == "42"


def test_service_metrics_use_pgw_metrics(reg: CollectorRegistry) -> None:
    from pgw.service.metrics import ServiceMetrics

    sm = ServiceMetrics(registry=reg)
    sm.request_done("t1", "ok", 0.01)
    sm.rows("t1", "qkv", 32)
    sm.batch("qkv", "cpu", 32, 0.002)
    body, _ = sm.render()
    assert b'pgw_service_rows_total{linear_kind="qkv",tenant="t1"} 32.0' in body
    # Same collectors as PGWMetrics on that registry.
    assert sm.get("pgw_service_rows_total") is PGWMetrics(reg).get("pgw_service_rows_total")


def test_factory_metrics_use_pgw_metrics(reg: CollectorRegistry) -> None:
    from pgw.factory.metrics import FactoryMetrics

    pm = PGWMetrics(reg)
    fm = FactoryMetrics(pm)
    fm.segment_done("ab" * 32, "K40A12", "cpu", 128, 64.0)
    assert fm.enabled
    text = pm.render()[0].decode()
    assert "pgw_factory_slots_generated_total" in text and "128.0" in text


def test_module_exports() -> None:
    assert set(m.__all__) >= {"METRIC_NAMES", "PGWMetrics", "default_registry", "safe_label"}
