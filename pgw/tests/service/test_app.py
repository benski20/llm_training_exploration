"""End-to-end tests of the service app (in-process ASGI, real wire client, tiny pack on CPU)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import (
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    ConfigError,
    ErrorCode,
    FingerprintMismatchError,
    PGWError,
    RingParams,
    check_output_range,
    mask_activations,
    pack_ring,
    quantize_activations,
    unmask_outputs,
    unpack_ring,
)
from pgw.service import ServiceSettings, create_app
from pgw.wire import CONTENT_TYPE, PROTOCOL_HEADER, ComputeItem, ComputeRequest, decode_message, encode_message
from pgw.wire.messages import ErrorResponse
from tests.service.conftest import (
    TOKEN_A,
    TOKEN_B,
    asgi_client,
    expected,
    http_client,
    make_req,
    rand_x,
    running_app,
)


def _settings(tiny_pack_dir: Path, **kw: Any) -> ServiceSettings:
    base: dict[str, Any] = {
        "packs": [tiny_pack_dir], "device": "cpu", "verify_packs": "fingerprint", "allow_unauthenticated": True,
        "batch_window_ms": 1.0, "warmup": False,
    }
    base.update(kw)
    return ServiceSettings(**base)


async def test_compute_exact_all_rings_and_items(tiny_pack_dir: Path, pack_fp: str,
                                                 pack_weights: dict[str, torch.Tensor]) -> None:
    async with running_app(_settings(tiny_pack_dir)) as app:
        c = asgi_client(app)
        models = await c.connect(pack_fp, "K40A12")
        info = models.find(pack_fp)
        assert info is not None and info.name == "tiny-qwen2"
        assert set(info.rings) == {"K40A12", "K48A20", "K32A8"}
        assert info.linear_dims() == {k: tuple(w.shape) for k, w in pack_weights.items()}
        assert models.server == "pgw-service"
        for ring in (RING_K32_A8, RING_K40_A12, RING_K48_A20):
            items = [(lid, rand_x(r, w.shape[1], ring, seed=i)) for i, (lid, w, r) in
                     enumerate((lid, w, 1 + i) for i, (lid, w) in enumerate(pack_weights.items()))]
            resp = await c.compute(make_req(pack_fp, ring, items, rid=f"req-{ring.name}"))
            assert resp.request_id == f"req-{ring.name}"
            for (lid, x), res in zip(items, resp.items, strict=True):
                m = pack_weights[lid].shape[0]
                y = unpack_ring(res.y_hat, (x.shape[0], m), ring)
                assert res.linear == lid and torch.equal(y, expected(x, pack_weights[lid], ring))
            assert resp.timing.batch_rows >= 1
        await c.aclose()


async def test_masked_protocol_roundtrip_matches_plain(tiny_pack_dir: Path, pack_fp: str, tiny_pack: Any) -> None:
    """Gateway-side mask -> service -> unmask gives exactly the plaintext integer product."""
    lw = tiny_pack.load_linear("1.gate_up")
    ring = RING_K40_A12
    x = torch.randn(7, lw.in_features)
    xq, _sx = quantize_activations(x, ring)
    r = rand_x(7, lw.in_features, ring, seed=42)
    async with running_app(_settings(tiny_pack_dir)) as app:
        c = asgi_client(app)
        x_hat = mask_activations(xq, r, ring)
        resp = await c.compute(make_req(pack_fp, ring, [("1.gate_up", x_hat)]))
        y_hat = unpack_ring(resp.items[0].y_hat, (7, lw.out_features), ring)
        wr = expected(r, lw.w8, ring)  # what the factory would have stored
        yq = unmask_outputs(y_hat, wr, ring, in_features=lw.in_features)
        check_output_range(yq, ring, lw.in_features)
        assert torch.equal(yq, xq @ lw.w8.to(torch.int64).T)
        await c.aclose()


async def test_concurrent_clients_are_batched(tiny_pack_dir: Path, pack_fp: str,
                                              pack_weights: dict[str, torch.Tensor]) -> None:
    async with running_app(_settings(tiny_pack_dir, batch_window_ms=30.0)) as app:
        c = asgi_client(app)
        ring = RING_K40_A12
        xs = [rand_x(1, 64, ring, seed=i) for i in range(12)]
        resps = await asyncio.gather(*(c.compute(make_req(pack_fp, ring, [("0.o", x)], rid=f"r{i}"))
                                       for i, x in enumerate(xs)))
        for x, resp in zip(xs, resps, strict=True):
            assert torch.equal(unpack_ring(resp.items[0].y_hat, (1, 64), ring), expected(x, pack_weights["0.o"], ring))
        assert max(r.timing.batch_rows for r in resps) > 1
        stats = app.state.pgw.batcher.stats
        assert stats.batches < 12
        await c.aclose()


async def test_errors_on_the_wire(tiny_pack_dir: Path, pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, rings=["K40A12", "K48A20"], max_rows_per_item=16)) as app:
        c = asgi_client(app, max_retries=0)
        await c.connect()
        ring = RING_K40_A12
        with pytest.raises(FingerprintMismatchError):
            await c.compute(make_req("ab" * 32, ring, [("0.o", rand_x(1, 64, ring))]))
        with pytest.raises(ConfigError):  # ring not served
            await c.compute(make_req(pack_fp, RING_K32_A8, [("0.o", rand_x(1, 64, RING_K32_A8))]))
        h = http_client(app)
        # unknown linear
        req = ComputeRequest("bad1", pack_fp, "K40A12", (ComputeItem("7.o", 1, b"\0" * 64 * 5),))
        r = await h.post("/pgw/v1/compute", content=encode_message(req), headers={"content-type": CONTENT_TYPE})
        assert r.status_code == 400 and decode_message(r.content, ErrorResponse).code == "protocol"
        # wrong payload length
        req = ComputeRequest("bad2", pack_fp, "K40A12", (ComputeItem("0.o", 2, b"\0" * 64 * 5),))
        r = await h.post("/pgw/v1/compute", content=encode_message(req), headers={"content-type": CONTENT_TYPE})
        assert r.status_code == 400 and decode_message(r.content, ErrorResponse).code == "bad_shape"
        # too many rows per item (server limit)
        x = rand_x(17, 64, ring)
        req = ComputeRequest("bad3", pack_fp, "K40A12", (ComputeItem("0.o", 17, pack_ring(x, ring)),))
        r = await h.post("/pgw/v1/compute", content=encode_message(req), headers={"content-type": CONTENT_TYPE})
        assert r.status_code == 413
        # unsupported protocol version
        r = await h.post("/pgw/v1/compute", content=b"", headers={"content-type": CONTENT_TYPE, PROTOCOL_HEADER: "2"})
        assert r.status_code == 426
        # garbage body
        r = await h.post("/pgw/v1/compute", content=b"\xc1\xc1", headers={"content-type": CONTENT_TYPE})
        assert r.status_code == 400
        # models advertise limits
        info = (await c.models()).models[0]
        assert info.max_rows_per_item == 16 and set(info.rings) == {"K40A12", "K48A20"}
        await h.aclose()
        await c.aclose()


async def test_auth_tokens(tiny_pack_dir: Path, pack_fp: str, tokens_file: Path) -> None:
    s = _settings(tiny_pack_dir, allow_unauthenticated=False, tokens_file=tokens_file)
    async with running_app(s) as app:
        ring = RING_K40_A12
        req = make_req(pack_fp, ring, [("0.o", rand_x(2, 64, ring))])
        anon = asgi_client(app, max_retries=0)
        with pytest.raises(PGWError) as ei:
            await anon.compute(req)
        assert ei.value.code == ErrorCode.UNAUTHENTICATED
        with pytest.raises(PGWError):
            await anon.models()
        bad = asgi_client(app, token="wrong", max_retries=0)
        with pytest.raises(PGWError) as ei:
            await bad.compute(req)
        assert ei.value.code == ErrorCode.UNAUTHENTICATED
        a = asgi_client(app, token=TOKEN_A)
        b = asgi_client(app, token=TOKEN_B)
        await a.compute(req)
        await b.compute(req)
        await a.compute(req)
        h = http_client(app)
        text = (await h.get("/metrics")).text
        assert 'pgw_service_requests_total{code="ok",tenant="tenant-a"} 2.0' in text
        assert 'pgw_service_requests_total{code="ok",tenant="tenant-b"} 1.0' in text
        assert 'pgw_service_requests_total{code="unauthenticated",tenant="unknown"}' in text
        assert 'pgw_service_rows_total{linear_kind="o",tenant="tenant-a"} 4.0' in text
        assert 'pgw_service_bytes_in_total{tenant="tenant-a"}' in text
        assert TOKEN_A not in text
        for cl in (anon, bad, a, b):
            await cl.aclose()
        await h.aclose()


async def test_no_credentials_configured_is_config_error(tiny_pack_dir: Path) -> None:
    with pytest.raises(ConfigError):
        create_app(_settings(tiny_pack_dir, allow_unauthenticated=False))


async def test_trusted_client_cert_header(tiny_pack_dir: Path, pack_fp: str, tmp_path: Path) -> None:
    m = tmp_path / "mtls.txt"
    m.write_text("tenant-x:gw-x\n")
    s = _settings(tiny_pack_dir, allow_unauthenticated=False, mtls_tenants_file=m,
                  trusted_client_cert_header="X-Client-Cert-Subject")
    async with running_app(s) as app:
        h = http_client(app)
        ring = RING_K40_A12
        body = encode_message(make_req(pack_fp, ring, [("0.o", rand_x(1, 64, ring))]))
        hdr = {"content-type": CONTENT_TYPE}
        r = await h.post("/pgw/v1/compute", content=body, headers={**hdr, "X-Client-Cert-Subject": "CN=gw-x,O=Acme"})
        assert r.status_code == 200
        r = await h.post("/pgw/v1/compute", content=body, headers={**hdr, "X-Client-Cert-Subject": "CN=evil"})
        assert r.status_code == 403
        r = await h.post("/pgw/v1/compute", content=body, headers=hdr)
        assert r.status_code == 401
        text = (await h.get("/metrics")).text
        assert 'tenant="tenant-x"' in text
        await h.aclose()


async def test_health_ready_and_drain(tiny_pack_dir: Path, pack_fp: str) -> None:
    app = create_app(_settings(tiny_pack_dir))
    h = http_client(app)
    assert (await h.get("/healthz")).status_code == 200
    r = await h.get("/readyz")
    assert r.status_code == 503 and r.json()["loaded"] is False  # before startup
    async with app.router.lifespan_context(app):
        r = await h.get("/readyz")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ready"] and body["devices"] == {"cpu": True} and body["models"][0]["fp"] == pack_fp[:12]
        state = app.state.pgw
        state.begin_drain()
        assert (await h.get("/readyz")).status_code == 503
        assert (await h.get("/healthz")).status_code == 200
        c = asgi_client(app, max_retries=0)
        with pytest.raises(PGWError) as ei:
            await c.compute(make_req(pack_fp, RING_K40_A12, [("0.o", rand_x(1, 64, RING_K40_A12))]))
        assert ei.value.code == ErrorCode.UNAVAILABLE and ei.value.retryable
        await c.aclose()
    assert not app.state.pgw.batcher.running
    await h.aclose()


async def test_shutdown_waits_for_inflight(tiny_pack_dir: Path, pack_fp: str,
                                           pack_weights: dict[str, torch.Tensor]) -> None:
    app = create_app(_settings(tiny_pack_dir, batch_window_ms=300.0, drain_timeout_s=10))
    ring = RING_K40_A12
    x = rand_x(3, 64, ring)
    async with app.router.lifespan_context(app):
        c = asgi_client(app, max_retries=0)
        fut = asyncio.ensure_future(c.compute(make_req(pack_fp, ring, [("0.o", x)])))
        for _ in range(100):
            if app.state.pgw.handler.inflight:
                break
            await asyncio.sleep(0.005)
        assert app.state.pgw.handler.inflight == 1
    # lifespan exit drained: the in-flight request completed with the right answer
    resp = await fut
    assert torch.equal(unpack_ring(resp.items[0].y_hat, (3, 64), ring), expected(x, pack_weights["0.o"], ring))
    await c.aclose()


async def test_admission_inflight_limit(tiny_pack_dir: Path, pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, max_inflight_requests=2, batch_window_ms=200.0)) as app:
        c = asgi_client(app, max_retries=0)
        ring = RING_K40_A12
        reqs = [make_req(pack_fp, ring, [("0.o", rand_x(1, 64, ring, seed=i))], rid=f"q{i}") for i in range(4)]
        res = await asyncio.gather(*(c.compute(r) for r in reqs), return_exceptions=True)
        errs = [r for r in res if isinstance(r, PGWError)]
        assert len(errs) == 2 and all(e.code == ErrorCode.UNAVAILABLE for e in errs)
        h = http_client(app)
        body = encode_message(reqs[0])
        # retry-after header on 503
        fut = [asyncio.ensure_future(c.compute(r)) for r in reqs[:2]]
        await asyncio.sleep(0.05)
        r = await h.post("/pgw/v1/compute", content=body, headers={"content-type": CONTENT_TYPE})
        assert r.status_code == 503 and r.headers.get("retry-after") == "1"
        await asyncio.gather(*fut)
        await h.aclose()
        await c.aclose()


async def test_tenant_quotas(tiny_pack_dir: Path, pack_fp: str, tokens_file: Path) -> None:
    s = _settings(tiny_pack_dir, allow_unauthenticated=False, tokens_file=tokens_file,
                  tenant_rows_per_s=1.0, tenant_burst_rows=4)
    async with running_app(s) as app:
        ring = RING_K40_A12
        a = asgi_client(app, token=TOKEN_A, max_retries=0)
        b = asgi_client(app, token=TOKEN_B, max_retries=0)
        req = make_req(pack_fp, ring, [("0.o", rand_x(3, 64, ring))])
        await a.compute(req)
        with pytest.raises(PGWError) as ei:
            await a.compute(req)
        assert ei.value.code == ErrorCode.RATE_LIMITED
        await b.compute(req)  # other tenant unaffected
        await a.aclose()
        await b.aclose()
    s2 = _settings(tiny_pack_dir, tenant_max_inflight=1, batch_window_ms=100.0)
    async with running_app(s2) as app:
        c = asgi_client(app, max_retries=0)
        reqs = [make_req(pack_fp, ring, [("0.o", rand_x(1, 64, ring, seed=i))], rid=f"t{i}") for i in range(3)]
        res = await asyncio.gather(*(c.compute(r) for r in reqs), return_exceptions=True)
        codes = sorted(str(r.code) if isinstance(r, PGWError) else "ok" for r in res)
        assert codes == ["ok", "rate_limited", "rate_limited"]
        await c.aclose()


async def test_deadline_exceeded(tiny_pack_dir: Path, pack_fp: str) -> None:
    async with running_app(_settings(tiny_pack_dir, batch_window_ms=500.0)) as app:
        c = asgi_client(app, max_retries=0)
        ring = RING_K40_A12
        with pytest.raises(PGWError) as ei:
            await c.compute(make_req(pack_fp, ring, [("0.o", rand_x(1, 64, ring))], deadline_ms=50))
        assert ei.value.code == ErrorCode.DEADLINE_EXCEEDED
        await asyncio.sleep(0.6)
        assert app.state.pgw.batcher.stats.dropped_rows == 1  # abandoned rows were never computed
        assert app.state.pgw.handler.inflight == 0
        await c.aclose()


async def test_no_payload_bytes_in_logs(tiny_pack_dir: Path, pack_fp: str, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    ring = RING_K48_A20
    x = rand_x(2, 64, ring, seed=7)
    payload = pack_ring(x, ring)
    async with running_app(_settings(tiny_pack_dir)) as app:
        c = asgi_client(app, max_retries=0)
        resp = await c.compute(make_req(pack_fp, ring, [("0.o", x)]))
        with pytest.raises(PGWError):  # error path too
            await c.compute(make_req("ab" * 32, ring, [("0.o", x)]))
        await c.aclose()
    y_bytes = resp.items[0].y_hat
    needles = [payload[:12].hex(), y_bytes[:12].hex(), str(int(x[0, 0])), str(int(x[1, 5])), repr(payload[:12])]
    for rec in caplog.records:
        msg = rec.getMessage() + " " + str(rec.__dict__)
        for n in needles:
            assert n not in msg, f"payload leaked into log record {rec.name}"


async def test_metrics_disabled_and_not_on_main_port(tiny_pack_dir: Path) -> None:
    async with running_app(_settings(tiny_pack_dir, metrics_enabled=False)) as app:
        h = http_client(app)
        assert (await h.get("/metrics")).status_code == 404
        await h.aclose()
    async with running_app(_settings(tiny_pack_dir, metrics_on_main_port=False)) as app:
        h = http_client(app)
        assert (await h.get("/metrics")).status_code == 404
        await h.aclose()
        import httpx

        from pgw.service import create_metrics_app

        mh = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_metrics_app(app.state.pgw)), base_url="http://m")
        r = await mh.get("/metrics")
        assert r.status_code == 200 and "pgw_service_inflight" in r.text
        assert (await mh.get("/other")).status_code == 404
        await mh.aclose()


async def test_startup_fails_on_bad_pack(tmp_path: Path) -> None:
    s = ServiceSettings(packs=[tmp_path / "missing"], device="cpu", allow_unauthenticated=True)
    app = create_app(s)
    with pytest.raises(ConfigError):
        async with app.router.lifespan_context(app):
            pass
    assert app.state.pgw.load_error and "missing" in app.state.pgw.load_error


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PGW_SERVICE_PACKS", f'["{tmp_path}"]')
    monkeypatch.setenv("PGW_SERVICE_BATCH_WINDOW_MS", "5")
    monkeypatch.setenv("PGW_SERVICE_RINGS", '["k48a20"]')
    monkeypatch.setenv("PGW_SERVICE_DEVICES", '["cpu"]')
    monkeypatch.setenv("PGW_SERVICE_TLS__REQUIRE_CLIENT_CERT", "false")
    s = ServiceSettings()
    assert s.packs == [tmp_path] and s.batch_window_ms == 5.0 and s.rings == ["K48A20"]
    assert s.device_list == ["cpu"] and s.ring_params == [RingParams(48, 20)]
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ServiceSettings(rings=["nope"])
    with pytest.raises(ValidationError):
        ServiceSettings(tls={"require_client_cert": True})
    with pytest.raises(ValidationError):
        ServiceSettings(port=9000, metrics_port=9000)
