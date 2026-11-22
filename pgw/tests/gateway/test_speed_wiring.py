"""Chat-speed phase A wired into the gateway: every setting keeps the answers bit-identical to the plaintext
quantized reference (masked mode, tiny model), the provider sees nothing new, and sessions release their
deferred checks.

Settings covered: ``local_linears`` (attention / auto), ``lane`` (auto without a service lane, auto and
required with the service's real UDS lane, tls with the service's TCP+TLS lane and without one), ``spec_rows`` + ``spec_strict_passes``, ``float_threads``, and the
prefill ``pipeline_blocks`` the TripEngine asks for.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from pydantic import ValidationError

import pgw.gateway.config as gw_config
from pgw.core import DEFAULT_RING
from pgw.engine import SamplingParams
from pgw.gateway import ApiKeyStore, GatewayModelConfig, create_app
from tests.gateway.conftest import ADMIN, MODEL, InProcessService, auth, gateway, masked_settings, plain_settings

CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "one two one two one two one two"}],
        "temperature": 0, "max_tokens": 8, "ignore_eos": True, "logprobs": True, "top_logprobs": 3}


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(None)


@pytest.fixture
def key(keys: ApiKeyStore) -> str:
    return keys.create("app")[1]


@pytest.fixture(scope="module")
def reference(tiny_pack_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The plaintext quantized reference answer (``mode = plain``: exact integer linears, no service)."""
    keys = ApiKeyStore(None)
    k = keys.create("ref")[1]
    tmp = tmp_path_factory.mktemp("ref")

    async def run() -> dict[str, Any]:
        app = create_app(plain_settings(tiny_pack_dir, tmp), key_store=keys)
        async with gateway(app) as c:
            r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(k))
            assert r.status_code == 200, r.text
            return r.json()["choices"][0]  # type: ignore[no-any-return]

    return asyncio.run(run())


async def _masked(tiny_pack_dir: Path, tmp_path: Path, store: Any, transport: Any, keys: ApiKeyStore, key: str,
                  model_kw: dict[str, Any], req: dict[str, Any] | None = None,
                  settings_kw: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], Any]:
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw=model_kw, **(settings_kw or {})),
                     mask_store=store,
                     transports={MODEL: transport}, key_store=keys)
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json=req or CHAT, headers=auth(key))
        assert r.status_code == 200, r.text
        rt = app.state.pgw.runtime.get(MODEL)
        await rt.generator.wait_drained()  # strict mode: the dummy passes after the answer
        status = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
    return r.json()["choices"][0], status, rt


# ---- local linears ---------------------------------------------------------------------------------------
async def test_local_attention_serves_from_segments_without_qkv_o(tiny_pack_dir: Path, tmp_path: Path,
                                                                    gw_pack: Any, make_store: Any, keys: ApiKeyStore,
                                                                    key: str, reference: dict[str, Any]) -> None:
    """Factory ``skip_linear_kinds = qkv,o`` + gateway ``local_linears = attention``: bit-identical answers from
    segments that hold no attention masks; a restart with local linears off retires them (never uses them)."""
    from pgw.factory import MaskGenerator

    store = make_store(0)
    gen = MaskGenerator(gw_pack, DEFAULT_RING, skip_kinds={"qkv", "o"})
    gen.generate_into_store(store, 256, chunk_slots=16, factory_id="test")
    svc = InProcessService(gw_pack)
    out, _status, rt = await _masked(tiny_pack_dir, tmp_path, store, svc.transport, keys, key,
                                    {"local_linears": "attention"})
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    assert sorted(k.rsplit(".", 1)[1] for k in rt.mask_linears_needed()) == sorted(
        ["gate_up", "down"] * (len(rt.mask_linears_needed()) // 2))
    avail = store.available_slots(gw_pack.fingerprint, DEFAULT_RING)
    assert avail > 0
    # restarted with every linear remote: the partial segment is retired (written off), not served from
    app = create_app(masked_settings(tiny_pack_dir, tmp_path / "off"), mask_store=store,
                     transports={MODEL: svc.transport}, key_store=keys)
    async with gateway(app) as c:
        assert store.available_slots(gw_pack.fingerprint, DEFAULT_RING) == 0
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code != 200  # no usable masks: refused, never served with missing masks


@pytest.mark.parametrize("mode", ["attention", "auto"])
async def test_local_linears_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                           keys: ApiKeyStore, key: str, reference: dict[str, Any], mode: str) -> None:
    svc = InProcessService(gw_pack)
    kw: dict[str, Any] = {"local_linears": mode}
    if mode == "auto":
        kw["local_rtt_ms"] = 10_000.0  # a slow link: local qkv + o always win
    out, status, _rt = await _masked(tiny_pack_dir, tmp_path, make_store(200), svc.transport, keys, key, kw)
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    sp = status["speed"]
    assert sp["local_linears"]["kinds"] == ["o", "qkv"]
    assert status["backend"]["local_calls"] > 0
    # the provider never saw an attention linear (half the round trips per layer)
    seen = {lid.rsplit(".", 1)[1] for lid, _ in svc.received}
    assert seen == {"gate_up", "down"}


async def test_local_linears_auto_stays_remote_on_a_fast_link(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                              make_store: Any, keys: ApiKeyStore, key: str,
                                                              reference: dict[str, Any]) -> None:
    svc = InProcessService(gw_pack)
    out, status, _ = await _masked(tiny_pack_dir, tmp_path, make_store(200), svc.transport, keys, key,
                                   {"local_linears": "auto", "local_rtt_ms": 1e-6})
    assert out["logprobs"] == reference["logprobs"]
    assert status["speed"]["local_linears"]["kinds"] == [] and status["backend"]["local_calls"] == 0
    assert {lid.rsplit(".", 1)[1] for lid, _ in svc.received} == {"qkv", "o", "gate_up", "down"}


async def test_local_linears_auto_measures_rtt_without_masks(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                             make_store: Any, keys: ApiKeyStore,
                                                             key: str) -> None:
    """Without local_rtt_ms the start-up measurement uses mask-free dummy calls: no mask slot is used."""
    store = make_store(200)
    svc = InProcessService(gw_pack)
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw={"local_linears": "auto"}), mask_store=store,
                     transports={MODEL: svc.transport}, key_store=keys)
    before = sum(i.slots_available for i in store.inventory(gw_pack.fingerprint, DEFAULT_RING))
    async with gateway(app) as c:
        st = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
    assert st["speed"]["measured_rtt_ms"] is not None and st["speed"]["measured_rtt_ms"] > 0
    assert st["speed"]["rtt_path"].split(" ")[0] in ("loopback", "private", "public")  # which path was measured
    assert st["speed"]["local_linears"]["mode"] == "auto" and "rtt_s" in st["speed"]["local_linears"]
    assert sum(i.slots_available for i in store.inventory(gw_pack.fingerprint, DEFAULT_RING)) == before
    # what the provider saw at start-up: 1-row first-layer qkv calls of uniform random ring elements only
    assert svc.received and all(lid == "0.qkv" and x.shape[0] == 1 for lid, x in svc.received)
    rows = torch.cat([x for _, x in svc.received])
    assert int(rows.min()) >= 0 and int(rows.max()) < DEFAULT_RING.modulus
    assert float((rows.abs() < 4096).double().mean()) < 0.01  # not plaintext-sized activations


# ---- float threads and the prefill pipeline ---------------------------------------------------------------
async def test_float_threads_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                           keys: ApiKeyStore, key: str, reference: dict[str, Any]) -> None:
    svc = InProcessService(gw_pack)
    out, status, rt = await _masked(tiny_pack_dir, tmp_path, make_store(200), svc.transport, keys, key,
                                    {"float_threads": 1})
    assert out["logprobs"] == reference["logprobs"]
    if torch.get_num_threads() != 1:  # already 1 thread: nothing to scope
        assert rt.model.float_threads == 1 and status["speed"]["float_threads"] == 1


@pytest.mark.parametrize("kw, env, want", [({}, None, 1), ({"float_threads": None}, None, None),
                                           ({}, "2", 2), ({"float_threads": 1}, "2", 1)])
async def test_float_threads_default_on(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                        keys: ApiKeyStore, key: str, reference: dict[str, Any],
                                        monkeypatch: pytest.MonkeyPatch, kw: dict[str, Any], env: str | None,
                                        want: int | None) -> None:
    """Default 1 (self-checked); explicit null keeps the process count; PGW_FLOAT_THREADS applies when the model
    entry does not set it; an explicit setting wins over the variable. Answers stay bit-identical."""
    if env is not None:
        monkeypatch.setenv("PGW_FLOAT_THREADS", env)
    else:
        monkeypatch.delenv("PGW_FLOAT_THREADS", raising=False)
    if torch.get_num_threads() <= 2:
        pytest.skip("needs a process thread count above 2 to tell the settings apart")
    svc = InProcessService(gw_pack)
    out, status, rt = await _masked(tiny_pack_dir, tmp_path, make_store(200), svc.transport, keys, key, kw)
    assert out["logprobs"] == reference["logprobs"] and out["message"] == reference["message"]
    assert rt.model.float_threads == want and status["speed"]["float_threads"] == want


async def test_trip_engine_prefill_pipeline_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                          make_store: Any, keys: ApiKeyStore, key: str,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """The TripEngine's 4-block prefill pipeline gives the same bits as the default 2 blocks."""
    long_chat = {**CHAT, "messages": [{"role": "user", "content": "alpha beta gamma delta " * 4}], "max_tokens": 4}
    base, _, rt0 = await _masked(tiny_pack_dir, tmp_path, make_store(400, name="s0"),
                                 InProcessService(gw_pack).transport, keys, key, {}, long_chat)
    assert rt0.model.pipeline_blocks == 2
    monkeypatch.setenv("PGW_TRIP_ENGINE", "1")
    out, status, rt1 = await _masked(tiny_pack_dir, tmp_path, make_store(400, name="s1"),
                                     InProcessService(gw_pack).transport, keys, key, {}, long_chat)
    if rt1.backend.trip_engine:  # needs the native trip kernels
        assert rt1.model.pipeline_blocks == 4 and status["speed"]["trip_engine"]
    assert out == base


# ---- strict speculation ----------------------------------------------------------------------------------
@pytest.mark.parametrize("source", ["lookup", "draft"])
async def test_strict_speculation_bit_identical_and_pass_count(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                               make_store: Any, keys: ApiKeyStore, key: str,
                                                               reference: dict[str, Any], source: str) -> None:
    src: dict[str, Any] = ({"spec_lookup": 2, "spec_lookup_min_n": 2} if source == "lookup"
                           else {"draft_pack_path": str(tiny_pack_dir)})
    svc0 = InProcessService(gw_pack)
    _, _, _ = await _masked(tiny_pack_dir, tmp_path, make_store(400, name="plain"), svc0.transport, keys, key, {})
    svc = InProcessService(gw_pack)
    store = make_store(400, name="strict")
    before = sum(i.slots_available for i in store.inventory(gw_pack.fingerprint, DEFAULT_RING))
    out, _status, rt = await _masked(tiny_pack_dir, tmp_path, store, svc.transport, keys, key,
                                    {**src, "spec_rows": 3, "spec_strict_passes": True})
    # reserved up front: every decode pass (dummy passes too) takes 3 masked rows, whatever the text
    used = before - sum(i.slots_available for i in store.inventory(gw_pack.fingerprint, DEFAULT_RING))
    prompt = len(rt.encode_chat(CHAT["messages"]))
    assert used == prompt + 8 + 7 * 2
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    assert rt.generator.spec_rows == 3 and rt.generator.spec_strict
    # the provider sees exactly as many passes as without speculation, and every decode pass has 3 rows
    passes0 = [x.shape[0] for lid, x in svc0.received if lid == "0.qkv"]
    passes = [x.shape[0] for lid, x in svc.received if lid == "0.qkv"]
    assert passes == [3 if rows == 1 else rows for rows in passes0]
    if source == "draft":  # the tiny model drafts for itself: every draft is accepted
        assert rt.generator.spec_stats["spec_accepted"] > 0
        assert rt.generator.spec_stats["dummy_passes"] > 0


@pytest.mark.parametrize("draft", [False, True])
async def test_strict_speculation_with_local_linears_is_exact(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                              make_store: Any, keys: ApiKeyStore, key: str,
                                                              reference: dict[str, Any], draft: bool) -> None:
    """The Modal demo's ``--spec-rows`` preset (``pgw_modal_boot.spec_options``) on top of its exact local
    linears: same bits; the provider sees only gate_up/down items, every decode pass with spec_rows rows, and as
    many passes as plain decoding (the local-linear run without speculation)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy" / "modal"))
    import pgw_modal_boot as boot

    svc0 = InProcessService(gw_pack)
    await _masked(tiny_pack_dir, tmp_path, make_store(400, name="plain"), svc0.transport, keys, key,
                  {"local_linears": "attention"})
    svc = InProcessService(gw_pack)
    spec = boot.spec_options(4, str(tiny_pack_dir) if draft else None)
    out, status, rt = await _masked(tiny_pack_dir, tmp_path, make_store(400, name="spec"), svc.transport, keys, key,
                                    {"local_linears": "attention", **spec})
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    assert status["speed"]["spec_rows"] == 4 and status["speed"]["local_linears"]["kinds"] == ["o", "qkv"]
    assert {lid.split(".")[-1] for lid, _ in svc.received} == {"gate_up", "down"}
    passes0 = [x.shape[0] for lid, x in svc0.received if lid == "0.gate_up"]
    passes = [x.shape[0] for lid, x in svc.received if lid == "0.gate_up"]
    assert passes == [4 if rows == 1 else rows for rows in passes0]
    if draft:  # the tiny model drafts for itself: drafts are accepted, and dummy passes fill the schedule
        assert rt.generator.spec_stats["spec_accepted"] > 0 and rt.generator.spec_stats["dummy_passes"] > 0


async def test_strict_speculation_orders_a_conversation(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                        make_store: Any, keys: ApiKeyStore, key: str) -> None:
    """Requests of one API key carry one conversation key; the engine gets it (dummy passes first)."""
    svc = InProcessService(gw_pack)
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw={
        "spec_lookup": 2, "spec_lookup_min_n": 2, "spec_rows": 2, "spec_strict_passes": True}),
        mask_store=make_store(600), transports={MODEL: svc.transport}, key_store=keys)
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        seen: list[Any] = []
        orig = rt.generator.astream

        def spy(*a: Any, **kw: Any) -> Any:
            seen.append(kw.get("conversation"))
            return orig(*a, **kw)

        rt.generator.astream = spy
        rs = await asyncio.gather(*[c.post("/v1/chat/completions", json={**CHAT, "user": u}, headers=auth(key))
                                    for u in ("u1", "u1", "u2")])
        assert all(r.status_code == 200 for r in rs)
        await rt.generator.wait_drained()
    assert len(seen) == 3 and sorted(seen.count(s) for s in set(seen)) == [1, 2]
    assert all(s is not None and s.startswith("key:") for s in seen)
    assert rs[0].json()["choices"][0]["logprobs"] == rs[2].json()["choices"][0]["logprobs"]


async def test_strict_speculation_seeded_sampling_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                               make_store: Any, keys: ApiKeyStore, key: str) -> None:
    req = {**CHAT, "temperature": 0.9, "top_p": 0.95, "seed": 1234}
    base, _, _ = await _masked(tiny_pack_dir, tmp_path, make_store(400, name="b"), InProcessService(gw_pack).transport,
                               keys, key, {}, req)
    out, _, _ = await _masked(tiny_pack_dir, tmp_path, make_store(400, name="s"), InProcessService(gw_pack).transport,
                              keys, key, {"draft_pack_path": str(tiny_pack_dir), "spec_rows": 4,
                                          "spec_strict_passes": True}, req)
    assert out == base


# ---- the trip lane ---------------------------------------------------------------------------------------
async def test_lane_auto_without_a_service_lane_is_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                                 make_store: Any, keys: ApiKeyStore, key: str,
                                                                 reference: dict[str, Any]) -> None:
    """``auto`` against a service without a lane: the stream channel; the TripEngine and deferred checks on."""
    svc = InProcessService(gw_pack)
    out, status, rt = await _masked(tiny_pack_dir, tmp_path, make_store(200), svc.transport, keys, key,
                                    {"lane": "auto"})
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    sp = status["speed"]
    assert sp["lane"] == "auto" and sp["lane_state"] == "unavailable" and sp["lane_stats"]["trips"] == 0
    assert sp["trip_engine"] == bool(rt.backend.trip_engine) and sp["sync_decode"]
    assert status["backend"]["integrity_failures"] == 0
    assert rt.backend.pending_checks == 0


async def test_lane_required_without_a_service_lane_does_not_start(tiny_pack_dir: Path, tmp_path: Path,
                                                                   gw_pack: Any, make_store: Any, keys: ApiKeyStore,
                                                                   key: str) -> None:
    svc = InProcessService(gw_pack)
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw={"lane": "required"}),
                     mask_store=make_store(50), transports={MODEL: svc.transport}, key_store=keys)
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        assert rt.state == "connecting" and "trip lane" in (rt.last_error or "")
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 503
        assert svc.handler.requests == 0  # nothing was computed


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="pgwgl"))  # short: AF_UNIX paths are limited to ~108 bytes
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _lane_service(tiny_pack_dir: Path, sock_dir: Path) -> Any:
    from pgw.service import ServiceSettings
    from pgw.service import create_app as service_app

    return service_app(ServiceSettings(packs=[tiny_pack_dir], device="cpu", verify_packs="fingerprint", warmup=False,
                                       rings=[DEFAULT_RING.name], lane="uds", lane_socket=sock_dir / "lane.sock",
                                       allow_unauthenticated=True))


@pytest.mark.parametrize("lane", ["auto", "required"])
async def test_lane_over_the_service_uds_lane_is_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                               make_store: Any, keys: ApiKeyStore, key: str,
                                                               reference: dict[str, Any], sock_dir: Path,
                                                               lane: str) -> None:
    from pgw.service.lane import lane_native_available

    if not lane_native_available():
        pytest.skip("pgw_fast without the service trip lane")
    svc_app = _lane_service(tiny_pack_dir, sock_dir)
    async with svc_app.router.lifespan_context(svc_app):
        out, status, _rt = await _masked(tiny_pack_dir, tmp_path, make_store(200), httpx.ASGITransport(app=svc_app),
                                        keys, key, {"lane": lane, "local_linears": "attention", "float_threads": 1})
        lane_requests = svc_app.state.pgw.handler.lane_requests
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    sp = status["speed"]
    assert sp["lane_state"] == "open" and sp["lane_stats"]["trips"] > 0
    assert lane_requests == sp["lane_stats"]["trips"]
    assert status["backend"]["integrity_failures"] == 0 and status["backend"]["freivalds_checked"] > 0
    # decode trips took the lane; prefill (more rows than the lane serves) the stream channel
    assert set(sp["lane_fallbacks"]) <= {"ineligible:rows", "ineligible:items", "off", "busy:pool"}


@pytest.fixture(scope="module")
def lane_pki(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from tests.service.test_lane_tls import _ca, _leaf, _pem

    d = tmp_path_factory.mktemp("gw-lane-pki")
    ca = _ca("svc-ca")
    cert, key = _leaf(ca, "127.0.0.1", server=True)
    out = {"ca": d / "ca.pem", "cert": d / "server.pem", "key": d / "server.key"}
    out["ca"].write_bytes(_pem(ca[0]))
    out["cert"].write_bytes(cert)
    out["key"].write_bytes(key)
    return out


def _tls_lane_service(tiny_pack_dir: Path, pki: dict[str, Path], lane: str = "tcp") -> Any:
    from pgw.core.settings import TLSConfig
    from pgw.service import ServiceSettings
    from pgw.service import create_app as service_app

    kw: dict[str, Any] = {"lane": lane, "lane_port": 0} if lane == "tcp" else {"lane": "off"}
    return service_app(ServiceSettings(packs=[tiny_pack_dir], device="cpu", verify_packs="fingerprint", warmup=False,
                                       rings=[DEFAULT_RING.name], host="127.0.0.1", allow_unauthenticated=True,
                                       tls=TLSConfig(cert_file=pki["cert"], key_file=pki["key"]), **kw))


@pytest.mark.parametrize("service_lane", ["tcp", "off"])
async def test_lane_tls_over_the_service_tcp_lane_is_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                                   make_store: Any, keys: ApiKeyStore, key: str,
                                                                   reference: dict[str, Any],
                                                                   lane_pki: dict[str, Path], service_lane: str) -> None:
    """``lane = "tls"``: decode trips over the service's TCP+TLS lane (native channel, server verified against the
    pinned CA); against a service without it, every call takes the stream path. Both bit-identical."""
    from pgw.service.lane import lane_native_available
    from pgw.wire.lane import native_channel_class

    native = native_channel_class()
    if not lane_native_available() or native is None or not hasattr(native, "connect_service_tls"):
        pytest.skip("pgw_fast without the TCP+TLS trip lane")
    svc_app = _tls_lane_service(tiny_pack_dir, lane_pki, service_lane)
    async with svc_app.router.lifespan_context(svc_app):
        out, status, _rt = await _masked(
            tiny_pack_dir, tmp_path, make_store(200), httpx.ASGITransport(app=svc_app), keys, key,
            {"lane": "tls", "local_linears": "attention", "float_threads": 1,
             "service_url": "https://127.0.0.1:18443", "service_ca_file": lane_pki["ca"]},
            settings_kw=_lane_egress(svc_app))
        lane_requests = svc_app.state.pgw.handler.lane_requests
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    sp = status["speed"]
    assert status["backend"]["integrity_failures"] == 0 and status["backend"]["freivalds_checked"] > 0
    if service_lane == "tcp":
        assert sp["lane_state"] == "open" and sp["lane_stats"]["trips"] > 0 and sp["lane_stats"]["native"]
        assert lane_requests == sp["lane_stats"]["trips"]
        assert set(sp["lane_fallbacks"]) <= {"ineligible:rows", "ineligible:items", "off", "busy:pool"}
    else:
        assert sp["lane_state"] == "stream" and sp["lane_stats"]["trips"] == 0 and lane_requests == 0


async def test_lane_tls_seeded_sampling_bit_identical(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                      make_store: Any, keys: ApiKeyStore, key: str,
                                                      lane_pki: dict[str, Path]) -> None:
    from pgw.wire.lane import native_channel_class

    native = native_channel_class()
    if native is None or not hasattr(native, "connect_service_tls"):
        pytest.skip("pgw_fast without the TCP+TLS trip lane")
    req = {**CHAT, "temperature": 0.9, "top_p": 0.95, "seed": 1234}
    base, _, _ = await _masked(tiny_pack_dir, tmp_path, make_store(400, name="b"), InProcessService(gw_pack).transport,
                               keys, key, {}, req)
    svc_app = _tls_lane_service(tiny_pack_dir, lane_pki)
    async with svc_app.router.lifespan_context(svc_app):
        out, status, _ = await _masked(
            tiny_pack_dir, tmp_path, make_store(400, name="t"), httpx.ASGITransport(app=svc_app), keys, key,
            {"lane": "tls", "service_url": "https://127.0.0.1:18443", "service_ca_file": lane_pki["ca"]}, req,
            settings_kw=_lane_egress(svc_app))
    assert status["speed"]["lane_stats"]["trips"] > 0
    assert out == base


async def test_lane_tls_port_off_the_egress_allowlist_is_not_used(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                                  make_store: Any, keys: ApiKeyStore, key: str,
                                                                  reference: dict[str, Any],
                                                                  lane_pki: dict[str, Path]) -> None:
    """The gateway connects to a TCP lane port only when it is on its egress allowlist for the service host
    (the service names the port): otherwise no lane connection is made and every call takes the stream path."""
    from pgw.wire.lane import native_channel_class

    native = native_channel_class()
    if native is None or not hasattr(native, "connect_service_tls"):
        pytest.skip("pgw_fast without the TCP+TLS trip lane")
    svc_app = _tls_lane_service(tiny_pack_dir, lane_pki)
    async with svc_app.router.lifespan_context(svc_app):
        out, status, _ = await _masked(
            tiny_pack_dir, tmp_path, make_store(200), httpx.ASGITransport(app=svc_app), keys, key,
            {"lane": "tls", "local_linears": "attention", "float_threads": 1,
             "service_url": "https://127.0.0.1:18443", "service_ca_file": lane_pki["ca"]})
        connections = svc_app.state.pgw.lane.svc.connections()
        unspent = svc_app.state.pgw.lane.svc.pending_tickets()
        lane_requests = svc_app.state.pgw.handler.lane_requests
    assert out["message"] == reference["message"] and out["logprobs"] == reference["logprobs"]
    sp = status["speed"]
    assert sp["lane_state"] == "stream" and sp["lane_stats"]["trips"] == 0
    assert lane_requests == 0 and connections == 0 and unspent >= 1  # tickets issued, none spent


def _lane_egress(svc_app: Any) -> dict[str, Any]:
    """Gateway settings allowing the service's TCP lane port (egress allowlist, as an operator would)."""
    lane = svc_app.state.pgw.lane
    return {"egress_allowlist": [f"https://127.0.0.1:{lane.svc.port}"]} if lane is not None else {}


def test_native_service_status_map_matches_python() -> None:
    from pgw.wire import service_lane as bridge

    fast = pytest.importorskip("pgw_fast")
    if not hasattr(fast, "lane_map_service_status"):
        pytest.skip("pgw_fast without the service-lane client")
    for code in range(0, 40):
        assert fast.lane_map_service_status(code) == bridge.map_service_status(code), code


def test_service_lane_bridge_mirrors_the_service_wire_format() -> None:
    from pgw.service import lane as svc_lane
    from pgw.wire import service_lane as bridge

    assert bridge.SERVICE_LANE_OPEN_PATH == svc_lane.LANE_OPEN_PATH
    assert bridge._REQUEST.format == svc_lane.REQUEST_HEADER.format
    assert bridge._REPLY.format == svc_lane.REPLY_HEADER.format
    assert bridge._REQUEST_MAGIC == svc_lane.REQUEST_MAGIC and bridge._REPLY_MAGIC == svc_lane.REPLY_MAGIC
    assert bridge._TICKET_LEN == svc_lane.TICKET_LEN and bridge._VERSION == svc_lane.LANE_VERSION
    from pgw.wire.lane import CLOSING_STATUSES, Status

    for st in svc_lane.LaneStatus:  # every service status maps; the ones the service closes after close here too
        mapped = bridge.map_service_status(int(st))
        if st == svc_lane.LaneStatus.OK:
            assert mapped == Status.OK
        elif st.fatal:
            assert mapped in CLOSING_STATUSES
        else:
            assert mapped not in CLOSING_STATUSES and mapped != Status.OK


# ---- session release and integrity failures ----------------------------------------------------------------
async def test_deferred_check_failure_discards_the_session(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                           make_store: Any, keys: ApiKeyStore, key: str) -> None:
    """A wrong gate_up result (checked deferred with the TripEngine) fails the request, trips the breaker,
    releases the session's checks and frees its KV cache; nothing of it is reused."""
    def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
        if lid.endswith("gate_up"):
            y = y.clone()
            y[0, 1] += 1
        return y

    svc = InProcessService(gw_pack, tamper=tamper)
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw={"lane": "auto"}), mask_store=make_store(200),
                     transports={MODEL: svc.transport}, key_store=keys)
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        rt.backend.defer_min_out = 0  # defer every check the TripEngine can (also qkv / o)
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 502 and r.json()["error"]["code"] == "integrity_check_failed"
        await rt.generator.wait_drained()
        assert rt.breaker.status()["open"]
        assert rt.backend.pending_checks == 0
        assert not rt.generator._active and not rt._sessions
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 503  # fail closed until an admin reset


async def test_session_end_releases_deferred_checks(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                    make_store: Any, keys: ApiKeyStore, key: str) -> None:
    svc = InProcessService(gw_pack)
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw={"lane": "auto"}), mask_store=make_store(300),
                     transports={MODEL: svc.transport}, key_store=keys)
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        released: list[Any] = []
        orig = rt.backend.release_session_checks

        def spy(session: Any) -> None:
            released.append(session)
            orig(session)

        rt.backend.release_session_checks = spy
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 200
        # a cancelled stream releases too
        async with c.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True, "max_tokens": 30},
                            headers=auth(key)) as rs:
            async for _ in rs.aiter_lines():
                break
        await asyncio.sleep(0.05)
        await rt.generator.wait_drained()
        # a session admitted but never handed to the engine releases as well
        sess = await rt.open_session([1, 2, 3], SamplingParams(max_tokens=2))
        await sess.close()
    assert len(released) == 3 and len({id(s) for s in released}) == 3
    assert rt.backend.pending_checks == 0


# ---- configuration -----------------------------------------------------------------------------------------
def _cfg(**kw: Any) -> GatewayModelConfig:
    base: dict[str, Any] = {"name": "m", "pack_path": Path("/nonexistent/pack"), "fingerprint": "ab" * 32,
                            "service_url": "https://compute.example.net:8443"}
    return GatewayModelConfig(**{**base, **kw})


def test_phase_a_settings_are_no_longer_pending() -> None:
    assert set(gw_config.PENDING_SPEED_SETTINGS) == {"float_ops", "head_mode", "schedule", "prefix_cache"}
    c = _cfg(lane="auto", lane_uds_dir=Path("/run/pgw"), lane_pool_wait_s=0.02, local_linears="auto",
             local_rtt_ms=12.5, spec_lookup=3, spec_rows=2, spec_strict_passes=True, float_threads=1)
    assert c.lane == "auto" and c.spec_rows == 2 and c.float_threads == 1
    for name, value in (("float_ops", "native"), ("head_mode", "provider_int8"), ("schedule", "fused2"),
                        ("prefix_cache", "conversation")):
        with pytest.raises(ValidationError, match="not available in this build"):
            _cfg(**{name: value})


@pytest.mark.parametrize(("kw", "match"), [
    ({"spec_rows": 3, "spec_lookup": 3}, "needs spec_strict_passes"),
    ({"spec_strict_passes": True}, "spec_strict_passes needs spec_rows"),
    ({"spec_rows": 3, "spec_strict_passes": True}, "needs a draft source"),
    ({"spec_rows": 3, "spec_strict_passes": True, "spec_lookup": 3, "spec_tokens": 0}, "spec_tokens > 0"),
    ({"spec_rows": 9, "spec_strict_passes": True, "spec_lookup": 3}, "less than or equal to 8"),
    ({"lane": "tcp"}, "lane"),
    ({"lane": "auto", "mode": "plain", "service_url": None}, "lane needs masked mode"),
    ({"local_linears": "always"}, "local_linears"),
    ({"local_rtt_ms": 0}, "greater than 0"),
    ({"lane_pool_wait_s": 2.0}, "less than or equal to 1"),
    ({"float_threads": 0}, "greater than or equal to 1"),
    # speculation verifies on the CPU float path only (a GPU target's verify passes are not bit-identical)
    ({"spec_rows": 4, "spec_strict_passes": True, "spec_lookup": 3, "device": "cuda"}, "needs device='cpu'"),
    ({"spec_lookup": 3, "device": "cuda:0"}, "needs device='cpu'"),
])
def test_speed_setting_validation(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        _cfg(**kw)


def test_network_path_labels() -> None:
    from pgw.gateway.config import network_path

    assert network_path("https://127.0.0.1:8443") == "loopback"
    assert network_path("http://localhost:1") == "loopback"
    assert network_path("https://[::1]:8443") == "loopback"
    assert network_path("https://[fdaa:0:1::5]:8443") == "private fdaa:0:1::5"
    assert network_path("https://10.1.2.3") == "private 10.1.2.3"
    assert network_path("https://r500.modal.host:4000") == "public r500.modal.host"
    assert network_path(None) == "unknown" and network_path("ftp://x") == "unknown"


def test_speculation_target_on_cpu_draft_anywhere() -> None:
    """The draft may sit on a GPU (it only proposes); without speculation a GPU target stays allowed."""
    c = _cfg(spec_rows=4, spec_strict_passes=True, spec_lookup=3, draft_device="cuda")
    assert c.speculates and c.device == "cpu"
    assert not _cfg(device="cuda").speculates
    assert not _cfg(device="cuda", spec_lookup=3, spec_tokens=0).speculates
