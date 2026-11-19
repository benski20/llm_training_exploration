"""Masked production path end to end: gateway -> real encrypted mask store -> pgw.wire HTTP -> in-process
compute service (exact ``W8 x_hat mod 2^K``). Covers exactness, single-use mask accounting, Freivalds
fail-closed behaviour with the circuit breaker, mask exhaustion, startup checks, batching and cancellation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import DEFAULT_RING, FingerprintMismatchError, RingParams
from pgw.gateway import ApiKeyStore, EgressGuardTransport, create_app
from pgw.gateway.runtime import InferenceRuntime
from tests.gateway.conftest import (
    ADMIN,
    MODEL,
    InProcessService,
    auth,
    gateway,
    masked_settings,
    parse_sse,
    plain_settings,
)

CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello there"}], "temperature": 0,
        "max_tokens": 6, "ignore_eos": True}


def available(store: Any, pack: Any, ring: str = DEFAULT_RING.name) -> int:
    return sum(i.slots_available for i in store.inventory(pack.fingerprint, ring))


def masked_app(tiny_pack_dir: Path, tmp_path: Path, store: Any, svc: InProcessService, keys: ApiKeyStore,
               **kw: Any) -> Any:
    return create_app(masked_settings(tiny_pack_dir, tmp_path, **kw), mask_store=store,
                      transports={MODEL: svc.transport}, key_store=keys)


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(None)


@pytest.fixture
def key(keys: ApiKeyStore) -> str:
    return keys.create("app")[1]


@pytest.mark.parametrize("ring", ["K40A12", "K48A20", "K32A12C27", "K32A12C18"])
async def test_masked_bit_identical_to_plain(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                             keys: ApiKeyStore, key: str, ring: str) -> None:
    """Masked logprobs (hence logits) equal the local exact-integer reference bit for bit.

    The capped rings use per-token adaptive activation bits; C18 is tight enough on the tiny model
    that many tokens lose bits, which must not break exactness between the two paths.
    """
    store = make_store(200, ring=ring)
    svc = InProcessService(gw_pack, rings=(ring,))
    req = {**CHAT, "logprobs": True, "top_logprobs": 5, "max_tokens": 8}
    masked = masked_app(tiny_pack_dir, tmp_path, store, svc, keys, model_kw={"ring": ring})
    async with gateway(masked) as c:
        rm = await c.post("/v1/chat/completions", json=req, headers=auth(key))
        assert rm.status_code == 200, rm.text
    plain = create_app(plain_settings(tiny_pack_dir, tmp_path, model_kw={"ring": ring}), key_store=keys)
    async with gateway(plain) as c:
        rp = await c.post("/v1/chat/completions", json=req, headers=auth(key))
    a, b = rm.json()["choices"][0], rp.json()["choices"][0]
    assert a["message"] == b["message"]
    assert a["logprobs"] == b["logprobs"]  # exact float equality, including top-5 alternatives
    assert rm.json()["system_fingerprint"] == rp.json()["system_fingerprint"] == gw_pack.fingerprint[:12]


async def test_single_use_mask_accounting_and_privacy(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                      make_store: Any, keys: ApiKeyStore, key: str) -> None:
    store = make_store(300)
    svc = InProcessService(gw_pack)
    before = available(store, gw_pack)
    app = masked_app(tiny_pack_dir, tmp_path, store, svc, keys)
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 200
        u = r.json()["usage"]
        # reserved = prompt + max_tokens; the unused slot is burned, never returned
        assert before - available(store, gw_pack) == u["prompt_tokens"] + 6
        # early stop: whole reservation is consumed as well
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 50, "ignore_eos": False,
                                                       "stop_token_ids": list(range(260))}, headers=auth(key))
        assert r.status_code == 200, r.text
        assert r.json()["usage"]["completion_tokens"] == 1
        assert before - available(store, gw_pack) == 2 * u["prompt_tokens"] + 6 + 50
        masks = (await c.get("/admin/masks", headers=ADMIN)).json()
        assert masks["models"][MODEL]["available_slots"] == available(store, gw_pack)
        assert masks["store"]["generation"] >= 2
        status = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
        be = status["backend"]
        assert be["freivalds_checked"] == be["round_trips"] > 0 and be["freivalds_skipped"] == 0
        assert status["masks_available"] == available(store, gw_pack)
        metrics = (await c.get("/metrics")).text
        assert "pgw_masks_available_slots" in metrics and "pgw_gateway_round_trip_seconds_count" in metrics
    # what the service saw: uniform ring noise, no duplicate rows (no mask reuse)
    ring = DEFAULT_RING
    rows = torch.cat([x.reshape(-1, x.shape[-1]) for lid, x in svc.received if lid.endswith(".qkv")])
    assert rows.min() >= 0 and rows.max() < ring.modulus
    assert len({tuple(r.tolist()) for r in rows}) == rows.shape[0]
    top = (rows >> (ring.k - 8)).flatten()
    hist = torch.bincount(top, minlength=256).double()
    exp = top.numel() / 256
    chi2 = float(((hist - exp) ** 2 / exp).sum())
    assert chi2 < 400  # 255 dof; plaintext-like data would be orders of magnitude off
    small = (rows.abs() < 4096).double().mean()  # quantized activations are |x| <= 2047
    assert float(small) < 0.01
    audit = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    inf = [a for a in audit if a["event"] == "inference"]
    assert inf[0]["masks_reserved"] == u["prompt_tokens"] + 6 and inf[0]["masks_consumed"] == u["prompt_tokens"] + 5
    assert all("Hello" not in json.dumps(a) for a in audit)


async def test_integrity_failure_fails_closed_and_trips_breaker(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                                make_store: Any, keys: ApiKeyStore,
                                                                key: str) -> None:
    store = make_store(400)
    svc = InProcessService(gw_pack)
    app = masked_app(tiny_pack_dir, tmp_path, store, svc, keys)
    async with gateway(app) as c:
        good = (await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))).json()

        def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
            if lid == "1.down":
                y = y.clone()
                y[0, 3] += 1  # a single wrong ring element
            return y

        svc.set_tamper(tamper)
        before = available(store, gw_pack)
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 502, r.text
        assert r.json()["error"]["code"] == "integrity_check_failed"
        assert available(store, gw_pack) < before  # masks of the failed request are burned, never reused
        # breaker is open: refused before any mask is reserved or anything is sent
        svc.set_tamper(None)
        n_req, before = svc.handler.requests, available(store, gw_pack)
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 503 and r.json()["error"]["code"] == "unavailable"
        # default: admin reset only (no cool-down), so no Retry-After hint
        assert "integrity" in r.json()["error"]["message"] and "Retry-After" not in r.headers
        assert svc.handler.requests == n_req and available(store, gw_pack) == before
        rs = await c.post("/v1/chat/completions", json={**CHAT, "stream": True}, headers=auth(key))
        assert rs.status_code == 503
        ready = await c.get("/readyz?models=all")
        assert ready.status_code == 503 and ready.json()["models"][MODEL]["reason"] == "integrity breaker open"
        # the process stays in the load balancer: other models / typed errors keep flowing
        assert (await c.get("/readyz")).status_code == 200
        m = (await c.get("/metrics")).text
        assert 'pgw_gateway_integrity_failures_total{check="freivalds",model="tiny"} 1.0' in m
        st = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
        assert st["breaker"]["open"] and st["breaker"]["trips"] == 1
        # admin reset restores service, results are correct again
        rr = await c.post(f"/admin/models/{MODEL}/reset", headers=ADMIN, json={"breaker": True})
        assert rr.json()["breaker_was_open"] is True
        again = (await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))).json()
        assert again["choices"][0]["message"] == good["choices"][0]["message"]
    audit = (tmp_path / "audit.jsonl").read_text()
    assert "integrity_check_failed" in audit and "admin.safety_reset" in audit


async def test_integrity_failure_mid_stream(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                            keys: ApiKeyStore, key: str) -> None:
    store = make_store(200)
    calls = {"n": 0}

    def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return y + 1 if calls["n"] > 20 else y  # fail during decode, after some tokens

    svc = InProcessService(gw_pack, tamper=tamper)
    async with gateway(masked_app(tiny_pack_dir, tmp_path, store, svc, keys)) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True, "max_tokens": 20}, headers=auth(key))
        ev = parse_sse(r.text)
        assert ev[-1] == "[DONE]" and ev[-2]["error"]["code"] == "integrity_check_failed"
        assert (await c.get("/readyz?models=all")).status_code == 503


async def test_mask_exhausted_fails_fast(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                         keys: ApiKeyStore, key: str) -> None:
    store = make_store(20)
    svc = InProcessService(gw_pack)
    async with gateway(masked_app(tiny_pack_dir, tmp_path, store, svc, keys)) as c:
        assert (await c.get("/readyz")).status_code == 200
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 503 and r.json()["error"]["code"] == "mask_exhausted"
        assert svc.handler.requests == 0 and available(store, gw_pack) == 20  # nothing consumed or sent
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True}, headers=auth(key))
        assert r.status_code == 503 and r.json()["error"]["code"] == "mask_exhausted"
        st = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
        assert st["sessions"]["active"] == 0


async def test_readyz_reports_empty_store(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                          keys: ApiKeyStore) -> None:
    store = make_store(0)
    svc = InProcessService(gw_pack)
    async with gateway(masked_app(tiny_pack_dir, tmp_path, store, svc, keys)) as c:
        r = await c.get("/readyz?models=all")
        assert r.status_code == 503 and r.json()["models"][MODEL]["reason"] == "no mask slots"
        r = await c.get("/readyz")
        assert r.status_code == 200 and r.json()["all_models_ready"] is False
        assert 'pgw_gateway_model_ready{model="tiny"} 0.0' in (await c.get("/metrics")).text


async def test_startup_refuses_wrong_fingerprint_or_ring(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                         make_store: Any, keys: ApiKeyStore) -> None:
    store = make_store(0)
    svc = InProcessService(gw_pack, fingerprint="ab" * 32)
    with pytest.raises(FingerprintMismatchError):
        async with gateway(masked_app(tiny_pack_dir, tmp_path, store, svc, keys)):
            pass
    svc2 = InProcessService(gw_pack, rings=("K48A20",))
    with pytest.raises(Exception, match="ring"):
        async with gateway(masked_app(tiny_pack_dir, tmp_path, store, svc2, keys)):
            pass
    svc3 = InProcessService(gw_pack)
    with pytest.raises(FingerprintMismatchError):  # pinned fingerprint does not match the pack
        async with gateway(masked_app(tiny_pack_dir, tmp_path, store, svc3, keys,
                                      model_kw={"fingerprint": "cd" * 32})):
            pass


async def test_unreachable_service_starts_degraded_and_reconnects(tiny_pack_dir: Path, tmp_path: Path,
                                                                    gw_pack: Any, make_store: Any,
                                                                    keys: ApiKeyStore, key: str,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """A compute-service outage at start-up must not crash-loop the gateway (reliability review)."""
    import httpx

    from pgw.gateway import runtime as runtime_mod

    svc = InProcessService(gw_pack)
    up = {"v": False}

    class Flaky(httpx.AsyncBaseTransport):
        async def handle_async_request(self, req: httpx.Request) -> httpx.Response:
            if not up["v"]:
                raise httpx.ConnectError("refused")
            return await svc.transport.handle_async_request(req)

    real_sleep = asyncio.sleep
    monkeypatch.setattr(runtime_mod.asyncio, "sleep", lambda d: real_sleep(min(d, 0.02)))
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw={"service_max_retries": 0}),
                     mask_store=make_store(200), transports={MODEL: Flaky()}, key_store=keys)
    async with gateway(app) as c:
        r = await c.get("/readyz")
        assert r.status_code == 200 and r.json()["models"][MODEL]["state"] == "connecting"
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 503 and r.json()["error"]["code"] == "unavailable"
        assert "Retry-After" in r.headers
        up["v"] = True
        for _ in range(200):
            if (await c.get("/readyz?models=all")).status_code == 200:
                break
            await real_sleep(0.02)
        assert (await c.get("/readyz?models=all")).status_code == 200
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 200, r.text


async def test_permanent_startup_failure_still_aborts(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                      make_store: Any, keys: ApiKeyStore) -> None:
    svc = InProcessService(gw_pack, fingerprint="ab" * 32)
    app = create_app(masked_settings(tiny_pack_dir, tmp_path), mask_store=make_store(0),
                     transports={MODEL: svc.transport}, key_store=keys)
    with pytest.raises(FingerprintMismatchError):
        async with gateway(app):
            pass


async def test_concurrent_requests_batched_and_exact(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                     make_store: Any, keys: ApiKeyStore, key: str) -> None:
    store = make_store(600)
    svc = InProcessService(gw_pack)
    prompts = ["Hello", "How are you?", "Tell me a story", "abc"]
    async with gateway(masked_app(tiny_pack_dir, tmp_path, store, svc, keys)) as c:
        def req(p: str) -> dict[str, Any]:
            return {**CHAT, "messages": [{"role": "user", "content": p}]}

        seq = []
        for p in prompts:
            seq.append((await c.post("/v1/chat/completions", json=req(p), headers=auth(key))).json())
        n_seq = svc.handler.requests
        conc = await asyncio.gather(*(c.post("/v1/chat/completions", json=req(p), headers=auth(key))
                                      for p in prompts))
        n_conc = svc.handler.requests - n_seq
    for a, b in zip(seq, conc, strict=True):
        assert b.status_code == 200
        assert a["choices"][0]["message"] == b.json()["choices"][0]["message"]
    assert n_conc < n_seq  # sessions at the same linear share round trips


async def test_stream_cancellation_burns_masks_and_frees_slot(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                             make_store: Any) -> None:
    """Closing a stream mid-generation stops it without disturbing a concurrent session."""
    from pgw.engine import SamplingParams

    store = make_store(300)
    svc = InProcessService(gw_pack)
    rt_all = InferenceRuntime(masked_settings(tiny_pack_dir, tmp_path), mask_store=store,
                              transports={MODEL: svc.transport})
    await rt_all.start()
    try:
        rt = rt_all.get(MODEL)
        assert isinstance(rt.client.inner._transport, EgressGuardTransport)
        ids = rt.encode_chat([{"role": "user", "content": "hi"}])
        params = SamplingParams(max_tokens=12, temperature=0, ignore_eos=True)
        before = available(store, gw_pack)
        s1 = await rt.open_session(ids, params)
        s2 = await rt.open_session(ids, params)
        assert rt.admission.active == 2
        it1 = s1.stream()

        async def consume_all() -> list[int]:
            return [ev.token_id async for ev in s2.stream()]

        task = asyncio.create_task(consume_all())
        got = []
        async for ev in it1:
            got.append(ev.token_id)
            if len(got) == 2:
                break
        await it1.aclose()
        full = await task
        assert len(full) == 12 and full[:2] == got
        await asyncio.sleep(0.05)
        assert rt.admission.active == 0 and s1.reservation.closed and s2.reservation.closed
        assert before - available(store, gw_pack) == 2 * (len(ids) + 12)
        # a session that is never started is cleaned up on close
        s3 = await rt.open_session(ids, params)
        await s3.close()
        assert s3.reservation.closed and rt.admission.active == 0
        # convenience APIs (non-streaming run() hands the reservation to the engine as well)
        res = await rt.chat([{"role": "user", "content": "hi"}], params)
        streamed = [ev.token_id async for ev in rt.chat_stream([{"role": "user", "content": "hi"}], params)]
        assert res.token_ids == full == streamed and res.masks_consumed == len(ids) + 11
        assert rt.admission.active == 0 and not rt._sessions
    finally:
        await rt_all.stop()


class FakeNoiseMonitor:
    """Alarms after ``trip_after`` guarded payloads (stands in for pgw.observability's monitor)."""

    trip_after = 10**9

    def __init__(self, ring: RingParams, *, sample_rate: float = 0.05, window_values: int = 1 << 16,
                 fail_closed: bool = True):
        self.n = 0
        self.alarm = False
        self.fail_closed = fail_closed

    def guard(self, x_hat: torch.Tensor, xq: torch.Tensor | None = None) -> None:
        from pgw.core import IntegrityError

        self.n += 1
        if self.n >= self.trip_after:
            self.alarm = True
        if self.alarm and self.fail_closed:
            raise IntegrityError("outbound payload failed the noise test")

    def report(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(alarm=self.alarm, samples=self.n, windows=0, duplicate_rows=0,
                               reasons=["chi2"] if self.alarm else [])

    def reset(self) -> None:
        self.alarm = False
        self.n = 0
        self.trip_after = 10**9


async def test_noise_alarm_blocks_until_admin_reset(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                    make_store: Any, keys: ApiKeyStore, key: str,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    import pgw.observability.noise as noise

    FakeNoiseMonitor.trip_after = 30
    monkeypatch.setattr(noise, "OutboundNoiseMonitor", FakeNoiseMonitor)
    store = make_store(300)
    svc = InProcessService(gw_pack)
    async with gateway(masked_app(tiny_pack_dir, tmp_path, store, svc, keys)) as c:
        st = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
        assert st["noise_monitor"]["state"] == "active"
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 502 and r.json()["error"]["code"] == "integrity_check_failed"
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 502 and "noise" in r.json()["error"]["message"]
        assert (await c.get("/readyz")).json()["models"][MODEL]["reason"] == "noise alarm"
        assert "pgw_gateway_noise_alarm{model=\"tiny\"} 1.0" in (await c.get("/metrics")).text
        rr = (await c.post(f"/admin/models/{MODEL}/reset", headers=ADMIN)).json()
        assert rr["noise_was_alarmed"] is True
        assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))).status_code == 200


def test_noise_monitor_required_but_unavailable(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                make_store: Any, keys: ApiKeyStore,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    import pgw.observability.noise as noise
    from pgw.core import ConfigError

    class Broken:
        def __init__(self, *a: Any, **k: Any) -> None:
            raise ImportError("numpy backend missing")  # an optional dependency is unavailable

    monkeypatch.setattr(noise, "OutboundNoiseMonitor", Broken)
    svc = InProcessService(gw_pack)
    app = masked_app(tiny_pack_dir, tmp_path, make_store(0), svc, keys, noise_monitor_required=True)

    async def run() -> None:
        async with gateway(app):
            pass

    with pytest.raises(ConfigError, match="noise monitor"):
        asyncio.run(run())


async def test_masked_reload_and_streaming(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                           keys: ApiKeyStore, key: str) -> None:
    store = make_store(200)
    svc = InProcessService(gw_pack)
    async with gateway(masked_app(tiny_pack_dir, tmp_path, store, svc, keys)) as c:
        full = (await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))).json()
        r = await c.post(f"/admin/models/{MODEL}/reload", headers=ADMIN)
        assert r.status_code == 200 and r.json()["fingerprint"] == gw_pack.fingerprint
        rs = await c.post("/v1/chat/completions", json={**CHAT, "stream": True}, headers=auth(key))
        ev = parse_sse(rs.text)
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev[:-1] if e["choices"])
        assert text == full["choices"][0]["message"]["content"]


# ---- security review regressions ------------------------------------------------------------------
def test_noise_monitor_required_by_default(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                           make_store: Any, keys: ApiKeyStore,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """Without allow_dev_modes, an unavailable noise monitor aborts start-up (no silent fallback)."""
    import pgw.observability.noise as noise
    from pgw.core import ConfigError

    class Broken:
        def __init__(self, *a: Any, **k: Any) -> None:
            raise ImportError("scipy missing")

    monkeypatch.setattr(noise, "OutboundNoiseMonitor", Broken)
    app = masked_app(tiny_pack_dir, tmp_path, make_store(0), InProcessService(gw_pack), keys)

    async def run() -> None:
        async with gateway(app):
            pass

    with pytest.raises(ConfigError, match="noise monitor"):
        asyncio.run(run())


async def test_weak_segments_refused_before_any_mask_is_used(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                             make_store: Any, keys: ApiKeyStore, key: str) -> None:
    """Segments below min_soundness_bits are refused before x_hat is sent or a mask row is taken."""
    from pgw.factory import MaskGenerator

    store = make_store(0)
    MaskGenerator(gw_pack, DEFAULT_RING).generate_into_store(store, 64, checks=1, chunk_slots=16, factory_id="weak")
    svc = InProcessService(gw_pack)
    app = masked_app(tiny_pack_dir, tmp_path, store, svc, keys, min_soundness_bits=40)
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code >= 500 and "soundness" in r.text
        assert svc.handler.requests == 0


def test_breaker_latches_after_auto_resets() -> None:
    from pgw.gateway.safety import CircuitBreaker

    now = [0.0]
    b = CircuitBreaker("m", 10.0, max_auto_resets=1, clock=lambda: now[0])
    b.trip()
    now[0] = 11.0
    assert not b.is_open  # first cool-down re-opens traffic
    b.trip()
    now[0] = 100.0
    assert b.is_open  # latched: admin reset only
    assert b.reset() and not b.is_open
    default = CircuitBreaker("m", None)
    default.trip()
    assert default.is_open


async def test_swapped_tokenizer_file_refused_at_startup(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                         make_store: Any, keys: ApiKeyStore) -> None:
    """Security review: the gateway hashes plain tensors and tokenizer files (pack_verify='full')."""
    import shutil

    from pgw.core import PGWError

    pack = tmp_path / "pack-copy"
    shutil.copytree(tiny_pack_dir, pack)
    tok = sorted(p for p in (pack / "tokenizer").iterdir() if p.is_file())[0]
    tok.write_bytes(tok.read_bytes() + b" ")
    app = masked_app(pack, tmp_path, make_store(0), InProcessService(gw_pack), keys,
                     model_kw={"fingerprint": gw_pack.fingerprint})
    with pytest.raises(PGWError):
        async with gateway(app):
            pass
