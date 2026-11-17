"""Hardening regressions: cross-task session close, cancellation while reserving, admin input validation,
production policy edges, and request-schema bounds."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from pgw.gateway import ApiKeyStore
from pgw.gateway.runtime import InferenceRuntime
from tests.gateway.conftest import MODEL, InProcessService, masked_settings


def available(store: Any, pack: Any, ring: str = "K40A12") -> int:
    return sum(i.slots_available for i in store.inventory(pack.fingerprint, ring))


async def test_stream_closed_from_another_task(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                               make_store: Any) -> None:
    """A stream iterated in one task can be closed (shutdown, reload) from another without errors."""
    from pgw.engine import SamplingParams

    store = make_store(200)
    svc = InProcessService(gw_pack)
    rt_all = InferenceRuntime(masked_settings(tiny_pack_dir, tmp_path, drain_timeout_s=0.0), mask_store=store,
                              transports={MODEL: svc.transport})
    await rt_all.start()
    rt = rt_all.get(MODEL)
    ids = rt.encode_chat([{"role": "user", "content": "hi"}])
    params = SamplingParams(max_tokens=40, temperature=0, ignore_eos=True)
    before = available(store, gw_pack)
    sess = await rt.open_session(ids, params)
    it = sess.stream()
    first = asyncio.Event()

    async def consume() -> int:
        n = 0
        async for _ in it:
            n += 1
            first.set()
            await asyncio.sleep(0.01)
        return n

    task = asyncio.create_task(consume())
    await asyncio.wait_for(first.wait(), 30)
    await sess.close()  # from this task, while the consumer task is suspended in the stream
    await rt_all.stop()
    res = await asyncio.gather(task, return_exceptions=True)
    assert not isinstance(res[0], ValueError), res[0]
    assert sess.reservation.closed and rt.admission.active == 0
    # the whole reservation is gone (consumed or burned), never returned
    assert before - available(store, gw_pack) == len(ids) + 40


async def test_cancel_while_reserving_burns_reservation(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                        make_store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request cancelled while the store reserves does not leak the reservation or the admission slot."""
    import threading

    from pgw.engine import SamplingParams

    store = make_store(100)
    svc = InProcessService(gw_pack)
    rt_all = InferenceRuntime(masked_settings(tiny_pack_dir, tmp_path), mask_store=store,
                              transports={MODEL: svc.transport})
    await rt_all.start()
    try:
        rt = rt_all.get(MODEL)
        gate = threading.Event()
        made: list[Any] = []
        real = store.reserve

        def slow_reserve(*a: Any, **k: Any) -> Any:
            gate.wait(10)
            r = real(*a, **k)
            made.append(r)
            return r

        monkeypatch.setattr(store, "reserve", slow_reserve)
        task = asyncio.create_task(rt.open_session([1, 2, 3], SamplingParams(max_tokens=4)))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.01)
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(200):
            if made and made[0].closed:
                break
            await asyncio.sleep(0.01)
        assert made and made[0].closed
        assert rt.admission.active == 0
    finally:
        await rt_all.stop()


async def test_admin_inputs_validated(tiny_pack_dir: Path, tmp_path: Path) -> None:
    from pgw.gateway import create_app
    from tests.gateway.conftest import ADMIN, gateway, plain_settings

    keys = ApiKeyStore(None)
    app = create_app(plain_settings(tiny_pack_dir, tmp_path), key_store=keys)
    async with gateway(app) as c:
        r = await c.post("/admin/keys", headers=ADMIN, json={"name": "x", "scopes": "admin"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "protocol"
        r = await c.post("/admin/keys", headers=ADMIN, json={"name": "x", "models": "tiny"})
        assert r.status_code == 400
        r = await c.post("/admin/keys", headers=ADMIN, json={"name": "x", "models": ["tiny", 3]})
        assert r.status_code == 400
        assert len(keys.list()) == 0
        r = await c.post(f"/admin/models/{MODEL}/reset", headers=ADMIN, json=[1, 2])
        assert r.status_code == 400 and r.json()["error"]["code"] == "protocol"
        r = await c.post(f"/admin/models/{MODEL}/reset", headers=ADMIN, content=b"{not json")
        assert r.status_code == 400
        # out-of-range seed is a 400, not a 500 from the sampler
        key = keys.create("app")[1]
        r = await c.post("/v1/completions", headers={"Authorization": f"Bearer {key}"},
                         json={"model": MODEL, "prompt": "hi", "seed": 1 << 70, "max_tokens": 1})
        assert r.status_code == 400


def test_policy_warns_on_plain_http_service_without_tls_requirement(tiny_pack_dir: Path,
                                                                     caplog: pytest.LogCaptureFixture) -> None:
    from pgw.gateway import GatewayModelConfig, GatewaySettings

    m = GatewayModelConfig(name=MODEL, pack_path=tiny_pack_dir, service_url="http://10.0.0.5:8443")
    with caplog.at_level("WARNING", logger="pgw.gateway.config"):
        GatewaySettings(models=[m], require_service_tls=False).validate_policy()
    assert any("plain-http" in r.getMessage() for r in caplog.records)


def test_admin_token_minimum_length() -> None:
    from pydantic import ValidationError

    from pgw.gateway import GatewaySettings

    with pytest.raises(ValidationError):
        GatewaySettings(admin_token="short")
    GatewaySettings(admin_token="x" * 16)


async def test_admission_no_lost_wakeup() -> None:
    """A waiter that times out after being notified passes the wake-up on to the next waiter."""
    from pgw.gateway import AdmissionController

    ac = AdmissionController(1, 4, 0.2)
    await ac.acquire()
    w1 = asyncio.create_task(ac.acquire())
    await asyncio.sleep(0.01)
    w2 = asyncio.create_task(ac.acquire())
    await asyncio.sleep(0.01)
    w1.cancel()  # w1 is cancelled; the slot released next must still reach w2
    await ac.release()
    with pytest.raises(asyncio.CancelledError):
        await w1
    waited = await asyncio.wait_for(w2, 1.0)
    assert waited >= 0 and ac.active == 1


async def test_breaker_stops_in_flight_sessions(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                make_store: Any) -> None:
    """Once the breaker is open, an already-running session makes no further round trips."""
    from pgw.core import PGWError
    from pgw.engine import SamplingParams

    store = make_store(200)
    svc = InProcessService(gw_pack)
    rt_all = InferenceRuntime(masked_settings(tiny_pack_dir, tmp_path), mask_store=store,
                              transports={MODEL: svc.transport})
    await rt_all.start()
    try:
        rt = rt_all.get(MODEL)
        ids = rt.encode_chat([{"role": "user", "content": "hi"}])
        sess = await rt.open_session(ids, SamplingParams(max_tokens=20, temperature=0, ignore_eos=True))
        got = 0
        with pytest.raises(PGWError) as ei:
            async for _ in sess.stream():
                got += 1
                if got == 2:
                    rt.breaker.trip(reason="test")
                    n_req = svc.handler.requests
        assert ei.value.code == "unavailable" and got == 2
        assert svc.handler.requests == n_req  # nothing more was sent
        for _ in range(100):  # the engine burns the reservation at its next safe point
            if sess.reservation.closed:
                break
            await asyncio.sleep(0.01)
        assert sess.reservation.closed and rt.admission.active == 0
    finally:
        await rt_all.stop()
