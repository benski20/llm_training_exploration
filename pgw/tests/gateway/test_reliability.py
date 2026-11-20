"""Reliability review regressions for the gateway: windowed masks, burn charging, atomic admission,
background start-up, process-level readiness, drain, mask store GC/ownership, deferred quota flush."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from pgw.core import DEFAULT_RING, MaskExhaustedError, MaskReuseError, ServiceUnavailableError
from pgw.gateway import ApiKeyStore, create_app
from pgw.gateway.auth import QuotaTracker
from pgw.gateway.masks import WindowedReservation
from pgw.gateway.safety import AdmissionController, AdmissionLimitError
from tests.gateway.conftest import ADMIN, MODEL, InProcessService, auth, gateway, masked_settings, parse_sse

CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello there"}], "temperature": 0,
        "max_tokens": 6, "ignore_eos": True}


def available(store: Any, pack: Any, ring: str = DEFAULT_RING.name) -> int:
    return sum(i.slots_available for i in store.inventory(pack.fingerprint, ring))


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(None)


@pytest.fixture
def key(keys: ApiKeyStore) -> str:
    return keys.create("app")[1]


def _app(tiny_pack_dir: Path, tmp_path: Path, store: Any, svc: InProcessService, keys: ApiKeyStore,
         **kw: Any) -> Any:
    return create_app(masked_settings(tiny_pack_dir, tmp_path, **kw), mask_store=store,
                      transports={MODEL: svc.transport}, key_store=keys)


# ---- windowed reservations ----------------------------------------------------------------------
async def test_windowed_reservation_is_exact_and_bounds_burn(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                             make_store: Any, keys: ApiKeyStore, key: str) -> None:
    body = {**CHAT, "max_tokens": 30}
    ref_store = make_store(1000, name="ref")
    async with gateway(_app(tiny_pack_dir, tmp_path / "a", ref_store, InProcessService(gw_pack), keys,
                            mask_window_tokens=0)) as c:
        ref = (await c.post("/v1/chat/completions", json=body, headers=auth(key))).json()
    store = make_store(1000, name="win")
    async with gateway(_app(tiny_pack_dir, tmp_path / "b", store, InProcessService(gw_pack), keys,
                            mask_window_tokens=4)) as c:
        got = (await c.post("/v1/chat/completions", json=body, headers=auth(key))).json()
        assert got["choices"][0]["message"] == ref["choices"][0]["message"]  # bit-identical across extensions
        used = got["usage"]["total_tokens"] - 1
        assert 1000 - available(store, gw_pack) <= used + 4
        # a stream cancelled after the first token (client disconnect) burns at most one window
        from pgw.engine import SamplingParams

        app_rt = c._transport.app.state.pgw.runtime.get(MODEL)  # type: ignore[attr-defined]
        rt = app_rt
        before = available(store, gw_pack)
        sess = await rt.open_session([1, 2, 3, 4], SamplingParams(max_tokens=500, temperature=0, ignore_eos=True))
        it = sess.stream()
        await it.__anext__()
        await it.aclose()
        await app_rt.generator.wait_closed()
        assert sess.masks_reserved <= 4 + 4 + 4 and sess.masks_burned <= 4
        assert before - available(store, gw_pack) <= 4 + 2 * 4
    audit = [json.loads(line) for line in (tmp_path / "b" / "audit.jsonl").read_text().splitlines()]
    inf = [a for a in audit if a.get("event") == "inference"]
    assert all("masks_burned" in a for a in inf)
    assert inf[0]["masks_burned"] <= 4


async def test_windowed_request_fails_fast_when_store_cannot_cover_it(tiny_pack_dir: Path, tmp_path: Path,
                                                                      gw_pack: Any, make_store: Any,
                                                                      keys: ApiKeyStore, key: str) -> None:
    store = make_store(60)
    svc = InProcessService(gw_pack)
    async with gateway(_app(tiny_pack_dir, tmp_path, store, svc, keys, mask_window_tokens=4)) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 200}, headers=auth(key))
        assert r.status_code == 503 and r.json()["error"]["code"] == "mask_exhausted"
        assert svc.handler.requests == 0 and available(store, gw_pack) == 60


class _FakeRes:
    def __init__(self, slots: int):
        self.slots = slots
        self.cur = 0
        self.closed = False

    def remaining(self, lid: Any) -> int:
        return self.slots - self.cur

    def taken(self, lid: Any) -> int:
        return self.cur

    def rows_until_boundary(self, lid: Any) -> int:
        return self.slots - self.cur

    def take(self, lid: Any, rows: int) -> tuple[Any, Any]:
        assert self.cur + rows <= self.slots
        self.cur += rows
        return rows, rows

    def checks(self, lid: Any) -> tuple[Any, Any]:
        return None, None

    def close(self) -> None:
        self.closed = True


def test_windowed_reservation_unit() -> None:
    made: list[_FakeRes] = []

    def reserve(n: int) -> _FakeRes:
        made.append(_FakeRes(n))
        return made[-1]

    w = WindowedReservation(_FakeRes(5), reserve, cap=20, window=4)
    assert w.remaining("x") == 20 and w.slots == 5
    w.ensure("x", 5)
    assert not made
    w.take("x", 5)
    w.ensure("x", 1)
    assert [m.slots for m in made] == [4] and w.rows_until_boundary("x") == 4
    w.ensure("x", 10)  # needs 6 more than the 4 held -> one extension of 6
    assert [m.slots for m in made] == [4, 6] and w.slots == 15
    with pytest.raises(MaskExhaustedError):
        w.ensure("x", 16)  # beyond the cap: refused before anything is taken
    w.close()
    assert all(m.closed for m in made)


# ---- charging burned masks -------------------------------------------------------------------------
async def test_rate_limit_charges_reserved_masks(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                 make_store: Any, keys: ApiKeyStore) -> None:
    key = keys.create("burner", tokens_per_minute=1_000_000)[1]
    store = make_store(1000)
    quotas = QuotaTracker(None)
    app = create_app(masked_settings(tiny_pack_dir, tmp_path, mask_window_tokens=0), mask_store=store,
                     transports={MODEL: InProcessService(gw_pack).transport}, key_store=keys, quotas=quotas)
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 100, "stop": ["\x00\x01"]},
                         headers=auth(key))
        assert r.status_code == 200
        reserved = r.json()["usage"]["prompt_tokens"] + 100
    used = quotas.usage()
    assert list(used.values()) == [reserved]  # charged for the burned masks, not just the tokens generated


# ---- admission ----------------------------------------------------------------------------------
async def test_multi_slot_admission_is_atomic() -> None:
    adm = AdmissionController(4, 8, 0.5)
    with pytest.raises(AdmissionLimitError):
        await adm.acquire(5)
    # three requests of 3 slots each on a 4-slot model: no hold-and-wait deadlock
    order: list[int] = []

    async def req(i: int) -> None:
        await adm.acquire(3)
        order.append(i)
        await asyncio.sleep(0.05)
        await adm.release(3)

    await asyncio.wait_for(asyncio.gather(req(0), req(1), req(2)), 3)
    assert sorted(order) == [0, 1, 2] and adm.active == 0
    await adm.acquire(4)
    with pytest.raises(ServiceUnavailableError):
        await adm.acquire(1)  # times out in the queue, holds nothing
    assert adm.active == 4 and adm.status()["rejected_by_reason"] == {"too_large": 1, "timeout": 1}


async def test_multi_prompt_requests_do_not_deadlock(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                     make_store: Any, keys: ApiKeyStore, key: str) -> None:
    store = make_store(3000)
    svc = InProcessService(gw_pack)
    app = _app(tiny_pack_dir, tmp_path, store, svc, keys,
               model_kw={"max_concurrent_sessions": 4, "queue_timeout_s": 20.0})
    body = {"model": MODEL, "prompt": ["a", "b", "c"], "max_tokens": 3, "temperature": 0}
    async with gateway(app) as c:
        rs = await asyncio.wait_for(asyncio.gather(
            *(c.post("/v1/completions", json=body, headers=auth(key)) for _ in range(3))), 60)
        assert [r.status_code for r in rs] == [200, 200, 200]
        too_many = {**body, "prompt": ["a"] * 5}
        r = await c.post("/v1/completions", json=too_many, headers=auth(key))
        assert r.status_code == 400
        m = (await c.get("/metrics")).text
        assert 'pgw_gateway_admission_rejected_total{model="tiny",reason="too_large"} 1.0' in m
        assert "pgw_gateway_queue_wait_seconds_count" in m
        assert 'pgw_gateway_service_attempts_total{code="ok",model="tiny"}' in m


# ---- readiness / drain ------------------------------------------------------------------------------
async def test_drain_fails_readyz_and_refuses_new_work(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                       make_store: Any, keys: ApiKeyStore, key: str) -> None:
    store = make_store(500)
    app = _app(tiny_pack_dir, tmp_path, store, InProcessService(gw_pack), keys)
    async with gateway(app) as c:
        assert (await c.get("/readyz")).status_code == 200
        app.state.pgw.runtime.begin_drain()
        assert (await c.get("/readyz")).status_code == 503
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 503
        ops = app.state.ops_app
        import httpx

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=ops), base_url="http://ops") as oc:
            assert (await oc.get("/healthz")).status_code == 200
            assert (await oc.get("/readyz")).status_code == 503
            assert "pgw_gateway_model_ready" in (await oc.get("/metrics")).text


def test_cli_passes_graceful_shutdown_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import uvicorn

    from pgw.gateway import cli
    from pgw.gateway.config import GatewaySettings

    seen: list[Any] = []

    async def fake_serve(self: Any, sockets: Any = None) -> None:
        seen.append(self.config)

    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)

    class _App:
        class state:  # noqa: N801
            ops_app = object()

    s = GatewaySettings(drain_timeout_s=25, ops_port=9091)
    cli.run_gateway_servers(_App(), s, "127.0.0.1", 8080, {})
    assert seen[0].timeout_graceful_shutdown == 25 and seen[1].port == 9091


# ---- mask store housekeeping ------------------------------------------------------------------------
async def test_gateway_runs_gc_and_owns_the_store(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                  make_store: Any, keys: ApiKeyStore) -> None:
    from pgw.core import DEFAULT_RING
    from pgw.maskstore import store as store_mod

    store = make_store(600)
    # an open lease of a crashed previous gateway pod (other hostname) holding a segment
    res = store.reserve(gw_pack.fingerprint, DEFAULT_RING, 512)
    with store._lock.hold(), store._txn() as conn:
        conn.execute("UPDATE reservations SET host=? WHERE id=?", ("old-pod|pid:[1]", res.id))
    store._open_reservations.pop(res.id, None)
    store_mod._live_discard(res.id)
    res._store = None
    res.close()
    app = _app(tiny_pack_dir, tmp_path, store, InProcessService(gw_pack), keys, maskstore_gc_interval_s=3600)
    async with gateway(app) as c:
        rt = app.state.pgw.runtime
        for _ in range(100):
            if rt.gc_runs:
                break
            await asyncio.sleep(0.05)
        assert rt.gc_runs >= 1 and store._owner_fd is not None
        st = store.status()
        assert st["segments_by_status"].get("shredded", 0) == 1  # the dead lease no longer blocks shredding
        m = (await c.get("/metrics")).text
        assert "pgw_masks_open_reservations 0.0" in m and "pgw_masks_unshredded_bytes 0.0" in m


def test_quota_tracker_deferred_flush(tmp_path: Path) -> None:
    from pgw.gateway.auth import ApiKey

    p = tmp_path / "q.json"
    q = QuotaTracker(None, p, deferred=True)
    k = ApiKey(key_id="k1", hashed="x" * 64, name="n")
    q.reserve(k, 10)
    q.settle(k, 10, 7)
    assert not p.exists()  # no fsync on the request path
    assert q.flush() and json.loads(p.read_text())["used"] == {"k1": 7}
    assert not q.flush()  # clean: nothing to write


async def test_streaming_still_works_with_windows(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                  make_store: Any, keys: ApiKeyStore, key: str) -> None:
    store = make_store(500)
    async with gateway(_app(tiny_pack_dir, tmp_path, store, InProcessService(gw_pack), keys,
                            mask_window_tokens=2)) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 9, "stream": True},
                         headers=auth(key))
        ev = parse_sse(r.text)
        assert ev[-1] == "[DONE]" and ev[-2]["choices"][0]["finish_reason"] == "length"
        assert (await c.get("/admin/status", headers=ADMIN)).status_code == 200


def test_windowed_take_checked_follows_parts() -> None:
    made: list[_FakeRes] = []

    def reserve(n: int) -> _FakeRes:
        made.append(_FakeRes(n))
        return made[-1]

    first = _FakeRes(3)
    w = WindowedReservation(first, reserve, cap=20, window=4)
    rows, _, _, chk = w.take_checked("x", 5, False)
    assert rows == 3 and chk is None and first.cur == 3  # never crosses a part
    with pytest.raises(MaskExhaustedError):
        w.take_checked("x", 1, False)  # nothing reserved beyond the first part yet
    w.ensure("x", 2)
    rows, *_ = w.take_checked("x", 5, False)
    assert rows == 4 and made[0].cur == 4
    for _ in range(5):  # many extensions: the current part is found without rescanning old ones
        w.ensure("x", 1)
        assert w.take_checked("x", 1, False)[0] == 1
    assert w.taken("x") == 12 and w.remaining("x") == 8
    w.close()
    with pytest.raises(MaskReuseError):
        w.take_checked("x", 1, False)
