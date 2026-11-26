"""Provider-side maskgen liveness (:mod:`pgw.hemask.progress`): the coarse public bit, the caller's own numbers, the
adaptive bound, and the handler's accounting of every item from admission to the end of its thread."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from pgw.hemask.progress import MaskgenProgress, stall_bound


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _prog(**kw: Any) -> tuple[MaskgenProgress, Clock]:
    c = Clock()
    return MaskgenProgress(stall_s=kw.pop("stall_s", 30.0), warmup_s=kw.pop("warmup_s", 300.0), clock=c), c


def _run(p: MaskgenProgress, c: Clock, tenant: str, seconds: float, shape: Any = "s") -> None:
    tok = p.admit(tenant, shape)
    p.start(tok)
    c.t += seconds
    p.finish(tok)


def test_stall_bound() -> None:
    assert stall_bound(30, None, False) == 30
    assert stall_bound(30, 20, False) == 80  # 4x the slowest recent item
    assert stall_bound(30, 1, True, 300) == 300  # the first item of a shape: warm-up bound


def test_a_wedged_item_stalls_after_the_bound_and_only_the_bit_is_public() -> None:
    p, c = _prog()
    _run(p, c, "a", 0.5)  # warm: the shape has finished once
    assert p.public_state() == "ok"
    tok = p.admit("a", "s")
    p.start(tok)
    c.t += 29
    assert p.public_state() == "ok" and not p.tenant_view("a")["tenant"]["stalled"]
    c.t += 2
    assert p.public_state() == "stalled"
    v = p.tenant_view("a")
    assert v["maskgen"] == "stalled" and v["tenant"]["stalled"] and v["tenant"]["running"] == 1
    assert v["tenant"]["oldest_running_s"] == 31.0 and v["tenant"]["done"] == 1
    p.finish(tok)
    assert p.public_state() == "ok" and p.tenant_view("a")["tenant"]["running"] == 0


def test_slow_items_that_finish_raise_the_bound() -> None:
    """A slow but healthy provider whose items grow slower (8 s, then 20 s, then 30 s each: a 72B linear on a slow
    device): never a stall; the bound follows the items."""
    p, c = _prog(stall_s=10.0)
    _run(p, c, "a", 0.2)
    for dur in (8, 8, 20, 20, 30, 30):
        tok = p.admit("a", "s")
        p.start(tok)
        for _ in range(dur):
            c.t += 1
            assert p.public_state() == "ok" and not p.tenant_view("a")["tenant"]["stalled"]
        p.finish(tok)
    tok = p.admit("a", "s")
    p.start(tok)
    c.t += 119
    assert p.public_state() == "ok" and p.tenant_view("a")["tenant"]["bound_s"] == 120.0
    c.t += 2
    assert p.public_state() == "stalled"


def test_the_first_item_of_a_shape_gets_the_warmup_bound() -> None:
    p, c = _prog(stall_s=10.0, warmup_s=120.0)
    tok = p.admit("a", ("cuda:0", 8, 4))
    p.start(tok)
    c.t += 100
    v = p.tenant_view("a")["tenant"]
    assert v["cold"] and v["bound_s"] == 120.0 and not v["stalled"] and p.public_state() == "ok"
    c.t += 21
    assert p.public_state() == "stalled"


def test_a_queue_that_does_not_move_is_a_stall() -> None:
    """Items wait for a device slot and nothing starts or finishes (leaked slots): the coarse bit says so."""
    p, c = _prog(stall_s=10.0)
    p.admit("a", "s")
    c.t += 9
    assert p.public_state() == "ok"
    c.t += 2
    assert p.public_state() == "stalled"
    assert p.operator_view()["waiting"] == 1


def test_a_tenant_sees_only_its_own_numbers() -> None:
    p, c = _prog()
    _run(p, c, "a", 1.0)
    _run(p, c, "tenant-other", 3.0)
    tb = p.admit("tenant-other", "s")
    p.start(tb)
    c.t += 5
    va = p.tenant_view("a")
    assert va["tenant"]["running"] == 0 and va["tenant"]["done"] == 1 and va["tenant"]["slowest_item_s"] == 1.0
    assert set(va) == {"maskgen", "stall_s", "warmup_s", "tenant"}
    assert "tenant-other" not in repr(va)
    vb = p.tenant_view("tenant-other")["tenant"]
    assert vb["running"] == 1 and vb["oldest_running_s"] == 5.0
    op = p.operator_view()
    assert op["done"] == 2 and op["running"] == 1 and "a" not in op


def test_failed_and_abandoned_items() -> None:
    p, _ = _prog()
    tok = p.admit("a", "s")
    p.finish(tok, ok=False)  # cancelled before it started: neither done nor failed
    tok = p.admit("a", "s")
    p.start(tok)
    p.finish(tok, ok=False)
    v = p.tenant_view("a")["tenant"]
    assert v["done"] == 0 and v["failed"] == 1 and v["running"] == v["waiting"] == 0


# ---- the handler ---------------------------------------------------------------------------------------------
def _handler(conc: int = 1) -> Any:
    from pgw.service import handler as handler_mod

    h = handler_mod.ServiceHandler.__new__(handler_mod.ServiceHandler)
    h.maskgen_concurrency = conc
    h._maskgen_slots = asyncio.Semaphore(conc)
    h._maskgen_dev_slots = {}
    h.maskgen_progress = MaskgenProgress(stall_s=0.3, warmup_s=0.3)
    return h


async def test_the_handler_counts_items_from_admission_to_the_end_of_their_thread() -> None:
    h = _handler(conc=1)
    gate = threading.Event()
    kern = SimpleNamespace(device="cpu", out_features=8, in_features=4)

    def item(k: Any, it: Any) -> bytes:
        if it == "hang":
            gate.wait(10)
        if it == "boom":
            raise RuntimeError("x")
        return b"y"

    t1 = asyncio.ensure_future(h._maskgen_in_slot(kern, "hang", item, tenant="a"))
    t2 = asyncio.ensure_future(h._maskgen_in_slot(kern, "ok", item, tenant="a"))
    await asyncio.sleep(0.1)
    v = h.maskgen_progress.tenant_view("a")["tenant"]
    assert v["running"] == 1 and v["waiting"] == 1
    await asyncio.sleep(0.4)
    assert h.maskgen_progress.public_state() == "stalled" and h.maskgen_progress.tenant_view("a")["tenant"]["stalled"]
    gate.set()
    assert await t1 == b"y" and await t2 == b"y"
    with pytest.raises(RuntimeError):
        await h._maskgen_in_slot(kern, "boom", item, tenant="a")
    v = h.maskgen_progress.tenant_view("a")["tenant"]
    assert v["done"] == 2 and v["failed"] == 1 and v["running"] == 0 and h.maskgen_progress.public_state() == "ok"


async def test_a_cancelled_request_keeps_its_item_counted_until_the_thread_ends() -> None:
    """A client that disconnects cannot stop the worker thread: the item stays running (and can stall) until it
    ends, like its device slot."""
    h = _handler(conc=1)
    gate = threading.Event()
    kern = SimpleNamespace(device="cpu", out_features=8, in_features=4)
    t = asyncio.ensure_future(h._maskgen_in_slot(kern, "x", lambda k, it: gate.wait(10) and b"y", tenant="a"))
    await asyncio.sleep(0.05)
    t.cancel()
    await asyncio.sleep(0.05)
    assert h.maskgen_progress.tenant_view("a")["tenant"]["running"] == 1
    gate.set()
    for _ in range(50):
        if h.maskgen_progress.tenant_view("a")["tenant"]["running"] == 0:
            break
        await asyncio.sleep(0.02)
    assert h.maskgen_progress.tenant_view("a")["tenant"]["running"] == 0


async def test_the_status_route_is_authenticated_and_per_tenant() -> None:
    import httpx
    from starlette.applications import Starlette

    from pgw.core.errors import AuthError
    from pgw.hemask.server import MASKGEN_STATUS_PATH, add_maskgen_route

    prog = MaskgenProgress(stall_s=30.0)
    tok = prog.admit("b", "s")
    prog.start(tok)

    class H:
        maskgen_progress = prog

        def authenticate(self, authorization: str | None, subject: str | None) -> Any:
            if authorization != "Bearer good":
                raise AuthError("invalid bearer token")
            return SimpleNamespace(tenant_id="a")

        async def maskgen(self, req: Any, ident: Any) -> Any:  # pragma: no cover - not called
            raise AssertionError

    app = Starlette()
    add_maskgen_route(app, H())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(MASKGEN_STATUS_PATH)
        assert r.status_code == 401 and r.json() == {"error": "unauthenticated"}
        r = await c.get(MASKGEN_STATUS_PATH, headers={"Authorization": "Bearer good"})
        assert r.status_code == 200
        body = r.json()
        assert body["maskgen"] == "ok" and body["tenant"]["running"] == 0  # tenant b's item is not a's
        assert r.headers["cache-control"] == "no-store"


def test_readyz_public_carries_only_the_bit() -> None:
    """The public /readyz gets ``maskgen: ok|stalled`` and nothing else; the metrics port the operator's numbers."""
    from pgw.service.app import ServiceState

    prog = MaskgenProgress(stall_s=30.0)
    st = ServiceState.__new__(ServiceState)
    st.loaded = False
    st.load_error = None
    st.lane = None
    st.registry = SimpleNamespace(models=lambda: [], __len__=lambda: 0)
    st.batcher = SimpleNamespace(queued_rows=0, running=True)
    st.handler = SimpleNamespace(draining=False, inflight=0, maskgen_progress=prog)

    async def go(public: bool) -> dict[str, Any]:
        return (await st.readiness(public=public))[1]

    pub = asyncio.run(go(True))
    assert pub["maskgen"] == "ok" and "maskgen_progress" not in pub and "inflight" not in pub
    op = asyncio.run(go(False))
    assert op["maskgen_progress"]["running"] == 0 and time.time() > 0
