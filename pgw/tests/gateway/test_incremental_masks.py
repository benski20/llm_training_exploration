"""Incremental mask reservations (``mask_reservation = "incremental"``): the prompt plus one public chunk up front,
whole chunks ahead of decoding, an in-memory entitlement ledger for admission.

Checked here: answers and logprobs bit-identical to an up-front reservation, the provider sees exactly the same
calls (linear and row count, in order), consumption is the rows used plus at most about one chunk, admission
refuses what the store cannot cover for every live request, an extension that runs short mid-answer ends the
answer cleanly (nothing reused, every part burned), faults and cancels never reuse a slot, and multi-turn chats
with conversation prefix reuse and strict speculation keep all of the above.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from pgw.core import DEFAULT_RING, MaskExhaustedError, MaskReuseError
from pgw.engine.model import native_float_available
from pgw.gateway import ApiKeyStore, create_app
from pgw.gateway.masks import EntitlementLedger, WindowedReservation, chunk_entitlement
from tests.gateway.conftest import ADMIN, MODEL, InProcessService, auth, gateway, masked_settings, parse_sse

CHUNK = 4
INC = {"mask_reservation": "incremental", "mask_chunk_slots": CHUNK}
UPFRONT = {"mask_window_tokens": 0}
CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello there"}], "temperature": 0,
        "max_tokens": 23, "ignore_eos": True, "logprobs": True, "top_logprobs": 2}
USER_TURNS = ["one two one two", "and three?", "say more", "thanks, bye"]


def available(store: Any, pack: Any) -> int:
    return sum(i.slots_available for i in store.inventory(pack.fingerprint, DEFAULT_RING))


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(None)


@pytest.fixture
def key(keys: ApiKeyStore) -> str:
    return keys.create("app")[1]


def _app(tiny_pack_dir: Path, tmp_path: Path, store: Any, svc: InProcessService, keys: ApiKeyStore,
         model_kw: dict[str, Any] | None = None, **kw: Any) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    return create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw=model_kw or {}, **kw), mask_store=store,
                      transports={MODEL: svc.transport}, key_store=keys)


def _calls(svc: InProcessService) -> list[tuple[str, tuple[int, ...]]]:
    """What the provider saw, in order: per run of calls to one linear, their row counts. A prompt pass's
    pipelined blocks (``TransformerModel.pipeline_blocks``) are concurrent requests whose arrival order varies
    from run to run in any configuration, so the row counts within such a run are compared as a sorted tuple."""
    out: list[tuple[str, list[int]]] = []
    for lid, x in svc.received:
        if out and out[-1][0] == lid:
            out[-1][1].append(int(x.shape[0]))
        else:
            out.append((lid, [int(x.shape[0])]))
    return [(lid, tuple(sorted(rows))) for lid, rows in out]


# ---- unit: the chunked chain and the ledger ---------------------------------------------------------------
class _Part:
    def __init__(self, slots: int):
        self.slots = slots
        self.cur = 0
        self.closed = False
        self.linear_keys = ("x",)

    def remaining(self, lid: Any) -> int:
        return self.slots - self.cur

    def taken(self, lid: Any) -> int:
        return self.cur

    def rows_until_boundary(self, lid: Any) -> int:
        return self.slots - self.cur

    def take(self, lid: Any, rows: int) -> tuple[Any, Any]:
        if self.closed:
            raise MaskReuseError("closed")
        assert self.cur + rows <= self.slots
        self.cur += rows
        return rows, rows

    def checks(self, lid: Any) -> tuple[Any, Any]:
        return None, None

    def close(self) -> None:
        self.closed = True


def _take(w: WindowedReservation, rows: int) -> None:
    w.ensure("x", rows)
    while rows:
        n = min(rows, w.rows_until_boundary("x"))
        w.take("x", n)
        rows -= n


def test_chunk_entitlement() -> None:
    assert chunk_entitlement(10, 10, 4) == 10
    assert chunk_entitlement(10, 11, 4) == 14
    assert chunk_entitlement(10, 18, 4) == 18
    assert chunk_entitlement(10, 3, 4) == 10  # the first part is never clipped to the cap


def test_chunked_extensions_are_whole_chunks_and_retire_in_background() -> None:
    made: list[_Part] = []

    def reserve(n: int) -> _Part:
        made.append(_Part(n))
        return made[-1]

    first = _Part(6)
    cap = chunk_entitlement(6, 6 + 13, 4)  # 22
    w = WindowedReservation(first, reserve, cap=cap, window=4, chunked=True)
    assert w.outstanding == 16
    _take(w, 6)
    _take(w, 1)  # needs 1 more -> one whole chunk
    assert [m.slots for m in made] == [4]
    _take(w, 9)  # holds 3, needs 6 more -> two whole chunks (never a partial one)
    assert [m.slots for m in made] == [4, 8] and w.slots == 18 and w.outstanding == 4
    with pytest.raises(MaskExhaustedError):
        w.ensure("x", 2 + 4 + 1)  # beyond the entitlement: refused before anything is taken
    _take(w, 2)
    _take(w, 1)  # step start: the first two parts are fully taken -> closed on the background thread
    w.close()  # joins the background closes, burns the rest
    assert first.closed and all(m.closed for m in made)
    assert w.outstanding == 0
    with pytest.raises(MaskReuseError):
        w.ensure("x", 1)


def test_ledger_admits_only_what_the_store_covers_for_everyone() -> None:
    led = EntitlementLedger()
    w = WindowedReservation(_Part(10), lambda n: _Part(n), cap=30, window=4, chunked=True)
    led.add(w)
    assert led.outstanding() == 20 and len(led) == 1
    assert led.admit(100, 80) and not led.admit(100, 81) and led.refused == 1
    w.close()
    assert led.outstanding() == 0 and len(led) == 0 and led.admit(100, 100)


# ---- gateway: exactness, provider view, consumption ------------------------------------------------------
@pytest.mark.parametrize("strict", [False, True])
async def test_incremental_is_bit_identical_and_the_provider_sees_the_same_calls(
        tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any, keys: ApiKeyStore, key: str,
        strict: bool) -> None:
    spec: dict[str, Any] = ({"draft_pack_path": str(tiny_pack_dir), "spec_rows": 3, "spec_strict_passes": True}
                            if strict else {})
    body = {**CHAT, "max_tokens": 23}
    ref_svc = InProcessService(gw_pack)
    ref_store = make_store(1500, name="ref")
    async with gateway(_app(tiny_pack_dir, tmp_path / "a", ref_store, ref_svc, keys, spec, **UPFRONT)) as c:
        ref = (await c.post("/v1/chat/completions", json=body, headers=auth(key))).json()
        await c._transport.app.state.pgw.runtime.get(MODEL).generator.wait_drained()  # type: ignore[attr-defined]
    svc = InProcessService(gw_pack)
    store = make_store(1500, name="inc")
    app = _app(tiny_pack_dir, tmp_path / "b", store, svc, keys, spec, **INC)
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json=body, headers=auth(key))
        assert r.status_code == 200, r.text
        got = r.json()
        rt = app.state.pgw.runtime.get(MODEL)
        await rt.generator.wait_drained()
        await rt.generator.wait_closed()
        status = (await c.get("/admin/status", headers=ADMIN)).json()
    assert got["choices"][0]["message"] == ref["choices"][0]["message"]
    assert got["choices"][0]["logprobs"] == ref["choices"][0]["logprobs"]
    assert _calls(svc) == _calls(ref_svc)  # same calls, same row counts, same order (incl. strict dummy passes)
    prompt, done = got["usage"]["prompt_tokens"], got["usage"]["completion_tokens"]
    rows = 3 if strict else 1
    used = prompt + (body["max_tokens"] - 1) * rows  # strict: every pass (dummy passes too) takes 3 rows
    if not strict:
        used = prompt + done - 1
    consumed = 1500 - available(store, gw_pack)
    chunk = rows * -(-CHUNK // rows)  # rounded up to whole passes in strict mode
    assert used <= consumed < used + 2 * chunk, (used, consumed)
    assert 1500 - available(ref_store, gw_pack) == prompt + body["max_tokens"] + (
        (body["max_tokens"] - 1) * 2 if strict else 0)  # up front: everything, unused burned
    assert not store._open_reservations  # every part closed
    audit = [json.loads(x) for x in (tmp_path / "b" / "audit.jsonl").read_text().splitlines()]
    inf = [a for a in audit if a.get("event") == "inference"]
    assert inf and inf[0]["masks_burned"] < 2 * chunk
    assert status["models"][0]["state"] == "ready"


@pytest.mark.skipif(not native_float_available(), reason="pgw_fast without the native float path")
async def test_multi_turn_with_prefix_reuse(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                            keys: ApiKeyStore, key: str) -> None:
    base = {"local_linears": "off", "float_ops": "native"}
    reuse = {**base, "prefix_cache": "conversation"}
    req = {"model": MODEL, "temperature": 0, "max_tokens": 9, "logprobs": True, "top_logprobs": 2}

    async def converse(app: Any, store: Any) -> tuple[list[dict[str, Any]], list[int]]:
        out, used = [], []
        async with gateway(app) as c:
            rt = app.state.pgw.runtime.get(MODEL)
            msgs: list[dict[str, str]] = []
            for t in USER_TURNS:
                msgs.append({"role": "user", "content": t})
                before = available(store, gw_pack)
                r = await c.post("/v1/chat/completions", json={**req, "messages": msgs}, headers=auth(key))
                assert r.status_code == 200, r.text
                await rt.generator.wait_drained()
                await rt.generator.wait_closed()
                used.append(before - available(store, gw_pack))
                out.append(r.json())
                msgs.append({"role": "assistant", "content": out[-1]["choices"][0]["message"]["content"]})
        return out, used

    ref_svc, ref_store = InProcessService(gw_pack), make_store(1500, name="ref")
    ref, ref_used = await converse(_app(tiny_pack_dir, tmp_path / "a", ref_store, ref_svc, keys, reuse, **UPFRONT),
                                   ref_store)
    svc, store = InProcessService(gw_pack), make_store(1500, name="inc")
    got, used = await converse(_app(tiny_pack_dir, tmp_path / "b", store, svc, keys, reuse, **INC), store)
    for a, b in zip(got, ref, strict=True):
        assert a["choices"][0]["message"] == b["choices"][0]["message"]
        assert a["choices"][0]["logprobs"] == b["choices"][0]["logprobs"]
        assert a["usage"] == b["usage"]
    assert _calls(svc) == _calls(ref_svc)
    for g, u, ru in zip(got, used, ref_used, strict=True):
        rows0 = g["usage"]["prompt_tokens"] - g["usage"]["prompt_tokens_details"]["cached_tokens"]
        need = rows0 + g["usage"]["completion_tokens"] - 1
        assert ru == rows0 + req["max_tokens"]
        assert need <= u < need + 2 * CHUNK


# ---- admission, exhaustion, faults, cancels ---------------------------------------------------------------
async def test_admission_counts_live_entitlements(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                  make_store: Any, keys: ApiKeyStore, key: str) -> None:
    """A request is admitted only if the store covers its prompt + max_tokens on top of what running requests
    may still reserve; the window mode would admit it on the stock alone and could run short mid-answer."""
    store = make_store(160)
    svc = InProcessService(gw_pack)
    app = _app(tiny_pack_dir, tmp_path, store, svc, keys, **INC)
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        from pgw.engine import SamplingParams

        sess = await rt.open_session([1, 2, 3, 4], SamplingParams(max_tokens=100, temperature=0, ignore_eos=True))
        it = sess.stream()
        await it.__anext__()
        assert rt.mask_ledger.outstanding() > 0 and available(store, gw_pack) > 60
        before = len(svc.received)
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 60}, headers=auth(key))
        assert r.status_code == 503 and r.json()["error"]["code"] == "mask_exhausted"
        assert "promised" in r.json()["error"]["message"]
        async for _ in it:
            pass
        await rt.generator.wait_closed()
        assert len(svc.received) > before  # only the running answer's passes
        assert rt.mask_ledger.outstanding() == 0
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 10}, headers=auth(key))
        assert r.status_code == 200


async def _ref_answer(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any, keys: ApiKeyStore,
                      key: str, body: dict[str, Any]) -> dict[str, Any]:
    store = make_store(400, name="ref")
    async with gateway(_app(tiny_pack_dir, tmp_path / "ref", store, InProcessService(gw_pack), keys,
                            **UPFRONT)) as c:
        r = await c.post("/v1/chat/completions", json=body, headers=auth(key))
        assert r.status_code == 200
        return r.json()  # type: ignore[no-any-return]


@pytest.mark.parametrize("fault", ["exhausted", "after_commit"])
async def test_extension_failure_mid_answer_ends_cleanly_and_never_reuses(
        tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any, keys: ApiKeyStore, key: str,
        fault: str) -> None:
    """An extension that fails mid-answer (the store runs short, or a crash-like fault after the durable commit)
    ends that answer with an error before the provider sees any row of the step it was for; every part is
    burned, nothing is handed out twice, and the next request is exact."""
    body = {**CHAT, "max_tokens": 20, "stream": True}
    ref = await _ref_answer(tiny_pack_dir, tmp_path, gw_pack, make_store, keys, key, {**CHAT, "max_tokens": 20})
    store = make_store(400, name="inc")
    svc = InProcessService(gw_pack)
    app = _app(tiny_pack_dir, tmp_path / "inc", store, svc, keys, **INC)
    calls = {"n": 0}
    real = store.reserve

    def flaky(*a: Any, **kw: Any) -> Any:
        calls["n"] += 1
        if calls["n"] >= 3:  # the first part and one extension succeed, then every extension fails
            if fault == "exhausted":
                raise MaskExhaustedError("not enough mask slots in the store")
            def hook(point: str) -> None:
                if point == "after_commit":
                    raise OSError("simulated crash after the durable commit")
            store._fault_hook = hook
            try:
                return real(*a, **kw)
            finally:
                store._fault_hook = None
        return real(*a, **kw)

    store.reserve = flaky  # type: ignore[method-assign]
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        r = await c.post("/v1/chat/completions", json=body, headers=auth(key))
        events = parse_sse(r.text)
        await rt.generator.wait_closed()
        text = "".join(e["choices"][0]["delta"].get("content") or "" for e in events
                       if isinstance(e, dict) and e.get("choices"))
        errs = [e for e in events if isinstance(e, dict) and "error" in e]
        assert errs, events  # the answer ended with an error, not a silent truncation
        assert errs[0]["error"]["code"] in ("mask_exhausted", "internal"), errs
        assert text  # it failed mid-answer
        assert ref["choices"][0]["message"]["content"].startswith(text)  # what was sent is exact
        assert not store._open_reservations  # every part closed (burned)
        # no partial pass reached the provider: every layer-0 call has its last-layer call
        n_layers = gw_pack.arch.num_layers
        first = sum(1 for lid, _ in svc.received if lid == "0.qkv")
        last = sum(1 for lid, _ in svc.received if lid == f"{n_layers - 1}.down")
        assert first == last
        store.reserve = real  # type: ignore[method-assign]
        avail = available(store, gw_pack)
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 20}, headers=auth(key))
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"] == ref["choices"][0]["message"]
        await rt.generator.wait_closed()
        assert available(store, gw_pack) < avail  # fresh slots, nothing returned or reused


async def test_cancel_burns_every_part(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                       keys: ApiKeyStore) -> None:
    store = make_store(600)
    app = _app(tiny_pack_dir, tmp_path, store, InProcessService(gw_pack), keys, **INC)
    async with gateway(app):
        rt = app.state.pgw.runtime.get(MODEL)
        from pgw.engine import SamplingParams

        before = available(store, gw_pack)
        sess = await rt.open_session([1, 2, 3, 4], SamplingParams(max_tokens=400, temperature=0, ignore_eos=True))
        it = sess.stream()
        for _ in range(11):
            await it.__anext__()
        await it.aclose()
        await rt.generator.wait_drained()  # the engine sees the cancel at its next step and burns the chain
        await rt.generator.wait_closed()
        assert sess.reservation.closed
        consumed = before - available(store, gw_pack)
        assert sess.masks_reserved == consumed and sess.masks_burned < 2 * CHUNK
        assert consumed < 4 + 11 + 2 * CHUNK  # never max_tokens
        assert not store._open_reservations and rt.mask_ledger.outstanding() == 0


async def test_concurrent_sessions_exact(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any, make_store: Any,
                                         keys: ApiKeyStore, key: str) -> None:
    """Several tenants' answers decoded in shared batches, each with its own chunked chain: every answer equals
    its up-front reference, and the store hands out each slot once (consumed == sum of reservations)."""
    bodies = [{**CHAT, "messages": [{"role": "user", "content": f"Count to {i}"}], "max_tokens": 12 + 3 * i}
              for i in range(4)]
    ref_store = make_store(1500, name="ref")
    async with gateway(_app(tiny_pack_dir, tmp_path / "a", ref_store, InProcessService(gw_pack), keys,
                            **UPFRONT)) as c:
        refs = [(await c.post("/v1/chat/completions", json=b, headers=auth(key))).json() for b in bodies]
    store = make_store(1500, name="inc")
    app = _app(tiny_pack_dir, tmp_path / "b", store, InProcessService(gw_pack), keys, **INC)
    async with gateway(app) as c:
        rs = await asyncio.gather(*[c.post("/v1/chat/completions", json=b, headers=auth(key)) for b in bodies])
        await app.state.pgw.runtime.get(MODEL).generator.wait_closed()
    for r, ref in zip(rs, refs, strict=True):
        assert r.status_code == 200
        assert r.json()["choices"][0] == ref["choices"][0]
    used = sum(r.json()["usage"]["total_tokens"] - 1 for r in rs)
    consumed = 1500 - available(store, gw_pack)
    assert used <= consumed < used + 4 * 2 * CHUNK
    assert not store._open_reservations


def test_settings_default_off() -> None:
    from pgw.gateway import GatewaySettings

    s = GatewaySettings(models=[], admin_token="x" * 32, allow_dev_modes=True)
    assert s.mask_reservation == "window" and s.mask_chunk_slots == 8

