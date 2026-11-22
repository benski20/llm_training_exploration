"""Conversation KV-prefix reuse on the gateway (``prefix_cache = "conversation"``): a multi-turn chat gives the
bits of recomputing every prompt, the provider sees only the new rows of a continued turn, masks are reserved for
them only, and the cache is bound to the API key, the OpenAI ``user`` and the conversation header, and wiped on
integrity failures."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from pgw.core import DEFAULT_RING, IntegrityError
from pgw.engine.model import native_float_available
from pgw.gateway import ApiKeyStore, GatewayModelConfig, create_app
from tests.gateway.conftest import ADMIN, MODEL, InProcessService, auth, gateway, masked_settings, parse_sse

pytestmark = pytest.mark.skipif(not native_float_available(), reason="pgw_fast without the native float path")

BASE = {"local_linears": "off", "float_ops": "native"}
REUSE = {**BASE, "prefix_cache": "conversation"}
USER_TURNS = ["one two one two", "and three?", "say more", "thanks, bye"]


def _req(messages: list[dict[str, str]], **kw: Any) -> dict[str, Any]:
    return {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 6, "logprobs": True,
            "top_logprobs": 2, **kw}


async def _converse(c: httpx.AsyncClient, key: str, turns: int, headers: dict[str, str] | None = None,
                    **kw: Any) -> list[dict[str, Any]]:
    """A chat of ``turns`` turns (each request resends the whole conversation)."""
    msgs: list[dict[str, str]] = []
    out = []
    for t in range(turns):
        msgs.append({"role": "user", "content": USER_TURNS[t % len(USER_TURNS)]})
        r = await c.post("/v1/chat/completions", json=_req(msgs, **kw), headers={**auth(key), **(headers or {})})
        assert r.status_code == 200, r.text
        body = r.json()
        out.append(body)
        msgs.append({"role": "assistant", "content": body["choices"][0]["message"]["content"]})
    return out


def _app(tiny_pack_dir: Path, tmp_path: Path, store: Any, svc: InProcessService, keys: ApiKeyStore,
         model_kw: dict[str, Any], **settings_kw: Any) -> Any:
    return create_app(masked_settings(tiny_pack_dir, tmp_path, model_kw=model_kw, **settings_kw), mask_store=store,
                      transports={MODEL: svc.transport}, key_store=keys)


def _slots(store: Any, fp: str) -> int:
    return sum(i.slots_available for i in store.inventory(fp, DEFAULT_RING))


@pytest.mark.parametrize("strict", [False, True])
async def test_multi_turn_chat_bit_identical_fewer_rows_and_masks(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                                  make_store: Any, strict: bool) -> None:
    keys = ApiKeyStore(None)
    key = keys.create("app")[1]
    spec: dict[str, Any] = ({"draft_pack_path": str(tiny_pack_dir), "spec_rows": 3, "spec_strict_passes": True}
                            if strict else {})
    turns = 4
    svc0 = InProcessService(gw_pack)
    store0 = make_store(1500, name="ref")
    app0 = _app(tiny_pack_dir, tmp_path, store0, svc0, keys, {**BASE, **spec})
    async with gateway(app0) as c:
        ref = await _converse(c, key, turns)
        await app0.state.pgw.runtime.get(MODEL).generator.wait_drained()
    svc = InProcessService(gw_pack)
    store = make_store(1500, name="reuse")
    app = _app(tiny_pack_dir, tmp_path, store, svc, keys, {**REUSE, **spec})
    used = []
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        got = []
        msgs: list[dict[str, str]] = []
        for t in range(turns):
            msgs.append({"role": "user", "content": USER_TURNS[t]})
            before = _slots(store, gw_pack.fingerprint)
            r = await c.post("/v1/chat/completions", json=_req(msgs), headers=auth(key))
            assert r.status_code == 200, r.text
            await rt.generator.wait_drained()
            used.append(before - _slots(store, gw_pack.fingerprint))
            got.append(r.json())
            msgs.append({"role": "assistant", "content": got[-1]["choices"][0]["message"]["content"]})
        status = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]
    for a, b in zip(got, ref, strict=True):
        assert a["choices"][0]["message"] == b["choices"][0]["message"]
        assert a["choices"][0]["logprobs"] == b["choices"][0]["logprobs"]
        assert a["usage"]["prompt_tokens"] == b["usage"]["prompt_tokens"]
    cached = [g["usage"]["prompt_tokens_details"]["cached_tokens"] for g in got]
    assert cached[0] == 0 and all(x > 0 for x in cached[1:])
    # the provider sees, per turn, prompt-pass rows for the uncached tokens only (a prompt pass may go out in
    # pipelined blocks); the decode passes are unchanged
    rows_dec = 3 if strict else 1
    first = [int(x.shape[0]) for lid, x in svc.received if lid == "0.qkv"]
    first0 = [int(x.shape[0]) for lid, x in svc0.received if lid == "0.qkv"]
    prompts = [g["usage"]["prompt_tokens"] for g in got]
    n_dec = sum(g["usage"]["completion_tokens"] - 1 for g in got)
    assert [x for x in first if x == rows_dec] == [x for x in first0 if x == rows_dec] == [rows_dec] * n_dec
    assert sum(x for x in first if x != rows_dec) == sum(prompts) - sum(cached)
    assert sum(x for x in first0 if x != rows_dec) == sum(prompts)
    # masks: the prompt pass takes rows for the new tokens only
    ref_need = [p + 6 + (5 * 2 if strict else 0) for p in prompts]
    assert used == [n - k for n, k in zip(ref_need, cached, strict=True)]
    st = status["speed"]["prefix_cache_stats"]
    assert status["speed"]["prefix_cache"] == "conversation"
    assert st["hits"] == turns - 1 and st["reused_tokens"] == sum(cached) and st["entries"] == 1


async def test_isolation_by_key_user_and_conversation_header(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                             make_store: Any) -> None:
    keys = ApiKeyStore(None)
    k1, k2 = keys.create("a")[1], keys.create("b")[1]
    app = _app(tiny_pack_dir, tmp_path, make_store(1500), InProcessService(gw_pack), keys, REUSE)
    async with gateway(app) as c:
        first = await _converse(c, k1, 1)
        msgs = [{"role": "user", "content": USER_TURNS[0]},
                {"role": "assistant", "content": first[0]["choices"][0]["message"]["content"]},
                {"role": "user", "content": USER_TURNS[1]}]

        async def cached(key: str, **kw: Any) -> int:
            headers = kw.pop("headers", {})
            r = await c.post("/v1/chat/completions", json=_req(msgs, **kw), headers={**auth(key), **headers})
            assert r.status_code == 200, r.text
            return int(r.json()["usage"]["prompt_tokens_details"]["cached_tokens"])

        assert await cached(k2) == 0  # another API key: never
        assert await cached(k1, user="someone") == 0  # another OpenAI user of the same key
        assert await cached(k1, headers={"x-pgw-conversation-id": "chat-2"}) == 0  # another conversation id
        assert await cached(k1) > 0  # the conversation itself
        rt = app.state.pgw.runtime.get(MODEL)
        assert rt.prefix_cache is not None
        r = await c.post("/v1/chat/completions", json=_req(msgs),
                         headers={**auth(k1), "x-pgw-conversation-id": "x" * 129})
        assert r.status_code == 400


async def test_streaming_reuse_and_integrity_failure_wipes(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                           make_store: Any) -> None:
    keys = ApiKeyStore(None)
    key = keys.create("a")[1]
    app = _app(tiny_pack_dir, tmp_path, make_store(1500), InProcessService(gw_pack), keys, REUSE)
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        await _converse(c, key, 1)
        for _ in range(200):  # the cache is committed after the reservation close (executor thread)
            if len(rt.prefix_cache) == 1:
                break
            await asyncio.sleep(0.01)
        assert len(rt.prefix_cache) == 1
        cache = next(iter(rt.prefix_cache._entries.values())).cache
        rt._on_integrity_failure(IntegrityError("Freivalds check failed (test)"))
        assert len(rt.prefix_cache) == 0 and not cache._k.any()
        rt.reset_safety(breaker=True, noise=False)
        # streaming turns store and reuse as well
        msgs = [{"role": "user", "content": USER_TURNS[0]}]
        for t in range(2):
            async with c.stream("POST", "/v1/chat/completions", headers=auth(key),
                                json=_req(msgs, stream=True, stream_options={"include_usage": True})) as rs:
                body = "".join([line + "\n" async for line in rs.aiter_lines()])
            events = parse_sse(body)
            usage = events[-2]["usage"]
            text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-2] if e["choices"])
            assert (usage["prompt_tokens_details"]["cached_tokens"] > 0) == (t == 1)
            msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": USER_TURNS[1]}]
            await asyncio.sleep(0.05)
            await rt.generator.wait_drained()


async def test_real_integrity_failure_discards(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                               make_store: Any) -> None:
    """A tampering service: the failed turn stores nothing and every cached conversation is wiped."""
    import torch

    bad = {"on": False}

    def tamper(lid: str, y: torch.Tensor) -> torch.Tensor:
        if bad["on"] and lid.endswith("down"):
            y = y.clone()
            y[0, 0] += 1
        return y

    keys = ApiKeyStore(None)
    key = keys.create("a")[1]
    app = _app(tiny_pack_dir, tmp_path, make_store(1500), InProcessService(gw_pack, tamper=tamper), keys, REUSE)
    async with gateway(app) as c:
        rt = app.state.pgw.runtime.get(MODEL)
        await _converse(c, key, 1)
        await _converse(c, key, 1, user="other")
        for _ in range(200):  # committed after the reservation close (executor thread)
            if len(rt.prefix_cache) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(rt.prefix_cache) == 2
        bad["on"] = True
        msgs = [{"role": "user", "content": "x"}]
        r = await c.post("/v1/chat/completions", json=_req(msgs), headers=auth(key))
        assert r.status_code == 502
        await rt.generator.wait_drained()
        assert len(rt.prefix_cache) == 0
        assert rt.prefix_cache.status()["wiped_all"] >= 1


def test_prefix_cache_config_validation() -> None:
    def cfg(**kw: Any) -> GatewayModelConfig:
        return GatewayModelConfig(name="m", pack_path=Path("/x"), service_url="https://compute.example.net:8443",
                                  fingerprint="ab" * 32, **kw)

    assert cfg().prefix_cache == "off"
    with pytest.raises(ValidationError, match="float_ops = 'native'"):
        cfg(prefix_cache="conversation")
    with pytest.raises(ValidationError, match="device = 'cpu'"):
        cfg(prefix_cache="conversation", float_ops="native", device="cuda")
    c = cfg(prefix_cache="conversation", float_ops="native")
    assert c.prefix_cache_ttl_s == 600 and c.prefix_cache_max_mb == 1024 and c.prefix_cache_wait_s == 30
