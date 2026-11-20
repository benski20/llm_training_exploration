"""Gateway with two-cohort ping-pong (model config ``engine_pingpong``): real encrypted mask store, pgw.wire
HTTP and an in-process service. Concurrent requests return exactly what sequential requests return, use the
same mask slots, and the admin status reports the split steps."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pgw.core import DEFAULT_RING
from pgw.gateway import ApiKeyStore, GatewayModelConfig, create_app
from tests.gateway.conftest import ADMIN, MODEL, InProcessService, auth, gateway, masked_settings

CHAT = {"model": MODEL, "temperature": 0, "max_tokens": 6, "ignore_eos": True, "logprobs": True, "top_logprobs": 2}
PROMPTS = ["Hello", "How are you?", "Tell me a story", "abc", "Why is the sky blue?", "Count to five"]


def available(store: Any, pack: Any) -> int:
    return sum(i.slots_available for i in store.inventory(pack.fingerprint, DEFAULT_RING.name))


def test_config_default_off() -> None:
    cfg = GatewayModelConfig(name="m", pack_path=Path("/nonexistent"), mode="plain")
    assert cfg.engine_pingpong is False and cfg.engine_pingpong_min_sessions == 5


async def test_pingpong_gateway_exact_and_same_masks(tiny_pack_dir: Path, tmp_path: Path, gw_pack: Any,
                                                     make_store: Any) -> None:
    store = make_store(900)
    svc = InProcessService(gw_pack)
    keys = ApiKeyStore(None)
    key = keys.create("app")[1]
    settings = masked_settings(tiny_pack_dir, tmp_path, model_kw={"engine_pingpong": True})
    app = create_app(settings, mask_store=store, transports={MODEL: svc.transport}, key_store=keys)
    async with gateway(app) as c:
        def req(p: str) -> dict[str, Any]:
            return {**CHAT, "messages": [{"role": "user", "content": p}]}

        a0 = available(store, gw_pack)
        seq = [(await c.post("/v1/chat/completions", json=req(p), headers=auth(key))).json() for p in PROMPTS]
        a1 = available(store, gw_pack)
        st0 = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]["engine"]
        assert st0["pingpong_steps"] == 0  # one request at a time: nothing to split
        conc = await asyncio.gather(*(c.post("/v1/chat/completions", json=req(p), headers=auth(key))
                                      for p in PROMPTS))
        a2 = available(store, gw_pack)
        st1 = (await c.get("/admin/status", headers=ADMIN)).json()["models"][0]["engine"]
    assert st1["pingpong_steps"] > 0
    for a, b in zip(seq, conc, strict=True):
        assert b.status_code == 200
        bj = b.json()
        assert a["choices"][0]["message"] == bj["choices"][0]["message"]
        assert a["choices"][0]["logprobs"] == bj["choices"][0]["logprobs"]  # bit-identical logits
    assert a0 - a1 == a1 - a2  # the same mask slots per request, split or not
