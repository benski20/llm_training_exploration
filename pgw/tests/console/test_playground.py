"""Console playground: same request/response as /v1/chat/completions, run as ``console-playground``."""

from __future__ import annotations

import json
from pathlib import Path

from pgw.gateway import ApiKeyStore
from tests.console.conftest import client, csrf, login, make_console_app
from tests.gateway.conftest import ADMIN, MODEL, auth, parse_sse

CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], "temperature": 0, "max_tokens": 6}


def _inference(tmp_path: Path) -> list[dict]:
    recs = [json.loads(ln) for ln in (tmp_path / "audit.jsonl").read_text().splitlines() if ln.strip()]
    return [r for r in recs if r["event"] == "inference"]


async def test_playground_matches_v1(tiny_pack_dir: Path, tmp_path: Path) -> None:
    keys = ApiKeyStore(None)
    _, key = keys.create("app")
    async with client(make_console_app(tiny_pack_dir, tmp_path, keys=keys)) as c:
        ref = (await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))).json()
        tok = await login(c)
        r = await c.post("/console/api/playground/chat", json=CHAT, headers=csrf(tok))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
        assert body["choices"][0]["message"] == ref["choices"][0]["message"] and body["usage"] == ref["usage"]
        assert r.headers["x-request-id"] == body["id"]

        r = await c.post("/console/api/playground/chat", headers=csrf(tok),
                         json={**CHAT, "stream": True, "stream_options": {"include_usage": True}})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.text)
        assert events[-1] == "[DONE]"
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in events[:-1] if e["choices"])
        assert text == ref["choices"][0]["message"]["content"]
        assert events[-2]["usage"] == ref["usage"]

        # bad bodies use the gateway's error shape
        r = await c.post("/console/api/playground/chat", json={"model": MODEL}, headers=csrf(tok))
        assert r.status_code == 400 and r.json()["error"]["code"] == "protocol"
        r = await c.post("/console/api/playground/chat", json={**CHAT, "model": "nope"}, headers=csrf(tok))
        assert r.status_code == 404
        # CSRF and a session are required
        assert (await c.post("/console/api/playground/chat", json=CHAT)).status_code == 403
        o = (await c.get("/admin/overview")).json()
        assert o["stream_requests_total"] == 1
        u = (await c.get("/admin/usage?window=1h")).json()
        assert u["by_key"]["console-playground"]["requests"] == 3
    keys2 = ApiKeyStore(None)
    _, key2 = keys2.create("app")
    c2 = make_console_app(tiny_pack_dir, tmp_path / "other", keys=keys2)
    async with client(c2) as c:
        assert (await c.post("/console/api/playground/chat", json=CHAT)).status_code == 401
        assert (await c.post("/console/api/playground/chat", json=CHAT, headers=auth(key2))).status_code == 403
        # API keys cannot use it; the admin token can (scripts)
        assert (await c.post("/console/api/playground/chat", json=CHAT, headers=ADMIN)).status_code == 200

    recs = _inference(tmp_path)
    play = [r for r in recs if r["key_id"] == "console-playground"]
    assert len(play) == 3 and [r["stream"] for r in play] == [False, True, False]
    assert play[0]["status"] == 200 and play[0]["completion_tokens"] > 0
    assert play[2]["status"] == 404


async def test_playground_is_quota_limited(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path, daily_token_quota=100)) as c:
        tok = await login(c)
        codes = []
        for _ in range(20):
            r = await c.post("/console/api/playground/chat", json=CHAT, headers=csrf(tok))
            codes.append(r.status_code)
        assert 200 in codes and codes[-1] == 429
        assert r.json()["error"]["type"] == "insufficient_quota"
