"""OpenAI-compatible API, auth, limits, admin and ops endpoints (dev 'plain' mode, no service).

The masked production path is covered in ``test_masked.py``; the HTTP surface is identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pgw.gateway import ApiKeyStore, QuotaTracker, RateLimiter, create_app
from tests.gateway.conftest import ADMIN, MODEL, auth, gateway, parse_sse, plain_settings

CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}], "temperature": 0, "max_tokens": 6}


@pytest.fixture
def keys() -> ApiKeyStore:
    return ApiKeyStore(None)


@pytest.fixture
def key(keys: ApiKeyStore) -> str:
    return keys.create("app")[1]


def make_app(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, **kw: Any) -> Any:
    extra = {k: kw.pop(k) for k in ("rate_limiter", "quotas") if k in kw}
    return create_app(plain_settings(tiny_pack_dir, tmp_path, **kw), key_store=keys, **extra)


# ---- chat ------------------------------------------------------------------------------------------
async def test_chat_completion_shape(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "chat.completion" and body["model"] == MODEL
        assert body["id"].startswith("chatcmpl-") and r.headers["x-request-id"] == body["id"]
        assert len(body["system_fingerprint"]) == 12
        ch = body["choices"][0]
        assert ch["index"] == 0 and ch["message"]["role"] == "assistant" and isinstance(ch["message"]["content"], str)
        assert ch["finish_reason"] in ("stop", "length") and ch["logprobs"] is None
        u = body["usage"]
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and 0 < u["completion_tokens"] <= 6
        # deterministic greedy decoding
        r2 = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r2.json()["choices"][0]["message"] == ch["message"]


async def test_chat_stream_matches_non_stream(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore,
                                              key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        full = (await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))).json()
        r = await c.post("/v1/chat/completions", headers=auth(key),
                         json={**CHAT, "stream": True, "stream_options": {"include_usage": True}})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(r.text)
        assert events[-1] == "[DONE]"
        chunks = events[:-1]
        assert all(e["object"] == "chat.completion.chunk" and e["id"] == chunks[0]["id"] for e in chunks)
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        text = "".join(e["choices"][0]["delta"].get("content", "") for e in chunks if e["choices"])
        assert text == full["choices"][0]["message"]["content"]
        finals = [e["choices"][0]["finish_reason"] for e in chunks if e["choices"] and e["choices"][0]["finish_reason"]]
        assert finals == [full["choices"][0]["finish_reason"]]
        assert chunks[-1]["choices"] == [] and chunks[-1]["usage"] == full["usage"]
        assert all(e["usage"] is None for e in chunks[:-1])


async def test_chat_stream_without_usage(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True}, headers=auth(key))
        events = parse_sse(r.text)
        assert events[-1] == "[DONE]" and all("usage" not in e for e in events[:-1])


async def test_chat_logprobs(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "logprobs": True, "top_logprobs": 3,
                                                       "ignore_eos": True}, headers=auth(key))
        ch = r.json()["choices"][0]
        content = ch["logprobs"]["content"]
        assert len(content) == r.json()["usage"]["completion_tokens"]
        for item in content:
            assert item["logprob"] <= 0 and len(item["top_logprobs"]) == 3
            assert item["top_logprobs"][0]["logprob"] >= item["logprob"] - 1e-6  # greedy picks the top token
        assert "".join(i["token"] for i in content) == ch["message"]["content"]
        rs = await c.post("/v1/chat/completions", json={**CHAT, "logprobs": True, "ignore_eos": True,
                                                        "stream": True}, headers=auth(key))
        lps = [e["choices"][0]["logprobs"] for e in parse_sse(rs.text)[:-1] if e["choices"][0]["logprobs"]]
        assert [x["content"][0]["logprob"] for x in lps] == pytest.approx([i["logprob"] for i in content])


async def test_chat_content_parts_stop_and_sampling(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore,
                                                    key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        parts = {**CHAT, "messages": [{"role": "developer", "content": "Be brief"},
                                      {"role": "user", "content": [{"type": "text", "text": "Hel"},
                                                                   {"type": "text", "text": "lo"}]}]}
        assert (await c.post("/v1/chat/completions", json=parts, headers=auth(key))).status_code == 200
        img = {**CHAT, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}
        r = await c.post("/v1/chat/completions", json=img, headers=auth(key))
        assert r.status_code == 400 and r.json()["error"]["code"] == "protocol"
        sampled = {**CHAT, "temperature": 0.9, "top_p": 0.9, "seed": 7, "max_tokens": 5, "ignore_eos": True}
        a = (await c.post("/v1/chat/completions", json=sampled, headers=auth(key))).json()
        b = (await c.post("/v1/chat/completions", json=sampled, headers=auth(key))).json()
        assert a["choices"][0]["message"] == b["choices"][0]["message"]  # seeded
        base = (await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 12, "ignore_eos": True},
                             headers=auth(key))).json()["choices"][0]["message"]["content"]
        if len(base) >= 3:
            stop = base[1:3]
            r = (await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 12, "ignore_eos": True,
                                                            "stop": [stop]}, headers=auth(key))).json()
            assert r["choices"][0]["finish_reason"] == "stop" and stop not in r["choices"][0]["message"]["content"]


async def test_max_tokens_clamped_to_context_and_limits(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore,
                                                        key: str) -> None:
    app = make_app(tiny_pack_dir, tmp_path, keys, max_prompt_tokens=50, model_kw={"max_context": 40})
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 500, "ignore_eos": True},
                         headers=auth(key))
        assert r.status_code == 200
        u = r.json()["usage"]
        assert u["prompt_tokens"] + u["completion_tokens"] <= 40 and r.json()["choices"][0]["finish_reason"] == "length"
        long = {**CHAT, "messages": [{"role": "user", "content": "x" * 45}]}
        r = await c.post("/v1/chat/completions", json=long, headers=auth(key))
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_shape"  # docs/api section 6
        r = await c.post("/v1/chat/completions", json={**CHAT, "max_tokens": 100_000}, headers=auth(key))
        assert r.status_code == 400 and r.json()["error"]["code"] == "protocol"  # rejected, not clamped


# ---- completions --------------------------------------------------------------------------------------
async def test_completions_echo_list_logprobs(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore,
                                             key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        req = {"model": MODEL, "prompt": "Once upon", "max_tokens": 5, "temperature": 0, "ignore_eos": True}
        r = await c.post("/v1/completions", json=req, headers=auth(key))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "text_completion" and body["id"].startswith("cmpl-")
        text = body["choices"][0]["text"]
        echo = (await c.post("/v1/completions", json={**req, "echo": True}, headers=auth(key))).json()
        assert echo["choices"][0]["text"] == "Once upon" + text
        multi = (await c.post("/v1/completions", json={**req, "prompt": ["Once upon", "abc"]},
                              headers=auth(key))).json()
        assert [ch["index"] for ch in multi["choices"]] == [0, 1]
        assert multi["choices"][0]["text"] == text
        single_b = (await c.post("/v1/completions", json={**req, "prompt": "abc"}, headers=auth(key))).json()
        assert multi["choices"][1]["text"] == single_b["choices"][0]["text"]
        assert multi["usage"]["prompt_tokens"] == body["usage"]["prompt_tokens"] + single_b["usage"]["prompt_tokens"]
        lp = (await c.post("/v1/completions", json={**req, "logprobs": 2}, headers=auth(key))).json()
        lpo = lp["choices"][0]["logprobs"]
        assert len(lpo["tokens"]) == len(lpo["token_logprobs"]) == len(lpo["top_logprobs"]) == 5
        assert "".join(lpo["tokens"]) == text and lpo["text_offset"][0] == 0
        assert all(len(t) <= 2 for t in lpo["top_logprobs"])


async def test_completions_stream(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        req = {"model": MODEL, "prompt": "Once upon", "max_tokens": 5, "temperature": 0, "ignore_eos": True}
        full = (await c.post("/v1/completions", json=req, headers=auth(key))).json()
        r = await c.post("/v1/completions", json={**req, "stream": True, "echo": True}, headers=auth(key))
        ev = parse_sse(r.text)
        assert ev[-1] == "[DONE]"
        text = "".join(e["choices"][0]["text"] for e in ev[:-1])
        assert text == "Once upon" + full["choices"][0]["text"]
        assert ev[-2]["choices"][0]["finish_reason"] == "length"
        r = await c.post("/v1/completions", json={**req, "stream": True, "prompt": ["a", "b"]}, headers=auth(key))
        assert r.status_code == 400
        r = await c.post("/v1/completions", json={**req, "suffix": "x"}, headers=auth(key))
        assert r.status_code == 400


# ---- auth / limits ----------------------------------------------------------------------------------------
async def test_auth_errors(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    _rec_admin_only, admin_only = keys.create("ops", scopes=("admin",))
    _, restricted = keys.create("restricted", models=("other",))
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        r = await c.post("/v1/chat/completions", json=CHAT)
        assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
        assert r.json()["error"]["code"] == "unauthenticated"
        r = await c.post("/v1/chat/completions", json=CHAT, headers={"Authorization": "Basic abc"})
        assert r.status_code == 401
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key[:-2] + "zz"))
        assert r.status_code == 401
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(admin_only))
        assert r.status_code == 403 and r.json()["error"]["code"] == "permission_denied"
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(restricted))
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
        r = await c.get("/v1/models", headers=auth(restricted))
        assert r.json()["data"] == []
        r = await c.post("/v1/chat/completions", json={**CHAT, "model": "nope"}, headers=auth(key))
        assert r.status_code == 404
        models = (await c.get("/v1/models", headers=auth(key))).json()
        assert models["object"] == "list" and [m["id"] for m in models["data"]] == [MODEL]
        assert (await c.get(f"/v1/models/{MODEL}", headers=auth(key))).json()["id"] == MODEL
        # admin endpoints: admin token or admin-scoped key, not inference keys
        assert (await c.get("/admin/status")).status_code == 401
        assert (await c.get("/admin/status", headers=auth(key))).status_code == 403
        assert (await c.get("/admin/status", headers=auth(admin_only))).status_code == 200
        assert (await c.get("/admin/status", headers={"Authorization": "Bearer wrong"})).status_code == 401
    audit = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    failed = [a for a in audit if a["event"] == "auth.failed"]
    assert len(failed) >= 5 and all("pgw-" not in json.dumps(a) for a in audit)


async def test_validation_errors(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        for bad in ({**CHAT, "n": 2}, {**CHAT, "temperature": 5}, {**CHAT, "messages": []},
                    {"messages": CHAT["messages"]}, {**CHAT, "stop": [""]}):
            r = await c.post("/v1/chat/completions", json=bad, headers=auth(key))
            assert r.status_code == 400, bad
            err = r.json()["error"]
            assert err["type"] == "invalid_request_error" and err["code"] == "protocol"
        r = await c.post("/v1/chat/completions", content=b"{not json", headers={**auth(key),
                                                                            "Content-Type": "application/json"})
        assert r.status_code == 400


async def test_body_limit(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys, max_body_bytes=2048)) as c:
        big = {**CHAT, "messages": [{"role": "user", "content": "x" * 5000}]}
        r = await c.post("/v1/chat/completions", json=big, headers=auth(key))
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"

        async def chunked() -> Any:  # no Content-Length: counted while streaming
            yield json.dumps(big).encode()[:1500]
            yield json.dumps(big).encode()[1500:]

        r = await c.post("/v1/chat/completions", content=chunked(),
                         headers={**auth(key), "Content-Type": "application/json"})
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"
        assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))).status_code == 200


async def test_rate_limit_429_with_retry_after(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore) -> None:
    _, k = keys.create("slow", requests_per_minute=2)
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        for _ in range(2):
            assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(k))).status_code == 200
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(k))
        assert r.status_code == 429 and r.json()["error"]["type"] == "rate_limit_error"
        assert r.json()["error"]["code"] == "rate_limited" and int(r.headers["Retry-After"]) >= 1


async def test_token_rate_limit(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    rl = RateLimiter(1000, 40)  # one ~27-token request fits, two do not
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys, rate_limiter=rl)) as c:
        assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))).status_code == 200
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 429 and "tokens" in r.json()["error"]["message"]


async def test_daily_quota(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore) -> None:
    _, k = keys.create("q", daily_token_quota=40)  # one ~27-token request fits
    quotas = QuotaTracker(None)
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys, quotas=quotas)) as c:
        assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(k))).status_code == 200
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(k))
        assert r.status_code == 429 and r.json()["error"]["type"] == "insufficient_quota"
        listing = (await c.get("/admin/keys", headers=ADMIN)).json()["keys"]
        assert next(x["tokens_today"] for x in listing if x["name"] == "q") > 0


async def test_per_key_concurrency_limit(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore) -> None:
    import asyncio

    _, k = keys.create("one-at-a-time", max_concurrent_requests=1)
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        req = {**CHAT, "max_tokens": 30, "ignore_eos": True}
        rs = await asyncio.gather(*(c.post("/v1/chat/completions", json=req, headers=auth(k)) for _ in range(3)))
        codes = sorted(r.status_code for r in rs)
        assert codes[0] == 200 and 429 in codes
        assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(k))).status_code == 200


async def test_anonymous_dev_access(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys, allow_anonymous=True)) as c:
        assert (await c.post("/v1/chat/completions", json=CHAT)).status_code == 200


# ---- admin / ops --------------------------------------------------------------------------------------------
async def test_admin_key_lifecycle(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        r = await c.post("/admin/keys", headers=ADMIN, json={"name": "svc", "models": [MODEL],
                                                             "requests_per_minute": 100})
        assert r.status_code == 201
        new_key, rec = r.json()["key"], r.json()["record"]
        assert "hashed" not in rec and rec["models"] == [MODEL]
        assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(new_key))).status_code == 200
        listing = (await c.get("/admin/keys", headers=ADMIN)).json()["keys"]
        assert [x["key_id"] for x in listing] == [rec["key_id"]] and new_key not in json.dumps(listing)
        assert (await c.delete(f"/admin/keys/{rec['key_id']}", headers=ADMIN)).status_code == 200
        assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(new_key))).status_code == 401
        assert (await c.delete("/admin/keys/0123456789ab", headers=ADMIN)).status_code == 404
        assert (await c.post("/admin/keys", headers=ADMIN, json={"name": "x", "scopes": ["root"]})).status_code == 400
        assert (await c.post("/admin/keys", headers=ADMIN, json={"nam": "x"})).status_code == 400
    audit = (tmp_path / "audit.jsonl").read_text()
    assert "admin.key_created" in audit and "admin.key_revoked" in audit and new_key not in audit


async def test_ops_endpoints(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys)) as c:
        assert (await c.get("/healthz")).json() == {"status": "ok"}
        ready = await c.get("/readyz")
        assert ready.status_code == 200 and ready.json()["models"][MODEL]["ready"]
        await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        m = (await c.get("/metrics")).text
        assert 'pgw_gateway_requests_total{endpoint="chat",model="tiny",status="200"} 1.0' in m
        assert "pgw_gateway_ttft_seconds_count" in m and "_created" not in m
        names = {line.split("{")[0].split(" ")[0] for line in m.splitlines() if line and not line.startswith("#")}
        from pgw.observability.metrics import METRIC_NAMES

        bases = {n.removesuffix("_total") for n in METRIC_NAMES}
        for n in names:
            assert any(n == b or n.startswith(b + "_") or n == b + "_total" for b in bases), n
        d = await c.get("/dashboard")
        assert d.status_code == 200 and "<html" in d.text.lower() and "http" not in d.text.split("<script>")[0][:20]
        st = (await c.get("/admin/status", headers=ADMIN)).json()
        assert st["models"][0]["mode"] == "plain" and st["production_policy"] is False
        assert st["telemetry"] == "disabled"
        assert (await c.get("/admin/masks", headers=ADMIN)).json() == {"store": None, "models": {}}
        r = await c.post(f"/admin/models/{MODEL}/reload", headers=ADMIN)
        assert r.status_code == 200 and r.json()["ready"]
        assert (await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))).status_code == 200
        assert (await c.post("/admin/models/nope/reload", headers=ADMIN)).status_code == 404
        assert (await c.get("/docs")).status_code == 200  # dev mode only


async def test_metrics_can_require_admin(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore) -> None:
    async with gateway(make_app(tiny_pack_dir, tmp_path, keys, metrics_require_admin=True)) as c:
        assert (await c.get("/metrics")).status_code == 401
        assert (await c.get("/metrics", headers=ADMIN)).status_code == 200


async def test_float_mode(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    app = create_app(plain_settings(tiny_pack_dir, tmp_path, mode="float"), key_store=keys)
    async with gateway(app) as c:
        r = await c.post("/v1/chat/completions", json=CHAT, headers=auth(key))
        assert r.status_code == 200


def test_prod_settings_refuse_plain_mode(tiny_pack_dir: Path, tmp_path: Path) -> None:
    from pgw.core import ConfigError

    with pytest.raises(ConfigError, match="development"):
        create_app(plain_settings(tiny_pack_dir, tmp_path, allow_dev_modes=False))


async def test_queue_full_returns_503(tiny_pack_dir: Path, tmp_path: Path, keys: ApiKeyStore, key: str) -> None:
    import asyncio

    app = make_app(tiny_pack_dir, tmp_path, keys,
                   model_kw={"max_concurrent_sessions": 1, "max_queue": 1, "queue_timeout_s": 60})
    async with gateway(app) as c:
        req = {**CHAT, "max_tokens": 40, "ignore_eos": True}
        rs = await asyncio.gather(*(c.post("/v1/chat/completions", json=req, headers=auth(key)) for _ in range(4)))
        codes = sorted(r.status_code for r in rs)
        assert codes.count(200) >= 2 and 503 in codes
        bad = next(r for r in rs if r.status_code == 503)
        assert bad.json()["error"]["code"] == "unavailable" and "Retry-After" in bad.headers
