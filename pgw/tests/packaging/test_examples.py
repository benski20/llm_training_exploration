"""``examples/`` run against a stand-in for the client API (``pgw.OpenAI`` / ``pgw.AsyncOpenAI``).

The stand-in implements only the OpenAI-style surface the examples use (``chat.completions.create``,
attribute access on responses, sync and async streaming). It checks the examples' use of that
interface without a provider; the real client is tested with the SDK (``tests/sdk``).
"""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import pgw

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


class _Msg(BaseModel):
    role: str = "assistant"
    content: str | None = None


class _Choice(BaseModel):
    index: int = 0
    message: _Msg | None = None
    delta: _Msg | None = None
    finish_reason: str | None = None


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _Completion(BaseModel):
    id: str = "chatcmpl-test"
    object: str = "chat.completion"
    model: str
    choices: list[_Choice]
    usage: _Usage | None = None


ANSWER = ["Private ", "inference ", "works."]
calls: list[dict[str, Any]] = []


def _check(kw: dict[str, Any]) -> None:
    assert isinstance(kw["model"], str) and kw["model"].startswith("qwen2.5-")
    msgs = kw["messages"]
    assert msgs and all(set(m) == {"role", "content"} and m["role"] in ("system", "user", "assistant") for m in msgs)
    calls.append(kw)


def _full(kw: dict[str, Any]) -> _Completion:
    return _Completion(model=kw["model"], choices=[_Choice(message=_Msg(content="".join(ANSWER)), finish_reason="stop")],
                       usage=_Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8))


def _chunks(kw: dict[str, Any]) -> Iterator[_Completion]:
    for i, piece in enumerate(ANSWER):
        yield _Completion(object="chat.completion.chunk", model=kw["model"],
                          choices=[_Choice(delta=_Msg(content=piece), finish_reason="stop" if i == 2 else None)])


class _Completions:
    def create(self, **kw: Any) -> Any:
        _check(kw)
        return _chunks(kw) if kw.get("stream") else _full(kw)


class _AsyncCompletions:
    async def create(self, **kw: Any) -> Any:
        _check(kw)
        if not kw.get("stream"):
            return _full(kw)

        async def gen() -> AsyncIterator[_Completion]:
            for c in _chunks(kw):
                await asyncio.sleep(0)
                yield c

        return gen()


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class FakeOpenAI:
    def __init__(self, **kw: Any) -> None:
        self.chat = _Chat(_Completions())


class FakeAsyncOpenAI:
    def __init__(self, **kw: Any) -> None:
        self.chat = _Chat(_AsyncCompletions())

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pgw, "OpenAI", FakeOpenAI, raising=False)
    monkeypatch.setattr(pgw, "AsyncOpenAI", FakeAsyncOpenAI, raising=False)
    monkeypatch.setenv("PGW_API_KEY", "pgw_test_notarealkey")
    calls.clear()


@pytest.mark.parametrize("name", ["chat.py", "streaming.py", "async_chat.py"])
def test_script_examples(name: str, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [name])
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    out = capsys.readouterr().out
    assert "Private inference works." in out
    assert calls


def test_every_example_is_tested() -> None:
    assert {p.name for p in EXAMPLES.glob("*.py")} == {"chat.py", "streaming.py", "async_chat.py", "fastapi_app.py"}


def test_fastapi_app_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("APP_TOKEN", "t0k")
    mod = runpy.run_path(str(EXAMPLES / "fastapi_app.py"), run_name="fastapi_app")
    with TestClient(mod["app"]) as tc:
        assert tc.post("/ask", json={"question": "hi"}).status_code == 401
        auth = {"Authorization": "Bearer t0k"}
        r = tc.post("/ask", json={"question": "hi"}, headers=auth)
        assert r.status_code == 200 and r.json() == {"answer": "Private inference works."}

        # OpenAI wire format, non-streaming: what OpenAI-compatible HTTP clients parse
        body = {"model": "qwen2.5-0.5b-instruct", "messages": [{"role": "user", "content": "hi"}]}
        r = tc.post("/v1/chat/completions", json=body, headers=auth)
        d = r.json()
        assert r.status_code == 200 and d["object"] == "chat.completion"
        assert d["choices"][0]["message"]["content"] == "Private inference works." and d["usage"]["total_tokens"] == 8

        # streaming: server-sent events, one chunk per data line, then [DONE]
        r = tc.post("/v1/chat/completions", json={**body, "stream": True}, headers=auth)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = [ln[len("data: "):] for ln in r.text.splitlines() if ln.startswith("data: ")]
        assert events[-1] == "[DONE]"
        text = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
        assert text == "Private inference works."
        assert tc.get("/healthz").json()["status"] == "ok"
