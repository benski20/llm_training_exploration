"""Fixtures of the ``pgw chat`` tests: a :class:`ChatServer` on a free loopback port with the fake backend."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from pgw.chat.server import ChatServer
from pgw.chat.store import DiskStore
from pgw.chat.testing import FakeBackend

MODEL = "qwen2.5-0.5b-instruct"


class Running:
    def __init__(self, server: ChatServer, backend: Any):
        self.server = server
        self.backend = backend
        self.base = f"http://127.0.0.1:{server.port}"
        self.origin = self.base
        self.http = httpx.Client(base_url=self.base, timeout=30, trust_env=False)

    def h(self, **extra: str) -> dict[str, str]:
        """The headers the page sends."""
        return {"Authorization": f"Bearer {self.server.token}", "Origin": self.origin,
                "Content-Type": "application/json", **extra}

    def chat(self, **body: Any) -> list[dict[str, Any]]:
        r = self.http.post("/api/chat", headers=self.h(), json={"model": MODEL, **body}, timeout=120)
        assert r.status_code == 200, r.text
        return parse_sse(r.text)


def parse_sse(raw: str) -> list[dict[str, Any]]:
    out = []
    for block in raw.split("\n\n"):
        data = [ln[6:] for ln in block.split("\n") if ln.startswith("data: ")]
        if data:
            out.append(json.loads("\n".join(data)))
    return out


def text_of(events: list[dict[str, Any]]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "delta")


@pytest.fixture
def running(tmp_path: Path) -> Iterator[Running]:
    backend = FakeBackend(delay=0.001)
    server = ChatServer(backend, DiskStore(tmp_path / "state" / "chat" / "conversations"), heartbeat_s=0.2,
                        history_dir=tmp_path / "state" / "chat" / "conversations", default_model=MODEL,
                        static_dir=tmp_path / "static").start()
    (tmp_path / "static" / "assets").mkdir(parents=True)
    (tmp_path / "static" / "index.html").write_text("<!doctype html><title>PGW Chat</title>")
    (tmp_path / "static" / "assets" / "app-abcdef12.js").write_text("console.log(1)")
    r = Running(server, backend)
    try:
        yield r
    finally:
        r.http.close()
        server.shutdown()
