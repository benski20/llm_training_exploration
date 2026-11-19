"""Console test helpers: an app in dev 'plain' mode, a client on a loopback origin, sign-in."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx

from pgw.gateway import ApiKeyStore, create_app
from tests.gateway.conftest import ADMIN_TOKEN, plain_settings

VIEWER_TOKEN = "viewer-token-abcdefghijklmnop"  # noqa: S105 - test fixture
LOCAL = "http://127.0.0.1:8080"


def make_console_app(tiny_pack_dir: Path, tmp_path: Path, *, static_dir: Path | None = None,
                     keys: ApiKeyStore | None = None, viewer: bool = True, **kw: Any) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if viewer and "console_viewer_tokens_file" not in kw:
        vf = tmp_path / "viewers.txt"
        vf.write_text(f"# console viewers\n{VIEWER_TOKEN}\n", encoding="utf-8")
        kw["console_viewer_tokens_file"] = vf
    if static_dir is None:
        static_dir = tmp_path / "no-static"
    return create_app(plain_settings(tiny_pack_dir, tmp_path, **kw), key_store=keys or ApiKeyStore(None),
                      console_static_dir=static_dir)


@contextlib.asynccontextmanager
async def client(app: Any, base_url: str = LOCAL) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url=base_url, timeout=120) as c:
            yield c


async def login(c: httpx.AsyncClient, token: str = ADMIN_TOKEN) -> str:
    """Sign in; returns the CSRF token (the session cookie stays in the client's jar)."""
    r = await c.post("/console/api/login", json={"token": token})
    assert r.status_code == 200, r.text
    return str(r.json()["csrf"])


def csrf(token: str) -> dict[str, str]:
    return {"X-PGW-CSRF": token}
