"""Static console app: SPA fallback, caching, security headers, path traversal."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.console.conftest import client, login, make_console_app
from tests.gateway.conftest import ADMIN

INDEX = "<!doctype html><title>PGW Console</title><script src=\"/console/assets/app-3f2a9c.js\"></script>"


@pytest.fixture
def static_dir(tmp_path: Path) -> Path:
    d = tmp_path / "static"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(INDEX, encoding="utf-8")
    (d / "assets" / "app-3f2a9c.js").write_text("console.log(1)", encoding="utf-8")
    (d / "assets" / "app-3f2a9c.css").write_text("body{}", encoding="utf-8")
    (d / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (d / ".hidden").write_text("nope", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOP-SECRET", encoding="utf-8")
    return d


def _security(headers: object) -> None:
    h = headers  # httpx.Headers
    csp = h["content-security-policy"]  # type: ignore[index]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "script-src 'self'" in csp
    assert h["x-frame-options"] == "DENY" and h["referrer-policy"] == "no-referrer"  # type: ignore[index]
    assert h["x-content-type-options"] == "nosniff"  # type: ignore[index]


async def test_index_assets_and_fallback(tiny_pack_dir: Path, tmp_path: Path, static_dir: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path, static_dir=static_dir)) as c:
        r = await c.get("/console", follow_redirects=False)
        assert r.status_code == 308 and r.headers["location"] == "/console/"
        r = await c.get("/console/")
        assert r.status_code == 200 and r.text == INDEX and r.headers["content-type"].startswith("text/html")
        assert r.headers["cache-control"] == "no-cache"
        _security(r.headers)
        r = await c.get("/console/assets/app-3f2a9c.js")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/javascript")
        assert r.headers["cache-control"] == "public, max-age=31536000, immutable"
        _security(r.headers)
        assert (await c.get("/console/assets/app-3f2a9c.css")).headers["content-type"].startswith("text/css")
        r = await c.get("/console/favicon.svg")
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
        # client-side routes get index.html
        for route in ("/console/keys", "/console/usage/24h", "/console/index.html"):
            r = await c.get(route)
            assert r.status_code == 200 and r.text == INDEX, route
        # missing assets are 404, not the app
        for missing in ("/console/assets/app-000000.js", "/console/assets/x", "/console/missing.png",
                        "/console/.hidden"):
            r = await c.get(missing)
            assert r.status_code == 404, missing
            _security(r.headers)
        # unknown API paths are JSON 404s, not the app
        r = await c.get("/console/api/nope")
        assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"
        assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("path", [
    "/console/%2e%2e/secret.txt",
    "/console/%2E%2E/%2E%2E/pyproject.toml",
    "/console/assets/%2e%2e/%2e%2e/secret.txt",
    "/console/..%2fsecret.txt",
    "/console/%2e%2e%2fsecret.txt",
    "/console/assets/..%5c..%5csecret.txt",
    "/console/%2fetc%2fpasswd",
    "/console//etc/passwd",
])
async def test_path_traversal(tiny_pack_dir: Path, tmp_path: Path, static_dir: Path, path: str) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path, static_dir=static_dir)) as c:
        r = await c.get(path)
        assert "TOP-SECRET" not in r.text and "root:" not in r.text
        assert r.status_code in (404, 200)
        if r.status_code == 200:  # only ever the app shell
            assert r.text == INDEX


async def test_raw_dotdot_path(tiny_pack_dir: Path, tmp_path: Path, static_dir: Path) -> None:
    """A raw ``..`` segment reaching the app (clients normally collapse it) is refused."""
    app = make_console_app(tiny_pack_dir, tmp_path, static_dir=static_dir)
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg: dict) -> None:
        sent.append(msg)

    scope = {"type": "http", "method": "GET", "path": "/console/../secret.txt", "raw_path": b"/console/../secret.txt",
             "query_string": b"", "headers": [(b"host", b"127.0.0.1")], "scheme": "http",
             "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80), "http_version": "1.1", "root_path": ""}
    await app(scope, receive, send)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert sent[0]["status"] == 404 and b"TOP-SECRET" not in body


async def test_not_built(tiny_pack_dir: Path, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    async with client(make_console_app(tiny_pack_dir, tmp_path, static_dir=empty)) as c:
        for path in ("/console/", "/console/keys"):
            r = await c.get(path)
            assert r.status_code == 200 and "not built" in r.text
            _security(r.headers)


async def test_api_and_admin_headers(tiny_pack_dir: Path, tmp_path: Path) -> None:
    async with client(make_console_app(tiny_pack_dir, tmp_path)) as c:
        await login(c)
        r = await c.get("/console/api/me")
        _security(r.headers)
        assert r.headers["cache-control"] == "no-store"
        r = await c.get("/admin/status", headers=ADMIN)
        assert r.headers["cache-control"] == "no-store"
        r = await c.get("/healthz")
        assert "content-security-policy" not in r.headers
