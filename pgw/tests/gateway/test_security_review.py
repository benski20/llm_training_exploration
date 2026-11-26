"""Regression tests of the 2026-11 security review (``research/reviews/security-review-2026-11.md``), customer gateway:
its own JSON routes stop reading a body at their limit, chunked bodies included (S13)."""

from __future__ import annotations

from pathlib import Path

from pgw.gateway import create_app
from pgw.gateway.setup import create_setup_app
from tests.gateway.conftest import ADMIN_TOKEN, plain_settings
from tests.gateway.test_setup import SETUP_TOKEN, _settings
from tests.service.test_security_review import _endless_chunked

_LIMIT = 64 << 10
_CHUNK = 16 << 10


async def test_setup_wizard_does_not_buffer_chunked_bodies(tmp_path: Path) -> None:
    """The setup wizard's sign-in takes no credential: before the fix it read the whole body, then compared it."""
    app = create_setup_app(_settings(tmp_path), setup_token=SETUP_TOKEN)
    try:
        status, pulled = await _endless_chunked(app, "/console/api/setup/login")
        assert status == 413 and pulled <= _LIMIT + _CHUNK, pulled
    finally:
        await app.aclose()


async def test_gateway_admin_and_console_routes_bound_their_bodies(tmp_path: Path, tiny_pack_dir: Path) -> None:
    app = create_app(plain_settings(tiny_pack_dir, tmp_path, console_enabled=True))
    admin = [(b"authorization", f"Bearer {ADMIN_TOKEN}".encode())]
    async with app.router.lifespan_context(app):
        for path, hdr, limit in (("/admin/keys", admin, _LIMIT), ("/admin/models/tiny/reset", admin, _LIMIT),
                                 ("/console/api/login", [], 16 << 10)):
            status, pulled = await _endless_chunked(app, path, headers=hdr)
            assert status == 413, (path, status)
            assert pulled <= limit + _CHUNK, (path, pulled)  # not the gateway-wide 4 MiB API limit
