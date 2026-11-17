"""Gateway docs match the code: routes (SPEC 9.1, ``create_app`` docstring) and request limits."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _routes(tiny_pack_dir: Path, tmp_path: Path) -> set[tuple[str, str]]:
    from pgw.gateway import ApiKeyStore, create_app
    from tests.gateway.conftest import plain_settings

    app = create_app(plain_settings(tiny_pack_dir, tmp_path), key_store=ApiKeyStore(None))
    out = set()
    for r in app.routes:
        path = getattr(r, "path", "")
        if path.startswith(("/v1/", "/admin/")):
            for m in getattr(r, "methods", ()) or ():
                if m in ("GET", "POST", "DELETE"):
                    out.add((m, path))
    return out


def test_every_route_is_documented(tiny_pack_dir: Path, tmp_path: Path) -> None:
    from pgw.gateway.app import create_app

    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
    section = spec[spec.index("### 9.1"):spec.index("### 9.2")]
    doc = create_app.__doc__ or ""
    routes = _routes(tiny_pack_dir, tmp_path)
    assert ("GET", "/v1/models/{name}") in routes and ("POST", "/admin/models/{name}/reset") in routes
    for _method, path in sorted(routes):
        assert f"{path}`" in section, f"SPEC 9.1 misses {path}"
        assert f"{path}``" in doc, f"create_app docstring misses {path}"


def test_max_tokens_and_prompt_length_rows_match_code() -> None:
    api = (ROOT / "docs/api/gateway-api.md").read_text(encoding="utf-8")
    row = next(line for line in api.splitlines() if line.startswith("| `max_completion_tokens` / `max_tokens`"))
    assert "rejected (400 `protocol`) above `max_tokens_limit`" in row and "clamped to the context left" in row
    bad_shape = next(line for line in api.splitlines() if line.startswith("| `bad_shape`"))
    assert re.search(r"max_prompt_tokens.*context", bad_shape)
    # the code raises exactly those errors (ShapeError -> bad_shape, InvalidRequestError -> protocol)
    import inspect

    from pgw.gateway.runtime import ModelRuntime

    src = inspect.getsource(ModelRuntime.clamp_max_tokens)
    assert src.count("raise ShapeError") == 2 and "max_tokens_limit" in src
