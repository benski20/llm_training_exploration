"""Load-test tool, dependency rule and statelessness checks."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pgw.service.loadtest import LoadResult, main, run_load
from tests.service.conftest import asgi_client, running_app

SRC = Path(__file__).resolve().parents[2] / "src" / "pgw" / "service"


async def test_run_load_in_process_verified(tiny_pack_dir: Path, pack_fp: str,
                                            pack_weights: dict[str, torch.Tensor]) -> None:
    from pgw.service import ServiceSettings

    s = ServiceSettings(packs=[tiny_pack_dir], device="cpu", verify_packs="fingerprint",
                        allow_unauthenticated=True, batch_window_ms=5.0)
    async with running_app(s) as app:
        c = asgi_client(app)
        res = await run_load(c, fingerprint=pack_fp, ring="K48A20", sessions=6, tokens=2, rows=2,
                             reference=pack_weights)
        await c.aclose()
    assert isinstance(res, LoadResult)
    assert res.errors == 0 and res.mismatches == 0
    assert res.calls == 6 * 2 * 8
    assert res.rows_per_s > 0 and res.latency_ms_p50 > 0 and res.latency_ms_p99 >= res.latency_ms_p50
    assert res.server_batch_rows_max >= 2
    assert res.bytes_up > 0 and res.bytes_down > 0
    json.loads(res.to_json())


@pytest.mark.slow
def test_loadtest_cli_self_contained(tiny_pack_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--pack", str(tiny_pack_dir), "--sessions", "4", "--tokens", "1", "--layers", "1", "--verify",
               "--device", "cpu"])
    out = capsys.readouterr().out
    assert rc == 0, out
    res = json.loads(out)
    assert res["calls"] == 4 * 4 and res["mismatches"] == 0 and res["errors"] == 0


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def test_service_never_imports_maskstore_factory_engine_gateway() -> None:
    forbidden = ("pgw.maskstore", "pgw.factory", "pgw.engine", "pgw.gateway")
    for f in SRC.glob("*.py"):
        for m in _imports(f):
            assert not m.startswith(forbidden), f"{f.name} imports {m}"
    code = (
        "import sys, pgw.service, pgw.service.loadtest, pgw.service.server;"
        "bad=[m for m in sys.modules if m.startswith(('pgw.maskstore','pgw.factory','pgw.gateway','pgw.engine'))];"
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)  # noqa: S603
    assert r.returncode == 0, f"transitively imported: {r.stdout} {r.stderr[-500:]}"


def test_no_payload_logging_calls_in_source() -> None:
    """Static guard: logging calls in the service never reference payload variables."""
    payload_names = {"x_hat", "y_hat", "payload", "body", "data", "x", "y"}
    for f in SRC.glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "log"):
                for arg in [*node.args, *(k.value for k in node.keywords)]:
                    for sub in ast.walk(arg):
                        if isinstance(sub, ast.Name):
                            assert sub.id not in payload_names, f"{f.name}:{node.lineno} logs {sub.id}"


async def test_service_is_stateless_between_requests(tiny_pack_dir: Path, pack_fp: str) -> None:
    """Identical requests give identical answers and leave no per-request state behind."""
    from pgw.core import RING_K40_A12
    from pgw.service import ServiceSettings
    from tests.service.conftest import make_req, rand_x

    s = ServiceSettings(packs=[tiny_pack_dir], device="cpu", verify_packs="fingerprint", allow_unauthenticated=True)
    async with running_app(s) as app:
        c = asgi_client(app)
        req = make_req(pack_fp, RING_K40_A12, [("0.qkv", rand_x(3, 64, RING_K40_A12))])
        a = await c.compute(req)
        b = await c.compute(req)  # a retry resends identical x_hat
        assert a.items[0].y_hat == b.items[0].y_hat
        st = app.state.pgw
        assert st.batcher.queued_rows == 0 and st.batcher.queued_items == 0 and st.handler.inflight == 0
        assert not st.batcher._open  # no batches retained
        await c.aclose()
