"""End-to-end through the CLI only: pack -> mask store -> factory -> service -> gateway -> chat.

This is the docker-compose demo in miniature (``deploy/compose``): every step is a ``pgw`` command,
the service and the gateway run as real subprocesses on loopback ports, and a chat completion goes
through the masked path. The masked answer must be identical to the plaintext-quantized answer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

from pgw.cli import cli

from .conftest import free_port

pytestmark = pytest.mark.slow


def _ok(r: object) -> None:
    res = r  # click Result
    assert res.exit_code == 0, res.output  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def flow(tiny_hf_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Pack, store, segments, service token, gateway API key and rendered models file."""
    d = tmp_path_factory.mktemp("flow")
    r = CliRunner()
    _ok(r.invoke(cli, ["pack", str(tiny_hf_dir), str(d / "pack"), "--name", "tiny", "-q"]))
    fp = json.loads((d / "pack" / "manifest.json").read_text())["fingerprint"]
    res = r.invoke(cli, ["inspect", str(d / "pack"), "--verify", "full", "--json"])
    _ok(res)
    info = json.loads(res.stdout)
    assert info["fingerprint"] == fp and info["linears"] == 8
    _ok(r.invoke(cli, ["maskstore", "init", "--root", str(d / "masks"), "--key-file", str(d / "kek")]))
    _ok(r.invoke(cli, ["factory", "generate", "--pack", str(d / "pack"), "--store", str(d / "masks"),
                       "--key-file", str(d / "kek"), "--slots", "128", "--ring", "K40A12"]))
    _ok(r.invoke(cli, ["service", "token", "create", "--tenant", "acme", "--tokens-file", str(d / "tokens"),
                       "--out", str(d / "svc_token")]))
    res = r.invoke(cli, ["gateway", "keys", "create", "--name", "app", "--keys-file", str(d / "keys.json")])
    _ok(res)
    api_key = json.loads(res.stdout.strip().splitlines()[-1])["key"]
    svc_port, gw_port = free_port(), free_port()
    tmpl = [
        {"name": "tiny", "pack_path": str(d / "pack"), "fingerprint": fp, "ring": "K40A12",
         "service_url": f"http://127.0.0.1:{svc_port}", "service_token": "${file:svc_token}"},
        {"name": "tiny-plain", "pack_path": str(d / "pack"), "mode": "plain", "ring": "K40A12"},
    ]
    (d / "models.tmpl.json").write_text(json.dumps(tmpl))
    _ok(r.invoke(cli, ["config", "render", str(d / "models.tmpl.json"), str(d / "run" / "models.json")]))
    rendered = json.loads((d / "run" / "models.json").read_text())
    assert rendered[0]["service_token"].startswith("pgwt-")
    return {"dir": d, "fp": fp, "api_key": api_key, "svc_port": svc_port, "gw_port": gw_port}


def _wait_ready(url: str, proc: subprocess.Popen[bytes], timeout: float = 90.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise AssertionError(f"process exited with {proc.returncode}:\n{out[-4000:]}")
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise AssertionError(f"{url} not ready after {timeout}s")


def _spawn(args: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
    base = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    base.update(env)
    return subprocess.Popen([sys.executable, "-m", "pgw.cli.main", *args], env=base,  # noqa: S603
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
            proc.wait()


@pytest.fixture(scope="module")
def running(flow: dict[str, object]) -> Iterator[dict[str, object]]:
    d = flow["dir"]
    assert isinstance(d, Path)
    svc = _spawn(["service", "serve", "--host", "127.0.0.1", "--port", str(flow["svc_port"])], {
        "PGW_SERVICE_PACKS": json.dumps([str(d / "pack")]),
        "PGW_SERVICE_TOKENS_FILE": str(d / "tokens"),
        "PGW_SERVICE_RINGS": '["K40A12"]',
        "PGW_LOG_JSON": "true",
    })
    gw = None
    try:
        _wait_ready(f"http://127.0.0.1:{flow['svc_port']}/readyz", svc)
        gw = _spawn(["gateway", "serve", "--host", "127.0.0.1", "--port", str(flow["gw_port"])], {
            "PGW_GATEWAY_MODELS_FILE": str(d / "run" / "models.json"),
            "PGW_GATEWAY_MASKSTORE_ROOT": str(d / "masks"),
            "PGW_MASKSTORE_KEY_FILE": str(d / "kek"),
            "PGW_GATEWAY_API_KEYS_FILE": str(d / "keys.json"),
            "PGW_GATEWAY_ALLOW_DEV_MODES": "true",  # only for the plaintext comparison model
        })
        _wait_ready(f"http://127.0.0.1:{flow['gw_port']}/readyz", gw)
        yield flow
    finally:
        if gw is not None:
            _stop(gw)
        _stop(svc)


def test_masked_chat_equals_plain(running: dict[str, object]) -> None:
    url = f"http://127.0.0.1:{running['gw_port']}/v1/chat/completions"
    hdr = {"Authorization": f"Bearer {running['api_key']}"}
    outs = {}
    for model in ("tiny", "tiny-plain"):
        body = {"model": model, "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 6,
                "temperature": 0}
        r = httpx.post(url, json=body, headers=hdr, timeout=120)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["usage"]["completion_tokens"] >= 1
        outs[model] = data["choices"][0]["message"]["content"]
    assert data["system_fingerprint"] == str(running["fp"])[:12]
    assert outs["tiny"] == outs["tiny-plain"]
    r = httpx.post(url, json={"model": "tiny", "messages": [{"role": "user", "content": "x"}]}, timeout=30)
    assert r.status_code == 401


def test_doctor_against_running_service(running: dict[str, object]) -> None:
    d = running["dir"]
    assert isinstance(d, Path)
    r = CliRunner().invoke(cli, ["doctor", "--no-gpu", "--json", "--pack", str(d / "pack"), "--ring", "K40A12",
                                 "--service-url", f"http://127.0.0.1:{running['svc_port']}",
                                 "--service-token-file", str(d / "svc_token"),
                                 "--maskstore", str(d / "masks"), "--key-file", str(d / "kek")])
    assert r.exit_code == 0, r.output
    rep = json.loads(r.stdout)
    by = {c["name"]: c for c in rep["checks"]}
    assert by["service"]["status"] == "ok" and by["service"]["data"]["models"][0]["fp"] == running["fp"]
    assert by["maskstore"]["status"] == "ok"
    inv = by["maskstore"]["data"]["status"]["inventory"][0]
    assert inv["slots_available"] < 128  # the chat consumed masks (single use)
    # wrong ring is detected
    r = CliRunner().invoke(cli, ["doctor", "--no-gpu", "--pack", str(d / "pack"), "--ring", "K48A20",
                                 "--service-url", f"http://127.0.0.1:{running['svc_port']}",
                                 "--service-token-file", str(d / "svc_token")])
    assert r.exit_code == 1 and "not ring K48A20" in r.output


def test_doctor_service_unreachable() -> None:
    port = free_port()
    r = CliRunner().invoke(cli, ["doctor", "--no-gpu", "--service-url", f"http://127.0.0.1:{port}"])
    assert r.exit_code == 1 and "[FAIL] service" in r.output
