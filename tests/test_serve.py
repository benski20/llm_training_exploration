"""Verification-gated inference server."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from test_global import hand_built_dyck
from vt.artifact import save_artifact
from vt.cli import main
from vt.serve import GateError, load_gated, make_server


@pytest.fixture
def proved(tmp_path):
    art, cert = str(tmp_path / "m.npz"), str(tmp_path / "c.json")
    save_artifact(hand_built_dyck(8), art)
    assert main(["verify", art, "--certificate", cert, "--workers", "1"]) == 0
    return art, cert


def _req(url, data=None):
    r = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode())
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_serves_verified_artifact(proved):
    P, digest, cert = load_gated(*proved)
    srv = make_server(P, digest, cert, "()", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        code, h = _req(base + "/health")
        assert code == 200 and h["verified"] and h["artifact_sha256"] == digest
        code, r = _req(base + "/predict", {"inputs": ["(()())()", "())(()()"]})
        assert code == 200 and [p["label"] for p in r["predictions"]] == [1, 0]
        assert _req(base + "/predict", {"inputs": ["(("]})[0] == 400        # wrong length
        assert _req(base + "/predict", {"inputs": ["abcdefgh"]})[0] == 400  # bad alphabet
        assert _req(base + "/predict", {"nope": 1})[0] == 400
    finally:
        srv.shutdown()


def test_gate_refuses_tampered_or_missing_certificate(proved, tmp_path):
    art, cert = proved
    with pytest.raises(GateError):
        load_gated(art, None)
    P = hand_built_dyck(8)
    P["b_out"][1] += 1e-9
    save_artifact(P, art)
    with pytest.raises(GateError):
        load_gated(art, cert)
    assert main(["serve", art, cert, "--port", "0"]) == 1
