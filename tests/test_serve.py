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


@pytest.fixture
def server(proved):
    P, digest, cert = load_gated(*proved)
    srv = make_server(P, digest, cert, "()", port=0, max_body=4096, timeout=1.0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def _raw(port, payload):
    import socket
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(payload)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    return data


def test_limits_and_malformed_requests(server):
    srv, base = server
    port = srv.server_port
    big = json.dumps({"inputs": ["(()())()"] * 2000}).encode()
    assert b" 413 " in _raw(port, b"POST /predict HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % len(big) + big)
    assert b" 411 " in _raw(port, b"POST /predict HTTP/1.1\r\nHost: x\r\n\r\n")
    for body in [b"\xff\xfe", b"[]", b"null", b'{"inputs": "(())(())"}', b'{"inputs": [1,2]}', b"{" * 100]:
        r = _raw(port, b"POST /predict HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        assert b" 400 " in r, (body, r[:80])
    assert _req(base + "/health")[0] == 200            # still alive


def test_fuzz_never_crashes(server):
    import random
    srv, base = server
    rng = random.Random(0)
    for _ in range(300):
        body = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 200)))
        r = _raw(srv.server_port, b"POST /predict HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n"
                 % len(body) + body)
        assert r.split(b" ")[1] in (b"200", b"400"), r[:60]
    assert _req(base + "/health")[0] == 200


def test_slow_client_does_not_block_others(server):
    import socket
    srv, base = server
    idle = socket.create_connection(("127.0.0.1", srv.server_port))   # sends nothing
    try:
        assert _req(base + "/predict", {"inputs": ["(()())()"]})[0] == 200
    finally:
        idle.close()


def test_concurrent_load(server):
    from concurrent.futures import ThreadPoolExecutor
    srv, base = server
    inputs = ["(()())()", "())(()()", "(((())))", "()()()()"]
    want = [1, 0, 1, 1]

    def one(_):
        code, r = _req(base + "/predict", {"inputs": inputs})
        return code == 200 and [p["label"] for p in r["predictions"]] == want
    with ThreadPoolExecutor(16) as ex:
        assert all(ex.map(one, range(400)))
    m = _req(base + "/metrics")[1]
    assert m["predictions"] >= 1600
