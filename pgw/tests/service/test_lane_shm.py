"""Opt-in latency knobs of the service trip lane's 1-row path (native listener and native client channel).

Covers: the shared-memory transport (``LaneService(shm=True)`` + ``TripChannel.attach_shm``) carries
byte-identical replies to the socket path for the same request bodies, keeps refusals and their statuses,
parks and wakes both ways (doorbell) after idle gaps, is refused (``False``, the connection stays on the
socket) by a service without it, and a closed service ends the channel; zero-copy socket replies
(``zero_copy=True``) are byte-identical; the adaptive client spin changes no bytes; ``queue_us`` /
``compute_us`` and the channel counters are reported.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from pathlib import Path
from typing import Any

import pytest

fast = pytest.importorskip("pgw_fast")
pytestmark = pytest.mark.skipif(not hasattr(fast.TripChannel, "attach_shm"),
                                reason="pgw_fast without the lane shm transport")

LINEARS = [(16, 24, "qkv"), (24, 16, "o"), (16, 64, "gate_up"), (32, 16, "down")]  # (n, m, kind)
BPE = 4


def _reply(linear: int, rows: int, body: bytes) -> bytes:
    """A deterministic function of the request body (so a wrong or torn body shows)."""
    m = LINEARS[linear][1]
    want = rows * m * BPE
    out = b""
    i = 0
    while len(out) < want:
        out += hashlib.sha256(bytes([linear, rows, i]) + body).digest()
        i += 1
    return out[:want]


def _service(td: Path, **kw: Any) -> tuple[Any, list[bytes]]:
    def execute(_model: int, linear: int, _ring: int, rows: int, body: bytes, _tenant: str, _left: float) -> Any:
        if rows == 3:
            return 3  # rate_limited: a refusal that keeps the connection
        return _reply(linear, rows, body)

    kw.setdefault("executor", execute)
    svc = fast.LaneService(str(td / "lane.sock"), max_rows=8, **kw)
    svc.set_model(0, [(n, m, kind) for n, m, kind in LINEARS], [("K32A12C27", BPE)])
    tickets = []
    for _ in range(6):
        t = secrets.token_bytes(32)
        svc.add_ticket(t, "t", 0, 60.0)
        tickets.append(t)
    return svc, tickets


def _bodies(n: int = 40) -> list[tuple[int, int, bytes]]:
    out = []
    for i in range(n):
        li = i % len(LINEARS)
        rows = 1 + (i % 4 == 3)  # mostly 1 row, some 2
        out.append((li, rows, secrets.token_bytes(rows * LINEARS[li][0] * BPE)))
    return out


def _trip(ch: Any, li: int, rows: int, body: bytes) -> tuple[int, bytes]:
    out = bytearray(rows * LINEARS[li][1] * BPE)
    st, _cu = ch.trip(0, li, rows, secrets.token_bytes(16), 5000, body, out, 5.0)
    return st, bytes(out)


def test_shm_replies_are_byte_identical_to_the_socket(tmp_path: Path) -> None:
    svc, tickets = _service(tmp_path, shm=True)
    try:
        sock = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), tickets[0], 5.0, 8, 50)
        shm = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), tickets[1], 5.0, 8, 50)
        assert sock.transport == "uds"
        assert shm.attach_shm() is True
        assert shm.transport == "shm"
        assert shm.attach_shm() is True  # idempotent
        for li, rows, body in _bodies():
            a, b = _trip(sock, li, rows, body), _trip(shm, li, rows, body)
            assert a == b == (0, _reply(li, rows, body))
        # a non-fatal refusal keeps the shm connection, with the socket path's status
        body = secrets.token_bytes(3 * LINEARS[0][0] * BPE)
        out = bytearray(3 * LINEARS[0][1] * BPE)
        st_sock = sock.trip(0, 0, 3, secrets.token_bytes(16), 5000, body, out, 5.0)[0]
        st_shm = shm.trip(0, 0, 3, secrets.token_bytes(16), 5000, body, out, 5.0)[0]
        assert st_sock == st_shm != 0
        assert _trip(shm, 1, 1, b"\x01" * 96) == (0, _reply(1, 1, b"\x01" * 96))
        st = shm.stats()
        assert st["trips"] == 41 and st["spin_hits"] + st["blocked"] == st["trips"]
        sock.close()
        shm.close()
        assert shm.closed
    finally:
        svc.stop(2.0)


def test_shm_parks_and_wakes_both_ways(tmp_path: Path) -> None:
    # server spin 1 ms, client spin 0: every reply wait parks the client; the idle gaps park the server
    svc, tickets = _service(tmp_path, shm=True, read_spin_us=1000)
    try:
        ch = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), tickets[0], 5.0, 8, 0)
        assert ch.attach_shm()
        for i, (li, rows, body) in enumerate(_bodies(12)):
            if i % 3 == 0:
                time.sleep(0.02)  # well past the server's spin: it parks on the doorbell
            assert _trip(ch, li, rows, body) == (0, _reply(li, rows, body))
        assert ch.stats()["blocked"] >= 1
        ch.close()
    finally:
        svc.stop(2.0)


def test_shm_refused_without_the_service_option(tmp_path: Path) -> None:
    svc, tickets = _service(tmp_path)  # shm off (the default)
    try:
        ch = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), tickets[0], 5.0, 8, 50)
        assert ch.attach_shm() is False
        assert ch.transport == "uds"
        li, rows, body = _bodies(1)[0]
        assert _trip(ch, li, rows, body) == (0, _reply(li, rows, body))
        ch.close()
    finally:
        svc.stop(2.0)


def test_shm_channel_ends_when_the_service_stops(tmp_path: Path) -> None:
    svc, tickets = _service(tmp_path, shm=True, poll_ms=20)
    ch = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), tickets[0], 2.0, 8, 0)
    assert ch.attach_shm()
    li, rows, body = _bodies(1)[0]
    assert _trip(ch, li, rows, body)[0] == 0
    svc.stop(2.0)
    with pytest.raises(OSError):
        _trip(ch, li, rows, body)
    assert ch.closed


def test_zero_copy_and_adaptive_spin_are_byte_identical(tmp_path: Path) -> None:
    # the native bench stub exercises the zero-copy sink (its reply comes from its own buffer: zeros)
    svc, tickets = _service(tmp_path, zero_copy=True, bench_stub_us=[5, 1, 20, 2])
    try:
        plain = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), tickets[0], 5.0, 8, 50)
        auto = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), tickets[1], 5.0, 8, 0)
        auto.set_spin_auto(500)
        for li, rows, body in _bodies():
            want = bytes(rows * LINEARS[li][1] * BPE)
            assert _trip(plain, li, rows, body) == _trip(auto, li, rows, body) == (0, want)
        st = auto.stats()
        assert st["trips"] == 40 and st["compute_us"] > 0 and st["spin_us"] > 0
        plain.close()
        auto.close()
    finally:
        svc.stop(2.0)
    # the Python executor (no zero-copy support) under zero_copy=True still answers from its copy
    svc, tickets = _service(tmp_path, zero_copy=True)
    try:
        ch = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), tickets[0], 5.0, 8, 50)
        for li, rows, body in _bodies(8):
            assert _trip(ch, li, rows, body) == (0, _reply(li, rows, body))
        ch.close()
    finally:
        svc.stop(2.0)


def test_zero_copy_large_reply_is_completed_after_a_partial_write(tmp_path: Path) -> None:
    # replies larger than a socket buffer: the non-blocking write takes part, the rest goes out after
    big = [(16, 1 << 17, "gate_up")]  # 512 KiB per row
    svc = fast.LaneService(str(tmp_path / "lane.sock"), max_rows=8, zero_copy=True, bench_stub_us=[0])
    svc.set_model(0, [(n, m, k) for n, m, k in big], [("K32A12C27", BPE)])
    t = secrets.token_bytes(32)
    svc.add_ticket(t, "t", 0, 60.0)
    try:
        ch = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), t, 5.0, 8, 0)
        for rows in (1, 2, 1):
            out = bytearray(rows * big[0][1] * BPE)
            out[:] = b"\xff" * len(out)
            st, _ = ch.trip(0, 0, rows, secrets.token_bytes(16), 5000, bytes(rows * 16 * BPE), out, 5.0)
            assert st == 0 and out == bytes(len(out))
        ch.close()
    finally:
        svc.stop(2.0)


def test_queue_and_compute_are_reported_separately(tmp_path: Path) -> None:
    svc, tickets = _service(tmp_path, bench_stub_us=[200])
    try:
        ch = fast.TripChannel.connect_service_uds(str(tmp_path / "lane.sock"), tickets[0], 5.0, 8, 50)
        for li, rows, body in _bodies(6):
            assert _trip(ch, li, rows, body)[0] == 0
        st = ch.stats()
        assert st["compute_us"] >= 6 * 200
        assert st["queue_us"] < st["compute_us"]
        ch.close()
    finally:
        svc.stop(2.0)
