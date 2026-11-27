"""WAN socket tuning of the TCP lane (docs/sdk/wan.md): keep-alive and user timeout on lane sockets, the env switch,
and the provider's read-only check of the kernel's slow-start-after-idle setting. No byte of the stream changes."""

from __future__ import annotations

import logging
import socket
import sys
from pathlib import Path

import pytest

from pgw.wire.lane import lane_keepalive_s, tune_wan_socket


def _pair() -> tuple[socket.socket, socket.socket, socket.socket]:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = socket.create_connection(srv.getsockname())
    a, _ = srv.accept()
    return srv, c, a


def test_keepalive_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_LANE_KEEPALIVE_S", raising=False)
    assert lane_keepalive_s() == 15
    monkeypatch.setenv("PGW_LANE_KEEPALIVE_S", "0")
    assert lane_keepalive_s() == 0
    monkeypatch.setenv("PGW_LANE_KEEPALIVE_S", "junk")
    assert lane_keepalive_s() == 15
    monkeypatch.setenv("PGW_LANE_KEEPALIVE_S", "99999")
    assert lane_keepalive_s() == 7200


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux socket options")
def test_tune_wan_socket_sets_keepalive_and_user_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_LANE_KEEPALIVE_S", "20")
    srv, c, a = _pair()
    try:
        tune_wan_socket(c)
        assert c.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
        assert c.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 20
        assert c.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 5
        assert c.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == 3
        assert c.getsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT) == 35_000
        c.sendall(b"same bytes")  # the stream itself is untouched
        assert a.recv(64) == b"same bytes"
        monkeypatch.setenv("PGW_LANE_KEEPALIVE_S", "0")
        tune_wan_socket(a)
        assert not a.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE)
    finally:
        for s in (srv, c, a):
            s.close()


def test_provider_wan_tcp_check(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from pgw.service.lane import wan_tcp_check

    (tmp_path / "tcp_slow_start_after_idle").write_text("1\n")
    (tmp_path / "tcp_congestion_control").write_text("cubic\n")
    with caplog.at_level(logging.INFO, logger="pgw.service.lane"):
        out = wan_tcp_check(tmp_path)
    assert out == {"tcp_slow_start_after_idle": "1", "tcp_congestion_control": "cubic"}
    assert any("tcp_slow_start_after_idle=1" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    (tmp_path / "tcp_slow_start_after_idle").write_text("0\n")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pgw.service.lane"):
        assert wan_tcp_check(tmp_path)["tcp_slow_start_after_idle"] == "0"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert wan_tcp_check(tmp_path / "missing")["tcp_slow_start_after_idle"] == "?"
