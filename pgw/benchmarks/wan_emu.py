"""A packet-level WAN between two network namespaces, for benches on one box without ``tc netem``.

Topology (needs root / CAP_NET_ADMIN, ``ip`` from iproute2 and ``/dev/net/tun``)::

    netns pgwwc (client)  tun pgwwc0 10.77.0.1  <->  pgw-wanemu  <->  tun pgwws0 10.77.0.2  netns pgwws (provider)

``rust/tools/wanemu`` (``pgw-wanemu``) forwards every IP packet between the two TUN devices with a per-direction
bottleneck rate, propagation delay, jitter and loss, so both kernels run real TCP end to end (slow start, delayed
ACKs, cwnd validation, loss recovery): unlike the TCP-terminating ``delay_proxy.py``, the transport behaves as on a
real path. Each namespace has its own TCP sysctls; :class:`WanLink` sets the congestion control (``cubic`` by
default, what a consumer OS and most clouds run) and ``tcp_slow_start_after_idle`` explicitly, so the box's own
settings (this box defaults to BBR) never leak into a measurement.

    with WanLink() as w:
        w.set(rtt_ms=20, rate_mbit=100)          # 10 ms each way, 100 Mbit/s both ways
        w.run_client([...]) / w.popen_server([...])
        w.stats()

The tun MTU is 1500 (an Ethernet path; set ``mtu``). Commands run inside a namespace through ``ip netns exec``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLIENT_NS, SERVER_NS = "pgwwc", "pgwws"
CLIENT_IP, SERVER_IP = "10.77.0.1", "10.77.0.2"
_TUN_C, _TUN_S = "pgwwc0", "pgwws0"


def wanemu_binary() -> str:
    """``$PGW_WANEMU_BIN``, else a release build of ``rust/tools/wanemu`` (built on first use, outside the tree)."""
    env = os.environ.get("PGW_WANEMU_BIN")
    if env:
        return env
    tgt = Path(os.environ.get("PGW_WANEMU_TARGET", str(Path(tempfile.gettempdir()) / "pgw-wanemu-target")))
    exe = tgt / "release" / "pgw-wanemu"
    if not exe.exists():
        subprocess.run(["cargo", "build", "--release", "--offline"], cwd=ROOT / "rust" / "tools" / "wanemu",  # noqa: S607
                       env={**os.environ, "CARGO_TARGET_DIR": str(tgt)}, check=True)
    return str(exe)


def _ip(*args: str, ns: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["ip", *(["netns", "exec", ns, "ip"] if ns else []), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)  # noqa: S603


def _sysctl(ns: str, key: str, value: str) -> None:
    subprocess.run(["ip", "netns", "exec", ns, "sysctl", "-q", "-w", f"{key}={value}"],  # noqa: S603, S607
                   check=True, capture_output=True)


def ns_cmd(ns: str, argv: list[str]) -> list[str]:
    return ["ip", "netns", "exec", ns, *argv]


class WanLink:
    """The two namespaces, the TUN pair and the emulator process (stdin control). ``cc`` / ``ssai``: congestion
    control and ``tcp_slow_start_after_idle`` in both namespaces."""

    def __init__(self, cc: str = "cubic", ssai: int = 1, mtu: int = 1500, server_cc: str | None = None,
                 server_ssai: int | None = None):
        self.cc, self.ssai, self.mtu = cc, ssai, mtu
        #: the provider namespace may differ (provider-side fixes: its kernel is ours, the client's is not)
        self.server_cc, self.server_ssai = server_cc, server_ssai
        self.proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------------------------------------------
    def __enter__(self) -> WanLink:
        self.up()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.down()

    def up(self) -> None:
        if shutil.which("ip") is None:
            raise SystemExit("needs iproute2 (`ip`)")
        self.down()
        for ns in (CLIENT_NS, SERVER_NS):
            _ip("netns", "add", ns)
            _ip("link", "set", "lo", "up", ns=ns)
        self.proc = subprocess.Popen([wanemu_binary(), _TUN_C, _TUN_S], stdin=subprocess.PIPE,  # noqa: S603
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        assert self.proc.stdout is not None
        if self.proc.stdout.readline().strip() != "ready":
            raise RuntimeError("pgw-wanemu did not start")
        for tun, ns, me, peer in ((_TUN_C, CLIENT_NS, CLIENT_IP, SERVER_IP), (_TUN_S, SERVER_NS, SERVER_IP, CLIENT_IP)):
            _ip("link", "set", tun, "netns", ns)
            _ip("addr", "add", me, "peer", peer, "dev", tun, ns=ns)
            _ip("link", "set", tun, "mtu", str(self.mtu), "up", ns=ns)
        self.set_tcp()

    def set_tcp(self, cc: str | None = None, ssai: int | None = None) -> None:
        """Congestion control / slow start after idle in both namespaces (the server's own values where set; new
        connections take them)."""
        if cc is not None:
            self.cc = cc
        if ssai is not None:
            self.ssai = ssai
        per_ns = {CLIENT_NS: (self.cc, self.ssai),
                  SERVER_NS: (self.server_cc or self.cc, self.ssai if self.server_ssai is None else self.server_ssai)}
        for ns, (c, si) in per_ns.items():
            # a namespace may only pick a congestion control the (global) allowed list names: extend it, never shrink
            allowed = Path("/proc/sys/net/ipv4/tcp_allowed_congestion_control").read_text().split()
            if c not in allowed:
                subprocess.run(["sysctl", "-q", "-w",  # noqa: S603, S607
                                "net.ipv4.tcp_allowed_congestion_control=" + " ".join([*allowed, c])], check=True)
            _sysctl(ns, "net.ipv4.tcp_congestion_control", c)
            _sysctl(ns, "net.ipv4.tcp_slow_start_after_idle", str(int(si)))

    def down(self) -> None:
        if self.proc is not None:
            try:
                self._cmd("quit", answer=False)
            except (OSError, ValueError):
                pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        for ns in (CLIENT_NS, SERVER_NS):
            _ip("netns", "del", ns, check=False)

    # -- control ---------------------------------------------------------------------------------------------
    def _cmd(self, line: str, answer: bool = True) -> str:
        assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        with self._lock:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
            if not answer:
                return ""
            out = self.proc.stdout.readline().strip()
        if out.startswith("error"):
            raise ValueError(out)
        return out

    def set(self, rtt_ms: float = 0.0, jitter_ms: float = 0.0, loss: float = 0.0, burst: float = 1.0,
            rate_mbit: float = 0.0, up_mbit: float | None = None, queue_kb: float = 4096.0) -> None:
        """Symmetric delay (``rtt_ms / 2`` each way), per-direction jitter and loss; ``rate_mbit`` the download
        (provider -> client) bottleneck, ``up_mbit`` the upload (default: the same)."""
        one = rtt_ms / 2.0
        up = rate_mbit if up_mbit is None else up_mbit
        self._cmd(f"set down delay_ms={one} jitter_ms={jitter_ms} loss={loss} burst={burst} "
                  f"rate_mbit={rate_mbit} queue_kb={queue_kb}")
        self._cmd(f"set up delay_ms={one} jitter_ms={jitter_ms} loss={loss} burst={burst} "
                  f"rate_mbit={up} queue_kb={queue_kb}")

    def stats(self) -> dict[str, Any]:
        return dict(json.loads(self._cmd("stats")))

    def reset(self) -> None:
        self._cmd("reset")

    # -- processes -------------------------------------------------------------------------------------------
    @staticmethod
    def client_cmd(argv: list[str]) -> list[str]:
        return ns_cmd(CLIENT_NS, argv)

    @staticmethod
    def server_cmd(argv: list[str]) -> list[str]:
        return ns_cmd(SERVER_NS, argv)
