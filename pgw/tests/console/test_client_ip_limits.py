"""Client address behind trusted proxies, and failure-only login / enrollment rate limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from pgw.console.sessions import LoginRateLimiter
from pgw.core.client_ip import ClientIPResolver, client_ip
from pgw.core.errors import ConfigError, RateLimitedError


def test_untrusted_peer_header_is_ignored() -> None:
    r = ClientIPResolver([])
    assert r.resolve("203.0.113.9", ["1.2.3.4"]) == "203.0.113.9"
    r = ClientIPResolver(["10.0.0.0/8"])
    assert r.resolve("203.0.113.9", ["1.2.3.4"]) == "203.0.113.9"  # peer is not a proxy we trust


def test_rightmost_untrusted_hop_is_the_client() -> None:
    r = ClientIPResolver(["10.0.0.0/8", "fd00::/8"])
    # client-supplied junk on the left is never used
    assert r.resolve("10.0.0.5", ["6.6.6.6, 198.51.100.7, 10.0.0.9"]) == "198.51.100.7"
    assert r.resolve("10.0.0.5", ["6.6.6.6", "198.51.100.7"]) == "198.51.100.7"  # repeated headers
    assert r.resolve("10.0.0.5", []) == "10.0.0.5"  # no header: the proxy itself
    assert r.resolve("10.0.0.5", ["10.1.1.1, 10.2.2.2"]) == "10.1.1.1"  # all trusted: leftmost
    assert r.resolve("10.0.0.5", ["1.1.1.1, garbage"]) == "10.0.0.5"  # unparsable hop: stop at the proxy
    assert r.resolve("10.0.0.5", ["[2001:db8::1]:443"]) == "2001:db8::1"
    assert r.resolve("10.0.0.5", ["198.51.100.7:5555"]) == "198.51.100.7"
    assert r.resolve("::ffff:10.0.0.5", ["198.51.100.7"]) == "198.51.100.7"  # v4-mapped peer
    assert r.resolve(None, ["1.1.1.1"]) == "unknown"


def test_scope_and_bad_config() -> None:
    r = ClientIPResolver(["127.0.0.1/32"])
    scope = {"client": ("127.0.0.1", 1), "headers": [(b"x-forwarded-for", b"192.0.2.44")]}
    assert client_ip(scope, r) == "192.0.2.44"
    with pytest.raises(ConfigError):
        ClientIPResolver(["not-a-net"])


def test_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.core.settings import CommonSettings

    monkeypatch.setenv("PGW_TRUSTED_PROXY_CIDRS", '["10.0.0.0/8", " 192.168.0.0/16 "]')
    assert CommonSettings().trusted_proxy_cidrs == ["10.0.0.0/8", "192.168.0.0/16"]
    monkeypatch.setenv("PGW_TRUSTED_PROXY_CIDRS", '["nope"]')
    with pytest.raises(ValueError):
        CommonSettings()
    monkeypatch.delenv("PGW_TRUSTED_PROXY_CIDRS")
    assert CommonSettings().trusted_proxy_cidrs == []


class _Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


def test_trusted_proxy_buckets_are_per_client() -> None:
    """Behind a trusted proxy, each real client gets its own failure bucket."""
    r = ClientIPResolver(["10.0.0.0/8"])
    lim = LoginRateLimiter(attempts=2, clock=_Clock())
    a = r.resolve("10.0.0.2", ["198.51.100.1"])
    b = r.resolve("10.0.0.2", ["198.51.100.2"])
    for _ in range(2):
        lim.blocked(a)
        lim.failure(a)
    with pytest.raises(RateLimitedError):
        lim.blocked(a)
    lim.blocked(b)  # a different client behind the same proxy is not affected
    # an untrusted peer cannot choose its bucket with the header
    assert r.resolve("198.51.100.1", ["198.51.100.99"]) == "198.51.100.1"


def test_only_failures_count_and_global_ceiling() -> None:
    clock = _Clock()
    lim = LoginRateLimiter(attempts=3, global_failures=5, clock=clock)
    for _ in range(50):
        lim.blocked("good")  # successes never count
    for i in range(5):
        lim.blocked(f"c{i}")
        lim.failure(f"c{i}")
    with pytest.raises(RateLimitedError) as ei:
        lim.blocked("fresh")  # global ceiling reached from rotating addresses
    assert "gateway" in ei.value.message
    clock.t += 61
    lim.blocked("fresh")


async def test_console_login_counts_failures_only(tiny_pack_dir: Path, tmp_path: Path) -> None:
    """End to end on the gateway console: right tokens never lock out, wrong ones do (per client)."""
    from tests.console.conftest import client, make_console_app
    from tests.gateway.conftest import ADMIN_TOKEN

    app = make_console_app(tiny_pack_dir, tmp_path)
    app.state.pgw.login_limiter = LoginRateLimiter(attempts=2)
    async with client(app) as c:
        for _ in range(4):
            assert (await c.post("/console/api/login", json={"token": ADMIN_TOKEN})).status_code == 200
        codes = [(await c.post("/console/api/login", json={"token": "wrong"})).status_code for _ in range(3)]
        assert codes == [401, 401, 429]
        assert (await c.post("/console/api/login", json={"token": ADMIN_TOKEN})).status_code == 429
