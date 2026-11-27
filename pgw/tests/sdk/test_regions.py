"""Region routing (pgw.sdk.regions): endpoint list validation, payload-free probing, selection with hysteresis,
failover order, mask-stock compatibility, and the provider's ``endpoints`` in the enrollment response."""

from __future__ import annotations

import socket
import threading

import pytest

from pgw.sdk.regions import Endpoint, Probe, choose, failover_order, parse_endpoints, probe, stock_compatible


def _p(url: str, *samples: float | None, region: str = "r", models: tuple[str, ...] = (), prio: int = 0) -> Probe:
    return Probe(Endpoint(region, url, frozenset(models), prio), list(samples))


def test_parse_valid_and_enrolled_first() -> None:
    eps = parse_endpoints([{"region": "eu", "url": "https://eu.example.net:8443"},
                           {"region": "us", "url": "https://us.example.net", "models": ["fp1"], "priority": 1},
                           {"region": "dup", "url": "https://eu.example.net:8443/"}],
                          enrolled_url="https://main.example.net:8443")
    assert [e.url for e in eps] == ["https://main.example.net:8443", "https://eu.example.net:8443",
                                    "https://us.example.net:443"]
    assert eps[0].region == "enrolled" and eps[2].models == frozenset({"fp1"}) and eps[2].port == 443


@pytest.mark.parametrize("bad", [
    [{"region": "x", "url": "http://a.example:1"}],
    [{"region": "x", "url": "https://user:pw@a.example"}],
    [{"region": "x", "url": "https://a.example/path"}],
    [{"region": "x", "url": "https://a.example?q=1"}],
    [{"region": "", "url": "https://a.example"}],
    [{"url": "https://a.example"}],
    [{"region": "x", "url": "https://a.example", "models": "fp"}],
    [{"region": "x", "url": "https://a.example", "priority": True}],
    ["https://a.example"],
    [{"region": "x", "url": f"https://h{i}.example"} for i in range(40)],
])
def test_parse_refuses_malformed(bad: list) -> None:
    with pytest.raises(ValueError):
        parse_endpoints(bad)


def test_choose_nearest_healthy_serving_model() -> None:
    ps = [_p("https://a:1", 20, 21, 19), _p("https://b:1", 5, 6, 5, models=("other",)),
          _p("https://c:1", 9, None, 9), _p("https://d:1", 12, 11, 12)]
    assert choose(ps, fingerprint="fp").url == "https://d:1"  # b serves another pack, c lost a probe
    assert choose(ps, fingerprint="other").url == "https://b:1"
    assert choose([_p("https://x:1", None)]) is None


def test_choose_hysteresis() -> None:
    cur = Endpoint("r", "https://a:1")
    near = [_p("https://a:1", 10, 10, 10), _p("https://b:1", 9.5, 9.5, 9.5)]
    assert choose(near, current=cur).url == "https://a:1"  # 5 % better: stay
    far = [_p("https://a:1", 10, 10, 10), _p("https://b:1", 4, 4, 4)]
    assert choose(far, current=cur).url == "https://b:1"
    tiny = [_p("https://a:1", 1.0, 1.0), _p("https://b:1", 0.5, 0.5)]
    assert choose(tiny, current=cur).url == "https://a:1"  # 50 % but only 0.5 ms: stay
    dead = [_p("https://a:1", None, 1), _p("https://b:1", 30, 30)]
    assert choose(dead, current=cur).url == "https://b:1"


def test_failover_order() -> None:
    ps = [_p("https://a:1", 3, 3), _p("https://b:1", None, None), _p("https://c:1", 8, 8), _p("https://d:1", 1, 1)]
    assert [e.url for e in failover_order(ps, exclude=["https://d:1"])] == ["https://a:1", "https://c:1",
                                                                            "https://b:1"]


def test_stock_compatible() -> None:
    st = {"fingerprint": "fp1", "ring": "K32A12C27"}
    assert stock_compatible(st, ["fp1", "fp2"], "K32A12C27")
    assert not stock_compatible(st, ["fp2"], "K32A12C27")
    assert not stock_compatible(st, ["fp1"], "K40A12")


def test_probe_sends_no_payload() -> None:
    """A real listener sees connections that carry zero bytes (no ClientHello, no SNI, no credentials)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    got: list[bytes] = []
    stop = threading.Event()

    def serve() -> None:
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                c, _ = srv.accept()
            except TimeoutError:
                continue
            c.settimeout(1.0)
            try:
                got.append(c.recv(65536))
            except TimeoutError:
                got.append(b"<timeout>")
            c.close()

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    port = srv.getsockname()[1]
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    dead_port = dead.getsockname()[1]
    dead.close()  # nothing listens: refused
    try:
        res = probe([Endpoint("live", f"https://127.0.0.1:{port}"), Endpoint("dead", f"https://127.0.0.1:{dead_port}")],
                    samples=3, timeout_s=0.5, gap_s=0)
    finally:
        import time

        time.sleep(0.5)
        stop.set()
        th.join(2)
        srv.close()
    assert res[0].healthy and res[0].rtt_ms < 500 and not res[1].healthy
    assert len(got) == 3 and all(b == b"" for b in got)
    assert choose(res).region == "live"


def test_enroll_response_endpoints_setting() -> None:
    """The provider setting is validated like the client parses it (a malformed list refuses to start)."""
    from pgw.service.config import ServiceSettings

    ok = ServiceSettings(endpoints=[{"region": "eu", "url": "https://eu.example.net:8443"}])
    assert ok.endpoints == [{"region": "eu", "url": "https://eu.example.net:8443"}]
    with pytest.raises(ValueError):
        ServiceSettings(endpoints=[{"region": "eu", "url": "http://eu.example.net"}])
