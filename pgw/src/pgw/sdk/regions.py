"""Region routing: pick the nearest healthy provider endpoint the installation is entitled to (docs/sdk/wan.md).

Every decode token costs dozens of sequential round trips, so the network RTT to the provider multiplies into the
token time: the client should talk to the closest GPU it may use. The provider lists its endpoints in the
enrollment response (``endpoints``, :func:`parse_endpoints`), on the pinned-CA TLS channel the enrollment already
uses; every endpoint is a full provider (HTTPS + trip lane) serving a subset of the models, under a TLS certificate
from the **same pinned provider CA** (the client verifies each one exactly as the enrolled URL), so a list entry can
only ever point at a host the provider's CA vouches for.

Probing (:func:`probe`) measures the RTT to each endpoint with bare TCP handshakes to its HTTPS port: SYN /
SYN-ACK / RST-free close, **no byte of payload** (no TLS ClientHello, so no SNI, no key, no token, no client
certificate, no model name). An endpoint learns only that this IP opened and closed a connection, which every
endpoint of the same operator learns anyway when it is used. Probes never run while a request is in flight on the
probed path (the caller schedules them between requests).

Selection (:func:`choose`): the lowest median handshake time among endpoints that serve the model and answered
every probe of the round; the current endpoint is kept unless another is faster by both ``switch_ratio`` and
``switch_min_ms`` (no flapping between two near-equal regions). :func:`failover_order` is the order to try on a
connection failure.

Masks across regions. A mask slot is ``W8 r mod 2^K`` for the pack's int8 weights ``W8`` and the client's secret
``r``; pks-v3 masks are made by the provider homomorphically under the **client's** key (the provider holds no key
of its own) and checked by the client. The stock is therefore bound to (pack fingerprint, ring, scheme), not to a
provider host: an endpoint serving the same pack fingerprint and ring consumes the same stock
(:func:`stock_compatible`); an endpoint with another fingerprint (a different revision or quantisation) needs its
own stock, and the client must not move a running request there. Nothing about masks or prompts is ever sent while probing.
"""

from __future__ import annotations

import math
import socket
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

__all__ = ["Endpoint", "Probe", "choose", "failover_order", "parse_endpoints", "probe", "stock_compatible"]

MAX_ENDPOINTS = 32


@dataclass(frozen=True)
class Endpoint:
    """One provider endpoint from the enrollment response."""

    region: str  # an operator label, e.g. "us-central-ks"; shown to the user, never used to connect
    url: str  # https://host:port, the service base URL (lane host = this host, port from its tickets)
    models: frozenset[str] = frozenset()  # pack fingerprints served (empty: every model of the enrollment)
    priority: int = 0  # operator preference among equally near endpoints (lower first)

    @property
    def host(self) -> str:
        return str(urlsplit(self.url).hostname)

    @property
    def port(self) -> int:
        return int(urlsplit(self.url).port or 443)

    def serves(self, fingerprint: str | None) -> bool:
        return not self.models or fingerprint is None or fingerprint in self.models


@dataclass
class Probe:
    """The handshake times of one endpoint in one probe round (ms; ``None`` = no answer within the timeout)."""

    endpoint: Endpoint
    samples: list[float | None] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return bool(self.samples) and all(s is not None for s in self.samples)

    @property
    def rtt_ms(self) -> float:
        ok = [s for s in self.samples if s is not None]
        return statistics.median(ok) if ok and self.healthy else math.inf


def parse_endpoints(raw: Any, *, enrolled_url: str | None = None) -> list[Endpoint]:
    """Validate the enrollment response's ``endpoints``: a list of ``{"region", "url", "models"?, "priority"?}``.
    Only ``https`` URLs with a host and no path, query, fragment or credentials; at most ``MAX_ENDPOINTS``; duplicate
    URLs dropped. The enrolled URL is always included (region ``"enrolled"``) so routing can only add choices.

    Raises:
        ValueError: a malformed entry (the whole list is refused: a provider bug, not something to half-trust).
    """
    out: list[Endpoint] = []
    seen: set[str] = set()
    items = list(raw or [])
    if len(items) > MAX_ENDPOINTS:
        raise ValueError(f"at most {MAX_ENDPOINTS} endpoints")
    for it in items:
        if not isinstance(it, Mapping):
            raise ValueError("an endpoint must be an object")
        url, region = it.get("url"), it.get("region")
        if not isinstance(url, str) or not isinstance(region, str) or not 0 < len(region) <= 64:
            raise ValueError("an endpoint needs a url and a region label")
        p = urlsplit(url)
        if p.scheme != "https" or not p.hostname or p.username or p.password or p.query or p.fragment \
                or p.path not in ("", "/"):
            raise ValueError(f"endpoint url must be https://host[:port]: {url!r}")
        try:
            _ = p.port
        except ValueError as exc:
            raise ValueError(f"endpoint url has a bad port: {url!r}") from exc
        models = it.get("models") or []
        if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError("endpoint models must be a list of fingerprints")
        prio = it.get("priority", 0)
        if not isinstance(prio, int) or isinstance(prio, bool):
            raise ValueError("endpoint priority must be an integer")
        norm = f"https://{p.hostname}:{p.port or 443}"
        if norm in seen:
            continue
        seen.add(norm)
        out.append(Endpoint(region, norm, frozenset(models), prio))
    if enrolled_url:
        p = urlsplit(enrolled_url)
        norm = f"{p.scheme}://{p.hostname}:{p.port or (443 if p.scheme == 'https' else 80)}"
        if norm not in seen:
            out.insert(0, Endpoint("enrolled", norm))
    return out


def _connect_ms(host: str, port: int, timeout_s: float) -> float | None:
    """One TCP handshake (connect, then close without sending a byte): ms, or ``None``."""
    t0 = time.perf_counter()
    try:
        s = socket.create_connection((host, port), timeout=timeout_s)
    except OSError:
        return None
    dt = (time.perf_counter() - t0) * 1e3
    s.close()
    return dt


def probe(endpoints: Sequence[Endpoint], *, samples: int = 5, timeout_s: float = 1.0, gap_s: float = 0.02,
          connect: Callable[[str, int, float], float | None] = _connect_ms) -> list[Probe]:
    """``samples`` handshakes to each endpoint, interleaved (endpoint order rotates per round, so a burst of local
    load does not favour one), ``gap_s`` apart. Bare TCP: no payload is ever sent."""
    res = [Probe(e) for e in endpoints]
    for i in range(samples):
        order = list(range(len(res)))
        k = i % max(1, len(order))
        for j in order[k:] + order[:k]:
            res[j].samples.append(connect(res[j].endpoint.host, res[j].endpoint.port, timeout_s))
        if gap_s and i + 1 < samples:
            time.sleep(gap_s)
    return res


def choose(probes: Iterable[Probe], *, fingerprint: str | None = None, current: Endpoint | None = None,
           switch_ratio: float = 0.8, switch_min_ms: float = 1.0) -> Endpoint | None:
    """The endpoint to use: healthy, serving ``fingerprint``, lowest median RTT (then ``priority``). The ``current``
    endpoint stays while healthy unless the best is faster by both ``switch_ratio`` and ``switch_min_ms``."""
    ok = [p for p in probes if p.healthy and p.endpoint.serves(fingerprint)]
    if not ok:
        return None
    ok.sort(key=lambda p: (p.rtt_ms, p.endpoint.priority))
    best = ok[0]
    if current is not None:
        cur = next((p for p in ok if p.endpoint.url == current.url), None)
        if cur is not None and not (best.rtt_ms <= cur.rtt_ms * switch_ratio
                                    and cur.rtt_ms - best.rtt_ms >= switch_min_ms):
            return cur.endpoint
    return best.endpoint


def failover_order(probes: Iterable[Probe], *, fingerprint: str | None = None,
                   exclude: Iterable[str] = ()) -> list[Endpoint]:
    """Endpoints to try after a connection failure: healthy ones by RTT, then the unanswered ones (a probe round can
    be stale), never the URLs in ``exclude``."""
    ex = set(exclude)
    ps = [p for p in probes if p.endpoint.serves(fingerprint) and p.endpoint.url not in ex]
    ps.sort(key=lambda p: (not p.healthy, p.rtt_ms, p.endpoint.priority))
    return [p.endpoint for p in ps]


def stock_compatible(stock: Mapping[str, Any], endpoint_fingerprints: Iterable[str], ring: str) -> bool:
    """Whether a mask stock made for ``stock["fingerprint"]`` / ``stock["ring"]`` serves an endpoint: the same pack
    fingerprint among the endpoint's models and the same ring (masks are ``W8 r``: bound to the weights, not the
    host)."""
    return stock.get("ring") == ring and stock.get("fingerprint") in set(endpoint_fingerprints)
