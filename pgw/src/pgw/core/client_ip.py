"""Client address behind reverse proxies (for rate limits and audit records).

The TCP peer is the client unless it is a trusted proxy (``PGW_TRUSTED_PROXY_CIDRS``, default: trust
nothing). From a trusted peer, the ``X-Forwarded-For`` chain is walked from the right and the first hop
that is not itself a trusted proxy is the client (the hops to its left were written by the client and
cannot be believed). The header is ignored from untrusted peers, so a client cannot pick its own rate
limit bucket by sending one.

This is the only forwarded-header mechanism: servers must not also rewrite the peer address (uvicorn
``proxy_headers``), or the two would disagree about which hop to trust.
"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

from pgw.core.errors import ConfigError

__all__ = ["ClientIPResolver", "client_ip", "default_resolver", "parse_cidrs"]

log = logging.getLogger(__name__)

_Net = ipaddress.IPv4Network | ipaddress.IPv6Network
_Addr = ipaddress.IPv4Address | ipaddress.IPv6Address
_MAX_HOPS = 32


def parse_cidrs(cidrs: Iterable[str]) -> tuple[_Net, ...]:
    """Parse CIDRs / single addresses (``10.0.0.0/8``, ``::1``).

    Raises:
        ConfigError: an entry is not a network or address.
    """
    out: list[_Net] = []
    for c in cidrs:
        c = str(c).strip()
        if not c:
            continue
        try:
            out.append(ipaddress.ip_network(c, strict=False))
        except ValueError as exc:
            raise ConfigError(f"trusted proxy {c!r} is not an IP network (e.g. 10.0.0.0/8)") from exc
    return tuple(out)


def _addr(text: str) -> _Addr | None:
    t = text.strip().strip('"')
    if t.startswith("["):  # [v6]:port
        t = t[1:].split("]", 1)[0]
    elif t.count(":") == 1:  # v4:port
        t = t.split(":", 1)[0]
    try:
        a = ipaddress.ip_address(t)
    except ValueError:
        return None
    if isinstance(a, ipaddress.IPv6Address) and a.ipv4_mapped is not None:
        return a.ipv4_mapped
    return a


class ClientIPResolver:
    """Resolve the client address of a request (see the module docstring).

    Args:
        trusted_proxy_cidrs: Networks of the reverse proxies in front of this process.
    """

    def __init__(self, trusted_proxy_cidrs: Iterable[str] = ()):
        self.trusted = parse_cidrs(trusted_proxy_cidrs)

    def is_trusted(self, addr: _Addr | None) -> bool:
        return addr is not None and any(addr in n for n in self.trusted)

    def resolve(self, peer: str | None, forwarded_for: Iterable[str] = ()) -> str:
        """The client address for a TCP ``peer`` and its ``X-Forwarded-For`` header value(s)."""
        if peer is None:
            return "unknown"
        p = _addr(peer)
        if p is None:
            return peer
        if not self.trusted or not self.is_trusted(p):
            return str(p)
        hops = [h for v in forwarded_for for h in str(v).split(",") if h.strip()]
        client = p
        for hop in reversed(hops[-_MAX_HOPS:]):
            a = _addr(hop)
            if a is None:  # garbage: the last trusted hop is the best we know
                break
            client = a
            if not self.is_trusted(a):
                break
        return str(client)

    def from_scope(self, scope: Mapping[str, Any]) -> str:
        """Client address of an ASGI ``scope`` (HTTP or WebSocket)."""
        c = scope.get("client")
        peer = str(c[0]) if c else None
        xff = [v.decode("latin-1") for k, v in scope.get("headers") or () if k.lower() == b"x-forwarded-for"]
        return self.resolve(peer, xff)


@lru_cache(maxsize=1)
def default_resolver() -> ClientIPResolver:
    """Resolver for ``PGW_TRUSTED_PROXY_CIDRS`` (``CommonSettings.trusted_proxy_cidrs``), built once."""
    from pgw.core.settings import CommonSettings

    cidrs = CommonSettings().trusted_proxy_cidrs
    if cidrs:
        log.info("trusting X-Forwarded-For from %d proxy network(s)", len(cidrs))
    return ClientIPResolver(cidrs)


def client_ip(request: Any, resolver: ClientIPResolver | None = None) -> str:
    """Client address of a Starlette request or an ASGI scope."""
    scope = request if isinstance(request, Mapping) else request.scope
    return (resolver or default_resolver()).from_scope(scope)
