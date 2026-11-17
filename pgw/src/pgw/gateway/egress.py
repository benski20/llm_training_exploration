"""Egress control: the gateway may only talk to its configured compute-service endpoints.

:class:`EgressGuardTransport` wraps the httpx transport of every outbound client the gateway
creates and refuses any request whose origin (scheme, host, port) is not on the allowlist. This is
defence in depth on top of a network-level egress firewall (recommended in every deployment).

:func:`disable_library_telemetry` switches off the network features and telemetry of the libraries
the gateway loads (Hugging Face hub/transformers), so the process never phones home.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import httpx

from pgw.core.errors import ErrorCode, PGWError

__all__ = ["EgressBlockedError", "EgressGuardTransport", "disable_library_telemetry"]

log = logging.getLogger(__name__)

#: Environment switches that keep HF libraries offline and silent.
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "DO_NOT_TRACK": "1",
    "DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


class EgressBlockedError(PGWError):
    """An outbound request to a host that is not on the egress allowlist was refused."""

    code = ErrorCode.PERMISSION_DENIED


def _origin(url: httpx.URL) -> tuple[str, str, int]:
    scheme = url.scheme.lower()
    port = url.port if url.port is not None else (443 if scheme == "https" else 80)
    return scheme, (url.host or "").lower(), int(port)


class EgressGuardTransport(httpx.AsyncBaseTransport):
    """httpx transport that only forwards requests to allowlisted origins.

    Args:
        inner: The real transport (``httpx.AsyncHTTPTransport`` in production, an
            ``ASGITransport`` in tests).
        allowed: Allowed ``(scheme, host, port)`` origins.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport, allowed: Iterable[tuple[str, str, int]]):
        self.inner = inner
        self.allowed = frozenset(allowed)
        self.blocked = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Forward or refuse."""
        origin = _origin(request.url)
        if origin not in self.allowed:
            self.blocked += 1
            log.error("egress blocked", extra={"scheme": origin[0], "host": origin[1], "port": origin[2]})
            raise EgressBlockedError(
                "outbound request to a non-allowlisted host refused",
                details={"host": origin[1], "port": origin[2]},
            )
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        """Close the inner transport."""
        await self.inner.aclose()


def disable_library_telemetry() -> None:
    """Force Hugging Face libraries offline (models and tokenizers load from the local pack only)."""
    for k, v in _OFFLINE_ENV.items():
        os.environ[k] = v
    try:  # already-imported huggingface_hub reads constants at import time
        import huggingface_hub.constants as hc

        hc.HF_HUB_OFFLINE = True
        hc.HF_HUB_DISABLE_TELEMETRY = True
    except Exception:  # pragma: no cover - optional dependency / attribute layout
        log.debug("huggingface_hub constants not patched")
