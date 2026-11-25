"""HTTP session to the provider's ``/pgw/v1/provision/...`` endpoints (credentials from the SDK bootstrap)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pgw.core.errors import ErrorCode, ModelNotFoundError, PGWError, ServiceUnavailableError

__all__ = ["ProvisionSession"]

PREFIX = "/pgw/v1/provision"


@dataclass
class ProvisionSession:
    """Where and how to reach the provider. ``token`` is the service token (never logged); ``ca_file`` pins the
    provider CA; ``client_cert_file``/``client_key_file`` are the enrolled client certificate. ``transport``: an
    ``httpx`` transport (tests)."""

    base_url: str
    token: str | None = field(default=None, repr=False)
    ca_file: Path | None = None
    client_cert_file: Path | None = None
    client_key_file: Path | None = None
    timeout_s: float = 600.0
    retries: int = 4
    transport: Any = None
    _client: Any = field(default=None, repr=False)

    def client(self) -> Any:
        import httpx

        if self._client is None:
            kw: dict[str, Any] = {"base_url": self.base_url, "timeout": self.timeout_s, "follow_redirects": False,
                                  "trust_env": True,
                                  "headers": {"PGW-Protocol": "1", "User-Agent": "pgw-provision"}}
            if self.token:
                kw["headers"]["Authorization"] = f"Bearer {self.token}"
            if self.transport is not None:
                kw["transport"] = self.transport
            else:
                kw["verify"] = str(self.ca_file) if self.ca_file is not None else True
                if self.client_cert_file is not None and self.client_key_file is not None:
                    kw["cert"] = (str(self.client_cert_file), str(self.client_key_file))
            self._client = httpx.Client(**kw)
        return self._client

    def close(self) -> None:
        c, self._client = self._client, None
        if c is not None:
            c.close()

    def request(self, method: str, path: str, *, params: dict[str, Any] | None = None, json: Any = None) -> Any:
        """One request with retries on transport errors and 503; raises :class:`PGWError` on an error reply."""
        import httpx

        last: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.client().request(method, PREFIX + path, params=params, json=json)
            except httpx.TransportError as exc:
                last = exc
                time.sleep(min(8.0, 0.25 * 2 ** attempt))
                continue
            if r.status_code == 503 and attempt < self.retries:
                last = ServiceUnavailableError(f"provider unavailable ({_message(r)})")
                time.sleep(min(8.0, 0.25 * 2 ** attempt))
                continue
            if r.status_code == 404:
                raise ModelNotFoundError(_message(r))
            if r.status_code >= 400:
                raise PGWError(f"provider refused {method} {path}: HTTP {r.status_code} ({_message(r)})",
                               code=_code(r))
            return r
        raise ServiceUnavailableError(f"provider unreachable for {method} {path}: {last!r}")

    def get_json(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, params=params or None).json()

    def get_bytes(self, path: str, **params: Any) -> bytes:
        return bytes(self.request("GET", path, params=params or None).content)


def _message(r: Any) -> str:
    try:
        return str(r.json().get("message", ""))[:300]
    except Exception:
        return r.text[:300] if hasattr(r, "text") else ""


def _code(r: Any) -> ErrorCode:
    try:
        return ErrorCode(r.json().get("code"))
    except Exception:
        return ErrorCode.BACKEND_UNAVAILABLE
