"""API keys and rate limiting for the gateway."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ApiKey", "ApiKeyStore", "RateLimiter"]


@dataclass(frozen=True)
class ApiKey:
    """Stored API key record (only the SHA-256 hash of the secret is stored)."""

    key_id: str
    hashed: str
    name: str
    scopes: tuple[str, ...] = ("inference",)
    models: tuple[str, ...] = ()  # empty = all
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    created_at: str = ""
    disabled: bool = False
    meta: dict[str, str] = field(default_factory=dict)


class ApiKeyStore:
    """File-backed key store. Keys look like ``pgw-<key_id>-<secret>``; verification is by
    ``key_id`` lookup + constant-time hash comparison."""

    def __init__(self, path: Path | None):
        raise NotImplementedError("gateway team: ApiKeyStore")

    def create(
        self, name: str, *, scopes: tuple[str, ...] = ("inference",), models: tuple[str, ...] = ()
    ) -> tuple[ApiKey, str]:
        """Create a key; returns the record and the plaintext key (shown once)."""
        raise NotImplementedError

    def verify(self, presented: str) -> ApiKey:
        """Return the key record or raise :class:`pgw.core.AuthError`."""
        raise NotImplementedError

    def revoke(self, key_id: str) -> None:
        """Disable a key."""
        raise NotImplementedError

    def list(self) -> list[ApiKey]:
        """All key records (no secrets)."""
        raise NotImplementedError


class RateLimiter:
    """Token-bucket limits per key: requests/minute and tokens/minute (prompt + completion)."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        raise NotImplementedError("gateway team: RateLimiter")

    def check_request(self, key: ApiKey, est_tokens: int) -> None:
        """Admit or raise :class:`pgw.core.RateLimitedError` (with ``retry_after_s`` in details)."""
        raise NotImplementedError

    def record_tokens(self, key: ApiKey, tokens: int) -> None:
        """Charge actual usage after completion."""
        raise NotImplementedError
