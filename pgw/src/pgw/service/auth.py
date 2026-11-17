"""Tenant authentication for the compute service (bearer tokens + optional mTLS identity)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["TenantAuthenticator", "TenantIdentity"]


@dataclass(frozen=True)
class TenantIdentity:
    """Authenticated caller (used for quotas, billing counters and metrics labels)."""

    tenant_id: str
    via: str  # "token" | "mtls" | "anonymous"


class TenantAuthenticator:
    """Verifies ``Authorization: Bearer <token>`` against SHA-256 hashes (constant-time compare)
    and/or the mTLS client certificate subject."""

    def __init__(self, tokens_file: Path | None, *, allow_unauthenticated: bool = False):
        raise NotImplementedError("service team: TenantAuthenticator")

    def authenticate(self, authorization: str | None, client_cert_subject: str | None = None) -> TenantIdentity:
        """Return the identity or raise :class:`pgw.core.AuthError`."""
        raise NotImplementedError
