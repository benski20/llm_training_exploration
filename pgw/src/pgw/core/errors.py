"""Error hierarchy shared by every PGW component.

Every error carries a stable :class:`ErrorCode`. The same codes are used on the wire
(``ErrorResponse.code``), in logs and in metrics labels, so a failure can be traced across the
gateway, mask factory and compute service. See SPEC.md section 6.6 for the table.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

__all__ = [
    "AuthError",
    "BackendUnavailableError",
    "ConfigError",
    "ErrorCode",
    "FingerprintMismatchError",
    "IntegrityError",
    "MaskExhaustedError",
    "MaskReuseError",
    "MaskStoreError",
    "ModelNotFoundError",
    "PGWError",
    "PayloadTooLargeError",
    "ProtocolError",
    "RateLimitedError",
    "RingOverflowError",
    "ServiceUnavailableError",
    "ShapeError",
    "VersionMismatchError",
    "error_from_code",
]


class ErrorCode(StrEnum):
    """Stable, wire-visible error codes. Never renumber or reuse a code."""

    INTERNAL = "internal"
    CONFIG = "config"
    BAD_SHAPE = "bad_shape"
    RING_OVERFLOW = "ring_overflow"
    INTEGRITY = "integrity_check_failed"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    MASK_EXHAUSTED = "mask_exhausted"
    MASK_REUSE = "mask_reuse"
    MASK_STORE = "mask_store"
    PROTOCOL = "protocol"
    VERSION_MISMATCH = "version_mismatch"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    UNAUTHENTICATED = "unauthenticated"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    MODEL_NOT_FOUND = "model_not_found"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    DEADLINE_EXCEEDED = "deadline_exceeded"


#: Default HTTP status for each code (used by service and gateway error handlers).
HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INTERNAL: 500,
    ErrorCode.CONFIG: 500,
    ErrorCode.BAD_SHAPE: 400,
    ErrorCode.RING_OVERFLOW: 400,
    ErrorCode.INTEGRITY: 502,
    ErrorCode.FINGERPRINT_MISMATCH: 409,
    ErrorCode.MASK_EXHAUSTED: 503,
    ErrorCode.MASK_REUSE: 500,
    ErrorCode.MASK_STORE: 500,
    ErrorCode.PROTOCOL: 400,
    ErrorCode.VERSION_MISMATCH: 426,
    ErrorCode.BACKEND_UNAVAILABLE: 503,
    ErrorCode.UNAUTHENTICATED: 401,
    ErrorCode.PERMISSION_DENIED: 403,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.UNAVAILABLE: 503,
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    ErrorCode.DEADLINE_EXCEEDED: 504,
}

#: Codes a client may retry (with backoff) without changing the request.
RETRYABLE: frozenset[ErrorCode] = frozenset(
    {ErrorCode.UNAVAILABLE, ErrorCode.RATE_LIMITED, ErrorCode.DEADLINE_EXCEEDED, ErrorCode.BACKEND_UNAVAILABLE}
)


class PGWError(Exception):
    """Base class of all PGW errors.

    Args:
        message: Human-readable description. Must never contain activations, masks or prompt text.
        code: Override the class default code.
        details: Small JSON-serialisable context (shapes, ids). Never payload data.
    """

    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(self, message: str, *, code: ErrorCode | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    @property
    def http_status(self) -> int:
        """HTTP status corresponding to :attr:`code`."""
        return HTTP_STATUS.get(self.code, 500)

    @property
    def retryable(self) -> bool:
        """Whether the same request may be retried."""
        return self.code in RETRYABLE

    def to_dict(self) -> dict[str, Any]:
        """Wire/JSON representation: ``{"code", "message", "retryable", "details"}``."""
        return {"code": str(self.code), "message": self.message, "retryable": self.retryable, "details": self.details}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!s}, message={self.message!r})"


class ConfigError(PGWError):
    """Invalid configuration or parameters."""

    code = ErrorCode.CONFIG


class ShapeError(PGWError):
    """Tensor shapes or dtypes do not match the contract."""

    code = ErrorCode.BAD_SHAPE


class RingOverflowError(PGWError):
    """A value would not be exactly representable in the chosen ring."""

    code = ErrorCode.RING_OVERFLOW


class IntegrityError(PGWError):
    """A compute result failed the Freivalds or range check (wrong or malicious server)."""

    code = ErrorCode.INTEGRITY


class FingerprintMismatchError(PGWError):
    """Peers disagree on the model pack fingerprint."""

    code = ErrorCode.FINGERPRINT_MISMATCH


class MaskExhaustedError(PGWError):
    """The mask store has no unconsumed masks for the requested model/ring."""

    code = ErrorCode.MASK_EXHAUSTED


class MaskReuseError(PGWError):
    """Attempt to use a mask slot twice. Always a bug or tampering; never retry."""

    code = ErrorCode.MASK_REUSE


class MaskStoreError(PGWError):
    """Mask store I/O, decryption or format failure."""

    code = ErrorCode.MASK_STORE


class ProtocolError(PGWError):
    """Malformed or unexpected wire message."""

    code = ErrorCode.PROTOCOL


class VersionMismatchError(PGWError):
    """Peer speaks an unsupported protocol version."""

    code = ErrorCode.VERSION_MISMATCH


class BackendUnavailableError(PGWError):
    """A requested compute backend (e.g. CUDA int8 GEMM) is not available."""

    code = ErrorCode.BACKEND_UNAVAILABLE


class AuthError(PGWError):
    """Missing or invalid credentials."""

    code = ErrorCode.UNAUTHENTICATED


class RateLimitedError(PGWError):
    """Caller exceeded its rate limit."""

    code = ErrorCode.RATE_LIMITED


class ServiceUnavailableError(PGWError):
    """Peer temporarily unavailable (overload, draining, network)."""

    code = ErrorCode.UNAVAILABLE


class ModelNotFoundError(PGWError):
    """Unknown model name or fingerprint."""

    code = ErrorCode.MODEL_NOT_FOUND


class PayloadTooLargeError(PGWError):
    """Request exceeds the configured size limits."""

    code = ErrorCode.PAYLOAD_TOO_LARGE


_BY_CODE: dict[ErrorCode, type[PGWError]] = {
    cls.code: cls
    for cls in (
        ConfigError, ShapeError, RingOverflowError, IntegrityError, FingerprintMismatchError,
        MaskExhaustedError, MaskReuseError, MaskStoreError, ProtocolError, VersionMismatchError,
        BackendUnavailableError, AuthError, RateLimitedError, ServiceUnavailableError,
        ModelNotFoundError, PayloadTooLargeError,
    )
}


def error_from_code(code: str, message: str, details: dict[str, Any] | None = None) -> PGWError:
    """Rebuild a typed exception from a wire error (unknown codes map to :class:`PGWError`)."""
    try:
        ec = ErrorCode(code)
    except ValueError:
        return PGWError(message, details={"unknown_code": code, **(details or {})})
    cls = _BY_CODE.get(ec, PGWError)
    return cls(message, code=ec, details=details)
