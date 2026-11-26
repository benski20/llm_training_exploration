"""Typed exceptions of the client library, named like the ``openai`` package's.

Every exception derives from :class:`PGWAPIError` (alias ``APIError``). ``code`` is the stable
:class:`pgw.core.errors.ErrorCode` value, ``status_code`` the HTTP status the gateway would have answered.

Retry policy (:attr:`PGWAPIError.retryable`): only transport failures and temporary unavailability are retried,
and only before any token of the answer was produced. :class:`IntegrityError` is never retried: a provider result
that fails its check is a hard error, and the model refuses traffic until the integrity breaker's cool-down ends.
"""

from __future__ import annotations

import errno
from typing import Any

from pgw.core.errors import ErrorCode, PGWError

__all__ = [
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "IntegrityError",
    "InternalServerError",
    "NotFoundError",
    "NotReadyError",
    "PGWAPIError",
    "PermissionDeniedError",
    "ProviderStalledError",
    "RateLimitError",
    "UnsupportedModelError",
    "from_pgw_error",
]


class PGWAPIError(Exception):
    """Base class of the library's errors.

    Attributes:
        message: Human-readable description (never contains prompts, completions or secrets).
        code: :class:`ErrorCode` value (string).
        status_code: The HTTP status an OpenAI-compatible server would have answered.
        request_id: The request's id, when one was assigned.
        details: Small JSON-serialisable context.
    """

    status_code: int = 500
    default_code: str = ErrorCode.INTERNAL.value
    retryable: bool = False

    def __init__(self, message: str, *, code: str | None = None, request_id: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.request_id = request_id
        self.details: dict[str, Any] = dict(details or {})

    @property
    def body(self) -> dict[str, Any]:
        """The OpenAI-shaped error body ``{"message", "type", "code"}``."""
        return {"message": self.message, "type": type(self).__name__, "code": self.code}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


APIError = PGWAPIError


class BadRequestError(PGWAPIError):
    """The request is invalid (unsupported feature, bad parameter)."""

    status_code = 400
    default_code = ErrorCode.PROTOCOL.value


class AuthenticationError(PGWAPIError):
    """The API key is missing, malformed, unknown or revoked."""

    status_code = 401
    default_code = ErrorCode.UNAUTHENTICATED.value


class PermissionDeniedError(PGWAPIError):
    """The key may not use this model or operation."""

    status_code = 403
    default_code = ErrorCode.PERMISSION_DENIED.value


class NotFoundError(PGWAPIError):
    """The model does not exist at the provider."""

    status_code = 404
    default_code = ErrorCode.MODEL_NOT_FOUND.value


class UnsupportedModelError(NotFoundError):
    """The model is not supported by this release of the library."""


class RateLimitError(PGWAPIError):
    """The provider rate-limited the key (retried with back-off; raised when retries are exhausted)."""

    status_code = 429
    default_code = ErrorCode.RATE_LIMITED.value
    retryable = True


class NotReadyError(PGWAPIError):
    """Not ready yet: provisioning, connecting to the provider, or no mask in stock after the wait budget."""

    status_code = 503
    default_code = ErrorCode.UNAVAILABLE.value
    retryable = True


class APIConnectionError(PGWAPIError):
    """The provider could not be reached (network, TLS, DNS)."""

    status_code = 503
    default_code = ErrorCode.BACKEND_UNAVAILABLE.value
    retryable = True


class ProviderStalledError(APIConnectionError):
    """The provider answers its health check but its mask generation makes no progress (a wedged GPU, a hung
    kernel): the maskgen watchdog of the background mask factory saw an item of this key run past its bound.
    Retryable (the factory resumes by itself once the provider's operator restarts it); ``details["reason"]`` is
    ``"provider_maskgen_stalled"``."""


class APITimeoutError(APIConnectionError):
    """The request did not finish within the client's timeout."""

    status_code = 504
    default_code = ErrorCode.DEADLINE_EXCEEDED.value


class IntegrityError(PGWAPIError):
    """A provider result failed its integrity check (Freivalds / prime-field / range check, or the
    outbound-noise monitor alarmed). Hard error: never retried, never returned as a success."""

    status_code = 502
    default_code = ErrorCode.INTEGRITY.value


class InternalServerError(PGWAPIError):
    """An unexpected error inside the library or at the provider."""

    status_code = 500


_BY_CODE: dict[str, type[PGWAPIError]] = {
    ErrorCode.PROTOCOL.value: BadRequestError,
    ErrorCode.BAD_SHAPE.value: BadRequestError,
    ErrorCode.RING_OVERFLOW.value: BadRequestError,
    ErrorCode.PAYLOAD_TOO_LARGE.value: BadRequestError,
    ErrorCode.UNAUTHENTICATED.value: AuthenticationError,
    ErrorCode.PERMISSION_DENIED.value: PermissionDeniedError,
    ErrorCode.MODEL_NOT_FOUND.value: NotFoundError,
    ErrorCode.RATE_LIMITED.value: RateLimitError,
    ErrorCode.UNAVAILABLE.value: NotReadyError,
    ErrorCode.MASK_EXHAUSTED.value: NotReadyError,
    ErrorCode.BACKEND_UNAVAILABLE.value: APIConnectionError,
    ErrorCode.DEADLINE_EXCEEDED.value: APITimeoutError,
    ErrorCode.INTEGRITY.value: IntegrityError,
    ErrorCode.FINGERPRINT_MISMATCH.value: IntegrityError,
    ErrorCode.MASK_REUSE.value: IntegrityError,
}


_DISK_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EROFS})


def from_pgw_error(exc: BaseException, *, request_id: str | None = None) -> PGWAPIError:
    """Map an internal :class:`PGWError` (or another exception) to the library's typed error."""
    if isinstance(exc, PGWAPIError):
        return exc
    if isinstance(exc, PGWError):
        cls = _BY_CODE.get(str(exc.code), InternalServerError)
        details = {k: v for k, v in exc.details.items() if isinstance(v, str | int | float | bool | None)}
        msg = exc.message
        low = msg.lower()
        if ("disk is full" in low or "no space left" in low) and "free disk space" not in low:
            msg += " (the disk holding the pgw cache is full: free disk space there and retry)"
        return cls(msg, code=str(exc.code), request_id=request_id, details=details)
    if isinstance(exc, TimeoutError):
        return APITimeoutError("the request timed out", request_id=request_id)
    if isinstance(exc, OSError) and not isinstance(exc, ConnectionError) and exc.errno in _DISK_ERRNOS:
        where = f" ({exc.filename})" if exc.filename else ""
        return InternalServerError(f"the disk holding the pgw cache is full or read-only{where}: "
                                   f"{exc.strerror or type(exc).__name__}. Free disk space there and retry",
                                   request_id=request_id)
    if isinstance(exc, ConnectionError | OSError):
        return APIConnectionError(f"connection failed ({type(exc).__name__})", request_id=request_id)
    return InternalServerError(f"internal error ({type(exc).__name__})", request_id=request_id)
