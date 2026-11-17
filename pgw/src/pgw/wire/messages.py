"""Protocol messages (version 1). Normative schema: SPEC.md section 6.3.

Every message is a msgpack map with string keys. Ring tensors travel as msgpack ``bin`` values in
the packed format of :func:`pgw.core.pack_ring` (K/8 little-endian bytes per element, row-major).
Unknown keys are ignored (forward compatibility within a major version); missing required keys
are a :class:`pgw.core.ProtocolError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pgw.core.errors import ErrorCode, PGWError, error_from_code

__all__ = [
    "ATTEMPT_HEADER",
    "COMPUTE_PATH",
    "MODELS_PATH",
    "PROTOCOL_HEADER",
    "PROTOCOL_VERSION",
    "REQUEST_ID_HEADER",
    "SUPPORTED_VERSIONS",
    "WIRE_HEADERS",
    "ComputeItem",
    "ComputeRequest",
    "ComputeResponse",
    "ComputeResult",
    "ErrorResponse",
    "LinearInfo",
    "ModelInfo",
    "ModelsResponse",
    "ServerTiming",
]

PROTOCOL_VERSION = 1
SUPPORTED_VERSIONS: tuple[int, ...] = (1,)
#: HTTP header carrying the protocol major version on every request and response.
PROTOCOL_HEADER = "PGW-Protocol"
#: Optional correlation / idempotency header (equals the request ``id``; identical on retries).
REQUEST_ID_HEADER = "PGW-Request-Id"
#: 1-based attempt counter sent by :class:`pgw.wire.ComputeClient` (diagnostics only).
ATTEMPT_HEADER = "PGW-Attempt"
#: Header names used by the protocol (informational).
WIRE_HEADERS: tuple[str, ...] = (PROTOCOL_HEADER, REQUEST_ID_HEADER, ATTEMPT_HEADER)
#: HTTP paths (SPEC 6.2).
COMPUTE_PATH = "/pgw/v1/compute"
MODELS_PATH = "/pgw/v1/models"


@dataclass(frozen=True)
class ComputeItem:
    """One masked linear evaluation. msgpack: ``{"linear": str, "rows": int, "x_hat": bin}``.

    ``x_hat`` holds ``rows * in_features`` ring elements; ``in_features`` is implied by ``linear``.
    """

    linear: str
    rows: int
    x_hat: bytes


@dataclass(frozen=True)
class ComputeRequest:
    """``POST /pgw/v1/compute`` body.

    msgpack: ``{"v": 1, "type": "compute", "id": str, "fp": str, "ring": str, "items": [item...],
    "deadline_ms": int?}``. ``id`` is a client-chosen request id (UUID hex) echoed in the response
    and used only for correlation. Items are independent and may name different linears.
    """

    request_id: str
    fingerprint: str
    ring: str
    items: tuple[ComputeItem, ...]
    deadline_ms: int | None = None
    version: int = PROTOCOL_VERSION
    type: str = "compute"

    @property
    def payload_bytes(self) -> int:
        """Total size of all ``x_hat`` blobs (excludes msgpack framing)."""
        return sum(len(it.x_hat) for it in self.items)

    @property
    def total_rows(self) -> int:
        """Sum of ``rows`` over all items."""
        return sum(int(it.rows) for it in self.items)


@dataclass(frozen=True)
class ComputeResult:
    """msgpack: ``{"linear": str, "rows": int, "y_hat": bin}`` (``rows * out_features`` elements)."""

    linear: str
    rows: int
    y_hat: bytes


@dataclass(frozen=True)
class ServerTiming:
    """msgpack: ``{"queue_ms": float, "compute_ms": float, "batch_rows": int}``."""

    queue_ms: float = 0.0
    compute_ms: float = 0.0
    batch_rows: int = 0


@dataclass(frozen=True)
class ComputeResponse:
    """msgpack: ``{"v": 1, "type": "compute_result", "id": str, "items": [result...], "timing": {...}}``.
    Results are in request item order."""

    request_id: str
    items: tuple[ComputeResult, ...]
    timing: ServerTiming = field(default_factory=ServerTiming)
    version: int = PROTOCOL_VERSION
    type: str = "compute_result"

    @property
    def payload_bytes(self) -> int:
        """Total size of all ``y_hat`` blobs."""
        return sum(len(it.y_hat) for it in self.items)


@dataclass(frozen=True)
class ErrorResponse:
    """msgpack (or JSON for non-msgpack clients): ``{"v": 1, "type": "error", "id": str?,
    "code": str, "message": str, "retryable": bool, "details": map}``. ``code`` is a
    :class:`pgw.core.ErrorCode` value."""

    code: str
    message: str
    retryable: bool = False
    request_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    version: int = PROTOCOL_VERSION
    type: str = "error"

    @classmethod
    def from_exception(cls, exc: BaseException, request_id: str | None = None) -> ErrorResponse:
        """Wire form of ``exc``. Non-:class:`PGWError` exceptions become a generic ``internal``
        error whose message and details are *not* leaked; the same holds for a :class:`PGWError`
        whose code is ``internal`` (SPEC 6.6: "bug; details not leaked")."""
        if isinstance(exc, PGWError) and exc.code != ErrorCode.INTERNAL:
            return cls(
                code=str(exc.code),
                message=exc.message,
                retryable=exc.retryable,
                request_id=request_id,
                details=dict(exc.details),
            )
        return cls(code=str(ErrorCode.INTERNAL), message="internal server error", request_id=request_id)

    def to_exception(self) -> PGWError:
        """Typed exception for this error (unknown codes become a plain :class:`PGWError`)."""
        details = dict(self.details)
        if self.request_id is not None:
            details.setdefault("request_id", self.request_id)
        return error_from_code(self.code, self.message, details)


@dataclass(frozen=True)
class LinearInfo:
    """msgpack: ``{"id": str, "m": int, "n": int}``."""

    id: str
    out_features: int
    in_features: int


@dataclass(frozen=True)
class ModelInfo:
    """One served model. msgpack: ``{"name", "fp", "rings": [str], "linears": [LinearInfo],
    "max_rows_per_item": int, "max_request_bytes": int}``."""

    name: str
    fingerprint: str
    rings: tuple[str, ...]
    linears: tuple[LinearInfo, ...]
    max_rows_per_item: int
    max_request_bytes: int

    def linear_dims(self) -> dict[str, tuple[int, int]]:
        """``{linear_id: (out_features m, in_features n)}``."""
        return {li.id: (li.out_features, li.in_features) for li in self.linears}


@dataclass(frozen=True)
class ModelsResponse:
    """``GET /pgw/v1/models``. msgpack: ``{"v": 1, "type": "models", "models": [...],
    "versions": [int], "server": str}``."""

    models: tuple[ModelInfo, ...]
    versions: tuple[int, ...] = SUPPORTED_VERSIONS
    server: str = ""
    version: int = PROTOCOL_VERSION
    type: str = "models"

    def find(self, fingerprint: str) -> ModelInfo | None:
        """The served model with ``fingerprint`` (``None`` if absent)."""
        for m in self.models:
            if m.fingerprint == fingerprint:
                return m
        return None
