"""Protocol messages (version 1). Normative schema: SPEC.md section 6.3.

Every message is a msgpack map with string keys. Ring tensors travel as msgpack ``bin`` values in
the packed format of :func:`pgw.core.pack_ring` (K/8 little-endian bytes per element, row-major).
Unknown keys are ignored (forward compatibility within a major version); missing required keys
are a :class:`pgw.core.ProtocolError`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PROTOCOL_HEADER",
    "PROTOCOL_VERSION",
    "SUPPORTED_VERSIONS",
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


@dataclass(frozen=True)
class ModelsResponse:
    """``GET /pgw/v1/models``. msgpack: ``{"v": 1, "type": "models", "models": [...],
    "versions": [int], "server": str}``."""

    models: tuple[ModelInfo, ...]
    versions: tuple[int, ...] = SUPPORTED_VERSIONS
    server: str = ""
    version: int = PROTOCOL_VERSION
    type: str = "models"
