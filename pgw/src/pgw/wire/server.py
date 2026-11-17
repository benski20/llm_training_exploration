"""Server-side helpers shared by the compute service (framework-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass

from pgw.wire.messages import ComputeRequest

__all__ = ["WireLimits", "check_protocol_header", "decode_compute_request", "error_response_bytes"]


@dataclass(frozen=True)
class WireLimits:
    """Request limits enforced before any compute."""

    max_request_bytes: int = 256 << 20
    max_items: int = 256
    max_rows_per_item: int = 8192


def check_protocol_header(value: str | None) -> int:
    """Parse the ``PGW-Protocol`` header; raise :class:`VersionMismatchError` if unsupported."""
    raise NotImplementedError("wire team: check_protocol_header")


def decode_compute_request(body: bytes, limits: WireLimits, linear_dims: dict[str, tuple[int, int]]) -> ComputeRequest:
    """Decode + validate a compute request against limits and the model's ``{linear: (m, n)}``
    (payload length must equal ``rows * n * K/8``)."""
    raise NotImplementedError("wire team: decode_compute_request")


def error_response_bytes(exc: BaseException, request_id: str | None = None) -> tuple[int, bytes]:
    """Map any exception to ``(http_status, msgpack ErrorResponse)``; internal errors are not leaked."""
    raise NotImplementedError("wire team: error_response_bytes")
