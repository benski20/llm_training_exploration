"""Gateway <-> compute service protocol: HTTP/2 + msgpack (SPEC.md section 6).

Owner: wire team.
"""

from pgw.wire.client import ClientConfig, ComputeClient
from pgw.wire.codec import CONTENT_TYPE, decode_message, encode_message
from pgw.wire.messages import (
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    SUPPORTED_VERSIONS,
    ComputeItem,
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    ErrorResponse,
    LinearInfo,
    ModelInfo,
    ModelsResponse,
    ServerTiming,
)
from pgw.wire.server import WireLimits, decode_compute_request, error_response_bytes

__all__ = [
    "CONTENT_TYPE", "PROTOCOL_HEADER", "PROTOCOL_VERSION", "SUPPORTED_VERSIONS", "ClientConfig",
    "ComputeClient", "ComputeItem", "ComputeRequest", "ComputeResponse", "ComputeResult",
    "ErrorResponse", "LinearInfo", "ModelInfo", "ModelsResponse", "ServerTiming", "WireLimits",
    "decode_compute_request", "decode_message", "encode_message", "error_response_bytes",
]
