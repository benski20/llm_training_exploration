"""Gateway <-> compute service protocol: HTTP/2 + msgpack (SPEC.md section 6).

* :mod:`pgw.wire.messages` - message dataclasses (protocol v1).
* :mod:`pgw.wire.codec` - strict msgpack codec with versioning and size limits.
* :mod:`pgw.wire.client` - async pooled HTTP/2 client (mTLS, retries, deadlines, splitting).
* :mod:`pgw.wire.server` - validation helpers and the FastAPI router the service mounts.

Owner: wire team.
"""

from pgw.wire.client import ClientConfig, ClientStats, ComputeClient, build_ssl_context
from pgw.wire.codec import (
    CONTENT_TYPE,
    JSON_CONTENT_TYPE,
    decode_any,
    decode_error_json,
    decode_message,
    encode_error_json,
    encode_message,
)
from pgw.wire.messages import (
    ATTEMPT_HEADER,
    COMPUTE_PATH,
    DEADLINE_HEADER,
    MODELS_PATH,
    PROTOCOL_HEADER,
    PROTOCOL_VERSION,
    REQUEST_ID_HEADER,
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
from pgw.wire.server import (
    ComputeHandler,
    RequestContext,
    WireLimits,
    check_content_type,
    check_protocol_header,
    create_router,
    decode_compute_request,
    error_headers,
    error_response_bytes,
    response_headers,
    validate_compute_response,
)

__all__ = [
    "ATTEMPT_HEADER",
    "COMPUTE_PATH",
    "CONTENT_TYPE",
    "DEADLINE_HEADER",
    "JSON_CONTENT_TYPE",
    "MODELS_PATH",
    "PROTOCOL_HEADER",
    "PROTOCOL_VERSION",
    "REQUEST_ID_HEADER",
    "SUPPORTED_VERSIONS",
    "ClientConfig",
    "ClientStats",
    "ComputeClient",
    "ComputeHandler",
    "ComputeItem",
    "ComputeRequest",
    "ComputeResponse",
    "ComputeResult",
    "ErrorResponse",
    "LinearInfo",
    "ModelInfo",
    "ModelsResponse",
    "RequestContext",
    "ServerTiming",
    "WireLimits",
    "build_ssl_context",
    "check_content_type",
    "check_protocol_header",
    "create_router",
    "decode_any",
    "decode_compute_request",
    "decode_error_json",
    "decode_message",
    "encode_error_json",
    "encode_message",
    "error_headers",
    "error_response_bytes",
    "response_headers",
    "validate_compute_response",
]
