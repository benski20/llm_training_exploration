"""msgpack encoding/decoding of :mod:`pgw.wire.messages` with strict validation.

The codec is the schema of the protocol (SPEC.md 6.1: there is no IDL). Decoding is defensive:
every message from the network is size-limited before parsing, msgpack limits bound container
sizes, ``ext`` types are rejected, and every field is type-checked. Unknown keys are ignored
(forward compatibility within a major version, SPEC 6.4); missing or mistyped required keys raise
:class:`pgw.core.ProtocolError`. No payload bytes ever appear in error messages.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any, TypeVar, cast

import msgpack

from pgw.core.errors import (
    ConfigError,
    PayloadTooLargeError,
    PGWError,
    ProtocolError,
    VersionMismatchError,
)
from pgw.core.ring import RingParams
from pgw.wire.messages import (
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

__all__ = [
    "CONTENT_TYPE",
    "JSON_CONTENT_TYPE",
    "MESSAGE_TYPES",
    "decode_any",
    "decode_error_json",
    "decode_message",
    "encode_error_json",
    "encode_message",
    "message_to_dict",
    "validate_request_id",
]

#: Media type of every PGW body. The protocol major version is also in the ``PGW-Protocol`` header.
CONTENT_TYPE = "application/vnd.pgw+msgpack"
#: Error bodies may also be JSON (for ``curl`` / non-msgpack clients).
JSON_CONTENT_TYPE = "application/json"

T = TypeVar("T")

# Bounds on msgpack containers (independent of payload size, which is bounded by max_bytes).
_MAX_MAP_LEN = 4096
_MAX_ARRAY_LEN = 1 << 20
_MAX_STR_LEN = 1 << 20
_MAX_ID_LEN = 128
_MAX_LINEAR_LEN = 64
_MAX_RING_LEN = 16
_MAX_DEPTH = 32
_FP_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
_LINEAR_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


# ---------------------------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------------------------
def _item_to_dict(it: ComputeItem) -> dict[str, Any]:
    return {"linear": it.linear, "rows": int(it.rows), "x_hat": bytes(it.x_hat)}


def _result_to_dict(it: ComputeResult) -> dict[str, Any]:
    return {"linear": it.linear, "rows": int(it.rows), "y_hat": bytes(it.y_hat)}


def _timing_to_dict(t: ServerTiming) -> dict[str, Any]:
    return {"queue_ms": float(t.queue_ms), "compute_ms": float(t.compute_ms), "batch_rows": int(t.batch_rows)}


def _linear_to_dict(li: LinearInfo) -> dict[str, Any]:
    return {"id": li.id, "m": int(li.out_features), "n": int(li.in_features)}


def _model_to_dict(m: ModelInfo) -> dict[str, Any]:
    return {
        "name": m.name,
        "fp": m.fingerprint,
        "rings": list(m.rings),
        "linears": [_linear_to_dict(li) for li in m.linears],
        "max_rows_per_item": int(m.max_rows_per_item),
        "max_request_bytes": int(m.max_request_bytes),
    }


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Make ``details`` values msgpack/JSON-safe (str keys; scalars, lists, maps; no bytes)."""
    if depth > 8:
        return str(value)
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_safe(v, depth + 1) for v in value]
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{len(value)} bytes>"  # never put raw bytes into error details
    return str(value)


def message_to_dict(msg: object) -> dict[str, Any]:
    """Plain-dict (wire key) form of a message dataclass, as it is msgpack-encoded."""
    if isinstance(msg, ComputeRequest):
        d: dict[str, Any] = {
            "v": int(msg.version),
            "type": "compute",
            "id": msg.request_id,
            "fp": msg.fingerprint,
            "ring": msg.ring,
            "items": [_item_to_dict(it) for it in msg.items],
        }
        if msg.deadline_ms is not None:
            d["deadline_ms"] = int(msg.deadline_ms)
        return d
    if isinstance(msg, ComputeResponse):
        return {
            "v": int(msg.version),
            "type": "compute_result",
            "id": msg.request_id,
            "items": [_result_to_dict(it) for it in msg.items],
            "timing": _timing_to_dict(msg.timing),
        }
    if isinstance(msg, ErrorResponse):
        d = {
            "v": int(msg.version),
            "type": "error",
            "code": str(msg.code),
            "message": str(msg.message),
            "retryable": bool(msg.retryable),
            "details": _json_safe(dict(msg.details)),
        }
        if msg.request_id is not None:
            d["id"] = msg.request_id
        return d
    if isinstance(msg, ModelsResponse):
        return {
            "v": int(msg.version),
            "type": "models",
            "versions": [int(v) for v in msg.versions],
            "server": msg.server,
            "models": [_model_to_dict(m) for m in msg.models],
        }
    raise TypeError(f"not a PGW wire message: {type(msg).__name__}")


def encode_message(msg: object) -> bytes:
    """Serialize any message dataclass to msgpack bytes (``use_bin_type=True``).

    Raises:
        TypeError: ``msg`` is not a message dataclass (a programming error).
    """
    out = msgpack.packb(message_to_dict(msg), use_bin_type=True)
    return cast(bytes, out)


def encode_error_json(err: ErrorResponse) -> bytes:
    """JSON body of an :class:`ErrorResponse` (for clients that do not accept msgpack)."""
    return json.dumps(message_to_dict(err), separators=(",", ":"), allow_nan=False).encode()


# ---------------------------------------------------------------------------------------------
# decoding helpers
# ---------------------------------------------------------------------------------------------
def _ext_hook(code: int, data: bytes) -> Any:
    raise ProtocolError(f"msgpack ext type {code} is not allowed")


def _unpack(data: bytes | bytearray | memoryview) -> Any:
    n = len(data)
    try:
        return msgpack.unpackb(
            data,
            raw=False,
            use_list=True,
            strict_map_key=True,
            ext_hook=_ext_hook,
            max_bin_len=n,
            max_str_len=min(n, _MAX_STR_LEN),
            max_array_len=min(n, _MAX_ARRAY_LEN),
            max_map_len=min(n, _MAX_MAP_LEN),
            max_ext_len=0,
        )
    except PGWError:
        raise
    except msgpack.ExtraData as exc:
        raise ProtocolError("trailing bytes after msgpack message") from exc
    except (ValueError, TypeError, msgpack.UnpackException, RecursionError, OverflowError) as exc:
        raise ProtocolError(f"malformed msgpack body: {type(exc).__name__}") from exc


def _check_depth(obj: Any, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise ProtocolError("message nesting too deep")
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, dict | list):
                _check_depth(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict | list):
                _check_depth(v, depth + 1)


def _where(ctx: str, key: str) -> str:
    return f"{ctx}.{key}" if ctx else key


def _req(d: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ProtocolError(f"missing required key {_where(ctx, key)!r}")
    return d[key]


def _map(v: Any, what: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise ProtocolError(f"{what} must be a map, got {type(v).__name__}")
    return v


def _list(v: Any, what: str) -> list[Any]:
    if not isinstance(v, list):
        raise ProtocolError(f"{what} must be an array, got {type(v).__name__}")
    return v


def _int(v: Any, what: str, *, lo: int | None = None, hi: int | None = None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ProtocolError(f"{what} must be an integer, got {type(v).__name__}")
    if lo is not None and v < lo:
        raise ProtocolError(f"{what} must be >= {lo}, got {v}")
    if hi is not None and v > hi:
        raise ProtocolError(f"{what} must be <= {hi}, got {v}")
    return v


def _float(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ProtocolError(f"{what} must be a number, got {type(v).__name__}")
    f = float(v)
    if not math.isfinite(f):
        raise ProtocolError(f"{what} must be finite")
    return f


def _bool(v: Any, what: str) -> bool:
    if not isinstance(v, bool):
        raise ProtocolError(f"{what} must be a boolean, got {type(v).__name__}")
    return v


def _str(v: Any, what: str, *, max_len: int = _MAX_STR_LEN, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(v, str):
        raise ProtocolError(f"{what} must be a string, got {type(v).__name__}")
    if len(v) > max_len:
        raise ProtocolError(f"{what} is too long ({len(v)} > {max_len})")
    if pattern is not None and not pattern.match(v):
        raise ProtocolError(f"{what} has an invalid format")
    return v


def _bin(v: Any, what: str) -> bytes:
    if not isinstance(v, bytes):
        raise ProtocolError(f"{what} must be msgpack bin, got {type(v).__name__}")
    return v


def validate_request_id(v: Any, what: str = "id") -> str:
    """Request ids: 1-128 chars of ``[A-Za-z0-9._:-]`` (UUID hex recommended)."""
    return _str(v, what, max_len=_MAX_ID_LEN, pattern=_ID_RE)


def _fp(v: Any, what: str) -> str:
    return _str(v, what, max_len=64, pattern=_FP_RE)


def _ring(v: Any, what: str) -> str:
    s = _str(v, what, max_len=_MAX_RING_LEN)
    try:
        params = RingParams.from_name(s)
    except ConfigError as exc:
        raise ProtocolError(f"{what} is not a valid ring name") from exc
    return params.name


def _linear(v: Any, what: str) -> str:
    return _str(v, what, max_len=_MAX_LINEAR_LEN, pattern=_LINEAR_RE)


# ---------------------------------------------------------------------------------------------
# per-type parsers (input: the top-level map, already checked for "v" and "type")
# ---------------------------------------------------------------------------------------------
def _parse_compute_request(d: dict[str, Any], v: int) -> ComputeRequest:
    items_raw = _list(_req(d, "items", ""), "items")
    if not items_raw:
        raise ProtocolError("compute request has no items")
    items = []
    for i, raw in enumerate(items_raw):
        ctx = f"items[{i}]"
        it = _map(raw, ctx)
        items.append(
            ComputeItem(
                linear=_linear(_req(it, "linear", ctx), f"{ctx}.linear"),
                rows=_int(_req(it, "rows", ctx), f"{ctx}.rows", lo=1),
                x_hat=_bin(_req(it, "x_hat", ctx), f"{ctx}.x_hat"),
            )
        )
    deadline = d.get("deadline_ms")
    if deadline is not None:
        deadline = _int(deadline, "deadline_ms", lo=0, hi=7 * 24 * 3600 * 1000)
    return ComputeRequest(
        request_id=validate_request_id(_req(d, "id", "")),
        fingerprint=_fp(_req(d, "fp", ""), "fp"),
        ring=_ring(_req(d, "ring", ""), "ring"),
        items=tuple(items),
        deadline_ms=deadline,
        version=v,
    )


def _parse_timing(raw: Any) -> ServerTiming:
    if raw is None:
        return ServerTiming()
    t = _map(raw, "timing")
    return ServerTiming(
        queue_ms=_float(t.get("queue_ms", 0.0), "timing.queue_ms"),
        compute_ms=_float(t.get("compute_ms", 0.0), "timing.compute_ms"),
        batch_rows=_int(t.get("batch_rows", 0), "timing.batch_rows", lo=0),
    )


def _parse_compute_response(d: dict[str, Any], v: int) -> ComputeResponse:
    items = []
    for i, raw in enumerate(_list(_req(d, "items", ""), "items")):
        ctx = f"items[{i}]"
        it = _map(raw, ctx)
        items.append(
            ComputeResult(
                linear=_linear(_req(it, "linear", ctx), f"{ctx}.linear"),
                rows=_int(_req(it, "rows", ctx), f"{ctx}.rows", lo=1),
                y_hat=_bin(_req(it, "y_hat", ctx), f"{ctx}.y_hat"),
            )
        )
    return ComputeResponse(
        request_id=validate_request_id(_req(d, "id", "")),
        items=tuple(items),
        timing=_parse_timing(d.get("timing")),
        version=v,
    )


def _parse_error(d: dict[str, Any], v: int) -> ErrorResponse:
    rid = d.get("id")
    details = d.get("details") or {}
    return ErrorResponse(
        code=_str(_req(d, "code", ""), "code", max_len=64),
        message=_str(_req(d, "message", ""), "message", max_len=16384),
        retryable=_bool(d.get("retryable", False), "retryable"),
        request_id=None if rid is None else _str(rid, "id", max_len=_MAX_ID_LEN),
        details=_map(details, "details"),
        version=v,
    )


def _parse_models(d: dict[str, Any], v: int) -> ModelsResponse:
    models = []
    for i, raw in enumerate(_list(_req(d, "models", ""), "models")):
        ctx = f"models[{i}]"
        m = _map(raw, ctx)
        linears = []
        for j, lraw in enumerate(_list(_req(m, "linears", ctx), f"{ctx}.linears")):
            lctx = f"{ctx}.linears[{j}]"
            li = _map(lraw, lctx)
            linears.append(
                LinearInfo(
                    id=_linear(_req(li, "id", lctx), f"{lctx}.id"),
                    out_features=_int(_req(li, "m", lctx), f"{lctx}.m", lo=1),
                    in_features=_int(_req(li, "n", lctx), f"{lctx}.n", lo=1),
                )
            )
        rings = tuple(_ring(r, f"{ctx}.rings[]") for r in _list(_req(m, "rings", ctx), f"{ctx}.rings"))
        models.append(
            ModelInfo(
                name=_str(_req(m, "name", ctx), f"{ctx}.name", max_len=256),
                fingerprint=_fp(_req(m, "fp", ctx), f"{ctx}.fp"),
                rings=rings,
                linears=tuple(linears),
                max_rows_per_item=_int(_req(m, "max_rows_per_item", ctx), f"{ctx}.max_rows_per_item", lo=1),
                max_request_bytes=_int(_req(m, "max_request_bytes", ctx), f"{ctx}.max_request_bytes", lo=1),
            )
        )
    versions = tuple(_int(x, "versions[]", lo=0) for x in _list(d.get("versions", [v]), "versions"))
    return ModelsResponse(
        models=tuple(models),
        versions=versions,
        server=_str(d.get("server", ""), "server", max_len=256),
        version=v,
    )


_PARSERS: dict[str, tuple[type[Any], Callable[[dict[str, Any], int], Any]]] = {
    "compute": (ComputeRequest, _parse_compute_request),
    "compute_result": (ComputeResponse, _parse_compute_response),
    "error": (ErrorResponse, _parse_error),
    "models": (ModelsResponse, _parse_models),
}

#: ``{"type" value: message class}``.
MESSAGE_TYPES: dict[str, type[Any]] = {k: cls for k, (cls, _) in _PARSERS.items()}


def _parse_top(obj: Any) -> tuple[str, int, dict[str, Any]]:
    d = _map(obj, "message")
    _check_depth(d)
    v = _int(_req(d, "v", ""), "v")
    if v not in SUPPORTED_VERSIONS:
        raise VersionMismatchError(
            f"unsupported protocol version {v}",
            details={"got": v, "versions": list(SUPPORTED_VERSIONS)},
        )
    mtype = _str(_req(d, "type", ""), "type", max_len=64)
    if mtype not in _PARSERS:
        raise ProtocolError(f"unknown message type {mtype!r}")
    return mtype, v, d


def _check_size(data: bytes | bytearray | memoryview, max_bytes: int | None) -> None:
    if not isinstance(data, bytes | bytearray | memoryview):
        raise ProtocolError(f"message body must be bytes, got {type(data).__name__}")
    if max_bytes is not None and len(data) > max_bytes:
        raise PayloadTooLargeError(
            f"message of {len(data)} bytes exceeds the limit of {max_bytes}",
            details={"bytes": len(data), "max_bytes": max_bytes},
        )
    if len(data) == 0:
        raise ProtocolError("empty message body")


def decode_any(data: bytes | bytearray | memoryview, *, max_bytes: int | None = None) -> Any:
    """Decode a message of any known type (no error-to-exception conversion)."""
    _check_size(data, max_bytes)
    mtype, v, d = _parse_top(_unpack(data))
    return _PARSERS[mtype][1](d, v)


def decode_message(data: bytes | bytearray | memoryview, expected: type[T], *, max_bytes: int | None = None) -> T:
    """Parse and validate a message of type ``expected``.

    Raises:
        ProtocolError: malformed msgpack, wrong ``type``, missing keys, bad field types.
        VersionMismatchError: ``v`` not in :data:`SUPPORTED_VERSIONS`.
        PayloadTooLargeError: ``len(data) > max_bytes``.
        ErrorResponse payloads are raised as the typed :class:`pgw.core.PGWError` when ``expected``
        is not :class:`ErrorResponse`.
    """
    if expected not in MESSAGE_TYPES.values():
        raise TypeError(f"{expected!r} is not a PGW wire message type")
    _check_size(data, max_bytes)
    mtype, v, d = _parse_top(_unpack(data))
    cls, parser = _PARSERS[mtype]
    msg = parser(d, v)
    if cls is expected:
        return cast(T, msg)
    if isinstance(msg, ErrorResponse):
        raise msg.to_exception()
    raise ProtocolError(f"expected a {expected.__name__} message, got type {mtype!r}")


def decode_error_json(data: bytes | bytearray | memoryview, *, max_bytes: int = 1 << 20) -> ErrorResponse:
    """Parse a JSON :class:`ErrorResponse` (``{"v":1,"type":"error",...}``)."""
    _check_size(data, max_bytes)
    try:
        obj = json.loads(bytes(data))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ProtocolError("malformed JSON error body") from exc
    mtype, v, d = _parse_top(obj)
    if mtype != "error":
        raise ProtocolError(f"expected an error message, got type {mtype!r}")
    return _parse_error(d, v)
