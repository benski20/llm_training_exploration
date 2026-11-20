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
import struct
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
    LANE_TRANSPORTS,
    SUPPORTED_VERSIONS,
    ComputeItem,
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    ErrorResponse,
    LaneOpenRequest,
    LaneTicket,
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
    "encode_message_parts",
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
    if isinstance(msg, LaneOpenRequest):
        return {
            "v": int(msg.version),
            "type": "lane_open",
            "id": msg.request_id,
            "fp": msg.fingerprint,
            "ring": msg.ring,
            "transports": [str(t) for t in msg.transports],
            "lane_versions": [int(x) for x in msg.lane_versions],
        }
    if isinstance(msg, LaneTicket):
        d = {
            "v": int(msg.version),
            "type": "lane_ticket",
            "id": msg.request_id,
            "lane_version": int(msg.lane_version),
            "transport": str(msg.transport),
            "ticket": bytes(msg.ticket),
            "ttl_ms": int(msg.ttl_ms),
            "max_connections": int(msg.max_connections),
            "max_rows": int(msg.max_rows),
            "fp": msg.fingerprint,
            "rings": [str(r) for r in msg.rings],
            "linears": [_linear_to_dict(li) for li in msg.linears],
        }
        if msg.path is not None:
            d["path"] = str(msg.path)
        if msg.port is not None:
            d["port"] = int(msg.port)
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
    # Compute requests/responses (one per masked round trip) are framed by hand: the payload is
    # copied once into the output instead of into msgpack's growing pack buffer, and no dicts are
    # built. The bytes are identical to the generic path (tests compare both).
    tm = type(msg)
    if tm is ComputeRequest or tm is ComputeResponse:
        fast = _encode_compute(cast(ComputeRequest | ComputeResponse, msg))
        if fast is not None:
            return fast
    out = msgpack.packb(message_to_dict(msg), use_bin_type=True)
    return cast(bytes, out)


_U64 = 1 << 64
_FLOAT64 = struct.Struct(">d")


def _mp_uint(v: int) -> bytes:
    if v < 0x80:
        return bytes((v,))
    if v < 0x100:
        return b"\xcc" + bytes((v,))
    if v < 0x10000:
        return b"\xcd" + v.to_bytes(2, "big")
    if v < 0x100000000:
        return b"\xce" + v.to_bytes(4, "big")
    return b"\xcf" + v.to_bytes(8, "big")


def _mp_str(s: str) -> bytes:
    b = s.encode("utf-8")
    n = len(b)
    if n < 32:
        return bytes((0xA0 | n,)) + b
    if n < 0x100:
        return b"\xd9" + bytes((n,)) + b
    if n < 0x10000:
        return b"\xda" + n.to_bytes(2, "big") + b
    return b"\xdb" + n.to_bytes(4, "big") + b


def _mp_bin_header(n: int) -> bytes:
    if n < 0x100:
        return b"\xc4" + bytes((n,))
    if n < 0x10000:
        return b"\xc5" + n.to_bytes(2, "big")
    return b"\xc6" + n.to_bytes(4, "big")


def _mp_array_header(n: int) -> bytes:
    if n < 16:
        return bytes((0x90 | n,))
    if n < 0x10000:
        return b"\xdc" + n.to_bytes(2, "big")
    return b"\xdd" + n.to_bytes(4, "big")


def _uint_ok(*vals: object) -> bool:
    return all(type(v) is int and 0 <= v < _U64 for v in vals)


def _str_ok(*vals: object) -> bool:
    return all(type(v) is str for v in vals)


def _payload(b: object) -> bytes | bytearray | memoryview | None:
    """A bin payload usable as-is (``None``: let the generic encoder handle it)."""
    if type(b) is bytes or type(b) is bytearray:
        return b
    if isinstance(b, memoryview) and b.c_contiguous:
        return b.cast("B")
    return None


_K_V, _K_TYPE, _K_ID, _K_FP, _K_RING, _K_ITEMS, _K_DEADLINE = (
    _mp_str(k) for k in ("v", "type", "id", "fp", "ring", "items", "deadline_ms"))
_K_LINEAR, _K_ROWS, _K_XHAT, _K_YHAT, _K_TIMING = (_mp_str(k) for k in ("linear", "rows", "x_hat", "y_hat", "timing"))
_V_COMPUTE, _V_COMPUTE_RESULT = _mp_str("compute"), _mp_str("compute_result")
_K_QUEUE, _K_COMPUTE_MS, _K_BATCH_ROWS = (_mp_str(k) for k in ("queue_ms", "compute_ms", "batch_rows"))


def encode_message_parts(msg: object) -> list[bytes | bytearray | memoryview]:
    """:func:`encode_message` as a list of buffers whose concatenation is exactly its bytes.

    Compute requests/responses come back as their msgpack framing pieces with every payload
    (``x_hat`` / ``y_hat``) as its own, uncopied buffer, so a transport that writes a buffer list
    (``writelines``: one ``writev``) sends a multi-megabyte prefill reply without assembling it
    first. Any other message is one buffer.
    """
    tm = type(msg)
    if tm is ComputeRequest or tm is ComputeResponse:
        fast = _compute_parts(cast(ComputeRequest | ComputeResponse, msg))
        if fast is not None:
            return fast
    return [encode_message(msg)]


def _encode_compute(msg: ComputeRequest | ComputeResponse) -> bytes | None:
    """msgpack bytes of a compute request/response, identical to ``packb(message_to_dict(msg))``;
    ``None`` when a field is outside what this encoder handles (the generic path then runs)."""
    parts = _compute_parts(msg)
    return None if parts is None else b"".join(parts)


def _compute_parts(msg: ComputeRequest | ComputeResponse) -> list[bytes | bytearray | memoryview] | None:
    """The buffers of :func:`_encode_compute` (payloads not copied); ``None`` like it."""
    parts: list[bytes | bytearray | memoryview] = []
    if isinstance(msg, ComputeRequest):
        if (not _uint_ok(msg.version) or (msg.deadline_ms is not None and not _uint_ok(msg.deadline_ms))
                or not _str_ok(msg.request_id, msg.fingerprint, msg.ring)):
            return None
        parts += [b"\x87" if msg.deadline_ms is not None else b"\x86",
                  _K_V, _mp_uint(msg.version), _K_TYPE, _V_COMPUTE, _K_ID, _mp_str(msg.request_id),
                  _K_FP, _mp_str(msg.fingerprint), _K_RING, _mp_str(msg.ring),
                  _K_ITEMS, _mp_array_header(len(msg.items))]
        for it in msg.items:
            data = _payload(it.x_hat)
            if data is None or not _uint_ok(it.rows) or not _str_ok(it.linear):
                return None
            parts += [b"\x83", _K_LINEAR, _mp_str(it.linear), _K_ROWS, _mp_uint(it.rows),
                      _K_XHAT, _mp_bin_header(len(data)), data]
        if msg.deadline_ms is not None:
            parts += [_K_DEADLINE, _mp_uint(msg.deadline_ms)]
        return parts
    t = msg.timing
    if (not _uint_ok(msg.version, t.batch_rows) or type(t.queue_ms) is not float
            or type(t.compute_ms) is not float or not _str_ok(msg.request_id)):
        return None
    parts += [b"\x85", _K_V, _mp_uint(msg.version), _K_TYPE, _V_COMPUTE_RESULT, _K_ID, _mp_str(msg.request_id),
              _K_ITEMS, _mp_array_header(len(msg.items))]
    for r in msg.items:
        data = _payload(r.y_hat)
        if data is None or not _uint_ok(r.rows) or not _str_ok(r.linear):
            return None
        parts += [b"\x83", _K_LINEAR, _mp_str(r.linear), _K_ROWS, _mp_uint(r.rows),
                  _K_YHAT, _mp_bin_header(len(data)), data]
    parts += [_K_TIMING, b"\x83", _K_QUEUE, b"\xcb" + _FLOAT64.pack(t.queue_ms),
              _K_COMPUTE_MS, b"\xcb" + _FLOAT64.pack(t.compute_ms), _K_BATCH_ROWS, _mp_uint(t.batch_rows)]
    return parts


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


def _reject_json_constant(name: str) -> Any:
    """``json.loads`` hook: ``NaN`` / ``Infinity`` are not valid JSON and are rejected."""
    raise ValueError(f"non-standard JSON constant {name}")


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
# Fast paths for the two per-round-trip messages: the same checks as the generic parsers below,
# without building context strings for error messages that are not raised. Anything they do not
# accept as-is (including every invalid message) goes to the generic parser, which raises the
# precise error.
_RING_CANON: dict[str, str] = {}
_MAX_DEADLINE_MS = 7 * 24 * 3600 * 1000


def _ring_canon(v: object) -> str | None:
    if type(v) is not str:
        return None
    hit = _RING_CANON.get(v)
    if hit is None and len(v) <= _MAX_RING_LEN:
        try:
            hit = RingParams.from_name(v).name
        except ConfigError:
            return None
        if len(_RING_CANON) < 64:
            _RING_CANON[v] = hit
    return hit


def _ok_str(v: object, max_len: int, pattern: re.Pattern[str]) -> bool:
    return type(v) is str and len(v) <= max_len and pattern.match(v) is not None


def _fast_compute_request(d: dict[str, Any], v: int) -> ComputeRequest | None:
    items_raw = d.get("items")
    rid, fp = d.get("id"), d.get("fp")
    ring = _ring_canon(d.get("ring"))
    if (type(items_raw) is not list or not items_raw or ring is None or not _ok_str(rid, _MAX_ID_LEN, _ID_RE)
            or not _ok_str(fp, 64, _FP_RE)):
        return None
    deadline = d.get("deadline_ms")
    if deadline is not None and (type(deadline) is not int or not 0 <= deadline <= _MAX_DEADLINE_MS):
        return None
    items = []
    for raw in items_raw:
        if type(raw) is not dict:
            return None
        linear, rows, x_hat = raw.get("linear"), raw.get("rows"), raw.get("x_hat")
        if (type(rows) is not int or rows < 1 or type(x_hat) is not bytes
                or not _ok_str(linear, _MAX_LINEAR_LEN, _LINEAR_RE)):
            return None
        items.append(ComputeItem(linear, rows, x_hat))  # type: ignore[arg-type]
    return ComputeRequest(request_id=rid, fingerprint=fp, ring=ring, items=tuple(items),  # type: ignore[arg-type]
                          deadline_ms=deadline, version=v)


def _fast_ms(x: object) -> float | None:
    if type(x) is float:
        return x if math.isfinite(x) else None
    return float(x) if type(x) is int else None


def _fast_compute_response(d: dict[str, Any], v: int) -> ComputeResponse | None:
    items_raw, rid, t = d.get("items"), d.get("id"), d.get("timing")
    if type(items_raw) is not list or not _ok_str(rid, _MAX_ID_LEN, _ID_RE):
        return None
    if t is None:
        timing = ServerTiming()
    else:
        if type(t) is not dict:
            return None
        q, c, b = _fast_ms(t.get("queue_ms", 0.0)), _fast_ms(t.get("compute_ms", 0.0)), t.get("batch_rows", 0)
        if q is None or c is None or type(b) is not int or b < 0:
            return None
        timing = ServerTiming(queue_ms=q, compute_ms=c, batch_rows=b)
    items = []
    for raw in items_raw:
        if type(raw) is not dict:
            return None
        linear, rows, y_hat = raw.get("linear"), raw.get("rows"), raw.get("y_hat")
        if (type(rows) is not int or rows < 1 or type(y_hat) is not bytes
                or not _ok_str(linear, _MAX_LINEAR_LEN, _LINEAR_RE)):
            return None
        items.append(ComputeResult(linear, rows, y_hat))  # type: ignore[arg-type]
    return ComputeResponse(request_id=rid, items=tuple(items), timing=timing, version=v)  # type: ignore[arg-type]


def _parse_compute_request(d: dict[str, Any], v: int) -> ComputeRequest:
    fast = _fast_compute_request(d, v)
    return fast if fast is not None else _parse_compute_request_generic(d, v)


def _parse_compute_request_generic(d: dict[str, Any], v: int) -> ComputeRequest:
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
    fast = _fast_compute_response(d, v)
    return fast if fast is not None else _parse_compute_response_generic(d, v)


def _parse_compute_response_generic(d: dict[str, Any], v: int) -> ComputeResponse:
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
        linears: list[LinearInfo] = []
        seen_ids: set[str] = set()
        for j, lraw in enumerate(_list(_req(m, "linears", ctx), f"{ctx}.linears")):
            lctx = f"{ctx}.linears[{j}]"
            li = _map(lraw, lctx)
            info = LinearInfo(
                id=_linear(_req(li, "id", lctx), f"{lctx}.id"),
                out_features=_int(_req(li, "m", lctx), f"{lctx}.m", lo=1),
                in_features=_int(_req(li, "n", lctx), f"{lctx}.n", lo=1),
            )
            if info.id in seen_ids:
                raise ProtocolError(f"{lctx}.id duplicates linear {info.id!r}")
            seen_ids.add(info.id)
            linears.append(info)
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
    fps = [m.fingerprint for m in models]
    if len(set(fps)) != len(fps):
        raise ProtocolError("models lists the same fingerprint more than once")
    versions = tuple(_int(x, "versions[]", lo=0) for x in _list(d.get("versions", [v]), "versions"))
    return ModelsResponse(
        models=tuple(models),
        versions=versions,
        server=_str(d.get("server", ""), "server", max_len=256),
        version=v,
    )


#: Lane ticket bounds (SPEC 6 lane section; :mod:`pgw.wire.lane`).
LANE_TICKET_BYTES = 32
_MAX_LANE_PATH = 4096
_MAX_LANE_ROWS = 65535
_MAX_LANE_LINEARS = 65535
_MAX_LANE_RINGS = 255


def _parse_lane_open(d: dict[str, Any], v: int) -> LaneOpenRequest:
    transports = tuple(_str(t, "transports[]", max_len=16) for t in _list(_req(d, "transports", ""), "transports"))
    if not transports or len(transports) > 8 or any(t not in LANE_TRANSPORTS for t in transports):
        raise ProtocolError(f"transports must be a non-empty subset of {list(LANE_TRANSPORTS)}")
    versions = tuple(_int(x, "lane_versions[]", lo=1, hi=0xFFFF)
                     for x in _list(_req(d, "lane_versions", ""), "lane_versions"))
    if not versions or len(versions) > 16:
        raise ProtocolError("lane_versions must list 1 to 16 versions")
    return LaneOpenRequest(
        request_id=validate_request_id(_req(d, "id", "")),
        fingerprint=_fp(_req(d, "fp", ""), "fp"),
        ring=_ring(_req(d, "ring", ""), "ring"),
        transports=transports,
        lane_versions=versions,
        version=v,
    )


def _parse_lane_ticket(d: dict[str, Any], v: int) -> LaneTicket:
    transport = _str(_req(d, "transport", ""), "transport", max_len=16)
    if transport not in LANE_TRANSPORTS:
        raise ProtocolError(f"transport must be one of {list(LANE_TRANSPORTS)}")
    ticket = _bin(_req(d, "ticket", ""), "ticket")
    if len(ticket) != LANE_TICKET_BYTES:
        raise ProtocolError(f"ticket must be {LANE_TICKET_BYTES} bytes")
    path = d.get("path")
    port = d.get("port")
    if transport == "uds":
        path = _str(_req(d, "path", ""), "path", max_len=_MAX_LANE_PATH)
        if not path or "\x00" in path:
            raise ProtocolError("path must be a non-empty socket path")
        port = None
    else:
        port = _int(_req(d, "port", ""), "port", lo=1, hi=65535)
        path = None
    rings = tuple(_ring(r, "rings[]") for r in _list(_req(d, "rings", ""), "rings"))
    if not rings or len(rings) > _MAX_LANE_RINGS:
        raise ProtocolError(f"rings must list 1 to {_MAX_LANE_RINGS} rings")
    linears: list[LinearInfo] = []
    seen: set[str] = set()
    raw_linears = _list(_req(d, "linears", ""), "linears")
    if not raw_linears or len(raw_linears) > _MAX_LANE_LINEARS:
        raise ProtocolError(f"linears must list 1 to {_MAX_LANE_LINEARS} linears")
    for j, lraw in enumerate(raw_linears):
        ctx = f"linears[{j}]"
        li = _map(lraw, ctx)
        info = LinearInfo(
            id=_linear(_req(li, "id", ctx), f"{ctx}.id"),
            out_features=_int(_req(li, "m", ctx), f"{ctx}.m", lo=1),
            in_features=_int(_req(li, "n", ctx), f"{ctx}.n", lo=1),
        )
        if info.id in seen:
            raise ProtocolError(f"{ctx}.id duplicates linear {info.id!r}")
        seen.add(info.id)
        linears.append(info)
    return LaneTicket(
        request_id=validate_request_id(_req(d, "id", "")),
        lane_version=_int(_req(d, "lane_version", ""), "lane_version", lo=1, hi=0xFFFF),
        transport=transport,
        ticket=ticket,
        ttl_ms=_int(_req(d, "ttl_ms", ""), "ttl_ms", lo=1, hi=_MAX_DEADLINE_MS),
        max_connections=_int(_req(d, "max_connections", ""), "max_connections", lo=1, hi=1024),
        max_rows=_int(_req(d, "max_rows", ""), "max_rows", lo=1, hi=_MAX_LANE_ROWS),
        fingerprint=_fp(_req(d, "fp", ""), "fp"),
        rings=rings,
        linears=tuple(linears),
        path=path,
        port=port,
        version=v,
    )


_PARSERS: dict[str, tuple[type[Any], Callable[[dict[str, Any], int], Any]]] = {
    "compute": (ComputeRequest, _parse_compute_request),
    "compute_result": (ComputeResponse, _parse_compute_response),
    "error": (ErrorResponse, _parse_error),
    "models": (ModelsResponse, _parse_models),
    "lane_open": (LaneOpenRequest, _parse_lane_open),
    "lane_ticket": (LaneTicket, _parse_lane_ticket),
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
        obj = json.loads(bytes(data), parse_constant=_reject_json_constant)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        # RecursionError: deeply nested arrays/objects (hostile or broken peer).
        raise ProtocolError(f"malformed JSON error body: {type(exc).__name__}") from exc
    mtype, v, d = _parse_top(obj)
    if mtype != "error":
        raise ProtocolError(f"expected an error message, got type {mtype!r}")
    return _parse_error(d, v)
