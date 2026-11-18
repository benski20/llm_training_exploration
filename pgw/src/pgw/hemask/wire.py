"""Wire messages for ``POST /pgw/v1/maskgen``: the provider multiplies MLWE ciphertexts by ``W8``.

Request (msgpack)::

    {"v": 1, "type": "maskgen", "id": str, "fp": str,
     "items": [{"linear": str, "cols": int, "b": bin}]}

``b`` is the ``(n, cols)`` matrix of ciphertext coefficients as little-endian uint64 (row-major), with
``n`` the linear's in_features and ``cols`` a multiple of 256 (one or more 256-slot groups). The
response carries ``y = W8 b mod 2^64`` as ``(m, cols)`` uint64 per item, in request order::

    {"v": 1, "type": "maskgen_result", "id": str, "items": [{"linear": str, "cols": int, "y": bin}]}

The provider learns nothing from ``b`` (MLWE samples) beyond the linear ids and sizes, which are
public metadata like on the compute path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgpack

from pgw.core.errors import ProtocolError
from pgw.hemask.lwe import DEGREE

__all__ = [
    "MASKGEN_PATH",
    "MAX_GROUPS_PER_ITEM",
    "MaskgenItem",
    "MaskgenRequest",
    "MaskgenResponse",
    "MaskgenResult",
    "decode_maskgen_request",
    "decode_maskgen_response",
    "encode_maskgen_request",
    "encode_maskgen_response",
]

MASKGEN_PATH = "/pgw/v1/maskgen"
MAX_GROUPS_PER_ITEM = 16
_MAX_ITEMS = 64
_VERSION = 1


@dataclass(frozen=True)
class MaskgenItem:
    linear: str
    cols: int
    b: bytes


@dataclass(frozen=True)
class MaskgenRequest:
    request_id: str
    fingerprint: str
    items: tuple[MaskgenItem, ...]

    @property
    def payload_bytes(self) -> int:
        return sum(len(it.b) for it in self.items)


@dataclass(frozen=True)
class MaskgenResult:
    linear: str
    cols: int
    y: bytes


@dataclass(frozen=True)
class MaskgenResponse:
    request_id: str
    items: tuple[MaskgenResult, ...]


def _check_cols(cols: Any, where: str) -> int:
    if not isinstance(cols, int) or isinstance(cols, bool) or cols <= 0 or cols % DEGREE \
            or cols > DEGREE * MAX_GROUPS_PER_ITEM:
        raise ProtocolError(f"{where}: cols must be a positive multiple of {DEGREE} up to "
                            f"{DEGREE * MAX_GROUPS_PER_ITEM}, got {cols!r}")
    return cols


def _str(v: Any, name: str, max_len: int = 128) -> str:
    if not isinstance(v, str) or not v or len(v) > max_len:
        raise ProtocolError(f"maskgen: {name} must be a non-empty string")
    return v


def _load(data: bytes | bytearray | memoryview, kind: str) -> dict[str, Any]:
    try:
        obj = msgpack.unpackb(bytes(data), raw=False, strict_map_key=True, max_bin_len=1 << 31)
    except Exception as exc:
        raise ProtocolError(f"maskgen: malformed msgpack ({type(exc).__name__})") from exc
    if not isinstance(obj, dict) or obj.get("v") != _VERSION or obj.get("type") != kind:
        raise ProtocolError(f"maskgen: expected a v{_VERSION} {kind!r} message")
    items = obj.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= _MAX_ITEMS:
        raise ProtocolError(f"maskgen: items must be a list of 1..{_MAX_ITEMS} entries")
    return obj


def encode_maskgen_request(req: MaskgenRequest) -> bytes:
    return bytes(msgpack.packb({
        "v": _VERSION, "type": "maskgen", "id": req.request_id, "fp": req.fingerprint,
        "items": [{"linear": it.linear, "cols": it.cols, "b": it.b} for it in req.items],
    }, use_bin_type=True))


def decode_maskgen_request(data: bytes | bytearray | memoryview) -> MaskgenRequest:
    obj = _load(data, "maskgen")
    items = []
    for i, it in enumerate(obj["items"]):
        if not isinstance(it, dict) or not isinstance(it.get("b"), bytes):
            raise ProtocolError(f"maskgen: items[{i}] must be a map with binary 'b'")
        cols = _check_cols(it.get("cols"), f"items[{i}]")
        if len(it["b"]) == 0 or len(it["b"]) % (8 * cols):
            raise ProtocolError(f"maskgen: items[{i}].b length is not a multiple of 8*cols")
        items.append(MaskgenItem(_str(it.get("linear"), "linear", 64), cols, it["b"]))
    return MaskgenRequest(_str(obj.get("id"), "id"), _str(obj.get("fp"), "fp"), tuple(items))


def encode_maskgen_response(resp: MaskgenResponse) -> bytes:
    return bytes(msgpack.packb({
        "v": _VERSION, "type": "maskgen_result", "id": resp.request_id,
        "items": [{"linear": it.linear, "cols": it.cols, "y": it.y} for it in resp.items],
    }, use_bin_type=True))


def decode_maskgen_response(data: bytes | bytearray | memoryview, req: MaskgenRequest,
                            out_features: dict[str, int] | None = None) -> MaskgenResponse:
    """Decode and check the response against ``req`` (item order, sizes when ``out_features`` known)."""
    obj = _load(data, "maskgen_result")
    if obj.get("id") != req.request_id:
        raise ProtocolError("maskgen: response id does not match the request")
    raw = obj["items"]
    if len(raw) != len(req.items):
        raise ProtocolError("maskgen: response has a different number of items")
    items = []
    for i, (q, it) in enumerate(zip(req.items, raw, strict=True)):
        if not isinstance(it, dict) or it.get("linear") != q.linear or it.get("cols") != q.cols \
                or not isinstance(it.get("y"), bytes):
            raise ProtocolError(f"maskgen: response items[{i}] does not match the request")
        y = it["y"]
        if out_features is not None and q.linear in out_features:
            if len(y) != 8 * q.cols * out_features[q.linear]:
                raise ProtocolError(f"maskgen: response items[{i}] has {len(y)} bytes, expected "
                                    f"{8 * q.cols * out_features[q.linear]}")
        elif len(y) == 0 or len(y) % (8 * q.cols):
            raise ProtocolError(f"maskgen: response items[{i}] length is not a multiple of 8*cols")
        items.append(MaskgenResult(q.linear, q.cols, y))
    return MaskgenResponse(req.request_id, tuple(items))
