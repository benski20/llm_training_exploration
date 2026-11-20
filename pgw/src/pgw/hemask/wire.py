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
import numpy as np

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
    "pack_y",
    "unpack_y",
    "y_round_offset",
    "y_wire_bytes",
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
    #: Low bits of ``y`` the provider rounds away before replying (0 or 16). With 16, ``y`` is sent as
    #: ``round(y / 2^16)`` in 6 bytes per element: 25% less download, decryption error +2^15.
    drop_bits: int = 0
    #: Ciphertext modulus bits: ``b`` carries ``q/8`` bytes per coefficient and only ``y mod 2^q`` matters.
    q_bits: int = 64

    @property
    def y_bytes_per_element(self) -> int:
        return (self.q_bits - self.drop_bits) // 8

    @property
    def b_bytes_per_element(self) -> int:
        return self.q_bits // 8


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
        "items": [{"linear": it.linear, "cols": it.cols, "b": it.b, **({"drop": it.drop_bits} if it.drop_bits else {}),
                   **({"q": it.q_bits} if it.q_bits != 64 else {})} for it in req.items],
    }, use_bin_type=True))


def decode_maskgen_request(data: bytes | bytearray | memoryview) -> MaskgenRequest:
    obj = _load(data, "maskgen")
    items = []
    for i, it in enumerate(obj["items"]):
        if not isinstance(it, dict) or not isinstance(it.get("b"), bytes):
            raise ProtocolError(f"maskgen: items[{i}] must be a map with binary 'b'")
        cols = _check_cols(it.get("cols"), f"items[{i}]")
        q = it.get("q", 64)
        if q not in (56, 64) or isinstance(q, bool):
            raise ProtocolError(f"maskgen: items[{i}].q must be 56 or 64")
        if len(it["b"]) == 0 or len(it["b"]) % ((q // 8) * cols):
            raise ProtocolError(f"maskgen: items[{i}].b length is not a multiple of (q/8)*cols")
        drop = it.get("drop", 0)
        if drop not in (0, 16) or isinstance(drop, bool):
            raise ProtocolError(f"maskgen: items[{i}].drop must be 0 or 16")
        items.append(MaskgenItem(_str(it.get("linear"), "linear", 64), cols, it["b"], int(drop), int(q)))
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
        nb = q.y_bytes_per_element
        if out_features is not None and q.linear in out_features:
            if len(y) != nb * q.cols * out_features[q.linear]:
                raise ProtocolError(f"maskgen: response items[{i}] has {len(y)} bytes, expected "
                                    f"{nb * q.cols * out_features[q.linear]}")
        elif len(y) == 0 or len(y) % (nb * q.cols):
            raise ProtocolError(f"maskgen: response items[{i}] length is not a multiple of 8*cols")
        items.append(MaskgenResult(q.linear, q.cols, y))
    return MaskgenResponse(req.request_id, tuple(items))


def _to_bytes(v: np.ndarray, nb: int) -> bytes:
    """Low ``nb`` little-endian bytes of each element of ``v`` (``(rows, cols)``), row-major."""
    a = np.ascontiguousarray(v, dtype="<u8")
    if nb == 8:
        return a.tobytes()
    return a.view(np.uint8).reshape(*a.shape, 8)[..., :nb].tobytes()  # one strided copy


def _from_bytes(data: bytes | bytearray | memoryview, shape: tuple[int, int], nb: int) -> np.ndarray:
    if nb == 8:
        return np.frombuffer(data, dtype="<u8").reshape(shape).astype(np.uint64)  # writable copy
    buf = np.zeros((*shape, 8), dtype=np.uint8)
    buf[..., :nb] = np.frombuffer(data, dtype=np.uint8).reshape(*shape, nb)
    return buf.view("<u8").reshape(shape).astype(np.uint64, copy=False)  # no copy on little-endian hosts


def y_wire_bytes(q_bits: int, drop_bits: int) -> tuple[int, int]:
    """Byte range ``[lo, hi)`` of each little-endian 64-bit word that :func:`pack_y` sends.

    For byte-aligned ``q`` and ``drop`` (the only ones the wire allows), ``pack_y(y)`` is exactly bytes
    ``drop/8 .. q/8 - 1`` of ``(y + y_round_offset(drop)) mod 2^64``: rounding by ``2^drop`` and the
    reduction mod ``2^(q - drop)`` only keep bits ``drop .. q-1`` of ``y + 2^(drop-1)``, and those
    depend on ``y mod 2^q`` only. Device kernels use this to cut the reply out without a separate pass.
    """
    if q_bits % 8 or drop_bits % 8 or not 0 <= drop_bits < q_bits <= 64:
        raise ValueError(f"q_bits={q_bits}, drop_bits={drop_bits} are not byte-aligned wire values")
    return drop_bits // 8, q_bits // 8


def y_round_offset(drop_bits: int) -> int:
    """The rounding term :func:`pack_y` adds before dropping ``drop_bits`` low bits (0 for none)."""
    return 1 << (drop_bits - 1) if drop_bits else 0


def pack_b(b: np.ndarray, q_bits: int = 64) -> bytes:
    """Ciphertext coefficients (``< 2^q``) as ``q/8`` little-endian bytes each."""
    return _to_bytes(b, q_bits // 8)


def unpack_b(data: bytes | bytearray | memoryview, shape: tuple[int, int], q_bits: int = 64) -> np.ndarray:
    return _from_bytes(data, shape, q_bits // 8)


def pack_y(y: np.ndarray, drop_bits: int, q_bits: int = 64) -> bytes:
    """Provider side: ``y mod 2^q``, rounded by ``2^drop_bits``, as ``(q - drop)/8`` little-endian bytes."""
    if q_bits % 8 == 0 and drop_bits % 8 == 0 and 0 <= drop_bits < q_bits <= 64:
        # Byte-aligned (every wire value): the masking and the shift are a byte slice (y_wire_bytes).
        lo, hi = y_wire_bytes(q_bits, drop_bits)
        yv = np.array(y, dtype="<u8", order="C", copy=True)
        if drop_bits:
            yv += np.uint64(y_round_offset(drop_bits))  # wraps mod 2^64; only bits drop..q-1 are kept
        return yv.view(np.uint8).reshape(*yv.shape, 8)[..., lo:hi].tobytes()
    yv = np.asarray(y, dtype=np.uint64)
    if q_bits < 64:
        yv = yv & np.uint64((1 << q_bits) - 1)
    if drop_bits:
        yv = (yv + np.uint64(1 << (drop_bits - 1))) >> np.uint64(drop_bits)
    return _to_bytes(yv, (q_bits - drop_bits) // 8)  # a rounding carry past bit q is dropped (mod 2^q)


def unpack_y(data: bytes | bytearray | memoryview, shape: tuple[int, int], drop_bits: int,
             q_bits: int = 64) -> np.ndarray:
    """Client side: inverse of :func:`pack_y` (the dropped low bits come back as zeros)."""
    v = _from_bytes(data, shape, (q_bits - drop_bits) // 8)
    if drop_bits:
        v <<= np.uint64(drop_bits)  # v is a fresh array: shift in place
    return v
