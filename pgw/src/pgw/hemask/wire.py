"""Wire messages for ``POST /pgw/v1/maskgen``: the provider multiplies MLWE ciphertexts by ``W8``.

Two message versions share the path. **v1** is always supported and is the fallback::

    {"v": 1, "type": "maskgen", "id": str, "fp": str, "vs": [int]?,
     "items": [{"linear": str, "cols": int, "b": bin, "q": int?, "drop": int?}]}

``b`` is the ``(n, cols)`` matrix of ciphertext coefficients as ``q/8`` little-endian bytes each
(row-major, ``q`` 56 or 64, default 64), with ``n`` the linear's in_features and ``cols`` a multiple
of 256 (one or more 256-slot groups). The response carries ``y = W8 b mod 2^q`` rounded by
``2^drop`` (``drop`` 0 or 16) as ``(q - drop)/8`` bytes per element of the ``(m, cols)`` result,
items in request order::

    {"v": 1, "type": "maskgen_result", "id": str, "vs": [int]?,
     "items": [{"linear": str, "cols": int, "y": bin}]}

**v2** is bit-packed and sized to what the customer needs. It is used only when both sides
advertise it::

    {"v": 2, "type": "maskgen", "id": str, "fp": str, "vs": [int]?,
     "items": [{"linear": str, "cols": int, "bb": int, "yb": int, "b": bin}]}

``b`` holds ``(n, cols)`` coefficients in ``[0, 2^bb)`` bit-packed at ``bb`` bits
(:func:`pack_bits`). The provider computes ``Y = W8 b mod 2^bb`` and returns
``floor((Y + 2^(d-1)) / 2^d) mod 2^yb`` with ``d = bb - yb`` (no offset when ``d = 0``),
bit-packed at ``yb`` bits (:func:`pack_y_v2`). That is ``n·cols·bb/8`` bytes up and
``m·cols·yb/8`` bytes down, ``8 <= yb <= bb <= 64`` and ``bb >= 32``. The v2 reply has the same shape as v1:
``{"v": 2, "type": "maskgen_result", ...}``. When ``bb`` is below the ciphertext modulus, the
customer has switched its own ciphertext down (:mod:`pgw.hemask.client`). The provider needs no
other parameter.

**v3** (scheme ``pks-v3``, :mod:`pgw.hemask.pks`; served only with ``PGW_SERVICE_MASKGEN_V3=true``, used only
by a customer that opted into the scheme) carries ring-LWE uploads and one reply key per 256-slot group, and
returns packed key-switched replies::

    {"v": 3, "type": "maskgen", "id": str, "fp": str, "vs": [int]?, "aseed": bin(32),
     "items": [{"linear": str, "cols": int, "b": bin, "ksk": bin}]}

``b`` is ``(n, cols)`` bit-packed at 56 bits (the upload modulus), ``ksk`` the ``cols/256`` reply keys, each
the NTT-domain residues ``(4 primes, 16, 8192)`` bit-packed per prime at 24, 24, 30 and 30 bits (1.77 MB per
group). ``aseed`` is the customer's public seed of the upload ring elements (the provider derives the
per-linear hint ``W8 A`` from it once and caches it). The reply ``{"v": 3, "type": "maskgen_result", ...}``
holds, per item, the ``(cols/256, m/32, 8192)`` rounded ``c0`` values bit-packed at 36 bits:
``m·cols·36/8`` bytes, the v2 size at ``yb = 36``.

``vs`` lists the maskgen versions the sender speaks. A client advertises it on its requests and a
server on its replies. Both are optional keys that v1-only peers ignore. A client sends v2 only
after a reply has advertised 2, and it falls back to v1 when a v2 request is refused. A server
rejects a version it does not serve with ``version_mismatch``.

The provider learns nothing from ``b`` (MLWE samples, or a customer-side randomised rounding of
them) beyond the linear ids and sizes, which are public metadata like on the compute path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeGuard

import msgpack
import numpy as np

from pgw.core.errors import ProtocolError, VersionMismatchError
from pgw.hemask.lwe import DEGREE

__all__ = [
    "MASKGEN_PATH",
    "MASKGEN_VERSIONS",
    "MAX_GROUPS_PER_ITEM",
    "V3_KSK_BYTES",
    "MaskgenItem",
    "MaskgenRequest",
    "MaskgenResponse",
    "MaskgenResult",
    "bit_fields",
    "decode_maskgen_request",
    "decode_maskgen_response",
    "encode_maskgen_request",
    "encode_maskgen_response",
    "encode_maskgen_response_parts",
    "le_words",
    "pack_bits",
    "pack_y",
    "pack_y_v2",
    "unpack_bits",
    "unpack_y",
    "y_round_offset",
    "y_wire_bytes",
]

MASKGEN_PATH = "/pgw/v1/maskgen"
MAX_GROUPS_PER_ITEM = 16
_MAX_ITEMS = 64
_VERSION = 1
#: Maskgen message versions this code speaks (a server may serve fewer, see ``vs``).
MASKGEN_VERSIONS: tuple[int, ...] = (1, 2, 3)
#: v3 (pks-v3) widths: 56-bit uploads, 36-bit replies, reply-key residues per prime.
V3_B_BITS = 56
V3_Y_BITS = 36
V3_KSK_BITS = (24, 24, 30, 30)
#: Bytes of one group's reply key on the wire (16 x 8192 coefficients per prime).
V3_KSK_BYTES = 16 * 8192 * sum(V3_KSK_BITS) // 8
#: Smallest v2 width (``8 <= yb <= bb <= 64``).
V2_MIN_BITS = 8
#: Smallest v2 upload width the provider accepts. The provider unpacks ``b`` into ``uint64``, so a
#: ``bb``-bit body costs ``64/bb`` times its size in host memory; ``bb >= 32`` caps that at 2x
#: (customers send ``q`` or ``q - 3``, i.e. 53..64).
V2_MIN_B_BITS = 32


@dataclass(frozen=True)
class MaskgenItem:
    linear: str
    cols: int
    b: bytes | bytearray  # v2 payloads are built in place (pack_bits)
    #: v1: low bits of ``y`` the provider rounds away before replying (0 or 16). With 16, ``y`` is sent as
    #: ``round(y / 2^16)`` in 6 bytes per element: 25% less download, decryption error +2^15.
    drop_bits: int = 0
    #: v1: ciphertext modulus bits: ``b`` carries ``q/8`` bytes per coefficient and only ``y mod 2^q`` matters.
    q_bits: int = 64
    #: v2 only (0 in v1): bits per uploaded coefficient (``b`` bit-packed; ``Y = W8 b mod 2^b_bits``).
    b_bits: int = 0
    #: v2 only (0 in v1): bits per returned element (``round(Y / 2^(b_bits - y_bits))``, bit-packed).
    y_bits: int = 0
    #: v3 only: the bit-packed reply keys of the item's groups (``b_bits`` 56, ``y_bits`` 36).
    ksk: bytes | bytearray | None = None

    @property
    def v3(self) -> bool:
        return self.ksk is not None

    @property
    def v2(self) -> bool:
        return self.b_bits > 0 and self.ksk is None

    @property
    def y_bytes_per_element(self) -> int:
        """v1: whole bytes per returned element (v2 uses :meth:`y_nbytes`)."""
        return (self.q_bits - self.drop_bits) // 8

    @property
    def b_bytes_per_element(self) -> int:
        """v1: whole bytes per uploaded coefficient (v2 uses :meth:`b_nbytes`)."""
        return self.q_bits // 8

    def b_nbytes(self, n: int) -> int:
        """Expected ``len(b)`` for a linear with ``n`` inputs (either version)."""
        return n * self.cols * self.b_bits // 8 if self.b_bits else self.b_bytes_per_element * n * self.cols

    def y_nbytes(self, m: int) -> int:
        """Expected ``len(y)`` of the reply for a linear with ``m`` outputs (either version)."""
        return m * self.cols * self.y_bits // 8 if self.b_bits else self.y_bytes_per_element * m * self.cols


@dataclass(frozen=True)
class MaskgenRequest:
    request_id: str
    fingerprint: str
    items: tuple[MaskgenItem, ...]
    #: Message version (1 or 2). Every item of a v2 request has ``b_bits`` and ``y_bits`` set.
    version: int = 1
    #: Versions the sender speaks (``vs``; empty: not advertised).
    versions: tuple[int, ...] = ()
    #: v3 only: the customer's public upload seed (32 bytes).
    a_seed: bytes = b""

    @property
    def payload_bytes(self) -> int:
        return sum(len(it.b) + len(it.ksk or b"") for it in self.items)


@dataclass(frozen=True)
class MaskgenResult:
    linear: str
    cols: int
    y: bytes | bytearray


@dataclass(frozen=True)
class MaskgenResponse:
    request_id: str
    items: tuple[MaskgenResult, ...]
    #: Message version (the request's).
    version: int = 1
    #: Versions the server speaks (``vs``; empty: not advertised, i.e. a v1-only server).
    versions: tuple[int, ...] = ()


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


def _int(v: Any) -> TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool)


def _versions(v: Any) -> tuple[int, ...]:
    """The optional ``vs`` list (unknown entries are kept; malformed lists read as not advertised)."""
    if not isinstance(v, list) or len(v) > 16 or not all(_int(x) for x in v):
        return ()
    return tuple(int(x) for x in v)


def _load(data: bytes | bytearray | memoryview, kind: str, versions: tuple[int, ...] = (_VERSION,)
          ) -> dict[str, Any]:
    try:
        # msgpack reads any buffer: no copy of a bytearray/memoryview body (``bin`` values are copied out)
        obj = msgpack.unpackb(data, raw=False, strict_map_key=True, max_bin_len=1 << 31)
    except Exception as exc:
        raise ProtocolError(f"maskgen: malformed msgpack ({type(exc).__name__})") from exc
    if not isinstance(obj, dict) or obj.get("type") != kind:
        raise ProtocolError(f"maskgen: expected a {kind!r} message")
    v = obj.get("v")
    if v not in versions or not _int(v):
        if _int(v) and v > 0:
            raise VersionMismatchError(f"maskgen: message version {v} is not supported",
                                       details={"versions": list(versions)})
        raise ProtocolError(f"maskgen: expected a v{'/v'.join(map(str, versions))} {kind!r} message")
    items = obj.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= _MAX_ITEMS:
        raise ProtocolError(f"maskgen: items must be a list of 1..{_MAX_ITEMS} entries")
    return obj


def _vs(versions: tuple[int, ...]) -> dict[str, Any]:
    return {"vs": list(versions)} if versions else {}


def encode_maskgen_request(req: MaskgenRequest) -> bytes:
    extra: dict[str, Any] = {}
    if req.version == 3:
        if not all(it.v3 for it in req.items) or len(req.a_seed) != 32:
            raise ValueError("every item of a v3 maskgen request needs ksk, and the request a 32-byte a_seed")
        items = [{"linear": it.linear, "cols": it.cols, "b": it.b, "ksk": it.ksk} for it in req.items]
        extra = {"aseed": bytes(req.a_seed)}
    elif req.version == 2:
        if not all(it.v2 for it in req.items):
            raise ValueError("every item of a v2 maskgen request needs b_bits and y_bits")
        items = [{"linear": it.linear, "cols": it.cols, "bb": it.b_bits, "yb": it.y_bits, "b": it.b}
                 for it in req.items]
    elif req.version == 1:
        if any(it.v2 for it in req.items):
            raise ValueError("v1 maskgen items cannot carry b_bits/y_bits")
        items = [{"linear": it.linear, "cols": it.cols, "b": it.b, **({"drop": it.drop_bits} if it.drop_bits else {}),
                  **({"q": it.q_bits} if it.q_bits != 64 else {})} for it in req.items]
    else:
        raise ValueError(f"unknown maskgen version {req.version}")
    return bytes(msgpack.packb({
        "v": req.version, "type": "maskgen", "id": req.request_id, "fp": req.fingerprint, **_vs(req.versions),
        **extra, "items": items,
    }, use_bin_type=True))


def _decode_item_v1(i: int, it: dict[str, Any]) -> MaskgenItem:
    cols = _check_cols(it.get("cols"), f"items[{i}]")
    q = it.get("q", 64)
    if q not in (56, 64) or isinstance(q, bool):
        raise ProtocolError(f"maskgen: items[{i}].q must be 56 or 64")
    if len(it["b"]) == 0 or len(it["b"]) % ((q // 8) * cols):
        raise ProtocolError(f"maskgen: items[{i}].b length is not a multiple of (q/8)*cols")
    drop = it.get("drop", 0)
    if drop not in (0, 16) or isinstance(drop, bool):
        raise ProtocolError(f"maskgen: items[{i}].drop must be 0 or 16")
    return MaskgenItem(_str(it.get("linear"), "linear", 64), cols, it["b"], int(drop), int(q))


def _decode_item_v2(i: int, it: dict[str, Any]) -> MaskgenItem:
    cols = _check_cols(it.get("cols"), f"items[{i}]")
    bb, yb = it.get("bb"), it.get("yb")
    if not (_int(bb) and _int(yb) and V2_MIN_BITS <= yb <= bb <= 64 and bb >= V2_MIN_B_BITS):
        raise ProtocolError(f"maskgen: items[{i}] needs integer bb and yb with {V2_MIN_BITS} <= yb <= bb <= 64 "
                            f"and bb >= {V2_MIN_B_BITS}")
    # cols is a multiple of 256, so n*cols*bb/8 is a whole number of bytes for every n.
    if len(it["b"]) == 0 or len(it["b"]) % (cols * bb // 8):
        raise ProtocolError(f"maskgen: items[{i}].b length is not a multiple of cols*bb/8")
    return MaskgenItem(_str(it.get("linear"), "linear", 64), cols, it["b"], 0, 64, int(bb), int(yb))


def _decode_item_v3(i: int, it: dict[str, Any]) -> MaskgenItem:
    cols = _check_cols(it.get("cols"), f"items[{i}]")
    ksk = it.get("ksk")
    if not isinstance(ksk, bytes) or len(ksk) != (cols // DEGREE) * V3_KSK_BYTES:
        raise ProtocolError(f"maskgen: items[{i}].ksk must hold {cols // DEGREE} reply keys of {V3_KSK_BYTES} bytes")
    if len(it["b"]) == 0 or len(it["b"]) % (cols * V3_B_BITS // 8):
        raise ProtocolError(f"maskgen: items[{i}].b length is not a multiple of cols*56/8")
    return MaskgenItem(_str(it.get("linear"), "linear", 64), cols, it["b"], 0, 64, V3_B_BITS, V3_Y_BITS, ksk)


def decode_maskgen_request(data: bytes | bytearray | memoryview,
                           versions: tuple[int, ...] = (1, 2)) -> MaskgenRequest:
    """Decode a maskgen request of any version in ``versions`` (others: ``version_mismatch``). v3 is only
    accepted when the caller lists it (the service serves it behind ``PGW_SERVICE_MASKGEN_V3``)."""
    obj = _load(data, "maskgen", versions)
    v = int(obj["v"])
    items = []
    a_seed = b""
    if v == 3:
        a_seed = obj.get("aseed")
        if not isinstance(a_seed, bytes) or len(a_seed) != 32:
            raise ProtocolError("maskgen v3: aseed must be 32 bytes")
    for i, it in enumerate(obj["items"]):
        if not isinstance(it, dict) or not isinstance(it.get("b"), bytes):
            raise ProtocolError(f"maskgen: items[{i}] must be a map with binary 'b'")
        items.append(_decode_item_v3(i, it) if v == 3 else _decode_item_v2(i, it) if v == 2
                     else _decode_item_v1(i, it))
    return MaskgenRequest(_str(obj.get("id"), "id"), _str(obj.get("fp"), "fp"), tuple(items), v,
                          _versions(obj.get("vs")), a_seed)


def encode_maskgen_response(resp: MaskgenResponse) -> bytes:
    return bytes(msgpack.packb({
        "v": resp.version, "type": "maskgen_result", "id": resp.request_id, **_vs(resp.versions),
        "items": [{"linear": it.linear, "cols": it.cols, "y": it.y} for it in resp.items],
    }, use_bin_type=True))


def _bin_header(n: int) -> bytes:
    """msgpack ``bin`` header for ``n`` bytes (the smallest form, as ``msgpack.packb`` writes it)."""
    if n < 1 << 8:
        return b"\xc4" + n.to_bytes(1, "big")
    if n < 1 << 16:
        return b"\xc5" + n.to_bytes(2, "big")
    if n < 1 << 32:
        return b"\xc6" + n.to_bytes(4, "big")
    raise ValueError("bin payload too large for msgpack")


def encode_maskgen_response_parts(resp: MaskgenResponse) -> list[bytes | bytearray | memoryview]:
    """:func:`encode_maskgen_response` as buffers whose concatenation is byte-identical, without copying ``y``.

    The ``y`` payloads (e.g. read-only views of pinned device-copy buffers) are returned as they are, so a
    sender can write them straight to the socket.
    """
    pk = msgpack.Packer(use_bin_type=True)
    head: dict[str, Any] = {"v": resp.version, "type": "maskgen_result", "id": resp.request_id, **_vs(resp.versions)}
    pre = bytearray(pk.pack_map_header(len(head) + 1))
    for k, v in head.items():
        pre += pk.pack(k)
        pre += pk.pack(v)
    pre += pk.pack("items")
    pre += pk.pack_array_header(len(resp.items))
    parts: list[bytes | bytearray | memoryview] = []
    for it in resp.items:
        pre += pk.pack_map_header(3)
        pre += pk.pack("linear")
        pre += pk.pack(it.linear)
        pre += pk.pack("cols")
        pre += pk.pack(it.cols)
        pre += pk.pack("y")
        y = memoryview(it.y).cast("B") if not isinstance(it.y, (bytes, bytearray)) else it.y
        pre += _bin_header(len(y))
        parts.append(bytes(pre))
        parts.append(y)
        pre = bytearray()
    if pre:
        parts.append(bytes(pre))
    return parts


def decode_maskgen_response(data: bytes | bytearray | memoryview, req: MaskgenRequest,
                            out_features: dict[str, int] | None = None) -> MaskgenResponse:
    """Decode and check the response against ``req`` (version, item order, sizes when ``out_features`` known)."""
    obj = _load(data, "maskgen_result", (req.version,))
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
            want = q.y_nbytes(out_features[q.linear])
            if len(y) != want:
                raise ProtocolError(f"maskgen: response items[{i}] has {len(y)} bytes, expected {want}")
        elif len(y) == 0 or len(y) % q.y_nbytes(1):
            raise ProtocolError(f"maskgen: response items[{i}] length is not a multiple of one output row")
        items.append(MaskgenResult(q.linear, q.cols, y))
    return MaskgenResponse(req.request_id, tuple(items), req.version, _versions(obj.get("vs")))


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


# ---- v2 bit packing ------------------------------------------------------------------------------
_FIELD_DTYPES: dict[int, np.dtype[Any]] = {
    64: np.dtype("<u8"), 32: np.dtype("<u4"), 16: np.dtype("<u2"), 8: np.dtype(np.uint8)}


def bit_fields(bits: int) -> tuple[list[tuple[int, int]], int]:
    """The ``(offset, width)`` fields and the tail width of the :func:`pack_bits` layout for ``bits``.

    Starting at bit 0, each field is the widest of 64, 32, 16 and 8 bits that still fits. The tail
    is the last ``bits mod 8`` bits. Examples: 35 = [32] + tail 3, 53 = [32, 16] + tail 5,
    56 = [32, 16, 8], 43 = [32, 8] + tail 3, 64 = [64].
    """
    if not 1 <= bits <= 64:
        raise ValueError(f"bit width must be in [1, 64], got {bits}")
    fields = []
    off = 0
    while bits - off >= 8:
        width = next(c for c in (64, 32, 16, 8) if c <= bits - off)
        fields.append((off, width))
        off += width
    return fields, bits - off


def _rep(mask: int, lane: int) -> np.uint64:
    return np.uint64(sum(mask << s for s in range(0, 64, lane)))


def _compress_tails(x: np.ndarray, t: int) -> np.ndarray:
    """Words whose byte ``k`` holds a ``t``-bit value -> the values at bits ``t·k`` (``8t`` bits per word)."""
    lo = (1 << t) - 1
    x = (x & _rep(lo, 16)) | ((x >> np.uint64(8 - t)) & _rep(lo << t, 16))
    lo2 = (1 << 2 * t) - 1
    x = (x & _rep(lo2, 32)) | ((x >> np.uint64(16 - 2 * t)) & _rep(lo2 << 2 * t, 32))
    lo4 = (1 << 4 * t) - 1
    return (x & np.uint64(lo4)) | ((x >> np.uint64(32 - 4 * t)) & np.uint64(lo4 << 4 * t))


def _spread_tails(x: np.ndarray, t: int) -> np.ndarray:
    """Inverse of :func:`_compress_tails`."""
    lo4 = (1 << 4 * t) - 1
    x = (x & np.uint64(lo4)) | ((x << np.uint64(32 - 4 * t)) & np.uint64(lo4 << 32))
    lo2 = (1 << 2 * t) - 1
    x = (x & _rep(lo2, 32)) | ((x << np.uint64(16 - 2 * t)) & _rep(lo2 << 16, 32))
    lo = (1 << t) - 1
    return (x & _rep(lo, 16)) | ((x << np.uint64(8 - t)) & _rep(lo << 8, 16))


def pack_bits(v: np.ndarray, bits: int) -> bytearray:
    """The low ``bits`` bits of every value of ``v`` (uint64, row-major), bit-packed: ``len(v)·bits/8`` bytes.

    Layout (:func:`bit_fields`): for each field ``(offset, width)`` in order, bits
    ``offset .. offset+width-1`` of every value as one contiguous little-endian ``uint{width}``
    array. Then the tail of ``t = bits mod 8`` bits per value as an LSB-first bit stream: group
    ``g`` of eight values is the ``t``-byte little-endian integer ``sum_k tail[8g + k] << (t·k)``.
    Each field is a cast, so the format is fast to produce and read with vector code. The value
    count must be a multiple of 8. Every maskgen operand qualifies, because ``cols`` is a multiple
    of 256. The fields are written straight into the returned buffer, which msgpack sends as ``bin``.
    """
    a = np.ascontiguousarray(v, dtype=np.uint64).reshape(-1)
    count = a.size
    if count % 8:
        raise ValueError(f"bit packing needs a multiple of 8 values, got {count}")
    fields, t = bit_fields(bits)
    buf = bytearray(count * bits // 8)
    dst = np.frombuffer(buf, dtype=np.uint8)
    tmp: np.ndarray | None = None  # one scratch array for the shifted values (no per-field allocation)

    def shifted(off: int) -> np.ndarray:
        nonlocal tmp
        if off == 0:
            return a
        tmp = np.empty_like(a) if tmp is None else tmp
        np.right_shift(a, np.uint64(off), out=tmp)
        return tmp

    pos = 0
    for off, width in fields:
        nb = count * width // 8
        np.copyto(dst[pos:pos + nb].view(_FIELD_DTYPES[width]), shifted(off), casting="unsafe")  # cast truncates
        pos += nb
    if t:
        tails = np.empty(count, dtype=np.uint8)
        np.copyto(tails, shifted(bits - t), casting="unsafe")
        tails &= np.uint8((1 << t) - 1)
        words = _compress_tails(tails.view("<u8"), t)
        dst[pos:].reshape(count // 8, t)[...] = words.view(np.uint8).reshape(count // 8, 8)[:, :t]
    return buf


def unpack_bits(data: bytes | bytearray | memoryview, count: int, bits: int) -> np.ndarray:
    """Inverse of :func:`pack_bits`: ``(count,)`` uint64 values below ``2^bits`` (a fresh, writable array)."""
    if count % 8:
        raise ValueError(f"bit packing needs a multiple of 8 values, got {count}")
    src = np.frombuffer(data, dtype=np.uint8)
    if src.size != count * bits // 8:
        raise ValueError(f"packed data has {src.size} bytes, expected {count * bits // 8}")
    fields, t = bit_fields(bits)
    out = np.empty(count, dtype=np.uint64)
    tmp: np.ndarray | None = None
    pos = 0
    for off, width in fields:  # the first field has offset 0
        nb = count * width // 8
        part = src[pos:pos + nb].view(_FIELD_DTYPES[width])
        pos += nb
        if off == 0:
            np.copyto(out, part)
        else:
            tmp = np.empty_like(out) if tmp is None else tmp
            np.copyto(tmp, part)
            tmp <<= np.uint64(off)
            out |= tmp
    if t:
        tb = np.zeros((count // 8, 8), dtype=np.uint8)
        tb[:, :t] = src[pos:].reshape(count // 8, t)
        tails = _spread_tails(tb.view("<u8").reshape(count // 8), t).view(np.uint8)
        if not fields:
            np.copyto(out, tails)
        else:
            tmp = np.empty_like(out) if tmp is None else tmp
            np.copyto(tmp, tails)
            tmp <<= np.uint64(bits - t)
            out |= tmp
    return out


def le_words(data: bytes | bytearray | memoryview, count: int, nb: int, *, copy: bool = True) -> np.ndarray:
    """``count`` little-endian ``nb``-byte unsigned integers (``1 <= nb <= 8``) as uint64.

    A fresh array, except with ``nb == 8`` and ``copy=False`` (a view of ``data``, read-only for ``bytes``).
    """
    if not 1 <= nb <= 8:
        raise ValueError(f"word size must be in [1, 8] bytes, got {nb}")
    if len(data) != count * nb:
        raise ValueError(f"{len(data)} bytes is not {count} words of {nb} bytes")
    if nb == 8:
        return np.frombuffer(data, dtype="<u8").astype(np.uint64, copy=copy)
    # Overlapping unaligned 8-byte reads at stride nb, then a mask: one pass, no per-byte copy.
    buf = bytes(data) + bytes(8 - nb)
    words = np.ndarray(shape=(count,), dtype="<u8", buffer=buf, offset=0, strides=(nb,))
    return words & np.uint64((1 << (8 * nb)) - 1)


def pack_y_v2(y: np.ndarray, b_bits: int, y_bits: int, *, inplace: bool = False) -> bytearray:
    """Provider side of v2: ``floor((Y + 2^(d-1)) / 2^d) mod 2^y_bits`` bit-packed, with ``Y = y mod 2^b_bits``.

    ``d = b_bits - y_bits``. ``y`` is ``W8 b`` modulo ``2^b_bits`` or modulo any multiple of it, for
    example the ``mod 2^56`` result of a 7-byte GEMM. Only bits ``d .. b_bits - 1`` of ``Y + 2^(d-1)``
    are sent, so a carry past bit ``b_bits`` is dropped. ``inplace=True`` lets the function overwrite
    ``y`` (a uint64 array).
    """
    if not V2_MIN_BITS <= y_bits <= b_bits <= 64:
        raise ValueError(f"b_bits={b_bits}, y_bits={y_bits} out of range")
    d = b_bits - y_bits
    own = inplace and y.dtype == np.uint64 and y.flags.c_contiguous and y.flags.writeable
    if not d:
        return pack_bits(y, y_bits)  # keeps bits 0 .. y_bits-1 only
    y = np.asarray(y, dtype=np.uint64).reshape(-1)
    v = np.add(y, np.uint64(1 << (d - 1)), out=y if own else None)  # wraps mod 2^64; one pass, also for read-only y
    v >>= np.uint64(d)  # bits from b_bits up are not packed
    return pack_bits(v, y_bits)
