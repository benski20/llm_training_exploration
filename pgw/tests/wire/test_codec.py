"""Codec round trips, strict validation, versioning and size limits (SPEC 6.3/6.4)."""

from __future__ import annotations

import json

import msgpack
import pytest

from pgw.core import (
    ErrorCode,
    FingerprintMismatchError,
    PayloadTooLargeError,
    PGWError,
    ProtocolError,
    RateLimitedError,
    VersionMismatchError,
    pack_ring,
    unpack_ring,
)
from pgw.wire import (
    CONTENT_TYPE,
    ComputeItem,
    ComputeRequest,
    ComputeResponse,
    ComputeResult,
    ErrorResponse,
    LinearInfo,
    ModelInfo,
    ModelsResponse,
    ServerTiming,
    decode_any,
    decode_error_json,
    decode_message,
    encode_error_json,
    encode_message,
)
from pgw.wire.codec import message_to_dict
from tests.wire.conftest import FP, make_request, random_x_hat


def _raw(d: dict) -> bytes:
    return msgpack.packb(d, use_bin_type=True)


def _req_dict(**over) -> dict:
    d = {
        "v": 1,
        "type": "compute",
        "id": "abc",
        "fp": FP,
        "ring": "K40A12",
        "items": [{"linear": "0.qkv", "rows": 2, "x_hat": b"\x01" * 160}],
    }
    d.update(over)
    return d


def test_content_type():
    assert CONTENT_TYPE == "application/vnd.pgw+msgpack"


def test_compute_request_round_trip(any_ring):
    req, xs = make_request(any_ring, [("0.qkv", 3), ("0.down", 1), ("0.qkv", 5)], deadline_ms=1500)
    data = encode_message(req)
    back = decode_message(data, ComputeRequest)
    assert back == req
    # tensors survive exactly
    for it, x in zip(back.items, xs, strict=True):
        assert (unpack_ring(it.x_hat, tuple(x.shape), any_ring) == x).all()


def test_compute_request_without_deadline_omits_key():
    req, _ = make_request(__import__("pgw.core", fromlist=["RING_K40_A12"]).RING_K40_A12, [("0.o", 1)])
    d = msgpack.unpackb(encode_message(req), raw=False)
    assert "deadline_ms" not in d
    assert d["v"] == 1 and d["type"] == "compute" and d["fp"] == FP and d["ring"] == "K40A12"
    assert isinstance(d["items"][0]["x_hat"], bytes)  # msgpack bin, not str


def test_wire_keys_match_spec():
    resp = ComputeResponse("r", (ComputeResult("0.o", 1, b"\x00" * 5),), ServerTiming(1.5, 2.5, 7))
    d = message_to_dict(resp)
    assert set(d) == {"v", "type", "id", "items", "timing"}
    assert d["timing"] == {"queue_ms": 1.5, "compute_ms": 2.5, "batch_rows": 7}
    models = ModelsResponse((ModelInfo("m", FP, ("K40A12",), (LinearInfo("0.o", 8, 16),), 10, 1000),), server="s")
    md = message_to_dict(models)
    assert md["models"][0]["linears"] == [{"id": "0.o", "m": 8, "n": 16}]
    assert md["versions"] == [1]


def test_compute_response_round_trip():
    resp = ComputeResponse(
        "req-9",
        (ComputeResult("1.down", 2, b"\xff" * 10), ComputeResult("1.o", 1, b"\x00")),
        ServerTiming(0.25, 3.0, 64),
    )
    assert decode_message(encode_message(resp), ComputeResponse) == resp


def test_models_round_trip_and_helpers():
    m = ModelInfo(
        "qwen", FP, ("K40A12", "K48A20"), (LinearInfo("0.qkv", 1152, 896), LinearInfo("0.o", 896, 896)), 8192, 256 << 20
    )
    resp = ModelsResponse((m,), server="pgw-service/1")
    back = decode_message(encode_message(resp), ModelsResponse)
    assert back == resp
    assert back.find(FP) == m and back.find("0" * 64) is None
    assert m.linear_dims() == {"0.qkv": (1152, 896), "0.o": (896, 896)}


def test_error_round_trip_and_typed_exception():
    err = ErrorResponse("rate_limited", "slow down", True, "rid", {"retry_after_s": 2})
    data = encode_message(err)
    assert decode_message(data, ErrorResponse) == err
    with pytest.raises(RateLimitedError) as ei:
        decode_message(data, ComputeResponse)
    assert ei.value.details["retry_after_s"] == 2 and ei.value.details["request_id"] == "rid"
    assert ei.value.retryable


def test_error_unknown_code_becomes_pgwerror():
    data = encode_message(ErrorResponse("brand_new_code", "?"))
    with pytest.raises(PGWError) as ei:
        decode_message(data, ComputeResponse)
    assert ei.value.details["unknown_code"] == "brand_new_code"


def test_error_from_exception_hides_internal():
    e = ErrorResponse.from_exception(ValueError("secret detail /path"), "x")
    assert e.code == "internal" and "secret" not in e.message
    f = ErrorResponse.from_exception(FingerprintMismatchError("nope", details={"a": 1}))
    assert f.code == "fingerprint_mismatch" and f.details == {"a": 1} and not f.retryable


def test_error_details_are_sanitized():
    err = ErrorResponse("internal", "m", details={"blob": b"\x00" * 99, "f": float("nan"), "t": (1, 2), 3: {"x": {1}}})
    back = decode_message(encode_message(err), ErrorResponse)
    assert back.details["blob"] == "<99 bytes>"
    assert back.details["f"] == "nan"
    assert back.details["t"] == [1, 2]
    assert back.details["3"] == {"x": [1]}


def test_error_json_round_trip():
    err = ErrorResponse("unavailable", "draining", True, None, {"k": "v"})
    body = encode_error_json(err)
    assert json.loads(body)["code"] == "unavailable"
    assert decode_error_json(body) == err
    with pytest.raises(ProtocolError):
        decode_error_json(b"{not json")
    with pytest.raises(ProtocolError):
        decode_error_json(json.dumps({"v": 1, "type": "models", "models": []}).encode())


def test_unknown_keys_are_ignored():
    d = _req_dict(future_field={"x": 1}, another=[1, 2])
    d["items"][0]["extra"] = "ignored"
    req = decode_message(_raw(d), ComputeRequest)
    assert req.request_id == "abc" and req.items[0].rows == 2


def test_decode_any_dispatches():
    assert isinstance(decode_any(_raw(_req_dict())), ComputeRequest)
    assert isinstance(decode_any(encode_message(ErrorResponse("protocol", "x"))), ErrorResponse)


def test_ring_name_is_canonicalized():
    assert decode_message(_raw(_req_dict(ring="k40a12")), ComputeRequest).ring == "K40A12"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.pop("id"),
        lambda d: d.pop("fp"),
        lambda d: d.pop("ring"),
        lambda d: d.pop("items"),
        lambda d: d.pop("type"),
        lambda d: d.pop("v"),
        lambda d: d.update(id=""),
        lambda d: d.update(id="x" * 200),
        lambda d: d.update(id="bad id with spaces"),
        lambda d: d.update(id=5),
        lambda d: d.update(fp="AB" * 32),
        lambda d: d.update(fp="ab" * 31),
        lambda d: d.update(ring="K41A12"),
        lambda d: d.update(ring="nonsense"),
        lambda d: d.update(ring=40),
        lambda d: d.update(items=[]),
        lambda d: d.update(items="x"),
        lambda d: d.update(items=[5]),
        lambda d: d.update(deadline_ms=-1),
        lambda d: d.update(deadline_ms="10"),
        lambda d: d.update(deadline_ms=1.5),
        lambda d: d.update(v="1"),
        lambda d: d.update(v=True),
        lambda d: d.update(type="nope"),
        lambda d: d["items"][0].update(rows=0),
        lambda d: d["items"][0].update(rows=True),
        lambda d: d["items"][0].update(rows=-3),
        lambda d: d["items"][0].update(x_hat="str not bin"),
        lambda d: d["items"][0].pop("x_hat"),
        lambda d: d["items"][0].update(linear="../etc"),
        lambda d: d["items"][0].update(linear=""),
    ],
)
def test_invalid_requests_raise_protocol_error(mutation):
    d = _req_dict()
    mutation(d)
    with pytest.raises(ProtocolError):
        decode_message(_raw(d), ComputeRequest)


def test_wrong_message_type():
    with pytest.raises(ProtocolError, match="expected a ComputeResponse"):
        decode_message(_raw(_req_dict()), ComputeResponse)


def test_not_a_map():
    for obj in ([1, 2], 5, "x", b"x", None):
        with pytest.raises(ProtocolError):
            decode_message(_raw(obj), ComputeRequest)  # type: ignore[arg-type]


@pytest.mark.parametrize("v", [0, 2, 99])
def test_version_mismatch(v):
    with pytest.raises(VersionMismatchError) as ei:
        decode_message(_raw(_req_dict(v=v)), ComputeRequest)
    assert ei.value.code == ErrorCode.VERSION_MISMATCH
    assert ei.value.details["versions"] == [1]
    assert ei.value.http_status == 426


def test_size_limit():
    data = _raw(_req_dict())
    decode_message(data, ComputeRequest, max_bytes=len(data))
    with pytest.raises(PayloadTooLargeError):
        decode_message(data, ComputeRequest, max_bytes=len(data) - 1)


def test_empty_garbage_trailing_and_ext():
    with pytest.raises(ProtocolError):
        decode_message(b"", ComputeRequest)
    with pytest.raises(ProtocolError):
        decode_message(b"\xc1", ComputeRequest)  # never-used msgpack byte
    with pytest.raises(ProtocolError):
        decode_message(_raw(_req_dict()) + b"\x00", ComputeRequest)
    ext = msgpack.packb({"v": 1, "type": "compute", "x": msgpack.ExtType(5, b"abc")}, use_bin_type=True)
    with pytest.raises(ProtocolError):
        decode_message(ext, ComputeRequest)
    with pytest.raises(ProtocolError):
        decode_message(msgpack.packb({1: 2}), ComputeRequest)  # non-str map key
    with pytest.raises(ProtocolError):
        decode_message("str", ComputeRequest)  # type: ignore[arg-type]


def test_deep_nesting_rejected():
    obj: object = 1
    for _ in range(200):
        obj = [obj]
    d = _req_dict(junk=obj)
    with pytest.raises(ProtocolError):
        decode_message(_raw(d), ComputeRequest)


def test_encode_rejects_non_messages():
    with pytest.raises(TypeError):
        encode_message({"v": 1})
    with pytest.raises(TypeError):
        decode_message(b"\x80", dict)  # type: ignore[arg-type]


def test_memoryview_and_bytearray_input(any_ring):
    req, _ = make_request(any_ring, [("0.o", 2)])
    data = encode_message(req)
    assert decode_message(memoryview(data), ComputeRequest) == req
    assert decode_message(bytearray(data), ComputeRequest) == req


def test_packing_contract_little_endian():
    from pgw.core import RING_K40_A12

    x = random_x_hat(2, 3, RING_K40_A12)
    b = pack_ring(x, RING_K40_A12)
    assert len(b) == 2 * 3 * 5
    v0 = int.from_bytes(b[:5], "little")
    assert v0 == int(x[0, 0])


def test_skeleton_constructor_compatible():
    req = ComputeRequest("id", "0" * 64, "K40A12", (ComputeItem("0.qkv", 1, b"\x00" * 5),))
    assert decode_message(encode_message(req), ComputeRequest) == req
    assert req.payload_bytes == 5 and req.total_rows == 1


@pytest.mark.parametrize("deadline", [None, 0, 250, 7 * 24 * 3600 * 1000])
def test_compute_messages_fast_encoding_is_byte_identical(deadline: int | None) -> None:
    """Compute requests/responses are framed by hand; the bytes equal the generic msgpack encoding."""
    blobs = [b"", b"\x01", bytes(range(256)), bytes(70000), bytearray(b"ab" * 40000), memoryview(bytes(300))]
    items = tuple(ComputeItem(f"{i}.qkv", rows, b) for i, (rows, b) in enumerate(zip([1, 127, 128, 65536, 2**32, 3],
                                                                                         blobs, strict=True)))
    req = ComputeRequest("r" * 40, FP, "K32A12C27", items, deadline_ms=deadline)
    resp = ComputeResponse("x" * 31, tuple(ComputeResult(it.linear, it.rows, it.x_hat) for it in items),
                           ServerTiming(0.0123, 1e300, 2**40))
    for msg in (req, resp, ComputeResponse("id", (), ServerTiming())):
        assert encode_message(msg) == msgpack.packb(message_to_dict(msg), use_bin_type=True)
    # fields the fast path does not handle as-is fall back to the generic encoder
    odd = ComputeResponse("id", (ComputeResult("0.o", 1, b"ab"),), ServerTiming(1, 2, 3))  # type: ignore[arg-type]
    assert encode_message(odd) == msgpack.packb(message_to_dict(odd), use_bin_type=True)
    assert decode_message(encode_message(req), ComputeRequest) == ComputeRequest(
        req.request_id, FP, "K32A12C27", tuple(ComputeItem(it.linear, it.rows, bytes(it.x_hat)) for it in items),
        deadline_ms=deadline)


def test_compute_messages_fast_decoding_matches_generic() -> None:
    """The fast parsers accept exactly what the generic ones accept, with the same result."""
    from pgw.core import RING_K32_A12_C27
    from pgw.wire import codec

    req, _ = make_request(RING_K32_A12_C27, [("0.o", 2), ("0.qkv", 1)], deadline_ms=900)
    body = message_to_dict(req)
    resp = ComputeResponse(req.request_id, (ComputeResult("0.o", 2, b"abcd"),), ServerTiming(1.5, 2.0, 7))
    cases: list[tuple[type, dict]] = [(ComputeRequest, body), (ComputeResponse, message_to_dict(resp))]
    variants = [
        lambda d: d.update(ring="k32a12c27"),  # canonicalised, like the generic parser does
        lambda d: d.update(deadline_ms=True),
        lambda d: d.update(id="bad id"),
        lambda d: d.update(fp=FP + "\n"),
        lambda d: d.update(timing={"queue_ms": 1, "compute_ms": float("inf"), "batch_rows": 0}),
        lambda d: d.update(timing={"queue_ms": 3, "compute_ms": 4, "batch_rows": 1}),
        lambda d: d.update(timing=None),
        lambda d: d["items"][0].update(rows=True),
        lambda d: d["items"][0].update(rows=0),
        lambda d: d["items"][0].update(linear="0/o"),
        lambda d: d.update(items=[]),
    ]
    for cls, base in cases:
        for change in [lambda d: None, *variants]:
            d = json.loads(json.dumps(base, default=lambda b: b.hex()))  # deep copy
            for it, orig in zip(d["items"], base["items"], strict=True):
                for k in ("x_hat", "y_hat"):
                    if k in it:
                        it[k] = orig[k]
            change(d)
            generic = (codec._parse_compute_request_generic if cls is ComputeRequest
                       else codec._parse_compute_response_generic)
            fast = codec._fast_compute_request if cls is ComputeRequest else codec._fast_compute_response
            try:
                want: object = generic(d, 1)
            except PGWError:
                want = None
            got = fast(d, 1)
            # the fast path accepts only what the generic parser accepts, with the same result; what
            # it leaves alone the generic parser decides (and raises for invalid input)
            assert got is None or got == want
            if change is variants[0] or change is variants[5] or change is variants[6]:
                assert got is not None  # well-formed messages take the fast path
