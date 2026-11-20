"""Trip lane wire format (chat-speed WP-A3): golden header vectors, the header-field allowlist, and the
``lane_open`` / ``lane_ticket`` messages.

``tests/wire/lane_golden.bin`` is the cross-implementation contract (Python here, ``pgw_fast`` lane.rs
``cargo test``, and the service listener ``rust/pgw-lane``): regenerate it only for a new lane version,
with ``PGW_REGEN_LANE_GOLDEN=1``.
"""

from __future__ import annotations

import inspect
import os
import struct
import uuid
from pathlib import Path

import pytest

from pgw.core import ProtocolError
from pgw.wire import LinearInfo, decode_message, encode_message
from pgw.wire import lane as L
from pgw.wire.messages import LaneOpenRequest, LaneTicket

GOLDEN = Path(__file__).with_name("lane_golden.bin")
FP = "ab" * 32


def _rid(seed: int) -> bytes:
    return bytes(((seed * 31 + i * 7) & 0xFF) for i in range(16))


# The valid records, field by field (the same values are hard-coded in lane.rs `expected()`).
HELLOS = [bytes(range(32)), b"\xa5" * 32]
REQUESTS = [  # ring, linear, rows, request_id, deadline_ms, body_len
    (0, 0, 1, _rid(1), 0, 14336),
    (1, 111, 8, _rid(2), 30000, 114688),
    (255, 65535, 65535, b"\xff" * 16, 0xFFFFFFFF, 0xFFFFFFFF),
]
REPLIES = [  # kind, status, linear, rows, compute_us, body_len, request_id
    (L.KIND_HELLO_ACK, L.Status.OK, 0, 8, 0, 0, bytes(16)),
    (L.KIND_HELLO_ACK, L.Status.BAD_TICKET, 0, 0, 0, 0, bytes(16)),
    (L.KIND_TRIP_REPLY, L.Status.OK, 0, 1, 287, 18432, _rid(1)),
    (L.KIND_TRIP_REPLY, L.Status.BUSY, 111, 8, 0, 0, _rid(2)),
]


def _invalid() -> list[tuple[int, bytes]]:
    h = L.pack_hello(HELLOS[0])
    q = L.pack_request(*REQUESTS[1])
    ack = L.pack_reply(*REPLIES[0])
    rep = L.pack_reply(*REPLIES[2])
    busy = L.pack_reply(*REPLIES[3])

    def put(b: bytes, at: int, v: bytes) -> bytes:
        return b[:at] + v + b[at + len(v):]

    return [
        (0x81, put(h, 0, b"PGWX")), (0x81, put(h, 4, struct.pack("<H", 2))), (0x81, put(h, 6, struct.pack("<H", 2))),
        (0x81, h[:-1]),
        (0x82, put(q, 0, b"PGWR")), (0x82, put(q, 4, struct.pack("<H", 0))), (0x82, put(q, 6, b"\x04")),
        (0x82, put(q, 10, struct.pack("<H", 0))), (0x82, q[:-1]), (0x82, q + b"\x00"),
        (0x83, put(rep, 0, b"PGWT")), (0x83, put(rep, 6, b"\x02")), (0x83, put(ack, 16, struct.pack("<I", 1))),
        (0x83, put(ack, 8, struct.pack("<H", 1))), (0x83, put(ack, 20, b"\x01")),
        (0x83, put(busy, 16, struct.pack("<I", 4))), (0x83, put(rep, 4, struct.pack("<H", 2))), (0x83, rep[:-1]),
    ]


def golden_bytes() -> bytes:
    recs = [(1, L.pack_hello(t)) for t in HELLOS]
    recs += [(2, L.pack_request(*r)) for r in REQUESTS]
    recs += [(3, L.pack_reply(*r)) for r in REPLIES]
    recs += _invalid()
    out = bytearray(b"PGWLGOLD" + struct.pack("<HH", L.LANE_VERSION, len(recs)))
    for tag, b in recs:
        out += bytes((tag, len(b))) + b
    return bytes(out)


def _records() -> list[tuple[int, bytes]]:
    data = GOLDEN.read_bytes()
    assert data[:8] == b"PGWLGOLD"
    version, n = struct.unpack_from("<HH", data, 8)
    assert version == L.LANE_VERSION
    at, recs = 12, []
    for _ in range(n):
        tag, ln = data[at], data[at + 1]
        recs.append((tag, data[at + 2:at + 2 + ln]))
        at += 2 + ln
    assert at == len(data)
    return recs


def test_golden_file_is_current() -> None:
    if os.environ.get("PGW_REGEN_LANE_GOLDEN") == "1":
        GOLDEN.write_bytes(golden_bytes())
    assert GOLDEN.read_bytes() == golden_bytes(), "lane header encoding changed: that is a new lane version"


def test_golden_records_decode() -> None:
    hellos, reqs, reps, bad = [], [], [], 0
    for tag, b in _records():
        if tag == 1:
            hellos.append(L.unpack_hello(b))
        elif tag == 2:
            d = L.unpack_request(b)
            reqs.append((d["ring"], d["linear"], d["rows"], d["request_id"], d["deadline_ms"], d["body_len"]))
        elif tag == 3:
            d = L.unpack_reply(b)
            reps.append((d["kind"], d["status"], d["linear"], d["rows"], d["compute_us"], d["body_len"],
                         d["request_id"]))
        else:
            fn = {0x81: L.unpack_hello, 0x82: L.unpack_request, 0x83: L.unpack_reply}[tag]
            with pytest.raises(L.LaneProtocolError):
                fn(b)
            bad += 1
    assert hellos == HELLOS and reqs == REQUESTS and reps == [tuple(r) for r in REPLIES]
    assert bad >= 8


def test_fixed_layout() -> None:
    assert (L.HELLO.size, L.REQUEST.size, L.REPLY.size) == (40, 36, 36)
    q = L.pack_request(3, 0x0102, 5, bytes(range(16)), 0x0A0B0C0D, 0x11223344)
    assert q == (b"PGWT" + b"\x01\x00" + b"\x02" + b"\x03" + b"\x02\x01" + b"\x05\x00" + b"\x0d\x0c\x0b\x0a"
                 + b"\x44\x33\x22\x11" + bytes(range(16)))


# ------------------------------------------------------------------------------- allowlist
PUBLIC_REQUEST_FIELDS = {
    "magic": "constant", "version": "constant", "kind": "constant",
    "ring": "index into the ticket's public ring table (the ring name is in every msgpack request)",
    "linear": "index into the public linear table (the linear id is in every msgpack request)",
    "rows": "row count (in every msgpack request)",
    "deadline_ms": "the request's deadline_ms, a gateway configuration constant (in every msgpack request)",
    "body_len": "len(x_hat) = rows * n * K/8, a function of the three above",
    "request_id": "uuid4 random, the msgpack request id",
}


def test_header_field_allowlist() -> None:
    """The request header holds exactly the public fields; nothing else can be added silently."""
    assert set(L.REQUEST_FIELDS) == set(PUBLIC_REQUEST_FIELDS)
    assert len(L.REQUEST_FIELDS) == len(L.REQUEST.unpack(bytes(L.REQUEST.size)))
    assert L.HELLO_FIELDS == ("magic", "version", "kind", "ticket")
    assert len(L.HELLO_FIELDS) == len(L.HELLO.unpack(bytes(L.HELLO.size)))
    params = list(inspect.signature(L.pack_request).parameters)
    assert params == ["ring", "linear", "rows", "request_id", "deadline_ms", "body_len"]
    # the format string has no padding or free bytes: every byte belongs to a named field
    assert "x" not in L.REQUEST.format and "x" not in L.HELLO.format


def test_header_carries_nothing_content_dependent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two calls with the same public shape but different x_hat give headers that differ only in the
    random request id, and the id is fresh uuid4 randomness (not a counter or a digest of the content)."""
    from pgw.wire.messages import ComputeItem, ComputeRequest

    sent: list[tuple[int, int, int, bytes, int, int]] = []

    class Rec:
        closed = False
        max_rows = 8

        def trip(self, ring, linear, rows, rid, deadline_ms, body, out, timeout_s):
            sent.append((ring, linear, rows, rid, deadline_ms, len(memoryview(body))))
            return L.Status.OK, 0

        def close(self) -> None:
            pass

    ticket = LaneTicket("t", 1, "uds", b"\x01" * 32, 60_000, 4, 8, FP, ("K32A12C27",),
                        (LinearInfo("0.qkv", 24, 16),), path="/nonexistent")
    lc = L.LaneComputeClient(_NoClient(), channel_factory=lambda t, s: Rec())
    assert lc._note_open(ticket, FP, "K32A12C27")
    for content in (b"\x00" * 64, b"\xff" * 64, os.urandom(64)):
        req = ComputeRequest(uuid.uuid4().hex, FP, "K32A12C27", (ComputeItem("0.qkv", 1, content),), deadline_ms=5000)
        lc.compute_sync(req)
    assert len(sent) == 3
    assert {s[:3] + s[4:] for s in sent} == {(0, 0, 1, 5000, 64)}
    assert len({s[3] for s in sent}) == 3
    assert all(uuid.UUID(bytes=s[3]).version == 4 for s in sent)


class _NoClient:
    """A wrapped client that must not be used (every call takes the lane)."""

    class config:  # noqa: N801
        base_url = "http://127.0.0.1:1"  # co-located: a UDS ticket is accepted
        request_timeout_s = 5.0

    def compute_sync(self, request):
        raise AssertionError("fallback not expected")


# ------------------------------------------------------------------------------- native == Python
def _native():
    try:
        import pgw_fast
    except ImportError:
        return None
    return pgw_fast if hasattr(pgw_fast, "lane_pack_request") else None


@pytest.mark.skipif(_native() is None, reason="pgw_fast without the lane module")
def test_native_header_codec_matches_python() -> None:
    nf = _native()
    import random

    rnd = random.Random(7)
    for _ in range(300):
        args = (rnd.randrange(256), rnd.randrange(1 << 16), rnd.randrange(1, 1 << 16), rnd.randbytes(16),
                rnd.randrange(1 << 32), rnd.randrange(1 << 32))
        b = L.pack_request(*args)
        assert nf.lane_pack_request(args[0], args[1], args[2], args[3], args[4], args[5]) == b
        assert nf.lane_unpack_request(b) == (args[0], args[1], args[2], args[4], args[5], args[3])
        rargs = (L.KIND_TRIP_REPLY, 0, rnd.randrange(1 << 16), rnd.randrange(1 << 16), rnd.randrange(1 << 32),
                 rnd.randrange(1 << 32), rnd.randbytes(16))
        rb = L.pack_reply(*rargs)
        assert nf.lane_pack_reply(*rargs) == rb
        assert nf.lane_unpack_reply(rb) == rargs
        t = rnd.randbytes(32)
        assert nf.lane_pack_hello(t) == L.pack_hello(t)
    for tag, b in _records():
        if tag == 0x82:
            with pytest.raises(ValueError):
                nf.lane_unpack_request(b)
        if tag == 0x83:
            with pytest.raises(ValueError):
                nf.lane_unpack_reply(b)


# ------------------------------------------------------------------------------- lane_open messages
def test_lane_messages_roundtrip() -> None:
    req = LaneOpenRequest("r1", FP, "K32A12C27")
    assert decode_message(encode_message(req), LaneOpenRequest) == req
    t = LaneTicket("r1", 1, "tcp", os.urandom(32), 60_000, 4, 8, FP, ("K32A12C27", "K40A12"),
                   (LinearInfo("0.qkv", 24, 16), LinearInfo("0.o", 16, 16)), port=9443)
    assert decode_message(encode_message(t), LaneTicket) == t
    u = LaneTicket("r1", 1, "uds", os.urandom(32), 60_000, 4, 8, FP, ("K32A12C27",),
                   (LinearInfo("0.qkv", 24, 16),), path="/run/pgw/lane.sock")
    assert decode_message(encode_message(u), LaneTicket) == u
    assert "redacted" in repr(t) and t.ticket.hex() not in repr(t)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(ticket=b"\x00" * 31),
    lambda d: d.update(transport="tcp-plain"),
    lambda d: d.pop("path"),
    lambda d: d.update(rings=[]),
    lambda d: d.update(linears=[{"id": "a", "m": 1, "n": 1}, {"id": "a", "m": 1, "n": 1}]),
    lambda d: d.update(lane_version=0),
    lambda d: d.update(max_rows=0),
])
def test_lane_ticket_rejects(mutate) -> None:
    import msgpack

    from pgw.wire.codec import message_to_dict

    u = LaneTicket("r1", 1, "uds", os.urandom(32), 60_000, 4, 8, FP, ("K32A12C27",),
                   (LinearInfo("0.qkv", 24, 16),), path="/run/pgw/lane.sock")
    d = message_to_dict(u)
    mutate(d)
    with pytest.raises(ProtocolError):
        decode_message(msgpack.packb(d, use_bin_type=True), LaneTicket)


def test_lane_open_rejects_unknown_transport() -> None:
    import msgpack

    from pgw.wire.codec import message_to_dict

    d = message_to_dict(LaneOpenRequest("r1", FP, "K32A12C27"))
    d["transports"] = ["tcp", "plain"]
    with pytest.raises(ProtocolError):
        decode_message(msgpack.packb(d, use_bin_type=True), LaneOpenRequest)


# ------------------------------------------------------------------------------- review hardening
@pytest.mark.parametrize("rid", ["a" * 32 + "\n", "a" * 32 + "\r\n", "A" * 32, "a" * 31, "a" * 33, " " + "a" * 32])
def test_request_id_must_be_exactly_32_lowercase_hex(rid: str) -> None:
    # `$` would accept a trailing newline, and the lane id would then differ from the msgpack id
    assert L.request_id_bytes(rid) is None
    assert L.request_id_bytes("0123456789abcdef" * 2) == bytes.fromhex("0123456789abcdef" * 2)


def test_lane_secret_not_in_repr() -> None:
    secret = bytearray(b"\xa5" * 32)
    lane = L._Lane(LaneTicket("r1", 1, "uds", bytes(32), 60_000, 4, 8, FP, ("K32A12C27",),
                              (LinearInfo("0.qkv", 24, 16),), path="/run/pgw/lane.sock"),
                   secret, 0.0, {}, {})
    text = repr(lane)
    assert "secret" not in text and "a5" not in text.lower() and "\\xa5" not in text


@pytest.mark.parametrize("base_url,uds_dir,path,ok", [
    ("http://127.0.0.1:8080", None, "/run/pgw/lane.sock", True),
    ("https://localhost:8443", None, "/run/pgw/lane.sock", True),
    ("https://[::1]:8443", None, "/run/pgw/lane.sock", True),
    ("https://compute.example.net:8443", None, "/run/pgw/lane.sock", False),  # remote service: no UDS
    ("https://10.0.0.5:8443", None, "/run/pgw/lane.sock", False),
    ("https://compute.example.net:8443", "/run/pgw", "/run/pgw/lane.sock", True),  # configured directory
    ("http://127.0.0.1:8080", "/run/pgw", "/run/pgw/sub/lane.sock", True),
    ("http://127.0.0.1:8080", "/run/pgw", "/run/pgw/../ssh/agent.sock", False),
    ("http://127.0.0.1:8080", "/run/pgw", "/run/pgwx/lane.sock", False),
    ("http://127.0.0.1:8080", "/run/pgw", "/run/pgw", False),
    ("http://127.0.0.1:8080", None, "\x00abstract", False),
])
def test_uds_ticket_only_from_a_local_service(base_url: str, uds_dir: str | None, path: str, ok: bool) -> None:
    from types import SimpleNamespace

    t = LaneTicket("r1", 1, "uds", os.urandom(32), 60_000, 4, 8, FP, ("K32A12C27",),
                   (LinearInfo("0.qkv", 24, 16),), path=path)
    lc = L.LaneComputeClient(SimpleNamespace(config=SimpleNamespace(base_url=base_url)),
                             L.LaneConfig(uds_dir=uds_dir))
    if ok:
        assert lc._accept(t, FP, "K32A12C27").ticket.path == path
    else:
        with pytest.raises(L.LaneProtocolError, match="UDS"):
            lc._accept(t, FP, "K32A12C27")
