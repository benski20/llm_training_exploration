"""Lane format 2: the ``multi`` frame (several answers' rows of one batched trip in one lane call).

Covers: a service ticket offers format 2 (``versions``, ``max_items``, ``max_multi_rows``) next to ``version: 1``;
a native multi call over TCP+TLS and over UDS answers exactly the bytes of the stream path's request with the same
items (several linears, an item longer than one call serves, extreme operands) and the exact integer product;
``PGW_SERVICE_LANE_MULTI=0`` gives format-1 tickets, and a multi frame sent anyway is refused like a format-1
service refuses it (closing); the client channel's argument checks; :class:`~pgw.wire.lane.LaneComputeClient`
sends a several-item request in one multi call only when it is on and offered (format-1 ticket, Python channel:
the regular path with the same request), with the response equal to the stream path's; ticket parsing of old and
new services.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from pgw.core import RingParams, pack_ring
from pgw.service.lane import LANE_OPEN_PATH, encode_multi, lane_native_available, parse_reply
from pgw.wire.lane import CLOSING_STATUSES, LaneComputeClient, LaneConfig, Status
from pgw.wire.messages import PROTOCOL_HEADER, PROTOCOL_VERSION
from pgw.wire.service_lane import ServiceLaneOpener

from .conftest import asgi_client, expected, http_client, make_req, rand_x, running_app
from .test_lane_tls import _settings as _tls_settings
from .test_lane_tls import pki  # noqa: F401 - module fixture

fast = pytest.importorskip("pgw_fast")
pytestmark = pytest.mark.skipif(
    not lane_native_available() or not hasattr(fast.TripChannel, "trip_multi")
    or not hasattr(fast.LaneService, "max_items"),
    reason="pgw_fast without lane format 2 (multi frames)")

K32 = RingParams.from_name("K32A8")


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    d = Path(tempfile.mkdtemp(prefix="pgwlm", dir="/tmp"))  # short: AF_UNIX path limit
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _uds_settings(tiny_pack_dir: Path, sock_dir: Path, **kw: Any) -> Any:
    from pgw.service import ServiceSettings

    base: dict[str, Any] = {"packs": [tiny_pack_dir], "device": "cpu", "verify_packs": "fingerprint",
                            "warmup": False, "rings": ["K32A8", "K40A12"], "lane": "uds",
                            "lane_socket": sock_dir / "lane.sock", "allow_unauthenticated": True}
    base.update(kw)
    return ServiceSettings(**base)


async def _open(app: Any, fp: str) -> dict[str, Any]:
    async with http_client(app) as c:
        r = await c.post(LANE_OPEN_PATH, content=json.dumps({"fingerprint": fp}),
                         headers={PROTOCOL_HEADER: str(PROTOCOL_VERSION), "content-type": "application/json"})
        assert r.status_code == 200, r.text
        return r.json()  # type: ignore[no-any-return]


def _items(info: dict[str, Any], dims: dict[str, tuple[int, int]]) -> list[tuple[str, torch.Tensor]]:
    """Items of one batched trip: two answers on linear 0, one on linear 1, a long item (a prompt's rows) and
    another answer on linear 0 again, with extreme operands."""
    names = info["linears"]
    out = []
    for k, (li, rows) in enumerate([(0, 1), (0, 1), (1, 2), (0, 11), (0, 1)]):
        _m, n = dims[names[li]]
        x = rand_x(rows, n, K32, seed=1000 + k)
        x[0, 0] = K32.modulus - 1
        out.append((names[li], x))
    return out


def _multi_args(info: dict[str, Any], items: list[tuple[str, torch.Tensor]],
                dims: dict[str, tuple[int, int]]) -> tuple[list[int], list[int], list[bytes], int]:
    ix = {name: i for i, name in enumerate(info["linears"])}
    lins = [ix[name] for name, _ in items]
    rows = [int(x.shape[0]) for _, x in items]
    bodies = [pack_ring(x, K32) for _, x in items]
    out_len = sum(r * dims[name][0] * 4 for (name, _), r in zip(items, rows, strict=True))
    return lins, rows, bodies, out_len


async def test_ticket_offers_format_2_and_tls_multi_call_equals_the_stream_path(
        tiny_pack_dir: Path, pki: dict[str, Any], pack_fp: str,  # noqa: F811
        pack_weights: dict[str, torch.Tensor]) -> None:
    async with running_app(_tls_settings(tiny_pack_dir, pki, lane_max_rows=4)) as app:
        info = await _open(app, pack_fp)
        # format 1 stays readable by every older client; format 2 is offered next to it
        assert info["version"] == 1 and info["versions"] == [1, 2]
        assert info["max_items"] == 64 and info["max_multi_rows"] == 256 and info["max_rows"] == 4
        dims = dict(app.state.pgw.registry.get(pack_fp).linear_dims)
        ch = await asyncio.to_thread(
            fast.TripChannel.connect_service_tls, "127.0.0.1", int(info["port"]), "127.0.0.1", pki["ca_pem"],
            bytes.fromhex(info["ticket"]), 5.0, int(info["max_rows"]), 50, None, None)
        client = asgi_client(app)
        try:
            items = _items(info, dims)
            lins, rows, bodies, out_len = _multi_args(info, items, dims)
            out = bytearray(out_len)
            st, _cu = await asyncio.to_thread(ch.trip_multi, 0, lins, rows, secrets.token_bytes(16), 0, bodies,
                                              out, 5.0)
            assert st == 0
            resp = await client.compute(make_req(pack_fp, K32, items))
            assert bytes(out) == b"".join(r.y_hat for r in resp.items)  # == the stream path's bytes, in order
            want = b"".join(pack_ring(expected(x, pack_weights[name], K32), K32) for name, x in items)
            assert bytes(out) == want  # the exact integer product, bit for bit
            # the connection serves single calls and further multi calls after it
            m, n = dims[info["linears"][0]]
            x = rand_x(1, n, K32, seed=5)
            one = bytearray(m * 4)
            st, _ = await asyncio.to_thread(ch.trip, 0, 0, 1, secrets.token_bytes(16), 0, pack_ring(x, K32), one, 5.0)
            assert st == 0 and bytes(one) == pack_ring(expected(x, pack_weights[info["linears"][0]], K32), K32)
            out2 = bytearray(out_len)
            st, _ = await asyncio.to_thread(ch.trip_multi, 0, lins, rows, secrets.token_bytes(16), 0, bodies,
                                            out2, 5.0)
            assert st == 0 and out2 == out and not ch.closed
        finally:
            await client.aclose()
            ch.close()


async def test_multi_off_service_issues_format_1_tickets_and_refuses_multi_frames(
        tiny_pack_dir: Path, sock_dir: Path, pack_fp: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_SERVICE_LANE_MULTI", "0")
    async with running_app(_uds_settings(tiny_pack_dir, sock_dir)) as app:
        info = await _open(app, pack_fp)
        assert info["version"] == 1 and "versions" not in info and "max_items" not in info
        dims = dict(app.state.pgw.registry.get(pack_fp).linear_dims)
        ch = await asyncio.to_thread(fast.TripChannel.connect_service_uds, info["socket"],
                                     bytes.fromhex(info["ticket"]), 5.0, int(info["max_rows"]))
        items = _items(info, dims)[:2]
        lins, rows, bodies, out_len = _multi_args(info, items, dims)
        st, _ = await asyncio.to_thread(ch.trip_multi, 0, lins, rows, secrets.token_bytes(16), 0, bodies,
                                        bytearray(out_len), 5.0)
        assert st in CLOSING_STATUSES and ch.closed  # Version: as a format-1 service answers
        # the raw frame on a fresh connection: the reply is the service's `version` status (21)
        import socket

        info = await _open(app, pack_fp)

        def raw() -> int:
            from pgw.service.lane import encode_open

            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect(info["socket"])
                s.sendall(encode_open(bytes.fromhex(info["ticket"])))
                assert parse_reply(s.recv(40)).status == 0
                s.sendall(encode_multi(secrets.token_bytes(16), [(li, r, b) for li, r, b in
                                                                 zip(lins, rows, bodies, strict=True)]))
                return int(parse_reply(s.recv(40)).status)

        assert await asyncio.to_thread(raw) == 21


async def test_uds_multi_raw_frame_matches_the_reference_encoder(tiny_pack_dir: Path, sock_dir: Path,
                                                                 pack_fp: str,
                                                                 pack_weights: dict[str, torch.Tensor]) -> None:
    """The Python reference encoder (``encode_multi``) and the native client speak the same frame."""
    import socket

    async with running_app(_uds_settings(tiny_pack_dir, sock_dir)) as app:
        info = await _open(app, pack_fp)
        dims = dict(app.state.pgw.registry.get(pack_fp).linear_dims)
        items = _items(info, dims)
        lins, rows, bodies, out_len = _multi_args(info, items, dims)

        def raw() -> bytes:
            from pgw.service.lane import encode_open

            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect(info["socket"])
                s.sendall(encode_open(bytes.fromhex(info["ticket"])))
                assert parse_reply(s.recv(40)).status == 0
                s.sendall(encode_multi(secrets.token_bytes(16), list(zip(lins, rows, bodies, strict=True))))
                hdr = b""
                while len(hdr) < 40:
                    hdr += s.recv(40 - len(hdr))
                r = parse_reply(hdr)
                assert r.status == 0 and r.body_len == out_len
                body = b""
                while len(body) < out_len:
                    body += s.recv(out_len - len(body))
                return body

        got = await asyncio.to_thread(raw)
        assert got == b"".join(pack_ring(expected(x, pack_weights[name], K32), K32) for name, x in items)


def test_native_multi_call_argument_checks(tmp_path: Path) -> None:
    d = Path(tempfile.mkdtemp(prefix="pgwlm", dir="/tmp"))
    calls: list[tuple[int, int]] = []

    def ex(_model: int, linear: int, _ring: int, rows: int, body: bytes, _tenant: str, _left: float) -> bytes:
        calls.append((linear, rows))
        return bytes(rows * 3 * 4)

    svc = fast.LaneService(str(d / "l.sock"), max_rows=4, executor=ex, max_items=8)
    try:
        assert svc.max_items == 8
        svc.set_model(0, [(2, 3, "qkv")], [("K32A8", 4)])
        t = secrets.token_bytes(32)
        svc.add_ticket(t, "t", 0, 60.0)
        ch = fast.TripChannel.connect_service_uds(str(d / "l.sock"), t, 5.0, 4)
        rid = secrets.token_bytes(16)
        with pytest.raises(ValueError):  # list lengths differ
            ch.trip_multi(0, [0, 0], [1], rid, 0, [b"\0" * 8, b"\0" * 8], bytearray(24), 5.0)
        with pytest.raises(ValueError):  # no items
            ch.trip_multi(0, [], [], rid, 0, [], bytearray(0), 5.0)
        buf = bytearray(32)
        with pytest.raises(ValueError):  # out overlaps a body
            ch.trip_multi(0, [0], [1], rid, 0, [memoryview(buf)[:8]], memoryview(buf)[4:16], 5.0)
        assert not ch.closed and calls == []  # nothing was sent
        st, _ = ch.trip_multi(0, [0, 0], [1, 1], rid, 0, [b"\x01" * 8, b"\x02" * 8], bytearray(24), 5.0)
        assert st == 0 and calls == [(0, 2)]  # one executor call for the same-linear run
        with pytest.raises(ValueError):  # out of the wrong size for the reply: the channel closes
            ch.trip_multi(0, [0, 0], [1, 1], secrets.token_bytes(16), 0, [b"\x01" * 8, b"\x02" * 8],
                          bytearray(23), 5.0)
        assert ch.closed
    finally:
        svc.stop(2.0)
        shutil.rmtree(d, ignore_errors=True)


async def _lane_client(app: Any, sock_dir: Path, fp: str, *, multi: bool, native: bool | None = None
                       ) -> tuple[LaneComputeClient, Any]:
    client = asgi_client(app)
    await client.connect(fp)
    cfg = LaneConfig(native=native, uds_dir=str(sock_dir), transports=("uds",), multi=multi)
    opener = ServiceLaneOpener(client, cfg)
    lc = LaneComputeClient(opener, cfg, channel_factory=opener.channel)
    assert await lc.open(fp, K32.name)
    return lc, client


@pytest.mark.parametrize("native", [None, False])
async def test_lane_client_sends_several_items_in_one_multi_call_only_when_on_and_offered(
        tiny_pack_dir: Path, sock_dir: Path, pack_fp: str, native: bool | None,
        monkeypatch: pytest.MonkeyPatch) -> None:
    async with running_app(_uds_settings(tiny_pack_dir, sock_dir)) as app:
        dims = dict(app.state.pgw.registry.get(pack_fp).linear_dims)
        info = await _open(app, pack_fp)
        items = _items(info, dims)
        req = make_req(pack_fp, K32, items, rid=secrets.token_hex(16))
        lc, client = await _lane_client(app, sock_dir, pack_fp, multi=True, native=native)
        try:
            assert lc.lane_ticket is not None and lc.lane_ticket.max_items == 64
            want = await client.compute(req)
            resp = await asyncio.to_thread(lc._try_lane, req, None)
            if native is False:  # the Python channel has no multi calls: the regular path, same request
                assert resp is None and lc.lane_stats.fallback_reasons == {"ineligible:channel": 1}
                return
            assert resp is not None and resp.request_id == req.request_id
            assert [(r.linear, r.rows) for r in resp.items] == [(it.linear, it.rows) for it in req.items]
            assert [bytes(r.y_hat) for r in resp.items] == [bytes(r.y_hat) for r in want.items]
            assert (lc.lane_stats.multi_trips, lc.lane_stats.multi_items, lc.lane_stats.fallbacks) == (1, 5, 0)
            # one item of more rows than a single call serves (a prompt's rows) also rides a multi call
            one = make_req(pack_fp, K32, items[3:4], rid=secrets.token_hex(16))
            r1 = await asyncio.to_thread(lc._try_lane, one, None)
            assert r1 is not None and bytes(r1.items[0].y_hat) == bytes(want.items[3].y_hat)
            assert lc.lane_stats.multi_trips == 2
        finally:
            lc.lane_close()
            await client.aclose()
        lc, client = await _lane_client(app, sock_dir, pack_fp, multi=False)  # off (the gateway's default)
        try:
            assert await asyncio.to_thread(lc._try_lane, req, None) is None
            assert lc.lane_stats.fallback_reasons == {"ineligible:items": 1} and lc.lane_stats.multi_trips == 0
        finally:
            lc.lane_close()
            await client.aclose()


async def test_lane_client_against_a_format_1_service_keeps_one_item_per_call(
        tiny_pack_dir: Path, sock_dir: Path, pack_fp: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_SERVICE_LANE_MULTI", "0")
    async with running_app(_uds_settings(tiny_pack_dir, sock_dir)) as app:
        dims = dict(app.state.pgw.registry.get(pack_fp).linear_dims)
        info = await _open(app, pack_fp)
        items = _items(info, dims)
        lc, client = await _lane_client(app, sock_dir, pack_fp, multi=True)
        try:
            assert lc.lane_ticket is not None and lc.lane_ticket.max_items == 0
            req = make_req(pack_fp, K32, items, rid=secrets.token_hex(16))
            assert await asyncio.to_thread(lc._try_lane, req, None) is None  # no multi frame is ever sent
            assert lc.lane_stats.fallback_reasons == {"ineligible:items": 1}
            single = make_req(pack_fp, K32, items[:1], rid=secrets.token_hex(16))
            r = await asyncio.to_thread(lc._try_lane, single, None)  # single items as before
            assert r is not None and lc.lane_stats.trips == 1
            big = make_req(pack_fp, K32, items[3:4], rid=secrets.token_hex(16))
            assert await asyncio.to_thread(lc._try_lane, big, None) is None  # rows above max_rows: regular path
            assert lc.lane_stats.fallback_reasons["ineligible:items"] == 2
        finally:
            lc.lane_close()
            await client.aclose()


def test_ticket_parsing_of_format_1_and_format_2_services() -> None:
    from pgw.wire.messages import LinearInfo

    class _Model:
        def linear_dims(self) -> dict[str, tuple[int, int]]:
            return {"0.qkv": (8, 4)}

    class _Client:
        model = _Model()
        config = None

    opener = ServiceLaneOpener(_Client(), LaneConfig(transports=("uds",)))
    base = {"version": 1, "transport": "uds", "socket": "/run/pgw/x.sock", "ticket": "ab" * 32, "ttl_s": 30.0,
            "fingerprint": "fp", "linears": ["0.qkv"], "rings": ["K32A8"], "max_rows": 8}
    t1 = opener._ticket(json.dumps(base).encode(), "r", "fp", "K32A8")
    assert (t1.max_items, t1.max_multi_rows) == (0, 0) and t1.linears == (LinearInfo("0.qkv", 8, 4),)
    t2 = opener._ticket(json.dumps({**base, "versions": [1, 2], "max_items": 999, "max_multi_rows": 99999}).encode(),
                        "r", "fp", "K32A8")
    assert (t2.max_items, t2.max_multi_rows) == (64, 256)  # clamped to the format's limits
    t3 = opener._ticket(json.dumps({**base, "versions": [1], "max_items": 8}).encode(), "r", "fp", "K32A8")
    assert t3.max_items == 0  # not offered without version 2
    assert Status.OK == 0
