"""Maskgen wire v2 (E5): bit packing, provider rounding, randomised upload switch, noise budget, negotiation.

The end-to-end tests run real HE parameters (K32 ring, rank 16, q = 2^56) on 7B-shaped linears
through the HTTP route, the service's item function and the factory's HE backend. They check that
v2 decrypts to exactly the same ``W8 r`` as v1 for the same randomness, and that the every-row
check accepts both.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import msgpack
import numpy as np
import pytest
import torch

import pgw.hemask.lwe as lwe_mod
from pgw.core.errors import ConfigError, ProtocolError, VersionMismatchError
from pgw.core.ring import RingParams
from pgw.core.types import LinearId, LinearSpec
from pgw.hemask import DEGREE, HEParams, decrypt_wr, encrypt_masks, matmul_mod64, public_matrix
from pgw.hemask.backend import HEMaskBackend, HintStore, verify_rows
from pgw.hemask.client import (
    UPLOAD_SWITCH_BITS,
    RemoteMulmod64,
    V2Format,
    check_v2,
    switch_down,
    v2_formats,
    v2_noise_bound,
)
from pgw.hemask.lwe import MaskCiphertext
from pgw.hemask.server import add_maskgen_route
from pgw.hemask.wire import (
    MASKGEN_PATH,
    MaskgenItem,
    MaskgenRequest,
    MaskgenResponse,
    MaskgenResult,
    bit_fields,
    decode_maskgen_request,
    decode_maskgen_response,
    encode_maskgen_request,
    encode_maskgen_response,
    le_words,
    pack_bits,
    pack_y_v2,
    unpack_bits,
)
from pgw.service import ServiceSettings, hash_token
from pgw.service.handler import _maskgen_item
from pgw.service.kernels import ComputeKernel
from tests.service.conftest import TOKEN_A, asgi_client, running_app

U64 = np.uint64
SIGMA_7B = 0.37  # max ||w8 row|| / (127 sqrt(n)) measured on Qwen2.5-1.5B (0.29 down .. 0.37 q/up)


@pytest.fixture
def svc_tokens(tmp_path: Path) -> Path:
    p = tmp_path / "tenants.txt"
    p.write_text(f"tenant-a:{hash_token(TOKEN_A)}\n")
    return p


@pytest.fixture(scope="module")
def tiny_weights(tiny_pack: Any) -> dict[str, torch.Tensor]:
    return {lw.id.key: lw.w8.clone() for lw in tiny_pack.iter_linears()}


@pytest.fixture(scope="module")
def tiny_fp(tiny_pack: Any) -> str:
    return str(tiny_pack.fingerprint)


# ---- bit packing ---------------------------------------------------------------------------------
@pytest.mark.parametrize("bits", list(range(1, 65)))
def test_pack_bits_roundtrip_and_tail_layout(bits: int) -> None:
    rng = np.random.default_rng(bits)
    v = rng.integers(0, 2**64 - 1, size=8 * 37, dtype=U64, endpoint=True)
    m = U64((1 << bits) - 1) if bits < 64 else U64(2**64 - 1)
    data = pack_bits(v, bits)
    assert len(data) == v.size * bits // 8
    assert np.array_equal(unpack_bits(data, v.size, bits), v & m)
    fields, t = bit_fields(bits)
    assert sum(w for _, w in fields) + t == bits and all(w in (8, 16, 32, 64) for _, w in fields)
    pos = 0
    for off, w in fields:  # each field is a contiguous little-endian uint{w} array
        want = ((v >> U64(off)) & U64((1 << w) - 1 if w < 64 else 2**64 - 1)).astype(f"<u{w // 8}")
        assert data[pos:pos + v.size * w // 8] == want.tobytes()
        pos += v.size * w // 8
    if t:  # the tail is an LSB-first bit stream of t bits per value
        ref = 0
        for i, x in enumerate(((v >> U64(bits - t)) & U64((1 << t) - 1)).tolist()):
            ref |= x << (t * i)
        assert data[pos:] == ref.to_bytes(v.size * t // 8, "little")


def test_bit_fields_examples_and_errors() -> None:
    assert bit_fields(35) == ([(0, 32)], 3)
    assert bit_fields(53) == ([(0, 32), (32, 16)], 5)
    assert bit_fields(56) == ([(0, 32), (32, 16), (48, 8)], 0)
    assert bit_fields(43) == ([(0, 32), (32, 8)], 3)
    assert bit_fields(64) == ([(0, 64)], 0)
    with pytest.raises(ValueError):
        pack_bits(np.zeros(12, dtype=U64), 35)  # not a multiple of 8 values
    with pytest.raises(ValueError):
        unpack_bits(bytes(10), 16, 35)
    with pytest.raises(ValueError):
        bit_fields(65)


@pytest.mark.parametrize("nb", range(1, 9))
def test_le_words(nb: int) -> None:
    rng = np.random.default_rng(nb)
    v = rng.integers(0, 2**(8 * nb) - 1, size=41, dtype=U64, endpoint=True)
    raw = v.view(np.uint8).reshape(41, 8)[:, :nb].tobytes()
    assert np.array_equal(le_words(raw, 41, nb), v)
    with pytest.raises(ValueError):
        le_words(raw[:-1], 41, nb)


@pytest.mark.parametrize("bb,yb", [(53, 35), (56, 35), (56, 56), (61, 43), (64, 43), (64, 64), (24, 8)])
def test_pack_y_v2_is_exact_round_half_up(bb: int, yb: int) -> None:
    rng = np.random.default_rng(bb * 100 + yb)
    y = rng.integers(0, 2**64 - 1, size=8 * 50, dtype=U64, endpoint=True)
    y[:8] = [0, 2**64 - 1, (1 << bb) - 1, 1 << max(bb - yb - 1, 0), (1 << bb) - (1 << max(bb - yb - 1, 0)), 1, 2, 3]
    d = bb - yb
    want = [(((int(x) % (1 << bb)) + ((1 << (d - 1)) if d else 0)) >> d) % (1 << yb) for x in y.tolist()]
    got = unpack_bits(pack_y_v2(y, bb, yb), y.size, yb)
    assert got.tolist() == want
    # y may be reduced modulo any multiple of 2^bb (e.g. a 7- or 8-byte GEMM): same reply
    if bb < 64:
        assert pack_y_v2(y & U64((1 << bb) - 1), bb, yb) == pack_y_v2(y, bb, yb)
    yc = y.copy()
    assert pack_y_v2(yc, bb, yb, inplace=True) == pack_y_v2(y, bb, yb)


# ---- messages -------------------------------------------------------------------------------------
def _item_v2(n: int = 5, cols: int = DEGREE, bb: int = 53, yb: int = 35) -> MaskgenItem:
    return MaskgenItem("0.qkv", cols, bytes(n * cols * bb // 8), 0, 64, bb, yb)


def test_v2_request_roundtrip_and_validation() -> None:
    req = MaskgenRequest("r1", "f" * 64, (_item_v2(),), 2, (1, 2))
    got = decode_maskgen_request(encode_maskgen_request(req))
    assert got == req and got.items[0].v2 and got.items[0].b_nbytes(5) == 5 * DEGREE * 53 // 8
    assert got.items[0].y_nbytes(7) == 7 * DEGREE * 35 // 8
    with pytest.raises(VersionMismatchError):  # a v1-only server
        decode_maskgen_request(encode_maskgen_request(req), (1,))

    def raw(**item: Any) -> bytes:
        it = {"linear": "0.qkv", "cols": DEGREE, "bb": 53, "yb": 35, "b": bytes(5 * DEGREE * 53 // 8), **item}
        return bytes(msgpack.packb({"v": 2, "type": "maskgen", "id": "r", "fp": "f", "items": [it]},
                                   use_bin_type=True))

    for bad in ({"bb": 65}, {"yb": 54}, {"yb": 7}, {"bb": True}, {"b": bytes(10)}, {"yb": "35"}):
        with pytest.raises(ProtocolError):
            decode_maskgen_request(raw(**bad))
    v3 = bytes(msgpack.packb({"v": 3, "type": "maskgen", "id": "r", "fp": "f", "items": [{}]}, use_bin_type=True))
    with pytest.raises(VersionMismatchError) as ei:
        decode_maskgen_request(v3)
    assert ei.value.details["versions"] == [1, 2]
    with pytest.raises(ValueError):  # v1 items in a v2 request and vice versa
        encode_maskgen_request(MaskgenRequest("r", "f", (MaskgenItem("x", DEGREE, b"\0" * 7 * DEGREE, 0, 56),), 2))
    with pytest.raises(ValueError):
        encode_maskgen_request(MaskgenRequest("r", "f", (_item_v2(),), 1))


def test_v1_messages_stay_compatible() -> None:
    """A v1 request without ``vs`` (old client) decodes as before; ``vs`` is an ignorable extra key."""
    item = MaskgenItem("0.qkv", DEGREE, bytes(7 * 3 * DEGREE), 16, 56)
    old = encode_maskgen_request(MaskgenRequest("r1", "f", (item,)))
    assert b"vs" not in old and decode_maskgen_request(old).versions == ()
    new = encode_maskgen_request(MaskgenRequest("r1", "f", (item,), 1, (1, 2)))
    obj = msgpack.unpackb(new, raw=False)
    assert obj["v"] == 1 and obj["vs"] == [1, 2] and obj["items"][0] == {"linear": "0.qkv", "cols": DEGREE,
                                                                         "b": item.b, "drop": 16, "q": 56}
    resp = encode_maskgen_response(MaskgenResponse("r1", (MaskgenResult("0.qkv", DEGREE, bytes(5 * 4 * DEGREE)),),
                                                   1, (1, 2)))
    got = decode_maskgen_response(resp, MaskgenRequest("r1", "f", (item,)), {"0.qkv": 4})
    assert got.versions == (1, 2) and got.version == 1
    with pytest.raises(VersionMismatchError):  # a reply in another version than the request
        decode_maskgen_response(resp, MaskgenRequest("r1", "f", (_item_v2(),), 2), {"0.qkv": 4})


# ---- upload switch and noise budget ---------------------------------------------------------------
@pytest.mark.parametrize("q_bits", [56, 64])
def test_switch_down_is_unbiased_and_bounded(q_bits: int) -> None:
    rng = np.random.default_rng(q_bits)
    b = rng.integers(0, 2**q_bits - 1, size=(64, 256), dtype=U64, endpoint=True)
    b[0, :4] = [0, (1 << q_bits) - 1, (1 << q_bits) - 4, 7]
    bb = q_bits - UPLOAD_SWITCH_BITS
    flat = b.ravel()

    def eps(out: np.ndarray) -> np.ndarray:
        """``b - 2^3 b' mod 2^q`` as a signed integer."""
        d = (flat - (out << U64(3))) & U64((1 << q_bits) - 1 if q_bits < 64 else 2**64 - 1)
        if q_bits < 64:
            d = np.where(d >= U64(1 << (q_bits - 1)), d.astype(np.int64) - (1 << q_bits), d.astype(np.int64))
        return d.astype(np.int64)

    out = switch_down(b, q_bits, bb, np.random.default_rng(1))
    assert out.shape == (b.size,) and int(out.max()) < (1 << bb)
    c = (flat & U64(7)).astype(np.int64)
    e = eps(out)
    assert np.all((e == c) | (e == c - 8))  # two values, 8 apart: |eps| < 8
    # E[eps | b] = 0: the mean over fresh dithers of one fixed b vanishes, coefficient by coefficient
    acc = np.zeros(flat.size)
    trials = 64
    for s in range(trials):
        acc += eps(switch_down(b, q_bits, bb, np.random.default_rng(100 + s)))
    mean = acc / trials
    assert abs(float(mean.mean())) < 0.05  # sd of the overall mean ~ 4 / sqrt(64 * 16384)
    assert float(np.abs(mean).max()) < 4.0
    assert np.array_equal(switch_down(b, q_bits, q_bits, rng), flat)


def test_noise_budget_formats_for_7b() -> None:
    p = HEParams(k=32, rank=16)  # q = 2^56, delta = 2^24
    assert p.q == 56 and p.delta_bits == 24
    for n in (3584, 18944):  # hidden-size and down-projection inputs of Qwen2.5-7B
        norm = SIGMA_7B * 127 * math.sqrt(n)
        fmts = v2_formats(p, norm)
        assert [(f.b_bits, f.y_bits) for f in fmts] == [(56, 35), (53, 35)]  # K + 3 reply, 53-bit upload
        for f in fmts:
            check_v2(p, norm, f)
            assert f.bound < 2.0 ** 22  # below half the scaling factor with a 1-bit margin
        # every term: W e, the switch, the reply rounding, the hint's low byte
        b = v2_noise_bound(p, norm, 53, 35)
        assert b == pytest.approx(24 * p.sigma * norm + 24 * 4 * norm + 2.0**20 + 255 * 4096)
    # worst-case int8 rows (all |w| = 127) on the down projection: the 53-bit upload needs a wider reply
    worst = 127 * math.sqrt(18944)
    assert [(f.b_bits, f.y_bits) for f in v2_formats(p, worst)] == [(56, 35), (53, 39)]
    assert [(f.b_bits, f.y_bits) for f in v2_formats(p, worst, upload_switch=False)] == [(56, 35)]
    # far beyond the budget: v2 is refused (v1's own check would refuse too)
    assert v2_formats(p, 1e6) == ()
    with pytest.raises(ConfigError):
        check_v2(p, 1e6, V2Format(56, 35, 0.0))
    with pytest.raises(ConfigError):  # margin 1 bit: 2^22 is not enough with these terms
        check_v2(p, worst, V2Format(53, 35, 0.0))
    # K40 (q = 2^64): 61-bit uploads and 43-bit replies
    p40 = HEParams(k=40, rank=16)
    assert [(f.b_bits, f.y_bits) for f in v2_formats(p40, SIGMA_7B * 127 * math.sqrt(3584))] == [(64, 43), (61, 43)]


def test_client_plans_per_linear_and_refuses_over_budget() -> None:
    p = HEParams(k=32, rank=16)
    out = {"0.qkv": 4608, "0.down": 3584, "0.up": 37888, "0.o": 3584}
    norms = {"0.qkv": 2800.0, "0.down": 1e6, "0.up": 2800.0}  # 0.down over budget, 0.o unknown
    r = RemoteMulmod64(None, "f" * 64, out, q_bits=56, params=p, row_norms=norms)
    assert r.versions == (1, 2)
    assert r.v2_format("0.qkv", 3584) == V2Format(53, 35, pytest.approx(r.v2_format("0.qkv", 3584).bound))
    assert r.v2_format("0.down", 18944) is None and r.v2_format("0.o", 3584) is None
    assert r.v2_format("0.up", 3584).b_bits == 53
    with pytest.raises(ConfigError):
        RemoteMulmod64(None, "f" * 64, out, q_bits=64, params=p, row_norms=norms)
    assert RemoteMulmod64(None, "f", out, q_bits=56).versions == (1,)  # no params: v1 only
    assert RemoteMulmod64(None, "f", out, q_bits=56, params=p, row_norms=norms, wire_v2=False).versions == (1,)


# ---- service item function -----------------------------------------------------------------------
def _kernel(m: int, n: int, seed: int = 0, worst: bool = False) -> tuple[ComputeKernel, np.ndarray]:
    rng = np.random.default_rng(seed)
    w = np.full((m, n), 127, dtype=np.int8) if worst else rng.integers(-127, 128, size=(m, n), dtype=np.int8)
    if worst:
        w[::2] = -127
    return ComputeKernel(LinearSpec(LinearId(0, "qkv"), m, n), torch.from_numpy(w.copy()), device="cpu"), w


@pytest.mark.parametrize("bb,yb", [(53, 35), (56, 35), (61, 43), (64, 43), (56, 56)])
def test_service_item_v2_matches_reference(bb: int, yb: int) -> None:
    m, n, cols = 9, 70, 2 * DEGREE
    kern, w = _kernel(m, n, seed=bb)
    rng = np.random.default_rng(yb)
    b = rng.integers(0, 2**bb - 1, size=(n, cols), dtype=U64, endpoint=True)
    item = MaskgenItem("0.qkv", cols, pack_bits(b, bb), 0, 64, bb, yb)
    want = pack_y_v2(matmul_mod64(w, b), bb, yb)
    assert _maskgen_item(kern, item) == want
    # the CUDA branch (8-byte words) and kernels without the wire entry point give the same bytes
    as_cuda = SimpleNamespace(in_features=n, out_features=m, device=SimpleNamespace(type="cuda"),
                              mulmod64_wire=kern.mulmod64_wire)
    assert _maskgen_item(as_cuda, item) == want
    no_wire = SimpleNamespace(in_features=n, out_features=m, mulmod64=kern.mulmod64)
    assert _maskgen_item(no_wire, item) == want


def test_handler_versions_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.service.handler import _maskgen_versions_default

    assert _maskgen_versions_default() == (1, 2)
    monkeypatch.setenv("PGW_SERVICE_MASKGEN_V2", "false")
    assert _maskgen_versions_default() == (1,)
    monkeypatch.setenv("PGW_SERVICE_MASKGEN_V2", "maybe")
    assert _maskgen_versions_default() == (1, 2)


# ---- end to end over HTTP with the service (tiny pack, negotiation) -------------------------------
def _settings(pack_dir: Any, tokens_file: Any) -> ServiceSettings:
    return ServiceSettings(packs=[pack_dir], device="cpu", rings=["K40A12"], tokens_file=tokens_file)


def _ref_wr(r: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    return np.array((r.astype(object) @ w.T.astype(object)) % (1 << k), dtype=np.int64)


def _copy_ct(ct: MaskCiphertext) -> MaskCiphertext:
    return MaskCiphertext(ct.b, ct.secret.copy(), ct.rows)


@pytest.mark.parametrize("k", [32, 40])
async def test_http_negotiates_v2_and_decrypts_identically(tiny_pack_dir: Any, svc_tokens: Any, tiny_fp: str,
                                                           tiny_weights: dict[str, torch.Tensor], k: int) -> None:
    lid, w8 = next(iter(tiny_weights.items()))
    w = w8.numpy().astype(np.int64)
    m, n = w.shape
    p = HEParams(k=k, rank=8)
    rng = np.random.default_rng(k)
    r = rng.integers(0, 1 << k, size=(DEGREE, n), dtype=np.int64)
    a = public_matrix(bytes(range(32)), n, p)
    hint = matmul_mod64(w, a)
    ct = encrypt_masks(r, a, p, _seed=bytes(32))  # test hook: the same randomness for every version
    norms = {key: float(np.sqrt((v.numpy().astype(np.float64) ** 2).sum(1)).max()) for key, v in tiny_weights.items()}
    out = {key: int(v.shape[0]) for key, v in tiny_weights.items()}
    async with running_app(_settings(tiny_pack_dir, svc_tokens)) as app:
        client = asgi_client(app, TOKEN_A)
        try:
            remote = RemoteMulmod64(client, tiny_fp, out, q_bits=p.q, params=p, row_norms=norms)
            assert remote.server_v2 is None
            y1 = await remote.acall(lid, ct.b)  # v1, advertising v2; the reply advertises v2
            up1, down1 = remote.bytes_up, remote.bytes_down
            assert remote.server_v2 is True and remote.calls_v2 == 0
            y2 = await remote.acall(lid, ct.b)  # v2
            assert remote.calls_v2 == 1
            up2, down2 = remote.bytes_up - up1, remote.bytes_down - down1
            v1only = RemoteMulmod64(client, tiny_fp, out, q_bits=p.q)  # no params: never v2
            await v1only.acall(lid, ct.b)
            await v1only.acall(lid, ct.b)
            assert v1only.calls_v2 == 0
        finally:
            await client.aclose()
    ref = _ref_wr(r, w, k)
    assert np.array_equal(decrypt_wr(y1, hint, _copy_ct(ct), p), ref)
    assert np.array_equal(decrypt_wr(y2, hint, _copy_ct(ct), p), ref)
    fmt = remote.v2_format(lid, n)
    assert fmt is not None and (fmt.b_bits, fmt.y_bits) in {(p.q, k + 3), (p.q - 3, k + 3)}
    # payload sizes: v1 q/8 bytes up and (q-16)/8 down; v2 bit-packed (msgpack framing < 200 bytes)
    assert p.q // 8 * n * DEGREE < up1 < p.q // 8 * n * DEGREE + 200
    assert fmt.b_bits * n * DEGREE // 8 < up2 < fmt.b_bits * n * DEGREE // 8 + 200
    assert (p.q - 16) // 8 * m * DEGREE < down1 < (p.q - 16) // 8 * m * DEGREE + 200
    assert fmt.y_bits * m * DEGREE // 8 < down2 < fmt.y_bits * m * DEGREE // 8 + 200
    assert down2 < down1


async def test_http_v2_turned_off_on_the_service_stays_v1(tiny_pack_dir: Any, svc_tokens: Any, tiny_fp: str,
                                                          tiny_weights: dict[str, torch.Tensor],
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_SERVICE_MASKGEN_V2", "false")
    lid, w8 = next(iter(tiny_weights.items()))
    n = int(w8.shape[1])
    p = HEParams(k=32, rank=8)
    norms = {key: 2000.0 for key in tiny_weights}
    out = {key: int(v.shape[0]) for key, v in tiny_weights.items()}
    b = np.zeros((n, DEGREE), dtype=U64)
    hdr = {"content-type": "application/vnd.pgw+msgpack", "authorization": f"Bearer {TOKEN_A}"}
    async with running_app(_settings(tiny_pack_dir, svc_tokens)) as app:
        client = asgi_client(app, TOKEN_A)
        try:
            remote = RemoteMulmod64(client, tiny_fp, out, q_bits=p.q, params=p, row_norms=norms)
            for _ in range(3):
                await remote.acall(lid, b)
            assert remote.server_v2 is False and remote.calls_v2 == 0 and remote.calls == 3
            # a v2 request anyway: 426 version_mismatch, listing the served versions
            req = MaskgenRequest("r9", tiny_fp, (MaskgenItem(lid, DEGREE, bytes(n * DEGREE * 53 // 8), 0, 64, 53, 35),),
                                 2, (1, 2))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as h:
                resp = await h.post(MASKGEN_PATH, content=encode_maskgen_request(req), headers=hdr)
            assert resp.status_code == 426
            assert msgpack.unpackb(resp.content, raw=False)["details"]["versions"] == [1]
        finally:
            await client.aclose()


class _ScriptedClient:
    """``post_raw_sync`` stand-in: answers v1 with a v2 advertisement and refuses v2 (an old or downgraded server)."""

    def __init__(self, m: int, refusal: type[Exception]):
        self.m = m
        self.refusal = refusal
        self.versions_seen: list[int] = []

    def post_raw_sync(self, path: str, body: bytes, rid: str, decode: Any, *, timeout_s: float) -> Any:
        req = decode_maskgen_request(body)
        self.versions_seen.append(req.version)
        if req.version == 2:
            raise self.refusal("maskgen: expected a v1 'maskgen' message")
        it = req.items[0]
        y = bytes(it.y_nbytes(self.m))
        return decode(encode_maskgen_response(MaskgenResponse(rid, (MaskgenResult(it.linear, it.cols, y),), 1, (1, 2))))


@pytest.mark.parametrize("refusal", [ProtocolError, VersionMismatchError])
def test_v2_refusal_falls_back_to_v1(refusal: type[Exception], monkeypatch: pytest.MonkeyPatch) -> None:
    import pgw.hemask.client as client_mod

    p = HEParams(k=32, rank=8)
    fake = _ScriptedClient(4, refusal)
    remote = RemoteMulmod64(fake, "f", {"0.qkv": 4}, q_bits=56, params=p, row_norms={"0.qkv": 500.0})
    b = np.zeros((6, DEGREE), dtype=U64)
    remote("0.qkv", b)  # v1, learns v2
    assert remote.server_v2 is True
    y = remote("0.qkv", b)  # v2 refused -> the same ciphertext again as v1
    assert y.shape == (4, DEGREE) and fake.versions_seen == [1, 2, 1] and remote.server_v2 is False
    remote("0.qkv", b)  # advertisements are ignored for V2_RETRY_S after a refusal
    assert fake.versions_seen[-1] == 1
    monkeypatch.setattr(client_mod, "V2_RETRY_S", 0.0)
    remote("0.qkv", b)  # retry window over: the next advertisement turns v2 back on
    assert remote.server_v2 is True


# ---- end to end at real parameters: factory HE backend, HTTP route, service item function --------
@dataclass
class _Linear:
    id: LinearId
    w8: torch.Tensor


class _Pack:
    """Minimal model pack: 7B-shaped linears (hidden 3584) sliced to a few output rows."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.fingerprint = "e5" * 32
        self.short_fingerprint = self.fingerprint[:12]
        self._w = weights

    def _ids(self) -> list[LinearId]:
        return [LinearId(i, kind) for i, kind in enumerate(self._w)]

    def linear_specs(self) -> list[LinearSpec]:
        return [LinearSpec(lid, *self._w[lid.kind].shape) for lid in self._ids()]

    def iter_linears(self) -> Any:
        for lid in self._ids():
            yield _Linear(lid, torch.from_numpy(self._w[lid.kind]))


class _KernelHandler:
    """Service-side maskgen on :class:`ComputeKernel` with the service's own item function."""

    def __init__(self, kernels: dict[str, ComputeKernel], versions: tuple[int, ...]):
        self.kernels = kernels
        self.maskgen_versions = versions

    async def maskgen(self, req: MaskgenRequest, identity: Any) -> MaskgenResponse:
        return MaskgenResponse(req.request_id, tuple(
            MaskgenResult(it.linear, it.cols, _maskgen_item(self.kernels[it.linear], it)) for it in req.items))


class _Recorder:
    """Wraps the remote: records what each call sent (the ciphertext) and got back."""

    def __init__(self, remote: RemoteMulmod64):
        self.remote = remote
        self.calls: list[tuple[str, np.ndarray]] = []

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        self.calls.append((linear, b.copy()))
        return self.remote(linear, b)


def _seeds() -> Any:
    ctr = itertools.count()
    return lambda: bytes([next(ctr) % 256]) * 32  # test-only: the same "fresh" seeds in every run


def test_real_parameters_v1_and_v2_decrypt_bit_identically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from starlette.applications import Starlette

    rng = np.random.default_rng(7)
    n = 3584  # Qwen2.5-7B hidden size
    weights = {
        # realistic rows (Qwen-like norm ratio) and worst-case rows (|w| = 127 everywhere)
        "qkv": np.clip(np.round(rng.standard_normal((48, n)) * 127 * SIGMA_7B), -127, 127).astype(np.int8),
        "gate_up": _kernel(32, n, worst=True)[1],
    }
    pack = _Pack(weights)
    ring = RingParams.from_name("K32A12C27")
    hints = HintStore.build(tmp_path / "hints", pack, 16, q_bits=56)
    params = HEParams(k=ring.k, rank=16)
    assert params.q == 56 and hints.rank == 16
    kernels = {s.id.key: ComputeKernel(s, torch.from_numpy(weights[s.id.kind].copy()), device="cpu")
               for s in pack.linear_specs()}
    app = Starlette()
    add_maskgen_route(app, _KernelHandler(kernels, (1, 2)))
    out = {s.id.key: s.out_features for s in pack.linear_specs()}
    rows = {"0.qkv": 300, "1.gate_up": DEGREE}  # two groups (one partial) and one group
    masks = {key: torch.from_numpy(rng.integers(0, 1 << ring.k, size=(rows[key], n), dtype=np.int64))
             for key in rows}

    def run(version: int) -> tuple[dict[str, np.ndarray], _Recorder, RemoteMulmod64]:
        client = asgi_client(app)
        try:
            remote = RemoteMulmod64(client, pack.fingerprint, out, q_bits=56, params=params,
                                    row_norms=hints.meta["row_norms"], wire_v2=version == 2)
            if version == 2:
                remote("0.qkv", np.zeros((n, DEGREE), dtype=U64))  # learn the service's versions first
                assert remote.server_v2 is True
                remote.bytes_up = remote.bytes_down = remote.calls = 0
            rec = _Recorder(remote)
            backend = HEMaskBackend(pack, ring, hints, rec, groups_per_call=1, max_inflight=1)
            monkeypatch.setattr(lwe_mod, "new_seed", _seeds())
            got = {key: backend.matmul(key, masks[key]).numpy() for key in rows}
            return got, rec, remote
        finally:
            client.close()

    got1, rec1, remote1 = run(1)
    got2, rec2, remote2 = run(2)
    assert remote1.calls_v2 == 0 and remote2.calls_v2 == remote2.calls == len(rec2.calls) == 3
    # same randomness: the provider was asked about the very same ciphertexts
    assert [k for k, _ in rec1.calls] == [k for k, _ in rec2.calls]
    assert all(np.array_equal(b1, b2) for (_, b1), (_, b2) in zip(rec1.calls, rec2.calls, strict=True))
    for key in rows:
        w = weights[key.split(".")[1]].astype(np.int64)
        ref = _ref_wr(masks[key].numpy(), w, ring.k)
        assert np.array_equal(got1[key], ref), key
        assert np.array_equal(got2[key], got1[key]), key  # bit-identical decrypted masks
        # the factory's every-row check (Freivalds with private vectors) accepts both, rejects a flip
        cu = rng.integers(0, 1 << ring.k, size=(w.shape[0], 2), dtype=np.int64).astype(U64)
        cw = (w.T.astype(U64) @ cu)
        assert not verify_rows(masks[key].numpy(), got2[key], cu, cw, ring.mask).any()
        bad = got2[key].copy()
        bad[5, 3] ^= 1
        assert verify_rows(masks[key].numpy(), bad, cu, cw, ring.mask).tolist() == [i == 5 for i in range(rows[key])]
    fmt = {key: remote2.v2_format(key, n) for key in rows}
    assert {(f.b_bits, f.y_bits) for f in fmt.values() if f is not None} == {(53, 35)}
    # wire bytes for the same work: v2 is the bit-packed size, about 9% less than v1 here
    assert remote2.bytes_up < remote1.bytes_up * 53 / 56 + 600
    assert remote2.bytes_down < remote1.bytes_down * 35 / 40 + 600
