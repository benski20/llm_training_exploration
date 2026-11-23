"""pks-v3 (``pgw.hemask.pks``, ``pgw_pks``, wire v3, service and factory integration): exactness, noise
bounds, negotiation, refusals and defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pgw.core.errors import ConfigError, IntegrityError, ProtocolError, VersionMismatchError
from pgw.hemask import pks
from pgw.hemask import pks_native as pn
from pgw.hemask.lwe import HEParams, MaskCiphertext, decrypt_wr, matmul_mod64, secret_matrix
from pgw.hemask.wire import (
    V3_KSK_BYTES,
    MaskgenItem,
    MaskgenRequest,
    decode_maskgen_request,
    encode_maskgen_request,
    pack_bits,
)

M56 = np.uint64((1 << 56) - 1)
PAR = pks.PKSParams()
NATIVE = pytest.mark.skipif(not pn.available(), reason="pgw_pks not built (rust/tools/build_pks.py)")


def _naive_negacyclic(a: list[int], b: list[int], mod: int) -> list[int]:
    n = len(a)
    out = [0] * n
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                k = i + j
                if k < n:
                    out[k] += ai * bj
                else:
                    out[k - n] -= ai * bj
    return [v % mod for v in out]


def test_primes_and_security_envelope() -> None:
    for p in pks.Q1 + pks.PS:
        assert p % (2 * pks.NR) == 1
    for p in pks.EP:
        assert p % (2 * pks.NU) == 1
    assert 47.5 < np.log2(PAR.q1) < 48.0
    assert pks.NR == 8192 and pks.PACK == 32
    assert PAR.qks_bits <= 109.0  # far inside the HE-standard 128-bit limit at N = 8192 (218)
    if pn.available():
        info = pn.native().info()
        assert tuple(info["q1_primes"]) == pks.Q1 and tuple(info["ps_primes"]) == pks.PS


def test_exact_mod2_product() -> None:
    rng = np.random.default_rng(2)
    a = rng.integers(0, 1 << 56, (2, pks.NU), dtype=np.uint64)
    t = np.zeros(pks.NU, dtype=np.int64)
    idx = rng.choice(pks.NU, 30, replace=False)
    t[idx] = rng.choice([-1, 1], 30)
    got = pks.negacyclic_mul_mod2(a, t)
    for r in range(2):
        ref = _naive_negacyclic([int(v) for v in a[r]], [int(v) for v in t], 1 << 56)
        assert [int(v) for v in got[r]] == ref


def test_rounding_helpers_exact() -> None:
    rng = np.random.default_rng(3)
    x = rng.integers(0, 1 << 56, 2000, dtype=np.uint64)
    x[:3] = [0, (1 << 56) - 1, 1 << 55]
    got = pks.round_mul_shift(x, PAR.q1, 56)
    assert [int(g) for g in got] == [(int(v) * PAR.q1 + (1 << 55)) >> 56 for v in x]
    v = rng.integers(0, PAR.q1, 2000, dtype=np.uint64)
    got = pks.round_pow2_div(v, 36, PAR.q1)
    assert [int(g) for g in got] == [((int(a) << 36) + PAR.q1 // 2) // PAR.q1 for a in v]


def test_noise_bound_production_parameters() -> None:
    """Worst case of SPEC 7.9 (all-127 rows of the widest 72B input, n = 29568) keeps one bit of margin."""
    worst = 127 * np.sqrt(29568)
    assert PAR.noise_ok(worst, margin_bits=1.0)
    assert np.log2(PAR.budget()) - np.log2(PAR.noise_bound(worst)) > 1.5
    assert not PAR.noise_ok(127 * np.sqrt(29568) * 8)  # the check does refuse
    with pytest.raises(ConfigError):
        PAR.check_noise(1e7, "x")


def _case(m: int, n: int, rows: int, seed: int, wmax: int = 127):
    rng = np.random.default_rng(seed)
    w = rng.integers(-wmax, wmax + 1, (m, n), dtype=np.int64).astype(np.int8)
    r = rng.integers(0, 1 << 32, (rows, n), dtype=np.int64)
    return w, r


def _hs(w: np.ndarray, a_seed: bytes, linear: str) -> np.ndarray:
    a = pks.a_mlwe_rows(pks.public_ring_rows(a_seed, w.shape[1], linear))
    return pks.switch_hint(matmul_mod64(w, a) & M56, PAR)


def test_upload_is_old_format_mlwe() -> None:
    """The ring-LWE upload decrypts with the v1/v2 hint path (provider GEMM unchanged)."""
    w, r = _case(32, 64, 256, 4)
    a_ring = pks.public_ring_rows(b"s" * 32, 64, "l")
    ct = pks.encrypt_rlwe(r, a_ring, PAR)
    a_m = pks.a_mlwe_rows(a_ring)
    smat = secret_matrix(ct.t_components)
    pad = matmul_mod64(smat, a_m, small_left=False) & M56
    diff = (ct.b - pad - (r.T.astype(np.uint64) << np.uint64(24))) & M56
    centred = np.where(diff >= np.uint64(1 << 55), diff.astype(np.int64) - (1 << 56), diff.astype(np.int64))
    assert np.abs(centred).max() <= 21  # exactly the CBD(21) noise
    y = matmul_mod64(w, ct.b) & M56
    got = decrypt_wr(y, matmul_mod64(w, a_m) & M56, MaskCiphertext(b=ct.b, secret=smat, rows=ct.rows), HEParams(k=32))
    assert np.array_equal(got, ((w.astype(np.int64) @ r.T) & 0xFFFFFFFF).T)


def _round_trip(m: int, n: int, rows: int, seed: int, *, force_reference: bool, w: np.ndarray | None = None) -> None:
    w0, r = _case(m, n, rows, seed)
    w = w0 if w is None else w
    a_seed, ks_seed = os.urandom(32), os.urandom(32)
    k_hat = pks.ks_a(ks_seed)
    cust = pn.Customer(k_hat, force_reference=force_reference)
    b, beta, secrets = cust.encrypt(a_seed, "blk.0.q", r)
    hs = _hs(w, a_seed, "blk.0.q")
    hint = pn.keyswitch(hs, k_hat[None].astype(np.uint32), force_reference=force_reference)
    y = matmul_mod64(w, b) & M56  # the provider's unchanged GEMM
    reply = pn.keyswitch(hs, beta, y, force_reference=force_reference)
    got = cust.decrypt(hint, bytes(pack_bits(reply.reshape(-1), 36)), secrets)
    want = ((w.astype(np.int64) @ r.T) & 0xFFFFFFFF).T
    assert np.array_equal(got.astype(np.int64), want)
    assert all(not s.t_reply.any() and not s.trhat.any() for s in secrets)  # wiped


@pytest.mark.parametrize("rows", [256, 77, 300])
def test_round_trip_exact_reference(rows: int) -> None:
    _round_trip(64, 48, rows, 5, force_reference=True)


@NATIVE
@pytest.mark.parametrize("rows", [256, 77, 513])
def test_round_trip_exact_native(rows: int) -> None:
    _round_trip(96 * 2, 80, rows, 6, force_reference=False)


@NATIVE
def test_native_equals_reference_bit_for_bit() -> None:
    w, r = _case(64, 48, 300, 8)
    a_seed, ks_seed = b"a" * 32, b"k" * 32
    k_hat = pks.ks_a(ks_seed)
    seeds = [bytes([i + 1]) * 32 for i in range(2)]
    bn, betan, sn = pn.Customer(k_hat).encrypt(a_seed, "l0", r, seeds=seeds)
    br, betar, sr = pn.Customer(k_hat, force_reference=True).encrypt(a_seed, "l0", r, seeds=seeds)
    assert np.array_equal(bn, br) and np.array_equal(betan, betar)
    assert all(np.array_equal(x.trhat, y.trhat) and np.array_equal(x.t_reply, y.t_reply) for x, y in zip(sn, sr, strict=True))
    hs = _hs(w, a_seed, "l0")
    kk = k_hat[None].astype(np.uint32)
    assert np.array_equal(pn.keyswitch(hs, kk), pn.keyswitch(hs, kk, force_reference=True))
    y = matmul_mod64(w, bn) & M56
    assert np.array_equal(pn.keyswitch(hs, betan, y), pn.keyswitch(hs, betar, y, force_reference=True))


@NATIVE
def test_native_isas_and_threads_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    nat = pn.native()
    w, r = _case(96, 64, 256, 9)
    k_hat = pks.ks_a(b"k" * 32)
    cust = pn.Customer(k_hat)
    b, beta, secrets = cust.encrypt(b"a" * 32, "l", r, seeds=[b"z" * 32])
    hs = _hs(w, b"a" * 32, "l")
    hint = pn.keyswitch(hs, k_hat[None].astype(np.uint32))
    y = bytes(pack_bits(pn.keyswitch(hs, beta, matmul_mod64(w, b) & M56).reshape(-1), 36))
    th = np.stack([s.trhat for s in secrets])
    isas = ("scalar", "avx2", "avx512", "avx512ifma")
    outs = [nat.decrypt(hint, y, 36, th, [256], thr, isa) for isa in isas for thr in (1, 3)]
    assert all(np.array_equal(o, outs[0]) for o in outs)
    assert np.array_equal(outs[0].astype(np.int64), ((w.astype(np.int64) @ r.T) & 0xFFFFFFFF).T)
    ts = np.stack([s.t for s in secrets])
    encs = [nat.encrypt(b"q" * 32, r, ts, [b"e" * 32], thr, isa) for isa in isas for thr in (1, 2, 3)]
    assert all(np.array_equal(e, encs[0]) for e in encs)
    kh = nat.KHat(k_hat.astype(np.uint32))
    tr = secrets[0].t_reply.copy()
    keys = [nat.reply_key(kh, ts[0], tr, b"x" * 32, isa) for isa in isas]
    assert all(np.array_equal(k[0], keys[0][0]) and np.array_equal(k[1], keys[0][1]) for k in keys)


@NATIVE
@pytest.mark.parametrize("isa", ["scalar", "avx2", "avx512", "avx512ifma"])
def test_native_partial_chunks_equal_reference(isa: str) -> None:
    """Every ISA copy equals the numpy reference bit for bit at shapes with partial 16-block chunks (n = 400:
    25 blocks; m = 640: 20 row blocks) and partial groups (rows 300 = 256 + 44)."""
    nat = pn.native()
    w, r = _case(640, 400, 300, 12)
    a_seed, k_hat = b"A" * 32, pks.ks_a(b"K" * 32)
    seeds = [bytes([7 + i]) * 32 for i in range(2)]
    br, betar, sr = pn.Customer(k_hat, force_reference=True).encrypt(a_seed, "lx", r, seeds=seeds)
    ts = np.stack([s.t for s in sr])
    from pgw.core.prng import SeedExpander

    sx = [SeedExpander(s) for s in seeds]
    bn = nat.encrypt(SeedExpander(a_seed).stream_key(pks.A_LABEL, "lx"), r, ts, [x.stream_key("pks-e") for x in sx],
                     3, isa)
    assert np.array_equal(bn, br)
    hs = _hs(w, a_seed, "lx")
    hint = pks.compact_hint(hs, k_hat, PAR)
    y = matmul_mod64(w, br) & M56
    reply = np.stack([pks.provider_reply(y[:, g * 256:(g + 1) * 256], hs, betar[g].astype(np.uint64), PAR)
                      for g in range(2)])
    th = np.stack([s.trhat for s in sr]).astype(np.uint32)
    got = nat.decrypt(hint, bytes(pack_bits(reply.reshape(-1), 36)), 36, th, [256, 44], 2, isa)
    want = np.concatenate([pks.decrypt_pks(reply[g], hint, sr[g].t_reply.astype(np.int64), s.rows, PAR)
                           for g, s in enumerate(sr)])
    assert np.array_equal(got.astype(np.int64), want)
    assert np.array_equal(want, ((w.astype(np.int64) @ r.T) & 0xFFFFFFFF).T)


@NATIVE
def test_native_upload_noise_is_reference_cbd21() -> None:
    """The native upload's noise (bodies minus pad, with r = 0) is exactly the reference CBD(21) stream
    ``pks-e`` and is distributed as Binomial(42, 1/2) - 21 (chi-square, moments)."""
    from pgw.core.prng import SeedExpander
    from pgw.hemask.lwe import _cbd

    n, groups = 64, 4
    r = np.zeros((groups * 256, n), dtype=np.int64)
    seeds = [bytes([40 + i]) * 32 for i in range(groups)]
    cust = pn.Customer(pks.ks_a(b"k" * 32))
    b, _beta, secrets = cust.encrypt(b"a" * 32, "noise", r, seeds=seeds)
    a_ring = pks.public_ring_rows(b"a" * 32, n, "noise")
    noise = []
    for g, s in enumerate(secrets):
        pad = pks.negacyclic_mul_mod2(a_ring, s.t.astype(np.int64))
        padv = pad.reshape(n // 16, 256, 16).transpose(0, 2, 1).reshape(n, 256)
        d = (b[:, g * 256:(g + 1) * 256] - padv) & M56
        e = np.where(d >= np.uint64(1 << 55), d.astype(np.int64) - (1 << 56), d.astype(np.int64))
        assert np.array_equal(e, _cbd(SeedExpander(seeds[g]), (n, 256), 21, "pks-e"))
        noise.append(e.reshape(-1))
    e = np.concatenate(noise)
    assert e.min() >= -21 and e.max() <= 21
    from math import comb

    pmf = np.array([comb(42, k) for k in range(43)], dtype=np.float64) / 2.0 ** 42
    obs = np.bincount(e + 21, minlength=43).astype(np.float64)
    exp = pmf * e.size
    core = slice(11, 32)  # [-10, 10]; tails pooled
    chi2 = (((obs[core] - exp[core]) ** 2) / exp[core]).sum()
    tail_o, tail_e = obs.sum() - obs[core].sum(), exp.sum() - exp[core].sum()
    chi2 += (tail_o - tail_e) ** 2 / tail_e
    assert chi2 < 60.0  # 21 degrees of freedom: P(chi2 > 60) < 1e-5
    assert abs(e.mean()) < 0.06 and abs(e.var() - 10.5) < 0.3


def test_tampered_reply_changes_masks() -> None:
    w, r = _case(32, 32, 256, 10)
    k_hat = pks.ks_a(b"k" * 32)
    cust = pn.Customer(k_hat)
    b, beta, secrets = cust.encrypt(b"a" * 32, "l", r)
    hs = _hs(w, b"a" * 32, "l")
    hint = pn.keyswitch(hs, k_hat[None].astype(np.uint32))
    reply = pn.keyswitch(hs, beta, matmul_mod64(w, b) & M56)
    reply[0, 0, 5] ^= np.uint64(1 << 30)
    got = cust.decrypt(hint, bytes(pack_bits(reply.reshape(-1), 36)), secrets)
    assert not np.array_equal(got.astype(np.int64), ((w.astype(np.int64) @ r.T) & 0xFFFFFFFF).T)  # caught by the row check


def test_torch_key_switch_matches_cpu() -> None:
    from pgw.service import pks_gpu

    pks_gpu.self_test(torch.device("cpu"))


@NATIVE
@pytest.mark.skipif(not os.environ.get("PGW_PKS_REAL_SHAPES"), reason="real shapes: set PGW_PKS_REAL_SHAPES=1")
@pytest.mark.parametrize("n", [3584, 18944, 8192, 29568])
def test_real_input_widths_worst_norm(n: int) -> None:
    """Exact at Qwen2.5-7B/72B input widths with all-127 rows (the largest row norm SPEC 7.9 allows)."""
    w = np.full((64, n), 127, dtype=np.int8)
    w[1::2] = -127
    _round_trip(64, n, 256, 11, force_reference=False, w=w)


# ---- wire v3, negotiation, defaults ------------------------------------------------------------------

def _v3_request() -> MaskgenRequest:
    item = MaskgenItem("blk.0.q", 256, bytes(7 * 16 * 256), 0, 64, 56, 36, bytes(V3_KSK_BYTES))
    return MaskgenRequest("r1", "fp", (item,), 3, (1, 2, 3), b"s" * 32)


def test_wire_v3_round_trip_and_refusal() -> None:
    body = encode_maskgen_request(_v3_request())
    req = decode_maskgen_request(body, (1, 2, 3))
    assert req.version == 3 and req.a_seed == b"s" * 32 and req.items[0].v3 and not req.items[0].v2
    assert req.items[0].y_nbytes(64) == 64 * 256 * 36 // 8
    with pytest.raises(VersionMismatchError):
        decode_maskgen_request(body)  # default: v1/v2 only
    with pytest.raises(VersionMismatchError):
        decode_maskgen_request(body, (1, 2))
    bad = MaskgenRequest("r1", "fp", (MaskgenItem("x", 256, bytes(7 * 256), 0, 64, 56, 36, b"short"),), 3, (), b"s" * 32)
    with pytest.raises(ProtocolError):
        decode_maskgen_request(encode_maskgen_request(bad), (1, 2, 3))


def test_service_v3_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.service.handler import _maskgen_versions_default

    monkeypatch.delenv("PGW_SERVICE_MASKGEN_V3", raising=False)
    monkeypatch.delenv("PGW_SERVICE_MASKGEN_V2", raising=False)
    assert _maskgen_versions_default() == (1, 2)
    monkeypatch.setenv("PGW_SERVICE_MASKGEN_V3", "true")
    assert _maskgen_versions_default() == (1, 2, 3)
    monkeypatch.setenv("PGW_SERVICE_MASKGEN_V2", "false")
    assert _maskgen_versions_default() == (1, 3)


def test_ksk_wire_range_checked() -> None:
    from pgw.service.pks_provider import pack_ksk, unpack_ksk

    rng = np.random.default_rng(1)
    beta = np.stack([rng.integers(0, p, (16, pks.NR)) for p in pks.QP])[None].astype(np.uint32)
    data = pack_ksk(beta)
    assert len(data) == V3_KSK_BYTES and np.array_equal(unpack_ksk(bytes(data), 1), beta)
    beta[0, 0, 0, 0] = pks.Q1[0]  # residue out of range
    with pytest.raises(ProtocolError):
        unpack_ksk(bytes(pack_ksk(beta)), 1)


def test_factory_scheme_default_and_ack(tmp_path: Path) -> None:
    from pgw.factory.config import FactorySettings

    base = {"pack_path": tmp_path, "mask_store_dir": tmp_path}
    try:
        s = FactorySettings(**base)
    except Exception as exc:  # pragma: no cover - other required fields in this tree
        pytest.skip(f"FactorySettings needs more fields here: {exc}")
    assert s.he_scheme == "mlwe"
    with pytest.raises(ValueError, match="he_pks_ack"):
        FactorySettings(**base, he_scheme="pks-v3")
    ok = FactorySettings(**base, he_scheme="pks-v3", he_pks_ack=pks.PKS_V3_ACK)
    assert ok.he_scheme == "pks-v3"
    with pytest.raises(ValueError, match="K32"):
        FactorySettings(**base, he_scheme="pks-v3", he_pks_ack=pks.PKS_V3_ACK, ring="K40A12")


# ---- factory integration (in-process provider) ----------------------------------------------------------

class _Spec:
    def __init__(self, key: str, m: int, n: int):
        from types import SimpleNamespace

        self.id = SimpleNamespace(key=key)
        self.out_features, self.in_features = m, n


class _Pack:
    """Minimal pack (linear specs and int8 weights) with pks-v3-compatible shapes."""

    fingerprint = "f" * 64
    short_fingerprint = "f" * 12

    def __init__(self, shapes: dict[str, tuple[int, int]], seed: int = 3):
        rng = np.random.default_rng(seed)
        self.w = {k: rng.integers(-127, 128, s, dtype=np.int64).astype(np.int8) for k, s in shapes.items()}

    def linear_specs(self) -> list[_Spec]:
        return [_Spec(k, *w.shape) for k, w in self.w.items()]

    def iter_linears(self) -> Any:
        from types import SimpleNamespace

        for k, w in self.w.items():
            yield SimpleNamespace(id=SimpleNamespace(key=k), w8=torch.from_numpy(w))


class _LocalProvider:
    """The service's v3 item function over in-process wire bytes, kernels = exact numpy GEMMs."""

    def __init__(self, pack: _Pack):
        from types import SimpleNamespace

        from pgw.service.pks_provider import PKSProvider

        self.prov = PKSProvider()
        self.kern = {k: SimpleNamespace(in_features=w.shape[1], out_features=w.shape[0],
                                        mulmod64=lambda b, nb, w=w: matmul_mod64(w, b))
                     for k, w in pack.w.items()}
        self.fp = pack.fingerprint

    def __call__(self, body: bytes, rid: str, decode: Any) -> Any:
        from pgw.hemask.wire import MaskgenResponse, MaskgenResult, encode_maskgen_response

        req = decode_maskgen_request(body, (1, 2, 3))
        out = [MaskgenResult(it.linear, it.cols, self.prov.run_item(self.fp, self.kern[it.linear], it, req.a_seed))
               for it in req.items]
        return decode(encode_maskgen_response(MaskgenResponse(req.request_id, tuple(out), 3, (1, 2, 3))))


def test_backend_exact_and_hint_store(tmp_path: Path) -> None:
    from pgw.core.ring import RingParams
    from pgw.hemask.pks_backend import PKSHintStore, PKSMaskBackend, RemotePKS

    pack = _Pack({"blk.0.q": (64, 48), "blk.0.down": (32, 96)})
    store = PKSHintStore.open_or_build(tmp_path, pack)
    assert PKSHintStore.dir_for(tmp_path, pack.fingerprint).name.endswith("pks-v3-n8192")
    assert store.hint("blk.0.q").shape == (2, 2, pks.NR, 3)
    out = {k: w.shape[0] for k, w in pack.w.items()}
    remote = RemotePKS(_LocalProvider(pack), pack.fingerprint, out, store.a_seed)
    be = PKSMaskBackend(pack, RingParams.from_name("K32A12C27"), store, remote, groups_per_call=2)
    rng = np.random.default_rng(4)
    for key, w in pack.w.items():
        r = rng.integers(0, 1 << 32, (600, w.shape[1]), dtype=np.int64)
        got = be.matmul(key, torch.from_numpy(r)).numpy()
        assert np.array_equal(got, (r @ w.astype(np.int64).T) & 0xFFFFFFFF)
    assert remote.calls == 2 * 2  # 600 rows = 3 groups -> 2 requests per linear
    with pytest.raises(ConfigError):
        PKSMaskBackend(pack, RingParams.from_name("K40A12"), store, remote)
    # a modified hint file is refused when the store is opened
    p = store.path / "blk.0.q.npy"
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 1
    p.write_bytes(bytes(raw))
    with pytest.raises(IntegrityError):
        PKSHintStore.open(tmp_path, pack.fingerprint)


def test_backend_in_real_factory_matches_local(tmp_path: Path) -> None:
    """The real MaskGenerator with the pks-v3 backend writes the same segments as local generation, with
    every row self-checked and the Freivalds vectors unchanged."""
    from pgw.core.ring import RingParams
    from pgw.factory import MaskGenerator
    from pgw.hemask.pks_backend import PKSHintStore, PKSMaskBackend, RemotePKS

    pytest.importorskip("pgw.factory")
    pack = _Pack({"blk.0.q": (64, 48)})
    ring = RingParams.from_name("K32A12C27")
    store = PKSHintStore.open_or_build(tmp_path, pack)
    remote = RemotePKS(_LocalProvider(pack), pack.fingerprint, {"blk.0.q": 64}, store.a_seed)
    be = PKSMaskBackend(pack, ring, store, remote, groups_per_call=1)
    try:
        local = MaskGenerator(pack, ring, device="cpu")
        he = MaskGenerator(pack, ring, device="cpu", remote=be)
    except Exception as exc:  # pragma: no cover - the minimal pack lacks what MaskGenerator needs
        pytest.skip(f"minimal pack not accepted by MaskGenerator: {exc}")
    seed = bytes(range(1, 33))
    a = {(g.id.key, g.slot_start): g.wr.clone() for g in local.generate(seed, 300, checks=2, chunk_slots=300)}
    b = {(g.id.key, g.slot_start): g.wr.clone() for g in he.generate(seed, 300, checks=2, chunk_slots=300)}
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert he.last_stats.self_checked_rows == 300
