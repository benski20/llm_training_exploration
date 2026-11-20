"""NTT ring products (``pgw.hemask.ntt``, native ``pgw_ntt``): bit-identical to the dense path.

Every product is checked three ways: against the dense production path (``fastmm`` int8 planes or
the float64 bit slices, whichever engine this host uses), against an independent wrapping uint64
numpy matmul, and across instruction sets. Shapes use the real parameters: rank 16 (dim 4096),
q = 2^56 (8 public-matrix planes, 6 hint planes holding bytes 1..6), 256-slot groups. Wrapper-level
tests also assert that the native NTT path actually ran (``calls``), so a silent fallback to the
dense path cannot make them pass.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pgw.hemask import DEGREE, HEParams, encrypt_masks, fastmm, lwe, ntt, public_matrix, secret_matrix

pytestmark = pytest.mark.skipif(not ntt.available(), reason="pgw_ntt not built (rust/tools/build_ntt.py)")


@pytest.fixture(autouse=True)
def _int8_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare with the production dense engine (int8 byte planes, ``cpu_int8``) when torch has exact
    CPU int8 GEMMs, whatever the speed probe says on this (shared) host. The float64 engine agrees
    on full 64-bit operands; on partial planes it differs in bit 63 only (see
    ``test_float64_engine_partial_planes``)."""
    from pgw.core.ringmm import cpu_int8_available

    if cpu_int8_available():
        monkeypatch.setenv("PGW_HEMASK_ENGINE", "cpu_int8")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Counts the native product calls (``mm_planes`` / ``mm_u64``) made while the test runs."""
    seen: Counter[str] = Counter()
    for name in ("mm_planes", "mm_u64"):
        fn = getattr(ntt.native, name)

        def spy(*args: object, _fn: object = fn, _name: str = name, **kw: object) -> object:
            seen[_name] += 1
            return _fn(*args, **kw)  # type: ignore[operator]

        monkeypatch.setattr(ntt.native, name, spy)
    return seen


RANK = 16
DIM = RANK * DEGREE
SEED = bytes(range(32))
BIT63 = np.uint64(2**63 - 1)


def _assert_dense(got: np.ndarray, planes: np.ndarray, smat: np.ndarray, first_byte: int | None = None) -> None:
    """``got`` equals the dense ``fastmm.big_planes_mm_small``. On partial planes the float64 engine
    (hosts without exact CPU int8 GEMMs) fills the missing top byte with 0x80 instead of 0, which
    changes bit 63 only (by 2^63 times the parity of the secret's column sums; see
    ``test_float64_engine_partial_planes``): bit 63 is then left out of this comparison. The tests
    also compare against the independent full 64-bit reference ``_ref``, on every host."""
    dense = fastmm.big_planes_mm_small(planes, smat, first_byte=first_byte)
    first = 8 - planes.shape[1] if first_byte is None else first_byte
    if first + planes.shape[1] < 8 and fastmm.engine() != "cpu_int8":
        got, dense = got & BIT63, dense & BIT63
    assert np.array_equal(got, dense)


def _secrets(rng: np.random.Generator, g: int, rank: int = RANK, fill: int | None = None) -> np.ndarray:
    if fill is not None:
        return np.full((g, rank, DEGREE), fill, dtype=np.int8)
    return rng.integers(-1, 2, size=(g, rank, DEGREE)).astype(np.int8)


def _smat(secrets: np.ndarray) -> np.ndarray:
    return np.concatenate([secret_matrix(s) for s in secrets], axis=1)


def _planes_u64(planes: np.ndarray, first: int) -> np.ndarray:
    """Words of byte planes holding bytes ``first .. first + P - 1`` (other bytes zero)."""
    out = np.zeros((planes.shape[0], planes.shape[2]), dtype=np.uint64)
    for d in range(planes.shape[1]):
        byte = (planes[:, d].view(np.uint8) ^ np.uint8(0x80)).astype(np.uint64)
        out |= byte << np.uint64(8 * (first + d))
    return out


def _ref(u: np.ndarray, smat: np.ndarray) -> np.ndarray:
    """Independent reference: numpy integer matmul wraps mod 2^64."""
    return np.asarray(u, dtype=np.uint64) @ smat.astype(np.int64).view(np.uint64)


# --- the exactness bound ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


def test_crt_range_covers_the_protocol_bound() -> None:
    """Proof obligations of the CRT: three NTT-friendly primes whose centred range covers every
    product the protocol can produce (|Z| <= (2^64 - 1) * 256 * rank for ternary secrets)."""
    inf = ntt.info()
    p0, p1, p2 = inf["primes"]
    for p in (p0, p1, p2):
        assert _is_prime(p) and p % 512 == 1 and 2**28 < p < 2**29
    assert p0 > p1 > p2 and p0 < 2 * p2  # the CRT's lazy reductions need r0 < 2 p1, 2 p2
    half = p0 * p1 * (p2 - 1) // 2
    assert inf["half"] == half
    assert half > 2**85.99
    # rank 16 encrypt (64-bit A) and decrypt (56-bit hint), all-max coefficients
    assert (2**64 - 1) * 256 * RANK <= half
    assert (2**56 - 1) * 256 * RANK <= half
    # the exact cut-off, enforced natively
    lmax = half // (2**64 - 1)
    assert ntt.native.exact_for(64, lmax) and not ntt.native.exact_for(64, lmax + 1)
    assert lmax // 256 > 10_000  # ternary ranks up to > 10k are exact (the protocol uses 8-16)


# --- encrypt: pad = A[:n] @ S ------------------------------------------------------------------------


@pytest.mark.parametrize("g", [1, 4])
def test_encrypt_pad_bit_identical(g: int, calls: Counter[str]) -> None:
    rng = np.random.default_rng(10 + g)
    params = HEParams(k=32, rank=RANK)
    a = public_matrix(SEED, 37, params)  # 37 rows: two full 16-row blocks + a partial one
    ap = fastmm.to_planes(a)
    smat = _smat(_secrets(rng, g))
    got = ntt.big_planes_mm_small(ap, smat)
    assert got.dtype == np.uint64 and got.shape == (37, g * DEGREE)
    assert np.array_equal(got, fastmm.big_planes_mm_small(ap, smat))
    assert np.array_equal(got, _ref(a, smat))
    assert np.array_equal(ntt.matmul_mod64(smat, a, small_left=False), got)
    assert np.array_equal(lwe.matmul_mod64(smat, a, small_left=False), got)
    assert calls == Counter(mm_planes=1, mm_u64=1)


def test_seven_plane_public_matrix(calls: Counter[str]) -> None:
    """q = 2^56 needs only bytes 0..6 of A (7 planes, first_byte=0)."""
    rng = np.random.default_rng(3)
    a = public_matrix(SEED, 20, HEParams(k=32, rank=RANK)) & np.uint64((1 << 56) - 1)
    ap7 = np.ascontiguousarray(fastmm.to_planes(a)[:, :7])
    smat = _smat(_secrets(rng, 2))
    got = ntt.big_planes_mm_small(ap7, smat, first_byte=0)
    assert calls["mm_planes"] == 1
    _assert_dense(got, ap7, smat, first_byte=0)
    assert np.array_equal(got, _ref(a, smat))


# --- decrypt: corr = H @ S (hint bytes 1..6) ---------------------------------------------------------


@pytest.mark.parametrize("g", [1, 4])
def test_decrypt_correction_bit_identical(g: int, calls: Counter[str]) -> None:
    rng = np.random.default_rng(20 + g)
    m = 33
    hint = rng.integers(-128, 128, size=(m, 6, DIM), dtype=np.int8)
    smat = _smat(_secrets(rng, g))
    got = ntt.big_planes_mm_small(hint, smat, first_byte=1)
    assert calls["mm_planes"] == 1
    _assert_dense(got, hint, smat, first_byte=1)
    assert np.array_equal(got, _ref(_planes_u64(hint, 1), smat))


def test_float64_engine_partial_planes(monkeypatch: pytest.MonkeyPatch, calls: Counter[str]) -> None:
    """The dense float64 engine (``PGW_HEMASK_ENGINE=cpu_f64``) fills planes above ``first_byte + P``
    with int8 0, i.e. byte 0x80, where the int8 engine (and the NTT) use byte 0. For the stored hint
    (bytes 1..6) that changes bit 63 of the correction only, which decryption never reads (it uses
    bits < q = 56). Full 8-plane operands agree exactly."""
    rng = np.random.default_rng(30)
    hint = rng.integers(-128, 128, size=(9, 6, DIM), dtype=np.int8)
    smat = _smat(_secrets(rng, 2))
    a = public_matrix(SEED, 9, HEParams(k=32, rank=RANK))
    ap = fastmm.to_planes(a)
    got_h = ntt.big_planes_mm_small(hint, smat, first_byte=1)
    got_a = ntt.big_planes_mm_small(ap, smat)
    assert calls["mm_planes"] == 2
    monkeypatch.setenv("PGW_HEMASK_ENGINE", "cpu_f64")
    dense_h = fastmm.big_planes_mm_small(hint, smat, first_byte=1)
    assert np.array_equal(got_h & BIT63, dense_h & BIT63)
    # the difference is exactly 2^63 times the parity of the secret's column sum
    parity = (smat.astype(np.int64).sum(0) & 1).astype(np.uint64) << np.uint64(63)
    assert np.array_equal(got_h ^ dense_h, np.broadcast_to(parity, got_h.shape))
    assert np.array_equal(got_a, fastmm.big_planes_mm_small(ap, smat))


# --- boundary values ---------------------------------------------------------------------------------


@pytest.mark.parametrize("fill", [1, -1])
def test_all_max_coefficients(fill: int, calls: Counter[str]) -> None:
    """Every coefficient at its maximum and a secret of maximal L1 norm (all +-1): the largest
    |Z| the protocol can produce at rank 16 (2^76 for A, 2^68 for the hint)."""
    smat = _smat(_secrets(np.random.default_rng(0), 2, fill=fill))
    a = np.full((17, DIM), np.uint64(2**64 - 1), dtype=np.uint64)
    ap = fastmm.to_planes(a)
    got = ntt.big_planes_mm_small(ap, smat)
    assert np.array_equal(got, _ref(a, smat))
    assert np.array_equal(got, fastmm.big_planes_mm_small(ap, smat))
    assert np.array_equal(ntt.matmul_mod64(smat, a, small_left=False), got)
    hint = np.full((5, 6, DIM), 0x7F, dtype=np.int8)  # every stored byte 0xFF
    got_h = ntt.big_planes_mm_small(hint, smat, first_byte=1)
    assert np.array_equal(got_h, _ref(_planes_u64(hint, 1), smat))
    _assert_dense(got_h, hint, smat, first_byte=1)
    assert calls == Counter(mm_planes=2, mm_u64=1)


@pytest.mark.parametrize("fill", [127, -127, 0])
def test_near_crt_limit(fill: int, calls: Counter[str]) -> None:
    """Products at the edge of the CRT range, not just the protocol's (2^76 at rank 16): rank 129 with
    every secret coefficient at +-127 (``fill=0``: random signs) and ``A = 2^64 - 1`` gives
    ``|Z_255| = (2^64 - 1) * 129 * 256 * 127 = 0.99998 HALF``, through several MAC chunks; rank 130
    is refused natively and falls back to the dense path in the wrappers."""
    rank = 129
    rng = np.random.default_rng(129)
    if fill:
        sec = np.full((1, rank, DEGREE), fill, dtype=np.int8)
    else:
        sec = (rng.integers(0, 2, size=(1, rank, DEGREE)) * 254 - 127).astype(np.int8)
    half = ntt.info()["half"]
    l1 = rank * DEGREE * 127
    assert 0.99997 * half <= (2**64 - 1) * l1 <= half
    assert ntt.exact(64, sec)
    a = rng.integers(0, 2**64, size=(3, rank * DEGREE), dtype=np.uint64)
    a[0] = np.uint64(2**64 - 1)
    smat = secret_matrix(sec[0])
    want = _ref(a, smat)
    if fill:  # Z_255 of the all-max row is (2^64 - 1) * sum(s), exactly, with Python integers
        z = (2**64 - 1) * int(sec.astype(np.int64).sum())
        assert abs(z) >= 0.99997 * half and int(want[0, DEGREE - 1]) == z % 2**64
    assert np.array_equal(ntt.matmul_mod64(smat, a, small_left=False), want)
    assert np.array_equal(ntt.big_planes_mm_small(fastmm.to_planes(a), smat), want)
    for isa in ("scalar", "avx2", "avx512"):
        assert np.array_equal(ntt.native.mm_u64(a, sec, 2, isa), want)
    assert calls == Counter(mm_u64=4, mm_planes=1)

    sec130 = np.full((1, 130, DEGREE), 127, dtype=np.int8)
    assert not ntt.exact(64, sec130)
    with pytest.raises(ValueError, match="exactness bound exceeded") as err:
        ntt.native.mm_u64(a[:1, :DEGREE].repeat(130, axis=1), sec130, 1, "auto")
    assert str(130 * DEGREE * 127) not in str(err.value)  # the secret's norm is not reported
    a130 = np.full((1, 130 * DEGREE), 2**64 - 1, dtype=np.uint64)
    smat130 = secret_matrix(sec130[0])
    assert np.array_equal(ntt.matmul_mod64(smat130, a130, small_left=False), _ref(a130, smat130))
    assert calls["mm_u64"] == 5  # only the direct native call above: the wrapper went dense


def test_zero_and_mixed_extremes(calls: Counter[str]) -> None:
    rng = np.random.default_rng(5)
    a = rng.choice(np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64), size=(19, DIM))
    s = _secrets(rng, 3)
    s[1] = 0  # an all-zero secret
    smat = _smat(s)
    got = ntt.matmul_mod64(smat, a, small_left=False)
    assert np.array_equal(got, _ref(a, smat))
    assert not got[:, DEGREE : 2 * DEGREE].any()
    assert calls["mm_u64"] == 1


# --- instruction sets, transformed operands, fallbacks -----------------------------------------------


def test_instruction_sets_agree() -> None:
    rng = np.random.default_rng(6)
    hint = rng.integers(-128, 128, size=(21, 6, DIM), dtype=np.int8)
    sec = _secrets(rng, 5)
    outs = [ntt.native.mm_planes(hint, sec, 1, 2, isa) for isa in ("scalar", "avx2", "avx512")]
    assert all(np.array_equal(outs[0], o) for o in outs[1:])
    assert np.array_equal(outs[0], _ref(_planes_u64(hint, 1), _smat(sec)))


def test_thread_count_does_not_change_bits() -> None:
    rng = np.random.default_rng(7)
    a = rng.integers(0, 2**64, size=(50, DIM), dtype=np.uint64)
    sec = _secrets(rng, 2)
    ref = ntt.native.mm_u64(a, sec, 1, "auto")
    for t in (2, 3, 8):
        assert np.array_equal(ntt.native.mm_u64(a, sec, t, "auto"), ref)


def test_transformed_operand_prefixes() -> None:
    rng = np.random.default_rng(8)
    params = HEParams(k=32, rank=RANK)
    a = public_matrix(SEED, 40, params)
    ap = fastmm.to_planes(a)
    t = ntt.TransformedOperand.from_planes(ap)
    t64 = ntt.TransformedOperand.from_u64(a)
    assert t.rows == 40 and t.rank == RANK
    assert t.nbytes == 48 * DIM * 12  # rows padded to 3 blocks of 16; 3 primes x 4 bytes per coefficient
    smat = _smat(_secrets(rng, 3))
    for n in (1, 16, 17, 40):
        ref = fastmm.big_planes_mm_small(ap[:n], smat)
        assert np.array_equal(t.mm(smat, rows=n), ref)
        assert np.array_equal(t64.mm(smat, rows=n), ref)
        assert np.array_equal(ref, _ref(a[:n], smat))
    hint = rng.integers(-128, 128, size=(9, 6, DIM), dtype=np.int8)
    th = ntt.TransformedOperand.from_planes(hint, first_byte=1)
    got_h = th.mm(smat)
    _assert_dense(got_h, hint, smat, first_byte=1)
    assert np.array_equal(got_h, _ref(_planes_u64(hint, 1), smat))


def test_non_negacyclic_small_falls_back_to_dense(calls: Counter[str]) -> None:
    rng = np.random.default_rng(9)
    small = rng.integers(-1, 2, size=(DIM, DEGREE)).astype(np.int8)  # not a secret matrix
    assert ntt.secrets_of(small) is None
    smat = _smat(_secrets(rng, 1))
    tampered = smat.copy()
    tampered[300, 7] = -tampered[300, 7] if tampered[300, 7] else 1
    assert ntt.secrets_of(tampered) is None
    a = public_matrix(SEED, 8, HEParams(k=32, rank=RANK))
    ap = fastmm.to_planes(a)
    assert np.array_equal(ntt.big_planes_mm_small(ap, small), fastmm.big_planes_mm_small(ap, small))
    assert np.array_equal(ntt.matmul_mod64(small, a, small_left=False), _ref(a, small))
    w = rng.integers(-127, 128, size=(5, 8))  # small_left=True (hint build) stays dense
    x = rng.integers(0, 2**64, size=(8, 3), dtype=np.uint64)
    assert np.array_equal(ntt.matmul_mod64(w, x), lwe.matmul_mod64(w, x))
    assert not calls


def test_disabled_module_uses_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ntt, "native", None)
    rng = np.random.default_rng(11)
    smat = _smat(_secrets(rng, 1))
    hint = rng.integers(-128, 128, size=(4, 6, DIM), dtype=np.int8)
    assert ntt.secrets_of(smat) is None and not ntt.available()
    assert np.array_equal(
        ntt.big_planes_mm_small(hint, smat, first_byte=1), fastmm.big_planes_mm_small(hint, smat, first_byte=1)
    )


def test_secrets_roundtrip() -> None:
    rng = np.random.default_rng(12)
    sec = _secrets(rng, 3)
    got = ntt.secrets_of(_smat(sec))
    assert got is not None and np.array_equal(got, sec)
    assert ntt.secrets_of(_smat(sec).astype(np.int64)) is not None
    fortran = np.asfortranarray(_smat(sec))
    got_f = ntt.secrets_of(fortran)
    assert got_f is not None and np.array_equal(got_f, sec)


def test_wrappers_leave_caller_arrays_intact(calls: Counter[str]) -> None:
    """The wrappers zero their own copies of the secret, never the caller's arrays."""
    rng = np.random.default_rng(14)
    sec = _secrets(rng, 2)
    smat = _smat(sec)
    keep = (sec.copy(), smat.copy())
    a = public_matrix(SEED, 5, HEParams(k=32, rank=RANK))
    ref = _ref(a, smat)
    assert np.array_equal(ntt.big_planes_mm_small(fastmm.to_planes(a), smat), ref)
    assert np.array_equal(ntt.matmul_mod64(smat.astype(np.int64), a, small_left=False), ref)
    assert np.array_equal(ntt.matmul_mod64(np.asfortranarray(smat), a, small_left=False), ref)
    t = ntt.TransformedOperand.from_u64(a)
    assert np.array_equal(t.mm(smat), ref)
    assert np.array_equal(t.mm_secrets(sec), ref)
    assert np.array_equal(t.mm_secrets(sec.astype(np.int16)), ref)
    assert ntt.exact(64, sec)
    assert np.array_equal(sec, keep[0]) and np.array_equal(smat, keep[1])
    assert calls == Counter(mm_planes=1, mm_u64=2)


def test_threads_are_shared_among_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_NTT_THREADS", raising=False)
    monkeypatch.setattr(ntt, "_cores", lambda: 4)
    with ntt._threads() as t1:
        with ntt._threads() as t2:
            with ntt._threads() as t3, ntt._threads() as t4, ntt._threads() as t5:
                assert (t3, t4, t5) == (1, 1, 1)
        assert (t1, t2) == (4, 2)
        with ntt._threads() as t2b:
            assert t2b == 2
    assert ntt._inflight == 0
    monkeypatch.setenv("PGW_NTT_THREADS", "3")
    with ntt._threads() as t:
        assert t == 3 and ntt._inflight == 0


# --- end to end: encryption + decryption with the NTT products ---------------------------------------


def test_mlwe_roundtrip_with_ntt_products(monkeypatch: pytest.MonkeyPatch, calls: Counter[str]) -> None:
    """Encryption (``lwe.encrypt_masks`` with A planes) and the factory backend's decryption
    (``HEMaskBackend._decrypt``: 6 hint planes, bytes 1..6) at K32, rank 16, q = 2^56, with the NTT
    products swapped in exactly where the hooks go (``fastmm.big_planes_mm_small`` for
    ``encrypt_masks``, ``backend.big_planes_mm_small`` for ``_decrypt``): ciphertexts and decrypted
    ``W8 r`` equal the dense run bit for bit, and the exact ``W8 r mod 2^32``."""
    from types import SimpleNamespace

    import pgw.hemask.backend as backend_mod
    import pgw.hemask.lwe as lwe_mod
    from pgw.core.ring import RING_K32_A12_C27

    rng = np.random.default_rng(13)
    params = HEParams(k=32, rank=RANK)
    m, n, rows = 24, 64, 300  # two 256-slot groups (the second one partial)
    w = rng.integers(-127, 128, size=(m, n))
    r = rng.integers(0, 1 << 32, size=(rows, n), dtype=np.int64)
    a = public_matrix(SEED, n, params)
    ap = fastmm.to_planes(a)
    hint = np.ascontiguousarray(fastmm.to_planes(lwe.matmul_mod64(w, a))[:, 1:7])  # HintStore v2 layout
    ring = RING_K32_A12_C27
    fake = SimpleNamespace(params=params, ring=ring, hints=SimpleNamespace(hint=lambda _key: hint))

    results = []
    for use_ntt in (False, True):
        seeds = iter([bytes([i]) * 32 for i in range(4)])
        monkeypatch.setattr(lwe_mod, "new_seed", seeds.__next__)  # same secrets and noise in both runs
        if use_ntt:
            monkeypatch.setattr(fastmm, "big_planes_mm_small", ntt.big_planes_mm_small)
            monkeypatch.setattr(backend_mod, "big_planes_mm_small", ntt.big_planes_mm_small)
        cts = [encrypt_masks(r[g0 : g0 + DEGREE], None, params, a_planes=ap) for g0 in range(0, rows, DEGREE)]
        assert calls["mm_planes"] == (2 if use_ntt else 0)  # one encryption per 256-slot group
        b = np.concatenate([ct.b for ct in cts], axis=1)
        y = lwe.matmul_mod64(w, b) & params.q_mask  # the provider's product
        wr = backend_mod.HEMaskBackend._decrypt(fake, "x", y, cts, m)  # type: ignore[arg-type]
        assert calls["mm_planes"] == (3 if use_ntt else 0)  # one decryption correction
        results.append((b, wr))
    (b0, wr0), (b1, wr1) = results
    assert np.array_equal(b0, b1)
    assert np.array_equal(wr0, wr1)
    ref = (r.astype(object) @ w.T.astype(object)) % (1 << 32)
    assert np.array_equal(wr1.astype(object), ref)
