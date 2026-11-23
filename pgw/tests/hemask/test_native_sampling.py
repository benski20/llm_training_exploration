"""Native HE customer path (``pgw_fast.he_*``) and the reworked backend are bit-identical to the old code.

The ``_old_*`` functions below are frozen copies of the Python implementation before the native
path (``encrypt_masks`` / ``HEMaskBackend._decrypt`` with 8 public-matrix planes, Python ChaCha
sampling, one decryption per reply). Every new configuration (native or not, ``cpu_int8`` or
``cpu_f64``, 7 or 8 planes, several requests in flight, several replies per hint pass) must give the
same ciphertexts for the same seed and the same plaintexts ``W8 r`` for the same ciphertexts.
"""

from __future__ import annotations

import threading
import types
from typing import Any

import numpy as np
import pytest
import torch

import pgw.core.native as native
from pgw.core.prng import SeedExpander
from pgw.core.ring import RingParams
from pgw.hemask import DEGREE, HEParams, encrypt_masks, matmul_mod64, public_matrix, secret_matrix
from pgw.hemask.backend import HEMaskBackend, HintStore, verify_rows
from pgw.hemask.fastmm import big_planes_mm_small, to_planes
from pgw.hemask.lwe import a_plane_count, sample_noise, sample_secret

_U64 = np.uint64
A_SEED = bytes(range(32))
NATIVE = native.native_available("he_ternary", "he_secret_matrix", "he_cbd", "he_encrypt_combine",
                                 "he_decrypt_combine", "he_check_rows")
needs_native = pytest.mark.skipif(not NATIVE, reason="pgw_fast without the he_* kernels (build rust/tools/build_fast.py)")


# ---- frozen copies of the previous implementation ------------------------------------------------
def _old_cbd(sx: SeedExpander, shape: tuple[int, ...], eta: int, label: str) -> np.ndarray:
    n = int(np.prod(shape))
    raw = np.frombuffer(sx.random_bytes(8 * n, label), dtype="<u8")
    mask = np.uint64((1 << eta) - 1)
    pos = np.bitwise_count(raw & mask).astype(np.int64)
    neg = np.bitwise_count((raw >> np.uint64(eta)) & mask).astype(np.int64)
    return (pos - neg).reshape(shape)


def _old_ternary(sx: SeedExpander, shape: tuple[int, ...], label: str) -> np.ndarray:
    n = int(np.prod(shape))
    out = np.empty(0, dtype=np.int64)
    ctr = 0
    while out.size < n:
        raw = np.frombuffer(sx.random_bytes(2 * (n - out.size) + 64, label, ctr), dtype=np.uint8)
        ctr += 1
        keep = raw[raw < 252].astype(np.int64) % 3 - 1
        out = np.concatenate([out, keep])
    return out[:n].reshape(shape)


def _old_encrypt(r: np.ndarray, a: np.ndarray, params: HEParams, seed: bytes) -> tuple[np.ndarray, np.ndarray]:
    """``encrypt_masks`` before the native path, with ``SeedExpander(new_seed())`` replaced by ``seed``."""
    rows, n = r.shape
    sx = SeedExpander(seed)
    s = _old_ternary(sx, (params.rank, DEGREE), "s")
    smat = secret_matrix(s)
    e = _old_cbd(sx, (n, DEGREE), params.eta, "e")
    msg = np.zeros((n, DEGREE), dtype=_U64)
    msg[:, :rows] = np.asarray(r, dtype=np.int64).astype(_U64).T
    pad = matmul_mod64(smat, a, small_left=False, fast=False)  # exact float64 reference
    b = (pad + e.astype(_U64) + (msg << _U64(params.delta_bits))) & params.q_mask
    return b, smat


def _old_decrypt(y: np.ndarray, hint: np.ndarray, secrets: list[np.ndarray], rows: list[int], params: HEParams,
                 ring_mask: int) -> np.ndarray:
    """``HEMaskBackend._decrypt`` before the native path (one reply)."""
    smat = np.concatenate(secrets, axis=1)
    corr = big_planes_mm_small(hint, smat, first_byte=1)
    v = y - corr
    half = np.uint64(1 << (params.delta_bits - 1))
    wr = ((v + half) >> np.uint64(params.delta_bits)) & np.uint64(ring_mask)
    return np.concatenate([wr[:, gi * DEGREE:gi * DEGREE + rw] for gi, rw in enumerate(rows)], axis=1).T


# ---- helpers -------------------------------------------------------------------------------------
@pytest.fixture(params=["cpu_int8", "cpu_f64"])
def engine_name(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("PGW_HEMASK_ENGINE", request.param)
    return str(request.param)


@pytest.fixture(params=[True, False], ids=["native", "python"])
def use_native(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> bool:
    if request.param and not NATIVE:
        pytest.skip("pgw_fast without the he_* kernels")
    if not request.param:
        monkeypatch.setattr(native, "fast", None)
    return bool(request.param)


def _ring_for(q_bits: int) -> RingParams:
    return RingParams.from_name("K32A12C27" if q_bits == 56 else "K40A12")


class _Spec:
    def __init__(self, key: str, n: int, m: int):
        self.id = types.SimpleNamespace(key=key)
        self.in_features, self.out_features = n, m


class _Pack:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.weights = weights

    def linear_specs(self) -> list[_Spec]:
        return [_Spec(k, w.shape[1], w.shape[0]) for k, w in self.weights.items()]


class _Hints(HintStore):
    """In-memory hint store (``W8 A`` planes 1 .. q/8-1, exactly what :meth:`HintStore.build` writes)."""

    def __init__(self, weights: dict[str, np.ndarray], q_bits: int, rank: int = 8):
        params = HEParams(k=16, rank=rank, q_bits=q_bits)
        planes = q_bits // 8 - 1
        a = public_matrix(A_SEED, max(w.shape[1] for w in weights.values()), params)
        super().__init__(None, {"a_seed": A_SEED.hex(), "rank": rank, "q_bits": q_bits,  # type: ignore[arg-type]
                                "row_norms": {k: float(np.sqrt((w.astype(np.float64) ** 2).sum(1)).max())
                                              for k, w in weights.items()}})
        self.h = {k: np.ascontiguousarray(to_planes(matmul_mod64(w, a[:w.shape[1]], fast=False))[:, 1:1 + planes])
                  for k, w in weights.items()}

    def hint(self, linear: str) -> np.ndarray:
        return self.h[linear]


class _Provider:
    """``W8 b mod 2^64`` with the wire's low-bit drop (``unpack_y(pack_y(y))``).

    With ``pair_s``, a call waits (up to that long) for a second concurrent call, so a client that
    really keeps several requests in flight is seen doing so regardless of machine speed.
    """

    def __init__(self, weights: dict[str, np.ndarray], q_bits: int, pair_s: float = 0.0):
        self.w, self.q_bits, self.pair_s = weights, q_bits, pair_s
        self.cond = threading.Condition()
        self.active = self.peak = self.calls = 0

    def __call__(self, linear: str, b: np.ndarray) -> np.ndarray:
        with self.cond:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
            self.cond.notify_all()
            if self.pair_s:
                self.cond.wait_for(lambda: self.active >= 2, timeout=self.pair_s)
        try:
            y = matmul_mod64(self.w[linear], b, fast=False)
            qm = np.uint64((1 << self.q_bits) - 1) if self.q_bits < 64 else np.uint64(2**64 - 1)
            drop = np.uint64(HEParams.DROP_BITS)
            return ((y & qm) >> drop) << drop
        finally:
            with self.cond:
                self.active -= 1


def _ref_wr(w: np.ndarray, r: np.ndarray, k: int) -> np.ndarray:
    return np.array((r.astype(object) @ w.T.astype(object)) % (1 << k), dtype=np.int64)


# ---- sampling ------------------------------------------------------------------------------------
@needs_native
@pytest.mark.parametrize("count", [0, 1, 255, 256 * 8, 256 * 16, 100_003])
def test_native_ternary_matches_python(count: int) -> None:
    for seed in (bytes(32), bytes(range(32)), bytes([0xA5]) * 32):
        ref = _old_ternary(SeedExpander(seed), (count,), "s")
        got = native.fast.he_ternary(seed, count)
        assert got.dtype == np.int8 and np.array_equal(got.astype(np.int64), ref)


@needs_native
@pytest.mark.parametrize("draw", [1, 7, 64, 300])
@pytest.mark.parametrize("count", [300, 1000])
def test_native_ternary_later_rounds_match_python(draw: int, count: int) -> None:
    """Rounds ``ctr >= 1`` (stream labels ``("s", 1)``, ...) with a forced short draw per round."""
    if not hasattr(native.fast, "_he_ternary_fixed_draw"):
        pytest.skip("pgw_fast without the ternary test hook")
    seed = bytes(range(9, 41))
    sx = SeedExpander(seed)
    ref = np.empty(0, dtype=np.int64)
    ctr = 0
    while ref.size < count:
        raw = np.frombuffer(sx.random_bytes(draw, "s", ctr), dtype=np.uint8)
        ctr += 1
        ref = np.concatenate([ref, raw[raw < 252].astype(np.int64) % 3 - 1])
    assert ctr > 1  # several rounds really ran
    got = native.fast._he_ternary_fixed_draw(seed, count, draw)
    assert got.dtype == np.int8 and np.array_equal(got.astype(np.int64), ref[:count])


@needs_native
@pytest.mark.parametrize("rank", [8, 16])
def test_native_secret_matrix_matches_python(rank: int) -> None:
    seed = bytes(range(3, 35))
    ref = secret_matrix(_old_ternary(SeedExpander(seed), (rank, DEGREE), "s"))
    got = native.fast.he_secret_matrix(seed, rank)
    assert got.dtype == np.int8 and got.shape == (rank * DEGREE, DEGREE) and np.array_equal(got, ref)
    assert np.array_equal(sample_secret(seed, rank), ref)


@needs_native
@pytest.mark.parametrize("eta", [1, 2, 21, 32])
@pytest.mark.parametrize("rows", [1, 37, 3000])
def test_native_cbd_matches_python(eta: int, rows: int) -> None:
    seed = bytes(range(7, 39))
    ref = _old_cbd(SeedExpander(seed), (rows, DEGREE), eta, "e")
    got = native.fast.he_cbd(seed, rows, DEGREE, eta)
    assert got.dtype == np.int64 and np.array_equal(got, ref)
    assert np.array_equal(sample_noise(seed, rows, eta), ref)


@needs_native
def test_native_sampling_distributions() -> None:
    """Moments and chi-square of the native samplers (CBD(21): mean 0, variance 10.5; ternary uniform)."""
    seed = bytes(range(11, 43))
    e = native.fast.he_cbd(seed, 4096, DEGREE, 21).ravel().astype(np.float64)  # 1M samples
    assert abs(e.mean()) < 0.02 and abs(e.var() - 10.5) < 0.1 and np.abs(e).max() <= 21
    # chi-square of the CBD histogram against the exact binomial-difference law
    from math import comb

    vals = np.arange(-21, 22)
    p = np.array([sum(comb(21, i) * comb(21, i - v) for i in range(22) if 0 <= i - v <= 21) for v in vals]) / 4.0**21
    hist = np.bincount((e + 21).astype(np.int64), minlength=43)
    keep = p * e.size >= 20
    chi2 = float((((hist[keep] - p[keep] * e.size) ** 2) / (p[keep] * e.size)).sum())
    assert chi2 < 3 * keep.sum()  # df ~ 30: far below a failing value, far above a biased sampler's
    t = native.fast.he_ternary(seed, 300_000).astype(np.int64)
    counts = np.bincount(t + 1, minlength=3)
    chi2_t = float(((counts - t.size / 3) ** 2 / (t.size / 3)).sum())
    assert chi2_t < 13.8  # df 2, p = 0.001


@needs_native
def test_native_samples_differ_per_seed() -> None:
    a = native.fast.he_secret_matrix(bytes(32), 8)
    b = native.fast.he_secret_matrix(bytes(31) + b"\x01", 8)
    assert not np.array_equal(a, b)


# ---- encryption ----------------------------------------------------------------------------------
@pytest.mark.parametrize("q_bits", [56, 64])
@pytest.mark.parametrize("rows", [DEGREE, 37])
def test_encrypt_bit_identical_to_old_code(engine_name: str, use_native: bool, q_bits: int, rows: int) -> None:
    k = 32 if q_bits == 56 else 40
    params = HEParams(k=k, rank=8, q_bits=q_bits)
    n = 300
    a = public_matrix(A_SEED, n, params)
    r = np.random.default_rng(q_bits + rows).integers(0, 1 << k, size=(rows, n), dtype=np.int64)
    for seed in (bytes(range(32)), bytes([0x5C]) * 32):
        ref_b, ref_s = _old_encrypt(r, a, params, seed)
        planes8 = to_planes(a)
        for kw in ({"a_planes": planes8}, {"a_planes": np.ascontiguousarray(planes8[:, :a_plane_count(params)])},
                   {}):
            ct = encrypt_masks(r, None if kw else a, params, _seed=seed, **kw)
            assert np.array_equal(ct.b, ref_b), (engine_name, use_native, list(kw))
            assert np.array_equal(ct.secret, ref_s)
            assert ct.rows == rows


def test_seven_planes_only_at_q56() -> None:
    p56, p64 = HEParams(k=32, rank=8), HEParams(k=40, rank=8)
    assert (a_plane_count(p56), a_plane_count(p64)) == (7, 8)
    a = public_matrix(A_SEED, 16, p64)
    with pytest.raises(Exception, match="shape"):
        encrypt_masks(np.zeros((4, 16), dtype=np.int64), None, p64, a_planes=to_planes(a)[:, :7])


# ---- decryption ----------------------------------------------------------------------------------
@pytest.mark.parametrize("q_bits", [56, 64])
def test_decrypt_bit_identical_to_old_code(engine_name: str, use_native: bool, q_bits: int) -> None:
    rng = np.random.default_rng(q_bits)
    weights = {"lin": rng.integers(-127, 128, size=(70, 200)).astype(np.int8)}
    ring = _ring_for(q_bits)
    hints = _Hints(weights, q_bits)
    provider = _Provider(weights, q_bits)
    be = HEMaskBackend(_Pack(weights), ring, hints, provider)
    rows = [DEGREE, 100, DEGREE, 7]  # groups of three replies: (256, 100), (256,), (7,)
    r = rng.integers(0, 1 << ring.k, size=(sum(rows), 200), dtype=np.int64)
    cts = be._encrypt(r[:356], 200) + be._encrypt(r[356:612], 200) + be._encrypt(r[612:], 200)
    assert [ct.rows for ct in cts] == rows
    replies = [cts[0:2], cts[2:3], cts[3:4]]
    ys = [provider("lin", np.concatenate([ct.b for ct in g], axis=1)) for g in replies]
    olds = [_old_decrypt(y, hints.h["lin"], [ct.secret for ct in g], [ct.rows for ct in g], be.params, ring.mask)
            for y, g in zip(ys, replies, strict=True)]
    ref = _ref_wr(weights["lin"].astype(np.int64), r, ring.k)
    assert np.array_equal(np.concatenate(olds), ref)  # the old code was exact on these inputs
    # one reply per call, as before
    for y, g, old in zip(ys, replies, olds, strict=True):
        assert np.array_equal(be._decrypt("lin", y, g, 70), old)
    # all three replies in one hint pass, into a shared output at slot offsets
    out = np.full((sum(rows), 70), -1, dtype=np.int64)
    be._decrypt_into("lin", [(ys[0], replies[0], 0), (ys[1], replies[1], 356), (ys[2], replies[2], 612)], out, 70)
    assert np.array_equal(out, ref)


# ---- whole backend: pipelining and passes --------------------------------------------------------
@pytest.mark.parametrize("q_bits", [56, 64])
@pytest.mark.parametrize("cfg", [(1, 1, 1), (1, 2, 16), (2, 4, 16), (1, 4, 2), (4, 2, 3)],
                         ids=lambda c: f"gpc{c[0]}-inflight{c[1]}-pass{c[2]}")
def test_backend_matmul_exact_for_every_schedule(engine_name: str, use_native: bool, q_bits: int,
                                                 cfg: tuple[int, int, int]) -> None:
    gpc, inflight, per_pass = cfg
    rng = np.random.default_rng(sum(cfg) + q_bits)
    weights = {"a": rng.integers(-127, 128, size=(40, 96)).astype(np.int8),
               "b": np.where(rng.random((24, 160)) < 0.5, -127, 127).astype(np.int8)}
    ring = _ring_for(q_bits)
    provider = _Provider(weights, q_bits, pair_s=1.0 if inflight > 1 else 0.0)
    be = HEMaskBackend(_Pack(weights), ring, _Hints(weights, q_bits), provider, groups_per_call=gpc,
                       max_inflight=inflight, groups_per_pass=per_pass)
    assert be._a.shape[1] == a_plane_count(be.params) or engine_name == "cpu_f64"
    for key, rows in (("a", 5 * DEGREE + 3), ("b", 77)):
        w = weights[key]
        r = rng.integers(0, 1 << ring.k, size=(rows, w.shape[1]), dtype=np.int64)
        got = be.matmul(key, torch.from_numpy(r))
        assert got.dtype == torch.int64 and np.array_equal(got.numpy(), _ref_wr(w.astype(np.int64), r, ring.k))
    groups = 6 + 1
    assert be.stats["groups"] == groups
    assert provider.peak <= inflight
    if inflight > 1:
        assert provider.peak > 1 and be.stats["max_inflight"] == min(inflight, -(-6 // gpc))
    requests = -(-6 // gpc) + 1
    assert be.stats["passes"] == -(-6 // per_pass) + 1  # full passes of exactly per_pass groups, then the rest
    assert provider.calls == requests


@pytest.mark.parametrize(("gpc", "per_pass"), [(12, 16), (5, 3), (16, 16)])
def test_backend_pass_size_is_a_hard_cap(use_native: bool, gpc: int, per_pass: int) -> None:
    """Replies are split across passes: no pass holds more than ``groups_per_pass`` groups."""
    rng = np.random.default_rng(gpc * per_pass)
    weights = {"x": rng.integers(-127, 128, size=(8, 16)).astype(np.int8)}
    be = HEMaskBackend(_Pack(weights), _ring_for(56), _Hints(weights, 56), _Provider(weights, 56),
                       groups_per_call=gpc, max_inflight=2, groups_per_pass=per_pass)
    sizes: list[int] = []
    orig = be._decrypt_into

    def spy(linear: str, items: Any, out: np.ndarray, m: int) -> None:
        sizes.append(sum(len(cts) for _, cts, _ in items))
        orig(linear, items, out, m)

    be._decrypt_into = spy  # type: ignore[method-assign]
    r = rng.integers(0, 1 << 32, size=(30 * DEGREE - 5, 16), dtype=np.int64)
    got = be.matmul("x", torch.from_numpy(r))
    assert np.array_equal(got.numpy(), _ref_wr(weights["x"].astype(np.int64), r, 32))
    assert sum(sizes) == 30 and max(sizes) <= per_pass
    assert sizes == [per_pass] * (30 // per_pass) + ([30 % per_pass] if 30 % per_pass else [])


def test_backend_one_inflight_still_overlaps_encryption() -> None:
    """``max_inflight=1``: batch ``k + 1`` is encrypted while request ``k`` is with the provider."""
    weights = {"x": np.ones((8, 16), dtype=np.int8)}
    encrypted = threading.Semaphore(0)
    overlapped: list[bool] = []
    inner = _Provider(weights, 56)

    def provider(linear: str, b: np.ndarray) -> np.ndarray:
        overlapped.append(encrypted.acquire(timeout=5.0))  # the next batch is encrypted meanwhile
        return inner(linear, b)

    be = HEMaskBackend(_Pack(weights), _ring_for(56), _Hints(weights, 56), provider, groups_per_call=1,
                       max_inflight=1)
    orig = be._encrypt
    calls = [0]

    def spy(rn: np.ndarray, n: int) -> Any:
        cts = orig(rn, n)
        calls[0] += 1
        if calls[0] > 1:
            encrypted.release()
        if calls[0] == 4:  # the last request has no next batch to wait for
            encrypted.release()
        return cts

    be._encrypt = spy  # type: ignore[method-assign]
    r = np.random.default_rng(1).integers(0, 1 << 32, size=(4 * DEGREE, 16), dtype=np.int64)
    got = be.matmul("x", torch.from_numpy(r))
    assert np.array_equal(got.numpy(), _ref_wr(weights["x"].astype(np.int64), r, 32))
    assert overlapped[:3] == [True, True, True] and inner.peak == 1


def test_backend_pass_budget_caps_groups() -> None:
    weights = {"x": np.ones((8, 16), dtype=np.int8)}
    be = HEMaskBackend(_Pack(weights), _ring_for(56), _Hints(weights, 56), _Provider(weights, 56),
                       groups_per_pass=16, pass_bytes=3 * 8 * DEGREE * 8)
    assert be._pass_groups(8) == 3 and be._pass_groups(10**9) == 1


def test_backend_wipes_secrets_on_provider_error() -> None:
    weights = {"x": np.ones((8, 16), dtype=np.int8)}
    seen: list[Any] = []

    def boom(linear: str, b: np.ndarray) -> np.ndarray:
        raise RuntimeError("provider down")

    be = HEMaskBackend(_Pack(weights), _ring_for(56), _Hints(weights, 56), boom, groups_per_call=1, max_inflight=3)
    orig = be._encrypt

    def spy(rn: np.ndarray, n: int) -> Any:
        cts = orig(rn, n)
        seen.extend(cts)
        return cts

    be._encrypt = spy  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        be.matmul("x", torch.zeros((3 * DEGREE, 16), dtype=torch.int64))
    assert seen and all(not ct.secret.any() for ct in seen)


# ---- row check -----------------------------------------------------------------------------------
@pytest.mark.parametrize("k", [32, 40])
def test_verify_rows_native_matches_numpy(use_native: bool, k: int) -> None:
    rng = np.random.default_rng(k)
    m, n, rows = 50, 70, 300
    w = rng.integers(-127, 128, size=(m, n))
    r = rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64)
    wr = _ref_wr(w, r, k)
    u = rng.integers(0, 1 << k, size=(2, m), dtype=np.int64)
    uw = np.array((u.astype(object) @ w.astype(object)) % (1 << k), dtype=np.int64)
    cu, cw = u.T.astype(np.uint64), uw.T.astype(np.uint64)
    mask = (1 << k) - 1
    assert not verify_rows(r, wr, cu, cw, mask).any()
    bad = wr.copy()
    bad[[3, 150, 299], [0, 10, 49]] ^= 1 << 5
    flags = verify_rows(r, bad, cu, cw, mask)
    assert flags.dtype == bool and set(np.flatnonzero(flags)) == {3, 150, 299}
    # the generator's numpy expression gives the same flags
    lhs = (bad.astype(np.uint64) @ cu) & np.uint64(mask)
    rhs = (r.astype(np.uint64) @ cw) & np.uint64(mask)
    assert np.array_equal(flags, (lhs != rhs).any(axis=1))


@pytest.mark.parametrize("c", [1, 2, 7, 8, 9, 14, 16, 17])
@pytest.mark.parametrize("k", [24, 32, 40])
def test_check_rows_simd_path_matches_wide_and_numpy(c: int, k: int) -> None:
    """The 32-bit SIMD row check (K <= 32, up to 16 vectors: the 40-bit self-check of encrypted factories)
    gives exactly the uint64 path's and numpy's flags, including high-valuation errors and row tails."""
    from pgw.core.native import fast, native_available

    if not native_available("he_check_rows"):
        pytest.skip("pgw_fast not built")
    try:
        fast.he_check_rows(np.zeros((1, 1), np.int64), np.zeros((1, 1), np.int64), np.zeros((1, 1), np.uint64),
                           np.zeros((1, 1), np.uint64), 1, True)
    except TypeError:
        pytest.skip("pgw_fast predates the 32-bit check path (rebuild: rust/tools/build_fast.py)")
    rng = np.random.default_rng(c * 100 + k)
    m, n, rows = 37, 45, 23  # rows not a multiple of the 4-row block
    mask = (1 << k) - 1
    w = rng.integers(-127, 128, size=(m, n))
    r = rng.integers(0, 1 << k, size=(rows, n), dtype=np.int64)
    wr = _ref_wr(w, r, k)
    u = rng.integers(0, 1 << k, size=(c, m), dtype=np.int64)
    uw = np.array((u.astype(object) @ w.astype(object)) % (1 << k), dtype=np.int64)
    cu, cw = np.ascontiguousarray(u.T.astype(np.uint64)), np.ascontiguousarray(uw.T.astype(np.uint64))
    bad = wr.copy()
    for t, (i, e) in enumerate([(0, 1), (5, 1 << (k - 1)), (36, 1 << (k - 3)), (2, 3 << (k - 2))]):
        bad[[t, rows - 1 - t][t % 2], i] = (bad[[t, rows - 1 - t][t % 2], i] + e) & mask
    lhs = (bad.astype(np.uint64) @ cu) & np.uint64(mask)
    rhs = (r.astype(np.uint64) @ cw) & np.uint64(mask)
    want = (lhs != rhs).any(axis=1)
    for wide in (False, True):
        got = fast.he_check_rows(r, bad, cu, cw, mask, wide).astype(bool)
        assert np.array_equal(got, want), wide
        assert not fast.he_check_rows(r, wr, cu, cw, mask, wide).any()
