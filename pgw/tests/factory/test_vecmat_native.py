"""The factory's native check-vector kernels (``pgw_pks.vecmat_wrap32`` / ``vecmat_p31`` / ``check_rows32``) give
exactly the integers / flags of the reference paths they replace (``ring_vecmat``'s int8 / float64 paths,
``prime_precompute``, ``verify_rows``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.core.freivalds import P31, prime_check_vectors, prime_precompute
from pgw.core.ring import RingParams
from pgw.factory import kernels
from pgw.factory.generator import check_flags
from pgw.factory.kernels import prime_vecmat, ring_vecmat
from pgw.hemask.backend import verify_rows

pgw_pks = pytest.importorskip("pgw_pks")
if not hasattr(pgw_pks, "vecmat_wrap32"):  # pragma: no cover - an older build
    pytest.skip("pgw_pks without the vecmat kernels", allow_module_level=True)

K32 = RingParams.from_name("K32A12C27")
SHAPES = [(1, 1), (3, 17), (16, 16), (300, 33), (257, 513), (1152, 896), (896, 4864)]


def _weight(rng: np.random.Generator, m: int, n: int) -> torch.Tensor:
    w = rng.integers(-128, 128, (m, n), dtype=np.int64).astype(np.int8)
    w.flat[:: 7] = -128  # extremes
    w.flat[3:: 11] = 127
    return torch.from_numpy(w)


@pytest.mark.parametrize(("m", "n"), SHAPES)
@pytest.mark.parametrize("c", [1, 2, 14, 17])
def test_ring_vecmat_native_equals_reference(m: int, n: int, c: int, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(m * 1000 + n + c)
    w = _weight(rng, m, n)
    u = torch.from_numpy(rng.integers(0, 1 << 32, (c, m), dtype=np.int64))
    u[0, 0] = (1 << 32) - 1
    got = ring_vecmat(u, w, K32)
    assert kernels._native_vecmat() is not None
    want = ring_vecmat(u, w, K32, method="f64")
    assert got.dtype == torch.int64 and torch.equal(got, want)
    exact = (u.numpy().astype(object) @ w.numpy().astype(object)) % (1 << 32)
    assert np.array_equal(got.numpy(), exact.astype(np.int64))
    monkeypatch.setenv("PGW_VECMAT_NATIVE", "0")  # the auto path without the kernel: same integers
    assert torch.equal(ring_vecmat(u, w, K32), got)


def test_ring_vecmat_native_threads_identical() -> None:
    rng = np.random.default_rng(5)
    w = _weight(rng, 700, 1100)
    x = rng.integers(0, 1 << 32, (14, 700), dtype=np.uint32)
    outs = [pgw_pks.vecmat_wrap32(x, w.numpy(), t) for t in (1, 2, 3, 8)]
    assert all(np.array_equal(outs[0], o) for o in outs[1:])


@pytest.mark.parametrize(("m", "n"), [*SHAPES, (4096, 40)])
@pytest.mark.parametrize("c", [1, 3, 9])
def test_prime_vecmat_equals_prime_precompute(m: int, n: int, c: int, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(m + 7 * n + c)
    w = _weight(rng, m, n)
    u = prime_check_vectors(bytes(range(32)), c, m, "0.qkv")
    u[0, : min(m, 300)] = P31 - 1  # the largest limbs over more rows than one 32-bit block
    got = prime_vecmat(u, w)
    want = prime_precompute(u, w)
    assert got.dtype == torch.int64 and torch.equal(got, want)
    monkeypatch.setenv("PGW_NATIVE", "0")
    assert torch.equal(prime_vecmat(u, w), want)


def test_prime_vecmat_refuses_out_of_field_vectors() -> None:
    w = torch.zeros((4, 4), dtype=torch.int8)
    with pytest.raises(Exception, match="2\\^31"):
        prime_vecmat(torch.full((1, 4), P31, dtype=torch.int64), w)


@pytest.mark.parametrize(("rows", "m", "n", "c"), [(1, 3, 5, 1), (64, 40, 17, 14), (97, 300, 64, 16), (200, 96, 33, 7)])
@pytest.mark.parametrize("form", ["int64", "uint32", "int32"])
def test_check_flags_native_equals_verify_rows(rows: int, m: int, n: int, c: int, form: str) -> None:
    rng = np.random.default_rng(rows + m + n + c)
    w = rng.integers(-128, 128, (m, n), dtype=np.int64)
    r = rng.integers(0, 1 << 32, (rows, n), dtype=np.int64)
    wr = (r @ w.T) & 0xFFFFFFFF
    v = rng.integers(0, 1 << 32, (c, m), dtype=np.int64)
    vw = (v @ w) & 0xFFFFFFFF
    bad_rows = rng.choice(rows, size=max(1, rows // 4), replace=False)
    for k, t in enumerate(bad_rows):  # errors of every valuation, including 2^31 in one entry
        wr[t, k % m] = (wr[t, k % m] + (1 << (k % 32))) & 0xFFFFFFFF
    cu, cw = np.ascontiguousarray(v.T.astype(np.uint32)), np.ascontiguousarray(vw.T.astype(np.uint32))
    w_in = {"int64": wr, "uint32": wr.astype(np.uint32), "int32": wr.astype(np.uint32).view(np.int32)}[form]
    got = check_flags(r, w_in, cu, cw, K32.mask)
    want = verify_rows(r, wr, cu.astype(np.uint64), cw.astype(np.uint64), K32.mask)
    assert got.dtype == bool and np.array_equal(got, want)
    assert want[bad_rows].sum() >= len(bad_rows) - 1  # (a 2^31 error escapes one vector set w.p. 2^-c)
    assert not want[np.setdiff1d(np.arange(rows), bad_rows)].any()
    # uint64 vectors (rings with K > 32 keep them) give the same flags through verify_rows
    assert np.array_equal(check_flags(r, w_in, cu.astype(np.uint64), cw.astype(np.uint64), K32.mask), want)
