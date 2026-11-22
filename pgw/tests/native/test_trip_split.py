"""Multi-core per-trip kernels (``PGW_TRIP_SPLIT``, opt-in): ``TripLease.prepare(..., split=c)`` cuts each row
into a fixed partition of column blocks over ``c`` tasks of the persistent pool, ``TripTicket.finish(..., split=c)``
splits the unmask and the prime output half by output columns. The bytes must not depend on ``c``: payload,
``x_hat``/``xq``, duplicate fingerprints, outputs, statuses and bad-row counts are identical to the one-thread
kernels for every ``c``, for the ring and the prime (``p31``) check formats; a tampered reply fails the same way;
``r`` derived ahead (any rows per trip) or inline gives the same bytes and is single use.
"""

from __future__ import annotations

import numpy as np
import pytest

from pgw.core import native

pytestmark = pytest.mark.skipif(not native.native_available("TripLease"), reason="pgw_fast TripLease not built")

P31 = (1 << 31) - 1
K, A, CAP = 32, 12, 27


def _has_split() -> bool:
    try:
        f = native.fast
        u = np.zeros((1, 4), np.uint32)
        lease = f.TripLease(bytes(32), 3, 4, K, A, CAP, 1.0, 0, [(bytearray(16), 0, 1)], 0, u, np.zeros((1, 3), np.uint32))
        return lease.prepare(np.zeros((1, 3), np.float32), False, False, 1, None, 2) is not None
    except TypeError:
        return False


def _lease(key: bytes, n: int, m: int, rows: int, wr: np.ndarray | None, u: np.ndarray, uw: np.ndarray, *,
           field: int = 0, r_ahead: bool = False, wnorm: float, slot0: int = 3) -> object:
    chunk = bytearray(rows * m * 4) if wr is None else bytearray(wr.astype("<u4").tobytes())
    return native.fast.TripLease(key, n, m, K, A, CAP, wnorm, slot0, [(chunk, 0, rows)], 0, u, uw, r_ahead, field)


def _setup(n: int, m: int, rows: int, field: int, seed: int = 0) -> dict:
    """Weights, check vectors and a lease's W r such that the genuine reply passes every check."""
    g = np.random.default_rng(seed)
    w = g.integers(-127, 128, (m, n), dtype=np.int64)
    key = g.integers(0, 256, 32, dtype=np.uint8).tobytes()
    c = 1 if field else 3
    hi = P31 if field else 1 << 32
    u = g.integers(0, hi, (c, m), dtype=np.uint64)
    if field:
        uw = (u.astype(np.int64) @ w) % P31  # |terms| < 2^38, m < 2^14: exact in int64
    else:
        uw = (u.astype(np.uint64) @ (w.astype(np.uint64))) & 0xFFFFFFFF
    u32, uw32 = u.astype(np.uint32), uw.astype(np.uint32)
    wnorm = float(np.sqrt((w.astype(np.float64) ** 2).sum(axis=1)).max())
    # r of every row: from a throwaway lease (sampled: r = x_hat - xq)
    probe = _lease(key, n, m, rows, None, u32, uw32, field=field, wnorm=wnorm)
    xs = g.standard_normal((rows, n)).astype(np.float32)
    xs[:, 7] *= 30.0  # a spike: a narrower width
    got = probe.prepare(np.zeros((rows, n), np.float32), False, True, 1, None)
    _, tk, xh, xq, _ = got
    r = (xh - xq) & 0xFFFFFFFF
    tk.wipe()
    wr = (w.astype(np.uint64) @ r.T.astype(np.uint64)).T & 0xFFFFFFFF
    return {"w": w, "key": key, "u": u32, "uw": uw32, "wnorm": wnorm, "wr": wr, "x": xs}


def _trip(s: dict, n: int, m: int, rows: int, field: int, split: int, *, r_ahead: bool = False, tamper: int = -1,
          per_trip: int = 1, sampled: bool = True) -> list:
    lease = _lease(s["key"], n, m, rows, s["wr"], s["u"], s["uw"], field=field, r_ahead=r_ahead, wnorm=s["wnorm"])
    dup = (np.arange(16, dtype=np.int64) * 7 % n, 5, np.arange(1, 17, dtype=np.uint64) * 0x9E3779B97F4A7C15, 4, 4)
    outs = []
    scale = np.full(m, 1e-3, np.float32)
    bias = np.linspace(-1, 1, m).astype(np.float32)
    bound = (1 << 31) - 1
    for t0 in range(0, rows, per_trip):
        x = np.ascontiguousarray(s["x"][t0:t0 + per_trip])
        payload, tk, xh, xq, keys = lease.prepare(x, True, sampled, 1, dup, split)
        # the service's genuine answer: W x_hat mod 2^32
        xhat = np.frombuffer(payload, "<u4").reshape(per_trip, n).astype(np.uint64)
        y = (s["w"].astype(np.uint64) @ xhat.T).T & 0xFFFFFFFF
        y = y.astype("<u4")
        if tamper >= 0:
            y.reshape(-1)[tamper] ^= 1
        st, bad, out, pend = tk.finish(y.tobytes(), scale, bias, bound, False, 1, split)
        assert pend is None
        outs.append((bytes(payload), None if xh is None else xh.tobytes(), None if xq is None else xq.tobytes(),
                     keys.tobytes(), st, bad, out.tobytes()))
    return outs


needs_split = pytest.mark.skipif(not _has_split(), reason="pgw_fast without the split per-trip kernels")


@needs_split
@pytest.mark.parametrize("field", [0, P31])
@pytest.mark.parametrize("rows", [1, 2])
def test_split_identical_for_any_core_count(field: int, rows: int) -> None:
    n, m = 4096 + 1000, 2 * 8192 + 300  # 3 column blocks; 3 output-column units
    s = _setup(n, m, rows, field, seed=rows)
    ref = _trip(s, n, m, rows, field, 0, per_trip=rows)
    assert all(o[4] == 0 for o in ref), [o[4:6] for o in ref]
    for cores in (2, 3, 4, 7):
        got = _trip(s, n, m, rows, field, cores, per_trip=rows)
        assert got == ref, cores


@needs_split
@pytest.mark.parametrize("field", [0, P31])
def test_split_tamper_fails_identically(field: int) -> None:
    n, m = 4096 + 5, 2 * 8192 + 17
    s = _setup(n, m, 1, field, seed=9)
    for pos in (0, m // 2, m - 1):
        ref = _trip(s, n, m, 1, field, 0, tamper=pos)
        assert ref[0][4] != 0 and ref[0][6] == bytes(4 * m)  # rejected, outputs zeroed
        for cores in (2, 4):
            assert _trip(s, n, m, 1, field, cores, tamper=pos) == ref


@needs_split
@pytest.mark.parametrize("per_trip", [1, 2])
def test_r_ahead_identical_and_single_use(per_trip: int) -> None:
    n, m, rows = 4096 + 64, 8192 + 64, 6
    s = _setup(n, m, rows, 0, seed=3)
    ref = _trip(s, n, m, rows, 0, 0, per_trip=per_trip)
    for split in (0, 3):
        assert _trip(s, n, m, rows, 0, split, r_ahead=True, per_trip=per_trip) == ref
    # every payload differs (no r reused across trips)
    assert len({o[0] for o in ref}) == len(ref)


@needs_split
def test_small_vectors_and_other_widths_fall_back() -> None:
    # below the split threshold (and for c > 16) the one-thread kernel runs: same bytes
    n, m = 300, 500
    s = _setup(n, m, 1, 0, seed=5)
    assert _trip(s, n, m, 1, 0, 4) == _trip(s, n, m, 1, 0, 0)


@needs_split
def test_split_nonfinite_and_ticket_wipe() -> None:
    n, m = 4200, 4100
    s = _setup(n, m, 2, 0, seed=11)
    lease = _lease(s["key"], n, m, 2, s["wr"], s["u"], s["uw"], wnorm=s["wnorm"])
    bad = s["x"][:1].copy()
    bad[0, 4000] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        lease.prepare(bad, True, False, 1, None, 4)
    assert lease.pos == 0  # nothing taken
    payload, tk, _, _, _ = lease.prepare(np.ascontiguousarray(s["x"][:1]), True, False, 1, None, 4)
    tk.wipe()
    with pytest.raises(ValueError, match="single use"):
        tk.finish(bytes(4 * m), np.ones(m, np.float32), None, 1 << 30, False, 1, 4)
    assert lease.pos == 1
