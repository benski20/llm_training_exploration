"""``pgw_fast.finish_chunk`` is bit-identical to the Python per-chunk finish of ``MaskedBackend._finish``.

The reference composes the Python functions the backend uses (``unpack_ring_u64``,
``freivalds_check_u64``, ``unmask_outputs_u32`` / ``unmask_outputs_u64``, ``dequantize_outputs``) in the
same order, so the verdict order (u all zero, Freivalds, range) is the same.

Cases are built from a sparse int8 weight so that ``y_hat = W8 x_hat`` and ``uw = u W8`` are cheap to
compute at real Qwen2.5-7B shapes; ``wr`` is then chosen as ``y_hat - yq`` so any ``yq`` (including the
range-check edges) can be planted without breaking the Freivalds relation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from pgw.core import native
from pgw.core.errors import IntegrityError
from pgw.core.freivalds import freivalds_check_u64
from pgw.core.masking import unmask_outputs_u32, unmask_outputs_u64
from pgw.core.packing import pack_ring, unpack_ring_u64
from pgw.core.ring import RingParams, dequantize_outputs

pytestmark = pytest.mark.skipif(
    not native.native_available("finish_chunk"), reason="pgw_fast.finish_chunk not built (rust/tools/build_fast.py)"
)

RINGS = {32: RingParams.from_name("K32A12C27"), 40: RingParams.from_name("K40A12"), 48: RingParams.from_name("K48A20")}
NNZ = 3  # nonzeros per weight row


@dataclass
class Case:
    ring: RingParams
    y_hat: bytes
    wr: np.ndarray  # int64 (rows, m)
    x_hat: np.ndarray | None  # int64 (rows, n)
    u: np.ndarray | None  # uint64 (c, m)
    uw: np.ndarray | None  # uint64 (c, n)
    sx: np.ndarray  # float64 (rows,)
    scale: np.ndarray  # float32 (m,)
    bias: np.ndarray | None  # float32 (m,)
    bound: int


def reference(cs: Case) -> tuple[np.ndarray | None, int, int]:
    """The Python path (loop body of ``MaskedBackend._finish``) with verdicts as status codes."""
    ring = cs.ring
    rows, m = cs.wr.shape
    u32 = ring.k == 32
    y_hat = None if u32 else unpack_ring_u64(cs.y_hat, (rows, m), ring)
    if cs.u is not None and cs.uw is not None:
        assert cs.x_hat is not None
        if y_hat is None:
            y_hat = np.frombuffer(cs.y_hat, dtype="<u4").reshape(rows, m).astype(np.uint64)
        if not cs.u.any():
            return None, 3, 0
        ok = freivalds_check_u64(cs.x_hat.view(np.uint64), y_hat, cs.u, cs.uw, ring)
        if not bool(ok.all()):
            return None, 1, int((~ok).sum())
    # Range check with the case's explicit bound (same comparison as the unmask functions do with
    # ``in_features``; see ``reference_with_in_features`` for that exact path).
    if u32:
        yq = unmask_outputs_u32(cs.y_hat, cs.wr, ring)
    else:
        assert y_hat is not None
        yq = unmask_outputs_u64(y_hat, cs.wr, ring)
    if yq.size and (int(yq.max()) > cs.bound or int(yq.min()) < -cs.bound):
        return None, 2, 0
    bias = None if cs.bias is None else torch.from_numpy(cs.bias)
    out = dequantize_outputs(
        torch.from_numpy(yq), torch.from_numpy(cs.sx), torch.from_numpy(cs.scale), bias, torch.float32
    )
    return out.numpy(), 0, 0


def reference_with_in_features(cs: Case, in_features: int) -> tuple[np.ndarray | None, int, int]:
    """Same, but with the range check done by the unmask functions themselves (``bound = output_bound(n)``)."""
    ring = cs.ring
    rows, m = cs.wr.shape
    try:
        if ring.k == 32:
            yq = unmask_outputs_u32(cs.y_hat, cs.wr, ring, in_features)
        else:
            yq = unmask_outputs_u64(unpack_ring_u64(cs.y_hat, (rows, m), ring), cs.wr, ring, in_features)
    except IntegrityError:
        return None, 2, 0
    bias = None if cs.bias is None else torch.from_numpy(cs.bias)
    return (
        dequantize_outputs(
            torch.from_numpy(yq), torch.from_numpy(cs.sx), torch.from_numpy(cs.scale), bias, torch.float32
        ).numpy(),
        0,
        0,
    )


def native_finish(cs: Case, threads: int = 0) -> tuple[np.ndarray | None, int, int]:
    return native.fast.finish_chunk(
        cs.y_hat, cs.wr, cs.x_hat, cs.u, cs.uw, cs.sx, cs.scale, cs.bias, cs.ring.k, cs.bound, threads
    )


def assert_same(cs: Case, threads: int = 0) -> tuple[int, int]:
    ref_out, ref_st, ref_bad = reference(cs)
    out, st, bad = native_finish(cs, threads)
    assert (st, bad) == (ref_st, ref_bad)
    if ref_out is None:
        assert out is None
    else:
        assert out is not None and out.dtype == np.float32 and out.shape == ref_out.shape
        assert out.flags.c_contiguous
        np.testing.assert_array_equal(out.view(np.int32), ref_out.view(np.int32))
    return st, bad


def make_case(
    rng: np.random.Generator,
    k: int,
    rows: int,
    m: int,
    n: int,
    c: int,
    *,
    bias: bool = True,
    yq: np.ndarray | None = None,
    bound: int | None = None,
) -> Case:
    """A genuine chunk: ``y_hat = W8 x_hat``, ``uw = u W8`` for a sparse random ``W8``; ``wr = y_hat - yq``."""
    ring = RINGS[k]
    mask = np.uint64(ring.mask)
    bound = ring.output_bound(n) if bound is None else bound
    x_hat = (rng.integers(0, 1 << 62, size=(rows, n), dtype=np.int64).view(np.uint64) & mask).view(np.int64)
    cols = rng.integers(0, n, size=(m, NNZ))
    vals = rng.integers(-127, 128, size=(m, NNZ)).astype(np.int64).view(np.uint64)
    xu = x_hat.view(np.uint64)
    y = np.zeros((rows, m), dtype=np.uint64)
    for d in range(NNZ):
        y += xu[:, cols[:, d]] * vals[None, :, d]
    y &= mask
    if yq is None:
        yq = rng.integers(-bound, bound + 1, size=(rows, m), dtype=np.int64)
    wr = ((y - yq.view(np.uint64)) & mask).view(np.int64)
    if c:
        u = rng.integers(0, 1 << 62, size=(c, m), dtype=np.int64).view(np.uint64) & mask
        uw = np.zeros((c, n), dtype=np.uint64)
        for d in range(NNZ):
            np.add.at(uw, (slice(None), cols[:, d]), u * vals[None, :, d])
        uw &= mask
        u.flags.writeable = False
        uw.flags.writeable = False
    else:
        u = uw = None
    sx = rng.uniform(1e-5, 1e-2, size=rows)
    scale = rng.uniform(-1e-3, 1e-3, size=m).astype(np.float32)
    scale[: min(m, 2)] = 0.0  # zero and negative-zero products
    b = rng.normal(0, 1, size=m).astype(np.float32) if bias else None
    return Case(ring, pack_ring(y.view(np.int64), ring), wr, x_hat if c else None, u, uw, sx, scale, b, bound)


SMALL = [(1, 1), (7, 5), (130, 70), (2051, 9)]
REAL = [(4608, 3584), (3584, 3584), (37888, 3584), (3584, 18944)]


@pytest.mark.parametrize("k", [32, 40, 48])
@pytest.mark.parametrize("rows", [1, 5, 40])
def test_genuine_small_shapes(k: int, rows: int) -> None:
    rng = np.random.default_rng(1000 * k + rows)
    for m, n in SMALL:
        for c in (0, 1, 3, 8):
            for bias in (True, False):
                cs = make_case(rng, k, rows, m, n, c, bias=bias)
                assert assert_same(cs) == (0, 0)
                assert assert_same(cs, threads=1) == (0, 0)
                assert assert_same(cs, threads=3) == (0, 0)


@pytest.mark.parametrize("k", [32, 40, 48])
@pytest.mark.parametrize("rows", [1, 5, 40])
def test_genuine_real_shapes(k: int, rows: int) -> None:
    rng = np.random.default_rng(7 + k * 100 + rows)
    for i, (m, n) in enumerate(REAL):
        c = 1 + (i * 3 + rows) % 8
        cs = make_case(rng, k, rows, m, n, c, bias=bool(i % 2))
        assert assert_same(cs) == (0, 0)
        # No check vectors, and the range check through the real ``in_features`` path.
        cs0 = replace(cs, x_hat=None, u=None, uw=None)
        assert assert_same(cs0) == (0, 0)
        out, st, _ = native_finish(cs0)
        ref_out, ref_st, _ = reference_with_in_features(cs0, n)
        assert st == ref_st == 0 and out is not None and ref_out is not None
        np.testing.assert_array_equal(out.view(np.int32), ref_out.view(np.int32))
        # Single bit flips anywhere in the payload: same verdict with and without check vectors.
        for _ in range(3):
            buf = bytearray(cs.y_hat)
            bit = int(rng.integers(0, len(buf) * 8))
            buf[bit // 8] ^= 1 << (bit % 8)
            assert assert_same(replace(cs, y_hat=bytes(buf)))[0] in (1, 2)
            assert_same(replace(cs0, y_hat=bytes(buf)))


@pytest.mark.parametrize("k", [32, 40, 48])
def test_range_edges(k: int) -> None:
    rng = np.random.default_rng(k)
    ring = RINGS[k]
    rows, m, n = 5, 300, 64
    bound = ring.output_bound(n)
    half = ring.half
    edges = [bound, -bound, 0, 1, -1]
    beyond = [bound + 1, -(bound + 1), half - 1, -half]
    if k == 32:
        beyond += [2**31 - 1, -(2**31)]
    for c in (0, 2):
        yq = rng.choice(np.array(edges, dtype=np.int64), size=(rows, m))
        assert assert_same(make_case(rng, k, rows, m, n, c, yq=yq, bound=bound)) == (0, 0)
        for v in beyond:
            for pos in [(0, 0), (rows - 1, m - 1), (2, 150)]:
                yq2 = yq.copy()
                yq2[pos] = v
                assert assert_same(make_case(rng, k, rows, m, n, c, yq=yq2, bound=bound)) == (2, 0)
        # Bound 0: only all-zero outputs pass. Bound >= half: nothing can fail.
        z = np.zeros((rows, m), dtype=np.int64)
        assert assert_same(make_case(rng, k, rows, m, n, c, yq=z, bound=0)) == (0, 0)
        ext = rng.choice(np.array([half - 1, -half], dtype=np.int64), size=(rows, m))
        assert assert_same(make_case(rng, k, rows, m, n, c, yq=ext, bound=half)) == (0, 0)
        assert assert_same(make_case(rng, k, rows, m, n, c, yq=ext, bound=2**62)) == (0, 0)


@pytest.mark.parametrize("k", [32, 40, 48])
def test_wr_extremes(k: int) -> None:
    rng = np.random.default_rng(50 + k)
    ring = RINGS[k]
    rows, m, n = 3, 257, 33
    for c in (0, 4):
        cs = make_case(rng, k, rows, m, n, c)
        # Plant y_hat - wr with wr = 0 and wr = 2^K - 1 (yq = y and y + 1 mod 2^K): mostly out of range.
        for wr_val in (0, ring.mask):
            wr = np.full((rows, m), wr_val, dtype=np.int64)
            assert_same(replace(cs, wr=wr))
            assert_same(replace(cs, wr=wr, bound=2**62))
        # Choose y_hat so that yq is in range for wr at the extremes (no checks: y_hat is free).
        if c == 0:
            for wr_val in (0, ring.mask):
                wr = np.full((rows, m), wr_val, dtype=np.int64)
                yq = rng.integers(-cs.bound, cs.bound + 1, size=(rows, m), dtype=np.int64)
                yb = pack_ring(((yq.view(np.uint64) + wr.view(np.uint64)) & np.uint64(ring.mask)).view(np.int64), ring)
                assert assert_same(replace(cs, wr=wr, y_hat=yb)) == (0, 0)


@pytest.mark.parametrize("k", [32, 40, 48])
@pytest.mark.parametrize("rows", [1, 5, 40])
def test_tampered(k: int, rows: int) -> None:
    rng = np.random.default_rng(300 + k + rows)
    ring = RINGS[k]
    seen: set[tuple[int, int]] = set()
    for m, n, c in [(130, 70, 1), (2051, 9, 3), (64, 16, 8)]:
        cs = make_case(rng, k, rows, m, n, c)
        for _ in range(12):
            buf = bytearray(cs.y_hat)
            nflip = int(rng.integers(1, 4))
            for _ in range(nflip):
                bit = int(rng.integers(0, len(buf) * 8))
                buf[bit // 8] ^= 1 << (bit % 8)
            for bound in (cs.bound, 2**62):
                seen.add(assert_same(replace(cs, y_hat=bytes(buf), bound=bound)))
                seen.add(assert_same(replace(cs, y_hat=bytes(buf), bound=bound, u=None, uw=None, x_hat=None)))
        # +2^31 on one element per tampered row (high 2-adic valuation; caught only when u_j is odd for K=32).
        y = unpack_ring_u64(cs.y_hat, (rows, m), ring)
        for _ in range(6):
            y2 = y.copy()
            bad = rng.choice(rows, size=int(rng.integers(1, rows + 1)), replace=False)
            y2[bad, rng.integers(0, m, size=bad.size)] += np.uint64(1 << 31)
            y2 &= np.uint64(ring.mask)
            yb = pack_ring(y2.view(np.int64), ring)
            for bound in (cs.bound, 2**62):
                seen.add(assert_same(replace(cs, y_hat=yb, bound=bound)))
        # Tamper with a small error the range check cannot see: only Freivalds catches it.
        y2 = y.copy()
        y2[rows - 1, 0] = (y2[rows - 1, 0] + np.uint64(1)) & np.uint64(ring.mask)
        st, _ = assert_same(replace(cs, y_hat=pack_ring(y2.view(np.int64), ring), bound=2**62))
        assert st in (0, 1)
    assert (1, 1) in seen and (2, 0) in seen
    if rows > 1:
        assert any(st == 1 and b > 1 for st, b in seen)


@pytest.mark.parametrize("k", [32, 40, 48])
def test_u_all_zero(k: int) -> None:
    rng = np.random.default_rng(k + 9)
    cs = make_case(rng, k, 5, 100, 20, 3)
    z = replace(cs, u=np.zeros_like(cs.u))
    assert assert_same(z) == (3, 0)
    # Takes precedence over a failing Freivalds or range check.
    bad = bytearray(cs.y_hat)
    bad[0] ^= 0x80
    assert assert_same(replace(z, y_hat=bytes(bad), bound=0)) == (3, 0)
    # Only the low K bits count (Python masks u to K bits when it builds the uint64 copy).
    if k < 64:
        hi = np.full_like(cs.u, np.uint64(1) << np.uint64(k))
        out, st, _ = native_finish(replace(cs, u=hi))
        assert st == 3 and out is None


def test_zero_rows() -> None:
    rng = np.random.default_rng(0)
    cs = make_case(rng, 32, 0, 10, 4, 2)
    out, st, bad = native_finish(cs)
    assert (st, bad) == (0, 0) and out is not None and out.shape == (0, 10)


def test_input_errors() -> None:
    rng = np.random.default_rng(1)
    cs = make_case(rng, 32, 2, 10, 4, 2)
    with pytest.raises(ValueError, match="bytes"):
        native_finish(replace(cs, y_hat=cs.y_hat[:-1]))
    with pytest.raises(ValueError):
        native.fast.finish_chunk(cs.y_hat, cs.wr, cs.x_hat, cs.u, cs.uw, cs.sx, cs.scale, cs.bias, 36, cs.bound)
    with pytest.raises(ValueError):
        native.fast.finish_chunk(cs.y_hat, cs.wr, cs.x_hat, cs.u, cs.uw, cs.sx, cs.scale, cs.bias, 64, cs.bound)
    with pytest.raises(ValueError):
        native_finish(replace(cs, wr=cs.wr.astype(np.int32)))
    with pytest.raises(ValueError):
        native_finish(replace(cs, wr=np.asfortranarray(cs.wr)))
    with pytest.raises(ValueError):
        native_finish(replace(cs, sx=cs.sx[:1]))
    with pytest.raises(ValueError):
        native_finish(replace(cs, sx=cs.sx.astype(np.float32)))
    with pytest.raises(ValueError):
        native_finish(replace(cs, scale=cs.scale[:-1]))
    with pytest.raises(ValueError):
        native_finish(replace(cs, scale=cs.scale.astype(np.float64)))
    with pytest.raises(ValueError):
        native_finish(replace(cs, bias=np.zeros(3, dtype=np.float32)))
    with pytest.raises(ValueError):
        native_finish(replace(cs, uw=None))
    with pytest.raises(ValueError):
        native_finish(replace(cs, x_hat=None))
    with pytest.raises(ValueError):
        native_finish(replace(cs, u=cs.u[:, :-1]))
    with pytest.raises(ValueError):
        native_finish(replace(cs, uw=cs.uw[:0]))
    with pytest.raises(ValueError):
        native_finish(replace(cs, u=cs.u[:0], uw=cs.uw[:0]))
    with pytest.raises(ValueError):
        native_finish(replace(cs, y_hat=bytearray(cs.y_hat)))  # type: ignore[arg-type]
    # uint64 views of x_hat / wr are accepted (same bits).
    a = native_finish(cs)
    b = native_finish(replace(cs, wr=cs.wr.view(np.uint64), x_hat=cs.x_hat.view(np.uint64)))  # type: ignore[union-attr]
    assert a[1:] == b[1:] == (0, 0)
    np.testing.assert_array_equal(a[0].view(np.int32), b[0].view(np.int32))


# ---- finish_group: several chunks (sessions) of one linear sharing check vectors -----------------
group_only = pytest.mark.skipif(not native.native_available("finish_group"), reason="pgw_fast.finish_group not built")


def make_group(rng: np.random.Generator, k: int, rows: list[int], m: int, n: int, c: int, *, bias: bool = True
               ) -> list[Case]:
    """Chunks of one linear (same sparse W8, u, uw, scale, bias), each with its own x_hat / W r."""
    base = make_case(rng, k, sum(rows), m, n, c, bias=bias)
    ring = RINGS[k]
    y = unpack_ring_u64(base.y_hat, (sum(rows), m), ring)
    parts, r0 = [], 0
    for r in rows:
        sl = slice(r0, r0 + r)
        parts.append(replace(
            base, y_hat=pack_ring(y[sl].view(np.int64), ring), wr=base.wr[sl].copy(),
            x_hat=None if base.x_hat is None else base.x_hat[sl].copy(), sx=base.sx[sl].copy(),
        ))
        r0 += r
    return parts


def _narrow(a: np.ndarray | None, k: int) -> np.ndarray | None:
    return None if a is None or k > 32 else a.astype(np.uint32)


def group_finish(parts: list[Case], *, x_as: str = "array", narrow: bool = False, wipe: bool = False,
                 threads: int = 0) -> tuple[np.ndarray | None, int, int]:
    p0 = parts[0]
    k = p0.ring.k
    items = []
    for cs in parts:
        x: object
        if cs.x_hat is None:
            x = None
        elif x_as == "bytes":
            x = pack_ring(cs.x_hat, cs.ring)
        else:
            x = cs.x_hat
        wr = cs.wr.astype(np.uint32) if narrow and k <= 32 else cs.wr
        items.append((cs.y_hat, wr if wipe else wr.copy(), x, cs.sx))
    u = _narrow(p0.u, k) if narrow else p0.u
    uw = _narrow(p0.uw, k) if narrow else p0.uw
    return native.fast.finish_group(items, u, uw, p0.scale, p0.bias, k, p0.bound, wipe, threads)


@group_only
@pytest.mark.parametrize("k", [32, 40, 48])
def test_group_matches_per_chunk(k: int) -> None:
    rng = np.random.default_rng(4000 + k)
    for m, n in [(130, 70), (2051, 9), (37888, 3584)]:
        for c in (0, 3, 7):
            parts = make_group(rng, k, [1, 3, 1, 2], m, n, c, bias=bool(c % 2))
            want = [native_finish(p) for p in parts]
            assert all(st == 0 for _, st, _ in want)
            ref = np.concatenate([o for o, _, _ in want])
            for x_as in ("array", "bytes"):
                for narrow in (False, True):
                    for threads in (0, 1, 3):
                        out, st, bad = group_finish(parts, x_as=x_as, narrow=narrow, threads=threads)
                        assert (st, bad) == (0, 0) and out is not None
                        np.testing.assert_array_equal(out.view(np.int32), ref.view(np.int32))
            # one part alone is exactly finish_chunk
            out, st, _ = group_finish(parts[1:2], x_as="bytes", narrow=True)
            assert st == 0 and out is not None and want[1][0] is not None
            np.testing.assert_array_equal(out.view(np.int32), want[1][0].view(np.int32))


@group_only
@pytest.mark.parametrize("k", [32, 48])
def test_group_wipes_wr_and_verdicts(k: int) -> None:
    rng = np.random.default_rng(77 + k)
    parts = make_group(rng, k, [2, 1, 3], 300, 40, 3)
    wrs = [p.wr.copy() for p in parts]
    out, st, _ = group_finish(parts, wipe=True)
    assert st == 0 and out is not None
    assert all(not p.wr.any() for p in parts)  # zeroed in place
    # Tamper with the second part only: Freivalds fails, its row is counted, every W r is still wiped.
    ring = RINGS[k]
    parts = [replace(p, wr=w.copy(), bound=2**62) for p, w in zip(parts, wrs, strict=True)]
    y = unpack_ring_u64(parts[1].y_hat, (1, 300), ring)
    y[0, 7] = (y[0, 7] + np.uint64(1)) & np.uint64(ring.mask)
    parts[1] = replace(parts[1], y_hat=pack_ring(y.view(np.int64), ring))
    out, st, bad = group_finish(parts, wipe=True)
    assert out is None and (st, bad) == (1, 1)
    assert all(not p.wr.any() for p in parts)
    # All-zero u: status 3, W r wiped as well.
    parts = [replace(p, wr=w.copy(), u=np.zeros_like(p.u)) for p, w in zip(parts, wrs, strict=True)]
    out, st, _ = group_finish(parts, wipe=True)
    assert out is None and st == 3 and all(not p.wr.any() for p in parts)
    # Range failure (no checks).
    parts = [replace(p, wr=w.copy(), u=None, uw=None, x_hat=None, bound=0) for p, w in zip(parts, wrs, strict=True)]
    out, st, _ = group_finish(parts, wipe=True)
    assert out is None and st == 2 and all(not p.wr.any() for p in parts)


@group_only
def test_group_input_errors() -> None:
    rng = np.random.default_rng(5)
    parts = make_group(rng, 32, [1, 2], 10, 4, 2)
    p0 = parts[0]
    assert p0.u is not None and p0.uw is not None
    fin = native.fast.finish_group
    good = [(p.y_hat, p.wr.copy(), p.x_hat, p.sx) for p in parts]
    with pytest.raises(ValueError, match="part 1"):
        fin([good[0], (good[1][0][:-1], *good[1][1:])], p0.u, p0.uw, p0.scale, p0.bias, 32, p0.bound)
    with pytest.raises(ValueError, match="x_hat is required"):
        fin([(good[0][0], good[0][1], None, good[0][3])], p0.u, p0.uw, p0.scale, p0.bias, 32, p0.bound)
    with pytest.raises(ValueError):
        fin(good, p0.u, None, p0.scale, p0.bias, 32, p0.bound)
    with pytest.raises(ValueError):  # uint32 needs K <= 32
        fin(good, p0.u.astype(np.uint32), p0.uw.astype(np.uint32), p0.scale, p0.bias, 40, p0.bound)
    ro = p0.wr.copy()
    ro.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        fin([(p0.y_hat, ro, p0.x_hat, p0.sx)], p0.u, p0.uw, p0.scale, p0.bias, 32, p0.bound, True)
    out, st, _ = fin([(p0.y_hat, ro, p0.x_hat, p0.sx)], p0.u, p0.uw, p0.scale, p0.bias, 32, p0.bound, False)
    assert st == 0 and out is not None and ro.any()  # read-only W r is fine without wiping
    # Errors raised before the kernel leave W r untouched (the caller wipes it).
    w = p0.wr.copy()
    with pytest.raises(ValueError):
        fin([(p0.y_hat, w, p0.x_hat, p0.sx[:0])], p0.u, p0.uw, p0.scale, p0.bias, 32, p0.bound)
    assert np.array_equal(w, p0.wr)
    out, st, _ = fin([], None, None, p0.scale, p0.bias, 32, p0.bound)
    assert st == 0 and out is not None and out.shape == (0, 10)
