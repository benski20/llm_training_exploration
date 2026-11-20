"""Chat-speed WP-A4a: the native trip kernels (``pgw_fast.trip_*``) are bit-identical to the existing path.

* ``trip_prepare_into`` == ``quantize_activations`` + ``mask_activations`` + ``pack_ring`` (and so
  == ``quantize_mask_pack``), written into reused caller buffers, with the constant-time width
  selection and the fused Freivalds input half ``uw . x_hat``.
* ``trip_finish_into`` == ``finish_group`` (outputs and statuses) with the check fused, split
  (input half from prepare) or off; ``trip_freivalds_output`` / ``trip_freivalds_submit`` give the
  same verdicts.
* masked == plain: prepare -> provider ``W x_hat mod 2^K`` -> finish equals the plain quantized
  product ``dequantize(W8 xq)`` bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pgw.core import native
from pgw.core.errors import RingOverflowError
from pgw.core.masking import mask_activations
from pgw.core.packing import pack_ring
from pgw.core.ring import RingParams, dequantize_outputs, quantize_activations

pytestmark = pytest.mark.skipif(
    not native.native_available("trip_prepare_into"), reason="pgw_fast.trip_prepare_into not built"
)

RINGS = ["K32A12C27", "K40A12", "K48A20", "K32A8", "K24A12C21", "K56A20C40", "K16A7"]
KINDS = ["normal", "outliers", "zero_rows", "ties", "tiny", "huge", "subnormal", "spike"]


def _activations(rng: np.random.Generator, rows: int, n: int, kind: str) -> np.ndarray:
    x = rng.normal(0, 1, size=(rows, n)).astype(np.float32)
    if kind == "outliers":
        ch = rng.choice(n, size=max(1, n // 500), replace=False)
        x[:, ch] *= 300
    elif kind == "zero_rows":
        x[::2] = 0
        if rows > 1:
            x[1, : n // 2] = -0.0
    elif kind == "ties":
        x = (rng.integers(-2 * 2047, 2 * 2047, size=(rows, n)) / 16.0).astype(np.float32)
        x[:, 0] = 2047 / 8
    elif kind == "tiny":
        x *= np.float32(1e-30)
    elif kind == "huge":
        x *= np.float32(1e30)
    elif kind == "subnormal":
        x = (rng.integers(-50, 50, size=(rows, n)) * np.float32(1e-45)).astype(np.float32)
    elif kind == "spike":  # low ||x|| / max|x|: the most bits fit
        x *= np.float32(1e-3)
        x[:, rng.integers(0, n)] = 50.0
    return x


def _wnorm(rng: np.random.Generator, n: int) -> float:
    w8 = np.clip(np.rint(rng.normal(0, 40, size=(4, n))), -127, 127)
    return float(np.sqrt((w8 * w8).sum(axis=1)).max())


def _reference(x: np.ndarray, r: np.ndarray, ring: RingParams, wnorm: float) -> tuple[bytes, np.ndarray, np.ndarray]:
    xq, sx = quantize_activations(torch.from_numpy(x), ring, wnorm if ring.cap else None)
    x_hat = mask_activations(xq, torch.from_numpy(r), ring)
    return pack_ring(x_hat, ring), x_hat.numpy(), sx.numpy()


class Buffers:
    """Caller-owned trip buffers, reused across calls of one shape."""

    def __init__(self, rows: int, n: int, ring: RingParams, c: int) -> None:
        self.payload = np.full(rows * n * (ring.k // 8), 0xA5, dtype=np.uint8)
        self.sx = np.full(rows, -1.0)
        self.xhat = np.full((rows, n), -1, dtype=np.int64)
        self.rhs = np.full((rows, c), 7, dtype=np.uint64)


def _trip_prepare(x, r, ring, wnorm, buf, uw=None, threads=1):  # noqa: ANN001, ANN202
    native.fast.trip_prepare_into(
        x, r, ring.k, ring.a, ring.cap, float(wnorm), buf.payload, buf.sx, buf.xhat,
        uw, None if uw is None else buf.rhs, threads,
    )


def _rhs_ref(x_hat: np.ndarray, uw: np.ndarray, ring: RingParams) -> np.ndarray:
    prod = (x_hat.astype(np.uint64) @ uw.astype(np.uint64).T)  # wraps mod 2^64
    return prod & np.uint64(ring.mask)


@pytest.mark.parametrize("ring_name", RINGS)
@pytest.mark.parametrize(("rows", "n"), [(1, 3584), (1, 18944), (5, 896), (40, 7), (3, 1)])
def test_prepare_into_matches_python(ring_name: str, rows: int, n: int) -> None:
    ring = RingParams.from_name(ring_name)
    rng = np.random.default_rng(hash((ring_name, rows, n)) & 0xFFFF_FFFF)
    c = 2
    buf = Buffers(rows, n, ring, c)  # one set of buffers for every call below
    for kind in KINDS:
        x = _activations(rng, rows, n, kind)
        r = rng.integers(0, 1 << ring.k, size=x.shape, dtype=np.int64)
        wn = _wnorm(rng, n)
        uw = rng.integers(0, 1 << ring.k, size=(c, n), dtype=np.uint64)
        try:
            want = _reference(x, r, ring, wn)
        except RingOverflowError:
            with pytest.raises(ValueError, match=r"^ring_overflow:"):
                _trip_prepare(x, r, ring, wn, buf, uw)
            assert not buf.payload.any() and not buf.rhs.any()
            continue
        _trip_prepare(x, r, ring, wn, buf, uw)
        assert buf.payload.tobytes() == want[0], kind
        assert np.array_equal(buf.xhat, want[1]), kind
        assert np.array_equal(buf.sx.view(np.uint64), want[2].view(np.uint64)), kind  # width choice, bitwise
        assert np.array_equal(buf.rhs, _rhs_ref(want[1], uw, ring)), kind
        # the existing native kernel agrees too
        old = native.fast.quantize_mask_pack(x, r, ring.k, ring.a, ring.cap, float(wn))
        assert old[0] == want[0]


@pytest.mark.parametrize("wnorm_scale", [1e-3, 0.3, 1.0, 3.0, 30.0, 1e3, 1e6])
def test_prepare_into_capped_widths(wnorm_scale: float) -> None:
    """Rows that need different widths (2..12 bits) in one call; uint32 ``uw`` for K=32."""
    ring = RingParams.from_name("K32A12C27")
    rng = np.random.default_rng(int(wnorm_scale * 1000) + 1)
    rows, n = 16, 3584
    x = _activations(rng, rows, n, "normal")
    x[::3] = _activations(rng, len(x[::3]), n, "spike")
    x[1::4] *= 1e-6
    r = rng.integers(0, 1 << 32, size=x.shape, dtype=np.int64)
    wn = _wnorm(rng, n) * wnorm_scale
    uw = rng.integers(0, 1 << 32, size=(3, n), dtype=np.uint32)
    buf = Buffers(rows, n, ring, 3)
    try:
        want = _reference(x, r, ring, wn)
    except RingOverflowError:
        with pytest.raises(ValueError, match=r"^ring_overflow:"):
            _trip_prepare(x, r, ring, wn, buf, uw)
        return
    for threads in (1, 0):
        _trip_prepare(x, r, ring, wn, buf, uw, threads)
        assert buf.payload.tobytes() == want[0]
        assert np.array_equal(buf.sx.view(np.uint64), want[2].view(np.uint64))
        assert np.array_equal(buf.rhs, _rhs_ref(want[1], uw, ring))


def test_prepare_into_prefill_blocks_on_pool() -> None:
    ring = RingParams.from_name("K32A12C27")
    rng = np.random.default_rng(9)
    rows, n = 40, 3584
    x = _activations(rng, rows, n, "outliers")
    r = rng.integers(0, 1 << 32, size=x.shape, dtype=np.int64)
    wn = _wnorm(rng, n)
    want = _reference(x, r, ring, wn)
    buf = Buffers(rows, n, ring, 1)
    _trip_prepare(x, r, ring, wn, buf, threads=0)
    assert buf.payload.tobytes() == want[0]
    assert np.array_equal(buf.sx.view(np.uint64), want[2].view(np.uint64))


def test_prepare_into_errors() -> None:
    ring = RingParams.from_name("K32A12C27")
    x = np.ones((2, 8), dtype=np.float32)
    r = np.zeros((2, 8), dtype=np.int64)
    buf = Buffers(2, 8, ring, 1)
    x[1, 3] = np.nan
    with pytest.raises(ValueError, match="^ring_overflow: non-finite"):
        _trip_prepare(x, r, ring, 100.0, buf)
    x[1, 3] = 1.0
    with pytest.raises(ValueError, match="^ring_overflow: K32A12C27"):
        _trip_prepare(x, r, ring, 1e30, buf)
    assert not buf.payload.any()
    with pytest.raises(ValueError, match="payload has"):
        native.fast.trip_prepare_into(x, r, 32, 12, 27, 100.0, np.zeros(5, np.uint8), buf.sx)
    with pytest.raises(ValueError, match="C-contiguous"):
        native.fast.trip_prepare_into(np.asfortranarray(np.ones((3, 8), np.float32)), np.zeros((3, 8), np.int64),
                                      32, 12, 27, 100.0, np.zeros(96, np.uint8), np.zeros(3))
    with pytest.raises(ValueError, match="both"):
        native.fast.trip_prepare_into(x, r, 32, 12, 27, 100.0, buf.payload, buf.sx, None,
                                      np.zeros((1, 8), np.uint64), None)
    ro = np.zeros(64, np.uint8)
    ro.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        native.fast.trip_prepare_into(x, r, 32, 12, 27, 100.0, ro, buf.sx)


def test_fortran_order_inputs() -> None:
    """Regression: numpy's ``as_slice`` accepts Fortran order, which the kernels would read transposed.
    The allocating kernels copy such inputs into C order; the trip kernels and finish refuse them."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 8)).astype(np.float32)
    r = rng.integers(0, 1 << 32, size=(3, 8), dtype=np.int64)
    xf, rf = np.asfortranarray(x), np.asfortranarray(r)
    want = native.fast.quantize_mask_pack(x, r, 32, 12, 27, 100.0)
    assert native.fast.quantize_mask_pack(xf, rf, 32, 12, 27, 100.0)[0] == want[0]
    assert native.fast.quantize_mask_pack_many([xf], [rf], 32, 12, 27, [100.0], [True])[0][0] == want[0]
    ring = RingParams.from_name("K32A12C27")
    call = Call(ring, 2, 64, 32, 2, seed=4)
    with pytest.raises(ValueError, match="C-contiguous"):
        native.fast.finish_group([(call.y_hat, call.wr.copy(), call.payload, call.sx)], np.asfortranarray(call.u),
                                 call.uw, call.scale, call.bias, 32, call.bound, True)
    with pytest.raises(ValueError, match="C-contiguous"):
        call.new(call.y_hat, call.u, "split", np.asfortranarray(np.empty((2, 64), np.float32)))


def test_width_window_is_public_and_small() -> None:
    for n, wn in [(3584, 1500.0), (3584, 4000.0), (18944, 5000.0), (896, 400.0)]:
        assert native.fast.trip_width_window(n, 12, 27, wn) == 2
    assert native.fast.trip_width_window(3584, 12, 0, 0.0) == 1  # uncapped
    assert native.fast.trip_width_window(3584, 12, 27, float("nan")) == 11  # degenerate: every width


# ---- finish -------------------------------------------------------------------------------------


class Call:
    """A genuine masked call: dense int8 W, prepared by the trip kernel, provider ``W x_hat``."""

    def __init__(self, ring: RingParams, rows: int, m: int, n: int, c: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.ring, self.rows, self.m, self.n = ring, rows, m, n
        mask = np.uint64(ring.mask)
        self.w8 = rng.integers(-127, 128, size=(m, n), dtype=np.int64)
        self.wnorm = float(np.sqrt((self.w8.astype(np.float64) ** 2).sum(axis=1)).max())
        self.x = _activations(rng, rows, n, "outliers")
        self.r = rng.integers(0, 1 << ring.k, size=(rows, n), dtype=np.int64)
        self.u = rng.integers(0, 1 << ring.k, size=(c, m), dtype=np.uint64)
        wu = self.w8.astype(np.uint64)
        self.uw = (self.u @ wu) & mask
        buf = Buffers(rows, n, ring, c)
        _trip_prepare(self.x, self.r, ring, self.wnorm, buf, self.uw)
        self.payload = buf.payload.tobytes()
        self.x_hat = buf.xhat.copy()
        self.sx = buf.sx.copy()
        self.rhs = buf.rhs.copy()
        y = (self.x_hat.astype(np.uint64) @ wu.T) & mask  # the provider: W x_hat mod 2^K
        self.y_hat = pack_ring(torch.from_numpy(y.astype(np.int64)), ring)
        self.wr = ((self.r.astype(np.uint64) @ wu.T) & mask).astype(np.int64)
        self.scale = rng.uniform(1e-4, 1e-2, size=m).astype(np.float32)
        self.bias = rng.normal(0, 1, size=m).astype(np.float32)
        self.bound = ring.output_bound(n)

    def old(self, y_hat: bytes, u: np.ndarray, bound: int | None = None):  # noqa: ANN201
        return native.fast.finish_group(
            [(y_hat, self.wr.copy(), self.payload, self.sx)], u, self.uw, self.scale, self.bias, self.ring.k,
            self.bound if bound is None else bound, True,
        )

    def new(self, y_hat, u, mode: str, out: np.ndarray, bound: int | None = None, threads: int = 1):  # noqa: ANN001, ANN201
        wr = self.wr.copy()
        kw = {}
        if mode == "full":
            kw = {"x_hat": self.payload, "u": u, "uw": self.uw}
        elif mode == "split":
            kw = {"u": u, "rhs": self.rhs}
        res = native.fast.trip_finish_into(
            y_hat, wr, self.sx, self.scale, self.bias, self.ring.k, self.bound if bound is None else bound, out,
            threads=threads, **kw,
        )
        assert not wr.any()  # W r wiped in place
        return res


@pytest.mark.parametrize("ring_name", ["K32A12C27", "K40A12", "K48A20", "K24A12C21", "K56A20C40"])
@pytest.mark.parametrize(("rows", "m", "n"), [(1, 300, 200), (4, 2100, 64)])
def test_finish_into_matches_finish_group(ring_name: str, rows: int, m: int, n: int) -> None:
    ring = RingParams.from_name(ring_name)
    call = Call(ring, rows, m, n, 2, seed=rows * 7 + m)
    out_old, st, _ = call.old(call.y_hat, call.u)
    assert st == 0
    out = np.full((rows, m), np.nan, dtype=np.float32)  # reused for every call below
    for mode in ("full", "split", "off"):
        for y in (call.y_hat, np.frombuffer(call.y_hat, dtype=np.uint8)):
            out.fill(np.nan)
            assert call.new(y, call.u, mode, out) == (0, 0)
            assert np.array_equal(out.view(np.uint32), out_old.view(np.uint32)), mode
    assert native.fast.trip_freivalds_output(call.y_hat, call.u, call.rhs, ring.k) == (0, 0)
    # tampered last row: same verdict, output zeroed
    bad = bytearray(call.y_hat)
    bad[-1] ^= 0x10
    bad = bytes(bad)
    want = call.old(bad, call.u)[1:]
    assert want == (1, 1)
    for mode in ("full", "split"):
        out.fill(1.0)
        assert call.new(bad, call.u, mode, out) == want
        assert not out.any()
    assert native.fast.trip_freivalds_output(bad, call.u, call.rhs, ring.k) == want
    # vectors wiped (reservation closed)
    uz = np.zeros_like(call.u)
    assert call.old(call.y_hat, uz)[1] == 3
    for mode in ("full", "split"):
        assert call.new(call.y_hat, uz, mode, out)[0] == 3
    assert native.fast.trip_freivalds_output(call.y_hat, uz, call.rhs, ring.k)[0] == 3
    # range check
    st_old = call.old(call.y_hat, call.u, bound=0)[1]
    for mode in ("full", "split", "off"):
        assert call.new(call.y_hat, call.u, mode, out, bound=0)[0] == st_old


def test_finish_into_prefill_on_pool() -> None:
    ring = RingParams.from_name("K32A12C27")
    call = Call(ring, 24, 9000, 16, 2, seed=5)
    out_old, st, _ = call.old(call.y_hat, call.u)
    assert st == 0
    out = np.empty((24, 9000), dtype=np.float32)
    for mode in ("full", "split", "off"):
        assert call.new(call.y_hat, call.u, mode, out, threads=0) == (0, 0)
        assert np.array_equal(out.view(np.uint32), out_old.view(np.uint32))


@pytest.mark.parametrize("ring_name", ["K32A12C27", "K48A20"])
def test_masked_equals_plain(ring_name: str) -> None:
    """prepare -> provider -> finish (split check) == dequantize(W8 @ xq) of the plain quantized path."""
    ring = RingParams.from_name(ring_name)
    call = Call(ring, 3, 257, 300, 2, seed=11)
    xq, sx = quantize_activations(torch.from_numpy(call.x), ring, call.wnorm if ring.cap else None)
    yq = xq.numpy() @ call.w8.T  # exact int64
    plain = dequantize_outputs(torch.from_numpy(yq), sx, torch.from_numpy(call.scale), torch.from_numpy(call.bias),
                               torch.float32).numpy()
    out = np.empty((3, 257), dtype=np.float32)
    assert call.new(call.y_hat, call.u, "split", out) == (0, 0)
    assert np.array_equal(out.view(np.uint32), plain.view(np.uint32))


def test_freivalds_on_helper_thread() -> None:
    ring = RingParams.from_name("K32A12C27")
    call = Call(ring, 2, 700, 100, 2, seed=3)
    bad = bytearray(call.y_hat)
    bad[5] ^= 1
    checks = [native.fast.trip_freivalds_submit(y, call.u, call.rhs, ring.k)
              for y in (call.y_hat, bytes(bad), np.frombuffer(call.y_hat, dtype=np.uint8))]
    assert [c.wait() for c in checks] == [(0, 0), (1, 1), (0, 0)]
    assert all(c.done() for c in checks)
    u32 = call.u.astype(np.uint32)
    assert native.fast.trip_freivalds_submit(call.y_hat, u32, call.rhs, ring.k).wait() == (0, 0)
    for _ in range(50):  # dropped without wait: the handle waits for its job before releasing the buffers
        native.fast.trip_freivalds_submit(bytes(call.y_hat), call.u.copy(), call.rhs, ring.k)
    with pytest.raises(ValueError, match="rhs shape"):
        native.fast.trip_freivalds_submit(call.y_hat, call.u, call.rhs[:, :1].copy(), ring.k)
    with pytest.raises(ValueError, match="y_hat has"):
        native.fast.trip_freivalds_output(call.y_hat[:-4], call.u, call.rhs, ring.k)


def test_finish_into_argument_errors() -> None:
    ring = RingParams.from_name("K32A12C27")
    call = Call(ring, 1, 64, 32, 1, seed=1)
    out = np.empty((1, 64), dtype=np.float32)
    f = native.fast.trip_finish_into
    with pytest.raises(ValueError, match="out shape"):
        f(call.y_hat, call.wr.copy(), call.sx, call.scale, call.bias, 32, call.bound, np.empty((2, 64), np.float32))
    with pytest.raises(ValueError, match="give u with uw"):
        f(call.y_hat, call.wr.copy(), call.sx, call.scale, call.bias, 32, call.bound, out, uw=call.uw)
    with pytest.raises(ValueError, match="x_hat is required"):
        f(call.y_hat, call.wr.copy(), call.sx, call.scale, call.bias, 32, call.bound, out, u=call.u, uw=call.uw)
    with pytest.raises(ValueError, match="ring payload"):
        f(call.y_hat[:-1], call.wr.copy(), call.sx, call.scale, call.bias, 32, call.bound, out)
