"""Native mask fetch (``pgw_fast.mask_rows`` / ``unpack_rows``) is bit-identical to the Python path."""

from __future__ import annotations

import numpy as np
import pytest

from pgw.core import native
from pgw.core.errors import ConfigError, ProtocolError, ShapeError
from pgw.core.packing import unpack_ring
from pgw.core.prng import SeedExpander, max_stream_rows

pytestmark = pytest.mark.skipif(
    not native.native_available("mask_rows", "mask_key", "mask_rows_with_key", "unpack_rows"),
    reason="pgw_fast store functions not built (python rust/tools/build_fast.py)",
)

SEED = bytes(range(32))
KS = (16, 24, 32, 40, 48, 56)
COLS = (1, 7, 3584, 18944)
LABELS = (["r", "3.qkv"], [b"x", 7], ("r", "0.down", -3), [bytearray(b"ba"), memoryview(b"mv")], [])


def _starts(cols: int, rows: int) -> list[int]:
    past_4gb = (1 << 32) // (8 * cols) + 3  # byte offset > 4 GB
    return [0, 1, 12345, past_4gb, max_stream_rows(cols) - rows]  # last: the end of the stream


def _ref(seed: bytes, labels, cols: int, k: int, start: int, rows: int) -> np.ndarray:
    return SeedExpander(seed).ring_uniform_rows(cols, k, start, rows, *labels).numpy()


def _check(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == np.int64 and got.flags.c_contiguous
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("cols", COLS)
def test_mask_rows_bit_identical(k: int, cols: int) -> None:
    fast = native.fast
    for rows in (1, 2, 17, 40):
        for start in _starts(cols, rows):
            want = _ref(SEED, ["r", "3.qkv"], cols, k, start, rows)
            _check(fast.mask_rows(SEED, ["r", "3.qkv"], cols, k, start, rows), want)
            assert int(want.max()) < 1 << k and int(want.min()) >= 0


@pytest.mark.parametrize("labels", LABELS)
def test_mask_rows_labels_and_all_row_counts(labels) -> None:
    fast = native.fast
    for cols in (1, 7, 64):
        for rows in range(1, 41):
            for start in (0, 1, 12345, (1 << 32) // (8 * cols) + 1):
                _check(fast.mask_rows(SEED, labels, cols, 40, start, rows), _ref(SEED, labels, cols, 40, start, rows))


def test_mask_key_variant_and_seed_types() -> None:
    fast = native.fast
    seed = bytes(range(100, 132))
    key = fast.mask_key(seed, ["r", "5.o"])
    assert isinstance(key, bytes) and len(key) == 32
    assert key == SeedExpander(seed)._key(("r", "5.o"))
    for cols, start, rows in ((3584, 0, 1), (18944, 777, 40), (7, (1 << 32) // 56 + 5, 3)):
        want = _ref(seed, ["r", "5.o"], cols, 48, start, rows)
        _check(fast.mask_rows_with_key(key, cols, 48, start, rows), want)
        _check(fast.mask_rows(bytearray(seed), ("r", "5.o"), cols, 48, start, rows), want)
    _check(fast.mask_rows(seed, ["r"], 5, 64, 3, 2), _ref(seed, ["r"], 5, 64, 3, 2))  # K=64 wraps like numpy
    assert fast.mask_rows(seed, ["r"], 5, 40, 3, 0).shape == (0, 5)


def test_mask_rows_errors() -> None:
    fast = native.fast
    with pytest.raises(ValueError):
        fast.mask_rows(b"short", ["r"], 4, 40, 0, 1)
    for cols, start, rows in ((0, 0, 1), (4, -1, 1), (4, 0, -1)):
        with pytest.raises(ConfigError):
            _ref(SEED, ["r"], cols, 40, start, rows)
        with pytest.raises(ValueError):
            fast.mask_rows(SEED, ["r"], cols, 40, start, rows)
    for k in (0, 65):
        with pytest.raises((ValueError, OverflowError)):
            fast.mask_rows(SEED, ["r"], 4, k, 0, 1)
    with pytest.raises(ValueError):  # beyond the 2^32-block ChaCha20 counter
        fast.mask_rows(SEED, ["r"], 4, 40, max_stream_rows(4), 1)
    with pytest.raises(ValueError):
        fast.mask_rows_with_key(bytes(31), 4, 40, 0, 1)
    for bad in ([True], [1.5], [None], [1 << 63]):
        with pytest.raises((TypeError, OverflowError)):
            fast.mask_rows(SEED, bad, 4, 40, 0, 1)
        with pytest.raises(Exception):
            _ref(SEED, bad, 4, 40, 0, 1)
    with pytest.raises(TypeError):
        fast.mask_rows(SEED, "r", 4, 40, 0, 1)  # a bare str is not a label list


@pytest.mark.parametrize("k", (8, 16, 24, 32, 40, 48, 56, 64))
def test_unpack_rows_matches_unpack_ring(k: int) -> None:
    fast = native.fast
    nb = k // 8
    rng = np.random.default_rng(k)
    for cols in (1, 7, 3584, 4608):
        total_rows = 45
        data = rng.integers(0, 256, size=total_rows * cols * nb, dtype=np.uint8).tobytes()
        row_bytes = cols * nb
        for off, rows in ((0, 1), (0, 40), (3, 17), (44, 1), (5, 40), (45, 0)):
            a, b = off * row_bytes, (off + rows) * row_bytes
            want = unpack_ring(data[a:b], (rows, cols), k).numpy()
            for buf in (data, bytearray(data), memoryview(data)):
                _check(fast.unpack_rows(buf, off, rows, cols, k), want)


def test_unpack_rows_large_parallel() -> None:
    fast = native.fast
    cols, k = 37888, 32
    data = np.random.default_rng(1).integers(0, 256, size=50 * cols * 4, dtype=np.uint8).tobytes()
    want = unpack_ring(data[3 * cols * 4 : 43 * cols * 4], (40, cols), k).numpy()
    _check(fast.unpack_rows(bytearray(data), 3, 40, cols, k), want)


def test_unpack_rows_errors() -> None:
    fast = native.fast
    buf = bytearray(10 * 7 * 5)  # 10 rows of 7 K=40 elements
    assert fast.unpack_rows(buf, 9, 1, 7, 40).shape == (1, 7)
    for off, rows in ((10, 1), (9, 2), (0, 11), (1 << 60, 1), (0, 1 << 60)):
        with pytest.raises(ValueError):
            fast.unpack_rows(buf, off, rows, 7, 40)
        with pytest.raises((ProtocolError, ValueError)):  # the Python path fails too
            unpack_ring(bytes(buf)[off * 35 : (off + rows) * 35], (rows, 7), 40)
    for off, rows, cols in ((-1, 1, 7), (0, -1, 7), (0, 1, -7)):
        with pytest.raises(ValueError):
            fast.unpack_rows(buf, off, rows, cols, 40)
    for k in (0, 12, 72):
        with pytest.raises((ValueError, OverflowError)):
            fast.unpack_rows(buf, 0, 1, 7, k)
        with pytest.raises(ShapeError):
            unpack_ring(b"", (0, 7), k)
    with pytest.raises(ValueError):  # non-contiguous view
        fast.unpack_rows(memoryview(bytes(700))[::2], 0, 1, 7, 40)
    with pytest.raises(TypeError):
        fast.unpack_rows("not bytes", 0, 1, 7, 40)
