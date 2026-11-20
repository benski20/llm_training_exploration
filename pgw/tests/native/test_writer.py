"""Native segment writer (``pgw_fast.seal_chunks`` / ``seal_ring_chunks``) is byte-identical to the
Python path (``AESGCM`` + ``pack_ring`` + ``pwrite``), and its segments read back with the existing
readers (``SegmentFile``, ``MaskReservation.take``, ``pgw_fast.unpack_rows``)."""

from __future__ import annotations

import hashlib
import os
import random
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pgw.core import RING_K32_A12_C27, RING_K40_A12, RING_K48_A20, RingParams, SeedExpander, native
from pgw.core.errors import MaskStoreError, ShapeError
from pgw.core.packing import pack_ring, unpack_ring
from pgw.maskstore import MaskStore, StaticKeyProvider
from pgw.maskstore.format import SEGMENT_SUFFIX, SegmentFile, chunk_aad, chunk_nonce
from pgw.maskstore.keys import WrappedKey
from pgw.maskstore.segment_io import RawSegmentWriter
from pgw.maskstore.store import SegmentWriter
from tests.maskstore.conftest import FP, ToyModel, make_model

pytestmark = pytest.mark.skipif(
    not native.native_available("seal_chunks", "seal_ring_chunks"),
    reason="pgw_fast writer not built (python rust/tools/build_fast.py)",
)

KEY = bytes(range(32))
PREFIX = bytes.fromhex("a1b2c3d4")
DIGEST = hashlib.sha256(b"header").digest()
KS = (8, 16, 24, 32, 40, 48, 56, 64)


def _ref_chunk(plain: bytes, index: int, key: bytes = KEY) -> bytes:
    return AESGCM(key).encrypt(chunk_nonce(PREFIX, index), plain, chunk_aad(DIGEST, index))


def _rand_ring(rng: np.random.Generator, shape: tuple[int, int], k: int) -> np.ndarray:
    u = rng.integers(0, 1 << 63, size=shape, dtype=np.int64).astype(np.uint64) << np.uint64(1)
    u |= rng.integers(0, 2, size=shape, dtype=np.int64).astype(np.uint64)
    if k < 64:
        u &= np.uint64((1 << k) - 1)
    return u.view(np.int64) if k < 64 else u  # K64 needs uint64 (>= 2^63 is not a valid int64 value)


def _open(path: Path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    return fd


# -------------------------------------------------------------------------------------------------
# the native functions themselves
# -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("threads", [0, 1, 3])
def test_seal_chunks_matches_aesgcm(tmp_path: Path, threads: int) -> None:
    rng = random.Random(1)
    data = os.urandom(300_000)
    jobs, off, cuts = [], 7, sorted(rng.sample(range(1, len(data)), 40))
    for i, (a, b) in enumerate(zip([0, *cuts], [*cuts, len(data)], strict=True)):
        jobs.append((i * 3 + 1, off, a, b - a))
        off += b - a + 16
    jobs.append((0xFFFFFFFF, off, 5, 0))  # empty chunk (tag only), largest index
    rng.shuffle(jobs)
    path = tmp_path / "f"
    fd = _open(path)
    try:
        native.fast.seal_chunks(fd, bytearray(KEY), PREFIX, DIGEST, memoryview(data), jobs, threads)
    finally:
        os.close(fd)
    got = path.read_bytes()
    for idx, o, a, n in jobs:
        assert got[o : o + n + 16] == _ref_chunk(data[a : a + n], idx)
    assert got[:7] == bytes(7)


@pytest.mark.parametrize("k", KS)
def test_seal_ring_chunks_matches_pack_and_aesgcm(tmp_path: Path, k: int) -> None:
    rng = np.random.default_rng(k)
    rows, cols = 37, 129
    vals = _rand_ring(rng, (rows, cols), k)
    row_bytes = cols * k // 8
    spans = [(0, 5), (5, 1), (6, 30), (36, 1)]
    jobs, off = [], 0
    for i, (r0, n) in enumerate(spans):
        jobs.append((10 + i, off, r0, n))
        off += n * row_bytes + 16
    for threads in (0, 1, 4):
        path = tmp_path / f"f{threads}"
        fd = _open(path)
        try:
            assert native.fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, vals, k, jobs, threads) is True
        finally:
            os.close(fd)
        got = path.read_bytes()
        for idx, o, r0, n in jobs:
            plain = pack_ring(vals[r0 : r0 + n], k)
            assert got[o : o + len(plain) + 16] == _ref_chunk(plain, idx)
            back = native.fast.unpack_rows(plain, 0, n, cols, k)  # the gateway's reader
            np.testing.assert_array_equal(back.view(np.uint64), vals[r0 : r0 + n].view(np.uint64))


@pytest.mark.parametrize("k", [8, 16, 24, 32])
@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.uint64])
def test_seal_ring_chunks_compact_dtypes(tmp_path: Path, k: int, dtype: type) -> None:
    rng = np.random.default_rng(3)
    vals64 = _rand_ring(rng, (9, 33), k)
    vals = vals64.astype(np.uint64).astype(dtype)  # same values (low 32 bits for int32)
    path = tmp_path / "f"
    fd = _open(path)
    try:
        assert native.fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, vals, k, [(1, 0, 0, 9)])
    finally:
        os.close(fd)
    assert path.read_bytes() == _ref_chunk(pack_ring(vals64, k), 1)


def test_seal_ring_chunks_range_check_writes_nothing(tmp_path: Path) -> None:
    rows, cols = 64, 1000
    path = tmp_path / "f"
    path.write_bytes(bytes(rows * cols * 4 + 64))
    for k, dtype, bad in [(32, np.int64, 1 << 32), (32, np.int64, -1), (40, np.int64, 1 << 40),
                          (24, np.int32, 1 << 24), (24, np.uint32, 1 << 31), (16, np.uint64, 1 << 63),
                          (64, np.int64, -5)]:
        vals = np.zeros((rows, cols), dtype=dtype)
        vals[rows - 1, cols - 1] = bad  # outside every job's rows: still refused
        fd = os.open(path, os.O_RDWR)
        try:
            ok = native.fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, vals, k, [(1, 0, 0, 4)], 2)
            assert ok is False, (k, dtype, bad)
            assert native.fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, vals, k, []) is False
        finally:
            os.close(fd)
        assert path.read_bytes() == bytes(rows * cols * 4 + 64)
    assert native.fast.seal_ring_chunks(1, KEY, PREFIX, DIGEST, np.zeros((2, 2), np.int64), 32, []) is True


def test_native_argument_errors(tmp_path: Path) -> None:
    fast = native.fast
    v = np.zeros((4, 4), dtype=np.int64)
    fd = _open(tmp_path / "f")
    try:
        with pytest.raises(ValueError, match="32 bytes"):
            fast.seal_chunks(fd, KEY[:31], PREFIX, DIGEST, b"x", [(0, 0, 0, 1)])
        with pytest.raises(ValueError, match="4 bytes"):
            fast.seal_chunks(fd, KEY, PREFIX + b"\0", DIGEST, b"x", [(0, 0, 0, 1)])
        with pytest.raises(ValueError, match="32 bytes"):
            fast.seal_chunks(fd, KEY, PREFIX, DIGEST[:8], b"x", [(0, 0, 0, 1)])
        with pytest.raises(ValueError, match="exceed"):
            fast.seal_chunks(fd, KEY, PREFIX, DIGEST, b"xy", [(0, 0, 1, 2)])
        with pytest.raises(ValueError, match="32-bit"):
            fast.seal_chunks(fd, KEY, PREFIX, DIGEST, b"x", [(1 << 32, 0, 0, 1)])
        with pytest.raises(ValueError, match="descriptor"):
            fast.seal_chunks(-1, KEY, PREFIX, DIGEST, b"x", [(0, 0, 0, 1)])
        with pytest.raises(ValueError, match="exceed"):
            fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, v, 32, [(0, 0, 3, 2)])
        with pytest.raises(ValueError, match="ring width"):
            fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, v, 12, [])
        with pytest.raises(ValueError, match="K <= 32"):
            fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, v.astype(np.int32), 40, [])
        with pytest.raises(ValueError, match="contiguous"):
            fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, np.zeros((4, 8), np.int64)[:, ::2], 32, [])
        with pytest.raises(TypeError):
            fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, v.astype(np.float64), 32, [])
    finally:
        os.close(fd)
    rd = os.open(tmp_path / "f", os.O_RDONLY)
    try:
        with pytest.raises(OSError):
            fast.seal_chunks(rd, KEY, PREFIX, DIGEST, b"x", [(0, 0, 0, 1)])
    finally:
        os.close(rd)


# -------------------------------------------------------------------------------------------------
# RawSegmentWriter: native vs Python, whole files
# -------------------------------------------------------------------------------------------------
def _header_and_dek(store: MaskStore, kp: StaticKeyProvider, model: ToyModel, ring: RingParams, slots: int,
                    chunk_slots: int, checks: int = 2):
    w = store.writer(model.fingerprint, ring, model.specs, slots=slots, chunk_slots=chunk_slots, checks=checks)
    header = w.header
    w.abort()
    dek = kp.unwrap(WrappedKey(header.kek_id, header.wrapped_dek_bytes))
    return header, dek


def test_raw_writer_native_equals_python(tmp_path: Path) -> None:
    kp = StaticKeyProvider(os.urandom(32))
    store = MaskStore.open(tmp_path / "s", kp, low_water_slots=0)
    try:
        model = make_model()
        header, dek = _header_and_dek(store, kp, model, RING_K40_A12, slots=11, chunk_slots=3)
    finally:
        store.close()
    files = {}
    for use_native in (True, False):
        raw = RawSegmentWriter(tmp_path / f"seg{use_native}{SEGMENT_SUFFIX}", header, dek, fsync=False,
                               use_native=use_native)
        assert raw.native is use_native
        refs = raw.layout.chunks()
        random.Random(5).shuffle(refs)  # any order
        for ref in refs:
            plain = hashlib.shake_256(struct.pack("<I", ref.index)).digest(ref.plain_len)
            # all accepted plaintext forms
            form = ref.index % 3
            raw.write_chunk(ref, plain if form == 0 else bytearray(plain) if form == 1 else memoryview(plain))
        with pytest.raises(MaskStoreError, match="twice"):
            raw.write_chunk(refs[0], hashlib.shake_256(struct.pack("<I", refs[0].index)).digest(refs[0].plain_len))
        raw.publish()
        assert raw._key == bytearray(len(raw._key))  # wiped
        files[use_native] = raw.final_path.read_bytes()
    assert files[True] == files[False]
    with SegmentFile(tmp_path / f"segTrue{SEGMENT_SUFFIX}", dek) as seg:
        for ref in seg.iter_chunks():
            assert seg.decrypt(ref) == hashlib.shake_256(struct.pack("<I", ref.index)).digest(ref.plain_len)


# -------------------------------------------------------------------------------------------------
# SegmentWriter (store): native vs Python, whole files and read back
# -------------------------------------------------------------------------------------------------
def _segment_rows(model: ToyModel, ring: RingParams, slots: int, seed: bytes, checks: int):
    exp = SeedExpander(seed)
    out = {}
    for s in model.specs:
        r = exp.ring_uniform_rows(s.in_features, ring.k, 0, slots, "r", s.id.key)
        u = exp.ring_uniform((checks, s.out_features), ring.k, "u", s.id.key)
        out[s.id.key] = (r, model.wr(s.id.key, r, ring), model.uw(s.id.key, u, ring))
    return out


def _write_all(w: SegmentWriter, model: ToyModel, rows: dict, slots: int, splits: list[int], compact: bool) -> None:
    for s in model.specs:
        _, wr, uw = rows[s.id.key]
        start = 0
        for n in splits:
            n = min(n, slots - start)
            if n <= 0:
                break
            part = wr[start : start + n]
            if compact:
                part = part.to(torch.int32)  # low 32 bits (K <= 32)
            w.write_linear(s.id, part, uw)
            start += n
        assert start == slots


@pytest.mark.parametrize(
    ("ring", "slots", "chunk_slots", "splits", "compact"),
    [
        (RING_K40_A12, 23, 4, [3] * 8, False),  # every call straddles chunks
        (RING_K40_A12, 23, 4, [8, 8, 7], False),  # whole chunks only, short last chunk
        (RING_K48_A20, 17, 5, [1, 2, 11, 3], False),
        (RING_K32_A12_C27, 30, 7, [30], False),  # one call, many chunks
        (RING_K32_A12_C27, 30, 7, [9, 13, 8], True),  # compact int32 input
        (RING_K32_A12_C27, 5, 64, [2, 3], False),  # chunk_slots > slots
    ],
)
def test_segment_writer_native_equals_python(
    tmp_path: Path, ring: RingParams, slots: int, chunk_slots: int, splits: list[int], compact: bool
) -> None:
    kp = StaticKeyProvider(os.urandom(32))
    model = make_model(layers=2, seed=slots)
    seed = os.urandom(32)
    rows = _segment_rows(model, ring, slots, seed, checks=2)
    native_store = MaskStore.open(tmp_path / "native", kp, low_water_slots=0, fsync=False)
    py_store = MaskStore.open(tmp_path / "python", kp, low_water_slots=0, fsync=False)
    try:
        w = native_store.writer(FP, ring, model.specs, slots=slots, seed=seed, checks=2, chunk_slots=chunk_slots)
        assert w._raw.native
        header = w.header
        dek = kp.unwrap(WrappedKey(header.kek_id, header.wrapped_dek_bytes))
        _write_all(w, model, rows, slots, splits, compact)
        w.commit()
        py_path = py_store._segment_dir(FP, ring.name) / f"{header.segment_id}{SEGMENT_SUFFIX}"
        pw = SegmentWriter(py_store, header, py_path, dek, seed, use_native=False)
        assert not pw._raw.native
        _write_all(pw, model, rows, slots, splits, compact)
        pw.commit()
        native_path = native_store._segment_dir(FP, ring.name) / f"{header.segment_id}{SEGMENT_SUFFIX}"
        assert native_path.read_bytes() == py_path.read_bytes()

        # read back: the raw reader + gateway unpacker, then the store's reservation path
        with SegmentFile(native_path, dek) as seg:
            for li, s in enumerate(model.specs):
                for j in range(seg.layout.chunks_per_linear):
                    plain = seg.decrypt(seg.layout.wr_chunk(li, j))
                    n = seg.layout.chunk_rows(j)
                    want = rows[s.id.key][1][j * chunk_slots : j * chunk_slots + n]
                    assert torch.equal(unpack_ring(plain, (n, s.out_features), ring), want)
                    got = native.fast.unpack_rows(plain, 0, n, s.out_features, ring.k)
                    np.testing.assert_array_equal(got, want.numpy())
        with native_store.reserve(FP, ring, slots) as res:  # consumes (and later shreds) the segment
            for s in model.specs:
                r_ref, wr_ref, _ = rows[s.id.key]
                r, wr = res.take(s.id, slots)
                assert torch.equal(r, r_ref) and torch.equal(wr, wr_ref)
    finally:
        native_store.close()
        py_store.close()


def test_segment_writer_native_errors_leave_state_unchanged(tmp_path: Path) -> None:
    """Every refused call writes and records nothing (the writer stays usable), like the Python path;
    an out-of-range ``W r`` is reported before a bad ``uW`` (same precedence)."""
    kp = StaticKeyProvider(os.urandom(32))
    ring = RING_K40_A12
    model = make_model(layers=1, seed=9)
    slots, chunk_slots = 10, 3
    seed = os.urandom(32)
    rows = _segment_rows(model, ring, slots, seed, checks=2)
    stores = [MaskStore.open(tmp_path / n, kp, low_water_slots=0, fsync=False) for n in ("a", "b")]
    try:
        w = stores[0].writer(FP, ring, model.specs, slots=slots, seed=seed, checks=2, chunk_slots=chunk_slots)
        header = w.header
        dek = kp.unwrap(WrappedKey(header.kek_id, header.wrapped_dek_bytes))
        pw = SegmentWriter(stores[1], header, stores[1]._segment_dir(FP, ring.name) / f"{header.segment_id}"
                           f"{SEGMENT_SUFFIX}", dek, seed, use_native=False)
        spec = model.specs[0]
        _, wr, uw = rows[spec.id.key]
        for writer in (w, pw):
            writer.write_linear(spec.id, wr[:2], uw)  # partial chunk pending
            bad = wr[2:7].clone()
            bad[4, 0] = 1 << 40
            with pytest.raises(ShapeError, match="canonical ring range"):
                writer.write_linear(spec.id, bad, uw)
            with pytest.raises(ShapeError, match="canonical ring range"):  # W r first, then uW
                writer.write_linear(spec.id, bad, uw + 1)
            with pytest.raises(MaskStoreError, match="changed"):
                writer.write_linear(spec.id, wr[2:7], uw + 1)
            with pytest.raises(ShapeError, match="canonical ring range"):
                writer.write_linear(spec.id, (wr[2:7] - (1 << 40)), uw)  # negative
            with pytest.raises(ShapeError, match="int64"):
                writer.write_linear(spec.id, wr[2:7].to(torch.int32), uw)  # int32 needs K <= 32
            with pytest.raises(ShapeError, match="shape"):
                writer.write_linear(spec.id, wr[2:7, :-1], uw)
            writer.write_linear(spec.id, wr[2:7], uw)
            writer.write_linear(spec.id, wr[7:], uw)
            for s in model.specs[1:]:
                _, wr2, uw2 = rows[s.id.key]
                writer.write_linear(s.id, wr2, uw2)
            writer.commit()
        a = stores[0]._segment_dir(FP, ring.name) / f"{header.segment_id}{SEGMENT_SUFFIX}"
        assert a.read_bytes() == pw._raw.final_path.read_bytes()
    finally:
        for s in stores:
            s.close()


def test_python_fallback_when_native_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.maskstore import segment_io

    monkeypatch.setattr(segment_io, "NATIVE_WRITER", False)
    kp = StaticKeyProvider(os.urandom(32))
    store = MaskStore.open(tmp_path / "s", kp, low_water_slots=0, fsync=False)
    try:
        model = make_model(layers=1)
        seed = os.urandom(32)
        rows = _segment_rows(model, RING_K40_A12, 6, seed, checks=2)
        w = store.writer(FP, RING_K40_A12, model.specs, slots=6, seed=seed, checks=2, chunk_slots=4)
        assert not w._raw.native and not w._raw._key
        _write_all(w, model, rows, 6, [5, 1], False)
        w.commit()
        with store.reserve(FP, RING_K40_A12, 6) as res:
            s = model.specs[0]
            assert torch.equal(res.take(s.id, 6)[1], rows[s.id.key][1])
    finally:
        store.close()
