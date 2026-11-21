"""Mask pipeline throughput paths: shared Freivalds vectors, chunk-buffer reuse, the in-place ChaCha20
pad and the batched claim (:func:`pgw.maskstore.take_checked_many`). Every fast path must give the
same rows, bit for bit, as the reference path, keep single use and zeroize what it drops."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from pgw.core import RING_K32_A12_C27, RING_K40_A12, MaskExhaustedError, MaskReuseError, MaskStoreError, RingParams
from pgw.core.prng import SeedExpander, _ring_stream_into
from pgw.maskstore import MaskStore, SegmentFile, StaticKeyProvider, take_checked_many
from pgw.maskstore import store as store_mod

from .conftest import FP, ToyModel, lid, write_segment

K32 = RING_K32_A12_C27


def _ref_stream(key: bytes, byte_offset: int, count: int, k: int) -> np.ndarray:
    """The normative construction, written out independently (copying form)."""
    block, skip = divmod(byte_offset, 64)
    enc = Cipher(algorithms.ChaCha20(key, block.to_bytes(4, "little") + bytes(12)), mode=None).encryptor()
    raw = np.frombuffer(enc.update(bytes(skip + 8 * count))[skip:], dtype="<u8")
    return (raw & np.uint64((1 << k) - 1) if k < 64 else raw).astype(np.int64)


@pytest.mark.parametrize("k", [8, 24, 32, 40, 64])
@pytest.mark.parametrize(("cols", "row_start", "rows"), [(1, 0, 3), (7, 5, 9), (13, 1, 1), (896, 3, 2),
                                                         (4864, 2, 30)])  # last: > 1 MiB (several pieces)
def test_ring_stream_into_matches_reference(k: int, cols: int, row_start: int, rows: int) -> None:
    key = os.urandom(32)
    got = SeedExpander.ring_uniform_rows_with_key(bytearray(key), cols, k, row_start, rows)
    want = _ref_stream(key, 8 * cols * row_start, cols * rows, k).reshape(rows, cols)
    assert got.dtype == torch.int64 and np.array_equal(got.numpy(), want)
    out = np.full(5, -1, dtype=np.int64)
    _ring_stream_into(key, 8 * 3 + 64 * 7, out, k)  # unaligned offset (skip inside a block)
    assert np.array_equal(out, _ref_stream(key, 8 * 3 + 64 * 7, 5, k))


def test_ring_uniform_matches_reference() -> None:
    seed = os.urandom(32)
    sx = SeedExpander(seed)
    got = sx.ring_uniform((3, 11), 32, "u", "0.qkv")
    want = _ref_stream(sx.stream_key("u", "0.qkv"), 0, 33, 32).reshape(3, 11)
    assert np.array_equal(got.numpy(), want)


def test_native_pad_choice_is_bit_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    key = bytearray(os.urandom(32))
    for cols, rows in ((64, 1), (3584, 1), (3584, 40)):
        monkeypatch.setattr(store_mod, "_NATIVE_PAD_SMALL", 0)
        monkeypatch.setattr(store_mod, "_NATIVE_PAD_LARGE", 1 << 62)
        a = store_mod._mask_rows_keyed(key, cols, 32, 17, rows)
        monkeypatch.setattr(store_mod, "_NATIVE_PAD_SMALL", 1 << 62)
        b = store_mod._mask_rows_keyed(key, cols, 32, 17, rows)
        assert torch.equal(a, b)


# ---- shared Freivalds vectors ------------------------------------------------------------------------
def test_checks_shared_per_segment_and_wiped_by_last_close(store: MaskStore, model: ToyModel) -> None:
    base = len(store_mod._SHARED_CHECKS)
    s1 = write_segment(store, model, 6, ring=K32)
    s2 = write_segment(store, model, 6, ring=K32)
    q = lid("1.down")
    a = store.reserve(FP, K32, 3)  # s1[0:3]
    b = store.reserve(FP, K32, 3)  # s1[3:6]
    c = store.reserve(FP, K32, 2)  # s2[0:2]
    ta, ua, uwa = a.checks_compact(q)
    tb, ub, uwb = b.checks_compact(q)
    tc, uc, _ = c.checks_compact(q)
    assert ta == tb == (s1, q.key) and tc == (s2, q.key)
    assert ua is ub and uwa is uwb  # one copy for the segment
    assert uc is not ua and not np.array_equal(uc, ua)
    u_a, _ = a.checks(q)
    u_b, _ = b.checks(q)
    assert torch.equal(u_a, u_b) and u_a.data_ptr() != u_b.data_ptr()  # checks() still returns copies
    assert len(store_mod._SHARED_CHECKS) == base + 2
    a.close()
    assert ub.any() and uwb.any()  # b still reads them
    r, wr = b.take(q, 3)
    u, uw = b.checks(q)
    assert torch.equal(wr, model.wr(q.key, r, K32))
    assert torch.equal(uw, model.uw(q.key, u, K32))
    b.close()
    assert not ub.any() and not uwb.any()  # last reservation of the segment closed: zeroed
    assert len(store_mod._SHARED_CHECKS) == base + 1
    c.close()
    assert not uc.any() and len(store_mod._SHARED_CHECKS) == base


def test_shared_checks_release_is_reentrant(store: MaskStore, model: ToyModel) -> None:
    """``__del__`` -> ``close`` -> ``release`` may run from a gc pass inside the registry's own critical
    section (same thread): the registry lock must be re-entrant, or that close deadlocks."""
    write_segment(store, model, 4, ring=K32)
    q = lid("1.down")
    a = store.reserve(FP, K32, 2)
    _, ua, _ = a.checks_compact(q)
    with store_mod._SHARED_CHECKS._lock:  # stands in for an allocation-triggered gc in acquire()
        a.close()
    assert not ua.any()


# ---- chunk buffer reuse ------------------------------------------------------------------------------
@pytest.mark.parametrize("out_features", [1, 3, 16, 40])  # chunks shorter than one AES block too
def test_decrypt_into_reuse_matches_and_zeroes_on_tamper(tmp_path: Path, kp: StaticKeyProvider,
                                                         out_features: int) -> None:
    from pgw.core import LinearId, LinearKind, LinearSpec

    st = MaskStore.open(tmp_path / "s", kp, low_water_slots=0)
    try:
        spec = LinearSpec(LinearId(0, LinearKind.QKV), out_features, 16)
        w = st.writer(FP, K32, [spec], slots=7, checks=0, chunk_slots=3)
        w.write_linear(spec.id, torch.randint(0, 1 << 32, (7, out_features), dtype=torch.int64),
                       torch.zeros((0, 16), dtype=torch.int64))
        sid = w.commit()
        path = next((tmp_path / "s").rglob(f"{sid}*"))
        seg = SegmentFile(path)
        seg.set_dek(st._dek_for(seg.header))
        refs = [seg.layout.wr_chunk(0, j) for j in range(seg.layout.chunks_per_linear)]
        buf = None
        for ref in refs:
            want = seg.decrypt(ref)
            if buf is not None and len(buf) == ref.plain_len:
                ident = id(buf)
                got = seg.decrypt_into(ref, buf)
                assert id(got) == ident  # decrypted in place
            else:
                got = seg.decrypt_into(ref)
            assert bytes(got) == want
            buf = got
        seg.close()
        # tamper with the first chunk's ciphertext: the reused buffer is zeroed, nothing leaks
        raw = bytearray(path.read_bytes())
        raw[refs[0].offset] ^= 1
        path.write_bytes(bytes(raw))
        seg = SegmentFile(path)
        seg.set_dek(st._dek_for(seg.header))
        out = bytearray(b"\xff" * refs[0].plain_len)
        with pytest.raises(MaskStoreError, match="authentication"):
            seg.decrypt_into(refs[0], out)
        assert not any(out)
        seg.close()
    finally:
        st.close()


# ---- batched claim -----------------------------------------------------------------------------------
def _pair(tmp_path: Path, model: ToyModel, ring: RingParams, slots: tuple[int, ...], chunk_slots: int) -> Any:
    """Two stores holding identical segments (same seeds), so a sequential and a batched consumer
    see the same slots."""
    stores = []
    seeds = [os.urandom(32) for _ in slots]
    for name in ("seq", "bat"):
        st = MaskStore.open(tmp_path / name, StaticKeyProvider(os.urandom(32)), low_water_slots=0)
        for n, seed in zip(slots, seeds, strict=True):
            write_segment(st, model, n, ring=ring, seed=seed, chunk_slots=chunk_slots)
        stores.append(st)
    return stores


def _same(a: Any, b: Any) -> None:
    assert a[0] == b[0]
    assert torch.equal(a[1], b[1]) and a[1].dtype == b[1].dtype == torch.int64
    assert torch.equal(a[2].to(torch.int64), b[2].to(torch.int64))
    assert (a[3] is None) == (b[3] is None)
    if a[3] is not None:
        # tokens name the segment id, which differs between the two stores (same seed, same slots)
        assert a[3][0][1] == b[3][0][1] and np.array_equal(a[3][1], b[3][1]) and np.array_equal(a[3][2], b[3][2])


@pytest.mark.parametrize("ring", [K32, RING_K40_A12], ids=lambda r: r.name)
@pytest.mark.parametrize("threads", [0, 1, 3])
def test_take_checked_many_equals_sequential(tmp_path: Path, model: ToyModel, ring: RingParams,
                                             threads: int, monkeypatch: pytest.MonkeyPatch) -> None:
    seq, bat = _pair(tmp_path, model, ring, (7, 9), chunk_slots=3)
    native_jobs = [0]
    if store_mod._native_claim:
        real = store_mod.native.fast.claim_rows_many

        def counting(jobs: Any, k: int, t: int) -> Any:
            native_jobs[0] += len(jobs)
            return real(jobs, k, t)

        monkeypatch.setattr(store_mod.native.fast, "claim_rows_many", counting)
    try:
        # three sessions; the second spans both segments (extent boundary at row 4)
        sizes = (3, 6, 5)
        rs = [seq.reserve(FP, ring, n, session_id=f"s{i}") for i, n in enumerate(sizes)]
        rb = [bat.reserve(FP, ring, n, session_id=f"s{i}") for i, n in enumerate(sizes)]
        keys = [s.id.key for s in model.specs]
        rb[1].prefetch(keys[0], 2)  # a staged linear takes the ordinary path inside the batch
        rs[1].prefetch(keys[0], 2)
        rng = np.random.default_rng(0)
        for step in range(8):
            reqs = []
            for si in range(len(sizes)):
                for key in keys:
                    if rs[si].remaining(key) == 0:
                        continue
                    rows = int(rng.integers(1, 4))  # 1..3 rows: some cross a 3-slot chunk
                    reqs.append((si, key, rows, bool((step + si) % 2)))
                    if step % 3 == 0 and rs[si].remaining(key) > rows:  # same pair twice in one batch
                        reqs.append((si, key, 1, True))
            if not reqs:
                break
            want = []
            for si, key, rows, chk in reqs:
                want.append(rs[si].take_checked(key, rows, chk))
            got = take_checked_many([(rb[si], key, rows, chk) for si, key, rows, chk in reqs], threads=threads)
            assert len(got) == len(want)
            for g, w in zip(got, want, strict=True):
                _same(g, w)
            for (_si, key, _, _), g in zip(reqs, got, strict=True):
                assert torch.equal(g[2].to(torch.int64), model.wr(key, g[1], ring))
        for si in range(len(sizes)):
            for key in keys:
                assert rs[si].remaining(key) == rb[si].remaining(key)
        if store_mod._native_claim and ring.k <= 32:
            assert native_jobs[0] > 20  # the batched native path did the work
        for r in (*rs, *rb):
            r.close()
    finally:
        seq.close()
        bat.close()


def test_take_checked_many_refuses_before_consuming(store: MaskStore, model: ToyModel) -> None:
    write_segment(store, model, 8, ring=K32, chunk_slots=4)
    a = store.reserve(FP, K32, 2)
    b = store.reserve(FP, K32, 2)
    c = store.reserve(FP, K32, 2)
    q, o = lid("0.qkv").key, lid("0.o").key
    c.close()
    with pytest.raises(MaskReuseError):
        take_checked_many([(a, q, 1, False), (c, q, 1, False)])
    assert a.remaining(q) == 2  # nothing of the batch was consumed
    with pytest.raises(MaskExhaustedError):
        take_checked_many([(a, q, 1, False), (b, o, 1, False), (b, o, 1, False), (b, o, 1, False)])
    assert a.remaining(q) == 2 and b.remaining(o) == 2
    got = take_checked_many([(a, q, 5, True), (b, o, 1, False), (b, o, 1, False)])
    assert [g[0] for g in got] == [2, 1, 1]
    assert a.remaining(q) == 0 and b.remaining(o) == 0
    with pytest.raises(MaskExhaustedError):
        take_checked_many([(a, q, 1, False)])
    a.close()
    b.close()


@pytest.mark.skipif(not store_mod._native_claim, reason="native claim_rows_many not built")
def test_take_checked_many_wipes_on_failure(store: MaskStore, model: ToyModel,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    write_segment(store, model, 8, ring=K32, chunk_slots=4)
    a = store.reserve(FP, K32, 4)
    q, o = lid("0.qkv").key, lid("0.o").key
    a.prefetch(q, 2)  # staged: handed out by the ordinary path before the native call
    wiped: list[Any] = []
    real_wipe = store_mod._wipe_claim

    def record(o_: Any) -> None:
        real_wipe(o_)
        wiped.append(o_)

    def boom(jobs: Any, k: int, threads: int) -> Any:
        raise ValueError("injected")

    monkeypatch.setattr(store_mod, "_wipe_claim", record)
    monkeypatch.setattr(store_mod.native.fast, "claim_rows_many", boom, raising=False)
    with pytest.raises(ValueError, match="injected"):
        take_checked_many([(a, q, 1, True), (a, o, 1, True)])
    assert len(wiped) == 1 and not wiped[0][1].any() and not wiped[0][2].any()  # the staged row: zeroed
    assert a.remaining(q) == 3 and a.remaining(o) == 3  # consumed (burned), never handed out again
    a.close()
    with pytest.raises(MaskReuseError):
        take_checked_many([(a, q, 1, True)])
