"""AES-256-GCM of the native segment writer and chunk reader across implementations.

Until 2026-11-26 ``pgw_fast`` used RustCrypto ``aes-gcm``; it now uses AWS-LC (``aws-lc-rs``). The at-rest format
must not change by a single bit:

* ``aead_golden_rustcrypto.bin`` was written by the RustCrypto build (``_write_golden`` with the ``pgw_fast`` of
  commit 9d574de). The current build must write the same bytes from the same inputs (new == old) and open every
  chunk of it (old -> new); ``cryptography``'s AESGCM opens every chunk too (the golden file is plain AES-GCM).
* Chunks sealed by the current build open with ``cryptography`` (new -> independent reader) and chunks sealed
  by ``cryptography`` open with ``chunk_open_into`` (the Python writer's stores -> native reader).
* Tampering with any bit of ciphertext or tag, or using another nonce, AAD or key, fails authentication and
  leaves the output zeroed; a short file is an ``OSError`` with the output zeroed.

The Rust tests in ``rust/pgw-fast/src/writer.rs`` seal and open across RustCrypto and AWS-LC in both
directions and check that the key schedule is wiped on drop.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pgw.core import native
from pgw.core.packing import pack_ring
from pgw.maskstore.format import chunk_aad, chunk_nonce

pytestmark = pytest.mark.skipif(
    not native.native_available("seal_chunks", "seal_ring_chunks", "chunk_open_into"),
    reason="pgw_fast writer not built (python rust/tools/build_fast.py)",
)

GOLDEN = Path(__file__).with_name("aead_golden_rustcrypto.bin")
#: SHA-256 of the golden file (guards against an accidental rewrite).
GOLDEN_SHA256 = "ce9e6dcd31401761368da0020ffdbe8208bceea9ab88f22fee990f1cf93abfed"
KEY = hashlib.sha256(b"pgw aead golden key").digest()
PREFIX = bytes.fromhex("5eed0a11")
DIGEST = hashlib.sha256(b"pgw aead golden header").digest()
#: plaintext lengths of the byte chunks: empty, partial / whole blocks, the assembly's 4- / 8-block strides
LENS = (0, 1, 15, 16, 17, 31, 63, 64, 65, 127, 128, 129, 255, 256, 1000, 4096, 4097, 16_385)
#: the ring chunks: (k, dtype, rows, cols, row spans)
RING = ((40, np.int64, 7, 33, ((0, 3), (3, 4))), (32, np.uint32, 5, 129, ((0, 5),)), (64, np.uint64, 3, 17, ((0, 1), (1, 2))))


def _next(index: int) -> int:
    """Chunk indices spread over the 32-bit AAD field (0, then pseudo-random, wrapping)."""
    return (index * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF


def _data(n: int, tag: str) -> bytes:
    return hashlib.shake_256(f"pgw aead golden {tag}".encode()).digest(n)


def _ring_vals(k: int, dtype: Any, rows: int, cols: int) -> np.ndarray:
    raw = np.frombuffer(_data(rows * cols * 8, f"ring{k}"), dtype=np.uint64).reshape(rows, cols)
    if k < 64:
        raw = raw & np.uint64((1 << k) - 1)
    return raw.astype(dtype) if dtype is not np.uint64 else raw.copy()


def _chunks() -> list[tuple[int, int, bytes]]:
    """(index, offset, plaintext) of every chunk of the golden file, in file order."""
    out, off, index = [], 0, 0
    for n in LENS:
        out.append((index, off, _data(n, f"chunk{index}")))
        off += n + 16
        index = _next(index)
    for k, dtype, rows, cols, spans in RING:
        vals = _ring_vals(k, dtype, rows, cols)
        for r0, nr in spans:
            out.append((index, off, pack_ring(vals[r0 : r0 + nr].astype(np.uint64), k)))
            off += nr * cols * k // 8 + 16
            index = _next(index)
    return out


def _write_golden(fd: int) -> None:
    """Write the golden file's chunks through the native writer (``seal_chunks`` and ``seal_ring_chunks``)."""
    fast = native.fast
    index, off = 0, 0
    jobs, blob = [], b""
    for n in LENS:
        jobs.append((index, off, len(blob), n))
        blob += _data(n, f"chunk{index}")
        off += n + 16
        index = _next(index)
    fast.seal_chunks(fd, bytearray(KEY), PREFIX, DIGEST, blob, jobs, 1)
    for k, dtype, rows, cols, spans in RING:
        vals = _ring_vals(k, dtype, rows, cols)
        rjobs = []
        for r0, nr in spans:
            rjobs.append((index, off, r0, nr))
            off += nr * cols * k // 8 + 16
            index = _next(index)
        assert fast.seal_ring_chunks(fd, KEY, PREFIX, DIGEST, vals, k, rjobs, 0)


def _open_native(path: Path, index: int, off: int, n: int, key: bytes = KEY, prefix: bytes = PREFIX,
                 digest: bytes = DIGEST) -> bytearray:
    out = bytearray(b"\xaa" * n)
    fd = os.open(path, os.O_RDONLY)
    try:
        native.fast.chunk_open_into(key, chunk_nonce(prefix, index), chunk_aad(digest, index), fd, off, n + 16, out)
    except (ValueError, OSError) as exc:
        exc.add_note(f"out zeroed: {out == bytearray(n)}")
        raise
    finally:
        os.close(fd)
    return out


def test_golden_file_is_plain_aes_gcm() -> None:
    data = GOLDEN.read_bytes()
    chunks = _chunks()
    assert len(data) == chunks[-1][1] + len(chunks[-1][2]) + 16
    aead = AESGCM(KEY)
    for index, off, plain in chunks:
        ct = data[off : off + len(plain) + 16]
        assert aead.decrypt(chunk_nonce(PREFIX, index), ct, chunk_aad(DIGEST, index)) == plain


def test_current_writer_writes_the_golden_bytes(tmp_path: Path) -> None:
    """new == old: the AWS-LC build writes exactly what the RustCrypto build wrote."""
    path = tmp_path / "seg"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_golden(fd)
    finally:
        os.close(fd)
    assert path.read_bytes() == GOLDEN.read_bytes()


def test_current_reader_opens_the_golden_file() -> None:
    """old -> new: every chunk the RustCrypto build wrote opens with the current ``chunk_open_into``."""
    for index, off, plain in _chunks():
        assert _open_native(GOLDEN, index, off, len(plain)) == plain


@pytest.mark.parametrize("n", [0, 1, 16, 4097, (1 << 20) + 3])
def test_native_and_cryptography_open_each_others_chunks(tmp_path: Path, n: int) -> None:
    plain = _data(n, f"x{n}")
    index = 0xFFFFFFFF if n % 2 else 12345
    nonce, aad = chunk_nonce(PREFIX, index), chunk_aad(DIGEST, index)
    # native seal -> cryptography open
    path = tmp_path / "n"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        native.fast.seal_chunks(fd, bytearray(KEY), PREFIX, DIGEST, plain, [(index, 3, 0, n)], 0)
    finally:
        os.close(fd)
    ct = path.read_bytes()[3:]
    assert AESGCM(KEY).decrypt(nonce, ct, aad) == plain
    # cryptography seal (the Python writer's stores) -> native open
    path2 = tmp_path / "p"
    path2.write_bytes(b"\x00" * 5 + AESGCM(KEY).encrypt(nonce, plain, aad))
    assert _open_native(path2, index, 5, n) == plain
    assert path2.read_bytes()[5:] == ct


def _expect_auth_failure(path: Path, index: int, off: int, n: int, **kw: bytes) -> None:
    with pytest.raises(ValueError, match="auth") as ei:
        _open_native(path, index, off, n, **kw)
    assert "out zeroed: True" in ei.value.__notes__


@pytest.mark.parametrize("n", [0, 1, 17, 4096, 65_537])
def test_tampering_fails_and_leaves_no_plaintext(tmp_path: Path, n: int) -> None:
    plain = _data(n, f"t{n}")
    index = 77
    good = AESGCM(KEY).encrypt(chunk_nonce(PREFIX, index), plain, chunk_aad(DIGEST, index))
    path = tmp_path / "c"
    path.write_bytes(good)
    assert _open_native(path, index, 0, n) == plain
    # every flipped bit position class: first / middle / last ciphertext byte, first / last tag byte
    for pos in sorted({0, n // 2, max(n - 1, 0), n, n + 15}):
        for bit in (0x01, 0x80):
            bad = bytearray(good)
            bad[pos] ^= bit
            path.write_bytes(bad)
            _expect_auth_failure(path, index, 0, n)
    path.write_bytes(good)
    _expect_auth_failure(path, index + 1, 0, n)  # another chunk's nonce + AAD (reordered chunks)
    _expect_auth_failure(path, index, 0, n, prefix=bytes.fromhex("5eed0a12"))  # another segment's nonce prefix
    _expect_auth_failure(path, index, 0, n, digest=hashlib.sha256(b"other header").digest())  # header changed
    _expect_auth_failure(path, index, 0, n, key=bytes(a ^ (i == 31) for i, a in enumerate(KEY)))  # other DEK
    if n:
        # a chunk cut by one byte (tag moved up) or with the wrong length
        path.write_bytes(good[: n - 1] + good[n:])
        _expect_auth_failure(path, index, 0, n - 1)
    # a short file: OSError, output zeroed
    path.write_bytes(good[:-1])
    with pytest.raises(OSError) as ei:
        _open_native(path, index, 0, n)
    assert "out zeroed: True" in ei.value.__notes__


def test_golden_sha256() -> None:
    assert hashlib.sha256(GOLDEN.read_bytes()).hexdigest() == GOLDEN_SHA256


if __name__ == "__main__":  # regenerate (only with the build the golden file stands for)
    fd = os.open(GOLDEN, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        _write_golden(fd)
    finally:
        os.close(fd)
    print(GOLDEN, hashlib.sha256(GOLDEN.read_bytes()).hexdigest())
