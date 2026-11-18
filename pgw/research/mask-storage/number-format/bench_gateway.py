"""Gateway CPU cost per 7B token (T=1, single thread) of: bit-unpacking a truncated mask, decoding,
Freivalds checks mod 2^M, and the optional wrap check. Compared to today's 5-byte K40 decode."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
import numpy as np

L = 28
SHAPES = {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}  # (m, n)
rng = np.random.default_rng(0)


def pack(vals: np.ndarray, bits: int) -> np.ndarray:
    """Pack uint64 values of `bits` bits little-endian into a byte array."""
    b = ((vals[:, None] >> np.arange(bits, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)
    return np.packbits(b.reshape(-1), bitorder="little")


def unpack(buf: np.ndarray, bits: int, count: int) -> np.ndarray:
    """Vectorised fixed-width unpack via 64-bit windows (bits <= 56)."""
    pad = np.concatenate([buf, np.zeros(8, np.uint8)])
    idx = np.arange(count, dtype=np.int64) * bits
    byte = idx >> 3
    sh = (idx & 7).astype(np.uint64)
    # gather 8 bytes little-endian
    w = np.zeros(count, np.uint64)
    for k in range(8):
        w |= pad[byte + k].astype(np.uint64) << np.uint64(8 * k)
    return (w >> sh) & np.uint64((1 << bits) - 1)


def bench(fn, reps=5):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


res = {}
for Ky, t in [(24, 10), (26, 10), (25, 10)]:
    bits = Ky - t
    data = {}
    for g, (m, n) in SHAPES.items():
        hi = rng.integers(0, 1 << bits, m, dtype=np.uint64)
        data[g] = (pack(hi, bits), m, rng.integers(0, 1 << 40, m, dtype=np.uint64))
    maskK = np.uint64((1 << Ky) - 1)

    def decode():
        for _ in range(L):
            for g, (buf, m, yhat) in data.items():
                hi = unpack(buf, bits, m)
                y = (yhat - (hi << np.uint64(t)) - np.uint64(1 << (t - 1))) & maskK
                yc = y.astype(np.int64)
                yc[yc >= (1 << (Ky - 1))] -= 1 << Ky
    res[f"decode_trunc_K{Ky}_t{t}_bits{bits}_ms"] = bench(decode, 3) * 1e3

# today's: 5-byte little-endian K40 store
data5 = {}
for g, (m, n) in SHAPES.items():
    wr = rng.integers(0, 1 << 40, m, dtype=np.uint64)
    data5[g] = (wr.view(np.uint8).reshape(m, 8)[:, :5].copy(), m, rng.integers(0, 1 << 40, m, dtype=np.uint64))
mask40 = np.uint64((1 << 40) - 1)


def decode5():
    for _ in range(L):
        for g, (b5, m, yhat) in data5.items():
            w = np.zeros((m, 8), np.uint8)
            w[:, :5] = b5
            wr = w.view(np.uint64)[:, 0]
            y = (yhat - wr) & mask40
            yc = y.astype(np.int64)
            yc[yc >= (1 << 39)] -= 1 << 40


res["decode_today_K40_5byte_ms"] = bench(decode5, 3) * 1e3

for c in (1, 2, 4):
    chk = {}
    for g, (m, n) in SHAPES.items():
        chk[g] = (rng.integers(0, 1 << 40, (c, m), dtype=np.uint64), rng.integers(0, 1 << 40, (c, n), dtype=np.uint64),
                  rng.integers(0, 1 << 40, (1, m), dtype=np.uint64), rng.integers(0, 1 << 40, (1, n), dtype=np.uint64))

    def fre():
        for _ in range(L):
            for g, (u, uw, yh, xh) in chk.items():
                lhs = (yh @ u.T) & mask40
                rhs = (xh @ uw.T) & mask40
                _ = (lhs == rhs).all()
    res[f"freivalds_c{c}_ms"] = bench(fre, 3) * 1e3
macs = sum(m + n for m, n in SHAPES.values()) * L
res["freivalds_MACs_per_vector_per_token"] = macs
print(res)
import json
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_gateway.json"), "w"), indent=1)
