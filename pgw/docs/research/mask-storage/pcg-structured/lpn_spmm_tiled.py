"""Column-grouped (CSC) tiled kernel: each table tile is streamed once per batch of B tokens."""
import os, sys, time
os.environ.setdefault("NUMBA_NUM_THREADS", "2"); os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, numba
K = 40; MASK = (1 << K) - 1
m, n = int(os.environ.get("M", 4096)), int(os.environ.get("N", 3584))
N = int(n * float(os.environ.get("RATIO", 2.0))); t = int(os.environ.get("T", 200)); B = int(os.environ.get("B", 512)); TILE = int(os.environ.get("TILE", 256))
rng = np.random.default_rng(3)
T = rng.integers(0, 1 << K, size=(N, m), dtype=np.uint64)
supp = np.stack([rng.choice(N, size=t, replace=False) for _ in range(B)]).astype(np.int64)
vals = (rng.integers(0, 1 << (K-1), size=(B, t), dtype=np.uint64) * np.uint64(2) + np.uint64(1))
# CSC grouping (cheap, O(B t log))
flat = supp.ravel(); order = np.argsort(flat, kind="stable")
cols = flat[order]; toks = (order // t).astype(np.int64); vv = vals.ravel()[order]
ptr = np.searchsorted(cols, np.arange(N + 1)).astype(np.int64)
@numba.njit(parallel=True, cache=True)
def kern(T, ptr, toks, vv, out, TILE):
    N, mm = T.shape; ntiles = (mm + TILE - 1) // TILE
    for ti in numba.prange(ntiles):
        k0 = ti * TILE; k1 = min(mm, k0 + TILE)
        for j in range(N):
            for h in range(ptr[j], ptr[j + 1]):
                b = toks[h]; v = vv[h]
                for k in range(k0, k1):
                    out[b, k] += T[j, k] * v
out = np.zeros((B, m), dtype=np.uint64)
kern(T, ptr, toks, vv, out, TILE); out[:] = 0
t0 = time.time(); kern(T, ptr, toks, vv, out, TILE); out &= np.uint64(MASK); ts = (time.time() - t0) / B
ref = (T[supp[5]] * vals[5][:, None]).sum(0) & np.uint64(MASK); assert np.array_equal(ref, out[5])
print(f"tiled m={m} n={n} N={N} t={t} B={B} tile={TILE}: {ts*1e3:.3f} ms/token ({ts/(m*t)*1e9:.3f} ns/mul-add), exact")
