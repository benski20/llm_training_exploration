"""Batched sparse W r = T @ E (T: one-time table, E: t-sparse per token) with numba, 2 threads,
vs dense exact ring matmul (pgw ringmm, cpu_f64 backend) on the same shapes. Exactness checked."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("NUMBA_NUM_THREADS", "2")
import numpy as np, torch, numba
torch.set_num_threads(2)
sys.path.insert(0, "/home/user/llm_training_exploration/pgw/src")
from pgw.core.ringmm import PreparedWeight
from pgw.core.ring import RingParams
K = 40; MASK = (1 << K) - 1; P = RingParams(k=K, a=12)
m, n = int(os.environ.get("M", 4096)), int(os.environ.get("N", 3584))
N = int(n * float(os.environ.get("RATIO", 2.0))); t = int(os.environ.get("T", 200)); B = int(os.environ.get("B", 64))
rng = np.random.default_rng(2)
T = rng.integers(0, 1 << K, size=(N, m), dtype=np.uint64)   # stands for [W^T ; (W H')^T] rows (column j of W H)
@numba.njit(parallel=True, fastmath=False, cache=True)
def spmm(T, supp, vals, out):
    Bn, tt = supp.shape; mm = T.shape[1]
    for b in numba.prange(Bn):
        acc = np.zeros(mm, dtype=np.uint64)
        for i in range(tt):
            row = T[supp[b, i]]; v = vals[b, i]
            for k in range(mm):
                acc[k] += row[k] * v
        for k in range(mm):
            out[b, k] = acc[k] & np.uint64(MASK)
supp = np.stack([rng.choice(N, size=t, replace=False) for _ in range(B)]).astype(np.int64)
vals = (rng.integers(0, 1 << (K-1), size=(B, t), dtype=np.uint64) * np.uint64(2) + np.uint64(1))
out = np.empty((B, m), dtype=np.uint64)
spmm(T, supp[:2], vals[:2], out[:2])
# exactness of kernel vs numpy
ref = (T[supp[0]] * vals[0][:, None]).sum(0) & np.uint64(MASK)
spmm(T, supp, vals, out); assert np.array_equal(ref, out[0])
t0 = time.time(); spmm(T, supp, vals, out); ts = (time.time() - t0) / B
W = torch.from_numpy(rng.integers(-127, 128, size=(m, n), dtype=np.int8)); pw = PreparedWeight(W, device="cpu")
R = torch.from_numpy(rng.integers(0, 1 << K, size=(B, n), dtype=np.int64)); pw.matmul(R[:2], P)
t0 = time.time(); pw.matmul(R, P); td = (time.time() - t0) / B
print(f"m={m} n={n} N={N} t={t} B={B}: sparse {ts*1e3:.3f} ms/token ({ts/(m*t)*1e9:.2f} ns/mul-add) | dense ringmm {td*1e3:.3f} ms/token ({td/(m*n)*1e12:.1f} ps/MAC) | speedup {td/ts:.1f}x")
