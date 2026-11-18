"""Prototype: dual-LPN structured masks r = H e (H = [I | H'] public, e t-sparse with odd entries).
Gateway computes W r = W[:,S1] e1 + (W H')[:,S2] e2 from a one-time table T = W H' (mod 2^K).
Checks exactness vs dense ring matmul and times both on 2 threads."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, "/home/user/llm_training_exploration/pgw/src")
from pgw.core.ringmm import ringmm, PreparedWeight
from pgw.core.ring import RingParams
K = 40; MASK = np.uint64((1 << K) - 1); P = RingParams(k=K, a=12)
rng = np.random.default_rng(1)
m, n = int(os.environ.get("M", 4096)), int(os.environ.get("N", 3584))
ratio = float(os.environ.get("RATIO", 2.0)); t = int(os.environ.get("T", 200))
Nn = int(n * ratio) - n                       # extra columns (H' is n x Nn)
W = rng.integers(-127, 128, size=(m, n), dtype=np.int8)
Hp = rng.integers(0, 1 << K, size=(n, Nn), dtype=np.uint64)          # public, seeded in practice
# one-time table T = W H' mod 2^K   (stored transposed: Nn x m so each column is contiguous)
t0 = time.time()
Wt = torch.from_numpy(W)
T_T = np.empty((Nn, m), dtype=np.uint64)
pw = PreparedWeight(Wt, device="cpu")
for j0 in range(0, Nn, 512):
    blk = torch.from_numpy(Hp[:, j0:j0+512].T.astype(np.int64))   # rows = columns of H'
    T_T[j0:j0+512] = pw.matmul(blk, P).numpy().astype(np.uint64) & MASK
t_table = time.time() - t0
W_T64 = W.T.astype(np.int64).astype(np.uint64)               # n x m, columns of W contiguous

def sample_e():
    supp = rng.choice(n + Nn, size=t, replace=False)
    vals = (rng.integers(0, 1 << (K - 1), size=t, dtype=np.uint64) * np.uint64(2) + np.uint64(1)) & MASK
    return supp, vals
def mask_from_e(supp, vals):
    r = np.zeros(n, dtype=np.uint64)
    s1 = supp < n; r[supp[s1]] += vals[s1]
    s2 = ~s1
    r += (Hp[:, supp[s2] - n] * vals[s2]).sum(axis=1)       # uint64 wraps mod 2^64
    return r & MASK
def Wr_sparse(supp, vals):
    s1 = supp < n; s2 = ~s1
    y = (W_T64[supp[s1]] * vals[s1, None]).sum(axis=0)
    y += (T_T[supp[s2] - n] * vals[s2, None]).sum(axis=0)
    return y & MASK

# exactness
for _ in range(3):
    supp, vals = sample_e(); r = mask_from_e(supp, vals)
    dense = ringmm(torch.from_numpy(r.astype(np.int64))[None], Wt, P).numpy()[0].astype(np.uint64) & MASK
    assert np.array_equal(dense, Wr_sparse(supp, vals)), "mismatch"
print("exactness: OK (3 random masks, W r via table == dense ring matmul)")
B = 32
es = [sample_e() for _ in range(B)]
t0 = time.time(); R = np.stack([mask_from_e(*e) for e in es]); t_rgen = (time.time()-t0)/B
t0 = time.time(); [Wr_sparse(*e) for e in es]; t_sp = (time.time()-t0)/B
Rt = torch.from_numpy(R.astype(np.int64))
ringmm(Rt[:2], Wt, P)
t0 = time.time(); pw.matmul(Rt, P); t_dense = (time.time()-t0)/B
print(f"m={m} n={n} N={n+Nn} t={t} threads=2")
print(f"table build (one-time, CPU): {t_table:.1f}s  table bytes (5B/entry) = {Nn*m*5/1e6:.1f} MB")
print(f"per-token: r regen {t_rgen*1e3:.2f} ms | W r sparse {t_sp*1e3:.2f} ms | W r dense ringmm (batch {B}) {t_dense*1e3:.2f} ms")
print(f"ns per output element: sparse {t_sp/m*1e9:.1f}  dense {t_dense/m*1e9:.1f}  speedup {t_dense/t_sp:.1f}x")
