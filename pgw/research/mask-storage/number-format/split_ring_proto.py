"""Prototype: storage ring K (mask W8 r stored mod 2^K) decoupled from the check ring Kw >= K.

Service gets x_hat = x + r mod 2^K (as integers in [0, 2^K)), returns z = W8 x_hat mod 2^Kw.
Gateway: Freivalds u.z == (uW8).x_hat mod 2^Kw  (u uniform in Z_{2^Kw}, secret),
         y = (z - W8 r) mod 2^K, centered; range check |y_j| <= B_t (per-token Cauchy-Schwarz bound).
Soundness per vector >= Kw - min(K-1, floor(log2(2 B_t))) bits; errors divisible by 2^K are harmless.
Compares with Kw = K (today's construction) at a small K where B_t ~ 2^(K-2).
"""
import numpy as np
rng = np.random.default_rng(1)
m, n = 64, 896
W = rng.integers(-127, 128, (m, n)).astype(np.int64)
wn = np.linalg.norm(W, axis=1).max()

def trial(K, Kw, err, trials=4000):
    M, Mw = (1 << K) - 1, (1 << Kw) - 1
    # pick activation bits so CS bound fits 2^(K-1): A adaptive
    x = rng.standard_normal(n); x[3] = 40.0
    for A in range(12, 1, -1):
        q = 2 ** (A - 1) - 1
        xq = np.round(x / np.abs(x).max() * q).astype(np.int64)
        B = wn * np.linalg.norm(xq)
        if B < 2 ** (K - 1):
            break
    fooled = harmful = 0
    for _ in range(trials):
        r = rng.integers(0, 1 << K, n)
        xh = (xq + r) & M
        z = (W @ xh) & Mw
        wr = (W @ r) & M
        e = np.zeros(m, dtype=np.int64); e[5] = err
        zt = (z + e) & Mw
        u = rng.integers(0, 1 << min(Kw, 62), m)  # Kw<=62 here; python ints avoid overflow
        uw = (u @ W) & Mw
        ok_f = (int((u.astype(object) @ zt.astype(object))) - int(uw.astype(object) @ xh.astype(object))) % (1 << Kw) == 0
        y = (zt - wr) & M
        y = np.where(y >= 1 << (K - 1), y - (1 << K), y)
        ok_r = bool((np.abs(y) <= B).all())
        wrong = not np.array_equal(y, W @ xq)
        harmful += wrong
        fooled += wrong and ok_f and ok_r
        assert err != 0 or (not wrong and ok_f and ok_r)
    return A, float(np.log2(B)), harmful / trials, fooled / trials

for K, Kw in ((22, 22), (22, 48)):
    for err in (0, 1, 1 << 20, 1 << 21, 1 << 22, (1 << 21) + (1 << 22)):
        A, lb, harm, fool = trial(K, Kw, err)
        print(f"K={K} Kw={Kw} err=2^{np.log2(err) if err else '-'}: A={A} log2B={lb:.2f} "
              f"harmful={harm:.3f} undetected-harmful={fool:.4f}")
