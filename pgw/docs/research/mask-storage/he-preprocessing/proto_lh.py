"""End-to-end correctness prototype of "hinted MLWE masking" (PGW-LH) for one linear layer.

Protocol (per linear, per group of G = N' tokens, MLWE rank k, ring Z_q[X]/(X^N'+1)):
  public : A = (a_{c,l}) c<n, l<k   ring elements, from a public seed (shared by all customers)
  once   : hint H = W . A  (m x k ring elements), computed by the service, Freivalds-verified by gateway
  group  : gateway samples fresh s in R^k (small), e_c in R (small) for every input coordinate c
           pad P_c = <a_c, s> + e_c      (coefficient j of P_c = pad of token j, coordinate c)
           share C_i = <H_i, s>           (coefficient j = token j's correction for output i)
  online : gateway sends v_j = Delta*x_j - P[:, j] mod q        (x_j integer activations)
           service returns z_j = W v_j mod q                    (int8 W, exact mod q)
           gateway: y_j = round_centered((z_j + C[:, j]) / Delta)   == W x_j exactly
           Freivalds over F_q: u.z == (uW).v  (u secret, uW precomputed)
Arithmetic is done exactly with int64 limb splits (q < 2^56).
"""
import sys, time
import numpy as np
import sympy

rng = np.random.default_rng(1)

def find_prime(bits, mod):
    c = (1 << bits) // mod
    while True:
        q = c * mod + 1
        if q < (1 << bits) and sympy.isprime(q):
            return q
        c -= 1

def negacyclic_matrix(s):
    """M such that (h @ M) = coefficients of h(X)*s(X) mod X^N+1, for small integer s."""
    N = len(s)
    M = np.zeros((N, N), dtype=np.int64)
    for i in range(N):              # row i: h_i X^i * s(X)
        for j in range(N):
            d = i + j
            if d < N:
                M[i, d] += s[j]
            else:
                M[i, d - N] -= s[j]
    return M

def mulmod_small(Hmat, Smat, q):
    """(Hmat @ Smat) mod q exactly for Hmat in [0,q) (q<2^56) and small Smat, via 28-bit limbs."""
    lo = Hmat & ((1 << 28) - 1)
    hi = Hmat >> 28
    rlo = (lo @ Smat) % q
    rhi = (hi @ Smat) % q
    return (rlo + (rhi * (1 << 28)) % q) % q  # rhi < 2^56, *2^28 overflows -> do it as python ints below

def mulmod_small_exact(Hmat, Smat, q):
    lo = Hmat & ((1 << 28) - 1)
    hi = Hmat >> 28
    rlo = (lo @ Smat) % q
    rhi = (hi @ Smat) % q
    t = (rhi.astype(object) * (1 << 28) + rlo.astype(object)) % q
    return t.astype(np.int64)

def W_times_mod(W, V, q):
    """W (m,n) int8-valued int64, V (n,G) in [0,q) -> W V mod q exactly via 4 x 14-bit limbs."""
    out = np.zeros((W.shape[0], V.shape[1]), dtype=object)
    for d in range(4):
        limb = (V >> (14 * d)) & ((1 << 14) - 1)
        part = (W @ limb)                      # |.| <= n*127*2^14 < 2^39
        out = out + part.astype(object) * (1 << (14 * d))
    return (out % q).astype(np.int64)

def run(m, n, Np, k, sigma_cbd_eta, t_bits, q_bits, worst_weights):
    q = find_prime(q_bits, 2 * Np)
    t = 1 << t_bits
    Delta = q // t
    W = (rng.choice([-127, 127], size=(m, n)) if worst_weights else rng.integers(-127, 128, size=(m, n))).astype(np.int64)
    # public A: n x k ring elements -> as matrix (n, k*Np)
    A = rng.integers(0, q, size=(n, k * Np), dtype=np.int64)
    # service computes hint H = W A mod q (m, k*Np): per column block exact
    H = W_times_mod(W, A, q)
    # gateway: fresh secret s (k ring elts, CBD eta), noise e (n ring elts)
    def cbd(shape):
        return (rng.binomial(2 * sigma_cbd_eta, 0.5, size=shape) - sigma_cbd_eta).astype(np.int64)
    s = cbd((k, Np))  # eta=2 secret
    S_big = np.concatenate([negacyclic_matrix(s[l]) for l in range(k)], axis=0)  # (k*Np, Np)
    e = (rng.binomial(42, 0.5, size=(n, Np)) - 21).astype(np.int64)  # sigma~3.24 like HE standard
    P = (mulmod_small_exact(A, S_big, q) + e) % q             # (n, Np): pads, col j = token j
    C = mulmod_small_exact(H, S_big, q)                       # (m, Np): corrections
    # online: G = Np tokens with worst-case activations
    qmax = (1 << 11) - 1
    X = rng.choice([-qmax, qmax], size=(n, Np)).astype(np.int64)
    Y_true = W.astype(object) @ X.astype(object)
    assert np.max(np.abs(Y_true.astype(np.float64))) < t // 2, "t too small"
    V = (Delta * X - P) % q                                   # sent to service (uniform-looking)
    Z = W_times_mod(W, V, q)                                  # service
    R = (Z.astype(object) + C.astype(object)) % q
    R = np.where(R > q // 2, R - q, R)                        # centered
    Y = np.vectorize(lambda r: (2 * r + Delta) // (2 * Delta))(R)  # round(r / Delta)
    ok = bool((Y == Y_true).all())
    noise = (R - Y_true * Delta)
    noise_max = max(abs(int(v)) for v in noise.flat)
    # Freivalds over F_q on the service product (u secret)
    u = rng.integers(0, q, size=(1, m), dtype=np.int64)
    uW = (u.astype(object) @ W.astype(object)) % q
    lhs = (u.astype(object) @ Z.astype(object)) % q
    rhs = (uW @ V.astype(object)) % q
    frei_ok = bool((lhs == rhs).all())
    Z_bad = Z.copy(); Z_bad[3, 5] = (Z_bad[3, 5] + 1) % q
    frei_catch = not bool((((u.astype(object) @ Z_bad.astype(object)) % q) == rhs).all())
    # uniformity sanity of V: mean of V/q should be ~0.5
    return dict(m=m, n=n, Np=Np, k=k, q_bits=q.bit_length(), q=q, t_bits=t_bits, Delta_bits=round(np.log2(Delta), 2),
                exact=ok, noise_max_bits=round(np.log2(noise_max + 1), 2), noise_budget_bits=round(np.log2(Delta / 2), 2),
                freivalds_honest_pass=frei_ok, freivalds_catches_1bit_error=frei_catch,
                v_mean_over_q=round(float(np.mean(V / q)), 4), worst_weights=worst_weights)

if __name__ == "__main__":
    t0 = time.time()
    for (m, n, ww) in [(32, 3584, True), (16, 18944, True), (32, 3584, False)]:
        r = run(m=m, n=n, Np=256, k=10, sigma_cbd_eta=2, t_bits=34, q_bits=56, worst_weights=ww)
        print(r, f"{time.time()-t0:.1f}s", flush=True)
