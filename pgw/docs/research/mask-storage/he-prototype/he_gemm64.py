"""RLWE mask preprocessing over power-of-two rings (PGW-native).  Q = 2^64, t = 2^40, N = 4096.

ct_i = (a_i, b_i = a_i*s + e_i + 2^24 * r_i  mod 2^64) packs mask r_i of N future tokens (coefficient i of
input feature i over tokens).  Server: Y = W @ B mod 2^64 (exactly PGW ringmm at K=64: 8 int8 GEMMs on GPU).
Download: b' = round(Y / 2^20) (44 bits).  Variant A also returns a' = round(W A / 2^14) (50 bits);
variant B keeps a_i fixed and the client holds the hint H = W A mod 2^64 (one-time).
Client: v = b'*2^20 - (a*s) mod 2^64 (or mod 2^50 for A), m = round(v / 2^24) mod 2^40 == (W r) mod 2^40.
"""
import os, sys, time, json
import numpy as np
U = np.uint64; F64 = np.float64
N = 4096; TB = 40; QB = 64; DB = QB - TB          # Delta = 2^24
KB = 20                                           # dropped bits of b on download -> 44-bit b'
KA = 14                                           # variant A: a' kept at 50 bits
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)

def polymul64(A, s, chunk=16, block=512):
    """Negacyclic A[r,:]*s mod (X^N+1, 2^64). A uint64, s small signed. Exact via float FFT on 16-bit chunks."""
    A = np.atleast_2d(A); S = np.fft.rfft(s.astype(F64), 2 * N)
    out = np.empty(A.shape, dtype=U)
    for r0 in range(0, A.shape[0], block):
        Ab = A[r0:r0 + block]; acc = np.zeros(Ab.shape, dtype=U)
        for c in range(64 // chunk):
            ch = ((Ab >> U(c * chunk)) & U((1 << chunk) - 1)).astype(F64)
            cv = np.rint(np.fft.irfft(np.fft.rfft(ch, 2 * N, axis=1) * S, 2 * N, axis=1)).astype(np.int64)
            cv = (cv[:, :N] - cv[:, N:]).astype(U)          # two's complement wrap == mod 2^64
            acc += cv << U(c * chunk)
        out[r0:r0 + block] = acc
    return out

def gemm64(Wf, X, chunk=22, colblock=512):
    """W @ X mod 2^64; X uint64. float64 GEMM per 22-bit chunk is exact (|sum| < 2^7*2^22*2^14.3 < 2^53)."""
    Y = np.empty((Wf.shape[0], X.shape[1]), dtype=U)
    for c0 in range(0, X.shape[1], colblock):
        Xb = X[:, c0:c0 + colblock]; nb = Xb.shape[1]
        parts = [((Xb >> U(k)) & U((1 << chunk) - 1)).astype(F64) for k in range(0, 64, chunk)]
        Z = Wf @ np.concatenate(parts, axis=1)
        acc = np.zeros((Wf.shape[0], nb), dtype=U)
        for i, k in enumerate(range(0, 64, chunk)):
            acc += np.rint(Z[:, i * nb:(i + 1) * nb]).astype(np.int64).astype(U) << U(k)
        Y[:, c0:c0 + nb] = acc
    return Y

def rshift_round(X, k):
    return (X + U(1 << (k - 1))) >> U(k)

def ref_wr(W, R):
    Wf = W.astype(F64); m = (1 << 20) - 1
    lo = np.rint(Wf @ (R & m).astype(F64)).astype(np.int64) % (1 << TB)
    hi = np.rint(Wf @ (R >> 20).astype(F64)).astype(np.int64) % (1 << TB)
    return (lo + (hi << 20)) % (1 << TB)

def run(W, label, variants=('A', 'B'), sigma=3.2, seed=0):
    rng = np.random.default_rng(seed)
    dout, din = W.shape; Wf = W.astype(F64)
    R = rng.integers(0, 1 << TB, size=(din, N), dtype=np.int64)
    ref = ref_wr(W, R)
    assert ((W @ R[:, :16]) % (1 << TB) == ref[:, :16]).all()
    res = dict(label=label, N=N, logQ=QB, t_bits=TB, d_in=din, d_out=dout, sigma=sigma,
               threads=os.environ.get('OPENBLAS_NUM_THREADS'))
    for var in variants:
        r = {}
        s = rng.integers(-1, 2, size=N)
        # public a: A = fresh per batch from seed; B = fixed forever
        aseed = 1234 if var == 'A' else 99   # B and C share the fixed public a (and the hint)
        t0 = time.perf_counter()
        a = np.random.default_rng(aseed).integers(0, 2**64, size=(din, N), dtype=np.uint64)
        t_gen = time.perf_counter() - t0
        if var in ('B', 'C'):
            t0 = time.perf_counter(); H = gemm64(Wf, a); r['hint_gen_s_one_time'] = time.perf_counter() - t0
            r['hint_bytes_this_linear'] = H.nbytes
        # ---- client encrypt ----
        t0 = time.perf_counter()
        e = np.clip(np.rint(rng.normal(0, sigma, size=(din, N))), -6 * sigma, 6 * sigma).astype(np.int64).astype(U)
        if var != 'C':
            B = polymul64(a, s) + e + (R.astype(U) << U(DB)); Rv, refv = R, ref
        else:
            # C: upload compression. u = public PRG; client sends only c (25 bits) with b = u + c, and the
            # mask is DEFINED as r = (b - a*s - e) / 2^24, i.e. the high 40 bits of u - a*s.
            u = np.random.default_rng(4242).integers(0, 2**64, size=(din, N), dtype=np.uint64)
            d = u - polymul64(a, s)
            rem = (d & U((1 << DB) - 1)).astype(np.int64); rem = np.where(rem >= (1 << (DB - 1)), rem - (1 << DB), rem)
            c = e.astype(np.int64) - rem                      # |c| < 2^23 + 6 sigma  -> 25-bit signed
            r['upload_c_max_abs_log2'] = float(np.log2(np.abs(c).max()))
            B = u + c.astype(U)
            Rv = (((d - rem.astype(U)) >> U(DB)) & U((1 << TB) - 1)).astype(np.int64)
            refv = ref_wr(W, Rv)
            r['derived_mask_mean_over_2^40'] = float(Rv.mean() / 2**TB)
        r['client_encrypt_s'] = time.perf_counter() - t0 + (t_gen if var == 'A' else 0)
        # ---- server ----
        t0 = time.perf_counter()
        Yb = gemm64(Wf, B)
        if var == 'A':
            a_srv = a  # server re-expands a from the same public seed (cost = t_gen, counted below)
            Ya = gemm64(Wf, a_srv)
        t_gemm = time.perf_counter() - t0 + (t_gen if var == 'A' else 0)
        t0 = time.perf_counter()
        Bd = rshift_round(Yb, KB)                      # 44-bit b'
        if var == 'A': Ad = rshift_round(Ya, KA) & U((1 << (QB - KA)) - 1)   # 50-bit a'
        t_sw = time.perf_counter() - t0
        # ---- client decrypt ----
        t0 = time.perf_counter()
        if var == 'A':
            mod = QB - KA                               # work mod 2^50
            v = ((Bd << U(KB - KA)) - polymul64(Ad, s)) & U((1 << mod) - 1)
            sh = mod - TB
        else:
            v = (Bd << U(KB)) - polymul64(H, s)         # mod 2^64 by wrap
            sh = DB
        m = (rshift_round(v, sh) & U((1 << TB) - 1)).astype(np.int64)
        r['client_decrypt_s'] = time.perf_counter() - t0
        # noise margin: |v - m*2^sh| / 2^sh  (must be < 0.5)
        if var == 'A':
            diff = (v.astype(np.int64) - (m.astype(np.int64) << sh)) 
            diff = ((diff + (1 << (mod - 1))) % (1 << mod)) - (1 << (mod - 1))
        else:
            diff = (v - (m.astype(U) << U(sh))).astype(np.int64)
        r['max_noise_over_Delta(<0.5 required)'] = float(np.abs(diff).max() / 2**sh)
        r['exact_match'] = bool((m == refv).all())
        r['server_gemm_s'] = t_gemm; r['server_switch_s'] = t_sw
        r['server_s_per_token'] = (t_gemm + t_sw) / N
        r['server_ns_per_token_MAC'] = (t_gemm + t_sw) / N / (din * dout) * 1e9
        r['client_s_per_token'] = (r['client_encrypt_s'] + r['client_decrypt_s']) / N
        r['client_ns_per_token_per_elem(enc per in, dec per out)'] = [r['client_encrypt_s'] / N / din * 1e9, r['client_decrypt_s'] / N / dout * 1e9]
        r['upload_bytes_per_token'] = din * (QB if var != 'C' else 25) / 8
        r['download_bytes_per_token'] = dout * ((QB - KB) + ((QB - KA) if var == 'A' else 0)) / 8
        r['stored_Wr_bytes_per_token_K40_today'] = dout * 5
        res['variant_' + var] = r
        print(label, var, json.dumps(r, indent=1), flush=True)
    return res

if __name__ == '__main__':
    which = sys.argv[1]; vs = tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 else ('A', 'B')
    if which == 'qwen05_gate':
        from safetensors.numpy import load_file
        Wall = load_file('/root/.pgw-client-demo/packs/qwen2.5-0.5b-instruct/linears-00000-of-00001.safetensors')['linears.0.gate_up.weight']
        out = run(Wall[:4864].astype(np.int64), 'qwen2.5-0.5b L0 gate 4864x896 real int8', vs)
    elif which == 'slice7b':
        W = np.random.default_rng(7).integers(-127, 128, size=(4608, 3584)).astype(np.int64)
        out = run(W, '7B qkv-shaped 4608x3584 random full-range int8', vs)
    elif which == 'down_noise':
        g = np.random.default_rng(8); W = np.where(g.random((64, 18944)) < .5, -127, 127).astype(np.int64)
        out = run(W, '7B down-shaped worst-case |W|=127, 64x18944', vs)
    json.dump(out, open(f'results64_{which}_{"".join(vs)}.json', 'w'), indent=1)
