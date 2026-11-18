"""Minimal RLWE (BFV-style, secret-key, coefficient-encoded) mask-preprocessing prototype.

Packing: one ciphertext per INPUT feature i, its N=4096 coefficients are the mask values r_i of N
different future tokens (token-batched column packing). Since W only multiplies ciphertexts by
integer scalars W_ji, the server work is a plain GEMM  W (d_out x d_in) @ C (d_in x N) mod q per
RNS limb -- no rotations, no NTT on the server, and the plaintext modulus can be t = 2^40 exactly
(coefficient encoding, scalar mults are coefficient-wise mod t), i.e. the PGW 2^K ring natively.

Q = q_d * q_e with Q == 1 (mod 2^40), so Delta = (Q-1)/t exactly and the unreduced W r wraps cleanly.

Variant A ("seeded"): fresh a_i per batch from a public seed; upload b only; server returns
  (W a, W b), modulus-switched Q -> q_d, b additionally compressed to `b_bits`.
Variant B ("hint"): a_i fixed forever (public seed); server returns W b only (switched + compressed);
  client keeps hint H = W A mod Q (one-time, per model) and computes H*s per batch.
Fresh secret s per batch in both variants (B: same public a_i, different s each batch = multi-secret RLWE).
"""
import os, time, json, sys
import numpy as np

T_BITS = 40
T = 1 << T_BITS
N = 4096
F64 = np.float64

def mulmod(x, y, q):
    """x*y mod q for int64 arrays/scalars with 0<=x,y<q<2^52 (float quotient estimate, uint64 wrap)."""
    x = np.asarray(x, dtype=np.int64); y = np.asarray(y, dtype=np.int64)
    qf = np.floor(x.astype(F64) * y.astype(F64) / F64(q)).astype(np.int64)
    r = (x.astype(np.uint64) * y.astype(np.uint64) - qf.astype(np.uint64) * np.uint64(q)).astype(np.int64)
    r = np.where(r < 0, r + q, r); r = np.where(r < 0, r + q, r)
    r = np.where(r >= q, r - q, r); r = np.where(r >= q, r - q, r)
    return r

class Ctx:
    def __init__(self, qd, qe, sigma=3.2, b_bits=None, seed=0):
        self.qd, self.qe = int(qd), int(qe)
        self.Q = self.qd * self.qe
        assert self.Q % T == 1
        self.delta = (self.Q - 1) // T
        self.dmod = [self.delta % self.qd, self.delta % self.qe]
        self.qe_inv_mod_qd = pow(self.qe, -1, self.qd)
        self.sigma = sigma
        self.b_bits = b_bits or self.qd.bit_length()
        self.k = self.qd.bit_length() - self.b_bits   # dropped low bits of b' on download
        self.rng = np.random.default_rng(seed)

    # ---------- polynomial arithmetic (client side) ----------
    def polymul(self, A, s, q, chunk_bits=17, block=1024):
        """Negacyclic A[r,:] * s mod (X^N+1, q); A int64 residues < q (<2^51), s small ints. Exact via float FFT."""
        A = np.atleast_2d(A)
        S = np.fft.rfft(s.astype(F64), 2 * N)
        out = np.empty_like(A)
        nch = -(-q.bit_length() // chunk_bits)
        mask = (1 << chunk_bits) - 1
        for r0 in range(0, A.shape[0], block):
            Ab = A[r0:r0 + block]
            acc = None
            for c in reversed(range(nch)):
                ch = ((Ab >> (c * chunk_bits)) & mask).astype(F64)
                cv = np.fft.irfft(np.fft.rfft(ch, 2 * N, axis=1) * S, 2 * N, axis=1)
                cv = np.rint(cv).astype(np.int64)
                cv = (cv[:, :N] - cv[:, N:]) % q
                acc = cv if acc is None else (mulmod(acc, 1 << chunk_bits, q) + cv) % q
            out[r0:r0 + block] = acc
        return out

    def keygen(self):
        return self.rng.integers(-1, 2, size=N).astype(np.int64)     # uniform ternary

    def noise(self, shape):
        e = np.rint(self.rng.normal(0, self.sigma, size=shape))
        return np.clip(e, -6 * self.sigma, 6 * self.sigma).astype(np.int64)

    def gen_a(self, seed, rows):
        g = np.random.default_rng(seed)
        return [g.integers(0, q, size=(rows, N), dtype=np.int64) for q in (self.qd, self.qe)]

    def encrypt_b(self, R, s, a):
        """b_i = a_i*s + e_i + Delta*r_i  (mod Q, RNS). R: (d_in,N) masks in [0,t)."""
        e = self.noise(R.shape)
        out = []
        for li, q in enumerate((self.qd, self.qe)):
            as_ = self.polymul(a[li], s, q)
            out.append((as_ + (e % q) + mulmod(R, self.dmod[li], q)) % q)
        return out

    # ---------- server ----------
    @staticmethod
    def gemm_mod(Wf, X, q, split=25, colblock=2048):
        """W @ X mod q, X int64 < q (<2^50). Exact float64 GEMM on 25-bit halves (|sum| < 2^53)."""
        dout = Wf.shape[0]; Y = np.empty((dout, X.shape[1]), dtype=np.int64)
        m = (1 << split) - 1
        for c0 in range(0, X.shape[1], colblock):
            Xb = X[:, c0:c0 + colblock]
            nb = Xb.shape[1]
            Z = Wf @ np.concatenate([(Xb & m).astype(F64), (Xb >> split).astype(F64)], axis=1)
            lo = np.rint(Z[:, :nb]).astype(np.int64) % q
            hi = np.rint(Z[:, nb:]).astype(np.int64) % q
            Y[:, c0:c0 + nb] = (mulmod(hi, 1 << split, q) + lo) % q
        return Y

    def modswitch(self, Yd, Ye):
        """RNS divide-and-round by q_e: Q -> q_d."""
        ec = np.where(Ye > self.qe // 2, Ye - self.qe, Ye)
        return mulmod((Yd - ec) % self.qd, self.qe_inv_mod_qd, self.qd)

    def compress_b(self, Bd):
        if self.k == 0: return Bd
        return ((Bd + (1 << (self.k - 1))) >> self.k) % (1 << self.b_bits)

    # ---------- client decode ----------
    def decode(self, v):
        """m = round(t*v/q_d) mod t; also return worst |fractional distance| (noise margin used)."""
        x = v.astype(F64) * (F64(T) / F64(self.qd))
        m = np.rint(x)
        frac = np.abs(x - m).max()
        return (m.astype(np.int64) % T), float(frac)

def ref_wr_mod_t(W, R):
    Wf = W.astype(F64); m = (1 << 20) - 1
    lo = np.rint(Wf @ (R & m).astype(F64)).astype(np.int64) % T
    hi = np.rint(Wf @ (R >> 20).astype(F64)).astype(np.int64) % T
    return (lo + (hi << 20)) % T

def run(W, label, variants=('A', 'B'), qd=None, qe=None, check_int_tokens=32):
    from find_primes import find
    if qd is None: qd, qe = find(50)
    ctx = Ctx(qd, qe, b_bits=45)
    dout, din = W.shape
    Wf = W.astype(F64)
    res = dict(label=label, N=N, t_bits=T_BITS, qd_bits=qd.bit_length(), qe_bits=qe.bit_length(),
               logQ=ctx.Q.bit_length(), d_in=din, d_out=dout, b_download_bits=ctx.b_bits,
               threads=os.environ.get('OPENBLAS_NUM_THREADS'))
    R = ctx.rng.integers(0, T, size=(din, N), dtype=np.int64)
    ref = ref_wr_mod_t(W, R)
    # extra exact int64 check on a few tokens (W r fits int64: |W r| < 2^7*2^40*2^14.3)
    ref_int = (W.astype(np.int64) @ R[:, :check_int_tokens]) % T
    assert (ref_int == ref[:, :check_int_tokens]).all()
    for var in variants:
        s = ctx.keygen()
        r = {}
        if var == 'A':
            t0 = time.perf_counter(); a = ctx.gen_a(seed=1234, rows=din); r['client_gen_a_s'] = time.perf_counter() - t0
        else:
            a = ctx.gen_a(seed=99, rows=din)   # fixed forever
            t0 = time.perf_counter()
            H = [Ctx.gemm_mod(Wf, a[0], ctx.qd), Ctx.gemm_mod(Wf, a[1], ctx.qe)]
            r['hint_gen_s(one-time, = 1 batch of server work on a)'] = time.perf_counter() - t0
            r['hint_bytes_per_linear(qd+qe bits packed)'] = dout * N * (ctx.Q.bit_length()) / 8
        t0 = time.perf_counter(); Bct = ctx.encrypt_b(R, s, a); r['client_encrypt_s'] = time.perf_counter() - t0
        # ---- server ----
        t0 = time.perf_counter()
        if var == 'A':
            a_srv = ctx.gen_a(seed=1234, rows=din)   # server re-expands a from the seed
            Ya = [Ctx.gemm_mod(Wf, a_srv[0], ctx.qd), Ctx.gemm_mod(Wf, a_srv[1], ctx.qe)]
        Yb = [Ctx.gemm_mod(Wf, Bct[0], ctx.qd), Ctx.gemm_mod(Wf, Bct[1], ctx.qe)]
        t_gemm = time.perf_counter() - t0
        t0 = time.perf_counter()
        Bsw = ctx.compress_b(ctx.modswitch(*Yb))
        if var == 'A': Asw = ctx.modswitch(*Ya)
        t_ms = time.perf_counter() - t0
        r['server_gemm_s'] = t_gemm; r['server_modswitch_s'] = t_ms
        # ---- client decrypt ----
        t0 = time.perf_counter()
        Bfull = (Bsw << ctx.k) % ctx.qd
        if var == 'A':
            v = (Bfull - ctx.polymul(Asw, s, ctx.qd)) % ctx.qd
        else:
            hs_d = ctx.polymul(H[0], s, ctx.qd); hs_e = ctx.polymul(H[1], s, ctx.qe)
            v = (Bfull - ctx.modswitch(hs_d, hs_e)) % ctx.qd
        m, frac = ctx.decode(v)
        r['client_decrypt_s'] = time.perf_counter() - t0
        r['exact_match'] = bool((m == ref).all())
        r['max_decode_frac(<0.5 required)'] = frac
        # per-token, per-element normalisations
        r['server_s_per_token'] = (t_gemm + t_ms) / N
        r['client_s_per_token'] = (r['client_encrypt_s'] + r['client_decrypt_s']) / N
        r['server_ns_per_token_MAC'] = (t_gemm + t_ms) / N / (din * dout) * 1e9
        up_bits = ctx.Q.bit_length()  # b mod Q per input element per token
        down_bits = ctx.b_bits + (ctx.qd.bit_length() if var == 'A' else 0)
        r['upload_bytes_per_token'] = din * up_bits / 8
        r['download_bytes_per_token'] = dout * down_bits / 8
        r['stored_Wr_bytes_per_token_K40'] = dout * 5
        res['variant_' + var] = r
        print(label, var, json.dumps(r, indent=1), flush=True)
    return res

if __name__ == '__main__':
    from safetensors.numpy import load_file
    which = sys.argv[1]
    out = {}
    if which == 'qwen05_gate':
        Wall = load_file('/root/.pgw-client-demo/packs/qwen2.5-0.5b-instruct/linears-00000-of-00001.safetensors')['linears.0.gate_up.weight']
        out = run(Wall[:4864].astype(np.int64), 'qwen2.5-0.5b layer0 gate 4864x896 (real int8)')
    elif which == 'slice7b':
        g = np.random.default_rng(7)
        W = g.integers(-127, 128, size=(4608, 3584)).astype(np.int64)
        out = run(W, '7B-shaped qkv 4608x3584 (random full-range int8)', variants=tuple(sys.argv[2].split(',')) if len(sys.argv) > 2 else ('A', 'B'))
    elif which == 'down_noise':
        g = np.random.default_rng(8)
        W = np.full((64, 18944), 127, dtype=np.int64) * np.where(g.random((64, 18944)) < 0.5, -1, 1)
        W[1::2] = g.integers(-127, 128, size=W[1::2].shape)
        out = run(W, '7B down-shaped noise worst case 64x18944 (|W|=127 rows + random rows)')
    json.dump(out, open(f'results_{which}.json', 'w'), indent=1)
