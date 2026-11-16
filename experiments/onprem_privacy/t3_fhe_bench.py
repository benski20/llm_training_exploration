"""T3: CKKS (TenSEAL) microbenchmarks on CPU.
(a) linear ops: encrypted query . plaintext embedding DB  (vector search)
(b) encrypted vector x plaintext matrix (one transformer linear layer), several widths
(c) polynomial approximations of non-linear ops (GELU, exp for softmax): accuracy vs degree, and encrypted cost
(d) a small encrypted classifier (logistic-regression head)
Plaintext timings measured on the same machine for the ratio."""
import json, time, numpy as np, tenseal as ts
rng = np.random.default_rng(0); R = {}
def ctx(N=8192, bits=(60, 40, 40, 60)):
    c = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N, coeff_mod_bit_sizes=list(bits))
    c.global_scale = 2**40; c.generate_galois_keys(); return c

# (a) vector search: 384-dim embeddings, 2000 docs. Pack the encrypted query, dot with each doc.
c = ctx(); D, ND = 384, 2000
db = rng.standard_normal((ND, D)); db /= np.linalg.norm(db, axis=1, keepdims=True)
q = db[17] + 0.1 * rng.standard_normal(D); q /= np.linalg.norm(q)
t = time.perf_counter(); eq = ts.ckks_vector(c, q.tolist()); t_enc = time.perf_counter() - t
# matrix form: one enc-vector x plain-matrix (D x ND chunked) is far faster than ND dot products
t = time.perf_counter()
scores_ct = [eq.mm(db[i:i+1000].T.tolist()) for i in range(0, ND, 1000)]
t_fhe = time.perf_counter() - t
scores = np.concatenate([np.array(s.decrypt()) for s in scores_ct])
t = time.perf_counter(); ps = db @ q; t_plain = time.perf_counter() - t
R["a_vector_search"] = {"docs": ND, "dim": D, "encrypt_query_s": t_enc, "fhe_search_s": t_fhe,
    "fhe_ms_per_doc": 1000*t_fhe/ND, "plain_s": t_plain, "slowdown_x": t_fhe/t_plain,
    "max_abs_err": float(np.abs(scores-ps).max()), "top1_match": int(scores.argmax()) == int(ps.argmax()),
    "ciphertext_query_bytes": len(eq.serialize())}
print("a", R["a_vector_search"])

# (b) one linear layer: enc(x) (1 x d) @ W (d x d). Qwen-0.5B hidden=896, Llama-8B hidden=4096.
R["b_linear_layer"] = []
for d in (128, 256, 512, 1024):
    W = rng.standard_normal((d, d)) / np.sqrt(d); x = rng.standard_normal(d)
    ex = ts.ckks_vector(c, x.tolist())
    t = time.perf_counter(); y = ex.mm(W.tolist()); tf = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(100): x @ W
    tp = (time.perf_counter() - t) / 100
    R["b_linear_layer"].append({"d": d, "fhe_s": tf, "plain_s": tp, "slowdown_x": tf/tp,
        "max_abs_err": float(np.abs(np.array(y.decrypt()) - x @ W).max())})
    print("b", R["b_linear_layer"][-1])
# fit cost ~ d^k and extrapolate to Llama-3-8B linear work per token per layer:
ds = np.array([r["d"] for r in R["b_linear_layer"]]); ts_ = np.array([r["fhe_s"] for r in R["b_linear_layer"]])
k, logc = np.polyfit(np.log(ds), np.log(ts_), 1)
f = lambda d: np.exp(logc) * d**k
# Llama-3-8B per layer per token: q(4096x4096), k,v(4096x1024 each), o(4096x4096), gate/up(4096x14336), down(14336x4096)
# express as equivalent count of 4096x4096 matvecs: (2*16.8M + 2*4.2M + 3*58.7M)/16.8M ~= 13 ; 32 layers
eq_mats = (2*4096*4096 + 2*4096*1024 + 3*4096*14336) / (4096*4096)
R["b_extrapolation"] = {"fit_exponent": float(k), "est_s_4096x4096_matvec": float(f(4096)),
    "equiv_4096_matvecs_per_layer": eq_mats,
    "est_linear_only_s_per_token_llama8b_1cpu_core": float(f(4096) * eq_mats * 32),
    "note": "naive TenSEAL diagonal method, single thread, linear layers only; ignores attention-over-KV, non-linearities, bootstrapping. Lower bound for this naive method; specialised GPU CKKS is 100-1000x faster."}
print("b-extrap", R["b_extrapolation"])

# (c) polynomial approximation accuracy for non-linear ops
from numpy.polynomial import chebyshev as C
gelu = lambda x: 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
R["c_poly_approx"] = {}
for name, fn, lo, hi in (("gelu", gelu, -8, 8), ("exp_softmax", np.exp, -16, 0)):
    xs = np.linspace(lo, hi, 4001); rows = []
    for deg in (3, 7, 15, 31):
        p = C.Chebyshev.fit(xs, fn(xs), deg)
        err = np.abs(p(xs) - fn(xs)); rows.append({"degree": deg, "max_err": float(err.max()),
            "mult_depth_needed": int(np.ceil(np.log2(deg+1)))})
    R["c_poly_approx"][name] = rows
    print("c", name, rows)
# encrypted cost of a degree-7 polynomial (needs deeper modulus chain)
c7 = ctx(16384, (60,) + (40,)*6 + (60,))
xv = rng.uniform(-4, 4, 4096); p = C.Chebyshev.fit(np.linspace(-8, 8, 4001), gelu(np.linspace(-8, 8, 4001)), 7)
coef = C.cheb2poly(p.convert(domain=[-8, 8]).coef) if False else np.polynomial.polynomial.Polynomial.cast(p).coef
ex = ts.ckks_vector(c7, xv.tolist())
t = time.perf_counter(); ey = ex.polyval(coef.tolist()); tpoly = time.perf_counter() - t
R["c_encrypted_poly"] = {"degree": 7, "slots": 4096, "fhe_s": tpoly,
    "max_err_vs_true_gelu": float(np.abs(np.array(ey.decrypt()) - gelu(xv)).max()),
    "note": "one GELU over 4096 values. Llama-8B MLP has 14336 activations x 32 layers per token; softmax needs exp+reciprocal over every attention score."}
print("c-enc", R["c_encrypted_poly"])

# (d) encrypted logistic-regression head: 384-dim embedding -> 5 classes (e.g. PII/risk/routing flag)
W = rng.standard_normal((384, 5)) / 20; x = rng.standard_normal(384)
ex = ts.ckks_vector(c, x.tolist())
t = time.perf_counter(); logits = ex.mm(W.tolist()); tcls = time.perf_counter() - t
R["d_encrypted_classifier"] = {"fhe_s": tcls, "argmax_match": int(np.argmax(logits.decrypt())) == int(np.argmax(x @ W))}
print("d", R["d_encrypted_classifier"])
json.dump(R, open("results/t3_fhe_bench.json", "w"), indent=1)
