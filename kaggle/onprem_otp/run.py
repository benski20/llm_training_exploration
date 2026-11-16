"""One-time-mask private inference with public weights.
Offline (customer, or a mask dealer): pick uniform r in Z_2^48 per linear call and compute W r. Store W r (r regenerated from a seed).
Online: customer sends x_hat = x + r (a one-time pad: uniform, reveals nothing) -> server returns W x_hat on its GPU ->
customer subtracts W r, does norms/attention/SiLU/LM head in plaintext, repeats.
Exact integer arithmetic: activations 20-bit fixed point, weights int8 per row; ring 2^48; server matmul = 6 int8 limb GEMMs.
Part A: server GPU cost per token and customer offline cost per token at Llama-3-8B / 70B shapes.
Part B: end-to-end Qwen2.5-0.5B, exactness vs plaintext, and statistical check that the server's view is uniform."""
import os, time, json, math, numpy as np, torch
T0 = time.time(); OUT = "/kaggle/working"; R = {}
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
def save(): json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=str)
GPU = "cuda" if torch.cuda.is_available() else "cpu"
R["device"] = torch.cuda.get_device_name(0) if GPU == "cuda" else "cpu"; R["cpu_count"] = os.cpu_count(); log(R)
K = 48; MOD = 1 << K; MASK = MOD - 1; LIMBS = 6; ABITS = 20
def sync():
    if GPU == "cuda": torch.cuda.synchronize()
def digits(v):                              # balanced base-256 digits of v in [0, 2^48): v = sum d_i 256^i mod 2^48
    out = []
    for _ in range(LIMBS):
        d = v & 255; d = torch.where(d >= 128, d - 256, d); out.append(d.to(torch.int8)); v = (v - d) >> 8
    return out
IMM = {}
def imm(a8, w8):                            # a8 (P,n) int8, w8 (m,n) int8 -> exact int64 a8 @ w8.T
    key = a8.device.type
    if IMM.get(key, True):
        try: return torch._int_mm(a8.contiguous(), w8.t()).long()
        except Exception as e:
            if key not in IMM: log("int8 GEMM unavailable on", key, "-> float64 path:", repr(e)[:150])
            IMM[key] = False
    return (a8.double() @ w8.double().t()).long()
def ringmm(X, W8):                          # (B,n) ring elements x (m,n) int8 -> (B,m) mod 2^48, exact
    B = X.shape[0]; P = max(32, -(-B // 8) * 8)
    Xp = torch.zeros((P, X.shape[1]), dtype=torch.int64, device=X.device); Xp[:B] = X
    acc = torch.zeros((P, W8.shape[0]), dtype=torch.int64, device=X.device)
    for i, d in enumerate(digits(Xp)): acc = (acc + (imm(d, W8) << (8 * i))) & MASK
    return acc[:B]
def center(v): v = v & MASK; return torch.where(v >= MOD // 2, v - MOD, v)
# self-test
g = torch.Generator().manual_seed(0)
W8 = torch.randint(-127, 128, (64, 96), dtype=torch.int8); x = torch.randint(-(1 << ABITS), 1 << ABITS, (5, 96))
r = torch.randint(0, MOD, (5, 96), generator=g); xh = (x + r) & MASK
for dev in ("cpu", GPU):
    y = center(ringmm(xh.to(dev), W8.to(dev)).cpu() - ringmm(r.to(dev), W8.to(dev)).cpu())
    R[f"selftest_exact_{dev}"] = bool((y == (x.double() @ W8.double().T).long()).all())
log("selftest", R); save()

# ================= Part A: per-token cost at 8B / 70B shapes =================
def bench(fn, n=5):
    fn(); sync(); ts = []
    for _ in range(n): t = time.time(); fn(); sync(); ts.append(time.time() - t)
    return float(np.median(ts))
SHAPES = {"llama-3-8b": (32, [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)]),
          "llama-3-70b": (80, [(10240, 8192), (8192, 8192), (57344, 8192), (8192, 28672)])}
A = {}
for mname, (L, mats) in SHAPES.items():
    srv1 = srv32 = fp16 = cpu_off = 0.0
    for m, n in mats:
        Wg = torch.randint(-127, 128, (m, n), dtype=torch.int8, device=GPU)
        x1 = torch.randint(0, MOD, (1, n), device=GPU); x32 = torch.randint(0, MOD, (32, n), device=GPU)
        srv1 += bench(lambda: ringmm(x1, Wg)); srv32 += bench(lambda: ringmm(x32, Wg))
        Wh = torch.randn((m, n), dtype=torch.float16, device=GPU); xf = torch.randn((n, 1), dtype=torch.float16, device=GPU)
        fp16 += bench(lambda: Wh @ xf)
        del Wg, Wh; torch.cuda.empty_cache() if GPU == "cuda" else None
        if mname == "llama-3-8b":               # customer-side offline mask generation on CPU: W r for a batch of 64 tokens
            Wc = torch.randint(-127, 128, (m, n), dtype=torch.int8); rc = torch.randint(0, MOD, (64, n))
            cpu_off += bench(lambda: ringmm(rc, Wc), n=2) / 64; del Wc
    out_vals = sum(m for m, n in mats) * L; in_vals = sum(n for m, n in mats) * L
    A[mname] = {"server_gpu_s_per_token_batch1": round(srv1 * L, 3), "server_gpu_s_per_token_amortized_batch32": round(srv32 * L / 32, 4),
                "plain_fp16_gpu_s_per_token_batch1": round(fp16 * L, 4),
                "mask_storage_MB_per_token": round(out_vals * 6 / 1e6, 2), "upload_MB_per_token": round(in_vals * 6 / 1e6, 2),
                "download_MB_per_token": round(out_vals * 6 / 1e6, 2), "rounds_per_token": 4 * L,
                "customer_online_work": "add r (from seed) and subtract stored W r: O(MB) of integer adds per token"}
    if cpu_off: A[mname]["customer_offline_cpu_s_per_token_mask_generation"] = round(cpu_off * L, 3)
    log("A", mname, A[mname]); R["A"] = A; save()
p8, p70 = 8.03e9 - 2 * 0.525e9, 70.6e9 - 2 * 1.05e9      # transformer-layer params (excl. embeddings / head)
if "customer_offline_cpu_s_per_token_mask_generation" in A["llama-3-8b"]:
    A["llama-3-70b"]["customer_offline_cpu_s_per_token_mask_generation_est"] = round(A["llama-3-8b"]["customer_offline_cpu_s_per_token_mask_generation"] * p70 / p8, 2)
for mname in A: A[mname]["tokens_per_TB_of_masks"] = int(1e12 / (A[mname]["mask_storage_MB_per_token"] * 1e6))
R["A"] = A; save()

# ================= Part B: end-to-end Qwen2.5-0.5B =================
from transformers import AutoTokenizer, AutoModelForCausalLM
MID = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(MID)
hf = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).eval(); cfg = hf.config
H, NH, KV, HD, L, I = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.num_hidden_layers, cfg.intermediate_size
sd = {k: v.detach().double().numpy() for k, v in hf.state_dict().items()}
E = sd["model.embed_tokens.weight"]; HEAD = sd.get("lm_head.weight", E)
THETA = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 1e6)
inv = 1.0 / (THETA ** (np.arange(0, HD, 2) / HD))
LIN = {}
for i in range(L):
    p = f"model.layers.{i}."
    LIN[f"{i}.qkv"] = (np.concatenate([sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"], sd[p + "self_attn.v_proj.weight"]]),
                       np.concatenate([sd[p + "self_attn.q_proj.bias"], sd[p + "self_attn.k_proj.bias"], sd[p + "self_attn.v_proj.bias"]]))
    LIN[f"{i}.o"] = (sd[p + "self_attn.o_proj.weight"], None)
    LIN[f"{i}.gu"] = (np.concatenate([sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]]), None)
    LIN[f"{i}.down"] = (sd[p + "mlp.down_proj.weight"], None)
Q = {}
for k, (W, b) in LIN.items():                    # int8 per-row quantization (public: both sides have it)
    sw = np.abs(W).max(1) / 127; W8 = np.round(W / sw[:, None]).astype(np.int8)
    Q[k] = (torch.from_numpy(W8), torch.from_numpy(W8).to(GPU), sw, b)
prompt = "Maria Lopez, grade 8, said she feels unsafe. The counselor will"
ids = np.array(tok(prompt).input_ids); NEW = 8; NTOK = len(ids) + NEW
# ---- OFFLINE (customer CPU or mask dealer): mask bank for NTOK tokens = seed + W r ----
t = time.time(); BANK = {}; bank_bytes = 0
for k, (W8c, _, _, _) in Q.items():
    gk = torch.Generator().manual_seed(hash(k) % (1 << 31))
    rr = torch.randint(0, MOD, (NTOK, W8c.shape[1]), generator=gk); BANK[k] = [rr, ringmm(rr, W8c), 0]; bank_bytes += NTOK * W8c.shape[0] * 6
R["B_offline"] = {"tokens": NTOK, "mask_gen_cpu_s_total": round(time.time() - t, 1), "mask_bank_MB": round(bank_bytes / 1e6, 2),
                  "mask_MB_per_token": round(bank_bytes / NTOK / 1e6, 3)}
log("B offline", R["B_offline"]); save()
TM = {"server_s": 0.0, "client_crypto_s": 0.0, "up_B": 0, "down_B": 0, "rounds": 0}; VIEW = []; PLAINQ = []
def linear(name, X, mode):
    W8c, W8g, sw, b = Q[name]
    if mode == "float": Y = X @ LIN[name][0].T
    else:
        sx = np.abs(X).max(1) / (2 ** ABITS - 1); sx[sx == 0] = 1.0
        xq = torch.from_numpy(np.round(X / sx[:, None]).astype(np.int64))
        if mode == "qplain": yq = (xq.double() @ W8c.double().T).numpy()
        else:
            t = time.time(); rr, wr, ptr = BANK[name]; T = xq.shape[0]
            xh = (xq + rr[ptr:ptr + T]) & MASK; TM["client_crypto_s"] += time.time() - t        # CUSTOMER: one-time pad
            VIEW.append(xh[:, :64].clone()); PLAINQ.append(xq[:, :64].clone())
            t = time.time(); yh = ringmm(xh.to(GPU), W8g).cpu(); sync(); TM["server_s"] += time.time() - t   # SERVER
            t = time.time(); yq = center(yh - wr[ptr:ptr + T]).double().numpy(); BANK[name][2] += T; TM["client_crypto_s"] += time.time() - t
            TM["up_B"] += xh.numel() * 6; TM["down_B"] += yh.numel() * 6; TM["rounds"] += 1
        Y = yq * sx[:, None] * sw[None, :]
    return Y + b[None, :] if b is not None else Y
def rms(x, w): return x / np.sqrt((x * x).mean(-1, keepdims=True) + cfg.rms_norm_eps) * w
def rope(x, pos):
    f = pos[:, None] * inv[None]; c, s = np.cos(np.concatenate([f, f], -1))[:, None], np.sin(np.concatenate([f, f], -1))[:, None]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]; return x * c + np.concatenate([-x2, x1], -1) * s
def forward(ids, cache, mode):
    T = len(ids); start = cache["len"]; pos = np.arange(start, start + T, dtype=np.float64); x = E[ids]
    for i in range(L):
        p = f"model.layers.{i}."; h = rms(x, sd[p + "input_layernorm.weight"]); qkv = linear(f"{i}.qkv", h, mode)
        q = rope(qkv[:, :H].reshape(T, NH, HD), pos); k = rope(qkv[:, H:H + KV * HD].reshape(T, KV, HD), pos); v = qkv[:, H + KV * HD:].reshape(T, KV, HD)
        cache.setdefault(i, [np.zeros((0, KV, HD)), np.zeros((0, KV, HD))])
        cache[i][0] = np.concatenate([cache[i][0], k]); cache[i][1] = np.concatenate([cache[i][1], v])
        K_ = np.repeat(cache[i][0], NH // KV, 1); V_ = np.repeat(cache[i][1], NH // KV, 1)
        sc = np.einsum("thd,shd->hts", q, K_) / np.sqrt(HD)
        sc[:, np.arange(K_.shape[0])[None, :] > (start + np.arange(T))[:, None]] = -1e30
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        x = x + linear(f"{i}.o", np.einsum("hts,shd->thd", sc, V_).reshape(T, H), mode)
        gu = linear(f"{i}.gu", rms(x, sd[p + "post_attention_layernorm.weight"]), mode)
        x = x + linear(f"{i}.down", gu[:, :I] / (1 + np.exp(-gu[:, :I])) * gu[:, I:], mode)
    cache["len"] += T
    return rms(x[-1:], sd["model.norm.weight"]) @ HEAD.T
def run(mode):
    cache = {"len": 0}; out, lgs, times = [], [], []
    t = time.time(); lg = forward(ids, cache, mode); times.append(time.time() - t)
    for _ in range(NEW):
        nxt = int(lg[-1].argmax()); out.append(nxt); lgs.append(lg[-1])
        t = time.time(); lg = forward(np.array([nxt]), cache, mode); times.append(time.time() - t)
    return out, np.array(lgs), times
out_f, lg_f, _ = run("float"); out_q, lg_q, _ = run("qplain")
for k in TM: TM[k] = 0 if isinstance(TM[k], int) else 0.0
out_m, lg_m, times = run("mask")
V_ = torch.cat([v.flatten() for v in VIEW]); P_ = torch.cat([v.flatten() for v in PLAINQ])
top = (V_ >> (K - 8)).numpy(); cnt = np.bincount(top, minlength=256); exp = len(top) / 256; chi2 = float(((cnt - exp) ** 2 / exp).sum())
corr = float(np.corrcoef(V_.double().numpy(), P_.double().numpy())[0, 1])
R["B"] = {"answer_float": tok.decode(out_f), "answer_masked": tok.decode(out_m),
    "masked_tokens_identical_to_quantized_plaintext": out_m == out_q, "masked_vs_quantized_max_logit_diff": float(np.abs(lg_m - lg_q).max()),
    "masked_tokens_identical_to_float_model": out_m == out_f, "masked_vs_float_max_logit_diff": float(np.abs(lg_m - lg_f).max()),
    "prefill_s": round(times[0], 2), "decode_s_per_token": round(float(np.mean(times[1:])), 3),
    "server_gpu_s_total": round(TM["server_s"], 2), "client_online_crypto_s_total": round(TM["client_crypto_s"], 3),
    "upload_MB_total": round(TM["up_B"] / 1e6, 2), "download_MB_total": round(TM["down_B"] / 1e6, 2), "rounds": TM["rounds"],
    "server_view_top_byte_chi2_255dof": round(chi2, 1), "server_view_corr_with_plaintext": round(corr, 5), "server_view_samples": int(len(top))}
R["minutes"] = round((time.time() - T0) / 60, 1); log("B", R["B"]); save(); log("done")
