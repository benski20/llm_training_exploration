"""Mask-generation cost for one-time-mask private inference.
T1 (GPU, 'gateway / mask partner'): W r mod 2^K for large batches, ring 2^48 (6 int8 limbs) vs 2^32 (4 limbs, needs 8-bit activations);
    8B layers resident on GPU; 70B layers streamed from host RAM each batch (GPU smaller than the model). -> tokens/s and $/M tokens.
T2 (CPU, 'laptop / office server'): best exact int8 GEMM path on this CPU -> effective TOPS -> s/token; NPU extrapolation.
T3 (quality): does W8A8 (8-bit activations, needed for the 32-bit ring) keep Qwen2.5-0.5B's predictions? vs 12/16/20-bit."""
import os, time, json, subprocess, numpy as np, torch
T0 = time.time(); OUT = "/kaggle/working"; R = {}
def log(*a): print(f"[{(time.time()-T0)/60:6.1f}m]", *a, flush=True)
def save(): json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=str)
GPU = "cuda"; R["gpu"] = torch.cuda.get_device_name(0)
try:
    lc = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    R["cpu_model"] = [l.split(":", 1)[1].strip() for l in lc.splitlines() if l.startswith("Model name")][0]
    fl = [l for l in lc.splitlines() if l.startswith("Flags")][0]
    R["cpu_flags"] = [f for f in ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8") if f" {f}" in fl]
except Exception as e: R["cpu_model"] = repr(e)
R["cpu_count"] = os.cpu_count(); R["torch_threads"] = torch.get_num_threads(); log(R); save()
def sync(): torch.cuda.synchronize()
def digits(v, limbs):
    out = []
    for _ in range(limbs):
        d = v & 255; d = torch.where(d >= 128, d - 256, d); out.append(d.to(torch.int8)); v = (v - d) >> 8
    return out
def ringmm(X, W8, K):                       # exact (B,n) x (m,n)^T mod 2^K via int8 tensor-core GEMMs
    mask = (1 << K) - 1; acc = None
    for i, d in enumerate(digits(X, K // 8)):
        y = (torch._int_mm(d, W8.t()).long() << (8 * i)) & mask
        acc = y if acc is None else (acc + y) & mask
    return acc
# exactness check for both rings
for K in (48, 32):
    W = torch.randint(-127, 128, (256, 512), dtype=torch.int8, device=GPU); X = torch.randint(0, 1 << K, (64, 512), device=GPU)
    ref = ((X.cpu().double() @ W.cpu().double().T) % (1 << K)) if K == 32 else None
    out = ringmm(X, W, K).cpu()
    if K == 32: R["exact_ring32"] = bool((out.double() == ref).all())
    else:
        xs = [X.cpu() & 0xFFFFFF, X.cpu() >> 24]; r = sum((((x.double() @ W.cpu().double().T).long() << (24 * j)) for j, x in enumerate(xs))) & ((1 << 48) - 1)
        R["exact_ring48"] = bool((out == r).all())
log("exact", R.get("exact_ring48"), R.get("exact_ring32")); save()

# ================= T1: GPU mask generation at large batch =================
SHAPES = {"llama-3-8b": (32, [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)]),
          "llama-3-70b": (80, [(10240, 8192), (8192, 8192), (57344, 8192), (8192, 28672)])}
PRICE = {"T4 on-demand (AWS g4dn.xlarge)": 0.526, "T4 spot (approx)": 0.20}
T1 = {}
for mname, (L, mats) in SHAPES.items():
    stream = mname == "llama-3-70b"
    for K in (48, 32):
        for B in (256, 1024, 4096):
            tot = 0.0; ok = True
            try:
                for m, n in mats:
                    Wc = torch.randint(-127, 128, (m, n), dtype=torch.int8).pin_memory() if stream else None
                    Wg = torch.randint(-127, 128, (m, n), dtype=torch.int8, device=GPU) if not stream else torch.empty((m, n), dtype=torch.int8, device=GPU)
                    X = torch.randint(0, 1 << K, (B, n), device=GPU)
                    def step():
                        if stream: Wg.copy_(Wc, non_blocking=True)
                        return ringmm(X, Wg, K)
                    step(); sync(); ts = []
                    for _ in range(3): t = time.time(); step(); sync(); ts.append(time.time() - t)
                    tot += float(np.median(ts)); del Wg, X, Wc; torch.cuda.empty_cache()
            except Exception as e: ok = False; T1[f"{mname}|ring{K}|B{B}"] = {"error": repr(e)[:200]}; torch.cuda.empty_cache()
            if ok:
                spt = tot * L / B; tps = 1 / spt
                T1[f"{mname}|ring{K}|B{B}"] = {"weights": "streamed from host RAM per batch" if stream else "resident on GPU",
                    "gpu_s_per_token": round(spt, 5), "tokens_per_hour": int(3600 * tps),
                    **{f"usd_per_M_tokens_{k}": round(p / (3600 * tps) * 1e6, 2) for k, p in PRICE.items()},
                    "mask_MB_per_token": round(sum(m for m, n in mats) * L * K / 8 / 1e6, 2)}
            log("T1", mname, K, B, T1[f"{mname}|ring{K}|B{B}"]); R["T1_gpu_mask_generation"] = T1; save()

# ================= T2: CPU mask generation (best exact path) =================
torch.set_num_threads(os.cpu_count())
m, n, B = 4096, 4096, 256
Wc = torch.randint(-127, 128, (m, n), dtype=torch.int8); Xc = torch.randint(-128, 128, (B, n), dtype=torch.int8)
useful_ops = 2 * m * n * B                       # one int8 limb GEMM
def tbest(fn, k=3):
    fn(); ts = []
    for _ in range(k): t = time.time(); fn(); ts.append(time.time() - t)
    return min(ts)
T2 = {}
try: T2["torch_int_mm_int8"] = round(useful_ops / tbest(lambda: torch._int_mm(Xc, Wc.t())) / 1e12, 3)
except Exception as e: T2["torch_int_mm_int8"] = repr(e)[:150]
Wf, Xf = Wc.float(), Xc.float()                  # exact if K-chunks of 1024: 127*128*1024 < 2^24
def f32chunks():
    acc = torch.zeros((B, m))
    for k0 in range(0, n, 1024): acc += (Xf[:, k0:k0 + 1024] @ Wf[:, k0:k0 + 1024].T).round()
    return acc
T2["mkl_fp32_exact_chunks"] = round(useful_ops / tbest(f32chunks) / 1e12, 3)
try:                                             # fbgemm int8 (VNNI if present) via quantized linear
    import torch.ao.nn.quantized as nnq
    ql = nnq.Linear(n, m); ql.set_weight_bias(torch.quantize_per_tensor(Wc.float(), 1.0, 0, torch.qint8), None)
    xq = torch.quantize_per_tensor(Xc.float() + 128, 1.0, 0, torch.quint8)
    T2["fbgemm_quantized_linear_int8"] = round(useful_ops / tbest(lambda: ql(xq)) / 1e12, 3)
except Exception as e: T2["fbgemm_quantized_linear_int8"] = repr(e)[:150]
best = max(v for v in T2.values() if isinstance(v, float))
T2["best_effective_int8_TOPS"] = best
p8, p70 = 6.98e9, 68.5e9
for K in (48, 32):
    for name, P in (("8b", p8), ("70b", p70)):
        ops = 2 * P * (K // 8)
        T2[f"cpu_s_per_token_{name}_ring{K}"] = round(ops / (best * 1e12), 2)
        for npu, tops in (("laptop NPU 40 TOPS at 30% util (estimate)", 12.0), ("laptop NPU 40 TOPS at 60% util (estimate)", 24.0)):
            T2[f"{npu}|s_per_token_{name}_ring{K}"] = round(ops / (tops * 1e12), 3)
R["T2_cpu_mask_generation"] = T2; log("T2", T2); save()

# ================= T3: does 8-bit activation quantization hurt quality? =================
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import pandas as pd
MID = "Qwen/Qwen2.5-0.5B-Instruct"; tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32).to(GPU).eval()
txt = pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"]
ids = tok(" ".join(t for t in txt[:400] if len(t) > 40), return_tensors="pt").input_ids[:, :2048].to(GPU)
lins = [mod for name, mod in model.model.layers.named_modules() if isinstance(mod, torch.nn.Linear)]
orig = {id(l): l.weight.data.clone() for l in lins}
@torch.no_grad()
def evaluate():
    lg = model(ids).logits[0, :-1].float(); tgt = ids[0, 1:]
    return lg, float(torch.exp(torch.nn.functional.cross_entropy(lg, tgt)))
ref_lg, ref_ppl = evaluate(); T3 = {"tokens": int(ids.shape[1]), "float_ppl": round(ref_ppl, 3)}
for l in lins:                                   # int8 per-output-row weights (used by every variant)
    W = orig[id(l)]; sw = W.abs().amax(1, keepdim=True) / 127; l.weight.data = (W / sw).round() * sw
def set_act_bits(bits):
    for l in lins:
        l._hook = getattr(l, "_hook", None)
        if l._hook: l._hook.remove()
        if bits:
            def pre(mod, inp, b=bits):
                x = inp[0]; s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / (2 ** (b - 1) - 1)
                return ((x / s).round() * s,)
            l._hook = l.register_forward_pre_hook(pre)
for bits in (None, 20, 16, 12, 8):
    set_act_bits(bits); lg, ppl = evaluate()
    T3[f"W8A{bits or 'float'}"] = {"ppl": round(ppl, 3), "top1_agreement_with_float": round(float((lg.argmax(-1) == ref_lg.argmax(-1)).float().mean()), 4)}
    log("T3", bits, T3[f"W8A{bits or 'float'}"])
R["T3_quality"] = T3; R["minutes"] = round((time.time() - T0) / 60, 1); save(); log("done")
