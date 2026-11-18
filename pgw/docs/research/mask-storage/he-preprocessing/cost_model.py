"""Cost model: today's mask factory vs (A) Delphi-style JIT HE preprocessing vs (B) hinted-MLWE masking (PGW-LH).
Labels: [m]=measured here or in PGW docs, [c]=cited, [e]=estimate."""
import json
MODELS = {  # layers, sum_m, sum_n per layer (qkv, o, gate_up, down), n_max
  "qwen2.5-0.5b": dict(L=24, sm=1152+896+9728+896, sn=896*3+4864, nmax=4864),
  "qwen2.5-7b":   dict(L=28, sm=4608+3584+37888+3584, sn=3584*3+18944, nmax=18944),
  "llama-3-70b":  dict(L=80, sm=10240+8192+57344+8192, sn=8192*3+28672, nmax=28672),
}
K = 40                    # today's ring bits
Q_BITS, T_BITS = 56, 34   # LH: prime q < 2^56, plaintext space for W x (|Wx| < 2^32.2 at n=18944, A12)
DIM, NP = 2560, 256       # MLWE total dimension (rank 10 x ring 256), group size G = NP tokens
NS_PER_ROW_TOKEN = 46.3   # [m] client_bench k=10, scalar C, 1 core
T4_FACTORY_TOK_H = 169e3  # [m] PGW factory, 7B, one T4
T4_ONLINE_S_TOK = 0.014   # [m] PGW sizing.md: 8B service T4 batch 32 (K/8 = 5 int8 digit GEMMs)
H100_ONLINE_S_TOK = 0.002 # [e] PGW sizing.md estimate for A100/H100 batch 32
H100_USD_H, T4_USD_H = 3.0, 0.35  # [e] on-demand-ish cloud prices
out = {}
for name, d in MODELS.items():
    L, sm, sn = d["L"], d["sm"], d["sn"]
    rows = L * (sm + sn)
    today_store = L * sm * K / 8
    up_today, down_today = L * sn * K / 8, L * sm * K / 8
    digits_today, digits_lh = K // 8, -(-Q_BITS // 8)
    r = dict(
      today_storage_MB_per_token=today_store / 1e6,
      today_storage_TB_per_1M=today_store / 1e6,
      today_factory_T4_h_per_1M=(1e6 / T4_FACTORY_TOK_H) * (L * sm * sn) / (28 * 49664 * 29696) if False else None,
      today_net_MB_per_token=(up_today + down_today) / 1e6,
      # ---- B: hinted MLWE masking
      LH_hint_GB=L * sm * DIM * 8 / 1e9,
      LH_A_public_MB=d["nmax"] * DIM * 8 / 1e6,
      LH_group_buffer_GB=NP * rows * 8 / 1e9,
      LH_breakeven_tokens=(L * sm * DIM * 8) / today_store,
      LH_client_cpu_s_per_token_1core=rows * NS_PER_ROW_TOKEN * 1e-9,
      LH_client_cores_for_1M_per_day=rows * NS_PER_ROW_TOKEN * 1e-9 * 1e6 / 86400,
      LH_hint_stream_MB_per_token=L * sm * DIM * 8 / NP / 1e6,
      LH_up_MB_per_token=L * sn * Q_BITS / 8 / 1e6,
      LH_down_full_z_MB_per_token=L * sm * Q_BITS / 8 / 1e6,
      LH_down_trunc_z_MB_per_token=L * sm * (T_BITS + 2) / 8 / 1e6,
      LH_gpu_digit_ratio_vs_online=digits_lh / digits_today,
      LH_extra_H100_s_per_1M=1e6 * H100_ONLINE_S_TOK * (digits_lh / digits_today - 1) * (L*sm*sn)/(28*49664*29696) * 0 + 0,
    )
    # scale online GPU time by MAC count relative to 7B (sizing numbers are for 8B ~ 7B scale)
    macs = L * (0)  # placeholder
    out[name] = r
# GPU: per-token MACs (exact)
W_MACS = {"qwen2.5-0.5b": 24*(1152*896+896*896+9728*896+896*4864),
          "qwen2.5-7b": 28*(4608*3584+3584*3584+37888*3584+3584*18944),
          "llama-3-70b": 80*(10240*8192+8192*8192+57344*8192+8192*28672)}
H100_INT8_TOPS = 1979e12  # [c] NVIDIA H100 SXM datasheet, dense INT8
for name in out:
    macs = W_MACS[name]
    r = out[name]
    r["W_MACs_per_token"] = macs
    for util in (0.1, 0.3):
        r[f"LH_H100_GPU_s_per_1M_at_{int(util*100)}pct"] = 1e6 * macs * 2 * 7 / (H100_INT8_TOPS * util)
        r[f"today_online_H100_GPU_s_per_1M_at_{int(util*100)}pct"] = 1e6 * macs * 2 * 5 / (H100_INT8_TOPS * util)
    # today factory on T4, scaled by MACs from the measured 7B rate
    r["today_factory_T4_h_per_1M"] = 1e6 / T4_FACTORY_TOK_H * macs / W_MACS["qwen2.5-7b"]
    r["A_JIT_HE_extra_up_MB_per_token"] = r["LH_up_MB_per_token"]
    r["A_JIT_HE_extra_down_MB_per_token"] = r["LH_down_full_z_MB_per_token"]
    r.pop("LH_extra_H100_s_per_1M"); r.pop("today_storage_TB_per_1M")
print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in out.items()}, indent=1))
