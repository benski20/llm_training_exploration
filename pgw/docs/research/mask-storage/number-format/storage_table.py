"""Mask-store bytes per token for per-group ring sizes K (bit-packed), and Freivalds check vectors needed."""
import json, math, sys
MODELS = {  # L, per-group output features m_g, input features n_g
    "qwen2.5-0.5b": (24, {"qkv": 1152, "o": 896, "gate_up": 9728, "down": 896}, {"qkv": 896, "o": 896, "gate_up": 896, "down": 4864}),
    "qwen2.5-1.5b": (28, {"qkv": 2048, "o": 1536, "gate_up": 17920, "down": 1536}, {"qkv": 1536, "o": 1536, "gate_up": 1536, "down": 8960}),
    "qwen2.5-7b": (28, {"qkv": 4608, "o": 3584, "gate_up": 37888, "down": 3584}, {"qkv": 3584, "o": 3584, "gate_up": 3584, "down": 18944}),
    "llama-3-70b": (80, {"qkv": 10240, "o": 8192, "gate_up": 57344, "down": 8192}, {"qkv": 8192, "o": 8192, "gate_up": 8192, "down": 28672}),
}

def mb_per_token(model, kg):
    L, m, _ = MODELS[model]
    return L * sum(m[g] * kg[g] for g in m) / 8 / 1e6

def checks_needed(h, target=20):
    return math.ceil(target / h) if h > 0 else float("inf")

def h_static(K, n, A=12):  # pgw.core.freivalds.soundness_bits per vector
    return max(0, K - math.ceil(math.log2(4 * (2 ** (A - 1) - 1) * 127 * n)))

def h_tight(K, Kw, log2B):  # Kw: ring the service returns / Freivalds runs in (>= K)
    return Kw - min(K - 1, math.floor(log2B) + 1)

if __name__ == "__main__":
    for model in MODELS:
        print(model, "K40 today", round(mb_per_token(model, {g: 40 for g in MODELS[model][1]}), 3),
              "| static h per vector at K40A12:", {g: h_static(40, n) for g, n in MODELS[model][2].items()})
