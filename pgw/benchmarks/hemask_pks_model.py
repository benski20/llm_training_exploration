"""Per-model sizing of the encrypted mask factory: today's scheme (SPEC 7.9) vs the pks-v3 prototype.

Inputs are per-unit CPU costs measured on this repository's kernels (defaults: the 2026-11-22 local run on a
shared 4-vCPU Xeon @ 2.3 GHz with AVX-512, single thread, process CPU time, see
``research/plans/he-compact-hint-pks.md``); everything else is arithmetic on the real Qwen2.5 shapes.

    python benchmarks/hemask_pks_model.py                      # measured unit costs
    python benchmarks/hemask_pks_model.py --pks-dec-ns 5 --enc-ns 18 --r-ns 3   # projected tuned kernels
"""

from __future__ import annotations

import argparse

MODELS = {  # layers, {kind: (m, n)} (masked LM head not included)
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "qwen2.5-32b": (64, {"qkv": (7168, 5120), "o": (5120, 5120), "gate_up": (55296, 5120), "down": (5120, 27648)}),
    "qwen2.5-72b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (59136, 8192), "down": (8192, 29568)}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--today-dec-ns", type=float, default=39.0, help="today, hint planes on disk/RAM (decrypt=ntt)")
    ap.add_argument("--today-pre-dec-ns", type=float, default=16.4, help="today, hint in NTT form (2x RAM)")
    ap.add_argument("--pks-dec-ns", type=float, default=8.8, help="pks-v3 decrypt16 kernel")
    ap.add_argument("--enc-ns", type=float, default=32.0, help="encryption per input and slot (same both)")
    ap.add_argument("--r-ns", type=float, default=5.8, help="r keystream per input and slot (same both)")
    ap.add_argument("--ksk-ms", type=float, default=1.7, help="pks-v3 reply key per (linear, group)")
    ap.add_argument("--rates", type=float, nargs="+", default=[50.0, 75.0, 90.0], help="slots/s")
    ap.add_argument("--reply-ring", type=int, default=4096, help="4096 (prototype) or 8192 (production)")
    a = ap.parse_args()
    ksk_bytes = 16 * a.reply_ring * 13.5  # 16 components, log2(q1 P) = 108 bits
    print("| model | sum m | sum n | linears | GEMM int8 MAC/slot | scheme | hint (customer) | up B/slot | down B/slot "
          "| CPU ms/slot | " + " | ".join(f"cores @{r:g}/s" for r in a.rates) + " |")
    print("|" + "---|" * (10 + len(a.rates)))
    for name, (layers, kinds) in MODELS.items():
        sm = layers * sum(m for m, _ in kinds.values())
        sn = layers * sum(n for _, n in kinds.values())
        lin = layers * len(kinds)
        mac = 7 * layers * sum(m * n for m, n in kinds.values())
        enc = sn * (a.enc_ns + a.r_ns) / 1e6
        rows = [
            ("today v2, hint planes", sm * 24576, sn * 6.625, sm * 4.375, sm * a.today_dec_ns / 1e6 + enc),
            ("today v2, NTT-form hint", sm * 49152, sn * 6.625, sm * 4.375, sm * a.today_pre_dec_ns / 1e6 + enc),
            ("pks-v3, compact hint", sm * 256 * 6, sn * 7 + lin * ksk_bytes / 256, sm * 4.5,
             sm * a.pks_dec_ns / 1e6 + enc + lin * a.ksk_ms / 256),
            ("pks-v3, hint streamed", 0, sn * 7 + lin * ksk_bytes / 256, sm * (4.5 + 6),
             sm * a.pks_dec_ns / 1e6 + enc + lin * a.ksk_ms / 256),
        ]
        for scheme, hint, up, down, ms in rows:
            cores = " | ".join(f"{ms * r / 1e3:.1f}" for r in a.rates)
            print(f"| {name} | {sm / 1e6:.2f}M | {sn / 1e6:.2f}M | {lin} | {mac / 1e9:.0f} G | {scheme} | "
                  f"{hint / 1e9:.1f} GB | {up / 1e6:.2f} MB | {down / 1e6:.2f} MB | {ms:.1f} | {cores} |")


if __name__ == "__main__":
    main()
