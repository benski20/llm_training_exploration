"""Per-model sizing of the encrypted mask factory: today's scheme (SPEC 7.9) vs pks-v3, before / after the
tuned customer kernels.

Inputs are per-unit CPU costs measured on this repository's kernels (single thread, ``rust/pgw-pks``
``pks-bench``; shared 4-vCPU Xeon @ 2.3 GHz with AVX-512, see ``research/plans/he-compact-hint-pks.md``
sections 7.5 and 7.7); everything else is arithmetic on the real Qwen2.5 shapes. "cores" = customer CPU
seconds per second at that single-thread speed (excluding the factory's every-row check and I/O).

    python benchmarks/hemask_pks_model.py                       # 25 and 50 slots/s, AVX-512 kernels
    python benchmarks/hemask_pks_model.py --isa avx2            # the AVX2 copies (CPUs without AVX-512)
    python benchmarks/hemask_pks_model.py --rates 25 50 90
"""

from __future__ import annotations

import argparse

MODELS = {  # layers, {kind: (m, n)} (masked LM head not included)
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "qwen2.5-32b": (64, {"qkv": (7168, 5120), "o": (5120, 5120), "gate_up": (55296, 5120), "down": (5120, 27648)}),
    "qwen2.5-72b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (59136, 8192), "down": (8192, 29568)}),
}

#: Measured unit costs (ns per output-slot, ns per input-slot incl. CBD noise, ms per reply key).
#: "before": 2026-11-23 kernels (auto-vectorised, section 7.5); "after": hand-written AVX-512 / AVX2 copies
#: (section 7.7), `pks-bench` single thread, groups = 4, 36-bit replies read in place.
KERNELS = {
    "before": {"dec": 10.0, "enc": 35.0, "ksk": 5.6},
    "avx512": {"dec": 6.1, "enc": 17.4, "ksk": 2.3},
    "avx2": {"dec": 9.5, "enc": 27.0, "ksk": 3.4},
}
KSK_BYTES = 16 * 8192 * (24 + 24 + 30 + 30) // 8  # wire bytes of one reply key (per linear and group)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isa", choices=["avx512", "avx2"], default="avx512", help="tuned kernel copy for 'after'")
    ap.add_argument("--today-dec-ns", type=float, default=39.0, help="today, hint planes on disk/RAM (decrypt=ntt)")
    ap.add_argument("--today-pre-dec-ns", type=float, default=16.4, help="today, hint in NTT form (2x RAM)")
    ap.add_argument("--enc-ns", type=float, default=32.0, help="today's encryption per input and slot")
    ap.add_argument("--r-ns", type=float, default=5.8, help="r keystream per input and slot (all schemes, unchanged)")
    ap.add_argument("--pks-dec-ns", type=float, help="override the 'after' pks-v3 decrypt ns per output-slot")
    ap.add_argument("--pks-enc-ns", type=float, help="override the 'after' pks-v3 encrypt ns per input-slot")
    ap.add_argument("--ksk-ms", type=float, help="override the 'after' reply-key ms per (linear, group)")
    ap.add_argument("--rates", type=float, nargs="+", default=[25.0, 50.0], help="slots/s")
    a = ap.parse_args()
    after = dict(KERNELS[a.isa])
    for k, v in (("dec", a.pks_dec_ns), ("enc", a.pks_enc_ns), ("ksk", a.ksk_ms)):
        if v is not None:
            after[k] = v
    print(f"pks-v3 unit costs: before {KERNELS['before']}, after ({a.isa}) {after}; r keystream {a.r_ns} ns")
    print("| model | scheme | customer storage (hint) | up / down MB per slot | CPU ms/slot | "
          + " | ".join(f"cores @{r:g}/s" for r in a.rates) + " | "
          + " | ".join(f"up+down MB/s @{r:g}/s" for r in a.rates) + " |")
    print("|" + "---|" * (5 + 2 * len(a.rates)))
    for name, (layers, kinds) in MODELS.items():
        sm = layers * sum(m for m, _ in kinds.values())
        sn = layers * sum(n for _, n in kinds.values())
        lin = layers * len(kinds)

        def pks(k: dict[str, float]) -> float:
            return sm * k["dec"] / 1e6 + sn * (k["enc"] + a.r_ns) / 1e6 + lin * k["ksk"] / 256

        enc_today = sn * (a.enc_ns + a.r_ns) / 1e6
        rows = [
            ("today v2, NTT-form hint", sm * 49152, sn * 6.625, sm * 4.375, sm * a.today_pre_dec_ns / 1e6 + enc_today),
            ("pks-v3, before", sm * 256 * 6, sn * 7 + lin * KSK_BYTES / 256, sm * 4.5, pks(KERNELS["before"])),
            (f"**pks-v3, after ({a.isa})**", sm * 256 * 6, sn * 7 + lin * KSK_BYTES / 256, sm * 4.5, pks(after)),
        ]
        for scheme, hint, up, down, ms in rows:
            cores = " | ".join(f"{ms * r / 1e3:.1f}" for r in a.rates)
            bw = " | ".join(f"{(up + down) * r / 1e6:.0f}" for r in a.rates)
            print(f"| {name} | {scheme} | {hint / 1e9:.1f} GB | {up / 1e6:.1f} / {down / 1e6:.1f} | {ms:.1f} | {cores} "
                  f"| {bw} |")
        print(f"|  | (reply keys: {lin} linears x {KSK_BYTES / 1e6:.2f} MB per 256-slot group = "
              f"{lin * KSK_BYTES / 256 / 1e6:.2f} MB of the upload per slot) |" + " |" * (3 + 2 * len(a.rates)))


if __name__ == "__main__":
    main()
