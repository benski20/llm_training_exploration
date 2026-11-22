"""Gateway HE on the GPU with NTTs (pgw.hemask.gpuntt) vs the dense device pass (devmm) and the CPU NTT.

Times full customer-side operations of the encrypted factory on one linear shape and extrapolates to a
model (per-row cost is the same for every linear):

* encryption of ``G`` groups (secret/noise sampling, pad, epilogue, ciphertext on the host):
  ``gpuntt`` (Triton and eager kernels), ``devmm`` (``DeviceHE.encrypt``, one int8 GEMM per request),
  ``ntt_pre`` (CPU, ``PretransformedEncryptor``, needs ``pgw_ntt``);
* decryption of one pass of ``G`` groups over an ``m``-row hint (hint on the device and streamed):
  ``gpuntt`` vs ``devmm``, and the CPU NTT hint product (``ntt.big_planes_mm_small``) for reference.

Every engine's output is checked against the others bit for bit (encryption with fixed seeds).

    python benchmarks/hemask_gpuntt_bench.py --preset qwen2.5-0.5b --rows 4864 --groups 4,16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable

import numpy as np
import torch

from pgw.hemask import gpuntt
from pgw.hemask.devmm import DeviceHE
from pgw.hemask.fastmm import to_planes
from pgw.hemask.lwe import DEGREE, HEParams, MaskCiphertext, a_plane_count, public_matrix, sample_secret

PRESETS = {  # (layers, {kind: (m, n)})
    "qwen2.5-0.5b": (24, {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)}),
    "qwen2.5-7b": (28, {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)}),
    "llama-70b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192), "down": (8192, 28672)}),
}


def timed(fn: Callable[[], object], reps: int, sync: bool) -> float:
    fn()
    if sync:
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen2.5-0.5b", choices=sorted(PRESETS))
    ap.add_argument("--rows", type=int, default=4864, help="encryption rows (linear inputs) timed")
    ap.add_argument("--hint-rows", type=int, default=4864, help="decryption hint rows (linear outputs) timed")
    ap.add_argument("--groups", default="4,16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--engines", default="triton,eager,devmm,ntt_pre")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    sync = dev.type == "cuda"
    layers, kinds = PRESETS[args.preset]
    sum_n = layers * sum(n for _m, n in kinds.values())
    sum_m = layers * sum(m for m, _n in kinds.values())
    p = HEParams(k=32, rank=16)
    n, m = args.rows, args.hint_rows
    gs = [int(x) for x in args.groups.split(",")]
    engines = args.engines.split(",")
    rng = np.random.default_rng(0)
    a = public_matrix(bytes(32), n, p)
    r = rng.integers(0, 1 << p.k, size=(DEGREE * max(gs), n), dtype=np.int64)
    hint = rng.integers(-128, 128, size=(m, p.q // 8 - 1, p.dim), dtype=np.int8)
    res: dict[str, object] = {"preset": args.preset, "device": str(dev), "rows": n, "hint_rows": m,
                              "sum_n": sum_n, "sum_m": sum_m, "cpu": os.cpu_count(),
                              "triton": gpuntt.triton_available(),
                              "gpu": torch.cuda.get_device_name(0) if sync else None}
    enc: dict[str, float] = {}  # seconds per row and group
    dec: dict[str, float] = {}
    check: dict[str, bool] = {}
    ref_b: dict[int, np.ndarray] = {}

    def seeds(g: int) -> list[bytes]:
        return [bytes([1 + i]) * 32 for i in range(g)]

    def record(name: str, g: int, cts: list[MaskCiphertext]) -> None:
        b = np.concatenate([ct.b for ct in cts], axis=1)
        if g not in ref_b:
            ref_b[g] = b
        else:
            check[f"{name}_g{g}"] = bool(np.array_equal(b, ref_b[g]))

    # ---- GPU NTT engines ----
    for kern in ("triton", "eager"):
        if kern not in engines or (kern == "triton" and (not sync or not gpuntt.triton_available())):
            continue
        t0 = time.perf_counter()
        ok = gpuntt.self_test(dev, kernel=kern)
        res[f"self_test_{kern}"] = {"ok": ok, "s": round(time.perf_counter() - t0, 2)}
        if not ok:
            res[f"self_test_{kern}_status"] = gpuntt.status()
            continue
        t0 = time.perf_counter()
        eng = gpuntt.GpuNTTHE(a, p, dev, kernel=kern, groups_per_pass=16)
        if sync:
            torch.cuda.synchronize()
        res[f"init_{kern}_s"] = round(time.perf_counter() - t0, 3)
        res["a_ntt_bytes"] = eng.nbytes
        for g in gs:
            rr = r[:DEGREE * g]
            record(kern, g, eng.encrypt(rr, _seeds=seeds(g)))
            enc[f"{kern}_g{g}"] = timed(lambda rr=rr: eng.encrypt(rr), args.reps, sync) / (n * g)
            cts = eng.encrypt(rr)
            y = rng.integers(0, 1 << 63, size=(m, DEGREE * g), dtype=np.int64).view(np.uint64)
            out = np.zeros((DEGREE * g, m), dtype=np.int64)
            for mode, budget in (("cached", 1 << 34), ("streamed", 0)):
                eng.cache.clear()
                eng.cache.budget = budget
                f = (lambda y=y, cts=cts, out=out: eng.decrypt_into("h", hint, [(y, cts, 0)], out, p.delta_bits, p.k))
                dec[f"{kern}_{mode}_g{g}"] = timed(f, args.reps, sync) / (m * g)
        # breakdown (device-only times; the CPU noise sampler every engine shares)
        g = max(gs)
        smats = [sample_secret(bytes([40 + i]) * 32, p.rank) for i in range(g)]
        sh = eng._secrets_on_device(smats, 56)
        hd = torch.from_numpy(hint).to(dev)
        mb = min(m, 4096)
        th = eng._fwd_planes(hd[:mb], 1)
        brk = {
            "mac_inv_crt_encrypt": timed(lambda: eng._mac(eng.t_a, 0, n, sh), args.reps, sync) / (n * g),
            "fwd_hint_planes": timed(lambda: eng._fwd_planes(hd[:mb], 1), args.reps, sync) / (mb * g),
            "mac_inv_crt_decrypt": timed(lambda: eng._mac(th, 0, mb, sh), args.reps, sync) / (mb * g),
            "secrets_transform": timed(lambda: eng._secrets_on_device(smats, 56), args.reps, sync) / (n * g),
        }
        res[f"breakdown_{kern}_g{g}_us_per_row_group"] = {k: round(v * 1e6, 3) for k, v in brk.items()}
        del hd, th, sh
        del eng
        if sync:
            torch.cuda.empty_cache()

    # CPU noise sampling (CBD, native when built), shared by every engine's encryption
    from pgw.hemask.lwe import sample_noise

    g = max(gs)
    res["sample_noise_us_per_row_group"] = round(1e6 * timed(
        lambda: [sample_noise(bytes([60 + i]) * 32, n, p.eta) for i in range(g)], args.reps, False) / (n * g), 3)

    # ---- dense device pass ----
    if "devmm" in engines:
        planes = to_planes(a)[:, :a_plane_count(p)]
        dv = DeviceHE(planes, p, dev)
        for g in gs:
            rr = r[:DEGREE * g]
            cts, b = dv.encrypt(rr, n, seeds=seeds(g))
            for gi, ct in enumerate(cts):
                ct.b = np.ascontiguousarray(b[:, gi * DEGREE:(gi + 1) * DEGREE])
            record("devmm", g, cts)
            enc[f"devmm_g{g}"] = timed(lambda rr=rr: dv.encrypt(rr, n), args.reps, sync) / (n * g)
            # decryption is compared on the same ciphertexts as the NTT engines (y is random: only the bits matter)
            for mode, budget in (("cached", 1 << 34), ("streamed", 0)):
                dv.cache.clear()
                dv.cache.budget = budget
                y = np.random.default_rng(g).integers(0, 1 << 63, size=(m, DEGREE * g), dtype=np.int64).view(np.uint64)
                out = np.zeros((DEGREE * g, m), dtype=np.int64)
                f = (lambda y=y, cts=cts, out=out: dv.decrypt_into("h", hint, [(y, cts, 0)], out, p.delta_bits, p.k))
                dec[f"devmm_{mode}_g{g}"] = timed(f, args.reps, sync) / (m * g)
        del dv
        if sync:
            torch.cuda.empty_cache()

    # decryption bit-exactness on identical inputs: GPU NTT vs devmm vs CPU NTT, one pass of the smallest G
    g0 = min(gs)
    smats = [sample_secret(bytes([9 + i]) * 32, p.rank) for i in range(g0)]
    cts0 = [MaskCiphertext(b=np.empty((0, DEGREE), dtype=np.uint64), secret=s, rows=DEGREE) for s in smats]
    y0 = rng.integers(0, 1 << 63, size=(min(m, 512), DEGREE * g0), dtype=np.int64).view(np.uint64)
    h0 = np.ascontiguousarray(hint[:y0.shape[0]])
    outs = {}
    for kern in ("triton", "eager"):
        if kern in engines and res.get(f"self_test_{kern}", {}).get("ok"):
            e = gpuntt.GpuNTTHE(a[:8], p, dev, kernel=kern, hint_budget=0)
            o = np.zeros((DEGREE * g0, y0.shape[0]), dtype=np.int64)
            e.decrypt_into("c", h0, [(y0, cts0, 0)], o, p.delta_bits, p.k)
            outs[kern] = o
    if "devmm" in engines:
        dv = DeviceHE(to_planes(a[:8])[:, :a_plane_count(p)], p, dev, hint_budget=0)
        o = np.zeros((DEGREE * g0, y0.shape[0]), dtype=np.int64)
        dv.decrypt_into("c", h0, [(y0, cts0, 0)], o, p.delta_bits, p.k)
        outs["devmm"] = o
    names = list(outs)
    for nm in names[1:]:
        check[f"decrypt_{nm}_vs_{names[0]}"] = bool(np.array_equal(outs[nm], outs[names[0]]))

    # ---- CPU NTT ----
    if "ntt_pre" in engines:
        from pgw.hemask import ntt
        from pgw.hemask.mlwe_ring import PretransformedEncryptor

        if ntt.available():
            cpu_rows = min(n, 2048)
            pe = PretransformedEncryptor(a[:cpu_rows], p)
            for g in gs:
                pe.groups_per_pass = min(g, 8)
                rr = r[:DEGREE * g, :cpu_rows]
                enc[f"ntt_pre_cpu_g{g}"] = timed(lambda rr=rr: pe.encrypt(rr), 1, False) / (cpu_rows * g)
            pe2 = PretransformedEncryptor(a, p)
            record("ntt_pre_cpu", gs[0], pe2.encrypt(r[:DEGREE * gs[0]], _seeds=seeds(gs[0])))
            mm = min(m, 1024)
            smat = np.concatenate(smats, axis=1)
            dec[f"ntt_cpu_hint_product_g{g0}"] = timed(
                lambda: ntt.big_planes_mm_small(hint[:mm], smat, first_byte=1), 1, False) / (mm * g0)
        else:
            res["ntt_pre_cpu"] = "pgw_ntt not built"

    res["check_bit_identical"] = check
    res["encrypt_us_per_row_group"] = {k: round(v * 1e6, 3) for k, v in enc.items()}
    res["decrypt_us_per_row_group"] = {k: round(v * 1e6, 3) for k, v in dec.items()}
    res["encrypt_slots_per_s_model"] = {k: round(DEGREE / (v * sum_n), 1) for k, v in enc.items()}
    res["decrypt_slots_per_s_model"] = {k: round(DEGREE / (v * sum_m), 1) for k, v in dec.items()}
    print(json.dumps(res, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
