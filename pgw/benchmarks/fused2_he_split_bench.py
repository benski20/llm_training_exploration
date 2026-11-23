"""Customer CPU per fused2 slot with the encrypted digit-split factory (K56 masks as digits on mlwe v2 / pks-v3),
at a checkpoint's shapes, against the plain CPU fused2 factory (opt-in research tool; ``pgw.factory.fused2_he``).

Per fused2 linear shape of one layer (``qkv/f2``, ``o``, ``gate_up/f2``, ``down``) with random public int8 weights
(clipped normal, a realistic row norm) the script runs ``--slots`` mask rows through

* ``plain``: :class:`pgw.factory.fused2_gen.Fused2MaskGenerator`'s exact CPU product (``cpu_f64`` digit GEMMs);
* ``mlwe``: :class:`pgw.factory.fused2_he.DigitSplitProducts` over :class:`pgw.hemask.backend.HEMaskBackend` at K40
  (4 x 14-bit digits; the provider is simulated in-process and its time is **excluded**: it is the provider's);
* ``pks-v3``: the same over :class:`pgw.hemask.pks_backend.PKSMaskBackend` at K32 (7 x 8-bit digits; the service's
  own v3 item code in-process, excluded likewise).

To bound disk and time, every linear keeps its full input width ``n`` (encryption cost) but only ``--rows`` output
rows (hints, decryption and plain GEMMs scale linearly in the output rows ``m``); the per-slot cost is extrapolated
to the full ``m``: ``customer(n, m) = encrypt(n) + split + (decrypt + recombine)(rows) * m / rows``. Every result is
checked bit for bit against the exact uint64 reference before it is timed. Prints JSON.

    python benchmarks/fused2_he_split_bench.py --model qwen2.5-7b --slots 256 --rows 256 --schemes plain,mlwe,pks-v3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

SHAPES = {"qwen2.5-7b": (3584, 4608, 37888, 18944, 28), "qwen2.5-0.5b": (896, 1152, 9728, 4864, 24),
          "qwen2.5-32b": (5120, 7168, 55296, 27648, 64), "qwen2.5-72b": (8192, 10240, 59136, 29568, 80)}


def _weights(model: str, rows: int, seed: int = 0) -> tuple[Any, dict[str, tuple[int, int]]]:
    d, qkv, gu, inter, _layers = SHAPES[model]
    full = {"0.qkv/f2": (qkv, d), "0.o": (d, d), "0.gate_up/f2": (gu, d), "0.down": (d, inter)}
    g = torch.Generator().manual_seed(seed)
    lins = []
    for key, (m, n) in full.items():
        mm = min(rows, m)
        w = (torch.randn((mm, n), generator=g) * 24.0).round().clamp(-127, 127).to(torch.int8)
        lins.append(SimpleNamespace(key=key, w8=w, m=mm, n=n))
    return SimpleNamespace(linears=lambda: lins), full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b", choices=sorted(SHAPES))
    ap.add_argument("--slots", type=int, default=256)
    ap.add_argument("--rows", type=int, default=256, help="output rows kept per linear (extrapolated to the full m)")
    ap.add_argument("--schemes", default="plain,mlwe,pks-v3")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--he-encrypt", default="dense", help="mlwe: dense or ntt_pre (demo)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--work-dir", default="", help="hint directory (default: a temp dir, deleted at the end)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    from pgw.core.prng import SeedExpander
    from pgw.core.ring import RingParams
    from pgw.core.ringmm import PreparedWeight
    from pgw.factory.fused2_gen import reference_wr
    from pgw.factory.fused2_he import DigitSplitProducts, Fused2HEPack, he_key
    from pgw.hemask.lwe import matmul_mod64

    w, full = _weights(args.model, args.rows)
    lins = {lin.key: lin for lin in w.linears()}
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="pgw-f2he-"))
    exp = SeedExpander(os.urandom(32))
    rs = {k: exp.ring_uniform_rows(lin.n, 56, 0, args.slots, "r", k) for k, lin in lins.items()}
    refs = {k: reference_wr(rs[k], lins[k].w8, 56) for k in lins}
    layers = SHAPES[args.model][4]
    out: dict[str, Any] = {"model": args.model, "slots": args.slots, "rows_kept": args.rows,
                           "torch_threads": torch.get_num_threads(), "cpus": os.cpu_count(), "schemes": {}}
    try:
        for scheme in [s.strip() for s in args.schemes.split(",") if s.strip()]:
            per_lin: dict[str, Any] = {}
            if scheme == "plain":
                for k, lin in lins.items():
                    pw = PreparedWeight(lin.w8, device="cpu", backend="auto", cache_float=True)
                    pw.matmul(rs[k][:2], 56)  # warm-up
                    t0 = time.perf_counter()
                    y = pw.matmul(rs[k], 56)
                    dt = time.perf_counter() - t0
                    assert np.array_equal(y.numpy().view(np.uint64), refs[k]), k
                    m_full = full[k][0]
                    per_lin[k] = {"gemm_s_per_slot_full_m": dt / args.slots * m_full / lin.m,
                                  "backend": pw.resolved_backend}
                cust = sum(v["gemm_s_per_slot_full_m"] for v in per_lin.values())
            else:
                fp = "f2" * 32
                adapter = Fused2HEPack(w, fp)
                provider_s = [0.0]
                if scheme == "mlwe":
                    from pgw.hemask.backend import HEMaskBackend, HintStore
                    from pgw.hemask.lwe import HEParams

                    q_bits = HEParams(k=40, rank=args.rank).q
                    t0 = time.perf_counter()
                    hints = HintStore.build(work / "mlwe", adapter, args.rank, q_bits=q_bits)
                    hint_s = time.perf_counter() - t0
                    wmap = {he_key(k): lin.w8.numpy() for k, lin in lins.items()}

                    def remote(linear: str, b: np.ndarray, _w: Any = wmap, _ps: Any = provider_s) -> np.ndarray:
                        t = time.perf_counter()
                        y = matmul_mod64(_w[linear], b)
                        _ps[0] += time.perf_counter() - t
                        return y

                    be: Any = HEMaskBackend(adapter, RingParams.from_name("K40A12"), hints, remote, device=None,
                                            groups_per_call=1, encrypt=args.he_encrypt)
                    k_he, b = 40, 14
                else:
                    from pgw.hemask.pks_backend import InProcessPKSProvider, PKSHintStore, PKSMaskBackend, RemotePKS

                    t0 = time.perf_counter()
                    store = PKSHintStore.build(work / "pks", adapter)
                    hint_s = time.perf_counter() - t0
                    inproc = InProcessPKSProvider(adapter)

                    def post(body: bytes, rid: str, decode: Any, _p: Any = inproc, _ps: Any = provider_s) -> Any:
                        t = time.perf_counter()
                        res = _p(body, rid, decode)
                        _ps[0] += time.perf_counter() - t
                        return res

                    remote_p = RemotePKS(post, fp, {he_key(k): lin.m for k, lin in lins.items()}, store.a_seed)
                    be = PKSMaskBackend(adapter, RingParams.from_name("K32A12C27"), store, remote_p, groups_per_call=1)
                    k_he, b = 32, 8
                shapes = {k: (lin.m, lin.n) for k, lin in lins.items()}
                dsp = DigitSplitProducts(be, k=56, k_he=k_he, digit_bits=b, shapes=shapes, scheme=scheme)
                for k, lin in lins.items():
                    # warm-up (engine choice, calibration), then the measured run on fresh stats
                    dsp.matmul(k, rs[k][:min(args.slots, 256)].clone())
                    for key in list(be.stats):
                        be.stats[key] = 0 if isinstance(be.stats[key], int) else 0.0
                    for key in list(dsp.stats):
                        dsp.stats[key] = 0 if isinstance(dsp.stats[key], int) else 0.0
                    provider_s[0] = 0.0
                    t0 = time.perf_counter()
                    y = dsp.matmul(k, rs[k].clone())
                    wall = time.perf_counter() - t0
                    assert np.array_equal(y.numpy().view(np.uint64), refs[k]), (scheme, k)
                    st = dict(be.stats)
                    m_full = full[k][0]
                    scale = m_full / lin.m
                    enc = float(st.get("encrypt_s", 0.0))
                    dec = float(st.get("decrypt_s", 0.0))
                    split, rec = dsp.stats["split_s"], dsp.stats["recombine_s"]
                    per_lin[k] = {
                        "digits": dsp.digits, "wall_s": wall, "provider_s": provider_s[0], "encrypt_s": enc,
                        "decrypt_s": dec, "split_s": split, "recombine_s": rec,
                        "customer_s_per_slot_full_m": (enc + split + (dec + rec) * scale) / args.slots,
                        "provider_s_per_slot_full_m": provider_s[0] * scale / args.slots,
                    }
                cust = sum(v["customer_s_per_slot_full_m"] for v in per_lin.values())
                per_lin["hint_build_s"] = hint_s
            out["schemes"][scheme] = {
                "per_linear": per_lin, "customer_s_per_slot_layer": cust,
                "customer_s_per_slot_full_model": cust * layers,
                "slots_per_s_per_core_full_model": 1.0 / (cust * layers * max(1, torch.get_num_threads())),
                "slots_per_s_full_model_this_box": 1.0 / (cust * layers),
            }
            print(json.dumps({scheme: out["schemes"][scheme]}, indent=1, default=float), flush=True)
    finally:
        if not args.work_dir:
            shutil.rmtree(work, ignore_errors=True)
    if "plain" in out["schemes"]:
        base = out["schemes"]["plain"]["customer_s_per_slot_layer"]
        out["x_plain"] = {s: v["customer_s_per_slot_layer"] / base for s, v in out["schemes"].items()}
    print(json.dumps({k: v for k, v in out.items() if k != "schemes"}, indent=1, default=float), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
