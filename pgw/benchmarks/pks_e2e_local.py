#!/usr/bin/env python3
"""Local end-to-end run of the pks-v3 encrypted mask factory on a real pack with the in-process provider.

The real :class:`pgw.factory.MaskGenerator` (every-row self-check, Freivalds check vectors) with
:class:`pgw.hemask.pks_backend.PKSMaskBackend`; the provider is the service's own v3 item function
(:class:`pgw.service.pks_provider.PKSProvider` on the service's :class:`ComputeKernel`) called over in-process
wire bytes. The segments are compared bit for bit with local plaintext generation (``W8 r mod 2^32``, which
the v1/v2 schemes also produce exactly). Reports the customer's stage CPU, bytes per slot and hint size.

    PYTHONPATH=src python benchmarks/pks_e2e_local.py --pack ~/.pgw-client-demo/packs/qwen2.5-0.5b-instruct \
        --slots 512 --skip gate_up,down --work /tmp/pks-e2e
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time
from pathlib import Path

import torch


def _cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--slots", type=int, default=512)
    ap.add_argument("--skip", default="", help="linear kinds to skip (comma list), e.g. gate_up,down")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--groups-per-call", type=int, default=2)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from pgw.core.prng import new_seed
    from pgw.core.ring import RingParams
    from pgw.factory import MaskGenerator
    from pgw.hemask.pks_backend import InProcessPKSProvider, PKSHintStore, PKSMaskBackend, RemotePKS, linear_keys
    from pgw.modelpack import ModelPack

    pack = ModelPack.open(a.pack, verify="full", components=("linears",))
    ring = RingParams.from_name(a.ring)
    skip = frozenset(k for k in a.skip.split(",") if k)
    keys = linear_keys(pack, skip)
    t0, c0 = time.perf_counter(), _cpu()
    store = PKSHintStore.open_or_build(a.work / "hints", pack, only=keys)
    t_hint, c_hint = time.perf_counter() - t0, _cpu() - c0
    from pgw.service.kernels import ComputeKernel

    kern = {}
    for lw in pack.iter_linears():
        if lw.id.key in keys:
            spec = next(s for s in pack.linear_specs() if s.id.key == lw.id.key)
            kern[lw.id.key] = ComputeKernel(spec, lw.w8, device="cpu")
    prov = InProcessPKSProvider(pack, kernels=kern)
    out = {s.id.key: s.out_features for s in pack.linear_specs()}
    remote = RemotePKS(prov, pack.fingerprint, out, store.a_seed)
    be = PKSMaskBackend(pack, ring, store, remote, groups_per_call=a.groups_per_call, skip_kinds=skip)
    seed = new_seed()
    local = MaskGenerator(pack, ring, device="cpu", skip_kinds=skip)
    want = {(g.id.key, g.slot_start): g.wr.clone() for g in local.generate(seed, a.slots, checks=2, chunk_slots=be.preferred_chunk_slots())}
    gen = MaskGenerator(pack, ring, device="cpu", remote=be, skip_kinds=skip)
    t0, c0 = time.perf_counter(), _cpu()
    got = {(g.id.key, g.slot_start): g.wr.clone() for g in gen.generate(seed, a.slots, checks=2, chunk_slots=be.preferred_chunk_slots())}
    wall, cpu = time.perf_counter() - t0, _cpu() - c0
    exact = want.keys() == got.keys() and all(torch.equal(want[k], got[k]) for k in want)
    if not exact:
        k0 = next(k for k in want if k not in got or not torch.equal(want[k], got[k]))
        a0, b0 = want[k0], got.get(k0)
        print("first mismatch", k0, a0.dtype, a0.shape, None if b0 is None else (b0.dtype, b0.shape),
              a0.flatten()[:4].tolist(), None if b0 is None else b0.flatten()[:4].tolist())
    specs = [s for s in pack.linear_specs() if s.id.key in keys]
    sm, sn = sum(s.out_features for s in specs), sum(s.in_features for s in specs)
    st = be.stats
    res = {
        "pack": pack.name, "linears": len(specs), "sum_m": sm, "sum_n": sn, "slots": a.slots, "skip": sorted(skip),
        "exact_vs_local_plaintext": exact, "self_checked_rows": gen.last_stats.self_checked_rows,
        "hint_bytes": store.nbytes, "hint_bytes_per_row": store.nbytes / sm,
        "hint_build_s": round(t_hint, 1), "hint_build_cpu_s": round(c_hint, 1),
        "wall_s": round(wall, 2), "process_cpu_s_incl_provider": round(cpu, 2),
        "customer_encrypt_ms_per_slot": round(st["encrypt_s"] * 1e3 / a.slots, 3),
        "customer_decrypt_ms_per_slot": round(st["decrypt_s"] * 1e3 / a.slots, 3),
        "provider_ms_per_slot": round(st["remote_s"] * 1e3 / a.slots, 3),
        "bytes_up_per_slot": remote.bytes_up / a.slots, "bytes_down_per_slot": remote.bytes_down / a.slots,
        "engine": be.engine, "provider_hint_builds": prov.provider.builds,
    }
    print(json.dumps(res, indent=1))
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    if not exact:
        raise SystemExit("pks-v3 masks differ from local generation")


if __name__ == "__main__":
    main()
