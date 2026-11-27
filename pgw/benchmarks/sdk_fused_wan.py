#!/usr/bin/env python3
"""The client library's ``schedule = "standard"`` vs ``"fused"`` over an emulated WAN link (CPU, loopback).

A real ``pgw service`` process (TLS, TCP+TLS trip lane, masked LM head, the head-fold fused2 lane entry and fused2
maskgen) is reached through two :class:`benchmarks.delay_proxy.DelayProxy` instances (its HTTPS port and its lane
port; the provider advertises the proxies' ports), each adding ``rtt / 2`` per direction. For each RTT and schedule a
fresh client (own cache, plain local masks so the factory is off the measured path) answers a warm-up request, then
``--reps`` greedy streamed answers of ``--tokens`` tokens; reported per configuration:

* trips per token (lane calls of the decode passes / decoded tokens, from the backend's counters),
* decode tokens/s (first streamed token to the last; median over reps),
* client CPU ms per token (this process's user + system CPU over the decode, ``resource.getrusage``; the factory
  child is a separate process and is not included),
* bytes up / down per token on the lane, and whether the fused answer equals the fused plaintext reference.

The CPU provider's own compute (K56 products in numpy / Triton-interpreter, a GPU's job) is part of the wall time
here: it is reported (``provider_ms_per_token`` from the lane's reported compute) so the RTT effect can be read.

    PYTHONPATH=src:. python benchmarks/sdk_fused_wan.py --rtt 0 5 20 --tokens 12 --json out.json
    PGW_SDK_E2E_MODEL=qwen ...   # the real Qwen2.5-0.5B-Instruct instead of the tiny test model
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

MODEL = "qwen2.5-0.5b-instruct"
PROMPT = [{"role": "user", "content": "Write one sentence about the sea."}]


def _cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _pack(work: Path) -> Path:
    from pgw.modelpack import pack_checkpoint

    if os.environ.get("PGW_SDK_E2E_MODEL", "tiny").lower() == "qwen":
        from huggingface_hub import snapshot_download

        src: Any = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    else:
        from tests.conftest import _save_tiny  # the tests' tiny random Qwen2 checkpoint

        (work / "tiny-hf").mkdir()
        src = _save_tiny(work / "tiny-hf", "qwen2", seed=0)
    out = work / "pack"
    pack_checkpoint(src, out, name=MODEL)
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    from delay_proxy import DelayProxy

    from tests.sdk.conftest import free_port, make_pki, start_provider, stop_provider
    from tests.sdk.test_fused_schedule import FUSED_ENV, fused_reference

    work = Path(tempfile.mkdtemp(prefix="pgw-fwan-"))
    pack = Path(args.pack) if args.pack else _pack(work)
    pki = make_pki(work)
    results: dict[str, Any] = {"model": os.environ.get("PGW_SDK_E2E_MODEL", "tiny"), "tokens": args.tokens,
                               "reps": args.reps, "configs": []}
    ref = fused_reference(pack, {"temperature": 0, "max_tokens": args.tokens, "logprobs": True,
                                 "extra_body": {"ignore_eos": True}}, PROMPT)
    ref_text = ref["choices"][0]["message"]["content"]
    for rtt in args.rtt:
        port, lane_port = free_port(), free_port()
        px = DelayProxy(("127.0.0.1", port), one_way_s=rtt / 2e3)
        pl = DelayProxy(("127.0.0.1", lane_port), one_way_s=rtt / 2e3)
        pport, plport = px.start(), pl.start()
        (work / f"prov-{rtt}").mkdir()
        prov = start_provider(work / f"prov-{rtt}", pack, pki, extra_env=FUSED_ENV, port=port, lane_port=lane_port,
                              public_port=pport, public_lane_port=plport)
        try:
            for sched in args.schedules:
                results["configs"].append(one(prov, sched, rtt, work, args, ref_text))
                print(json.dumps(results["configs"][-1]), flush=True)
        finally:
            stop_provider(prov)
            px.stop()
            pl.stop()
    return results


def one(prov: Any, sched: str, rtt: float, work: Path, args: argparse.Namespace, ref_text: str) -> dict[str, Any]:
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    root = work / f"client-{sched}-{rtt}"
    c = OpenAI(api_key=prov.new_key(f"wan-{sched}-{rtt}"), provisioner=LocalPackProvisioner({MODEL: prov.pack_dir}),
               cache_dir=root / "cache", state_dir=root / "state", mask_scheme="plain", mask_workers=1,
               mask_target_slots=2048, mask_low_water_slots=256, mask_segment_slots=512, timeout=1200,
               ready_timeout_s=1800, schedule=sched, usage_reports=False)
    try:
        kw = {"temperature": 0, "max_tokens": args.tokens, "extra_body": {"ignore_eos": True}}
        c.chat.completions.create(model=MODEL, messages=PROMPT, **kw)  # warm: provisioning, lane, first masks
        rt = c._engine.models[MODEL].runtime  # type: ignore[attr-defined]
        be = rt.backend
        prov_obj = getattr(be, "provider", None)
        rates, cpus, trips, provider_ms, ups, downs = [], [], [], [], [], []
        text = ""
        for _ in range(args.reps):
            calls0 = _calls(rt)
            cu0 = sum(getattr(prov_obj, "compute_us", []) or []) if prov_obj is not None else 0
            b0 = _bytes(rt)
            first = last = None
            n = 0
            parts = []
            cpu0 = None
            for ch in c.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, **kw):
                if ch.choices and ch.choices[0].delta.content:
                    now = time.perf_counter()
                    if first is None:
                        first, cpu0, calls_first = now, _cpu(), _calls(rt)
                    last = now
                    n += 1
                    parts.append(ch.choices[0].delta.content)
            cpu1 = _cpu()
            text = "".join(parts)
            dec = max(1, n - 1)
            rates.append(dec / max(1e-9, (last - first)))
            cpus.append((cpu1 - cpu0) * 1e3 / dec)
            trips.append((_calls(rt) - calls_first) / dec)
            b1 = _bytes(rt)
            ups.append((b1[0] - b0[0]) / n)
            downs.append((b1[1] - b0[1]) / n)
            if prov_obj is not None and hasattr(prov_obj, "compute_us"):
                provider_ms.append((sum(prov_obj.compute_us) - cu0) / 1e3 / n)
            del calls0
        st = c.status()["models"][MODEL]
        out = {"schedule": sched, "rtt_ms": rtt, "trips_per_token": statistics.median(trips),
               "decode_tok_s": statistics.median(rates), "client_cpu_ms_per_token": statistics.median(cpus),
               "bytes_up_per_token": statistics.median(ups), "bytes_down_per_token": statistics.median(downs),
               "provider_ms_per_token": statistics.median(provider_ms) if provider_ms else None,
               "native_decode": (st.get("fused2") or {}).get("native_decode", st.get("native", {}).get("decode_native")),
               "integrity_failures": st["integrity_failures"], "ring": st["ring"]}
        if sched == "fused":
            out["equals_fused_reference"] = text == ref_text
        return out
    finally:
        c.close()


def _calls(rt: Any) -> int:
    be = rt.backend
    p = getattr(be, "provider", None)
    if p is not None and hasattr(p, "calls"):  # fused2 lane provider: one call per round trip
        return int(p.calls)
    return int(be.stats.snapshot().get("round_trips", 0))


def _bytes(rt: Any) -> tuple[int, int]:
    be = rt.backend
    f2 = getattr(be, "f2", None)
    if f2 is not None:
        return int(f2.bytes_up), int(f2.bytes_down)
    s = be.stats.snapshot()
    return int(s.get("bytes_up", 0)), int(s.get("bytes_down", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtt", type=float, nargs="+", default=[0.0, 5.0, 20.0])
    ap.add_argument("--schedules", nargs="+", default=["standard", "fused"])
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pack", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = run(args)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
