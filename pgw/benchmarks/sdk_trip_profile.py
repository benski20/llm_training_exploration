"""Round trips per decoded token and per-trip overhead of the client library's default path (CPU, loopback TLS).

Starts the SDK tests' provider (real ``pgw service``: TLS, TCP+TLS trip lane, masked LM head, accounts), then one
``pgw.OpenAI`` client in this process (default stack unless ``--opt k=v`` overrides an :class:`SDKOptions` field,
e.g. ``--opt spec=lookup``). For a warm, streamed, greedy chat it reports from the runtime's own counters:

* trips per decoded token (lane trips + stream-lane trips, counted exactly; the prompt's trips are subtracted by
  running the same prompt with ``max_tokens = 1`` first),
* the decode step time per token and per trip, and its split into the provider's reported compute + queue time
  and the rest (``o``: framing, TLS records, syscalls, Python and native hops on both ends, gateway masking),
* the client's CPU per token (all threads of this process), the native driver's phase counters per trip
  (prepare / finish / float ops per linear slot) and the lane channel's split of a trip (wait beyond the provider's
  compute, send + reply read, spin budget),
* the bare lane floor: the median round trip of mask-free 1-row dummy calls over the same lane (the gateway's
  start-up RTT probe), which is network + TLS + framing + provider dispatch without the GEMM.

    PYTHONPATH=src:. python benchmarks/sdk_trip_profile.py --tokens 48 --runs 3 [--opt key=value ...]

The provider computes on the same 4-vCPU box, so absolute numbers include the provider's CPU GEMMs; compare
variants. Output: one JSON document on stdout.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
MODEL = "qwen2.5-0.5b-instruct"
CODE = """def summarize(orders):
    total = 0
    count = 0
    for order in orders:
        if order.status != "paid":
            continue
        total += order.amount
        count += 1
    average = total / count if count else 0.0
    return {"total": total, "count": count, "average": average}
"""
PROMPTS = {
    "chat": [{"role": "user", "content": "Write one sentence about the sea."}],
    # a copy-heavy request (code edit): what prompt lookup is for
    "edit": [{"role": "user", "content": "Rename the variable `total` to `paid_total` in this function. Reply with "
              "the full function only, no explanation.\n\n" + CODE}],
}
PROMPT = PROMPTS["chat"]


def _pack(out: Path) -> Path:
    from huggingface_hub import snapshot_download

    from pgw.modelpack import pack_checkpoint
    from pgw.sdk.provisioning import PINNED_MODELS

    pin = PINNED_MODELS[MODEL]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    src = snapshot_download(pin.hf_repo, revision=pin.revision, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    pack_checkpoint(src, out, name=MODEL, revision=pin.revision)
    return out


def _counters(rt: Any) -> dict[str, float]:
    be = rt.backend
    st = be.stats.snapshot()
    out = {"backend_round_trips": st["round_trips"], "remote_s": st["remote_seconds"], "local_s": st["local_seconds"],
           "decode_calls": st["decode_calls"], "rows": st["rows"], "calls": st["calls"],
           "local_calls": st["local_calls"], "pad_rows": st["pad_rows"], "dummy_calls": st["dummy_calls"]}
    ls = getattr(rt.client, "lane_stats", None)
    if ls is None:
        ls = getattr(getattr(rt.client, "inner", None), "lane_stats", None)
    if ls is not None:
        out["lane_trips"] = ls.trips
        out["lane_fallbacks"] = ls.fallbacks
        out["lane_bytes"] = ls.bytes_sent + ls.bytes_received
    for obj in (rt.client, getattr(rt.client, "inner", None)):
        cs = getattr(obj, "channel_stats", None)
        if callable(cs):
            for k, v in cs().items():
                if isinstance(v, int | float):
                    out["ch_" + k] = float(v)
            break
    inst = rt.client
    out["instrumented_trips"] = int(getattr(inst, "round_trips", 0))
    tim = getattr(inst, "timing", None)
    if tim is not None:
        snap = tim.snapshot()
        for k in ("rtt_s", "queue_s", "compute_s", "wire_s", "trips"):
            out["timing_" + k] = sum(v[k] for v in snap.values())
    nd = getattr(be, "native_decode", None)
    if nd is not None and getattr(nd, "totals", None) is not None:
        t = list(nd.totals)
        out["native_passes"], out["native_trips"] = float(t[0]), float(t[2])
    return out


def _phases() -> list[int] | None:
    from pgw.core.native import fast

    f = getattr(fast, "decode_phase_ns", None) if fast is not None else None
    return list(f()) if f is not None else None


def _phase_summary(a: list[int] | None, b: list[int] | None) -> dict[str, float] | None:
    """Mean us per native trip of each driver phase per linear slot (prepare / finish / float ops)."""
    if a is None or b is None:
        return None
    d = [y - x for x, y in zip(a, b, strict=True)]
    out: dict[str, float] = {}
    for i, name in enumerate(("qkv", "o", "gate_up", "down")):
        n = max(1, d[12 + i])
        out[name + "_prepare"] = d[i] / n / 1e3
        out[name + "_finish"] = d[4 + i] / n / 1e3
        out[name + "_float"] = d[8 + i] / n / 1e3
    out["trips"] = float(sum(d[12:16]))
    return out


def _delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: b[k] - a.get(k, 0.0) for k in b}


def _chat(c: Any, tokens: int) -> tuple[list[float], list[Any]]:
    times, lp = [], []
    t0 = time.perf_counter()
    for ch in c.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, temperature=0, max_tokens=tokens,
                                        logprobs=True, extra_body={"ignore_eos": True}):
        if ch.choices and ch.choices[0].logprobs:
            times.append(time.perf_counter())
            lp += [(x.token, x.logprob) for x in ch.choices[0].logprobs.content]
    return [t0, *times], lp


STEPS: list[tuple[float, float, int]] = []
PARTS: dict[str, float] = {}


def _instrument() -> None:
    """Record (start, end, sequences) of every engine forward pass (gateway-local timing; nothing else changes)."""
    from pgw.engine.model import TransformerModel

    orig = TransformerModel.forward_batch

    def timed(self: Any, seqs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig(self, seqs)
        finally:
            STEPS.append((t0, time.perf_counter(), len(seqs)))

    TransformerModel.forward_batch = timed  # type: ignore[method-assign]
    from pgw.engine.backends import MaskedBackend
    from pgw.engine.decode_native import NativeDecode

    def wrap(cls: Any, name: str) -> None:
        f = getattr(cls, name)

        def w(*a: Any, **k: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return f(*a, **k)
            finally:
                PARTS[name] = PARTS.get(name, 0.0) + time.perf_counter() - t0
                PARTS[name + "_n"] = PARTS.get(name + "_n", 0.0) + 1

        setattr(cls, name, w)

    wrap(NativeDecode, "_handback")
    wrap(NativeDecode, "run")
    wrap(MaskedBackend, "forward_many")


def run(args: argparse.Namespace, key: str, pack: Path, root: Path) -> dict[str, Any]:
    from pgw import OpenAI

    _instrument()
    from pgw.sdk import LocalPackProvisioner

    opts: dict[str, Any] = {}
    for kv in args.opt:
        k, v = kv.split("=", 1)
        opts[k] = json.loads(v) if v[:1] in "[{0123456789-" or v in ("true", "false", "null") else v
    c = OpenAI(api_key=key, cache_dir=root / "c", state_dir=root / "s", mask_scheme="plain", mask_workers=1,
               mask_target_slots=8 * (args.tokens + 64) * (args.runs + 2), mask_segment_slots=128,
               provisioner=LocalPackProvisioner({MODEL: pack}), **opts)
    c.warmup(MODEL)
    c.chat.completions.create(model=MODEL, messages=PROMPT, max_tokens=1, temperature=0)
    eng = c._engine
    rt = eng.models[MODEL].runtime
    # prompt-only trips: the same prompt with one token
    a = _counters(rt)
    _chat(c, 1)
    eng.call(rt.generator.wait_drained(), timeout_s=600)
    b = _counters(rt)
    prompt_d = _delta(a, b)
    res: list[dict[str, Any]] = []
    ref = None
    for i in range(args.runs + 1):
        a = _counters(rt)
        STEPS.clear()
        PARTS.clear()
        ph0 = _phases()
        cpu0 = time.process_time()
        times, lp = _chat(c, args.tokens)
        cpu1 = time.process_time()
        ph1 = _phases()
        steps = [s_ for s_ in STEPS if times[1] <= s_[0] and s_[1] <= times[-1] + 1e-3]  # passes of the answer
        eng.call(rt.generator.wait_drained(), timeout_s=600)  # strict speculation: the dummy passes still due
        all_passes = len(STEPS)
        b = _counters(rt)
        d = _delta(a, b)
        ref = lp if ref is None else ref
        assert lp == ref, "the same request gave different tokens"
        if i == 0:
            continue
        gaps = [t2 - t1 for t1, t2 in itertools.pairwise(times[1:])]
        n_dec = len(times) - 2  # tokens after the first (the first comes with the prompt pass)
        dec = {k: d[k] - prompt_d.get(k, 0.0) for k in d}
        trips = dec.get("lane_trips", 0.0) + max(0.0, dec.get("instrumented_trips", 0.0) - dec.get("lane_trips", 0.0))
        trips = max(trips, dec.get("backend_round_trips", 0.0))
        res.append({"ttft_ms": 1e3 * (times[1] - times[0]), "decode_ms_per_token": 1e3 * statistics.median(gaps),
                    "decode_tok_s": n_dec / (times[-1] - times[1]),
                    "trips_per_token": trips / n_dec, "backend_trips_per_token": dec["backend_round_trips"] / n_dec,
                    "lane_trips_per_token": dec.get("lane_trips", 0.0) / n_dec,
                    "remote_ms_per_trip": 1e3 * dec["remote_s"] / max(1.0, dec["backend_round_trips"]),
                    "service_ms_per_trip": 1e3 * (dec.get("timing_queue_s", 0.0) + dec.get("timing_compute_s", 0.0))
                    / max(1.0, dec.get("timing_trips", 0.0)),
                    "lane_trip_us": dec.get("ch_trip_us", 0.0) / max(1.0, dec.get("ch_trips", 0.0)),
                    "provider_compute_us": dec.get("ch_compute_us", 0.0) / max(1.0, dec.get("ch_trips", 0.0)),
                    "provider_queue_us": dec.get("ch_queue_us", 0.0) / max(1.0, dec.get("ch_trips", 0.0)),
                    "gateway_local_us_per_trip": 1e6 * dec["local_s"] / max(1.0, dec["backend_round_trips"]),
                    "gap_p90_ms": 1e3 * sorted(gaps)[int(0.9 * len(gaps))], "gap_max_ms": 1e3 * max(gaps),
                    "passes_per_token": len(steps) / n_dec,
                    "tokens_per_pass": n_dec / max(1, len(steps)),
                    "critical_trips_per_token": (dec["backend_round_trips"] / max(1, all_passes - 1)) * len(steps)
                    / n_dec,
                    "provider_passes_per_token": (all_passes - 1) / n_dec,
                    "pass_ms": 1e3 * statistics.median(e - b for b, e, _ in steps) if steps else 0.0,
                    "outside_pass_ms_per_token": (1e3 * (times[-1] - times[1]) - 1e3 * sum(e - b for b, e, _ in steps))
                    / n_dec,
                    "parts_ms_per_token": {k: (1e3 * v if not k.endswith("_n") else v) / n_dec
                                           for k, v in PARTS.items()},
                    "rows_per_trip": dec["rows"] / max(1.0, dec["backend_round_trips"]),
                    "client_cpu_ms_per_token": 1e3 * (cpu1 - cpu0) / (n_dec + 1),
                    # the client lane's own view per trip: reply wait beyond the provider's reported compute
                    # (network, provider lane I/O, wake-up), send + reply read (TLS, syscalls), spin budget
                    "lane_wait_beyond_compute_us": (dec.get("ch_wait_us", 0.0) - dec.get("ch_compute_us", 0.0)
                                                    - dec.get("ch_queue_us", 0.0)) / max(1.0, dec.get("ch_trips", 0.0)),
                    "lane_io_us": (dec.get("ch_trip_us", 0.0) - dec.get("ch_wait_us", 0.0))
                    / max(1.0, dec.get("ch_trips", 0.0)),
                    "lane_spin_us": dec.get("ch_spin_us", 0.0) / max(1.0, dec.get("ch_trips", 0.0)),
                    "native_phase_us_per_trip": _phase_summary(ph0, ph1),
                    "local_calls_per_token": dec["local_calls"] / n_dec,
                    "raw": dec})
    rtt = rt._measure_rtt(trips=31) if hasattr(rt, "_measure_rtt") else None
    st = c.status()["models"][MODEL]
    skip = ("raw", "parts_ms_per_token", "native_phase_us_per_trip")
    summary = {k: statistics.median(r[k] for r in res) for k in res[0] if k not in skip}
    summary["native_phase_us_per_trip"] = res[-1]["native_phase_us_per_trip"]
    summary["parts_ms_per_token"] = res[-1]["parts_ms_per_token"]
    summary["ms_per_trip_wall"] = summary["decode_ms_per_token"] / max(1e-9, summary["trips_per_token"])
    summary["lane_floor_ms"] = None if rtt is None else 1e3 * rtt
    out = {"opts": opts, "summary": summary, "runs": res, "lane": st.get("lane"), "native": st.get("native"),
           "masked_head": st.get("masked_head"), "spec": rt.status().get("speed", {}).get("spec_stats"),
           "logprobs": ref}
    c.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="chat")
    ap.add_argument("--opt", action="append", default=[], help="SDKOptions override key=value (repeatable)")
    ap.add_argument("--pack", default=None, help="an existing 0.5B pack directory (default: pack into a temp dir)")
    ap.add_argument("--provider-env", action="append", default=[], help="extra provider env KEY=VALUE")
    args = ap.parse_args()
    global PROMPT
    PROMPT = PROMPTS[args.prompt]
    from tests.sdk.conftest import make_pki, start_provider, stop_provider

    with tempfile.TemporaryDirectory(prefix="pgw-trip-prof-") as d:
        root = Path(d)
        for sub in ("pki", "provider", "client"):
            (root / sub).mkdir()
        pack = Path(args.pack) if args.pack else _pack(root / "pack" / "pack")
        extra = dict(kv.split("=", 1) for kv in args.provider_env)

        prov = start_provider(root / "provider", pack, make_pki(root / "pki"), extra_env=extra)
        try:
            out = run(args, prov.new_key(), pack, root / "client")
        finally:
            stop_provider(prov)
    lp = out.pop("logprobs")
    out["logprobs_sha"] = __import__("hashlib").sha256(json.dumps(lp).encode()).hexdigest()[:16]
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
