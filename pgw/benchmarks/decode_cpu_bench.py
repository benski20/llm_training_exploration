"""Gateway CPU per generated token of one chat (decode, 1 row), on the gateway's production decode path.

Mirrors the Modal co-located gateway (``pgw_modal_boot.GATEWAY_MODEL_OPTIONS``): ``Generator`` +
``TransformerModel`` + ``MaskedBackend`` with the TripEngine, deferred checks and synchronous decode
passes (what ``lane = "stream"`` turns on), the outbound-noise monitor at the gateway defaults,
``float_threads = 1`` and ``local_linears = attention`` (qkv and o computed on the gateway, which is what
``auto`` chose on Modal). The service is a stub that answers instantly with zeros (all linear weights are
zero), so every CPU second measured is the gateway's: masking, unmasking, Freivalds, noise monitor, local
linears, float ops, LM head, sampling and Python.

The model has the real hidden, intermediate, head and vocabulary shapes but only ``--layers`` layers;
``extrapolated`` scales the per-layer part (everything but the LM head and sampling, measured separately)
to the preset's full layer count.

Usage:
    PYTHONPATH=src python benchmarks/decode_cpu_bench.py --preset 0.5b --layers 4 --tokens 32 --repeat 3 \
        --workdir /tmp/pgw-dec [--local attention|off] [--json out.json] [--profile]

Reports ``cpu_ms_per_token`` (process CPU time, all threads), ``wall_ms_per_token`` and the backend's
``prepare+finish`` share, per token; numbers are for this machine only.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import json
import os
import pstats
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")  # as the pgw CLI does

import torch  # after the OpenMP setting on purpose

PRESETS: dict[str, dict[str, int]] = {
    "0.5b": {"hidden_size": 896, "intermediate_size": 4864, "num_attention_heads": 14,
             "num_key_value_heads": 2, "full_layers": 24, "vocab": 151936},
    "7b": {"hidden_size": 3584, "intermediate_size": 18944, "num_attention_heads": 28,
           "num_key_value_heads": 4, "full_layers": 28, "vocab": 152064},
}


def build_pack(work: Path, preset: str, layers: int) -> Path:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from pgw.modelpack import pack_checkpoint

    p = PRESETS[preset]
    ck, pk = work / f"ckpt-{preset}-{layers}", work / f"pack-{preset}-{layers}"
    if (pk / "manifest.json").exists():
        return pk
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.conftest import _save_byte_tokenizer  # type: ignore[import-not-found]

    cfg = Qwen2Config(
        vocab_size=p["vocab"], hidden_size=p["hidden_size"], intermediate_size=p["intermediate_size"],
        num_hidden_layers=layers, num_attention_heads=p["num_attention_heads"],
        num_key_value_heads=p["num_key_value_heads"], max_position_embeddings=2048,
        tie_word_embeddings=False, torch_dtype="bfloat16",
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16)
    with torch.no_grad():  # zero linear weights: the stub service's all-zero result is the exact answer
        for name, prm in model.named_parameters():
            if "layers." in name and "proj" in name:
                prm.zero_()
    model.eval().save_pretrained(ck, safe_serialization=True)
    _save_byte_tokenizer(ck)
    pack_checkpoint(ck, pk, name=f"synthetic-{preset}", overwrite=True, plain_dtype="bfloat16")
    import shutil

    shutil.rmtree(ck, ignore_errors=True)  # only the pack is needed (disk)
    return pk


def _thread_cpu() -> dict[str, tuple[str, float]]:
    """``{tid: (thread name, CPU seconds)}`` of this process's threads (Linux ``/proc``; ``{}`` elsewhere)."""
    out: dict[str, tuple[str, float]] = {}
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    base = Path("/proc/self/task")
    if not base.is_dir():
        return out
    for t in base.iterdir():
        try:
            name = (t / "comm").read_text().strip()
            f = (t / "stat").read_text().rsplit(")", 1)[1].split()
            out[t.name] = (name.rstrip("0123456789-_") or name, (int(f[11]) + int(f[12])) / tick)
        except OSError:
            continue
    return out


def _cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from pgw.core import RingParams
    from pgw.engine import Generator, MaskedBackend, SamplingParams, TransformerModel
    from pgw.factory import MaskGenerator
    from pgw.gateway.masks import WindowedReservation
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter
    from pgw.modelpack import ModelPack
    from pgw.observability.noise import OutboundNoiseMonitor
    from pgw.wire.messages import ComputeResponse, ComputeResult

    work = Path(args.workdir or tempfile.mkdtemp(prefix="pgw-dec-"))
    work.mkdir(parents=True, exist_ok=True)
    pack = ModelPack.open(build_pack(work, args.preset, args.layers))
    ring = RingParams.from_name(args.ring)
    prompt_len = int(args.prompt_len)
    per = prompt_len + args.tokens
    slots = (per + args.window) * (args.repeat + 1) + 32 + 4 * (prompt_len + 1 + args.window)
    kp = FileKeyProvider(work / "kek", create=True)
    store_dir = work / f"store-{time.time_ns()}"
    store = MaskStore.open(store_dir, kp, fsync=False,
                           counter=FileMonotonicCounter(work / f"counter-{time.time_ns()}", fsync=False))
    t0 = time.perf_counter()
    MaskGenerator(pack, ring).generate_into_store(store, slots, chunk_slots=16, factory_id="dec")
    print(f"factory: {slots} slots in {time.perf_counter() - t0:.1f}s", file=sys.stderr)

    outm = {sp.id.key: sp.out_features for sp in pack.linear_specs()}
    nb = ring.k // 8

    class StubLane:
        """Instant all-zero service answers (zero weights), shaped like the blocking stream lane."""

        sync_stream = True

        def _answer(self, req: Any) -> Any:
            return ComputeResponse(req.request_id, tuple(
                ComputeResult(i.linear, i.rows, bytes(i.rows * outm[i.linear] * nb)) for i in req.items))

        def compute_sync(self, req: Any) -> Any:
            if args.service_delay_ms > 0:  # stand-in for network + provider time (GIL released, like recv)
                time.sleep(args.service_delay_ms / 1e3)
            return self._answer(req)

        async def compute(self, req: Any) -> Any:
            return self._answer(req)

    noise = None if args.no_noise else OutboundNoiseMonitor(ring, sample_rate=0.05, fail_closed=True)
    backend = MaskedBackend(pack, None, StubLane(), ring, pack.fingerprint, min_soundness_bits=20,
                            noise_monitor=noise)
    if args.local != "off":
        backend.configure_local_linears(args.local)
    backend.warm_trip_engine()
    model = TransformerModel(pack, backend)
    model.sync_decode = True
    if args.float_threads:
        model.enable_float_threads(args.float_threads)
    fp = pack.fingerprint

    def factory(n: int) -> Any:
        first = min(n, prompt_len + args.window)
        res = store.reserve(fp, ring, first)
        if first >= n:
            return res
        return WindowedReservation(res, lambda s: store.reserve(fp, ring, s), cap=n, window=args.window)

    gen = Generator(model, None, max_batch=8, session_factory=factory)
    params = SamplingParams(max_tokens=args.tokens, temperature=args.temperature, seed=7, ignore_eos=True,
                            logprobs=args.logprobs)
    prompt = [(3 * i) % (pack.arch.vocab_size - 1) + 1 for i in range(prompt_len)]

    # LM head + sampling alone (no layers), for the extrapolation
    head_ms = _time_head(model, args.logprobs)

    async def once() -> tuple[float, float, Any]:
        c, t = _cpu_s(), time.perf_counter()
        out = await gen.agenerate(prompt, params)
        return time.perf_counter() - t, _cpu_s() - c, out

    timers = _install_timers(backend, model) if args.breakdown else None
    await once()  # warm-up
    if timers is not None:
        timers.clear()
    rows: list[dict[str, Any]] = []
    prof = cProfile.Profile() if args.profile and not args.profile_prefill else None
    outs = []
    thr0 = _thread_cpu()
    for _ in range(args.repeat):
        st0 = backend.stats.snapshot()
        if prof:
            prof.enable()
        wall, cpu, out = await once()
        if prof:
            prof.disable()
        outs.append((list(out.token_ids), list(out.logprobs or [])))
        st1 = backend.stats.snapshot()
        toks = len(out.token_ids)
        rows.append({
            "wall_ms_per_token": 1e3 * wall / toks,
            "cpu_ms_per_token": 1e3 * cpu / toks,
            "backend_ms_per_token": 1e3 * (st1["local_seconds"] - st0["local_seconds"]) / toks,
            "round_trips_per_token": (st1["round_trips"] - st0["round_trips"]) / toks,
        })
    thr1 = _thread_cpu()
    ntk = max(1, args.repeat * args.tokens)
    by_name: dict[str, float] = {}
    for tid, (name, c1) in thr1.items():
        c0 = thr0.get(tid, (name, 0.0))[1]
        by_name[name] = by_name.get(name, 0.0) + (c1 - c0)
    thread_cpu = {k: round(1e3 * v / ntk, 2) for k, v in sorted(by_name.items(), key=lambda kv: -kv[1]) if v > 0}
    # prefill (+ the first token): the prompt's pass alone, a few times (time to first token on the gateway)
    pre_rows: list[float] = []
    pre_timers: dict[str, Any] = {}
    dec_timers: dict[str, list[float]] = {}
    if timers is not None:
        dec_timers = {k: list(v) for k, v in timers.items()}
        timers.clear()
    pparams = SamplingParams(max_tokens=1, temperature=args.temperature, seed=7, ignore_eos=True,
                             logprobs=args.logprobs)
    if args.profile_prefill:
        prof = cProfile.Profile()
    for _ in range(3):
        if prof is not None and args.profile_prefill:
            prof.enable()
        t = time.perf_counter()
        await gen.agenerate(prompt, pparams)
        pre_rows.append(1e3 * (time.perf_counter() - t))
        if prof is not None and args.profile_prefill:
            prof.disable()
    if timers is not None:
        pre_timers = {k: {"wall": round(1e3 * v[0] / 3, 3), "calls": round(v[2] / 3, 1)}
                      for k, v in sorted(timers.items())}
    await gen.aclose()
    ntok = args.repeat * args.tokens
    best = min(rows, key=lambda r: r["cpu_ms_per_token"])
    med = sorted(rows, key=lambda r: r["cpu_ms_per_token"])[len(rows) // 2]
    full, lay = PRESETS[args.preset]["full_layers"], args.layers

    def extra(v: float) -> float:
        return round(head_ms + (v - head_ms) * full / lay, 2)

    result = {
        "preset": args.preset, "layers": lay, "full_layers": full, "ring": args.ring, "local": args.local,
        "noise_monitor": noise is not None, "trip_engine": backend.trip_engine,
        "deferred_check": backend.deferred_check, "float_threads": model.float_threads,
        "local_kinds": sorted(backend.local_policy.get("kinds", ())),
        "tokens": args.tokens, "temperature": args.temperature, "logprobs": args.logprobs,
        "best": {k: round(v, 3) for k, v in best.items()},
        "median": {k: round(v, 3) for k, v in med.items()},
        "head_and_sample_ms": round(head_ms, 3),
        "extrapolated_full_model": {
            "cpu_ms_per_token": extra(med["cpu_ms_per_token"]),
            "wall_ms_per_token": extra(med["wall_ms_per_token"]),
            "backend_ms_per_token": round(med["backend_ms_per_token"] * full / lay, 2),
        },
        "outputs_repeat_identical": all(o == outs[0] for o in outs),
        "prompt_len": prompt_len,
        "thread_cpu_ms_per_token": thread_cpu,
        "prefill_ms": round(sorted(pre_rows)[1], 2),
        **({"prefill_breakdown_ms": pre_timers} if pre_timers else {}),
        **({"breakdown_ms_per_token": {k: {"wall": round(1e3 * v[0] / ntok, 3),
                                           "thread_cpu": round(1e3 * v[1] / ntok, 3),
                                           "calls_per_token": round(v[2] / ntok, 2)}
                                       for k, v in sorted(dec_timers.items())}} if timers is not None else {}),
        "first_tokens": outs[0][0][:8],
    }
    if prof:
        s = io.StringIO()
        st = pstats.Stats(prof, stream=s).sort_stats(args.profile_sort)
        if args.profile_dump:
            prof.dump_stats(args.profile_dump)
        st.print_stats(args.profile_top)
        print(s.getvalue(), file=sys.stderr)
        ntok = 3 if args.profile_prefill else max(1, args.repeat * args.tokens)
        for want in args.callees:  # compact callee table, microseconds per generated token
            for key, val in st.stats.items():  # type: ignore[attr-defined]
                if key[2] != want:
                    continue
                print(f"== {Path(key[0]).name}:{key[1]}({key[2]}) cum {1e3 * val[3] / ntok:.3f} ms/token",
                      file=sys.stderr)
                rows_c = []
                for ckey, cval in st.stats.items():  # type: ignore[attr-defined]
                    if key in cval[4]:
                        nc, _p, tt, ct = cval[4][key]
                        rows_c.append((ct, nc, tt, f"{Path(ckey[0]).name}:{ckey[1]}({ckey[2]})"))
                for ct, nc, tt, name in sorted(rows_c, reverse=True)[:18]:
                    print(f"   {1e3 * ct / ntok:8.3f} cum {1e3 * tt / ntok:8.3f} own  {nc / ntok:6.1f}/tok  {name}",
                          file=sys.stderr)
    model.close()
    store.close()
    if not args.keep_store:
        import shutil

        shutil.rmtree(store_dir, ignore_errors=True)
    return result


def _install_timers(backend: Any, model: Any) -> dict[str, list[float]]:
    """Wrap the gateway's per-token stages with wall + calling-thread CPU timers (``--breakdown``)."""
    from pgw.engine import sampling

    timers: dict[str, list[float]] = {}

    def wrap(obj: Any, name: str, label: str) -> None:
        fn = getattr(obj, name)

        def timed(*a: Any, **k: Any) -> Any:
            t, c = time.perf_counter(), time.thread_time()
            try:
                return fn(*a, **k)
            finally:
                e = timers.setdefault(label, [0.0, 0.0, 0])
                e[0] += time.perf_counter() - t
                e[1] += time.thread_time() - c
                e[2] += 1

        setattr(obj, name, timed)

    wrap(backend, "_prepare", "trip_prepare")
    wrap(backend, "_finish", "trip_finish")
    wrap(backend, "forward_local", "local_linears")
    wrap(backend.client, "compute_sync", "stub_service")
    wrap(model, "_logits", "lm_head")
    wrap(model, "forward_batch", "forward_total")
    wrap(sampling.Sampler, "sample", "sample")
    return timers


def _time_head(model: Any, logprobs: int, reps: int = 7) -> float:
    """Median ms of the LM head (one row) + sampling, on this model's head."""
    from pgw.engine.lm_head import lm_head_logits
    from pgw.engine.sampling import Sampler, SamplingParams

    head = model._head
    h = torch.randn(1, head.shape[1] if hasattr(head, "shape") else model.arch.hidden_size)
    sampler = Sampler(SamplingParams(temperature=0.0, logprobs=logprobs, max_tokens=1))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        logits = lm_head_logits(h, head)
        sampler.sample(logits[0], [])
        ts.append(time.perf_counter() - t)
    return 1e3 * sorted(ts)[len(ts) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="0.5b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--prompt-len", type=int, default=8, help="prompt tokens (prefill_ms: prompt pass + 1 token)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--local", default="attention", help="local_linears mode (off, attention, attention_down, all)")
    ap.add_argument("--float-threads", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--logprobs", type=int, default=0)
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--service-delay-ms", type=float, default=0.0, help="stub round-trip latency (sleep, GIL free)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-prefill", action="store_true", help="with --profile: profile the prefill runs instead")
    ap.add_argument("--profile-dump", default=None, help="with --profile: write the pstats file here")
    ap.add_argument("--breakdown", action="store_true", help="per-stage wall and thread CPU per token")
    ap.add_argument("--profile-top", type=int, default=30)
    ap.add_argument("--profile-sort", default="tottime")
    ap.add_argument("--callees", nargs="*", default=[], help="with --profile: print the callees of these functions")
    ap.add_argument("--keep-store", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = asyncio.run(run(args))
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
