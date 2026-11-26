#!/usr/bin/env python3
"""The client library against a REMOTE provider (a RunPod GPU pod), measured from a client machine in the same data
center: first-use setup, TTFT, decode tokens/s, round trips per token, client CPU, with strict speculation off /
lookup / draft_head, then N concurrent streams.

    export PGW_API_KEY=...            # the test SDK key (never on the command line)
    python deploy/runpod/sdk_remote_bench.py --model qwen2.5-7b-instruct --spec off,lookup,draft_head \\
        --draft-head extras/draft_head_7b.pt --conc 1,4,16 --out results/sdk-7b

What it does, in one process (the native switches are process-wide), one cache directory for every stage:

1. ``setup``: ``OpenAI(...).warmup(model)`` (enroll, provision: gateway pack + factory pack, fingerprint-checked) and
   the first answer (waits for the local pks-v3 hint build and the first masks): seconds and the client's CPU.
2. ``spec_<mode>`` for each ``--spec`` mode, a fresh client with ``speculation=<mode>``: ``--runs`` warm, streamed,
   greedy chats per prompt kind (``chat``: open chat; ``edit``: a copy-heavy code edit, what lookup is for), first
   run discarded: TTFT, decode tokens/s, trips per decoded token (the runtime's counters; prompt trips subtracted),
   tokens per pass, the provider's ``compute_us`` per trip, lane fixed cost seen from here, client CPU per token
   (this process, the mask factory children, and the whole box). Every mode must give the same tokens and
   log-probabilities as ``off`` (strict speculation is exact): checked and recorded.
3. ``conc_<N>`` for each ``--conc``: N threads, one streamed greedy request each (four prompts), against a stock
   filled first: aggregate tokens/s, per-stream tokens/s, TTFT p50/p95, errors, bit-identity with each prompt alone.

Results: ``<out>/results.json`` (updated after every stage; a stopped run keeps what it measured) and
``<out>/status.jsonl`` (the client's status every 30 s). The key is read from ``PGW_API_KEY`` and never printed.
``--fingerprint`` pins a model the library does not pin itself (e.g. 32B before its pin ships): a measurement-only
override that the caller states explicitly; pinned models ignore it.

CPU smoke test (a local CPU provider with the tiny test model, plain masks; numbers meaningless)::

    PYTHONPATH=src:. python deploy/runpod/sdk_remote_bench.py --smoke --spec off,lookup --conc 1,2 --tokens 8 \\
        --runs 1 --out /tmp/sdkr
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "benchmarks")]

CODE = (Path(ROOT / "benchmarks" / "sdk_trip_profile.py").read_text().split('CODE = """', 1)[1].split('"""', 1)[0]
        if (ROOT / "benchmarks" / "sdk_trip_profile.py").exists() else "def f(x):\n    return x\n")
PROMPTS = {
    "chat": [{"role": "user", "content": "Write a short paragraph about why the sky is blue."}],
    "edit": [{"role": "user", "content": "Rename the variable `total` to `paid_total` in this function. Reply with "
              "the full function only, no explanation.\n\n" + CODE}],
}
CONC_PROMPTS = ["Write one sentence about the sea.", "Name three colours.", "What is two plus two?",
                "Describe a cat in a few words."]
R: dict[str, Any] = {"stages": {}, "errors": []}
OUT = Path(".")


def log(*a: Any) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def save() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "results.json.tmp"
    tmp.write_text(json.dumps(R, indent=1, default=str))
    os.replace(tmp, OUT / "results.json")


def _box_cpu() -> tuple[float, float]:
    """(busy, total) jiffies of the whole machine (``/proc/stat``)."""
    try:
        f = [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        return float(sum(f) - f[3] - f[4]), float(sum(f))
    except (OSError, ValueError, IndexError):
        return 0.0, 1.0


def _children_cpu() -> float:
    from sdk_concurrency_bench import _tree_cpu

    return _tree_cpu(os.getpid()) - _self_cpu()


def _self_cpu() -> float:
    from sdk_concurrency_bench import _proc_cpu

    return _proc_cpu(os.getpid())


def _cpu_snapshot() -> dict[str, float]:
    busy, tot = _box_cpu()
    return {"self": time.process_time(), "children": _children_cpu(), "box_busy": busy, "box_total": tot,
            "wall": time.perf_counter()}


def _cpu_delta(a: dict[str, float], b: dict[str, float], tokens: int) -> dict[str, float]:
    wall = max(1e-9, b["wall"] - a["wall"])
    ncpu = os.cpu_count() or 1
    box = (b["box_busy"] - a["box_busy"]) / max(1.0, b["box_total"] - a["box_total"]) * ncpu
    return {"client_cpu_ms_per_token": round(1e3 * (b["self"] - a["self"]) / max(1, tokens), 2),
            "factory_cpu_ms_per_token": round(1e3 * (b["children"] - a["children"]) / max(1, tokens), 2),
            "box_cores_busy": round(box, 2), "wall_s": round(wall, 2)}


def make_client(args: argparse.Namespace, **over: Any) -> Any:
    from pgw import OpenAI

    kw: dict[str, Any] = {"cache_dir": Path(args.cache) / "c", "state_dir": Path(args.cache) / "s",
                          "mask_store_gb": args.mask_store_gb, "timeout_s": 3600.0, "mask_wait_s": 1800.0,
                          "max_concurrent": max(8, max(args.conc_list)), "mask_admission": args.admission}
    if args.mask_workers:
        kw["mask_workers"] = args.mask_workers
    if args.fingerprint:
        kw["model_fingerprints"] = {args.model: args.fingerprint}
    if args.smoke:
        from pgw.sdk import LocalPackProvisioner

        kw.update(mask_scheme="plain", mask_workers=1, mask_segment_slots=128,
                  provisioner=LocalPackProvisioner({args.model: Path(args.pack)}))
    kw.update(over)
    return OpenAI(api_key=os.environ["PGW_API_KEY"], **kw)


def _stream(c: Any, model: str, messages: Any, tokens: int, **extra: Any) -> tuple[list[float], list[Any]]:
    times, lp = [], []
    t0 = time.perf_counter()
    for ch in c.chat.completions.create(model=model, messages=messages, stream=True, temperature=0,
                                        max_tokens=tokens, logprobs=True, extra_body={"ignore_eos": True}, **extra):
        if ch.choices and ch.choices[0].logprobs:
            times.append(time.perf_counter())
            lp += [(x.token, x.logprob) for x in ch.choices[0].logprobs.content]
    return [t0, *times], lp


def _stock(c: Any, model: str) -> int:
    with contextlib.suppress(Exception):
        return int(c.status()["models"][model].get("masks_available") or 0)
    return 0


def wait_stock(c: Any, model: str, slots: int, timeout_s: float) -> float:
    t0 = time.monotonic()
    while _stock(c, model) < slots and time.monotonic() - t0 < timeout_s:
        time.sleep(1.0)
    return round(time.monotonic() - t0, 1)


def stage_setup(args: argparse.Namespace) -> None:
    a = _cpu_snapshot()
    c = make_client(args)
    t0 = time.perf_counter()
    c.warmup(args.model)
    t_warm = time.perf_counter() - t0
    times, _ = _stream(c, args.model, PROMPTS["chat"], 8)
    b = _cpu_snapshot()
    st = c.status()["models"].get(args.model, {})
    R["stages"]["setup"] = {"warmup_s": round(t_warm, 1), "first_answer_s": round(times[-1] - t0, 1),
                            "first_ttft_s": round(times[1] - times[0], 2) if len(times) > 1 else None,
                            "cpu": _cpu_delta(a, b, 1), "lane": st.get("lane"), "native": st.get("native"),
                            "masked_head": st.get("masked_head"), "scheme": st.get("mask_scheme")}
    c.close()
    save()


def stage_spec(args: argparse.Namespace, mode: str, refs: dict[str, Any]) -> None:
    from sdk_trip_profile import _counters, _delta

    over: dict[str, Any] = {}
    if mode != "off":
        over.update(speculation=mode, spec_rows=args.spec_rows)
    if mode == "draft_head":
        over["draft_head_path"] = Path(args.draft_head)
    c = make_client(args, **over)
    c.warmup(args.model)
    rows = 1 if mode == "off" else args.spec_rows
    fill = wait_stock(c, args.model, min(int(1.2 * rows * (args.tokens + 300) * 2), args.max_fill_slots),
                      args.fill_timeout)
    rt = c._engine.models[args.model].runtime
    out: dict[str, Any] = {"stock_wait_s": fill, "spec_rows": rows}
    for kind, msgs in PROMPTS.items():
        _stream(c, args.model, msgs, 1)  # prompt-only trips of this prompt
        c._engine.call(rt.generator.wait_drained(), timeout_s=600)
        a0 = _counters(rt)
        _stream(c, args.model, msgs, 1)
        c._engine.call(rt.generator.wait_drained(), timeout_s=600)
        prompt_d = _delta(a0, _counters(rt))
        runs = []
        for i in range(args.runs + 1):
            a = _counters(rt)
            ca = _cpu_snapshot()
            times, lp = _stream(c, args.model, msgs, args.tokens)
            cb = _cpu_snapshot()
            c._engine.call(rt.generator.wait_drained(), timeout_s=600)  # strict speculation's due dummy passes
            d = _delta(a, _counters(rt))
            ref = refs.setdefault(kind, lp)
            if i == 0:
                continue
            n_dec = max(1, len(times) - 2)
            dec = {k: d[k] - prompt_d.get(k, 0.0) for k in d}
            gaps = [t2 - t1 for t1, t2 in itertools.pairwise(times[1:])]
            ch_trips = max(1.0, dec.get("ch_trips", 0.0))
            runs.append({
                "ttft_ms": round(1e3 * (times[1] - times[0]), 1),
                "decode_tok_s": round(n_dec / max(1e-9, times[-1] - times[1]), 2),
                "decode_ms_per_token_p50": round(1e3 * statistics.median(gaps), 2) if gaps else None,
                "trips_per_token": round(dec.get("backend_round_trips", 0.0) / n_dec, 2),
                "rows_per_trip": round(dec.get("rows", 0.0) / max(1.0, dec.get("backend_round_trips", 0.0)), 2),
                "provider_compute_us": round(dec.get("ch_compute_us", 0.0) / ch_trips, 1),
                "lane_trip_us": round(dec.get("ch_trip_us", 0.0) / ch_trips, 1),
                "lane_wait_beyond_compute_us": round((dec.get("ch_wait_us", 0.0) - dec.get("ch_compute_us", 0.0)
                                                      - dec.get("ch_queue_us", 0.0)) / ch_trips, 1),
                "bit_identical_to_off": lp == ref, **_cpu_delta(ca, cb, n_dec + 1)})
        out[kind] = {"runs": runs, "summary": {k: statistics.median(r[k] for r in runs) for k in runs[0]
                                               if isinstance(runs[0][k], int | float) and not isinstance(
                                                   runs[0][k], bool)} if runs else {},
                     "bit_identical_to_off": all(r["bit_identical_to_off"] for r in runs)}
        log(f"spec {mode} {kind}: {out[kind]['summary']}")
    with contextlib.suppress(Exception):
        out["spec_stats"] = rt.status().get("speed", {}).get("spec_stats")
    c.close()
    R["stages"][f"spec_{mode}"] = out
    save()


def stage_conc(args: argparse.Namespace, n: int, refs: dict[int, Any]) -> None:
    c = make_client(args)
    c.warmup(args.model)
    for j in range(len(CONC_PROMPTS)):
        if j not in refs:
            _, refs[j] = _stream(c, args.model, [{"role": "user", "content": CONC_PROMPTS[j]}], args.conc_tokens)
    fill = wait_stock(c, args.model, min(int(1.2 * n * (args.conc_tokens + 64)), args.max_fill_slots),
                      args.fill_timeout)
    be = c._engine.models[args.model].runtime.backend
    s0 = be.stats.snapshot()
    res: list[dict[str, Any]] = []
    errs: list[str] = []

    def one(i: int) -> None:
        j = i % len(CONC_PROMPTS)
        try:
            times, lp = _stream(c, args.model, [{"role": "user", "content": CONC_PROMPTS[j]}], args.conc_tokens,
                                user=f"user-{i}")
            res.append({"ttft": times[1] - times[0], "tps": (len(times) - 2) / max(1e-9, times[-1] - times[1]),
                        "n": len(lp), "same": lp == refs[j]})
        except Exception as exc:  # recorded
            errs.append(f"{type(exc).__name__}: {str(exc)[:200]}")

    ca = _cpu_snapshot()
    ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    cb = _cpu_snapshot()
    s1 = be.stats.snapshot()
    toks = sum(r["n"] for r in res)
    ttft = sorted(r["ttft"] for r in res)
    rt_ = s1["round_trips"] - s0["round_trips"]
    R["stages"][f"conc_{n}"] = {
        "streams": n, "stock_wait_s": fill, "errors": len(errs), "error_samples": errs[:3],
        "bit_identical": all(r["same"] for r in res) and not errs, "tokens": toks,
        "agg_tok_s": round(toks / max(1e-9, cb["wall"] - ca["wall"]), 2),
        "stream_tok_s_p50": round(statistics.median(r["tps"] for r in res), 2) if res else None,
        "ttft_p50_s": round(ttft[len(ttft) // 2], 3) if ttft else None,
        "ttft_p95_s": round(ttft[min(len(ttft) - 1, int(0.95 * len(ttft)))], 3) if ttft else None,
        "rows_per_trip": round((s1["rows"] - s0["rows"]) / max(1, rt_), 2), "round_trips": rt_,
        **_cpu_delta(ca, cb, max(1, toks))}
    log(f"conc {n}: {R['stages'][f'conc_{n}']}")
    c.close()
    save()


def status_sampler(args: argparse.Namespace, stop: threading.Event) -> None:
    from pgw.sdk.cli import status_report

    while not stop.wait(30.0):
        with contextlib.suppress(Exception), (OUT / "status.jsonl").open("a") as f:
            rep = status_report(Path(args.cache) / "c", Path(args.cache) / "s")
            f.write(json.dumps({"t": time.time(), "status": rep}, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen2.5-7b-instruct")
    ap.add_argument("--spec", default="off,lookup", help="comma list of off, lookup, draft_head")
    ap.add_argument("--spec-rows", type=int, default=3)
    ap.add_argument("--draft-head", default=None, help="draft head file (needed for draft_head)")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conc", default="1,4,16")
    ap.add_argument("--conc-tokens", type=int, default=64)
    ap.add_argument("--mask-store-gb", type=float, default=24.0)
    ap.add_argument("--max-fill-slots", type=int, default=3000, help="cap of the stock waited for before a stage")
    ap.add_argument("--fill-timeout", type=float, default=1800.0)
    ap.add_argument("--mask-workers", type=int, default=0, help="0: the library's default")
    ap.add_argument("--admission", default="entitled", choices=["entitled", "overcommit"])
    ap.add_argument("--fingerprint", default=None, help="measurement-only pin for a model the library does not pin")
    ap.add_argument("--cache", default=None, help="client cache root (default <out>/cache)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="setup,spec,conc")
    ap.add_argument("--smoke", action="store_true", help="local CPU provider, tiny model, plain masks")
    ap.add_argument("--pack", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    OUT = Path(args.out)
    args.cache = args.cache or str(OUT / "cache")
    args.conc_list = [int(x) for x in args.conc.split(",") if x]
    modes = [m for m in args.spec.split(",") if m]
    if "draft_head" in modes and not (args.draft_head and Path(args.draft_head).is_file()):
        log("draft_head skipped: no --draft-head file")
        R.setdefault("notes", []).append("draft_head skipped: no draft head file")
        modes.remove("draft_head")
    R.update(model=args.model, host_cpus=os.cpu_count(), args={k: v for k, v in vars(args).items()
                                                                 if k not in ("conc_list",)}, started=time.time())
    stages = set(args.stages.split(","))
    prov = None
    tmp = None
    if args.smoke:
        from sdk_concurrency_bench import _pack
        from tests.sdk.conftest import make_pki, start_provider

        tmp = tempfile.TemporaryDirectory(prefix="pgw-sdkr-")
        root = Path(tmp.name)
        for sub in ("pki", "provider"):
            (root / sub).mkdir()
        args.model = "qwen2.5-0.5b-instruct"
        PROMPTS["edit"] = [{"role": "user", "content": "Rename x to y:\n" + "\n".join(CODE.splitlines()[:3])}]
        args.pack = str(_pack("tiny", root / "pack" / "pack"))
        prov = start_provider(root / "provider", Path(args.pack), make_pki(root / "pki"))
        os.environ["PGW_API_KEY"] = prov.new_key()
    if not os.environ.get("PGW_API_KEY"):
        raise SystemExit("PGW_API_KEY is not set")
    stop = threading.Event()
    threading.Thread(target=status_sampler, args=(args, stop), daemon=True).start()
    try:
        for name, fn in (("setup", lambda: stage_setup(args)),):
            if name in stages:
                log("stage setup")
                fn()
        refs: dict[str, Any] = {}
        if "spec" in stages:
            for m in (["off"] + [x for x in modes if x != "off"]):
                log(f"stage spec {m}")
                try:
                    stage_spec(args, m, refs)
                except Exception as exc:  # recorded; the next stage still runs
                    R["errors"].append(f"spec {m}: {type(exc).__name__}: {str(exc)[:300]}")
                    save()
        crefs: dict[int, Any] = {}
        if "conc" in stages:
            for n in args.conc_list:
                log(f"stage conc {n}")
                try:
                    stage_conc(args, n, crefs)
                except Exception as exc:
                    R["errors"].append(f"conc {n}: {type(exc).__name__}: {str(exc)[:300]}")
                    save()
    finally:
        stop.set()
        R["finished"] = time.time()
        save()
        if prov is not None:
            from tests.sdk.conftest import stop_provider

            stop_provider(prov)
        if tmp is not None:
            tmp.cleanup()
    log(f"results in {OUT / 'results.json'}")
    return 0 if not R["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
