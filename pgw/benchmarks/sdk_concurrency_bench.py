"""Client library (``pgw.OpenAI``) under many concurrent streams, one or several client processes.

Starts the provider of the SDK tests (TLS, TCP+TLS trip lane, masked LM head; CPU kernels) on loopback, then, for
each concurrency level ``N``, runs ``N`` streamed chats at once through the library (``--procs P``: ``P`` client
processes with the same key and cache directory, gunicorn-style, ``N / P`` threads each). The mask stock is filled
before each level (the factory is then stopped from adding more by its target) so the run measures the gateway, the
lane and the provider, not mask production; ``--live-factory`` keeps the factory running and starts from an empty
stock instead (measures mask production under load).

Every stream's tokens and logprobs are compared with the same request run alone (bit-identical under concurrency).
Per level: aggregate decode tokens/s, per-stream tokens/s, TTFT p50 / p95, rows per provider trip (cross-session
batching), CPU seconds per token of the client process by thread (event loop, backend worker, native helpers, ...),
of the factory and of the provider, and the mask slots consumed per generated token.

    PYTHONPATH=src:. python benchmarks/sdk_concurrency_bench.py --model tiny --streams 1,4,16,64 --tokens 32

The provider computes on the same CPU here: compare levels and components, not absolute GPU numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
MODEL = "qwen2.5-0.5b-instruct"
PROMPTS = ["Write one sentence about the sea.", "Name three colours.", "What is two plus two?",
           "Describe a cat in a few words."]


def _pack(kind: str, out: Path) -> Path:
    from pgw.modelpack import pack_checkpoint

    if kind == "tiny":
        from tests.conftest import _save_tiny

        (out.parent / "hf").mkdir(parents=True, exist_ok=True)
        src: Any = _save_tiny(out.parent / "hf", "qwen2", seed=0)
    else:
        from huggingface_hub import snapshot_download

        src = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    pack_checkpoint(src, out, name=MODEL)
    return out


def _thread_cpu() -> dict[str, float]:
    """CPU seconds of this process by thread name (``/proc/self/task``; Linux)."""
    tick = os.sysconf("SC_CLK_TCK")
    names = {t.native_id: t.name for t in threading.enumerate()}
    out: dict[str, float] = {}
    for t in Path("/proc/self/task").iterdir():
        try:
            name = names.get(int(t.name)) or "native:" + (t / "comm").read_text().strip()
            f = (t / "stat").read_text().rsplit(")", 1)[1].split()
            cpu = (int(f[11]) + int(f[12])) / tick
        except (OSError, IndexError, ValueError):
            continue
        key = name.split(" (")[0].rstrip("0123456789_-") or name
        out[key] = out.get(key, 0.0) + cpu
    return out


def _proc_cpu(pid: int) -> float:
    """CPU seconds of process ``pid`` and its reaped children."""
    tick = os.sysconf("SC_CLK_TCK")
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return sum(int(x) for x in f[11:15]) / tick
    except (OSError, ValueError):
        return 0.0


def _tree_cpu(pid: int) -> float:
    """CPU seconds of ``pid`` and its live descendants."""
    total = _proc_cpu(pid)
    try:
        kids = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        kids = []
    return total + sum(_tree_cpu(int(k)) for k in kids)


def _request(i: int, tokens: int) -> dict[str, Any]:
    j = i % len(PROMPTS)
    return {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[j]}], "stream": True,
            "temperature": 0.8, "seed": 1000 + j, "max_tokens": tokens, "logprobs": True, "user": f"user-{i}",
            "extra_body": {"ignore_eos": True}}


def _one(c: Any, i: int, tokens: int) -> dict[str, Any]:
    t0 = time.perf_counter()
    times: list[float] = []
    lp: list[list[Any]] = []
    for ch in c.chat.completions.create(**_request(i, tokens)):
        if ch.choices and ch.choices[0].logprobs:
            times.append(time.perf_counter())
            lp += [[x.token, x.logprob] for x in ch.choices[0].logprobs.content]
    return {"i": i, "t0": t0, "first": times[0], "last": times[-1], "n": len(lp), "lp": lp}


def _client(args: argparse.Namespace, wid: int) -> None:
    """One client process: prints a JSON line per level (``--worker``)."""
    from pgw import OpenAI
    from pgw.sdk import LocalPackProvisioner

    root = Path(args.root)
    kw: dict[str, Any] = {}
    if args.store_slots:
        # refill whenever the stock is below its target (each level starts from a full stock)
        kw.update(mask_target_slots=args.store_slots, mask_low_water_slots=max(1, args.store_slots - 1))
    c = OpenAI(api_key=os.environ["PGW_API_KEY"], cache_dir=root / "c", state_dir=root / "s",
               mask_scheme=args.scheme, mask_workers=args.mask_workers, mask_segment_slots=args.segment,
               max_concurrent=args.max_concurrent, mask_wait_s=600.0, timeout_s=3600.0, mask_admission=args.admission,
               provisioner=LocalPackProvisioner({MODEL: Path(args.pack)}), **kw)
    c.warmup(MODEL)
    eng = c._engine
    sync = Path(args.sync)

    def stock() -> int:
        return int(c.status()["models"][MODEL].get("masks_available") or 0)

    def barrier(tag: str) -> None:  # file barrier across the client processes
        (sync / f"{tag}.{wid}").touch()
        while len(list(sync.glob(f"{tag}.*"))) < args.procs:
            time.sleep(0.05)

    refs: dict[int, list[Any]] = {}
    if wid == 0:  # references: each distinct request alone
        for j in range(len(PROMPTS)):
            refs[j] = _one(c, j, args.tokens)["lp"]
        (sync / "refs.json").write_text(json.dumps(refs))
    barrier("refs")
    refs = {int(k): v for k, v in json.loads((sync / "refs.json").read_text()).items()}
    for n in [int(x) for x in args.streams.split(",")]:
        mine = [i for i in range(n) if i % args.procs == wid]
        if not args.live_factory:  # a full stock: measure the gateway, not mask production
            need = args.fill_slots(n) if callable(getattr(args, "fill_slots", None)) else 0
            t_w = time.monotonic()
            while stock() < min(args.store_slots or 0, need) and time.monotonic() - t_w < args.fill_timeout:
                time.sleep(0.5)
        barrier(f"go{n}")
        h = eng.models[MODEL]
        be = h.runtime.backend
        s0 = be.stats.snapshot()
        ls = getattr(h.runtime.client, "lane_stats", None)
        l0 = (ls.trips, ls.fallbacks) if ls is not None else (0, 0)
        gen = getattr(h.runtime, "generator", None)
        sc0 = dict(getattr(gen, "sched_stats", {}) or {})
        nd = getattr(h.runtime.model, "_native_decode", None) if getattr(h.runtime, "model", None) else None
        nd0 = list(nd.totals) if nd is not None else None
        from pgw.maskstore import store as _mstore

        lc0 = dict(getattr(_mstore, "LEASE_COUNTS", {}))
        cpu0, wall0, epoch0 = _thread_cpu(), time.perf_counter(), time.time()
        res: list[dict[str, Any]] = []
        errs: list[str] = []

        def run(i: int, res: list[dict[str, Any]] = res, errs: list[str] = errs) -> None:
            try:
                res.append(_one(c, i, args.tokens))
            except Exception as exc:  # reported
                errs.append(f"{type(exc).__name__}: {exc}")

        ts = [threading.Thread(target=run, args=(i,)) for i in mine]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        wall = time.perf_counter() - wall0
        cpu1, s1 = _thread_cpu(), be.stats.snapshot()
        same = all(r["lp"] == refs[r["i"] % len(PROMPTS)] for r in res)
        sc1 = dict(getattr(gen, "sched_stats", {}) or {})
        nd = getattr(h.runtime.model, "_native_decode", None) if getattr(h.runtime, "model", None) else None
        nd1 = list(nd.totals) if nd is not None else None
        print(json.dumps({
            "worker": wid, "n": n, "streams": len(mine), "errors": errs[:3], "n_errors": len(errs),
            "bit_identical": same, "wall_s": wall, "epoch0": epoch0, "epoch1": time.time(),
            "tokens": sum(r["n"] for r in res),
            "ttft": [r["first"] - r["t0"] for r in res],
            "stream_tps": [(r["n"] - 1) / max(1e-9, r["last"] - r["first"]) for r in res],
            "first_token_at": [r["first"] for r in res], "last_token_at": [r["last"] for r in res],
            "cpu": {k: round(cpu1.get(k, 0.0) - cpu0.get(k, 0.0), 3) for k in cpu1
                    if cpu1.get(k, 0.0) - cpu0.get(k, 0.0) > 0.01},
            "round_trips": s1["round_trips"] - s0["round_trips"], "rows": s1["rows"] - s0["rows"],
            "decode_calls": s1["decode_calls"] - s0["decode_calls"],
            "lane_trips": (ls.trips - l0[0]) if ls is not None else 0,
            "lane_fallbacks": (ls.fallbacks - l0[1]) if ls is not None else 0,
            "lane_fallback_reasons": dict(ls.fallback_reasons) if ls is not None else {},
            # the generator's steps and the native decode driver's calls and trips (one or several answers)
            "leases": {k: v - lc0.get(k, 0) for k, v in getattr(_mstore, "LEASE_COUNTS", {}).items()},
            "sched": {k: sc1.get(k, 0) - sc0.get(k, 0) for k in sc1 if sc1.get(k, 0) != sc0.get(k, 0)},
            "native_decode": ([b - a for a, b in zip(nd0 or [0] * len(nd1), nd1, strict=True)]
                              if nd1 is not None else None),
        }), flush=True)
        barrier(f"done{n}")
    c.close()


def _summ(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["tiny", "qwen0.5b"], default="tiny")
    ap.add_argument("--streams", default="1,4,16,64")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--max-concurrent", type=int, default=64)
    ap.add_argument("--scheme", default="plain", choices=["plain", "pks-v3"])
    ap.add_argument("--mask-workers", type=int, default=1)
    ap.add_argument("--segment", type=int, default=256)
    ap.add_argument("--store-slots", type=int, default=0, help="factory target (0: from --tokens and N)")
    ap.add_argument("--live-factory", action="store_true")
    ap.add_argument("--admission", default="entitled", choices=["entitled", "overcommit"])
    ap.add_argument("--fill-timeout", type=float, default=900.0)
    ap.add_argument("--client-env", action="append", default=[], help="client process env KEY=VALUE (repeatable)")
    ap.add_argument("--provider-env", action="append", default=[], help="provider env KEY=VALUE (repeatable)")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pack", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--sync", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    nmax = max(int(x) for x in args.streams.split(","))
    if not args.store_slots:  # prompt (~30 rows) + tokens + two chunks + the ledger's spare chunk, per stream
        args.store_slots = max(512, int(1.3 * nmax * (args.tokens + 64)))
    args.fill_slots = lambda n: int(1.1 * n * (args.tokens + 64))
    if args.worker is not None:
        _client(args, args.worker)
        return
    from tests.sdk.conftest import make_pki, start_provider, stop_provider

    with tempfile.TemporaryDirectory(prefix="pgw-sdk-conc-") as d:
        root = Path(d)
        for sub in ("pki", "provider", "sync"):
            (root / sub).mkdir()
        pack = _pack(args.model, root / "pack" / "pack")
        prov = start_provider(root / "provider", pack, make_pki(root / "pki"),
                              extra_env=dict(kv.split("=", 1) for kv in args.provider_env))
        out: dict[str, Any] = {"model": args.model, "tokens": args.tokens, "cpus": os.cpu_count(), "procs": args.procs,
                               "scheme": args.scheme, "store_slots": args.store_slots, "levels": {}}
        try:
            env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
            env["PGW_API_KEY"] = prov.new_key()
            env.update(dict(kv.split("=", 1) for kv in args.client_env))
            env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
            cmd = [sys.executable, __file__, "--root", str(root), "--pack", str(pack), "--sync", str(root / "sync"),
                   "--streams", args.streams, "--tokens", str(args.tokens), "--procs", str(args.procs),
                   "--max-concurrent", str(args.max_concurrent), "--scheme", args.scheme,
                   "--mask-workers", str(args.mask_workers), "--segment", str(args.segment),
                   "--store-slots", str(args.store_slots), "--fill-timeout", str(args.fill_timeout),
                   "--admission", args.admission]
            if args.live_factory:
                cmd.append("--live-factory")
            procs = [subprocess.Popen([*cmd, "--worker", str(w)], env=env, stdout=subprocess.PIPE,  # noqa: S603
                                      stderr=subprocess.PIPE, text=True) for w in range(args.procs)]
            lines: dict[int, list[dict[str, Any]]] = {}
            samples: list[tuple[float, float, float]] = []  # (time, provider CPU, client children CPU)
            # sample the provider's and the clients' process trees while levels run
            stop = threading.Event()

            def sampler() -> None:
                while not stop.is_set():
                    samples.append((time.time(), _tree_cpu(prov.proc.pid),
                                    sum(_tree_cpu(p.pid) - _proc_cpu(p.pid) for p in procs)))
                    stop.wait(0.2)

            th = threading.Thread(target=sampler, daemon=True)
            th.start()
            outs = []
            for p in procs:
                o, e = p.communicate(timeout=7200)
                if p.returncode != 0:
                    raise SystemExit(f"client failed:\n{e[-4000:]}")
                outs.append(o)
            stop.set()
            for o in outs:
                for ln in o.splitlines():
                    if ln.startswith("{"):
                        r = json.loads(ln)
                        lines.setdefault(r["n"], []).append(r)
            for n, rs in sorted(lines.items()):
                e0, e1 = min(r["epoch0"] for r in rs), max(r["epoch1"] for r in rs)
                wall = e1 - e0
                win = [s for s in samples if e0 <= s[0] <= e1]
                pcpu = (win[-1][1] - win[0][1]) if len(win) > 1 else 0.0
                fcpu = (win[-1][2] - win[0][2]) if len(win) > 1 else 0.0
                toks = sum(r["tokens"] for r in rs)
                ttft = [x for r in rs for x in r["ttft"]]
                tps = [x for r in rs for x in r["stream_tps"]]
                cpu: dict[str, float] = {}
                for r in rs:
                    for k, v in r["cpu"].items():
                        cpu[k] = round(cpu.get(k, 0.0) + v, 3)
                rt, rows = sum(r["round_trips"] for r in rs), sum(r["rows"] for r in rs)
                lvl = {"streams": n, "errors": sum(r["n_errors"] for r in rs),
                       "error_samples": [e for r in rs for e in r["errors"]][:3],
                       "bit_identical": all(r["bit_identical"] for r in rs), "wall_s": round(wall, 2),
                       "tokens": toks, "agg_tok_s": round(toks / wall, 2),
                       "stream_tok_s_p50": round(statistics.median(tps), 2) if tps else None,
                       "ttft_p50_s": round(_summ(ttft, 0.5), 3), "ttft_p95_s": round(_summ(ttft, 0.95), 3),
                       "rows_per_trip": round(rows / max(1, rt), 2), "round_trips": rt,
                       "trips_per_s": round(rt / wall, 1),
                       "client_cpu_ms_per_tok_total": round(1e3 * sum(cpu.values()) / max(1, toks), 3),
                       "client_cpu_s_total": round(sum(cpu.values()), 3),
                       "lane_trips": sum(r.get("lane_trips", 0) for r in rs),
                       "lane_fallbacks": sum(r.get("lane_fallbacks", 0) for r in rs),
                       "lane_fallback_reasons_cumulative": [r.get("lane_fallback_reasons") for r in rs][:2],
                       "client_cpu_s_per_tok": {k: round(v / max(1, toks), 5) for k, v in
                                                sorted(cpu.items(), key=lambda kv: -kv[1])[:8]},
                       "client_cpu_util": {k: round(v / wall, 2) for k, v in
                                           sorted(cpu.items(), key=lambda kv: -kv[1])[:8]}}
                sched: dict[str, float] = {}
                for r in rs:
                    for k, v in (r.get("sched") or {}).items():
                        sched[k] = sched.get(k, 0) + v
                lvl["sched"] = sched
                lvl["leases"] = {k: sum((r.get("leases") or {}).get(k, 0) for r in rs)
                                 for k in rs[0].get("leases", {})}
                nds = [r["native_decode"] for r in rs if r.get("native_decode")]
                if nds:  # [driver calls, trips, native local ns, lane ns, run CPU s]
                    lvl["native_decode"] = [round(sum(x), 3) for x in zip(*nds, strict=True)]
                lvl["provider_cpu_util"] = round(pcpu / wall, 2)
                lvl["provider_cpu_s_per_tok"] = round(pcpu / max(1, toks), 5)
                lvl["factory_cpu_util"] = round(fcpu / wall, 2)
                out["levels"][n] = lvl
                print(json.dumps(lvl), file=sys.stderr, flush=True)
        finally:
            stop_provider(prov)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
