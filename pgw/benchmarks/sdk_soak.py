"""Soak test of the client library: many sequential and concurrent requests for a long time, with faults.

Starts a real ``pgw service`` on loopback (the test provider of ``tests/sdk``: TLS, TCP+TLS lane, accounts) behind a
fault-injection proxy and one client (``from pgw import OpenAI``) with its background mask factory, then loops:
sequential requests, streamed requests, concurrent requests from threads, abandoned streams, cancelled async
streams, client churn (an ``AsyncOpenAI`` per round), and every ``--fault-every-s`` a dropped connection. Every
greedy answer must equal the first one (bit-identical). Every ``--sample-s`` it records the client process's RSS,
open file descriptors and threads, the same for its child processes (factory, workers) and the provider, the
mask-store disk use, request counts and latencies (JSON lines in ``--out``).

    PYTHONPATH=src:. python benchmarks/sdk_soak.py --minutes 32 --out /tmp/soak.jsonl

The tiny random model by default (``--model 0.5b`` packs Qwen2.5-0.5B-Instruct from the local Hugging Face cache).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

PROMPT = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hello there, who are you?"}]
GREEDY = {"temperature": 0, "max_tokens": 16, "logprobs": True, "extra_body": {"ignore_eos": True}}


def _proc_stats(pid: int) -> dict[str, Any] | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        rss = int(next(ln.split()[1] for ln in status.splitlines() if ln.startswith("VmRSS:")))
        threads = int(next(ln.split()[1] for ln in status.splitlines() if ln.startswith("Threads:")))
        fds = len(os.listdir(f"/proc/{pid}/fd"))
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:80]
    except (OSError, StopIteration, ValueError):
        return None
    return {"pid": pid, "rss_mb": round(rss / 1024, 1), "fds": fds, "threads": threads, "cmd": cmd}


def _descendants(pid: int) -> list[int]:
    out: list[int] = []
    todo = [pid]
    while todo:
        p = todo.pop()
        kids: list[str] = []
        try:  # children are listed per thread (the one that forked them)
            for t in Path(f"/proc/{p}/task").iterdir():
                with contextlib.suppress(OSError):
                    kids += (t / "children").read_text().split()
        except OSError:
            continue
        for k in kids:
            out.append(int(k))
            todo.append(int(k))
    return out


def _store_breakdown(root: Path) -> dict[str, float]:
    """MB per kind of file under the session directory (segments, state database, logs, other)."""
    out: dict[str, float] = {}
    for p in root.rglob("*"):
        with contextlib.suppress(OSError):
            if not p.is_file():
                continue
            kind = ("segments" if p.suffix == ".pgwm" else "state_db" if "sqlite" in p.name else
                    "logs" if p.suffix == ".log" else "other")
            out[kind] = out.get(kind, 0.0) + p.stat().st_size / 2**20
    return {k: round(v, 2) for k, v in out.items()}


def _du(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        with contextlib.suppress(OSError):
            if p.is_file():
                total += p.stat().st_size
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=32.0)
    ap.add_argument("--sample-s", type=float, default=60.0)
    ap.add_argument("--fault-every-s", type=float, default=300.0)
    ap.add_argument("--model", choices=["tiny", "0.5b"], default="tiny")
    ap.add_argument("--scheme", default="pks-v3")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("soak.jsonl"))
    ap.add_argument("--work", type=Path, default=None)
    args = ap.parse_args()

    from tests.conftest import _save_tiny
    from tests.sdk.chaos import ChaosProxy
    from tests.sdk.conftest import MODEL, free_port, make_pki, start_provider, stop_provider

    work = args.work or Path(tempfile.mkdtemp(prefix="pgw-soak-"))
    work.mkdir(parents=True, exist_ok=True)
    (work / "pki").mkdir(exist_ok=True)
    pki = make_pki(work / "pki")
    from pgw.modelpack import pack_checkpoint

    if args.model == "tiny":
        src: Any = _save_tiny(work / "hf", "qwen2", 0)
    else:
        from huggingface_hub import snapshot_download

        src = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    pack = work / "pack"
    if not pack.exists():
        pack_checkpoint(src, pack, name=MODEL)
    (work / "prov").mkdir(exist_ok=True)
    port, lane = free_port(), free_port()
    px, pl = ChaosProxy(port), ChaosProxy(lane)
    prov = start_provider(work / "prov", pack, pki, port=port, lane_port=lane, public_port=px.port,
                          public_lane_port=pl.port)
    key = prov.new_key("soak")

    from pgw import AsyncOpenAI, OpenAI
    from pgw.sdk import LocalPackProvisioner

    opts: dict[str, Any] = {"cache_dir": work / "c" / "cache", "state_dir": work / "c" / "state",
                            "mask_scheme": args.scheme, "mask_workers": 1, "mask_target_slots": 1200,
                            "mask_low_water_slots": 600, "mask_segment_slots": 256, "timeout": 120,
                            "ready_timeout_s": 900, "masked_head": "off" if args.scheme == "pks-v3" else "auto"}
    prov_fn = LocalPackProvisioner({MODEL: pack})
    c = OpenAI(api_key=key, provisioner=prov_fn, **opts)
    t0 = time.monotonic()
    want = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY).choices[0].message.content
    first_s = time.monotonic() - t0
    out = args.out.open("w")
    counts: Counter[str] = Counter()
    lat: list[float] = []
    mu = threading.Lock()
    mismatches = 0

    def note(kind: str, dt: float | None = None) -> None:
        with mu:
            counts[kind] += 1
            if dt is not None:
                lat.append(dt)

    def one(i: int) -> None:
        nonlocal mismatches
        t = time.monotonic()
        try:
            if i % 3 == 0:
                text = "".join(ch.choices[0].delta.content or "" for ch in c.chat.completions.create(
                    model=MODEL, messages=PROMPT, stream=True, **GREEDY) if ch.choices)
            else:
                text = c.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY).choices[0].message.content
            if text != want:
                with mu:
                    mismatches += 1
            note("ok", time.monotonic() - t)
        except Exception as exc:
            note(f"err:{type(exc).__name__}:{getattr(exc, 'code', '')}")

    def sampled() -> None:
        t = time.monotonic()
        try:
            c.chat.completions.create(model=MODEL, messages=PROMPT, temperature=0.9, top_p=0.95,
                                      seed=random.randrange(1 << 30), max_tokens=random.randint(4, 40))  # noqa: S311
            note("ok", time.monotonic() - t)
        except Exception as exc:
            note(f"err:{type(exc).__name__}:{getattr(exc, 'code', '')}")

    def abandoned() -> None:
        try:
            s = c.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, max_tokens=150, temperature=0)
            for k, _ in enumerate(s):
                if k == 2:
                    break
            del s
            note("abandoned")
        except Exception as exc:
            note(f"err:{type(exc).__name__}:{getattr(exc, 'code', '')}")

    async def async_round() -> None:
        async with AsyncOpenAI(api_key=key, provisioner=prov_fn, **opts) as ac:
            r = await ac.chat.completions.create(model=MODEL, messages=PROMPT, **GREEDY)
            if r.choices[0].message.content != want:
                nonlocal mismatches
                mismatches += 1
            s = await ac.chat.completions.create(model=MODEL, messages=PROMPT, stream=True, max_tokens=150,
                                                 temperature=0)

            async def consume() -> None:
                async for _ in s:
                    pass

            task = asyncio.create_task(consume())
            await asyncio.sleep(0.05)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            note("ok")
            note("cancelled")

    me = os.getpid()
    t_start = time.monotonic()
    next_sample = t_start
    next_fault = t_start + args.fault_every_s
    end = t_start + args.minutes * 60
    rnd = 0
    ex = ThreadPoolExecutor(args.concurrency)
    try:
        while time.monotonic() < end:
            rnd += 1
            for i in range(4):
                one(i)
            sampled()
            list(ex.map(one, range(args.concurrency)))
            abandoned()
            try:
                asyncio.run(async_round())
            except Exception as exc:
                note(f"err:{type(exc).__name__}:{getattr(exc, 'code', '')}")
            if time.monotonic() >= next_fault:
                next_fault += args.fault_every_s
                px.drop()
                pl.drop()
                note("fault:drop")
            if time.monotonic() >= next_sample:
                next_sample += args.sample_s
                gc.collect()
                st = c.status()["models"][MODEL]
                kids = [s for s in (_proc_stats(p) for p in _descendants(me)) if s]
                with mu:
                    rec = {
                        "t_s": round(time.monotonic() - t_start, 1), "round": rnd, "counts": dict(counts),
                        "mismatches": mismatches,
                        "p50_ms": round(1e3 * statistics.median(lat), 1) if lat else None,
                        "p99_ms": round(1e3 * sorted(lat)[int(0.99 * (len(lat) - 1))], 1) if lat else None,
                        "client": _proc_stats(me), "py_threads": threading.active_count(),
                        "children": kids, "provider": _proc_stats(prov.proc.pid),
                        "masks_available": st.get("masks_available"),
                        "mask_store_mb": round(_du(work / "c" / "cache" / "sessions") / 2**20, 2),
                        "store_breakdown_mb": _store_breakdown(work / "c" / "cache" / "sessions"),
                        "factory": {k: st.get("factory", {}).get(k) for k in ("running", "starts",
                                                                                "consecutive_failures")},
                        "lane": st.get("lane"),
                    }
                    lat.clear()
                out.write(json.dumps(rec) + "\n")
                out.flush()
                cl = rec["client"] or {}
                print(f"t={rec['t_s']:7.1f}s round={rnd} ok={counts['ok']} errs="
                      f"{sum(v for k, v in counts.items() if k.startswith('err'))} rss={cl.get('rss_mb')}MB "
                      f"fds={cl.get('fds')} threads={cl.get('threads')} kids={len(kids)} "
                      f"stock={rec['masks_available']} p50={rec['p50_ms']}ms", flush=True)
    finally:
        ex.shutdown()
        t1 = time.monotonic()
        c.close()
        close_s = time.monotonic() - t1
        time.sleep(3)
        left = [s for s in (_proc_stats(p) for p in _descendants(me)) if s and "provider_main" not in s["cmd"]]
        summary = {"summary": True, "first_request_s": round(first_s, 1), "rounds": rnd, "counts": dict(counts),
                   "mismatches": mismatches, "close_s": round(close_s, 2), "children_after_close": left,
                   "client_after_close": _proc_stats(me), "py_threads_after_close": threading.active_count()}
        out.write(json.dumps(summary) + "\n")
        out.close()
        print(json.dumps(summary), flush=True)
        stop_provider(prov)
        px.close()
        pl.close()
        if args.work is None:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
