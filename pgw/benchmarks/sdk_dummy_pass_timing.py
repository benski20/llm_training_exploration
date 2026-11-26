"""Strict speculation: can the provider tell the dummy passes from the real ones by their timing? (review S20)

Runs the client library with ``speculation="lookup"`` against a real ``pgw service`` process on loopback (the
provider of the SDK tests: TLS, TCP+TLS trip lane, masked LM head; CPU) and puts a TCP relay in front of the trip
lane. The relay records what the provider's network sees: when each trip's request arrives, its size, when the
reply leaves and its size. Nothing else is used for the provider's features. In the client, the start of every
pass is tagged real / dummy / prefill (``Generator._plan_fixed``); each trip is assigned to the pass that was
planned last before it arrived. Per decode pass, the provider-visible features are:

* ``gap_ms``: from the last reply of the previous pass to the first request of this pass (client work between
  passes: sampling, drafting, emitting tokens; the feature S20 asks about);
* ``intra_ms``: mean client gap between the trips of the pass (layer work);
* ``proc_ms``: mean time from a request to its reply (provider processing as seen on the wire);
* ``dur_ms``: first request to last reply;
* ``bytes``: request and reply bytes of the pass (TLS records).

For each feature it reports the real and dummy distributions, the Kolmogorov-Smirnov distance and the balanced
accuracy of the best single-threshold classifier (0.5 = indistinguishable), and a per-request attack: the
change point of ``gap_ms`` along the request's passes, as an estimate of the number of real passes (the S4 leak),
against a baseline that always guesses the median count.

    PYTHONPATH=src:. python benchmarks/sdk_dummy_pass_timing.py --model tiny --requests 24 --tokens 40
    PYTHONPATH=src:. python benchmarks/sdk_dummy_pass_timing.py --model qwen0.5b --requests 12 --tokens 48
"""

from __future__ import annotations

import argparse
import asyncio
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
MODEL = "qwen2.5-0.5b-instruct"

TEXT = ("The lighthouse keeper climbed the spiral stairs every evening at dusk, lit the great lamp, and wrote the "
        "weather, the ships and the state of the sea into a leather book that nobody ever read.")
PROMPTS = [
    [{"role": "user", "content": f"Repeat this text exactly: {TEXT}"}],
    [{"role": "user", "content": "Write a short poem about the sea."}],
    [{"role": "user", "content": f"Summarise in one line: {TEXT}"}],
    [{"role": "user", "content": "List five fruits, one per line."}],
]


# ---- the provider's view: a TCP relay in front of the trip lane ----------------------------------------------
class Relay:
    """Forwards 127.0.0.1:<port> to the lane and records (time, connection, direction, bytes) of every read."""

    def __init__(self, upstream_port: int):
        self.upstream_port = upstream_port
        self.events: list[tuple[float, int, str, int]] = []
        self._ids = itertools.count()
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self.port = 0
        threading.Thread(target=self._run, name="relay", daemon=True).start()
        self._ready.wait(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def start() -> None:
            srv = await asyncio.start_server(self._handle, "127.0.0.1", 0)
            self.port = srv.sockets[0].getsockname()[1]
            self._ready.set()

        self._loop.run_until_complete(start())
        self._loop.run_forever()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        cid = next(self._ids)
        ur, uw = await asyncio.open_connection("127.0.0.1", self.upstream_port)
        for w in (writer, uw):
            sock = w.get_extra_info("socket")
            if sock is not None:
                import socket

                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        async def pipe(r: asyncio.StreamReader, w: asyncio.StreamWriter, d: str) -> None:
            try:
                while True:
                    data = await r.read(1 << 20)
                    if not data:
                        break
                    self.events.append((time.monotonic(), cid, d, len(data)))
                    w.write(data)
                    await w.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                w.close()

        await asyncio.gather(pipe(reader, uw, "up"), pipe(ur, writer, "down"))


def trips(events: list[tuple[float, int, str, int]], since: float) -> list[dict[str, float]]:
    """Request/reply cycles per connection: a trip starts with client bytes after a reply."""
    out: list[dict[str, float]] = []
    by: dict[int, list[tuple[float, str, int]]] = {}
    for t, cid, d, n in events:
        by.setdefault(cid, []).append((t, d, n))
    for cid, evs in by.items():
        cur: dict[str, float] | None = None
        for t, d, n in evs:
            if d == "up":
                if cur is None or cur["down"] > 0:
                    if cur is not None:
                        out.append(cur)
                    cur = {"t_req": t, "up": 0, "t_resp": 0.0, "t_end": 0.0, "down": 0, "conn": cid}
                cur["up"] += n
            elif cur is not None:
                if not cur["down"]:
                    cur["t_resp"] = t
                cur["down"] += n
                cur["t_end"] = t
        if cur is not None and cur["down"]:
            out.append(cur)
    return sorted((x for x in out if x["t_req"] >= since), key=lambda x: x["t_req"])


# ---- statistics -------------------------------------------------------------------------------------------
def ks(a: list[float], b: list[float]) -> float:
    pts = sorted(set(a) | set(b))
    sa, sb = sorted(a), sorted(b)
    import bisect

    return max(abs(bisect.bisect_right(sa, x) / len(sa) - bisect.bisect_right(sb, x) / len(sb)) for x in pts)


def best_threshold(a: list[float], b: list[float]) -> float:
    """Balanced accuracy of the best rule 'x > t -> class a' or 'x <= t -> class a'."""
    import bisect

    sa, sb = sorted(a), sorted(b)
    best = 0.5
    for t in sorted(set(a) | set(b)):
        fa = bisect.bisect_right(sa, t) / len(sa)  # P(a <= t)
        fb = bisect.bisect_right(sb, t) / len(sb)
        best = max(best, 0.5 * ((1 - fa) + fb), 0.5 * (fa + (1 - fb)))
    return best


def summary(xs: list[float]) -> dict[str, float]:
    q = statistics.quantiles(xs, n=10, method="inclusive") if len(xs) >= 2 else [xs[0]] * 9
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 3), "median": round(statistics.median(xs), 3),
            "p10": round(q[0], 3), "p90": round(q[-1], 3), "stdev": round(statistics.pstdev(xs), 3)}


def change_point(xs: list[float]) -> int:
    """Split index k (1..n-1) minimising the within-segment squared error of xs[:k] and xs[k:]."""
    best, arg = float("inf"), len(xs)
    for k in range(1, len(xs)):
        a, b = xs[:k], xs[k:]
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        sse = sum((x - ma) ** 2 for x in a) + sum((x - mb) ** 2 for x in b)
        if sse < best:
            best, arg = sse, k
    return arg


# ---- the run ----------------------------------------------------------------------------------------------
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


def run(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    from tests.sdk.conftest import free_port, make_pki, start_provider, stop_provider

    for sub in ("pki", "provider"):
        (root / sub).mkdir()
    pack = Path(args.pack) if args.pack else _pack(args.model, root / "pack" / "pack")
    # the lane is advertised on the relay's port: every trip of the client passes through the relay
    lane_port = free_port()
    relay = Relay(lane_port)
    prov = start_provider(root / "provider", pack, make_pki(root / "pki"),
                          extra_env={"PGW_SERVICE_LANE_PORT": str(lane_port),
                                     "PGW_SERVICE_LANE_ADVERTISED_PORT": str(relay.port)})
    try:
        return measure(args, root, prov, pack, relay)
    finally:
        stop_provider(prov)


def measure(args: argparse.Namespace, root: Path, prov: Any, pack: Path, relay: Relay) -> dict[str, Any]:
    from pgw import OpenAI
    from pgw.engine import generate as G
    from pgw.sdk import LocalPackProvisioner

    plans: list[tuple[float, str]] = []
    orig = G.Generator._plan_fixed

    def plan(self: Any, batch: Any) -> None:
        orig(self, batch)
        kind = "prefill" if any(not st.passes for st in batch) else ("dummy" if all(st.dummy for st in batch)
                                                                      else "real")
        plans.append((time.monotonic(), kind))

    G.Generator._plan_fixed = plan  # type: ignore[method-assign]
    http_calls: list[tuple[float, str]] = []
    if args.dump:  # debugging: compute calls that do not use the lane
        from pgw.gateway import runtime as R

        oc, ocs = R._InstrumentedClient.compute, R._InstrumentedClient._compute_sync

        async def comp(self: Any, request: Any) -> Any:
            http_calls.append((time.monotonic(), ",".join(str(getattr(i, "linear", "")) for i in request.items)))
            return await oc(self, request)

        def comp_sync(self: Any, request: Any) -> Any:
            http_calls.append((time.monotonic(), ",".join(str(getattr(i, "linear", "")) for i in request.items)))
            return ocs(self, request)

        R._InstrumentedClient.compute, R._InstrumentedClient._compute_sync = comp, comp_sync
    passes_per_req = args.tokens
    from pgw.sdk.engine import Engine

    if args.experiment == "no_status":  # diagnosis: no status-file write at the end of a request
        Engine._write_status_soon = lambda self: None  # type: ignore[method-assign]
    elif args.experiment == "before_fix":  # the status write right after the answer, as before the S20 fix
        Engine._status_after_passes = lambda self, rt: self._write_status_soon()  # type: ignore[method-assign]
    c = OpenAI(api_key=prov.new_key(), cache_dir=root / "c", state_dir=root / "s", mask_scheme="plain",
               mask_workers=max(1, (os.cpu_count() or 2) // 2), speculation="lookup", spec_rows=args.rows,
               mask_target_slots=(args.rows * passes_per_req + 160) * (args.requests + 2),
               mask_low_water_slots=(args.rows * passes_per_req + 160) * 2, mask_segment_slots=256,
               timeout=900, ready_timeout_s=1800, provisioner=LocalPackProvisioner({MODEL: pack}),
               **({} if args.gap_ms is None else {"spec_pass_gap_ms": args.gap_ms}))
    try:
        c.warmup(MODEL)
        rt = c._engine.models[MODEL].runtime
        body = {"max_tokens": args.tokens, "temperature": 0, "extra_body": {"ignore_eos": True}}
        c.chat.completions.create(model=MODEL, messages=PROMPTS[0], **body)  # warm up (connections, caches)
        c._engine.call(rt.generator.wait_drained(), timeout_s=600)
        need = (args.rows * passes_per_req + 160) * args.requests
        t_wait = time.monotonic()
        while (c.status()["models"][MODEL].get("masks_available") or 0) < need and time.monotonic() - t_wait < 3600:
            time.sleep(2)  # the whole run from the stock: no factory refills in the middle of a pass
        print(f"mask stock {c.status()['models'][MODEL].get('masks_available')} after "
              f"{time.monotonic() - t_wait:.0f}s", file=sys.stderr)
        time.sleep(1.0)
        since = time.monotonic()
        plans.clear()
        reqs: list[dict[str, Any]] = []
        for i in range(args.requests):
            msgs = PROMPTS[i % len(PROMPTS)]
            t0 = time.monotonic()
            if args.stream:
                n = sum(1 for _ in c.chat.completions.create(model=MODEL, messages=msgs, stream=True, **body))
            else:
                n = c.chat.completions.create(model=MODEL, messages=msgs, **body).usage.completion_tokens
            t1 = time.monotonic()
            _wait_idle(rt.generator)  # the dummy passes of this request (polled from this thread, not the loop)
            reqs.append({"t0": t0, "t_answer": t1, "t_drained": time.monotonic(), "tokens": n})
            time.sleep(args.pause)
        spec = dict(rt.generator.spec_stats)
    finally:
        G.Generator._plan_fixed = orig  # type: ignore[method-assign]
        c.close()
    tr = trips(relay.events, since)
    if args.dump:
        Path(args.dump).write_text(json.dumps({"trips": tr, "plans": plans, "reqs": reqs, "http": http_calls},
                                              default=str))
    return analyse(tr, plans, reqs, spec, args)


def _wait_idle(gen: Any, timeout_s: float = 600.0) -> None:
    """Until the generator has no sequence and no task (checked from the calling thread: no wake-ups of the
    client's event loop during the dummy passes, which would perturb what is measured)."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        tasks = [t for t in (gen._task, gen._lane_task) if t is not None and not t.done()]
        if not (gen._active or gen._lane_wait or gen._lane_busy or gen._draining or tasks):
            return
        time.sleep(0.05)
    raise TimeoutError("the generator did not go idle")


def analyse(tr: list[dict[str, float]], plans: list[tuple[float, str]], reqs: list[dict[str, Any]],
            spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import bisect

    starts = [p[0] for p in plans]
    per: list[dict[str, Any]] = [{"kind": k, "t_plan": t, "trips": []} for t, k in plans]
    stray = 0
    for x in tr:
        j = bisect.bisect_right(starts, x["t_req"]) - 1
        if j < 0:
            stray += 1
            continue
        per[j]["trips"].append(x)
    feats: list[dict[str, Any]] = []
    prev_end: float | None = None
    for p in per:
        ts = p["trips"]
        if not ts:
            continue
        f: dict[str, Any] = {"kind": p["kind"], "t": ts[0]["t_req"], "n_trips": len(ts),
                             "up": int(sum(x["up"] for x in ts)), "down": int(sum(x["down"] for x in ts)),
                             "sizes": tuple(int(x["up"]) for x in ts),
                             "dur_ms": (ts[-1]["t_end"] - ts[0]["t_req"]) * 1e3,
                             "proc_ms": statistics.fmean(x["t_resp"] - x["t_req"] for x in ts) * 1e3,
                             "intra_ms": (statistics.fmean(b["t_req"] - a["t_end"] for a, b in itertools.pairwise(ts))
                                          * 1e3 if len(ts) > 1 else 0.0),
                             "gap_ms": (ts[0]["t_req"] - prev_end) * 1e3 if prev_end is not None else None}
        prev_end = ts[-1]["t_end"]
        feats.append(f)
    # requests: passes between t0 and the next request's t0
    for i, r in enumerate(reqs):
        hi = reqs[i + 1]["t0"] if i + 1 < len(reqs) else float("inf")
        r["passes"] = [f for f in feats if r["t0"] <= f["t"] < hi]
        # the first dummy pass of a request (the answer -> dummy transition) is reported on its own
        seen = False
        for f in r["passes"]:
            if f["kind"] == "dummy" and not seen:
                f["first_dummy"] = True
                seen = True
            if f["kind"] == "prefill":
                f["gap_ms"] = None  # the gap before a prefill is the application's pause
        if r["passes"] and r["passes"][0]["kind"] == "prefill" and len(r["passes"]) > 1:
            r["passes"][1]["after_prefill"] = True
    real = [f for f in feats if f["kind"] == "real"]
    dummy = [f for f in feats if f["kind"] == "dummy"]
    out: dict[str, Any] = {"model": args.model, "rows": args.rows, "gap_ms": args.gap_ms,
                           "experiment": args.experiment, "tokens": args.tokens, "requests": len(reqs),
                           "stream": args.stream, "cpus": os.cpu_count(), "stray_trips": stray,
                           "passes": {"real": len(real), "dummy": len(dummy),
                                      "prefill": sum(f["kind"] == "prefill" for f in feats)},
                           "spec_stats": {k: spec.get(k) for k in ("spec_passes", "spec_accepted", "dummy_passes")},
                           "trips_per_pass": sorted({f["n_trips"] for f in real + dummy}),
                           "request_sizes_equal": len({f["sizes"] for f in real + dummy}) == 1,
                           "features": {}}
    for name in ("gap_ms", "intra_ms", "proc_ms", "dur_ms", "up", "down"):
        def vals(fs: list[dict[str, Any]], name: str = name, steady: bool = False) -> list[float]:
            return [float(f[name]) for f in fs if f.get(name) is not None
                    and not (steady and (f.get("first_dummy") or f.get("after_prefill")))]

        a, b = vals(real, steady=True), vals(dummy, steady=True)
        if not a or not b:
            continue
        e: dict[str, Any] = {"real": summary(a), "dummy": summary(b), "ks": round(ks(a, b), 3),
                             "ks_critical_5pct": round(1.36 * ((len(a) + len(b)) / (len(a) * len(b))) ** 0.5, 3),
                             "threshold_accuracy": round(best_threshold(a, b), 3)}
        if name == "gap_ms":
            fd = vals([f for f in dummy if f.get("first_dummy")])
            if fd:
                e["first_dummy"] = summary(fd)
                e["first_dummy_vs_real"] = {"ks": round(ks(fd, a), 3),
                                            "threshold_accuracy": round(best_threshold(fd, a), 3),
                                            "ks_critical_5pct": round(1.36 * ((len(fd) + len(a))
                                                                              / (len(fd) * len(a))) ** 0.5, 3)}
            ap = vals([f for f in real if f.get("after_prefill")])
            if ap:
                e["first_real_after_prefill"] = summary(ap)
        out["features"][name] = e
    # control: the same statistic between two halves of one kind (early vs late passes of each request); time and
    # position effects (load of other processes on the host) show up here as well, not only real vs dummy
    ctrl: dict[str, Any] = {}
    for kind in ("real", "dummy"):
        early: list[float] = []
        late: list[float] = []
        for r in reqs:
            xs = [f["gap_ms"] for f in r["passes"] if f["kind"] == kind and f.get("gap_ms") is not None
                  and not (f.get("first_dummy") or f.get("after_prefill"))]
            early += xs[: len(xs) // 2]
            late += xs[len(xs) // 2:]
        if early and late:
            ctrl[kind] = {"ks": round(ks(early, late), 3), "threshold_accuracy": round(best_threshold(early, late), 3),
                          "ks_critical_5pct": round(1.36 * ((len(early) + len(late)) / (len(early) * len(late)))
                                                    ** 0.5, 3)}
    out["features"]["gap_ms"]["control_early_vs_late"] = ctrl
    per_pass = [f["gap_ms"] + f["dur_ms"] for f in real + dummy if f.get("gap_ms") is not None]
    out["pass_period_ms"] = summary(per_pass) if per_pass else None  # the cost of a gap floor shows here
    # the S4 question: how many real passes did a request have? (truth vs a change point on gap_ms)
    errs, base = [], []
    truth = [sum(f["kind"] == "real" for f in r["passes"]) for r in reqs]
    med = statistics.median(truth) if truth else 0
    for r, t in zip(reqs, truth, strict=True):
        xs = [f["gap_ms"] for f in r["passes"] if f["kind"] in ("real", "dummy") and f["gap_ms"] is not None]
        if len(xs) < 3:
            continue
        errs.append(abs(change_point(xs) - t))
        base.append(abs(med - t))
    # the largest gap of a request as the answer -> dummy boundary
    hits = 0
    n_req = 0
    for r in reqs:
        ps = [f for f in r["passes"] if f["kind"] in ("real", "dummy") and f["gap_ms"] is not None]
        if not any(f.get("first_dummy") for f in ps):
            continue
        n_req += 1
        hits += bool(max(ps, key=lambda f: f["gap_ms"]).get("first_dummy"))
    out["real_pass_count_attack"] = {
        "truth_real_passes": truth,
        "change_point_mean_abs_error": round(statistics.fmean(errs), 2) if errs else None,
        "median_guess_mean_abs_error": round(statistics.fmean(base), 2) if base else None,
        "largest_gap_is_boundary": f"{hits}/{n_req}",
        "chance": round(statistics.fmean(1 / max(1, sum(f["kind"] in ("real", "dummy") for f in r["passes"]) - 1)
                                         for r in reqs), 3) if reqs else None}
    from collections import Counter

    out["trips_per_pass_counts"] = {k: dict(Counter(f["n_trips"] for f in fs))
                                    for k, fs in (("real", real), ("dummy", dummy))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["tiny", "qwen0.5b"], default="tiny")
    ap.add_argument("--pack", default=None, help="An existing pack directory (skips packing)")
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--pause", type=float, default=0.2)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--gap-ms", type=float, default=None, help="spec_pass_gap_ms (default: the library's)")
    ap.add_argument("--out", default=None, help="Write the JSON result here too")
    ap.add_argument("--experiment", default=None, choices=["no_status", "before_fix"], help=argparse.SUPPRESS)
    ap.add_argument("--dump", default=None, help="Write the raw trips and pass starts here (JSON)")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="pgw-s20-") as d:
        res = run(args, Path(d))
    text = json.dumps(res, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
