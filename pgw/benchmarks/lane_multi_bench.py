"""Lane cost of one batched trip of K answers: the design choice between multi-item lane calls and parallel
single-item lane calls (``docs/sdk/scaling.md``, bottleneck 4), measured against the SDK tests' real provider.

Starts the SDK tests' provider (TLS, TCP+TLS trip lane, CPU kernels) and one ``pgw.OpenAI`` client, takes the
client's lane, and for K = 1, 2, 4, 8, 16 answers of one row each on one linear measures one "step" (the K rows of
one batched trip) as:

* ``multi``: one multi-item lane call (lane format 2: K items of 1 row in one frame), when the service offers it;
* ``rows``: one single-item call of K rows (K <= the lane's max rows): the floor a multi call can reach;
* ``parallel``: K single-item calls at once over K lane connections, one Python thread each (the GIL is released
  for each call; the provider serves each connection on its own thread);
* ``serial``: K single-item calls one after the other on one connection;
* ``stream``: the K items in one request on the client's fallback channel (the blocking stream lane), as today.

Per step: wall time (median), client CPU (process CPU time per step, all threads) and provider CPU per step. With
``--busy N``, N extra Python threads spin on the GIL meanwhile (the client's plaintext and helper threads under
load). Bodies are random bytes: the provider computes on them as on masked rows; nothing is checked here.

    PYTHONPATH=src:. python benchmarks/lane_multi_bench.py --model tiny --ks 1,4,16 --steps 200 [--busy 2]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def _proc_cpu(pid: int) -> float:
    tick = os.sysconf("SC_CLK_TCK")
    try:
        f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return sum(int(x) for x in f[11:13]) / tick
    except (OSError, ValueError):
        return 0.0


class _Pool:
    """K persistent threads, each owning one lane channel; ``run`` starts one call on each and waits for all."""

    def __init__(self, chans: list[Any]):
        self.chans = chans
        self.go = [threading.Event() for _ in chans]
        self.done = [threading.Event() for _ in chans]
        self.job: Any = None
        self.stop = False
        self.ths = [threading.Thread(target=self._loop, args=(i,), daemon=True) for i in range(len(chans))]
        for t in self.ths:
            t.start()

    def _loop(self, i: int) -> None:
        while True:
            self.go[i].wait()
            self.go[i].clear()
            if self.stop:
                return
            self.job(i, self.chans[i])
            self.done[i].set()

    def run(self, job: Any) -> None:
        self.job = job
        for e in self.go:
            e.set()
        for e in self.done:
            e.wait()
            e.clear()

    def close(self) -> None:
        self.stop = True
        for e in self.go:
            e.set()


class _Spinners:
    """``n`` Python threads spinning on the GIL (busy plaintext / helper threads); their CPU is reported apart."""

    def __init__(self, n: int):
        self.stop = threading.Event()
        self.ths = [threading.Thread(target=self._spin, daemon=True) for _ in range(n)]
        for t in self.ths:
            t.start()

    def _spin(self) -> None:
        x = 0
        while not self.stop.is_set():
            x += 1

    def cpu(self) -> float:
        total = 0.0
        for t in self.ths:
            if t.ident is not None and t.is_alive():
                total += time.clock_gettime(time.pthread_getcpuclockid(t.ident))
        return total

    def close(self) -> None:
        self.stop.set()


def _level(lc: Any, lane: Any, chans: list[tuple[Any, int]], k: int, shape: dict[str, Any], steps: int,
           modes: list[str], prov_pid: int, spin: _Spinners) -> dict[str, Any]:
    from pgw.wire.messages import ComputeItem, ComputeRequest

    ring_ix, lin_ix, lin_id, m, n, nb, ring = (shape[x] for x in ("ring_ix", "lin_ix", "lin_id", "m", "n", "nb",
                                                                   "ring"))
    bodies = [secrets.token_bytes(n * nb) for _ in range(k)]
    joined = b"".join(bodies)
    ch0 = chans[0][0]
    inner = lc._client

    def one(ch: Any, rows: int, body: bytes) -> None:
        o = bytearray(rows * m * nb)
        st, _ = ch.trip(ring_ix, lin_ix, rows, secrets.token_bytes(16), 0, body, o, 30.0)
        assert st == 0, st

    def step_multi() -> None:
        o = bytearray(k * m * nb)
        st, _ = ch0.trip_multi(ring_ix, [lin_ix] * k, [1] * k, secrets.token_bytes(16), 0, bodies, o, 30.0)
        assert st == 0, st

    def step_rows() -> None:
        one(ch0, k, joined)

    def step_serial() -> None:
        for b in bodies:
            one(ch0, 1, b)

    pool = _Pool([c for c, _ in chans[:k]]) if "parallel" in modes else None

    def step_parallel() -> None:
        assert pool is not None
        pool.run(lambda i, ch: one(ch, 1, bodies[i]))

    def step_stream() -> None:
        req = ComputeRequest(secrets.token_hex(16), lane.ticket.fingerprint, ring,
                             tuple(ComputeItem(lin_id, 1, b) for b in bodies))
        inner.compute_sync(req)

    fns = {"multi": step_multi, "rows": step_rows, "serial": step_serial, "parallel": step_parallel,
           "stream": step_stream}
    res: dict[str, Any] = {}
    try:
        for mode in modes:
            if mode == "rows" and k > int(lane.ticket.max_rows):
                continue
            if mode == "multi" and (not hasattr(ch0, "trip_multi") or not getattr(lane.ticket, "max_items", 0)):
                continue
            fn = fns[mode]
            for _ in range(10):
                fn()
            walls = []
            c0, s0, p0 = time.process_time(), spin.cpu(), _proc_cpu(prov_pid)
            for _ in range(steps):
                t0 = time.perf_counter()
                fn()
                walls.append(time.perf_counter() - t0)
            c1, s1, p1 = time.process_time(), spin.cpu(), _proc_cpu(prov_pid)
            res[mode] = {"wall_us_p50": round(1e6 * statistics.median(walls), 1),
                         "wall_us_p90": round(1e6 * sorted(walls)[int(0.9 * len(walls))], 1),
                         "steps_per_s": round(len(walls) / sum(walls), 1),
                         # the process's CPU without the spinner threads': the lane's own cost per step
                         "client_cpu_us_per_step": round(1e6 * ((c1 - c0) - (s1 - s0)) / steps, 1),
                         "provider_cpu_us_per_step": round(1e6 * (p1 - p0) / steps, 1)}
    finally:
        if pool is not None:
            pool.close()
    return res


def measure(lc: Any, fingerprint: str, ring_name: str, prov_pid: int, ks: list[int], steps: int, busy: int,
            modes: list[str]) -> dict[str, Any]:
    from pgw.core.ring import RingParams

    lane = lc._lane
    if lane is None:
        lc.open_sync(fingerprint, ring_name, if_needed=True)
        lane = lc._lane
    assert lane is not None, f"no lane: {type(lc).__name__} {getattr(lc, 'lane_stats', None)}"
    ring = lane.ticket.rings[0]
    # the widest linear of layer 0 (gate_up)
    lin_id = max((k for k in lane.linears if k.startswith("0.")), key=lambda k: lane.linears[k][1])
    lin_ix, m, n = lane.linears[lin_id]
    shape = {"ring_ix": lane.rings[ring], "lin_ix": lin_ix, "lin_id": lin_id, "m": m, "n": n,
             "nb": RingParams.from_name(ring).k // 8, "ring": ring}
    spin = _Spinners(busy)
    out: dict[str, Any] = {"linear": lin_id, "m": m, "n": n, "busy_threads": busy,
                           "max_rows": int(lane.ticket.max_rows)}
    chans: list[tuple[Any, int]] = []
    try:
        for k in ks:
            while len(chans) < k:
                got = lc._acquire(lane, 5.0)
                if got is None:  # pool limit: open more connections directly
                    chans.append((lc._factory(lane.ticket, bytearray(lane.secret)), -1))
                else:
                    chans.append(got)
            res = _level(lc, lane, chans, k, shape, steps, modes, prov_pid, spin)
            out[str(k)] = res
            print(json.dumps({"k": k, **res}), file=sys.stderr, flush=True)
    finally:
        spin.close()
        for c, gen in chans:
            if gen >= 0:
                lc._release(c, gen)
            else:
                c.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["tiny", "qwen0.5b"], default="tiny")
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--busy", type=int, default=0)
    ap.add_argument("--modes", default="multi,rows,parallel,serial,stream")
    args = ap.parse_args()
    from benchmarks.sdk_concurrency_bench import MODEL, _pack
    from tests.sdk.conftest import make_pki, start_provider, stop_provider

    with tempfile.TemporaryDirectory(prefix="pgw-lane-multi-") as d:
        root = Path(d)
        for sub in ("pki", "provider", "client"):
            (root / sub).mkdir()
        pack = _pack(args.model, root / "pack" / "pack")
        prov = start_provider(root / "provider", pack, make_pki(root / "pki"))
        try:
            from pgw import OpenAI
            from pgw.sdk import LocalPackProvisioner

            c = OpenAI(api_key=prov.new_key(), cache_dir=root / "client" / "c", state_dir=root / "client" / "s",
                       mask_scheme="plain", mask_workers=1, mask_target_slots=64,
                       provisioner=LocalPackProvisioner({MODEL: pack}))
            c.warmup(MODEL)
            c.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=2)
            rt = c._engine.models[MODEL].runtime
            out = measure(rt.client, rt.fingerprint, rt.ring.name, prov.proc.pid,
                          [int(x) for x in args.ks.split(",")], args.steps, args.busy, args.modes.split(","))
            out["model"] = args.model
            c.close()
        finally:
            stop_provider(prov)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
