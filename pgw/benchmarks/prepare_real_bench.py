"""The client's one-time setup of a real pack end to end on CPU: what ``pgw sdk prepare`` does, old vs new code.

A compute service on loopback serves ``--pack`` with provisioning on (in this process); the client runs in a child
process (its own peak RSS) with ``--src`` on its ``PYTHONPATH`` (a checkout of the code to measure) and calls
``provision(hint="local", factory_pack=True, masked_head=...)``: the gateway pack, the factory pack (int8 weights,
fingerprint-checked) and the pks-v3 hint. With the new code the hint is built while each linear's weights are in
memory (``resume=True`` as ``pgw sdk prepare``); the old code built the factory pack, then the hint from it.

Reported: wall seconds (total, per stage), the client's CPU seconds, peak RSS, peak disk of the cache directory
(sampled every 0.2 s), bytes downloaded, sizes. Run (each mode in its own call; the box's load shows in the wall time):

    PYTHONPATH=src python benchmarks/prepare_real_bench.py --pack <0.5B pack> --src src --label new
    PYTHONPATH=src python benchmarks/prepare_real_bench.py --pack <0.5B pack> --src <old checkout>/src --label old
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

TOKEN = "prepare-bench-token-0123456789abcdef"  # noqa: S105 - loopback benchmark credential


def _du(p: Path) -> int:
    tot = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                tot += os.lstat(os.path.join(root, f)).st_blocks * 512
            except OSError:
                pass
    return tot


def client(url: str, fp: str, cache: Path, masked_head: bool, resume: bool) -> None:
    import inspect

    from pgw.provision.client import provision
    from pgw.provision.session import ProvisionSession

    peak = {"disk": 0}
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(0.2):
            peak["disk"] = max(peak["disk"], _du(cache))

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    kw = {"resume": True} if resume and "resume" in inspect.signature(provision).parameters else {}
    t0, c0 = time.perf_counter(), time.process_time()
    pm = provision(fp, session=ProvisionSession(url, token=TOKEN, retries=0), cache_dir=cache, hint="local",
                   masked_head=masked_head, factory_pack=True, **kw)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    stop.set()
    th.join()
    peak["disk"] = max(peak["disk"], _du(cache))
    out = {"wall_s": round(wall, 1), "client_cpu_s": round(cpu, 1),
           "peak_rss_GB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
           "peak_disk_GB": round(peak["disk"] / 1e9, 3), "final_disk_GB": round(_du(cache) / 1e9, 3),
           "sizes_GB": {k: round(v / 1e9, 3) for k, v in pm.sizes.items()},
           "timings_s": {k: round(v, 1) for k, v in pm.timings.items()}, "hint_source": pm.hint_source,
           "a_seed": pm.hint_store.a_seed.hex()}
    pm.close()
    print("RESULT " + json.dumps(out), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--src", type=Path, required=True, help="the client code to measure (a checkout's src dir)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--no-masked-head", action="store_true")
    ap.add_argument("--client", nargs=3, help=argparse.SUPPRESS)
    ap.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.client:
        url, fp, cache = args.client
        client(url, fp, Path(cache), not args.no_masked_head, args.resume)
        return

    from tests.provision.test_provision_e2e import _free_port, _Server

    from pgw.modelpack import ModelPack
    from pgw.service.auth import hash_token
    from pgw.service.config import ServiceSettings

    with tempfile.TemporaryDirectory(prefix="pgw-prepare-bench-") as d:
        root = Path(d)
        tok = root / "tenants.txt"
        tok.write_text(f"tenant-a:{hash_token(TOKEN)}\n")
        port = _free_port()
        s = ServiceSettings(host="127.0.0.1", port=port, metrics_port=None, packs=[args.pack], device="cpu",
                            tokens_file=tok, batch_window_ms=1.0, rings=["K32A12C27"], drain_timeout_s=2.0,
                            provision=True, masked_head="also", warmup=False)
        srv = _Server(s)
        url = f"http://127.0.0.1:{port}"
        srv.start(url)
        try:
            with ModelPack.open(args.pack) as p:
                fp = p.fingerprint
            env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
            env["PYTHONPATH"] = str(args.src.resolve())
            cmd = [sys.executable, os.path.abspath(__file__), "--pack", str(args.pack), "--src", str(args.src),
                   "--client", url, fp, str(root / "cache"), "--resume"]
            if args.no_masked_head:
                cmd.append("--no-masked-head")
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200, check=False)  # noqa: S603
            line = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
            if r.returncode != 0 or not line:
                raise SystemExit(f"client failed:\n{r.stderr[-4000:]}")
            res = json.loads(line[-1][7:])
            res["label"] = args.label
            print(json.dumps(res, indent=1))
        finally:
            srv.stop()
            shutil.rmtree(root / "cache", ignore_errors=True)


if __name__ == "__main__":
    main()
