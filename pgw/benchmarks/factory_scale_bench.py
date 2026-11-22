#!/usr/bin/env python3
"""Encrypted mask factory scale-out: aggregate slots/s vs factory worker processes and provider services.

Sets up, on one box: a synthetic checkpoint with real layer shapes (``--preset``, ``--layers`` decoder
layers, random weights), its pack and one-time HE hint; ``--providers`` real compute services
(``pgw service serve``, plain HTTP on loopback, one per ``--provider-devices`` entry: ``cpu`` here, ``cuda:0``
``cuda:1`` on a T4x2); then, for each ``--workers`` count, a fresh mask store filled by a
:class:`pgw.factory.fleet.FactoryFleet` of that many worker processes (``he_service_urls`` = all services,
sharded by segment) until ``workers * --segments-per-worker`` segments are committed.

Reported per worker count (measured): ``steady_slots_per_s`` = slots committed after every worker's first
segment / time from the W-th to the last commit (segment ``added_at`` in the store; excludes start-up:
pack open, hint map, connection set-up), ``wall_slots_per_s`` (including start-up), and the per-worker
seconds per segment from each worker's ``factory-state.json``. ``model_slots_per_s`` scales by
``layers / model_layers`` (work per layer is identical): an estimate for the whole model.

Customer-side HE work runs on the CPU unless ``--factory-gpu`` (the worker processes do not see the GPUs).

    PYTHONPATH=src python benchmarks/factory_scale_bench.py --preset qwen2.5-0.5b --layers 2 \
        --workers 1 2 4 --providers 1 --provider-devices cpu --segment-slots 512 --json out.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

PRESETS = {  # hidden, intermediate, heads, kv heads, model layers
    "qwen2.5-0.5b": (896, 4864, 14, 2, 24),
    "qwen2.5-7b": (3584, 18944, 28, 4, 28),
    "tiny": (64, 128, 4, 2, 2),
}
TOKEN = "bench-token-0123456789abcdef"  # noqa: S105 - loopback benchmark credential


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def make_pack(work: Path, preset: str, layers: int) -> Path:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from pgw.modelpack import pack_checkpoint

    hidden, inter, heads, kv, _ = PRESETS[preset]
    ckpt, pack = work / "ckpt", work / "pack"
    if (pack / "manifest.json").exists():
        return pack
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv, vocab_size=512,
                      max_position_embeddings=512, tie_word_embeddings=True)
    with torch.device("cpu"):
        model = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.normal_(0, 0.02)
    model.eval().save_pretrained(ckpt, safe_serialization=True)
    del model
    pack_checkpoint(ckpt, pack, name=f"bench-{preset}-l{layers}")
    return pack


def build_hint(pack_dir: Path, hint_dir: Path, ring: str, rank: int) -> float:
    from pgw.core.ring import RingParams
    from pgw.hemask.backend import HintStore
    from pgw.hemask.lwe import HEParams
    from pgw.modelpack import ModelPack

    pack = ModelPack.open(pack_dir, verify="none", components=("linears",))
    q = HEParams(k=RingParams.from_name(ring).k, rank=rank).q
    t0 = time.perf_counter()
    HintStore.open_or_build(hint_dir, pack, rank, q)
    return time.perf_counter() - t0


class Services:
    """``pgw service serve`` subprocesses on loopback, one per device."""

    def __init__(self, work: Path, pack: Path, devices: list[str], ring: str, delay_ms: float):
        from pgw.service.auth import hash_token

        self.tokens = work / "tokens.txt"
        self.tokens.write_text(f"bench:{hash_token(TOKEN)}\n")
        self.token_file = work / "token"
        self.token_file.write_text(TOKEN)
        self.procs: list[subprocess.Popen[bytes]] = []
        self.urls: list[str] = []
        self.proxies: list[Any] = []
        self.logs: list[Path] = []
        for i, dev in enumerate(devices):
            port = _free_port()
            env = dict(os.environ, PGW_SERVICE_DEVICE=dev, PGW_SERVICE_TOKENS_FILE=str(self.tokens),
                       PGW_SERVICE_RINGS=json.dumps([ring]), PGW_SERVICE_HOST="127.0.0.1",
                       PGW_SERVICE_PORT=str(port), PGW_SERVICE_PACKS=json.dumps([str(pack)]))
            logf = work / f"service{i}.log"
            self.logs.append(logf)
            fh = open(logf, "wb")
            self.procs.append(subprocess.Popen([sys.executable, "-m", "pgw.cli", "service", "serve"], env=env,
                                               stdout=fh, stderr=subprocess.STDOUT))
            if delay_ms > 0:
                sys.path.insert(0, str(Path(__file__).resolve().parent))
                from delay_proxy import DelayProxy

                prx = DelayProxy(("127.0.0.1", port), one_way_s=delay_ms / 2e3)
                self.proxies.append(prx)
                port = prx.start()
            self.urls.append(f"http://127.0.0.1:{port}")
        self._direct = [u if not self.proxies else None for u in self.urls]

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        deadline = time.monotonic() + timeout_s
        for i, url in enumerate(self.urls):
            while True:
                if self.procs[i].poll() is not None:
                    raise RuntimeError(f"service {i} exited: {self.logs[i].read_text()[-3000:]}")
                with contextlib.suppress(OSError):
                    with urllib.request.urlopen(f"{url}/readyz", timeout=2) as r:  # noqa: S310 - loopback
                        if r.status == 200:
                            break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"service {i} not ready")
                time.sleep(0.5)

    def stop(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
        for prx in self.proxies:
            prx.stop()


def _commits(store_root: Path) -> list[tuple[float, int]]:
    con = sqlite3.connect(f"file:{store_root / 'state.sqlite'}?mode=ro", uri=True)
    try:
        return [(float(t), int(s)) for t, s in con.execute("SELECT added_at, slots FROM segments ORDER BY added_at")]
    finally:
        con.close()


def run_point(work: Path, pack: Path, svc: Services, workers: int, args: argparse.Namespace) -> dict[str, Any]:
    from pgw.factory.config import FactorySettings
    from pgw.factory.fleet import FactoryFleet
    from pgw.maskstore import FileKeyProvider, MaskStore, write_key_file

    tag = f"w{workers}"
    root = work / f"store-{tag}"
    key = work / f"kek-{tag}"
    write_key_file(key)
    st = MaskStore.open(root, FileKeyProvider(key), low_water_slots=0, min_free_bytes=0)
    n_seg = workers * args.segments_per_worker
    target = n_seg * args.segment_slots
    s = FactorySettings(
        pack_path=pack, rings=(args.ring,), segment_slots=args.segment_slots, low_water_slots=target,
        target_slots=target, poll_interval_s=0.5, workers=workers, he_service_urls=tuple(svc.urls),
        he_token_file=svc.token_file, he_hint_dir=work / "hints", he_rank=args.rank, he_encrypt=args.he_encrypt,
        he_groups_per_call=args.groups_per_call, state_dir=work / f"state-{tag}", min_free_bytes=0,
        cpu_threads=args.cpu_threads,
    )
    from pgw.modelpack import ModelPack

    fp = ModelPack.open(pack, verify="none", components=("linears",)).fingerprint
    from pgw.core.ring import RingParams

    rp = RingParams.from_name(args.ring)
    t_start = time.time()
    fleet = FactoryFleet(s, store_overrides={"root": root, "key_file": key, "key_provider": "file",
                                             "min_free_bytes": 0}).start()
    try:
        deadline = time.monotonic() + args.timeout_s
        while st.available_slots(fp, rp) < target:
            fleet.supervise_once()
            if time.monotonic() > deadline:
                raise TimeoutError(f"{tag}: {st.available_slots(fp, rp)} of {target} slots after {args.timeout_s}s")
            time.sleep(0.1)
        t_end = time.time()
    finally:
        fleet.stop(timeout_s=60)
    commits = _commits(root)
    times = [t for t, _ in commits]
    steady = None
    if len(commits) > workers:
        span = times[-1] - times[workers - 1]
        steady = sum(sl for _, sl in commits[workers:]) / span if span > 0 else None
    per_worker = []
    for k in range(workers):
        p = fleet.root / f"w{k}" / "factory-state.json"
        if p.exists():
            d = json.loads(p.read_text())
            tot = next(iter(d.get("totals", {}).values()), {})
            if tot:
                per_worker.append({"worker": k, "segments": int(tot["segments"]),
                                   "s_per_segment": round(float(tot["seconds"]) / max(1, int(tot["segments"])), 3)})
    st.close()
    scale = args.layers / PRESETS[args.preset][4]
    out = {
        "workers": workers, "providers": len(svc.urls), "segments": len(commits), "slots": sum(s for _, s in commits),
        "overshoot_slots": sum(s for _, s in commits) - target,
        "wall_s": round(t_end - t_start, 2),
        "wall_slots_per_s": round(target / (t_end - t_start), 2),
        "steady_slots_per_s": None if steady is None else round(steady, 2),
        "model_steady_slots_per_s_est": None if steady is None else round(steady * scale, 2),
        "per_worker": per_worker,
    }
    shutil_rm(root)
    return out


def shutil_rm(p: Path) -> None:
    import shutil

    shutil.rmtree(p, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="qwen2.5-0.5b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--provider-devices", nargs="+", default=["cpu"],
                    help="one compute service per entry (cpu, cuda:0, cuda:1, ...)")
    ap.add_argument("--segment-slots", type=int, default=512)
    ap.add_argument("--segments-per-worker", type=int, default=3)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--he-encrypt", default="dense", choices=["dense", "ntt_pre"])
    ap.add_argument("--groups-per-call", type=int, default=2)
    ap.add_argument("--cpu-threads", type=int, default=0, help="per worker (0 = cores / workers)")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="added round-trip latency per service (proxy)")
    ap.add_argument("--factory-gpu", action="store_true", help="let factory workers use the GPUs for HE")
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    import logging

    logging.basicConfig(level=logging.WARNING)
    work = args.work or Path(tempfile.mkdtemp(prefix="pgw-fscale-"))
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    pack = make_pack(work, args.preset, args.layers)
    t_pack = time.perf_counter() - t0
    t_hint = build_hint(pack, work / "hints", args.ring, args.rank)
    print(f"pack {t_pack:.1f}s, hint {t_hint:.1f}s", flush=True)
    svc = Services(work, pack, args.provider_devices, args.ring, args.delay_ms)
    results: dict[str, Any] = {"preset": args.preset, "layers": args.layers, "model_layers": PRESETS[args.preset][4],
                               "segment_slots": args.segment_slots, "ring": args.ring, "he_encrypt": args.he_encrypt,
                               "groups_per_call": args.groups_per_call, "provider_devices": args.provider_devices,
                               "delay_ms": args.delay_ms, "cpus": os.cpu_count(), "hint_build_s": round(t_hint, 1),
                               "points": []}
    try:
        svc.wait_ready()
        if not args.factory_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned workers inherit: customer-side HE on the CPU
        for w in args.workers:
            r = run_point(work, pack, svc, w, args)
            print(json.dumps(r), flush=True)
            results["points"].append(r)
            if args.json:
                args.json.write_text(json.dumps(results, indent=1))
    finally:
        svc.stop()
    base = next((p["steady_slots_per_s"] for p in results["points"] if p["workers"] == 1), None)
    print(f"{'workers':>8} {'providers':>9} {'steady slots/s':>15} {'speedup':>8} {'wall slots/s':>13} "
          f"{'overshoot':>9}")
    for p in results["points"]:
        sp = p["steady_slots_per_s"]
        print(f"{p['workers']:>8} {p['providers']:>9} {sp if sp is not None else '-':>15} "
              f"{(sp / base) if (sp and base) else float('nan'):>8.2f} {p['wall_slots_per_s']:>13} "
              f"{p['overshoot_slots']:>9}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
