"""Gateway CPU per masked trip on the decode path: ``PGW_TRIP_FAST`` vs ``PGW_TRIP_NATIVE`` (one native call
per trip on each side of the lane call), per linear kind, at real shapes.

A synthetic pack (``decode_cpu_bench.build_pack``: zero linear weights, so the stub service's all-zero answer
is exact), a disk mask store, a windowed reservation as the gateway makes, the outbound-noise monitor at the
gateway defaults and a stub lane that answers instantly. Each trip is ``forward_many([call])`` on 1 row, in
model order (one "token" = one trip of every linear), so the store, lease refills and window extensions are
amortized as in decode. Reports microseconds per trip (wall, single thread) and ms per token of trips.

Usage:
    PYTHONPATH=src python benchmarks/trip_native_bench.py --preset 7b --layers 2 --tokens 48 \
        --workdir /dev/shm/pgw-tn [--masked-head] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_cpu_bench import PRESETS, build_pack, set_wnorm


def run(args: argparse.Namespace, native_on: bool, split: int = 0, driver: bool = False) -> dict[str, Any]:
    os.environ["PGW_TRIP_FAST"] = "1"
    os.environ["PGW_TRIP_NATIVE"] = "1" if native_on else "0"
    os.environ["PGW_TRIP_SPLIT"] = str(split)
    os.environ["PGW_TRIP_DRIVER"] = "1" if driver else "0"
    from pgw.core import RingParams
    from pgw.engine import MaskedBackend
    from pgw.engine.backends import LinearCall
    from pgw.factory import MaskGenerator
    from pgw.gateway.masks import WindowedReservation
    from pgw.maskstore import FileKeyProvider, MaskStore
    from pgw.maskstore.counters import FileMonotonicCounter
    from pgw.modelpack import ModelPack
    from pgw.observability.noise import OutboundNoiseMonitor
    from pgw.wire.messages import ComputeResponse, ComputeResult

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    pack: Any = ModelPack.open(build_pack(work, args.preset, args.layers))
    if args.masked_head:
        from pgw.engine.masked_head import MaskedHeadPack
        from pgw.modelpack.loader import LinearWeights

        class ZeroHeadPack(MaskedHeadPack):
            def _head(self) -> Any:
                if self._w is None:
                    s = self._spec
                    self._w = LinearWeights(self._lid, torch.zeros((s.out_features, s.in_features), dtype=torch.int8),
                                            torch.ones(s.out_features), None)
                return self._w

        pack = ZeroHeadPack(pack)
    ring = RingParams.from_name(args.ring)
    specs = list(pack.linear_specs())
    slots = (args.tokens + args.warm) * args.rows + 8
    # one generated store per process, copied for every run: the runs mask with the same rows (payloads comparable)
    import shutil

    tkey = f"tn-template-{os.getpid()}-{args.prime}-{slots}"
    if not (work / tkey).exists():
        (work / f"{tkey}-ctr").mkdir(parents=True, exist_ok=True)
        tmpl = MaskStore.open(work / tkey, FileKeyProvider(work / "kek", create=True), fsync=False,
                              counter=FileMonotonicCounter(work / f"{tkey}-ctr" / "c", fsync=False))
        MaskGenerator(pack, ring, check_field="p31" if args.prime else None).generate_into_store(
            tmpl, slots, chunk_slots=16, factory_id="tn")
        tmpl.close()
    tag = f"{time.time_ns()}"
    shutil.copytree(work / tkey, work / f"tn-store-{tag}")
    shutil.copytree(work / f"{tkey}-ctr", work / f"tn-ctr-{tag}")
    store = MaskStore.open(work / f"tn-store-{tag}", FileKeyProvider(work / "kek", create=True), fsync=False,
                           counter=FileMonotonicCounter(work / f"tn-ctr-{tag}" / "c", fsync=False))
    from pgw.maskstore import store as store_mod

    store_mod.TRIP_DECRYPT_AHEAD = bool(args.decrypt_ahead and native_on)
    store_mod.TRIP_R_AHEAD = bool(args.r_ahead)
    outm = {sp.id.key: sp.out_features for sp in specs}
    nb = ring.k // 8
    zeros = {k: bytes(m * nb) for k, m in outm.items()}

    import hashlib

    sent = hashlib.sha256()

    class Stub:
        sync_stream = True

        def compute_sync(self, req: Any) -> Any:
            for i in req.items:
                sent.update(i.x_hat)
            return ComputeResponse(req.request_id, tuple(
                ComputeResult(i.linear, i.rows, zeros[i.linear] if i.rows == 1 else bytes(i.rows * outm[i.linear] * nb))
                for i in req.items))

    client: Any = Stub()
    lane_srv = None
    if args.lane:  # the reference lane server over a Unix socket (its own threads), native channel
        import tempfile
        import uuid
        from types import SimpleNamespace

        from pgw.wire import LinearInfo
        from pgw.wire.lane import LaneComputeClient, LaneConfig, LaneStubServer
        from pgw.wire.messages import LaneOpenRequest

        sock = Path(tempfile.mkdtemp(prefix="pgwtn")) / "lane.sock"

        def lane_compute(li: Any, rows: int, ring_name: str, x_hat: bytes) -> bytes:
            sent.update(x_hat)
            return zeros[li.id] if rows == 1 else bytes(rows * outm[li.id] * nb)

        lane_srv = LaneStubServer(lane_compute, fingerprint=pack.fingerprint, rings=(ring.name,),
                                  linears=tuple(LinearInfo(sp.id.key, sp.out_features, sp.in_features) for sp in specs),
                                  transport="uds", path=str(sock), max_rows=8, record=False)
        stub = client

        class Wrapped:
            sync_stream = True
            config = SimpleNamespace(base_url="http://127.0.0.1:9", request_timeout_s=10.0)

            def compute_sync(self, req: Any) -> Any:
                return stub.compute_sync(req)

            def close(self) -> None:
                pass

            def lane_open_sync(self, fp: str, rname: str, *, transports: Any = ("uds",), lane_versions: Any = (1,)) -> Any:
                return lane_srv.issue_ticket(LaneOpenRequest(uuid.uuid4().hex, fp, rname, tuple(transports),
                                                             tuple(lane_versions)))

        client = LaneComputeClient(Wrapped(), LaneConfig(native=True))
    noise = OutboundNoiseMonitor(ring, sample_rate=0.05, fail_closed=True)
    be = MaskedBackend(pack, None, client, ring, pack.fingerprint, min_soundness_bits=20, noise_monitor=noise,
                       **({"prime_check": True} if args.prime else {}))
    set_wnorm(be, pack, args.wnorm_frac)
    be.warm_trip_engine()
    fp = pack.fingerprint
    res = WindowedReservation(store.reserve(fp, ring, args.window), lambda s: store.reserve(fp, ring, s),
                              cap=slots - 4, window=args.window)
    g = torch.Generator().manual_seed(0)
    xs = {sp.id.key: torch.randn(args.rows, sp.in_features, generator=g) for sp in specs}
    per: dict[str, list[float]] = {}
    outs: list[bytes] = []
    import resource

    def proc_cpu() -> float:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return ru.ru_utime + ru.ru_stime

    main_cpu = all_cpu = 0.0
    prof = None
    try:
        for tok in range(args.warm + args.tokens):
            if tok == args.warm:
                if args.profile and native_on:
                    import cProfile

                    prof = cProfile.Profile()
                    prof.enable()
                c0, p0 = time.thread_time(), proc_cpu()
            for sp in specs:
                c = LinearCall(sp.id, xs[sp.id.key], res)
                t0 = time.perf_counter()
                y = be.forward_many([c])[0]
                dt = time.perf_counter() - t0
                if tok >= args.warm:
                    per.setdefault(sp.id.kind.value, []).append(dt)
                if tok == args.warm:
                    outs.append(y.numpy().tobytes())
            be.join_checks()
        main_cpu, all_cpu = time.thread_time() - c0, proc_cpu() - p0
        if prof is not None:
            prof.disable()
            import io
            import pstats

            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(18)
            print(f"---- profile (driver={driver}, split={split}) ----", file=sys.stderr)
            print(buf.getvalue(), file=sys.stderr)
    finally:
        res.close()
        store.close()
        if lane_srv is not None:
            client.close()
            lane_srv.close()
    kinds = {k: {"us_median": round(1e6 * sorted(v)[len(v) // 2], 1), "us_mean": round(1e6 * sum(v) / len(v), 1),
                 "calls": len(v)} for k, v in per.items()}
    tot = sum(sum(v) for v in per.values()) / args.tokens
    full, lay = PRESETS[args.preset]["full_layers"], args.layers
    head = sum(per.get("lm_head", [])) / args.tokens
    return {"native": native_on, "per_kind": kinds, "ms_per_token_trips": round(1e3 * tot, 3),
            "main_thread_cpu_ms_per_token": round(1e3 * main_cpu / args.tokens, 3),
            "process_cpu_ms_per_token": round(1e3 * all_cpu / args.tokens, 3),
            "ms_per_token_trips_full_model": round(1e3 * (head + (tot - head) * full / lay), 2),
            "main_thread_cpu_ms_full_model": round(1e3 * (head + (main_cpu / args.tokens - head) * full / lay), 2),
            "process_cpu_ms_full_model": round(1e3 * (head + (all_cpu / args.tokens - head) * full / lay), 2),
            "trips_per_token": len(specs), "rows_per_trip": args.rows, "split": split,
            "us_per_trip": round(1e6 * tot / len(specs), 1),
            "process_cpu_us_per_trip": round(1e6 * all_cpu / args.tokens / len(specs), 1),
            "outputs": outs, "payload_sha256": sent.hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="0.5b", choices=sorted(PRESETS))
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--ring", default="K32A12C27")
    ap.add_argument("--wnorm-frac", type=float, default=0.3)
    ap.add_argument("--masked-head", action="store_true")
    ap.add_argument("--prime", action="store_true", help="prime-field check segments (p31) + prime_check")
    ap.add_argument("--decrypt-ahead", action="store_true", help="PGW_TRIP_DECRYPT_AHEAD on the native run")
    ap.add_argument("--only", choices=["fast", "native"], default=None)
    ap.add_argument("--rows", type=int, default=1, help="rows per trip (R)")
    ap.add_argument("--split", type=int, default=0, help="also run the native path with PGW_TRIP_SPLIT=N")
    ap.add_argument("--r-ahead", type=int, default=1, help="PGW_TRIP_R_AHEAD (r of the next trip on a helper)")
    ap.add_argument("--profile", action="store_true", help="cProfile the native runs (top functions to stderr)")
    ap.add_argument("--lane", action="store_true", help="trips over the reference lane server (UDS, native channel)")
    ap.add_argument("--driver", action="store_true", help="also run the native run with PGW_TRIP_DRIVER=1 (--lane)")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    a = run(args, False) if args.only != "native" else {"outputs": None, "payload_sha256": None}
    b = run(args, True) if args.only != "fast" else {"outputs": None, "payload_sha256": None}
    c = run(args, True, args.split) if args.split >= 2 else None
    b_out, b_sha = b.pop("outputs"), b.pop("payload_sha256")
    res = {"preset": args.preset, "layers": args.layers, "masked_head": args.masked_head, "prime": args.prime,
           "decrypt_ahead": args.decrypt_ahead, "r_ahead": args.r_ahead, "rows": args.rows,
           "payloads_identical": a.pop("payload_sha256") == b_sha,
           "outputs_identical": a.pop("outputs") == b_out, "fast": a, "native": b}
    if c is not None:
        res["split_payloads_identical"] = c.pop("payload_sha256") == b_sha
        res["split_outputs_identical"] = c.pop("outputs") == b_out
        res["native_split"] = c
    if args.driver:
        dr = run(args, True, args.split, driver=True)
        res["driver_payloads_identical"] = dr.pop("payload_sha256") == b_sha
        res["driver_outputs_identical"] = dr.pop("outputs") == b_out
        res["native_driver"] = dr
    print(json.dumps(res, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
