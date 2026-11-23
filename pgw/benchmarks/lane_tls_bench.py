#!/usr/bin/env python3
"""Per-trip latency of the service trip lane's transports (package L1: the TCP+TLS lane), without a GPU.

The service side is the real native listener (``pgw_fast.LaneService``, ``rust/pgw-lane``) in a subprocess,
serving every call with a zero-cost Python stub executor (a cached reply of the right size: one GIL
acquire per call, about what CUDA-graph replay costs minus the GPU, which needs no GIL at all). The client
side is what the gateway's decode path uses, in this process:

* ``native-tls``: ``pgw_fast.TripChannel.connect_service_tls`` (rustls TLS 1.3, A1 frames, GIL released for
  the whole trip, spin-then-block wait), against the TCP+TLS listener (server certificate from a throwaway CA
  for 127.0.0.1, verified; ``--mtls``: a client certificate too, bound to the ticket);
* ``native-uds``: ``TripChannel.connect_service_uds`` against the Unix-domain-socket listener (no TLS);
* ``py-uds``: :class:`pgw.wire.service_lane.ServiceLaneChannel` (the Python channel the gateway's UDS lane
  used before) against the UDS listener.
* ``native-shm``: ``connect_service_uds`` + ``attach_shm()`` (opt-in shared-memory transport: the same frames
  in a sealed memfd segment passed over the socket; the socket only as the doorbell) against a listener with
  ``shm=True``.

``--stub-us 44,40,191,79``: instead of the Python stub, the listener's native benchmark executor
(``LaneService(bench_stub_us=...)``) busy-waits that long per linear (qkv, o, gate_up, down: the H100's
measured 1-row graph times at 7B) and answers from its own buffer, so the client sees a realistic reply
delay with no GIL on the service side. ``--zero-copy``: the listener's zero-copy replies;
``--spin-auto-us N``: the client's adaptive spin capped at N us; ``--read-spin-us``: the listener's request
spin (shm: 500 us when 0).

The stream-lane baseline (TLS, msgpack, WebSocket, the Python service) is
``wire_overhead_bench.py --tls --sync stream``; the full lane through the real service app (Python
executor hop to the event loop on CPU) is ``wire_overhead_bench.py --tls --sync lane --service-lane tcp``.

``--gil-load N`` adds N Python threads in the client process doing pure-Python work 1 ms on, 1 ms off (a
gateway whose other threads hold the GIL): a Python transport waits for the GIL at every socket call, the
native one once per trip.

Prints one JSON line per run and a summary (median p50/mean over ``--repeat`` interleaved runs). Numbers
from a shared 4-vCPU box are relative; the lead's Modal run decides.

Usage::

    python benchmarks/lane_tls_bench.py [--modes native-tls,native-uds,py-uds] [--preset 0.5b|7b]
        [--rows 1] [--calls 3000] [--repeat 3] [--spin-us 50] [--gil-load 0] [--mtls]
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SHAPES = {
    "0.5b": {"qkv": (1152, 896), "o": (896, 896), "gate_up": (9728, 896), "down": (896, 4864)},
    "7b": {"qkv": (4608, 3584), "o": (3584, 3584), "gate_up": (37888, 3584), "down": (3584, 18944)},
}
LAYERS = 2
BPE = 4  # K = 32


def _linears(preset: str) -> list[tuple[str, int, int]]:
    return [(f"{layer}.{k}", m, n) for layer in range(LAYERS) for k, (m, n) in SHAPES[preset].items()]


def _client_cert(td: Path, ca_dir: Path) -> tuple[Path, Path, Path]:
    """A client CA and a client certificate signed by it: ``(client_ca, cert, key)``."""
    import datetime as dt

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = dt.datetime.now(dt.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "bench client CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
          .not_valid_after(now + dt.timedelta(days=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                       content_commitment=False, key_encipherment=False, data_encipherment=False,
                                       key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
          .sign(ca_key, hashes.SHA256()))
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "gw")]))
            .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=1))
            .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    paths = (td / "client-ca.pem", td / "client.pem", td / "client.key")
    paths[0].write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    paths[1].write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    paths[2].write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    return paths


# ---- service process ---------------------------------------------------------------------------------
def _serve(args: argparse.Namespace) -> None:
    import pgw_fast as fast

    lins = _linears(args.preset)
    cache: dict[tuple[int, int], bytes] = {}

    def execute(_model: int, linear: int, _ring: int, rows: int, _body: bytes, _tenant: str, _left: float) -> bytes:
        out = cache.get((linear, rows))
        if out is None:
            out = cache[(linear, rows)] = bytes(rows * lins[linear][1] * BPE)
        return out

    td = Path(args.serve_dir)
    kw: dict[str, Any] = {"max_rows": 8, "executor": execute}
    if args.read_spin_us:
        kw["read_spin_us"] = int(args.read_spin_us)
    if args.stub_us:
        kw["bench_stub_us"] = [int(v) for v in args.stub_us.split(",")]
    if args.zero_copy:
        kw["zero_copy"] = True
    if args.serve_shm:
        kw["shm"] = True
    if args.serve == "tcp":
        kw.update(tls_addr="127.0.0.1:0", tls_cert_pem=(td / "server.pem").read_bytes(),
                  tls_key_pem=(td / "server.key").read_bytes())
        if (td / "client-ca.pem").exists():
            kw.update(tls_client_ca_pem=(td / "client-ca.pem").read_bytes(), require_client_cert=True)
        svc = fast.LaneService("", **kw)
    else:
        svc = fast.LaneService(str(td / "lane.sock"), **kw)
    svc.set_model(0, [(n, m, name.split(".")[1]) for name, m, n in lins], [("K32A12C27", BPE)])
    tickets = []
    cert_der = None
    if (td / "client.pem").exists():
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        cert_der = x509.load_pem_x509_certificate((td / "client.pem").read_bytes()).public_bytes(
            serialization.Encoding.DER)
    for _ in range(4):
        t = secrets.token_bytes(32)
        svc.add_ticket(t, "bench", 0, 600.0, cert_der)
        tickets.append(t.hex())
    print(json.dumps({"port": svc.port, "socket": str(td / "lane.sock"), "tickets": tickets}), flush=True)
    sys.stdin.read()  # until the client closes the pipe
    svc.stop(2.0)


# ---- client ------------------------------------------------------------------------------------------
def _channel(mode: str, info: dict[str, Any], td: Path, timeout: float, spin_us: int, spin_auto_us: int = 0) -> Any:
    ch = _channel0(mode, info, td, timeout, spin_us)
    if spin_auto_us and hasattr(ch, "set_spin_auto"):
        ch.set_spin_auto(spin_auto_us)
    if mode == "native-shm" and not ch.attach_shm():
        raise SystemExit("the listener refused the shm transport")
    return ch


def _channel0(mode: str, info: dict[str, Any], td: Path, timeout: float, spin_us: int) -> Any:
    import pgw_fast as fast

    from pgw.wire.service_lane import ServiceLaneChannel

    ticket = bytes.fromhex(info["tickets"][0])
    if mode == "native-tls":
        cert = key = None
        if (td / "client.pem").exists():
            cert, key = (td / "client.pem").read_bytes(), (td / "client.key").read_bytes()
        return fast.TripChannel.connect_service_tls("127.0.0.1", int(info["port"]), "127.0.0.1",
                                                    (td / "ca.pem").read_bytes(), ticket, timeout, 8, spin_us,
                                                    cert, key)
    if mode in ("native-uds", "native-shm"):
        return fast.TripChannel.connect_service_uds(info["socket"], ticket, timeout, 8, spin_us)
    if mode == "py-uds":
        return ServiceLaneChannel(info["socket"], ticket, timeout, 8)
    raise SystemExit(f"unknown mode {mode!r}")


def _one_run(args: argparse.Namespace, mode: str, rows: int) -> dict[str, Any]:
    import numpy as np
    from wire_overhead_bench import _make_tls, _proc_stat

    with tempfile.TemporaryDirectory(prefix="pgwlt") as tds:
        td = Path(tds)
        _make_tls(td)
        if args.mtls and mode == "native-tls":
            _client_cert(td, td)
        serve = "tcp" if mode.endswith("tls") else "uds"
        cmd = [sys.executable, __file__, "--serve", serve, "--serve-dir", tds, "--preset", args.preset,
               "--read-spin-us", str(args.read_spin_us)]
        if args.stub_us:
            cmd += ["--stub-us", args.stub_us]
        if args.zero_copy:
            cmd.append("--zero-copy")
        if mode == "native-shm":
            cmd.append("--serve-shm")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)  # noqa: S603
        try:
            assert proc.stdout is not None
            info = json.loads(proc.stdout.readline())
            ch = _channel(mode, info, td, 5.0, args.spin_us, args.spin_auto_us)
            lins = _linears(args.preset)
            rng = np.random.default_rng(0)
            bodies = [rng.integers(0, 1 << 32, size=rows * n, dtype=np.uint64).astype("<u4").tobytes()
                      for _name, _m, n in lins]
            outs = [bytearray(rows * m * BPE) for _name, m, _n in lins]
            stop = threading.Event()

            def gil_load() -> None:
                while not stop.is_set():
                    end = time.perf_counter() + 1e-3
                    while time.perf_counter() < end:
                        sum(range(200))
                    time.sleep(1e-3)

            loaders = [threading.Thread(target=gil_load, daemon=True) for _ in range(args.gil_load)]
            for t in loaders:
                t.start()

            gap = args.gap_us * 1e-6

            ovh: list[float] = []  # trip minus the service's compute_us (transport + waits)

            def run(n: int, lat: list[float] | None) -> None:
                for i in range(n):
                    if gap:  # the gateway's work between two trips (busy, holding the GIL)
                        end = time.perf_counter() + gap
                        while time.perf_counter() < end:
                            pass
                    li = i % len(lins)
                    rid = secrets.token_bytes(16)
                    t0 = time.perf_counter()
                    st, cu = ch.trip(0, li, rows, rid, 10_000, bodies[li], outs[li], 5.0)
                    if lat is not None:
                        lat.append((time.perf_counter() - t0) * 1e6)
                        ovh.append(lat[-1] - cu)
                    if st != 0:
                        raise SystemExit(f"lane refused a trip (status {st})")

            run(300, None)
            lat: list[float] = []
            (c0, cf0), (s0, sf0) = _proc_stat(os.getpid()), _proc_stat(proc.pid)
            t0 = time.perf_counter()
            run(args.calls, lat)
            wall = time.perf_counter() - t0
            (c1, cf1), (s1, sf1) = _proc_stat(os.getpid()), _proc_stat(proc.pid)
            cst = ch.stats() if hasattr(ch, "stats") else {}
            stop.set()
            for t in loaders:
                t.join()
            ch.close()
        finally:
            if proc.stdin is not None:
                proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    lat.sort()
    if cst:  # the channel's counters cover warm-up and timed trips
        n = max(1, cst["trips"])
        cst = {"blocked_frac": round(cst["blocked"] / n, 3), "wait_us": round(cst["wait_us"] / n, 1),
               "svc_queue_us": round(cst["queue_us"] / n, 1), "svc_compute_us": round(cst["compute_us"] / n, 1),
               "spin_budget_us": round(cst["spin_us"] / n, 1)}

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1)

    return {"mode": mode, "preset": args.preset, "rows": rows, "calls": len(lat), "mtls": bool(args.mtls),
            "spin_us": args.spin_us, "read_spin_us": args.read_spin_us, "gap_us": args.gap_us,
            "gil_load": args.gil_load, "p50_us": pct(0.5), "p90_us": pct(0.9),
            "p99_us": pct(0.99), "mean_us": round(statistics.fmean(lat), 1),
            "wall_us_per_call": round(wall / len(lat) * 1e6, 1),
            "client_cpu_us_per_call": round((c1 - c0 + cf1 - cf0) / len(lat) * 1e6, 1),
            "server_cpu_us_per_call": round((s1 - s0 + sf1 - sf0) / len(lat) * 1e6, 1),
            "stub_us": args.stub_us, "zero_copy": bool(args.zero_copy), "spin_auto_us": args.spin_auto_us,
            "channel": cst, "overhead_p50_us": round(statistics.median(ovh), 1),
            "overhead_mean_us": round(statistics.fmean(ovh), 1)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", default="native-tls,native-uds,py-uds")
    ap.add_argument("--preset", default="0.5b", choices=sorted(SHAPES))
    ap.add_argument("--rows", default="1")
    ap.add_argument("--calls", type=int, default=3000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--spin-us", type=int, default=50)
    ap.add_argument("--gil-load", type=int, default=0)
    ap.add_argument("--gap-us", type=float, default=0.0, help="client busy-waits this long between trips")
    ap.add_argument("--read-spin-us", type=int, default=0, help="server: poll the socket this long before blocking")
    ap.add_argument("--mtls", action="store_true", help="native-tls: present a client certificate bound to the ticket")
    ap.add_argument("--stub-us", default="", help="native stub delays per linear, us (e.g. 44,40,191,79)")
    ap.add_argument("--zero-copy", action="store_true", help="server: zero-copy replies")
    ap.add_argument("--spin-auto-us", type=int, default=0, help="client: adaptive spin cap (0 = fixed --spin-us)")
    ap.add_argument("--serve-shm", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--serve", choices=["tcp", "uds"], help=argparse.SUPPRESS)
    ap.add_argument("--serve-dir", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.serve:
        _serve(args)
        return
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows_list = [int(r) for r in args.rows.split(",") if r.strip()]
    runs = []
    for rows in rows_list:
        for _ in range(args.repeat):
            for mode in modes:  # interleaved: drifts of the shared host hit every mode alike
                r = _one_run(args, mode, rows)
                runs.append(r)
                print(json.dumps(r), flush=True)
    summary: dict[str, dict[str, Any]] = {}
    for rows in rows_list:
        for mode in modes:
            sel = [r for r in runs if r["rows"] == rows and r["mode"] == mode]
            summary.setdefault(str(rows), {})[mode] = {
                "p50_us": round(statistics.median(r["p50_us"] for r in sel), 1),
                "mean_us": round(statistics.median(r["mean_us"] for r in sel), 1),
                "p99_us": round(statistics.median(r["p99_us"] for r in sel), 1),
                "client_cpu_us": round(statistics.median(r["client_cpu_us_per_call"] for r in sel), 1),
                "server_cpu_us": round(statistics.median(r["server_cpu_us_per_call"] for r in sel), 1)}
    print(json.dumps({"summary_median": summary, "repeat": args.repeat, "preset": args.preset,
                      "gil_load": args.gil_load, "spin_us": args.spin_us}))


if __name__ == "__main__":
    main()
