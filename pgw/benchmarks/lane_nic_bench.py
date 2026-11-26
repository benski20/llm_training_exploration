#!/usr/bin/env python3
"""Lane fixed cost per trip over a real NIC: a GPU provider pod serves the native trip lane (TCP + TLS 1.3, real
CUDA-graph replay at 7B / 32B / 72B layer widths, random int8 weights), a client pod in the same data center drives
it with the gateway's native ``TripChannel``. Two sides, one script:

``serve`` (GPU pod)::

    PYTHONPATH=src python benchmarks/lane_nic_bench.py serve --preset 72b --host "$RUNPOD_PUBLIC_IP" \\
        --port 8444 --stub-port 8445 --out /workspace/lane_nic.json [--read-spin-us 2000] [--zero-copy] [--coalesce]

  starts two listeners: the **GPU lane** (graphs captured and self-checked for rows ``--rows``, exactly what the
  service runs) and a **stub lane** (same TLS listener and frames, the native benchmark executor answering after 0
  us: transport only). Writes the CA, the single-use tickets and the table to ``--out`` (random weights; the
  tickets only open these throwaway listeners) and serves until ``--seconds`` or SIGTERM.

``client`` (client pod, same data center; copy the JSON over)::

    PYTHONPATH=src python benchmarks/lane_nic_bench.py client --info lane_nic.json --connect <ip> \\
        --port <public port of 8444> --stub-port <public port of 8445> --conc 1,4,16,64 --iters 400 --json out.json

  For each listener and concurrency ``c`` (``c`` connections, one thread each, cycling ``qkv, o, gate_up, down``
  with 1 row): trip p50/p90/p99, the service's ``compute_us`` (device lock + body copy + launch + wait + reply copy)
  and the aggregate trips/s. Plus the TCP connect time to the same port (one network RTT, p50 of 50).

Derived (the numbers the 72B model needs, ``benchmarks/round_trip_model.py``):

* ``transport_us = stub trip p50 - rtt p50``: TLS records, socket wake-ups, frames, both sides' copies;
* ``lane_fixed_us = gpu trip p50 - compute_us p50 - rtt p50``: everything a trip costs beyond the network and the
  device (the 0.13 ms estimate); ``device_us`` is ``compute_us`` (with the launch and copies);
* throughput at ``c`` connections (one GPU serializes lane calls per device; ``--coalesce`` merges concurrent
  calls of the same linear into one launch, each reply cut back to its own rows).

CPU smoke test (no GPU: the Python executor serves; numbers meaningless)::

    PYTHONPATH=src python benchmarks/lane_nic_bench.py serve --cpu --preset tiny --host 127.0.0.1 --port 0 \\
        --stub-port 0 --out /tmp/l.json --seconds 60 &
    PYTHONPATH=src python benchmarks/lane_nic_bench.py client --info /tmp/l.json --conc 1,2 --iters 20
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

KIND_ORDER = ("qkv", "o", "gate_up", "down")


def _tls_material(host: str) -> dict[str, bytes]:
    """A throwaway CA and a server certificate for ``host`` (IP or DNS name): PEM bytes."""
    import datetime as dt
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = dt.datetime.now(dt.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "lane nic bench CA")])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name).public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now - dt.timedelta(minutes=5))
          .not_valid_after(now + dt.timedelta(days=2))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True,
                                       content_commitment=False, key_encipherment=False, data_encipherment=False,
                                       key_agreement=False, encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    key = ec.generate_private_key(ec.SECP256R1())
    try:
        san: list[x509.GeneralName] = [x509.IPAddress(ipaddress.ip_address(host))]
    except ValueError:
        san = [x509.DNSName(host)]
    cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)]))
            .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5)).not_valid_after(now + dt.timedelta(days=2))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    return {"ca": ca.public_bytes(serialization.Encoding.PEM),
            "cert": cert.public_bytes(serialization.Encoding.PEM),
            "key": key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                     serialization.NoEncryption())}


# ---- serve (GPU pod) -------------------------------------------------------------------------------------
def serve(a: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from provider_scale_bench import PRESETS, RING, _build, _Mgr

    from pgw.core.native import fast
    from pgw.core.packing import pack_ring
    from pgw.core.ring import RingParams
    from pgw.service.gpu_exec import device_executor
    from pgw.service.lane import capture_lane_graphs

    if fast is None or not hasattr(fast, "LaneService"):
        raise SystemExit("pgw_fast with the lane is not installed")
    ring = RingParams.from_name(RING)
    tls = _tls_material(a.host)
    t0 = time.perf_counter()
    model, kerns = _build(a.preset, 1, a.cpu)
    build_s = time.perf_counter() - t0

    def py_execute(_m: int, li: int, _r: int, rows: int, body: bytes, _t: str, _left: float) -> bytes:
        x = torch.from_numpy(np.frombuffer(body, dtype="<u4").astype(np.int64).reshape(rows, -1))
        return pack_ring(kerns[li].run_compact(x, ring), ring)

    common = {"max_rows": 8, "tls_cert_pem": tls["cert"], "tls_key_pem": tls["key"],
              "max_connections": max(64, 2 * a.tickets)}
    kw: dict[str, Any] = dict(common, tls_addr=f"{a.bind}:{a.port}", use_graphs=not a.cpu, executor=py_execute)
    if a.read_spin_us:
        kw["read_spin_us"] = int(a.read_spin_us)
    if a.zero_copy:
        kw["zero_copy"] = True
    if a.coalesce:
        kw["coalesce"] = True
    svc = fast.LaneService("", **kw)
    mgr = _Mgr(svc, model, 8)
    rep: dict[str, Any] = {}
    if not a.cpu:
        for d in {str(dv) for k in kerns for dv in getattr(k, "devices", [k.device])}:
            dev = torch.device(d)
            device_executor(dev).share_lock(svc.device_lock(int(dev.index or 0)))
        os.environ["PGW_SERVICE_LANE_OWN_LOCK"] = "0"
        t1 = time.perf_counter()
        r = capture_lane_graphs(mgr, rows_list=[int(x) for x in a.rows.split(",")])  # type: ignore[arg-type]
        rep = {"captured": r.captured, "checked": r.checked, "disabled": r.disabled,
               "s": round(time.perf_counter() - t1, 1)}
        if r.disabled:
            raise SystemExit(f"lane graphs disabled: {r.disabled}")
    stub_kw: dict[str, Any] = dict(common, tls_addr=f"{a.bind}:{a.stub_port}", executor=py_execute,
                                   bench_stub_us=[0] * len(model.specs))
    stub = fast.LaneService("", **stub_kw)
    stub.set_model(0, [(s.in_features, s.out_features, s.id.kind.value) for s in model.specs], [(RING, 4)])
    tickets, stub_tickets = [], []
    for svc_, lst in ((svc, tickets), (stub, stub_tickets)):
        for _ in range(a.tickets):
            t = secrets.token_bytes(32)
            svc_.add_ticket(t, "bench", 0, float(a.seconds))
            lst.append(t.hex())
    info = {"host": a.host, "port": int(svc.port), "stub_port": int(stub.port), "ca_pem": tls["ca"].decode(),
            "tickets": tickets, "stub_tickets": stub_tickets, "preset": a.preset,
            "shapes": {k: list(v) for k, v in PRESETS[a.preset].items()}, "ring": RING, "build_s": round(build_s, 1),
            "graphs": rep, "server": {"read_spin_us": a.read_spin_us, "zero_copy": a.zero_copy, "coalesce": a.coalesce},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and not a.cpu else "cpu"}
    tmp = Path(a.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(info))
    os.chmod(tmp, 0o600)
    tmp.replace(a.out)
    print(json.dumps({k: v for k, v in info.items() if "ticket" not in k and k != "ca_pem"}), flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait(a.seconds)
    stub.stop(2.0)
    svc.clear_graphs()
    svc.stop(2.0)
    mgr._graphs.clear()
    return 0


# ---- client (client pod) ---------------------------------------------------------------------------------
def _rtt_us(host: str, port: int, n: int = 50) -> float:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        with socket.create_connection((host, port), timeout=5):
            out.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(out), 1)


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1) if s else 0.0


def _drive(info: dict[str, Any], host: str, port: int, tickets: list[str], conc: int, iters: int,
           spin_auto_us: int) -> dict[str, Any]:
    from pgw.core.native import fast

    shapes = [info["shapes"][k] for k in KIND_ORDER]
    ca = info["ca_pem"].encode()
    chans = []
    for i in range(conc):
        ch = fast.TripChannel.connect_service_tls(host, port, info["host"], ca, bytes.fromhex(tickets[i]), 10.0, 8, 50)
        if spin_auto_us:
            ch.set_spin_auto(int(spin_auto_us))
        chans.append(ch)
    trip: list[list[float]] = [[] for _ in range(conc)]
    comp: list[list[float]] = [[] for _ in range(conc)]
    errors: list[str] = []
    barrier = threading.Barrier(conc + 1)

    def run(j: int) -> None:
        ch = chans[j]
        bodies = [os.urandom(n * 4) for _, n in shapes]
        outs = [bytearray(m * 4) for m, _ in shapes]
        barrier.wait()
        try:
            for it in range(iters + 20):
                for li in range(4):
                    t0 = time.perf_counter()
                    st, cu = ch.trip(0, li, 1, secrets.token_bytes(16), 10_000, bodies[li], outs[li], 10.0)
                    dt = (time.perf_counter() - t0) * 1e6
                    if st != 0:
                        raise RuntimeError(f"lane status {st}")
                    if it >= 20:
                        trip[j].append(dt)
                        comp[j].append(float(cu))
        except Exception as exc:  # report, keep the others going
            errors.append(f"{type(exc).__name__}: {exc}")

    ths = [threading.Thread(target=run, args=(j,), daemon=True) for j in range(conc)]
    for t in ths:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    for ch in chans:
        ch.close()
    alltrip = [x for xs in trip for x in xs]
    allcomp = [x for xs in comp for x in xs]
    return {"conc": conc, "trips": len(alltrip), "errors": errors[:3], "trip_p50_us": _pct(alltrip, 0.5),
            "trip_p90_us": _pct(alltrip, 0.9), "trip_p99_us": _pct(alltrip, 0.99),
            "compute_p50_us": _pct(allcomp, 0.5), "trips_per_s": round(len(alltrip) / wall, 1) if wall else 0.0}


def client(a: argparse.Namespace) -> int:
    info = json.loads(Path(a.info).read_text())
    host = a.connect or info["host"]
    port = a.port or int(info["port"])
    stub_port = a.stub_port or int(info["stub_port"])
    concs = [int(c) for c in a.conc.split(",")]
    if sum(concs) > min(len(info["tickets"]), len(info["stub_tickets"])):
        raise SystemExit(f"{sum(concs)} connections need as many tickets; serve --tickets more")
    res: dict[str, Any] = {"preset": info["preset"], "gpu": info.get("gpu"), "server": info.get("server"),
                           "host": host, "rtt_us": _rtt_us(host, port), "runs": []}
    used = {"gpu": 0, "stub": 0}
    for target, p, key in (("stub", stub_port, "stub_tickets"), ("gpu", port, "tickets")):
        for c in concs:
            tk = info[key][used[target]:used[target] + c]
            used[target] += c
            r = _drive(info, host, p, tk, c, a.iters, a.spin_auto_us)
            r["target"] = target
            res["runs"].append(r)
            print(json.dumps(r), flush=True)
    by = {(r["target"], r["conc"]): r for r in res["runs"]}
    if ("stub", 1) in by and ("gpu", 1) in by:
        rtt = res["rtt_us"]
        res["derived_1conn"] = {
            "rtt_us": rtt, "transport_us": round(by[("stub", 1)]["trip_p50_us"] - rtt, 1),
            "device_us": by[("gpu", 1)]["compute_p50_us"],
            "lane_fixed_us": round(by[("gpu", 1)]["trip_p50_us"] - by[("gpu", 1)]["compute_p50_us"] - rtt, 1)}
        print(json.dumps(res["derived_1conn"]), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--preset", default="7b")
    s.add_argument("--host", required=True, help="the address clients use (certificate SAN)")
    s.add_argument("--bind", default="0.0.0.0")  # noqa: S104 - the NIC under test; TLS + single-use tickets
    s.add_argument("--port", type=int, default=8444)
    s.add_argument("--stub-port", type=int, default=8445)
    s.add_argument("--rows", default="1,2,4,8")
    s.add_argument("--tickets", type=int, default=256)
    s.add_argument("--read-spin-us", type=int, default=0)
    s.add_argument("--zero-copy", action="store_true")
    s.add_argument("--coalesce", action="store_true")
    s.add_argument("--seconds", type=float, default=3600.0)
    s.add_argument("--out", required=True)
    s.add_argument("--cpu", action="store_true")
    c = sub.add_parser("client")
    c.add_argument("--info", required=True)
    c.add_argument("--connect", default="")
    c.add_argument("--port", type=int, default=0)
    c.add_argument("--stub-port", type=int, default=0)
    c.add_argument("--conc", default="1,4,16,64")
    c.add_argument("--iters", type=int, default=400)
    c.add_argument("--spin-auto-us", type=int, default=0)
    c.add_argument("--json", default="")
    a = ap.parse_args()
    return serve(a) if a.cmd == "serve" else client(a)


if __name__ == "__main__":
    sys.exit(main())
