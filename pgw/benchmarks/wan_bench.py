#!/usr/bin/env python3
"""WAN ground truth: the trip lane and the client library through an emulated wide-area path (``wan_emu.py``).

The client runs in one network namespace and the provider in another; every packet between them crosses
``pgw-wanemu`` (``rust/tools/wanemu``: bottleneck rate, propagation delay, jitter, loss), so both kernels run real
TCP end to end. Needs root (namespaces, TUN), ``ip`` and ``cargo`` (the emulator is built on first use). Two
benches, each a sweep over RTTs:

``lane`` (transport only, no model): the real native TLS listener (``pgw_fast.LaneService``, ``rust/pgw-lane``)
with its benchmark executor (answers at once from its own buffer: no compute) serving the request / reply sizes of
every linear at 0.5B / 7B / 32B (and the masked LM head), driven by the gateway's native ``TripChannel``; next to
it a bare TCP request/reply of the same sizes (``raw``). Per RTT and payload: trip p50 / p90 / p99 and
``excess = p50 - RTT``; ``lane_over_raw = lane p50 - raw p50`` is what the lane stack (TLS records, frames, both
sides' copies and wake-ups) adds over the kernel's transport of the same bytes, independent of the emulator. Also
``--idle-ms``: the first 7B gate_up trip after an idle gap (TCP's restart after idle), and ``--loss``.

    python benchmarks/wan_bench.py lane --rtts 0,1,2,5,20,50 [--rate-mbit 0] [--idle-ms 0,100,300,1000,3000]

``sdk`` (end to end): the SDK tests' provider (``tests/sdk/provider_main.py``: TLS, TCP+TLS lane, masked LM
head, real accounts and enrollment; CPU kernels) in the provider namespace, and ``pgw.OpenAI`` in the client
namespace running ``sdk_trip_profile.py``'s measurement (warm greedy streamed chats, the runtime's own counters):
decode tokens/s, TTFT, trips per token, ms per trip, the lane floor; plus the cold start (key -> enrolled,
provisioned, connected, first masks). ``excess_ms_per_trip = (ms/trip at RTT) - (ms/trip at RTT 0) - RTT``: what
a trip over the WAN costs beyond one network RTT (the provider's CPU compute cancels out).

    python benchmarks/wan_bench.py sdk --rtts 0,1,2,5,20,50 --tokens 16 --runs 2 [--pack DIR]

``--rate-mbit`` / ``--up-mbit`` add a bottleneck (a consumer access link); ``--cc`` / ``--ssai`` set the
namespaces' congestion control (default cubic) and ``tcp_slow_start_after_idle`` (default 1, every OS's default).
Output: one JSON document on stdout (progress on stderr). The CPU box's numbers are relative for the compute part;
the network part is the point.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(HERE)]

from wan_emu import SERVER_IP, WanLink  # noqa: E402

RING = "K32A8"
#: (out_features m, in_features n) per linear, 1 row; the masked head: vocab x hidden
SHAPES: dict[str, dict[str, tuple[int, int]]] = {
    "0.5b": {"qkv": ((14 + 2 * 2) * 64, 896), "o": (896, 896), "gate_up": (2 * 4864, 896), "down": (896, 4864),
             "head": (151936, 896)},
    "7b": {"qkv": ((28 + 2 * 4) * 128, 3584), "o": (3584, 3584), "gate_up": (2 * 18944, 3584), "down": (3584, 18944),
           "head": (152064, 3584)},
    "32b": {"qkv": ((40 + 2 * 8) * 128, 5120), "o": (5120, 5120), "gate_up": (2 * 27648, 5120),
            "down": (5120, 27648), "head": (152064, 5120)},
}
LAYERS = {"0.5b": 24, "7b": 28, "32b": 64}
LANE_PORT, RAW_PORT = 8461, 8462


def log(*a: Any) -> None:
    print(*a, file=sys.stderr, flush=True)


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 3) if s else 0.0


def _specs() -> list[tuple[str, str, int, int]]:
    """(preset, linear, m, n) in the stub model's linear order."""
    return [(p, k, m, n) for p, d in SHAPES.items() for k, (m, n) in d.items()]


# ---- lane: server side (provider namespace) ----------------------------------------------------------------
def _recv_exact(c: socket.socket, n: int, buf: bytearray) -> bool:
    mv = memoryview(buf)
    got = 0
    while got < n:
        k = c.recv_into(mv[got:n])
        if k == 0:
            return False
        got += k
    return True


def _raw_server(port: int) -> None:
    """Bare TCP request/reply: 8-byte header (request bytes, reply bytes), then the request; answers the reply."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((SERVER_IP, port))
    srv.listen(16)
    zero = bytes(64 << 20)

    def serve(c: socket.socket) -> None:
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        buf = bytearray(64 << 20)
        try:
            while True:
                if not _recv_exact(c, 8, buf):
                    return
                n, m = int.from_bytes(buf[:4], "little"), int.from_bytes(buf[4:8], "little")
                if not _recv_exact(c, n, buf):
                    return
                c.sendall(memoryview(zero)[:m])
        except OSError:
            return

    while True:
        c, _ = srv.accept()
        threading.Thread(target=serve, args=(c,), daemon=True).start()


def lane_serve(a: argparse.Namespace) -> int:
    from lane_nic_bench import _tls_material

    from pgw.core.native import fast

    if fast is None or not hasattr(fast, "LaneService"):
        raise SystemExit("pgw_fast with the lane is not installed")
    tls = _tls_material(SERVER_IP)
    stub = fast.LaneService("", tls_addr=f"{SERVER_IP}:{LANE_PORT}", max_rows=8, tls_cert_pem=tls["cert"],
                            tls_key_pem=tls["key"], max_connections=256, executor=lambda *x: b"",
                            bench_stub_us=[0] * len(_specs()))
    stub.set_model(0, [(n, m, "gate_up" if k == "head" else k) for _, k, m, n in _specs()], [(RING, 4)])
    tickets = []
    for _ in range(a.tickets):
        t = secrets.token_bytes(32)
        stub.add_ticket(t, "bench", 0, 86400.0)
        tickets.append(t.hex())
    threading.Thread(target=_raw_server, args=(RAW_PORT,), daemon=True).start()
    Path(a.info).write_text(json.dumps({"ca_pem": tls["ca"].decode(), "tickets": tickets, "port": int(stub.port)}))
    os.chmod(a.info, 0o600)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    stop.wait()
    stub.stop(1.0)
    return 0


# ---- lane: client side (client namespace) ------------------------------------------------------------------
class _Raw:
    def __init__(self) -> None:
        self.s = socket.create_connection((SERVER_IP, RAW_PORT), timeout=10)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = bytearray(64 << 20)

    def trip(self, body: bytes, m: int) -> None:
        self.s.sendall(len(body).to_bytes(4, "little") + m.to_bytes(4, "little") + body)
        if not _recv_exact(self.s, m, self.buf):
            raise EOFError


def _tcp_info(s: socket.socket) -> dict[str, int]:
    """snd_cwnd and rtt (us) of a connected socket (Linux ``struct tcp_info``)."""
    b = s.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 256)
    import struct

    # u8 x8, then u32 fields: rto ato snd_mss rcv_mss unacked sacked lost retrans fackets last_data_sent
    # last_ack_sent last_data_recv last_ack_recv pmtu rcv_ssthresh rtt rttvar snd_ssthresh snd_cwnd ...
    f = struct.unpack_from("<8B24I", b)
    return {"rtt_us": f[8 + 15], "snd_cwnd": f[8 + 18], "retrans_total": f[8 + 23] if len(f) > 31 else 0}


def lane_client(a: argparse.Namespace) -> int:
    from pgw.core.native import fast

    info = json.loads(Path(a.info).read_text())
    tk = iter(info["tickets"][a.ticket_offset:])
    ca = info["ca_pem"].encode()

    def connect() -> Any:
        return fast.TripChannel.connect_service_tls(SERVER_IP, int(info["port"]), SERVER_IP, ca,
                                                    bytes.fromhex(next(tk)), 30.0, 64, 50)

    # connection setup: TCP connect, TLS 1.3 handshake + lane open, each timed
    t0 = time.perf_counter()
    s = socket.create_connection((SERVER_IP, RAW_PORT), timeout=10)
    tcp_ms = (time.perf_counter() - t0) * 1e3
    s.close()
    t0 = time.perf_counter()
    ch = connect()
    lane_open_ms = (time.perf_counter() - t0) * 1e3
    raw = _Raw()
    specs = _specs()
    presets = a.presets.split(",")
    rows_list = [int(r) for r in a.rows.split(",")]
    out: dict[str, Any] = {"tcp_connect_ms": round(tcp_ms, 3), "lane_connect_ms": round(lane_open_ms, 3),
                           "payloads": []}
    idx = {(p, k): i for i, (p, k, _, _) in enumerate(specs)}
    for p in presets:
        for rows in rows_list:
            for k, (m, n) in SHAPES[p].items():
                if k == "head" and rows > 4:
                    continue
                li = idx[(p, k)]
                body = os.urandom(rows * n * 4)
                buf = bytearray(rows * m * 4)
                lt, rt = [], []
                iters = a.iters if rows * (m + n) * 4 < 4 << 20 else max(5, a.iters // 8)
                for it in range(iters + 3):  # interleaved, so both see the same link state
                    t1 = time.perf_counter()
                    st, _cu = ch.trip(0, li, rows, secrets.token_bytes(16), 30_000, body, buf, 30.0)
                    t2 = time.perf_counter()
                    if st != 0:
                        raise RuntimeError(f"lane status {st}")
                    raw.trip(body, rows * m * 4)
                    t3 = time.perf_counter()
                    if it >= 3:
                        lt.append((t2 - t1) * 1e3)
                        rt.append((t3 - t2) * 1e3)
                r = {"preset": p, "linear": k, "rows": rows, "up_bytes": rows * n * 4, "down_bytes": rows * m * 4,
                     "lane_p50_ms": _pct(lt, 0.5), "lane_p90_ms": _pct(lt, 0.9), "lane_p99_ms": _pct(lt, 0.99),
                     "raw_p50_ms": _pct(rt, 0.5), "raw_p99_ms": _pct(rt, 0.99)}
                r["lane_over_raw_ms"] = round(r["lane_p50_ms"] - r["raw_p50_ms"], 3)
                out["payloads"].append(r)
                log(json.dumps(r))
    # one decode pass per preset: every trip of a token in order (layers x 4 + head), 1 row
    out["pass"] = []
    for p in presets:
        seq = [k for _ in range(LAYERS[p]) for k in ("qkv", "o", "gate_up", "down")] + ["head"]
        bodies = {k: os.urandom(n * 4) for k, (m, n) in SHAPES[p].items()}
        bufs = {k: bytearray(m * 4) for k, (m, n) in SHAPES[p].items()}
        ts = []
        for _ in range(a.passes):
            t1 = time.perf_counter()
            for k in seq:
                ch.trip(0, idx[(p, k)], 1, secrets.token_bytes(16), 30_000, bodies[k], bufs[k], 30.0)
            ts.append((time.perf_counter() - t1) * 1e3)
        r = {"preset": p, "trips": len(seq), "pass_p50_ms": _pct(ts, 0.5),
             "ms_per_trip": round(_pct(ts, 0.5) / len(seq), 4),
             "bytes_per_token": sum(SHAPES[p][k][0] * 4 + SHAPES[p][k][1] * 4 for k in seq)}
        out["pass"].append(r)
        log(json.dumps(r))
    # TCP restart after idle: the first 7B gate_up reply (148 KB) after a gap, on a warm connection
    out["idle"] = []
    for gap in [float(x) for x in a.idle_ms.split(",") if x]:
        m, n = SHAPES["7b"]["gate_up"]
        body, buf = os.urandom(n * 4), bytearray(m * 4)
        lt, rt, cw = [], [], []
        for _ in range(a.idle_reps):
            for _w in range(20):  # warm the window with back-to-back trips first
                ch.trip(0, idx[("7b", "gate_up")], 1, secrets.token_bytes(16), 30_000, body, buf, 30.0)
                raw.trip(body, m * 4)
            time.sleep(gap / 1e3)
            t1 = time.perf_counter()
            ch.trip(0, idx[("7b", "gate_up")], 1, secrets.token_bytes(16), 30_000, body, buf, 30.0)
            lt.append((time.perf_counter() - t1) * 1e3)
            time.sleep(gap / 1e3)
            cw.append(_tcp_info(raw.s)["snd_cwnd"])
            t1 = time.perf_counter()
            raw.trip(body, m * 4)
            rt.append((time.perf_counter() - t1) * 1e3)
        r = {"gap_ms": gap, "lane_first_trip_p50_ms": _pct(lt, 0.5), "raw_first_trip_p50_ms": _pct(rt, 0.5),
             "client_cwnd_before": int(statistics.median(cw))}
        out["idle"].append(r)
        log(json.dumps(r))
    out["raw_tcp_info"] = _tcp_info(raw.s)
    ch.close()
    print(json.dumps(out))
    return 0


def lane_bench(a: argparse.Namespace) -> dict[str, Any]:
    res: dict[str, Any] = {"bench": "lane", "cc": a.cc, "ssai": a.ssai, "server_cc": a.server_cc,
                           "server_ssai": a.server_ssai, "rate_mbit": a.rate_mbit,
                           "up_mbit": a.up_mbit, "runs": []}
    with tempfile.TemporaryDirectory(prefix="pgw-wan-") as d, WanLink(cc=a.cc, ssai=a.ssai, server_cc=a.server_cc,
                                                                   server_ssai=a.server_ssai) as w:
        info = Path(d) / "info.json"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT), str(HERE)])}
        srv = subprocess.Popen(w.server_cmd([sys.executable, __file__, "lane-serve", "--info", str(info),  # noqa: S603
                                             "--tickets", str(4 * len(a.rtts.split(",")) * len(a.loss.split(",")))]),
                               env=env)
        try:
            for _ in range(600):
                if info.exists() or srv.poll() is not None:
                    break
                time.sleep(0.1)
            if not info.exists():
                raise SystemExit("lane listener did not start")
            used = 0
            for loss in [float(x) for x in a.loss.split(",")]:
                for rtt in [float(x) for x in a.rtts.split(",")]:
                    w.set(rtt_ms=rtt, jitter_ms=a.jitter_ms, loss=loss, rate_mbit=a.rate_mbit, up_mbit=a.up_mbit,
                          queue_kb=a.queue_kb)
                    w.reset()
                    log(f"-- RTT {rtt} ms, loss {loss}")
                    r = subprocess.run(w.client_cmd([sys.executable, __file__, "lane-client", "--info", str(info),  # noqa: S603
                                                     "--ticket-offset", str(used), "--presets", a.presets,
                                                     "--rows", a.rows, "--iters", str(a.iters),
                                                     "--passes", str(a.passes),
                                                     "--idle-ms", a.idle_ms if loss == 0 else "",
                                                     "--idle-reps", str(a.idle_reps)]),
                                       env=env, capture_output=True, text=True, timeout=7200)
                    used += 2
                    if r.returncode != 0:
                        raise SystemExit(f"lane client failed:\n{r.stderr[-4000:]}")
                    one = json.loads(r.stdout.strip().splitlines()[-1])
                    one.update({"rtt_ms": rtt, "loss": loss, "emulator": w.stats()})
                    for p in one["payloads"]:
                        p["lane_excess_ms"] = round(p["lane_p50_ms"] - rtt, 3)
                        p["raw_excess_ms"] = round(p["raw_p50_ms"] - rtt, 3)
                    for p in one["pass"]:
                        p["excess_ms_per_trip"] = round(p["ms_per_trip"] - rtt, 4)
                    res["runs"].append(one)
        finally:
            srv.terminate()
            srv.wait(timeout=10)
    return res


# ---- sdk: provider side (provider namespace) ---------------------------------------------------------------
def sdk_serve(a: argparse.Namespace) -> int:
    from tests.sdk.conftest import free_port, make_pki, start_provider, stop_provider

    root = Path(a.root)
    for sub in ("pki", "provider"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    port = free_port()
    extra = {"PGW_SERVICE_HOST": "0.0.0.0",  # noqa: S104 - the provider namespace only
             "PGW_SERVICE_PUBLIC_URL": f"https://{SERVER_IP}:{port}"}
    extra.update(dict(kv.split("=", 1) for kv in a.provider_env))
    prov = start_provider(root / "provider", Path(a.pack), make_pki(root / "pki", ips=("127.0.0.1", SERVER_IP)),
                          extra_env=extra, port=port)
    try:
        key = prov.new_key()
        Path(a.info).write_text(json.dumps({"key": key, "url": f"https://{SERVER_IP}:{port}"}))
        os.chmod(a.info, 0o600)
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        stop.wait()
    finally:
        stop_provider(prov)
    return 0


def sdk_client(a: argparse.Namespace) -> int:
    import sdk_trip_profile as tp

    from pgw import OpenAI

    tp.PROMPT = tp.PROMPTS[a.prompt]
    info = json.loads(Path(a.info).read_text())
    t_warm: list[float] = []
    orig = OpenAI.warmup

    def warmup(self: Any, *x: Any, **k: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig(self, *x, **k)
        finally:
            t_warm.append(time.perf_counter() - t0)

    OpenAI.warmup = warmup  # type: ignore[method-assign]
    ns = argparse.Namespace(tokens=a.tokens, runs=a.runs, opt=a.opt)
    out = tp.run(ns, info["key"], Path(a.pack), Path(a.root))
    lp = out.pop("logprobs")
    out["logprobs_sha"] = __import__("hashlib").sha256(json.dumps(lp).encode()).hexdigest()[:16]
    out["cold_setup_s"] = round(t_warm[0], 3) if t_warm else None
    out.pop("runs", None)
    print(json.dumps(out, default=str))
    return 0


def _pack_05b(out: Path) -> Path:
    import sdk_trip_profile as tp

    return Path(tp._pack(out))


def sdk_bench(a: argparse.Namespace) -> dict[str, Any]:
    res: dict[str, Any] = {"bench": "sdk", "cc": a.cc, "ssai": a.ssai, "server_cc": a.server_cc,
                           "server_ssai": a.server_ssai, "rate_mbit": a.rate_mbit,
                           "up_mbit": a.up_mbit, "tokens": a.tokens, "prompt": a.prompt, "opt": a.opt, "runs": []}
    with tempfile.TemporaryDirectory(prefix="pgw-wan-sdk-") as d, WanLink(cc=a.cc, ssai=a.ssai, server_cc=a.server_cc,
                                                                   server_ssai=a.server_ssai) as w:
        root = Path(d)
        pack = Path(a.pack) if a.pack else _pack_05b(root / "pack" / "pack")
        info = root / "info.json"
        # no PGW_ settings and no HTTP(S) proxy: the namespaces reach only each other (an inherited proxy would
        # take the client's HTTPS off the emulated path)
        env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_") and "proxy" not in k.lower()}
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), str(HERE)])
        env.setdefault("HF_HUB_OFFLINE", "1")
        srv = subprocess.Popen(w.server_cmd([sys.executable, __file__, "sdk-serve", "--info", str(info),  # noqa: S603
                                             "--root", str(root / "srv"), "--pack", str(pack),
                                             *[x for kv in a.provider_env for x in ("--provider-env", kv)]]), env=env)
        try:
            for _ in range(3000):
                if info.exists() or srv.poll() is not None:
                    break
                time.sleep(0.1)
            if not info.exists():
                raise SystemExit("provider did not start")
            ref = None
            for i, rtt in enumerate([float(x) for x in a.rtts.split(",")]):
                w.set(rtt_ms=rtt, jitter_ms=a.jitter_ms, loss=a.loss, rate_mbit=a.rate_mbit, up_mbit=a.up_mbit,
                      queue_kb=a.queue_kb)
                w.reset()
                log(f"-- RTT {rtt} ms")
                croot = root / f"client{i}"
                croot.mkdir()
                r = subprocess.run(w.client_cmd([sys.executable, __file__, "sdk-client", "--info", str(info),  # noqa: S603
                                                 "--root", str(croot), "--pack", str(pack), "--tokens", str(a.tokens),
                                                 "--runs", str(a.runs), "--prompt", a.prompt,
                                                 *[x for o in a.opt for x in ("--opt", o)]]),
                                   env=env, capture_output=True, text=True, timeout=14400)
                if r.returncode != 0:
                    raise SystemExit(f"sdk client failed at RTT {rtt}:\n{r.stderr[-6000:]}")
                one = json.loads(r.stdout.strip().splitlines()[-1])
                s = one["summary"]
                ref = one["logprobs_sha"] if ref is None else ref
                row = {"rtt_ms": rtt, "bit_identical": one["logprobs_sha"] == ref, "cold_setup_s": one["cold_setup_s"],
                       "lane": one.get("lane"), "emulator": w.stats(), "summary": s}
                for k in ("decode_tok_s", "ttft_ms", "trips_per_token", "critical_trips_per_token", "ms_per_trip_wall",
                          "lane_trip_us", "provider_compute_us", "lane_floor_ms", "client_cpu_ms_per_token"):
                    row[k] = s.get(k)
                res["runs"].append(row)
                log(json.dumps({k: v for k, v in row.items() if k not in ("summary", "emulator")}))
                shutil_rm(croot)
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                srv.kill()
    base = next((r for r in res["runs"] if r["rtt_ms"] == 0), None)
    if base is not None:
        for r in res["runs"]:
            r["excess_ms_per_trip"] = round(r["ms_per_trip_wall"] - base["ms_per_trip_wall"] - r["rtt_ms"], 3)
            r["ttft_excess_ms"] = round(r["ttft_ms"] - base["ttft_ms"], 1)
    return res


# ---- quic: QUIC (quinn) vs TCP for the lane's trip pattern (rust/tools/quicbench) ----------------------------
def quicbench_binary() -> str:
    """``$PGW_QUICBENCH_BIN``, else a release build of ``rust/tools/quicbench`` (fetches quinn from crates.io)."""
    env = os.environ.get("PGW_QUICBENCH_BIN")
    if env:
        return env
    tgt = Path(os.environ.get("PGW_QUICBENCH_TARGET", str(Path(tempfile.gettempdir()) / "pgw-quicbench-target")))
    exe = tgt / "release" / "pgw-quicbench"
    if not exe.exists():
        subprocess.run(["cargo", "build", "--release"], cwd=ROOT / "rust" / "tools" / "quicbench",  # noqa: S607
                       env={**os.environ, "CARGO_TARGET_DIR": str(tgt)}, check=True)
    return str(exe)


def quic_bench(a: argparse.Namespace) -> dict[str, Any]:
    """Per RTT and loss: the same 7B trip sizes and decode pass over TCP (the client's and provider's kernel
    congestion control, cubic) and over QUIC with each ``--quic-cc`` (quinn's controller on both ends)."""
    from lane_nic_bench import _tls_material

    exe = quicbench_binary()
    shapes = SHAPES[a.preset]
    sizes = ",".join(f"{n * 4}:{m * 4}" for k, (m, n) in shapes.items() if k != "head")
    seq = [k for _ in range(LAYERS[a.preset]) for k in ("qkv", "o", "gate_up", "down")] + ["head"]
    pas = ",".join(f"{shapes[k][1] * 4}:{shapes[k][0] * 4}" for k in seq)
    res: dict[str, Any] = {"bench": "quic", "preset": a.preset, "cc": a.cc, "server_cc": a.server_cc,
                           "server_ssai": a.server_ssai, "runs": []}
    variants = [("tcp", a.cc)] + [("quic", c) for c in a.quic_cc.split(",") if c]
    with tempfile.TemporaryDirectory(prefix="pgw-wan-quic-") as d, WanLink(
            cc=a.cc, ssai=a.ssai, server_cc=a.server_cc, server_ssai=a.server_ssai) as w:
        tls = _tls_material(SERVER_IP)
        for k in ("ca", "cert", "key"):
            (Path(d) / f"{k}.pem").write_bytes(tls[k])
        port = 9600
        for loss in [float(x) for x in a.loss.split(",")]:
            for rtt in [float(x) for x in a.rtts.split(",")]:
                w.set(rtt_ms=rtt, jitter_ms=a.jitter_ms, loss=loss, rate_mbit=a.rate_mbit, up_mbit=a.up_mbit,
                      queue_kb=a.queue_kb)
                for tr, cc in variants:
                    port += 1
                    common = ["--transport", tr, "--addr", f"{SERVER_IP}:{port}", "--cc", cc]
                    if a.init_window and tr == "quic":
                        common += ["--init-window", str(a.init_window)]
                    srv = subprocess.Popen(w.server_cmd([exe, "serve", *common, "--cert", f"{d}/cert.pem",  # noqa: S603
                                                         "--key", f"{d}/key.pem"]), stdout=subprocess.PIPE, text=True)
                    try:
                        assert srv.stdout is not None
                        if srv.stdout.readline().strip() != "ready":
                            raise SystemExit("quicbench server did not start")
                        log(f"-- RTT {rtt} ms, loss {loss}, {tr} {cc}")
                        r = subprocess.run(w.client_cmd([exe, "client", *common, "--ca", f"{d}/ca.pem",  # noqa: S603
                                                         "--name", SERVER_IP, "--sizes", sizes, "--iters",
                                                         str(a.iters), "--pass", pas, "--passes", str(a.passes)]),
                                           capture_output=True, text=True, timeout=7200)
                    finally:
                        srv.kill()
                        srv.wait()
                    if r.returncode != 0:
                        raise SystemExit(f"quicbench client failed:\n{r.stderr[-3000:]}")
                    one = json.loads(r.stdout.strip().splitlines()[-1])
                    one.update({"rtt_ms": rtt, "loss": loss, "variant": f"{tr}-{cc}",
                                "ms_per_trip": round(one["pass_p50_ms"] / max(1, one["pass_trips"]), 3)})
                    one["excess_ms_per_trip"] = round(one["ms_per_trip"] - rtt, 3)
                    res["runs"].append(one)
                    log(json.dumps({k: v for k, v in one.items() if k != "sizes"}))
    return res


def shutil_rm(p: Path) -> None:
    import shutil

    shutil.rmtree(p, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("lane", "sdk"):
        s = sub.add_parser(name)
        s.add_argument("--rtts", default="0,1,2,5,20,50")
        s.add_argument("--jitter-ms", type=float, default=0.0)
        s.add_argument("--rate-mbit", type=float, default=0.0, help="download bottleneck (0 = none)")
        s.add_argument("--up-mbit", type=float, default=None, help="upload bottleneck (default: --rate-mbit)")
        s.add_argument("--queue-kb", type=float, default=4096.0)
        s.add_argument("--cc", default="cubic")
        s.add_argument("--ssai", type=int, default=1)
        s.add_argument("--server-cc", default=None, help="the provider namespace's congestion control (default --cc)")
        s.add_argument("--server-ssai", type=int, default=None, help="the provider's slow start after idle")
        s.add_argument("--json", default="")
    s = sub.choices["lane"]
    s.add_argument("--presets", default="0.5b,7b,32b")
    s.add_argument("--rows", default="1")
    s.add_argument("--iters", type=int, default=60)
    s.add_argument("--passes", type=int, default=3)
    s.add_argument("--loss", default="0")
    s.add_argument("--idle-ms", default="0,100,300,1000,3000")
    s.add_argument("--idle-reps", type=int, default=5)
    s = sub.choices["sdk"]
    s.add_argument("--loss", type=float, default=0.0)
    s.add_argument("--tokens", type=int, default=16)
    s.add_argument("--runs", type=int, default=2)
    s.add_argument("--prompt", default="chat")
    s.add_argument("--pack", default="")
    s.add_argument("--opt", action="append", default=[])
    s.add_argument("--provider-env", action="append", default=[])
    s = sub.add_parser("quic")
    for k, v in (("--rtts", "5,20"), ("--loss", "0,0.001,0.01"), ("--cc", "cubic"), ("--quic-cc", "cubic,bbr"),
                 ("--preset", "7b")):
        s.add_argument(k, default=v)
    s.add_argument("--ssai", type=int, default=1)
    s.add_argument("--server-cc", default=None)
    s.add_argument("--server-ssai", type=int, default=None)
    s.add_argument("--jitter-ms", type=float, default=0.0)
    s.add_argument("--rate-mbit", type=float, default=0.0)
    s.add_argument("--up-mbit", type=float, default=None)
    s.add_argument("--queue-kb", type=float, default=4096.0)
    s.add_argument("--iters", type=int, default=40)
    s.add_argument("--passes", type=int, default=5)
    s.add_argument("--init-window", type=int, default=0, help="QUIC initial window in bytes (0 = quinn's)")
    s.add_argument("--json", default="")
    s = sub.add_parser("lane-serve")
    s.add_argument("--info", required=True)
    s.add_argument("--tickets", type=int, default=64)
    s = sub.add_parser("lane-client")
    s.add_argument("--info", required=True)
    s.add_argument("--ticket-offset", type=int, default=0)
    s.add_argument("--presets", default="0.5b,7b,32b")
    s.add_argument("--rows", default="1")
    s.add_argument("--iters", type=int, default=60)
    s.add_argument("--passes", type=int, default=3)
    s.add_argument("--idle-ms", default="")
    s.add_argument("--idle-reps", type=int, default=5)
    s = sub.add_parser("sdk-serve")
    s.add_argument("--info", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--pack", required=True)
    s.add_argument("--provider-env", action="append", default=[])
    s = sub.add_parser("sdk-client")
    s.add_argument("--info", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--pack", required=True)
    s.add_argument("--tokens", type=int, default=16)
    s.add_argument("--runs", type=int, default=2)
    s.add_argument("--prompt", default="chat")
    s.add_argument("--opt", action="append", default=[])
    a = ap.parse_args()
    if a.cmd == "lane-serve":
        return lane_serve(a)
    if a.cmd == "lane-client":
        return lane_client(a)
    if a.cmd == "sdk-serve":
        return sdk_serve(a)
    if a.cmd == "sdk-client":
        return sdk_client(a)
    res = lane_bench(a) if a.cmd == "lane" else sdk_bench(a) if a.cmd == "sdk" else quic_bench(a)
    txt = json.dumps(res, indent=1)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
