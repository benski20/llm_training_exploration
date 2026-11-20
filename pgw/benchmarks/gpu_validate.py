#!/usr/bin/env python3
"""GPU validation and benchmark driver for one model (used by the Kaggle kernel, also runs on CPU).

Everything goes through the real ``pgw`` CLI and real processes, as an operator would run it:

1. ``pgw pack`` the HF checkpoint (int8 pack + SHA-256 fingerprint);
2. QuantPlain reference: an in-process :class:`pgw.engine.PlainBackend` engine (subprocess, so its
   memory is released) generates the answer greedily with logprobs;
3. ``pgw maskstore init`` + ``pgw factory generate --device <dev>`` into the encrypted store (timed);
   ``pgw bench factory --pack`` for tokens/hour;
4. ``pgw service serve`` on ``<dev>`` and ``pgw gateway serve`` (engine on CPU) on 127.0.0.1;
5. chat completions (non-streaming and streaming) compared token for token and logprob for logprob
   with the reference; wire bytes, round trips, gateway CPU time from ``/admin/status`` and /proc;
6. concurrent chat requests (gateway engine batching) for batched end-to-end throughput;
7. ``pgw bench service`` against the running service: 1 session (decode, batch 1) and N sessions.

Accounting (streaming request): ``decode_*`` fields cover only the decode steps. They come from the
backend's decode counters (``decode_calls``/``decode_remote_seconds``/``decode_local_seconds``) when the
gateway has them, else from the backend stats delta between the first streamed token and the end. The
older whole-request averages (prefill included, divided by prompt + generated tokens) are kept as
``*_incl_prefill``. ``trip_split`` breaks a round trip into the client RTT, the service-reported queue and
compute time, and the rest (``wire_ms`` = RTT - queue - compute, clamped at 0: network, framing, Python),
by rows per request; for multi-item requests the service reports compute as a sum and queue as a max.

Writes ``<out>/<label>.json``. Usage::

    python benchmarks/gpu_validate.py --model Qwen/Qwen2.5-0.5B-Instruct --label qwen0.5b \
        --device cuda --out results/
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROMPT = "Explain in three sentences why the sky is blue."
ADMIN = "bench-admin-token-0123456789abcdef"


def log(*a: Any) -> None:
    print(f"[gpu_validate {time.strftime('%H:%M:%S')}]", *a, flush=True)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = int(s.getsockname()[1])
    s.close()
    return p


def base_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PGW_LOG_LEVEL"] = "INFO"
    return env


def cli(*args: str, env: dict[str, str] | None = None, timeout: float = 7200, log_file: Path | None = None) -> str:
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-m", "pgw.cli", *args], env=env or base_env(),  # noqa: S603
                       capture_output=True, text=True, timeout=timeout)
    if log_file is not None:
        log_file.write_text(r.stdout + "\n---stderr---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"pgw {' '.join(args)} exited {r.returncode}:\n{r.stdout[-3000:]}\n{r.stderr[-5000:]}")
    log(f"pgw {' '.join(args[:3])} ... ok ({time.monotonic() - t0:.1f}s)")
    return r.stdout


def proc_cpu_seconds(pid: int) -> float:
    """utime+stime of a process and its threads, in seconds."""
    f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def gpu_mem_mb() -> list[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],  # noqa: S607
                             capture_output=True, text=True, timeout=30).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def cpu_info() -> dict[str, Any]:
    """CPU count, affinity, cgroup quota and ``lscpu`` (model, topology): needed to compare runs."""
    out: dict[str, Any] = {"cpu_count": os.cpu_count()}
    try:
        out["affinity_cpus"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    try:
        out["cgroup_cpu_max"] = Path("/sys/fs/cgroup/cpu.max").read_text().strip()
    except OSError:
        pass
    try:
        js = json.loads(subprocess.run(["lscpu", "-J"], capture_output=True, text=True, timeout=30).stdout)  # noqa: S607
        out["lscpu"] = {str(e.get("field", "")).rstrip(":"): e.get("data") for e in js.get("lscpu", [])}
    except Exception as exc:
        out["lscpu_error"] = type(exc).__name__
    try:
        rows = subprocess.run(["lscpu", "-p=CPU,CORE,SOCKET,NODE"], capture_output=True, text=True,  # noqa: S607
                              timeout=30).stdout.splitlines()
        cpus = [r.split(",") for r in rows if r and not r.startswith("#")]
        out["topology"] = {"logical_cpus": len(cpus), "cores": len({(c[2], c[1]) for c in cpus}),
                           "sockets": len({c[2] for c in cpus}), "numa_nodes": len({c[3] for c in cpus})}
    except Exception as exc:
        out["topology_error"] = type(exc).__name__
    return out


def num_delta(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, float]:
    """``b - a`` for the numeric fields of ``b`` (missing in ``a``: 0)."""
    a, b = a or {}, b or {}
    return {k: v - (a.get(k) or 0) for k, v in b.items() if isinstance(v, int | float) and not isinstance(v, bool)}


def trip_split(t0: dict[str, Any] | None, t1: dict[str, Any] | None) -> dict[str, Any]:
    """Per-trip client RTT, service queue/compute and the rest, by rows per request (``/admin/status``
    ``client_timing`` delta). Empty if the gateway does not report it."""
    out: dict[str, Any] = {}
    for cls, b in (t1 or {}).items():
        d = num_delta((t0 or {}).get(cls), b)
        n = d.get("trips", 0)
        if n <= 0:
            continue
        row: dict[str, Any] = {"trips": int(n), "rows_per_trip": round(d["rows"] / n, 2),
                               "multi_item_trips": int(d["multi_item"])}
        for k in ("rtt", "queue", "compute", "wire"):
            row[f"{k}_ms"] = round(d[f"{k}_s"] / n * 1e3, 4)
        if d["multi_item"]:
            row["note"] = "multi-item trips: compute_ms sums the items, queue_ms is their max"
        out[f"rows_{cls}"] = row
    return out


def decode_accounting(b_first: dict[str, Any], b_end: dict[str, Any], db: dict[str, float], n_dec: int,
                      dec_s: float | None) -> dict[str, Any]:
    """Decode-only remote/local seconds per token and per trip (see the module docstring)."""
    dd = num_delta(b_first, b_end)
    trips = dd.get("round_trips", 0)
    out: dict[str, Any] = {"decode_tokens": n_dec}
    if n_dec <= 0 or trips <= 0:
        return out
    trips_per_token = trips / n_dec
    out["decode_round_trips_per_token"] = round(trips_per_token, 2)
    out["decode_rows_per_trip"] = round(dd.get("rows", 0) / trips, 3)
    after_first: dict[str, Any] = {
        "remote_s_per_token": round(dd.get("remote_seconds", 0) / n_dec, 4),
        "local_s_per_token": round(dd.get("local_seconds", 0) / n_dec, 4),
        "remote_ms_per_trip": round(dd.get("remote_seconds", 0) / trips * 1e3, 4),
        "local_ms_per_trip": round(dd.get("local_seconds", 0) / trips * 1e3, 4),
        "note": "backend stats delta from the first streamed token to the end; a step in flight at the "
                "first-token snapshot is partly counted",
    }
    out["decode_after_first_token"] = after_first
    src = after_first
    calls = db.get("decode_calls", 0)
    if "decode_calls" in b_end and calls > 0:
        rem, loc = db.get("decode_remote_seconds", 0.0), db.get("decode_local_seconds", 0.0)
        out["decode_counters"] = {
            "calls": int(calls),
            "remote_ms_per_trip": round(rem / calls * 1e3, 4),
            "local_ms_per_trip": round(loc / calls * 1e3, 4),
            "remote_s_per_token": round(rem / calls * trips_per_token, 4),
            "local_s_per_token": round(loc / calls * trips_per_token, 4),
        }
        src = out["decode_counters"]
        out["decode_source"] = "backend decode counters"
    else:
        out["decode_source"] = "stats delta after the first token"
    for k in ("remote_s_per_token", "local_s_per_token", "remote_ms_per_trip", "local_ms_per_trip"):
        out[f"decode_{k}"] = src[k]
    if dec_s:
        acc = src["remote_s_per_token"] + src["local_s_per_token"]
        out["decode_accounted_s_per_token"] = round(acc, 4)
        out["decode_unaccounted_s_per_token"] = round(dec_s - acc, 4)  # LM head, sampling, Python, streaming
        out["decode_accounted_fraction"] = round(acc / dec_s, 3)
    return out


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# ---- reference (runs in a subprocess) -------------------------------------------------------------
def reference_main(args: argparse.Namespace) -> None:
    import torch

    from pgw.core import RingParams
    from pgw.engine import ChatTokenizer, Generator, PlainBackend, SamplingParams, TransformerModel
    from pgw.modelpack import ModelPack

    torch.set_num_threads(os.cpu_count() or 4)
    pack = ModelPack.open(args.pack, expected_fingerprint=args.fingerprint)
    ring = RingParams.from_name(args.ring)
    backend = PlainBackend(pack, ring, device=args.ref_device)
    model = TransformerModel(pack, backend)  # type: ignore[arg-type]
    tok = ChatTokenizer.from_pack(pack)
    gen = Generator(model, tok, batching=False)
    ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}], add_generation_prompt=True)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, logprobs=2, ignore_eos=True)
    t0 = time.perf_counter()
    events = list(gen.stream(ids, params))
    dt = time.perf_counter() - t0
    out = {
        "prompt_ids": list(ids),
        "token_ids": [e.token_id for e in events],
        "tokens": [str(tok.decode([e.token_id], skip_special_tokens=False)) for e in events],
        "logprobs": [e.logprob for e in events],
        "top": [[(str(tok.decode([i], skip_special_tokens=False)), lp) for i, lp in (e.top_logprobs or [])]
                for e in events],
        "text": "".join(e.text or "" for e in events),
        "seconds": dt,
        "device": args.ref_device,
    }
    Path(args.ref_out).write_text(json.dumps(out))


# ---- main driver ---------------------------------------------------------------------------------
class Stack:
    def __init__(self, work: Path):
        self.work = work
        self.procs: list[subprocess.Popen[bytes]] = []

    def start(self, name: str, args: list[str], env: dict[str, str]) -> subprocess.Popen[bytes]:
        fh = (self.work / "logs" / f"{name}.log").open("ab")
        p = subprocess.Popen([sys.executable, "-m", "pgw.cli", *args], env=env, stdout=fh,  # noqa: S603
                             stderr=subprocess.STDOUT)
        self.procs.append(p)
        return p

    def wait_ready(self, url: str, p: subprocess.Popen[bytes], name: str, timeout: float = 900) -> float:
        import httpx

        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if p.poll() is not None:
                tail = (self.work / "logs" / f"{name}.log").read_text(errors="replace")[-5000:]
                raise RuntimeError(f"{name} exited {p.returncode}:\n{tail}")
            try:
                if httpx.get(url, timeout=5.0, trust_env=False).status_code == 200:
                    return time.monotonic() - t0
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        raise RuntimeError(f"{name} not ready at {url}")

    def stop(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()


def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    R: dict[str, Any] = {"model": args.model, "label": args.label, "device": args.device, "ring": args.ring,
                         "max_tokens": args.max_tokens, "prompt": args.prompt}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.label}.json"

    def save() -> None:
        out_file.write_text(json.dumps(R, indent=1, default=str))

    work = Path(args.work) / args.label
    (work / "logs").mkdir(parents=True, exist_ok=True)
    env = base_env()
    # External monotonic counter on a separate directory (gateways refuse masks without one).
    (work / "counter").mkdir(parents=True, exist_ok=True)
    env["PGW_MASKSTORE_COUNTER"] = "file"
    env["PGW_MASKSTORE_COUNTER_FILE"] = str(work / "counter" / "maskstore.ctr")
    try:
        import torch

        R["torch"] = torch.__version__
        R["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        R["torch_error"] = type(exc).__name__
    R["cpu"] = cpu_info()
    R["cpu_count"] = R["cpu"].get("cpu_count")

    # 1. download + pack
    src = args.model
    if not Path(src).exists():
        from huggingface_hub import snapshot_download

        t0 = time.monotonic()
        src = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        R["download_s"] = round(time.monotonic() - t0, 1)
    pack = work / "pack"
    t0 = time.monotonic()
    if not (pack / "manifest.json").exists():
        cli("pack", str(src), str(pack), "--name", args.label, "--plain-dtype", args.plain_dtype, "-q", env=env,
            log_file=work / "logs" / "pack.log")
    R["pack_s"] = round(time.monotonic() - t0, 1)
    info = json.loads(cli("inspect", str(pack), "--json", env=env))
    fp = info["fingerprint"]
    R["fingerprint"] = fp
    R["pack_bytes"] = dir_bytes(pack)
    R["pack_info"] = {k: info[k] for k in info if k not in ("linears", "files", "tensors")}
    if args.delete_hf and Path(src) != Path(args.model):
        shutil.rmtree(Path(src).resolve().parents[1], ignore_errors=True)  # free disk (blobs dir)
    draft_pack: Path | None = None
    if args.draft_model:
        dsrc = args.draft_model
        if not Path(dsrc).exists():
            from huggingface_hub import snapshot_download

            dsrc = snapshot_download(args.draft_model, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        draft_pack = work / "draft_pack"
        if not (draft_pack / "manifest.json").exists():
            cli("pack", str(dsrc), str(draft_pack), "--name", f"{args.label}-draft", "-q", env=env,
                log_file=work / "logs" / "pack_draft.log")
        R["draft"] = {"model": args.draft_model, "spec_tokens": args.spec_tokens,
                      "devices": args.draft_device or ["cpu"], "pack_bytes": dir_bytes(draft_pack),
                      "spec_min_tokens": args.spec_min_tokens, "spec_max_batch": args.spec_max_batch}
    save()

    # 2. reference
    ref_out = work / "reference.json"
    t0 = time.monotonic()
    subprocess.run([sys.executable, __file__, "--phase", "reference", "--pack", str(pack), "--fingerprint", fp,  # noqa: S603
                    "--ring", args.ring, "--ref-device", args.ref_device, "--max-tokens", str(args.max_tokens),
                    "--prompt", args.prompt, "--ref-out", str(ref_out)], env=env, check=True, timeout=7200)
    ref = json.loads(ref_out.read_text())
    R["reference"] = {"text": ref["text"], "prompt_tokens": len(ref["prompt_ids"]), "seconds": round(ref["seconds"], 2),
                      "device": ref["device"], "wall_s": round(time.monotonic() - t0, 1)}
    log("reference:", repr(ref["text"]))
    save()

    # 3. mask store + factory
    n_prompt = len(ref["prompt_ids"])
    per_req = n_prompt + args.max_tokens
    concurrent = [c for c in args.concurrent if c > 1]
    n_requests = 2 + sum(concurrent)  # non-streaming, streaming, concurrent batches
    slots = per_req * n_requests + 16
    spec_variants = _spec_variants(args, draft_pack)
    for _, _, spec_conc in spec_variants:  # room for speculation rows (spec_row_budget <= 3 per token)
        slots += (n_prompt + args.max_tokens * 3) * (2 + sum(spec_conc))
    # A/B: the same concurrency levels again on a gateway with two-cohort ping-pong (model config engine_pingpong)
    pingpong_variants: list[tuple[str, dict[str, Any], list[int]]] = []
    if args.pingpong:
        pp: dict[str, Any] = {"engine_pingpong": True}
        if args.pingpong_min_sessions is not None:
            pp["engine_pingpong_min_sessions"] = args.pingpong_min_sessions
        pingpong_variants.append(("pingpong_", pp, concurrent))
        slots += per_req * (2 + sum(concurrent))
    store, kek = work / "masks", work / "secrets" / "kek"
    kek.parent.mkdir(mode=0o700, exist_ok=True)
    if not (store / "state.sqlite").exists():
        cli("maskstore", "init", "--root", str(store), "--key-file", str(kek), env=env)
    before = dir_bytes(store)
    t0 = time.monotonic()
    fac = cli("factory", "generate", "--pack", str(pack), "--store", str(store), "--key-file", str(kek), "--ring",
              args.ring, "--slots", str(slots), "--device", args.device, env=env,
              log_file=work / "logs" / "factory.log")
    fac_s = time.monotonic() - t0
    store_bytes = dir_bytes(store) - before
    R["factory_generate"] = {"slots": slots, "wall_s": round(fac_s, 2), "tokens_per_hour_wall_incl_startup":
                             round(slots / fac_s * 3600), "store_bytes": store_bytes,
                             "store_MB_per_token": round(store_bytes / slots / 1e6, 3)}
    try:
        R["factory_generate"]["output"] = json.loads(fac)
    except json.JSONDecodeError:
        R["factory_generate"]["output"] = fac[-2000:]
    save()
    try:
        fb = cli("bench", "factory", "--pack", str(pack), "--ring", args.ring, "--device", args.device, "--slots",
                 str(args.factory_bench_slots), "--chunk-slots", str(min(256, args.factory_bench_slots)), "--json",
                 env=env, log_file=work / "logs" / "bench_factory.log")
        R["factory_bench"] = json.loads(fb[fb.index("{"):])
    except Exception as exc:
        R["factory_bench_error"] = repr(exc)[-3000:]
    save()

    # 4. credentials, service, gateway
    sec = work / "secrets"
    if not (sec / "service_token").exists():
        cli("service", "token", "create", "--tenant", "bench", "--tokens-file", str(sec / "tokens"), "--out",
            str(sec / "service_token"), env=env)
    keys = work / "state" / "api_keys.json"
    keys.parent.mkdir(exist_ok=True)
    api_key = json.loads(cli("gateway", "keys", "create", "--name", f"b{int(time.time())}", "--keys-file", str(keys),
                             env=env))["key"]
    stack = Stack(work)
    try:
        svc_port = free_port()
        svc_env = {**env, "PGW_SERVICE_HOST": "127.0.0.1", "PGW_SERVICE_PORT": str(svc_port),
                   "PGW_SERVICE_PACKS": json.dumps([str(pack)]), "PGW_SERVICE_RINGS": json.dumps([args.ring]),
                   "PGW_SERVICE_TOKENS_FILE": str(sec / "tokens"), "PGW_SERVICE_DEVICE": args.device,
                   "PGW_SERVICE_VERIFY_PACKS": "full", "PGW_SERVICE_BATCH_WINDOW_MS": str(args.batch_window_ms)}
        svc = stack.start("service", ["service", "serve"], svc_env)
        R["service_startup_s"] = round(stack.wait_ready(f"http://127.0.0.1:{svc_port}/readyz", svc, "service"), 1)
        R["gpu_mem_MB_service_loaded"] = gpu_mem_mb()
        svc_url = f"http://127.0.0.1:{svc_port}"
        models = sec / "models.json"
        variants: list[tuple[str, dict[str, Any], list[int]]] = [("", {}, concurrent), *pingpong_variants,
                                                                 *spec_variants]
        gw: subprocess.Popen[bytes] | None = None
        for prefix, extra, conc_list in variants:
            if gw is not None:  # the next gateway reuses the store; stop the previous one first
                gw.terminate()
                gw.wait(timeout=120)
            models.write_text(json.dumps([{"name": args.label, "pack_path": str(pack), "fingerprint": fp,
                                           "ring": args.ring, "service_url": svc_url,
                                           "service_token_file": str(sec / "service_token"),
                                           "max_concurrent_sessions":
                                               max(4, *args.concurrent, args.spec_concurrent or 0),
                                           "service_timeout_s": 600, **extra}]))
            models.chmod(0o600)
            gw_port = free_port()
            gw_env = {**env, "PGW_GATEWAY_HOST": "127.0.0.1", "PGW_GATEWAY_PORT": str(gw_port),
                      "PGW_GATEWAY_MODELS_FILE": str(models), "PGW_GATEWAY_API_KEYS_FILE": str(keys),
                      "PGW_GATEWAY_ADMIN_TOKEN": ADMIN, "PGW_GATEWAY_MASKSTORE_ROOT": str(store),
                      "PGW_GATEWAY_AUDIT_LOG_FILE": str(work / "state" / "audit.jsonl"),
                      "PGW_GATEWAY_TOKENS_PER_MINUTE": "10000000", "PGW_GATEWAY_REQUESTS_PER_MINUTE": "10000",
                      "PGW_MASKSTORE_ROOT": str(store), "PGW_MASKSTORE_KEY_FILE": str(kek),
                      "PGW_MASKSTORE_LOW_WATER_SLOTS": "1"}
            # gateway setting (plan interface C3), only when given and only for the speculative runs
            if "draft_pack_path" in extra and args.spec_max_batch is not None:
                gw_env["PGW_GATEWAY_SPEC_MAX_BATCH"] = str(args.spec_max_batch)
            gw = stack.start(f"gateway{prefix}", ["gateway", "serve"], gw_env)
            ready_s = stack.wait_ready(f"http://127.0.0.1:{gw_port}/readyz", gw, f"gateway{prefix}")
            R[prefix + "gateway_startup_s"] = round(ready_s, 1)
            gw_url = f"http://127.0.0.1:{gw_port}"
            H = {"Authorization": f"Bearer {api_key}"}

            def status(gw_url: str = gw_url) -> dict[str, Any]:
                r = httpx.get(f"{gw_url}/admin/status", headers={"Authorization": f"Bearer {ADMIN}"}, timeout=60,
                              trust_env=False)
                r.raise_for_status()
                return dict(r.json()["models"][0])

            body = {"model": args.label, "messages": [{"role": "user", "content": args.prompt}], "temperature": 0,
                    "max_tokens": args.max_tokens, "ignore_eos": True, "logprobs": True, "top_logprobs": 2}

            # 5a. non-streaming, compared with the reference
            s0, c0, t0 = status(), proc_cpu_seconds(gw.pid), time.perf_counter()
            sc0 = proc_cpu_seconds(svc.pid)
            r = httpx.post(f"{gw_url}/v1/chat/completions", json=body, headers=H, timeout=7200, trust_env=False)
            wall = time.perf_counter() - t0
            c1, sc1, s1 = proc_cpu_seconds(gw.pid), proc_cpu_seconds(svc.pid), status()
            if r.status_code != 200:
                raise RuntimeError(f"chat failed {r.status_code}: {r.text[:2000]}")
            js = r.json()
            content = js["choices"][0]["logprobs"]["content"]
            got_tokens = [c["token"] for c in content]
            got_lps = [c["logprob"] for c in content]
            got_top = [[(t["token"], t["logprob"]) for t in c["top_logprobs"]] for c in content]
            b0, b1 = s0["backend"], s1["backend"]
            ntok = n_prompt + args.max_tokens
            d = {k: b1[k] - b0[k] for k in b1 if isinstance(b1[k], (int, float))}
            R[prefix + "chat"] = {
                "text": js["choices"][0]["message"]["content"],
                "usage": js["usage"],
                "tokens_identical": got_tokens == ref["tokens"],
                "logprobs_identical": got_lps == ref["logprobs"],
                "top2_identical": ([[tuple(x) for x in t] for t in got_top]
                                   == [[tuple(x) for x in t] for t in ref["top"]]),
                "text_identical": js["choices"][0]["message"]["content"] == ref["text"],
                "max_abs_logprob_diff": max((abs(a - b) for a, b in zip(got_lps, ref["logprobs"], strict=False)),
                                            default=None),
                "wall_s": round(wall, 2),
                "backend_delta": d,
                "gateway_cpu_s": round(c1 - c0, 2),
                "service_cpu_s": round(sc1 - sc0, 2),
                "slots_used": ntok,
                "wire_bytes_per_token": round((d.get("bytes_up", 0) + d.get("bytes_down", 0)) / ntok),
                "wire_up_bytes_per_token": round(d.get("bytes_up", 0) / ntok),
                "wire_down_bytes_per_token": round(d.get("bytes_down", 0) / ntok),
                "gateway_cpu_s_per_token_incl_prefill": round((c1 - c0) / ntok, 4),
                "trip_split_incl_prefill": trip_split(s0.get("client_timing"), s1.get("client_timing")),
                "status_after": {k: v for k, v in s1.items() if k != "backend"},
            }
            if got_tokens != ref["tokens"]:
                R[prefix + "chat"]["got_tokens"], R[prefix + "chat"]["ref_tokens"] = got_tokens, ref["tokens"]
            log(prefix + "chat:", repr(R[prefix + "chat"]["text"]), "identical:",
                R[prefix + "chat"]["tokens_identical"], R[prefix + "chat"]["logprobs_identical"])
            save()

            # 5b. streaming: TTFT (prefill) and decode s/token
            s0, c0 = status(), proc_cpu_seconds(gw.pid)
            s_first: dict[str, Any] | None = None  # status right after the first token: decode-only deltas
            times: list[float] = []
            text: list[str] = []
            lps: list[float] = []
            t0 = time.perf_counter()
            with httpx.stream("POST", f"{gw_url}/v1/chat/completions", json={**body, "stream": True}, headers=H,
                              timeout=7200, trust_env=False) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise RuntimeError(f"stream failed {resp.status_code}: {resp.text[:2000]}")
                for line in resp.iter_lines():
                    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                        continue
                    ev = json.loads(line[6:])
                    if ev.get("choices"):
                        ch = ev["choices"][0]
                        piece = ch["delta"].get("content")
                        if ch.get("logprobs"):
                            for c in ch["logprobs"]["content"]:
                                lps.append(c["logprob"])
                                times.append(time.perf_counter() - t0)
                            if s_first is None and times:
                                s_first = status()
                        if piece:
                            text.append(piece)
            c1, s1 = proc_cpu_seconds(gw.pid), status()
            db = num_delta(s0["backend"], s1["backend"])
            dec = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else None
            R[prefix + "stream"] = {
                "text_identical": "".join(text) == ref["text"],
                "logprobs_identical": lps == ref["logprobs"],
                "ttft_s": round(times[0], 3) if times else None,
                "prefill_s_per_prompt_token": round(times[0] / n_prompt, 4) if times else None,
                "decode_s_per_token": round(dec, 4) if dec else None,
                "decode_tokens_per_s": round(1 / dec, 3) if dec else None,
                "wall_s": round(time.perf_counter() - t0, 2),
                "gateway_cpu_s": round(c1 - c0, 2),
                "backend_delta": db,
            }
            if dec:
                # whole-request averages (prefill included, divided by prompt + generated tokens): the old fields
                R[prefix + "stream"]["gateway_local_engine_s_per_token_incl_prefill"] = round(
                    db.get("local_seconds", 0) / ntok, 4)
                R[prefix + "stream"]["remote_s_per_token_incl_prefill"] = round(db.get("remote_seconds", 0) / ntok, 4)
            if s_first is not None:
                R[prefix + "stream"].update(decode_accounting(s_first["backend"], s1["backend"], db,
                                                              len(times) - 1, dec))
                R[prefix + "stream"]["decode_trip_split"] = trip_split(s_first.get("client_timing"),
                                                                       s1.get("client_timing"))
            R[prefix + "stream"]["prefill_trip_split"] = trip_split(s0.get("client_timing"),
                                                                    (s_first or s1).get("client_timing"))
            log(prefix + "stream:", R[prefix + "stream"])
            save()

            # 6. concurrent requests (gateway engine batches sessions; service batches rows)
            for n_conc in conc_list:
                s0, c0, t0 = status(), proc_cpu_seconds(gw.pid), time.perf_counter()

                def one(_: int, gw_url: str = gw_url, body: dict[str, Any] = body,
                        H: dict[str, str] = H) -> dict[str, Any]:
                    rr = httpx.post(f"{gw_url}/v1/chat/completions", json=body, headers=H, timeout=7200,
                                    trust_env=False)
                    return {"status": rr.status_code, "json": rr.json() if rr.status_code == 200 else rr.text[:500]}

                with cf.ThreadPoolExecutor(n_conc) as ex:
                    res = list(ex.map(one, range(n_conc)))
                wall = time.perf_counter() - t0
                c1, s1 = proc_cpu_seconds(gw.pid), status()
                db = num_delta(s0["backend"], s1["backend"])
                ok = [x for x in res if x["status"] == 200]
                same = [[c["token"] for c in x["json"]["choices"][0]["logprobs"]["content"]] == ref["tokens"]
                        for x in ok]
                rc = {
                    "requests": n_conc, "ok": len(ok), "tokens_identical": same,
                    "errors": [x for x in res if x["status"] != 200][:3],
                    "wall_s": round(wall, 2),
                    "generated_tokens_per_s": round(len(ok) * args.max_tokens / wall, 3),
                    "all_tokens_per_s": round(len(ok) * ntok / wall, 3),
                    "gateway_cpu_s_per_token_incl_prefill": round((c1 - c0) / max(1, len(ok) * ntok), 4),
                    "backend_delta": db,
                    "trip_split_incl_prefill": trip_split(s0.get("client_timing"), s1.get("client_timing")),
                    # engine steps and how many were split in two cohorts (ping-pong); empty on older gateways
                    "engine_delta": num_delta(s0.get("engine"), s1.get("engine")),
                }
                # "<prefix>concurrent" keeps the first level (the field older runs have); every level is
                # also under "<prefix>concurrent_<n>"
                if n_conc == conc_list[0]:
                    R[prefix + "concurrent"] = rc
                R[f"{prefix}concurrent_{n_conc}"] = rc
                log(f"{prefix}concurrent {n_conc}:", {k: v for k, v in rc.items()
                                                     if k not in ("backend_delta", "trip_split_incl_prefill",
                                                                  "engine_delta")})
                save()

        # 7. service load test (decode rows, all layers), batch 1 and batched
        R["service_loadtest"] = {}
        for cfg in args.loadtest:
            sessions, rows = (int(v) for v in cfg.lower().split("x"))
            lt = None
            try:
                lt = subprocess.run([sys.executable, "-m", "pgw.service.loadtest", "--url", svc_url, "--token-file",  # noqa: S603
                                     str(sec / "service_token"), "--fingerprint", fp, "--ring", args.ring,
                                     "--sessions", str(sessions), "--tokens", str(args.loadtest_tokens),
                                     "--rows", str(rows), "--timeout", "600"],
                                    env=env, capture_output=True, text=True, timeout=3600)
                txt = lt.stdout
                res_lt = json.loads(txt[txt.index("{"):])
                # tokens_per_s counts token-rows; one decode step (all layers) of one session takes:
                res_lt["s_per_decode_step"] = round(sessions * rows / res_lt["tokens_per_s"], 4)
                res_lt["s_per_token_amortized"] = round(1 / res_lt["tokens_per_s"], 5)
                R["service_loadtest"][f"{sessions}x{rows}"] = res_lt
                log(f"loadtest sessions={sessions} rows={rows}:", res_lt.get("tokens_per_s"), "token-rows/s")
            except Exception as exc:
                R["service_loadtest"][f"{sessions}x{rows}"] = {
                    "error": repr(exc)[-1500:], "stderr": lt.stderr[-3000:] if lt is not None else ""}
            save()
        R["gpu_mem_MB_end"] = gpu_mem_mb()
    finally:
        stack.stop()
        for lf in sorted((work / "logs").glob("*.log")):
            if lf.stem == "service" or lf.stem.startswith("gateway"):
                shutil.copy(lf, out_dir / f"{args.label}_{lf.name}")
        for name in ("factory.log", "bench_factory.log", "pack.log"):
            lf = work / "logs" / name
            if lf.exists():
                shutil.copy(lf, out_dir / f"{args.label}_{name}")
    R["ok"] = True
    save()
    return R


def _spec_variants(args: argparse.Namespace, draft_pack: Path | None) -> list[tuple[str, dict[str, Any], list[int]]]:
    """Speculative gateway runs as ``(result prefix, model config extras, concurrency levels)``, one per
    (draft device, k). With a single k the prefixes are the older ``spec_`` / ``spec_cuda1_``."""
    if draft_pack is None:
        return []
    out: list[tuple[str, dict[str, Any], list[int]]] = []
    ks = list(args.spec_tokens)
    conc = [c for c in ([args.concurrent[0]] if args.spec_concurrent is None else [args.spec_concurrent]) if c > 1]
    only = set(args.spec_concurrent_variants or [])
    for dev in args.draft_device or ["cpu"]:
        for k in ks:
            dk: dict[str, Any] = {"draft_pack_path": str(draft_pack), "spec_tokens": k, "draft_device": dev}
            if args.spec_min_tokens is not None:  # model config field, only when given
                dk["spec_min_tokens"] = min(args.spec_min_tokens, k)
            prefix = "spec_" if dev == "cpu" else f"spec_{dev.replace(':', '')}_"
            if len(ks) > 1:
                prefix += f"k{k}_"
            out.append((prefix, dk, conc if not only or f"{k}@{dev}" in only else []))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="main", choices=["main", "reference"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--label", default="qwen0.5b")
    ap.add_argument("--device", default="cuda", help="service and factory device")
    ap.add_argument("--ref-device", default="cpu", help="device of the QuantPlain reference products")
    ap.add_argument("--ring", default="K40A12")
    ap.add_argument("--plain-dtype", default="float32")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--concurrent", type=int, nargs="+", default=[4],
                    help="concurrent request counts (one batch each); the first is also reported as 'concurrent'")
    ap.add_argument("--loadtest", nargs="*", default=["1x1", "32x1", "1x32"],
                    help="service load tests as SESSIONSxROWS (rows per call: 1 = one decode row; a gateway "
                         "batching N sessions sends N rows per call)")
    ap.add_argument("--loadtest-tokens", type=int, default=4)
    ap.add_argument("--factory-bench-slots", type=int, default=256)
    ap.add_argument("--batch-window-ms", type=float, default=0.0)
    ap.add_argument("--draft-model", default=None, help="also run with speculative decoding (this draft model)")
    ap.add_argument("--spec-tokens", type=int, nargs="+", default=[4],
                    help="draft lengths k; one speculative gateway run per (draft device, k)")
    ap.add_argument("--spec-min-tokens", type=int, default=None,
                    help="model config spec_min_tokens, capped at each run's k (forwarded only when given; "
                         "a value >= k gives one fixed draft size)")
    ap.add_argument("--spec-max-batch", type=int, default=None,
                    help="gateway setting spec_max_batch: speculate only in steps with at most this many "
                         "sequences (forwarded as PGW_GATEWAY_SPEC_MAX_BATCH only when given)")
    ap.add_argument("--spec-concurrent", type=int, default=None,
                    help="concurrent requests in speculative runs (default: the first --concurrent; 0 or 1 skips)")
    ap.add_argument("--spec-concurrent-variants", nargs="*", default=None,
                    help="only these speculative runs do the concurrent batch, as K@DEVICE (e.g. 2@cuda:1)")
    ap.add_argument("--draft-device", nargs="+", default=None,
                    help="draft model device(s) on the gateway; one speculative run each (default cpu)")
    ap.add_argument("--pingpong", action="store_true",
                    help="A/B: also run the single-stream and --concurrent levels on a second gateway with two-cohort "
                         "ping-pong (model config engine_pingpong; results under 'pingpong_*')")
    ap.add_argument("--pingpong-min-sessions", type=int, default=None,
                    help="model config engine_pingpong_min_sessions for the --pingpong run (default: the gateway's, 5)")
    ap.add_argument("--delete-hf", action="store_true", help="delete the HF download after packing")
    ap.add_argument("--work", default="/tmp/pgw_bench")  # noqa: S108 - scratch space of a throwaway VM
    ap.add_argument("--out", default="results")
    # reference phase
    ap.add_argument("--pack")
    ap.add_argument("--fingerprint")
    ap.add_argument("--ref-out")
    args = ap.parse_args()
    if args.phase == "reference":
        reference_main(args)
        return 0
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        err = Path(args.out) / f"{args.label}.error.txt"
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
