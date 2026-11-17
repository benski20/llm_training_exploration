"""Client demo: the full PGW stack on one machine plus a web UI that shows what each side sees.

    python scripts/client_demo/run.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--port 8080] [--slots 800]

Starts, in order: model pack, encrypted mask store + factory, compute service, a *wire tap* between
gateway and service (it forwards every compute call unchanged, keeps a sample of the bytes for the UI
and can corrupt one reply on request), the gateway, and the demo web UI on ``--port``.

Everything generated here (KEK, tokens, API key) is demo-only. Gateway and service talk plain HTTP on
loopback; production uses TLS/mTLS (docs/operations).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def log(msg: str) -> None:
    print(f"\033[1;34m[client-demo]\033[0m {msg}", file=sys.stderr, flush=True)


def pgw(*args: str, env: dict[str, str], out: Path | None = None) -> str:
    cmd = [sys.executable, "-m", "pgw.cli", *args]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)  # noqa: S603
    if out is not None:
        out.write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"pgw {' '.join(args[:2])} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def wait_ready(url: str, proc: subprocess.Popen[bytes], name: str, logf: Path) -> None:
    for _ in range(1200):
        if proc.poll() is not None:
            raise SystemExit(f"{name} exited:\n{logf.read_text()[-3000:]}")
        try:
            if httpx.get(url, timeout=2.0, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"{name} not ready at {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--name", default="qwen2.5-0.5b-instruct", help="model name shown in the API")
    ap.add_argument("--work", default=str(Path.home() / ".pgw-client-demo"))
    ap.add_argument("--ring", default="K40A12")
    ap.add_argument("--slots", type=int, default=800, help="mask slots (one per prompt or generated token)")
    ap.add_argument("--port", type=int, default=8080, help="demo UI port")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--pack-dir", default=None, help="use (or create) the model pack here instead of <work>/packs")
    ap.add_argument("--pack-only", action="store_true", help="build the model pack and exit (image build step)")
    ap.add_argument("--refill-low", type=int, default=0,
                    help="run the mask factory daemon: refill when fewer slots than this remain (0 = off)")
    ap.add_argument("--refill-target", type=int, default=0, help="factory daemon fills up to this many slots")
    args = ap.parse_args()

    work = Path(args.work)
    for d in ("packs", "secrets", "state", "logs", "counter"):
        (work / d).mkdir(parents=True, exist_ok=True)
    (work / "secrets").chmod(0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PGW_")}
    env.update({"PGW_LOG_LEVEL": "INFO", "PGW_MASKSTORE_COUNTER": "file",
                "PGW_MASKSTORE_COUNTER_FILE": str(work / "counter" / "generation.counter")})

    pack = Path(args.pack_dir) if args.pack_dir else work / "packs" / args.name
    if not (pack / "manifest.json").exists():
        log(f"packing {args.model} (int8 weights + SHA-256 fingerprint)")
        pgw("pack", args.model, str(pack), "--name", args.name, env=env, out=work / "logs" / "pack.log")
    fp = json.loads(pgw("inspect", str(pack), "--json", env=env))["fingerprint"]
    log(f"pack fingerprint {fp[:16]}...")
    if args.pack_only:
        return

    masks, kek = work / "masks", work / "secrets" / "kek"
    if not (masks / "state.sqlite").exists():
        pgw("maskstore", "init", "--root", str(masks), "--key-file", str(kek), env=env)
    log(f"mask factory: generating {args.slots} one-time mask slots ({args.ring})")
    t = time.time()
    pgw("factory", "generate", "--pack", str(pack), "--store", str(masks), "--key-file", str(kek),
        "--ring", args.ring, "--slots", str(args.slots), env=env, out=work / "logs" / "factory.log")
    log(f"  done in {time.time() - t:.0f}s")

    if not (work / "secrets" / "service_token").exists():
        pgw("service", "token", "create", "--tenant", "demo", "--tokens-file", str(work / "secrets" / "tokens"),
            "--out", str(work / "secrets" / "service_token"), env=env)
    keys = work / "state" / "api_keys.json"
    if not keys.exists():
        (work / "secrets" / "api_key.json").write_text(
            pgw("gateway", "keys", "create", "--name", "demo", "--keys-file", str(keys), env=env))
    api_key = json.loads((work / "secrets" / "api_key.json").read_text())["key"]
    admin = "demo-admin-" + secrets.token_hex(16)

    procs: list[subprocess.Popen[bytes]] = []

    def start(name: str, cmd: list[str], penv: dict[str, str]) -> subprocess.Popen[bytes]:
        fh = (work / "logs" / f"{name}.log").open("wb")
        p = subprocess.Popen(cmd, env=penv, stdout=fh, stderr=subprocess.STDOUT)  # noqa: S603
        procs.append(p)
        return p

    def stop(*_: object) -> None:
        for p in reversed(procs):
            p.terminate()
        for p in procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    svc_port, gw_port, tap_port = free_port(), free_port(), free_port()
    log(f"compute service on 127.0.0.1:{svc_port} (our side: sees only masked ring elements)")
    svc = start("service", [sys.executable, "-m", "pgw.cli", "service", "serve"], {
        **env, "PGW_SERVICE_HOST": "127.0.0.1", "PGW_SERVICE_PORT": str(svc_port), "PGW_SERVICE_DEVICE": "cpu",
        "PGW_SERVICE_PACKS": json.dumps([str(pack)]), "PGW_SERVICE_RINGS": json.dumps([args.ring]),
        "PGW_SERVICE_TOKENS_FILE": str(work / "secrets" / "tokens")})
    wait_ready(f"http://127.0.0.1:{svc_port}/readyz", svc, "service", work / "logs" / "service.log")

    models = work / "secrets" / "models.json"
    models.write_text(json.dumps([{
        "name": args.name, "pack_path": str(pack), "fingerprint": fp, "ring": args.ring,
        # the gateway talks to the service *through the demo's wire tap*
        "service_url": f"http://127.0.0.1:{tap_port}", "service_token_file": str(work / "secrets" / "service_token"),
    }]))
    models.chmod(0o600)

    ui_env = {**env, "DEMO_SERVICE_URL": f"http://127.0.0.1:{svc_port}", "DEMO_TAP_PORT": str(tap_port),
              "DEMO_GATEWAY_URL": f"http://127.0.0.1:{gw_port}", "DEMO_API_KEY": api_key, "DEMO_ADMIN": admin,
              "DEMO_MODEL": args.name, "DEMO_RING": args.ring, "DEMO_FP": fp, "DEMO_PORT": str(args.port),
              "DEMO_HOST": args.host}
    ui = start("ui", [sys.executable, str(HERE / "server.py")], ui_env)
    wait_ready(f"http://127.0.0.1:{args.port}/api/ping", ui, "ui", work / "logs" / "ui.log")

    log(f"gateway on 127.0.0.1:{gw_port} (customer side: plaintext model, masks, Freivalds checks)")
    gw = start("gateway", [sys.executable, "-m", "pgw.cli", "gateway", "serve"], {
        **env, "PGW_GATEWAY_HOST": "127.0.0.1", "PGW_GATEWAY_PORT": str(gw_port),
        "PGW_GATEWAY_MODELS_FILE": str(models), "PGW_GATEWAY_API_KEYS_FILE": str(keys),
        "PGW_GATEWAY_ADMIN_TOKEN": admin, "PGW_GATEWAY_MASKSTORE_ROOT": str(masks),
        "PGW_GATEWAY_AUDIT_LOG_FILE": str(work / "state" / "audit.jsonl"),
        "PGW_MASKSTORE_ROOT": str(masks), "PGW_MASKSTORE_KEY_FILE": str(kek), "PGW_MASKSTORE_LOW_WATER_SLOTS": "16"})
    wait_ready(f"http://127.0.0.1:{gw_port}/readyz?models=all", gw, "gateway", work / "logs" / "gateway.log")

    if args.refill_low > 0:
        target = max(args.refill_target, args.refill_low * 2)
        log(f"mask factory daemon: refill below {args.refill_low} slots, up to {target}")
        start("factory", [sys.executable, "-m", "pgw.cli", "factory", "run", "--pack", str(pack), "--store", str(masks),
                          "--key-file", str(kek), "--ring", args.ring, "--low-water", str(args.refill_low),
                          "--target", str(target), "--segment-slots", "256", "--nice", "10", "--threads", "1", "--poll", "10"], env)

    log(f"READY: demo UI on http://127.0.0.1:{args.port}/   (logs in {work / 'logs'})")
    while True:
        for p in procs:
            if p.poll() is not None:
                log(f"a process exited ({p.args}); see {work / 'logs'}")
                stop()
        time.sleep(2)


if __name__ == "__main__":
    main()
