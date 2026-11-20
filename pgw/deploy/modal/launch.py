"""Start (or test) the two-party PGW demo on Modal with one command.

    python pgw/deploy/modal/launch.py              # fresh pair left in setup mode (default 80 min)
    python pgw/deploy/modal/launch.py --ttl-min 45
    python pgw/deploy/modal/launch.py --e2e        # scripted wizard + masked chat + exactness check, then stop
    python pgw/deploy/modal/launch.py --show       # print what is live now
    python pgw/deploy/modal/launch.py --stop       # stop both apps and clear the secrets in the Dict

It deploys ``provider.py`` (app ``pgw-provider``, T4) and ``customer.py`` (app ``pgw-customer``, CPU),
spawns one call of each and waits until both have published their details in the ``pgw-live``
Dict. Both calls end by themselves after ``--ttl-min`` minutes (hard cap: the 88-minute function
timeout); deployed apps with no running call cost nothing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
DICT_NAME = "pgw-live"
APPS = ("pgw-provider", "pgw-customer")
# Modal list prices (modal billing rates, 2026-11): T4 0.59/h, CPU 0.0473/core/h, memory 0.008/GiB/h.
_GPU = bool(os.environ.get("PGW_MODAL_GPU"))
COST_PER_H = {"pgw-provider": (0.59 + 1 * 0.0473 if _GPU else 4 * 0.0473) + 6 * 0.008,
              "pgw-customer": 4 * 0.0473 + 8 * 0.008}


def _ts(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def deploy() -> None:
    for f in ("provider.py", "customer.py"):
        subprocess.run(["modal", "deploy", str(HERE / f)], check=True)  # noqa: S603, S607


def stop() -> None:
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    live["stop"] = "all"
    for a in APPS:
        subprocess.run(["modal", "app", "stop", a, "-y"], check=False)  # noqa: S603, S607
    for k in ("provider", "customer", "e2e"):
        live.pop(k, None)
    live.pop("stop", None)
    print("stopped both apps and cleared the live details")


def show(live: modal.Dict) -> None:
    p, c = live.get("provider") or {}, live.get("customer") or {}
    print(json.dumps({"provider": p, "customer": c}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttl-min", type=float, default=80.0, help="lifetime of the pair (max 85)")
    ap.add_argument("--e2e", action="store_true", help="scripted end-to-end test, then stop")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-deploy", action="store_true", help="reuse the deployed apps")
    args = ap.parse_args()
    if args.stop:
        stop()
        return
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    if args.show:
        show(live)
        return
    ttl_s = int(min(args.ttl_min, 85.0) * 60)
    if not args.no_deploy:
        deploy()
    run_id = secrets.token_hex(4)
    for k in ("provider", "customer", "e2e", "stop"):
        live.pop(k, None)
    prov = modal.Function.from_name("pgw-provider", "serve")
    cust = modal.Function.from_name("pgw-customer", "gateway")
    t0 = time.time()
    pcall = prov.spawn(run_id, ttl_s)
    ccall = cust.spawn(run_id, ttl_s, args.e2e)
    print(f"run {run_id}: spawned provider ({pcall.object_id}) and customer ({ccall.object_id})", flush=True)
    if args.e2e:
        try:
            res = ccall.get(timeout=ttl_s + 300)
        finally:
            live["stop"] = run_id
        print(json.dumps({k: v for k, v in res.items() if k != "log_tail"}, indent=1, default=str))
        print("---- gateway log tail ----\n" + res.get("log_tail", ""))
        try:
            pres = pcall.get(timeout=120)
            print("---- service log tail ----\n" + pres.get("log_tail", "")[-2500:])
        except Exception as exc:
            print(f"provider result: {type(exc).__name__}: {exc}")
        stop()
        print(f"elapsed {time.time() - t0:.0f}s")
        sys.exit(0 if res.get("ok") else 1)
    while True:
        p, c = live.get("provider") or {}, live.get("customer") or {}
        if p.get("run_id") == run_id and c.get("run_id") == run_id:
            break
        if time.time() - t0 > 1200:
            raise SystemExit("timed out waiting for the pair to start; run --stop")
        time.sleep(5)
    exp = min(p["expires_at"], c["expires_at"])
    cost = sum(COST_PER_H.values()) * (exp - t0) / 3600
    print(f"""
PGW two-party demo is up (run {run_id}).

  Customer console (setup wizard): {c['console_url']}
  One-time setup token:            {c['setup_token']}
  Enrollment code (from provider): {p['code']}
      code valid until {_ts(p['code_expires_at'])}, single use
  Provider compute service:        {p['service_url']}  (TLS end to end, CA sha256 {p['ca_sha256'][:16]}...)

Both containers stop by themselves at {_ts(exp)} (cost if left running until then: about ${cost:.2f}).
Stop earlier with: python {HERE / 'launch.py'} --stop
""")


if __name__ == "__main__":
    main()
