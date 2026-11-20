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

If either call ends (crash, OOM, timeout) after it has published its details for this run, the
launcher stops the other one and exits instead of waiting; on a start timeout it stops both. The
one-time setup token is printed once and then replaced by ``(shown)`` in the Dict, so ``--show``
cannot print it again (start a new pair if it is lost).
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


def _call_finished(call: modal.FunctionCall) -> tuple[bool, str]:
    """``(finished, how)`` without blocking: a FunctionCall that returned, raised or was cancelled."""
    try:
        call.get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError):
        return False, ""
    except Exception as exc:  # the call ended with an error (crash, OOM, cancelled, function timeout)
        return True, f"{type(exc).__name__}: {str(exc)[:200]}"
    return True, "returned"


class PeerWatch:
    """Dead-peer detection for one run: a party counts as dead only after it has published its details
    for ``run_id`` at least once and its FunctionCall has then finished (a call still starting up or
    queued for a GPU is not dead)."""

    def __init__(self, live: modal.Dict, run_id: str, calls: dict[str, modal.FunctionCall]):
        self.live, self.run_id, self.calls = live, run_id, calls
        self.seen: set[str] = set()

    def dead(self) -> str | None:
        """Description of the first dead party, or ``None``."""
        for name, call in self.calls.items():
            if name not in self.seen and (self.live.get(name) or {}).get("run_id") == self.run_id:
                self.seen.add(name)
            if name in self.seen:
                done, how = _call_finished(call)
                if done:
                    return f"{name} ended ({how})"
        return None


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
    watch = PeerWatch(live, run_id, {"provider": pcall, "customer": ccall})
    if args.e2e:
        try:
            while True:
                try:
                    res = ccall.get(timeout=10)
                    break
                except (TimeoutError, modal.exception.TimeoutError):
                    pass
                if time.time() - t0 > ttl_s + 300:
                    stop()
                    raise SystemExit("timed out waiting for the end-to-end test; stopped both apps")
                dead = watch.dead()
                if dead and dead.startswith("provider"):
                    stop()
                    raise SystemExit(f"{dead} during the end-to-end test; stopped both apps")
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
        dead = watch.dead()
        if dead:
            stop()
            raise SystemExit(f"{dead} before the pair was up; stopped both apps")
        if time.time() - t0 > 1200:
            stop()
            raise SystemExit("timed out waiting for the pair to start; stopped both apps")
        time.sleep(5)
    exp = min(p["expires_at"], c["expires_at"])
    cost = sum(COST_PER_H.values()) * (exp - t0) / 3600
    print(f"""
PGW two-party demo is up (run {run_id}).

  Customer console (setup wizard): {c['console_url']}
  One-time setup token:            {c['setup_token']}   (shown once: --show will not repeat it)
  Enrollment code (from provider): {p['code']}
      code valid until {_ts(p['code_expires_at'])}, single use
  Provider compute service:        {p['service_url']}  (TLS end to end, CA sha256 {p['ca_sha256'][:16]}...)

Both containers stop by themselves at {_ts(exp)} (cost if left running until then: about ${cost:.2f}).
Stop earlier with: python {HERE / 'launch.py'} --stop
""", flush=True)
    # Shown once: the token is a credential for the setup wizard, so it does not stay readable in the
    # Dict for anyone with access to the workspace (and --show no longer prints it).
    cur = live.get("customer") or {}
    if cur.get("run_id") == run_id:
        live["customer"] = {**cur, "setup_token": "(shown)"}


if __name__ == "__main__":
    main()
