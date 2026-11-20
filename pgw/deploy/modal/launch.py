"""Start (or test) the two-party PGW demo on Modal with one command.

    python pgw/deploy/modal/launch.py              # fresh pair left in setup mode (default 80 min)
    python pgw/deploy/modal/launch.py --ttl-min 45
    python pgw/deploy/modal/launch.py --e2e        # scripted wizard + masked chat + exactness check, then stop
    python pgw/deploy/modal/launch.py --e2e --gpu A10G --i6pn --max-trip-ms 1   # latency-gated run
    python pgw/deploy/modal/launch.py --show       # print what is live now
    python pgw/deploy/modal/launch.py --stop       # stop both apps and clear the secrets in the Dict

It deploys ``provider.py`` (app ``pgw-provider``; CPU, or the GPU named by ``--gpu``) and ``customer.py``
(app ``pgw-customer``, CPU), spawns one call of each and waits until both have published their details
in the ``pgw-live`` Dict. Both calls end by themselves after ``--ttl-min`` minutes (hard cap: the
88-minute function timeout); deployed apps with no running call cost nothing.

If either call ends (crash, OOM, timeout) after it has published its details for this run, the
launcher stops the other one and exits instead of waiting; on a start timeout it stops both. The
one-time setup token is printed once and then replaced by ``(shown)`` in the Dict, so ``--show``
cannot print it again (start a new pair if it is lost).

Network: by default the gateway reaches the provider through the provider's public TCP tunnel. With
``--i6pn`` (or ``PGW_MODAL_I6PN=1``) both parties run on Modal's private IPv6 network instead; if Modal
has not scheduled both containers within ``--i6pn-schedule-timeout-min`` minutes, the launcher cancels
the pair and relaunches it on the tunnel.

Latency gate (``--e2e`` only): ``--max-trip-ms X`` reads the gateway's client-reported round-trip
latency over the warm chat (median, from its ``/metrics`` histogram; see ``trip_gate.py``) and, if it is
above ``X``, stops the pair and launches a fresh one, at most ``--tries`` pairs in all (default 3). A
pair whose masked answer is not bit-identical to the reference is never retried.
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trip_gate  # noqa: E402  (no Modal imports; also shipped into the customer image)

DICT_NAME = "pgw-live"
APPS = ("pgw-provider", "pgw-customer")
#: Dict keys that belong to one run (cleared before a new pair starts and by --stop).
RUN_KEYS = ("provider", "customer", "provider_boot", "customer_boot", "e2e", "stop")
#: Function names per network: (provider, customer).
FUNCTIONS = {False: ("serve", "gateway"), True: ("serve_i6pn", "gateway_i6pn")}
# Modal list prices (modal.com/pricing, 2026-11; check before quoting): per GPU-hour, per core-hour,
# per GiB-hour. L4 is deliberately absent (see provider.py: no faster than a T4 for the decode GEMV).
GPU_PRICE_H = {"T4": 0.59, "A10G": 1.10, "L40S": 1.95, "H100": 3.95}
CORE_H, GIB_H = 0.0473, 0.008


def cost_per_h(gpu: str | None) -> dict[str, float]:
    """Hourly list price of each party (provider: 2 cores with a GPU, 4 without; see provider.py)."""
    prov = (GPU_PRICE_H[gpu] + 2 * CORE_H) if gpu else 4 * CORE_H
    return {"pgw-provider": prov + 6 * GIB_H, "pgw-customer": 4 * CORE_H + 8 * GIB_H}


def _ts(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def deploy(gpu: str | None) -> None:
    """``modal deploy`` both apps; the provider's GPU is fixed at deploy time (``PGW_MODAL_GPU``)."""
    env = dict(os.environ)
    if gpu:
        env["PGW_MODAL_GPU"] = gpu
    else:
        env.pop("PGW_MODAL_GPU", None)
    for f in ("provider.py", "customer.py"):
        subprocess.run(["modal", "deploy", str(HERE / f)], check=True, env=env)  # noqa: S603, S607


def stop() -> None:
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    live["stop"] = "all"
    for a in APPS:
        subprocess.run(["modal", "app", "stop", a, "-y"], check=False)  # noqa: S603, S607
    for k in RUN_KEYS:
        live.pop(k, None)
    print("stopped both apps and cleared the live details")


def _call_finished(call: Any) -> tuple[bool, str]:
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

    def __init__(self, live: Any, run_id: str, calls: dict[str, Any]):
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


@dataclass
class Pair:
    """One spawned provider/customer pair."""

    run_id: str
    i6pn: bool
    pcall: Any
    ccall: Any
    t0: float
    note: str = ""

    @property
    def network(self) -> str:
        return "i6pn" if self.i6pn else "tunnel"

    def watch(self, live: Any) -> PeerWatch:
        return PeerWatch(live, self.run_id, {"provider": self.pcall, "customer": self.ccall})


def spawn_pair(live: Any, i6pn: bool, ttl_s: int, e2e: bool,
               lookup: Callable[[str, str], Any] | None = None) -> Pair:
    """Clear the per-run Dict keys and spawn one call of each party on the chosen network."""
    lookup = lookup or modal.Function.from_name
    run_id = secrets.token_hex(4)
    for k in RUN_KEYS:
        live.pop(k, None)
    pfn, cfn = FUNCTIONS[i6pn]
    prov, cust = lookup("pgw-provider", pfn), lookup("pgw-customer", cfn)
    pcall = prov.spawn(run_id, ttl_s)
    ccall = cust.spawn(run_id, ttl_s, e2e)
    print(f"run {run_id} ({'i6pn' if i6pn else 'tunnel'}): spawned provider ({pcall.object_id}) and "
          f"customer ({ccall.object_id})", flush=True)
    return Pair(run_id, i6pn, pcall, ccall, time.time())


def unscheduled(live: Any, run_id: str) -> list[str]:
    """Parties whose function has not started running for ``run_id`` yet (no ``<party>_boot`` marker)."""
    return [p for p in ("provider", "customer") if (live.get(f"{p}_boot") or {}).get("run_id") != run_id]


def wait_scheduled(live: Any, pair: Pair, timeout_s: float, *, poll_s: float = 5.0,
                   clock: Callable[[], float] = time.time,
                   sleep: Callable[[float], None] = time.sleep) -> list[str]:
    """Wait until both functions of ``pair`` run; returns the parties still unscheduled at the timeout
    (``[]`` once both run). Raises ``RuntimeError`` if a call ends before it has even started."""
    end = clock() + timeout_s
    while True:
        missing = unscheduled(live, pair.run_id)
        if not missing:
            return []
        for name, call in (("provider", pair.pcall), ("customer", pair.ccall)):
            done, how = _call_finished(call)
            if done:
                raise RuntimeError(f"{name} ended before it started ({how})")
        if clock() >= end:
            return missing
        sleep(poll_s)


def cancel_pair(live: Any, pair: Pair) -> None:
    """Stop one pair without undeploying the apps (``stop`` would ``modal app stop`` them)."""
    live["stop"] = pair.run_id
    for call in (pair.pcall, pair.ccall):
        try:
            try:
                call.cancel(terminate_containers=True)
            except TypeError:  # older modal clients
                call.cancel()
        except Exception as exc:  # already finished, or never started
            print(f"cancel {getattr(call, 'object_id', '?')}: {type(exc).__name__}", flush=True)


def launch_pair(live: Any, *, i6pn: bool, ttl_s: int, e2e: bool, schedule_timeout_s: float,
                spawn: Callable[..., Pair] = spawn_pair,
                wait: Callable[..., list[str]] = wait_scheduled,
                cancel: Callable[[Any, Pair], None] = cancel_pair) -> Pair:
    """Spawn a pair; with i6pn, fall back to the tunnel when Modal does not schedule both in time."""
    pair = spawn(live, i6pn, ttl_s, e2e)
    if not i6pn:
        return pair
    try:
        missing = wait(live, pair, schedule_timeout_s)
    except RuntimeError as exc:
        cancel(live, pair)
        raise SystemExit(f"{exc}; cancelled the pair") from exc
    if not missing:
        return pair
    why = (f"i6pn: {', '.join(missing)} not scheduled within {schedule_timeout_s / 60:g} min; "
           "relaunching on the public tunnel")
    print(why, flush=True)
    cancel(live, pair)
    fallback = spawn(live, False, ttl_s, e2e)
    fallback.note = why
    return fallback


def wait_e2e(live: Any, pair: Pair, ttl_s: int) -> dict[str, Any]:
    """Wait for the customer's end-to-end result; stops everything on a timeout or a dead provider."""
    watch = pair.watch(live)
    try:
        while True:
            try:
                return dict(pair.ccall.get(timeout=10))
            except (TimeoutError, modal.exception.TimeoutError):
                pass
            if time.time() - pair.t0 > ttl_s + 300:
                stop()
                raise SystemExit("timed out waiting for the end-to-end test; stopped both apps")
            dead = watch.dead()
            if dead and dead.startswith("provider"):
                stop()
                raise SystemExit(f"{dead} during the end-to-end test; stopped both apps")
    finally:
        live["stop"] = pair.run_id


def e2e_verdict(res: dict[str, Any], max_trip_ms: float | None) -> tuple[str, str]:
    """``(verdict, reason)`` for one end-to-end result: ``exactness_failed`` (never retried), ``ok``,
    ``gate_failed`` (relaunch while tries remain) or ``unmeasured`` (gate asked, no trips measured)."""
    if not res.get("ok"):
        return "exactness_failed", str(res.get("error") or "masked answer differs from the reference")
    if max_trip_ms is None:
        return "ok", "no latency gate"
    lat = (((res.get("e2e") or {}).get("trip_latency")) or {}).get("warm")
    passed, why = trip_gate.gate(lat, max_trip_ms)
    if passed is None:
        return "unmeasured", why
    return ("ok" if passed else "gate_failed"), why


def _print_latency(res: dict[str, Any], pair: Pair) -> None:
    lat = ((res.get("e2e") or {}).get("trip_latency") or {}).get("warm") or {}
    if not lat.get("trips"):
        print(f"trip latency ({pair.network}): not measured", flush=True)
        return
    print(f"trip latency ({pair.network}, warm chat): {lat['trips']} trips, p50 {lat.get('p50_ms')} ms "
          f"(bucket {lat.get('p50_bucket_ms')}), p90 {lat.get('p90_ms')} ms, mean {lat.get('mean_ms')} ms; "
          f"1-row mean {lat.get('rows_1_mean_ms')} ms = service {lat.get('rows_1_service_ms')} + "
          f"rest {lat.get('rows_1_wire_ms')}", flush=True)


def run_e2e(live: Any, args: argparse.Namespace, ttl_s: int) -> int:
    """One or more end-to-end pairs (relaunched while the latency gate fails); returns the exit code."""
    t_all = time.time()
    tries = max(1, args.tries if args.max_trip_ms is not None else 1)
    code = 1
    for attempt in range(1, tries + 1):
        pair = launch_pair(live, i6pn=args.i6pn, ttl_s=ttl_s, e2e=True,
                           schedule_timeout_s=args.i6pn_schedule_timeout_min * 60)
        res = wait_e2e(live, pair, ttl_s)
        print(json.dumps({k: v for k, v in res.items() if k != "log_tail"}, indent=1, default=str))
        print("---- gateway log tail ----\n" + res.get("log_tail", ""))
        try:
            pres = pair.pcall.get(timeout=120)
            print("---- service log tail ----\n" + pres.get("log_tail", "")[-2500:])
        except Exception as exc:
            print(f"provider result: {type(exc).__name__}: {exc}")
        if pair.note:
            print(f"note: {pair.note}")
        _print_latency(res, pair)
        verdict, why = e2e_verdict(res, args.max_trip_ms)
        print(f"attempt {attempt}/{tries}: {verdict}: {why}", flush=True)
        if verdict == "ok":
            code = 0
            break
        if verdict == "exactness_failed":
            code = 1
            break
        code = 2  # latency gate failed or unmeasured
        if verdict == "unmeasured" or attempt == tries:
            break
        if not pair.i6pn and args.max_trip_ms < 3:
            print("hint: the public tunnel adds about 5 ms per round trip; a gate this tight passes only on "
                  "i6pn (--i6pn)", flush=True)
        cancel_pair(live, pair)  # make sure the old provider is gone: max_containers=1 would queue the new one
        print("relaunching a fresh pair (new placement)", flush=True)
    stop()
    print(f"elapsed {time.time() - t_all:.0f}s")
    return code


def show(live: Any) -> None:
    p, c = live.get("provider") or {}, live.get("customer") or {}
    print(json.dumps({"provider": p, "customer": c}, indent=1, default=str))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ttl-min", type=float, default=80.0, help="lifetime of the pair (max 85)")
    ap.add_argument("--e2e", action="store_true", help="scripted end-to-end test, then stop")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-deploy", action="store_true",
                    help="reuse the deployed apps (their GPU is whatever was deployed)")
    ap.add_argument("--deploy-only", action="store_true",
                    help="build the images and deploy both apps, start nothing (no running cost)")
    ap.add_argument("--gpu", choices=["none", *GPU_PRICE_H], default=os.environ.get("PGW_MODAL_GPU") or "none",
                    help="provider GPU, fixed at deploy time (default: none = CPU int8 backend; "
                         "PGW_MODAL_GPU sets the default). Not L4 (see README)")
    ap.add_argument("--i6pn", action="store_true", default=os.environ.get("PGW_MODAL_I6PN", "0") == "1",
                    help="run both parties on Modal's private IPv6 network (off by default; PGW_MODAL_I6PN=1)")
    ap.add_argument("--i6pn-schedule-timeout-min", type=float, default=6.0,
                    help="with --i6pn: relaunch on the public tunnel if Modal has not started both "
                         "containers within this many minutes")
    ap.add_argument("--max-trip-ms", type=float, default=None,
                    help="with --e2e: relaunch while the median client-reported round trip of the warm chat "
                         "is above this (off by default)")
    ap.add_argument("--tries", type=int, default=3, help="with --max-trip-ms: pairs to try in all (default 3)")
    ap.add_argument("--local-linears", choices=["off", "attention", "auto"], default="off",
                    help="placeholder for exact local linears (E2); only 'off' until WP-A4b/A5 land")
    args = ap.parse_args(argv)
    if args.gpu == "none":
        args.gpu = None
    if args.max_trip_ms is not None and not args.e2e:
        ap.error("--max-trip-ms needs --e2e (the latency is measured on the scripted chat)")
    if args.max_trip_ms is not None and args.max_trip_ms <= 0:
        ap.error("--max-trip-ms must be positive")
    if args.local_linears != "off":
        ap.error(f"--local-linears {args.local_linears}: exact local linears (E2) are not available in this build "
                 "(pending WP-A4b/A5); see 'Exact local linears' in deploy/modal/README.md")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.deploy_only:
        deploy(args.gpu)
        print("deployed; start a pair later with --no-deploy")
        return
    if args.stop:
        stop()
        return
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    if args.show:
        show(live)
        return
    ttl_s = int(min(args.ttl_min, 85.0) * 60)
    if not args.no_deploy:
        deploy(args.gpu)
    if args.e2e:
        sys.exit(run_e2e(live, args, ttl_s))
    pair = launch_pair(live, i6pn=args.i6pn, ttl_s=ttl_s, e2e=False,
                       schedule_timeout_s=args.i6pn_schedule_timeout_min * 60)
    watch = pair.watch(live)
    while True:
        p, c = live.get("provider") or {}, live.get("customer") or {}
        if p.get("run_id") == pair.run_id and c.get("run_id") == pair.run_id:
            break
        dead = watch.dead()
        if dead:
            stop()
            raise SystemExit(f"{dead} before the pair was up; stopped both apps")
        if time.time() - pair.t0 > 1200:
            stop()
            raise SystemExit("timed out waiting for the pair to start; stopped both apps")
        time.sleep(5)
    exp = min(p["expires_at"], c["expires_at"])
    cost = sum(cost_per_h(args.gpu).values()) * (exp - pair.t0) / 3600
    note = f"\n  Note: {pair.note}" if pair.note else ""
    print(f"""
PGW two-party demo is up (run {pair.run_id}, network {pair.network}, provider GPU {p.get('gpu', '?')}).{note}

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
    if cur.get("run_id") == pair.run_id:
        live["customer"] = {**cur, "setup_token": "(shown)"}


if __name__ == "__main__":
    main()
