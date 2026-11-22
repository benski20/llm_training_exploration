"""Start (or test) the two-party PGW demo on Modal with one command.

    python pgw/deploy/modal/launch.py              # fresh pair left in setup mode (default 80 min)
    python pgw/deploy/modal/launch.py --ttl-min 45
    python pgw/deploy/modal/launch.py --e2e        # scripted wizard + masked chat + exactness check, then stop
    python pgw/deploy/modal/launch.py --e2e --gpu A10G --i6pn --max-trip-ms 1   # latency-gated run
    python pgw/deploy/modal/launch.py --e2e --gpu T4 --placement colocated      # both parties, one container
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

Placement (``--placement``, ``PGW_MODAL_PLACEMENT``): ``tunnel`` (default) = two containers, the
gateway reaches the provider through the provider's public TCP tunnel; ``i6pn`` (also ``--i6pn`` or
``PGW_MODAL_I6PN=1``) = two containers on Modal's private IPv6 network; ``colocated`` = one container
(app ``pgw-colocated``, ``colocated.py``) running both parties as separate Unix users over loopback TLS
(lowest latency; see the trust caveat there and in the README). If Modal has not started an ``i6pn`` or
``colocated`` pair within ``--schedule-timeout-min`` minutes, the launcher cancels it and relaunches the
pair on the tunnel. The window runs until each function writes its ``*_boot``
marker, so it covers image pull and container cold start as well as Modal's scheduling; the default is
therefore longer with a GPU (larger CUDA image, scarcer machines).

Latency gate (``--e2e`` only): ``--max-trip-ms X`` reads the gateway's client-reported round-trip
latency over the warm chat (median, from its ``/metrics`` histogram; see ``trip_gate.py``) and, if it is
above ``X``, stops the pair and launches a fresh one, at most ``--tries`` pairs in all (default 3). A
pair whose masked answer is not bit-identical to the reference is never retried, and a tunnel pair is
not relaunched for a limit below :data:`TUNNEL_FLOOR_MS` (the tunnel alone adds about 5 ms per trip).

Gateway speed settings: ``--local-linears`` (default ``auto``) is written into the customer's
``models.json`` by the setup wizard together with ``float_threads = 1`` (``pgw_modal_boot.GATEWAY_MODEL_OPTIONS``);
``--e2e`` prints the decision and the round trips per token of the warm chat.

Strict speculation (off by default): ``--spec-rows R`` presets ``spec_rows = R`` with strict passes and prompt
lookup for the wizard's ``models.json``: every decode pass carries exactly R masked rows (pending token, drafts,
masked random fillers) and the provider sees as many passes as without speculation (SPEC 3.4), while the answer
streams in fewer sequential passes. ``--model`` (deploy time) serves a larger Qwen2.5 target and ``--spec-draft``
(deploy time) bakes the 0.5B draft model into the customer image. Mask budget: R slots per generated token (see
:func:`spec_budget`). README section "Strict speculation" has the measurements and when it pays off.
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
#: Where the two parties run and how the gateway reaches the provider (``--placement``).
PLACEMENTS = ("tunnel", "i6pn", "colocated")
#: Function names per two-container network: (provider, customer). Kept keyed by "i6pn?" for the tests.
FUNCTIONS = {False: ("serve", "gateway"), True: ("serve_i6pn", "gateway_i6pn")}
#: The co-located placement's single function: (app, function).
COLOCATED_FUNCTION = ("pgw-colocated", "pair")
#: Apps per placement (deployed and stopped by the launcher; the tunnel apps are always deployed, as the
#: fallback of the other placements).
PLACEMENT_APPS = {"tunnel": APPS, "i6pn": APPS, "colocated": (*APPS, COLOCATED_FUNCTION[0])}
# Modal list prices (modal.com/pricing, 2026-11; check before quoting): per GPU-hour, per core-hour,
# per GiB-hour. L4 is deliberately absent (see provider.py: no faster than a T4 for the decode GEMV).
GPU_PRICE_H = {"T4": 0.59, "A10G": 1.10, "L40S": 1.95, "H100": 3.95}
CORE_H, GIB_H = 0.0473, 0.008
#: A ``--max-trip-ms`` limit below this cannot pass on the public tunnel (about 5 ms per round trip, 6.8 ms
#: measured on 2026-11-20), so a tunnel pair is not relaunched for it.
TUNNEL_FLOOR_MS = 3.0
#: Default ``--schedule-timeout-min`` (minutes until both ``*_boot`` markers): without / with a GPU.
I6PN_SCHEDULE_TIMEOUT_MIN = {False: 6.0, True: 10.0}
#: Targets ``--model`` accepts (the same table as ``pgw_modal_boot.MODELS`` / ``provider.py``).
MODELS = ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct")
DEFAULT_MODEL = MODELS[0]


def spec_budget(spec_rows: int, prompt_tokens: int, max_tokens: int) -> dict[str, int]:
    """Mask slots of one request with strict speculation at ``spec_rows`` masked rows per pass (the gateway's
    ``Generator.mask_rows_needed``): ``prompt + max_tokens + (max_tokens - 1) * (spec_rows - 1)`` reserved up
    front, the same consumed whatever the text (dummy passes take ``spec_rows`` rows too), against
    ``prompt + max_tokens`` without speculation."""
    r = max(1, int(spec_rows))
    base = int(prompt_tokens) + int(max_tokens)
    return {"plain": base, "spec": base + max(0, int(max_tokens) - 1) * (r - 1)}


def normalize_placement(value: str | bool | None) -> str:
    """``--placement`` value (or the legacy ``i6pn`` flag) to one of :data:`PLACEMENTS`."""
    if isinstance(value, bool):
        return "i6pn" if value else "tunnel"
    v = (value or "tunnel").strip().lower()
    if v not in PLACEMENTS:
        raise ValueError(f"{value!r}: unknown placement; use one of {', '.join(PLACEMENTS)}")
    return v


def provider_cpus(gpu: str | None) -> float:
    """Provider cores as ``provider.py`` sets them: ``PGW_MODAL_PROVIDER_CPUS``, else 2 with a GPU, 4 without."""
    return float(os.environ.get("PGW_MODAL_PROVIDER_CPUS") or (2.0 if gpu else 4.0))


def customer_gib() -> int:
    """Customer memory as ``customer.py`` sets it: 16 GiB with a target larger than the 0.5B or the draft
    (``PGW_MODAL_MODEL`` / ``PGW_MODAL_DRAFT``), else 8."""
    model = (os.environ.get("PGW_MODAL_MODEL") or DEFAULT_MODEL).strip().lower()
    return 16 if os.environ.get("PGW_MODAL_DRAFT", "0") == "1" or model != DEFAULT_MODEL else 8


def cost_per_h(gpu: str | None, cpus: float | None = None) -> dict[str, float]:
    """Hourly list price of each party (provider cores: ``cpus``, else :func:`provider_cpus`)."""
    cores = provider_cpus(gpu) if cpus is None else float(cpus)
    prov = (GPU_PRICE_H[gpu] if gpu else 0.0) + cores * CORE_H
    if gpu and os.environ.get("PGW_MODAL_GATEWAY_GPU", "0") == "1":  # co-located --gateway-gpu: a second GPU
        prov += GPU_PRICE_H[gpu]
    cgpu = normalize_gpu(os.environ.get("PGW_MODAL_CUSTOMER_GPU"))
    cust = (GPU_PRICE_H[cgpu] if cgpu else 0.0) + 4 * CORE_H + customer_gib() * GIB_H
    return {"pgw-provider": prov + 6 * GIB_H, "pgw-customer": cust}


def normalize_gpu(value: str | None) -> str | None:
    """``--gpu`` / ``PGW_MODAL_GPU`` to a tier of :data:`GPU_PRICE_H` (case-insensitive), ``None`` for
    none/empty. Raises ``ValueError`` for L4, a GPU count (``T4:2``) or an unknown name."""
    v = (value or "").strip()
    if v.lower() in ("", "none", "cpu"):
        return None
    if ":" in v:
        raise ValueError(f"{v!r}: one GPU only (no ':count'); the provider serves one device")
    tiers = {t.upper(): t for t in GPU_PRICE_H}
    if v.upper() == "L4":
        raise ValueError("L4 is not supported for the provider (no faster than a T4 for the bandwidth-bound "
                         f"decode GEMV, and dearer); use one of {', '.join(GPU_PRICE_H)}")
    if v.upper() not in tiers:
        raise ValueError(f"{v!r}: unknown GPU; use none or one of {', '.join(GPU_PRICE_H)}")
    return tiers[v.upper()]


def gpu_tier_of_device(device: str | None) -> str | None:
    """Tier of a CUDA device name as the provider reports it (``"NVIDIA A10G"``, ``"Tesla T4"``; ``"none"``
    for the CPU backend). ``None`` for no GPU; raises ``KeyError`` for a device outside the price table."""
    if not device or device == "none":
        return None
    up = device.upper()
    for tier in sorted(GPU_PRICE_H, key=len, reverse=True):
        if tier in up:
            return tier
    raise KeyError(device)


def _ts(t: float) -> str:
    return dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def deploy(gpu: str | None, customer_gpu: str | None = None, placement: str = "tunnel",
           region: str | None = None, model: str | None = None, draft: bool = False) -> None:
    """``modal deploy`` both apps (and ``colocated.py`` for that placement); the GPUs are fixed at deploy
    time (``PGW_MODAL_GPU`` for the provider, ``PGW_MODAL_CUSTOMER_GPU`` for the customer gateway's mask
    factory), and so is the region (``region``: ``None`` keeps ``PGW_MODAL_REGION`` / the apps' default
    ``us-east``; ``""`` lifts the pin), the target the provider packs (``PGW_MODAL_MODEL``) and the speculation
    draft in the customer image (``PGW_MODAL_DRAFT=1``)."""
    env = dict(os.environ)
    for var, val in (("PGW_MODAL_GPU", gpu), ("PGW_MODAL_CUSTOMER_GPU", customer_gpu),
                     ("PGW_MODAL_MODEL", model if model and model != DEFAULT_MODEL else None),
                     ("PGW_MODAL_DRAFT", "1" if draft else None)):
        if val:
            env[var] = val
        else:
            env.pop(var, None)
    if region is not None:
        env["PGW_MODAL_REGION"] = region
    files = ["provider.py", "customer.py"] + (["colocated.py"] if placement == "colocated" else [])
    for f in files:
        subprocess.run(["modal", "deploy", str(HERE / f)], check=True, env=env)  # noqa: S603, S607


def stop() -> None:
    live = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    live["stop"] = "all"
    for a in PLACEMENT_APPS["colocated"]:
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
    """One spawned provider/customer pair (``placement``: one of :data:`PLACEMENTS`, or the legacy
    ``i6pn`` bool). In the co-located placement both calls are the same FunctionCall."""

    run_id: str
    placement: str | bool
    pcall: Any
    ccall: Any
    t0: float
    note: str = ""

    def __post_init__(self) -> None:
        self.placement = normalize_placement(self.placement)

    @property
    def network(self) -> str:
        return str(self.placement)

    @property
    def i6pn(self) -> bool:
        return self.placement == "i6pn"

    @property
    def colocated(self) -> bool:
        return self.placement == "colocated"

    def watch(self, live: Any) -> PeerWatch:
        return PeerWatch(live, self.run_id, {"provider": self.pcall, "customer": self.ccall})


def spawn_pair(live: Any, placement: str | bool, ttl_s: int, e2e: bool,
               lookup: Callable[[str, str], Any] | None = None, *, local_linears: str = "auto",
               spec_rows: int = 0, lane: str = "stream", gateway_env: dict[str, str] | None = None,
               provider_env: dict[str, str] | None = None, sustain_s: float = 0.0) -> Pair:
    """Clear the per-run Dict keys and spawn the pair in the chosen placement (two calls, or one
    co-located call; ``placement`` may be the legacy ``i6pn`` bool)."""
    lookup = lookup or modal.Function.from_name
    placement = normalize_placement(placement)
    run_id = secrets.token_hex(4)
    for k in RUN_KEYS:
        live.pop(k, None)
    if placement == "colocated":
        extra: tuple[Any, ...] = (lane,) if lane != "stream" else ()  # --lane tls: the provider's TCP+TLS lane
        if gateway_env or provider_env or sustain_s:  # --gateway-env / --provider-env / --sustain-s (colocated)
            extra = (lane, dict(gateway_env or {}), dict(provider_env or {}), float(sustain_s))
        call = lookup(*COLOCATED_FUNCTION).spawn(run_id, ttl_s, e2e, local_linears, spec_rows, *extra)
        print(f"run {run_id} (colocated): spawned both parties in one container ({call.object_id})", flush=True)
        return Pair(run_id, placement, call, call, time.time())
    pfn, cfn = FUNCTIONS[placement == "i6pn"]
    prov, cust = lookup("pgw-provider", pfn), lookup("pgw-customer", cfn)
    pcall = prov.spawn(run_id, ttl_s)
    ccall = cust.spawn(run_id, ttl_s, e2e, local_linears, spec_rows)
    print(f"run {run_id} ({placement}): spawned provider ({pcall.object_id}) and "
          f"customer ({ccall.object_id})", flush=True)
    return Pair(run_id, placement, pcall, ccall, time.time())


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
    for call in dict.fromkeys((pair.pcall, pair.ccall)):  # the co-located pair has one call
        try:
            try:
                call.cancel(terminate_containers=True)
            except TypeError:  # older modal clients
                call.cancel()
        except Exception as exc:  # already finished, or never started
            print(f"cancel {getattr(call, 'object_id', '?')}: {type(exc).__name__}", flush=True)


def launch_pair(live: Any, *, placement: str | bool | None = None, ttl_s: int, e2e: bool,
                schedule_timeout_s: float, spawn: Callable[..., Pair] = spawn_pair,
                wait: Callable[..., list[str]] = wait_scheduled,
                cancel: Callable[[Any, Pair], None] = cancel_pair, i6pn: bool | None = None) -> Pair:
    """Spawn a pair; for i6pn and colocated, fall back to the tunnel when Modal does not start it in time
    (``i6pn`` is the legacy spelling of ``placement``)."""
    placement = normalize_placement(placement if placement is not None else bool(i6pn))
    pair = spawn(live, placement, ttl_s, e2e)
    if placement == "tunnel":
        return pair
    try:
        missing = wait(live, pair, schedule_timeout_s)
    except RuntimeError as exc:
        cancel(live, pair)
        raise SystemExit(f"{exc}; cancelled the pair") from exc
    if not missing:
        return pair
    why = (f"{placement}: {', '.join(missing)} not scheduled within {schedule_timeout_s / 60:g} min; "
           "relaunching on the public tunnel")
    print(why, flush=True)
    cancel(live, pair)
    fallback = spawn(live, "tunnel", ttl_s, e2e)
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


def _print_speed(res: dict[str, Any]) -> None:
    sp = (res.get("e2e") or {}).get("speed") or {}
    if not sp:
        return
    ll = sp.get("local_linears") or {}
    print(f"gateway speed settings: local_linears {ll.get('mode')} -> {ll.get('kinds') or 'none'} "
          f"(start-up rtt {sp.get('measured_rtt_ms')} ms over {sp.get('rtt_path') or '?'}; "
          f"{ll.get('reason', '')}), float_threads "
          f"{sp.get('float_threads')}, lane {sp.get('lane')} ({sp.get('lane_state')}), strict speculation "
          f"{sp.get('spec_rows') or 'off'}; {sp.get('warm_trips_per_token', '?')} round trips per token "
          f"(warm chat), {sp.get('warm_tokens_per_s', '?')} tokens/s (warm chat wall time; "
          f"{sp.get('warm_loopback_tokens_per_s', '?')} sent on loopback, without Modal's HTTPS edge)", flush=True)
    if sp.get("spec_rows"):
        print(f"strict speculation: warm answer {sp.get('warm_answer_tokens_per_s', '?')} tokens/s; dummy passes "
              f"after the first chat took {sp.get('spec_drain_wait_s', '?')} s", flush=True)


def path_label(res: dict[str, Any], pair: Pair) -> str:
    """``placement`` plus the network path the gateway actually used (``loopback``, ``private <addr>``,
    ``public <host>``; from the service URL it enrolled with), e.g. ``tunnel via public r500.modal.host``."""
    tl = (res.get("e2e") or {}).get("trip_latency") or {}
    path = tl.get("path") or (res.get("e2e") or {}).get("network_path")
    return f"{pair.network} via {path}" if path else pair.network


def _print_latency(res: dict[str, Any], pair: Pair) -> None:
    _print_speed(res)
    lat = ((res.get("e2e") or {}).get("trip_latency") or {}).get("warm") or {}
    where = path_label(res, pair)
    if not lat.get("trips"):
        print(f"trip latency ({where}): not measured", flush=True)
        return
    print(f"trip latency ({where}, warm chat): {lat['trips']} trips, p50 {lat.get('p50_ms')} ms "
          f"(bucket {lat.get('p50_bucket_ms')}), p90 {lat.get('p90_ms')} ms, mean {lat.get('mean_ms')} ms; "
          f"1-row mean {lat.get('rows_1_mean_ms')} ms = service {lat.get('rows_1_service_ms')} + "
          f"rest {lat.get('rows_1_wire_ms')}", flush=True)


#: Variable prefixes ``--gateway-env`` / ``--provider-env`` may set (PGW settings and thread-count knobs only).
ENV_PREFIXES = ("PGW_", "OMP_", "MKL_", "GOMP_")


def parse_env_pairs(items: list[str]) -> dict[str, str]:
    """``["KEY=VALUE", ...]`` -> dict; keys must start with one of :data:`ENV_PREFIXES`."""
    out: dict[str, str] = {}
    for it in items:
        key, sep, val = it.partition("=")
        key = key.strip()
        if not sep or not key.startswith(ENV_PREFIXES):
            raise ValueError(f"environment setting {it!r}: use KEY=VALUE with KEY starting with {ENV_PREFIXES}")
        out[key] = val
    return out


def _spawner(args: argparse.Namespace) -> Callable[..., Pair]:
    """:func:`spawn_pair` with the customer gateway's ``--local-linears`` and ``--spec-rows`` choices."""
    local = getattr(args, "local_linears", "auto")
    spec_rows = int(getattr(args, "spec_rows", 0) or 0)
    lane = getattr(args, "lane", "stream") or "stream"
    genv = parse_env_pairs(getattr(args, "gateway_env", None) or [])
    pin = int(getattr(args, "float_threads", 0) or 0)
    if pin > 0:  # --float-threads: the gateway's torch thread count (the reference follows it, colocated.py)
        genv.setdefault("OMP_NUM_THREADS", str(pin))
    penv = parse_env_pairs(getattr(args, "provider_env", None) or [])
    sustain = float(getattr(args, "sustain_s", 0.0) or 0.0)

    def spawn(live: Any, placement: str | bool, ttl_s: int, e2e: bool) -> Pair:
        # a tunnel fallback of a co-located --lane tls run uses the stream lane (no provider lane port there)
        if normalize_placement(placement) != "colocated":
            return spawn_pair(live, placement, ttl_s, e2e, local_linears=local, spec_rows=spec_rows, lane="stream")
        return spawn_pair(live, placement, ttl_s, e2e, local_linears=local, spec_rows=spec_rows, lane=lane,
                          gateway_env=genv, provider_env=penv, sustain_s=sustain)

    return spawn


def run_e2e(live: Any, args: argparse.Namespace, ttl_s: int) -> int:
    """One or more end-to-end pairs (relaunched while the latency gate fails); returns the exit code."""
    t_all = time.time()
    tries = max(1, args.tries if args.max_trip_ms is not None else 1)
    code = 1
    for attempt in range(1, tries + 1):
        pair = launch_pair(live, placement=args.placement, ttl_s=ttl_s, e2e=True,
                           schedule_timeout_s=args.i6pn_schedule_timeout_min * 60, spawn=_spawner(args))
        res = wait_e2e(live, pair, ttl_s)
        print(json.dumps({k: v for k, v in res.items() if k not in ("log_tail", "service_log_tail")}, indent=1,
                         default=str))
        print("---- gateway log tail ----\n" + res.get("log_tail", ""))
        if pair.colocated:  # one call: the service's log comes with the result
            print("---- service log tail ----\n" + str(res.get("service_log_tail", ""))[-2500:])
        else:
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
        if pair.network == "tunnel" and args.placement == "tunnel" and args.max_trip_ms < TUNNEL_FLOOR_MS:
            # every relaunch would be another tunnel pair (a full paid e2e run) with no chance of passing
            print(f"not relaunching: the public tunnel adds about 5 ms per round trip, so a limit below "
                  f"{TUNNEL_FLOOR_MS:g} ms passes only on i6pn or co-located (--placement)", flush=True)
            break
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
    ap.add_argument("--customer-gpu", default=os.environ.get("PGW_MODAL_CUSTOMER_GPU") or "none",
                    help="GPU for the customer gateway (fixed at deploy time; default none): runs the encrypted "
                         "mask factory's encryption on the customer's own GPU")
    ap.add_argument("--gpu", default=os.environ.get("PGW_MODAL_GPU") or "none", metavar="{none," +
                    ",".join(GPU_PRICE_H) + "}",
                    help="provider GPU, fixed at deploy time (default: none = CPU int8 backend; "
                         "PGW_MODAL_GPU sets the default; case-insensitive). Not L4 (see README)")
    ap.add_argument("--placement", choices=PLACEMENTS, default=None,
                    help="where the parties run: 'tunnel' (default; two containers, public TCP tunnel), 'i6pn' "
                         "(two containers, Modal's private IPv6 network), 'colocated' (one container, both "
                         "parties as separate users over loopback TLS; see the README's trust caveat). "
                         "PGW_MODAL_PLACEMENT sets the default")
    ap.add_argument("--i6pn", action="store_true", default=os.environ.get("PGW_MODAL_I6PN", "0") == "1",
                    help="same as --placement i6pn (PGW_MODAL_I6PN=1)")
    ap.add_argument("--schedule-timeout-min", "--i6pn-schedule-timeout-min", dest="i6pn_schedule_timeout_min",
                    type=float, default=None,
                    help="with --placement i6pn or colocated: relaunch on the public tunnel if the pair has not "
                         "started within this many minutes (Modal scheduling + image pull + cold start; default "
                         f"{I6PN_SCHEDULE_TIMEOUT_MIN[False]:g} without a GPU, "
                         f"{I6PN_SCHEDULE_TIMEOUT_MIN[True]:g} with one)")
    ap.add_argument("--region", default=None,
                    help="Modal region of the deployed apps (fixed at deploy time; default: PGW_MODAL_REGION, else "
                         "us-east). 'any' lifts the pin (risk: the parties land far apart). A region group such as "
                         "us-east, not a cloud zone such as us-east-1")
    ap.add_argument("--max-trip-ms", type=float, default=None,
                    help="with --e2e: relaunch while the median client-reported round trip of the warm chat "
                         "is above this (off by default)")
    ap.add_argument("--tries", type=int, default=3, help="with --max-trip-ms: pairs to try in all (default 3)")
    ap.add_argument("--model", default=os.environ.get("PGW_MODAL_MODEL") or DEFAULT_MODEL, choices=MODELS,
                    help="target the provider packs and serves (fixed at deploy time; PGW_MODAL_MODEL sets the "
                         f"default; default {DEFAULT_MODEL})")
    ap.add_argument("--spec-rows", type=int, default=0,
                    help="strict speculation with this many masked rows per decode pass (2-8; default 0 = off): "
                         "prompt lookup, plus the draft model with --spec-draft. Exact; the provider sees fixed rows "
                         "and the same pass count. Uses spec-rows mask slots per generated token")
    ap.add_argument("--spec-mask", default=os.environ.get("PGW_MODAL_SPEC_MASK", ""),
                    help="with --spec-rows: padded strict speculation with a public mask charge, 'C' or 'C/B' (deploy "
                         "time; PGW_MODAL_SPEC_MASK): filler rows and dummy passes are padding (no masks) and every "
                         "decode pass charges C mask slots (1 <= C <= spec-rows), the draft credit capped at B slots. "
                         "Default '' = masked filler rows (spec-rows slots per token)")
    ap.add_argument("--spec-draft", action="store_true", default=os.environ.get("PGW_MODAL_DRAFT", "0") == "1",
                    help="bake the Qwen2.5-0.5B draft model into the customer image (deploy time; needs --spec-rows "
                         "and a --model larger than the 0.5B; PGW_MODAL_DRAFT=1). On a CPU customer it runs on the "
                         "gateway's CPU, with --customer-gpu on its GPU")
    ap.add_argument("--local-linears", choices=["off", "attention", "attention_down", "auto", "auto_kinds"],
                    default=None,
                    help="exact local linears (E2) of the customer gateway (default: 'attention_down' co-located, "
                         "measured fastest there; 'auto' otherwise): 'auto' = qkv and o on the "
                         "gateway when its start-up round-trip measurement says they are cheaper than two trips "
                         "(the tunnel: yes); 'attention' = always; 'attention_down' = also down "
                         "(one round trip per layer); "
                         "'auto_kinds' = each kind local iff cheaper than a trip; 'off' = every linear on the provider")
    ap.add_argument("--lane", choices=["stream", "tls"], default=None,
                    help="customer gateway's trip lane (default: 'tls' co-located, 'stream' otherwise): 'stream' = "
                         "the blocking stream lane (WebSocket, msgpack, Python TLS); 'tls' (--placement colocated "
                         "only) = the provider also serves its "
                         "native trip lane over TLS 1.3 on 127.0.0.1:8444 and the gateway uses it (rustls, binary "
                         "header, GIL released per trip; same masked bytes, request ids and schedule; misses fall "
                         "back to the stream lane). Exact; the e2e comparison still applies")
    ap.add_argument("--gateway-env", action="append", default=[], metavar="KEY=VALUE",
                    help="--placement colocated: extra environment of the customer gateway party (and the mask "
                         "factory it starts), e.g. PGW_FACTORY_SKIP_LINEAR_KINDS=qkv,o,down; PGW_/OMP_/MKL_ keys only")
    ap.add_argument("--provider-env", action="append", default=[], metavar="KEY=VALUE",
                    help="--placement colocated: extra environment of the provider's service (PGW_/OMP_/MKL_ keys)")
    ap.add_argument("--sustain-s", type=float, default=0.0,
                    help="--e2e --placement colocated: after the exactness check, chat back to back for this many "
                         "seconds and report tokens/s and the mask factory's segments finished meanwhile (slots/s "
                         "while a chat decodes)")
    ap.add_argument("--float-threads", type=int, default=None,
                    help="--placement colocated: the pinned float formulation. The gateway's process runs torch on "
                         "exactly N threads (OMP_NUM_THREADS=N) and the e2e's plaintext reference on the same N, so "
                         "thread-count-dependent float kernels (CPU SDPA) give the same bits in both (default: 1 "
                         "co-located; 0 = each process's torch default, as before)")
    ap.add_argument("--gateway-gpu", action="store_true",
                    default=os.environ.get("PGW_MODAL_GATEWAY_GPU", "0") == "1",
                    help="--placement colocated with --gpu: a second GPU of the same tier for the gateway (deploy "
                         "time; PGW_MODAL_GATEWAY_GPU=1). Each party's process sees only its own GPU; the gateway "
                         "computes its exact local linears there (self-checked bit-identical, else the CPU). "
                         "Default off")
    args = ap.parse_args(argv)
    try:
        parse_env_pairs(args.gateway_env)
        parse_env_pairs(args.provider_env)
    except ValueError as exc:
        ap.error(str(exc))
    try:
        env_placement = os.environ.get("PGW_MODAL_PLACEMENT") or None
        if args.placement is None:
            args.placement = normalize_placement("i6pn" if args.i6pn else env_placement)
        elif args.i6pn and args.placement != "i6pn":
            ap.error(f"--i6pn contradicts --placement {args.placement}")
    except ValueError as exc:
        ap.error(f"PGW_MODAL_PLACEMENT: {exc}")
    args.i6pn = args.placement == "i6pn"
    if args.region is not None:
        args.region = "" if args.region.strip().lower() in ("any", "none", "") else args.region.strip()
    try:
        args.customer_gpu = normalize_gpu(args.customer_gpu)
        if args.placement == "colocated" and args.customer_gpu:
            ap.error("--customer-gpu applies to the two-container placements only (the co-located gateway "
                     "runs on CPU)")
        if args.customer_gpu:  # cost estimates read it
            os.environ["PGW_MODAL_CUSTOMER_GPU"] = args.customer_gpu
        else:
            os.environ.pop("PGW_MODAL_CUSTOMER_GPU", None)
    except ValueError as exc:
        ap.error(f"--customer-gpu / PGW_MODAL_CUSTOMER_GPU: {exc}")
    try:  # also validates a PGW_MODAL_GPU default, which argparse would not check against choices
        args.gpu = normalize_gpu(args.gpu)
    except ValueError as exc:
        ap.error(f"--gpu / PGW_MODAL_GPU: {exc}")
    if args.i6pn_schedule_timeout_min is None:
        args.i6pn_schedule_timeout_min = I6PN_SCHEDULE_TIMEOUT_MIN[args.gpu is not None]
    elif args.i6pn_schedule_timeout_min <= 0:
        ap.error("--schedule-timeout-min must be positive")
    if args.max_trip_ms is not None and not args.e2e:
        ap.error("--max-trip-ms needs --e2e (the latency is measured on the scripted chat)")
    if args.local_linears is None:
        args.local_linears = "attention_down" if args.placement == "colocated" else "auto"
    if args.lane is None:
        args.lane = "tls" if args.placement == "colocated" else "stream"
    if args.lane == "tls" and args.placement != "colocated":
        ap.error("--lane tls needs --placement colocated (the provider's lane port is on its loopback only)")
    if args.float_threads is None:
        args.float_threads = 1 if args.placement == "colocated" else 0
    if args.float_threads < 0 or (args.float_threads and args.placement != "colocated"):
        ap.error("--float-threads needs --placement colocated and N >= 0")
    if args.gateway_gpu and (args.placement != "colocated" or not args.gpu):
        ap.error("--gateway-gpu needs --placement colocated and a provider --gpu (two GPUs of that tier)")
    if args.max_trip_ms is not None and args.max_trip_ms <= 0:
        ap.error("--max-trip-ms must be positive")
    args.model = args.model.strip().lower()
    if args.model not in MODELS:
        ap.error(f"--model / PGW_MODAL_MODEL: use one of {', '.join(MODELS)}")
    if args.spec_rows != 0 and not 2 <= args.spec_rows <= 8:
        ap.error("--spec-rows must be 0 (off) or in [2, 8]")
    if args.spec_draft and not args.spec_rows:
        ap.error("--spec-draft needs --spec-rows (the draft is only used by strict speculation)")
    if args.spec_mask:
        if not args.spec_rows:
            ap.error("--spec-mask needs --spec-rows")
        c_s, _, b_s = args.spec_mask.strip().partition("/")
        try:  # the same parsing as pgw_modal_boot.spec_mask_options (in the container)
            c_v, b_v = float(c_s), int(b_s) if b_s.strip() else 0
        except ValueError:
            ap.error("--spec-mask: use C or C/B, e.g. 4/160")
        if not 1 <= c_v <= args.spec_rows or b_v < 0:
            ap.error("--spec-mask: need 1 <= C <= --spec-rows and B >= 0")
    if args.spec_draft and args.model == DEFAULT_MODEL:
        # the target itself as its own draft: the gateway would compute the whole model in plaintext
        ap.error("--spec-draft drafts with the 0.5B itself: use a larger --model (the draft must be much cheaper "
                 "than the target, or the provider computes nothing the gateway does not)")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # the deploy-time choices also drive the cost estimate (customer memory) in this process
    for var, val in (("PGW_MODAL_MODEL", args.model if args.model != DEFAULT_MODEL else None),
                     ("PGW_MODAL_DRAFT", "1" if args.spec_draft else None),
                     ("PGW_MODAL_GATEWAY_GPU", "1" if args.gateway_gpu else None),
                     ("PGW_MODAL_SPEC_MASK", args.spec_mask.strip() or None)):
        if val:
            os.environ[var] = val
        else:
            os.environ.pop(var, None)
    if args.spec_rows:
        b = spec_budget(args.spec_rows, 41, 32)
        print(f"strict speculation: {args.spec_rows} masked rows per decode pass, i.e. {args.spec_rows} mask slots per "
              f"generated token (the e2e chat reserves {b['spec']} slots instead of {b['plain']}); pass max_tokens "
              "explicitly: a request reserves prompt + max_tokens + (max_tokens - 1) x (spec_rows - 1) slots",
              flush=True)
    if args.deploy_only:
        deploy(args.gpu, args.customer_gpu, args.placement, args.region, args.model, args.spec_draft)
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
        deploy(args.gpu, args.customer_gpu, args.placement, args.region, args.model, args.spec_draft)
    if args.e2e:
        sys.exit(run_e2e(live, args, ttl_s))
    pair = launch_pair(live, placement=args.placement, ttl_s=ttl_s, e2e=False,
                       schedule_timeout_s=args.i6pn_schedule_timeout_min * 60, spawn=_spawner(args))
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
    # Price what is running (with --no-deploy the deployed GPU and cores may differ from these arguments).
    try:
        tier = gpu_tier_of_device(p.get("gpu")) if "gpu" in p else args.gpu
    except KeyError:
        tier = args.gpu
    cost = sum(cost_per_h(tier, p.get("cpus")).values()) * (exp - pair.t0) / 3600
    note = f"\n  Note: {pair.note}" if pair.note else ""
    print(f"""
PGW two-party demo is up (run {pair.run_id}, placement {pair.network}, path {p.get('network_path') or '?'},
provider GPU {p.get('gpu', '?')}).{note}

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
