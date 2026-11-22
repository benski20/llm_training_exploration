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

#: ``PGW_MODAL_APP_SUFFIX`` (deploy time, default empty): appended to every app name and to the Dict name, so
#: parallel experiments in one workspace do not replace each other's apps (the containers get it in their image env).
APP_SUFFIX = os.environ.get("PGW_MODAL_APP_SUFFIX", "").strip()
DICT_NAME = "pgw-live" + APP_SUFFIX
APPS = ("pgw-provider" + APP_SUFFIX, "pgw-customer" + APP_SUFFIX)
#: Dict keys that belong to one run (cleared before a new pair starts and by --stop).
RUN_KEYS = ("provider", "customer", "provider_boot", "customer_boot", "e2e", "stop")
#: Where the two parties run and how the gateway reaches the provider (``--placement``).
PLACEMENTS = ("tunnel", "i6pn", "colocated")
#: Function names per two-container network: (provider, customer). Kept keyed by "i6pn?" for the tests.
FUNCTIONS = {False: ("serve", "gateway"), True: ("serve_i6pn", "gateway_i6pn")}
#: The co-located placement's single function: (app, function).
COLOCATED_FUNCTION = ("pgw-colocated" + APP_SUFFIX, "pair")
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
MODELS = ("qwen2.5-0.5b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-3b-instruct", "qwen2.5-7b-instruct",
          "qwen2.5-32b-instruct")
#: Targets served only by the co-located placement (``colocated.py`` sizes memory and disk for them; the
#: two-container apps do not): ``--model`` refuses them elsewhere and ``--placement colocated`` deploys only
#: ``colocated.py`` for them (no tunnel fallback).
COLOCATED_ONLY_MODELS = ("qwen2.5-7b-instruct", "qwen2.5-32b-instruct")
#: Co-located memory of the large targets, (provider GiB, gateway GiB) (= colocated.py COLOCATED_MEM_GIB; a test
#: checks it). Requests, not limits; the 32B's gateway keeps the 114 GB hint and 37.5 GB pack in the page cache.
COLOCATED_MEM_GIB = {"qwen2.5-7b-instruct": (16, 48), "qwen2.5-32b-instruct": (48, 160)}
#: Provider GPUs a target fits (its int8 linears plus maskgen scratch on one device); default: every tier.
#: Qwen2.5-32B: 31.2 GB of int8 linears; only the H100 (80 GB) is planned (the L40S's 48 GB is untested).
MODEL_GPUS = {"qwen2.5-32b-instruct": ("H100",)}
#: Targets whose provider pack lives in a provider-side Volume (= colocated.py / pgw_modal_boot.VOLUME_PACK_MODELS)
#: and whose encryption hint must be cached (= pgw_modal_boot.HINT_CACHE_REQUIRED_MODELS): ``--prepare-cache`` fills
#: both Volumes, and a run checks them first (:func:`cache_preflight`, free: no container).
VOLUME_PACK_MODELS = ("qwen2.5-32b-instruct",)
PROVIDER_PACK_VOLUME = os.environ.get("PGW_MODAL_PROVIDER_PACK_VOLUME") or "pgw-provider-packs"
CACHE_VOLUME = os.environ.get("PGW_MODAL_CACHE_VOLUME") or "pgw-gateway-cache"
#: ``--mask-source`` (= pgw_modal_boot.MASK_SOURCES). ``gpu-service`` is a hook for the customer-side GPU mask
#: service (built in a separate package): refused here until that package wires it and sets this to True.
MASK_SOURCES = ("factory", "gpu-service")
GPU_MASK_SERVICE_READY = False
#: Port of the provider's bulk maskgen channel in the co-located container (``--he-bulk``; 127.0.0.1 only).
HE_BULK_PORT = 8445
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
    if model in COLOCATED_MEM_GIB:  # colocated.py: the large targets' gateway party (pack, hint pages, factory)
        return COLOCATED_MEM_GIB[model][1]
    return 16 if os.environ.get("PGW_MODAL_DRAFT", "0") == "1" or model != DEFAULT_MODEL else 8


def provider_gib() -> int:
    """Provider memory: 6 GiB, more for a co-located-only target (its int8 pack is staged in host memory)."""
    model = (os.environ.get("PGW_MODAL_MODEL") or DEFAULT_MODEL).strip().lower()
    return COLOCATED_MEM_GIB[model][0] if model in COLOCATED_MEM_GIB else 6


def cost_per_h(gpu: str | None, cpus: float | None = None) -> dict[str, float]:
    """Hourly list price of each party (provider cores: ``cpus``, else :func:`provider_cpus`)."""
    cores = provider_cpus(gpu) if cpus is None else float(cpus)
    prov = (GPU_PRICE_H[gpu] if gpu else 0.0) + cores * CORE_H
    if gpu and os.environ.get("PGW_MODAL_GATEWAY_GPU", "0") == "1":  # co-located --gateway-gpu: a second GPU
        prov += GPU_PRICE_H[gpu]
    cgpu = normalize_gpu(os.environ.get("PGW_MODAL_CUSTOMER_GPU"))
    cust = (GPU_PRICE_H[cgpu] if cgpu else 0.0) + 4 * CORE_H + customer_gib() * GIB_H
    return {APPS[0]: prov + provider_gib() * GIB_H, APPS[1]: cust}


def colocated_cost_per_h(gpu: str | None, model: str | None = None) -> dict[str, float]:
    """Hourly list price of the co-located container as ``colocated.py`` sizes it (deploy-time environment):
    the provider's GPU (two with ``--gateway-gpu``), provider + gateway cores and memory."""
    model = (model or os.environ.get("PGW_MODAL_MODEL") or DEFAULT_MODEL).strip().lower()
    draft = os.environ.get("PGW_MODAL_DRAFT", "0") == "1"
    pcpu = provider_cpus(gpu)
    gcpu = float(os.environ.get("PGW_MODAL_GATEWAY_CPUS") or 12.0)
    pdef, gdef = COLOCATED_MEM_GIB.get(model, (6, 16 if draft or model != DEFAULT_MODEL else 8))
    pmem = float(os.environ.get("PGW_MODAL_PROVIDER_MEM_GIB") or pdef)
    gmem = float(os.environ.get("PGW_MODAL_GATEWAY_MEM_GIB") or gdef)
    ngpu = 2 if gpu and os.environ.get("PGW_MODAL_GATEWAY_GPU", "0") == "1" else 1
    gp = (GPU_PRICE_H[gpu] * ngpu) if gpu else 0.0
    out = {"gpu": gp, "cores": (pcpu + gcpu) * CORE_H, "memory": (pmem + gmem) * GIB_H}
    out["total"] = sum(out.values())
    out.update(provider_cpus=pcpu, gateway_cpus=gcpu, memory_gib=pmem + gmem)
    return out


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
    if placement == "colocated" and (model in COLOCATED_ONLY_MODELS
                                     or os.environ.get("PGW_MODAL_DEPLOY_COLOCATED_ONLY", "0") == "1"):
        # the two-container apps would pack the target too (minutes of image build) for a fallback that cannot
        # serve it; PGW_MODAL_DEPLOY_COLOCATED_ONLY=1 does the same for any target (no tunnel fallback then)
        files = ["colocated.py"]
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
    prov, cust = lookup(APPS[0], pfn), lookup(APPS[1], cfn)
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
    if sp.get("warm_stream_decode_tokens_per_s") is not None:
        print(f"streamed warm chat (loopback): time to first token {sp.get('warm_stream_ttft_s')} s, decode "
              f"{sp.get('warm_stream_decode_tokens_per_s')} tokens/s ({sp.get('warm_stream_decode_ms_per_token')} "
              f"ms/token after the first), wall {sp.get('warm_stream_wall_s')} s; stream identical "
              f"{(res.get('e2e') or {}).get('stream_identical')}", flush=True)
    for row in (res.get("e2e") or {}).get("ab") or []:  # --gateway-env PGW_MODAL_AB=...: in-run A/B diagnostics
        print(f"A/B {row.get('phase')}: decode ms/token {row.get('decode_ms_per_token')}, TTFT s "
              f"{row.get('ttft_s')}, identical {row.get('identical')}; 1-row trip "
              f"{(row.get('trips') or {}).get('rows_1_mean_ms')} ms (service "
              f"{(row.get('trips') or {}).get('rows_1_service_ms')}), local linear "
              f"{row.get('local_decode_us_per_call')} us/call {row.get('error') or ''}", flush=True)
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
    if getattr(args, "float_ops", "torch") == "native":  # the reference follows it too (colocated.py)
        genv.setdefault("PGW_FLOAT_OPS", "native")
    penv = parse_env_pairs(getattr(args, "provider_env", None) or [])
    sustain = float(getattr(args, "sustain_s", 0.0) or 0.0)

    colocated_only = (getattr(args, "model", DEFAULT_MODEL) in COLOCATED_ONLY_MODELS
                      or os.environ.get("PGW_MODAL_DEPLOY_COLOCATED_ONLY", "0") == "1")
    # CPU-only gateway + strict speculation (README "CPU-only 7B gateway with strict speculation"): run-time
    # model options and gateway settings, carried to colocated.py as PGW_MODAL_* / PGW_* gateway variables
    if getattr(args, "draft_backend", None):
        genv["PGW_MODAL_DRAFT_BACKEND"] = args.draft_backend
    if getattr(args, "spec_pipeline_steps", None) is not None:
        genv["PGW_MODAL_SPEC_PIPELINE_STEPS"] = str(int(args.spec_pipeline_steps))
    if (getattr(args, "spec_mask", "") or "").strip():
        genv.setdefault("PGW_MODAL_SPEC_MASK", args.spec_mask.strip())
    if getattr(args, "trip_native", False):  # one native call per side of each trip (with PGW_TRIP_FAST, on co-located)
        genv.setdefault("PGW_TRIP_NATIVE", "1")
    if getattr(args, "prime_check", False):  # one-vector Freivalds check over F_p: factory format + gateway acceptance
        genv.setdefault("PGW_FACTORY_CHECK_FIELD", "p31")
        genv.setdefault("PGW_GATEWAY_PRIME_CHECK", "true")
    if (getattr(args, "mask_source", None) or "factory") != "factory":  # hook (pgw_modal_boot.mask_source_env)
        genv["PGW_MODAL_MASK_SOURCE"] = args.mask_source
    if getattr(args, "masked_head", False):  # the masked LM head: service, factory and gateway model option
        penv.setdefault("PGW_SERVICE_MASKED_HEAD", "also")
        genv["PGW_MODAL_MASKED_HEAD"] = "1"
        genv.setdefault("PGW_FACTORY_MASKED_HEAD", "true")
    if getattr(args, "ab_plain", None) is not None:  # in-run A/B: the same run again without speculation
        genv["PGW_MODAL_AB_PLAIN"] = f"{float(args.ab_plain):g}"
    if getattr(args, "he_bulk", False):  # --he-bulk: the provider's bulk maskgen channel, used by the factory
        penv.setdefault("PGW_SERVICE_MASKGEN_BULK_PORT", str(HE_BULK_PORT))
        genv.setdefault("PGW_FACTORY_HE_BULK_PORT", str(HE_BULK_PORT))
        genv.setdefault("PGW_FACTORY_HE_BULK_SPLIT", str(int(getattr(args, "he_bulk_split", 2) or 1)))

    def spawn(live: Any, placement: str | bool, ttl_s: int, e2e: bool) -> Pair:
        if colocated_only and normalize_placement(placement) != "colocated":
            raise SystemExit("the co-located pair was not scheduled and only colocated.py is deployed (target "
                             f"{getattr(args, 'model', '?')}): no tunnel fallback; try again later")
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
        sus = (res.get("e2e") or {}).get("sustain") or {}
        if sus:  # --sustain-s: one line on decode rate and the mask factory's keep-up
            ku = sus.get("keep_up") or {}
            print(f"sustain: {sus.get('chats')} chats in {sus.get('seconds')}s, {sus.get('tokens_per_s')} tok/s "
                  f"overall, streamed decode median {sus.get('decode_tokens_per_s_median')} (min "
                  f"{sus.get('decode_tokens_per_s_min')}) tok/s, identical {sus.get('identical')}, error "
                  f"{sus.get('error')}; masks: consumption {ku.get('consumption_slots_per_s')} slots/s, factory "
                  f"{ku.get('production_slots_per_s')} slots/s, stock {ku.get('stock_start')} -> "
                  f"{ku.get('stock_end')} (min {ku.get('stock_min')}), segment-to-segment "
                  f"{ku.get('production_commit_to_commit_slots_per_s')} slots/s, stock slope "
                  f"{ku.get('stock_slope_slots_per_s')} slots/s; token-weighted decode "
                  f"{sus.get('decode_tokens_per_s_overall')} tok/s", flush=True)
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
        _print_ab_plain(res)
        for k in ("cache_restore", "cache_restore_s", "cache_save", "cache_save_s"):
            if k in res:
                print(f"{k}: {res[k]}", flush=True)
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


def prepare_cache(force: bool = False) -> dict[str, Any]:
    """Run ``colocated.py``'s ``prepare_cache`` (a T4 container, no provider process): fill the gateway side's
    cache Volume with the deployed target's pack and encrypted-mask hint, so the next co-located run with
    ``--gateway-cache`` restores them instead of building them (README "Gateway cache")."""
    model = (os.environ.get("PGW_MODAL_MODEL") or DEFAULT_MODEL).strip().lower()
    fn = modal.Function.from_name(COLOCATED_FUNCTION[0], "prepare_cache")
    t0 = time.time()
    # a Volume-pack target (the 32B): the provider side's pack in its own CPU container, in parallel (no shared Volume)
    pcall = (modal.Function.from_name(COLOCATED_FUNCTION[0], "prepare_provider_pack").spawn(force)
             if model in VOLUME_PACK_MODELS else None)
    out: dict[str, Any] = {"gateway_cache": dict(fn.remote(force))}
    if pcall is not None:
        out["provider_pack"] = dict(pcall.get())
        out["preflight"] = cache_preflight(model)
    out["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1, default=str), flush=True)
    return out


def prepare_cost_per_h(model: str) -> dict[str, float]:
    """Hourly list price of ``--prepare-cache``'s containers as ``colocated.py`` sizes them (PREP_*): the gateway
    side's T4 container and, for a Volume-pack target, the provider side's CPU container."""
    big = model in VOLUME_PACK_MODELS
    cpus, gib = (16, 48) if big else (8, 32 if model == "qwen2.5-7b-instruct" else 16)
    out = {"prepare_cache (T4)": GPU_PRICE_H["T4"] + cpus * CORE_H + gib * GIB_H}
    if big:
        out["prepare_provider_pack (CPU)"] = cpus * CORE_H + gib * GIB_H
    return out


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    """``--dry-run``: what a launch with these arguments would deploy and pass, without any Modal call."""
    import pgw_modal_boot as boot

    calls: list[dict[str, Any]] = []
    spawn = _spawner(args)
    real = globals()["spawn_pair"]
    globals()["spawn_pair"] = lambda *a, **kw: calls.append(kw)  # capture the run-time environment only
    try:
        if args.placement == "colocated":
            spawn(None, "colocated", int(min(args.ttl_min, 85.0) * 60), args.e2e)
    finally:
        globals()["spawn_pair"] = real
    deploy_env = {k: os.environ[k] for k in sorted(os.environ) if k.startswith("PGW_MODAL_")}
    sizes = boot.model_sizes(args.model)
    plan = {"model": args.model, "gpu": args.gpu, "placement": args.placement, "deploy_env": deploy_env,
            "run": calls[-1] if calls else None,
            "prepare_cache": args.prepare_cache, "volume_pack": args.model in VOLUME_PACK_MODELS,
            "sizes_gb": {k: round(v / 1e9, 2) for k, v in sizes.items() if k.endswith("bytes")},
            "slot_mb": round(sizes["slot_bytes"] / 1e6, 2) if sizes else None,
            "factory_keep_up": boot.factory_keep_up(args.model)}
    print(json.dumps(plan, indent=1, default=str), flush=True)
    print("dry run: nothing deployed or started", flush=True)
    return plan


def _volume_json(volume: str, path: str) -> dict[str, Any] | None:
    """A small JSON file from a Modal Volume (no container; ``None`` if the Volume or file is missing)."""
    try:
        vol = modal.Volume.from_name(volume)
        return json.loads(b"".join(vol.read_file(path)))
    except Exception:
        return None


def _volume_dirs(volume: str, path: str) -> list[str]:
    try:
        return [e.path.rstrip("/").rsplit("/", 1)[-1] for e in modal.Volume.from_name(volume).listdir(path)]
    except Exception:
        return []


def cache_preflight(model: str, read_json: Callable[[str, str], dict[str, Any] | None] | None = None,
                    list_dirs: Callable[[str, str], list[str]] | None = None,
                    need_hint: bool = True) -> dict[str, Any]:
    """Before an expensive run of a Volume-pack target (free: reads three small files, starts no container): the
    provider's pack, the gateway cache's pack and its hint exist and carry the same fingerprint. A mismatch (e.g. a
    pack format change between the two preparations) would make the wizard download and pack the checkpoint and
    build the hint on the gateway's CPU, or the run refuse after paying for the H100's start. Returns
    ``{"ok", "problems", "fingerprints"}``."""
    read_json = read_json or _volume_json
    list_dirs = list_dirs or _volume_dirs
    prov = read_json(PROVIDER_PACK_VOLUME, f"{model}/pack/manifest.json")
    gw = read_json(CACHE_VOLUME, f"{model}/pack/manifest.json")
    problems: list[str] = []
    fps = {"provider_pack": (prov or {}).get("fingerprint"), "gateway_pack": (gw or {}).get("fingerprint")}
    if prov is None:
        problems.append(f"no provider pack in Volume {PROVIDER_PACK_VOLUME} ({model}/pack)")
    if gw is None:
        problems.append(f"no gateway pack in Volume {CACHE_VOLUME} ({model}/pack)")
    if prov is not None and gw is not None and fps["provider_pack"] != fps["gateway_pack"]:
        problems.append("the provider's and the gateway cache's packs differ in fingerprint")
    if need_hint:
        hints = [d for d in list_dirs(CACHE_VOLUME, f"{model}/hemask") if not d.endswith(".partial")]
        fp = fps["gateway_pack"] or fps["provider_pack"] or ""
        match = [d for d in hints if fp and d.startswith(fp[:16] + "-")]
        fps["hints"] = hints
        if not match:
            problems.append(f"no encryption hint for fingerprint {fp[:16] or '?'} in Volume {CACHE_VOLUME} "
                            f"({model}/hemask: {hints or 'empty'})")
    return {"ok": not problems, "problems": problems, "fingerprints": fps}


def _print_ab_plain(res: dict[str, Any]) -> None:
    """One summary of the in-run A/B (``--ab-plain``): speculative vs plain decode, trips and the factory."""
    e = res.get("e2e") or {}
    sp = e.get("speed") or {}
    pl = e.get("plain")
    if not pl and not sp.get("stream_spec"):
        return
    ss = sp.get("stream_spec") or {}
    print(f"speculative phase: warm stream decode {sp.get('warm_stream_decode_tokens_per_s')} tok/s, TTFT "
          f"{sp.get('warm_stream_ttft_s')} s, tokens/pass {ss.get('tokens_per_pass')}, drafter hit rate "
          f"{ss.get('drafter_hit_rate')}, draft ms per background step {ss.get('draft_ms_per_bg_step')}", flush=True)
    for name, sus in (("speculative", e.get("sustain")), ("plain", (pl or {}).get("sustain"))):
        if not sus:
            continue
        d = sus.get("derived") or {}
        print(f"{name} sustain: decode token-weighted {sus.get('decode_tokens_per_s_overall')} tok/s (median "
              f"{sus.get('decode_tokens_per_s_median')}, min {sus.get('decode_tokens_per_s_min')}), TTFT median "
              f"{sus.get('ttft_s_median')} s, identical {sus.get('identical')}; trips "
              f"{(sus.get('trips') or {}).get('rows_1_mean_ms')} ms (1-row mean), {d.get('trips_per_token')} "
              f"trips/token, "
              f"gateway CPU {d.get('gateway_cpu_ms_per_trip')} ms/trip; {d.get('slots_per_token')} slots/token; "
              f"factory {d.get('factory_vs_consumption')}; tokens/pass "
              f"{(sus.get('spec') or {}).get('tokens_per_pass')}",
              flush=True)
    if pl:
        print(f"plain phase: restart {pl.get('restart_s')} s, decode {pl.get('decode_tokens_per_s')} tok/s, TTFT "
              f"{pl.get('ttft_s')} s, identical {pl.get('identical')}, trips {pl.get('trips')}, "
              f"{pl.get('trips_per_token')} trips/token {pl.get('error') or ''}", flush=True)


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
    ap.add_argument("--lane", choices=["stream", "tls", "uds"], default=None,
                    help="customer gateway's trip lane (default: 'uds' co-located, 'stream' otherwise): 'stream' = "
                         "the blocking stream lane (WebSocket, msgpack, Python TLS); 'tls' (--placement colocated "
                         "only) = the provider also serves its "
                         "native trip lane over TLS 1.3 on 127.0.0.1:8444 and the gateway uses it (rustls, binary "
                         "header, GIL released per trip; same masked bytes, request ids and schedule; misses fall "
                         "back to the stream lane); 'uds' (--placement colocated only) = the product's same-host "
                         "Unix-socket lane between the two party users (the service accepts the gateway user's uid "
                         "only; socket 0660 and its directory 0710 to that user's group). Exact; the e2e "
                         "comparison still applies")
    ap.add_argument("--gateway-env", action="append", default=[], metavar="KEY=VALUE",
                    help="--placement colocated: extra environment of the customer gateway party (and the mask "
                         "factory it starts), e.g. PGW_FACTORY_SKIP_LINEAR_KINDS=qkv,o,down; PGW_/OMP_/MKL_ keys only")
    ap.add_argument("--provider-env", action="append", default=[], metavar="KEY=VALUE",
                    help="--placement colocated: extra environment of the provider's service (PGW_/OMP_/MKL_ keys)")
    ap.add_argument("--sustain-s", type=float, default=0.0,
                    help="--e2e --placement colocated: after the exactness check, chat back to back for this many "
                         "seconds and report tokens/s and the mask factory's segments finished meanwhile (slots/s "
                         "while a chat decodes)")
    ap.add_argument("--float-ops", choices=("torch", "native"), default=None,
                    help="--placement colocated: the gateway's float path. 'native' = the pinned native formulation "
                         "(pgw_fast: RMSNorm, RoPE, attention, SiLU in a fixed order; PGW_FLOAT_OPS=native), which "
                         "the e2e's plaintext reference then uses too (a new reference). Default: native co-located, "
                         "torch otherwise")
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
    ap.add_argument("--he-bulk", action=argparse.BooleanOptionalAction, default=None,
                    help="--placement colocated: the encrypted mask factory sends its maskgen requests over the "
                         f"provider's bulk channel (threaded TLS sockets on 127.0.0.1:{HE_BULK_PORT}; the same "
                         "request and reply bodies, authentication and admission as POST /pgw/v1/maskgen; a "
                         "transport failure resends over HTTP). Default: on co-located (measured exact), "
                         "--no-he-bulk = HTTP only")
    ap.add_argument("--he-bulk-split", type=int, default=2,
                    help="with --he-bulk: a multi-group maskgen request goes out as up to N concurrent whole-group "
                         "requests (PGW_FACTORY_HE_BULK_SPLIT; default 2)")
    ap.add_argument("--draft-backend", choices=("auto", "native_int8", "torch_int8", "float"), default=None,
                    help="--spec-draft: the draft model's CPU kernels (run time; GatewayModelConfig.draft_backend). "
                         "'native_int8' = the native exact int8 kernels on the gateway's cores (3-5x torch int8)")
    ap.add_argument("--spec-pipeline-steps", type=int, default=None, metavar="M",
                    help="--spec-draft: pipelined drafting (run time; spec_pipeline_steps): the next block is drafted "
                         "while the pass is in flight, M draft steps between passes whatever the acceptance (0 = none "
                         "on the critical path); turns prompt lookup off")
    ap.add_argument("--trip-native", action="store_true",
                    help="--placement colocated: the native per-trip path (PGW_TRIP_NATIVE=1 next to the demo's "
                         "PGW_TRIP_FAST=1; library default off)")
    ap.add_argument("--prime-check", action="store_true",
                    help="--placement colocated: one-vector Freivalds check over F_p (factory "
                         "PGW_FACTORY_CHECK_FIELD=p31, gateway PGW_GATEWAY_PRIME_CHECK). Not with --spec-mask: padding "
                         "rows refuse prime extents")
    ap.add_argument("--masked-head", action="store_true",
                    help="--placement colocated: the masked LM head (opt-in; head_mode = 'provider_int8'): the "
                         "provider serves each pack also under its derived masked-head fingerprint "
                         "(PGW_SERVICE_MASKED_HEAD=also; the int8 head derived from its own pack), the gateway's "
                         "factory makes masks for it (PGW_FACTORY_MASKED_HEAD; its hint extends the base hint), the "
                         "gateway sends the head as one more masked linear per pass and never loads the float head, "
                         "and the e2e's plaintext reference uses the same int8 head (a new pinned reference). "
                         "Memory: provider +vocab x hidden int8 (0.55 GB at 7B), gateway -1.09 GB (bf16 head) at 7B")
    ap.add_argument("--ab-plain", type=float, default=None, metavar="SUSTAIN_S",
                    help="--e2e --placement colocated --spec-rows: after the speculative phase, restart the gateway "
                         "without speculation (same pack, mask store, keys) and measure plain decoding in the same "
                         "container: 3 streamed chats, then SUSTAIN_S seconds back to back (0 = chats only)")
    ap.add_argument("--gateway-cpus", type=float, default=None,
                    help="--placement colocated: the gateway party's cores (deploy time; PGW_MODAL_GATEWAY_CPUS, "
                         "default 12)")
    ap.add_argument("--provider-cpus", type=float, default=None,
                    help="the provider's cores (deploy time; PGW_MODAL_PROVIDER_CPUS; default 2 with a GPU, 4 without)")
    ap.add_argument("--gateway-mem-gib", type=int, default=None,
                    help="--placement colocated: the gateway party's memory (deploy time; default 8/16, 48 for the 7B)")
    ap.add_argument("--provider-mem-gib", type=int, default=None,
                    help="--placement colocated: the provider party's memory (deploy time; default 6, 16 for the 7B)")
    ap.add_argument("--gateway-cache", action="store_true",
                    default=os.environ.get("PGW_MODAL_GATEWAY_CACHE", "0") == "1",
                    help="--placement colocated: the gateway side's cache Volume of public set-up artefacts (the "
                         "target's pack and encrypted-mask hint), restored before the wizard and filled after a run "
                         "(deploy time; PGW_MODAL_GATEWAY_CACHE=1). See README 'Gateway cache'")
    ap.add_argument("--prepare-cache", action="store_true",
                    help="with --gateway-cache: fill the cache for the deployed target on a T4 container (pack + "
                         "hint built on its GPU), then exit; run it once before an expensive co-located run")
    ap.add_argument("--mask-source", choices=MASK_SOURCES, default="factory",
                    help="--placement colocated: where the gateway's masks come from: 'factory' (default; the "
                         "encrypted MLWE mask factory, which needs the hint) or 'gpu-service' (the customer-side GPU "
                         "mask service: a hook, refused until that package wires it; no hint needed then)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the arguments, print the deploy-time sizing, the run-time environment and the "
                         "cost estimate, then exit (no Modal call, nothing deployed or spent)")
    ap.add_argument("--allow-self-draft", action="store_true",
                    help="plumbing tests only: allow --spec-draft with the 0.5B target (the draft is the target "
                         "itself, so acceptance is near 1 and nothing is learned about speed)")
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
        args.lane = "uds" if args.placement == "colocated" else "stream"  # S4: exact, 0.29 vs 0.44 ms per trip
    if args.lane in ("tls", "uds") and args.placement != "colocated":
        ap.error(f"--lane {args.lane} needs --placement colocated (the provider's lane is on its host only)")
    if args.float_threads is None:
        args.float_threads = 1 if args.placement == "colocated" else 0
    if args.float_threads < 0 or (args.float_threads and args.placement != "colocated"):
        ap.error("--float-threads needs --placement colocated and N >= 0")
    if args.float_ops is None:
        args.float_ops = "native" if args.placement == "colocated" else "torch"
    if args.float_ops != "torch" and args.placement != "colocated":
        ap.error("--float-ops needs --placement colocated")
    if args.gateway_gpu and (args.placement != "colocated" or not args.gpu):
        ap.error("--gateway-gpu needs --placement colocated and a provider --gpu (two GPUs of that tier)")
    if args.max_trip_ms is not None and args.max_trip_ms <= 0:
        ap.error("--max-trip-ms must be positive")
    args.model = args.model.strip().lower()
    if args.model not in MODELS:
        ap.error(f"--model / PGW_MODAL_MODEL: use one of {', '.join(MODELS)}")
    if args.model in COLOCATED_ONLY_MODELS and args.placement != "colocated":
        ap.error(f"--model {args.model} is served by --placement colocated only (memory and disk are sized there)")
    if args.he_bulk is None:
        args.he_bulk = args.placement == "colocated"
    if args.he_bulk and args.placement != "colocated":
        ap.error("--he-bulk needs --placement colocated (the provider's bulk port is on its host only)")
    if not 1 <= args.he_bulk_split <= 8:
        ap.error("--he-bulk-split must be in [1, 8]")
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
    if args.spec_draft and args.model == DEFAULT_MODEL and not args.allow_self_draft:
        # the target itself as its own draft: the gateway would compute the whole model in plaintext
        ap.error("--spec-draft drafts with the 0.5B itself: use a larger --model (the draft must be much cheaper "
                 "than the target, or the provider computes nothing the gateway does not)")
    if (args.draft_backend or args.spec_pipeline_steps is not None) and not args.spec_draft:
        ap.error("--draft-backend / --spec-pipeline-steps need --spec-draft (the draft model)")
    if args.spec_pipeline_steps is not None and not 0 <= args.spec_pipeline_steps <= max(0, args.spec_rows - 1):
        ap.error("--spec-pipeline-steps must be in [0, --spec-rows - 1]")
    colo_only = {"--trip-native": args.trip_native, "--prime-check": args.prime_check,
                 "--masked-head": args.masked_head,
                 "--ab-plain": args.ab_plain is not None, "--gateway-cache": args.gateway_cache,
                 "--gateway-cpus": args.gateway_cpus is not None, "--gateway-mem-gib": args.gateway_mem_gib is not None,
                 "--provider-mem-gib": args.provider_mem_gib is not None}
    for flag, on in colo_only.items():
        if on and args.placement != "colocated":
            ap.error(f"{flag} needs --placement colocated")
    if args.prime_check and args.spec_mask:
        ap.error("--prime-check with --spec-mask: padding rows refuse prime-field extents (fail closed); use the "
                 "ring check with padded speculation")
    if args.ab_plain is not None and (not args.e2e or not args.spec_rows or args.ab_plain < 0):
        ap.error("--ab-plain needs --e2e and --spec-rows (it measures the same run without speculation), and >= 0")
    if args.prepare_cache and not args.gateway_cache:
        ap.error("--prepare-cache needs --gateway-cache")
    if args.mask_source != "factory":
        if args.placement != "colocated":
            ap.error("--mask-source needs --placement colocated")
        if not GPU_MASK_SERVICE_READY:
            ap.error("--mask-source gpu-service: the GPU mask service is not wired into the Modal demo yet "
                     "(hook: launch.GPU_MASK_SERVICE_READY / pgw_modal_boot.mask_source_env)")
    if args.model in MODEL_GPUS and not (args.stop or args.show) and args.gpu not in MODEL_GPUS[args.model]:
        ap.error(f"--model {args.model} needs --gpu {' or '.join(MODEL_GPUS[args.model])} (its int8 linears and the "
                 "maskgen scratch on one device)")
    if args.model in VOLUME_PACK_MODELS and not args.gateway_cache and not (args.stop or args.show):
        ap.error(f"--model {args.model} needs --gateway-cache (its pack and 114 GB encryption hint are prepared once "
                 "with --prepare-cache; built in the run they would not fit the TTL)")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # the deploy-time choices also drive the cost estimate (customer memory) in this process
    for var, val in (("PGW_MODAL_MODEL", args.model if args.model != DEFAULT_MODEL else None),
                     ("PGW_MODAL_DRAFT", "1" if args.spec_draft else None),
                     ("PGW_MODAL_GATEWAY_GPU", "1" if args.gateway_gpu else None),
                     ("PGW_MODAL_SPEC_MASK", args.spec_mask.strip() or None),
                     ("PGW_MODAL_GATEWAY_CACHE", "1" if args.gateway_cache else None),
                     ("PGW_MODAL_GATEWAY_CPUS", f"{args.gateway_cpus:g}" if args.gateway_cpus else None),
                     ("PGW_MODAL_PROVIDER_CPUS", f"{args.provider_cpus:g}" if args.provider_cpus
                      else os.environ.get("PGW_MODAL_PROVIDER_CPUS")),
                     ("PGW_MODAL_GATEWAY_MEM_GIB", str(args.gateway_mem_gib) if args.gateway_mem_gib else None),
                     ("PGW_MODAL_PROVIDER_MEM_GIB", str(args.provider_mem_gib) if args.provider_mem_gib else None)):
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
    if args.placement == "colocated":
        c = colocated_cost_per_h(args.gpu, args.model)
        print(f"co-located container: {args.gpu or 'no GPU'}, {c['provider_cpus']:g} + {c['gateway_cpus']:g} cores, "
              f"{c['memory_gib']:g} GiB: about ${c['total']:.2f}/h at list prices (GPU ${c['gpu']:.2f}, cores "
              f"${c['cores']:.2f}, memory ${c['memory']:.2f}); at most ${c['total'] * min(args.ttl_min, 85) / 60:.2f} "
              f"for --ttl-min {args.ttl_min:g}", flush=True)
    if args.prepare_cache:
        pc = prepare_cost_per_h(args.model)
        print(f"--prepare-cache: {', '.join(f'{k} about ${v:.2f}/h' for k, v in pc.items())} at list prices, billed "
              "while they run (estimate in README)", flush=True)
    if args.dry_run:
        dry_run(args)
        return
    if (args.model in VOLUME_PACK_MODELS and not args.prepare_cache and not args.deploy_only
            and not (args.stop or args.show)):
        pf = cache_preflight(args.model, need_hint=args.mask_source == "factory")
        if not pf["ok"]:
            raise SystemExit(f"{args.model}: not prepared ({'; '.join(pf['problems'])}); run the same flags with "
                             "--prepare-cache first (nothing was deployed or started)")
        print(f"cache preflight ok: {pf['fingerprints']}", flush=True)
    if args.deploy_only:
        deploy(args.gpu, args.customer_gpu, args.placement, args.region, args.model, args.spec_draft)
        print("deployed; start a pair later with --no-deploy")
        return
    if args.prepare_cache:
        if not args.no_deploy:
            deploy(args.gpu, args.customer_gpu, args.placement, args.region, args.model, args.spec_draft)
        prepare_cache()
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
