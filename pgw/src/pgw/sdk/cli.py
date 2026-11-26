"""``pgw sdk status`` and ``pgw sdk doctor``: the client library's state and environment, without secrets.

``status`` reads the cache directory only (no network): enrolled sessions, provisioned models, the mask stock,
the factories (running, halted, provider integrity failures) and the last integrity check of each model.
``doctor`` checks the installation: native kernels of the proven stack, directories and permissions, CPU and
disk, the API key's format (never printed), and with ``--online`` the provider's CA against the key's pin.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import click

__all__ = ["doctor_report", "sdk_group", "status_report"]


@click.group("sdk")
def sdk_group() -> None:
    """The in-process client library (``from pgw import OpenAI``)."""


def _layout(cache_dir: Path | None, state_dir: Path | None) -> Any:
    from pgw.sdk.cache import CacheLayout

    return CacheLayout(cache_dir, state_dir)


def _mask_stock(layout: Any, key_id: str) -> dict[str, Any] | None:
    """Available slots per (fingerprint, ring) of a session's mask store (opens it with its KEK; read only)."""
    root = layout.masks_dir(key_id)
    kek = layout.session_state_dir(key_id) / "masks.kek"
    if not (root / "state.sqlite").exists() or not kek.exists():
        return None
    try:
        from pgw.maskstore.admin import open_store
        from pgw.maskstore.config import MaskStoreSettings

        s = MaskStoreSettings(root=root, key_provider="file", key_file=kek, counter="file",
                              counter_file=layout.session_state_dir(key_id) / "masks.counter",
                              verify_on_open=False, lock_timeout_s=10.0)
        with open_store(s) as store:
            st = store.status()
        inv = st.get("inventory") or st.get("models") or []
        return {"inventory": inv, "generation": st.get("generation")}
    except Exception as exc:  # status must never fail on a busy or locked store
        return {"error": getattr(exc, "message", type(exc).__name__)}


def status_report(cache_dir: Path | None = None, state_dir: Path | None = None, *,
                  masks: bool = True) -> dict[str, Any]:
    """The data of ``pgw sdk status``."""
    from pgw.sdk.engine import read_status

    lay = _layout(cache_dir, state_dir)
    rep = read_status(lay)
    if masks:
        for s in rep["sessions"]:
            stock = _mask_stock(lay, s["key_id"])
            if stock is not None:
                s["mask_stock"] = stock
    return rep


def _age(t: Any) -> str:
    if not isinstance(t, int | float) or not t:
        return "never"
    d = max(0.0, time.time() - float(t))
    for unit, n in (("d", 86400), ("h", 3600), ("min", 60)):
        if d >= n:
            return f"{d / n:.0f} {unit} ago"
    return f"{d:.0f} s ago"


def _print_status(rep: dict[str, Any]) -> None:
    click.echo(f"cache  {rep['cache_dir']}")
    click.echo(f"state  {rep['state_dir']}")
    if not rep["sessions"]:
        click.echo("no enrolled API key yet (the first request of a client enrolls)")
        return
    for s in rep["sessions"]:
        e = s.get("enrollment") or {}
        click.echo(f"\nkey {s['key_id']}  provider {e.get('service_url', '?')}  tenant {e.get('tenant', '?')}  "
                   f"enrolled {_age(e.get('enrolled_at'))}")
        click.echo(f"  provider models: {', '.join(e.get('models') or []) or '-'}")
        models = (s.get("status") or {}).get("models") or {}
        for name, m in sorted(models.items()):
            prov = m.get("provisioned") or {}
            provd = f"yes ({prov.get('source', '?')})" if prov else "no"
            click.echo(f"  {name}: {m.get('state')}  provisioned={provd}  masks={m.get('masks_available', '?')}  "
                       f"lane={m.get('lane', '?')}  masked_head={m.get('masked_head')}")
            click.echo(f"    integrity: last check ok {_age(m.get('last_integrity_check_ok_at'))}, "
                       f"{m.get('verified_requests', 0)} verified request(s), "
                       f"{m.get('integrity_failures', 0)} failure(s)"
                       + (f", last failure {_age(m.get('last_integrity_failure_at'))}"
                          if m.get("last_integrity_failure_at") else ""))
            if m.get("state") not in ("ready", None) and m.get("progress"):
                click.echo(f"    preparing: {m['progress']}")
            if m.get("state") == "ready" and m.get("masks_available") in (0, None) and models:
                click.echo("    (no masks yet: on first use the background factory prepares the one-time mask setup;"
                           " see the factory line)")
            if m.get("error"):
                click.echo(f"    error: {m['error']}")
        for name, hrec in sorted((s.get("integrity_halts") or {}).items()):
            if hrec.get("halted"):
                click.echo(f"  {name}: HALTED (persisted): {hrec.get('reason')}; "
                           f"clear with: pgw sdk clear-halt --model {name}")
        for fp, f in sorted((s.get("factories") or {}).items()):
            click.echo(f"  factory {fp}: halted={f.get('halted')}  provider_integrity_failures="
                       f"{f.get('provider_integrity_failures', 0)}  last run {_age(f.get('last_run_at'))}"
                       + (f"  last error: {f['last_error']}" if f.get("last_error") else ""))
        stock = s.get("mask_stock")
        if stock:
            if "error" in stock:
                click.echo(f"  mask store: {stock['error']}")
            else:
                for it in stock.get("inventory") or []:
                    if isinstance(it, dict):
                        click.echo(f"  mask stock {str(it.get('fingerprint', ''))[:12]} {it.get('ring', '')}: "
                                   f"{it.get('slots_available', it.get('available', '?'))} slot(s)")


@sdk_group.command("status")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None,
              help="Default: PGW_CACHE_DIR or ~/.cache/pgw")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None, help="Default: PGW_STATE_DIR")
@click.option("--no-masks", is_flag=True, help="Do not open the mask stores (faster)")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
def status_cmd(cache_dir: Path | None, state_dir: Path | None, no_masks: bool, as_json: bool) -> None:
    """Provisioning, mask stock, factories and the last integrity check (reads the cache; no network)."""
    rep = status_report(cache_dir, state_dir, masks=not no_masks)
    if as_json:
        click.echo(json.dumps(rep, indent=2, default=str))
    else:
        _print_status(rep)


# ---- doctor --------------------------------------------------------------------------------------------
def _check(name: str, ok: bool | None, detail: str) -> dict[str, Any]:
    return {"check": name, "status": "ok" if ok else ("warn" if ok is None else "fail"), "detail": detail}


def doctor_report(cache_dir: Path | None = None, state_dir: Path | None = None, *, api_key: str | None = None,
                  online: bool = False) -> dict[str, Any]:
    """The checks of ``pgw sdk doctor`` (``ok`` is false when any check failed)."""
    import stat as st

    checks: list[dict[str, Any]] = []
    checks.append(_check("python", sys.version_info >= (3, 12), sys.version.split()[0]))
    try:
        import pgw_fast

        need = ("TripLease", "DecodeDriver", "trip_prime_output", "TripChannel")
        missing = [n for n in need if not hasattr(pgw_fast, n)]
        tls = hasattr(getattr(pgw_fast, "TripChannel", None), "connect_service_tls")
        checks.append(_check("native kernels", not missing,
                             "pgw_fast: native trip / decode drivers and p31 checks present" if not missing
                             else f"pgw_fast lacks {', '.join(missing)} (slower Python path; still exact)"))
        checks.append(_check("TLS trip lane", True if tls else None,
                             "native TCP+TLS lane client present" if tls else "no native TLS lane (stream lane used)"))
    except ImportError:
        checks.append(_check("native kernels", None, "pgw_fast not installed: the Python path is used (exact, slower)"))
    try:
        import pgw_ntt  # noqa: F401

        checks.append(_check("pks-v3 kernels", True, "pgw_ntt present"))
    except ImportError:
        checks.append(_check("pks-v3 kernels", None, "pgw_ntt not installed (mask factory uses slower kernels)"))
    cpus = os.cpu_count() or 1
    from pgw.sdk.options import default_mask_workers

    checks.append(_check("cpus", cpus >= 2, f"{cpus} CPU(s); default mask workers {default_mask_workers(cpus)}"))
    lay = _layout(cache_dir, state_dir)
    for label, d in (("cache dir", lay.root), ("state dir", lay.state)):
        if d.exists():
            mode = st.S_IMODE(d.stat().st_mode)
            own = (not hasattr(os, "getuid")) or d.stat().st_uid == os.getuid()
            checks.append(_check(label, mode == 0o700 and own, f"{d} (mode {mode:o}{'' if own else ', other owner'})"))
        else:
            checks.append(_check(label, True, f"{d} (created on first use, 0700)"))
    probe = lay.root if lay.root.exists() else lay.root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    checks.append(_check("disk", True if free >= 8 << 30 else None,
                         f"{free / 2**30:.1f} GiB free at {probe} (models and the mask stock need several GiB)"))
    key_text = api_key if api_key is not None else os.environ.get("PGW_API_KEY")
    key = None
    if key_text:
        from pgw.sdk.keys import parse_key

        try:
            key = parse_key(key_text)
            pin = f"pinned {key.ca_fingerprint_hex[:16]}..." if key.pinned else "not pinned"
            checks.append(_check("api key", True,
                                 f"valid format: key id {key.key_id_hex}, provider {key.service_url}, CA {pin}"))
        except Exception as exc:
            checks.append(_check("api key", False, getattr(exc, "message", "invalid key")))
    else:
        checks.append(_check("api key", None, "PGW_API_KEY not set (pass api_key= to the client)"))
    if online and key is not None:
        from pgw.enroll.client import _fetch_pinned_ca

        try:
            _fetch_pinned_ca(key, None, 15.0)  # type: ignore[arg-type]
            checks.append(_check("provider", True, f"{key.service_url} reachable; CA matches the key's pin"
                                 if key.pinned else f"{key.service_url} reachable (no pin: loopback http)"))
        except Exception as exc:
            checks.append(_check("provider", False, getattr(exc, "message", type(exc).__name__)))
    return {"ok": all(c["status"] != "fail" for c in checks), "checks": checks}


@sdk_group.command("doctor")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None)
@click.option("--state-dir", type=click.Path(path_type=Path), default=None)
@click.option("--online", is_flag=True, help="Also contact the provider and check its CA against the key's pin")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
def doctor_cmd(cache_dir: Path | None, state_dir: Path | None, online: bool, as_json: bool) -> None:
    """Check the installation for the client library (the API key comes from PGW_API_KEY and is never printed)."""
    rep = doctor_report(cache_dir, state_dir, online=online)
    if as_json:
        click.echo(json.dumps(rep, indent=2))
    else:
        for c in rep["checks"]:
            click.echo(f"[{c['status']:>4}] {c['check']}: {c['detail']}")
        click.echo("ok" if rep["ok"] else "problems found")
    if not rep["ok"]:
        sys.exit(1)


# ---- clear-halt ----------------------------------------------------------------------------------------
_CLEAR_WARNING = (
    "WARNING: this re-opens masked traffic to a provider whose answers FAILED the integrity checks.\n"
    "An honest provider never fails them: a failure means wrong or tampered results. Clear the halt only after\n"
    "investigating (provider incident report, service logs, a changed provider deployment). Processes that are\n"
    "already running keep the model blocked until they restart or call reset_integrity_breaker().")


@sdk_group.command("clear-halt")
@click.option("--model", "model", required=True, help="Model whose integrity halt to clear")
@click.option("--key-id", default=None, help="Only this SDK key (16 hex digits, see pgw sdk status); default: every "
              "key with a halt of the model")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None,
              help="Default: PGW_STATE_DIR, else $XDG_STATE_HOME/pgw or ~/.local/state/pgw")
@click.option("--note", default="", help="Why the halt is cleared (kept in the halt record's event log)")
@click.option("--yes", is_flag=True, help="Confirm that the provider was investigated")
def clear_halt_cmd(model: str, key_id: str | None, state_dir: Path | None, note: str, yes: bool) -> None:
    """Clear the persisted integrity halt of a model (operator action, after investigating the provider).

    After an integrity failure of the provider the client library blocks the model for this API key, and the block
    survives restarts (it is stored, MAC-protected, in the state directory). A halt record that was modified or
    whose MAC key is missing also blocks the model (fail closed) and is cleared the same way."""
    import re

    from pgw.sdk.halt import clear_halt
    from pgw.sdk.options import canonical_model

    try:
        name = canonical_model(model)
    except Exception as exc:
        raise click.BadParameter(getattr(exc, "message", str(exc)), param_hint="--model") from None
    if key_id is not None and not re.fullmatch(r"[0-9a-f]{16}", key_id):
        raise click.BadParameter("expected 16 lowercase hex digits", param_hint="--key-id")
    click.echo(_CLEAR_WARNING, err=True)
    if not yes:
        raise click.UsageError("investigate the compute provider first, then pass --yes")
    lay = _layout(None, state_dir)
    cleared = clear_halt(lay.state, name, key_id=key_id, note=note)
    if not cleared:
        click.echo(f"no integrity halt of {name} recorded under {lay.state}")
        return
    for c in cleared:
        extra = f" (record was unverifiable: {c['tampered']})" if c.get("tampered") else ""
        click.echo(f"cleared {name} for key {c['key_id']}: {c['failures']} integrity failure(s) recorded{extra}")


# ---- prepare -------------------------------------------------------------------------------------------
@sdk_group.command("prepare")
@click.option("--model", "models", multiple=True, required=True, help="Model to prepare (repeatable)")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None,
              help="Default: PGW_CACHE_DIR or ~/.cache/pgw")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None, help="Default: PGW_STATE_DIR")
@click.option("--no-hint", is_flag=True, help="Artifacts and weights only (the factory builds the hint on first use)")
@click.option("--threads", type=int, default=0, help="CPU threads for the hint build (default: every core)")
@click.option("--masked-head", type=click.Choice(["auto", "on", "off"]), default=None,
              help="As the client's masked_head option (default auto); use the clients' setting")
@click.option("--pin", "pins", multiple=True, metavar="MODEL=FINGERPRINT",
              help="Extra fingerprint pin, as the client's model_fingerprints (repeatable)")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable summary")
def prepare_cmd(models: tuple[str, ...], cache_dir: Path | None, state_dir: Path | None, no_hint: bool, threads: int,
                masked_head: str | None, pins: tuple[str, ...], as_json: bool) -> None:
    """One-time setup of a model ahead of the first request: artifacts, weights and the pks-v3 hint, on every core,
    resumable (run it again after an interruption). The API key comes from PGW_API_KEY and is never printed.

    To set up on a big host once and copy: run this there, then copy <cache>/models/<fingerprint[:16]> and
    <cache>/keys/pks-hint.key to the same places on the application hosts (details: pgw.sdk.prepare)."""
    from pgw.sdk.options import SDKOptions
    from pgw.sdk.prepare import prepare

    if threads > 0:
        os.environ["PGW_PKS_THREADS"] = str(threads)
        import torch

        torch.set_num_threads(threads)
    fps: dict[str, str] = {}
    for p in pins:
        name, sep, fp = p.partition("=")
        if not sep or not name.strip() or not fp.strip():
            raise click.BadParameter(f"--pin expects MODEL=FINGERPRINT, not {p!r}")
        fps[name.strip().lower()] = fp.strip().lower()
    opts = SDKOptions.from_env(cache_dir=cache_dir, state_dir=state_dir, masked_head=masked_head,
                               model_fingerprints=fps or None)
    last: dict[str, Any] = {"t": 0.0, "stage": ""}

    def say(line: str) -> None:
        now, stage = time.monotonic(), line.split(":")[0]
        if stage != last["stage"] or now - float(last["t"]) > 2.0:
            last.update(t=now, stage=stage)
            click.echo(f"  {line}", err=True)

    out = []
    for m in models:
        click.echo(f"preparing {m} ...", err=True)
        r = prepare(m, options=opts, progress=say, build_hint=False if no_hint else None)
        out.append(r)
        if not as_json:
            click.echo(f"{r['model']}: fingerprint {r['fingerprint'][:12]}, masked head {r['masked_head']}, hint "
                       f"{'ready' if r['hint_dir'] else 'not built (the factory builds it on first use)'}, "
                       f"{r['seconds']:.0f} s; pack {r.get('pack_bytes', 0) / 1e9:.2f} GB, factory pack "
                       f"{r.get('factory_pack_bytes', 0) / 1e9:.2f} GB, hint {r.get('hint_dir_bytes', 0) / 1e9:.2f} GB")
    if as_json:
        click.echo(json.dumps(out, indent=2))
