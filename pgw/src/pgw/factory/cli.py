"""``pgw factory`` commands: ``generate``, ``run``, ``status``, ``benchmark``.

Mounted into the main CLI as ``pgw factory`` (see ``INTERFACE_NOTES.md``) and runnable standalone
as ``python -m pgw.factory``. Settings come from ``PGW_FACTORY_*`` / ``PGW_MASKSTORE_*``; command
line options override them. Exit codes: 0 ok, 1 :class:`pgw.core.PGWError`, 3 not implemented
(a dependency such as the mask store is unfinished), 130 interrupted.
"""

from __future__ import annotations

import contextlib
import functools
import json
import signal
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import click
from pydantic import ValidationError

from pgw.core.errors import ConfigError, PGWError
from pgw.core.fingerprint import short_fingerprint
from pgw.factory.governor import FactoryStopped, Throttle

__all__ = ["factory_group", "main"]

F = TypeVar("F", bound=Callable[..., Any])


def _handle_errors(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except FactoryStopped:
            click.echo("stopped; the partial segment was discarded", err=True)
            sys.exit(130)
        except PGWError as exc:
            click.echo(f"error [{exc.code}]: {exc.message}", err=True)
            sys.exit(1)
        except ValidationError as exc:
            click.echo(f"error [config]: {exc}", err=True)
            sys.exit(1)
        except KeyboardInterrupt:
            # the partial segment (if any) was aborted by generate_into_store
            click.echo("interrupted", err=True)
            sys.exit(130)

    return wrapper  # type: ignore[return-value]


def _factory_settings(**overrides: Any) -> Any:
    from pgw.factory.config import FactorySettings

    return FactorySettings(**{k: v for k, v in overrides.items() if v is not None and v != ()})


def _store_overrides(store_root: Path | None, key_file: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if store_root is not None:
        out["root"] = store_root
    if key_file is not None:
        out["key_file"] = key_file
        out["key_provider"] = "file"
    return out


@contextlib.contextmanager
def _signal_handlers(stop: Callable[[], None]) -> Iterator[None]:
    """Route SIGINT/SIGTERM to ``stop`` (graceful abort of the current segment); restore on exit."""
    if threading.current_thread() is not threading.main_thread():  # pragma: no cover - e.g. worker threads
        yield
        return

    def handler(signum: int, frame: Any) -> None:
        click.echo(f"received signal {signum}; stopping after aborting the current segment", err=True)
        stop()

    previous = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, h in previous.items():
            signal.signal(sig, h)


@click.group("factory", context_settings={"help_option_names": ["-h", "--help"]})
def factory_group() -> None:
    """Mask factory (customer-run or independent partner): generate one-time masks offline."""
    from pgw.core.proc_tuning import apply_malloc_tuning

    apply_malloc_tuning()  # large mask buffers: keep them on the heap (PGW_MALLOC_TUNING=0 turns it off)


_common = [
    click.option("--ring", "rings", multiple=True, help="Ring preset(s), e.g. K32A12C27 (repeatable)"),
    click.option("--device", default=None, help="auto | cpu | cuda | cuda:N"),
    click.option("--backend", type=click.Choice(["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"]), default=None),
    click.option("--chunk-slots", type=int, default=None, help="Rows per GEMM batch / encrypted chunk"),
    click.option("--checks", type=int, default=None, help="Freivalds vectors per linear"),
    click.option("--store", "store_root", type=click.Path(path_type=Path), default=None, help="Mask store root"),
    click.option("--key-file", type=click.Path(path_type=Path), default=None, help="Store KEK file (file provider)"),
    click.option("--export-dir", type=click.Path(path_type=Path), default=None, help="Partner mode: export directory"),
    click.option("--recipient-key", type=click.Path(path_type=Path), default=None,
                 help="Partner mode: customer X25519 public key file"),
    click.option("--partner-key", type=click.Path(path_type=Path), default=None,
                 help="Partner mode: Ed25519 key ('pgw maskstore partner-key create') signing each segment"),
    click.option("--destination-store", "destination_store", default=None,
                 help="Partner mode: store_id of the customer mask store the segments are for"),
    click.option("--he-service", "he_service", default=None,
                 help="Encrypted mode: compute service URL that multiplies MLWE-encrypted masks (no local GEMMs)"),
    click.option("--he-token-file", "he_token_file", type=click.Path(path_type=Path), default=None,
                 help="Encrypted mode: service bearer token file"),
    click.option("--he-hint-dir", "he_hint_dir", type=click.Path(path_type=Path), default=None,
                 help="Encrypted mode: directory of the one-time hint W8.A"),
    click.option("--schedule", "schedule", type=click.Choice(["standard", "fused2"]), default=None,
                 help="Masks for the gateway's schedule: standard (default) or fused2 (pair-mask segments under the "
                      "fused2 lane fingerprint; ring e.g. K56A12C53)"),
]


def _with_common(fn: F) -> F:
    for opt in reversed(_common):
        fn = opt(fn)
    return fn


def _common_overrides(rings: tuple[str, ...], device: str | None, backend: str | None, chunk_slots: int | None,
                      checks: int | None, export_dir: Path | None, recipient_key: Path | None,
                      partner_key: Path | None = None, destination_store: str | None = None,
                      he_service: str | None = None, he_token_file: Path | None = None,
                      he_hint_dir: Path | None = None, schedule: str | None = None) -> dict[str, Any]:
    o: dict[str, Any] = {"device": device, "backend": backend, "chunk_slots": chunk_slots, "checks": checks,
                         "schedule": schedule}
    if he_service is not None:
        o["he_service_url"] = he_service
        o["he_token_file"] = he_token_file
        o["he_hint_dir"] = he_hint_dir
    if rings:
        o["rings"] = tuple(rings)
        o["ring"] = rings[0]
    if export_dir is not None:
        o["output"] = "export"
        o["export_dir"] = export_dir
        o["recipient_public_key_file"] = recipient_key
        if partner_key is not None:
            o["partner_signing_key_file"] = partner_key
        if destination_store is not None:
            o["destination_store_id"] = destination_store
    elif partner_key is not None or destination_store is not None:
        raise click.UsageError("--partner-key / --destination-store only apply to partner mode (--export-dir)")
    return o


@factory_group.command("generate")
@click.option("--pack", "packs", multiple=True, type=click.Path(path_type=Path), help="Model pack (repeatable)")
@click.option("--slots", type=int, default=None, help="Slots per segment")
@click.option("--segments", type=int, default=1, show_default=True)
@_with_common
@_handle_errors
def generate_cmd(packs: tuple[Path, ...], slots: int | None, segments: int, rings: tuple[str, ...], device: str | None,
                 backend: str | None, chunk_slots: int | None, checks: int | None, store_root: Path | None,
                 key_file: Path | None, export_dir: Path | None, recipient_key: Path | None,
                 partner_key: Path | None, destination_store: str | None, he_service: str | None = None,
                 he_token_file: Path | None = None, he_hint_dir: Path | None = None,
                 schedule: str | None = None) -> None:
    """Generate SEGMENTS fresh segments per pack and ring, then exit."""
    from pgw.factory.runtime import build_generators, build_sink, open_packs, open_store

    if segments <= 0:
        raise ConfigError("--segments must be positive")
    s = _factory_settings(
        pack_path=packs[0] if packs else None, extra_pack_paths=tuple(packs[1:]) or None, segment_slots=slots,
        **_common_overrides(rings, device, backend, chunk_slots, checks, export_dir, recipient_key, partner_key,
                                destination_store, he_service, he_token_file, he_hint_dir, schedule),
    )
    store = None if s.output == "export" else open_store(**_store_overrides(store_root, key_file))
    sink = build_sink(s, store)
    throttle = Throttle()
    gens = build_generators(s, open_packs(s), throttle)
    with _signal_handlers(throttle.stop_event.set):
        for (fp, ring), gen in gens.items():
            for i in range(segments):
                seg = gen.generate_into_store(sink, s.segment_slots, checks=s.checks, chunk_slots=s.chunk_slots,
                                              factory_id=s.factory_id)
                click.echo(f"{short_fingerprint(fp)} {ring} segment {i + 1}/{segments}: {seg} "
                           f"({s.segment_slots} slots, {gen.tokens_per_second():.1f} slots/s)")


@factory_group.command("run")
@click.option("--pack", "packs", multiple=True, type=click.Path(path_type=Path), help="Model pack (repeatable)")
@click.option("--once", is_flag=True, help="Run a single plan and exit")
@click.option("--segment-slots", type=int, default=None)
@click.option("--low-water", type=int, default=None, help="Refill below this many available slots")
@click.option("--target", type=int, default=None, help="Refill up to this many slots")
@click.option("--max-slots-per-hour", type=float, default=None, help="Rate limit (0 = unlimited)")
@click.option("--duty-cycle", type=float, default=None, help="Fraction of time computing, (0, 1]")
@click.option("--nice", type=int, default=None, help="CPU niceness increment (low priority)")
@click.option("--threads", "cpu_threads", type=int, default=None, help="torch CPU threads")
@click.option("--poll", "poll_interval_s", type=float, default=None, help="Inventory poll interval (s)")
@click.option("--workers", type=int, default=None,
              help="Worker processes feeding the store concurrently (pgw.factory.fleet; default "
                   "PGW_FACTORY_WORKERS or 1)")
@_with_common
@_handle_errors
def run_cmd(packs: tuple[Path, ...], once: bool, segment_slots: int | None, low_water: int | None, target: int | None,
            max_slots_per_hour: float | None, duty_cycle: float | None, nice: int | None, cpu_threads: int | None,
            poll_interval_s: float | None, workers: int | None, rings: tuple[str, ...], device: str | None,
            backend: str | None,
            chunk_slots: int | None, checks: int | None, store_root: Path | None, key_file: Path | None,
            export_dir: Path | None, recipient_key: Path | None,
                 partner_key: Path | None, destination_store: str | None, he_service: str | None = None,
                 he_token_file: Path | None = None, he_hint_dir: Path | None = None,
                 schedule: str | None = None) -> None:
    """Keep inventories between low water and target (settings from PGW_FACTORY_* / PGW_MASKSTORE_*)."""
    s = _factory_settings(
        pack_path=packs[0] if packs else None, extra_pack_paths=tuple(packs[1:]) or None,
        segment_slots=segment_slots, low_water_slots=low_water, target_slots=target,
        max_slots_per_hour=max_slots_per_hour, duty_cycle=duty_cycle, nice=nice, cpu_threads=cpu_threads,
        poll_interval_s=poll_interval_s, workers=workers,
        **_common_overrides(rings, device, backend, chunk_slots, checks, export_dir, recipient_key, partner_key,
                                destination_store, he_service, he_token_file, he_hint_dir, schedule),
    )
    _run_factory(s, once, store_root, key_file)


def _run_factory(s: Any, once: bool, store_root: Path | None, key_file: Path | None) -> None:
    """Body of ``run`` and ``mask-service``: a fleet, a single scheduler forever, or one plan (``once``)."""
    from pgw.factory.governor import apply_process_priority
    from pgw.factory.runtime import build_scheduler, open_store

    if s.workers > 1:
        if once:
            raise click.UsageError("--once runs a single process; use --workers 1")
        from pgw.factory.fleet import run_fleet

        click.echo(f"starting a factory fleet of {s.workers} worker processes", err=True)
        run_fleet(s, _store_overrides(store_root, key_file))
        return
    apply_process_priority(s.nice, s.cpu_threads)
    store = None if s.output == "export" else open_store(**_store_overrides(store_root, key_file))
    sched = build_scheduler(s, store)
    health = None
    if s.metrics_port is not None and not once:
        from pgw.factory.health import FactoryHealthServer

        health = FactoryHealthServer(sched, s.metrics_host, s.metrics_port).start()
    with _signal_handlers(sched.stop), contextlib.ExitStack() as stack:
        if health is not None:
            stack.callback(health.stop)
        if once:
            with sched:
                ids = sched.run_once()  # FactoryStopped -> exit 130 via _handle_errors
            click.echo(f"produced {len(ids)} segment(s)")
            for i in ids:
                click.echo(f"  {i}")
            return
        sched.run_forever()


@factory_group.command("mask-service")
@click.option("--pack", "packs", multiple=True, type=click.Path(path_type=Path), help="Model pack (repeatable)")
@click.option("--once", is_flag=True, help="Run a single refill plan and exit")
@click.option("--check-only", is_flag=True, help="Run the bit-exact start-up check against the CPU reference and exit")
@click.option("--allow-cpu", is_flag=True, default=None,
              help="Allow a CPU device (measurement, small models); default: refuse without a GPU")
@click.option("--segment-slots", type=int, default=None)
@click.option("--low-water", type=int, default=None, help="Refill below this many available slots")
@click.option("--target", type=int, default=None, help="Refill up to this many slots")
@click.option("--check-field", type=click.Choice(["ring", "p31"]), default=None, help="Segment check format")
@click.option("--json", "as_json", is_flag=True, help="--check-only: print the report as JSON")
@_with_common
@_handle_errors
def mask_service_cmd(packs: tuple[Path, ...], once: bool, check_only: bool, allow_cpu: bool | None,
                     segment_slots: int | None, low_water: int | None, target: int | None, check_field: str | None,
                     as_json: bool, rings: tuple[str, ...], device: str | None, backend: str | None,
                     chunk_slots: int | None, checks: int | None, store_root: Path | None, key_file: Path | None,
                     export_dir: Path | None, recipient_key: Path | None, partner_key: Path | None,
                     destination_store: str | None, he_service: str | None = None, he_token_file: Path | None = None,
                     he_hint_dir: Path | None = None, schedule: str | None = None) -> None:
    """Customer-cloud GPU mask service: plain W8 r on this machine's GPU from the public pack (opt-in mode).

    Refuses to start unless a throw-away segment generated on the device is bit-identical to an independent
    CPU reference. Output: the gateway's mask store on this host (--store) or signed, HPKE-sealed export files
    for a gateway on another VM (--export-dir ... ; serve them with 'pgw factory outbox-serve')."""
    if he_service is not None:
        raise click.UsageError("the mask service computes masks locally; --he-service does not apply")
    s = _factory_settings(
        pack_path=packs[0] if packs else None, extra_pack_paths=tuple(packs[1:]) or None,
        segment_slots=segment_slots, low_water_slots=low_water, target_slots=target, check_field=check_field,
        mask_service=True, mask_service_allow_cpu=allow_cpu,
        **_common_overrides(rings, device, backend, chunk_slots, checks, export_dir, recipient_key, partner_key,
                            destination_store, schedule=schedule),
    )
    if check_only:
        from pgw.factory.runtime import build_generators, open_packs

        gens = build_generators(s, open_packs(s))  # runs the start-up check; raises on a mismatch
        reports = [g.mask_service_check.to_dict() for g in gens.values()]
        if as_json:
            click.echo(json.dumps(reports, indent=1, default=str))
        else:
            for r in reports:
                click.echo(f"{short_fingerprint(r['fingerprint'])} {r['ring']} ({r['check_field']}) on {r['device']}/"
                           f"{r['backend']} gpu_prng={r['gpu_prng']}: OK, {len(r['linears_checked'])} linears x "
                           f"{r['rows_per_linear']} rows bit-identical ({r['wr_elements_compared']:,} W r and "
                           f"{r['uw_elements_compared']:,} u W elements)")
        return
    _run_factory(s, once, store_root, key_file)


@factory_group.command("outbox-serve")
@click.option("--export-dir", type=click.Path(path_type=Path), required=True, help="The mask service's export dir")
@click.option("--token-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True,
              help="Shared bearer token (the gateway's 'pgw maskstore pull --token-file')")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address (the VPC-internal address)")
@click.option("--port", type=int, default=8470, show_default=True)
@click.option("--tls-cert", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
@click.option("--tls-key", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
@_handle_errors
def outbox_serve_cmd(export_dir: Path, token_file: Path, host: str, port: int, tls_cert: Path | None,
                     tls_key: Path | None) -> None:
    """Serve the export outbox to the gateway VM (sealed + signed segments; the gateway deletes after import)."""
    from pgw.factory.courier import OutboxServer

    srv = OutboxServer(export_dir / "outbox", token_file, host, port, tls_cert_file=tls_cert, tls_key_file=tls_key)
    click.echo(f"serving {export_dir / 'outbox'} on {srv.url}", err=True)
    stop = threading.Event()
    srv.start()
    try:
        with _signal_handlers(stop.set):
            while not stop.wait(1.0):
                pass
    finally:
        srv.stop()


@factory_group.command("status")
@click.option("--pack", "packs", multiple=True, type=click.Path(path_type=Path), help="Model pack(s) to plan for")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@click.option("--low-water", type=int, default=None)
@click.option("--target", type=int, default=None)
@click.option("--segment-slots", type=int, default=None)
@_with_common
@_handle_errors
def status_cmd(packs: tuple[Path, ...], as_json: bool, low_water: int | None, target: int | None,
               segment_slots: int | None, rings: tuple[str, ...], device: str | None, backend: str | None,
               chunk_slots: int | None, checks: int | None, store_root: Path | None, key_file: Path | None,
               export_dir: Path | None, recipient_key: Path | None,
                 partner_key: Path | None, destination_store: str | None, he_service: str | None = None,
                 he_token_file: Path | None = None, he_hint_dir: Path | None = None,
                 schedule: str | None = None) -> None:
    """Inventory, refill plan and lifetime totals (does not generate anything)."""
    from pgw.core.ring import RingParams
    from pgw.factory.config import FactorySettings
    from pgw.factory.generator import slot_bytes
    from pgw.factory.runtime import build_sink, open_store
    from pgw.factory.scheduler import FactoryScheduler
    from pgw.modelpack import ModelPack

    overrides = _common_overrides(rings, device, backend, chunk_slots, checks, export_dir, recipient_key, partner_key,
                                destination_store, he_service, he_token_file, he_hint_dir)
    pack_paths = list(packs)
    try:
        s = _factory_settings(pack_path=pack_paths[0] if pack_paths else None,
                              extra_pack_paths=tuple(pack_paths[1:]) or None,
                              low_water_slots=low_water, target_slots=target, segment_slots=segment_slots, **overrides)
    except ValidationError:
        if pack_paths:
            raise
        # no pack configured: report the sink only
        s = FactorySettings(pack_path=Path("."), **{k: v for k, v in {"low_water_slots": low_water,
                            "target_slots": target, **overrides}.items() if v is not None})
        pack_paths = []
    else:
        pack_paths = list(s.all_pack_paths) if (packs or _env_has_pack()) else []
    store = None if s.output == "export" else open_store(**_store_overrides(store_root, key_file))
    sink = build_sink(s, store)
    plan_only: dict[tuple[str, str], Any] = {}
    for p in pack_paths:
        pack = ModelPack.open(p, verify="fingerprint", components=("linears",))
        if s.masked_head:  # segments are made for the derived masked-head fingerprint
            from pgw.engine.masked_head import MaskedHeadPack

            pack = MaskedHeadPack(pack)
        specs = pack.linear_specs()
        for r in s.ring_names:
            rp = RingParams.from_name(r)
            plan_only[(pack.fingerprint, rp.name)] = SimpleNamespace(
                key=(pack.fingerprint, rp.name), pack=pack, bytes_per_slot=slot_bytes(specs, rp), device="-")
    report: dict[str, Any]
    if plan_only:
        report = FactoryScheduler(s, sink, plan_only, metrics=False).status()
    else:
        report = {"sink": getattr(sink, "kind", "store"), "inventory": _raw_inventory(sink)}
        state_dir = s.state_dir or sink.state_dir()
        if state_dir is not None:
            from pgw.factory.scheduler import STATE_FILE, FactoryState

            st = FactoryState.load(Path(state_dir) / STATE_FILE)
            report["totals"] = st.totals
            report["crash_recoveries"] = st.crash_recoveries
            report["last_error"] = st.last_error
    if as_json:
        click.echo(json.dumps(report, indent=1, default=str))
        return
    _print_status(report)


def _env_has_pack() -> bool:
    import os

    return bool(os.environ.get("PGW_FACTORY_PACK_PATH"))


def _raw_inventory(sink: Any) -> list[dict[str, Any]]:
    if getattr(sink, "kind", "") == "export":
        agg: dict[tuple[str, str], dict[str, Any]] = {}
        for seg in sink.segments():
            row = agg.setdefault((seg.fingerprint, seg.ring), {"fingerprint": seg.fingerprint, "ring": seg.ring,
                                                               "segments": 0, "slots_available": 0, "bytes": 0})
            row["segments"] += 1
            row["slots_available"] += seg.slots
            row["bytes"] += seg.size
        return list(agg.values())
    return [{"fingerprint": i.fingerprint, "ring": i.ring, "segments": i.segments, "slots_available": i.slots_available,
             "slots_total": i.slots_total, "bytes": i.bytes_on_disk} for i in sink.store.inventory()]


def _print_status(report: dict[str, Any]) -> None:
    click.echo(f"sink: {report.get('sink')}")
    if "models" in report:
        if report.get("running_elsewhere"):
            click.echo("factory: RUNNING (lock held by another process)")
        for m in report["models"]:
            flag = "  LOW" if m["below_low_water"] else ""
            click.echo(
                f"  {m['model'] or '?'} {short_fingerprint(m['fingerprint'])} {m['ring']}: "
                f"{m['available_slots']:,} available "
                f"(low {m['low_water_slots']:,}, target {m['target_slots']:,}){flag}; "
                f"generated {m['segments_generated']} segments / {m['slots_generated']:,} slots"
            )
        for p in report["plan"]:
            click.echo(f"  plan: {short_fingerprint(p['fingerprint'])} {p['ring']} x{p['segments']} "
                       f"segments of {p['slots_per_segment']} slots ({p['reason']})")
        if not report["plan"]:
            click.echo("  plan: nothing to do")
    else:
        for row in report.get("inventory", []):
            click.echo(f"  {short_fingerprint(row['fingerprint'])} {row['ring']}: {row['slots_available']:,} slots "
                       f"available in {row['segments']} segment(s)")
        if not report.get("inventory"):
            click.echo("  (empty)")
    if report.get("crash_recoveries"):
        click.echo(f"crash recoveries: {report['crash_recoveries']}")
    if report.get("last_error"):
        click.echo(f"last error: {report['last_error']}")


@factory_group.command("benchmark")
@click.option("--model", "preset", default=None, help="Architecture preset (e.g. llama3-8b, qwen2.5-0.5b)")
@click.option("--pack", "pack_path", type=click.Path(path_type=Path), default=None, help="Benchmark a real pack")
@click.option("--ring", default="K32A12C27", show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--backend", type=click.Choice(["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"]), default="auto")
@click.option("--slots", type=int, default=256, show_default=True, help="Slots to time")
@click.option("--chunk-slots", type=int, default=256, show_default=True)
@click.option("--checks", type=int, default=2, show_default=True)
@click.option("--layers", type=int, default=1, show_default=True, help="Preset: layers to time (extrapolated)")
@click.option("--segment-slots", type=int, default=4096, show_default=True, help="Segment size for amortisation")
@click.option("--price-per-hour", type=float, default=None, help="USD/hour of this host (default: rough assumption)")
@click.option("--with-store", is_flag=True, help="Pack mode: include encryption into a temporary mask store")
@click.option("--json", "as_json", is_flag=True)
@_handle_errors
def benchmark_cmd(preset: str | None, pack_path: Path | None, ring: str, device: str, backend: str, slots: int,
                  chunk_slots: int, checks: int, layers: int, segment_slots: int, price_per_hour: float | None,
                  with_store: bool,
                  as_json: bool) -> None:
    """Measure tokens/hour and estimate $ per million tokens of masks."""
    from pgw.factory.bench import MODEL_PRESETS, benchmark_pack, benchmark_preset, format_report

    if (preset is None) == (pack_path is None):
        raise ConfigError(f"give exactly one of --model {{{','.join(sorted(MODEL_PRESETS))}}} or --pack")
    if slots <= 0 or chunk_slots <= 0:
        raise ConfigError("--slots and --chunk-slots must be positive")
    if preset is not None:
        res = benchmark_preset(preset, ring=ring, device=device, backend=backend, slots=slots, chunk_slots=chunk_slots,
                               checks=checks, layers=layers, segment_slots=segment_slots,
                               price_per_hour=price_per_hour)
    else:
        from pgw.modelpack import ModelPack

        pack = ModelPack.open(pack_path, verify="fingerprint", components=("linears",))  # type: ignore[arg-type]
        if with_store:
            from pgw.maskstore import FileKeyProvider, MaskStore

            with tempfile.TemporaryDirectory(prefix="pgw-bench-") as tmp:
                kp = FileKeyProvider(Path(tmp) / "kek", create=True)
                store = MaskStore.open(Path(tmp) / "store", kp)
                try:
                    res = benchmark_pack(pack, ring=ring, device=device, backend=backend, slots=slots,
                                         chunk_slots=chunk_slots, checks=checks, store=store,
                                         segment_slots=segment_slots,
                                         price_per_hour=price_per_hour)
                finally:
                    store.close()
        else:
            res = benchmark_pack(pack, ring=ring, device=device, backend=backend, slots=slots, chunk_slots=chunk_slots,
                                 checks=checks, segment_slots=segment_slots, price_per_hour=price_per_hour)
    if as_json:
        click.echo(json.dumps(res.to_dict(), indent=1))
    else:
        click.echo(format_report(res))


def main(argv: list[str] | None = None) -> None:
    """Standalone entry point (``python -m pgw.factory``)."""
    from pgw.core.log import configure_logging

    configure_logging("INFO")
    factory_group.main(args=argv, prog_name="pgw factory")
