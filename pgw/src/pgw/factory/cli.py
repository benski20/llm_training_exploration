"""``pgw factory`` commands: ``generate``, ``run``, ``status``, ``benchmark``.

Mounted into the main CLI as ``pgw factory`` (see ``INTERFACE_NOTES.md``) and runnable standalone
as ``python -m pgw.factory``. Settings come from ``PGW_FACTORY_*`` / ``PGW_MASKSTORE_*``; command
line options override them. Exit codes: 0 ok, 1 :class:`pgw.core.PGWError`, 3 not implemented
(a dependency such as the mask store is unfinished), 130 interrupted.
"""

from __future__ import annotations

import functools
import json
import signal
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

import click
from pydantic import ValidationError

from pgw.core.errors import ConfigError, PGWError
from pgw.core.fingerprint import short_fingerprint

__all__ = ["factory_group", "main"]

F = TypeVar("F", bound=Callable[..., Any])


def _handle_errors(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            click.echo(f"error: not implemented yet: {exc}", err=True)
            sys.exit(3)
        except PGWError as exc:
            click.echo(f"error [{exc.code}]: {exc.message}", err=True)
            sys.exit(1)
        except ValidationError as exc:
            click.echo(f"error [config]: {exc}", err=True)
            sys.exit(1)

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


def _install_signal_handlers(stop: Callable[[], None]) -> None:
    if threading.current_thread() is not threading.main_thread():  # pragma: no cover - e.g. CliRunner threads
        return

    def handler(signum: int, frame: Any) -> None:
        click.echo(f"received signal {signum}; stopping after aborting the current segment", err=True)
        stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)


@click.group("factory", context_settings={"help_option_names": ["-h", "--help"]})
def factory_group() -> None:
    """Mask factory (customer-run or independent partner): generate one-time masks offline."""


_common = [
    click.option("--ring", "rings", multiple=True, help="Ring preset(s), e.g. K40A12 (repeatable)"),
    click.option("--device", default=None, help="auto | cpu | cuda | cuda:N"),
    click.option("--backend", type=click.Choice(["auto", "cuda_int8", "cpu_int8", "cpu_f64", "cpu_f32"]), default=None),
    click.option("--chunk-slots", type=int, default=None, help="Rows per GEMM batch / encrypted chunk"),
    click.option("--checks", type=int, default=None, help="Freivalds vectors per linear"),
    click.option("--store", "store_root", type=click.Path(path_type=Path), default=None, help="Mask store root"),
    click.option("--key-file", type=click.Path(path_type=Path), default=None, help="Store KEK file (file provider)"),
    click.option("--export-dir", type=click.Path(path_type=Path), default=None, help="Partner mode: export directory"),
    click.option("--recipient-key", type=click.Path(path_type=Path), default=None,
                 help="Partner mode: customer X25519 public key file"),
]


def _with_common(fn: F) -> F:
    for opt in reversed(_common):
        fn = opt(fn)
    return fn


def _common_overrides(rings: tuple[str, ...], device: str | None, backend: str | None, chunk_slots: int | None,
                      checks: int | None, export_dir: Path | None, recipient_key: Path | None) -> dict[str, Any]:
    o: dict[str, Any] = {"device": device, "backend": backend, "chunk_slots": chunk_slots, "checks": checks}
    if rings:
        o["rings"] = tuple(rings)
        o["ring"] = rings[0]
    if export_dir is not None:
        o["output"] = "export"
        o["export_dir"] = export_dir
        o["recipient_public_key_file"] = recipient_key
    return o


@factory_group.command("generate")
@click.option("--pack", "packs", multiple=True, type=click.Path(path_type=Path), help="Model pack (repeatable)")
@click.option("--slots", type=int, default=None, help="Slots per segment")
@click.option("--segments", type=int, default=1, show_default=True)
@_with_common
@_handle_errors
def generate_cmd(packs: tuple[Path, ...], slots: int | None, segments: int, rings: tuple[str, ...], device: str | None,
                 backend: str | None, chunk_slots: int | None, checks: int | None, store_root: Path | None,
                 key_file: Path | None, export_dir: Path | None, recipient_key: Path | None) -> None:
    """Generate SEGMENTS fresh segments per pack and ring, then exit."""
    from pgw.factory.runtime import build_generators, build_sink, open_packs, open_store

    if segments <= 0:
        raise ConfigError("--segments must be positive")
    s = _factory_settings(
        pack_path=packs[0] if packs else None, extra_pack_paths=tuple(packs[1:]) or None, segment_slots=slots,
        **_common_overrides(rings, device, backend, chunk_slots, checks, export_dir, recipient_key),
    )
    store = None if s.output == "export" else open_store(**_store_overrides(store_root, key_file))
    sink = build_sink(s, store)
    gens = build_generators(s, open_packs(s))
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
@_with_common
@_handle_errors
def run_cmd(packs: tuple[Path, ...], once: bool, segment_slots: int | None, low_water: int | None, target: int | None,
            max_slots_per_hour: float | None, duty_cycle: float | None, nice: int | None, cpu_threads: int | None,
            poll_interval_s: float | None, rings: tuple[str, ...], device: str | None, backend: str | None,
            chunk_slots: int | None, checks: int | None, store_root: Path | None, key_file: Path | None,
            export_dir: Path | None, recipient_key: Path | None) -> None:
    """Keep inventories between low water and target (settings from PGW_FACTORY_* / PGW_MASKSTORE_*)."""
    from pgw.factory.governor import FactoryStopped, apply_process_priority
    from pgw.factory.runtime import build_scheduler, open_store

    s = _factory_settings(
        pack_path=packs[0] if packs else None, extra_pack_paths=tuple(packs[1:]) or None,
        segment_slots=segment_slots, low_water_slots=low_water, target_slots=target,
        max_slots_per_hour=max_slots_per_hour, duty_cycle=duty_cycle, nice=nice, cpu_threads=cpu_threads,
        poll_interval_s=poll_interval_s,
        **_common_overrides(rings, device, backend, chunk_slots, checks, export_dir, recipient_key),
    )
    apply_process_priority(s.nice, s.cpu_threads)
    store = None if s.output == "export" else open_store(**_store_overrides(store_root, key_file))
    sched = build_scheduler(s, store)
    _install_signal_handlers(sched.stop)
    if once:
        try:
            with sched:
                ids = sched.run_once()
        except FactoryStopped:
            click.echo("stopped", err=True)
            sys.exit(130)
        click.echo(f"produced {len(ids)} segment(s)")
        for i in ids:
            click.echo(f"  {i}")
        return
    sched.run_forever()


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
               export_dir: Path | None, recipient_key: Path | None) -> None:
    """Inventory, refill plan and lifetime totals (does not generate anything)."""
    from pgw.core.ring import RingParams
    from pgw.factory.config import FactorySettings
    from pgw.factory.generator import slot_bytes
    from pgw.factory.runtime import build_sink, open_store
    from pgw.factory.scheduler import FactoryScheduler
    from pgw.modelpack import ModelPack

    overrides = _common_overrides(rings, device, backend, chunk_slots, checks, export_dir, recipient_key)
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
@click.option("--ring", default="K40A12", show_default=True)
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
