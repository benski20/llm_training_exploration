"""``pgw`` command tree.

Commands delegate to subpackage APIs. ``NotImplementedError`` from an unfinished component is
reported as exit code 3; :class:`pgw.core.PGWError` as exit code 1 with its error code.
"""

from __future__ import annotations

import functools
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click

from pgw import PROTOCOL_VERSION, __version__
from pgw.core.errors import PGWError
from pgw.core.log import configure_logging

__all__ = ["cli", "main"]

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

    return wrapper  # type: ignore[return-value]


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--log-level", default="INFO", show_default=True, envvar="PGW_LOG_LEVEL")
@click.option("--log-json/--no-log-json", default=False, envvar="PGW_LOG_JSON")
def cli(log_level: str, log_json: bool) -> None:
    """PGW: private LLM inference over public weights with one-time masks."""
    configure_logging(log_level.upper(), json_lines=log_json)


@cli.command()
def version() -> None:
    """Print package and protocol versions."""
    click.echo(f"pgw {__version__} (protocol v{PROTOCOL_VERSION})")


@cli.command()
@click.option("--pack", "pack_path", type=click.Path(path_type=Path), default=None, help="Also verify a model pack")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@_handle_errors
def doctor(pack_path: Path | None, as_json: bool) -> None:
    """Check the environment: exact ring arithmetic on every backend, devices, optional pack."""
    import torch

    from pgw.core.selftest import run_selftest

    res = run_selftest()
    report: dict[str, Any] = {
        "pgw": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_threads": torch.get_num_threads(),
        "selftest_ok": res.ok,
        "selftest_failures": [k for k, v in res.checks.items() if not v],
        "selftest_ms": res.timings_ms,
    }
    if pack_path is not None:
        from pgw.modelpack import ModelPack

        pack = ModelPack.open(pack_path, verify="full")
        report["pack"] = {"name": pack.name, "fingerprint": pack.fingerprint}
    if as_json:
        click.echo(json.dumps(report, indent=1))
    else:
        for k, v in report.items():
            click.echo(f"{k:18s} {v}")
    if not res.ok:
        sys.exit(1)


@cli.command()
@click.argument("source")
@click.argument("out_dir", type=click.Path(path_type=Path))
@click.option("--name", default=None, help="Served model name (default: derived from source)")
@click.option("--revision", default=None)
@click.option("--plain-dtype", type=click.Choice(["float32", "float16", "bfloat16"]), default="float32")
@click.option("--overwrite", is_flag=True)
@_handle_errors
def pack(source: str, out_dir: Path, name: str | None, revision: str | None, plain_dtype: str, overwrite: bool) -> None:
    """Build a model pack from an HF checkpoint (repo id or local dir)."""
    from pgw.modelpack import pack_checkpoint

    m = pack_checkpoint(source, out_dir, name=name, revision=revision, plain_dtype=plain_dtype,  # type: ignore[arg-type]
                        overwrite=overwrite)
    click.echo(f"packed {m.name} fingerprint {m.fingerprint}")


# ---- factory -------------------------------------------------------------------------------
@cli.group()
def factory() -> None:
    """Mask factory (customer-run or partner)."""


@factory.command("generate")
@click.option("--pack", "pack_path", type=click.Path(path_type=Path), required=True)
@click.option("--store", "store_root", type=click.Path(path_type=Path), required=True)
@click.option("--slots", type=int, default=4096, show_default=True)
@click.option("--ring", default="K40A12", show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--segments", type=int, default=1, show_default=True)
@_handle_errors
def factory_generate(pack_path: Path, store_root: Path, slots: int, ring: str, device: str, segments: int) -> None:
    """Generate segments into a local mask store."""
    raise NotImplementedError("factory team: pgw factory generate")


@factory.command("run")
@_handle_errors
def factory_run() -> None:
    """Run the refill scheduler (settings from PGW_FACTORY_* / PGW_MASKSTORE_*)."""
    raise NotImplementedError("factory team: pgw factory run")


# ---- mask store ------------------------------------------------------------------------------
@cli.group()
def maskstore() -> None:
    """Mask store administration."""


@maskstore.command("init")
@click.option("--root", type=click.Path(path_type=Path), required=True)
@click.option("--key-file", type=click.Path(path_type=Path), required=True)
@_handle_errors
def maskstore_init(root: Path, key_file: Path) -> None:
    """Create a store and a file KEK (0600)."""
    raise NotImplementedError("maskstore team: pgw maskstore init")


@maskstore.command("status")
@click.option("--root", type=click.Path(path_type=Path), required=True)
@_handle_errors
def maskstore_status(root: Path) -> None:
    """Show inventory per model fingerprint and ring."""
    raise NotImplementedError("maskstore team: pgw maskstore status")


@maskstore.command("import")
@click.option("--root", type=click.Path(path_type=Path), required=True)
@click.argument("files", nargs=-1, type=click.Path(path_type=Path))
@_handle_errors
def maskstore_import(root: Path, files: tuple[Path, ...]) -> None:
    """Import segment files from a partner factory."""
    raise NotImplementedError("maskstore team: pgw maskstore import")


@maskstore.command("gc")
@click.option("--root", type=click.Path(path_type=Path), required=True)
@_handle_errors
def maskstore_gc(root: Path) -> None:
    """Shred fully consumed segments."""
    raise NotImplementedError("maskstore team: pgw maskstore gc")


# ---- service -------------------------------------------------------------------------------
@cli.group()
def service() -> None:
    """Masked compute service."""


@service.command("serve")
@_handle_errors
def service_serve() -> None:
    """Serve (settings from PGW_SERVICE_*); HTTP/2 via hypercorn."""
    raise NotImplementedError("service team: pgw service serve")


# ---- gateway -------------------------------------------------------------------------------
@cli.group()
def gateway() -> None:
    """Customer-side OpenAI-compatible gateway."""


@gateway.command("serve")
@_handle_errors
def gateway_serve() -> None:
    """Serve (settings from PGW_GATEWAY_*)."""
    raise NotImplementedError("gateway team: pgw gateway serve")


@gateway.group("keys")
def gateway_keys() -> None:
    """Manage API keys."""


@gateway_keys.command("create")
@click.option("--name", required=True)
@_handle_errors
def gateway_keys_create(name: str) -> None:
    """Create an API key (printed once)."""
    raise NotImplementedError("gateway team: pgw gateway keys create")


@gateway_keys.command("list")
@_handle_errors
def gateway_keys_list() -> None:
    """List API keys."""
    raise NotImplementedError("gateway team: pgw gateway keys list")


@gateway_keys.command("revoke")
@click.argument("key_id")
@_handle_errors
def gateway_keys_revoke(key_id: str) -> None:
    """Revoke an API key."""
    raise NotImplementedError("gateway team: pgw gateway keys revoke")


def main() -> None:
    """Console entry point."""
    cli(prog_name="pgw")


if __name__ == "__main__":  # pragma: no cover
    main()
