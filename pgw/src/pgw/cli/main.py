"""``pgw`` command tree (SPEC section 12).

::

    pgw version [--json]
    pgw doctor [--role R] [--pack P [--deep] [--mask-supply]] [--maskstore ROOT] [--service-url URL] ...
    pgw pack SRC OUT                      HF checkpoint -> int8 model pack
    pgw inspect PACK                      manifest summary, rings, mask bytes per slot
    pgw factory generate|run|status|benchmark      (pgw.factory.cli)
    pgw maskstore init|status|import|pull|gc|rotate-key
    pgw service serve | service token create|hash | service enroll create|list|revoke
    pgw gateway enroll CODE --config-dir DIR   (pgw.gateway.cli)
    pgw gateway serve | gateway keys create|list|revoke  (pgw.gateway.cli)
    pgw chat [--model M] [--no-browser]   private chat on this computer (pgw.chat.cli)
    pgw bench ring|factory|service
    pgw config show ROLE | config render TEMPLATE OUT

Component groups are imported lazily so that each deployment image only needs its own extra.
Exit codes: 0 ok, 1 :class:`pgw.core.PGWError` / invalid configuration, 3 not implemented,
130 interrupted.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click

from pgw import PROTOCOL_VERSION, __version__
from pgw.cli._common import EXIT_ERROR, LazyGroup, echo_json, handle_errors
from pgw.core.errors import ConfigError
from pgw.core.log import configure_logging

__all__ = ["cli", "main"]

_LAZY = {
    "factory": ("pgw.factory.cli", "factory_group", "factory"),
    "gateway": ("pgw.gateway.cli", "gateway_group", "gateway"),
    "sdk": ("pgw.sdk.cli", "sdk_group", "gateway"),
    "chat": ("pgw.chat.cli", "chat_command", "client"),
}


@click.group(cls=LazyGroup, lazy_subcommands=_LAZY, context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--log-level", default="INFO", show_default=True, envvar="PGW_LOG_LEVEL",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
@click.option("--log-json/--no-log-json", default=False, envvar="PGW_LOG_JSON", help="JSON log lines (production)")
@click.version_option(__version__, "-V", "--version", prog_name="pgw",
                      message=f"%(prog)s %(version)s (protocol v{PROTOCOL_VERSION})")
def cli(log_level: str, log_json: bool) -> None:
    """PGW: private LLM inference over public weights with one-time masks."""
    configure_logging(log_level.upper(), json_lines=log_json)


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
def version(as_json: bool) -> None:
    """Print package and protocol versions."""
    if as_json:
        import platform

        echo_json({"pgw": __version__, "protocol": PROTOCOL_VERSION, "python": platform.python_version()})
    else:
        click.echo(f"pgw {__version__} (protocol v{PROTOCOL_VERSION})")


# ---- doctor ------------------------------------------------------------------------------------
@cli.command()
@click.option("--role", type=click.Choice(["all", "gateway", "service", "factory", "pack"]), default="all",
              show_default=True, help="Deployment role: its extras and PGW_* settings become mandatory")
@click.option("--pack", "pack_path", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None,
              help="Fully verify a model pack (SHA-256 of every file)")
@click.option("--ring", default=None, help="Ring preset to check against the pack / service (e.g. K32A12C27)")
@click.option("--maskstore", "maskstore_root", type=click.Path(path_type=Path, exists=True, file_okay=False),
              default=None, help="Open and verify a mask store (KEK from PGW_MASKSTORE_* or --key-file)")
@click.option("--key-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
              help="Mask store KEK file (permissions are checked)")
@click.option("--service-url", default=None, help="Probe a compute service (readyz, models, fingerprint)")
@click.option("--service-token-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
@click.option("--service-ca-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
@click.option("--settings", "check_settings", is_flag=True, help="Validate the role's PGW_* settings")
@click.option("--no-gpu", is_flag=True, help="Skip the GPU arithmetic self test")
@click.option("--strict", is_flag=True, help="Warnings count as failures (exit 1)")
@click.option("--deep", is_flag=True,
              help="With --pack: recompute weight-row norms from the int8 weights (install time; slow)")
@click.option("--mask-supply", is_flag=True,
              help="With --pack: time one mask chunk into a temporary store (heavy; not for start-up checks)")
@click.option("--expected-tps", "expected_tps", type=float, default=None,
              help="Expected mask consumption (tokens/s) for --mask-supply and the maskstore.read_rate check "
                   "(with --maskstore; default PGW_DOCTOR_EXPECTED_TOKENS_PER_S)")
@click.option("--maskstore-read-mb", "maskstore_read_mb", type=click.IntRange(min=0), default=None,
              help="Size in MiB of the maskstore.read_rate probe written and read next to the --maskstore store "
                   "(default 64; 0 skips the check)")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@handle_errors
def doctor(role: str, pack_path: Path | None, ring: str | None, maskstore_root: Path | None, key_file: Path | None,
           service_url: str | None, service_token_file: Path | None, service_ca_file: Path | None,
           check_settings: bool, no_gpu: bool, strict: bool, deep: bool, mask_supply: bool,
           expected_tps: float | None, maskstore_read_mb: int | None, as_json: bool) -> None:
    """Check the environment: dependencies, CSPRNG, exact ring arithmetic, devices, pack, store, service."""
    from pgw.cli.doctor import format_report, run_doctor

    if ring is not None:
        from pgw.core.ring import RingParams

        ring = RingParams.from_name(ring).name
    report = run_doctor(role=role, pack=pack_path, ring=ring, maskstore_root=maskstore_root, key_file=key_file,
                        service_url=service_url, service_token_file=service_token_file,
                        service_ca_file=service_ca_file, include_gpu=not no_gpu, check_settings=check_settings,
                        strict=strict, deep=deep, mask_supply=mask_supply,
                        expected_tokens_per_second=expected_tps,
                        maskstore_read_bytes=None if maskstore_read_mb is None else maskstore_read_mb << 20)
    if as_json:
        echo_json(report.to_dict())
    else:
        click.echo(format_report(report))
    if not report.ok:
        sys.exit(EXIT_ERROR)


# ---- model packs -------------------------------------------------------------------------------
@cli.command()
@click.argument("source")
@click.argument("out_dir", type=click.Path(path_type=Path))
@click.option("--name", default=None, help="Served model name (default: derived from source)")
@click.option("--revision", default=None, help="Hub revision (branch, tag or commit)")
@click.option("--plain-dtype", type=click.Choice(["source", "float32", "float16", "bfloat16"]), default="source",
              show_default=True, help="dtype of embeddings / LM head in plain.safetensors: 'source' keeps a bfloat16 "
              "checkpoint's bfloat16 (pack format 2, exact); float32 reproduces format-1 packs")
@click.option("--shard-gib", type=float, default=2.0, show_default=True, help="Max size of a linears shard")
@click.option("--license", "license_", default=None, help="License identifier recorded in the manifest")
@click.option("--overwrite", is_flag=True, help="Replace an existing pack directory")
@click.option("--quiet", "-q", is_flag=True, help="No progress output")
@click.option("--allow-unsupported-rope", is_flag=True,
              help="Pack even if the RoPE scaling type is not implemented by the PGW engine (not servable)")
@handle_errors
def pack(source: str, out_dir: Path, name: str | None, revision: str | None, plain_dtype: str, shard_gib: float,
         license_: str | None, overwrite: bool, quiet: bool, allow_unsupported_rope: bool) -> None:
    """Build a model pack from an HF checkpoint (hub repo id or local directory)."""
    from pgw.modelpack import pack_checkpoint

    if shard_gib <= 0:
        raise ConfigError("--shard-gib must be positive")
    progress = None if quiet else (lambda msg: click.echo(f"  {msg}", err=True))
    m = pack_checkpoint(source, out_dir, name=name, revision=revision, plain_dtype=plain_dtype,  # type: ignore[arg-type]
                        shard_bytes=int(shard_gib * 2**30), overwrite=overwrite, license=license_, progress=progress,
                        allow_unsupported_rope=allow_unsupported_rope)
    click.echo(f"packed {m.name} fingerprint {m.fingerprint}")


@cli.command()
@click.argument("pack_dir", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--verify", type=click.Choice(["fingerprint", "full"]), default="fingerprint", show_default=True)
@click.option("--json", "as_json", is_flag=True)
@handle_errors
def inspect(pack_dir: Path, verify: str, as_json: bool) -> None:
    """Summarize a model pack: fingerprint, architecture, rings, mask and wire bytes per token."""
    from pgw.core.ring import RingParams
    from pgw.modelpack import ModelPack

    with ModelPack.open(pack_dir, verify=verify) as p:  # type: ignore[arg-type]
        specs = p.linear_specs()
        m_total = sum(s.out_features for s in specs)
        n_total = sum(s.in_features for s in specs)
        rings = p.supported_rings()
        info: dict[str, Any] = {
            "name": p.name,
            "fingerprint": p.fingerprint,
            "arch": p.arch.arch,
            "layers": p.num_layers,
            "hidden_size": p.arch.hidden_size,
            "vocab_size": p.arch.vocab_size,
            "linears": len(specs),
            "max_in_features": p.max_in_features(),
            "supported_rings": rings,
            "per_token": {
                r: {"mask_bytes": m_total * RingParams.from_name(r).k // 8,
                    "upload_bytes": n_total * RingParams.from_name(r).k // 8,
                    "download_bytes": m_total * RingParams.from_name(r).k // 8,
                    "round_trips": len(specs)}
                for r in rings
            },
            "verified": verify,
        }
    if as_json:
        echo_json(info)
        return
    click.echo(f"{info['name']}  fingerprint {info['fingerprint']}")
    click.echo(f"  arch {info['arch']}, {info['layers']} layers, hidden {info['hidden_size']}, "
               f"vocab {info['vocab_size']}, {info['linears']} masked linears")
    for r, d in info["per_token"].items():
        click.echo(f"  {r}: masks {_fmt_bytes(d['mask_bytes'])}/token, wire up {_fmt_bytes(d['upload_bytes'])} "
                   f"down {_fmt_bytes(d['download_bytes'])} per token, {d['round_trips']} round trips")


def _fmt_bytes(n: int) -> str:
    """``1536`` -> ``'1.5 KiB'``."""
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if v < 1024 or unit == "GiB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.2f} {unit}"
        v /= 1024
    return f"{v:.2f} GiB"  # pragma: no cover


# ---- mask store --------------------------------------------------------------------------------
def _store_overrides(root: Path | None, key_file: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if root is not None:
        out["root"] = root
    if key_file is not None:
        out["key_file"] = key_file
        out["key_provider"] = "file"
    return out


_root_opt = click.option("--root", type=click.Path(path_type=Path), default=None,
                         help="Store root (default PGW_MASKSTORE_ROOT)")
_key_opt = click.option("--key-file", type=click.Path(path_type=Path), default=None,
                        help="KEK file, file provider (default PGW_MASKSTORE_KEY_FILE)")


@cli.group()
def maskstore() -> None:
    """Mask store administration (settings from PGW_MASKSTORE_*)."""


@maskstore.command("init")
@click.option("--root", type=click.Path(path_type=Path), required=True)
@click.option("--key-file", type=click.Path(path_type=Path), required=True,
              help="KEK file; created (32 random bytes, mode 0600) if absent")
@click.option("--json", "as_json", is_flag=True)
@handle_errors
def maskstore_init(root: Path, key_file: Path, as_json: bool) -> None:
    """Create a store and a file KEK (0600)."""
    from pgw.maskstore.admin import init_store

    status = init_store(root, key_file)
    if as_json:
        echo_json(status)
    else:
        click.echo(f"initialized mask store at {root} (KEK file {key_file})")


@maskstore.command("status")
@_root_opt
@_key_opt
@click.option("--json", "as_json", is_flag=True)
@handle_errors
def maskstore_status(root: Path | None, key_file: Path | None, as_json: bool) -> None:
    """Show inventory per model fingerprint and ring (no secrets)."""
    from pgw.maskstore.admin import store_status

    status = store_status(**_store_overrides(root, key_file))
    if as_json:
        echo_json(status)
        return
    _print_store_status(status)


def _print_store_status(status: dict[str, Any]) -> None:
    for k, v in status.items():
        if isinstance(v, list | dict):
            continue
        click.echo(f"{k:24s} {v}")
    inv = status.get("inventory")
    if isinstance(inv, list):
        click.echo("inventory:")
        if not inv:
            click.echo("  (empty)")
        for e in inv:
            if isinstance(e, dict):
                click.echo("  " + ", ".join(f"{k}={v}" for k, v in e.items()))
    for k, v in status.items():
        if isinstance(v, dict):
            click.echo(f"{k}: {json.dumps(v, sort_keys=True, default=str)}")


@maskstore.command("import")
@_root_opt
@_key_opt
@click.option("--keep-source", is_flag=True,
              help="Keep each file (and its .sig) after a successful import (default: remove them, so a copy "
              "can never be imported into a second store)")
@click.option("--delete-source", is_flag=True, hidden=True, help="Deprecated: deleting is the default now")
@click.argument("files", nargs=-1, required=True, type=click.Path(path_type=Path, exists=True, dir_okay=False))
@handle_errors
def maskstore_import(root: Path | None, key_file: Path | None, keep_source: bool, delete_source: bool,
                     files: tuple[Path, ...]) -> None:
    """Import segment files from a partner factory (verifies every chunk, re-wraps the key).

    Partner segments must be signed by a key in PGW_MASKSTORE_TRUSTED_PARTNER_KEYS and name this
    store (see 'pgw maskstore status' store_id) as their destination.
    """
    from pgw.maskstore.admin import import_files

    if keep_source and delete_source:
        raise click.UsageError("--keep-source and --delete-source are mutually exclusive")
    results = import_files(files, delete_source=not keep_source, **_store_overrides(root, key_file))
    echo_json(results)
    failed = [r for r in results if not r.get("ok")]
    if failed:
        click.echo(f"error: {len(failed)} of {len(results)} file(s) failed to import", err=True)
        sys.exit(EXIT_ERROR)


@maskstore.command("pull")
@_root_opt
@_key_opt
@click.option("--from", "from_url", required=True, help="Mask service outbox URL, e.g. https://10.0.2.15:8470")
@click.option("--token-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), required=True)
@click.option("--ca-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
              help="CA of the outbox server's TLS certificate")
@click.option("--inbox", type=click.Path(path_type=Path), required=True, help="Private download directory")
@click.option("--max-available-slots", type=int, default=None,
              help="Do not pull while the store holds this many available slots for the model and ring")
@click.option("--loop", "interval_s", type=float, default=None, help="Repeat every N seconds (default: once)")
@handle_errors
def maskstore_pull(root: Path | None, key_file: Path | None, from_url: str, token_file: Path, ca_file: Path | None,
                   inbox: Path, max_available_slots: int | None, interval_s: float | None) -> None:
    """Pull signed, HPKE-sealed segments from a GPU mask service in the same VPC and import them.

    Each segment is verified and imported exactly like 'pgw maskstore import' (trusted partner key, destination
    store id, every chunk's AEAD), then deleted locally and on the mask service."""
    import time

    from pgw.factory.courier import pull_once

    while True:
        try:
            results = pull_once(from_url, token_file, inbox, ca_file=ca_file,
                                max_available_slots=max_available_slots,
                                store_overrides=_store_overrides(root, key_file))
        except Exception as exc:
            if interval_s is None:
                raise
            click.echo(f"pull failed: {exc!r}", err=True)
            results = []
        if results or interval_s is None:
            echo_json(results)
        if interval_s is None:
            if any(not r.get("ok") for r in results):
                sys.exit(EXIT_ERROR)
            return
        time.sleep(interval_s)


@maskstore.command("gc")
@_root_opt
@_key_opt
@handle_errors
def maskstore_gc(root: Path | None, key_file: Path | None) -> None:
    """Shred fully consumed segments."""
    from pgw.maskstore.admin import gc_store

    freed = gc_store(**_store_overrides(root, key_file))
    click.echo(f"freed {freed} bytes")


@maskstore.command("rotate-key")
@_root_opt
@_key_opt
@click.argument("new_key_file", type=click.Path(path_type=Path))
@click.option("--no-reencrypt", is_flag=True, help="Only re-wrap segment keys (default also re-MACs state)")
@handle_errors
def maskstore_rotate_key(root: Path | None, key_file: Path | None, new_key_file: Path, no_reencrypt: bool) -> None:
    """Move the store to a new KEK file (offline: stop gateways and factories first)."""
    from pgw.maskstore.admin import rotate_store_kek

    res = rotate_store_kek(new_key_file, reencrypt=not no_reencrypt, **_store_overrides(root, key_file))
    echo_json(res)
    click.echo(f"now set PGW_MASKSTORE_KEY_FILE={new_key_file} and restart gateways and factories", err=True)


@maskstore.group("recipient-key")
def maskstore_recipient_key() -> None:
    """X25519 key pair for importing partner-factory segments (SPEC 7.7)."""


@maskstore_recipient_key.command("create")
@click.argument("private_key_file", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--public-out", type=click.Path(path_type=Path, dir_okay=False), default=None,
              help="Also write the public key as hex (give this file to the partner)")
@handle_errors
def maskstore_recipient_key_create(private_key_file: Path, public_out: Path | None) -> None:
    """Generate the private key (mode 0600) and print the public key and its id.

    Set PGW_MASKSTORE_RECIPIENT_KEY_FILE to the private key on the gateway host. Hand the public key
    to the partner over an authenticated channel and confirm the key id out of band.
    """
    from pgw.maskstore import RecipientKeyPair

    if private_key_file.exists():
        raise ConfigError(f"{private_key_file} already exists; refusing to overwrite a recipient key")
    private_key_file.parent.mkdir(parents=True, exist_ok=True)
    kp = RecipientKeyPair.generate(private_key_file)
    pub_hex = kp.public_bytes().hex()
    if public_out is not None:
        public_out.parent.mkdir(parents=True, exist_ok=True)
        public_out.write_text(pub_hex + "\n", encoding="utf-8")
    echo_json({"key_id": kp.key_id, "kek_id": kp.kek_id, "public_key_hex": pub_hex})


@maskstore_recipient_key.command("show")
@click.argument("private_key_file", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@handle_errors
def maskstore_recipient_key_show(private_key_file: Path) -> None:
    """Print the public key and key id of an existing recipient private key."""
    from pgw.maskstore import RecipientKeyPair

    kp = RecipientKeyPair(private_key_file)
    echo_json({"key_id": kp.key_id, "kek_id": kp.kek_id, "public_key_hex": kp.public_bytes().hex()})


@maskstore.group("partner-key")
def maskstore_partner_key() -> None:
    """Ed25519 signing key of a partner mask factory (segment signatures, SPEC 7.7)."""


@maskstore_partner_key.command("create")
@click.argument("private_key_file", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--public-out", type=click.Path(path_type=Path, dir_okay=False), default=None,
              help="Also write the public key as hex (give this file to the customer)")
@handle_errors
def maskstore_partner_key_create(private_key_file: Path, public_out: Path | None) -> None:
    """Generate the partner's signing key (mode 0600) and print the public key and its id.

    The partner factory signs every exported segment with it (PGW_FACTORY_PARTNER_SIGNING_KEY_FILE or
    'pgw factory ... --partner-key'). The customer adds the public key to PGW_MASKSTORE_TRUSTED_PARTNER_KEYS
    after confirming the key id over an authenticated channel.
    """
    from pgw.maskstore import PartnerSigningKey

    if private_key_file.exists():
        raise ConfigError(f"{private_key_file} already exists; refusing to overwrite a partner signing key")
    private_key_file.parent.mkdir(parents=True, exist_ok=True)
    key = PartnerSigningKey.generate(private_key_file)
    pub_hex = key.public_bytes().hex()
    if public_out is not None:
        public_out.parent.mkdir(parents=True, exist_ok=True)
        public_out.write_text(pub_hex + "\n", encoding="utf-8")
    echo_json({"key_id": key.key_id, "public_key_hex": pub_hex})


@maskstore_partner_key.command("show")
@click.argument("private_key_file", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@handle_errors
def maskstore_partner_key_show(private_key_file: Path) -> None:
    """Print the public key and key id of an existing partner signing key."""
    from pgw.maskstore import PartnerSigningKey

    key = PartnerSigningKey(private_key_file)
    echo_json({"key_id": key.key_id, "public_key_hex": key.public_bytes().hex()})


# ---- compute service ---------------------------------------------------------------------------
@cli.group()
def service() -> None:
    """Masked compute service (operator side; settings from PGW_SERVICE_*)."""


@service.command("serve")
@click.option("--host", default=None, help="Bind address (default PGW_SERVICE_HOST)")
@click.option("--port", type=int, default=None, help="Port (default PGW_SERVICE_PORT)")
@click.option("--pack", "packs", multiple=True, type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Model pack to serve (repeatable; default PGW_SERVICE_PACKS)")
@handle_errors
def service_serve(host: str | None, port: int | None, packs: tuple[Path, ...]) -> None:
    """Serve over HTTP/2 (hypercorn) with graceful drain on SIGTERM."""
    try:
        from pgw.service import ServiceSettings, serve
    except ImportError as exc:
        raise ConfigError(f"'pgw service serve' needs the 'service' extra (pip install 'pgw-sdk[service]'): {exc}") from exc

    overrides: dict[str, Any] = {}
    if host is not None:
        overrides["host"] = host
    if port is not None:
        overrides["port"] = port
    if packs:
        overrides["packs"] = list(packs)
    serve(ServiceSettings(**overrides) if overrides else None)


@service.group("token")
def service_token() -> None:
    """Tenant bearer tokens (the service stores SHA-256 hashes only)."""


def _tokens_file(tokens_file: Path | None) -> Path:
    if tokens_file is not None:
        return tokens_file
    env = os.environ.get("PGW_SERVICE_TOKENS_FILE")
    if not env:
        raise ConfigError("no tokens file: pass --tokens-file or set PGW_SERVICE_TOKENS_FILE")
    return Path(env)


_tokens_file_opt = click.option("--tokens-file", type=click.Path(path_type=Path), default=None,
                                help="Tokens file (default PGW_SERVICE_TOKENS_FILE)")


@service_token.command("create")
@click.option("--tenant", required=True, help="Tenant id ([A-Za-z0-9._-], max 64)")
@click.option("--tokens-file", type=click.Path(path_type=Path), default=None,
              help="Append 'tenant:token_id:sha256' here (default PGW_SERVICE_TOKENS_FILE); created with mode 0600")
@click.option("--out", "out_file", type=click.Path(path_type=Path), default=None,
              help="Write the plaintext token to this file (0600) instead of stdout")
@click.option("--token-only", is_flag=True,
              help="Valid without a client certificate even when the service requires mTLS for compute")
@click.option("--ttl", "ttl", default=None, help="Lifetime, e.g. 90d, 12h (default: until revoked)")
@handle_errors
def service_token_create(tenant: str, tokens_file: Path | None, out_file: Path | None, token_only: bool,
                         ttl: str | None) -> None:
    """Create a tenant token; the plaintext is shown once (give it to the customer's gateway)."""
    from pgw.enroll.store import parse_ttl
    from pgw.service.auth import issue_token_record, new_token, valid_tenant_id

    if not valid_tenant_id(tenant):
        raise ConfigError("tenant id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    path = _tokens_file(tokens_file)
    ttl_s = parse_ttl(ttl) if ttl else None
    token = new_token()
    if out_file is not None:
        # Write the plaintext first: if this fails, no orphan hash is registered.
        _write_private(out_file, token + "\n", append=False)
    _, rec = issue_token_record(tenant, path, token_only=token_only, ttl_s=ttl_s, token=token)
    out: dict[str, Any] = {"tenant": tenant, "token_id": rec.token_id}
    if rec.not_after is not None:
        out["not_after"] = int(rec.not_after)
    if out_file is not None:
        out["token_file"] = str(out_file)
    else:
        out["token"] = token
    click.echo(json.dumps(out))


@service_token.command("list")
@_tokens_file_opt
@handle_errors
def service_token_list(tokens_file: Path | None) -> None:
    """List tokens (id, tenant, expiry, flags; never the token or its hash)."""
    import time as _time

    from pgw.service.auth import read_token_records

    now = _time.time()
    rows = [{"token_id": r.token_id, "tenant": r.tenant,
             "not_after": int(r.not_after) if r.not_after is not None else None,
             "expired": r.expired(now), "token_only": r.token_only, "legacy_format": r.legacy}
            for r in read_token_records(_tokens_file(tokens_file))]
    click.echo(json.dumps(rows, indent=2))


@service_token.command("revoke")
@click.argument("token_id")
@_tokens_file_opt
@handle_errors
def service_token_revoke(token_id: str, tokens_file: Path | None) -> None:
    """Revoke a token by id (atomic rewrite of the tokens file). A running service stops accepting it
    within its reload interval and closes the compute channels it opened."""
    from pgw.service.auth import revoke_token

    rec = revoke_token(_tokens_file(tokens_file), token_id)
    click.echo(json.dumps({"revoked": rec.token_id, "tenant": rec.tenant}))


def _write_private(path: Path, text: str, *, append: bool) -> None:
    """Write ``text`` to ``path`` with mode 0600, refusing symlinks and tightening a looser existing mode."""
    from pgw.core.fileio import write_private

    write_private(path, text, append=append)


@service.group("enroll")
def service_enroll() -> None:
    """Single-use enrollment codes for new customer gateways (POST /pgw/v1/enroll)."""


def _enroll_store(enrollment_file: Path | None) -> Any:
    from pgw.enroll.store import EnrollmentStore

    path = enrollment_file or (Path(os.environ["PGW_SERVICE_ENROLLMENT_FILE"])
                               if os.environ.get("PGW_SERVICE_ENROLLMENT_FILE") else None)
    if path is None:
        raise ConfigError("no enrollment file: pass --enrollment-file or set PGW_SERVICE_ENROLLMENT_FILE")
    return EnrollmentStore(path)


_enroll_file_opt = click.option("--enrollment-file", type=click.Path(path_type=Path, dir_okay=False), default=None,
                                help="Enrollment records (default PGW_SERVICE_ENROLLMENT_FILE)")


@service_enroll.command("create")
@click.option("--tenant", required=True, help="Tenant id the gateway will belong to ([A-Za-z0-9._-], max 64)")
@click.option("--ttl", default="24h", show_default=True, help="Validity: e.g. 30m, 24h, 7d (max 30d)")
@click.option("--models", default="", help="Comma-separated model names the code grants (default: all served)")
@click.option("--note", default=None, help="Free-text note shown by 'enroll list' (e.g. the customer's name)")
@click.option("--service-url", default=None, help="Public service URL in the code (default PGW_SERVICE_PUBLIC_URL)")
@click.option("--ca-file", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
              help="Service CA certificate to pin (default PGW_SERVICE_ENROLLMENT_CA_FILE)")
@_enroll_file_opt
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@handle_errors
def service_enroll_create(tenant: str, ttl: str, models: str, note: str | None, service_url: str | None,
                          ca_file: Path | None, enrollment_file: Path | None, as_json: bool) -> None:
    """Create a single-use enrollment code (shown once; only a hash of its secret is stored)."""
    import datetime as dt

    from pgw.enroll.code import EnrollmentCode, validate_service_url
    from pgw.enroll.service import ca_fingerprint, load_ca_pem
    from pgw.enroll.store import parse_ttl

    url = service_url or os.environ.get("PGW_SERVICE_PUBLIC_URL")
    if not url:
        raise ConfigError("no service URL: pass --service-url or set PGW_SERVICE_PUBLIC_URL")
    url = validate_service_url(url)
    ca = ca_file or (Path(os.environ["PGW_SERVICE_ENROLLMENT_CA_FILE"])
                     if os.environ.get("PGW_SERVICE_ENROLLMENT_CA_FILE") else None)
    fp = ca_fingerprint(load_ca_pem(ca)) if ca is not None else None
    if fp is None and url.startswith("https://"):
        raise ConfigError("an https service URL needs the service CA to pin: pass --ca-file or set "
                          "PGW_SERVICE_ENROLLMENT_CA_FILE")
    store = _enroll_store(enrollment_file)
    rec, secret = store.create(tenant, parse_ttl(ttl), [m for m in models.split(",") if m.strip()], note=note)
    code = EnrollmentCode(url, fp or bytes(32), secret).encode()
    if as_json:
        echo_json({"id": rec.id, "tenant": rec.tenant, "expires_at": rec.expires_at, "models": rec.models,
                   "code": code})
        return
    exp = dt.datetime.fromtimestamp(rec.expires_at, dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    click.echo(code)
    click.echo(f"\nid {rec.id}, tenant {rec.tenant}, single use, valid until {exp}"
               + (f", models {', '.join(rec.models)}" if rec.models else ", all served models"), err=True)
    click.echo("Send the code to the customer's administrator over a trusted channel. It is shown only once; "
               "revoke it with 'pgw service enroll revoke ID' if it leaks.", err=True)


@service_enroll.command("list")
@_enroll_file_opt
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@handle_errors
def service_enroll_list(enrollment_file: Path | None, as_json: bool) -> None:
    """List enrollment codes and their state (no secrets)."""
    import datetime as dt

    recs = [r.public() for r in _enroll_store(enrollment_file).list()]
    if as_json:
        echo_json(recs)
        return
    if not recs:
        click.echo("(no enrollment codes)")
    for r in recs:
        exp = dt.datetime.fromtimestamp(r["expires_at"], dt.UTC).strftime("%Y-%m-%d %H:%M")
        used = f" by {r['used_by']}" if r.get("used_by") else ""
        click.echo(f"{r['id']}  {r['tenant']:<20} {r['state']:<8}{used}  expires {exp} UTC  "
                   f"models={','.join(r['models']) or '*'}" + (f"  note={r['note']}" if r.get("note") else ""))


@service_enroll.command("revoke")
@click.argument("record_id")
@_enroll_file_opt
@handle_errors
def service_enroll_revoke(record_id: str, enrollment_file: Path | None) -> None:
    """Revoke an unused enrollment code (takes effect immediately, also in a running service)."""
    rec = _enroll_store(enrollment_file).revoke(record_id)
    click.echo(json.dumps({"revoked": rec.id, "tenant": rec.tenant, "state": rec.state()}))


from pgw.cli.accounts_cli import register as _register_accounts  # noqa: E402

_register_accounts(service)


@service_token.command("hash")
@handle_errors
def service_token_hash() -> None:
    """Read a token from stdin and print its SHA-256 (the form stored in the tokens file)."""
    from pgw.service.auth import hash_token

    token = sys.stdin.readline().strip()
    if not token:
        raise ConfigError("no token on stdin")
    click.echo(hash_token(token))


# ---- benchmarks --------------------------------------------------------------------------------
@cli.group()
def bench() -> None:
    """Benchmarks: ring matmul kernels, mask factory throughput, service load."""


@bench.command("ring")
@click.option("--ring", "rings", multiple=True, default=("K32A12C27",), show_default=True)
@click.option("--backend", "backends", multiple=True, help="Backend(s) (default: all available)")
@click.option("--shape", default="llama3-8b.qkv", show_default=True,
              help="Preset name or 'MxN' (out x in features)")
@click.option("--rows", "rows_list", multiple=True, type=int, default=(1, 32), show_default=True,
              help="Rows per call (1 = decode; repeatable)")
@click.option("--repeat", type=int, default=3, show_default=True)
@click.option("--json", "as_json", is_flag=True)
@handle_errors
def bench_ring_cmd(rings: tuple[str, ...], backends: tuple[str, ...], shape: str, rows_list: tuple[int, ...],
                   repeat: int, as_json: bool) -> None:
    """Exact W8 . x_hat mod 2^K timing per backend (the service's per-call work)."""
    from pgw.cli.bench import SHAPE_PRESETS, available_backends, bench_ring, format_ring_results

    if shape in SHAPE_PRESETS:
        m, n = SHAPE_PRESETS[shape]
    else:
        try:
            m_s, n_s = shape.lower().split("x")
            m, n = int(m_s), int(n_s)
        except ValueError as exc:
            raise ConfigError(f"--shape must be one of {', '.join(SHAPE_PRESETS)} or 'MxN'") from exc
    results = [bench_ring(ring=r, backend=b, m=m, n=n, rows=rows, repeat=repeat)
               for r in rings for b in (backends or available_backends()) for rows in rows_list]
    if as_json:
        echo_json([r.to_dict() for r in results])
    else:
        click.echo(format_ring_results(results))
    if not all(r.exact for r in results):
        click.echo("error [internal]: a backend produced an inexact result", err=True)
        sys.exit(EXIT_ERROR)


@bench.command("factory", context_settings={"ignore_unknown_options": True, "allow_extra_args": True,
                                            "help_option_names": []}, add_help_option=False)
@click.pass_context
def bench_factory(ctx: click.Context) -> None:
    """Mask generation throughput and $/M tokens (same options as 'pgw factory benchmark')."""
    try:
        from pgw.factory.cli import factory_group
    except ImportError as exc:
        click.echo(f"error [config]: 'pgw bench factory' needs the 'factory' extra: {exc}", err=True)
        ctx.exit(EXIT_ERROR)
        return
    cmd = factory_group.commands["benchmark"]
    cmd.main(args=list(ctx.args), prog_name="pgw bench factory", standalone_mode=True)


@bench.command("service", context_settings={"ignore_unknown_options": True, "allow_extra_args": True,
                                            "help_option_names": []}, add_help_option=False)
@click.pass_context
def bench_service(ctx: click.Context) -> None:
    """Load-test a compute service (same options as 'python -m pgw.service.loadtest')."""
    try:
        from pgw.service.loadtest import main as loadtest_main
    except ImportError as exc:
        click.echo(f"error [config]: 'pgw bench service' needs the 'service' extra: {exc}", err=True)
        ctx.exit(EXIT_ERROR)
        return
    ctx.exit(loadtest_main(list(ctx.args)))


# ---- configuration -----------------------------------------------------------------------------
@cli.group()
def config() -> None:
    """Inspect settings and render config templates."""


@config.command("show")
@click.argument("role", type=click.Choice(["common", "gateway", "service", "factory", "maskstore"]))
@handle_errors
def config_show(role: str) -> None:
    """Print the effective PGW_* settings of ROLE as JSON (secrets masked)."""
    from pgw.cli.configtools import settings_dump

    echo_json(settings_dump(role))


@config.command("render")
@click.argument("template", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.argument("out", type=click.Path(path_type=Path, dir_okay=False))
@handle_errors
def config_render(template: Path, out: Path) -> None:
    """Expand ${env:NAME} / ${file:PATH} references in TEMPLATE into OUT (mode 0600, atomic)."""
    from pgw.cli.configtools import render_template_file

    n = render_template_file(template, out)
    click.echo(f"wrote {out} ({n} bytes)", err=True)


def main() -> None:
    """Console entry point."""
    cli(prog_name="pgw")


if __name__ == "__main__":  # pragma: no cover
    main()
