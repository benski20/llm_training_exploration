"""``pgw gateway`` commands: ``serve``, ``enroll`` and ``keys create|list|revoke``.

``serve`` starts the first-run setup wizard (:mod:`pgw.gateway.setup`) instead of the API when no
model is configured, a wizard run under ``config_dir`` is unfinished, or ``setup_mode`` is set.

The top-level CLI (``pgw.cli``) mounts :data:`gateway_group`; it is also runnable as
``python -m pgw.gateway``. Exit codes: 0 ok, 1 :class:`pgw.core.PGWError`, 3 not implemented.
"""

from __future__ import annotations

import functools
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click

from pgw.core.errors import ConfigError, PGWError

__all__ = ["gateway_group", "main"]

F = TypeVar("F", bound=Callable[..., Any])


def _errors(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except PGWError as exc:
            click.echo(f"error [{exc.code}]: {exc.message}", err=True)
            sys.exit(1)

    return wrapper  # type: ignore[return-value]


def _key_store(keys_file: Path | None) -> Any:
    from pgw.gateway.auth import ApiKeyStore
    from pgw.gateway.config import GatewaySettings

    path = keys_file or GatewaySettings().api_keys_file
    if path is None:
        raise ConfigError("no API key file: pass --keys-file or set PGW_GATEWAY_API_KEYS_FILE")
    return ApiKeyStore(path)


@click.group("gateway")
def gateway_group() -> None:
    """Customer-side OpenAI-compatible gateway."""


@gateway_group.command("serve")
@click.option("--host", default=None, help="Bind address (default PGW_GATEWAY_HOST)")
@click.option("--port", type=int, default=None, help="Port (default PGW_GATEWAY_PORT)")
@_errors
def serve(host: str | None, port: int | None) -> None:
    """Serve the gateway (settings from PGW_GATEWAY_* / PGW_MASKSTORE_*)."""
    from pgw.core.log import configure_logging
    from pgw.core.proc_tuning import apply_malloc_tuning

    apply_malloc_tuning()  # before the model, masks and buffers are allocated
    from pgw.core.settings import CommonSettings
    from pgw.gateway.app import create_app
    from pgw.gateway.config import GatewaySettings

    common = CommonSettings()
    configure_logging(common.log_level, json_lines=common.log_json)
    from pgw.core.stack_sampler import maybe_start

    maybe_start()  # PGW_STACK_SAMPLER (diagnostics; off by default)
    settings = _settings_with_wizard_env(GatewaySettings())
    from pgw.gateway.setup import should_run_setup

    supervisor = None
    if should_run_setup(settings):
        app = _setup_app(settings, host or settings.host, port or settings.port)
    else:
        app = create_app(settings)
        if settings.factory_env_file is not None:
            from pgw.gateway.setup_jobs import FactorySupervisor, read_env_file

            supervisor = FactorySupervisor(read_env_file(settings.factory_env_file))
            supervisor.start()

    bind = host or settings.host
    if not settings.tls.enabled and bind not in ("127.0.0.1", "::1", "localhost"):
        logging.getLogger("pgw.gateway").warning(
            "serving the gateway API without TLS on %s: prompts and completions cross the network in "
            "plaintext unless a TLS-terminating proxy sits in front (set PGW_GATEWAY_TLS__*)", bind
        )
    tls: dict[str, Any] = {}
    if settings.tls.enabled:
        tls = {"ssl_certfile": str(settings.tls.cert_file), "ssl_keyfile": str(settings.tls.key_file)}
        if settings.tls.key_password is not None:
            tls["ssl_keyfile_password"] = settings.tls.key_password.get_secret_value()
        if settings.tls.ca_file is not None:
            import ssl

            tls["ssl_ca_certs"] = str(settings.tls.ca_file)
            if settings.tls.require_client_cert:
                tls["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    try:
        run_gateway_servers(app, settings, bind, port or settings.port, tls)
    finally:
        if supervisor is not None:
            supervisor.stop()


def _settings_with_wizard_env(settings: Any) -> Any:
    """Settings plus ``<config_dir>/gateway.env`` written by a finished setup wizard.

    Only read when the wizard manages ``config_dir`` (``setup/state.json`` exists); variables already
    set in the environment win over the file.
    """
    import os

    from pgw.gateway.config import GatewaySettings
    from pgw.gateway.setup import load_setup_state
    from pgw.gateway.setup_jobs import read_env_file

    st = load_setup_state(settings.config_dir)
    env_file = Path(settings.config_dir) / "gateway.env"
    if st is None or not st.get("finished") or not env_file.is_file():
        return settings
    for k, v in read_env_file(env_file).items():
        os.environ.setdefault(k, v)
    return GatewaySettings()


def _setup_app(settings: Any, host: str, port: int) -> Any:
    from pgw.gateway.setup import create_setup_app, new_setup_token

    token = new_setup_token()
    app = create_setup_app(settings, setup_token=token)
    shown = "localhost" if host in ("0.0.0.0", "::", "127.0.0.1") else host  # noqa: S104
    scheme = "https" if settings.tls.enabled else "http"
    banner = (
        "\n" + "=" * 72 + "\n"
        "PGW gateway setup mode: no model is configured yet.\n"
        f"Open {scheme}://{shown}:{port}/console/ and enter this one-time setup token:\n\n"
        f"    {token}\n\n"
        "The token is valid until this process stops. Keep it private.\n" + "=" * 72 + "\n"
    )
    click.echo(banner)
    logging.getLogger("pgw.gateway.setup").warning("setup mode: setup token printed to stdout; open /console/")
    return app


@gateway_group.command("enroll")
@click.argument("code")
@click.option("--config-dir", type=click.Path(path_type=Path, file_okay=False), required=True,
              help="Where to write the credentials, models.json and enrollment.json")
@click.option("--name", "gateway_name", default=None, help="Gateway name shown to the provider (default: host name)")
@click.option("--packs-dir", type=click.Path(path_type=Path, file_okay=False), default=None,
              help="Model pack directory referenced from models.json (default: <config-dir>/packs)")
@click.option("--no-client-cert", is_flag=True, help="Do not request an mTLS client certificate")
@click.option("--force", is_flag=True, help="Replace the files of an earlier enrollment")
@_errors
def enroll(code: str, config_dir: Path, gateway_name: str | None, packs_dir: Path | None, no_client_cert: bool,
           force: bool) -> None:
    """Join the compute service with a one-time enrollment code (PGW1-...).

    The code is checked locally first (typos are caught before any network call). The service's
    certificate authority must match the fingerprint in the code. The gateway's private key is
    generated here and never leaves this machine.
    """
    from pgw.enroll.client import enroll_gateway

    res = enroll_gateway(code, config_dir, gateway_name=gateway_name, force=force, packs_dir=packs_dir,
                         request_client_cert=not no_client_cert, progress=lambda m: click.echo(f"  {m}", err=True))
    click.echo(f"Enrolled as tenant {res.tenant} with {res.service_url}.")
    for k, p in res.files.items():
        click.echo(f"  wrote {p}" + (" (mode 0600)" if k in ("token", "client_key", "models") else ""))
    click.echo("\nNext steps:")
    for m in res.models:
        packs = packs_dir or config_dir / "packs"
        rev = f" --revision {m['revision']}" if m.get("revision") else ""
        click.echo(f"  1. Build the model pack: pgw pack {m.get('hf_repo')} {packs / m['name']} "
                   f"--name {m.get('pack_name') or m['name']}{rev}  (fingerprint must be {m['fingerprint'][:12]}...)")
    click.echo(f"  2. Initialize the mask store: pgw maskstore init --root /var/lib/pgw/masks "
               f"--key-file {config_dir / 'secrets' / 'maskstore.kek'}")
    click.echo(f"  3. Start the gateway: PGW_GATEWAY_MODELS_FILE={res.files['models']} pgw gateway serve")
    click.echo("  Or skip all of this: start 'pgw gateway serve' with no models and use the setup wizard at /console/.")


def run_gateway_servers(app: Any, settings: Any, host: str, port: int, tls: dict[str, Any]) -> None:
    """Run the API server (and the optional ops listener) with a graceful SIGTERM drain.

    On SIGTERM/SIGINT the runtime first stops admitting work and ``/readyz`` turns 503 (so the
    load balancer stops routing here), then uvicorn waits up to ``drain_timeout_s`` for open
    requests and SSE streams before closing them, and only then the lifespan shutdown burns
    leftover reservations, flushes quotas and closes the audit log. Kubernetes'
    ``terminationGracePeriodSeconds`` must exceed ``drain_timeout_s`` plus store close time.
    """
    import asyncio

    import uvicorn

    from pgw.core.proc_tuning import apply_malloc_tuning, new_event_loop

    apply_malloc_tuning()
    state = getattr(app.state, "pgw", None)

    class _DrainingServer(uvicorn.Server):
        def handle_exit(self, sig: int, frame: Any) -> None:
            if state is not None:
                state.runtime.begin_drain()
            super().handle_exit(sig, frame)

    grace = float(settings.drain_timeout_s)
    cfg = uvicorn.Config(app, host=host, port=port, log_config=None, server_header=False, date_header=False,
                         timeout_graceful_shutdown=int(grace + 0.999) if grace > 0 else None, **tls)
    servers = [_DrainingServer(cfg)]
    ops_app = getattr(app.state, "ops_app", None)
    if settings.ops_port is not None and ops_app is not None:
        ops_cfg = uvicorn.Config(ops_app, host=settings.ops_host, port=settings.ops_port, log_config=None,
                                 server_header=False, date_header=False, lifespan="off",
                                 timeout_graceful_shutdown=1)
        servers.append(_DrainingServer(ops_cfg))

    async def _main() -> None:
        tasks = [asyncio.create_task(s.serve()) for s in servers]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for s in servers:  # one stopped (signal or failure): stop the other as well
            s.should_exit = True
        await asyncio.gather(*tasks, return_exceptions=True)
        for t in done:
            t.result()

    asyncio.run(_main(), loop_factory=new_event_loop)


@gateway_group.group("keys")
def keys() -> None:
    """Manage API keys (hashed key file)."""


@keys.command("create")
@click.option("--name", required=True, help="Human-readable owner/app name")
@click.option("--scope", "scopes", multiple=True, default=("inference",), show_default=True,
              type=click.Choice(["inference", "admin"]))
@click.option("--model", "models", multiple=True, help="Restrict to these models (default: all)")
@click.option("--rpm", type=int, default=None, help="Requests per minute (default: gateway setting)")
@click.option("--tpm", type=int, default=None, help="Tokens per minute (default: gateway setting)")
@click.option("--daily-tokens", type=int, default=None, help="Daily token quota")
@click.option("--max-concurrent", type=int, default=None, help="Concurrent requests")
@click.option("--keys-file", type=click.Path(path_type=Path), default=None)
@_errors
def keys_create(name: str, scopes: tuple[str, ...], models: tuple[str, ...], rpm: int | None, tpm: int | None,
                daily_tokens: int | None, max_concurrent: int | None, keys_file: Path | None) -> None:
    """Create an API key (the key is printed once and cannot be recovered)."""
    store = _key_store(keys_file)
    rec, plaintext = store.create(name, scopes=scopes, models=models, requests_per_minute=rpm,
                                  tokens_per_minute=tpm, daily_token_quota=daily_tokens,
                                  max_concurrent_requests=max_concurrent)
    click.echo(json.dumps({"key": plaintext, "key_id": rec.key_id, "name": rec.name}))


@keys.command("list")
@click.option("--keys-file", type=click.Path(path_type=Path), default=None)
@_errors
def keys_list(keys_file: Path | None) -> None:
    """List API keys (no secrets)."""
    store = _key_store(keys_file)
    click.echo(json.dumps([k.public() for k in store.list()], indent=1))


@keys.command("revoke")
@click.argument("key_id")
@click.option("--keys-file", type=click.Path(path_type=Path), default=None)
@_errors
def keys_revoke(key_id: str, keys_file: Path | None) -> None:
    """Revoke an API key (takes effect in running gateways on their next request)."""
    _key_store(keys_file).revoke(key_id)
    click.echo(json.dumps({"revoked": key_id}))


def main() -> None:
    """``python -m pgw.gateway``."""
    gateway_group(prog_name="pgw gateway")
