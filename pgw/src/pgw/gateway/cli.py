"""``pgw gateway`` commands: ``serve`` and ``keys create|list|revoke``.

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
        except NotImplementedError as exc:
            click.echo(f"error: not implemented yet: {exc}", err=True)
            sys.exit(3)
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
    from pgw.core.settings import CommonSettings
    from pgw.gateway.app import create_app
    from pgw.gateway.config import GatewaySettings

    common = CommonSettings()
    configure_logging(common.log_level, json_lines=common.log_json)
    settings = GatewaySettings()
    app = create_app(settings)
    import uvicorn

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
    uvicorn.run(app, host=bind, port=port or settings.port, log_config=None,
                server_header=False, date_header=False, **tls)


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
