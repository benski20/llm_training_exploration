"""Shared CLI plumbing: error handling / exit codes, lazy sub-command loading, output helpers.

Exit codes (SPEC section 12): ``0`` ok, ``1`` :class:`pgw.core.PGWError` or invalid configuration,
``3`` not implemented, ``130`` interrupted (SIGINT).
"""

from __future__ import annotations

import functools
import importlib
import json
import logging
import sys
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

import click

from pgw.core.errors import PGWError

__all__ = [
    "EXIT_ERROR",
    "EXIT_INTERRUPTED",
    "EXIT_NOT_IMPLEMENTED",
    "EXIT_OK",
    "LazyGroup",
    "echo_json",
    "handle_errors",
    "short_validation",
]

log = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_IMPLEMENTED = 3
EXIT_INTERRUPTED = 130

F = TypeVar("F", bound=Callable[..., Any])


def _validation_error_type() -> type[Exception] | None:
    try:
        from pydantic import ValidationError
    except ImportError:  # pragma: no cover - pydantic is a core dependency
        return None
    return ValidationError


def handle_errors(fn: F) -> F:
    """Map expected failures to exit codes and one-line messages on stderr (no tracebacks).

    * ``NotImplementedError`` -> exit 3 (an unfinished component).
    * :class:`PGWError` -> exit 1, printed as ``error [<code>]: <message>``.
    * pydantic ``ValidationError`` (bad ``PGW_*`` settings) -> exit 1 as ``error [config]``.
    * ``KeyboardInterrupt`` -> exit 130.
    * ``OSError`` (file system / permission problems) -> exit 1 as ``error [os]``.
    """
    validation_error = _validation_error_type()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            click.echo(f"error: not implemented yet: {exc}", err=True)
            sys.exit(EXIT_NOT_IMPLEMENTED)
        except PGWError as exc:
            click.echo(f"error [{exc.code}]: {exc.message}", err=True)
            sys.exit(EXIT_ERROR)
        except KeyboardInterrupt:
            click.echo("interrupted", err=True)
            sys.exit(EXIT_INTERRUPTED)
        except OSError as exc:  # missing file, permission denied, disk full: one line, no traceback
            log.debug("OS error in %s", fn.__name__, exc_info=True)
            where = f" ({exc.filename})" if exc.filename else ""
            click.echo(f"error [os]: {exc.strerror or type(exc).__name__}{where}", err=True)
            sys.exit(EXIT_ERROR)
        except Exception as exc:
            if validation_error is not None and isinstance(exc, validation_error):
                click.echo(f"error [config]: {short_validation(exc)}", err=True)
                sys.exit(EXIT_ERROR)
            raise

    return wrapper  # type: ignore[return-value]


def short_validation(exc: Exception) -> str:
    """Compact rendering of a pydantic ValidationError (field paths + messages, no input values)."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    parts = []
    for e in errors(include_input=False, include_url=False):
        loc = ".".join(str(x) for x in e.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {e.get('msg', 'invalid')}")
    title = getattr(exc, "title", "settings")
    return f"{title}: " + "; ".join(parts)


def echo_json(obj: Any) -> None:
    """Print ``obj`` as indented, key-sorted JSON (non-JSON values via ``str``)."""
    click.echo(json.dumps(obj, indent=1, sort_keys=True, default=str))


class LazyGroup(click.Group):
    """A click group whose sub-commands are imported on first use.

    ``lazy_subcommands`` maps a command name to ``(module, attribute, extra)``. Importing lazily
    keeps ``pgw --help`` fast and lets one CLI ship in images that install only some extras (the
    service image has no gateway dependencies, for example). If a lazy command cannot be imported
    the user gets a clear message naming the missing extra instead of a traceback.
    """

    def __init__(self, *args: Any, lazy_subcommands: Mapping[str, tuple[str, str, str]] | None = None,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lazy_subcommands: dict[str, tuple[str, str, str]] = dict(lazy_subcommands or {})

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self.lazy_subcommands})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name in self.lazy_subcommands:
            return self._load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _load(self, name: str) -> click.Command:
        module_name, attr, extra = self.lazy_subcommands[name]
        try:
            module = importlib.import_module(module_name)
            cmd = getattr(module, attr)
        except ImportError as exc:
            log.debug("cannot import %s: %s", module_name, exc)
            return _unavailable_command(name, extra, exc)
        if not isinstance(cmd, click.Command):  # pragma: no cover - programming error
            raise TypeError(f"{module_name}.{attr} is not a click command")
        return cmd


def _unavailable_command(name: str, extra: str, exc: ImportError) -> click.Command:
    missing = getattr(exc, "name", None) or str(exc)

    @click.command(name, context_settings={"ignore_unknown_options": True, "allow_extra_args": True,
                                           "help_option_names": []},
                   help=f"(unavailable: install pgw-sdk[{extra}])", add_help_option=False)
    @click.pass_context
    def _cmd(ctx: click.Context) -> None:
        click.echo(f"error [config]: 'pgw {name}' needs the '{extra}' extra (missing module: {missing}); "
                   f"install it with: pip install 'pgw-sdk[{extra}]'", err=True)
        ctx.exit(EXIT_ERROR)

    return _cmd
