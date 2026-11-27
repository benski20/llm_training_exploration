"""``pgw chat``: the private chat app on this machine (localhost server + the browser)."""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

import click

__all__ = ["chat_command", "harden_process"]


def harden_process() -> None:
    """No core dumps and no ptrace attach by other processes of the same user (Linux), so a crash or a stray debugger
    does not copy conversation text out of this process's memory."""
    with contextlib.suppress(Exception):
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sys.platform.startswith("linux"):
        with contextlib.suppress(Exception):
            import ctypes

            ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 4


def write_opener(path: Path, url: str) -> None:
    """A 0600 file that sends the browser to ``url``: the token never appears on a command line (``ps``) of the
    browser launcher, where other users of the machine could read it."""
    from pgw.core.fileio import atomic_write_text

    html = (f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={url}">'
            f'<title>PGW Chat</title><p><a href="{url}">Open PGW Chat</a></p>\n')
    atomic_write_text(path, html, mode=0o600)


@click.command("chat")
@click.option("--model", default=None, help="Model selected first (default: the one recommended for this machine)")
@click.option("--port", type=int, default=0, show_default=True, help="Port on 127.0.0.1 (0: a free one)")
@click.option("--no-browser", is_flag=True, help="Print the link instead of opening the browser")
@click.option("--no-history", is_flag=True, help="Keep conversations in memory only (nothing written to disk)")
@click.option("--factory-weights", type=click.Choice(["pack", "stream"]), default=None,
              help="stream: keep no int8 weights on disk (less disk, a little more network) [PGW_FACTORY_WEIGHTS]")
@click.option("--prefix-cache", type=click.Choice(["auto", "on", "off"]), default="auto", show_default=True,
              help="Reuse earlier turns' KV state (on: the provider can tell a turn continued an earlier one)")
@click.option("--speculation", type=click.Choice(["off", "draft_head", "lookup"]), default=None,
              help="Strict speculative decoding (same answers; fewer round trips per token) [PGW_SPECULATION]")
@click.option("--usage-reports/--no-usage-reports", default=True, show_default=True,
              help="Send the provider content-free request records (counts and timing it sees anyway)")
@click.option("--max-tokens", type=click.IntRange(1, 8192), default=1024, show_default=True)
@click.option("--demo", is_flag=True, help="Canned answers, no model or provider (UI work and screenshots)")
@click.option("--cache-dir", type=click.Path(path_type=Path), default=None, help="[PGW_CACHE_DIR]")
@click.option("--state-dir", type=click.Path(path_type=Path), default=None, help="[PGW_STATE_DIR]")
def chat_command(model: str | None, port: int, no_browser: bool, no_history: bool, factory_weights: str | None,
                 prefix_cache: str, speculation: str | None, usage_reports: bool, max_tokens: int, demo: bool,
                 cache_dir: Path | None, state_dir: Path | None) -> None:
    """Private chat on this computer: plaintext stays here, the GPU sees only masked data.

    Starts a server on 127.0.0.1 that only this browser session can use (a random token per launch) and opens the
    chat in your browser. Needs your SDK key in PGW_API_KEY (or asks for it).
    """
    from pgw.chat.server import ChatServer
    from pgw.chat.store import DiskStore, MemoryStore
    from pgw.sdk.cache import CacheLayout
    from pgw.sdk.options import SUPPORTED_MODELS

    harden_process()
    if model is not None and model not in SUPPORTED_MODELS:
        raise click.BadParameter(f"supported: {', '.join(SUPPORTED_MODELS)}", param_hint="--model")
    layout = CacheLayout(cache_dir, state_dir)
    history_dir = layout.state / "chat" / "conversations"
    if not no_history or not no_browser:
        from pgw.sdk.cache import ensure_private_dir

        ensure_private_dir(layout.state / "chat")  # 0700, like the SDK's own directories
    store = MemoryStore() if no_history else DiskStore(history_dir)
    if demo:
        from pgw.chat.testing import FakeBackend

        backend: Any = FakeBackend()
    else:
        from pgw.chat.backend import SDKBackend
        from pgw.sdk.client import OpenAI

        key = os.environ.get("PGW_API_KEY")
        if not key and sys.stdin.isatty():
            key = click.prompt("Your PGW key (pgw_...)", hide_input=True)
        if not key:
            raise click.ClickException("set PGW_API_KEY to your key")
        opts: dict[str, Any] = {"usage_reports": usage_reports,
                                "prefix_cache": {"auto": "auto", "on": True, "off": False}[prefix_cache]}
        for k, v in (("cache_dir", cache_dir), ("state_dir", state_dir), ("factory_weights", factory_weights),
                     ("speculation", speculation)):
            if v is not None:
                opts[k] = v
        try:
            backend = SDKBackend(OpenAI(api_key=key, **opts))
        except Exception as exc:
            raise click.ClickException(getattr(exc, "message", None) or type(exc).__name__) from None
        del key
    server = ChatServer(backend, store, port=port, default_model=model, max_tokens=max_tokens,
                        history_dir=history_dir if not no_history else None, demo=demo)
    if not demo:
        backend.warmup(server.default_model)  # first-run setup starts now; the page shows its progress
    opener = None
    click.echo(f"PGW Chat on {server.url}  (Ctrl-C to stop)", err=True)
    if no_browser:
        click.echo(f"Open: {server.open_url}", err=True)
    else:
        opener = layout.state / "chat" / "open.html"
        opener.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_opener(opener, server.open_url)
        import webbrowser

        if not webbrowser.open(opener.as_uri()):
            click.echo(f"Open: {server.open_url}", err=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    server.start()
    try:
        while not stop.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        click.echo("stopping PGW Chat", err=True)
        server.shutdown()
        with contextlib.suppress(Exception):
            backend.close()
        if opener is not None:
            with contextlib.suppress(OSError):
                opener.unlink()
