"""Every ``pgw`` invocation shipped in ``deploy/`` and ``scripts/`` parses with the real CLI.

Commands and options are resolved against :data:`pgw.cli.main.cli` with click's own parser (no
command is executed), so a renamed command or option breaks this test instead of a demo or a
production unit file.
"""

from __future__ import annotations

import json
import re
import shlex
import warnings
from pathlib import Path

import click
import pytest
import yaml

from pgw.cli.main import cli

PGW_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = PGW_ROOT / "deploy"

_PGW_CALL = re.compile(r"(?:^|[\s\"'(=/])pgw((?: [^|;&>)\n#`]+)+)")


def _resolve(args: list[str]) -> click.Command:
    """Parse ``args`` with click down to the leaf command; raise on unknown commands/options."""
    cmd: click.Command = cli
    ctx = cli.make_context("pgw", list(args), resilient_parsing=False)
    with ctx, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # click 8.2: protected_args
        while isinstance(cmd, click.Group):
            rest = [*ctx.protected_args, *ctx.args] if hasattr(ctx, "protected_args") else list(ctx.args)
            if not rest:
                break
            name, sub, rest2 = cmd.resolve_command(ctx, rest)
            assert sub is not None, name
            cmd = sub
            ctx = sub.make_context(name or "", rest2, parent=ctx)
    return cmd


def _norm(arg: str) -> str:
    """Replace shell/compose variables with a harmless placeholder value."""
    arg = re.sub(r"\$\{PGW_DEMO_SLOTS:-?(\d+)\}", r"\1", arg)
    arg = re.sub(r"\$\{?[A-Z_]*(SLOTS|PORT|SEGMENTS)[A-Z_]*\}?", "1", arg)  # integer-valued variables
    arg = re.sub(r"\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*", "X", arg)
    return arg if arg else "X"


def _shell_calls(text: str) -> list[list[str]]:
    out = []
    text = text.replace("\\\n", " ").replace('"${PGW[@]}"', "pgw")  # line continuations; demo script's pgw array
    for line in text.splitlines():
        if line.lstrip().startswith("#") or line.lstrip().startswith("echo"):
            continue
        for m in _PGW_CALL.finditer(line):
            try:
                parts = shlex.split(m.group(1))
            except ValueError:
                continue
            if parts and re.fullmatch(r"[a-z][a-z-]*", parts[0]):
                out.append([_norm(p) for p in parts])
    return out


def _collect() -> list[tuple[str, list[str]]]:
    calls: list[tuple[str, list[str]]] = []
    compose = yaml.safe_load((DEPLOY / "compose" / "docker-compose.yml").read_text())
    for name, svc in compose["services"].items():
        c = svc.get("command")
        if isinstance(c, list) and c and c[0] == "pgw":
            calls.append((f"compose:{name}", [_norm(str(a)) for a in c[1:]]))
    for df in sorted((DEPLOY / "docker").glob("*.Dockerfile")):
        for m in re.finditer(r'^CMD (\[.*\])', df.read_text(), re.M):
            c = json.loads(m.group(1))
            if c[0] == "pgw":
                calls.append((df.name, c[1:]))
    files = [DEPLOY / "compose" / "demo-init.sh", *sorted((DEPLOY / "systemd").glob("*.service")),
             DEPLOY / "vmware" / "pgw-ovf-config.sh", PGW_ROOT / "scripts" / "e2e_demo.sh"]
    for f in files:
        if f.is_file():
            calls += [(f.name, c) for c in _shell_calls(f.read_text())]
    return calls


CALLS = _collect()


def test_found_the_expected_invocations() -> None:
    where = {w for w, _ in CALLS}
    assert {"compose:factory", "compose:factory-refill", "demo-init.sh", "e2e_demo.sh",
            "gateway.Dockerfile", "pgw-gateway.service"} <= where, where
    assert len(CALLS) >= 15


@pytest.mark.parametrize(("where", "args"), CALLS, ids=[f"{w}:{' '.join(a[:3])}" for w, a in CALLS])
def test_invocation_parses_with_the_real_cli(where: str, args: list[str]) -> None:
    try:
        cmd = _resolve(args)
    except click.BadParameter as exc:
        if "does not exist" not in exc.format_message():  # paths of the target host are fine
            pytest.fail(f"{where}: 'pgw {' '.join(args)}' does not parse: {exc.format_message()}")
        return
    except click.UsageError as exc:  # unknown command/option, missing argument ...
        pytest.fail(f"{where}: 'pgw {' '.join(args)}' does not parse: {exc.format_message()}")
    assert not isinstance(cmd, click.Group) or args[-1] in ("--help", "-h"), f"{where}: {args} is a group"
