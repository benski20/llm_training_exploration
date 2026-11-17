"""Fixtures for the CLI and deployment tests."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from click.testing import CliRunner

PGW_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = PGW_ROOT / "deploy"


@pytest.fixture
def runner() -> CliRunner:
    """Click runner (stderr kept separate from stdout)."""
    return CliRunner()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every ``PGW_*`` variable so tests see default settings."""
    import os

    for k in list(os.environ):
        if k.startswith("PGW_") and k != "PGW_OFFLINE":
            monkeypatch.delenv(k, raising=False)
    return monkeypatch


def free_port() -> int:
    """An unused localhost TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
