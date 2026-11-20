"""pgw.core.proc_tuning: glibc mallopt tuning and the event-loop factory."""

from __future__ import annotations

import asyncio
import ctypes
import sys
from pathlib import Path
from typing import Any

import pytest

from pgw.core import proc_tuning as pt

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "_applied", None)
    monkeypatch.delenv("PGW_MALLOC_TUNING", raising=False)
    monkeypatch.delenv("PGW_UVLOOP", raising=False)


class _FakeGlibc:
    def __init__(self, ret: int = 1) -> None:
        self.calls: list[tuple[int, int]] = []
        self.ret = ret

        def version() -> bytes:
            return b"2.36"

        self.gnu_get_libc_version = version

    def mallopt(self, param: ctypes.c_int, value: ctypes.c_int) -> int:
        self.calls.append((param.value, value.value))
        return self.ret


class _FakeMusl:
    """A C library without ``gnu_get_libc_version`` (musl, macOS libSystem)."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def mallopt(self, *a: Any) -> int:  # pragma: no cover - must not be called
        self.calls.append(a)
        return 1


def test_sets_all_three_parameters_on_glibc() -> None:
    lib = _FakeGlibc()
    assert pt.apply_malloc_tuning(libc=lib) is True
    assert lib.calls == [(pt.M_TRIM_THRESHOLD, 256 << 20), (pt.M_TOP_PAD, 64 << 20), (pt.M_MMAP_THRESHOLD, 32 << 20)]
    # idempotent: a second call does not touch the allocator again
    assert pt.apply_malloc_tuning(libc=lib) is True
    assert len(lib.calls) == 3
    assert pt.apply_malloc_tuning(libc=lib, force=True) is True and len(lib.calls) == 6


def test_noop_on_non_glibc_platform() -> None:
    lib = _FakeMusl()
    assert pt.glibc_version(lib) is None
    assert pt.apply_malloc_tuning(libc=lib) is False
    assert lib.calls == []


def test_noop_when_process_libc_is_not_glibc(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = _FakeMusl()
    monkeypatch.setattr(pt, "_libc", lambda: lib)
    assert pt.apply_malloc_tuning() is False and lib.calls == []
    monkeypatch.setattr(pt, "_libc", lambda: None)
    assert pt.apply_malloc_tuning(force=True) is False


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_disabled_by_env(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("PGW_MALLOC_TUNING", value)
    lib = _FakeGlibc()
    assert pt.apply_malloc_tuning(libc=lib) is False
    assert lib.calls == []


def test_rejected_parameter_reports_false() -> None:
    lib = _FakeGlibc(ret=0)
    assert pt.apply_malloc_tuning(libc=lib) is False
    assert len(lib.calls) == 3


@pytest.mark.skipif(not sys.platform.startswith("linux") or pt.glibc_version() is None, reason="needs glibc")
def test_real_glibc_accepts_the_parameters() -> None:
    assert pt.glibc_version() is not None
    assert pt.apply_malloc_tuning() is True


def test_event_loop_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = pt.new_event_loop()
    try:
        assert isinstance(loop, asyncio.AbstractEventLoop)
        assert loop.run_until_complete(asyncio.sleep(0, result=7)) == 7
    finally:
        loop.close()
    try:
        import uvloop
    except ImportError:
        assert pt.loop_name() == "asyncio"
    else:
        assert pt.loop_name() == "uvloop"
        loop = pt.new_event_loop()
        assert isinstance(loop, uvloop.Loop)
        loop.close()
    monkeypatch.setenv("PGW_UVLOOP", "0")
    assert pt.loop_name() == "asyncio"
    loop = pt.new_event_loop()
    try:
        assert type(loop).__module__.startswith("asyncio")
    finally:
        loop.close()
    assert asyncio.run(asyncio.sleep(0, result=1), loop_factory=pt.new_event_loop) == 1


def test_deploy_env_matches_module_values() -> None:
    """Dockerfiles and systemd units set the same MALLOC_* values that apply_malloc_tuning uses."""
    files = sorted((ROOT / "deploy" / "docker").glob("*.Dockerfile"))
    files += [ROOT / "deploy" / "systemd" / f"pgw-{u}.service" for u in ("gateway", "factory")]
    assert len(files) >= 7
    for f in files:
        text = f.read_text()
        for k, v in pt.MALLOC_ENV.items():
            assert f"{k}={v}" in text, f"{f.name}: {k}={v} missing"
