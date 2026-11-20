#!/usr/bin/env python3
"""Benchmark-only timing probe for a live gateway process: splits the decode "unaccounted" time.

``gpu_validate.py`` measures a streaming request's decode seconds per token and the backend's
masked-trip counters (remote = round trips, local = prepare + finish). What is left (run 27: 98 ms of
349 ms per 7B token) is gateway plaintext work and loop overhead. This launcher starts the real
``pgw`` CLI with timers wrapped around the engine functions that make up that rest:

=================  ==========================================================================
``attention``      ``pgw.engine.model._attend`` (SDPA / GQA over the KV cache)
``norm``           ``pgw.engine.model.rms_norm`` (layer norms; the final norm counts under ``head``)
``rope``           ``apply_rope`` + ``RotaryEmbedding.cos_sin``
``kv``             ``KVCache.append``
``head``           ``TransformerModel._logits`` (final norm + exact LM head)
``consume``        ``Generator._consume_batch`` (sampling, logprobs, stop scan, detokenization)
``draft``          ``Generator._plan_drafts`` (speculation only; the draft model's own float ops
                   and head count here, not above)
``step``           ``Generator._run_step`` (wall time of one engine step, async)
=================  ==========================================================================

Timing is exclusive: a wrapped call inside another wrapped call on the same thread is not timed
again, so the components never double count. Overhead is about 1-2 us per wrapped call (about 200
calls per 7B token, i.e. well under 1 ms per token).

Snapshots: on ``SIGUSR2`` the probe writes ``{"seq", "counters": {name: [seconds, calls]},
"missing": [...]}`` atomically to ``--probe-file``; :func:`snapshot` sends the signal and waits for
the new ``seq``. A wrapped name that no longer exists (a refactor) is listed under ``missing`` and
simply not timed; the gateway itself always starts.

Usage (gpu_validate does this when ``--attribute`` is on, the default)::

    python benchmarks/gateway_probe.py --probe-file /tmp/probe.json -- gateway serve
"""

from __future__ import annotations

import functools
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

#: (module, attribute path, component). Attribute paths with a dot are class attributes.
TARGETS: tuple[tuple[str, str, str], ...] = (
    ("pgw.engine.model", "_attend", "attention"),
    ("pgw.engine.model", "rms_norm", "norm"),
    ("pgw.engine.model", "apply_rope", "rope"),
    ("pgw.engine.rope", "RotaryEmbedding.cos_sin", "rope"),
    ("pgw.engine.kv_cache", "KVCache.append", "kv"),
    ("pgw.engine.model", "TransformerModel._logits", "head"),
    ("pgw.engine.generate", "Generator._consume_batch", "consume"),
    ("pgw.engine.generate", "Generator._plan_drafts", "draft"),
)
ASYNC_TARGETS: tuple[tuple[str, str, str], ...] = (
    ("pgw.engine.generate", "Generator._run_step", "step"),
)

_lock = threading.RLock()  # re-entrant: the SIGUSR2 handler may interrupt _add on the main thread
_counters: dict[str, list[float]] = {}
_missing: list[str] = []
_undo: list[tuple[Any, str, Any]] = []
_tls = threading.local()
_seq = 0


def _add(name: str, dt: float) -> None:
    with _lock:
        c = _counters.setdefault(name, [0.0, 0])
        c[0] += dt
        c[1] += 1


def _timed(fn: Any, name: str) -> Any:
    @functools.wraps(fn)
    def wrapper(*a: Any, **k: Any) -> Any:
        if getattr(_tls, "depth", 0):
            return fn(*a, **k)
        _tls.depth = 1
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _tls.depth = 0
            _add(name, time.perf_counter() - t)

    return wrapper


def _timed_async(fn: Any, name: str) -> Any:
    @functools.wraps(fn)
    async def wrapper(*a: Any, **k: Any) -> Any:
        t = time.perf_counter()
        try:
            return await fn(*a, **k)
        finally:
            _add(name, time.perf_counter() - t)

    return wrapper


def _patch(module: str, path: str, name: str, wrap: Any) -> None:
    import importlib

    try:
        owner: Any = importlib.import_module(module)
        *parents, attr = path.split(".")
        for p in parents:
            owner = getattr(owner, p)
        # class attributes: wrap what the class defines itself (staticmethod/classmethod are not used here)
        fn = owner.__dict__[attr] if isinstance(owner, type) else getattr(owner, attr)
        if not callable(fn):
            raise TypeError(f"{module}.{path} is not callable")
        _undo.append((owner, attr, fn))
        setattr(owner, attr, wrap(fn, name))
    except Exception as exc:  # a renamed function must never stop the gateway
        _missing.append(f"{module}.{path}: {type(exc).__name__}")


def _write(path: Path) -> None:
    global _seq
    _seq += 1
    data = {"seq": _seq, "t": time.perf_counter(), **counters()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def install(probe_file: Path | None) -> None:
    """Wrap the targets; with ``probe_file``, also install the ``SIGUSR2`` snapshot handler (main thread only)
    and write the first snapshot."""
    import pgw.cli  # noqa: F401 - its environment defaults (OMP_WAIT_POLICY) must apply before torch loads

    for module, path, name in TARGETS:
        _patch(module, path, name, _timed)
    for module, path, name in ASYNC_TARGETS:
        _patch(module, path, name, _timed_async)
    if probe_file is not None:
        signal.signal(signal.SIGUSR2, lambda _s, _f: _write(probe_file))
        _write(probe_file)


def counters() -> dict[str, Any]:
    """Current counters ``{name: [seconds, calls]}`` and the targets that could not be wrapped."""
    with _lock:
        return {"counters": {k: list(v) for k, v in _counters.items()}, "missing": list(_missing)}


def uninstall() -> None:
    """Restore every wrapped function and clear the counters (tests)."""
    while _undo:
        owner, attr, fn = _undo.pop()
        setattr(owner, attr, fn)
    with _lock:
        _counters.clear()
        _missing.clear()


def snapshot(pid: int, probe_file: Path, timeout: float = 10.0) -> dict[str, Any] | None:
    """Ask the probed process ``pid`` for a snapshot and return it (None if it does not answer)."""
    try:
        before = json.loads(probe_file.read_text())["seq"]
    except (OSError, ValueError, KeyError):
        return None
    try:
        os.kill(pid, signal.SIGUSR2)
    except OSError:
        return None
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            data = json.loads(probe_file.read_text())
            if data["seq"] > before:
                return dict(data)
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.005)
    return None


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 3 or argv[0] != "--probe-file" or "--" not in argv:
        raise SystemExit("usage: gateway_probe.py --probe-file FILE -- <pgw cli arguments>")
    cut = argv.index("--")
    install(Path(argv[1]))
    import runpy

    sys.argv = ["pgw", *argv[cut + 1 :]]
    runpy.run_module("pgw.cli", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
