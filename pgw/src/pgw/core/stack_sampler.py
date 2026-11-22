"""Opt-in wall-clock stack sampler for profiling a running gateway (diagnostics only; off by default).

``PGW_STACK_SAMPLER=<path>`` starts a daemon thread that samples every Python thread's stack each
``PGW_STACK_SAMPLER_MS`` milliseconds (default 10) and rewrites ``<path>`` (JSON) every few seconds with, per
thread, the most frequent leaf frames (exclusive) and frames on the stack (inclusive). A thread in a native
call that released the GIL shows the Python frame that made the call. Only code locations are recorded
(file, function, line), never values. A file ``<path>.reset`` (touched by an operator) restarts the counts,
so a window can be measured alone.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path

__all__ = ["maybe_start"]

_started = False


def _frame_key(f: object) -> str:
    code = f.f_code  # type: ignore[attr-defined]
    return f"{os.path.basename(code.co_filename)}:{code.co_name}:{f.f_lineno}"  # type: ignore[attr-defined]


def maybe_start() -> bool:
    """Start the sampler if ``PGW_STACK_SAMPLER`` names an output file (once per process)."""
    global _started
    out = os.environ.get("PGW_STACK_SAMPLER", "").strip()
    if not out or _started:
        return False
    _started = True
    period = max(0.0002, float(os.environ.get("PGW_STACK_SAMPLER_MS", "10") or 10) / 1e3)
    path = Path(out)
    reset = Path(out + ".reset")

    def run() -> None:
        me = threading.get_ident()
        excl: dict[str, Counter[str]] = {}
        incl: dict[str, Counter[str]] = {}
        samples: Counter[str] = Counter()
        t_start = time.time()
        last_dump = time.time()
        while True:
            time.sleep(period)
            if reset.exists():
                try:
                    reset.unlink()
                except OSError:
                    pass
                excl.clear()
                incl.clear()
                samples.clear()
                t_start = time.time()
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                if ident == me:
                    continue
                name = names.get(ident, str(ident))
                samples[name] += 1
                excl.setdefault(name, Counter())[_frame_key(frame)] += 1
                seen: set[str] = set()
                f = frame
                depth = 0
                while f is not None and depth < 60:
                    code = f.f_code
                    k = f"{os.path.basename(code.co_filename)}:{code.co_name}"
                    if k not in seen:
                        seen.add(k)
                        incl.setdefault(name, Counter())[k] += 1
                    f = f.f_back
                    depth += 1
            now = time.time()
            if now - last_dump >= 3.0:
                last_dump = now
                rep = {"since": t_start, "at": now, "period_ms": period * 1e3,
                       "threads": {n: {"samples": c, "exclusive": excl[n].most_common(25),
                                       "inclusive": incl[n].most_common(40)}
                                   for n, c in samples.most_common(64)}}
                try:
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(rep))
                    tmp.replace(path)
                except OSError:
                    pass

    threading.Thread(target=run, name="pgw-stack-sampler", daemon=True).start()
    return True
