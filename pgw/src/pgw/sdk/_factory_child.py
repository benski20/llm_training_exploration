"""Entry point of the background mask factory process (``python -m pgw.sdk._factory_child``).

Runs the existing factory (:mod:`pgw.factory`, settings from ``PGW_FACTORY_*`` / ``PGW_MASKSTORE_*`` set by
:mod:`pgw.sdk.factory`) and exits when the process that started it exits: on Linux through
``PR_SET_PDEATHSIG`` (SIGTERM; the factory aborts its current segment, nothing partial is committed), elsewhere
through a parent-pid watch thread.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time


def _die_with_parent(parent: int) -> None:
    if sys.platform.startswith("linux"):
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)  # PR_SET_PDEATHSIG
        except Exception:  # pragma: no cover - platform dependent
            pass
    if parent and os.getppid() != parent:  # the parent died before the prctl took effect
        os._exit(0)

    def watch() -> None:
        while True:
            time.sleep(2.0)
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    if parent:
        threading.Thread(target=watch, name="pgw-parent-watch", daemon=True).start()


def main() -> None:
    """Run the factory until SIGTERM (or the parent's exit)."""
    _die_with_parent(int(os.environ.get("PGW_SDK_PARENT_PID", "0") or 0))
    from pgw.core.log import configure_logging

    configure_logging(level=os.environ.get("PGW_LOG_LEVEL", "INFO"), json_lines=True)
    from pgw.factory.cli import _factory_settings, _handle_errors, _run_factory

    @_handle_errors
    def run() -> None:
        _run_factory(_factory_settings(), False, None, None)

    run()


if __name__ == "__main__":
    main()
