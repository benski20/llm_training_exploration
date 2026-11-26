"""Entry point of the background mask factory process (``python -m pgw.sdk._factory_child``).

Runs the existing factory (:mod:`pgw.factory`, settings from ``PGW_FACTORY_*`` / ``PGW_MASKSTORE_*`` set by
:mod:`pgw.sdk.factory`) and exits when the process that started it exits: on Linux through
``PR_SET_PDEATHSIG`` (SIGTERM; the factory aborts its current segment, nothing partial is committed) and a
parent-pid watch thread, with a hard exit if the graceful stop hangs (:func:`pgw.core.proc_tuning.exit_with_parent`).
Its fleet workers do the same with the factory process as their parent.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

#: The child touches this file in its state directory every few seconds; the supervisor restarts a child whose
#: heartbeat is older than its hang limit (stopped, or stuck with the interpreter lock held).
HEARTBEAT_FILE = "heartbeat"
HEARTBEAT_S = 5.0


def _heartbeat(path: Path) -> None:
    def beat() -> None:
        while True:
            try:
                path.touch(mode=0o600, exist_ok=True)
                os.utime(path)
            except OSError:  # a full or read-only disk is reported by the factory itself
                pass
            time.sleep(HEARTBEAT_S)

    threading.Thread(target=beat, name="pgw-factory-heartbeat", daemon=True).start()


def main() -> None:
    """Run the factory until SIGTERM (or the parent's exit)."""
    from pgw.core.proc_tuning import exit_with_parent

    # SIGTERM with the parent (the factory aborts its current segment, nothing partial is committed); a hard exit
    # if that stop does not finish within the grace period
    exit_with_parent(int(os.environ.get("PGW_SDK_PARENT_PID", "0") or 0) or None)
    _heartbeat(Path(os.environ.get("PGW_FACTORY_STATE_DIR", ".")) / HEARTBEAT_FILE)
    from pgw.core.log import configure_logging

    configure_logging(level=os.environ.get("PGW_LOG_LEVEL", "INFO"), json_lines=True)
    from pgw.factory.cli import _factory_settings, _handle_errors, _run_factory

    @_handle_errors
    def run() -> None:
        _run_factory(_factory_settings(), False, None, None)

    run()


if __name__ == "__main__":
    main()
