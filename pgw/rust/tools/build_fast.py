#!/usr/bin/env python3
"""Build the native hot path (``pgw_fast``) and install it next to the ``pgw`` package.

    python rust/tools/build_fast.py            # release build -> src/pgw_fast.abi3.so

Needs a Rust toolchain (``rustup``) and the Python headers of the interpreter running this script.
``pgw`` works without it (pure Python fallback); ``PGW_NATIVE=0`` disables it at runtime.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # pgw/


def main() -> int:
    env = {**os.environ, "PYO3_PYTHON": sys.executable}
    subprocess.run(["cargo", "build", "--release", "-p", "pgw-fast"], cwd=ROOT / "rust", env=env, check=True)  # noqa: S603, S607
    lib = ROOT / "rust" / "target" / "release" / ("pgw_fast.dll" if os.name == "nt" else "libpgw_fast.so")
    if sys.platform == "darwin":
        lib = lib.with_suffix(".dylib")
    dst = ROOT / "src" / ("pgw_fast.pyd" if os.name == "nt" else "pgw_fast.abi3.so")
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(lib, tmp)
    os.replace(tmp, dst)  # atomic: a running process keeps its mapped copy
    print(f"installed {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
