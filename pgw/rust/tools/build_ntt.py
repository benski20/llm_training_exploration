#!/usr/bin/env python3
"""Build the NTT extension (``pgw_ntt``, crate ``rust/pgw-ntt``) and install it next to ``pgw``.

    python rust/tools/build_ntt.py            # release build -> src/pgw_ntt.abi3.so
    python rust/tools/build_ntt.py --site     # ... and also into site-packages (non-editable installs)

Needs a Rust toolchain and the Python headers of the interpreter running this script. ``pgw`` works
without it: :mod:`pgw.hemask.ntt` then falls back to the dense products (``PGW_NTT=0`` or
``PGW_NATIVE=0`` force that at run time). AVX-512 / AVX2 kernels are compiled in and picked at run
time, so one build runs on any x86-64 CPU.
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
    subprocess.run(["cargo", "build", "--release", "-p", "pgw-ntt"], cwd=ROOT / "rust", env=env, check=True)  # noqa: S607
    lib = ROOT / "rust" / "target" / "release" / ("pgw_ntt.dll" if os.name == "nt" else "libpgw_ntt.so")
    if sys.platform == "darwin":
        lib = lib.with_suffix(".dylib")
    dst = ROOT / "src" / ("pgw_ntt.pyd" if os.name == "nt" else "pgw_ntt.abi3.so")
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(lib, tmp)
    os.replace(tmp, dst)  # atomic: a running process keeps its mapped copy
    # An editable install (setup.py) leaves e.g. pgw_fast.cpython-313-*.so, which Python would import first.
    for stale in dst.parent.glob(dst.name.split(".")[0] + ".*"):
        if stale != dst and stale.suffix in (".so", ".pyd"):
            stale.unlink()
    print(f"installed {dst}")
    if "--site" in sys.argv[1:]:
        import sysconfig

        site = Path(sysconfig.get_paths()["purelib"]) / dst.name
        shutil.copy2(dst, site)
        print(f"installed {site}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
