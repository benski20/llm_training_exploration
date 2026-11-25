#!/usr/bin/env python3
"""Build the pks-v3 extension (``pgw_pks``, crate ``rust/pgw-pks``) and install it next to ``pgw``.

    python rust/tools/build_pks.py            # release build -> src/pgw_pks.abi3.so
    python rust/tools/build_pks.py --site     # ... and also into site-packages (non-editable installs)

Needs a Rust toolchain and the Python headers of the interpreter running this script. ``pgw`` works
without it: :mod:`pgw.hemask.pks_native` then uses the numpy reference :mod:`pgw.hemask.pks` (same bits,
far slower; ``PGW_NATIVE=0`` forces that). AVX-512 / AVX2 copies are compiled in and picked at run time
(``PGW_PKS_ISA`` caps them), so one build runs on any x86-64 CPU.
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
    subprocess.run(["cargo", "build", "--release", "-p", "pgw-pks"], cwd=ROOT / "rust", env=env, check=True)  # noqa: S607
    lib = ROOT / "rust" / "target" / "release" / ("pgw_pks.dll" if os.name == "nt" else "libpgw_pks.so")
    if sys.platform == "darwin":
        lib = lib.with_suffix(".dylib")
    dst = ROOT / "src" / ("pgw_pks.pyd" if os.name == "nt" else "pgw_pks.abi3.so")
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
