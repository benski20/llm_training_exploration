#!/usr/bin/env python3
"""Smoke test of an INSTALLED pgw-sdk (wheel in a fresh venv). Run it with that venv's python, from a
directory outside the repository, so ``import pgw`` cannot pick up the source tree:

    cd /tmp && /path/to/venv/bin/python /path/to/pgw/scripts/smoke_installed.py [--allow-pure] [--require-sdk]

Checks: the package imports from site-packages; ``pgw.__version__`` matches the distribution metadata;
``py.typed`` and the license files ship; the three native extensions load (unless ``--allow-pure``) and
an NTT product is bit-identical to an independent numpy reference; ``pgw doctor`` reports no failure;
the client API (``from pgw import OpenAI, AsyncOpenAI``) imports when ``pgw.sdk`` is part of the build
(required with ``--require-sdk``). No network access, no secrets. Exit code 0 = all good.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

DIST = "pgw-sdk"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-pure", action="store_true", help="accept missing native extensions (fallback)")
    ap.add_argument("--require-sdk", action="store_true", help="fail if pgw.sdk (from pgw import OpenAI) is absent")
    args = ap.parse_args()

    import pgw

    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    pkg = Path(pgw.__file__).resolve().parent
    check("import location", pkg.parent == site, str(pkg))
    check("version", pgw.__version__ == md.version(DIST), f"pgw.__version__={pgw.__version__}, dist={md.version(DIST)}")
    check("py.typed", (pkg / "py.typed").is_file())
    lic = {Path(str(f)).name for f in (md.files(DIST) or []) if "licenses" in str(f)}
    check("license files", {"LICENSE", "THIRD_PARTY_NOTICES.txt"} <= lic, ", ".join(sorted(lic)))
    meta = md.metadata(DIST)
    check("license metadata", meta.get("License-Expression") == "LicenseRef-Proprietary",
          str(meta.get("License-Expression")))

    # --- native extensions -------------------------------------------------------------------------
    from pgw.core import native as fast_mod
    from pgw.hemask import ntt, pks_native

    loaded = {"pgw_fast": fast_mod.fast, "pgw_ntt": ntt.native, "pgw_pks": pks_native.native()}
    for name, mod in loaded.items():
        if mod is None:
            check(f"native {name}", args.allow_pure, "not loaded (pure-Python fallback)")
        else:
            where = Path(mod.__file__).resolve()
            check(f"native {name}", where.parent == site and ".abi3." in where.name, where.name)
    if ntt.native is not None:
        import numpy as np

        from pgw.hemask import lwe

        rng = np.random.default_rng(7)
        smat = lwe.secret_matrix(rng.integers(-1, 2, size=(2, ntt.DEGREE), dtype=np.int8))
        a = rng.integers(0, 2**64, size=(5, smat.shape[0]), dtype=np.uint64)
        calls = 0
        real = ntt.native.mm_u64

        def spy(*a_: object, **kw: object) -> object:
            nonlocal calls
            calls += 1
            return real(*a_, **kw)

        ntt.native.mm_u64 = spy
        try:
            got = ntt.matmul_mod64(smat, a, small_left=False)
        finally:
            ntt.native.mm_u64 = real
        want = a @ smat.astype(np.int64).view(np.uint64)  # numpy wraps mod 2^64
        check("native NTT bit-identical", calls == 1 and bool(np.array_equal(got, want)),
              f"native calls={calls}, isa={ntt.info().get('isa', '?')}")

    # --- doctor ------------------------------------------------------------------------------------
    from pgw.cli.doctor import format_report, run_doctor

    rep = run_doctor()
    fails = [c.name for c in rep.checks if c.status == "fail"]
    check("pgw doctor", not fails, "no failures" if not fails else "failed: " + ", ".join(fails))
    if fails:
        print(format_report(rep))
    exe = Path(sys.executable).parent / "pgw"
    if exe.exists():
        out = subprocess.run([str(exe), "version"], capture_output=True, text=True, check=False)  # noqa: S603
        check("pgw CLI", out.returncode == 0 and pgw.__version__ in out.stdout, out.stdout.strip())

    # --- client API --------------------------------------------------------------------------------
    if importlib.util.find_spec("pgw.sdk") is not None:
        from pgw import AsyncOpenAI, OpenAI  # type: ignore[attr-defined,unused-ignore]

        check("client API", callable(OpenAI) and callable(AsyncOpenAI), "from pgw import OpenAI, AsyncOpenAI")
    else:
        check("client API", not args.require_sdk, "pgw.sdk not in this build (skipped)")

    print(f"\n{'SMOKE TEST PASSED' if not failures else 'SMOKE TEST FAILED: ' + ', '.join(failures)}"
          f" (python {sys.version.split()[0]}, PGW_NATIVE={os.environ.get('PGW_NATIVE', '1')})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
