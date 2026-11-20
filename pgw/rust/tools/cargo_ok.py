"""Exit 0 when ``cargo`` is on PATH and at least the given version (e.g. ``1.89``), else 1."""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    want = tuple(int(x) for x in sys.argv[1].split("."))
    exe = shutil.which("cargo")
    if exe is None:
        print("cargo not found")
        return 1
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False).stdout.split()  # noqa: S603
    try:
        have = tuple(int(x) for x in out[1].split("-")[0].split(".")[: len(want)])
    except (IndexError, ValueError):
        print(f"cannot read the cargo version: {' '.join(out)!r}")
        return 1
    print(f"cargo {'.'.join(map(str, have))}, need {sys.argv[1]}")
    return 0 if have >= want else 1


if __name__ == "__main__":
    sys.exit(main())
