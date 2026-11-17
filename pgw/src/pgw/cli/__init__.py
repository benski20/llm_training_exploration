"""``pgw`` command line (entry point ``pgw.cli.main:main``).

Owner: cli team (command bodies delegate to the owning subpackages). Contract: SPEC.md section 12.
"""

import os

# Before torch loads its OpenMP runtime: idle OpenMP workers sleep instead of spin-waiting. The
# servers alternate many small torch ops with network waits; spinning workers burned ~2/3 of the
# gateway's CPU and competed with the co-located service for cores (measured: 63 -> 24 ms CPU per
# token on the tiny model, same output). Operators can still override it in the environment.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from pgw.cli.main import cli, main  # after the OpenMP setting on purpose

__all__ = ["cli", "main"]
