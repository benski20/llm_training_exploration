"""``python -m pgw.factory`` -> the ``pgw factory`` command group."""

import os

# Same as ``pgw.cli``: idle OpenMP workers sleep instead of spinning (set before torch loads). Spinning workers
# made CPU factory runs up to 20x slower on a busy host.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from pgw.factory.cli import main  # after the OpenMP setting on purpose

if __name__ == "__main__":
    main()
