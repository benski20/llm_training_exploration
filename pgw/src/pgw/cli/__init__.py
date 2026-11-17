"""``pgw`` command line (entry point ``pgw.cli.main:main``).

Owner: cli team (command bodies delegate to the owning subpackages). Contract: SPEC.md section 12.
"""

from pgw.cli.main import cli, main

__all__ = ["cli", "main"]
