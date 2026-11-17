"""``python -m pgw.service``: run the compute service with settings from ``PGW_SERVICE_*``."""

from __future__ import annotations

import sys

from pydantic import ValidationError

from pgw.core.errors import PGWError


def main() -> int:
    """Run the service; exit code 0 on clean shutdown, 1 on :class:`PGWError`."""
    from pgw.service.server import serve

    try:
        serve()
    except PGWError as exc:
        print(f"pgw service: {exc.code}: {exc.message}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"pgw service: invalid configuration: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
