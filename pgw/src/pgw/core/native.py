"""Optional native hot path (Rust extension ``pgw_fast``, built by ``rust/tools/build_fast.py``).

Every native function is bit-identical to the Python code it replaces (``tests/native``), so the
choice only affects speed. ``PGW_NATIVE=0`` forces the Python path.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["fast", "native_available"]


def _load() -> Any:
    if os.environ.get("PGW_NATIVE", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        import pgw_fast  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return None
    return pgw_fast


#: The loaded ``pgw_fast`` module, or ``None`` (use the Python path).
fast: Any = _load()


def native_available(*names: str) -> bool:
    """True if the extension is loaded and provides every function in ``names``."""
    return fast is not None and all(hasattr(fast, n) for n in names)
