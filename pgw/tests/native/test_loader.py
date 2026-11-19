"""The native extension loads (when built) and can be disabled."""

from __future__ import annotations

import pytest

from pgw.core import native


def test_loader_reports_version_when_built() -> None:
    if native.fast is None:
        pytest.skip("pgw_fast not built (python rust/tools/build_fast.py)")
    assert isinstance(native.fast.version(), str)
    assert native.native_available("version") and not native.native_available("no_such_function")
