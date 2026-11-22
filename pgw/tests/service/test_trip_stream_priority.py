"""``PGW_SERVICE_TRIP_STREAM_PRIORITY`` parsing (the stream itself needs a GPU)."""

from __future__ import annotations

import pytest

from pgw.core.errors import ConfigError
from pgw.service.gpu_exec import ENV_TRIP_STREAM_PRIORITY, trip_stream_priority


def test_default_off_and_high(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_TRIP_STREAM_PRIORITY, raising=False)
    assert trip_stream_priority() == 0
    for v in ("0", "off", ""):
        monkeypatch.setenv(ENV_TRIP_STREAM_PRIORITY, v)
        assert trip_stream_priority() == 0
    monkeypatch.setenv(ENV_TRIP_STREAM_PRIORITY, "high")
    assert trip_stream_priority() < 0
    monkeypatch.setenv(ENV_TRIP_STREAM_PRIORITY, "sideways")
    with pytest.raises(ConfigError):
        trip_stream_priority()
