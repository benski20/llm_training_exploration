"""``pgw doctor``'s throw-away mask store ignores PGW_MASKSTORE_* environment overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from pgw.cli.doctor import _settings_without_env
from pgw.maskstore.store import MaskStoreSettings


def test_throwaway_store_settings_ignore_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_MASKSTORE_FSYNC", "false")
    monkeypatch.setenv("PGW_MASKSTORE_COUNTER", "file")
    monkeypatch.setenv("PGW_MASKSTORE_MIN_FREE_BYTES", "123456789")
    base = MaskStoreSettings(root=tmp_path / "a", key_provider="file", key_file=tmp_path / "k")
    assert base.fsync is False  # the env does reach normal settings
    s = _settings_without_env(MaskStoreSettings, {"root": tmp_path / "s", "key_provider": "file",
                                                  "key_file": tmp_path / "k", "counter": "none",
                                                  "min_free_bytes": 0})
    assert isinstance(s, MaskStoreSettings)
    assert s.fsync is True and s.counter == "none" and s.min_free_bytes == 0
