"""The chat's default schedule: ``auto`` unless the user chose one or an option the fused schedule refuses."""

from __future__ import annotations

import pytest

from pgw.chat.cli import chat_schedule
from pgw.sdk.options import SDKOptions


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PGW_SCHEDULE", raising=False)
    monkeypatch.delenv("PGW_FACTORY_WEIGHTS", raising=False)


def test_default_is_auto_and_valid() -> None:
    assert chat_schedule(None, "auto", None) == "auto"
    SDKOptions(schedule="auto").validate()  # the chat's other defaults combine with auto


@pytest.mark.parametrize("explicit", ["standard", "fused", "auto"])
def test_explicit_choice_wins(explicit: str) -> None:
    assert chat_schedule(explicit, "on", "stream") == explicit


def test_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_SCHEDULE", "standard")
    assert chat_schedule(None, "auto", None) is None  # left to the SDK, which reads PGW_SCHEDULE


@pytest.mark.parametrize("prefix_cache, weights, env", [("on", None, None), ("auto", "stream", None),
                                                         ("auto", None, "stream")])
def test_incompatible_options_fall_back_to_standard(monkeypatch: pytest.MonkeyPatch, prefix_cache: str,
                                                    weights: str | None, env: str | None) -> None:
    if env:
        monkeypatch.setenv("PGW_FACTORY_WEIGHTS", env)
    assert chat_schedule(None, prefix_cache, weights) == "standard"
