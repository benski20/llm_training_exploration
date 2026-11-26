"""Measurement-only preview of pinned, unsupported models (``PGW_SDK_PREVIEW_MODELS``): off by default, only for
models with a baked pin, and never a way around the pin."""

from __future__ import annotations

import pytest

from pgw.sdk.errors import UnsupportedModelError
from pgw.sdk.options import PREVIEW_ENV, SUPPORTED_MODELS, SDKOptions, canonical_model, preview_models
from pgw.sdk.provisioning import PINNED_MODELS

PREVIEW = "qwen2.5-32b-instruct"


def test_preview_candidates_are_pinned_and_unsupported() -> None:
    assert PREVIEW in PINNED_MODELS and PINNED_MODELS[PREVIEW].fingerprint
    assert PREVIEW not in SUPPORTED_MODELS


def test_default_refuses_a_pinned_unsupported_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PREVIEW_ENV, raising=False)
    assert preview_models() == ()
    with pytest.raises(UnsupportedModelError, match="not supported"):
        canonical_model(PREVIEW)
    monkeypatch.setenv(PREVIEW_ENV, "")
    with pytest.raises(UnsupportedModelError):
        canonical_model("Qwen/Qwen2.5-32B-Instruct")


def test_opt_in_accepts_a_pinned_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREVIEW_ENV, " Qwen/Qwen2.5-32B-Instruct , qwen2.5-7b-instruct")
    assert preview_models() == (PREVIEW,)  # a supported name in the list is not a preview
    assert canonical_model("Qwen/Qwen2.5-32B-Instruct") == PREVIEW
    assert canonical_model("qwen2.5-7b-instruct") == "qwen2.5-7b-instruct"
    with pytest.raises(UnsupportedModelError):  # only the listed ones
        canonical_model("qwen2.5-72b-instruct")
    SDKOptions().validate()


def test_opt_in_refuses_an_unpinned_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREVIEW_ENV, "qwen2.5-110b-instruct")
    with pytest.raises(UnsupportedModelError, match="no baked pin"):
        canonical_model("qwen2.5-110b-instruct")
    with pytest.raises(UnsupportedModelError, match="no baked pin"):
        preview_models()
    assert canonical_model("qwen2.5-7b-instruct") == "qwen2.5-7b-instruct"  # supported models never need it


def test_a_pin_without_a_fingerprint_is_not_previewable(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.sdk import provisioning

    monkeypatch.setitem(provisioning.PINNED_MODELS, "qwen2.5-99b-instruct",
                        provisioning.ModelPin("Qwen/Qwen2.5-99B-Instruct", "0" * 40, None))
    with pytest.raises(UnsupportedModelError, match="no baked pin"):
        preview_models("qwen2.5-99b-instruct")


def test_a_preview_model_keeps_its_baked_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PREVIEW_ENV, PREVIEW)
    with pytest.raises(ValueError, match="baked pin only"):
        SDKOptions(model_fingerprints={PREVIEW: "ab" * 32}).validate()
    SDKOptions(model_fingerprints={"my-custom-pack": "ab" * 32}).validate()
    monkeypatch.delenv(PREVIEW_ENV)
    SDKOptions(model_fingerprints={PREVIEW: "ab" * 32}).validate()  # off: the model is refused at use anyway


def test_the_provisioner_checks_a_preview_model_against_its_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The preview changes which names the library accepts, not what it checks: the provisioner's pins are the
    release's, so a provider serving another pack under the preview name is refused before any download."""
    from pgw.sdk.provisioning import ProviderProvisioner

    monkeypatch.setenv(PREVIEW_ENV, PREVIEW)
    prov = ProviderProvisioner.from_options(SDKOptions())
    assert prov.expected_fingerprint(PREVIEW) == PINNED_MODELS[PREVIEW].fingerprint
