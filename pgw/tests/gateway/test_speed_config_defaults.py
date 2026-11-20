"""Chat-speed settings (WP-A0 scaffold): every default is today's behaviour, and a mode that has not
been built yet cannot be switched on (no configuration silently asks for e.g. strict-pass speculation
that does not happen)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import pgw.gateway.config as gw_config
import pgw.service.config as svc_config
from pgw.gateway.config import GatewayModelConfig, load_models_file
from pgw.service.config import ServiceSettings

PACK = Path("/nonexistent/pack")

#: Field -> today's behaviour. Changing a default here is a behaviour change of every deployment.
GATEWAY_DEFAULTS: dict[str, object] = {
    "lane": "off",
    "local_linears": "off",
    "spec_rows": None,
    "spec_strict_passes": False,
    "float_ops": "torch",
    "head_mode": "gateway",
    "schedule": "standard",
    "prefix_cache": "off",
}
#: A non-default value per field (refused while pending).
GATEWAY_ON: dict[str, object] = {
    "lane": "auto",
    "local_linears": "attention",
    "spec_rows": 4,
    "spec_strict_passes": True,
    "float_ops": "native",
    "head_mode": "provider_int8",
    "schedule": "fused2",
    "prefix_cache": "conversation",
}
SERVICE_DEFAULTS: dict[str, object] = {"lane": "off", "lane_socket": None, "lane_port": None, "lane_max_rows": 8}


def masked(**kw: object) -> GatewayModelConfig:
    return GatewayModelConfig(name="m", pack_path=PACK, service_url="https://compute.example.net:8443",  # type: ignore[arg-type]
                              fingerprint="ab" * 32, **kw)


def test_gateway_speed_defaults_are_todays_behaviour() -> None:
    fields = GatewayModelConfig.model_fields
    for name, default in GATEWAY_DEFAULTS.items():
        assert fields[name].default == default, name
    m = masked()
    for name, default in GATEWAY_DEFAULTS.items():
        assert getattr(m, name) == default
    # the existing speculation knobs keep their values (today's adaptive, variable-row speculation)
    assert fields["spec_tokens"].default == 2 and fields["spec_min_tokens"].default == 2
    assert fields["spec_lookup"].default == 0 and fields["draft_pack_path"].default is None


def test_every_new_gateway_field_is_gated_until_built() -> None:
    assert set(GATEWAY_ON) == set(GATEWAY_DEFAULTS)
    assert set(gw_config.PENDING_SPEED_SETTINGS) <= set(GATEWAY_DEFAULTS)
    for name in gw_config.PENDING_SPEED_SETTINGS:
        kw: dict[str, object] = {name: GATEWAY_ON[name]}
        with pytest.raises(ValidationError, match="not available in this build"):
            masked(**kw)
    # the default spelled out explicitly is fine
    masked(**GATEWAY_DEFAULTS)


def test_gateway_speed_fields_validate_once_built(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gw_config, "PENDING_SPEED_SETTINGS", {})
    m = masked(**GATEWAY_ON, spec_lookup=3)
    assert m.spec_rows == 4 and m.lane == "auto"
    assert masked(spec_rows=4, spec_strict_passes=True, spec_lookup=3).spec_strict_passes
    with pytest.raises(ValidationError, match="spec_strict_passes needs spec_rows"):
        masked(spec_strict_passes=True)
    with pytest.raises(ValidationError, match="spec_rows needs spec_strict_passes"):
        masked(spec_rows=4, spec_lookup=3)
    for bad in (1, 9):
        with pytest.raises(ValidationError):
            masked(spec_rows=bad, spec_strict_passes=True, spec_lookup=3)
    with pytest.raises(ValidationError):
        masked(lane="sometimes")


def test_models_file_without_the_new_fields_still_loads(tmp_path: Path) -> None:
    f = tmp_path / "models.json"
    f.write_text('[{"name": "m", "pack_path": "/nonexistent/pack", "fingerprint": "' + "ab" * 32
                 + '", "service_url": "https://compute.example.net:8443"}]')
    (m,) = load_models_file(f)
    assert {k: getattr(m, k) for k in GATEWAY_DEFAULTS} == GATEWAY_DEFAULTS


def test_service_lane_defaults_off_and_is_gated() -> None:
    fields = ServiceSettings.model_fields
    for name, default in SERVICE_DEFAULTS.items():
        assert fields[name].default == default, name
    s = ServiceSettings()
    assert s.lane == "off"
    assert set(svc_config.PENDING_SPEED_SETTINGS) <= set(SERVICE_DEFAULTS)
    if "lane" in svc_config.PENDING_SPEED_SETTINGS:
        with pytest.raises(ValidationError, match="not available in this build"):
            ServiceSettings(lane="uds", lane_socket=Path("/run/pgw/lane.sock"))
    else:  # WP-A1 landed: the UDS lane is accepted (tests/service/test_lane_server.py)
        assert ServiceSettings(lane="uds", lane_socket=Path("/run/pgw/lane.sock")).lane == "uds"


def test_service_lane_validation_once_built(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(svc_config, "PENDING_SPEED_SETTINGS", {})
    assert ServiceSettings(lane="uds", lane_socket=tmp_path / "lane.sock").lane == "uds"
    with pytest.raises(ValidationError, match="needs lane_socket"):
        ServiceSettings(lane="uds")
    with pytest.raises(ValidationError, match="needs lane_port"):
        ServiceSettings(lane="tcp")
    with pytest.raises(ValidationError, match="never plain TCP"):
        ServiceSettings(lane="tcp", lane_port=9444)
    with pytest.raises(ValidationError):
        ServiceSettings(lane_max_rows=9)
