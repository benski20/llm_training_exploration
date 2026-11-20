"""Modal demo bootstrap (``deploy/modal/pgw_modal_boot.py``, no Modal imports): the customer gateway's speed
settings reach the setup wizard as valid ``setup_model_options``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODAL_DIR = Path(__file__).resolve().parents[2] / "deploy" / "modal"


def _boot():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("pgw_modal_boot_t", MODAL_DIR / "pgw_modal_boot.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_customer_gateway_env_presets_speed_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.gateway.config import GatewayModelConfig, GatewaySettings

    boot = _boot()
    env = boot.gateway_env(tmp_path / "config", tmp_path / "data", 8080)
    opts = json.loads(env["PGW_GATEWAY_SETUP_MODEL_OPTIONS"])
    assert opts == {"local_linears": "auto", "float_threads": 1}
    assert env["PGW_FLOAT_THREADS"] == "1"
    assert "lane" not in opts and "spec_rows" not in opts  # remote provider; no draft model in the demo
    for k, v in env.items():
        if k.startswith("PGW_GATEWAY_"):
            monkeypatch.setenv(k, v)
    gs = GatewaySettings()
    assert gs.setup_model_options == opts
    GatewayModelConfig(name="m", pack_path=tmp_path, fingerprint="ab" * 32,
                       service_url="https://compute.example.net:8443", **gs.setup_model_options)
