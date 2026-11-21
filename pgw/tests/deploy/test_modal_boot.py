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


def test_spec_options_are_valid_strict_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``launch.py --spec-rows`` presets: strict fixed rows (never the variable-row mode), prompt lookup, and the
    draft pack only when the image has one; accepted by the wizard's preset check and by GatewayModelConfig."""
    from pgw.gateway.config import GatewayModelConfig, GatewaySettings

    boot = _boot()
    assert boot.spec_options(0) == {}
    with pytest.raises(ValueError):
        boot.spec_options(1)
    with pytest.raises(ValueError):
        boot.spec_options(9)
    for opts in (boot.spec_options(3), boot.spec_options(4, boot.DRAFT_PACK, "cuda")):
        assert opts["spec_rows"] in (3, 4) and opts["spec_strict_passes"] is True and opts["spec_lookup"] > 0
        merged = {**boot.GATEWAY_MODEL_OPTIONS, **opts}
        env = boot.gateway_env(tmp_path / "config", tmp_path / "data", 8080, merged)
        for k, v in env.items():
            if k.startswith("PGW_GATEWAY_"):
                monkeypatch.setenv(k, v)
        gs = GatewaySettings()
        assert gs.setup_model_options == json.loads(json.dumps(merged))
        cfg = GatewayModelConfig(name="m", pack_path=tmp_path, fingerprint="ab" * 32,
                                 service_url="https://compute.example.net:8443", **gs.setup_model_options)
        assert cfg.spec_rows == opts["spec_rows"] and cfg.spec_strict_passes
    assert "draft_pack_path" not in boot.spec_options(3)
    assert boot.spec_options(4, boot.DRAFT_PACK, "cuda")["draft_device"] == "cuda"


def test_provider_model_table_matches_boot() -> None:
    """provider.py keeps its own copy of the target table (no module-level import of this file in its
    container); both parties must pack the same pinned revision."""
    import ast

    boot = _boot()
    tree = ast.parse((MODAL_DIR / "provider.py").read_text())
    table = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "MODELS" for t in n.targets))
    assert table == boot.MODELS
    assert all(len(rev) == 40 for _, rev in boot.MODELS.values())
    assert boot.MODELS[boot.MODEL_NAME][0] == boot.MODEL_REPO
