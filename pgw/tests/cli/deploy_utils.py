"""Helpers shared by the deployment-artifact tests: map ``PGW_*`` env vars onto settings classes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

#: Env vars consumed by something other than a settings class.
NON_SETTINGS_ENV = {
    "PGW_RENDER",             # pgw-entrypoint: config templates to render
    "PGW_MASKSTORE_KEK",      # EnvKeyProvider (MaskStoreSettings.key_env_var default)
    "PGW_HEALTHCHECK_CERT_FILE",
    "PGW_HEALTHCHECK_KEY_FILE",
    "PGW_OFFLINE",
}


def settings_classes() -> list[tuple[str, type[BaseModel]]]:
    """``(env prefix, class)``, longest prefix first."""
    from pgw.core.settings import CommonSettings
    from pgw.factory.config import FactorySettings
    from pgw.gateway.config import GatewaySettings
    from pgw.maskstore.config import MaskStoreSettings
    from pgw.service.config import ServiceSettings

    return [
        ("PGW_MASKSTORE_", MaskStoreSettings),
        ("PGW_GATEWAY_", GatewaySettings),
        ("PGW_SERVICE_", ServiceSettings),
        ("PGW_FACTORY_", FactorySettings),
        ("PGW_", CommonSettings),
    ]


def _field_exists(cls: type[BaseModel], path: list[str]) -> bool:
    fields = cls.model_fields
    head = path[0]
    if head not in fields:
        return False
    if len(path) == 1:
        return True
    ann: Any = fields[head].annotation
    for cand in (ann, *getattr(ann, "__args__", ())):
        if isinstance(cand, type) and issubclass(cand, BaseModel):
            return _field_exists(cand, path[1:])
    return False


def unknown_env_names(names: list[str]) -> list[str]:
    """``PGW_*`` names that no settings class (or known consumer) reads; these would be ignored silently."""
    bad = []
    for name in names:
        if not name.startswith("PGW_") or name in NON_SETTINGS_ENV:
            continue
        for prefix, cls in settings_classes():
            if name.startswith(prefix):
                path = name[len(prefix):].lower().split("__")
                if not _field_exists(cls, path):
                    bad.append(name)
                break
    return bad


def construct_settings(env: dict[str, str], monkeypatch: Any) -> dict[str, BaseModel]:
    """Instantiate every settings class whose prefix occurs in ``env`` with exactly that environment."""
    import os

    for k in list(os.environ):
        if k.startswith("PGW_") and k != "PGW_OFFLINE":
            monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out: dict[str, BaseModel] = {}
    for prefix, cls in settings_classes():
        if prefix == "PGW_" or any(k.startswith(prefix) for k in env):
            out[prefix] = cls()
    return out
