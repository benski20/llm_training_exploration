"""``pgw config``: inspect effective settings and render config templates with secrets.

``expand_placeholders`` turns a template (for example a gateway ``models.yaml`` kept in a Kubernetes
ConfigMap) into the runtime file by substituting secret references, so secrets live only in
Secret objects / files and never in the template, the image or the command line:

* ``${env:NAME}``  value of environment variable ``NAME`` (error if unset)
* ``${file:PATH}`` contents of ``PATH`` with one trailing newline removed (error if unreadable)
* ``$${``          a literal ``${``

The output is written atomically with mode 0600.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

from pgw.core.errors import ConfigError

__all__ = ["ROLE_SETTINGS", "expand_placeholders", "render_template_file", "settings_dump"]

_PLACEHOLDER = re.compile(r"\$\$\{|\$\{(env|file):([^}]*)\}")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: role -> (settings module, class name)
ROLE_SETTINGS: dict[str, tuple[str, str]] = {
    "common": ("pgw.core.settings", "CommonSettings"),
    "service": ("pgw.service.config", "ServiceSettings"),
    "gateway": ("pgw.gateway.config", "GatewaySettings"),
    "factory": ("pgw.factory.config", "FactorySettings"),
    "maskstore": ("pgw.maskstore.config", "MaskStoreSettings"),
}


def expand_placeholders(text: str, *, env: dict[str, str] | None = None, base_dir: Path | None = None) -> str:
    """Substitute ``${env:..}`` / ``${file:..}`` references in ``text`` (see module docstring).

    Raises:
        ConfigError: unknown variable, unreadable file or malformed placeholder. The message names
            the reference, never its value.
    """
    environ = os.environ if env is None else env

    def repl(m: re.Match[str]) -> str:
        if m.group(0) == "$${":
            return "${"
        kind, ref = m.group(1), m.group(2).strip()
        if kind == "env":
            if not _ENV_NAME.match(ref):
                raise ConfigError(f"bad environment variable name in placeholder: {ref!r}")
            if ref not in environ:
                raise ConfigError(f"environment variable {ref} referenced by the template is not set")
            return environ[ref]
        p = Path(ref)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        try:
            data = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read file {p} referenced by the template: {exc.strerror}") from exc
        return data[:-1] if data.endswith("\n") else data

    # Validate the template before substituting, so errors never depend on (or echo) secret values.
    leftover = re.search(r"\$\{[^}\n]*\}?", _PLACEHOLDER.sub("", text))
    if leftover:
        raise ConfigError(f"unsupported placeholder {leftover.group(0)!r}: unknown kind or unterminated "
                          "(use ${env:NAME}, ${file:PATH} or $${ for a literal ${)")
    return _PLACEHOLDER.sub(repl, text)


def render_template_file(src: Path, dest: Path, *, mode: int = 0o600, env: dict[str, str] | None = None) -> int:
    """Expand ``src`` into ``dest`` atomically (temp file + rename) with ``mode``; return bytes written."""
    try:
        text = src.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read template {src}: {exc.strerror}") from exc
    out = expand_placeholders(text, env=env, base_dir=src.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(out)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return len(out.encode("utf-8"))


def settings_dump(role: str) -> dict[str, Any]:
    """Effective ``PGW_*`` settings of ``role`` as JSON-able data; secrets are masked."""
    import importlib

    if role not in ROLE_SETTINGS:
        raise ConfigError(f"unknown role {role!r}; expected one of {', '.join(ROLE_SETTINGS)}")
    module, cls_name = ROLE_SETTINGS[role]
    try:
        cls = getattr(importlib.import_module(module), cls_name)
    except ImportError as exc:
        raise ConfigError(f"settings for {role!r} need the '{role}' extra: {exc}") from exc
    data: dict[str, Any] = cls().model_dump(mode="json")  # SecretStr dumps as '**********'
    return data
