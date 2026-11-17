"""``pgw gateway keys`` commands (the group is mounted by ``pgw.cli``; see INTERFACE_NOTES)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from pgw.gateway import ApiKeyStore
from pgw.gateway.cli import gateway_group


def test_keys_create_list_revoke(tmp_path: Path) -> None:
    kf = tmp_path / "keys.json"
    runner = CliRunner()
    r = runner.invoke(gateway_group, ["keys", "create", "--name", "billing", "--model", "tiny", "--rpm", "30",
                                      "--keys-file", str(kf)])
    assert r.exit_code == 0, r.output
    created = json.loads(r.output)
    assert created["key"].startswith(f"pgw-{created['key_id']}-")
    assert ApiKeyStore(kf).verify(created["key"]).requests_per_minute == 30
    r = runner.invoke(gateway_group, ["keys", "list", "--keys-file", str(kf)])
    listing = json.loads(r.output)
    assert listing[0]["key_id"] == created["key_id"] and created["key"] not in r.output
    r = runner.invoke(gateway_group, ["keys", "revoke", created["key_id"], "--keys-file", str(kf)])
    assert r.exit_code == 0
    r = runner.invoke(gateway_group, ["keys", "revoke", "0123456789ab", "--keys-file", str(kf)])
    assert r.exit_code == 1 and "unknown API key" in r.output


def test_keys_file_from_env_or_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    monkeypatch.delenv("PGW_GATEWAY_API_KEYS_FILE", raising=False)
    r = runner.invoke(gateway_group, ["keys", "list"])
    assert r.exit_code == 1 and "PGW_GATEWAY_API_KEYS_FILE" in r.output
    monkeypatch.setenv("PGW_GATEWAY_API_KEYS_FILE", str(tmp_path / "k.json"))
    r = runner.invoke(gateway_group, ["keys", "create", "--name", "x", "--scope", "admin"])
    assert r.exit_code == 0 and (tmp_path / "k.json").exists()


def test_serve_refuses_invalid_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGW_GATEWAY_MODELS", json.dumps([{"name": "m", "pack_path": str(tmp_path), "mode": "plain"}]))
    r = CliRunner().invoke(gateway_group, ["serve"])
    assert r.exit_code == 1 and "development" in r.output
