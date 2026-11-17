"""``pgw doctor`` checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from pgw.cli import cli
from pgw.cli.doctor import Check, DoctorReport, format_report, run_doctor


def test_basic_report_is_healthy(clean_env: pytest.MonkeyPatch) -> None:
    rep = run_doctor()
    names = [c.name for c in rep.checks]
    assert names[:5] == ["python", "packages", "extras", "entropy", "selftest"]
    assert rep.ok, format_report(rep)
    assert rep.get("selftest") is not None and rep.get("selftest").status == "ok"  # type: ignore[union-attr]
    d = rep.to_dict()
    assert d["ok"] and d["summary"]["fail"] == 0
    json.dumps(d)


def test_report_semantics() -> None:
    rep = DoctorReport(checks=[Check("a", "ok"), Check("b", "warn", "meh")])
    assert rep.ok
    rep.strict = True
    assert not rep.ok
    rep = DoctorReport(checks=[Check("a", "fail", "boom")])
    assert not rep.ok
    assert "PROBLEMS FOUND" in format_report(rep) and "[FAIL] a" in format_report(rep)


def test_unknown_role() -> None:
    with pytest.raises(ValueError, match="role"):
        run_doctor(role="nope")


def test_check_exceptions_become_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import pgw.cli.doctor as d

    def boom() -> tuple[str, str, dict[str, object]]:
        raise RuntimeError("kaputt")

    monkeypatch.setattr(d, "_check_entropy", boom)
    rep = run_doctor()
    c = rep.get("entropy")
    assert c is not None and c.status == "fail" and "kaputt" in c.detail
    assert not rep.ok


def test_selftest_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from pgw.core import selftest

    res = selftest.SelfTestResult(ok=False, checks={"K40A12/cpu_f64/exact": False})
    monkeypatch.setattr(selftest, "run_selftest", lambda **kw: res)
    rep = run_doctor()
    c = rep.get("selftest")
    assert c is not None and c.status == "fail" and "K40A12/cpu_f64/exact" in c.detail


def test_cli_exit_codes(runner: CliRunner, clean_env: pytest.MonkeyPatch, monkeypatch: pytest.MonkeyPatch) -> None:
    r = runner.invoke(cli, ["doctor", "--json", "--no-gpu"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["ok"] is True
    import pgw.cli.doctor as d

    monkeypatch.setattr(d, "_check_memory", lambda: ("warn", "low", {}))
    assert runner.invoke(cli, ["doctor"]).exit_code == 0
    r = runner.invoke(cli, ["doctor", "--strict"])
    assert r.exit_code == 1 and "[WARN] memory" in r.output


def test_key_file_permissions(tmp_path: Path) -> None:
    kf = tmp_path / "kek"
    kf.write_bytes(os.urandom(32))
    kf.chmod(0o644)
    rep = run_doctor(key_file=kf)
    c = rep.get("maskstore.key_file")
    assert c is not None and c.status == "fail" and "0600" in c.detail
    kf.chmod(0o600)
    c = run_doctor(key_file=kf).get("maskstore.key_file")
    assert c is not None and c.status == "ok"


def test_pack_check(tiny_pack_dir: Path, runner: CliRunner) -> None:
    rep = run_doctor(pack=tiny_pack_dir, ring="K40A12")
    c = rep.get("pack")
    assert c is not None and c.status == "ok", c
    assert c.data["supported_rings"] == ["K32A8", "K40A12", "K48A20"]
    assert c.data["mask_bytes_per_slot"]["K40A12"] > 0
    r = runner.invoke(cli, ["doctor", "--pack", str(tiny_pack_dir), "--ring", "k40a12", "--json"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli, ["doctor", "--pack", str(tiny_pack_dir), "--ring", "bogus"])
    assert r.exit_code == 1 and "error [config]" in r.output


def test_pack_check_detects_tampering(tiny_pack_dir: Path, tmp_path: Path) -> None:
    import shutil

    bad = tmp_path / "pack"
    shutil.copytree(tiny_pack_dir, bad)
    shard = next(bad.glob("linears-*.safetensors"))
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.write_bytes(bytes(data))
    c = run_doctor(pack=bad).get("pack")
    assert c is not None and c.status == "fail"


def test_role_service_settings(clean_env: pytest.MonkeyPatch, tmp_path: Path, tiny_pack_dir: Path) -> None:
    c = run_doctor(role="service").get("settings.service")
    assert c is not None and c.status == "fail" and "credentials" in c.detail
    clean_env.setenv("PGW_SERVICE_ALLOW_UNAUTHENTICATED", "true")
    c = run_doctor(role="service").get("settings.service")
    assert c is not None and c.status == "warn" and "no packs" in c.detail and "dev only" in c.detail
    clean_env.delenv("PGW_SERVICE_ALLOW_UNAUTHENTICATED")
    clean_env.setenv("PGW_SERVICE_PACKS", json.dumps([str(tiny_pack_dir)]))
    tf = tmp_path / "tokens"
    tf.write_text("t:" + "0" * 64 + "\n")
    clean_env.setenv("PGW_SERVICE_TOKENS_FILE", str(tf))
    c = run_doctor(role="service").get("settings.service")
    assert c is not None and c.status == "warn" and "TLS disabled" in c.detail
    clean_env.setenv("PGW_SERVICE_PACKS", json.dumps([str(tmp_path / "missing")]))
    c = run_doctor(role="service").get("settings.service")
    assert c is not None and c.status == "fail" and "not found" in c.detail
    clean_env.setenv("PGW_SERVICE_PORT", "not-a-port")
    c = run_doctor(role="service").get("settings.service")
    assert c is not None and c.status == "fail" and "port" in c.detail and "not-a-port" not in c.detail


def test_role_gateway_settings(clean_env: pytest.MonkeyPatch, tmp_path: Path, tiny_pack_dir: Path) -> None:
    models = [{"name": "m", "pack_path": str(tiny_pack_dir), "service_url": "http://compute.example:8443",
               "fingerprint": "ab" * 32}]
    clean_env.setenv("PGW_GATEWAY_MODELS", json.dumps(models))
    c = run_doctor(role="gateway").get("settings.gateway")
    assert c is not None and c.status == "fail" and "https" in c.detail  # policy: TLS to remote services
    models[0]["service_url"] = "https://compute.example:8443"
    clean_env.setenv("PGW_GATEWAY_MODELS", json.dumps(models))
    c = run_doctor(role="gateway").get("settings.gateway")
    assert c is not None and c.status == "warn" and "admin token" in c.detail
    assert c.data["models"] == ["m"]
    clean_env.setenv("PGW_GATEWAY_ADMIN_TOKEN", "x" * 32)
    clean_env.setenv("PGW_GATEWAY_API_KEYS_FILE", str(tmp_path / "keys.json"))
    c = run_doctor(role="gateway").get("settings.gateway")
    assert c is not None and c.status == "ok", c


def test_role_factory_settings(clean_env: pytest.MonkeyPatch, tiny_pack_dir: Path, tmp_path: Path) -> None:
    c = run_doctor(role="factory").get("settings.factory")
    assert c is not None and c.status == "fail"
    clean_env.setenv("PGW_FACTORY_PACK_PATH", str(tiny_pack_dir))
    c = run_doctor(role="factory").get("settings.factory")
    assert c is not None and c.status == "ok", c
    clean_env.setenv("PGW_FACTORY_OUTPUT", "export")
    c = run_doctor(role="factory").get("settings.factory")
    assert c is not None and c.status == "fail" and "export_dir" in c.detail and "input" not in c.detail


def test_role_extras_required(monkeypatch: pytest.MonkeyPatch) -> None:
    import pgw.cli.doctor as d

    real = d._importable
    monkeypatch.setattr(d, "_importable", lambda m: False if m == "hypercorn" else real(m))
    c = run_doctor(role="all").get("extras")
    assert c is not None and c.status == "warn" and "pgw[service]" in c.detail
    rep = run_doctor(role="pack")
    assert rep.get("extras").status == "ok"  # type: ignore[union-attr]
    monkeypatch.setenv("PGW_SERVICE_ALLOW_UNAUTHENTICATED", "true")
    c = run_doctor(role="service").get("extras")
    assert c is not None and c.status == "fail"


def test_maskstore_without_counter_fails_and_dev_flags_warn(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    """Security review: no external counter = gateways refuse to serve, so doctor reports 'fail'."""
    from pgw.maskstore.admin import init_store

    kf = tmp_path / "kek"
    init_store(tmp_path / "store", kf)
    clean_env.delenv("PGW_MASKSTORE_ALLOW_NO_EXTERNAL_COUNTER", raising=False)
    c = run_doctor(maskstore_root=tmp_path / "store", key_file=kf).get("maskstore")
    assert c is not None and c.status == "fail" and "counter" in c.detail
    clean_env.setenv("PGW_MASKSTORE_COUNTER", "file")
    clean_env.setenv("PGW_MASKSTORE_COUNTER_FILE", str(tmp_path / "other-volume" / "counter"))
    c = run_doctor(maskstore_root=tmp_path / "store", key_file=kf).get("maskstore")
    assert c is not None and c.status == "ok", c
    clean_env.setenv("PGW_MASKSTORE_ALLOW_UNSIGNED_PARTNER_SEGMENTS", "true")
    rep = run_doctor(maskstore_root=tmp_path / "store", key_file=kf)
    c = rep.get("maskstore")
    assert c is not None and c.status == "warn" and "allow_unsigned_partner_segments" in c.detail
    rep.strict = True
    assert not rep.ok
