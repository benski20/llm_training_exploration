"""Unit tests for the ``pgw`` command tree: wiring, exit codes, config tools, tokens, bench."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from pgw.cli import cli
from pgw.cli._common import EXIT_ERROR, EXIT_INTERRUPTED, EXIT_NOT_IMPLEMENTED, LazyGroup, handle_errors
from pgw.core.errors import ConfigError

TOP_LEVEL = {"version", "doctor", "pack", "inspect", "factory", "maskstore", "service", "gateway", "bench", "config", "sdk",
             "chat"}


def test_help_lists_every_command(runner: CliRunner) -> None:
    r = runner.invoke(cli, ["--help"])
    assert r.exit_code == 0, r.output
    for name in TOP_LEVEL:
        assert name in r.output
    assert set(cli.list_commands(click.Context(cli))) == TOP_LEVEL


@pytest.mark.parametrize("path", [
    ["factory", "--help"], ["factory", "generate", "--help"], ["factory", "run", "--help"],
    ["factory", "status", "--help"], ["factory", "benchmark", "--help"],
    ["chat", "--help"],
    ["gateway", "--help"], ["gateway", "serve", "--help"], ["gateway", "keys", "create", "--help"],
    ["maskstore", "init", "--help"], ["maskstore", "status", "--help"], ["maskstore", "import", "--help"],
    ["maskstore", "gc", "--help"], ["maskstore", "rotate-key", "--help"],
    ["service", "serve", "--help"], ["service", "token", "create", "--help"],
    ["bench", "ring", "--help"], ["sdk", "status", "--help"], ["sdk", "doctor", "--help"], ["config", "show", "--help"], ["config", "render", "--help"],
    ["doctor", "--help"], ["pack", "--help"], ["inspect", "--help"],
])
def test_subcommand_help(runner: CliRunner, path: list[str]) -> None:
    r = runner.invoke(cli, path)
    assert r.exit_code == 0, r.output
    assert "Usage:" in r.output


def test_version(runner: CliRunner) -> None:
    r = runner.invoke(cli, ["version"])
    assert r.exit_code == 0 and "protocol v1" in r.output
    r = runner.invoke(cli, ["version", "--json"])
    assert json.loads(r.stdout)["protocol"] == 1
    r = runner.invoke(cli, ["-V"])
    assert r.exit_code == 0 and "pgw" in r.output


def test_log_level_validated(runner: CliRunner) -> None:
    r = runner.invoke(cli, ["--log-level", "LOUD", "version"])
    assert r.exit_code == 2


# ---- error handling ------------------------------------------------------------------------------
def _cmd(exc: BaseException) -> click.Command:
    @click.command()
    @handle_errors
    def c() -> None:
        raise exc

    return c


@pytest.mark.parametrize(("exc", "code", "text"), [
    (ConfigError("bad thing"), EXIT_ERROR, "error [config]: bad thing"),
    (NotImplementedError("later"), EXIT_NOT_IMPLEMENTED, "not implemented yet: later"),
    (KeyboardInterrupt(), EXIT_INTERRUPTED, "interrupted"),
])
def test_handle_errors_exit_codes(exc: BaseException, code: int, text: str) -> None:
    r = CliRunner().invoke(_cmd(exc), [])
    assert r.exit_code == code
    assert text in r.output


def test_handle_errors_validation_error_hides_values(clean_env: pytest.MonkeyPatch) -> None:
    from pgw.service.config import ServiceSettings

    @click.command()
    @handle_errors
    def c() -> None:
        ServiceSettings(port=-5)

    r = CliRunner().invoke(c, [])
    assert r.exit_code == EXIT_ERROR
    assert "error [config]" in r.output and "port" in r.output
    assert "-5" not in r.output  # input values are not echoed


def test_unexpected_errors_propagate() -> None:
    r = CliRunner().invoke(_cmd(RuntimeError("bug")), [])
    assert isinstance(r.exception, RuntimeError)


def test_lazy_group_missing_extra_is_reported() -> None:
    grp = LazyGroup("x", lazy_subcommands={"ghost": ("pgw_does_not_exist.cli", "group", "ghostly")})
    assert "ghost" in grp.list_commands(click.Context(grp))
    r = CliRunner().invoke(grp, ["ghost", "anything", "--flag"])
    assert r.exit_code == EXIT_ERROR
    assert "pip install 'pgw-sdk[ghostly]'" in r.output


def test_lazy_groups_are_the_component_groups() -> None:
    from pgw.factory.cli import factory_group
    from pgw.gateway.cli import gateway_group

    ctx = click.Context(cli)
    assert cli.get_command(ctx, "factory") is factory_group
    assert cli.get_command(ctx, "gateway") is gateway_group


# ---- config tools --------------------------------------------------------------------------------
def test_expand_placeholders(tmp_path: Path) -> None:
    from pgw.cli.configtools import expand_placeholders

    (tmp_path / "tok").write_text("s3cret\n")
    out = expand_placeholders('a: "${env:FOO}"\nb: "${file:tok}"\nc: "$${env:LITERAL}"\n',
                              env={"FOO": "bar"}, base_dir=tmp_path)
    assert out == 'a: "bar"\nb: "s3cret"\nc: "${env:LITERAL}"\n'


@pytest.mark.parametrize(("text", "msg"), [
    ("${env:MISSING_VAR}", "MISSING_VAR"),
    ("${env:bad name}", "bad environment variable name"),
    ("${file:/nonexistent/x}", "cannot read file"),
    ("${vault:x}", "unsupported placeholder"),
])
def test_expand_placeholders_errors(text: str, msg: str) -> None:
    from pgw.cli.configtools import expand_placeholders

    with pytest.raises(ConfigError, match=msg):
        expand_placeholders(text, env={})


def test_config_render_cli_writes_0600(runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmpl = tmp_path / "models.tmpl.yaml"
    tmpl.write_text("- name: m\n  service_token: '${env:PGW_TEST_TOKEN}'\n")
    monkeypatch.setenv("PGW_TEST_TOKEN", "tok-123")
    out = tmp_path / "run" / "models.yaml"
    r = runner.invoke(cli, ["config", "render", str(tmpl), str(out)])
    assert r.exit_code == 0, r.output
    assert "tok-123" in out.read_text()
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert "tok-123" not in r.output
    monkeypatch.delenv("PGW_TEST_TOKEN")
    r = runner.invoke(cli, ["config", "render", str(tmpl), str(out)])
    assert r.exit_code == EXIT_ERROR and "PGW_TEST_TOKEN" in r.output


def test_config_show_masks_secrets(runner: CliRunner, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("PGW_GATEWAY_ADMIN_TOKEN", "very-secret-admin")
    clean_env.setenv("PGW_GATEWAY_PORT", "9090")
    r = runner.invoke(cli, ["config", "show", "gateway"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data["port"] == 9090
    assert "very-secret-admin" not in r.output
    r = runner.invoke(cli, ["config", "show", "service"])
    assert json.loads(r.stdout)["port"] == 8443


def test_config_show_reports_bad_settings(runner: CliRunner, clean_env: pytest.MonkeyPatch) -> None:
    r = runner.invoke(cli, ["config", "show", "factory"])  # pack_path is required
    assert r.exit_code == EXIT_ERROR and "pack_path" in r.output
    clean_env.setenv("PGW_RING", "K41A3")
    r = runner.invoke(cli, ["config", "show", "common"])
    assert r.exit_code == EXIT_ERROR


# ---- service tokens -------------------------------------------------------------------------------
def test_service_token_create_compatible_with_authenticator(runner: CliRunner, tmp_path: Path) -> None:
    from pgw.service.auth import TenantAuthenticator, hash_token

    tf = tmp_path / "tokens"
    r = runner.invoke(cli, ["service", "token", "create", "--tenant", "acme", "--tokens-file", str(tf)])
    assert r.exit_code == 0, r.output
    token = json.loads(r.stdout)["token"]
    assert stat.S_IMODE(tf.stat().st_mode) == 0o600
    tenant, token_id, digest = tf.read_text().strip().split(":")
    assert (tenant, digest) == ("acme", hash_token(token)) and token_id == json.loads(r.stdout)["token_id"]
    out = tmp_path / "tok2"
    r = runner.invoke(cli, ["service", "token", "create", "--tenant", "beta", "--tokens-file", str(tf),
                            "--out", str(out)])
    assert r.exit_code == 0 and "pgwt-" not in r.output
    assert out.read_text().startswith("pgwt-")
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    auth = TenantAuthenticator(tf)
    assert len(tf.read_text().splitlines()) == 2
    assert auth is not None
    r = runner.invoke(cli, ["service", "token", "hash"], input=token + "\n")
    assert r.stdout.strip() == hash_token(token)


def test_service_token_create_validation(runner: CliRunner, tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    r = runner.invoke(cli, ["service", "token", "create", "--tenant", "bad tenant", "--tokens-file",
                            str(tmp_path / "t")])
    assert r.exit_code == EXIT_ERROR and "tenant id" in r.output
    r = runner.invoke(cli, ["service", "token", "create", "--tenant", "ok"])
    assert r.exit_code == EXIT_ERROR and "PGW_SERVICE_TOKENS_FILE" in r.output
    clean_env.setenv("PGW_SERVICE_TOKENS_FILE", str(tmp_path / "env_tokens"))
    r = runner.invoke(cli, ["service", "token", "create", "--tenant", "ok"])
    assert r.exit_code == 0 and (tmp_path / "env_tokens").exists()
    r = runner.invoke(cli, ["service", "token", "hash"], input="\n")
    assert r.exit_code == EXIT_ERROR


def test_service_serve_bad_settings_exit_1(runner: CliRunner, clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("PGW_SERVICE_MAX_BATCH_ROWS", "-3")
    r = runner.invoke(cli, ["service", "serve"])
    assert r.exit_code == EXIT_ERROR and "max_batch_rows" in r.output


# ---- bench ---------------------------------------------------------------------------------------
def test_bench_ring_all_backends_exact(runner: CliRunner) -> None:
    from pgw.cli.bench import available_backends

    r = runner.invoke(cli, ["bench", "ring", "--shape", "96x160", "--rows", "1", "--rows", "40",
                            "--ring", "K40A12", "--ring", "K48A20", "--repeat", "1", "--json"])
    assert r.exit_code == 0, r.output
    res = json.loads(r.stdout)
    assert len(res) == 2 * 2 * len(available_backends())
    assert all(x["exact"] and x["ms_per_call"] > 0 for x in res)
    assert {x["backend"] for x in res} == set(available_backends())


def test_bench_ring_table_and_errors(runner: CliRunner) -> None:
    r = runner.invoke(cli, ["bench", "ring", "--shape", "tiny", "--backend", "cpu_f64", "--rows", "2"])
    assert r.exit_code == 0 and "cpu_f64" in r.output and "yes" in r.output
    r = runner.invoke(cli, ["bench", "ring", "--shape", "huge"])
    assert r.exit_code == EXIT_ERROR and "--shape" in r.output
    r = runner.invoke(cli, ["bench", "ring", "--shape", "tiny", "--backend", "cuda_int8_nope"])
    assert r.exit_code == EXIT_ERROR and "not available" in r.output


def test_bench_ring_detects_headroom_violation() -> None:
    from pgw.cli.bench import bench_ring
    from pgw.core.errors import RingOverflowError

    with pytest.raises(RingOverflowError):
        bench_ring(ring="K32A8", backend="cpu_f64", m=8, n=1 << 20, rows=1)


def test_bench_factory_delegates(runner: CliRunner) -> None:
    r = runner.invoke(cli, ["bench", "factory", "--model", "qwen2.5-0.5b", "--slots", "4", "--chunk-slots", "4",
                            "--layers", "1", "--device", "cpu", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.stdout)
    assert data


def test_bench_service_needs_target(runner: CliRunner) -> None:
    r = runner.invoke(cli, ["bench", "service", "--sessions", "1"])
    assert r.exit_code != 0


def test_bench_service_in_process(runner: CliRunner, tiny_pack_dir: Path) -> None:
    r = runner.invoke(cli, ["bench", "service", "--pack", str(tiny_pack_dir), "--sessions", "2", "--tokens", "1",
                            "--verify"])
    assert r.exit_code == 0, r.output
    res = json.loads(r.stdout)
    assert res["errors"] == 0 and res["mismatches"] == 0 and res["calls"] == 2 * 8


# ---- mask store wiring ---------------------------------------------------------------------------
def test_maskstore_commands(runner: CliRunner, tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    root, kek = tmp_path / "masks", tmp_path / "kek"
    r = runner.invoke(cli, ["maskstore", "init", "--root", str(root), "--key-file", str(kek)])
    assert r.exit_code == 0, r.output
    assert stat.S_IMODE(kek.stat().st_mode) == 0o600
    r = runner.invoke(cli, ["maskstore", "status", "--root", str(root), "--key-file", str(kek), "--json"])
    assert r.exit_code == 0, r.output
    st = json.loads(r.stdout)
    assert st["inventory"] == [] and "kek" not in json.dumps(st).lower().replace("kek_id", "")
    r = runner.invoke(cli, ["maskstore", "status", "--root", str(root), "--key-file", str(kek)])
    assert r.exit_code == 0 and "(empty)" in r.output
    r = runner.invoke(cli, ["maskstore", "gc", "--root", str(root), "--key-file", str(kek)])
    assert r.exit_code == 0 and "freed 0 bytes" in r.output
    # env-configured store
    clean_env.setenv("PGW_MASKSTORE_ROOT", str(root))
    clean_env.setenv("PGW_MASKSTORE_KEY_FILE", str(kek))
    r = runner.invoke(cli, ["maskstore", "status", "--json"])
    assert r.exit_code == 0, r.output
    # wrong key: integrity / key error -> exit 1, not a traceback
    other = tmp_path / "other_kek"
    other.write_bytes(os.urandom(32))
    other.chmod(0o600)
    r = runner.invoke(cli, ["maskstore", "status", "--root", str(root), "--key-file", str(other)])
    assert r.exit_code == EXIT_ERROR and "error [" in r.output
    # rotate
    new = tmp_path / "kek2"
    r = runner.invoke(cli, ["maskstore", "rotate-key", "--root", str(root), "--key-file", str(kek), str(new)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli, ["maskstore", "status", "--root", str(root), "--key-file", str(new)])
    assert r.exit_code == 0, r.output


def test_maskstore_import_failure_exit_1(runner: CliRunner, tmp_path: Path) -> None:
    root, kek = tmp_path / "masks", tmp_path / "kek"
    assert runner.invoke(cli, ["maskstore", "init", "--root", str(root), "--key-file", str(kek)]).exit_code == 0
    junk = tmp_path / "junk.pgwm"
    junk.write_bytes(b"not a segment")
    r = runner.invoke(cli, ["maskstore", "import", "--root", str(root), "--key-file", str(kek), str(junk)])
    assert r.exit_code == EXIT_ERROR
    assert json.loads(r.stdout)[0]["ok"] is False


def test_maskstore_recipient_key(runner: CliRunner, tmp_path: Path) -> None:
    from pgw.factory.sinks import load_recipient_public_key

    priv, pub = tmp_path / "keys" / "recipient.key", tmp_path / "recipient.pub.hex"
    r = runner.invoke(cli, ["maskstore", "recipient-key", "create", str(priv), "--public-out", str(pub)])
    assert r.exit_code == 0, r.output
    info = json.loads(r.stdout)
    assert info["kek_id"] == "hpke:" + info["key_id"] and len(info["public_key_hex"]) == 64
    assert stat.S_IMODE(priv.stat().st_mode) & 0o077 == 0
    assert load_recipient_public_key(pub).hex() == info["public_key_hex"]
    r = runner.invoke(cli, ["maskstore", "recipient-key", "show", str(priv)])
    assert json.loads(r.stdout) == info
    r = runner.invoke(cli, ["maskstore", "recipient-key", "create", str(priv)])
    assert r.exit_code == EXIT_ERROR and "refusing to overwrite" in r.output


# ---- review hardening ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["${env:FOO", "a ${unterminated", "x: ${}"])
def test_expand_placeholders_rejects_malformed(text: str) -> None:
    from pgw.cli.configtools import expand_placeholders

    with pytest.raises(ConfigError, match="placeholder"):
        expand_placeholders(text, env={"FOO": "secret-value"})


def test_expand_placeholders_error_never_contains_values(tmp_path: Path) -> None:
    from pgw.cli.configtools import expand_placeholders

    with pytest.raises(ConfigError) as ei:
        expand_placeholders("${env:FOO} ${bogus}", env={"FOO": "secret-value"})
    assert "secret-value" not in str(ei.value)
    # substituted values are never re-expanded
    assert expand_placeholders("${env:FOO}", env={"FOO": "${env:BAR}"}) == "${env:BAR}"


def test_handle_errors_maps_oserror() -> None:
    r = CliRunner().invoke(_cmd(PermissionError(13, "Permission denied", "/x/y")), [])
    assert r.exit_code == EXIT_ERROR
    assert "error [os]: Permission denied (/x/y)" in r.output


def test_service_token_files_are_tightened_and_symlinks_refused(runner: CliRunner, tmp_path: Path) -> None:
    tf = tmp_path / "tokens"
    tf.write_text("")
    tf.chmod(0o644)
    out = tmp_path / "tok"
    out.write_text("old")
    out.chmod(0o644)
    r = runner.invoke(cli, ["service", "token", "create", "--tenant", "acme", "--tokens-file", str(tf),
                            "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert stat.S_IMODE(tf.stat().st_mode) == 0o600
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    target = tmp_path / "elsewhere"
    target.write_text("")
    link = tmp_path / "link"
    link.symlink_to(target)
    r = runner.invoke(cli, ["service", "token", "create", "--tenant", "acme", "--tokens-file", str(tf),
                            "--out", str(link)])
    assert r.exit_code == EXIT_ERROR and "cannot write" in r.output
    assert target.read_text() == ""
    assert len(tf.read_text().splitlines()) == 1  # no orphan hash for the token that was never written
