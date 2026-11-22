"""Modal demo bootstrap (``deploy/modal/pgw_modal_boot.py``, no Modal imports): the customer gateway's speed
settings reach the setup wizard as valid ``setup_model_options``."""

from __future__ import annotations

import importlib.util
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
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
    assert opts == {"local_linears": "auto", "float_threads": 1, "lane": "stream"}
    assert env["PGW_FLOAT_THREADS"] == "1"
    assert "spec_rows" not in opts  # no draft model in the demo; lane = stream works with a remote provider
    for k, v in env.items():
        if k.startswith("PGW_GATEWAY_"):
            monkeypatch.setenv(k, v)
    gs = GatewaySettings()
    assert gs.setup_model_options == opts
    GatewayModelConfig(name="m", pack_path=tmp_path, fingerprint="ab" * 32,
                       service_url="https://compute.example.net:8443", **gs.setup_model_options)


def test_network_path_labels() -> None:
    boot = _boot()
    assert boot.network_path("https://127.0.0.1:8443") == "loopback"
    assert boot.network_path("https://localhost:8443") == "loopback"
    assert boot.network_path("https://[::1]:8443") == "loopback"
    assert boot.network_path("https://[fdaa:0:1::5]:8443") == "private fdaa:0:1::5"
    assert boot.network_path("https://10.0.0.7:8443") == "private 10.0.0.7"
    assert boot.network_path("https://r500.modal.host:41234") == "public r500.modal.host"
    assert boot.network_path("https://8.8.8.8:443") == "public 8.8.8.8"
    assert boot.network_path(None) == "unknown" and boot.network_path("not a url") == "unknown"


def test_children_never_inherit_the_modal_runtime_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    boot = _boot()
    monkeypatch.setenv("MODAL_TASK_ID", "ta-123")
    monkeypatch.setenv("MODAL_SERVER_URL", "https://x")
    monkeypatch.setenv("PGW_SOMETHING", "1")
    pki = {k: tmp_path / k for k in ("server_cert", "server_key", "client_ca", "service_ca", "client_ca_key")}
    for env in (boot.clean_env(), boot.service_env(tmp_path, tmp_path, pki, "https://h:1", 1),
                boot.gateway_env(tmp_path / "c", tmp_path / "d", 1)):
        assert not any(k.startswith("MODAL_") for k in env)
        assert "PGW_SOMETHING" not in env
    assert boot.as_user({}, None) == {}


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0, reason="needs root (drops to another user)")
def test_party_users_cannot_read_each_others_directories() -> None:
    """The co-located placement's isolation primitives: a child started with ``as_user`` runs as that
    user (no supplementary groups, umask 077), writes in its own ``party_dir`` and can neither list a
    0700 directory of another party (here root's) nor read a log file the parent wrote."""
    try:
        pw = pwd.getpwnam("nobody")
    except KeyError:
        pytest.skip("no 'nobody' user")
    boot = _boot()
    base = Path(tempfile.mkdtemp(prefix="pgw-party-", dir="/tmp"))
    try:
        os.chmod(base, 0o711)  # noqa: S103 - traversable (not listable) for the dropped-privilege child
        mine = boot.party_dir(base / "mine", "nobody")
        other = boot.party_dir(base / "other", None)
        (other / "secret").write_text("x")
        log = base / "log.txt"
        boot.start_logged(["/bin/true"], dict(os.environ), log).wait(10)
        assert (log.stat().st_mode & 0o777) == 0o600
        assert mine.stat().st_uid == pw.pw_uid and (mine.stat().st_mode & 0o777) == 0o700
        env = boot.clean_env()
        kw = boot.as_user(env, "nobody", mine)
        code = ("import os, sys\n"
                "f = os.path.join(os.environ['HOME'], 'f')\n"
                "open(f, 'w').write('y')\n"
                "assert os.stat(f).st_mode & 0o777 == 0o600\n"
                "for p in sys.argv[1:]:\n"
                "    try:\n"
                "        os.listdir(p) if os.path.isdir(p) else open(p).read()\n"
                "    except PermissionError:\n"
                "        continue\n"
                "    sys.exit('read ' + p)\n"
                "print(os.getuid(), *os.getgroups())\n")
        r = subprocess.run([sys.executable, "-c", code, str(other), str(log)], env=env, capture_output=True,  # noqa: S603
                           text=True, timeout=60, check=False, **kw)
        assert r.returncode == 0, r.stderr
        ids = [int(x) for x in r.stdout.split()]
        assert ids[0] == pw.pw_uid and 0 not in ids[1:]
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_colocated_party_starters_isolate_the_parties() -> None:
    """``start_provider_party`` binds the service to loopback only and hands its directory to the provider's
    user after the PKI and state files exist; ``start_gateway_party`` hides the GPU from the gateway. Both
    children run through ``as_user`` and log to a directory of the parent."""
    import ast

    text = (MODAL_DIR / "pgw_modal_boot.py").read_text()
    defs = {n.name: ast.get_source_segment(text, n) or "" for n in ast.parse(text).body
            if isinstance(n, ast.FunctionDef)}
    prov, gw = defs["start_provider_party"], defs["start_gateway_party"]
    assert 'env["PGW_SERVICE_HOST"] = "127.0.0.1"' in prov
    assert prov.index("make_pki") < prov.index("party_dir(work, user)")
    assert prov.index("service_env") < prov.index("party_dir(work, user)")
    assert "user_kw=kw" in prov and "as_user(env, user" in prov
    assert 'env["CUDA_VISIBLE_DEVICES"] = ""' in gw and "user_kw=kw" in gw and "as_user(env, user" in gw


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
    for name in ("provider.py", "colocated.py"):  # the co-located container packs the same target
        tree = ast.parse((MODAL_DIR / name).read_text())
        table = next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "MODELS" for t in n.targets))
        assert table == boot.MODELS, name
    assert all(len(rev) == 40 for _, rev in boot.MODELS.values())
    assert boot.MODELS[boot.MODEL_NAME][0] == boot.MODEL_REPO


def test_e2e_waits_for_both_strict_chats_slots() -> None:
    """The scripted e2e chats (32 tokens, prompt under 64 tokens) fail fast without their whole strict
    reservation: the wizard step waits for both chats' slots (and keeps its old 64 without speculation)."""
    boot = _boot()
    assert boot.e2e_min_masks(0) == 64
    assert boot.e2e_min_masks(4) == 2 * (64 + 32 + 31 * 3)
    assert boot.e2e_min_masks(4, prompt_tokens=41, chats=1) == 41 + 32 + 31 * 3


def test_customer_memory_matches_boot() -> None:
    """customer.py computes its memory without importing this file at module level: same rule."""
    import re

    boot = _boot()
    src = (MODAL_DIR / "customer.py").read_text()
    m = re.search(r"MEMORY_MIB = (\d+) if DRAFT or MODEL != \"([^\"]+)\" else (\d+)", src)
    assert m is not None
    assert int(m.group(1)) == boot.CUSTOMER_MEMORY_MIB["large"] == boot.customer_memory_mib(None, draft=True)
    assert int(m.group(3)) == boot.CUSTOMER_MEMORY_MIB["default"] == boot.customer_memory_mib(None)
    assert m.group(2) == boot.MODEL_NAME
    assert boot.customer_memory_mib("qwen2.5-1.5b-instruct") == boot.CUSTOMER_MEMORY_MIB["large"]
    # the co-located container: the provider's 6 GiB plus the same gateway rule
    col = (MODAL_DIR / "colocated.py").read_text()
    m = re.search(r"MEMORY_MIB = 6144 \+ \((\d+) if DRAFT or MODEL_NAME != \"([^\"]+)\" else (\d+)\)", col)
    assert m is not None
    assert (int(m.group(1)), m.group(2), int(m.group(3))) == (
        boot.CUSTOMER_MEMORY_MIB["large"], boot.MODEL_NAME, boot.CUSTOMER_MEMORY_MIB["default"])


def test_lane_tls_options_validate_against_the_gateway_config(tmp_path: Path) -> None:
    from pgw.gateway.config import GatewayModelConfig

    boot = _boot()
    assert boot.lane_options("stream") == {"lane": "stream"} and boot.lane_options("tls") == {"lane": "tls"}
    with pytest.raises(ValueError):
        boot.lane_options("quic")
    # --lane uds: the product's same-host lane, taken by the gateway's "auto" (no egress entry: no TCP port)
    assert boot.lane_options("uds") == {"lane": "auto"} and boot.lane_egress("uds", None) == []
    assert GatewayModelConfig(name="m", pack_path=tmp_path, fingerprint="ab" * 32, service_url="https://127.0.0.1:8443",
                              **{**boot.GATEWAY_MODEL_OPTIONS, **boot.lane_options("uds")}).lane == "auto"
    assert 'lane_socket=LANE_SOCKET if lane == "uds" else None' in (MODAL_DIR / "colocated.py").read_text()
    c = GatewayModelConfig(name="m", pack_path=tmp_path, fingerprint="ab" * 32,
                           service_url="https://127.0.0.1:8443",
                           **{**boot.GATEWAY_MODEL_OPTIONS, **boot.lane_options("tls")})
    assert c.lane == "tls"
    col = (MODAL_DIR / "colocated.py").read_text()
    assert "lane_port=LANE_PORT if lane == \"tls\" else None" in col and "boot.lane_options(lane)" in col
    # the gateway connects to a TCP lane port only when it is on its egress allowlist
    assert "egress=boot.lane_egress(lane, LANE_PORT)" in col
    assert boot.lane_egress("tls", 8444) == ["https://127.0.0.1:8444"] and boot.lane_egress("stream", None) == []
    with pytest.raises(ValueError):
        boot.lane_egress("tls", None)
    from pgw.gateway import GatewaySettings

    gs = GatewaySettings(egress_allowlist=boot.lane_egress("tls", 8444), models=[c], admin_token="x" * 32)
    assert ("https", "127.0.0.1", 8444) in gs.egress_origins()


def test_spec_mask_preset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``launch.py --spec-mask C/B`` (PGW_MODAL_SPEC_MASK): padded strict speculation presets that the wizard's
    preset check and GatewayModelConfig accept; unset = masked filler rows (no new keys)."""
    from pgw.gateway.config import GatewayModelConfig, GatewaySettings

    boot = _boot()
    monkeypatch.delenv("PGW_MODAL_SPEC_MASK", raising=False)
    assert "spec_mask_rows" not in boot.spec_options(4)
    assert boot.spec_mask_options("") == {}
    assert boot.spec_mask_options("1.5") == {"spec_mask_rows": 1.5}
    monkeypatch.setenv("PGW_MODAL_SPEC_MASK", "4/160")
    opts = boot.spec_options(4)
    assert opts["spec_mask_rows"] == 4.0 and opts["spec_mask_credit"] == 160
    GatewaySettings(setup_model_options=opts)  # the wizard's preset check accepts them
    cfg = GatewayModelConfig(name="m", pack_path=tmp_path, fingerprint="ab" * 32,
                             service_url="https://compute.example.net:8443", **opts)
    assert cfg.spec_mask_rows == 4.0 and cfg.spec_mask_credit == 160
