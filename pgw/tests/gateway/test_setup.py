"""First-run setup mode: access control (setup token, CSRF), production guard, and the full wizard
against a real TLS compute service with a tiny model (pack built from a local checkpoint)."""

from __future__ import annotations

import asyncio
import json
import stat
import threading
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from pgw.gateway.config import GatewaySettings
from pgw.gateway.setup import SetupController, SetupOptions, create_setup_app, load_setup_state, should_run_setup
from tests.enroll_helpers import EnrollService, enroll_service

pytestmark = [pytest.mark.timeout(300)]

SETUP_TOKEN = "pgw-setup-test-token-0123456789"  # noqa: S105 - test credential


def _settings(tmp_path: Path, **kw: Any) -> GatewaySettings:
    return GatewaySettings(config_dir=tmp_path / "etc", data_dir=tmp_path / "data", **kw)


class InThreadFactory:
    """Test launcher: one factory run in a thread (the default launcher spawns 'pgw factory run')."""

    def __init__(self, env: dict[str, str]):
        self.env = env
        self.done = threading.Event()
        self.error: BaseException | None = None

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        from pgw.factory.config import FactorySettings
        from pgw.factory.runtime import build_scheduler
        from pgw.maskstore.admin import open_store
        from pgw.maskstore.config import MaskStoreSettings

        f = {k[len("PGW_FACTORY_"):].lower(): v for k, v in self.env.items() if k.startswith("PGW_FACTORY_")}
        f.update(low_water_slots=256, target_slots=512, segment_slots=256)
        m = {k[len("PGW_MASKSTORE_"):].lower(): v for k, v in self.env.items() if k.startswith("PGW_MASKSTORE_")}
        try:
            store = open_store(MaskStoreSettings(**m))
            try:
                with build_scheduler(FactorySettings(**f), store) as sched:
                    sched.run_once()
            finally:
                store.close()
        except BaseException as exc:  # surfaced by the test
            self.error = exc
        finally:
            self.done.set()

    def stop(self) -> None:
        pass

    def status(self) -> dict[str, Any]:
        return {"running": not self.done.is_set(), "restarts": 0, "last_exit": None, "started_at": None}


@pytest.fixture
async def setup_client(tmp_path: Path) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    app = create_setup_app(_settings(tmp_path), setup_token=SETUP_TOKEN)
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 5000))
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8080", timeout=120) as c:
        yield c, app
    await app.aclose()


async def _login(c: httpx.AsyncClient, token: str = SETUP_TOKEN) -> str:
    r = await c.post("/console/api/setup/login", json={"token": token})
    assert r.status_code == 200, r.text
    return str(r.json()["csrf"])


# ---------------------------------------------------------------------------------------------- access
async def test_setup_mode_serves_only_setup_routes(setup_client: tuple[httpx.AsyncClient, Any]) -> None:
    c, _ = setup_client
    assert (await c.get("/healthz")).status_code == 200
    r = await c.get("/readyz")
    assert r.status_code == 503 and r.json()["setup_mode"] is True
    for path in ("/v1/models", "/admin/status", "/metrics", "/dashboard"):
        assert (await c.get(path)).status_code == 503, path
    assert (await c.post("/v1/chat/completions", json={})).status_code == 503
    r = await c.get("/console/setup")
    assert r.status_code == 200 and "Content-Security-Policy" in r.headers
    assert (await c.get("/console/api/setup/status")).json() == {"setup_mode": True, "signed_in": False}


async def test_setup_requires_the_setup_token_and_csrf(setup_client: tuple[httpx.AsyncClient, Any]) -> None:
    c, _ = setup_client
    assert (await c.get("/console/api/setup/state")).status_code == 401
    assert (await c.post("/console/api/setup/enroll", json={"code": "x"})).status_code == 401
    r = await c.post("/console/api/setup/login", json={"token": "wrong-token-0123456789"})
    assert r.status_code == 401
    csrf = await _login(c)
    cookie = c.cookies.get("pgw_console")
    assert cookie
    set_cookie = r.headers.get("set-cookie", "")
    assert "pgw-setup" not in set_cookie
    assert (await c.get("/console/api/setup/state")).status_code == 200
    # writes need the CSRF token
    r = await c.post("/console/api/setup/maskstore", json={})
    assert r.status_code == 403 and "CSRF" in r.json()["error"]["message"]
    r = await c.post("/console/api/setup/maskstore", json={}, headers={"X-PGW-CSRF": "nope"})
    assert r.status_code == 403
    me = (await c.get("/console/api/me")).json()
    assert me["setup"] is True and me["csrf"] == csrf


async def test_setup_login_is_rate_limited(setup_client: tuple[httpx.AsyncClient, Any]) -> None:
    c, _ = setup_client
    codes = [(await c.post("/console/api/setup/login", json={"token": f"wrong-{i:020d}"})).status_code
             for i in range(11)]
    assert codes[:10] == [401] * 10 and codes[10] == 429


async def test_setup_refuses_dev_options(setup_client: tuple[httpx.AsyncClient, Any], tmp_path: Path) -> None:
    c, _ = setup_client
    h = {"X-PGW-CSRF": await _login(c)}
    for path, body in (("/console/api/setup/admin", {"allow_dev_modes": True}),
                       ("/console/api/setup/maskstore", {"allow_no_external_counter": True}),
                       ("/console/api/setup/model", {"name": "m", "mode": "plain"}),
                       ("/console/api/setup/enroll", {"code": "PGW1-AAAAA", "freivalds_rate": 0.1}),
                       ("/console/api/setup/verify", {"noise_monitor": False}),
                       ("/console/api/setup/finish", {"require_service_tls": False})):
        r = await c.post(path, json=body, headers=h)
        assert r.status_code == 400, (path, r.text)
        assert "not a setup option" in r.json()["error"]["message"]
    # an enrollment code for plain http to a non-loopback host cannot even be expressed
    from pgw.enroll.code import EnrollmentCode

    with pytest.raises(Exception, match="https"):
        EnrollmentCode("http://compute.example.net", bytes(32), bytes(16))
    # a typo in the code is a 400 with a plain message, not a session error
    r = await c.post("/console/api/setup/enroll", json={"code": "PGW1-AAAAA-AAAAA-AAAAA-AAAAA"}, headers=h)
    assert r.status_code == 400 and "code" in r.json()["error"]["message"]
    # steps run in order
    r = await c.post("/console/api/setup/model", json={"name": "x"}, headers=h)
    assert r.status_code == 400 and "enroll" in r.json()["error"]["message"]


def test_factory_supervisor_restarts_and_stops(tmp_path: Path) -> None:
    import sys

    from pgw.gateway.setup_jobs import FactorySupervisor, read_env_file

    marker = tmp_path / "runs"
    sup = FactorySupervisor({"PGW_TEST_MARKER": str(marker)},
                            argv=[sys.executable, "-c", "import os; open(os.environ['PGW_TEST_MARKER'], 'a').write('x')"])
    sup.start()
    t0 = time.monotonic()
    while sup.last_exit is None and time.monotonic() - t0 < 30:
        time.sleep(0.05)
    sup.stop(timeout_s=10)
    assert marker.read_text().startswith("x") and sup.last_exit == 0 and not sup.running
    env = tmp_path / "f.env"
    env.write_text("# comment\nPGW_FACTORY_RING=K32A12C27\n\nPGW_MASKSTORE_COUNTER=file\n")
    assert read_env_file(env) == {"PGW_FACTORY_RING": "K32A12C27", "PGW_MASKSTORE_COUNTER": "file"}


def test_should_run_setup(tmp_path: Path, tiny_pack_dir: Path) -> None:
    from pgw.gateway import GatewayModelConfig

    assert should_run_setup(_settings(tmp_path))  # no models
    model = GatewayModelConfig(name="m", pack_path=tiny_pack_dir, mode="masked", service_url="https://x.example.net",
                               fingerprint="0" * 64)
    assert not should_run_setup(_settings(tmp_path, models=[model]))
    assert should_run_setup(_settings(tmp_path, models=[model], setup_mode=True))
    SetupController(_settings(tmp_path))  # persists an unfinished run
    assert should_run_setup(_settings(tmp_path, models=[model]))


# -------------------------------------------------------------------------------------------- wizard
@pytest.fixture(scope="module")
def svc(tiny_pack_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[EnrollService]:
    with enroll_service(tiny_pack_dir, tmp_path_factory.mktemp("setup-svc")) as s:
        yield s


async def _wait(c: httpx.AsyncClient, pred: Any, what: str, timeout_s: float = 180.0) -> dict[str, Any]:
    t0 = time.monotonic()
    st: dict[str, Any] = {}
    while time.monotonic() - t0 < timeout_s:
        st = (await c.get("/console/api/setup/state")).json()
        if pred(st):
            return st
        await asyncio.sleep(0.3)
    raise AssertionError(f"timed out waiting for {what}: {json.dumps(st)[:2000]}")


async def test_wizard_happy_path(svc: EnrollService, tmp_path: Path, tiny_hf_dir: Path, tiny_pack: Any,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
    resolved: list[tuple[str, str | None]] = []

    def resolver(repo: str, revision: str | None) -> Path:
        resolved.append((repo, revision))
        return tiny_hf_dir  # instead of the HF hub

    factories: list[InThreadFactory] = []

    def launcher(env: dict[str, str]) -> InThreadFactory:
        factories.append(InThreadFactory(env))
        return factories[-1]

    settings = _settings(tmp_path)
    app = create_setup_app(settings, setup_token=SETUP_TOKEN,
                           options=SetupOptions(checkpoint_resolver=resolver, factory_launcher=launcher))
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 5000))
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8080", timeout=240) as c:
            h = {"X-PGW-CSRF": await _login(c)}
            code, _ = svc.create_code("acme")
            # 1. enrollment
            r = await c.post("/console/api/setup/enroll", json={"code": code, "gateway_name": "gw-wizard"}, headers=h)
            assert r.status_code == 200, r.text
            assert r.json()["tenant"] == "acme" and r.json()["client_cert"] is True
            token = (tmp_path / "etc" / "secrets" / "service_token").read_text().strip()
            assert token not in r.text and "pgwt-" not in r.text
            # 2. model: pack built in the background from the injected source, fingerprint checked
            r = await c.post("/console/api/setup/model", json={"name": "tiny-qwen2"}, headers=h)
            assert r.status_code == 200, r.text
            st = await _wait(c, lambda s: s["model"]["pack_status"] in ("ready", "failed"), "pack build")
            assert st["model"]["pack_status"] == "ready", st["jobs"]["pack"]
            assert st["jobs"]["pack"]["phase"] == "done" and st["jobs"]["pack"]["total"] == 8
            assert resolved and resolved[0][0] == tiny_hf_dir.name
            assert st["estimate"]["basis"] == "model" and st["estimate"]["hint_bytes"] > 0
            # 3. mask store
            r = await c.post("/console/api/setup/maskstore", json={}, headers=h)
            assert r.status_code == 200, r.text
            kek = Path(r.json()["key_file"])
            assert stat.S_IMODE(kek.stat().st_mode) == 0o600
            # 4. admin token, shown once
            r = await c.post("/console/api/setup/admin", json={}, headers=h)
            assert r.status_code == 200, r.text
            admin_token = r.json()["admin_token"]
            assert admin_token not in (await c.get("/console/api/setup/state")).text
            # 5. verify: starts the hint build and the factory; checks pass once masks are in
            r = await c.post("/console/api/setup/verify", json={}, headers=h)
            assert r.status_code == 200, r.text
            await _wait(c, lambda s: (s["masks"] or 0) > 0 or (factories and factories[0].done.is_set()),
                        "first masks")
            assert factories and factories[0].error is None, factories[0].error
            # a new store gets a small first segment, so the gateway can serve before a full one is made
            assert factories[0].env["PGW_FACTORY_FIRST_SEGMENT_SLOTS"] == "256"
            r = await c.post("/console/api/setup/verify", json={}, headers=h)
            checks = {x["id"]: x for x in r.json()["checks"]}
            assert r.json()["ok"], checks
            assert checks["service"]["status"] == "ok" and "pinned TLS" in checks["service"]["label"]
            assert checks["masks"]["status"] == "ok" and checks["freivalds"]["status"] == "ok"
            assert checks["egress"]["status"] == "info"
            # 6. finish: config written, gateway switches to normal mode in-process
            r = await c.post("/console/api/setup/finish", json={}, headers=h)
            assert r.status_code == 200, r.text
            assert r.json()["mode"] == "normal", r.json()
            env = (tmp_path / "etc" / "gateway.env").read_text()
            assert stat.S_IMODE((tmp_path / "etc" / "gateway.env").stat().st_mode) == 0o600
            assert "ALLOW_DEV" not in env and admin_token not in env and "PGW_MASKSTORE_COUNTER=file" in env
            assert load_setup_state(tmp_path / "etc")["finished"] is True  # type: ignore[index]
            # normal mode: setup routes are gone, admin token works, a masked request succeeds
            assert (await c.get("/console/api/setup/state")).status_code == 404
            admin = {"Authorization": f"Bearer {admin_token}"}
            st = await _wait_status(c, admin)
            assert st["models"][0]["fingerprint"] == tiny_pack.fingerprint
            r = await c.post("/admin/keys", headers=admin, json={"name": "first-app"})
            assert r.status_code == 201, r.text
            key = r.json()["key"]
            r = await c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                             json={"model": "tiny-qwen2", "messages": [{"role": "user", "content": "hi"}],
                                   "max_tokens": 3})
            assert r.status_code == 200, r.text
            assert r.json()["usage"]["completion_tokens"] >= 1
    finally:
        await app.aclose()
    # a restarted gateway reads gateway.env by itself and starts in normal mode with the written settings
    import os

    from pgw.gateway.cli import _settings_with_wizard_env

    monkeypatch.setattr(os, "environ", dict(os.environ))
    restarted = _settings_with_wizard_env(_settings(tmp_path))
    assert restarted.models_file == tmp_path / "etc" / "models.json"
    assert restarted.factory_env_file == tmp_path / "etc" / "factory.env"
    assert restarted.admin_token is not None and restarted.admin_token.get_secret_value() == admin_token
    assert not should_run_setup(restarted)
    restarted.validate_policy()


async def _wait_status(c: httpx.AsyncClient, admin: dict[str, str]) -> dict[str, Any]:
    t0 = time.monotonic()
    body: dict[str, Any] = {}
    while time.monotonic() - t0 < 120:
        r = await c.get("/admin/status", headers=admin)
        body = r.json()
        if r.status_code == 200 and body["models"] and body["models"][0].get("ready"):
            return body
        await asyncio.sleep(0.3)
    raise AssertionError(f"model not ready: {json.dumps(body)[:2000]}")


async def test_wizard_refuses_a_pack_that_differs_from_the_service(svc: EnrollService, tmp_path: Path,
                                                                    tiny_llama_hf_dir: Path) -> None:
    settings = _settings(tmp_path)
    ctl = SetupController(settings, SetupOptions(checkpoint_resolver=lambda repo, rev: tiny_llama_hf_dir))
    code, _ = svc.create_code("acme")
    from pgw.gateway.setup import EnrollBody, ModelBody

    ctl.enroll(EnrollBody(code=code))
    ctl.choose_model(ModelBody(name="tiny-qwen2"))
    assert ctl.pack_job is not None
    ctl.pack_job.join(120)
    st = ctl.view()
    assert st["model"]["pack_status"] == "failed"
    assert "not the one the service runs" in st["jobs"]["pack"]["error"]
    assert not Path(st["model"]["pack_path"]).exists()
    # state survives a restart (the wizard resumes where it was)
    again = SetupController(settings).view()
    assert again["enroll"]["tenant"] == "acme" and again["model"]["pack_status"] == "failed"
