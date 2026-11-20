"""Real-browser smoke test of the first-run setup wizard.

A real TLS compute service with enrollment runs in a thread; the gateway runs in setup mode under
uvicorn. Headless Chromium signs in with the setup token and walks through every step: enrollment
code, model (packed from the local tiny checkpoint instead of the HF hub), mask store, admin token,
checks (hint build + an in-thread mask factory) and finish. Fails on console errors, page errors,
CSP violations or unexpected failed API calls.

Set ``PGW_SETUP_SCREENSHOTS=<dir>`` to keep a screenshot of each step. Skipped without Playwright.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pgw.console.routes import DEFAULT_STATIC_DIR
from tests.console.test_console_browser import TIMEOUT_MS, Watch, _check, _free_port, _new_page, browser  # noqa: F401
from tests.enroll_helpers import EnrollService, enroll_service
from tests.gateway.test_setup import SETUP_TOKEN, InThreadFactory

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300)]

pytest.importorskip("playwright.sync_api", reason="playwright (Python) is not installed")


@pytest.fixture(scope="module")
def svc(tiny_pack_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[EnrollService]:
    with enroll_service(tiny_pack_dir, tmp_path_factory.mktemp("setup-browser-svc")) as s:
        yield s


@pytest.fixture(scope="module")
def setup_url(tiny_hf_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    import uvicorn

    from pgw.gateway.config import GatewaySettings
    from pgw.gateway.setup import SetupOptions, create_setup_app

    if not (DEFAULT_STATIC_DIR / "index.html").is_file():
        pytest.skip("console is not built (src/pgw/console/static/index.html missing)")
    tmp = tmp_path_factory.mktemp("setup-browser")
    settings = GatewaySettings(config_dir=tmp / "etc", data_dir=tmp / "data")
    app = create_setup_app(settings, setup_token=SETUP_TOKEN, options=SetupOptions(
        checkpoint_resolver=lambda repo, rev: tiny_hf_dir, factory_launcher=InThreadFactory))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, log_level="warning",
                                           lifespan="on"))
    thread = threading.Thread(target=server.run, name="setup-browser-gateway", daemon=True)
    thread.start()
    deadline = time.monotonic() + 60
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("gateway did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=60)


def _shot(page: Any, name: str) -> None:
    d = os.environ.get("PGW_SETUP_SCREENSHOTS")
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(d) / f"setup-{name}.png"), full_page=True)


def test_setup_wizard_walkthrough(browser: Any, setup_url: str, svc: EnrollService) -> None:  # noqa: F811
    ctx, page, w = _new_page(browser)
    try:
        page.goto(f"{setup_url}/console/")
        page.wait_for_selector("text=Set up your PGW gateway")
        _shot(page, "1-token")
        # a wrong setup token is explained (expected 401)
        page.get_by_label("Setup token").fill("pgw-setup-wrong-token-000000")
        page.click("button:has-text('Start setup')")
        page.wait_for_selector("text=was not accepted")
        w.problems = [p for p in w.problems if "/console/api/setup/login" not in p and "/console/api/me" not in p]
        page.get_by_label("Setup token").fill(SETUP_TOKEN)
        page.click("button:has-text('Start setup')")
        page.wait_for_selector("text=Gateway setup")

        # 1. enrollment code (client-side check first, then the real one)
        code, _ = svc.create_code("acme")
        page.get_by_label("Enrollment code").fill("hello")
        page.click(".setup-main button[type=submit]:has-text('Connect')")
        page.wait_for_selector(".field-error:has-text('starts with PGW1-')")
        page.get_by_label("Enrollment code").fill(code)
        _shot(page, "2-enroll")
        page.click(".setup-main button[type=submit]:has-text('Connect')")

        # 2. model
        page.wait_for_selector(".setup-main button:has-text('Download and prepare'), .setup-main .field-error",
                               timeout=60_000)
        assert page.locator(".setup-main .field-error").count() == 0, page.inner_text(".setup-main")
        page.click(".setup-main button:has-text('Download and prepare')")
        page.wait_for_selector("text=is ready", timeout=120_000)
        _shot(page, "3-model")
        page.click(".setup-main button:has-text('Continue')")

        # 3. mask store (cloud KMS shown as coming soon, disabled)
        page.wait_for_selector("text=Where to keep the masks")
        assert page.locator("input[name=kms][value=aws-kms]").is_disabled()
        _shot(page, "4-maskstore")
        page.click("button:has-text('Create mask store')")

        # 4. admin token shown once; continue needs the confirmation
        page.wait_for_selector("button:has-text('Generate admin token')")
        page.click("button:has-text('Generate admin token')")
        token = page.locator("pre[aria-label='Admin token']")
        token.wait_for()
        assert token.inner_text().startswith("pgw-admin-")
        cont = page.locator(".setup-main button:has-text('Continue')")
        assert cont.is_disabled()
        page.get_by_label("I have saved the admin token in a safe place").check()
        _shot(page, "5-admin")
        cont.click()

        # 5. checks (hint build and the first masks happen in the background)
        page.wait_for_selector("button:has-text('Run the checks')")
        page.click("button:has-text('Run the checks')")
        deadline = time.monotonic() + 180
        while True:
            page.wait_for_selector(".setup-checks")
            if page.locator("[data-check=masks].setup-check-ok").count():
                break
            if time.monotonic() > deadline:
                raise AssertionError(page.inner_text(".setup-main"))
            page.wait_for_timeout(1000)
            page.click(".setup-main button:has-text('Check again')")
            page.wait_for_load_state("networkidle")
        assert page.locator(".setup-check-fail").count() == 0, page.inner_text(".setup-checks")
        _shot(page, "6-verify")
        page.click(".setup-main button:has-text('Continue')")

        # 6. finish: the gateway switches to normal mode
        page.click("button:has-text('Finish setup')")
        page.wait_for_selector("text=Setup complete", timeout=120_000)
        _shot(page, "7-finish")
        page.click("text=Go to sign-in")
        page.wait_for_selector(".login-card")
        page.wait_for_load_state("networkidle")
        w.problems = [p for p in w.problems if "/console/api/me" not in p]  # pre-login probe of the console
        _check(page, w)
    except BaseException:
        _shot(page, "failure")
        raise
    finally:
        ctx.close()
