#!/usr/bin/env python3
"""Screenshots of every developer-platform page (light and dark, desktop and phone).

    python scripts/platform_screenshots.py [--out docs/design/screenshots] [--only overview,keys]

Serves ``python -m pgw.platform --demo`` in-process on a free loopback port, signs in with the demo token and
captures each page with headless Chromium (Python Playwright; browsers from PLAYWRIGHT_BROWSERS_PATH).
Fails on any console error or CSP violation, so a screenshot run doubles as a strict-CSP check.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PAGES = [
    ("login", "/platform/login"),
    ("overview", "/platform/"),
    ("quickstart", "/platform/quickstart"),
    ("keys", "/platform/keys"),
    ("keys-new", "/platform/keys/new"),
    ("keys-created", None),  # create a key, screenshot the one-time reveal
    ("keys-revoke", "/platform/keys/key_stag03/revoke"),
    ("models", "/platform/models"),
    ("usage", "/platform/usage"),
    ("usage-by-key", "/platform/usage?by=key"),
    ("requests", "/platform/requests"),
    ("settings", "/platform/settings"),
]
VIEWPORTS = {"desktop": (1440, 900), "phone": (390, 844)}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _shrink(path: Path) -> None:
    """Quantize to a 256-colour palette (flat UI survives it; files are about 3x smaller in git)."""
    try:
        from PIL import Image
    except ImportError:
        return
    im = Image.open(path).convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)


def _serve(port: int) -> None:
    import uvicorn

    from pgw.platform.__main__ import build

    cfg = uvicorn.Config(build(True), host="127.0.0.1", port=port, log_level="warning")  # type: ignore[arg-type]
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            return
        except OSError:
            time.sleep(0.05)
    raise SystemExit("platform demo server did not start")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/design/screenshots"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {x for x in args.only.split(",") if x}

    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for vp_name, (w, h) in VIEWPORTS.items():
            for scheme in ("light", "dark"):
                # a fresh demo organisation per run, so a key created in one run does not show up in the next
                port = _free_port()
                _serve(port)
                base = f"http://127.0.0.1:{port}"
                ctx = browser.new_context(viewport={"width": w, "height": h}, color_scheme=scheme,
                                          device_scale_factor=2 if vp_name == "phone" else 1)
                page = ctx.new_page()
                page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type == "error" else None)
                page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
                for name, path in PAGES:
                    if only and name not in only and name != "login":  # always sign in
                        continue
                    if name == "login":
                        page.goto(base + path)
                    elif name == "keys-created":
                        page.goto(base + "/platform/keys/new")
                        page.fill("#key-name", "Data pipeline")
                        page.check("input[name=models][value='qwen2.5-7b-instruct']", force=True)
                        page.click("button[type=submit].btn-primary")
                    else:
                        page.goto(base + path)
                    page.wait_for_load_state("networkidle")
                    page.evaluate("document.fonts.ready")
                    if not only or name in only:
                        shot = out / f"platform-{name}-{scheme}-{vp_name}.png"
                        page.screenshot(path=str(shot), full_page=True)
                        _shrink(shot)
                        print(shot.relative_to(ROOT) if shot.is_relative_to(ROOT) else shot)
                    if name == "login":
                        page.fill("#token", "demo")
                        page.click("button[type=submit]")
                        page.wait_for_url(base + "/platform/")
                ctx.close()
        browser.close()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(f"{len(errors)} console error(s)")


if __name__ == "__main__":
    main()
