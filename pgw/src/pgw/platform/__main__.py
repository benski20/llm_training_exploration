"""Run the developer platform locally: ``python -m pgw.platform --demo [--port 8090]``.

``--demo`` serves the in-memory sample organisation (:class:`pgw.platform.fixtures.FixtureAccounts`;
sign in with the token ``demo``, or ``demo-viewer`` for the read-only role). Without ``--demo`` the
accounts API comes from ``PGW_PLATFORM_ACCOUNTS_URL`` / ``PGW_PLATFORM_ACCOUNTS_TOKEN_FILE``.
Binds to 127.0.0.1 only.
"""

from __future__ import annotations

import argparse


def build(demo: bool) -> object:
    from fastapi import FastAPI
    from starlette.responses import RedirectResponse

    from pgw.platform.app import PlatformSettings, mount_platform
    from pgw.platform.client import AccountsClient

    settings = PlatformSettings()
    accounts: AccountsClient | None
    if demo:
        from pgw.platform.fixtures import FixtureAccounts

        accounts = FixtureAccounts()
    else:
        accounts = settings.build_accounts()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    mount_platform(app, accounts, settings)
    prefix = "/" + settings.path_prefix.strip("/")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(prefix + "/", status_code=307)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m pgw.platform", description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true", help="serve the sample organisation (no accounts API needed)")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()
    import uvicorn

    uvicorn.run(build(args.demo), host="127.0.0.1", port=args.port, log_level="warning")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
