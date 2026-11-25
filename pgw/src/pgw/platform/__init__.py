"""PGW developer platform: the provider-side web console for SDK customers.

Sign in, overview, API keys, usage, request log (metadata only), models, quickstart and settings,
server-rendered with the shared PGW design system (:mod:`pgw.design`). Data comes from the provider's
accounts API through :class:`pgw.platform.client.AccountsClient`.

Mount it on the provider service with :func:`mount_platform`; run a local demo with sample data with
``python -m pgw.platform --demo``.
"""

from pgw.platform.app import PlatformSettings, create_platform_app, mount_platform
from pgw.platform.client import AccountsClient, AccountsError, HttpAccountsClient

__all__ = ["AccountsClient", "AccountsError", "HttpAccountsClient", "PlatformSettings", "create_platform_app",
           "mount_platform"]
