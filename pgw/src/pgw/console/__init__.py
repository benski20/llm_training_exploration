"""PGW Console: the operator web console served by the gateway at ``/console/``.

The single-page app is built from ``pgw/console`` into :data:`~pgw.console.routes.DEFAULT_STATIC_DIR`
(``static/``). This package holds its server side: sign-in sessions (:mod:`~pgw.console.sessions`),
usage and audit views over the audit log (:mod:`~pgw.console.usage`) and the routes
(:mod:`~pgw.console.routes`). Contract: ``docs/api/console-api.md``.
"""

from pgw.console.sessions import COOKIE_NAME, CSRF_HEADER, ConsoleSessions, LoginRateLimiter

__all__ = ["COOKIE_NAME", "CSRF_HEADER", "ConsoleSessions", "LoginRateLimiter"]
