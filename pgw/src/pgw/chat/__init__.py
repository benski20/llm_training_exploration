"""PGW Chat: a private chat app whose plaintext exists only on the user's own device (``pgw chat``).

The client library (``pgw.OpenAI``) runs in this process on the user's machine and talks directly to the hosted
GPU provider, which sees only masked data and whose every result is checked here. ``pgw chat`` serves the chat web
app on ``127.0.0.1`` (a random token per launch, Host / Origin checks against DNS rebinding, a strict CSP) and keeps
conversations in 0600 files under the SDK state directory. No server of ours ever sees a prompt or an answer.

Design: ``research/reviews/encrypted-chat-app.md``; user guide: ``docs/sdk/chat.md``.

Modules: ``server`` (the localhost HTTP server), ``backend`` (the SDK), ``store`` (conversations), ``status``
(readiness and error views), ``cli`` (``pgw chat``), ``testing`` (a fake backend for tests and screenshots).
"""
