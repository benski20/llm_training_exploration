"""What ``pgw chat`` tells the browser about the client library: a model's readiness and its errors.

:func:`summarize` turns one model of :meth:`pgw.OpenAI.status` into a small, content-free view for the UI's setup
screen and privacy indicator. :func:`error_view` maps the library's typed errors to what the UI shows. Neither ever
includes content (the library's statuses and error messages carry none by design).
"""

from __future__ import annotations

from typing import Any

__all__ = ["error_view", "summarize"]


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else None


def summarize(status: dict[str, Any], model: str, *, warming: bool = False,
              warm_error: dict[str, Any] | None = None) -> dict[str, Any]:
    """One model's readiness.

    ``phase`` is one of ``idle`` (nothing started), ``starting``, ``enrolling``, ``provisioning``, ``connecting``,
    ``setup`` (first-time mask setup on this machine), ``preparing`` (waiting for the first masks), ``ready``,
    ``stalled`` (the provider is unreachable or its mask generation makes no progress), ``blocked`` (an integrity
    failure halted the model), ``failed``.
    """
    models: dict[str, Any] = status["models"] if isinstance(status.get("models"), dict) else {}
    h: dict[str, Any] = models.get(model) or {}
    out: dict[str, Any] = {
        "model": model, "phase": "idle", "title": "Not started", "detail": None, "percent": None, "eta_s": None,
        "masks_available": _num(h.get("masks_available")), "requests": int(h.get("requests") or 0),
        "verified_requests": int(h.get("verified_requests") or 0),
        "integrity_failures": int(h.get("integrity_failures") or 0),
        "last_verified_at": _num(h.get("last_integrity_check_ok_at")),
        "lane": h.get("lane"), "masked_head": bool(h.get("masked_head")),
        "prefix_cache": isinstance(h.get("prefix_cache"), dict),
        "mask_scheme": status.get("mask_scheme"), "error": None,
    }
    state = h.get("state")
    def sub(d: dict[str, Any], key: str) -> dict[str, Any]:
        v = d.get(key)
        return v if isinstance(v, dict) else {}

    fac, halt, breaker = sub(h, "factory"), sub(h, "integrity_halt"), sub(h, "breaker")
    health = sub(fac, "health")

    def put(phase: str, title: str, detail: str | None = None) -> dict[str, Any]:
        out.update(phase=phase, title=title, detail=detail)
        return out

    if halt.get("halted") or breaker.get("open"):
        return put("blocked", "Paused after a failed integrity check",
                   "A result from the provider failed its check and was discarded. The model refuses traffic "
                   "until the cool-down ends; report it to the provider.")
    if not state or state == "new":
        if warm_error is not None:
            out["error"] = warm_error
            return put("failed", warm_error.get("title") or "Could not start", warm_error.get("message"))
        return put("starting", "Starting") if warming else out
    if state == "enrolling":
        return put("enrolling", "Connecting your key", "Registering this machine with the provider.")
    if state == "provisioning":
        return put("provisioning", "Downloading the public model files",
                   h.get("progress") or "Checking every file against the pinned fingerprint.")
    if state == "connecting":
        return put("connecting", "Connecting to the provider")
    if state == "failed":
        if warm_error is not None:
            out["error"] = warm_error
        return put("failed", (warm_error or {}).get("title") or "Could not start",
                   (warm_error or {}).get("message") or h.get("error") or "The model failed to start.")
    # ready: the mask factory decides whether a request can start now
    stalled = health.get("stalled")
    if stalled:
        maskgen = health.get("phase") == "maskgen_stalled"
        out["error"] = {"kind": "provider_stalled" if maskgen else "connection", "retryable": True}
        return put("stalled", "Provider is not making progress" if maskgen else "Provider unreachable",
                   str(stalled)[:400])
    setup = sub(health, "setup") or None
    if setup is not None:
        from pgw.sdk.factory import describe_setup

        out["percent"] = _num(setup.get("percent"))
        out["eta_s"] = _num(setup.get("eta_s"))
        line = describe_setup(health) or "first-time setup in progress"
        return put("setup", "First-time setup on this machine", line[0].upper() + line[1:])
    masks = out["masks_available"]
    if masks is not None and masks <= 0:
        seg = sub(health, "segment")
        if seg:
            out["eta_s"] = _num(seg.get("eta_s"))
        return put("preparing", "Preparing masks", "The background mask factory is making the next batch.")
    return put("ready", "Ready")


# (title, what the user can do) per error class; messages come from the library and contain no content
_KINDS: dict[str, tuple[str, str, bool]] = {
    "AuthenticationError": ("auth", "Your API key was refused", False),
    "PermissionDeniedError": ("permission", "This key may not use that model", False),
    "UnsupportedModelError": ("model", "Model not supported", False),
    "NotFoundError": ("model", "Model not available", False),
    "RateLimitError": ("rate_limited", "Rate limited by the provider", True),
    "NotReadyError": ("not_ready", "Not ready yet", True),
    "ProviderStalledError": ("provider_stalled", "Provider is not making progress", True),
    "APITimeoutError": ("timeout", "Timed out", True),
    "APIConnectionError": ("connection", "Provider unreachable", True),
    "IntegrityError": ("integrity", "Integrity check failed: answer discarded", False),
    "BadRequestError": ("bad_request", "Request not accepted", False),
    "InternalServerError": ("internal", "Something went wrong", True),
}


def error_view(exc: BaseException) -> dict[str, Any]:
    """``{"kind", "title", "message", "code", "retryable"}`` for one exception (unknown ones become ``internal``
    with a generic message: their text could carry anything)."""
    from pgw.sdk.errors import PGWAPIError, from_pgw_error

    err = exc if isinstance(exc, PGWAPIError) else from_pgw_error(exc)
    for cls in type(err).__mro__:
        if cls.__name__ in _KINDS:
            kind, title, retry = _KINDS[cls.__name__]
            break
    else:
        kind, title, retry = "internal", "Something went wrong", True
    msg = err.message if isinstance(err.message, str) else ""
    if kind == "integrity":
        msg = ("A result from the provider failed its integrity check, so it was discarded and nothing unverified "
               "was shown. The model pauses until the cool-down ends. Treat this as a security event and report "
               "it to the provider." + (f" ({msg})" if msg else ""))
    return {"kind": kind, "title": title, "message": msg[:600], "code": str(err.code),
            "retryable": retry}
