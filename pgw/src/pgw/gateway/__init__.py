"""Customer-side OpenAI-compatible gateway (sees plaintext; the only component that does).

Owner: gateway team. Contract: SPEC.md section 9.

Routes: ``POST /v1/chat/completions``, ``POST /v1/completions`` (both with SSE ``stream=true``),
``GET /v1/models``, admin endpoints under ``/admin``, ``/healthz``, ``/readyz``, ``/metrics``,
``/dashboard``.

Modules: :mod:`~pgw.gateway.app` (FastAPI app), :mod:`~pgw.gateway.runtime` (engine + mask store +
compute client per model), :mod:`~pgw.gateway.auth` (API keys, rate limits, quotas),
:mod:`~pgw.gateway.safety` (integrity circuit breaker, request queue), :mod:`~pgw.gateway.egress`
(egress allowlist, no telemetry), :mod:`~pgw.gateway.audit` (content-free audit log),
:mod:`~pgw.gateway.metrics`, :mod:`~pgw.gateway.cli`.
"""

from pgw.gateway.app import create_app
from pgw.gateway.audit import AuditLog
from pgw.gateway.auth import ApiKey, ApiKeyStore, QuotaTracker, RateLimiter
from pgw.gateway.config import GatewayModelConfig, GatewaySettings
from pgw.gateway.egress import EgressBlockedError, EgressGuardTransport
from pgw.gateway.metrics import GatewayMetrics
from pgw.gateway.runtime import ActiveSession, InferenceRuntime, InvalidRequestError, ModelRuntime
from pgw.gateway.safety import AdmissionController, BreakerOpenError, CircuitBreaker

__all__ = [
    "ActiveSession", "AdmissionController", "ApiKey", "ApiKeyStore", "AuditLog", "BreakerOpenError",
    "CircuitBreaker", "EgressBlockedError", "EgressGuardTransport", "GatewayMetrics", "GatewayModelConfig",
    "GatewaySettings", "InferenceRuntime", "InvalidRequestError", "ModelRuntime", "QuotaTracker", "RateLimiter",
    "create_app",
]
