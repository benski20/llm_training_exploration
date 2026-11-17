"""Customer-side OpenAI-compatible gateway (sees plaintext; the only component that does).

Owner: gateway team. Contract: SPEC.md section 9.

Routes: ``POST /v1/chat/completions``, ``POST /v1/completions`` (both with SSE ``stream=true``),
``GET /v1/models``, admin endpoints under ``/admin``, ``/healthz``, ``/readyz``, ``/metrics``,
``/dashboard``.
"""

from pgw.gateway.app import create_app
from pgw.gateway.auth import ApiKey, ApiKeyStore, RateLimiter
from pgw.gateway.config import GatewayModelConfig, GatewaySettings
from pgw.gateway.runtime import InferenceRuntime, ModelRuntime

__all__ = [
    "ApiKey", "ApiKeyStore", "GatewayModelConfig", "GatewaySettings", "InferenceRuntime",
    "ModelRuntime", "RateLimiter", "create_app",
]
