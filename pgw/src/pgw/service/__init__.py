"""Stateless masked-compute service (operated by us; sees only uniform ring noise).

Owner: service team. Contract: SPEC.md section 10.

Holds public int8 weights for registered model packs and answers ``POST /pgw/v1/compute`` with
``W8 x_hat mod 2^K``. No KV cache, no sessions, no payload logging, no persistence of requests.

Modules:

* :mod:`~pgw.service.kernels` - exact ring GEMM per linear (``cuda_int8``, ``cpu_int8``, ``cpu_f64``, ``cpu_f32``).
* :mod:`~pgw.service.registry` - packs by fingerprint, multi-GPU placement by layer.
* :mod:`~pgw.service.tensor_parallel` - opt-in output-row tensor parallel kernels (one linear over N GPUs).
* :mod:`~pgw.service.persistent` - opt-in persistent GPU trip worker (descriptor ring, no launch per trip).
* :mod:`~pgw.service.batcher` - cross-request/tenant dynamic batching per ``(fp, ring, linear)``.
* :mod:`~pgw.service.auth` - bearer-token (SHA-256 hashed) and mTLS tenant identity.
* :mod:`~pgw.service.handler` - admission control, quotas, the wire ``ComputeHandler``.
* :mod:`~pgw.service.app` / :mod:`~pgw.service.server` - ASGI app, hypercorn runner, graceful drain.
* :mod:`~pgw.service.loadtest` - load generator (``python -m pgw.service.loadtest``).
"""

from pgw.service.app import ServiceState, create_app, create_metrics_app
from pgw.service.auth import TenantAuthenticator, TenantIdentity, hash_token
from pgw.service.batcher import BatchScheduler, BatchStats, BatchTiming
from pgw.service.config import ServiceSettings
from pgw.service.handler import ServiceHandler, TokenBucket
from pgw.service.kernels import ComputeKernel, cpu_int8_available
from pgw.service.metrics import ServiceMetrics
from pgw.service.registry import ModelRegistry, RegisteredModel, place_layers
from pgw.service.server import build_hypercorn_config, serve, serve_async
from pgw.service.tensor_parallel import ShardedKernel, TPGroup

__all__ = [
    "BatchScheduler", "BatchStats", "BatchTiming", "ComputeKernel", "ModelRegistry", "RegisteredModel",
    "ServiceHandler", "ServiceMetrics", "ServiceSettings", "ServiceState", "ShardedKernel", "TPGroup",
    "TenantAuthenticator",
    "TenantIdentity", "TokenBucket", "build_hypercorn_config", "cpu_int8_available", "create_app",
    "create_metrics_app", "hash_token", "place_layers", "serve", "serve_async",
]
