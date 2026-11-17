"""Stateless masked-compute service (operated by us; sees only uniform ring noise).

Owner: service team. Contract: SPEC.md section 10.

Holds public int8 weights for registered model packs and answers ``POST /pgw/v1/compute`` with
``W8 x_hat mod 2^K``. No KV cache, no sessions, no payload logging, no persistence of requests.
"""

from pgw.service.app import create_app
from pgw.service.auth import TenantAuthenticator, TenantIdentity
from pgw.service.batcher import BatchScheduler
from pgw.service.config import ServiceSettings
from pgw.service.kernels import ComputeKernel
from pgw.service.registry import ModelRegistry, RegisteredModel

__all__ = [
    "BatchScheduler", "ComputeKernel", "ModelRegistry", "RegisteredModel", "ServiceSettings",
    "TenantAuthenticator", "TenantIdentity", "create_app",
]
