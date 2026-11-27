"""Minimal RunPod API client (REST v1 for pods and network volumes, GraphQL for GPU prices). Stdlib only.

The API key comes from ``RUNPOD_API_KEY`` (or the constructor), is sent only in the ``Authorization`` header to
RunPod's own hosts, and never appears in a message, a ``repr`` or an exception (:meth:`RunPodAPI._redact`).
``transport`` is injectable: tests replace it with a fake and make no network call.

Field names follow RunPod's REST API v1 (``rest.runpod.io/v1``; docs.runpod.io/api-reference). They are in this
one module so a change on RunPod's side is a one-place fix; ``launch.py --dry-run`` prints the exact payloads.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

__all__ = ["CATALOG_URL", "GRAPHQL_URL", "REST_URL", "RunPodAPI", "RunPodError"]

REST_URL = "https://rest.runpod.io/v1"
CATALOG_URL = "https://api.runpod.io/v2/catalog/datacenters"
GRAPHQL_URL = "https://api.runpod.io/graphql"

#: ``transport(method, url, headers, body_bytes | None) -> (status, body_bytes)``
Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, bytes]]


class RunPodError(RuntimeError):
    """A RunPod API call failed (message without the API key)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)  # noqa: S310 - https RunPod URLs
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https RunPod URLs
            return int(r.status), r.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read() or b""


class RunPodAPI:
    """RunPod pods, network volumes and GPU prices.

    Args:
        api_key: Default ``RUNPOD_API_KEY`` from the environment.
        transport: HTTP transport (tests).
    """

    def __init__(self, api_key: str | None = None, *, transport: Transport | None = None):
        key = api_key if api_key is not None else os.environ.get("RUNPOD_API_KEY", "")
        if not key.strip():
            raise RunPodError("RUNPOD_API_KEY is not set (create one at runpod.io > Settings > API Keys and export it; "
                              "it is never printed)")
        self._key = key.strip()
        self._transport = transport or _urllib_transport

    def __repr__(self) -> str:
        return "RunPodAPI(key=<hidden>)"

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "<RUNPOD_API_KEY>") if self._key else text

    def _call(self, method: str, url: str, payload: Any = None) -> Any:
        headers = {"Authorization": f"Bearer {self._key}", "Accept": "application/json",
                   "User-Agent": "pgw-runpod-launcher"}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        try:
            status, raw = self._transport(method, url, headers, body)
        except OSError as exc:
            raise RunPodError(self._redact(f"{method} {url}: {type(exc).__name__}: {exc}")) from None
        text = raw.decode("utf-8", "replace") if raw else ""
        if not 200 <= status < 300:
            raise RunPodError(self._redact(f"{method} {url} -> HTTP {status}: {text[:500]}"), status)
        if not text.strip():
            return None
        try:
            return json.loads(text)
        except ValueError as exc:
            raise RunPodError(self._redact(f"{method} {url}: not JSON: {text[:200]}"), status) from exc

    # ---- pods ---------------------------------------------------------------------------------------
    def create_pod(self, spec: dict[str, Any]) -> dict[str, Any]:
        """``POST /pods`` (see :func:`launch.pod_spec` for the body). Returns the pod (``id``, ...)."""
        out = self._call("POST", f"{REST_URL}/pods", spec)
        if not isinstance(out, dict) or not out.get("id"):
            raise RunPodError(f"pod creation returned no id: {self._redact(json.dumps(out)[:300])}")
        return out

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        """``GET /pods/{id}``: ``desiredStatus``, ``publicIp``, ``portMappings`` ({"8443": 41234}), ``costPerHr``..."""
        out = self._call("GET", f"{REST_URL}/pods/{pod_id}")
        return out if isinstance(out, dict) else {}

    def list_pods(self) -> list[dict[str, Any]]:
        out = self._call("GET", f"{REST_URL}/pods")
        return out if isinstance(out, list) else []

    def stop_pod(self, pod_id: str) -> None:
        """``POST /pods/{id}/stop``: the GPU is released; the pod's volume disk keeps (and bills a little)."""
        self._call("POST", f"{REST_URL}/pods/{pod_id}/stop", {})

    def terminate_pod(self, pod_id: str) -> None:
        """``DELETE /pods/{id}``: gone, including its container and pod volume (a network volume stays)."""
        self._call("DELETE", f"{REST_URL}/pods/{pod_id}")

    # ---- network volumes ----------------------------------------------------------------------------
    def list_network_volumes(self) -> list[dict[str, Any]]:
        out = self._call("GET", f"{REST_URL}/networkvolumes")
        return out if isinstance(out, list) else []

    def create_network_volume(self, name: str, size_gb: int, data_center_id: str) -> dict[str, Any]:
        out = self._call("POST", f"{REST_URL}/networkvolumes",
                         {"name": name, "size": int(size_gb), "dataCenterId": data_center_id})
        if not isinstance(out, dict) or not out.get("id"):
            raise RunPodError("network volume creation returned no id")
        return out

    # ---- account (read-only) --------------------------------------------------------------------------
    def account(self) -> dict[str, Any]:
        """``{"clientBalance", "currentSpendPerHr", "spendLimit"}`` (USD; GraphQL ``myself``). Read-only."""
        out = self._call("POST", GRAPHQL_URL, {"query": "query { myself { clientBalance currentSpendPerHr "
                                                         "spendLimit } }"})
        if isinstance(out, dict) and out.get("errors"):
            raise RunPodError(self._redact(f"account query refused: {json.dumps(out['errors'])[:300]}"))
        me = ((out or {}).get("data") or {}).get("myself") or {}
        return dict(me) if isinstance(me, dict) else {}

    # ---- data centres (read-only) --------------------------------------------------------------------
    def datacenters(self, regions: str | None = None) -> list[dict[str, Any]]:
        """``GET /v2/catalog/datacenters?include=GPU_AVAILABILITY,CPU_AVAILABILITY`` (the key needs the
        ``catalog:read`` scope): every data centre with its region and GPU / CPU availability. Read-only."""
        q = "?include=GPU_AVAILABILITY,CPU_AVAILABILITY" + (f"&regions={regions}" if regions else "")
        out = self._call("GET", CATALOG_URL + q)
        if isinstance(out, dict):
            out = out.get("dataCenters") or out.get("data") or out.get("items") or []
        return [r for r in out if isinstance(r, dict)] if isinstance(out, list) else []

    # ---- prices -------------------------------------------------------------------------------------
    def gpu_prices(self) -> dict[str, dict[str, Any]]:
        """``{gpu type id: {"displayName", "memoryInGb", "securePrice", "communityPrice"}}`` (USD per GPU-hour)."""
        q = "query { gpuTypes { id displayName memoryInGb securePrice communityPrice } }"
        out = self._call("POST", GRAPHQL_URL, {"query": q})
        rows = ((out or {}).get("data") or {}).get("gpuTypes") or []
        return {r["id"]: r for r in rows if isinstance(r, dict) and r.get("id")}
