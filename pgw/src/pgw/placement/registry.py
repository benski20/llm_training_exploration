"""Pod registration: boot -> verify pack -> load -> self-tests -> health -> register -> serve (and drain -> remove).

Pod side (:class:`PodBoot`, run by ``deploy/runpod/pod_boot.py placement``): each step is timed; any failure stops
the boot *before* registration, so a pod that did not verify its pack against the pinned fingerprint, or failed a
self-test, is never listed.

Controller side (:class:`Registry`):

1. ``expect(pod_name, dc, model)`` when the controller creates a pod: a one-time registration token
   (:class:`pgw.placement.certs.RegistrationTokens`) for the pod's environment.
2. ``register(request, public_host=...)``: the pod presents the token, its CSR, the fingerprints it serves and its
   self-test results. The controller checks the token (single use, bound to pod name / data centre / model), that
   every fingerprint is the **pinned** one for the model, that every self-test passed, and issues a short-lived leaf
   for the pod's public address **as RunPod's API reports it** (``public_host``), never an address the pod claims.
3. ``activate(pod_id, probe)``: the controller connects to the pod like a client would (TLS verified against the
   pinned root only, the issued leaf exactly, ``/readyz`` 200, the served fingerprints listed) and only then adds it
   to the directory (a new signed version).
4. ``drain(pod_id)`` marks it ``draining`` in the next directory (clients move between requests); ``remove``
   deletes it from the directory before the pod is terminated.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pgw.placement.certs import IssuedCert, PodIssuer, RegistrationTokens, TokenError

__all__ = ["BootStep", "PodBoot", "PodRecord", "RegistrationError", "Registry"]


class RegistrationError(PermissionError):
    """A registration or activation was refused (the pod is not listed)."""


@dataclass
class PodRecord:
    pod_id: str
    pod_name: str
    dc: str
    model: str
    host: str
    port: int
    lane_port: int | None
    fingerprints: tuple[str, ...]
    cert: IssuedCert
    state: str = "certified"  # certified -> serving -> draining -> removed
    registered_at: float = 0.0
    boot_timings: dict[str, float] = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"https://{self.host}:{self.port}"

    def endpoint(self, region: str) -> dict[str, Any]:
        e: dict[str, Any] = {"region": region, "url": self.url, "models": list(self.fingerprints),
                             "dc": self.dc, "state": "draining" if self.state == "draining" else "serving"}
        if self.lane_port:
            e["lane_port"] = self.lane_port
        return e


#: probe(url, chain the pod must serve) -> {"ready": bool, "fingerprints": [...], "leaf_sha256": hex}
Probe = Callable[[str, IssuedCert], Mapping[str, Any]]


class Registry:
    def __init__(self, issuer: PodIssuer, pinned: Mapping[str, str], *, tokens: RegistrationTokens | None = None,
                 on_change: Callable[[list[dict[str, Any]]], None] | None = None,
                 region_of: Callable[[str], str] | None = None, clock: Callable[[], float] = time.time):
        """``pinned``: model -> pinned pack fingerprint (a pod may also serve that pack's masked-head fingerprint,
        :func:`pgw.modelpack.masked_head.head_fingerprint`). ``on_change``: called with the new endpoint list
        whenever it changes (sign and publish it). ``region_of``: data centre -> region label."""
        self.issuer, self.pinned = issuer, dict(pinned)
        self.tokens = tokens or RegistrationTokens()
        self.on_change = on_change
        self.region_of = region_of or (lambda dc: dc.lower())
        self.clock = clock
        self.pods: dict[str, PodRecord] = {}

    def _allowed(self, model: str) -> set[str]:
        from pgw.modelpack.masked_head import head_fingerprint

        fp = self.pinned.get(model)
        if not fp:
            raise RegistrationError(f"model {model!r} has no pinned fingerprint")
        return {fp, head_fingerprint(fp)}

    def expect(self, pod_name: str, dc: str, model: str) -> str:
        self._allowed(model)
        return self.tokens.issue(pod_name, dc, model)

    def register(self, req: Mapping[str, Any], *, public_host: str, public_port: int,
                 public_lane_port: int | None) -> PodRecord:
        """Validate a pod's registration and issue its leaf (see the module docstring)."""
        try:
            pod_id, pod_name = str(req["pod_id"]), str(req["pod_name"])
            dc, model = str(req["dc"]), str(req["model"])
            fps = tuple(str(f).lower() for f in req["fingerprints"])
            tests = dict(req["selftests"])
            csr = req["csr_pem"].encode() if isinstance(req["csr_pem"], str) else bytes(req["csr_pem"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistrationError("malformed registration") from exc
        try:
            self.tokens.redeem(str(req.get("token", "")), pod_name=pod_name, dc=dc, model=model)
        except TokenError as exc:
            raise RegistrationError(str(exc)) from exc
        allowed = self._allowed(model)
        if not fps or any(f not in allowed for f in fps):
            raise RegistrationError("the pod serves a pack fingerprint that is not the pinned one")
        failed = sorted(k for k, v in tests.items() if v is not True)
        if not tests or failed:
            raise RegistrationError(f"self-tests failed or missing: {failed or 'none reported'}")
        cert = self.issuer.issue(csr, pod_id=pod_id, hosts=[public_host])
        rec = PodRecord(pod_id, pod_name, dc, model, public_host, int(public_port), public_lane_port, fps, cert,
                        registered_at=self.clock(),
                        boot_timings={str(k): float(v) for k, v in dict(req.get("boot_timings") or {}).items()})
        self.pods[pod_id] = rec
        return rec

    def activate(self, pod_id: str, probe: Probe) -> PodRecord:
        rec = self.pods.get(pod_id)
        if rec is None or rec.state != "certified":
            raise RegistrationError("no certified pod with that id")
        got = probe(rec.url, rec.cert)
        if not got.get("ready"):
            raise RegistrationError("the pod is not ready")
        if got.get("leaf_sha256") != rec.cert.sha256:
            raise RegistrationError("the pod does not serve the certificate issued to it")
        if not set(rec.fingerprints) <= {str(f).lower() for f in got.get("fingerprints") or []}:
            raise RegistrationError("the pod does not serve the fingerprints it registered")
        rec.state = "serving"
        self._changed()
        return rec

    def drain(self, pod_id: str) -> None:
        rec = self.pods[pod_id]
        if rec.state == "serving":
            rec.state = "draining"
            self._changed()

    def remove(self, pod_id: str) -> None:
        rec = self.pods.pop(pod_id, None)
        if rec is not None and rec.state in ("serving", "draining"):
            self._changed()

    def endpoints(self) -> list[dict[str, Any]]:
        return [r.endpoint(self.region_of(r.dc)) for r in sorted(self.pods.values(), key=lambda r: r.pod_id)
                if r.state in ("serving", "draining")]

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change(self.endpoints())


# ---- pod side ---------------------------------------------------------------------------------------------------
@dataclass
class BootStep:
    name: str
    run: Callable[[dict[str, Any]], Any]


class PodBoot:
    """Runs the boot steps in order, timing each; ``ctx`` carries results between steps. A failing step raises and
    no later step (in particular no registration) runs."""

    def __init__(self, steps: Sequence[BootStep], *, clock: Callable[[], float] = time.perf_counter):
        self.steps, self.clock = list(steps), clock
        self.timings: dict[str, float] = {}

    def run(self, ctx: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = ctx if ctx is not None else {}
        ctx["boot_timings"] = self.timings
        for st in self.steps:
            t0 = self.clock()
            try:
                st.run(ctx)
            finally:
                self.timings[st.name] = round(self.clock() - t0, 3)
        return ctx


def registration_request(ctx: Mapping[str, Any]) -> dict[str, Any]:
    """The body a pod POSTs to the controller (no secrets but the one-time token; never the private key)."""
    return {"token": ctx["token"], "pod_id": ctx["pod_id"], "pod_name": ctx["pod_name"], "dc": ctx["dc"],
            "model": ctx["model"], "fingerprints": list(ctx["fingerprints"]), "selftests": dict(ctx["selftests"]),
            "csr_pem": ctx["csr_pem"].decode() if isinstance(ctx["csr_pem"], bytes) else ctx["csr_pem"],
            "boot_timings": dict(ctx.get("boot_timings") or {})}
