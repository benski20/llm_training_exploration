"""Gateway wiring of ``schedule = "fused2"`` (opt-in, default off): weights, lane ticket, provider.

The gateway service runs fused2 (``research/plans/fused2-two-trips-per-layer.md``) exactly like the harness
(``benchmarks/fused2_e2e.py --masks store``), through its production pieces:

* **public weights** (:func:`load_fused2_weights`): derived at start-up from the verified pack (the int8 fold with
  pack-time migration multipliers, ``fused2_migrate_bits``; the same derivation as the service's
  ``PGW_SERVICE_FUSED2_FROM_PACK``) or loaded from a ``Fused2Weights.save()`` file whose base fingerprint is the
  pack's. The gateway computes the **fused2 lane fingerprint** from them (:func:`pgw.service.fused2_lane.
  fused2_fingerprint`: base pack, ring, every item's ``w8`` and ``hmul``), then keeps only the weight-free tables
  (:func:`pgw.engine.fused2_masks.gateway_tables`: scales, biases, bounds, multipliers; no int8 matrix);
* **lane ticket** (:func:`open_fused2_ticket`): ``POST /pgw/v1/lane/open`` for the fused2 fingerprint over the
  model's authenticated compute client (the same route, authentication and ticket as every lane model). The
  reply is checked against the gateway's own public tables: version, transport, ticket length, the fingerprint
  (a service serving other weights is refused), the item table in pass order, the ring, the row limit; a UDS
  socket only from a service on this host or inside ``lane_uds_dir``, a TCP port only from the egress allowlist;
* **provider** (:func:`connect_fused2_lane`): the native ``pgw_fast.TripChannel`` over the service's Unix socket
  (``PGW_LANE_SHM=1``: the shared-memory transport; ``PGW_LANE_SPIN_AUTO_US``: adaptive spin) or TLS 1.3 to the
  service host (``lane = "tls"``, the service verified against ``service_ca_file``), wrapped in
  :class:`pgw.engine.fused2_lane.LaneFused2Provider`.

Masks: the mask store's fused2 pair-mask segments (``schedule = "fused2"``) under the fused2 lane fingerprint,
reserved per request like today's masks (window or incremental; ``store.reserve(..., schedule="fused2")``). The
provider sees uniformly masked ring elements of public shapes only; every output of both stages of every trip is
range- and Freivalds-checked (``F_p``, ``p = 2^61 - 1``) before use (:class:`pgw.engine.fused2.Fused2MaskedBackend`
and the native driver).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from types import SimpleNamespace
from typing import Any

from pgw.core.errors import ConfigError, ProtocolError, ServiceUnavailableError

__all__ = ["connect_fused2_lane", "fused2_item_names", "load_fused2_weights", "open_fused2_ticket"]

log = logging.getLogger(__name__)

_TICKET_LEN = 32
_MAX_TICKET_BODY = 1 << 20


def load_fused2_weights(pack: Any, config: Any) -> Any:
    """The public fused2 weights of ``pack`` (full: ``w8`` included; the caller reduces them to gateway tables
    after computing the lane fingerprint)."""
    from pgw.engine.fused2 import Fused2Weights, migration_multipliers

    # head_mode = "provider_int8" with fused2: the LM head folded into the last trip (item table v2)
    head = getattr(config, "head_mode", "gateway") == "provider_int8"
    path = getattr(config, "fused2_weights_path", None)
    if path is not None:
        w = Fused2Weights.load(path)
        if str(getattr(w, "fingerprint", "")) != str(pack.fingerprint):
            raise ConfigError("fused2_weights_path holds weights of another pack (base fingerprint mismatch)")
        if any(float(lin.scale.abs().sum()) == 0.0 for lin in w.all_linears()):
            raise ConfigError("fused2_weights_path is a provider-only file (no scales): the gateway needs the full "
                              "public weights (Fused2Weights.save(path) without provider_only)")
        if int(w.num_layers) != int(pack.arch.num_layers):
            raise ConfigError("fused2_weights_path does not match the pack's layer count")
        if (w.head is not None) != head:
            raise ConfigError("fused2_weights_path: the folded head must be present exactly with head_mode = "
                              "'provider_int8' (head fold)")
        return w
    bits = int(getattr(config, "fused2_migrate_bits", 8))
    t0 = time.perf_counter()
    w = Fused2Weights(pack, None, multipliers=migration_multipliers(pack, bits, head=head) if bits else None,
                      head=head)
    log.info("fused2 weights derived from pack %s (migration %d bits) in %.1f s", str(pack.fingerprint)[:12], bits,
             time.perf_counter() - t0)
    return w


def fused2_item_names(weights: Any) -> list[str]:
    """The lane item table of ``weights`` in pass order (what a ticket must list)."""
    from pgw.service.fused2_lane import fused2_items

    return [it.name for it in fused2_items(weights)]


def _check_ticket(raw: bytes, fingerprint: str, ring: Any, names: list[str], transports: tuple[str, ...],
                  items_version: int = 1) -> dict[str, Any]:
    try:
        d = json.loads(raw)
    except ValueError as exc:
        raise ProtocolError("fused2 lane ticket is not JSON") from exc
    if not isinstance(d, dict) or int(d.get("version", 0)) != 1 or d.get("transport") not in transports:
        raise ProtocolError("unsupported fused2 lane ticket (version or transport)")
    try:
        secret = bytes.fromhex(str(d["ticket"]))
        max_rows = int(d["max_rows"])
        linears = [str(x) for x in d["linears"]]
        rings = [str(x) for x in d["rings"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"fused2 lane ticket is malformed ({type(exc).__name__})") from exc
    if len(secret) != _TICKET_LEN:
        raise ProtocolError("fused2 lane ticket is malformed")
    if d.get("fingerprint") != fingerprint:
        # the lane fingerprint binds every public weight and multiplier: another value means other weights
        raise ConfigError("the service serves another fused2 weight set (lane fingerprint mismatch): check "
                          "PGW_SERVICE_FUSED2_FROM_PACK / fused2_migrate_bits and the ring",
                          details={"expected": fingerprint, "served": str(d.get("fingerprint"))})
    if linears != names:
        raise ProtocolError("fused2 lane ticket: the service's item table differs from the gateway's")
    if int(d.get("fused2_items", 1)) != int(items_version):
        raise ProtocolError(f"fused2 lane ticket: item table version {d.get('fused2_items', 1)} != {items_version}")
    if rings != [ring.name]:
        raise ProtocolError(f"fused2 lane ticket: rings {rings} != [{ring.name}]")
    if not 1 <= max_rows <= 1024:
        raise ProtocolError("fused2 lane ticket: bad max_rows")
    return {"transport": d["transport"], "socket": d.get("socket"), "port": d.get("port"),
            "alpn": d.get("alpn", "pgw-lane/1"), "secret": bytearray(secret), "max_rows": max_rows,
            "linears": linears,
            # lane format 2 (multi frames: several items in one call) when the service offers it
            "max_items": int(d.get("max_items", 0) or 0) if 2 in (d.get("versions") or ()) else 0,
            "max_multi_rows": int(d.get("max_multi_rows", 0) or 0)}


async def open_fused2_ticket(client: Any, fingerprint: str, ring: Any, weights: Any, *,
                             transports: tuple[str, ...] = ("uds",)) -> dict[str, Any]:
    """A lane ticket for the fused2 lane model ``fingerprint`` (one attempt, authenticated like a compute call).

    ``client``: the model's connected :class:`pgw.wire.client.ComputeClient`. Raises :class:`ConfigError` for a
    service serving other weights (permanent), :class:`ServiceUnavailableError` / :class:`ProtocolError` else."""
    from pgw.wire.service_lane import SERVICE_LANE_OPEN_PATH

    call = getattr(client, "_call", None)
    if not callable(call):
        raise ConfigError("fused2 needs the HTTP compute client (lane/open)")
    names = fused2_item_names(weights)
    rid = uuid.uuid4().hex
    deadline = time.monotonic() + float(getattr(getattr(client, "config", None), "request_timeout_s", 60.0))
    raw: bytes = await call("POST", SERVICE_LANE_OPEN_PATH, body=json.dumps({"fingerprint": fingerprint}).encode(),
                            request_id=rid, deadline=deadline, decode=lambda b: bytes(b),
                            max_response_bytes=_MAX_TICKET_BODY, max_retries=0)
    from pgw.service.fused2_lane import items_version

    return _check_ticket(raw, fingerprint, ring, names, transports, items_version(weights))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return min(100_000, max(0, int(raw))) if raw else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def connect_fused2_lane(ticket: dict[str, Any], weights: Any, ring: Any, *, client_config: Any, base_url: str,
                        uds_dir: str | None = None, tcp_ports: Any = None, timeout_s: float = 60.0) -> Any:
    """Connect the native lane channel of ``ticket`` and wrap it as the fused2 provider.

    UDS: the socket must be inside ``uds_dir`` or (without one) belong to a service on a loopback URL; the
    shared-memory transport with ``PGW_LANE_SHM=1`` (refused by the service -> error). TCP: TLS 1.3 to the host
    of ``base_url`` on the ticket's port, which must be in ``tcp_ports`` (the egress allowlist), the server
    verified against ``client_config.ca_file``. The ticket secret is wiped after use."""
    from pgw.core.native import fast
    from pgw.engine.fused2_lane import LaneFused2Provider, items_of
    from pgw.wire.lane import _uds_allowed

    if fast is None or not hasattr(fast, "TripChannel"):
        raise ConfigError("schedule = 'fused2' needs the pgw_fast extension with the native lane client")
    spin = _env_int("PGW_LANE_SPIN_US", 50)
    secret: bytearray = ticket["secret"]
    max_rows = int(ticket["max_rows"])
    try:
        if ticket["transport"] == "tcp":
            from pgw.wire.service_lane import _connect_tls

            port = int(ticket["port"] or 0)
            if not 0 < port < 65536 or ticket.get("alpn") != "pgw-lane/1":
                raise ProtocolError("fused2 lane ticket names a bad port or protocol")
            if tcp_ports is not None and port not in set(tcp_ports):
                raise ProtocolError(f"fused2 lane ticket names port {port}, which is not on the gateway's egress "
                                    "allowlist for the service host (add https://<host>:<port>)")
            ch = _connect_tls(fast, client_config, SimpleNamespace(port=port, max_rows=max_rows), secret,
                              float(timeout_s), spin)
        else:
            path = str(ticket["socket"] or "")
            if not _uds_allowed(path, base_url, uds_dir):
                raise ProtocolError("fused2 UDS lane ticket from a remote service or outside lane_uds_dir")
            ch = fast.TripChannel.connect_service_uds(path, bytes(secret), float(timeout_s), max_rows, spin)
            if os.environ.get("PGW_LANE_SHM", "0").strip().lower() in ("1", "true", "yes", "on"):
                if not ch.attach_shm(float(timeout_s)):
                    ch.close()
                    raise ServiceUnavailableError("fused2 lane: the service refused the shared-memory transport")
        auto = _env_int("PGW_LANE_SPIN_AUTO_US", 0)
        if auto and hasattr(ch, "set_spin_auto"):
            ch.set_spin_auto(auto)
    finally:
        secret[:] = bytes(len(secret))
    from pgw.engine.fused2_lane import single_multipliers

    return LaneFused2Provider(ch, ticket["linears"], items_of(weights), ring.k, max_rows=max_rows,
                              timeout_s=float(timeout_s), hmul=single_multipliers(weights),
                              max_items=int(ticket.get("max_items", 0) or 0) if ticket["transport"] == "tcp"
                              or not ch_is_shm(ch) else 0,
                              max_multi_rows=int(ticket.get("max_multi_rows", 0) or 0))


def ch_is_shm(ch: Any) -> bool:
    """Whether a native lane channel moved to the shared-memory transport (multi frames need the socket)."""
    try:
        return str(ch.transport()) == "shm" if callable(getattr(ch, "transport", None)) else False
    except Exception:  # pragma: no cover - defensive
        return True
