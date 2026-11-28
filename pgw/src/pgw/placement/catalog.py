"""Data centres, metros and the RTT between them (estimates; replace with measurements as they arrive).

The RunPod data-centre ids are RunPod's (``deploy/runpod/jobs.py datacenters`` lists the live catalog with GPU
availability). Their **locations are approximate** (the metro each id is understood to be in; RunPod does not
publish street addresses): the controller only needs them to rank data centres for a metro, and the ranking is
corrected by the clients' measured RTT buckets (:mod:`pgw.placement.telemetry`) once real traffic exists.

RTT model (docs/sdk/wan.md, "Where the user is"): ``access_ms + peering_ms + km x detour / 100`` (light in fibre
covers ~200 km per ms one way, i.e. 1 ms of RTT per 100 km of path; routes run 1.5-2x the great circle).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

__all__ = [
    "ACCESS_MS",
    "DATACENTERS",
    "DEFAULT_GPU",
    "GPU_PRICE_H",
    "METROS",
    "POD_STREAMS",
    "DataCenter",
    "Metro",
    "great_circle_km",
    "ranked",
    "rtt_ms",
]

#: Secure Cloud list prices, USD per GPU-hour: the same table as ``deploy/runpod/launch.GPU_PRICE_H`` (a test keeps
#: them equal); the controller uses the live API price when it has one.
GPU_PRICE_H: dict[str, float] = {
    "NVIDIA H100 80GB HBM3": 2.99, "NVIDIA H100 PCIe": 2.39, "NVIDIA H100 NVL": 2.79, "NVIDIA A100 80GB PCIe": 1.64,
    "NVIDIA A100-SXM4-80GB": 1.74, "NVIDIA L40S": 0.86, "NVIDIA L40": 0.99, "NVIDIA RTX A6000": 0.49,
    "NVIDIA A40": 0.40, "NVIDIA GeForce RTX 4090": 0.69,
}
DEFAULT_GPU = "NVIDIA H100 80GB HBM3"
#: Concurrent decode streams one pod serves at full per-stream speed (estimate, not measured: decode batches the
#: streams' rows through one weight read, so the GPU is not the bound; pks-v3 maskgen is: ~450 / 135 / 81 slots/s
#: at 7B / 32B / 72B estimated on an H100, at ~1.2 slots per token and ~20 tok/s per stream near the GPU).
POD_STREAMS: dict[str, int] = {"7B": 16, "32B": 5, "72B": 3}


@dataclass(frozen=True)
class DataCenter:
    id: str
    city: str
    lat: float
    lon: float
    gpus: frozenset[str] = frozenset({DEFAULT_GPU})  # GPU types listed there (the live catalog overrides)


@dataclass(frozen=True)
class Metro:
    name: str
    lat: float
    lon: float
    #: share of the user population by access class (fibre / cable / mobile / in-cloud server): sets access_ms
    access_mix: Mapping[str, float] = field(default_factory=lambda: {"fibre": 1.0})
    utc_offset_h: float = 0.0


ACCESS_MS = {"in-cloud": 0.3, "fibre": 2.0, "cable": 10.0, "mobile": 25.0}

DATACENTERS: dict[str, DataCenter] = {d.id: d for d in [
    DataCenter("US-KS-2", "Kansas City", 39.10, -94.58),
    DataCenter("US-TX-3", "Dallas", 32.78, -96.80),
    DataCenter("US-GA-1", "Atlanta", 33.75, -84.39),
    DataCenter("US-GA-2", "Atlanta", 33.75, -84.39),
    DataCenter("US-IL-1", "Chicago", 41.88, -87.63),
    DataCenter("US-DE-1", "Wilmington", 39.74, -75.55),
    DataCenter("US-CA-2", "San Jose", 37.34, -121.89),
    DataCenter("US-WA-1", "Seattle", 47.61, -122.33),
    DataCenter("CA-MTL-3", "Montreal", 45.50, -73.57),
    DataCenter("EU-NL-1", "Amsterdam", 52.37, 4.90),
    DataCenter("EU-SE-1", "Stockholm", 59.33, 18.07),
    DataCenter("EU-CZ-1", "Prague", 50.08, 14.44),
    DataCenter("EU-RO-1", "Bucharest", 44.43, 26.10),
    DataCenter("EU-FR-1", "Paris", 48.86, 2.35),
    DataCenter("AP-JP-1", "Tokyo", 35.68, 139.69),
    DataCenter("OC-AU-1", "Sydney", -33.87, 151.21),
]}

METROS: dict[str, Metro] = {m.name: m for m in [
    Metro("new-york", 40.71, -74.01, {"in-cloud": 0.3, "fibre": 0.4, "cable": 0.3}, -5),
    Metro("san-francisco", 37.77, -122.42, {"in-cloud": 0.3, "fibre": 0.4, "cable": 0.3}, -8),
    Metro("chicago", 41.88, -87.63, {"in-cloud": 0.2, "fibre": 0.4, "cable": 0.4}, -6),
    Metro("dallas", 32.78, -96.80, {"in-cloud": 0.2, "fibre": 0.4, "cable": 0.4}, -6),
    Metro("london", 51.51, -0.13, {"in-cloud": 0.3, "fibre": 0.5, "cable": 0.2}, 0),
    Metro("frankfurt", 50.11, 8.68, {"in-cloud": 0.4, "fibre": 0.4, "cable": 0.2}, 1),
    Metro("stockholm", 59.33, 18.07, {"in-cloud": 0.2, "fibre": 0.7, "cable": 0.1}, 1),
    Metro("tokyo", 35.68, 139.69, {"in-cloud": 0.3, "fibre": 0.6, "cable": 0.1}, 9),
]}


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, a)))


def rtt_ms(metro: Metro, dc: DataCenter, *, access: str = "fibre", detour: float = 1.6,
           peering_ms: float = 0.5) -> float:
    """Estimated idle RTT from a user of ``access`` class in ``metro`` to ``dc``."""
    km = great_circle_km(metro.lat, metro.lon, dc.lat, dc.lon)
    return ACCESS_MS[access] + peering_ms + km * detour / 100.0


def ranked(metro: Metro, dcs: Iterable[DataCenter], **kw: object) -> list[tuple[float, DataCenter]]:
    """Data centres by estimated RTT from ``metro`` (nearest first)."""
    return sorted(((rtt_ms(metro, d, **kw), d) for d in dcs), key=lambda t: (t[0], t[1].id))  # type: ignore[arg-type]
