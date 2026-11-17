"""Mask factory: offline generation of one-time masks ``(seed -> r, W r)`` and Freivalds vectors.

Owner: factory team. Contract: SPEC.md section 7.4. Runs customer-side or at an independent
partner; never at the compute-service operator.

Typical use::

    pack = ModelPack.open("packs/qwen0.5b", components=("linears",))
    gen = MaskGenerator(pack, "K40A12", device="auto")
    seg_id = gen.generate_into_store(store, slots=4096)

or, as a service, ``pgw factory run`` (:class:`FactoryScheduler`).
"""

from pgw.factory.config import FactorySettings
from pgw.factory.generator import GeneratedLinear, GenerationStats, MaskGenerator, slot_bytes
from pgw.factory.governor import FactoryStopped, Throttle, apply_process_priority
from pgw.factory.kernels import RingKernel, cpu_int8_available, resolve_backend, ring_vecmat
from pgw.factory.scheduler import FactoryScheduler, FactoryState, RefillPlan
from pgw.factory.sinks import ExportSink, RecipientWrapProvider, SegmentSink, StoreSink
from pgw.factory.streaming import WeightStreamer

__all__ = [
    "ExportSink", "FactoryScheduler", "FactorySettings", "FactoryState", "FactoryStopped", "GeneratedLinear",
    "GenerationStats", "MaskGenerator", "RecipientWrapProvider", "RefillPlan", "RingKernel", "SegmentSink",
    "StoreSink", "Throttle", "WeightStreamer", "apply_process_priority", "cpu_int8_available", "resolve_backend",
    "ring_vecmat", "slot_bytes",
]
