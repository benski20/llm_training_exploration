"""Mask factory: offline generation of one-time masks ``(seed -> r, W r)`` and Freivalds vectors.

Owner: factory team. Contract: SPEC.md section 7.4. Runs customer-side or at an independent
partner; never at the compute-service operator.
"""

from pgw.factory.config import FactorySettings
from pgw.factory.generator import GeneratedLinear, MaskGenerator
from pgw.factory.scheduler import FactoryScheduler, RefillPlan
from pgw.factory.streaming import WeightStreamer

__all__ = ["FactoryScheduler", "FactorySettings", "GeneratedLinear", "MaskGenerator", "RefillPlan", "WeightStreamer"]
