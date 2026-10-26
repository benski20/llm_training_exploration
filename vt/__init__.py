"""vt: train small models that can be formally verified.

    from vt import CegisConfig, train_until_proved, DyckSpec, verify_global
    model, report = train_until_proved(DyckSpec(), CegisConfig(L=16, ternary=False))
    P = model.export()
    assert verify_global(P, DyckSpec(), workers=4)["status"] == "proved"
"""
__version__ = "0.1.0"

from .artifact import load_artifact, make_certificate, save_artifact  # noqa: F401
from .global_verify import SPECS, DyckSpec, SortedSpec, exhaustive_check, verify_global  # noqa: F401
from .verify import np_forward, np_forward_batch, verify  # noqa: F401


_LAZY = {"CegisConfig": ("vt.cegis", "CegisConfig"), "train_until_proved": ("vt.cegis", "cegis"),
         "MixerNet": ("vt.model", "MixerNet"), "GPT": ("vt.lm", "GPT"), "GPTConfig": ("vt.lm", "GPTConfig")}


def __getattr__(name):
    # torch-dependent pieces load lazily so verification works without torch
    if name in _LAZY:
        import importlib
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(name)
