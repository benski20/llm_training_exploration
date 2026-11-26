"""Peak tensor memory of a piece of torch code, measured on CPU (test helper).

:class:`AllocPeak` is a ``TorchDispatchMode`` that sees every tensor an operator returns, counts each new
storage once (views share their base's storage) and forgets it when the last tensor object holding it is
collected. ``peak`` is the largest number of bytes alive at once among storages created inside the block.
Storages that existed before (inputs, weights) are not counted: pass them as ``exclude`` when an operator may
hand them back (``x.contiguous()`` of a contiguous ``x``, views).

This measures what the code *allocates*, so the device memory model (:mod:`pgw.service.memplan`,
:func:`pgw.core.ringmm.offset_byte_work_bytes`) can be checked without a GPU: the same torch code runs on CPU
and on CUDA with the same tensor shapes. Operator-internal scratch (cuBLAS workspaces) is not visible here; the
memory plan reserves it separately.
"""

from __future__ import annotations

import weakref
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class AllocPeak(TorchDispatchMode):
    """``with AllocPeak(exclude=[x, w]) as t: ...`` then ``t.peak`` (bytes)."""

    def __init__(self, exclude: list[torch.Tensor] | tuple[torch.Tensor, ...] = ()):
        super().__init__()
        self._skip = {int(t.untyped_storage().data_ptr()) for t in exclude}
        self._live: dict[int, list[int]] = {}  # ptr -> [nbytes, tensor objects alive]
        self.current = 0
        self.peak = 0

    def _drop(self, ptr: int) -> None:
        e = self._live.get(ptr)
        if e is None:
            return
        e[1] -= 1
        if e[1] <= 0:
            self.current -= e[0]
            del self._live[ptr]

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        out = func(*args, **(kwargs or {}))
        flat, _ = tree_flatten(out)
        for t in flat:
            if not isinstance(t, torch.Tensor):
                continue
            st = t.untyped_storage()
            ptr = int(st.data_ptr())
            if ptr == 0 or ptr in self._skip:
                continue
            e = self._live.get(ptr)
            if e is None:
                e = self._live[ptr] = [int(st.nbytes()), 0]
                self.current += e[0]
                self.peak = max(self.peak, self.current)
            e[1] += 1
            weakref.finalize(t, self._drop, ptr)
        return out
