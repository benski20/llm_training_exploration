"""Client-side transformer runtime (runs inside the gateway, sees plaintext).

Owner: engine team. Contract: SPEC.md section 9.

The engine executes embeddings, RMSNorm, RoPE, attention with KV cache, SiLU*up, final norm, LM
head and sampling in plaintext, and delegates every one of the ``4 * num_layers`` linear groups to
a pluggable :class:`LinearBackend`:

* :class:`PlainBackend`  - exact quantized integer product locally (reference / dev mode).
  Also exported as :data:`QuantPlainBackend`.
* :class:`FloatBackend`  - dequantized float weights, no activation quantization (quality baseline).
* :class:`MaskedBackend` - one-time-masked remote compute via ``pgw.wire`` + ``pgw.maskstore``.

``PlainBackend`` and ``MaskedBackend`` must produce bit-identical logits for the same pack and ring.

Typical gateway wiring::

    pack = ModelPack.open(path)
    backend = MaskedBackend(pack, None, client, ring, pack.fingerprint)
    model = TransformerModel(pack, backend)
    gen = Generator(model, ChatTokenizer.from_pack(pack),
                    session_factory=lambda n: store.reserve(pack.fingerprint, ring, n))
    async for ev in gen.astream(ids, SamplingParams(max_tokens=64)):
        ...

For development without a service or mask store, :class:`LocalComputeClient` and
:class:`LocalMaskSource` provide in-process equivalents, and :class:`InMemoryPack` builds a pack
directly from a Hugging Face checkpoint.
"""

from pgw.engine.backends import (
    BackendStats,
    FloatBackend,
    LinearBackend,
    LinearCall,
    MaskedBackend,
    PlainBackend,
    QuantPlainBackend,
    exact_int_matmul,
)
from pgw.engine.generate import FinishReason, GenerationResult, Generator, SessionFactory, TokenEvent
from pgw.engine.kv_cache import ContextOverflowError, KVCache
from pgw.engine.local import LocalComputeClient, LocalMaskReservation, LocalMaskSource
from pgw.engine.model import SequenceInput, TransformerModel, rms_norm
from pgw.engine.pack import EnginePack, InMemoryPack, arch_config_from_hf, linear_split_sizes
from pgw.engine.rope import RotaryEmbedding, rope_inv_freq
from pgw.engine.sampling import Sampler, SamplingParams
from pgw.engine.tokenizer import ChatTokenizer, IncrementalDecoder

__all__ = [
    "BackendStats", "ChatTokenizer", "ContextOverflowError", "EnginePack", "FinishReason", "FloatBackend",
    "GenerationResult", "Generator", "InMemoryPack", "IncrementalDecoder", "KVCache", "LinearBackend",
    "LinearCall", "LocalComputeClient", "LocalMaskReservation", "LocalMaskSource", "MaskedBackend",
    "PlainBackend", "QuantPlainBackend", "RotaryEmbedding", "Sampler", "SamplingParams", "SequenceInput",
    "SessionFactory", "TokenEvent", "TransformerModel", "arch_config_from_hf", "exact_int_matmul",
    "linear_split_sizes", "rms_norm", "rope_inv_freq",
]
