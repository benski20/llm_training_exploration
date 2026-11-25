"""PGW (Private Gateway): private LLM inference over public open weights with one-time masks.

Subpackages:
    core          ring arithmetic, exact ring matmul, mask math, Freivalds, CSPRNG, shared types/errors
    modelpack     HF checkpoint -> int8 model pack + manifest/fingerprint; loader
    engine        client-side transformer runtime with pluggable linear backends
    factory       offline mask generation (W r, Freivalds vectors)
    maskstore     encrypted single-use mask store
    wire          protocol messages, codec, HTTP/2 client and server helpers
    service       stateless masked-compute server
    gateway       OpenAI-compatible customer-side API
    observability metrics, outbound-noise monitor, dashboard
    cli           ``pgw`` command line
    sdk           in-process client library: ``from pgw import OpenAI, AsyncOpenAI``
"""

__version__ = "0.1.0"
PROTOCOL_VERSION = 1

_SDK_EXPORTS = ("OpenAI", "AsyncOpenAI")


def __getattr__(name: str) -> object:
    """Lazy ``pgw.OpenAI`` / ``pgw.AsyncOpenAI``: ``import pgw`` stays light (nothing heavy is imported)."""
    if name in _SDK_EXPORTS:
        from pgw import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module 'pgw' has no attribute {name!r}")
