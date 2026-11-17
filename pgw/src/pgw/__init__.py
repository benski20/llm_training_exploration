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
"""

__version__ = "0.1.0"
PROTOCOL_VERSION = 1
