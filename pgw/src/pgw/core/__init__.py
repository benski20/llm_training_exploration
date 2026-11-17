"""Shared core of PGW: ring arithmetic, exact ring matmul, one-time-mask math, Freivalds checks,
CSPRNG seed expansion, packing, fingerprints, shared types and errors.

This package is fully implemented and stable; other subpackages build on it. Changes to anything
exported here require a note in ``INTERFACE_NOTES.md`` and a SPEC update.
"""

from pgw.core.errors import (
    AuthError,
    BackendUnavailableError,
    ConfigError,
    ErrorCode,
    FingerprintMismatchError,
    IntegrityError,
    MaskExhaustedError,
    MaskReuseError,
    MaskStoreError,
    ModelNotFoundError,
    PayloadTooLargeError,
    PGWError,
    ProtocolError,
    RateLimitedError,
    RingOverflowError,
    ServiceUnavailableError,
    ShapeError,
    VersionMismatchError,
    error_from_code,
)
from pgw.core.fingerprint import (
    canonical_json,
    compute_fingerprint,
    require_same_fingerprint,
    sha256_file,
    short_fingerprint,
    validate_fingerprint,
)
from pgw.core.freivalds import freivalds_check, freivalds_precompute, freivalds_verify, soundness_bits
from pgw.core.masking import check_output_range, mask_activations, unmask_outputs
from pgw.core.packing import pack_ring, packed_size, unpack_ring
from pgw.core.prng import SEED_BYTES, SeedExpander, new_seed
from pgw.core.ring import (
    DEFAULT_RING,
    RING_K32_A8,
    RING_K40_A12,
    RING_K48_A20,
    RingParams,
    balanced_digits,
    center,
    dequantize_outputs,
    quantize_activations,
    quantize_weight_rows,
    ring_add,
    ring_sub,
    to_ring,
)
from pgw.core.ringmm import (
    PreparedWeight,
    cpu_int8_available,
    cuda_int8_available,
    int8_ring_matmul,
    ringmm,
    ringmm_reference,
    select_backend,
)
from pgw.core.types import LINEAR_ORDER, LinearId, LinearKind, LinearSpec, ModelArch, linear_ids

__all__ = [
    "DEFAULT_RING", "LINEAR_ORDER", "RING_K32_A8", "RING_K40_A12", "RING_K48_A20", "SEED_BYTES",
    "AuthError", "BackendUnavailableError", "ConfigError", "ErrorCode", "FingerprintMismatchError",
    "IntegrityError", "LinearId", "LinearKind", "LinearSpec", "MaskExhaustedError", "MaskReuseError",
    "MaskStoreError", "ModelArch", "ModelNotFoundError", "PGWError", "PayloadTooLargeError",
    "PreparedWeight", "ProtocolError", "RateLimitedError", "RingOverflowError", "RingParams",
    "SeedExpander", "ServiceUnavailableError", "ShapeError", "VersionMismatchError",
    "balanced_digits", "canonical_json", "center", "check_output_range", "compute_fingerprint",
    "cpu_int8_available", "cuda_int8_available", "dequantize_outputs", "error_from_code", "freivalds_check",
    "freivalds_precompute", "freivalds_verify", "int8_ring_matmul", "linear_ids", "mask_activations", "new_seed",
    "pack_ring", "packed_size", "quantize_activations", "quantize_weight_rows",
    "require_same_fingerprint", "ring_add", "ring_sub", "ringmm", "ringmm_reference",
    "select_backend", "sha256_file", "short_fingerprint", "soundness_bits", "to_ring",
    "unmask_outputs", "unpack_ring", "validate_fingerprint",
]
