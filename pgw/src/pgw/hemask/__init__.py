"""Encrypted mask generation: the compute provider produces ``W8 r`` on MLWE-encrypted masks.

Replaces the customer-side mask factory's GEMMs (and its large mask store) with a just-in-time,
single-provider protocol whose privacy rests on module-LWE. See :mod:`pgw.hemask.lwe`.
"""

from pgw.hemask.lwe import (
    DEGREE,
    HEParams,
    MaskCiphertext,
    decrypt_wr,
    encrypt_masks,
    matmul_mod64,
    public_matrix,
    secret_matrix,
)

__all__ = [
    "DEGREE",
    "HEParams",
    "MaskCiphertext",
    "decrypt_wr",
    "encrypt_masks",
    "matmul_mod64",
    "public_matrix",
    "secret_matrix",
]
