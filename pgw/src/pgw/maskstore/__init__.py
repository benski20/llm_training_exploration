"""Encrypted, single-use mask store (lives on the customer side next to the gateway).

Owner: maskstore team. Contract: SPEC.md section 7.

The factory writes immutable encrypted *segments* (``SegmentWriter``); the gateway *reserves*
slots (``MaskStore.reserve``), which advances a durable, monotonic consumption cursor BEFORE any
mask is returned, so a crash can lose masks but never reuse them.
"""

from pgw.maskstore.config import MaskStoreSettings
from pgw.maskstore.format import (
    SEGMENT_MAGIC,
    SEGMENT_SUFFIX,
    SEGMENT_VERSION,
    SegmentHeader,
    SegmentLinear,
)
from pgw.maskstore.keys import (
    EnvKeyProvider,
    FileKeyProvider,
    KeyProvider,
    KmsKeyProvider,
    Pkcs11KeyProvider,
    RecipientKeyPair,
    WrappedKey,
    hpke_wrap_for_recipient,
)
from pgw.maskstore.store import (
    Inventory,
    LowWaterListener,
    MaskReservation,
    MaskStore,
    SegmentWriter,
    SlotExtent,
)

__all__ = [
    "SEGMENT_MAGIC", "SEGMENT_SUFFIX", "SEGMENT_VERSION", "EnvKeyProvider", "FileKeyProvider",
    "Inventory", "KeyProvider", "KmsKeyProvider", "LowWaterListener", "MaskReservation", "MaskStore",
    "MaskStoreSettings", "Pkcs11KeyProvider", "RecipientKeyPair", "SegmentHeader", "SegmentLinear", "SegmentWriter",
    "SlotExtent", "WrappedKey", "hpke_wrap_for_recipient",
]
