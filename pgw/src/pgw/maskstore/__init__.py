"""Encrypted, single-use mask store (lives on the customer side next to the gateway).

Owner: maskstore team. Contract: SPEC.md section 7.

The factory writes immutable encrypted *segments* (``SegmentWriter``); the gateway *reserves*
slots (``MaskStore.reserve``), which advances a durable, monotonic consumption cursor BEFORE any
mask is returned, so a crash can lose masks but never reuse them. Rollback of the state database
is detected with an HMAC-chained generation journal and optional monotonic counters
(:mod:`pgw.maskstore.counters`).
"""

from pgw.maskstore.config import MaskStoreSettings
from pgw.maskstore.counters import (
    FileMonotonicCounter,
    GenerationJournal,
    MonotonicCounter,
    ProviderCounter,
    TpmNvCounter,
)
from pgw.maskstore.format import (
    SEGMENT_MAGIC,
    SEGMENT_SUFFIX,
    SEGMENT_VERSION,
    SegmentFile,
    SegmentHeader,
    SegmentLayout,
    SegmentLinear,
)
from pgw.maskstore.keys import (
    EnvKeyProvider,
    FileKeyProvider,
    KeyProvider,
    KmsClient,
    KmsKeyProvider,
    Pkcs11KeyProvider,
    RecipientKeyPair,
    StaticKeyProvider,
    WrappedKey,
    hpke_wrap_for_recipient,
    write_key_file,
)
from pgw.maskstore.partner import PartnerSigningKey, TrustedPartners, sign_segment
from pgw.maskstore.store import (
    IntegrityReport,
    Inventory,
    LowWaterListener,
    MaskReservation,
    MaskStore,
    SegmentWriter,
    SlotExtent,
    take_checked_many,
)

__all__ = [
    "SEGMENT_MAGIC", "SEGMENT_SUFFIX", "SEGMENT_VERSION", "EnvKeyProvider", "FileKeyProvider",
    "FileMonotonicCounter", "GenerationJournal", "IntegrityReport", "Inventory", "KeyProvider", "KmsClient",
    "KmsKeyProvider", "LowWaterListener", "MaskReservation", "MaskStore", "MaskStoreSettings", "MonotonicCounter",
    "PartnerSigningKey", "Pkcs11KeyProvider", "ProviderCounter", "RecipientKeyPair", "SegmentFile", "SegmentHeader",
    "SegmentLayout", "SegmentLinear", "SegmentWriter", "SlotExtent", "StaticKeyProvider", "TpmNvCounter",
    "TrustedPartners", "WrappedKey", "hpke_wrap_for_recipient", "sign_segment", "take_checked_many",
    "write_key_file",
]
