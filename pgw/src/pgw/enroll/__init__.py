"""Zero-friction onboarding: single-use enrollment codes (service side and gateway side).

* :mod:`pgw.enroll.code`: the ``PGW1-...`` code format (URL, pinned CA fingerprint, secret, checksum).
* :mod:`pgw.enroll.store`: service-side single-use records (hashed secrets, expiry, revocation).
* :mod:`pgw.enroll.service`: ``GET /pgw/v1/enroll/ca`` and ``POST /pgw/v1/enroll`` on the compute service.
* :mod:`pgw.enroll.client`: ``pgw gateway enroll`` (CA pinning, local key + CSR, credential files).
"""

from pgw.enroll.code import EnrollmentCode, EnrollmentCodeError

__all__ = ["EnrollmentCode", "EnrollmentCodeError"]
