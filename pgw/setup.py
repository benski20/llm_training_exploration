"""Build hook for the native extensions; all other metadata lives in pyproject.toml.

The wheel carries three Rust extensions (top-level modules, imported by ``pgw.core.native``,
``pgw.hemask.ntt`` and ``pgw.hemask.pks_native``):

    pgw_fast   gateway/service hot path (rust/pgw-fast)
    pgw_ntt    negacyclic NTT products of the encrypted mask factory (rust/pgw-ntt)
    pgw_pks    pks-v3 compact-hint kernels (rust/pgw-pks)

They are built against the stable ABI (abi3, CPython >= 3.11), so one wheel per platform serves
Python 3.11-3.13. Every native function is bit-identical to its Python fallback; the extensions only
change speed.

Environment:
    PGW_REQUIRE_NATIVE=1   a failed extension build fails the build (CI and release wheels).
                           Default: optional, so ``pip install`` from an sdist without a Rust
                           toolchain still installs (pure-Python path, much slower).
"""

from __future__ import annotations

import os

from setuptools import setup
from setuptools_rust import Binding, RustExtension, Strip

REQUIRE_NATIVE = os.environ.get("PGW_REQUIRE_NATIVE", "0").strip().lower() in ("1", "true", "yes", "on")

NATIVE = [
    ("pgw_fast", "rust/pgw-fast/Cargo.toml"),
    ("pgw_ntt", "rust/pgw-ntt/Cargo.toml"),
    ("pgw_pks", "rust/pgw-pks/Cargo.toml"),
]

setup(
    rust_extensions=[
        RustExtension(
            target,
            path=path,
            binding=Binding.PyO3,
            py_limited_api="auto",
            optional=not REQUIRE_NATIVE,
            # Release profile from rust/Cargo.toml (overflow checks on, no debug info).
            debug=False,
            strip=Strip.Debug,
        )
        for target, path in NATIVE
    ],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
    zip_safe=False,
)
