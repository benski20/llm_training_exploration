# Packaging shortcuts (see docs/sdk/ and scripts/release_check.sh).
.PHONY: wheel release-check release-check-full notices

## Build sdist + wheel into dist/ (native extensions required).
wheel:
	PGW_REQUIRE_NATIVE=1 python -m build --outdir dist .

## Build, install into a fresh venv (CPU torch), smoke test.
release-check:
	scripts/release_check.sh

## Same, plus native bit-identity tests against the install and the no-Rust fallback build.
release-check-full:
	scripts/release_check.sh --tests --fallback

## Regenerate THIRD_PARTY_NOTICES.txt from rust/Cargo.lock.
notices:
	python scripts/third_party_notices.py
