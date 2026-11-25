#!/usr/bin/env bash
# Release check: build the sdist and a wheel (from the sdist, native extensions required), repair it to a
# manylinux tag (Linux), install it into a FRESH venv with the CPU build of torch, and run the smoke test.
#
#   scripts/release_check.sh                 build + install + smoke test (python3)
#   PYTHON=python3.11 scripts/release_check.sh
#   scripts/release_check.sh --tests         ... and run the native bit-identity tests against the install
#   scripts/release_check.sh --fallback      ... and check an sdist build WITHOUT Rust (pure-Python wheel)
#
# Environment:
#   PYTHON          interpreter for the fresh venv (default python3)
#   WORK            scratch directory (default build/release-check; the wheelhouse/ is left there)
#   TORCH_INDEX     extra index for the CPU torch build (default https://download.pytorch.org/whl/cpu on Linux)
#   PGW_ZIG=0       on Linux, do not use cargo-zigbuild (the wheel then gets the host's glibc tag)
#   MANYLINUX       manylinux tag to target with zig (default manylinux_2_28)
#
# Output: dist files in $WORK/wheelhouse, sizes, and SMOKE TEST PASSED. Exit code 0 = releasable.
# Needs: a Rust toolchain (cargo >= 1.89), network access to PyPI (and the PyTorch CPU index).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RUN_TESTS=0
FALLBACK=0
for a in "$@"; do
    case "$a" in
        --tests) RUN_TESTS=1 ;;
        --fallback) FALLBACK=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown option $a" >&2; exit 2 ;;
    esac
done
WORK="${WORK:-$ROOT/build/release-check}"
mkdir -p "$WORK"
OS="$(uname -s)"
if [ "$OS" = "Linux" ]; then TORCH_INDEX="${TORCH_INDEX-https://download.pytorch.org/whl/cpu}"; else TORCH_INDEX="${TORCH_INDEX-}"; fi
MANYLINUX="${MANYLINUX:-manylinux_2_28}"
say() { printf '\n== %s\n' "$*" >&2; }

say "tools ($WORK/buildenv)"
"$PYTHON" -m venv "$WORK/buildenv"
B="$WORK/buildenv/bin"
"$B/pip" install -q --disable-pip-version-check build "setuptools>=77" wheel "setuptools-rust>=1.9" twine
USE_ZIG=0
if [ "$OS" = "Linux" ]; then
    "$B/pip" install -q --disable-pip-version-check auditwheel patchelf
    if [ "${PGW_ZIG:-1}" = "1" ] && "$B/pip" install -q --disable-pip-version-check ziglang cargo-zigbuild; then
        USE_ZIG=1
    fi
fi

say "third-party notices"
"$B/python" "$ROOT/scripts/third_party_notices.py" --check

say "sdist"
rm -rf "$WORK/dist" "$WORK/wheelhouse" "$WORK/src"
"$B/python" -m build --sdist --outdir "$WORK/dist" "$ROOT" >"$WORK/sdist.log" 2>&1 || { tail -40 "$WORK/sdist.log"; exit 1; }
SDIST="$(ls "$WORK"/dist/*.tar.gz)"

say "wheel from the sdist (native extensions required)"
mkdir -p "$WORK/src" && tar -xzf "$SDIST" -C "$WORK/src"
SRC="$(ls -d "$WORK"/src/*/)"
if [ "$USE_ZIG" = 1 ]; then
    ARCH="$(uname -m)"
    GLIBC="${MANYLINUX#manylinux_}"; GLIBC="${GLIBC/_/.}"
    (cd "$SRC" && PATH="$B:$PATH" PGW_REQUIRE_NATIVE=1 CARGO=cargo-zigbuild \
        CARGO_BUILD_TARGET="$ARCH-unknown-linux-gnu.$GLIBC" \
        "$B/python" -m build --wheel --no-isolation --outdir "$WORK/dist" . >"$WORK/build.log" 2>&1) \
        || { tail -40 "$WORK/build.log"; exit 1; }
else
    (cd "$SRC" && PGW_REQUIRE_NATIVE=1 "$B/python" -m build --wheel --no-isolation --outdir "$WORK/dist" . \
        >"$WORK/build.log" 2>&1) || { tail -40 "$WORK/build.log"; exit 1; }
fi
mkdir -p "$WORK/wheelhouse"
cp "$SDIST" "$WORK/wheelhouse/"
if [ "$OS" = "Linux" ]; then
    PLAT="$MANYLINUX"_"$(uname -m)"
    [ "$USE_ZIG" = 1 ] || PLAT="$("$B/auditwheel" show "$WORK"/dist/*.whl 2>/dev/null | grep -o 'manylinux_[0-9_]*_[a-z0-9_]*' | head -1)"
    PATH="$B:$PATH" "$B/auditwheel" repair --plat "$PLAT" -w "$WORK/wheelhouse" "$WORK"/dist/*.whl 2>&1 | tail -1
else
    cp "$WORK"/dist/*.whl "$WORK/wheelhouse/"
fi
WHEEL="$(ls "$WORK"/wheelhouse/*.whl)"
"$B/twine" check --strict "$WORK"/wheelhouse/*

say "wheel contents"
"$B/python" - "$WHEEL" <<'EOF'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
need = ["pgw/__init__.py", "pgw/py.typed"]
missing = [n for n in need if n not in names]
for ext in ("pgw_fast", "pgw_ntt", "pgw_pks"):
    if not any(n.startswith(ext + ".") and n.endswith((".so", ".pyd")) and "abi3" in n for n in names):
        missing.append(ext + ".abi3.so")
for lic in ("LICENSE", "THIRD_PARTY_NOTICES.txt"):
    if not any(n.endswith("dist-info/licenses/" + lic) for n in names):
        missing.append(lic)
stray = [n for n in names if "__pycache__" in n or n.startswith(("tests/", "rust/"))]
if missing or stray:
    sys.exit(f"wheel check failed: missing={missing} stray={stray[:5]}")
print(f"{len(names)} files; native extensions, py.typed and licenses present")
EOF

say "fresh venv ($PYTHON) + install"
rm -rf "$WORK/venv"
"$PYTHON" -m venv "$WORK/venv"
V="$WORK/venv/bin"
"$V/pip" install -q --disable-pip-version-check --upgrade pip
IDX=()
[ -n "$TORCH_INDEX" ] && IDX=(--extra-index-url "$TORCH_INDEX")
"$V/pip" install -q --disable-pip-version-check ${IDX[@]+"${IDX[@]}"} "$WHEEL"
if "$V/pip" list 2>/dev/null | grep -qi '^nvidia-'; then
    echo "client install pulled CUDA libraries (nvidia-*): the CPU torch index was not used" >&2; exit 1
fi

say "smoke test (installed package, outside the source tree)"
mkdir -p "$WORK/cwd"
(cd "$WORK/cwd" && "$V/python" "$ROOT/scripts/smoke_installed.py")
(cd "$WORK/cwd" && PGW_NATIVE=0 "$V/python" "$ROOT/scripts/smoke_installed.py" --allow-pure)

say "sizes (client install)"
du -h "$WORK"/wheelhouse/*
SITE="$("$V/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "installed site-packages: $(du -sh "$SITE" | cut -f1) (torch: $(du -sh "$SITE/torch" | cut -f1))"

if [ "$RUN_TESTS" = 1 ]; then
    say "native bit-identity tests against the installed wheel"
    # Some of these tests start the compute service's lane in-process: they need the service extra.
    "$V/pip" install -q --disable-pip-version-check ${IDX[@]+"${IDX[@]}"} "$WHEEL[service]" \
        pytest pytest-asyncio pytest-timeout
    # -p no:cacheprovider: never write into the source tree; rootdir conftest only adds fixtures.
    # test_fused2_driver_with_split_and_prefetch drives the service's CPU emulation of its GPU kernels,
    # which needs triton (shipped with CUDA torch only): not part of a CPU client install.
    (cd "$ROOT" && PGW_OFFLINE=1 "$V/python" -m pytest -q -p no:cacheprovider \
        tests/native tests/hemask/test_ntt.py tests/hemask/test_pks.py \
        --deselect tests/native/test_scale32_knobs.py::test_fused2_driver_with_split_and_prefetch \
        -o addopts="" --import-mode=importlib)
fi

if [ "$FALLBACK" = 1 ]; then
    say "sdist build without a Rust toolchain (pure-Python fallback wheel)"
    rm -rf "$WORK/nort" && mkdir -p "$WORK/nort/src"
    tar -xzf "$SDIST" -C "$WORK/nort/src"  # fresh tree: no build/ left over from the native build
    (cd "$(ls -d "$WORK"/nort/src/*/)" && env -u CARGO -u CARGO_BUILD_TARGET PATH="/usr/bin:/bin" CARGO_HOME=/nonexistent \
        RUSTUP_HOME=/nonexistent "$B/python" -m build --wheel --no-isolation --outdir "$WORK/nort/dist" . \
        >"$WORK/nort/build.log" 2>&1) || { tail -40 "$WORK/nort/build.log"; exit 1; }
    "$B/python" - "$WORK"/nort/dist/*.whl <<'EOF'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
so = [n for n in names if n.endswith((".so", ".pyd"))]
assert not so and "pgw/__init__.py" in names, so
print("pure-Python wheel builds without Rust:", sys.argv[1].rsplit("/", 1)[-1])
EOF
fi

echo "release check passed: $WHEEL"
