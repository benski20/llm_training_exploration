#!/usr/bin/env python3
"""Build Kaggle GPU validation kernels with the current pgw source embedded as a base64 tarball.

Usage::

    python benchmarks/kaggle/build.py <out_root> [--owner KAGGLE_USER]

creates ``<out_root>/<kernel>/{run.py,kernel-metadata.json}`` for each job in ``JOBS``; push them with
``kaggle kernels push -p <out_root>/<kernel>``. The tarball leaves out rust/ (build artifacts), caches and
benchmark results.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PGW = HERE.parents[1]
EXCLUDE_DIRS = {"node_modules", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", "target", "results", "build", "dist"}
#: Top-level folders the GPU jobs never use (Kaggle rejects scripts above about 1 MB).
#: Paths (relative to pgw/) the GPU jobs never use: the console UI and its tests.
EXCLUDE_PATHS = ("tests/console", "src/pgw/console/static", "tests/cli", "tests/docs", "tests/deploy")
EXCLUDE_TOP = {"docs", "requirements", ".github", "ci", "research", "console", "deploy"}

QWEN05 = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN7 = "Qwen/Qwen2.5-7B-Instruct"
QWEN3 = "Qwen/Qwen2.5-3B-Instruct"

#: Builds the Rust gateway kernels (pgw_fast) on the Kaggle VM.
BUILD_NATIVE = {
    "kind": "shell", "name": "build_native",
    "cmd": "(command -v cargo || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal) && "
           ". $HOME/.cargo/env && python rust/tools/build_fast.py && "
           "python -c 'from pgw.core import native; print(\"pgw_fast\", native.fast and native.fast.version())'",
    "timeout": 1200,
}

JOBS: dict[str, dict] = {
    # run 26 (after the phase-1 merge): 7B single stream with the decode-only accounting and per-trip split,
    # aggregate throughput at 4/8/16 streams, the network floor and the gateway CPU profile (all layers warm).
    # Checks: decode remote ms per trip x 112 + local + LM head within 10% of s/token; outputs bit-identical.
    "pgw-run26-7b": {"steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; cat /sys/fs/cgroup/cpu.max || true", "timeout": 60},
        {"kind": "pytest", "name": "pytest_gpu", "args": ["tests", "-m", "gpu", "-q", "-rs", "--timeout", "900"],
         "timeout": 1800},
        {"kind": "shell", "name": "net_floor_bench",
         "cmd": "python benchmarks/net_floor_bench.py --iters 5000 --json /kaggle/working/net_floor.json",
         "timeout": 300},
        {"kind": "shell", "name": "gpu_kernel_bench",
         "cmd": "python benchmarks/gpu_kernel_bench.py --iters 300", "timeout": 1800},
        {"kind": "shell", "name": "profile_gateway_cpu",
         "cmd": "python benchmarks/profile_gateway.py --preset 7b --layers 2 --steps 24 --ring K32A12C27 --profile "
                "--workdir /tmp/pgw_prof", "timeout": 1800},
        {"kind": "validate", "label": "qwen7b", "args": [
            "--model", QWEN7, "--label", "qwen7b", "--device", "cuda", "--ref-device", "cuda", "--ring", "K32A12C27",
            "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "64", "--concurrent", "4", "8", "16",
            "--loadtest", "1x1", "1x5", "4x1", "8x1", "16x1", "--loadtest-tokens", "4", "--factory-bench-slots", "64"],
         "timeout": 7800},
    ]},
    # phase 2 P2-b A/B: two-cohort ping-pong (engine_pingpong, steps of > 4 sequences) vs the default engine at
    # 4 / 8 / 16 streams on the same service and mask store (results "concurrent_<n>" vs "pingpong_concurrent_<n>";
    # at 4 streams ping-pong does not engage: a control). 32 tokens keep the store near run 26's size (two gateway
    # runs: about 4.3k slots, 24 GB at 7B). Checks: tokens_identical everywhere; engine_delta.pingpong_steps > 0
    # at 8 and 16 streams only.
    "pgw-pingpong-7b": {"steps": [
        BUILD_NATIVE,
        {"kind": "pytest", "name": "pytest_sched", "args": ["tests/engine/test_generate.py",
                                                            "tests/engine/test_scheduling_pingpong.py", "-q", "-rs",
                                                            "--timeout", "900"], "timeout": 1500},
        {"kind": "validate", "label": "qwen7b_pingpong", "args": [
            "--model", QWEN7, "--label", "qwen7b_pingpong", "--device", "cuda", "--ref-device", "cuda",
            "--ring", "K32A12C27", "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "32",
            "--concurrent", "4", "8", "16", "--pingpong", "--loadtest", "1x1", "--loadtest-tokens", "2",
            "--factory-bench-slots", "64"],
         "timeout": 7800},
    ]},
    # speculation sweep: k = 1, 2, 3 with a 0.5B draft on the gateway CPU and on the second T4 (cuda:1), one
    # fixed draft size per run (spec_min_tokens capped at k), spec_max_batch 1; single stream for every run and
    # 4 streams for no-spec and k=2 on cuda:1 (spec at 4 streams should equal no-spec). 48 tokens keep the
    # mask store near run 25's size (about 3.5k slots, 19 GB at 7B).
    "pgw-spec-sweep-7b": {"steps": [
        BUILD_NATIVE,
        {"kind": "pytest", "name": "pytest_spec", "args": ["tests/engine/test_speculative.py", "-q", "-rs",
                                                           "--timeout", "900"], "timeout": 1500},
        {"kind": "validate", "label": "qwen7b_spec", "args": [
            "--model", QWEN7, "--label", "qwen7b_spec", "--device", "cuda", "--ref-device", "cuda",
            "--ring", "K32A12C27", "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "48",
            "--concurrent", "4", "--draft-model", QWEN05, "--spec-tokens", "1", "2", "3", "--spec-min-tokens", "16",
            "--spec-max-batch", "1", "--draft-device", "cpu", "cuda:1", "--spec-concurrent", "4",
            "--spec-concurrent-variants", "2@cuda:1", "--loadtest", "1x1", "--loadtest-tokens", "2",
            "--factory-bench-slots", "64"],
         "timeout": 7800},
    ]},
    # fused Triton ring GEMV (K <= 32, <= 16 rows): GPU exactness tests + timing vs the digit-GEMM path, 7B shapes
    "pgw-triton-gemv": {"exclude": ["rust", "deploy", "scripts", "tests/vectors", "tests/gateway", "tests/cli",
                                    "tests/maskstore", "tests/factory", "tests/docs", "tests/wire"], "steps": [
        {"kind": "pytest", "name": "pytest_triton_gemv",
         "args": ["tests/core/test_triton_gemv.py", "tests/core/test_ringmm.py", "tests/service/test_kernels.py",
                  "tests/service/test_gpu_exec.py", "tests/service/test_gpu_graphs.py", "-v", "-rs", "--timeout", "1800"],
         "timeout": 2400},
        {"kind": "shell", "name": "triton_gemv_bench",
         "cmd": "python benchmarks/triton_gemv_bench.py --iters 300 --sweep --json /kaggle/working/triton_gemv_bench.json",
         "timeout": 3600},
        {"kind": "shell", "name": "gpu_kernel_bench",
         "cmd": "python benchmarks/gpu_kernel_bench.py --iters 300", "timeout": 1800},
    ]},
    # 7B quality of the 32-bit capped ring vs today's K40A12 (int8 weights + per-token activations on GPU)
    "pgw-quality-7b": {"steps": [
        {"kind": "shell", "name": "quality_7b",
         "cmd": "python benchmarks/quant_quality_gpu.py --model Qwen/Qwen2.5-7B-Instruct --configs w8a12 c27 c25 c23 "
                "--eval-tokens 16384", "timeout": 7000},
    ]},
    # encrypted mask generation: CUDA mulmod64 exactness + cost at 7B / 8B / 0.5B shapes
    "pgw-he-gpu": {"steps": [
        {"kind": "pytest", "name": "pytest_hemask_gpu", "args": ["tests/hemask", "-v", "-rs", "--timeout", "1800"],
         "timeout": 2400},
        {"kind": "pytest", "name": "pytest_hemask_cuda_engine", "args": ["tests/hemask", "tests/service/test_maskgen.py",
                                                                         "-q", "-rs", "--timeout", "1800"],
         "env": {"PGW_HEMASK_ENGINE": "cuda"}, "timeout": 2400},
        {"kind": "shell", "name": "hemask_bench_7b",
         "cmd": "PGW_HEMASK_ENGINE=cuda python benchmarks/hemask_bench.py --preset qwen2.5-7b --device cuda --groups 4",
         "timeout": 3600},
        {"kind": "shell", "name": "hemask_bench_7b_k32",
         "cmd": "PGW_HEMASK_ENGINE=cuda python benchmarks/hemask_bench.py --preset qwen2.5-7b --device cuda "
                "--groups 4 --k 32", "timeout": 3600},
        {"kind": "shell", "name": "hemask_bench_7b_cpu_f64",
         "cmd": "PGW_HEMASK_ENGINE=cpu_f64 python benchmarks/hemask_bench.py --preset qwen2.5-7b --device cuda "
                "--groups 4", "timeout": 3600},
        {"kind": "shell", "name": "hemask_bench_05b",
         "cmd": "PGW_HEMASK_ENGINE=cuda python benchmarks/hemask_bench.py --preset qwen2.5-0.5b --device cuda "
                "--groups 4", "timeout": 1800},
    ]},
    # (a) GPU kernel tests + full suite with the e2e service on CUDA + ring benches + 0.5B end to end
    "pgw-gpu-tests": {"steps": [
        {"kind": "shell", "name": "probe_int_mm", "cmd": "python benchmarks/kaggle/probe_int_mm.py", "timeout": 300},
        {"kind": "doctor"},
        {"kind": "pytest", "name": "pytest_gpu", "args": ["tests", "-m", "gpu", "-v", "-rs", "--timeout", "900"],
         "timeout": 1800},
        {"kind": "validate", "label": "qwen0.5b", "args": [
            "--model", QWEN05, "--label", "qwen0.5b", "--device", "cuda", "--ref-device", "cpu",
            "--max-tokens", "32", "--concurrent", "8", "--loadtest", "1x1", "8x1", "32x1", "1x8", "1x32", "4x32", "1x128",
            "--loadtest-tokens", "8", "--factory-bench-slots", "1024"], "timeout": 3600},
        {"kind": "ring", "shapes": ["llama3-8b.qkv", "llama3-8b.gate_up", "llama3-8b.down",
                                    "qwen2.5-0.5b.gate_up"], "rings": ["K40A12", "K48A20", "K32A8"]},
        {"kind": "pytest", "name": "pytest_full", "args": ["tests", "-q", "-rs", "--timeout", "1200"],
         "env": {"PGW_E2E_SERVICE_DEVICE": "cuda"}, "timeout": 3600},
    ]},
    # (b) 7B end to end on one T4
    "pgw-gpu-7b": {"steps": [
        {"kind": "shell", "name": "probe_int_mm", "cmd": "python benchmarks/kaggle/probe_int_mm.py", "timeout": 300},
        {"kind": "doctor"},
        {"kind": "validate", "label": "qwen7b", "args": [
            "--model", QWEN7, "--label", "qwen7b", "--device", "cuda", "--ref-device", "cuda",
            "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "32", "--concurrent", "8",
            "--loadtest", "1x1", "8x1", "32x1", "1x8", "1x32", "4x32", "1x128", "--loadtest-tokens", "2", "--factory-bench-slots", "256"],
         "timeout": 6600},
    ]},
    # re-run after fixes: GPU tests, the tests that failed in the full suite, e2e with the service on CUDA
    "pgw-gpu-verify": {"steps": [
        {"kind": "pytest", "name": "pytest_gpu", "args": ["tests", "-m", "gpu", "-v", "-rs", "--timeout", "900"],
         "timeout": 1800},
        {"kind": "pytest", "name": "pytest_regressions", "args": [
            "tests/core/test_misc.py", "tests/factory/test_bench_cli.py", "tests/vectors", "-q", "-rs",
            "--timeout", "900"], "timeout": 1800},
        {"kind": "pytest", "name": "pytest_e2e_gpu_service", "args": ["tests/e2e", "-v", "-rs", "--timeout", "1800"],
         "env": {"PGW_E2E_SERVICE_DEVICE": "cuda"}, "timeout": 2400},
    ]},
    # speed pass: GPU kernel exactness (stacked digit GEMM) + 7B end to end, compare with run4_7b
    "pgw-speed-7b": {"steps": [
        {"kind": "shell", "name": "probe_int_mm", "cmd": "python benchmarks/kaggle/probe_int_mm.py", "timeout": 300},
        {"kind": "pytest", "name": "pytest_gpu", "args": ["tests", "-m", "gpu", "-v", "-rs", "--timeout", "900"],
         "timeout": 1800},
        {"kind": "validate", "label": "qwen7b", "args": [
            "--model", QWEN7, "--label", "qwen7b", "--device", "cuda", "--ref-device", "cuda",
            "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "32", "--concurrent", "8",
            "--loadtest", "1x1", "8x1", "32x1", "1x8", "1x32", "4x32", "1x128", "--loadtest-tokens", "2",
            "--factory-bench-slots", "256"],
         "timeout": 6600},
    ]},
    # fallback if 7B does not fit / is too slow
    "pgw-gpu-3b": {"steps": [
        {"kind": "validate", "label": "qwen3b", "args": [
            "--model", QWEN3, "--label", "qwen3b", "--device", "cuda", "--ref-device", "cuda",
            "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "32", "--concurrent", "8",
            "--loadtest", "1x1", "8x1", "32x1", "1x8", "1x32", "4x32", "1x128", "--loadtest-tokens", "2", "--factory-bench-slots", "256"],
         "timeout": 6600},
    ]},
}


def source_tarball(exclude: tuple[str, ...] = ()) -> bytes:
    """xz tarball of pgw/ without the excluded folders (plus ``exclude``: job-specific paths under pgw/)."""
    buf = io.BytesIO()
    skip = EXCLUDE_PATHS + tuple(exclude)

    def filt(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(ti.name).parts
        if len(parts) > 1 and parts[1] in EXCLUDE_TOP:
            return None
        rel = "/".join(parts[1:])
        if any(rel == p or rel.startswith(p + "/") for p in skip):
            return None
        if any(p in EXCLUDE_DIRS or p.endswith(".egg-info") for p in parts) or ti.name.endswith((".whl", ".so")):
            return None
        return ti

    with tarfile.open(fileobj=buf, mode="w:xz", preset=9) as tf:
        tf.add(PGW, arcname="pgw", filter=filt)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--owner", default="bensliv")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    template = (HERE / "kernel_template.py").read_text()
    tarballs: dict[tuple[str, ...], str] = {}
    for slug, job in JOBS.items():
        if args.only and slug not in args.only:
            continue
        excl = tuple(job.get("exclude", ()))
        if excl not in tarballs:
            raw = base64.b64encode(source_tarball(excl)).decode()
            tarballs[excl] = "\n".join(raw[i : i + 100] for i in range(0, len(raw), 100))  # short lines for Kaggle
        tgz = tarballs[excl]
        d = Path(args.out_root) / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.py").write_text(template.replace("__JOB__", json.dumps(job)).replace("__PGW_TGZ_B64__", tgz))
        meta = {"id": f"{args.owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": True, "machine_shape": "NvidiaTeslaP100",
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        print(d, f"{(d / 'run.py').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
