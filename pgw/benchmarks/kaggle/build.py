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
QWEN15 = "Qwen/Qwen2.5-1.5B-Instruct"

#: Builds the Rust gateway kernels (pgw_fast) and the HE customer's NTT products (pgw_ntt) on the Kaggle VM.
BUILD_NATIVE = {
    "kind": "shell", "name": "build_native", "required": True,
    # rust-version is 1.89 (stable AVX-512 intrinsics): install rustup when cargo is missing or older.
    "cmd": "{ [ -f $HOME/.cargo/env ] && . $HOME/.cargo/env; true; } && "
           "(python rust/tools/cargo_ok.py 1.89 || "
           "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable) && "
           "{ [ -f $HOME/.cargo/env ] && . $HOME/.cargo/env; true; } && cargo --version && python rust/tools/build_fast.py && "
           "python -c 'from pgw.core import native; assert native.fast is not None, \"pgw_fast not loaded\"; "
           "print(\"pgw_fast\", native.fast.version())' && python rust/tools/build_ntt.py && "
           "python -c 'from pgw.hemask import ntt; assert ntt.available(), \"pgw_ntt not loaded\"; "
           "print(\"pgw_ntt\", ntt.info())'",
    "timeout": 1200,
}

#: X4 concurrency job: only gpu_validate.py (and its probe) of the benchmarks, no tests or top-level docs (keeps the
#: bundle under Kaggle's size limit).
X4_EXCLUDE = ["scripts", "tests", "SPEC.md", "INTERFACE_NOTES.md", "STATUS.md", "src/pgw/moe",
              "benchmarks/kaggle/probe_int_mm.py",
              *sorted(f"benchmarks/{p.name}" for p in HERE.parent.glob("*.py")
                      if p.name not in ("gpu_validate.py", "gateway_probe.py"))]

JOBS: dict[str, dict] = {
    # run 27 (after the phase-2 merge; run 26 had the same steps after phase 1): 7B single stream with the decode-only accounting and per-trip split,
    # aggregate throughput at 4/8/16 streams, the network floor and the gateway CPU profile (all layers warm).
    # Checks: decode remote ms per trip x 112 + local + LM head within 10% of s/token; outputs bit-identical.
    "pgw-run27-7b": {"exclude": ["scripts", "tests/gateway", "tests/e2e", "tests/maskstore", "tests/enroll"], "steps": [
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
    "pgw-pingpong-7b": {"exclude": ["scripts", "tests/gateway", "tests/e2e", "tests/maskstore", "tests/enroll"], "steps": [
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
    # phase 2 package D: the gateway's plaintext work per decode token (LM head, norms, RoPE, attention,
    # sampling, hops) at 7B shapes with all 28 layers and a stub service (CPU only, about 2 GB). A/B on the
    # Kaggle CPU: new paths (AVX-512 / register-blocked LM head, fast RoPE and sampler helpers) vs the
    # reference forms (PGW_LM_HEAD_ISA=avx + --reference-paths), at 1, 4 and 16 rows; plus spinning OpenMP
    # workers (OMP_WAIT_POLICY=ACTIVE) to size their effect. Outputs are bit-identical in every variant.
    "pgw-token-profile-7b": {"gpu": False, "exclude": ["deploy", "scripts", "tests"], "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; cat /sys/fs/cgroup/cpu.max || true", "timeout": 60},
        *[{"kind": "shell", "name": f"profile_token_{tag}_r{rows}",
           "cmd": f"{env} python benchmarks/profile_token.py --steps 30 --rows {rows} {flags} "
                  f"--json /kaggle/working/profile_token_{tag}_r{rows}.json",
           "timeout": 1200}
          for rows in (1, 4, 16)
          for tag, env, flags in (("new", "", ""), ("ref", "PGW_LM_HEAD_ISA=avx", "--reference-paths"),
                                  ("avx2", "PGW_LM_HEAD_ISA=avx2", ""),
                                  ("omp_active", "OMP_WAIT_POLICY=ACTIVE", ""))],
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
    # factory r expansion on the GPU (ChaCha20 + digit split in one Triton kernel): byte-identity vs the CPU
    # expander on CUDA, then 7B (K32A12C27) factory slots/h with PGW_FACTORY_GPU_PRNG=1 vs 0 (same seeds/work).
    # Target (STATUS "Factory throughput"): >= 500k slots/h on a T4 (was 169k with CPU ChaCha20).
    "pgw-factory-gpu-prng": {"exclude": ["scripts", "tests/gateway", "tests/e2e", "tests/maskstore", "tests/enroll",
                                         "tests/wire", "tests/service", "tests/console"], "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; nvidia-smi || true", "timeout": 60},
        {"kind": "pytest", "name": "pytest_factory_gpu",
         # test_runtime_he_mtls imports tests.wire, which this bundle leaves out
         "args": ["tests/factory", "-m", "gpu", "--ignore", "tests/factory/test_runtime_he_mtls.py",
                  "-v", "-rs", "--timeout", "1800"], "timeout": 2400},
        {"kind": "pytest", "name": "pytest_gpu_prng_all",
         "args": ["tests/factory/test_gpu_prng.py", "tests/factory/test_generator.py",
                  "tests/factory/test_self_check.py", "-q", "-rs", "--timeout", "1800"], "timeout": 2400},
        *[{"kind": "shell", "name": f"bench_factory_7b_k32_l{layers}_gpu_prng{flag}",
           "cmd": f"PGW_FACTORY_GPU_PRNG={flag} python -m pgw.cli bench factory --model qwen2.5-7b --ring K32A12C27 "
                  f"--device cuda --layers {layers} --slots 1024 --chunk-slots 256 --json",
           "timeout": 2400}
          for layers in (2, 28) for flag in (1, 0)],
        *[{"kind": "shell", "name": f"bench_factory_7b_k40_gpu_prng{flag}",
           "cmd": f"PGW_FACTORY_GPU_PRNG={flag} python -m pgw.cli bench factory --model qwen2.5-7b --ring K40A12 "
                  "--device cuda --layers 2 --slots 1024 --chunk-slots 256 --json",
           "timeout": 2400}
          for flag in (1, 0)],
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
    # encrypted masks with a GPU on the customer gateway (Modal --customer-gpu): exactness of every hemask path
    # with the CUDA engine picked automatically (the CPU NTT must step aside), then encrypt/decrypt cost per slot
    # at the Modal demo's 0.5B K32 shapes on the GPU vs the CPU int8 engine (same box).
    "pgw-he-gateway-gpu": {"exclude": ["scripts", "tests/gateway", "tests/e2e", "tests/maskstore", "tests/enroll",
                                       "tests/console", "tests/factory", "tests/engine", "tests/wire", "tests/native",
                                       "tests/core", "tests/cli", "tests/docs", "tests/deploy", "tests/benchmarks",
                                       "docs", "research", "console", "tests/vectors", "SPEC.md",
                                       "INTERFACE_NOTES.md", "tests/service", "tests/modelpack", "tests/observability",
                                       "STATUS.md", "benchmarks/gpu_validate.py", "benchmarks/profile_token.py"],
                           "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; nvidia-smi || true", "timeout": 60},
        {"kind": "pytest", "name": "pytest_hemask_auto_cuda", "args": ["tests/hemask", "-q", "-rs", "--timeout", "1800"],
         "timeout": 2400},
        *[{"kind": "shell", "name": f"hemask_bench_05b_k32_{eng}",
           "cmd": f"PGW_HEMASK_ENGINE={eng} python benchmarks/hemask_bench.py --preset qwen2.5-0.5b --device cuda "
                  "--groups 4 --k 32", "timeout": 1800}
          for eng in ("cuda", "cpu_int8")],
    ]},
    # Gateway encryption: dense (T4 GPU / CPU int8) vs public matrix in NTT form (mlwe_ring, CPU)
    "pgw-he-encrypt-ntt": {"exclude": ["scripts", "tests/gateway", "tests/e2e", "tests/maskstore", "tests/enroll",
                                       "tests/console", "tests/factory", "tests/engine", "tests/wire", "tests/native",
                                       "tests/core", "tests/cli", "tests/docs", "tests/deploy", "tests/benchmarks",
                                       "docs", "research", "console", "tests/vectors", "SPEC.md",
                                       "INTERFACE_NOTES.md", "tests/service", "tests/modelpack", "tests/observability",
                                       "STATUS.md", "benchmarks/gpu_validate.py", "benchmarks/profile_token.py",
                                       "benchmarks/results", "benchmarks/RESULTS.md", "ci", "deploy"],
                           "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; free -g; nvidia-smi || true", "timeout": 60},
        {"kind": "pytest", "name": "pytest_mlwe_ring",
         "args": ["tests/hemask/test_mlwe_ring.py", "tests/hemask/test_ntt.py", "-q", "-rs", "--timeout", "1800"],
         "timeout": 2400},
        *[{"kind": "shell", "name": f"encrypt_bench_{preset}_{eng}",
           "cmd": f"PGW_HEMASK_ENGINE={eng} python benchmarks/hemask_encrypt_bench.py --preset {preset} "
                  f"--rows 2048 --dense-rows {256 if eng == 'cuda' else 128} --groups 1,4,8", "timeout": 1800}
          for preset in ("qwen2.5-0.5b", "qwen2.5-7b") for eng in ("cuda", "cpu_int8")],
        {"kind": "shell", "name": "encrypt_bench_7b_rank12",
         "cmd": "PGW_HEMASK_ENGINE=cuda python benchmarks/hemask_encrypt_bench.py --preset qwen2.5-7b --rank 12 "
                "--rows 2048 --dense-rows 256 --groups 4", "timeout": 1800},
    ]},
    # package M2: provider-side encrypted maskgen throughput (device-side v2 unpack/pack, ceil(bb/8) planes,
    # fused Triton combine, per-device slots). GPU exactness tests, then legacy vs fused slots/s and TOPS at
    # 0.5B / 7B / 70B layer shapes, layers sharded over both T4s, and decode-GEMM latency next to maskgen.
    "pgw-maskgen-provider": {"machine_shape": "NvidiaTeslaT4x2",
                             "exclude": ["scripts", "tests/gateway", "tests/e2e", "tests/maskstore", "tests/enroll",
                                         "tests/console", "tests/factory", "tests/engine", "tests/wire",
                                         "tests/native", "tests/core", "tests/cli", "tests/docs", "tests/deploy",
                                         "tests/benchmarks", "docs", "research", "console", "tests/vectors",
                                         "SPEC.md", "INTERFACE_NOTES.md", "tests/modelpack", "tests/observability",
                                         "STATUS.md", "benchmarks/gpu_validate.py", "benchmarks/profile_token.py",
                                         "rust", "tests/hemask"],
                             "steps": [
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu | head -20; nproc; free -g; nvidia-smi || true",
         "timeout": 60},
        {"kind": "pytest", "name": "pytest_maskgen_gpu",
         "args": ["tests/service/test_maskgen_gpu.py", "tests/service/test_maskgen_fast.py",
                  "tests/service/test_maskgen.py", "-q", "-rs", "--timeout", "1800"], "timeout": 2400},
        {"kind": "shell", "name": "maskgen_provider_bench",
         # run 1 (v1 of this kernel) measured legacy vs fused; run 2 measures the fused path after the
         # host-copy cuts (parallel pinned upload, zero-copy reply), smaller yield blocks, and the stage profile
         "cmd": "python benchmarks/maskgen_provider_bench.py --presets qwen2.5-0.5b qwen2.5-7b llama3-70b "
                "--modes fused --conc 2 4 --rounds 4 --decode --profile "
                "--json /kaggle/working/maskgen_provider.json", "timeout": 5400},
    ]},
    # scale plan (research/plans/scale-north-star.md): provider ring-GEMM throughput at decode batches 1..1024 for
    # Llama-3-70B and Mixtral-8x22B shapes (K32, exact spot check), oblivious-MoE overhead, plain factory rate.
    "pgw-scale-provider": {"exclude": ["rust", "scripts", "tests/gateway", "tests/e2e", "tests/maskstore",
                                       "tests/enroll", "tests/console", "tests/factory", "tests/engine", "tests/wire",
                                       "tests/native", "tests/cli", "tests/docs", "tests/deploy", "tests/benchmarks",
                                       "tests/hemask", "tests/service", "tests/modelpack", "tests/observability",
                                       "tests/vectors", "tests/integration", "SPEC.md", "INTERFACE_NOTES.md",
                                       "STATUS.md", "benchmarks/gpu_validate.py", "benchmarks/profile_token.py"],
                           "steps": [
        {"kind": "shell", "name": "gpu_info", "cmd": "nvidia-smi || true; nproc", "timeout": 60},
        {"kind": "pytest", "name": "pytest_moe", "args": ["tests/moe", "-q", "-rs", "--timeout", "600"],
         "timeout": 900},
        {"kind": "shell", "name": "scale_provider_bench",
         "cmd": "python benchmarks/scale_provider_bench.py --iters 5 --json /kaggle/working/scale_provider_bench.json",
         "timeout": 3600},
    ]},
    # gateway mask pipeline (factory writer -> disk store -> batched decode claims -> wipe) on a quiet 4-vCPU
    # box: A/B against a base commit, plus the batched native claim. CPU only. Before building, write the diff
    # from the base to this tree of the files that differ (not committed):
    #   git diff 2078450 --relative -- src/pgw/core/prng.py src/pgw/maskstore > benchmarks/kaggle/_ab_base.patch
    # Results: benchmarks/RESULTS.md, "Gateway mask pipeline".
    "pgw-mask-pipeline": {"gpu": False,
                          "exclude": ["scripts", "tests", "docs", "research", "console", "SPEC.md",
                                      "INTERFACE_NOTES.md", "STATUS.md", "src/pgw/hemask", "src/pgw/console",
                                      "src/pgw/service",
                                      "benchmarks/gpu_validate.py", "benchmarks/profile_token.py"],
                          "steps": [
        {"kind": "shell", "name": "build_fast", "required": True,
         "cmd": "{ [ -f $HOME/.cargo/env ] && . $HOME/.cargo/env; true; } && "
                "(python rust/tools/cargo_ok.py 1.89 || "
                "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable) && "
                "{ [ -f $HOME/.cargo/env ] && . $HOME/.cargo/env; true; } && python rust/tools/build_fast.py && "
                "python -c 'import pgw_fast; assert hasattr(pgw_fast, \"claim_rows_many\")'",
         "timeout": 1500},
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; free -g; df -h /tmp /kaggle/working",
         "timeout": 60},
        {"kind": "shell", "name": "make_base", "required": True,
         "cmd": "rm -rf /tmp/pgwbase && mkdir -p /tmp/pgwbase && cp -r src /tmp/pgwbase/ && "
                "patch -d /tmp/pgwbase -p1 -R < benchmarks/kaggle/_ab_base.patch && "
                "PYTHONPATH=/tmp/pgwbase/src python -c 'import pgw.maskstore.store as s; "
                "assert not hasattr(s, \"take_checked_many\"); print(\"base ok\")'", "timeout": 120},
        *[{"kind": "shell", "name": f"pipeline_{tag}_{side}_{rep}",
           "cmd": f"PGW_OFFLINE=1 PYTHONPATH={pp} python benchmarks/mask_pipeline_bench.py {args} --root /tmp/mpb "
                  f"--json /kaggle/working/pipeline_{tag}_{side}_{rep}.json{extra}",
           "timeout": 1800}
          for rep in (1, 2)
          for tag, args in (("05b", "--model 0.5b --slots 512 --segments 1 --sessions 32 --tokens 16"),
                            ("7b", "--model 7b --slots 256 --segments 1 --sessions 32 --tokens 8"),
                            ("70b_l8", "--model 70b --layers 8 --slots 256 --segments 1 --sessions 32 --tokens 8"))
          for side, pp, extra in (("base", "/tmp/pgwbase/src", ""), ("new", "src", ""),  # noqa: S108
                                  ("new_claim", "src", " --claim-threads 0"))],
    ]},
    # co-located placement of the Modal demo (deploy/modal/colocated.py) on a T4: the same pgw_modal_boot party
    # starters (provider service on the GPU as user pgwprov, gateway as pgwcust without the GPU, loopback TLS),
    # the setup wizard and two greedy chats of Qwen2.5-0.5B checked against the plaintext reference; then the
    # same with 17 ms added per round trip (the Modal tunnel) through a loopback delay proxy. Reports trip
    # latency, round trips per token and tokens/s. Plus the wire bench with TLS and the TCP_NODELAY check.
    "pgw-colocated-t4": {"include_top": ["deploy"], "exclude": [
        "scripts", "tests", "docs", "research", "console", "SPEC.md", "INTERFACE_NOTES.md", "STATUS.md",
        "deploy/helm", "deploy/docker", "deploy/compose", "deploy/hf-space", "deploy/marketplace", "deploy/monitoring",
        "deploy/systemd", "deploy/vmware",
        # every benchmark but the wire bench and its delay proxy
        "benchmarks/RESULTS.md", "benchmarks/chat_speed_model.py", "benchmarks/e2e_local.py",
        "benchmarks/gateway_probe.py", "benchmarks/gpu_kernel_bench.py", "benchmarks/gpu_validate.py",
        "benchmarks/hemask_bench.py", "benchmarks/hemask_ntt_bench.py", "benchmarks/kaggle",
        "benchmarks/lane_bench.py", "benchmarks/lm_head_bench.py", "benchmarks/native_finish_bench.py",
        "benchmarks/native_prepare_bench.py", "benchmarks/native_store_bench.py", "benchmarks/net_floor_bench.py",
        "benchmarks/profile_gateway.py", "benchmarks/profile_prefill.py", "benchmarks/profile_token.py",
        "benchmarks/quant_quality.py", "benchmarks/quant_quality_gpu.py", "benchmarks/spec_cost_bench.py",
        "benchmarks/stack_bench.py", "benchmarks/triton_gemv_bench.py"],
        "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; nvidia-smi || true; df -h / /tmp", "timeout": 60},
        {"kind": "shell", "name": "wire_tls_nodelay",
         "cmd": "python benchmarks/wire_overhead_bench.py --calls 1000 --tls --check-nodelay && "
                "python benchmarks/wire_overhead_bench.py --calls 1000 --tls --check-nodelay --delay-ms 0 && "
                "python benchmarks/wire_overhead_bench.py --calls 300 --tls --check-nodelay --delay-ms 17",
         "timeout": 900},
        {"kind": "shell", "name": "colocated_users_t4",
         "cmd": "python deploy/modal/local_rehearsal.py --work /tmp/reh --device auto --users "
                "--json /kaggle/working/colocated_t4.json", "timeout": 3000},
        {"kind": "shell", "name": "tunnel_emulated_17ms_t4",
         "cmd": "rm -rf /tmp/reh/provider /tmp/reh/customer /tmp/reh/logs && "
                "python deploy/modal/local_rehearsal.py --work /tmp/reh --device auto --users --delay-ms 17 "
                "--json /kaggle/working/tunnel17_t4.json", "timeout": 3000},
    ]},
    # provider GPU wiring on a T4 with Modal's torch (2.15.0): which fused ring GEMV variant compiles and passes
    # its self-check, GPU exactness tests, the service time per call at the Modal demo's 0.5B shapes (Triton vs
    # digit GEMM vs CUDA graphs, 1-16 rows), and the real service per round trip over TLS at 1 / 2 / all cores.
    "pgw-g1-provider-t4": {"exclude": ["rust", "scripts", "tests/gateway", "tests/e2e", "tests/maskstore",
                                       "tests/enroll", "tests/console", "tests/factory", "tests/engine", "tests/wire",
                                       "tests/native", "tests/cli", "tests/docs", "tests/deploy", "tests/benchmarks",
                                       "tests/hemask", "tests/vectors", "tests/modelpack", "tests/observability",
                                       "docs", "research", "console", "SPEC.md", "INTERFACE_NOTES.md", "STATUS.md",
                                       "benchmarks/gpu_validate.py", "benchmarks/profile_token.py",
                                       "benchmarks/chat_speed_model.py", "benchmarks/RESULTS.md"],
                           "steps": [
        {"kind": "shell", "name": "torch_modal_version",
         "cmd": "python -m pip install -q torch==2.15.0 && python -c 'import torch, triton; "
                "assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, triton.__version__, "
                "torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))' || "
                "(python -m pip install -q torch==2.15.0 --index-url https://download.pytorch.org/whl/cu126 && "
                "python -c 'import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__, "
                "torch.cuda.is_available())')", "timeout": 1500},
        {"kind": "shell", "name": "cpu_info", "cmd": "nproc; lscpu | head -20; nvidia-smi", "timeout": 60},
        {"kind": "pytest", "name": "pytest_gpu_kernels",
         "args": ["tests/core/test_triton_gemv.py", "tests/core/test_triton_gemv_multirow.py",
                  "tests/service/test_gpu_exec.py", "tests/service/test_kernels.py", "tests/service/test_gpu_graphs.py",
                  "-q", "-rs", "--timeout", "1800"],
         "timeout": 2400},
        {"kind": "shell", "name": "provider_trip_kernel",
         "cmd": "python benchmarks/provider_trip_bench.py --iters 300 --json /kaggle/working/provider_trip_kernel.json",
         "timeout": 1800},
        {"kind": "shell", "name": "provider_trip_service",
         "cmd": "python benchmarks/provider_trip_bench.py --no-kernel --service --tls --cores 0 0,1 all --tokens 20 "
                "--json /kaggle/working/provider_trip_service.json", "timeout": 2400},
    ]},
    # G2: gateway GPU wiring. Device passes of the encrypted factory (exactness + rate) vs the per-call CUDA
    # products and the CPU int8 engine, the local GPU factory, and the local qkv/o linears per token.
    "pgw-g2-gateway-gpu": {"exclude": ["scripts", "tests/gateway", "tests/e2e", "tests/maskstore", "tests/enroll",
                                       "tests/console", "tests/engine", "tests/wire", "tests/native",
                                       "tests/core", "tests/cli", "tests/docs", "tests/deploy", "tests/benchmarks",
                                       "docs", "research", "console", "tests/vectors", "SPEC.md",
                                       "INTERFACE_NOTES.md", "tests/service", "tests/modelpack", "tests/observability",
                                       "STATUS.md", "benchmarks/gpu_validate.py", "benchmarks/profile_token.py",
                                       "tests/factory/test_export_e2e.py", "tests/factory/test_reliability.py",
                                       "tests/factory/test_scheduler.py", "tests/factory/test_sinks.py",
                                       *[f"benchmarks/{b}.py" for b in (
                                           "chat_speed_model", "e2e_local", "gateway_probe", "gpu_kernel_bench",
                                           "hemask_ntt_bench", "lane_bench", "lm_head_bench", "native_finish_bench",
                                           "native_prepare_bench", "native_store_bench", "net_floor_bench",
                                           "profile_gateway", "profile_prefill", "quant_quality",
                                           "quant_quality_gpu", "spec_cost_bench", "stack_bench",
                                           "triton_gemv_bench", "wire_overhead_bench")]],
                           "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; free -g; nvidia-smi || true", "timeout": 60},
        {"kind": "pytest", "name": "pytest_devmm_cuda",
         "args": ["tests/hemask/test_devmm.py", "tests/hemask/test_gpu.py", "tests/hemask/test_factory_he.py",
                  "-q", "-rs",
                  "--timeout", "1800"],
         "timeout": 2400},
        *[{"kind": "shell", "name": name, "cmd": f"{env} python benchmarks/gateway_gpu_bench.py {args}",
           "timeout": 2400} for name, env, args in (
            ("he_device_24l", "", "--mode he --layers 24 --slots 4096"),
            ("he_device_4l", "", "--mode he --layers 4 --slots 2048"),
            ("he_device_stream_4l", "PGW_HEMASK_GPU_HINT_MB=0", "--mode he --layers 4 --slots 4096"),
            ("he_cuda_percall_4l", "PGW_HEMASK_DEVICE_PASSES=0", "--mode he --layers 4 --slots 2048"),
            # run 1 (kept for reference, slow on Kaggle's CPU): he_cpu_int8 (--device cpu, PGW_HEMASK_ENGINE=cpu_int8),
            # local_cuda_factory_24l (--mode local), linears_1row / linears_32rows (--mode linears)
        )],
    ]},
    # S1: draft acceptance of exact strict speculation on chat prompts (benchmarks/spec_accept_bench.py): tokens
    # per masked pass for each R (greedy, simulated over every R) and the real Generator in strict mode (greedy and
    # sampled; tokens must equal plain decoding). Targets 0.5B (prompt lookup only: the demo today), 1.5B and 3B
    # with the 0.5B draft on the GPU, and the 1.5B with the 0.5B draft on the CPU (draft time per pass on a
    # CPU-only gateway). All on the exact PlainBackend (CUDA), which is bit-identical to the masked path.
    "pgw-spec-accept": {"exclude": ["scripts", "tests", "docs", "research", "console", "SPEC.md", "INTERFACE_NOTES.md",
                                    "STATUS.md", "benchmarks/gpu_validate.py", "benchmarks/profile_token.py",
                                    "benchmarks/results"],
                        "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; nvidia-smi || true", "timeout": 60},
        *[{"kind": "shell", "name": f"pack_{tag}", "required": True,
           "cmd": f"python -m pgw.cli pack {repo} /tmp/packs/{tag} --name {tag} -q && rm -rf $HOME/.cache/huggingface",
           "timeout": 1800} for tag, repo in (("q05", QWEN05), ("q15", QWEN15), ("q3", QWEN3))],
        {"kind": "shell", "name": "accept_q05_lookup",
         "cmd": "python benchmarks/spec_accept_bench.py --target-pack /tmp/packs/q05 --device cuda --label q05 "
                "--check 3,4 --json /kaggle/working/accept_q05.json --dump-tokens /kaggle/working/tokens_q05.json",
         "timeout": 3600},
        {"kind": "shell", "name": "accept_q15_d05",
         "cmd": "python benchmarks/spec_accept_bench.py --target-pack /tmp/packs/q15 --draft-pack /tmp/packs/q05 "
                "--device cuda --label q15_d05 --check 4,5 --json /kaggle/working/accept_q15_d05.json "
                "--dump-tokens /kaggle/working/tokens_q15.json", "timeout": 5400},
        {"kind": "shell", "name": "accept_q3_d05",
         "cmd": "python benchmarks/spec_accept_bench.py --target-pack /tmp/packs/q3 --draft-pack /tmp/packs/q05 "
                "--device cuda --label q3_d05 --check 4 --json /kaggle/working/accept_q3_d05.json "
                "--dump-tokens /kaggle/working/tokens_q3.json", "timeout": 5400},
        {"kind": "shell", "name": "accept_q15_d05_cpu",
         "cmd": "python benchmarks/spec_accept_bench.py --target-pack /tmp/packs/q15 --draft-pack /tmp/packs/q05 "
                "--device cuda --draft-device cpu --label q15_d05_cpu --prompts 4 --rs 4 --check 4 --check-prompts 3 "
                "--temperature 0 --json /kaggle/working/accept_q15_d05_cpu.json", "timeout": 3600},
    ]},
    # S1 follow-up: the gateway's real configuration (target float ops on the CPU, the only device that keeps verify
    # passes bit-identical; the 0.5B draft on the T4): Generator exactness (greedy and sampled) and tokens per pass of
    # the 1.5B target at R = 4 and 5, draft time per pass on the GPU next to a CPU target, and the gateway-side
    # verify cost per row (spec_cost_bench, 1.5B shapes, masking included) on the Kaggle CPU.
    "pgw-spec-accept-cpu": {"exclude": ["scripts", "tests", "docs", "research", "console", "SPEC.md",
                                        "INTERFACE_NOTES.md", "STATUS.md", "benchmarks/gpu_validate.py",
                                        "benchmarks/results"],
                            "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; nvidia-smi || true", "timeout": 60},
        *[{"kind": "shell", "name": f"pack_{tag}", "required": True,
           "cmd": f"python -m pgw.cli pack {repo} /tmp/packs/{tag} --name {tag} -q && rm -rf $HOME/.cache/huggingface",
           "timeout": 1800} for tag, repo in (("q05", QWEN05), ("q15", QWEN15))],
        {"kind": "shell", "name": "accept_q15_cpu_d05_gpu",
         "cmd": "python benchmarks/spec_accept_bench.py --target-pack /tmp/packs/q15 --draft-pack /tmp/packs/q05 "
                "--device cpu --draft-device cuda --label q15_cpu_d05_gpu --prompts 4 --check-prompts 4 "
                "--max-tokens 96 --rs 2,3,4,5,6,8 --check 4,5 --json /kaggle/working/accept_q15_cpu.json",
         "timeout": 9000},
        {"kind": "shell", "name": "spec_cost_15b",
         "cmd": "python benchmarks/spec_cost_bench.py --preset 1.5b --layers 4 --rows 4,5 --masking "
                "--workdir /tmp/sc15 --json /kaggle/working/spec_cost_15b.json", "timeout": 3600},
        {"kind": "shell", "name": "spec_cost_05b",
         "cmd": "python benchmarks/spec_cost_bench.py --preset 0.5b --layers 4 --rows 4,5 --masking "
                "--workdir /tmp/sc05 --json /kaggle/working/spec_cost_05b.json", "timeout": 3600},
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
    # X4: aggregate online throughput vs concurrent chats on one T4 (gateway co-located over loopback, masks
    # pre-filled by the factory before the gateway starts, so maskgen is not the limit). 0.5B then 1.5B: 1, 8, 32,
    # 64 identical greedy requests at once on one gateway (every output compared with the plaintext reference),
    # then 64 requests over 2 and 3 gateway processes on the same store and service (request i to gateway i mod N),
    # then the service alone (pgw.service.loadtest, N sessions x 1 row: the provider's ceiling without a gateway).
    # A/B: the old thread count (float_threads null) as variant "ft_off" against the default 1; the sharded
    # gateways use it too. Reported per level: generated tokens/s, gateway and service CPU cores used, engine steps.
    "pgw-x4-concurrency": {"exclude": X4_EXCLUDE, "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; nvidia-smi || true; df -h /tmp", "timeout": 60},
        {"kind": "validate", "label": "qwen05_conc", "args": [
            "--model", QWEN05, "--label", "qwen05_conc", "--device", "cuda", "--ref-device", "cuda",
            "--ring", "K32A12C27", "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "24",
            "--concurrent", "1", "8", "32", "64", "--variant", 'ft_off={"float_threads": null}',
            "--shard-gateways", "2", "3", "--shard-concurrent", "64", "--shard-config", '{"float_threads": 1}',
            "--loadtest", "1x1", "8x1", "32x1", "64x1", "--loadtest-tokens", "4", "--factory-bench-slots", "64"],
         "timeout": 3600},
        {"kind": "shell", "name": "free_disk", "cmd": "rm -rf /tmp/pgw_bench/qwen05_conc; df -h /tmp", "timeout": 300},
        {"kind": "validate", "label": "qwen15_conc", "args": [
            "--model", QWEN15, "--label", "qwen15_conc", "--device", "cuda", "--ref-device", "cuda",
            "--ring", "K32A12C27", "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "16",
            "--concurrent", "1", "64", "--variant", 'ft_off={"float_threads": null}',
            "--shard-gateways", "2", "--shard-concurrent", "64", "--shard-config", '{"float_threads": 1}',
            "--loadtest", "1x1", "64x1", "--loadtest-tokens", "4", "--factory-bench-slots", "64"],
         "timeout": 3600},
    ]},
    # X4 review run (short): 0.5B only, 1/16/64 chats on one gateway with the new default and the old thread count
    # (ft_off), 64 chats over 2 gateway processes, the service alone at 1 and 64 sessions.
    "pgw-x4-concurrency-short": {"exclude": X4_EXCLUDE, "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; nvidia-smi || true; df -h /tmp", "timeout": 60},
        {"kind": "validate", "label": "qwen05_conc", "args": [
            "--model", QWEN05, "--label", "qwen05_conc", "--device", "cuda", "--ref-device", "cuda",
            "--ring", "K32A12C27", "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "24",
            "--concurrent", "1", "16", "64", "--variant", 'ft_off={"float_threads": null}',
            "--shard-gateways", "2", "--shard-concurrent", "64", "--shard-config", '{"float_threads": 1}',
            "--loadtest", "1x1", "64x1", "--loadtest-tokens", "4", "--factory-bench-slots", "64"],
         "timeout": 3000},
    ]},
    # X4 at 7B (after the float_threads default became 1): 1, 16 and 32 chats on one gateway, 32 chats over 2
    # gateway processes, and the service alone at 1 and 32 sessions. About 22 GB of masks.
    "pgw-x4-concurrency-7b": {"exclude": X4_EXCLUDE, "steps": [
        BUILD_NATIVE,
        {"kind": "shell", "name": "cpu_info", "cmd": "lscpu; nproc; nvidia-smi || true; df -h /tmp", "timeout": 60},
        {"kind": "validate", "label": "qwen7b_conc", "args": [
            "--model", QWEN7, "--label", "qwen7b_conc", "--device", "cuda", "--ref-device", "cuda",
            "--ring", "K32A12C27", "--plain-dtype", "bfloat16", "--delete-hf", "--max-tokens", "16",
            "--concurrent", "1", "16", "32", "--shard-gateways", "2", "--shard-concurrent", "32",
            "--loadtest", "1x1", "32x1", "--loadtest-tokens", "2", "--factory-bench-slots", "64"],
         "timeout": 5400},
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


def source_tarball(exclude: tuple[str, ...] = (), include_top: tuple[str, ...] = ()) -> bytes:
    """xz tarball of pgw/ without the excluded folders (plus ``exclude``: job-specific paths under pgw/;
    ``include_top``: top-level folders of :data:`EXCLUDE_TOP` this job needs after all)."""
    buf = io.BytesIO()
    skip = EXCLUDE_PATHS + tuple(exclude)
    top = EXCLUDE_TOP - set(include_top)

    def filt(ti: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(ti.name).parts
        if len(parts) > 1 and parts[1] in top:
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
    tarballs: dict[tuple[tuple[str, ...], tuple[str, ...]], str] = {}
    for slug, job in JOBS.items():
        if args.only and slug not in args.only:
            continue
        excl = (tuple(job.get("exclude", ())), tuple(job.get("include_top", ())))
        if excl not in tarballs:
            raw = base64.b64encode(source_tarball(*excl)).decode()
            tarballs[excl] = "\n".join(raw[i : i + 100] for i in range(0, len(raw), 100))  # short lines for Kaggle
        tgz = tarballs[excl]
        d = Path(args.out_root) / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "run.py").write_text(template.replace("__JOB__", json.dumps(job)).replace("__PGW_TGZ_B64__", tgz))
        meta = {"id": f"{args.owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": job.get("gpu", True),
                "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
        if meta["enable_gpu"]:
            meta["machine_shape"] = job.get("machine_shape", "NvidiaTeslaT4")
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        print(d, f"{(d / 'run.py').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
