#!/usr/bin/env python3
"""Numbers for research/reviews/consumer-model-depth.md (stdlib only; no I/O).

    python research/reviews/consumer_depth_numbers.py            # every table, markdown
    python research/reviews/consumer_depth_numbers.py --model qwen3-14b

Model shapes are the public ``config.json`` of each checkpoint (fetched 2026-11-27; Gemma 3 and Llama 3.3 are
gated, their shapes are from the model cards). Quality numbers are copied from the cited reports, never measured
here. Everything else is arithmetic on the shapes:

* sequential non-linear stops per token ``S``: a standard block has two (attention, the SiLU gate), a parallel
  attention+MLP block one; an MoE block the same as its dense form when every expert runs on every token
  (routing-oblivious dispatch, section 5 of the review);
* trips per pass: today ``4L + 1`` (``2L + 1`` for parallel blocks: ``qkv`` and ``gate_up`` share a trip, ``o``
  and ``down`` another); fused (fused2-style residual and norm folding, the other workstream) ``S + 1``;
* tok/s on the RTT alone: ``tau * 1000 / (trips * (RTT + o))``, ``o`` = per-trip overhead beyond the RTT
  (0.25 ms: lane 0.13 + gateway 0.1 at 7B, measured, ``benchmarks/round_trip_model.py``);
* wire per generated token: ring elements in and out of every masked linear, 4 bytes each at K32 (today),
  7 bytes at fused2's K56 ring; ``R`` rows per pass for ``tau`` tokens per pass (strict speculation);
* masks per token on the client: one output word per masked output element per row (``W r``), the same element
  count as the download.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

O_MS = 0.25  # per-trip overhead beyond the RTT (lane + gateway work), 7B measured
SERVE_MS = 2.0  # per pass: sampling, hand-off (measured at 0.5B, rounded up)
RING_BYTES = {"today": 4, "fused": 7}  # K32 today; fused2 needs K56C53 (research/plans/fused2-two-trips-per-layer.md)
TAUS = {1.7: 3, 3.0: 5}  # tokens per pass -> rows per pass (shipped draft head R=3 ~1.64-1.72; EAGLE-class R=5)
RTTS = (5, 20, 50)


@dataclass
class M:
    key: str
    repo: str
    L: int
    d: int
    I: int
    H: int
    KV: int
    hd: int
    V: int
    licence: str
    quality: str  # short, cited
    engine: str  # engine/packer work
    parallel: bool = False
    E: int = 0  # routed experts
    k: int = 0
    Ie: int = 0  # expert intermediate
    Is: int = 0  # shared expert intermediate (total)
    dense_layers: int = 0  # leading dense layers of an MoE
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def q(self) -> int:
        return self.H * self.hd

    @property
    def kv(self) -> int:
        return self.KV * self.hd

    def stops(self) -> int:
        return self.L if self.parallel else 2 * self.L

    def trips_today(self) -> int:
        return (2 if self.parallel else 4) * self.L + 1

    def trips_fused(self) -> int:
        return self.stops() + 1

    def elems(self) -> tuple[int, int]:
        """(up, down) ring elements per row-token over every masked linear, masked LM head included."""
        up = down = 0
        for i in range(self.L):
            up += self.d + self.q + self.d  # qkv in, o in, gate_up (+ router) in
            down += self.q + 2 * self.kv + self.d
            if self.E and i >= self.dense_layers:
                # oblivious: every expert's gate_up on the one masked row; down: one row per expert (unselected =
                # masked zeros) and, with the provider summing the experts' outputs, one d-vector back
                up += self.E * self.Ie + self.Is
                down += self.E * 2 * self.Ie + self.E + 2 * self.Is + self.d
            else:
                up += self.I
                down += 2 * self.I + self.d
        return up + self.d, down + self.V

    def active_elems_down(self) -> int:
        """Download elements if only the routed experts ran (leaks the routing; reference only)."""
        if not self.E:
            return self.elems()[1]
        down = 0
        for i in range(self.L):
            down += self.q + 2 * self.kv + self.d
            if i >= self.dense_layers:
                down += self.k * 2 * self.Ie + self.E + 2 * self.Is + self.d
            else:
                down += 2 * self.I + self.d
        return down + self.V

    def params_b(self) -> float:
        per = 0
        for i in range(self.L):
            per += self.d * (self.q + 2 * self.kv) + self.q * self.d
            if self.E and i >= self.dense_layers:
                per += self.E * 3 * self.d * self.Ie + 3 * self.d * self.Is + self.d * self.E
            else:
                per += 3 * self.d * self.I
        return (per + 2 * self.V * self.d) / 1e9

    def masked_mn_b(self, active: bool = True) -> float:
        """Int8 weight products ``W r`` the mask factory computes per slot (billions of MACs)."""
        per = 0
        for i in range(self.L):
            per += self.d * (self.q + 2 * self.kv) + self.q * self.d
            if self.E and i >= self.dense_layers:
                e = self.k if active else self.E
                per += e * 3 * self.d * self.Ie + 3 * self.d * self.Is + self.d * self.E
            else:
                per += 3 * self.d * self.I
        return (per + self.V * self.d) / 1e9


QW = "Qwen3 tech report (arXiv 2505.09388) Tables 14-20, non-thinking: Arena-Hard v1 / MMLU-Redux"
MODELS = [
    # ---- dense, sequential (standard blocks) ----
    M("llama-3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", 16, 2048, 8192, 32, 8, 64, 128256,
      "Llama 3.2 Community (commercial, 700M MAU cap, AUP)", "MMLU 49.3 (card)",
      "none: llama arch, tied head, llama3 rope already supported; needs a pin", tags=("small",)),
    M("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", 24, 896, 4864, 14, 2, 64, 151936, "Apache-2.0",
      "MMLU-Redux ~ 30s (weak)", "supported (pinned)", tags=("small",)),
    M("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct", 28, 1536, 8960, 12, 2, 128, 151936, "Apache-2.0",
      "AH 9.0 / MR 50.7", "supported (pinned)"),
    M("qwen3-1.7b", "Qwen/Qwen3-1.7B", 28, 2048, 6144, 16, 8, 128, 151936, "Apache-2.0", "AH 36.9 / MR 64.4",
      "qwen3: per-head q/k RMSNorm (client side), no qkv bias"),
    M("llama-3.2-3b", "meta-llama/Llama-3.2-3B-Instruct", 28, 3072, 8192, 24, 8, 128, 128256,
      "Llama 3.2 Community", "MMLU 63.4 (card)", "none (llama); needs a pin"),
    M("qwen2.5-3b", "Qwen/Qwen2.5-3B-Instruct", 36, 2048, 11008, 16, 2, 128, 151936,
      "Qwen Research (NON-commercial)", "AH 23.7 / MR 64.4", "supported (pinned)"),
    M("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", 28, 3584, 18944, 28, 4, 128, 152064, "Apache-2.0",
      "AH 52.0 / MR 75.4", "supported (pinned)"),
    M("qwen3-4b-2507", "Qwen/Qwen3-4B-Instruct-2507", 36, 2560, 9728, 32, 8, 128, 151936, "Apache-2.0",
      "AH 66.2 / MR 77.3 (Qwen3-4B); 2507: AH-v2 43.4", "qwen3"),
    M("llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", 32, 4096, 14336, 32, 8, 128, 128256,
      "Llama 3.1 Community", "AH 30.1 / MR 61.7", "none (llama); needs a pin"),
    M("qwen3-8b", "Qwen/Qwen3-8B", 36, 4096, 12288, 32, 8, 128, 151936, "Apache-2.0", "AH 79.6 / MR 79.5", "qwen3"),
    M("qwen3-14b", "Qwen/Qwen3-14B", 40, 5120, 17408, 40, 8, 128, 151936, "Apache-2.0", "AH 86.3 / MR 82.0",
      "qwen3", tags=("32b-class",)),
    M("phi-4", "microsoft/phi-4", 40, 5120, 17920, 40, 10, 128, 100352, "MIT", "AH 75.4 / MR 85.3 (IFEval 62.1)",
      "phi3: fused qkv_proj / gate_up_proj split in the packer; otherwise llama-like", tags=("32b-class",)),
    M("mistral-small-24b", "mistralai/Mistral-Small-3.2-24B-Instruct-2506", 40, 5120, 32768, 32, 8, 128, 131072,
      "Apache-2.0", "MMLU 80.5, MMLU-Pro 69.1, AH-v2 43.1 (card)",
      "mistral = llama arch with explicit head_dim; 3.1/3.2 ship as a multimodal 'mistral3' wrapper (text tower only)",
      tags=("32b-class",)),
    M("qwen2.5-14b", "Qwen/Qwen2.5-14B-Instruct", 48, 5120, 13824, 40, 8, 128, 152064, "Apache-2.0",
      "AH 68.3 / MR 80.0", "pinned, preview"),
    M("gemma-3-12b", "google/gemma-3-12b-it", 48, 3840, 15360, 16, 8, 256, 262208, "Gemma Terms (custom)",
      "AH 82.6 / MR 77.8", "large: GeGLU, sandwich (post) norms break fused residual folding, sliding windows, "
      "soft caps, scaled embeddings"),
    M("gemma-3-27b", "google/gemma-3-27b-it", 62, 5376, 21504, 32, 16, 128, 262208, "Gemma Terms (custom)",
      "AH 86.8 / MR 82.6", "large (as gemma-3-12b)", tags=("32b-class",)),
    M("qwen2.5-32b", "Qwen/Qwen2.5-32B-Instruct", 64, 5120, 27648, 40, 8, 128, 152064, "Apache-2.0",
      "AH 74.5 / MR 83.9", "pinned, preview", tags=("32b-class",)),
    M("qwen3-32b", "Qwen/Qwen3-32B", 64, 5120, 25600, 64, 8, 128, 151936, "Apache-2.0", "AH 92.8 / MR 85.7",
      "qwen3", tags=("32b-class",)),
    M("qwen2.5-72b", "Qwen/Qwen2.5-72B-Instruct", 80, 8192, 29568, 64, 8, 128, 152064,
      "Qwen licence (commercial < 100M MAU)", "AH 81.2 / MR 86.8", "pinned, preview"),
    # ---- MoE (sequential blocks) ----
    M("gpt-oss-20b", "openai/gpt-oss-20b", 24, 2880, 0, 64, 8, 64, 201088, "Apache-2.0",
      "MMLU 80.4 low / 85.3 high effort, GPQA-D 56.8 / 71.5 (arXiv 2508.10925); reasoning model",
      "large: MoE (oblivious dispatch), attention sinks, alternating 128-token windows, biases, clamped SwiGLU, "
      "MXFP4 weights, harmony format", E=32, k=4, Ie=2880, tags=("32b-class", "moe")),
    M("deepseek-v2-lite", "deepseek-ai/DeepSeek-V2-Lite-Chat", 27, 2048, 10944, 16, 16, 192, 102400,
      "DeepSeek model licence (commercial, use restrictions)", "MMLU 55.7 (card)",
      "large: MLA attention, MoE", E=64, k=6, Ie=1408, Is=2 * 1408, dense_layers=1, tags=("moe",)),
    M("mixtral-8x7b", "mistralai/Mixtral-8x7B-Instruct-v0.1", 32, 4096, 0, 32, 8, 128, 32000, "Apache-2.0",
      "MMLU 70.6 (paper); AH ~ 23", "MoE (oblivious dispatch)", E=8, k=2, Ie=14336, tags=("moe",)),
    M("qwen3-30b-a3b-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507", 48, 2048, 0, 32, 4, 128, 151936, "Apache-2.0",
      "AH 88.0 / MR 84.1 (Qwen3-30B-A3B); 2507: AH-v2 69.0", "qwen3_moe: MoE + q/k norm",
      E=128, k=8, Ie=768, tags=("32b-class", "moe")),
    M("gpt-oss-120b", "openai/gpt-oss-120b", 36, 2880, 0, 64, 8, 64, 201088, "Apache-2.0",
      "MMLU 85.9 low / 90.0 high (arXiv 2508.10925)", "as gpt-oss-20b", E=128, k=4, Ie=2880, tags=("moe",)),
    # ---- native parallel blocks (commercially usable) ----
    M("phi-2", "microsoft/phi-2", 32, 2560, 10240, 32, 32, 80, 51200, "MIT", "MMLU 56.7 (base, no chat tune)",
      "phi: parallel block, GELU, LayerNorm with bias, partial rotary (32 of 80)", parallel=True,
      tags=("parallel",)),
    M("stablelm-2-12b-chat", "stabilityai/stablelm-2-12b-chat", 40, 5120, 13824, 32, 8, 160, 100352,
      "Stability AI Community (commercial < $1M revenue)", "MMLU ~ 62 (card, base 12B)",
      "stablelm: parallel residual, LayerNorm, partial rotary, per-head qk norm", parallel=True,
      tags=("parallel",)),
    M("falcon-11b", "tiiuae/falcon-11B", 60, 4096, 16384, 32, 8, 128, 65024, "Falcon 2 TII licence (Apache-based)",
      "MMLU 58.4 (card, base)", "falcon: parallel attn, LayerNorm, GELU", parallel=True, tags=("parallel",)),
    M("falcon-40b", "tiiuae/falcon-40b", 60, 8192, 32768, 128, 8, 64, 65024, "Apache-2.0", "MMLU ~ 55 (base)",
      "falcon: parallel attn, multi-group", parallel=True, tags=("parallel",)),
    M("gpt-j-6b", "EleutherAI/gpt-j-6b", 28, 4096, 16384, 16, 16, 256, 50400, "Apache-2.0", "MMLU ~ 27 (base)",
      "gptj: parallel, GELU, LayerNorm, partial rotary", parallel=True, tags=("parallel",)),
    M("pythia-12b", "EleutherAI/pythia-12b", 36, 5120, 20480, 40, 40, 128, 50688, "Apache-2.0", "MMLU ~ 26 (base)",
      "gpt_neox: parallel residual, GELU, LayerNorm, partial rotary", parallel=True, tags=("parallel",)),
]


def hypotheticals() -> list[M]:
    """Converted / pruned students (quality is the open question; see the review)."""
    base = {m.key: m for m in MODELS}
    out = []
    for src, L, par, label in (("qwen3-32b", 40, False, "qwen3-32b pruned to 40L"),
                               ("qwen3-32b", 64, True, "qwen3-32b -> parallel"),
                               ("qwen3-32b", 40, True, "qwen3-32b -> parallel + pruned to 40L"),
                               ("qwen3-32b", 32, True, "qwen3-32b -> parallel + pruned to 32L"),
                               ("qwen3-14b", 40, True, "qwen3-14b -> parallel"),
                               ("qwen3-14b", 30, True, "qwen3-14b -> parallel + pruned to 30L"),
                               ("mistral-small-24b", 40, True, "mistral-small-24b -> parallel"),
                               ("gpt-oss-20b", 24, True, "gpt-oss-20b -> parallel"),
                               ("qwen2.5-7b", 28, True, "qwen2.5-7b -> parallel")):
        m = base[src]
        out.append(M(label, m.repo, L, m.d, m.I, m.H, m.KV, m.hd, m.V, m.licence, "open (to be measured)",
                     m.engine + "; + parallel-block variant", parallel=par, E=m.E, k=m.k, Ie=m.Ie, Is=m.Is,
                     dense_layers=m.dense_layers, tags=("hypothetical",)))
    return out


def tok_s(trips: int, rtt: float, tau: float, o: float = O_MS) -> float:
    return tau * 1000.0 / (trips * (rtt + o) + SERVE_MS)


def wire_mb(m: M, tau: float, ring: str) -> tuple[float, float]:
    rows = TAUS[tau]
    up, down = m.elems()
    b = RING_BYTES[ring] * rows / tau / 1e6
    return up * b, down * b


def link_tok_s(m: M, tau: float, ring: str, up_mbit: float, down_mbit: float) -> float:
    u, d = wire_mb(m, tau, ring)
    return 1.0 / max(u * 8 / up_mbit, d * 8 / down_mbit)


def table_main(ms: list[M]) -> None:
    print("| model | L | S | trips today (4L+1) | trips fused (S+1) | params B | " +
          " | ".join(f"tok/s RTT {r} tau {t}" for t in TAUS for r in RTTS) + " | quality | licence |")
    print("|---|---:|---:|---:|---:|---:|" + "---:|" * (len(TAUS) * len(RTTS)) + "---|---|")
    for m in ms:
        cells = [f"{tok_s(m.trips_fused(), r, t):.1f}" for t in TAUS for r in RTTS]
        print(f"| {m.key} | {m.L} | {m.stops()} | {m.trips_today()} | {m.trips_fused()} | {m.params_b():.1f} | "
              + " | ".join(cells) + f" | {m.quality} | {m.licence} |")


def table_today(ms: list[M]) -> None:
    print("| model | trips today | " + " | ".join(f"tok/s RTT {r} tau {t}" for t in TAUS for r in RTTS) + " |")
    print("|---|---:|" + "---:|" * (len(TAUS) * len(RTTS)))
    for m in ms:
        print(f"| {m.key} | {m.trips_today()} | " +
              " | ".join(f"{tok_s(m.trips_today(), r, t):.1f}" for t in TAUS for r in RTTS) + " |")


def table_wire(ms: list[M]) -> None:
    links = (("cable 300/30", 30, 300), ("fiber 1000/1000", 1000, 1000), ("DC 25G", 25000, 25000))
    print("| model | up/down elems per row (M) | MB/token up+down, K32 tau 1.7 | same fused K56 | " +
          " | ".join(f"wire-bound tok/s {n} (fused, tau 1.7)" for n, _, _ in links) +
          " | client masks MB/token (fused, tau 1.7) | factory GMAC/slot |")
    print("|---|---:|---:|---:|" + "---:|" * len(links) + "---:|---:|")
    for m in ms:
        up, down = m.elems()
        a = wire_mb(m, 1.7, "today")
        f = wire_mb(m, 1.7, "fused")
        cells = [f"{link_tok_s(m, 1.7, 'fused', u, d):.1f}" for _, u, d in links]
        masks = m.active_elems_down() * RING_BYTES["fused"] * TAUS[1.7] / 1.7 / 1e6  # routed rows only
        print(f"| {m.key} | {up / 1e6:.2f} / {down / 1e6:.2f} | {sum(a):.1f} | {sum(f):.1f} | " + " | ".join(cells)
              + f" | {masks:.1f} | {m.masked_mn_b():.1f} |")


def needed_rtt(trips: int, tau: float, target: float = 20.0) -> float:
    return (tau * 1000.0 / target - SERVE_MS) / trips - O_MS


def table_target(ms: list[M]) -> None:
    print("| model | trips fused | max RTT for 20 tok/s, tau 1.7 | tau 3 | wire for 20 tok/s down / up (Mbit/s, fused, "
          "tau 1.7) | client mask cores at 20 tok/s (est.) |")
    print("|---|---:|---:|---:|---:|---:|")
    for m in ms:
        u, d = wire_mb(m, 1.7, "fused")
        # client CPU per mask slot scales with the masked outputs (sum_m) at fixed width; 32B measured 112 ms/slot
        # per core at 4.65M outputs/slot (STATUS 2026-11-26, K32); fused K56 rings assumed 1.75x
        ms_slot = 112.0 * (m.active_elems_down() / 4.65e6) * 1.75  # masks cover routed rows only
        slots_per_tok = TAUS[1.7] / 1.7
        cores = 20 * slots_per_tok * ms_slot / 1000
        print(f"| {m.key} | {m.trips_fused()} | {needed_rtt(m.trips_fused(), 1.7):.2f} ms | "
              f"{needed_rtt(m.trips_fused(), 3.0):.2f} ms | {d * 8 * 20:.0f} / {u * 8 * 20:.0f} | {cores:.1f} |")


def moe_table(ms: list[M]) -> None:
    print("| model | E / k | provider FLOPs vs routed | down elems per row: oblivious / routed (leaky) | "
          "factory GMAC/slot routed / all experts |")
    print("|---|---|---:|---:|---:|")
    for m in ms:
        if not m.E:
            continue
        print(f"| {m.key} | {m.E} / {m.k} | {m.E / m.k:.1f}x on expert linears | "
              f"{m.elems()[1] / 1e6:.2f}M / {m.active_elems_down() / 1e6:.2f}M | "
              f"{m.masked_mn_b(True):.1f} / {m.masked_mn_b(False):.1f} |")


#: (name, RTT ms, up Mbit/s, down Mbit/s)
SCENARIOS = (("consumer cable, RTT 20, 300/30", 20.0, 30.0, 300.0),
             ("consumer fiber, RTT 20, 1000/1000", 20.0, 1000.0, 1000.0),
             ("metro edge GPU, RTT 5, 1000/1000", 5.0, 1000.0, 1000.0),
             ("same region (topology B), RTT 0.3, 25G", 0.3, 25000.0, 25000.0))


def combined_tok_s(m: M, tau: float, rtt: float, up_mbit: float, down_mbit: float, ring: str = "fused") -> float:
    """Per pass: trips x (RTT + o) + every byte up / U + every byte down / D + serving + provider device time
    (transfers inside the trips serialize with them; no overlap assumed), divided by the tokens per pass."""
    rows = TAUS[tau]
    up, down = m.elems()
    b = RING_BYTES[ring] * rows
    trips = m.trips_fused() if ring == "fused" else m.trips_today()
    ms = (trips * (rtt + O_MS) + up * b * 8 / (up_mbit * 1e3) + down * b * 8 / (down_mbit * 1e3) + SERVE_MS
          + device_ms(m, ring))
    return tau * 1000.0 / ms


def device_ms(m: M, ring: str = "fused") -> float:
    """Provider GPU time per pass: every int8 weight read once (MoE oblivious: all experts), H100 K32 trips measured at
    0.186 ms per 233 MB (scaling_model.DEVICE_MS_PER_MB); fused2 pair trips 1.3x (measured 1.07-1.30)."""
    return m.params_b() * 1000.0 * (0.186 / 233.0) * (1.3 if ring == "fused" else 1.0)


def table_combined(ms: list[M]) -> None:
    print("| model | trips fused | " + " | ".join(f"{n}, tau 1.7 / 3" for n, *_ in SCENARIOS) + " |")
    print("|---|---:|" + "---:|" * len(SCENARIOS))
    for m in ms:
        cells = [f"{combined_tok_s(m, 1.7, r, u, d):.1f} / {combined_tok_s(m, 3.0, r, u, d):.1f}"
                 for _, r, u, d in SCENARIOS]
        print(f"| {m.key} | {m.trips_fused()} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    a = ap.parse_args()
    ms = MODELS + hypotheticals()
    if a.model:
        ms = [m for m in ms if m.key == a.model]
    print("## Trips and RTT-bound tok/s (fused trips S+1; o = 0.25 ms per trip, 2 ms per pass)\n")
    table_main(ms)
    print("\n## Today's trips (4L+1, 2L+1 parallel), RTT-bound tok/s\n")
    table_today(ms)
    print("\n## Wire per token and link-bound tok/s\n")
    table_wire(ms)
    print("\n## 20 tok/s: what RTT and bandwidth it needs\n")
    table_target(ms)
    print("\n## RTT and wire together (fused trips, K56 ring), tok/s\n")
    table_combined(ms)
    print("\n## MoE: hiding the routing\n")
    moe_table(ms)


if __name__ == "__main__":
    main()
