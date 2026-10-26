# vt — training models that are proved correct

Most AI verification work takes a finished model and asks a solver to prove
something about it, and the solver usually gives up. **vt puts the verifier
inside training.** A model comes out of `vt prove` only with a machine-checked
proof that it computes its specification on **every possible input**, plus a
certificate bound to the exact weights that get deployed.

**Headline results.** Everything here ran on a 4-core CPU.

* **Exact proofs, with no floating-point solver in the trust base, at lengths
  testing can't reach.** With *windowed-uniform* mixing (a few local taps plus
  one shared weight for all older positions, i.e. causal conv + sum
  attention), the exact verifier `vt/exact.py` decides correctness by a
  dynamic program in exact integer arithmetic. It runs in polynomial time in
  L. Results:
  * One trained Dyck-1 (balanced brackets) model is **proved correct at every
    even length 2–64**: 32 certificates; L=64 is 1.8×10¹⁹ inputs; 4 minutes
    end to end.
  * One trained **sortedness** model (V=4) is proved at **every even length
    2–32**: 4³² ≈ 1.8×10¹⁹ inputs at L=32; proved in round 0, about 3
    minutes.
  * The verifier itself proves a structured Dyck network on **2¹⁰²⁴ inputs in
    40 s**.
  * With a **length curriculum** (warm-start from the proved 2–64 model and
    extend), one model is proved exactly at **every even length 2–128**: 64
    certificates; L=128 is 3.4×10³⁸ inputs; 414 s. Training 2–128 from
    scratch did not converge. A third stage extends it to **every even
    length 2–256**: 128 certificates; L=256 is 1.2×10⁷⁷ inputs; exact min
    margin 3.45; 39 min. That model is also proved beyond its training
    range, out to L=274.
  * **Provable length generalisation.** The 2–64 model is proved exactly out
    to L=78, past anything it was trained on. The verifier also pins the
    first failure: L=80, nesting depth 40.
* **Your own specs.** Any regular language, given as a JSON DFA
  (`--spec-file`), is trained and exactly proved the same way. The DFA is
  embedded in the certificate.
* **Weight tying is what makes learned networks provable at all.** With
  general weight-tied (Toeplitz) mixing and the MILP verifier, a learned
  Dyck-1 model is proved on all 2³² inputs in 119 s; brute force would take
  about 49 core-hours. A second weight-tied model is proved at every even
  length 2–30. Untied, position-specific models never got there, after 25+
  counterexample-guided rounds across float, causal, 4-bit and 6-bit variants.
* **Test accuracy is not correctness.** Five standard training recipes all
  reach ≥ 99% held-out accuracy at length 16. Every one of them is wrong on
  80–4,749 of the 65,536 possible inputs.
* **The recipe carries over to GPTs.** For a byte-level GPT, a closed-form
  *neuron-margin loss* halves the binary variables needed to verify each MLP
  block exactly, and makes that verification 10–15× faster, for about 0.5%
  extra LM loss. It costs O(params) per step, so it scales to any model size.
  Ternary (BitNet b1.58) weights compose with it. The trainer is tested with
  2-process DDP (`torchrun`) and runs at the `gpt2-124m` preset.
* **Deployment is gated on proofs.** Certificates bind each verdict to the
  SHA-256 of the exact weights. `vt check` re-verifies independently, and
  `vt serve` refuses to serve an artifact without a matching "proved"
  certificate.

```
pip install -e ".[dev]"
vt prove --spec dyck --length 64 --also-lengths $(seq -s, 2 2 62) --uniform --float --out runs/dyck64  # exact proofs, ~4 min
vt prove --spec dyck --length 32 --shared --float --out runs/dyck32   # MILP proof (~4.5 min, 4 cores)
vt check runs/dyck64/model.npz runs/dyck64/certificate.json           # independent re-check
vt serve runs/dyck64/model.npz runs/dyck64/certificate.json --port 8000
curl -X POST localhost:8000/predict -d '{"inputs": ["(()(()))()()(())((()))()()(())()(()(()))()()(())((()))()()(())()"]}'
```

---

## 1. How `vt prove` works

```
         ┌──────── counterexamples (+ 1-token mutants, + 1-3-token neighbourhood) ─────────┐
         ▼                                                                                  │
  train MixerNet ──► falsify: hard-family sampler ──► clean? ──► MILP proof, per spec case ─┤
  (CE + margin hinge,   (or exhaustive enumeration     │ errors        │ proved            │ counterexample
   annealed lr)          for short lengths)            └───────────────┼───────────────────┘
                                                                       ▼
                                          model.npz (SHA-256) + certificate.json
```

| component | file | what it does |
|---|---|---|
| Model | `vt/model.py` | `MixerNet`: token mixing `s ← (I+M)s` plus per-position ReLU MLPs with residuals. It has no softmax or LayerNorm, so it is piecewise-linear and encodes **exactly** as a MILP. Options: `causal`, `shared` (weight-tied: Toeplitz `M[t,j] = m[t−j]`, pooled + last-position readout), `uniform` + `window` (kernel free on the last `w` offsets, shared beyond: exactly verifiable), `bits` (k-bit QAT), `ternary`. |
| Spec | `vt/global_verify.py` | Specs are encoded as MILP constraints. For each class *c*, the solver looks for an input whose label ≠ *c* but where the model predicts *c* with margin ≥ −τ. If none exists in any case, the model is proved. Specs: `dyck`, `sorted` (V=4). |
| Bounds | `vt/bounds.py` | Symbolic linear bounds, concretised exactly over the one-hot input domain. With interval bounds alone, all 512 neurons of a Dyck-16 model were unstable and the solver timed out. With symbolic bounds the same queries become decidable. |
| Case splitting | `vt/global_verify.py` | Each "label ≠ c" region is split into disjoint, tightly constrained cases. For Dyck: the depth first reaches −1 at even position k, or never dips but ends at ≥ 2. Cases solve in parallel, with early exit on a counterexample. Dyck-16 proof time: 63 s → about 12 s on 4 workers. No-good cuts can collect several counterexamples per case. |
| CEGIS loop | `vt/cegis.py` | Hard-family sampling (deep nesting, single shallow dips), neighbourhood search around verifier counterexamples, per-round cosine-annealed lr, time-limit escalation (60 s × 4ⁿ), QAT schedule, and multi-length training for weight-tied models. Short lengths are decided by exhaustive enumeration, which is itself a complete proof. |
| **Exact verifier** | `vt/exact.py` | For one-block models whose causal kernel is free on the last `w` offsets and constant beyond, position t's output depends only on (its token, the last w−1 tokens, the counts of older tokens). The margin is additive over positions. A DP over (counts, recent tokens, spec DFA state) computes the **exact** worst-case margin over all V^L inputs, in Python big-int arithmetic at scale 2^K (float64 weights are dyadic). Counterexamples are reconstructed by back-pointers and re-checked with an exact rational forward pass. It is polynomial in L for fixed V and w. `--verifier auto` uses it whenever the structure allows. |
| User specs | `vt/dfaspec.py` | `DFASpec`: any regular language from JSON (`specs/*.json`). It validates completeness. The sampler mixes uniform strings, uniformly random *accepted* strings (via suffix-acceptance counts) and one-token near misses. |
| Serving | `vt/serve.py` | `vt serve`: a threaded HTTP server with `/health` and `/predict`, input validation, and a gate that refuses unproved or tampered artifacts. |
| Artifacts | `vt/artifact.py`, `vt/cli.py` | The certificate binds the verdict to the SHA-256 of the float64 `.npz`. `vt predict` runs that file through the forward pass the verifier encoded, so the proved artifact is exactly what is deployed. Change any weight, even by 1e-9, and `vt check` fails. |

### Results (`results/cegis/`, `experiments/summarize_cegis.py`)

| model | L | inputs | outcome |
|---|---|---|---|
| **uniform (w=1), one model, exact** | 2–64 (even) | up to **1.8e19** | **proved exactly at all 32 lengths in round 0** (4 min); 32 certificates (`exact_dyck_L2-64`) |
| **uniform, curriculum (init from 2–64), one model, exact** | 2–128 (even) | up to **3.4e38** | **proved exactly at all 64 lengths in round 1** (414 s) (`exact_dyck_L2-128_curriculum`) |
| **uniform, curriculum (init from 2–128), one model, exact** | 2–256 (even) | up to **1.2e77** | **proved exactly at all 128 lengths in round 2** (39 min) (`exact_dyck_L2-256_curriculum`) |
| uniform, 2–256 model *evaluated beyond training* | 258–400 | — | exactly **proved at 258–274**; first counterexample at L=276 |
| uniform, 2–64 model *evaluated beyond training* | 66–256 | — | exactly **proved at 66–78**; first counterexample at L=80 (depth-40 nesting) |
| uniform, 2–128 from scratch | 2–128 | 3.4e38 | not converged (errors at every length); fixed by the curriculum |
| **user DFA `no_bb`** (no two consecutive b), window 2 | 4–24 | 1.7e7 | **proved exactly at all 6 lengths in round 0** (15 s) |
| user DFA `mod3_a` (#a ≡ 0 mod 3), uniform | 4–24 | 1.7e7 | not proved: the readout must be periodic in a count up to L, which an 8-unit ReLU MLP can't represent (~2L/3 kinks needed) |
| **sortedness, window 2, one model, exact** | 2–32 (even) | up to **1.8e19** | **proved exactly at all 16 lengths in round 0** (194 s) (`exact_sorted_w2_L2-32`) |
| sortedness, window 2, exact | 2–16 | 4.3e9 | not proved: the same configuration with a different data draw got stuck (2,014 errors after 30 rounds; optimisation, not representability) |
| sortedness, uniform w=1 | 2–16 | 4.3e9 | not proved: without a previous-token tap it can't compare neighbours; this motivated windowed kernels |
| dense, float | 16 | 6.6e4 | **proved** in round 0 (63 s verify); exhaustive re-check passes |
| causal, **4-bit** QAT | 16 | 6.6e4 | **proved** in round 3 (11 s verify, 45 s total); min margin 1.67 |
| **weight-tied**, float | 16 | 6.6e4 | **proved** in round 1 (17 s verify) |
| **weight-tied**, float | 32 | **4.3e9** | **proved in round 0** (119 s proof, 230 s total) |
| **weight-tied, multi-length** (one model) | 2–30 | up to 1.1e9 | **proved at all 15 even lengths in round 2** (2–16 enumeration, 18–30 MILP in 9–106 s); 15 certificates; L=22 also confirmed by brute force |
| weight-tied, multi-length 2–32 | 2–32 | 4.3e9 | 2–30 proved in 3 consecutive rounds; L=32 hit the 600 s limit in the joint model (the single-length L=32 model is proved) |
| dense / causal, float | 32 | 4.3e9 | not proved: 25 rounds × 4 attempts plateau at 1–14 rare errors per round |
| causal, 4-bit / 6-bit QAT | 32 | 4.3e9 | not proved: 4-bit too coarse (thousands of errors); 6-bit plateaus like float |
| ternary QAT (dense or causal) | 16 | 6.6e4 | not proved: the STE oscillates (hundreds of errors per round) |
| sortedness (V=4), weight-tied | 16 | 4.3e9 | not proved: a jump at the last position conflicts with the last-position head (d=4, h=8); the wider model's proofs were too slow to finish here |
| weight-tied, float | 64 | 1.8e19 | not converged (d=4, h=8 is likely too small at this length) |
| hand-built integer net | 128 | 3.4e38 | proved in 52 s (L=32: 0.7 s, L=64: 4.5 s) |

What we learned, in order of importance:

1. **Weight tying is what made L=32 work.** Exact counting over 32 positions
   needs the mixing weights to be *exactly* uniform. Position-specific
   parameters get there only approximately, and the leftover slack shows up
   as a few rare errors that move around with every update. Tying the
   weights removes that slack by construction. It also cuts the parameter
   count from O(L²) to O(L) and makes the network's structure uniform, which
   is what lets the solver prove it.
2. Proof engineering decided whether proofs happened at all. Symbolic bounds
   took queries from always timing out to decidable. Spec-guided case
   splitting gave a further 5×.
3. 4-bit QAT helps at L=16, but quantisation alone does not fix the length
   problem. Ternary STE is too unstable for networks this small.
4. The verifier finds errors that sampling misses. At L=32, the falsifier
   repeatedly returned 0 errors from 200k hard-family samples, and the MILP
   then produced a real counterexample.

## 2. Verification-friendly GPTs

`vt/lm.py` is a standard GPT (RMSNorm, causal SDPA attention, ReLU MLP).
`experiments/train_lm.py` trains it from a CPU toy up to GPT-2 sizes:
presets up to `gpt2-774m`, raw-byte or pre-tokenised `uint16` data,
gradient accumulation, `torch.compile`, bf16, DDP via `torchrun`, and
resumable checkpoints. It adds two knobs:

* `--ternary`: every linear layer is a BitLinear with weights in {−a, 0, +a}.
* `--margin-weight`: neuron *i* with `z = w·x + b` cannot change sign under
  any `‖δ‖∞ ≤ ε` iff `|z| > ε‖w‖₁`. The loss is `relu(1 − |z| / (ε‖w‖₁))`.
  It is the closed-form single-layer form of the ReLU-stability loss, so it
  needs no bound propagation through attention.

Tiny Shakespeare, byte-level, 0.87M params, 3,000 steps, ε = 0.05. "Binaries"
and "solve" come from exact MILP verification of each MLP block over the
ε-ball around real validation activations: 16 tokens × 4 layers, 30 s limit
(`experiments/verify_lm_blocks.py`).

| model | val loss (nats) | stable neurons | binaries / block | median solve | solved |
|---|---|---|---|---|---|
| base | 1.519 | 87.2% | 72.5 | 5.58 s | 83% |
| **+ margin** | 1.526 (+0.5%) | **93.5%** | **32.5** | **0.50 s** | **100%** |
| ternary | 1.668 | 83.8% | 105.2 | 8.14 s | 77% |
| **ternary + margin** | 1.703 | **93.1%** | **41.5** | **0.55 s** | **100%** |

Scale-up recipe (GPT-2-small class, nanoGPT `.bin` data):

```bash
torchrun --nproc_per_node 8 experiments/train_lm.py --data fineweb_train.bin --dtype uint16 \
    --preset gpt2-124m --block-size 1024 --batch-size 16 --grad-accum 4 --steps 20000 \
    --lr 6e-4 --ternary --margin-weight 0.5 --compile
python experiments/verify_lm_blocks.py runs/gpt --data fineweb_val.bin
```

Tested here: 2-process DDP with `torchrun` (gloo, CPU) is covered by a CI
test, and a 10.9M-parameter `small`-preset run used DDP with gradient
accumulation. The `gpt2-124m` preset trains end to end on CPU (85.5M
parameters with the byte vocabulary; 124M with the 50K-token BPE vocabulary
via `--dtype uint16`). GPU throughput and convergence at that scale have not
been run in this environment.

What scales and what doesn't: no solver will prove a billion-parameter GPT
correct end to end. The margin loss and ternary weights scale (O(params) per
step), and they make exact per-block, per-token verification cheap.
Global proofs (track 1) apply to small, weight-tied, piecewise-linear modules.
Such modules could serve as verified components alongside an LM, for
example a bracket or format checker.

## 3. Earlier result: local robustness (`results/dyck16`)

Exact local-robustness verification (ε = 0.03) of 512-ReLU Dyck models, 3
seeds. The ReLU-stability loss cuts unstable neurons from 207 to 24 and the
mean MILP time from ≥ 20 s to 2.5 s, at 99.6% accuracy.
Ternary + stability verifies 82% of points with no timeouts.

## Use

```bash
pip install -e ".[dev]"; pytest -q        # 44 tests, about 1.5 minutes on CPU; CI runs them on every push

vt prove   --spec dyck --length 32 --shared --float --out runs/d32 [--also-lengths $(seq -s, 2 2 30)]
vt verify  model.npz --spec dyck --certificate cert.json --workers 4
vt check   model.npz cert.json [--exhaustive]
vt predict model.npz "(()())()(())(())"
```

```python
from vt import CegisConfig, train_until_proved, DyckSpec, verify_global
model, report = train_until_proved(DyckSpec(), CegisConfig(L=32, ternary=False, shared=True))
assert verify_global(model.export(), DyckSpec(), workers=4)["status"] == "proved"
```

Reproduce: `experiments/run_sweep.py` (local robustness),
`experiments/cegis_sweep.sh`, `kbit_runs.sh` and `shared_runs.sh` (MILP proofs),
`experiments/exact_runs.sh`, `exact_sorted_runs.sh` and `exact_curriculum.sh` (exact proofs),
`experiments/lm_sweep.sh` and `verify_lm_blocks.py` (GPT),
`experiments/length_generalization.py`. Each one resumes after an
interruption.

## Soundness and trust

* **Tests compare the verifiers with ground truth.** MILP verdicts must
  agree with exhaustive enumeration (Dyck L=8, sorted V=4 L=5, trained models
  in both outcomes) and with a hand-built correct network. Counterexamples
  must reproduce under a float64 forward pass. Symbolic bounds must contain
  sampled activations. No-good cuts must return distinct real
  counterexamples. Certificates must reject tampered artifacts.
* `vt check --exhaustive` gives an independent second check for small `V^L`.
* **Limits.**
  * **Exact-DP certificates** (`method: exact-dp`) are exact. They depend
    only on Python integer arithmetic and the DFA encoding of the spec, which
    tests check against the spec on every input of small lengths.
  * **MILP certificates** (`method: milp`) are as sound as HiGHS's
    floating-point MILP. The margin τ = 1e-4 and re-checked counterexamples
    guard against tolerance issues, and the tests and `vt check --exhaustive`
    compare the MILP with exact methods. Prefer windowed-uniform models when
    an exact certificate is needed.
  * The specs are small, formal languages.
  * GPT-scale models are verified per block, not globally.

## Related work

ReLU-stability training (Xiao et al. 2019); IBP certified training (Gowal et
al. 2018); MILP verification of ReLU networks (Tjeng et al. 2019); symbolic
bound propagation (DeepPoly, CROWN); BitNet b1.58 (Ma et al. 2024);
counterexample-guided synthesis (Solar-Lezama 2008) and neural-network
repair; compact proofs via mechanistic interpretability (Gross et al. 2024);
RASP / Tracr. What is combined here:
* CEGIS with a *complete* global-correctness verifier over discrete
  sequences,
* exact one-hot bound concretisation and spec-guided case splitting,
* the finding that weight tying is what makes learned networks provable at
  long lengths,
* hash-bound certificates,
* a closed-form stability loss for GPT MLPs.

We have not found this combination elsewhere, but treat that as a claim to
check.
