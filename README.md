# vt — training models that are proved correct

Most AI verification work takes a finished model and asks a solver to prove
something about it, and the solver usually gives up. **vt puts the verifier
inside training.** A model comes out of `vt prove` only with a machine-checked
proof that it computes its specification on **every possible input**, plus a
certificate bound to the exact weights that get deployed.

**Headline results.** Everything here ran on a 4-core CPU.

* **Weight-tied architectures make learned networks provable at scale.** A
  learned Dyck-1 (balanced-bracket) recogniser was **proved correct on all
  2³² = 4.3 billion inputs**. The proof took 119 s; brute force would take
  about 49 core-hours. Untied, position-specific models of the same size
  never got there, after 25+ counterexample-guided rounds across float,
  causal, 4-bit and 6-bit variants. A single weight-tied model trained jointly on lengths 2–32 is proved at **every even length from 2 to 30**: up to 16 by exhaustive enumeration, 18–30 by MILP in 5–273 s each. Length 32 is still in progress.
* **Test accuracy is not correctness.** Five standard training recipes all
  reach ≥ 99% held-out accuracy at length 16. Every one of them is wrong on
  80–4,749 of the 65,536 possible inputs.
* **The proof engine scales when weights are structured.** A hand-built
  integer network is proved correct on 2¹²⁸ inputs in 52 s. The limit is how
  learned weights are structured, not how big the input space is. That is
  what the training recipe targets.
* **The recipe carries over to GPTs.** For a byte-level GPT, a closed-form
  *neuron-margin loss* halves the binary variables needed to verify each MLP
  block exactly, and makes that verification 10–15× faster, for about 0.5%
  extra LM loss. It costs O(params) per step, so it scales to any model size.
  Ternary (BitNet b1.58) weights compose with it.

```
pip install -e ".[dev]"
vt prove --spec dyck --length 32 --shared --float --out runs/dyck32   # train → proof → certificate
vt check runs/dyck32/model.npz runs/dyck32/certificate.json           # independent re-check
vt predict runs/dyck32/model.npz "(()(()))()()(())((()))()()(())()"
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
| Model | `vt/model.py` | `MixerNet`: token mixing `s ← (I+M)s` plus per-position ReLU MLPs with residuals. It has no softmax or LayerNorm, so it is piecewise-linear and encodes **exactly** as a MILP. Options: `causal`, `shared` (weight-tied: Toeplitz `M[t,j] = m[t−j]`, pooled + last-position readout), `bits` (k-bit QAT), `ternary`. |
| Spec | `vt/global_verify.py` | Specs are encoded as MILP constraints. For each class *c*, the solver looks for an input whose label ≠ *c* but where the model predicts *c* with margin ≥ −τ. If none exists in any case, the model is proved. Specs: `dyck`, `sorted` (V=4). |
| Bounds | `vt/bounds.py` | Symbolic linear bounds, concretised exactly over the one-hot input domain. With interval bounds alone, all 512 neurons of a Dyck-16 model were unstable and the solver timed out. With symbolic bounds the same queries become decidable. |
| Case splitting | `vt/global_verify.py` | Each "label ≠ c" region is split into disjoint, tightly constrained cases. For Dyck: the depth first reaches −1 at even position k, or never dips but ends at ≥ 2. Cases solve in parallel, with early exit on a counterexample. Dyck-16 proof time: 63 s → about 12 s on 4 workers. No-good cuts can collect several counterexamples per case. |
| CEGIS loop | `vt/cegis.py` | Hard-family sampling (deep nesting, single shallow dips), neighbourhood search around verifier counterexamples, per-round cosine-annealed lr, time-limit escalation (60 s × 4ⁿ), QAT schedule, and multi-length training for weight-tied models. Short lengths are decided by exhaustive enumeration, which is itself a complete proof. |
| Artifacts | `vt/artifact.py`, `vt/cli.py` | The certificate binds the verdict to the SHA-256 of the float64 `.npz`. `vt predict` runs that file through the forward pass the verifier encoded, so the proved artifact is exactly what is deployed. Change any weight, even by 1e-9, and `vt check` fails. |

### Results (`results/cegis/`, `experiments/summarize_cegis.py`)

| model | L | inputs | outcome |
|---|---|---|---|
| dense, float | 16 | 6.6e4 | **proved** in round 0 (63 s verify); exhaustive re-check passes |
| causal, **4-bit** QAT | 16 | 6.6e4 | **proved** in round 3 (11 s verify, 45 s total); min margin 1.67 |
| **weight-tied**, float | 16 | 6.6e4 | **proved** in round 1 (17 s verify) |
| **weight-tied**, float | 32 | **4.3e9** | **proved in round 0** (119 s proof, 230 s total) |
| **weight-tied, multi-length** (one model) | 2–30 | up to 1.1e9 | **proved at every even length 2–30**, for two consecutive rounds; L=32: counterexample, then solver timeout (in progress) |
| dense / causal, float | 32 | 4.3e9 | not proved: 25 rounds × 4 attempts plateau at 1–14 rare errors per round |
| causal, 4-bit / 6-bit QAT | 32 | 4.3e9 | not proved: 4-bit too coarse (thousands of errors); 6-bit plateaus like float |
| ternary QAT (dense or causal) | 16 | 6.6e4 | not proved: the STE oscillates (hundreds of errors per round) |
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
pip install -e ".[dev]"; pytest -q        # 27 tests, about 1 minute on CPU; CI runs them on every push

vt prove   --spec dyck --length 32 --shared --float --out runs/d32 [--also-lengths 2,4,...,30]
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
`experiments/cegis_sweep.sh`, `kbit_runs.sh` and `shared_runs.sh` (proofs),
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
  * Proofs are as sound as HiGHS's floating-point MILP. The margin τ = 1e-4
    and re-checked counterexamples guard against tolerance issues, but this
    is not an exact-arithmetic certificate. An exact rational re-check is
    the natural next hardening step.
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
