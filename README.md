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
* **Proved components give guarantees about GPT outputs.** A GPT can't be
  proved, but what it emits can be. A small GPT trained on bracket strings
  produced 11.7% invalid strings (2,343 of 20,000 samples). Filtering with
  the *proved* guard (`vt/guard.py`) passed 0 invalid strings and rejected 0
  valid ones, and its certificate rules out leaks on every one of the 2¹⁶
  inputs. An *unverified* guard with 99.6% held-out accuracy leaked an
  invalid string in the same sample; over all inputs it accepts 647 invalid
  strings (`experiments/guarded_generation.py`).
* **Deployment is gated on proofs.** Certificates bind each verdict to the
  SHA-256 of the exact weights. `vt check` re-verifies independently, and
  `vt serve` refuses to serve an artifact without a matching "proved"
  certificate. The server is hardened: body-size cap, socket timeouts,
  strict parsing, fuzz-tested, no 5xx on 300 random bodies. On 4 cores it
  handles **392 req/s, i.e. 12.6k sequences/s, with p99 latency of 83 ms**
  (`experiments/bench_serve.py`). It ships as a torch-free, non-root Docker
  image with a healthcheck. CI builds the image, serves a certified model,
  and checks that a tampered one is refused.

```
pip install -e ".[dev]"
vt prove --spec dyck --length 64 --also-lengths $(seq -s, 2 2 62) --uniform --float --out runs/dyck64  # exact proofs, ~4 min
vt prove --spec dyck --length 32 --shared --float --out runs/dyck32   # MILP proof (~4.5 min, 4 cores)
vt check runs/dyck64/model.npz runs/dyck64/certificate.json           # independent re-check
vt serve runs/dyck64/model.npz runs/dyck64/certificate.json --spec dyck --port 8000   # re-proves on startup
curl -X POST localhost:8000/predict -d '{"inputs": ["(()(()))()()(())((()))()()(())()(()(()))()()(())((()))()()(())()"]}'
```

---

## VeriChat: a chatbot that reports how sure it is, and where each claim came from

The second half of this repo applies the same idea, verification built into the model, to ordinary
open-weight chat models (Qwen2.5 0.5B–14B, DeepSeek-R1-Distill). Results with confidence intervals
are in `VISION.md`.

* **Every claim scored for free.** A linear probe reads the hidden state the model has already
  computed when it writes a name, number or date, and gives P(correct). It costs no extra pass
  (measured: about 0% overhead). On Qwen2.5-7B, across five formats (short answers, lists, multi-turn,
  chain-of-thought, GSM8K), it ranks claims at AUROC 0.856, against 0.760 for token probability. In
  the most confident quarter of its claims, 7% are wrong, against 18%.
* **Exact provenance.** With the model's decisions fixed (norm scales, attention patterns, SiLU
  gates), the network is linear. One extra pass splits each claim's logit exactly between the
  documents, the question and the model's own memory, and cites the document sentence that
  contributed most.
* **Doubt-triggered behaviour.** It can withhold or mark a doubted claim before writing it. It can
  also look the question up only when it doubts itself: at half the retrievals it gets 83% of the
  gain of always retrieving, and beats token-probability triggers.
* **Judge any model's answer.** A frozen 3B reader with probes scores replies written by other models
  (GPT, Claude, ...). It gets RAGTruth AUROC 0.86 (F1 0.71) and 0.91 over six benchmarks, where
  output log-probability is at chance. It needs in-domain training data: zero-shot on unseen domains
  it gets 0.61. A 7B reader is no better on the same sources. A 7B head trained on ten sources
  (`results/kaggle/detector-7b2/`) adds numerical reasoning (0.92) and biomedical QA (0.85); `vt adapt`
  fits a new domain from labelled replies.

```bash
pip install -e ".[verichat]"
vt verichat --model Qwen/Qwen2.5-7B-Instruct --head results/kaggle/claims-7b/claims/Qwen2.5-7B-Instruct_claims_head.pt \
    --detector results/kaggle/detector-3b/detector/Qwen2.5-3B-Instruct_detector.pt [--corpus my_docs/] --port 8080
#   GET /  chat page    POST /chat {"messages": [...], "documents": [...], "mode": "hedge"|"stop"|"flag"}
#     "stream": true -> server-sent events of checked text; "thinking": true -> a reasoning model (Qwen3)
#     thinks first; its <think> block is streamed and returned apart ("reasoning"), never scored or
#     counted as the reply (--thinking makes it the default, --think-budget caps it, default 1024 tokens)
#     sampling: Qwen3's advice by default (--temperature 0.7 --top-p 0.8 --top-k 20; thinking 0.6/0.95/20;
#     --temperature 0 for greedy); "temperature" and "seed" per request; --presence-penalty against loops
#   POST /judge {"messages": [...], "response": "an answer from any model", "documents": [...]}
vt judge --detector results/kaggle/detector-3b/detector/Qwen2.5-3B-Instruct_detector.pt \
    --question "Who designed the Eiffel Tower?" --answer "Thomas Edison, in 1920." --doc source.txt
modal deploy deploy/modal_app.py                      # public demo on a Modal GPU (scales to zero; modal>=1.0)
docker build -f Dockerfile.verichat -t verichat .     # CPU image, heads included; built and health-checked
# public deployments: --rate-per-minute N (default 20 per client); API-only: VERICHAT_API_KEY=... (Bearer token)
```

```python
from vt.verichat import VeriChat
bot = VeriChat.load("Qwen/Qwen2.5-7B-Instruct", "results/kaggle/claims-7b/claims/Qwen2.5-7B-Instruct_claims_head.pt")
out = bot.reply([{"role": "user", "content": "When was the Eiffel Tower completed?"}],
                documents=["The Eiffel Tower was completed in 1889 ..."], mode="hedge")
for c in out["claims"]:
    print(c["text"], c["verdict"], round(c["score"], 2), c.get("source"), c.get("citation"))
```

Heads are trained on Kaggle GPUs by `kaggle/claims/run.py` (claim probe, any model) and
`kaggle/detector/run.py` (general detector), and evaluated with bootstrap CIs and leave-one-out
splits.

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
| Guards | `vt/guard.py` | `Guard(artifact, certificate)`: accept or reject candidate outputs with a proved classifier; refuses unproved guards unless told otherwise. |
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

## 3. Certified generators

`vt gen-train` trains a generator and proves its support exactly: it never
emits an invalid string (sound), it can emit every valid one (complete), and
it never gets stuck (live). Count-state generators (`vt/gen.py`) use
relu-normalised probabilities, so invalid continuations get *exactly* zero
mass. A softmax generator can never be sound. The support loss covers every
live state of the enumerable product graph, not a sample of it.

* Dyck-1: proved at all 32 lengths 2–64 (`results/gen/dyck_L2-64`).
* `no_bb`, a JSON DFA: proved at all 16 lengths 2–32.
* Sortedness, window 2: proved at only 3 of 8 lengths.

The verifier also computes the exact violation probability of unsound
generators, as a rational. Two extensions:

* **Beyond regular languages.** Counter-automaton specs
  (`vt/counterspec.py`, `specs/anbn.json`, `specs/anbncn.json`). A learned
  network is proved exactly to recognise aⁿbⁿcⁿ, which is not context-free.
* **Typed nesting.** Stack-state models (`vt/stack.py`) for Dyck-k with
  bounded depth, the JSON-nesting milestone. Their exact verifiers agree with
  brute force; a Dyck-2 generator is certified at L = 4, 8, 12.

## 4. The semantic layer: truthfulness, reasoning, harm, tone

These properties have no formal definition, so vt designs the model so they
become checkable, and labels every guarantee with its strength:

* **Interpretability by design** (`vt/epistemic.py`). Named
  residual-stream channels, such as "grounded", are trained on signals
  computable during training. Intervention training makes them causally
  load-bearing. On grounded answering, the built-in channel beats a post-hoc
  probe of the same model:

  | measure | post-hoc probe | built-in channel |
  |---|---|---|
  | accuracy | 29% | 43% |
  | hallucinated | 47% | 29% |
  | flags its own errors (AUROC) | 0.15 | 0.71 |

  The channel is causal in both directions (`results/grounding_by_design_v2_single.json`).
* **Provenance by architecture** (`experiments/provenance_by_design.py`).
  Fact-bearing tokens can only be copied from context, and every copy's
  source is checked exactly. **0 false answers released, both in-distribution
  and under distribution shift**, with every correct answer kept.
* **Plan → Check → Realize** (`vt/pcr`, `vt respond`). A planner emits a
  structured plan:
  * claims are checked against a knowledge base;
  * reasoning steps are evaluated exactly by a whitelist interpreter, never
    `eval`;
  * the declared intent is checked against policy;
  * a realizer states only checked items in an approved register;
  * an independent verifier confirms the text is in the licensed language;
  * subjective properties use conformally calibrated judges (`vt/pcr/judge.py`).

  With a simulated noisy planner, false claims released fell from 932 to 0,
  wrong arithmetic from 406 to 0, and harmful responses from 102 to 0, with
  all correct content kept.

* **Glass-box LM** (`vt/glassbox.py`): a transformer built to be read
  mechanistically. It has an evidence stream (an exact account of how
  content was mixed from each input position and memory read), a starved core
  with a product-key memory as the only fact store, and a self-knowledge
  variable k trained on the model's own correctness that alone drives
  abstention.

  On a synthetic knowledge world (`experiments/glassbox_world.py`, six
  versions):
  * the explicit memory answers facts it was never asked about 3-4x better
    than a matched GPT;
  * k matches or beats post-hoc probes, by more under shift, but only reaches
    0.66-0.75 AUROC;
  * placing the read after the last attention block makes the memory record
    causal (64% of answers follow a patched read);
  * hallucinations turned out to be reads that fall onto heavily used "hub"
    slots;
  * abstention on never-seen facts was not learned.

  `experiments/glassbox_chat.py` runs the same comparison as a general chat
  model (Alpaca, WikiText-2 and a fact world in natural language).

* **A chatbot with its own verification report** (`vt/chat_serve.py`). `vt chat`
  and `vt chat-serve` serve a model whose every reply carries per-token
  confidence, low-confidence spans and abstention. Glass-box models also report
  the evidence split per token: from the prompt, from the reply so far, or from
  memory. Packages are pinned by sha256 and refused if any file changes.

  Backends:
  * glass-box chat checkpoints;
  * Glassbox-2/3 packages (`vt-glassbox2-chat/1`, `vt-glassbox3-chat/1`), such as
    the student trained on Kaggle by `kaggle/distill/`. Per token they report
    confidence, an exact certificate that the token was made without stored
    memory, and the most active features. Glassbox-3 also reports its epistemic
    uncertainty u and whether the reply opened with `<unsure>`;
  * a pretrained open model retrofitted on a GPU by
    `kaggle/verifiable_chat/train.py`: LoRA on Qwen2.5-1.5B-Instruct with
    self-labelled abstention, made-up-name training and a confidence head.

  `kaggle/launch.sh` runs the retrofit as a private Kaggle T4 kernel.
  `kaggle/distill/launch.sh` trains glass-box students from scratch on Kaggle:
  kernel 1 writes teacher data with Qwen2.5-1.5B-Instruct, and kernel 2 trains
  a Glassbox-2 and a GPT student on it. Then
  `bash kaggle/distill/fetch_package.sh` downloads the trained package, and
  `vt chat-serve results/kaggle/glassbox-distill/glassbox_package` serves it.

Limits, measured: these are small CPU models on synthetic tasks. Under
distribution shift the base model's capability collapsed, and statistical
bounds hold only when calibration matches deployment. See VISION.md for the
roadmap.

## 5. Earlier result: local robustness (`results/dyck16`)

Exact local-robustness verification (ε = 0.03) of 512-ReLU Dyck models, 3
seeds. The ReLU-stability loss cuts unstable neurons from 207 to 24 and the
mean MILP time from ≥ 20 s to 2.5 s, at 99.6% accuracy.
Ternary + stability verifies 82% of points with no timeouts.

## Use

```bash
pip install -e ".[dev]"; pytest -q        # 115 tests, about 4 minutes on CPU; CI runs them on every push

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

## Scope: what this is and isn't

* **It is** a working system for training small sequence models that come
  with machine-checked global-correctness certificates. Those certificates
  are exact for the windowed-uniform architecture class, which is what
  `--uniform` trains. Certified artifacts can be deployed and served under a
  proof gate, and used as proved guards around unverified generators such as
  GPTs.
* **It isn't** a proof that a large language model is correct. No current
  method proves end-to-end properties of billion-parameter networks. Here
  the GPT side gets a cheap, scalable *verification-friendliness* recipe
  (margin loss + ternary weights; exact per-block checks) plus output
  guarantees via proved guards.
* **Not done in this environment:**
  * GPU-scale training: CPU only; the code paths are tested with DDP on
    CPU and at the gpt2-124m preset.
  * Exact proofs for multi-block or arbitrary mixing: these get labelled MILP
    certificates instead.
  * Specs that need periodic counting, such as `mod3_a`: training collapses
    to a constant even at width 32, the known parity/modular-counting
    difficulty.

## Security model

The **verifier code is the root of trust, not the certificate.** A
certificate is just JSON, so `vt serve`, `vt check` and `Guard` never accept
one on its own word. Before serving, they:
1. read the artifact once and hash exactly those bytes, so the file can't be
   swapped between hashing and loading (no TOCTOU);
2. reject compressed or oversized zip members and validate every array shape;
3. require the certificate's spec to be consistent (an embedded DFA must
   match its name), and require it to match the spec the operator pinned
   (`--spec`, `--spec-file`). A certificate can't decide what "correct"
   means; DFA specs must always be pinned;
4. **re-prove the artifact** against that spec with vt's own verifier. With
   exact certificates this takes seconds.

The optional fast path is a certificate signed with Ed25519 (`vt keygen`,
`vt sign`, `--trusted-key`). `vt sign` re-verifies before it signs, so a
signature means "the signer's verifier proved this artifact".

The server limits:
* body size (413), and requests without a Content-Length get 411;
* JSON nesting, checked before parsing;
* concurrent connections (503 beyond `--max-concurrency`);
* time per socket operation and per connection (`--deadline`), so
  trickling clients are cut off.

An independent review of the serving path found seven issues: a forgeable
certificate, a hash/load TOCTOU, a spec swap via an embedded DFA, unbounded
threads and slow-client holding, a `RecursionError` on deep JSON, a zip-bomb
risk, and image/CI pinning. All have been fixed, and each has a regression
test. CI runs `bandit` and `pip-audit`. The Docker image pins the base image
by digest and the dependencies by version. The CI token is read-only. CI also
checks that the container refuses both a tampered artifact and a forged
certificate.

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
