# Training small models to be easy to verify

**Question:** can we add terms to training that make a small model *cheap to
formally verify*, without giving up accuracy?

Today verification is attempted after training and usually fails because
solvers blow up. The cost of exact verification of a ReLU network is driven by
the number of **unstable ReLUs** (neurons that can be either on or off inside
the input region): each one becomes a binary variable in a MILP, and the
search is exponential in them. So we train against that directly, and combine
it with **ternary weights** (`{-a, 0, +a}`, BitNet b1.58-style), which make
the model small/cheap and the MILP sparse.

Everything runs on a CPU.

## Setup

| piece | choice |
|---|---|
| Task | Dyck-1 (balanced brackets), length 16. Hard negatives: equal counts but prefix goes negative, or one flipped token. Class-balanced, train/test disjoint by string. |
| Model | `MixerNet` (`vt/model.py`): transformer-shaped but fully piecewise-linear — linear token mixing `s ← (I+M)s` + per-position ReLU MLP with residuals, 2 blocks, d=8, hidden=16 (512 ReLUs). No softmax/LayerNorm, so the model is *exactly* encodable as a MILP. |
| Property | Local robustness: every input within L∞ distance ε of the one-hot test input gets the same label. |
| Verifier | `vt/verify.py`: interval bound propagation (IBP) first; if inconclusive, an exact big-M MILP solved by HiGHS (scipy). Each point ends as **verified**, **falsified** (counterexample re-checked by a float64 forward pass) or **timeout**. |
| Regularisers | `rs`: ReLU-stability loss `−tanh(1 + l·u)` on IBP pre-activation bounds (Xiao et al. 2019). `ibp`: certified training (Gowal et al. 2018), as a reference point. ε is ramped up over the first half of training. |

Variants: `baseline`, `rs`, `ternary`, `ternary_rs`, `ibp`.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q tests                # soundness checks (~20 s)
python experiments/run_sweep.py          # full sweep, ~30 min on 4 CPU cores
python experiments/run_sweep.py --seeds 1 --n-verify 10 --time-limit 5 --epochs 20   # smoke test
```

Outputs go to `results/<name>/summary.md` and `results.json` (per-sample
verdicts, times, unstable-neuron counts).

## Results

See [`results/dyck16/summary.md`](results/dyck16/summary.md).

RESULTS_PLACEHOLDER

## Soundness

`tests/test_soundness.py` checks that the exported numpy weights reproduce the
torch model, that IBP bounds contain sampled perturbations, and that MILP
verdicts agree with brute-force sampling (verified ⇒ no sampled flip;
falsified ⇒ counterexample is inside the ball and really flips the label),
for both float and ternary models. Verification is sound up to
floating-point tolerance of the solver; it is not a fully rigorous
(exact-arithmetic) certificate.

## Next steps

1. **Scale the claim**: longer sequences, more blocks, a second task
   (modular arithmetic), and plot verified-accuracy vs. solver-time Pareto
   fronts over a sweep of `rs_weight`.
2. **Learned solver-cost loss**: replace the `tanh` proxy with a predictor of
   actual MILP time (or penalise only neurons the MILP branches on).
3. **Tighter bounds** (CROWN / LP-refined big-M) in both training and
   verification — currently plain IBP, which is loose for the baseline.
4. **Global properties**, not just local robustness: e.g. prove "every
   balanced string is accepted" over the whole discrete input space, which is
   where exactly-encodable ternary models should shine.
5. **Attention**: replace linear mixing with hard/piecewise-linear attention
   (e.g. argmax or ReLU attention) and check what the verifier cost does.
