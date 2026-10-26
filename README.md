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

Dyck-1, L=16, ε=0.03, 3 seeds × 60 test points, 20 s MILP limit, 4 CPU cores
(mean ± std over seeds):

| variant | clean acc | IBP-verified | unstable ReLUs | zero weights | MILP verified | falsified | timeout | mean time (s) | median time (s) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.996 ± 0.004 | 0.007 ± 0.006 | 207.2 ± 19.9 | 0.00 ± 0.00 | 0.41 ± 0.06 | 0.07 ± 0.02 | 0.52 ± 0.04 | 19.94 ± 0.03 | 20.03 ± 0.00 |
| rs | 0.996 ± 0.003 | 0.125 ± 0.064 | 24.2 ± 5.1 | 0.00 ± 0.00 | 0.68 ± 0.07 | 0.31 ± 0.07 | 0.01 ± 0.01 | 2.47 ± 1.05 | 1.80 ± 0.45 |
| ternary | 0.997 ± 0.001 | 0.052 ± 0.032 | 178.4 ± 11.2 | 0.37 ± 0.00 | 0.68 ± 0.09 | 0.12 ± 0.02 | 0.21 ± 0.07 | 13.17 ± 0.14 | 13.92 ± 0.49 |
| ternary_rs | 0.990 ± 0.006 | 0.065 ± 0.025 | 51.3 ± 12.0 | 0.41 ± 0.01 | 0.82 ± 0.05 | 0.18 ± 0.05 | 0.00 ± 0.00 | 2.37 ± 0.85 | 1.85 ± 0.52 |
| ibp | 0.922 ± 0.018 | 0.884 ± 0.022 | 58.2 ± 9.1 | 0.00 ± 0.00 | 0.93 ± 0.01 | 0.07 ± 0.01 | 0.00 ± 0.00 | 0.04 ± 0.04 | 0.00 ± 0.00 |

**What this shows**

- **Exact verification of the plain model mostly fails.** Even at 99.6%
  clean accuracy and only 512 ReLUs, 52% of baseline points hit the 20 s
  timeout. Its true robust accuracy is somewhere between 41% and 93%, and we
  can't say where.
- **The ReLU-stability loss makes the model decidable** at no accuracy cost.
  Unstable ReLUs drop 207 → 24. The solver settles 99% of points, at a mean
  of 2.5 s versus ≥ 20 s. That speedup is a lower bound, because baseline
  times are capped by the timeout.
- **Ternary weights alone help a little** (37% of weights become exactly 0,
  timeouts fall from 52% to 21%). They help most combined with RS:
  `ternary_rs` has the most exactly-verified points of the high-accuracy
  models (82%), no timeouts, and 99.0% clean accuracy.
- **IBP certified training is the reference to beat.** It certifies 88% of the
  full test set with IBP alone (93% exactly on the 60-point subset). It costs
  about 7 points of clean accuracy (92.2%). `ternary_rs`
  sits at a different point on that trade-off: higher accuracy, exact
  verdicts on every point, and a ternary model.
- Caveats: one small task, local robustness in one-hot input space, 60
  points per run, and big-M constants from plain IBP. Of the RS models'
  verdicts, 31% are counterexamples. This doesn't show they are less robust
  than the baseline, whose true falsified rate is hidden by its timeouts.

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
