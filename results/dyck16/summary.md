# dyck L=16, eps=0.03, 3 seeds, 60 test points, 20s MILP limit

Mean ± std over seeds. Verified/falsified/timeout are fractions of the verified test points; every point gets one of the three.

| variant | clean acc | IBP-verified | unstable ReLUs | zero weights | MILP verified | falsified | timeout | mean time (s) | median time (s) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.996 ± 0.004 | 0.007 ± 0.006 | 207.2 ± 19.9 | 0.00 ± 0.00 | 0.41 ± 0.06 | 0.07 ± 0.02 | 0.52 ± 0.04 | 19.94 ± 0.03 | 20.03 ± 0.00 |
| rs | 0.996 ± 0.003 | 0.125 ± 0.064 | 24.2 ± 5.1 | 0.00 ± 0.00 | 0.68 ± 0.07 | 0.31 ± 0.07 | 0.01 ± 0.01 | 2.47 ± 1.05 | 1.80 ± 0.45 |
| ternary | 0.997 ± 0.001 | 0.052 ± 0.032 | 178.4 ± 11.2 | 0.37 ± 0.00 | 0.68 ± 0.09 | 0.12 ± 0.02 | 0.21 ± 0.07 | 13.17 ± 0.14 | 13.92 ± 0.49 |
| ternary_rs | 0.990 ± 0.006 | 0.065 ± 0.025 | 51.3 ± 12.0 | 0.41 ± 0.01 | 0.82 ± 0.05 | 0.18 ± 0.05 | 0.00 ± 0.00 | 2.37 ± 0.85 | 1.85 ± 0.52 |
| ibp | 0.922 ± 0.018 | 0.884 ± 0.022 | 58.2 ± 9.1 | 0.00 ± 0.00 | 0.93 ± 0.01 | 0.07 ± 0.01 | 0.00 ± 0.00 | 0.04 ± 0.04 | 0.00 ± 0.00 |
