| run | spec | L | inputs | mixing | weights | status | rounds | verifier calls | last verify (s) | total (s) | certificate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L16_causal_4bit | dyck | 16 | 6.55e+04 | causal | 4-bit | proved | 4 | 1 | 11 | 45 | `b06fb9e8a677` |
| L16_causal_ternary_qat | dyck | 16 | 6.55e+04 | causal | ternary | stopped | 12 | 0 | - | 78 |  |
| L16_float | dyck | 16 | 6.55e+04 | dense | float | proved | 1 | 1 | 63 | 992 | `e1871bb297c1` |
| L16_shared | dyck | 16 | 6.55e+04 | tied (Toeplitz) | float | proved | 2 | 1 | 17 | 710 | `fca91d71729d` |
| L16_shared_v1_fastdecay | dyck | 16 | 6.55e+04 | tied (Toeplitz) | float | not_proved | 15 | 0 | - | 94 |  |
| L2-30_shared | dyck | 2..30 | 1.07e+09 | tied (Toeplitz) | float | proved | 3 | 3 | 254 | 2415 | `d61c44e1effb` |
| L2-32_shared_v1_L32timeout | dyck | 32 | 4.29e+09 | tied (Toeplitz) | float | stopped | 4 | 3 | 1810 | 5714 |  |
| L24_float_v1_nocases | dyck | 24 | 1.68e+07 | dense | float | stopped | 6 | 2 | 601 | 1532 |  |
| L24_float_v2_firstcex | dyck | 24 | 1.68e+07 | dense | float | stopped | 6 | 2 | 271 | 575 |  |
| L24_float_v3_allcex | dyck | 24 | 1.68e+07 | dense | float | stopped | 5 | 1 | 1497 | 2195 |  |
| L32_causal_4bit_v1_noconverge | dyck | 32 | 4.29e+09 | causal | 4-bit | stopped | 10 | 0 | - | 221 |  |
| L32_causal_6bit | dyck | 32 | 4.29e+09 | causal | 6-bit | stopped | 20 | 2 | 300 | 767 |  |
| L32_causal_float | dyck | 32 | 4.29e+09 | causal | float | not_proved | 25 | 3 | 112 | 725 |  |
| L32_causal_float_v1_constlr | dyck | 32 | 4.29e+09 | causal | float | stopped | 10 | 0 | - | 142 |  |
| L32_float | dyck | 32 | 4.29e+09 | dense | float | stopped | 7 | 3 | 266 | 1075 |  |
| L32_float_v1 | dyck | 32 | 4.29e+09 | dense | float | not_proved | 10 | 2 | 900 | 1431 |  |
| L32_float_v2_constlr | dyck | 32 | 4.29e+09 | dense | float | not_proved | 25 | 2 | 58 | 848 |  |
| L32_float_v3_1cex | dyck | 32 | 4.29e+09 | dense | float | stopped | 7 | 3 | 269 | 822 |  |
| L32_shared | dyck | 32 | 4.29e+09 | tied (Toeplitz) | float | proved | 1 | 1 | 119 | 230 | `59b734b3a876` |
| L64_shared_v1_noconverge | dyck | 64 | 1.84e+19 | tied (Toeplitz) | float | stopped | 4 | 0 | - | 237 |  |
| exact_dyck_L2-128_curriculum | dyck | 2..128 | 3.40e+38 | tied (Toeplitz) | float | proved | 2 | 2 | 15 | 414 | `f86bb350f72a` |
| exact_dyck_L2-128_v1_noconverge | dyck | 128 | 3.40e+38 | dense | float | stopped | 1 | 0 | - | 1532 |  |
| exact_dyck_L2-256_curriculum | dyck | 2..256 | 1.16e+77 | tied (Toeplitz) | float | proved | 3 | 2 | 117 | 2326 | `85cddf9871bf` |
| exact_dyck_L2-64 | dyck | 2..64 | 1.84e+19 | tied (Toeplitz) | float | proved | 1 | 1 | 13 | 237 | `22e935d843df` |
| exact_sorted_L2-16_v1_uniform_w1 | dyck | 16 | 6.55e+04 | dense | float | stopped | 23 | 0 | - | 610 |  |
| exact_sorted_w2_L2-16 | dyck | 2..16 | 6.55e+04 | tied (Toeplitz) | float | not_proved | 30 | 0 | - | 672 |  |
| exact_sorted_w2_L2-32 | dyck | 2..32 | 4.29e+09 | tied (Toeplitz) | float | proved | 1 | 1 | 140 | 194 | `6365e8325a48` |
| sorted_L16_shared_v1_small | sorted | 16 | 4.29e+09 | tied (Toeplitz) | float | not_proved | 30 | 13 | 20 | 687 |  |
| sorted_L16_shared_v2_wide | sorted | 16 | 4.29e+09 | tied (Toeplitz) | float | stopped | 6 | 4 | 39 | 1097 |  |
