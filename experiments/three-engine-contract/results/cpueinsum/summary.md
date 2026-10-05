# cpueinsum final benchmark (tprims-rs#62)

Apple M5 Max, macOS 26.5.1, rustc 1.96.0 (ac68faa20 2026-05-25), 1 thread, release (thin LTO, codegen-units 1). cpueinsum-rs 6364da4, tprims-rs main 4a44a7f0 (FaerLimit #65, execute_slices #66). Arms: `ce_exec` = prebuilt EinsumPlan + kept Scratch, views built per program call; `ce_call` = EinsumSpec + einsum_into + fresh output per call. Other arms as in ../README.md. Three sessions, 11 samples of about 20 ms each.

| case | tc_exec | tp_exec | ce_exec | tf_exec | tc_call | tp_call | ce_call | tf_call | ce_call/tc_call |
|---|---|---|---|---|---|---|---|---|---|
| mps_chain_L32_chi4 | 362 | 184 | 108 | 480 | 1369 | 1390* | 1460 | 4766 | 1.07 |
| mps_chain_L32_chi8 | 656 | 331 | 268 | 630 | 1734 | 1548 | 1604 | 4927 | 0.92 |
| mps_chain_L32_chi16 | 2119 | 1759* | 1617 | 2039 | 3321 | 2990 | 3009 | 6254 | 0.91 |
| mps_chain_L32_chi32 | 11197 | 11387 | 11269 | 11795 | 12641 | 12841 | 12677 | 16294 | 1.00 |
| mps_chain_L32_chi64 | 75070 | 75002 | 75751 | 88422 | 77812 | 80318 | 80700 | 93419 | 1.04 |
| ikb_knb_inb_n2_b16 | 907 | 267 | 292 | 642 | 2041 | 1540 | 2019 | 4912 | 0.99 |
| ikb_knb_inb_n4_b16 | 1475 | 317* | 342* | 691 | 2676 | 1646* | 2030 | 4810 | 0.76 |
| ikb_knb_inb_n8_b16 | 2569 | 769 | 784 | 1188 | 3891 | 2074 | 2509 | 5253 | 0.64 |
| ikb_knb_inb_n16_b16 | 6072 | 4627 | 4664 | 5000 | 7711 | 6231 | 6664 | 9161 | 0.86 |
| ikb_knb_inb_n2_b64 | 2915 | 686 | 709 | 1368 | 4158 | 1967 | 2398* | 5467* | 0.58 |
| ikb_knb_inb_n4_b64 | 5110 | 888 | 915 | 1592 | 6402 | 2184 | 2668 | 5734 | 0.42 |
| ikb_knb_inb_n8_b64 | 9393 | 2685 | 2718 | 3584 | 11186 | 4275 | 4681 | 7682 | 0.42 |
| ikb_knb_inb_n16_b64 | 23043 | 18231 | 18216 | 19056 | 25872 | 20881 | 21467 | 23352 | 0.83 |
| ikb_knb_inb_n2_b256 | 10890 | 2374 | 2330 | 4234 | 12585 | 3626 | 4067 | 8314 | 0.32 |
| ikb_knb_inb_n4_b256 | 19883 | 3150 | 3199 | 5214 | 21906 | 4843 | 5282 | 9550 | 0.24 |
| ikb_knb_inb_n8_b256 | 36835 | 10444 | 10447 | 13324 | 40307 | 13009 | 13509* | 17566 | 0.34 |
| ikb_knb_inb_n16_b256 | 90794 | 72422 | 72284 | 74233 | 96041 | 76796 | 77361 | 78950 | 0.81 |
| ijk_jkl_il_8x16x8 | 1895 | 434 | 458 | 716 | 3038 | 1613 | 2044 | 4916 | 0.67 |
| ij_jk_ik_c64_n32 | 6190 | 5372 | 5400 | 5565 | 7385 | 6557 | 7050 | 9153 | 0.95 |
| ij_jk_ik_f64_n64 | 12303 | 10183 | 10165 | 10432 | 14138 | 11784 | 12142 | 14271 | 0.86 |
| ij_jk_kl_il_n64 | 12136 | 10130 | 10117 | 10431 | 13668 | 11698 | 12064 | 14085 | 0.88 |

ns per pairwise step (session median of per-call medians); * = cross-session spread > 10%
