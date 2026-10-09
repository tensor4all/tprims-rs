# faer-limit-sweep

Per-call cost of the two engines the `tprims-contract` planner can choose for a
copy-free fusable contraction, on the same `Problem`, at one and four threads:

```text
D = alpha * A * B          alpha = 1
```

| arm | what one call pays |
| --- | --- |
| `default` | `PlanConfig::default()` — the planner's own route: faer below the dtype's `FaerLimit`, the packed driver above it, elementwise for all-batch |
| `packed` | `PlanConfig::packed()` — the packed TBLIS-style driver, forced |

The question is [tprims-rs#63](https://github.com/tensor4all/tprims-rs/issues/63)
(the size/dtype threshold on the copy-free fusion rule) and its follow-up
[#69](https://github.com/tensor4all/tprims-rs/issues/69): where the crossover
between the two engines lies, per dtype and per thread count, on the classes
that fuse to one strided batched GEMM. This probe only records the two arms;
the `FaerLimit` values and the decision-log row are decided elsewhere.

## Timed boundary

Only `Plan::execute_into` is timed, on a prebuilt `Plan` and a preallocated
output. Building the `Problem`, planning it, building the views and allocating
`A`/`B`/`D` all happen outside the clock, and one arm is measured at a time.

| threads | executor |
| --- | --- |
| 1 | `Exec::serial()`, process pinned to one CPU |
| 4 | a host-owned `rayon::ThreadPool` of 4 workers built once, borrowed through `tprims_exec::Pool::borrow`, handed to `Exec::rayon` |

The binary asserts the pool width, the `Exec` budget and that
`Cpus_allowed_list` holds exactly the requested CPUs, and refuses
`RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`MKL_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` that contradict `--threads`.

Before any timing, every case is checked against a naive label oracle and the
two arms against each other (`rel < 1e-10` for `f64`/`c64`, `< 1e-5` for
`f32`/`c32`). A mismatch aborts the run instead of publishing a number. The
`CHECK` lines are in `results/session*.txt`.

## Shapes

Column-major contiguous; the labels are the ones
`experiments/three-engine-contract/src/main.rs` already uses. All four dtypes
`DType::{F32, F64, C32, C64}` are measured for every shape.

| class | equation | extents |
| --- | --- | --- |
| `gemm` | `D[i,k] = sum_j A[i,j] B[j,k]` | `n` in {32, 64, 128, 256} |
| `gemm_batched` | `D[i,k,b] = sum_j A[i,j,b] B[j,k,b]` | `(n, batch)` in {(16,16), (32,16), (64,16), (32,64), (64,64)} |
| `mps_env` | `ab,asc->bsc` | `A = chi x chi`, `B = chi x 2 x chi`, `D = chi x 2 x chi` |
| `mps_site` | `bsc,bsd->cd` | `A = B = chi x 2 x chi`, `D = chi x chi` |

`chi` is {16, 24, 32, 48, 64, 96, 128} for the two MPS classes. `mnk` in the
table below is the fused `m * n * k` per batch item — the quantity the planner's
`FaerLimit` compares — read from the forced-packed plan's `PlanStats` for both
rows. The MPS steps both have `m * n * k = 2 chi^3`.

`mps_env`'s output extents follow the label order of the equation (`bsc`), so
`D` is `chi x 2 x chi`; the "2 x chi x chi" written in the task text is not the
order the labels give. `results/manifest.txt` records this.

## Run

```bash
bash experiments/faer-limit-sweep/run.sh 3
```

`run.sh` builds once, then runs three sessions; each session is two guarded,
pinned processes — a 1T run on `idle_cpus.py pick 1` and a 4T run on
`idle_cpus.py pick 4` — through `benchmarks/scripts/pinned.sh` of this
checkout. A sub-run whose guard log does not say "idle before and after" is
discarded and retried. Per arm: 500 ms of untimed wall-clock priming, then the
best of 5 calls. `run.sh` prints the summary table below.

## Recorded run

`results/manifest.txt` carries the revision, host, CPU sets, guard and method.
`results/session{1,2,3}.{csv,txt,guard}` are the raw per-session rows (368
each), the `CHECK` lines and the guard logs. All six guarded sub-runs passed at
attempt 1.

`default/packed < 1` means the planner's own route was faster; `> 1` means the
forced-packed arm was faster. `spread` is the larger of the two arms'
`(max-min)/median` over the three sessions.

| threads | class | dtype | params | mnk | default | packed | ns default | ns packed | default/packed | spread |
|---|---|---|---|---:|---|---|---:|---:|---:|---:|
| 1 | `gemm` | f32 | n=32 | 32768 | faer | packed | 742 | 2554 | 0.291 | 8% |
| 1 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1402 | 6412 | 0.219 | 2% |
| 1 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2705 | 7454 | 0.363 | 2% |
| 1 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5299 | 7794 | 0.680 | 1% |
| 1 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5179 | 13475 | 0.384 | 1% |
| 1 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10309 | 22772 | 0.453 | 0% |
| 1 | `gemm` | c32 | n=64 | 262144 | faer | packed | 20508 | 33783 | 0.607 | 1% |
| 1 | `gemm` | c64 | n=64 | 262144 | packed | packed | 61345 | 61225 | 1.002 | 1% |
| 1 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 40375 | 69389 | 0.582 | 0% |
| 1 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 80460 | 128469 | 0.626 | 1% |
| 1 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 160920 | 205905 | 0.782 | 0% |
| 1 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 427688 | 426246 | 1.003 | 1% |
| 1 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 320969 | 397441 | 0.808 | 1% |
| 1 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 644252 | 863060 | 0.746 | 1% |
| 1 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 1284768 | 1353035 | 0.950 | 0% |
| 1 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 2694577 | 2696541 | 0.999 | 1% |
| 1 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2044 | 11732 | 0.174 | 9% |
| 1 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3616 | 9969 | 0.363 | 5% |
| 1 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9658 | 19787 | 0.488 | 1% |
| 1 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 19316 | 19457 | 0.993 | 1% |
| 1 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 11401 | 34965 | 0.326 | 12% |
| 1 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 21901 | 46276 | 0.473 | 1% |
| 1 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 42639 | 62156 | 0.686 | 1% |
| 1 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 84618 | 119092 | 0.711 | 1% |
| 1 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 82453 | 172602 | 0.478 | 2% |
| 1 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 166661 | 236832 | 0.704 | 0% |
| 1 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 329054 | 398994 | 0.825 | 0% |
| 1 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 775457 | 776239 | 0.999 | 1% |
| 1 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 45665 | 121576 | 0.376 | 3% |
| 1 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 88986 | 184013 | 0.484 | 1% |
| 1 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 171691 | 251990 | 0.681 | 1% |
| 1 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 343260 | 484864 | 0.708 | 1% |
| 1 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 332180 | 693653 | 0.479 | 0% |
| 1 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 669970 | 985188 | 0.680 | 1% |
| 1 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1328108 | 1622797 | 0.818 | 1% |
| 1 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 3719421 | 3704311 | 1.004 | 1% |
| 1 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 320 | 1452 | 0.220 | 3% |
| 1 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 461 | 1503 | 0.307 | 3% |
| 1 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 801 | 1884 | 0.425 | 2% |
| 1 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1492 | 2735 | 0.546 | 1% |
| 1 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 881 | 1773 | 0.497 | 1% |
| 1 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1242 | 2324 | 0.534 | 1% |
| 1 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2355 | 5110 | 0.461 | 1% |
| 1 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4648 | 8085 | 0.575 | 1% |
| 1 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1432 | 4378 | 0.327 | 2% |
| 1 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2795 | 5901 | 0.474 | 2% |
| 1 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5390 | 7855 | 0.686 | 1% |
| 1 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10630 | 14117 | 0.753 | 1% |
| 1 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4528 | 7153 | 0.633 | 1% |
| 1 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8867 | 11923 | 0.744 | 0% |
| 1 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 17512 | 28223 | 0.620 | 0% |
| 1 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 41607 | 41587 | 1.000 | 0% |
| 1 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10360 | 20949 | 0.495 | 1% |
| 1 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 20869 | 28463 | 0.733 | 1% |
| 1 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 41237 | 50384 | 0.818 | 1% |
| 1 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 96239 | 96209 | 1.000 | 0% |
| 1 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 34594 | 43852 | 0.789 | 0% |
| 1 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 68899 | 79428 | 0.867 | 0% |
| 1 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 137106 | 153957 | 0.891 | 0% |
| 1 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 301462 | 300501 | 1.003 | 0% |
| 1 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 81512 | 108282 | 0.753 | 1% |
| 1 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 162704 | 203240 | 0.801 | 0% |
| 1 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 323924 | 357547 | 0.906 | 0% |
| 1 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 704134 | 702912 | 1.002 | 2% |
| 1 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 311 | 1723 | 0.180 | 6% |
| 1 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 471 | 1443 | 0.326 | 2% |
| 1 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 801 | 2675 | 0.299 | 1% |
| 1 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1473 | 2695 | 0.547 | 1% |
| 1 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 882 | 2976 | 0.296 | 6% |
| 1 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1292 | 2374 | 0.544 | 2% |
| 1 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2364 | 4879 | 0.485 | 0% |
| 1 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4659 | 8085 | 0.576 | 0% |
| 1 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1453 | 4628 | 0.314 | 9% |
| 1 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2845 | 6222 | 0.457 | 1% |
| 1 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5380 | 8676 | 0.620 | 0% |
| 1 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10790 | 15078 | 0.716 | 0% |
| 1 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4608 | 7404 | 0.622 | 1% |
| 1 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9217 | 12203 | 0.755 | 0% |
| 1 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 17683 | 29024 | 0.609 | 0% |
| 1 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 41257 | 41207 | 1.001 | 0% |
| 1 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 10669 | 22362 | 0.477 | 0% |
| 1 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 21430 | 29575 | 0.725 | 2% |
| 1 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 41657 | 51556 | 0.808 | 0% |
| 1 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 95558 | 95508 | 1.001 | 0% |
| 1 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 35065 | 44844 | 0.782 | 0% |
| 1 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 69700 | 80520 | 0.866 | 0% |
| 1 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 137316 | 156752 | 0.876 | 0% |
| 1 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 302925 | 302194 | 1.002 | 1% |
| 1 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 82805 | 111989 | 0.739 | 1% |
| 1 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 167392 | 207057 | 0.808 | 2% |
| 1 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 326739 | 379267 | 0.862 | 1% |
| 1 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 720635 | 718541 | 1.003 | 2% |
| 4 | `gemm` | f32 | n=32 | 32768 | faer | packed | 751 | 2244 | 0.335 | 14% |
| 4 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1412 | 3266 | 0.432 | 1% |
| 4 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2705 | 4278 | 0.632 | 1% |
| 4 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5300 | 7794 | 0.680 | 1% |
| 4 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5180 | 11161 | 0.464 | 0% |
| 4 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10289 | 14888 | 0.691 | 1% |
| 4 | `gemm` | c32 | n=64 | 262144 | faer | packed | 14838 | 12453 | 1.192 | 13% |
| 4 | `gemm` | c64 | n=64 | 262144 | packed | packed | 21149 | 20809 | 1.016 | 12% |
| 4 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 38672 | 20288 | 1.906 | 59% |
| 4 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 32100 | 42459 | 0.756 | 8% |
| 4 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 57256 | 54432 | 1.052 | 3% |
| 4 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 98614 | 98935 | 0.997 | 2% |
| 4 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 98434 | 142456 | 0.691 | 56% |
| 4 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 185015 | 204431 | 0.905 | 4% |
| 4 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 353960 | 348560 | 1.015 | 0% |
| 4 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 689095 | 688806 | 1.000 | 1% |
| 4 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2024 | 11722 | 0.173 | 8% |
| 4 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3486 | 9968 | 0.350 | 8% |
| 4 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9568 | 19697 | 0.486 | 0% |
| 4 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 19326 | 19065 | 1.014 | 1% |
| 4 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 6883 | 15318 | 0.449 | 11% |
| 4 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 11462 | 20688 | 0.554 | 1% |
| 4 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 14236 | 19657 | 0.724 | 6% |
| 4 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 24606 | 33472 | 0.735 | 1% |
| 4 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 24135 | 47228 | 0.511 | 2% |
| 4 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 44823 | 62657 | 0.715 | 2% |
| 4 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 240599 | 102190 | 2.354 | 7% |
| 4 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 196927 | 196496 | 1.002 | 1% |
| 4 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 14948 | 39303 | 0.380 | 1% |
| 4 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 24997 | 49853 | 0.501 | 4% |
| 4 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 46396 | 65853 | 0.705 | 1% |
| 4 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 88195 | 122498 | 0.720 | 1% |
| 4 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 86341 | 177131 | 0.487 | 1% |
| 4 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 171199 | 243034 | 0.704 | 1% |
| 4 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1018981 | 401940 | 2.535 | 1% |
| 4 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 936949 | 932430 | 1.005 | 1% |
| 4 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 320 | 1433 | 0.223 | 3% |
| 4 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 471 | 1432 | 0.329 | 2% |
| 4 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 791 | 1854 | 0.427 | 2% |
| 4 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1483 | 2695 | 0.550 | 1% |
| 4 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 871 | 1743 | 0.500 | 1% |
| 4 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1242 | 2234 | 0.556 | 2% |
| 4 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2364 | 5050 | 0.468 | 1% |
| 4 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4628 | 8075 | 0.573 | 1% |
| 4 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1433 | 4308 | 0.333 | 1% |
| 4 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2775 | 5830 | 0.476 | 1% |
| 4 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5370 | 7795 | 0.689 | 1% |
| 4 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10630 | 14056 | 0.756 | 0% |
| 4 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4499 | 7094 | 0.634 | 1% |
| 4 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8886 | 11872 | 0.748 | 0% |
| 4 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 13997 | 15338 | 0.913 | 25% |
| 4 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 15569 | 15549 | 1.001 | 5% |
| 4 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10590 | 13195 | 0.803 | 41% |
| 4 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 15950 | 17883 | 0.892 | 26% |
| 4 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 18103 | 19847 | 0.912 | 29% |
| 4 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 33252 | 33302 | 0.998 | 10% |
| 4 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 27621 | 18966 | 1.456 | 14% |
| 4 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 38020 | 30497 | 1.247 | 15% |
| 4 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 63097 | 46426 | 1.359 | 47% |
| 4 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 110596 | 110416 | 1.002 | 0% |
| 4 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 33372 | 35386 | 0.943 | 36% |
| 4 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 54983 | 75881 | 0.725 | 14% |
| 4 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 99025 | 97542 | 1.015 | 17% |
| 4 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 185827 | 186268 | 0.998 | 1% |
| 4 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 311 | 1644 | 0.189 | 4% |
| 4 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 481 | 1343 | 0.358 | 6% |
| 4 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 801 | 2645 | 0.303 | 2% |
| 4 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1472 | 2615 | 0.563 | 2% |
| 4 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 891 | 2925 | 0.305 | 10% |
| 4 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1262 | 2284 | 0.553 | 1% |
| 4 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2344 | 4859 | 0.482 | 1% |
| 4 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4608 | 8015 | 0.575 | 1% |
| 4 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1452 | 4548 | 0.319 | 1% |
| 4 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2805 | 6152 | 0.456 | 1% |
| 4 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5350 | 8596 | 0.622 | 0% |
| 4 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10700 | 14978 | 0.714 | 1% |
| 4 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4548 | 7303 | 0.623 | 1% |
| 4 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9217 | 12102 | 0.762 | 1% |
| 4 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 21671 | 14036 | 1.544 | 10% |
| 4 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 21780 | 21420 | 1.017 | 7% |
| 4 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 12403 | 15659 | 0.792 | 9% |
| 4 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 17803 | 17273 | 1.031 | 51% |
| 4 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 27110 | 21911 | 1.237 | 20% |
| 4 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 33572 | 33433 | 1.004 | 5% |
| 4 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 21671 | 19156 | 1.131 | 89% |
| 4 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 75721 | 28824 | 2.627 | 49% |
| 4 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 97472 | 50294 | 1.938 | 37% |
| 4 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 109975 | 109425 | 1.005 | 2% |
| 4 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 61505 | 40936 | 1.502 | 45% |
| 4 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 59801 | 77695 | 0.770 | 32% |
| 4 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 112179 | 103192 | 1.087 | 2% |
| 4 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 192278 | 192239 | 1.000 | 1% |

Across the three sessions the 1T rows repeat within a median 0.4% / p90 1.9% /
max 12.3% spread; the 4T rows repeat within a median 1.2% / p90 13.8% /
max 89.1%, so 4T differences below that band are not findings.

## What this does not cover

- One host, one `rustc`, one library revision, one session order. This is a
  laptop APU under the `powersave` governor: no fixed frequency and no isolated
  cores, which is where most of the 4T spread comes from.
- Only `execute_into` (the overwrite form); `execute_into_accum` with a
  separate or an in-place `C` is not measured.
- The problem is built with `CSpec::Absent`, so the faer strategy's separate-`C`
  and in-place-`beta` paths are not exercised.
- Whole-shape timings only: no split between packing, the micro-kernel and
  write-back inside either engine, and no counter-based attribution.
- **For `c64` above the provisional limit the two arms are the same engine.**
  The default route already sends `c64` to the packed driver at
  `mnk = 262144` (`FaerLimit::default()` bounds `c64` at `2^17`), so the c64
  rows above that volume compare packed against packed (ratio ~1.00) rather
  than faer against packed. The probe as specified can see the c64 crossover
  only below that volume (`chi <= 32`, `n = 32`); a third arm with
  `FaerLimit::NONE` would be needed to observe it above.
- Routing actually exercised: the default arm chose `faer` with reason `fused`
  for every `f32`/`f64`/`c32` case and for `c64` below the limit, and `packed`
  with reason `above_faer_limit` above it. `NotFusable`, `AllBatch` and the
  elementwise route never appeared (every case is a copy-free fusion).
- Nothing here chooses a `FaerLimit`; the per-dtype values and the
  decision-log row are the main agent's to write from these numbers.
