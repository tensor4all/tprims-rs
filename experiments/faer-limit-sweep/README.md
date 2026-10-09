# faer-limit-sweep

Per-call cost of the two engines the `tprims-contract` planner can choose for a
copy-free fusable contraction, on the same `Problem`, at one and four threads:

```text
D = alpha * A * B          alpha = 1
```

Two C modes are recorded for every case:

| `c_mode` | problem | timed call |
| --- | --- | --- |
| `absent` | `CSpec::Absent`, a zero `D` | `Plan::execute_into` — `D = alpha * A * B`, overwriting |
| `output` | `CSpec::Output(Op::Identity)`, a non-zero `D` | `Plan::execute_into_accum` with `beta = 1` and `AccumulationSource::Output` — `D += alpha * A * B` in place |

`output` is the form an MPS step (`D += A*B`) pays; `absent` is the overwrite
form the earlier recorded run used.

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

Only the plan execution is timed, on a prebuilt `Plan` and a preallocated
output: `Plan::execute_into` in the `absent` mode and
`Plan::execute_into_accum(.., AccumulationSource::Output)` in the `output` mode.
Building the `Problem`, planning it, building the views and allocating
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
`CHECK` lines are in `results/cmode-session*.txt` (both C modes) and in the
earlier `results/session*.txt` (`absent` only).

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
best of 5 calls. The one binary records both C modes for every case; each
sub-run writes `results/cmode-session{s}.{csv,txt,guard}`, and `run.sh` prints
the summary table below.

## Recorded runs

`results/manifest.txt` carries the revision, host, CPU sets, guard and method.

There are two recorded runs. The earlier one measured only the `absent` mode:
`results/session{1,2,3}.{csv,txt,guard}` are its raw per-session rows (368
each), its `CHECK` lines and its guard logs. The new one measures both C modes:
`results/cmode-session{1,2,3}.{csv,txt,guard}` are its raw per-session rows (736
each: 23 shapes x 4 dtypes x 2 arms x 2 thread counts x 2 C modes), its `CHECK`
lines (368 per session) and its guard logs. All six guarded sub-runs of each run
passed at attempt 1.

The `absent` rows of the new run re-measure the same configuration as the
earlier run (at the newer library revision named in `results/manifest.txt`), so
the earlier `absent` table and the new run's `absent` table are comparable; the
`output` rows are the new dimension.

### Earlier run (overwrite `absent` only)

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



### New run (both C modes)

The new run re-measured the `absent` rows and added the `output` mode, on the
library revision named in `results/manifest.txt`. Both modes' rows are below;
`default/packed < 1` means the planner's own route was faster, `> 1` the
forced-packed arm.

#### `absent` (overwrite)

| c_mode | threads | class | dtype | params | mnk | default | packed | ns default | ns packed | default/packed | spread |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|
| absent | 1 | `gemm` | f32 | n=32 | 32768 | faer | packed | 741 | 2565 | 0.289 | 0% |
| absent | 1 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1403 | 6532 | 0.215 | 4% |
| absent | 1 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2705 | 7494 | 0.361 | 3% |
| absent | 1 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5300 | 7815 | 0.678 | 1% |
| absent | 1 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5190 | 13425 | 0.387 | 9% |
| absent | 1 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10339 | 22713 | 0.455 | 0% |
| absent | 1 | `gemm` | c32 | n=64 | 262144 | faer | packed | 20529 | 33764 | 0.608 | 1% |
| absent | 1 | `gemm` | c64 | n=64 | 262144 | packed | packed | 60654 | 60544 | 1.002 | 2% |
| absent | 1 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 40445 | 69349 | 0.583 | 2% |
| absent | 1 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 80500 | 129040 | 0.624 | 1% |
| absent | 1 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 161093 | 205903 | 0.782 | 0% |
| absent | 1 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 426856 | 426475 | 1.001 | 0% |
| absent | 1 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 321239 | 396930 | 0.809 | 1% |
| absent | 1 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 646715 | 861857 | 0.750 | 1% |
| absent | 1 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 1286609 | 1353624 | 0.950 | 0% |
| absent | 1 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 2703971 | 2692579 | 1.004 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2084 | 11161 | 0.187 | 64% |
| absent | 1 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3496 | 9889 | 0.354 | 28% |
| absent | 1 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9558 | 19997 | 0.478 | 0% |
| absent | 1 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 19336 | 19155 | 1.009 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 11361 | 34214 | 0.332 | 1% |
| absent | 1 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 21781 | 45715 | 0.476 | 1% |
| absent | 1 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 42719 | 61895 | 0.690 | 0% |
| absent | 1 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 85129 | 119074 | 0.715 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 82604 | 170800 | 0.484 | 0% |
| absent | 1 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 166270 | 236715 | 0.702 | 0% |
| absent | 1 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 328346 | 395197 | 0.831 | 1% |
| absent | 1 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 774023 | 773873 | 1.000 | 0% |
| absent | 1 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 45806 | 135765 | 0.337 | 11% |
| absent | 1 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 88406 | 183612 | 0.481 | 2% |
| absent | 1 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 172171 | 247442 | 0.696 | 0% |
| absent | 1 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 343175 | 482540 | 0.711 | 1% |
| absent | 1 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 333011 | 686599 | 0.485 | 1% |
| absent | 1 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 669979 | 991923 | 0.675 | 2% |
| absent | 1 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1331532 | 1626492 | 0.819 | 2% |
| absent | 1 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 3751546 | 3770130 | 0.995 | 2% |
| absent | 1 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 320 | 1432 | 0.223 | 6% |
| absent | 1 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 461 | 1683 | 0.274 | 5% |
| absent | 1 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 792 | 1983 | 0.399 | 1% |
| absent | 1 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1492 | 2765 | 0.540 | 1% |
| absent | 1 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 871 | 1883 | 0.463 | 1% |
| absent | 1 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1252 | 2414 | 0.519 | 4% |
| absent | 1 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2364 | 5290 | 0.447 | 1% |
| absent | 1 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4629 | 8146 | 0.568 | 1% |
| absent | 1 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1443 | 4448 | 0.324 | 2% |
| absent | 1 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2785 | 5851 | 0.476 | 1% |
| absent | 1 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5370 | 7985 | 0.673 | 0% |
| absent | 1 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10740 | 14206 | 0.756 | 1% |
| absent | 1 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4528 | 7303 | 0.620 | 0% |
| absent | 1 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8937 | 12043 | 0.742 | 0% |
| absent | 1 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 17583 | 28433 | 0.618 | 0% |
| absent | 1 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 41558 | 41499 | 1.001 | 0% |
| absent | 1 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10410 | 21020 | 0.495 | 0% |
| absent | 1 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 20899 | 28754 | 0.727 | 1% |
| absent | 1 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 41298 | 50515 | 0.818 | 1% |
| absent | 1 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 95940 | 95960 | 1.000 | 1% |
| absent | 1 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 34595 | 43833 | 0.789 | 0% |
| absent | 1 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 68939 | 79579 | 0.866 | 1% |
| absent | 1 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 136989 | 154238 | 0.888 | 0% |
| absent | 1 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 300097 | 300230 | 1.000 | 1% |
| absent | 1 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 81293 | 108784 | 0.747 | 1% |
| absent | 1 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 162806 | 204050 | 0.798 | 1% |
| absent | 1 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 324540 | 358328 | 0.906 | 1% |
| absent | 1 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 702018 | 701166 | 1.001 | 1% |
| absent | 1 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 300 | 1774 | 0.169 | 1% |
| absent | 1 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 471 | 1563 | 0.301 | 5% |
| absent | 1 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 801 | 2885 | 0.278 | 2% |
| absent | 1 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1473 | 2846 | 0.518 | 2% |
| absent | 1 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 892 | 3086 | 0.289 | 1% |
| absent | 1 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1262 | 2515 | 0.502 | 0% |
| absent | 1 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2344 | 5129 | 0.457 | 0% |
| absent | 1 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4608 | 8286 | 0.556 | 0% |
| absent | 1 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1453 | 4668 | 0.311 | 1% |
| absent | 1 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2835 | 6291 | 0.451 | 0% |
| absent | 1 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5340 | 8866 | 0.602 | 1% |
| absent | 1 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10701 | 15249 | 0.702 | 0% |
| absent | 1 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4558 | 7494 | 0.608 | 1% |
| absent | 1 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9197 | 12323 | 0.746 | 1% |
| absent | 1 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 17734 | 29195 | 0.607 | 1% |
| absent | 1 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 41278 | 41417 | 0.997 | 0% |
| absent | 1 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 10610 | 22091 | 0.480 | 0% |
| absent | 1 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 21470 | 29585 | 0.726 | 0% |
| absent | 1 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 41689 | 51667 | 0.807 | 0% |
| absent | 1 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 95631 | 95750 | 0.999 | 1% |
| absent | 1 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 35126 | 45005 | 0.780 | 0% |
| absent | 1 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 69731 | 80812 | 0.863 | 1% |
| absent | 1 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 137437 | 157156 | 0.875 | 0% |
| absent | 1 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 299395 | 299374 | 1.000 | 0% |
| absent | 1 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 82444 | 113654 | 0.725 | 1% |
| absent | 1 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 167355 | 206818 | 0.809 | 3% |
| absent | 1 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 328442 | 376556 | 0.872 | 1% |
| absent | 1 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 719596 | 718405 | 1.002 | 0% |
| absent | 4 | `gemm` | f32 | n=32 | 32768 | faer | packed | 761 | 2484 | 0.306 | 4% |
| absent | 4 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1413 | 3206 | 0.441 | 2% |
| absent | 4 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2725 | 4228 | 0.645 | 2% |
| absent | 4 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5330 | 7725 | 0.690 | 1% |
| absent | 4 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5219 | 11081 | 0.471 | 2% |
| absent | 4 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10259 | 14898 | 0.689 | 1% |
| absent | 4 | `gemm` | c32 | n=64 | 262144 | faer | packed | 15439 | 13676 | 1.129 | 28% |
| absent | 4 | `gemm` | c64 | n=64 | 262144 | packed | packed | 20008 | 19967 | 1.002 | 16% |
| absent | 4 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 21210 | 21019 | 1.009 | 88% |
| absent | 4 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 39614 | 42459 | 0.933 | 35% |
| absent | 4 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 59992 | 53129 | 1.129 | 7% |
| absent | 4 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 99798 | 98696 | 1.011 | 2% |
| absent | 4 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 113654 | 142267 | 0.799 | 24% |
| absent | 4 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 183606 | 203193 | 0.904 | 5% |
| absent | 4 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 347615 | 348216 | 0.998 | 0% |
| absent | 4 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 696076 | 690485 | 1.008 | 3% |
| absent | 4 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2084 | 11120 | 0.187 | 0% |
| absent | 4 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3486 | 9628 | 0.362 | 2% |
| absent | 4 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9568 | 20017 | 0.478 | 0% |
| absent | 4 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 19306 | 19066 | 1.013 | 1% |
| absent | 4 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 6623 | 15118 | 0.438 | 7% |
| absent | 4 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 11281 | 20147 | 0.560 | 3% |
| absent | 4 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 14197 | 19587 | 0.725 | 0% |
| absent | 4 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 24716 | 33593 | 0.736 | 3% |
| absent | 4 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 24125 | 46847 | 0.515 | 2% |
| absent | 4 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 44584 | 61985 | 0.719 | 2% |
| absent | 4 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 253774 | 102122 | 2.485 | 6% |
| absent | 4 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 196840 | 197969 | 0.994 | 1% |
| absent | 4 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 14848 | 38061 | 0.390 | 1% |
| absent | 4 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 24896 | 49383 | 0.504 | 4% |
| absent | 4 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 46297 | 65814 | 0.703 | 1% |
| absent | 4 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 88224 | 123412 | 0.715 | 1% |
| absent | 4 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 86391 | 175209 | 0.493 | 1% |
| absent | 4 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 170758 | 243494 | 0.701 | 3% |
| absent | 4 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1015983 | 401273 | 2.532 | 1% |
| absent | 4 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 943176 | 945794 | 0.997 | 1% |
| absent | 4 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 310 | 1413 | 0.219 | 3% |
| absent | 4 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 481 | 1412 | 0.341 | 2% |
| absent | 4 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 811 | 1803 | 0.450 | 2% |
| absent | 4 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1512 | 2655 | 0.569 | 3% |
| absent | 4 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 871 | 1744 | 0.499 | 2% |
| absent | 4 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1272 | 2235 | 0.569 | 5% |
| absent | 4 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2384 | 5039 | 0.473 | 3% |
| absent | 4 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4608 | 8005 | 0.576 | 1% |
| absent | 4 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1462 | 4258 | 0.343 | 4% |
| absent | 4 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2755 | 5741 | 0.480 | 2% |
| absent | 4 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5350 | 7744 | 0.691 | 1% |
| absent | 4 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10580 | 13986 | 0.756 | 1% |
| absent | 4 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4488 | 7133 | 0.629 | 1% |
| absent | 4 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8856 | 11862 | 0.747 | 0% |
| absent | 4 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 11913 | 13896 | 0.857 | 25% |
| absent | 4 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 15739 | 15600 | 1.009 | 5% |
| absent | 4 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10028 | 12163 | 0.824 | 40% |
| absent | 4 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 18244 | 17703 | 1.031 | 33% |
| absent | 4 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 23714 | 19507 | 1.216 | 51% |
| absent | 4 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 32542 | 32270 | 1.008 | 10% |
| absent | 4 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 32671 | 18194 | 1.796 | 93% |
| absent | 4 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 58720 | 27602 | 2.127 | 36% |
| absent | 4 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 63610 | 46216 | 1.376 | 13% |
| absent | 4 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 110917 | 110347 | 1.005 | 2% |
| absent | 4 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 44223 | 35376 | 1.250 | 29% |
| absent | 4 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 65543 | 76052 | 0.862 | 20% |
| absent | 4 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 98954 | 98423 | 1.005 | 6% |
| absent | 4 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 186560 | 185788 | 1.004 | 1% |
| absent | 4 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 321 | 1613 | 0.199 | 7% |
| absent | 4 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 481 | 1402 | 0.343 | 4% |
| absent | 4 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 811 | 2665 | 0.304 | 1% |
| absent | 4 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1483 | 2605 | 0.569 | 3% |
| absent | 4 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 892 | 2895 | 0.308 | 2% |
| absent | 4 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1262 | 2275 | 0.555 | 2% |
| absent | 4 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2344 | 4839 | 0.484 | 2% |
| absent | 4 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4608 | 7965 | 0.579 | 2% |
| absent | 4 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1472 | 4489 | 0.328 | 1% |
| absent | 4 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2835 | 6051 | 0.469 | 2% |
| absent | 4 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5380 | 8536 | 0.630 | 1% |
| absent | 4 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10730 | 14998 | 0.715 | 1% |
| absent | 4 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4589 | 7324 | 0.627 | 1% |
| absent | 4 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9217 | 12072 | 0.764 | 1% |
| absent | 4 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 18385 | 14107 | 1.303 | 23% |
| absent | 4 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 21059 | 21481 | 0.980 | 5% |
| absent | 4 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 11752 | 15338 | 0.766 | 13% |
| absent | 4 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 17783 | 16762 | 1.061 | 13% |
| absent | 4 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 29946 | 21260 | 1.409 | 27% |
| absent | 4 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 32661 | 33613 | 0.972 | 9% |
| absent | 4 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 23033 | 19106 | 1.206 | 94% |
| absent | 4 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 76935 | 28273 | 2.721 | 47% |
| absent | 4 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 88815 | 50184 | 1.770 | 40% |
| absent | 4 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 109907 | 110096 | 0.998 | 1% |
| absent | 4 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 49533 | 41869 | 1.183 | 52% |
| absent | 4 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 60443 | 76703 | 0.788 | 24% |
| absent | 4 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 109805 | 102332 | 1.073 | 11% |
| absent | 4 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 194825 | 192061 | 1.014 | 1% |

#### `output` (in-place accumulation)

| c_mode | threads | class | dtype | params | mnk | default | packed | ns default | ns packed | default/packed | spread |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|
| output | 1 | `gemm` | f32 | n=32 | 32768 | faer | packed | 751 | 2624 | 0.286 | 1% |
| output | 1 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1413 | 3477 | 0.406 | 1% |
| output | 1 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2715 | 7654 | 0.355 | 3% |
| output | 1 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5330 | 8546 | 0.624 | 2% |
| output | 1 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5180 | 11271 | 0.460 | 1% |
| output | 1 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10359 | 15359 | 0.674 | 0% |
| output | 1 | `gemm` | c32 | n=64 | 262144 | faer | packed | 20659 | 29635 | 0.697 | 0% |
| output | 1 | `gemm` | c64 | n=64 | 262144 | packed | packed | 64220 | 63679 | 1.008 | 1% |
| output | 1 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 40606 | 56466 | 0.719 | 1% |
| output | 1 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 80511 | 102132 | 0.788 | 1% |
| output | 1 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 161882 | 198430 | 0.816 | 0% |
| output | 1 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 436895 | 436734 | 1.000 | 1% |
| output | 1 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 322361 | 399375 | 0.807 | 1% |
| output | 1 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 649119 | 710013 | 0.914 | 1% |
| output | 1 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 1290636 | 1424827 | 0.906 | 0% |
| output | 1 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 2764845 | 2739847 | 1.009 | 2% |
| output | 1 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2175 | 11442 | 0.190 | 60% |
| output | 1 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3556 | 10109 | 0.352 | 28% |
| output | 1 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9117 | 24246 | 0.376 | 1% |
| output | 1 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 18114 | 21691 | 0.835 | 1% |
| output | 1 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 11452 | 34955 | 0.328 | 1% |
| output | 1 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 21781 | 46757 | 0.466 | 1% |
| output | 1 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 42930 | 80590 | 0.533 | 0% |
| output | 1 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 86632 | 130365 | 0.665 | 1% |
| output | 1 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 82774 | 173066 | 0.478 | 0% |
| output | 1 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 167552 | 240992 | 0.695 | 1% |
| output | 1 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 330837 | 469723 | 0.704 | 0% |
| output | 1 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 817444 | 817674 | 1.000 | 0% |
| output | 1 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 46637 | 138400 | 0.337 | 12% |
| output | 1 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 89759 | 189233 | 0.474 | 2% |
| output | 1 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 173354 | 322130 | 0.538 | 1% |
| output | 1 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 351801 | 528005 | 0.666 | 1% |
| output | 1 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 333667 | 694105 | 0.481 | 0% |
| output | 1 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 684606 | 1028538 | 0.666 | 5% |
| output | 1 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1346330 | 1926598 | 0.699 | 6% |
| output | 1 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 4038360 | 4042171 | 0.999 | 2% |
| output | 1 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 330 | 1613 | 0.205 | 6% |
| output | 1 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 461 | 1734 | 0.266 | 4% |
| output | 1 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 801 | 2454 | 0.326 | 1% |
| output | 1 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1492 | 3216 | 0.464 | 4% |
| output | 1 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 1042 | 2104 | 0.495 | 1% |
| output | 1 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1252 | 2474 | 0.506 | 3% |
| output | 1 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2375 | 6112 | 0.389 | 1% |
| output | 1 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4739 | 8957 | 0.529 | 2% |
| output | 1 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1452 | 4809 | 0.302 | 1% |
| output | 1 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2785 | 6172 | 0.451 | 1% |
| output | 1 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5420 | 10400 | 0.521 | 1% |
| output | 1 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10841 | 15680 | 0.691 | 0% |
| output | 1 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4518 | 7474 | 0.604 | 0% |
| output | 1 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8916 | 12303 | 0.725 | 0% |
| output | 1 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 17643 | 33532 | 0.526 | 1% |
| output | 1 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 44494 | 44504 | 1.000 | 0% |
| output | 1 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 10409 | 21380 | 0.487 | 1% |
| output | 1 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 20779 | 29425 | 0.706 | 1% |
| output | 1 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 41398 | 59942 | 0.691 | 1% |
| output | 1 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 101240 | 101330 | 0.999 | 1% |
| output | 1 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 34625 | 44414 | 0.780 | 0% |
| output | 1 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 68990 | 80431 | 0.858 | 1% |
| output | 1 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 138041 | 175029 | 0.789 | 0% |
| output | 1 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 312332 | 312021 | 1.001 | 1% |
| output | 1 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 81672 | 110547 | 0.739 | 1% |
| output | 1 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 162816 | 204344 | 0.797 | 1% |
| output | 1 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 325862 | 394845 | 0.825 | 1% |
| output | 1 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 725342 | 724690 | 1.001 | 1% |
| output | 1 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 300 | 1813 | 0.165 | 2% |
| output | 1 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 471 | 1613 | 0.292 | 2% |
| output | 1 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 791 | 3196 | 0.247 | 3% |
| output | 1 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1483 | 3035 | 0.489 | 2% |
| output | 1 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 951 | 3156 | 0.301 | 1% |
| output | 1 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1262 | 2585 | 0.488 | 2% |
| output | 1 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2344 | 5761 | 0.407 | 1% |
| output | 1 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4678 | 8666 | 0.540 | 1% |
| output | 1 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1452 | 4749 | 0.306 | 1% |
| output | 1 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2795 | 6382 | 0.438 | 1% |
| output | 1 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5340 | 10039 | 0.532 | 0% |
| output | 1 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10781 | 15920 | 0.677 | 1% |
| output | 1 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4558 | 7514 | 0.607 | 1% |
| output | 1 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9227 | 12384 | 0.745 | 0% |
| output | 1 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 17893 | 31880 | 0.561 | 0% |
| output | 1 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 42941 | 42911 | 1.001 | 0% |
| output | 1 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 10640 | 22502 | 0.473 | 0% |
| output | 1 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 21431 | 29847 | 0.718 | 0% |
| output | 1 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 41618 | 56385 | 0.738 | 0% |
| output | 1 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 98704 | 98475 | 1.002 | 1% |
| output | 1 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 35096 | 45165 | 0.777 | 0% |
| output | 1 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 69862 | 81232 | 0.860 | 0% |
| output | 1 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 137960 | 167332 | 0.824 | 0% |
| output | 1 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 305169 | 305106 | 1.000 | 0% |
| output | 1 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 82705 | 114204 | 0.724 | 0% |
| output | 1 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 167886 | 212328 | 0.791 | 2% |
| output | 1 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 328844 | 394696 | 0.833 | 0% |
| output | 1 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 730102 | 730327 | 1.000 | 0% |
| output | 4 | `gemm` | f32 | n=32 | 32768 | faer | packed | 751 | 2575 | 0.292 | 1% |
| output | 4 | `gemm` | f64 | n=32 | 32768 | faer | packed | 1413 | 3437 | 0.411 | 2% |
| output | 4 | `gemm` | c32 | n=32 | 32768 | faer | packed | 2725 | 5420 | 0.503 | 2% |
| output | 4 | `gemm` | c64 | n=32 | 32768 | faer | packed | 5390 | 8496 | 0.634 | 1% |
| output | 4 | `gemm` | f32 | n=64 | 262144 | faer | packed | 5210 | 11291 | 0.461 | 2% |
| output | 4 | `gemm` | f64 | n=64 | 262144 | faer | packed | 10309 | 15279 | 0.675 | 1% |
| output | 4 | `gemm` | c32 | n=64 | 262144 | faer | packed | 15288 | 15379 | 0.994 | 17% |
| output | 4 | `gemm` | c64 | n=64 | 262144 | packed | packed | 22552 | 21771 | 1.036 | 8% |
| output | 4 | `gemm` | f32 | n=128 | 2097152 | faer | packed | 26940 | 20488 | 1.315 | 49% |
| output | 4 | `gemm` | f64 | n=128 | 2097152 | faer | packed | 33563 | 43441 | 0.773 | 25% |
| output | 4 | `gemm` | c32 | n=128 | 2097152 | faer | packed | 57979 | 61015 | 0.950 | 5% |
| output | 4 | `gemm` | c64 | n=128 | 2097152 | packed | packed | 104356 | 104216 | 1.001 | 3% |
| output | 4 | `gemm` | f32 | n=256 | 16777216 | faer | packed | 120807 | 143860 | 0.840 | 27% |
| output | 4 | `gemm` | f64 | n=256 | 16777216 | faer | packed | 180658 | 205894 | 0.877 | 4% |
| output | 4 | `gemm` | c32 | n=256 | 16777216 | faer | packed | 355039 | 378997 | 0.937 | 2% |
| output | 4 | `gemm` | c64 | n=256 | 16777216 | packed | packed | 707789 | 708785 | 0.999 | 1% |
| output | 4 | `gemm_batched` | f32 | n=16 batch=16 | 4096 | faer | packed | 2134 | 11441 | 0.187 | 2% |
| output | 4 | `gemm_batched` | f64 | n=16 batch=16 | 4096 | faer | packed | 3557 | 9928 | 0.358 | 2% |
| output | 4 | `gemm_batched` | c32 | n=16 batch=16 | 4096 | faer | packed | 9137 | 24156 | 0.378 | 0% |
| output | 4 | `gemm_batched` | c64 | n=16 batch=16 | 4096 | faer | packed | 18164 | 21851 | 0.831 | 2% |
| output | 4 | `gemm_batched` | f32 | n=32 batch=16 | 32768 | faer | packed | 6742 | 15168 | 0.444 | 8% |
| output | 4 | `gemm_batched` | f64 | n=32 batch=16 | 32768 | faer | packed | 11401 | 20989 | 0.543 | 2% |
| output | 4 | `gemm_batched` | c32 | n=32 batch=16 | 32768 | faer | packed | 13716 | 24376 | 0.563 | 7% |
| output | 4 | `gemm_batched` | c64 | n=32 batch=16 | 32768 | faer | packed | 25167 | 37009 | 0.680 | 4% |
| output | 4 | `gemm_batched` | f32 | n=64 batch=16 | 262144 | faer | packed | 24135 | 47239 | 0.511 | 4% |
| output | 4 | `gemm_batched` | f64 | n=64 batch=16 | 262144 | faer | packed | 44904 | 63669 | 0.705 | 2% |
| output | 4 | `gemm_batched` | c32 | n=64 batch=16 | 262144 | faer | packed | 247501 | 120775 | 2.049 | 6% |
| output | 4 | `gemm_batched` | c64 | n=64 batch=16 | 262144 | packed | packed | 208258 | 208092 | 1.001 | 1% |
| output | 4 | `gemm_batched` | f32 | n=32 batch=64 | 32768 | faer | packed | 14857 | 38823 | 0.383 | 5% |
| output | 4 | `gemm_batched` | f64 | n=32 batch=64 | 32768 | faer | packed | 25618 | 50766 | 0.505 | 1% |
| output | 4 | `gemm_batched` | c32 | n=32 batch=64 | 32768 | faer | packed | 46647 | 84269 | 0.554 | 1% |
| output | 4 | `gemm_batched` | c64 | n=32 batch=64 | 32768 | faer | packed | 89878 | 134492 | 0.668 | 1% |
| output | 4 | `gemm_batched` | f32 | n=64 batch=64 | 262144 | faer | packed | 86692 | 177394 | 0.489 | 1% |
| output | 4 | `gemm_batched` | f64 | n=64 batch=64 | 262144 | faer | packed | 171572 | 248927 | 0.689 | 2% |
| output | 4 | `gemm_batched` | c32 | n=64 batch=64 | 262144 | faer | packed | 1022256 | 476448 | 2.146 | 2% |
| output | 4 | `gemm_batched` | c64 | n=64 batch=64 | 262144 | packed | packed | 998510 | 998191 | 1.000 | 1% |
| output | 4 | `mps_env` | f32 | chi=16 | 8192 | faer | packed | 340 | 1493 | 0.228 | 9% |
| output | 4 | `mps_env` | f64 | chi=16 | 8192 | faer | packed | 471 | 1543 | 0.305 | 4% |
| output | 4 | `mps_env` | c32 | chi=16 | 8192 | faer | packed | 821 | 2224 | 0.369 | 4% |
| output | 4 | `mps_env` | c64 | chi=16 | 8192 | faer | packed | 1503 | 3025 | 0.497 | 2% |
| output | 4 | `mps_env` | f32 | chi=24 | 27648 | faer | packed | 1012 | 1923 | 0.526 | 2% |
| output | 4 | `mps_env` | f64 | chi=24 | 27648 | faer | packed | 1252 | 2294 | 0.546 | 1% |
| output | 4 | `mps_env` | c32 | chi=24 | 27648 | faer | packed | 2374 | 5851 | 0.406 | 1% |
| output | 4 | `mps_env` | c64 | chi=24 | 27648 | faer | packed | 4728 | 8776 | 0.539 | 1% |
| output | 4 | `mps_env` | f32 | chi=32 | 65536 | faer | packed | 1443 | 4619 | 0.312 | 2% |
| output | 4 | `mps_env` | f64 | chi=32 | 65536 | faer | packed | 2766 | 5941 | 0.466 | 2% |
| output | 4 | `mps_env` | c32 | chi=32 | 65536 | faer | packed | 5380 | 10088 | 0.533 | 1% |
| output | 4 | `mps_env` | c64 | chi=32 | 65536 | faer | packed | 10730 | 15449 | 0.695 | 3% |
| output | 4 | `mps_env` | f32 | chi=48 | 221184 | faer | packed | 4508 | 7284 | 0.619 | 1% |
| output | 4 | `mps_env` | f64 | chi=48 | 221184 | faer | packed | 8896 | 12083 | 0.736 | 0% |
| output | 4 | `mps_env` | c32 | chi=48 | 221184 | faer | packed | 12513 | 16200 | 0.772 | 26% |
| output | 4 | `mps_env` | c64 | chi=48 | 221184 | packed | packed | 16481 | 16641 | 0.990 | 6% |
| output | 4 | `mps_env` | f32 | chi=64 | 524288 | faer | packed | 13235 | 13275 | 0.997 | 23% |
| output | 4 | `mps_env` | f64 | chi=64 | 524288 | faer | packed | 20708 | 18435 | 1.123 | 22% |
| output | 4 | `mps_env` | c32 | chi=64 | 524288 | faer | packed | 18294 | 23203 | 0.788 | 28% |
| output | 4 | `mps_env` | c64 | chi=64 | 524288 | packed | packed | 35256 | 35526 | 0.992 | 5% |
| output | 4 | `mps_env` | f32 | chi=96 | 1769472 | faer | packed | 33423 | 18435 | 1.813 | 78% |
| output | 4 | `mps_env` | f64 | chi=96 | 1769472 | faer | packed | 41799 | 28694 | 1.457 | 7% |
| output | 4 | `mps_env` | c32 | chi=96 | 1769472 | faer | packed | 67156 | 51756 | 1.298 | 44% |
| output | 4 | `mps_env` | c64 | chi=96 | 1769472 | packed | packed | 118563 | 118743 | 0.998 | 1% |
| output | 4 | `mps_env` | f32 | chi=128 | 4194304 | faer | packed | 28633 | 36337 | 0.788 | 55% |
| output | 4 | `mps_env` | f64 | chi=128 | 4194304 | faer | packed | 54182 | 77436 | 0.700 | 2% |
| output | 4 | `mps_env` | c32 | chi=128 | 4194304 | faer | packed | 112642 | 114977 | 0.980 | 11% |
| output | 4 | `mps_env` | c64 | chi=128 | 4194304 | packed | packed | 197247 | 197320 | 1.000 | 1% |
| output | 4 | `mps_site` | f32 | chi=16 | 8192 | faer | packed | 320 | 1663 | 0.192 | 6% |
| output | 4 | `mps_site` | f64 | chi=16 | 8192 | faer | packed | 481 | 1452 | 0.331 | 2% |
| output | 4 | `mps_site` | c32 | chi=16 | 8192 | faer | packed | 812 | 2925 | 0.278 | 2% |
| output | 4 | `mps_site` | c64 | chi=16 | 8192 | faer | packed | 1482 | 2795 | 0.530 | 1% |
| output | 4 | `mps_site` | f32 | chi=24 | 27648 | faer | packed | 952 | 2965 | 0.321 | 1% |
| output | 4 | `mps_site` | f64 | chi=24 | 27648 | faer | packed | 1282 | 2324 | 0.552 | 2% |
| output | 4 | `mps_site` | c32 | chi=24 | 27648 | faer | packed | 2374 | 5490 | 0.432 | 1% |
| output | 4 | `mps_site` | c64 | chi=24 | 27648 | faer | packed | 4688 | 8365 | 0.560 | 1% |
| output | 4 | `mps_site` | f32 | chi=32 | 65536 | faer | packed | 1462 | 4568 | 0.320 | 1% |
| output | 4 | `mps_site` | f64 | chi=32 | 65536 | faer | packed | 2836 | 6201 | 0.457 | 1% |
| output | 4 | `mps_site` | c32 | chi=32 | 65536 | faer | packed | 5380 | 9738 | 0.552 | 1% |
| output | 4 | `mps_site` | c64 | chi=32 | 65536 | faer | packed | 10850 | 15650 | 0.693 | 1% |
| output | 4 | `mps_site` | f32 | chi=48 | 221184 | faer | packed | 4609 | 7414 | 0.622 | 2% |
| output | 4 | `mps_site` | f64 | chi=48 | 221184 | faer | packed | 9267 | 12223 | 0.758 | 0% |
| output | 4 | `mps_site` | c32 | chi=48 | 221184 | faer | packed | 17653 | 15559 | 1.135 | 25% |
| output | 4 | `mps_site` | c64 | chi=48 | 221184 | packed | packed | 22231 | 22171 | 1.003 | 4% |
| output | 4 | `mps_site` | f32 | chi=64 | 524288 | faer | packed | 13265 | 15398 | 0.861 | 22% |
| output | 4 | `mps_site` | f64 | chi=64 | 524288 | faer | packed | 22222 | 17813 | 1.248 | 37% |
| output | 4 | `mps_site` | c32 | chi=64 | 524288 | faer | packed | 31789 | 23284 | 1.365 | 20% |
| output | 4 | `mps_site` | c64 | chi=64 | 524288 | packed | packed | 35456 | 35407 | 1.001 | 0% |
| output | 4 | `mps_site` | f32 | chi=96 | 1769472 | faer | packed | 29275 | 19627 | 1.492 | 29% |
| output | 4 | `mps_site` | f64 | chi=96 | 1769472 | faer | packed | 74218 | 29165 | 2.545 | 12% |
| output | 4 | `mps_site` | c32 | chi=96 | 1769472 | faer | packed | 89948 | 53070 | 1.695 | 18% |
| output | 4 | `mps_site` | c64 | chi=96 | 1769472 | packed | packed | 115417 | 115315 | 1.001 | 1% |
| output | 4 | `mps_site` | f32 | chi=128 | 4194304 | faer | packed | 61044 | 41948 | 1.455 | 19% |
| output | 4 | `mps_site` | f64 | chi=128 | 4194304 | faer | packed | 63640 | 77594 | 0.820 | 21% |
| output | 4 | `mps_site` | c32 | chi=128 | 4194304 | faer | packed | 108233 | 110898 | 0.976 | 12% |
| output | 4 | `mps_site` | c64 | chi=128 | 4194304 | packed | packed | 199114 | 197779 | 1.007 | 1% |

The new run's `absent` rows reproduce the earlier table within the session
spread. Over the 184 `(threads, class, dtype, params)` keys, the
median-over-sessions `output`/`absent` ns ratio is 0.65 / 1.01 / 1.32
(min / median / max) for the `default` arm and 0.53 / 1.04 / 1.30 for the
`packed` arm: for most rows the per-call `output` cost is within the session
noise of `absent`, and the largest shifts are on the small / 1T rows.

Across the three sessions of the new run the `absent` rows repeat within a
median 0.4% / p90 1.8% / max 63.9% spread at 1T and a median 1.4% / p90 22.6% /
max 94.4% at 4T; the `output` rows within a median 0.5% / p90 2.0% / max 59.9%
at 1T and a median 1.4% / p90 17.6% / max 78.4% at 4T. As before, 4T differences
below that band are not findings.

The `CHECK` lines in `results/cmode-session{1,2,3}.txt` (368 per session: one
per case, dtype, thread count and C mode) show both modes matching their
references. The worst `rel` over all cases and sessions is 1.13e-6 for `absent`
and 1.11e-6 for `output`, both on the `f32`/`c32` rows (tol 1e-5); the two arms
agree to at most 4.6e-7. No case failed.

## What this does not cover

- One host, one `rustc`, one library revision, one session order. This is a
  laptop APU under the `powersave` governor: no fixed frequency and no isolated
  cores, which is where most of the 4T spread comes from. The earlier and the
  new run also sit on different library revisions (`results/manifest.txt`),
  so a ratio read across the two tables mixes that change; the new run's own
  `absent` and `output` rows are same-revision and directly comparable.
- Two C modes are measured: `absent` (overwrite) and `output` (in-place
  accumulation with `CSpec::Output(Op::Identity)` and `beta = 1`). The third
  mode, `CSpec::Separate` (a separately described `C`), is not measured.
- `output` is `beta = 1`, `alpha = 1` and `Op::Identity` only; other `beta`
  values, `op_C` conjugation, and `alpha != 1` are not measured.
- In the `output` mode the priming and timed calls accumulate into the same `D`
  without a reset between calls (as a sequence of MPS steps does). The per-call
  cost is the same, but `D` grows across the priming window; a reset outside the
  clock would measure the same call.
- Whole-shape timings only: no split between packing, the micro-kernel and
  write-back inside either engine, and no counter-based attribution.
- **For `c64` above the provisional limit the two arms are the same engine.**
  The default route already sends `c64` to the packed driver at
  `mnk = 262144` (`FaerLimit::default()` bounds `c64` at `2^17`), so the c64
  rows above that volume compare packed against packed (ratio ~1.00) rather
  than faer against packed. The probe as specified can see the c64 crossover
  only below that volume (`chi <= 32`, `n = 32`); a third arm with
  `FaerLimit::NONE` would be needed to observe it above.
- Routing actually exercised: in both C modes the default arm chose `faer` with
  reason `fused` for every `f32`/`f64`/`c32` case and for `c64` below the limit,
  and `packed` with reason `above_faer_limit` above it. `NotFusable`, `AllBatch`
  and the elementwise route never appeared (every case is a copy-free fusion).
- Nothing here chooses a `FaerLimit`; the per-dtype values and the
  decision-log row are the main agent's to write from these numbers.
