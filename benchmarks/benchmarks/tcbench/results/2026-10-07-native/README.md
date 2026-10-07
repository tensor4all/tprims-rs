# Native tprims / tensorprimitives / TBLIS comparison

## Identity and validity

Measured 2026-10-07 on **AMD Ryzen AI 9 HX 470**, 12 physical cores / 24
logical CPUs. tprims base commit
`d726474701dce6ac571758755a8d60d6600df546` plus local benchmark-only changes
([source patch](source.patch), [complete harness snapshot](harness-source.tar.gz),
binary hash/compiler/topology in [manifest](manifest.txt)). Library sources
were unchanged. Release: opt-level=3, thin LTO, codegen-units=1;
**`RUSTFLAGS=-C target-cpu=native`** across both Rust implementations.
Upstream tensorprimitives-rs:
`8cda75e11ed26f46c0c22f9629004c84dbabc8e5` (Lukas Devos).

TBLIS **2.0**: `20cc0bcdb13ddf9fbf619138b6081a16e9e48b7f`;
bundled static BLIS **3.0**: `358e689cadd6757f564a2992cf46a2f7d6fa6bb0`.
CMake Release, C/C++ `-O3 -march=native`, explicit **BLIS zen3 / AVX2**
configuration, BLIS pthread / TCI OpenMP. Auto-detection chose low-performance
generic on this newer CPU; that configuration was rejected before timing.
The pinned BLIS has no Zen4/Zen5 config. Thus this is **not an equal-ISA
AVX512 comparison**. System BLIS 0.9.0 is installed but not used by TBLIS.
See [provider manifest](tblis-manifest.txt),
[CMake settings](tblis-CMakeCache.txt), [configuration log](tblis-configure.txt).
Installation was local under `/tmp/tprims-tblis-install`, no system changes.

Affinity, one hardware thread per physical core (no SMT):

| Budget | Linux CPUs | Actual L3 domains |
| --- | --- | --- |
| 1T | 4 | 1 |
| 4T | 4-7 | 1 |
| 8T | 4-11 | 1 |
| 12T | 0-11 | 2 |

12T is the user's full-physical-core scaling arm, explicitly outside the
single-L3 condition. The CPU has heterogeneous core/cache domains; 8T-to-12T
is not pure same-core thread scaling. Legacy `tcbench info` cache/domain
estimates are the library's model, not actual affinity: use the captured
`lscpu` topology. Frequency/turbo and OS scheduling were not locked, so the
idle windows do not eliminate all runtime variance.

Complete **49-case TCCG corpus**, f64/c64, contiguous layouts, nominal
1/16 MiB sizing (rounded/minimum extents; see per-row M/N/K).
All engines consume identical seeded buffers, alpha=1,beta=0.
`plan` is tprims's default strategy; `packed` is diagnostic only.
Original `upstream` is a separate git dependency, not relabelled tprims code.
One warm-up + five repetitions, best execution wall time. Plans, input
creation and tprims pool creation excluded; packing, entry/FFI and
provider-owned scratch/allocation/thread creation included. tprims retains
its scratch/pool; upstream keeps its default per-call scoped threads.

**Correctness passed**: known-value GEMM plus full corpus at both sizes and
all four budgets, finite Frobenius residual <=1e-10.
Complete suite measured twice (A/A), sequentially, no concurrent build or
benchmark. Every pre/post idle guard passed on attempt 1 (3-second window,
<=5% busy on selected CPUs). 6,272 measured CSV rows and 784 combined
case/dtype/size/thread rows; no mismatches, missing cases or selected exclusions.

An earlier run exceeded its 600-second controller deadline during 16 MiB 1T.
Its incomplete files remain in `../2026-10-07-native-interrupted/`; **none**
of those times entered this complete rerun. Protocol and verification:
[work log](../../../../../docs/worklogs/2026-10-07-three-provider-benchmark.md).

## All-case summaries

Ratios are geometric means over all 49 cases, using the geometric mean of
the two complete measurements for each engine. **A time ratio >1 means
tprims is faster**. A/A columns aggregate all four engines (196 rows per
size/dtype/budget). `packed` contributes only to noise reporting here.

| MiB target | dtype | T | upstream / tprims time | TBLIS / tprims time | tprims 1T speedup | A/A median | A/A max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | f64 | 1 | 1.05 | 1.47 | 1.00 | 0.9% | 11.1% |
| 1 | f64 | 4 | 1.14 | 1.48 | 3.11 | 1.6% | 70.9% |
| 1 | f64 | 8 | 1.35 | 1.39 | 4.53 | 1.5% | 83.8% |
| 1 | f64 | 12 | 1.52 | 1.47 | 4.59 | 12.1% | 9486.2% |
| 1 | c64 | 1 | 1.03 | 1.22 | 1.00 | 0.6% | 9.2% |
| 1 | c64 | 4 | 1.06 | 1.20 | 3.43 | 0.7% | 32.6% |
| 1 | c64 | 8 | 1.16 | 1.13 | 5.33 | 1.1% | 31.7% |
| 1 | c64 | 12 | 1.35 | 1.11 | 6.00 | 9.6% | 10714.2% |
| 16 | f64 | 1 | 1.05 | 1.29 | 1.00 | 0.4% | 10.7% |
| 16 | f64 | 4 | 1.05 | 1.28 | 3.27 | 0.7% | 11.8% |
| 16 | f64 | 8 | 1.04 | 1.31 | 5.06 | 0.8% | 17.8% |
| 16 | f64 | 12 | 1.19 | 1.21 | 5.49 | 4.3% | 414.0% |
| 16 | c64 | 1 | 1.02 | 1.18 | 1.00 | 0.3% | 8.1% |
| 16 | c64 | 4 | 1.02 | 1.21 | 3.73 | 0.2% | 16.4% |
| 16 | c64 | 8 | 1.02 | 1.14 | 5.84 | 0.6% | 9.1% |
| 16 | c64 | 12 | 1.18 | 1.10 | 6.96 | 3.1% | 173.0% |

### 16 MiB scaling relative to each implementation's own 1T

| dtype | T | tprims plan | upstream tensorprimitives | TBLIS |
| --- | --- | --- | --- | --- |
| f64 | 1 | 1.00 | 1.00 | 1.00 |
| f64 | 4 | 3.27 | 3.27 | 3.31 |
| f64 | 8 | 5.06 | 5.09 | 5.02 |
| f64 | 12 | 5.49 | 4.82 | 5.89 |
| c64 | 1 | 1.00 | 1.00 | 1.00 |
| c64 | 4 | 3.73 | 3.73 | 3.61 |
| c64 | 8 | 5.84 | 5.85 | 6.05 |
| c64 | 12 | 6.96 | 6.03 | 7.44 |

## Interpretation and limitations

- At 16 MiB, 1T-8T tprims and original tensorprimitives are close (2-5%
  geometric-mean time differences); do not claim universal superiority from
  these small differences. Per-case A/A spreads can exceed them.
- Against **this Zen3/AVX2 TBLIS build**, tprims's 16 MiB mean time advantage
  at 1T-8T is 1.28-1.31x in f64 and 1.14-1.21x in c64. This does not isolate
  driver design from differing kernel/ISA choices.
- 12T is unstable, especially at 1 MiB. TBLIS's `abjc-cbka-kj` c64 execution
  changed from 0.277 ms to 29.991 ms (108x) between complete runs despite
  passing idle windows. tprims also has large small-case variance; the cause
  was not established. **12T winner/ranking is inconclusive**, not evidence
  that a library is reliably slower/faster. All such rows remain in the report.
- More threads do not win every individual case. The complete per-case
  latency, 1T scaling and A/A spread are in [comparison.csv](comparison.csv),
  including unfavorable and noisy cases. Means are not a replacement for it.
- These results do not cover padded/negative strides, f32/c32, cold caches,
  planning cost, N-ary contraction ordering or a Zen5-tuned TBLIS build.

Recreate the table and validate the complete result set:

```sh
python3 benchmarks/benchmarks/tcbench/summarize.py \
  benchmarks/benchmarks/tcbench/results/2026-10-07-native
```

The runner and source-of-truth skill are linked from the [benchmark README](../../README.md).
