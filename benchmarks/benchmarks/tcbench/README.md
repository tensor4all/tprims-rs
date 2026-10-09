# TCCG contraction benchmark

`tcbench` reuses the full 49-case TCCG corpus (CCSD, AO2MO, InTensLi,
CCSD(T), transposed output). Shapes, signed element strides, seeded random
inputs, alpha=1, beta=0 and output buffers are identical across engines.
The algorithms originate in Matthews, *High-Performance Tensor Contraction
without Transposition*, [arXiv:1607.00291](https://arxiv.org/abs/1607.00291).

| Engine | What is measured |
| --- | --- |
| `plan` | tprims's default planner: packed or copy-free faer |
| `packed` | tprims packed driver forced (diagnostic) |

## Fixed procedure

The source of truth is
[the tprims-benchmark skill](../../../.agents/skills/tprims-benchmark/SKILL.md).
Use `RUSTFLAGS="-C target-cpu=native"` and release for all Rust code.

```sh
export CARGO_BUILD_JOBS=8  # choose for the host
# Ryzen AI 9 HX 470: 1/4/8 physical cores in L3 #1; 12 physical cores in two L3s.
bash benchmarks/benchmarks/tcbench/run.sh \
  benchmarks/benchmarks/tcbench/results/native-12t 4 4-7 4-11 0-11
```

The runner builds once, checks known-value GEMM and all corpus outputs at
1/16 MiB and 1T/4T/8T/12T, then measures the complete suite twice sequentially.
Every process goes through `pinned.sh`; do not run another benchmark or build
alongside it. A failed idle/correctness gate stops the suite; keep failed
observations and classify the pair inconclusive. Core lists are explicit
arguments: validate them against `lscpu`, not a guessed core count. 12T across
L3s is a user-requested full-physical-core exception to single-L3 measurement.
No SMT or 16T arm.

At least 1500 ms of untimed, wall-clock priming per arm (`--prime-ms`, echoed by
`run`) then five repetitions, best wall time, including packing,
executor/FFI entry and provider-internal allocation. Priming is time-based, not
a call count: a fixed number of warm-up calls removes a different share of the
clock ramp in every arm, most in the fastest one. Input generation,
planning/descriptor construction and tprims pool construction are excluded.
tprims retains a caller-owned scratch arena/pool.
These are library execution comparisons, not isolated microkernel timings.

The default is 1500 ms, not 500: at 500 ms the *first* arm measured for a
case read 40% low on this host (2.89 ms against 2.06 ms once settled, for a call
of about 2 ms), which made the arm measured after it look 29% faster than the
same work measured on its own — a position bias, not a kernel difference. At
1500 ms the same case reads 2.06 ms whether measured alone or after another arm.
The built-in check: two arms that resolve to the same driver and grid must agree.


CSV rows include the thread budget, dtype, equivalent M/N/K, GFLOP/s, latency
and tprims strategy/kernel notes. `manifest.txt` records the base commit,
dirty status, binary SHA256, compiler, flags and actual topology;
`source.patch` records tracked local harness edits. Archive new source files
next to results too. The legacy `info` cache
and domain estimates describe the library's model, **not** actual affinity;
use the captured `lscpu` topology to interpret this heterogeneous host.

[2026-10-07 native run](results/2026-10-07-native/README.md).
