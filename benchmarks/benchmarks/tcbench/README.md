# Three-provider contraction benchmark

`tcbench` reuses the full 49-case TCCG corpus (CCSD, AO2MO, InTensLi,
CCSD(T), transposed output). Shapes, signed element strides, seeded random
inputs, alpha=1, beta=0 and output buffers are identical across engines.
The optional upstream baseline calls Lukas Devos's
[tensorprimitives-rs](https://github.com/lkdvos/tensorprimitives-rs) at
`8cda75e11ed26f46c0c22f9629004c84dbabc8e5`, not tprims's imported copy.
The algorithms originate in Matthews, *High-Performance Tensor Contraction
without Transposition*, [arXiv:1607.00291](https://arxiv.org/abs/1607.00291).

| Engine | What is measured |
| --- | --- |
| `plan` | tprims's default planner: packed or copy-free faer |
| `packed` | tprims packed driver forced (diagnostic) |
| `upstream` | Original tensorcontract, explicit `.with_threads(N)`, default scoped-thread policy |
| `tblis` | Actual C++ TBLIS via the existing direct FFI adapter |

Do not label `packed` as external TBLIS or upstream tensorprimitives.

## Fixed procedure

The source of truth is
[the tprims-benchmark skill](../../../.agents/skills/tprims-benchmark/SKILL.md).
Use `RUSTFLAGS="-C target-cpu=native"` and release for all Rust code; TBLIS
must be a Release/native build. TBLIS 2.x requires `--features upstream,tblis`;
1.3 instead requires `upstream,tblis13` (not supported by this 2.x runner).
Installation options: [RESTGroup tblis-rs](https://github.com/RESTGroup/tblis-rs#installation).
System BLIS alone is not TBLIS.

```sh
export TBLIS_ROOT=/path/to/native-release-tblis
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

At least 500 ms of untimed, wall-clock priming per arm (`--prime-ms`, echoed by
`run`) then five repetitions, best wall time, including packing,
executor/FFI entry and provider-internal allocation. Priming is time-based, not
a call count: a fixed number of warm-up calls removes a different share of the
clock ramp in every arm, most in the fastest one. Input generation,
planning/descriptor construction and tprims pool construction are excluded.
tprims retains a caller-owned scratch arena/pool; upstream retains its default
per-call scoped-thread/allocation policy; TBLIS retains its internal policy.
These are library execution comparisons, not isolated microkernel timings.

CSV rows include the thread budget, dtype, equivalent M/N/K, GFLOP/s, latency
and tprims strategy/kernel notes. `manifest.txt` records the base commit,
dirty status, binary SHA256, compiler, flags and actual topology;
`source.patch` records tracked local harness edits. Archive new source files
and TBLIS's build configuration next to results too. The legacy `info` cache
and domain estimates describe the library's model, **not** actual affinity;
use the captured `lscpu` topology to interpret this heterogeneous host.

[2026-10-07 native run](results/2026-10-07-native/README.md).
