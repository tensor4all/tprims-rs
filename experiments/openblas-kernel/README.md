# openblas-kernel — isolated f64 kernel-import experiment

Imports OpenBLAS's `dgemm_kernel_16x2_skylakex` (the SkylakeX/Cooperlake f64
micro-kernel) as an external [`KernelFamily`] for the tprims packed driver, to
measure it against the project's own kernels on equal footing (same driver,
packers, write-back and cache blocking).

## What is imported

- `c/dgemm_kernel_16x2_skylakex.c` — vendored **byte-for-byte** from OpenBLAS
  commit `31e82fa8c509e6f0d96288de3c20d8916d894e72`
  (`kernel/x86_64/dgemm_kernel_16x2_skylakex.c`), BSD-3-Clause (see `LICENSE`
  and `PROVENANCE.md`).
- `c/common.h` — a **project-owned shim** (MIT OR Apache-2.0) supplying only
  `typedef int64_t BLASLONG`. OpenBLAS's own `common.h` is *not* vendored.

The symbol name is fixed on the compile line (`-DCNAME=...`), exactly as
OpenBLAS's own `Makefile.system` does, so the vendored source needs no edit.

## Zen 5 note

Zen 5 has no Zen-5-exclusive f64 kernel: OpenBLAS maps Zen 5 to the
Cooperlake/SkylakeX f64 kernel, which is this one. The family id and ISA label
say so honestly; this is "the kernel OpenBLAS uses on Zen 5", not "a Zen 5
kernel".

## Adapter

The kernel is a `16 x 2` panel product over column-major packed panels, which
matches tprims's `PackFormat::Real` / `TileFormat::Real` contract exactly
(A `A[i + 16p]`, B `B[j + 2p]`, tile `C[j*16 + i]`). The only adapter is the
scratch-tile wrapper: zero the tile, then call the kernel with `alpha = 1.0`,
`ldc = 16` (the kernel always accumulates `C += alpha*A*B`; there is no beta
parameter at this level). No packing or contraction logic is reimplemented —
the tprims driver and packers are reused. **No overhead adapter**: the packing
and tile shapes are a direct match, so the only cost is one `32`-element
zero-fill per tile.

## Layout

- `src/lib.rs` — the `KernelFamily` descriptor, the tile wrapper, and the
  `KernelCatalog` admission.
- `src/bin/selfcheck.rs` — correctness: tile product vs naive reference
  (K0/K1/tails), exact known values, and end-to-end (default tprims vs
  OpenBLAS) against a naive contraction, matrix and strided cases.
- `src/bin/bench.rs` — CSV A/B comparison at `--threads 1,4,8,12`.
- `build.rs` — one `cc -O3 -march=native -DCNAME=... -c` invocation (no `cc`
  crate).

## Results

`results/2026-10-10/` (1T, `--reps 5`, 1500 ms of priming per arm, one core of one L3
domain, the default family versus the imported kernel through the same driver, packers,
blocking and write-back):

| case | tprims (`avx512.f64.real.24x8`) | OpenBLAS (`dgemm_kernel_16x2_skylakex`) | ratio |
|---|---|---|---|
| `gemm-1024-col` | 47.6 GFLOP/s | 29.1 GFLOP/s | 1.64x |
| `gemm-512-col` | 47.1 | 29.4 | 1.60x |
| `strided-1024-rowa` | 47.2 | 29.2 | 1.62x |
| `gemm-92160x40x48-col` | 29.4 | 21.1 | 1.39x |
| `gemm-3456x3456x24-col` | 39.0 | 19.9 | 1.96x |

Two readings:

- **The default kernel is faster than this production kernel on every shape tested,
  including the class the tensor corpus separates on.** `gemm-92160x40x48-col` is the
  effective GEMM of `abjc-cbka-kj` at 16 MiB (`m` = 92160, `n` = 40, `k` = 48), where the
  reference beats this library by 1.5x on the tensor corpus; here the default kernel is
  1.39x *ahead* of the imported one. The micro-kernel is therefore not what that loss is
  made of, and swapping kernels would make it worse.
- **`results/2026-10-07/` was taken with a single warm-up call per arm**, before this
  bench primed by wall clock. That leaves the arm measured first reading up to 40% low
  on this host (see `PERFORMANCE_TIPS.md`), and the default family is always that arm
  here, so those numbers understated it. They stay as history; use the 2026-10-10 run.

What this experiment does **not** show: every case here is a *contiguous* column-major
GEMM, and the tensor corpus is not. In `abjc-cbka-kj` the folded `m` axis is `j`, whose
stride in `A = abjc` is `a*b`, and the contracted group `(a,b,c)` has strides `(1, a,
a*b*j)`, so the packing reads a strided, interleaved panel rather than a matrix. The
same dimensions run at 29.4 GFLOP/s contiguously here and at 9.3 GFLOP/s inside that
tensor case, which is where the loss lives; reproducing it needs a tensor `Problem`
with the corpus's label sets, not a matrix.

## Traversal attribution

`src/bin/traversal.rs` runs **one** contraction - the campaign's worst losing TCCG case,
`abjc-cbka-kj` f64 16 MiB, effective GEMM m=92160 n=40 k=48 - under four operand layouts:
the case's own plan-recorded "corpus" strides versus compact column-major matrices, for
`A` and for the output independently. Same family, same blocking, same thread budget; only
the layout changes, so the four rows attribute the cost to reading `A`, writing the
output, both, or neither.

`results/2026-10-10-traversal/` (1T, best of 5, 1500 ms per variant):

| variant | ms | GFLOP/s |
|---|---|---|
| `corpus-a-corpus-d` | 11.62 | 30.5 |
| `corpus-a-compact-d` | 11.41 | 31.0 |
| `compact-a-corpus-d` | 34.29 | 10.3 |
| `compact-a-compact-d` | 33.92 | 10.4 |

The lesson is not the one the file was written to test. The same dimensions **and the same
operand layouts** as the campaign case ran here in 11.6 ms, while the campaign's own plan
takes 38.1 ms (9.3 GFLOP/s) on that case with the identical family (`avx512.f64.real.24x8`)
and blocking (264/1536/256). The difference is the *plan*: this binary's plan folds the
`m` group into two axes (`PlanStats.m_axes`), the campaign's keeps three. So a plan-level
traversal decision is worth 3.3x on this problem, and the pack-read ordering - which is
what a reader would expect to matter here - is not what separates these rows: swapping `A`
between the corpus and compact layouts moves nothing (11.62 vs 11.41).

Read the `residual` column with care: the variants place the same result in differently
permuted buffers, so the column compares unequal positions and its value is meaningless as
a correctness check. The `A`/`D` layouts and the dimensions are faithful to the plan's
record; the *plan* built from them is not the campaign's.

## Build & run

```sh
# from this directory
cargo build --release
cargo run --release --bin selfcheck
cargo run --release --bin bench -- --threads 1,4,8,12
```

## Limitations

- **x86-64 + AVX-512F only.** `-march=native` compiles the inline-asm kernel;
  on a host without AVX-512 the build fails at compile time. The family
  declares `required.avx512f`.
- **Not a Zen 5 kernel** (see above); it is the SkylakeX/Cooperlake f64 kernel
  OpenBLAS selects for Zen 5.
- The kernel does **not** apply `alpha`/`beta` or conjugation; the tprims
  driver's write-back does, and conjugation is unsupported (`caps.conj_* =
  false`), which real f64 never needs.
- `bench.rs` does no affinity pinning; run it under the `tprims-benchmark`
  skill protocol for any number worth recording.
- Every case is a contiguous column-major GEMM; the tensor corpus's strided,
  interleaved panels are not covered here (see Results).
- `traversal.rs` covers those panels, but its plan is not the campaign's plan
  (see Traversal attribution); it attributes a *decision*, not a cell.

[`KernelFamily`]: https://docs.rs/tprims-kernel/latest/tprims_kernel/struct.KernelFamily.html
