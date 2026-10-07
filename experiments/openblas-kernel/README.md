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

[`KernelFamily`]: https://docs.rs/tprims-kernel/latest/tprims_kernel/struct.KernelFamily.html
