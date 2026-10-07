# Provenance

## Imported artifact

| | |
|---|---|
| Source repository | https://github.com/OpenMathLib/OpenBLAS |
| Commit | `31e82fa8c509e6f0d96288de3c20d8916d894e72` |
| Upstream path | `kernel/x86_64/dgemm_kernel_16x2_skylakex.c` |
| Local path | `c/dgemm_kernel_16x2_skylakex.c` |
| Import method | byte-for-byte copy, no edits |
| License | BSD-3-Clause (full text in `LICENSE`) |
| Copyright | Copyright (c) 2011-2014, The OpenBLAS Project |

The upstream file carries no per-file copyright header; the project LICENSE
(applied by OpenBLAS to the whole tree) is the notice, retained verbatim as
`LICENSE`.

## Callable symbols

Exactly one C symbol is compiled and linked:

- `dgemm_kernel_16x2_skylakex`
  `(BLASLONG m, BLASLONG n, BLASLONG k, double alpha, double *A, double *B, double *C, BLASLONG ldc)`
  where `BLASLONG` is `int64_t` (project shim, `c/common.h`).

`CNAME` is fixed to `dgemm_kernel_16x2_skylakex` on the compile line
(`-DCNAME=...`), mirroring OpenBLAS's `Makefile.system` (`-DCNAME=$(*F)`).

## Project-owned files

- `c/common.h` — `typedef int64_t BLASLONG;` only. MIT OR Apache-2.0
  (tensor4all). Not derived from OpenBLAS's `common.h`.
- `build.rs`, `src/`, `README.md`, `PROVENANCE.md` — MIT OR Apache-2.0.

## Zen 5 mapping

OpenBLAS's parameter selection maps Zen 5 to the Cooperlake/SkylakeX f64
kernel; there is no Zen-5-exclusive f64 kernel. This import is therefore
labelled `skylakex`, not `zen5`, and the family id records that.
