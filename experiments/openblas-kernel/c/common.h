/* Project-owned shim (tensor4all, MIT OR Apache-2.0).
 *
 * The vendored OpenBLAS kernel (c/dgemm_kernel_16x2_skylakex.c) includes
 * "common.h" and <stdint.h>. This file supplies the two things the kernel
 * needs and nothing else:
 *
 *   - BLASLONG, the integer width of the kernel's ABI. OpenBLAS's own
 *     common.h types this as `long long` (or `long`); we pin it to int64_t
 *     so the Rust extern declaration is `i64` on every platform we build.
 *   - CNAME is *not* defined here. It is supplied on the compile line
 *     (-DCNAME=dgemm_kernel_16x2_skylakex), exactly as OpenBLAS's Makefile
 *     does (Makefile.system passes -DCNAME=$(*F)). That keeps the vendored
 *     file byte-for-byte identical to upstream while fixing the symbol name.
 */
#ifndef OPENBLAS_KERNEL_COMMON_H
#define OPENBLAS_KERNEL_COMMON_H

#include <stdint.h>

typedef int64_t BLASLONG;

#endif /* OPENBLAS_KERNEL_COMMON_H */
