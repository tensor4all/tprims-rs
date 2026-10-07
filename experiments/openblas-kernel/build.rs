//! Compile the vendored OpenBLAS kernel with the system C compiler.
//!
//! No `cc` crate: a single `cc -c` invocation is all we need. `-march=native`
//! is required because the kernel is inline-assembly over AVX-512 (zmm)
//! registers; without it the compiler rejects the register clobbers. This makes
//! the experiment x86-64/AVX-512-only by construction, which is recorded in
//! README.md and PROVENANCE.md.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cdir = manifest.join("c");
    let src = cdir.join("dgemm_kernel_16x2_skylakex.c");
    let obj = out.join("dgemm_kernel_16x2_skylakex.o");

    // CNAME is set exactly as OpenBLAS's own Makefile sets it (-DCNAME=$(*F)),
    // so the vendored source needs no textual change.
    let status = Command::new("cc")
        .arg("-O3")
        .arg("-march=native")
        .arg("-DCNAME=dgemm_kernel_16x2_skylakex")
        .arg("-I")
        .arg(&cdir)
        .arg("-c")
        .arg(&src)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to spawn `cc`");

    assert!(
        status.success(),
        "cc failed to compile the OpenBLAS kernel (needs an x86-64 AVX-512 host)"
    );

    // Link the object straight into every binary that links this crate.
    println!("cargo:rustc-link-arg={}", obj.display());
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", cdir.join("common.h").display());
}
