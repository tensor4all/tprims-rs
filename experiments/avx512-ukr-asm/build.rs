//! Assemble the frozen kernel `.S` with the system C compiler and link it.
//!
//! One `cc -c` invocation, mirroring the `openblas-kernel` experiment. No
//! `-march=native`: the `.S` carries explicit AVX-512 (EVEX) instructions, so
//! the assembler needs no target hint — it encodes exactly what is written.
//! This makes the experiment x86-64/AVX-512-only by construction (recorded in
//! README.md and PROVENANCE.md).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let asm = manifest.join("asm").join("avx512_f64_real_24x8.S");
    let obj = out.join("avx512_f64_real_24x8.o");

    let status = Command::new("cc")
        .arg("-c")
        .arg(&asm)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to spawn `cc`");

    assert!(
        status.success(),
        "cc failed to assemble the frozen kernel (needs an x86-64 AVX-512 host)"
    );

    println!("cargo:rustc-link-arg={}", obj.display());
    println!("cargo:rerun-if-changed={}", asm.display());
}
