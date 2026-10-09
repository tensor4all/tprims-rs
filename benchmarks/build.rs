//! Locate the optional C baselines.
//!
//! * `blas`  feature: needs `OPENBLAS_ROOT` or `BLAS_ROOT`, or a system
//!   `libopenblas` on the default search path.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=OPENBLAS_ROOT");
    println!("cargo:rerun-if-env-changed=BLAS_ROOT");

    if env::var("CARGO_FEATURE_BLAS").is_ok() {
        // Accelerate is a framework, not a library on a search path, and it
        // supplies the classic CBLAS entry points this crate declares. The
        // `framework=` link kind is the only way to express `-framework`;
        // `dylib=openblas` cannot.
        if env::var("CARGO_FEATURE_ACCELERATE").is_ok() {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        } else {
            if let Ok(root) = env::var("OPENBLAS_ROOT").or_else(|_| env::var("BLAS_ROOT")) {
                for sub in ["lib", "lib64"] {
                    println!("cargo:rustc-link-search=native={root}/{sub}");
                }
            }
            println!("cargo:rustc-link-lib=dylib=openblas");
        }
    }
}
