//! Correctness self-check: intrinsic kernel and frozen-assembly kernel against
//! a naive Rust reference, for `kc` in `{0,1,2,3,7,16,33}` and for exact
//! small-integer inputs, using a relative Frobenius residual.
//!
//! Both kernels must match the reference (relative Frobenius residual within
//! tolerance) and — because the frozen body is a copy of the native intrinsic
//! body — must be bit-identical to each other.

use avx512_ukr_asm::{frozen_ukr, intrinsic_ukr, reference_real, Ukr, MR, NR};

fn check(what: &str, ok: bool) {
    if ok {
        println!("ok    {what}");
    } else {
        println!("FAIL  {what}");
        std::process::exit(1);
    }
}

/// Relative Frobenius residual `||got - want||_F / max(||want||_F, eps)`.
/// `want == 0` (only possible at `kc = 0`) demands `got == 0` exactly.
fn rel_frob(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let num = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w) * (g - w))
        .sum::<f64>()
        .sqrt();
    let den = want.iter().map(|w| w * w).sum::<f64>().sqrt();
    if den == 0.0 {
        if num == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        num / den
    }
}

/// Deterministic `(-1, 1)` values, no external RNG dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x as f64 / u64::MAX as f64) * 2.0 - 1.0
    }
}

/// Run one kernel on the given panels and return the `MR x NR` tile.
///
/// # Safety
/// Panels and tile have the exact extents the kernel declares.
unsafe fn run(ukr: Ukr, kc: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut ab = vec![f64::NAN; MR * NR];
    // SAFETY: caller-sized panels/tile.
    unsafe { ukr(kc, a.as_ptr(), b.as_ptr(), ab.as_mut_ptr()) };
    ab
}

fn tile_case(kc: usize, rng: &mut Rng) {
    let a: Vec<f64> = (0..MR * kc).map(|_| rng.next()).collect();
    let b: Vec<f64> = (0..NR * kc).map(|_| rng.next()).collect();
    let want = reference_real(kc, &a, &b);
    // SAFETY: exact sized panels and exclusive tile.
    let got_i = unsafe { run(intrinsic_ukr(), kc, &a, &b) };
    // SAFETY: exact sized panels and exclusive tile.
    let got_f = unsafe { run(frozen_ukr(), kc, &a, &b) };
    check(
        &format!("intrinsic kc={kc}"),
        rel_frob(&got_i, &want) <= 1e-12,
    );
    check(
        &format!("frozen    kc={kc}"),
        rel_frob(&got_f, &want) <= 1e-12,
    );
    check(&format!("bit-identical kc={kc}"), got_i == got_f);
}

/// Exact small-integer panels: every product and partial sum is an exactly
/// representable `f64`, so both kernels must match the reference exactly.
fn exact_case() {
    for kc in [0usize, 1, 2, 3, 7, 16, 33] {
        let a: Vec<f64> = (0..MR * kc)
            .map(|n| {
                let i = n % MR;
                let p = n / MR;
                (i as f64 + 1.0 + 2.0 * p as f64) as f64
            })
            .collect();
        let b: Vec<f64> = (0..NR * kc)
            .map(|n| {
                let j = n % NR;
                let p = n / NR;
                (j as f64 + 3.0 - p as f64) as f64
            })
            .collect();
        let want = reference_real(kc, &a, &b);
        // SAFETY: exact sized panels and exclusive tile.
        let got_i = unsafe { run(intrinsic_ukr(), kc, &a, &b) };
        // SAFETY: exact sized panels and exclusive tile.
        let got_f = unsafe { run(frozen_ukr(), kc, &a, &b) };
        check(&format!("exact intrinsic kc={kc}"), got_i == want);
        check(&format!("exact frozen    kc={kc}"), got_f == want);
    }
}

fn main() {
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("this host has no AVX-512F; cannot run the kernels");
        std::process::exit(2);
    }
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for kc in [0usize, 1, 2, 3, 7, 16, 33] {
        tile_case(kc, &mut rng);
    }
    exact_case();
    println!("all self-checks passed");
}
