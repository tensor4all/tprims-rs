//! Shared plumbing for the harness: element traits, timing, and the
//! per-engine runners.

pub mod run;
pub mod verify;

use std::time::Instant;

use num_complex::Complex;
use rand::Rng;
use tprims_contract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use tprims_kernel::Element;

use crate::blas::GemmScalar;
use crate::corpus::{Layout, Sized};

// Concrete implementations avoid mixing the two libraries' Element traits.
macro_rules! upstream_method {
    () => {
        #[cfg(feature = "upstream")]
        fn upstream(
            s: &Sized,
            threads: usize,
            a: &[Self],
            b: &[Self],
            d: &mut [Self],
            reps: usize,
            prime_ms: u64,
        ) -> f64 {
            crate::upstream::run(s, threads, a, b, d, reps, prime_ms)
        }
    };
}

/// An element type the harness can drive through every engine.
#[allow(dead_code)] // some members are only used under optional features
pub trait BenchElem: Scalar + GemmScalar {
    const NAME: &'static str;
    /// The same shape in the corresponding real type, for ratio reporting.
    const REAL_NAME: &'static str;

    fn sample(rng: &mut impl Rng) -> Self;
    fn from_f64(v: f64) -> Self;
    #[cfg(feature = "upstream")]
    fn upstream(
        s: &Sized,
        threads: usize,
        a: &[Self],
        b: &[Self],
        d: &mut [Self],
        reps: usize,
        prime_ms: u64,
    ) -> f64;
}

impl BenchElem for f32 {
    upstream_method!();
    const NAME: &'static str = "f32";
    const REAL_NAME: &'static str = "f32";
    fn sample(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl BenchElem for f64 {
    upstream_method!();
    const NAME: &'static str = "f64";
    const REAL_NAME: &'static str = "f64";
    fn sample(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl BenchElem for Complex<f32> {
    upstream_method!();
    const NAME: &'static str = "c32";
    const REAL_NAME: &'static str = "f32";
    fn sample(rng: &mut impl Rng) -> Self {
        Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn from_f64(v: f64) -> Self {
        Complex::new(v as f32, 0.0)
    }
}

impl BenchElem for Complex<f64> {
    upstream_method!();
    const NAME: &'static str = "c64";
    const REAL_NAME: &'static str = "f64";
    fn sample(rng: &mut impl Rng) -> Self {
        Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn from_f64(v: f64) -> Self {
        Complex::new(v, 0.0)
    }
}

fn operand(l: &Layout) -> OperandSpec {
    let dims: Vec<usize> = l.extents().iter().map(|&e| e as usize).collect();
    let strides: Vec<isize> = l.strides().iter().map(|&s| s as isize).collect();
    OperandSpec::new(LayoutSpec::new(&dims, &strides, 0).expect("corpus layouts are valid"))
}

/// The validated problem of a sized corpus case in storage type `T`: `D` is
/// overwritten (`beta = 0`), so there is no `C`.
pub fn problem_of<T: BenchElem>(s: &Sized) -> Result<Problem, tprims_contract::Error> {
    Problem::from_labels(
        T::STORAGE,
        operand(&s.la),
        operand(&s.lb),
        CSpec::Absent,
        operand(&s.lc),
        &Labels::new(&s.idx_a, &s.idx_b, &s.idx_c),
    )
}

/// Best-of-`reps` wall time in seconds, after time-based priming.
///
/// Priming is a wall-clock duration, not a call count: after an idle gate this
/// host reads up to 25% low for the first one to two seconds of sustained
/// AVX-512 work, and a fixed number of warm-up calls removes a different share
/// of that bias in every arm (largest in the fastest arm). `prime_ms == 0`
/// skips priming.
///
/// Minimum rather than mean: these are deterministic compute kernels, so the
/// spread is machine noise (frequency, interrupts, other tenants) and the
/// minimum is the least contaminated estimator.
pub fn timed(reps: usize, prime_ms: u64, mut f: impl FnMut()) -> f64 {
    let until = Instant::now() + std::time::Duration::from_millis(prime_ms);
    while Instant::now() < until {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

/// GFLOP/s given a multiply-accumulate count and a time.
pub fn gflops<T: Element>(macs: u64, secs: f64) -> f64 {
    (macs as f64) * (<T as Element>::FLOPS_PER_MAC as f64) / secs / 1e9
}

/// Relative Frobenius error `||got - want|| / ||want||`.
pub fn rel_error<T: Element>(got: &[T], want: &[T]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (&g, &w) in got.iter().zip(want) {
        num += Element::sub(g, w).norm().powi(2);
        den += Element::norm(w).powi(2);
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

/// Set provider budgets.
pub fn configure_threads(threads: usize) {
    let _ = threads;
    // Accelerate has no equivalent: it reads `VECLIB_MAXIMUM_THREADS` once at
    // first use, so pinning it is the caller's job and cannot be done from here.
    #[cfg(all(feature = "blas", not(feature = "accelerate")))]
    unsafe {
        crate::blas::openblas_set_num_threads(
            threads.try_into().expect("BLAS thread count fits i32"),
        )
    };
}
