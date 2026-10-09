//! Shared plumbing for the harness: element traits, timing and the per-engine
//! runners.

pub mod run;
pub mod verify;

use std::time::Instant;

use num_complex::Complex;
use rand::Rng;
use tprims_contract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use tprims_kernel::Element;

use crate::corpus::{col_major_strides, label_ids, Program, Step};

#[cfg(feature = "tblis")]
use std::os::raw::c_int;

// Concrete implementations avoid mixing the two libraries' Element traits.
macro_rules! upstream_method {
    () => {
        #[cfg(feature = "upstream")]
        fn upstream(
            p: &Program,
            threads: usize,
            inputs: &[Vec<Self>],
            slots: &mut [Vec<Self>],
            reps: usize,
            prime_ms: u64,
        ) -> f64 {
            crate::upstream::run(p, threads, inputs, slots, reps, prime_ms)
        }
    };
}

/// An element type the harness can drive through every engine.
pub trait BenchElem: Scalar {
    const NAME: &'static str;
    fn sample(rng: &mut impl Rng) -> Self;
    /// Multiply by a real scalar: the MPS chain's inputs are scaled.
    fn real_scaled(self, s: f64) -> Self;
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar;
    #[cfg(feature = "upstream")]
    fn upstream(
        p: &Program,
        threads: usize,
        inputs: &[Vec<Self>],
        slots: &mut [Vec<Self>],
        reps: usize,
        prime_ms: u64,
    ) -> f64;
}

impl BenchElem for f64 {
    upstream_method!();
    const NAME: &'static str = "f64";
    fn sample(rng: &mut impl Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn real_scaled(self, s: f64) -> Self {
        self * s
    }
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int = crate::tblis::TYPE_DOUBLE;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar {
        crate::tblis::tblis_scalar::f64(v)
    }
}

impl BenchElem for Complex<f64> {
    upstream_method!();
    const NAME: &'static str = "c64";
    fn sample(rng: &mut impl Rng) -> Self {
        Complex::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn real_scaled(self, s: f64) -> Self {
        Complex::new(self.re * s, self.im * s)
    }
    #[cfg(feature = "tblis")]
    const TBLIS_TYPE: c_int = crate::tblis::TYPE_DCOMPLEX;
    #[cfg(feature = "tblis")]
    fn tblis_scalar(v: f64) -> crate::tblis::tblis_scalar {
        crate::tblis::tblis_scalar::c64(v, 0.0)
    }
}

fn operand(dims: &[usize]) -> OperandSpec {
    let strides = col_major_strides(dims);
    OperandSpec::new(LayoutSpec::new(dims, &strides, 0).expect("corpus layouts are valid"))
}

/// The validated problem of step `k` in storage type `T`: `D` is overwritten
/// (`beta = 0`), so there is no `C`.
pub fn problem_of<T: BenchElem>(p: &Program, k: usize) -> Result<Problem, tprims_contract::Error> {
    let st = &p.steps[k];
    Problem::from_labels(
        T::STORAGE,
        operand(&p.shapes[st.lhs]),
        operand(&p.shapes[st.rhs]),
        CSpec::Absent,
        operand(&p.shapes[p.n_in() + k]),
        &Labels::new(&label_ids(&st.a), &label_ids(&st.b), &label_ids(&st.d)),
    )
}

/// Run a program's steps in order, handing each step's two input slots (already
/// written by earlier steps, or an input tensor) and a mutable output slot to
/// `f`. Each step writes a slot no later step has read yet, so an engine that
/// keeps the intermediates preallocated reuses this one loop.
pub fn steps_with<T, F>(p: &Program, inputs: &[Vec<T>], slots: &mut [Vec<T>], mut f: F)
where
    F: FnMut(usize, &Step, &[T], &[T], &mut [T]),
{
    let n_in = p.n_in();
    for (k, st) in p.steps.iter().enumerate() {
        let (done, rest) = slots.split_at_mut(k);
        let get = |i: usize| -> &[T] {
            if i < n_in {
                &inputs[i]
            } else {
                &done[i - n_in]
            }
        };
        f(k, st, get(st.lhs), get(st.rhs), &mut rest[0]);
    }
}

/// Sample every input tensor of a program once, with the corpus's MPS scaling.
pub fn sample_inputs<T: BenchElem>(p: &Program, seed: u64) -> Vec<Vec<T>> {
    use rand::SeedableRng as _;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    p.inputs
        .iter()
        .map(|d| {
            (0..crate::corpus::numel(d))
                .map(|_| T::sample(&mut rng).real_scaled(p.scale))
                .collect()
        })
        .collect()
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

/// Maximum relative error `max|got - want| / max|want|` — the metric the
/// experiment of #61 checked every arm with. A norm ratio rather than the
/// Frobenius relative error `tcbench` uses, because these programs are chains
/// and a per-step residual is not what the corpus records.
pub fn max_rel_err<T: Element>(got: &[T], want: &[T]) -> f64 {
    assert_eq!(got.len(), want.len());
    let scale = want
        .iter()
        .map(|v| v.norm())
        .fold(0.0f64, f64::max)
        .max(f64::MIN_POSITIVE);
    got.iter()
        .zip(want)
        .map(|(&g, &w)| Element::sub(g, w).norm())
        .fold(0.0f64, f64::max)
        / scale
}

/// Tolerance for [`max_rel_err`], by the real type's width.
pub fn tol<T: Element>() -> f64 {
    if core::mem::size_of::<T::Real>() == 4 {
        1e-3
    } else {
        1e-10
    }
}

/// Set provider budgets and confirm the linked TBLIS ABI.
pub fn configure_threads(threads: usize) {
    let _ = threads;
    #[cfg(feature = "tblis")]
    unsafe {
        crate::tblis::tblis_set_num_threads(
            threads.try_into().expect("TBLIS thread count fits u32"),
        );
        assert_eq!(crate::tblis::tblis_get_num_threads() as usize, threads);
        if let Err(e) = crate::tblis::verify_type_tags() {
            eprintln!("FATAL: {e}");
            std::process::exit(2);
        }
    };
}
