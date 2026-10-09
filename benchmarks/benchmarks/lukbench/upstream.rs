//! Calls Lukas Devos's tensorprimitives-rs at the Cargo.toml-pinned revision;
//! no upstream source is copied. Algorithm: Matthews, arXiv:1607.00291.
//! One prebuilt `Plan` per step into preallocated slots, and the whole program
//! is one timed call — the same raw-execution boundary the `plan` arm uses.
use tensorcontract::{kernel::KernelSet, Element, Layout, Operand, Plan};

use crate::corpus::{dims_strides_i64, label_ids, Program};

pub fn run<T: Element>(
    p: &Program,
    threads: usize,
    inputs: &[Vec<T>],
    slots: &mut [Vec<T>],
    reps: usize,
    prime_ms: u64,
) -> f64
where
    T::Real: KernelSet,
{
    let n_in = p.n_in();
    let layouts: Vec<Layout> = p
        .shapes
        .iter()
        .map(|d| {
            let (extents, strides) = dims_strides_i64(d);
            Layout::new(extents, strides).expect("valid corpus layout")
        })
        .collect();
    let plans: Vec<Plan> = p
        .steps
        .iter()
        .enumerate()
        .map(|(k, st)| {
            Plan::new(
                Operand::new(&layouts[st.lhs], &label_ids(&st.a)),
                Operand::new(&layouts[st.rhs], &label_ids(&st.b)),
                None,
                Operand::new(&layouts[n_in + k], &label_ids(&st.d)),
            )
            .expect("valid corpus plan")
            .with_threads(threads)
        })
        .collect();
    assert!(
        plans.iter().all(|pl| pl.threads() == threads),
        "tensorcontract threads != {threads}"
    );
    let mut execute = || {
        for (k, st) in p.steps.iter().enumerate() {
            let (done, rest) = slots.split_at_mut(k);
            let get = |i: usize| -> &[T] {
                if i < n_in {
                    &inputs[i]
                } else {
                    &done[i - n_in]
                }
            };
            // SAFETY: steps write each slot at most once and read only slots an
            // earlier step wrote, the buffers span their layouts, D is exclusive
            // and beta = 0 leaves C unread.
            unsafe {
                plans[k].run_raw(
                    T::one(),
                    get(st.lhs).as_ptr(),
                    get(st.rhs).as_ptr(),
                    T::zero(),
                    std::ptr::null(),
                    rest[0].as_mut_ptr(),
                );
            }
        }
    };
    if reps == 0 {
        execute();
        0.0
    } else {
        crate::engines::timed(reps, prime_ms, execute)
    }
}
