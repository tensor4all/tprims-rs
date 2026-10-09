//! Calls Lukas Devos's tensorprimitives-rs at the Cargo.toml-pinned revision;
//! no upstream source is copied. Algorithm: Matthews, arXiv:1607.00291.
//! Uses the same corpus buffers and raw execution boundary as tprims.
use tensorcontract::{kernel::KernelSet, Element, Layout, Operand, Plan};

use crate::corpus::Sized;

pub fn run<T: Element>(
    s: &Sized,
    threads: usize,
    a: &[T],
    b: &[T],
    d: &mut [T],
    reps: usize,
    prime_ms: u64,
) -> f64
where
    T::Real: KernelSet,
{
    let layout = |l: &crate::corpus::Layout| {
        Layout::new(l.extents().to_vec(), l.strides().to_vec()).expect("valid corpus layout")
    };
    let (la, lb, ld) = (layout(&s.la), layout(&s.lb), layout(&s.lc));
    let p = Plan::new(
        Operand::new(&la, &s.idx_a),
        Operand::new(&lb, &s.idx_b),
        None,
        Operand::new(&ld, &s.idx_c),
    )
    .expect("valid corpus plan")
    .with_threads(threads);
    assert_eq!(p.threads(), threads);
    let mut execute = || {
        // SAFETY: corpus buffers span these layouts, D is exclusive, beta=0
        // leaves C unread. Same raw-pointer boundary as the tprims engine.
        unsafe {
            p.run_raw(
                T::one(),
                a.as_ptr(),
                b.as_ptr(),
                T::zero(),
                std::ptr::null(),
                d.as_mut_ptr(),
            );
        }
    };
    if reps == 0 {
        execute();
        0.0
    } else {
        super::engines::timed(reps, prime_ms, execute)
    }
}
