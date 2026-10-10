//! Optional per-phase timing of the packed driver.
//!
//! Off unless the `phase-timing` feature is on, and every call site is behind
//! `#[cfg(feature = "phase-timing")]`, so a default build pays nothing. It is an
//! instrument in the same spirit as `benchmarks/scripts/idle_cpus.py`: it answers
//! "which phase of the packed route absorbs this case's time", which nothing else
//! in the tree can on this host - `perf` events are blocked
//! (`kernel.perf_event_paranoid` is 4), `/proc/<pid>/stat` masks the user PC, and
//! Yama refuses the ptrace a sampler would need.
//!
//! Read it a call late: the counters are printed and cleared at the *start* of the
//! next `execute_raw`, so one line describes the call that just finished. The last
//! four numbers are the phases; `setup` is the whole call minus those four, i.e.
//! the loops, the epoch bookkeeping and everything between the phases.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Phase names, indexed like the counters.
pub const NAMES: [&str; 5] = ["pack_a", "pack_b", "kernel", "writeback", "setup"];

static NANOS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static TOTAL: AtomicU64 = AtomicU64::new(0);

/// Accumulates one phase while it is alive.
#[derive(Debug)]
pub struct Scope(usize, Instant);

impl Drop for Scope {
    fn drop(&mut self) {
        NANOS[self.0].fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

/// Start timing phase `i` of [`NAMES`].
#[inline]
pub fn scope(i: usize) -> Scope {
    assert!(i < 4, "phase index");
    Scope(i, Instant::now())
}

/// Record a whole `execute_raw` call, so the remainder can be reported as setup.
#[inline]
pub fn add_total(nanos: u64) {
    TOTAL.fetch_add(nanos, Ordering::Relaxed);
}

/// Print the previous call's phase totals and clear them, when `TPRIMS_PHASE` is set.
pub fn report_previous() {
    if std::env::var_os("TPRIMS_PHASE").is_none() {
        return;
    }
    let total = TOTAL.swap(0, Ordering::Relaxed);
    let mut ns = [0u64; 5];
    let mut accounted = 0u64;
    for i in 0..4 {
        ns[i] = NANOS[i].swap(0, Ordering::Relaxed);
        accounted += ns[i];
    }
    ns[4] = total.saturating_sub(accounted);
    if total == 0 {
        return;
    }
    let parts: Vec<String> = NAMES
        .iter()
        .zip(ns)
        .map(|(n, v)| {
            format!(
                "{n}={:.3}ms({:>4.1}%)",
                v as f64 / 1e6,
                100.0 * v as f64 / total as f64
            )
        })
        .collect();
    eprintln!(
        "PHASE total={:.3}ms {}",
        total as f64 / 1e6,
        parts.join(" ")
    );
}
