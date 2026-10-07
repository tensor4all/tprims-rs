//! The micro-kernel's rate when its panels do *not* live in L1.
//!
//! `multicore.rs` reuses one A panel and one B panel, so the kernel reads from
//! L1 and reports ~52.7 GFLOP/s per core. A real GEMM walks panels out of a
//! buffer that is far larger than L1, so the kernel waits on L2/L3/DRAM
//! instead. This probe holds the kernel, the tile shape and the instruction
//! sequence fixed and varies only the **A working-set size**, which separates
//! "the driver's byte volume is expensive" from "the kernel is slower when fed
//! from memory".
//!
//! It measures the kernel, not a contraction: there is no packing, no output
//! write-back and no blocking here.
//!
//! Run pinned, with the same warm-up discipline as `multicore.rs`:
//!     benchmarks/scripts/pinned.sh 4 -- ./target/release/stream

use std::hint::black_box;
use std::time::Instant;

use avx512_ukr_asm::{intrinsic_ukr, Ukr, MR, NR};

const A_PANEL: usize = MR * 1024; // sized for the largest kc tried
const B_PANEL: usize = NR * 1024;

fn fill(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = seed | 1;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    (0..n).map(|_| next()).collect()
}

/// ns per call over `calls` calls that cycle through `panels` A panels.
/// `stride_a`/`stride_b` are the panel strides for this `kc`.
fn run(
    ukr: Ukr,
    kc: usize,
    a: &[f64],
    b: &[f64],
    panels: usize,
    ab: &mut [f64],
    calls: usize,
) -> f64 {
    let stride_a = MR * kc;
    let stride_b = NR * kc;
    let b_panels = b.len() / stride_b;
    let t = Instant::now();
    for i in 0..calls {
        let p = i % panels;
        let bp = (i % b_panels) * stride_b;
        // SAFETY: p < panels and the buffer holds panels*stride_a; the B panel
        // is in range by construction; ab is MR*NR.
        unsafe {
            ukr(
                kc,
                a.as_ptr().add(p * stride_a),
                b.as_ptr().add(bp),
                ab.as_mut_ptr(),
            )
        };
    }
    black_box(&*ab);
    t.elapsed().as_secs_f64() * 1e9 / calls as f64
}

fn main() {
    let ukr = intrinsic_ukr();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if std::env::args().any(|a| a == "--help") {
        eprintln!("usage: stream THREADS [KC...]");
        return;
    }
    let threads: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(1);
    let kcs: Vec<usize> = args
        .iter()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let kcs = if kcs.is_empty() {
        vec![64, 256]
    } else {
        kcs
    };
    let a = fill(32 << 20, 0x2545_F491_4F6C_DD1D);
    let b = fill(4 << 20, 0x9E37_79B9_7F4A_7C15);
    println!("threads,kc,a_bytes,b_bytes,a_panels,b_panels,ns_per_call,gflops,per_core");
    for &kc in &kcs {
        let flops = 2.0 * kc as f64 * MR as f64 * NR as f64;
        let calls = (40_000_000 / kc).max(20_000);
        let a_panels = (a.len() / (MR * kc)).max(1);
        let b_panels = (b.len() / (NR * kc)).max(1);
        // One timed pass with every thread reading the same two buffers: this
        // is the shared-panel case, where B is shared and A is per-thread.
        let pass = || {
            let mut per_thread = vec![0.0f64; threads * MR * NR];
            // Addresses, not pointers: a raw pointer is not `Send`, and the
            // threads below each own exactly one MR x NR slot.
            let base = per_thread.as_mut_ptr() as usize;
            let slots: Vec<usize> = (0..threads)
                .map(|i| base + i * MR * NR * std::mem::size_of::<f64>())
                .collect();
            let t = Instant::now();
            std::thread::scope(|s| {
                for slot in slots {
                    let (a, b) = (&a, &b);
                    s.spawn(move || {
                        let mut ab = unsafe {
                            // SAFETY: each thread gets its own slot, and no two
                            // slots overlap.
                            std::slice::from_raw_parts_mut(slot as *mut f64, MR * NR)
                        };
                        ab.fill(0.0);
                        run(ukr, kc, a, b, a_panels, &mut ab, calls)
                    });
                }
            });
            t.elapsed().as_secs_f64()
        };
        // Warm-up long enough that the clock has settled; see the worklog note.
        let deadline = Instant::now() + std::time::Duration::from_millis(1500);
        while Instant::now() < deadline {
            let _ = pass();
        }
        let best = (0..3).map(|_| pass()).fold(f64::INFINITY, f64::min);
        // Every thread performs the same number of calls, and they run
        // concurrently, so the aggregate multiplies by the thread count.
        let ns = best * 1e9 / calls as f64;
        let aggregate = flops * threads as f64 / ns;
        println!(
            "{threads},{kc},{},{},{a_panels},{b_panels},{ns:.3},{aggregate:.1},{:.1}",
            a.len() * 8,
            b.len() * 8,
            aggregate / threads as f64
        );
    }
}
