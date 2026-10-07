//! Does the micro-kernel keep its single-core rate when every core runs it?
//!
//! A 24x8 f64 micro-kernel reaches ~52.7 GFLOP/s on one core here, but a
//! 1024^3 GEMM reaches only ~231 GFLOP/s on 8 cores, i.e. ~29 per core. The
//! two candidate explanations are (a) the driver/packing costs the difference,
//! or (b) the cores run slower when all of them are busy. This probe separates
//! them: it runs the *same kernel, same data, same loop* on N threads and
//! reports the per-thread rate. If per-thread throughput falls toward 29
//! GFLOP/s as threads are added, the gap is core clock/power, not the driver.
//!
//! It measures nothing about contractions and makes no claim beyond the kernel.
//! Pin with the project protocol, one L3 domain first:
//!     benchmarks/scripts/pinned.sh 4-7  -- ./target/release/multicore 4
//!     benchmarks/scripts/pinned.sh 4-11 -- ./target/release/multicore 8

use std::hint::black_box;
use std::time::Instant;

use avx512_ukr_asm::{intrinsic_ukr, Ukr, MR, NR};

const KC: usize = 64;

fn batch_ns(ukr: Ukr, kc: usize, a: &[f64], b: &[f64], ab: &mut [f64], iters: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..iters {
        // SAFETY: sized panels for this kc, exclusive tile.
        unsafe { ukr(kc, a.as_ptr(), b.as_ptr(), ab.as_mut_ptr()) };
    }
    black_box(&*ab);
    t.elapsed().as_secs_f64() * 1e9 / iters as f64
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    assert!(threads >= 1 && threads <= 16);
    let iters = 20_000_000 / KC;
    let ukr = intrinsic_ukr();

    let workers: Vec<_> = (0..threads)
        .map(|_| {
            std::thread::spawn(move || {
                let mut rng = 0x2545_F491_4F6C_DD1Du64;
                let mut next = || {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
                };
                let a: Vec<f64> = (0..MR * KC).map(|_| next()).collect();
                let b: Vec<f64> = (0..NR * KC).map(|_| next()).collect();
                let mut ab = vec![0.0f64; MR * NR];
                // Warm the core for a fixed wall time before anything is
                // reported. A short FMA burst started right after an idle gate
                // measures the clock ramp, not the core: the same probe reads
                // ~40 GFLOP/s over ~1 s and ~52.7 GFLOP/s once the clock has
                // settled, which is why this loop is time-based rather than a
                // fixed number of calls.
                let deadline = Instant::now() + std::time::Duration::from_millis(2000);
                while Instant::now() < deadline {
                    let _ = batch_ns(ukr, KC, &a, &b, &mut ab, iters / 4);
                }
                (0..5)
                    .map(|_| batch_ns(ukr, KC, &a, &b, &mut ab, iters))
                    .fold(f64::INFINITY, f64::min)
            })
        })
        .collect();
    let per_thread: Vec<f64> = workers.into_iter().map(|w| w.join().unwrap()).collect();

    // flops per nanosecond is already GFLOP/s.
    let flops_per_call = 2.0 * KC as f64 * MR as f64 * NR as f64;
    let best = per_thread.iter().cloned().fold(f64::INFINITY, f64::min);
    let worst = per_thread.iter().cloned().fold(0.0f64, f64::max);
    let per_core = flops_per_call / best;
    let aggregate = per_thread.iter().map(|&ns| flops_per_call / ns).sum::<f64>();
    println!("threads={threads} kc={KC}");
    for (i, ns) in per_thread.iter().enumerate() {
        println!(
            "  thread {i}: {ns:8.3} ns/call  {:.1} GFLOP/s",
            flops_per_call / ns
        );
    }
    println!(
        "best-per-core={per_core:.1} GFLOP/s  worst-per-core={:.1} GFLOP/s  aggregate={aggregate:.1} GFLOP/s",
        flops_per_call / worst
    );
}
