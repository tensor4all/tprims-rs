//! How fast this core retires 512-bit FMAs, as a ceiling for the micro-kernel.
//!
//! The 24x8 f64 micro-kernel measured ~52.7 GFLOP/s at 1T. That number is only
//! interpretable against the machine's FMA ceiling, which is what this probe
//! measures: independent accumulator chains, no memory traffic in the loop, so
//! the only limit is FMA issue width.
//!
//! It is a *ceiling probe*, not a kernel, and makes no claim about contractions.
//! Every run uses an opaque and different iteration count, and the sum is
//! printed, so the loop cannot be folded away or common-subexpression
//! eliminated across runs.
//!
//! Run pinned: `benchmarks/scripts/pinned.sh 4 ./target/release/peak`

use std::hint::black_box;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[target_feature(enable = "avx512f")]
unsafe fn chains<const ACCS: usize>(iters: usize) -> f64 {
    let mut acc = [_mm512_setzero_pd(); ACCS];
    let a = _mm512_set1_pd(black_box(1.0 + f64::EPSILON));
    let b = _mm512_set1_pd(black_box(1.0 - f64::EPSILON));
    for _ in 0..iters {
        for j in 0..ACCS {
            acc[j] = _mm512_fmadd_pd(a, b, acc[j]);
        }
    }
    let mut sum = 0.0;
    for x in acc {
        let mut lane = [0.0f64; 8];
        // SAFETY: `x` is a 512-bit register and `lane` holds 8 f64.
        unsafe { _mm512_storeu_pd(lane.as_mut_ptr(), x) };
        sum += lane.iter().sum::<f64>();
    }
    sum
}

#[cfg(target_arch = "x86_64")]
fn measure<const ACCS: usize>(iters: usize) -> (f64, f64) {
    let mut checksum = 0.0;
    let mut best = f64::INFINITY;
    for r in 0..7 {
        // Opaque and run-dependent, so no two calls can be merged or hoisted.
        let n = black_box(iters + r * 4099);
        let t = Instant::now();
        // SAFETY: the caller checked `avx512f`.
        let v = unsafe { chains::<ACCS>(n) };
        let s = t.elapsed().as_secs_f64();
        checksum += black_box(v);
        best = best.min(s);
    }
    black_box(checksum);
    let flops = iters as f64 * ACCS as f64 * 16.0;
    (best, flops / best / 1e9)
}

fn main() {
    #[cfg(not(target_arch = "x86_64"))]
    eprintln!("peak: x86_64 only");

    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx512f") {
            eprintln!("peak: no avx512f");
            return;
        }
        const ITERS: usize = 2_000_000;
        // Clock ramp: a short burst after the idle gate reports the ramp, not
        // the core. Warm every arm the same way before it is timed.
        let deadline = Instant::now() + std::time::Duration::from_millis(1500);
        while Instant::now() < deadline {
            let _ = black_box(unsafe { chains::<4>(ITERS / 8) });
        }
        println!("AVX-512 FMA ceiling, 8 f64 lanes, 2 flop per FMA");
        for accs in [1usize, 4, 6, 8, 12, 16, 24] {
            let (best, gflops) = match accs {
                1 => measure::<1>(ITERS / 4),
                4 => measure::<4>(ITERS / 4),
                6 => measure::<6>(ITERS / 4),
                8 => measure::<8>(ITERS / 4),
                12 => measure::<12>(ITERS / 4),
                16 => measure::<16>(ITERS / 4),
                _ => measure::<24>(ITERS / 8),
            };
            println!("accs={accs:2}  best={:8.3} ms  {gflops:7.1} GFLOP/s", best * 1e3);
        }
    }
}
