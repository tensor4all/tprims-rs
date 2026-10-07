//! Pinned microbenchmark: intrinsic `tramp_real::<3,8>` vs the frozen-assembly
//! `frozen_real_24x8`, for `kc` in `{16, 64, 256}`, single-threaded, as CSV.
//!
//! Performs no pinning itself; run it under the project's protocol:
//!
//! ```sh
//! benchmarks/scripts/pinned.sh 4 -- cargo run --release --bin bench
//! ```
//!
//! (CPU 4 for 1T on this host; never 0-3. `pinned.sh` checks the core is idle
//! before and after and discards spoiled runs.) Each measurement is a warm-up
//! batch followed by `--reps` timed batches; `best_ns` is the minimum batch
//! time, `mean_ns` the arithmetic mean.

use std::hint::black_box;
use std::time::Instant;

use avx512_ukr_asm::{frozen_ukr, intrinsic_ukr, reference_real, Ukr, MR, NR};

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

fn rel_frob(got: &[f64], want: &[f64]) -> f64 {
    let num = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w) * (g - w))
        .sum::<f64>()
        .sqrt();
    let den = want.iter().map(|w| w * w).sum::<f64>().sqrt();
    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Wall-clock time per call (ns) for one timed batch of `iters` calls.
fn batch_ns(ukr: Ukr, kc: usize, a: &[f64], b: &[f64], ab: &mut [f64], iters: usize) -> f64 {
    let t0 = Instant::now();
    for _ in 0..iters {
        // SAFETY: exact sized panels and exclusive tile.
        unsafe { ukr(kc, a.as_ptr(), b.as_ptr(), ab.as_mut_ptr()) };
    }
    black_box(&*ab);
    t0.elapsed().as_secs_f64() / iters as f64 * 1e9
}

fn main() {
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("this host has no AVX-512F; cannot run the kernels");
        std::process::exit(2);
    }
    let args: Vec<String> = std::env::args().collect();
    let reps = args
        .iter()
        .position(|a| a == "--reps")
        .map_or(7, |i| args[i + 1].parse::<usize>().unwrap());
    assert!(reps > 0, "reps must be positive");

    let cases = [16usize, 64, 256];
    // Roughly constant wall time per batch (~87 ms) across kc.
    let iters_for = |kc: usize| (20_000_000 / kc).max(10_000);

    println!("# threads: requested=1 (pin with benchmarks/scripts/pinned.sh 4)");
    println!("kc,kernel,best_ns,mean_ns,gflops");

    for kc in cases {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let a: Vec<f64> = (0..MR * kc).map(|_| rng.next()).collect();
        let b: Vec<f64> = (0..NR * kc).map(|_| rng.next()).collect();
        let mut ab = vec![0.0f64; MR * NR];
        let iters = iters_for(kc);
        let flops = 2.0 * kc as f64 * MR as f64 * NR as f64;

        let arms: [(&str, Ukr); 2] = [("intrinsic", intrinsic_ukr()), ("frozen", frozen_ukr())];

        // Correctness gate at the measured kc before timing.
        let want = reference_real(kc, &a, &b);
        for (name, ukr) in &arms {
            // SAFETY: exact sized panels and exclusive tile.
            unsafe { (*ukr)(kc, a.as_ptr(), b.as_ptr(), ab.as_mut_ptr()) };
            let r = rel_frob(&ab, &want);
            assert!(
                r.is_finite() && r <= 1e-12,
                "{name} kc={kc}: rel_frob={r}"
            );
        }
        eprintln!(
            "CHECK kc={kc} intrinsic_vs_frozen rel_frob={:.3e}",
            rel_frob(&ab, &want)
        );

        for (name, ukr) in &arms {
            // Warm-up: one full batch.
            let _ = batch_ns(*ukr, kc, &a, &b, &mut ab, iters);
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                samples.push(batch_ns(*ukr, kc, &a, &b, &mut ab, iters));
            }
            let best = samples.iter().cloned().fold(f64::INFINITY, f64::min);
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let gflops = flops / best; // flops / (best_ns * 1e-9) / 1e9
            println!("{kc},{name},{best:.3},{mean:.3},{gflops:.3}");
            eprintln!(
                "TIMED {name} kc={kc} iters={iters} reps={reps} best={best:.2}ns mean={mean:.2}ns {gflops:.2} GF/s"
            );
        }
    }
}
