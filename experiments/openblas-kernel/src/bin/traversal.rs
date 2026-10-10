//! Attribute the cost of an interleaved operand layout.
//!
//! One contraction, one kernel, one blocking, one thread budget: the only thing
//! that changes between the four rows is where the operands live in memory. The
//! dimensions match the campaign's worst losing TCCG case, `abjc-cbka-kj` f64 at
//! 16 MiB - effective GEMM m=92160, n=40, k=48 - and the "corpus" layouts are
//! that case's own plan record:
//!
//! * `A` 4 axes, extents 48x40x48x48, `m` axes at strides (92160, 48, 1) and the
//!   contracted axis at 1920 — so the fastest-varying `m` axis, the one the pack
//!   walk steps first, is 92160 elements away in `A`;
//! * `D` 4 axes, extents 48x40x48x40, `m` axes at (1, 48, 76800) and `n` at 1920.
//!
//! The "compact" variants put the same dimension on a plain column-major matrix
//! (`A` = [m, k] at (1, 92160), `D` = [m, n] at (1, m)), which is what the
//! kernel A/B in `bench.rs` measures. Comparing the four rows says whether the
//! packed route's cost lives in reading `A`, in writing `D`, in both, or in
//! neither.
//!
//! Run it under the `tprims-benchmark` skill protocol (pinned idle cores, one
//! L3 domain, an A/A noise floor); it does no pinning itself.
//!
//! Usage: `traversal [--threads N] [--reps N] [--prime-ms N]`

use std::hint::black_box;
use std::time::Instant;

use tprims_bench::threads::BenchThreads;
use tprims_contract::api::{CSpec, DType, Labels, LayoutSpec, OperandSpec, Problem};
use tprims_contract::{Plan, PlanConfig};
const LABELS: ([i64; 4], [i64; 2], [i64; 4]) = ([0, 1, 2, 3], [4, 3], [2, 1, 0, 4]);
const A_DIMS: [usize; 4] = [48, 40, 48, 48];
const D_DIMS: [usize; 4] = [48, 40, 48, 40];

/// `A`'s strides: corpus order (m axes strided) or a compact [m, k] matrix.
const A_CORPUS: [isize; 4] = [92160, 48, 1, 1920];
const A_COMPACT: [isize; 4] = [1, 48, 1920, 92160];
/// `D`'s strides: corpus order or a compact [m, n] matrix.
const D_CORPUS: [isize; 4] = [1, 48, 76800, 1920];
const D_COMPACT: [isize; 4] = [1, 48, 1920, 92160];

fn spec(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

fn span(dims: &[usize], strides: &[isize]) -> usize {
    dims.iter()
        .zip(strides)
        .fold(1usize, |n, (&e, &s)| n + (e - 1) * s.unsigned_abs())
}

fn problem(a: &[isize; 4], d: &[isize; 4]) -> Problem {
    let (ia, ib, id) = LABELS;
    Problem::from_labels(
        DType::F64,
        spec(&A_DIMS, a),
        spec(&[40, 48], &[48, 1]),
        CSpec::Absent,
        spec(&D_DIMS, d),
        &Labels::new(&ia, &ib, &id),
    )
    .unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads = BenchThreads::from_args();
    threads.verify();
    let wanted = tprims_bench::threads::parse_threads(&args).unwrap_or(1);
    let reps = args
        .iter()
        .position(|a| a == "--reps")
        .map_or(5, |i| args[i + 1].parse::<usize>().unwrap());
    let prime_ms = args
        .iter()
        .position(|a| a == "--prime-ms")
        .map_or(1500, |i| args[i + 1].parse::<u64>().unwrap());

    let a_len = span(&A_DIMS, &A_CORPUS).max(span(&A_DIMS, &A_COMPACT));
    let d_len = span(&D_DIMS, &D_CORPUS).max(span(&D_DIMS, &D_COMPACT));
    let mut a = vec![0.0f64; a_len];
    let mut b = vec![0.0f64; 40 * 48];
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng as f64 / u64::MAX as f64) * 2.0 - 1.0
    };
    for x in a.iter_mut().chain(b.iter_mut()) {
        *x = next();
    }
    let flops = 2.0 * 92160.0 * 40.0 * 48.0;

    let variants: [(&str, [isize; 4], [isize; 4]); 4] = [
        ("corpus-a-corpus-d", A_CORPUS, D_CORPUS),
        ("corpus-a-compact-d", A_CORPUS, D_COMPACT),
        ("compact-a-corpus-d", A_COMPACT, D_CORPUS),
        ("compact-a-compact-d", A_COMPACT, D_COMPACT),
    ];

    println!("variant,threads,best_s,gflops,residual");
    let mut reference: Option<Vec<f64>> = None;
    for (name, a_s, d_s) in variants {
        let p = problem(&a_s, &d_s);
        let plan = Plan::<f64>::new(&p, &PlanConfig::packed()).unwrap();
        let mut d = vec![0.0f64; d_len];
        eprintln!("PLAN {name} {:?}", plan.report());
        let (secs, residual) = threads.with_exec(|exec, _| {
            let until = Instant::now() + std::time::Duration::from_millis(prime_ms);
            while Instant::now() < until {
                plan.execute_slices(exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0))
                    .unwrap();
            }
            let mut best = f64::INFINITY;
            for _ in 0..reps.max(1) {
                let t = Instant::now();
                plan.execute_slices(exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0))
                    .unwrap();
                best = best.min(t.elapsed().as_secs_f64());
                black_box(&d);
            }
            // Every variant computes the same contraction, so the compact pair is
            // the oracle for the interleaved ones: a layout cannot change the
            // answer, and a nonzero residual here means it did.
            let residual = match &reference {
                None => {
                    reference = Some(d.clone());
                    0.0
                }
                Some(want) => {
                    let num: f64 = d.iter().zip(want).map(|(x, y)| (x - y).powi(2)).sum();
                    let den: f64 = want.iter().map(|x| x * x).sum();
                    (num / den).sqrt()
                }
            };
            (best, residual)
        });
        println!(
            "{name},{},{secs:.9},{:.4},{residual:.2e}",
            wanted,
            flops / secs / 1e9
        );
    }
}
