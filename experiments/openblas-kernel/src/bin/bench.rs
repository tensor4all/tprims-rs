//! A/B comparison: default packed tprims vs the external OpenBLAS family, on
//! identical representative f64 cases, at `--threads 1,4,8,12`, reported as CSV.
//!
//! Provided for measurement but must be run under the `tprims-benchmark` skill
//! protocol (pinned idle cores in one L3 domain, paired thread counts, an A/A
//! noise floor). It performs no pinning itself.
//!
//! Usage: `bench [--threads 1,4,8,12] [--reps N]`

use std::hint::black_box;
use std::time::Instant;

use openblas_kernel::{catalog, OPENBLAS_FAMILY, OPENBLAS_WIDE_FAMILY};
use tprims_bench::threads::BenchThreads;
use tprims_contract::api::{DType, DotGeneral, LayoutSpec, OperandSpec, Problem};
use tprims_contract::{Chooser, Plan, PlanConfig};
use tprims_exec::Exec;
use tprims_kernel::SelectError;

fn layout(dims: &[usize], strides: &[isize]) -> OperandSpec {
    OperandSpec::new(LayoutSpec::new(dims, strides, 0).unwrap())
}

struct Case {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
    a_s: [isize; 2],
    b_s: [isize; 2],
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "gemm-1024-col",
            m: 1024,
            n: 1024,
            k: 1024,
            a_s: [1, 1024],
            b_s: [1, 1024],
        },
        Case {
            name: "gemm-512-col",
            m: 512,
            n: 512,
            k: 512,
            a_s: [1, 512],
            b_s: [1, 512],
        },
        Case {
            name: "strided-1024-rowa",
            m: 1024,
            n: 1024,
            k: 1024,
            a_s: [1024, 1],
            b_s: [1, 1024],
        },
    ]
}

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

type Prepared = (Plan<f64>, Plan<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn build_plans(case: &Case, family_id: &str) -> Prepared {
    let dot = DotGeneral::new(&[1], &[0], &[], &[]);
    let problem = Problem::from_dot_general(
        DType::F64,
        layout(&[case.m, case.k], &case.a_s),
        layout(&[case.k, case.n], &case.b_s),
        layout(&[case.m, case.n], &[1, case.m as isize]),
        &dot,
    )
    .unwrap();

    let mut a = vec![0.0f64; case.m * case.k];
    let mut b = vec![0.0f64; case.k * case.n];
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for x in &mut a {
        *x = rng.next();
    }
    for x in &mut b {
        *x = rng.next();
    }

    let cfg = PlanConfig::packed();
    let dflt = Plan::<f64>::new(&problem, &cfg).unwrap();

    let cat = catalog();
    let mut selector = |_: &tprims_contract::SelectionContext<'_>,
                        cands: &[tprims_contract::KernelCandidate<f64>]|
     -> Result<tprims_kernel::KernelHandle<f64>, SelectError> {
        cands
            .iter()
            .map(|c| c.handle)
            .find(|h| h.id() == family_id)
            .ok_or(SelectError::NoCandidates { dtype: "f64" })
    };
    let openblas = Plan::<f64>::new_with_selector(
        &problem,
        &cfg,
        &cat,
        &mut selector as &mut Chooser<'_, f64>,
    )
    .unwrap();

    (dflt, openblas, a, b, vec![0.0f64; case.m * case.n])
}

/// Minimum wall time over `reps` for each backend, reusing one output buffer.
fn measure(
    dflt: &Plan<f64>,
    openblas: &Plan<f64>,
    a: &[f64],
    b: &[f64],
    d: &mut [f64],
    exec: &Exec<'_>,
    reps: usize,
) -> (f64, f64) {
    // One untimed warm-up per arm; planning and pool construction are outside.
    for plan in [dflt, openblas] {
        plan.execute_slices(exec, 1.0, (a, 0), (b, 0), (d, 0))
            .unwrap();
        black_box(&*d);
    }
    let mut best_dflt = f64::INFINITY;
    let mut best_ob = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        dflt.execute_slices(exec, 1.0, (a, 0), (b, 0), (d, 0))
            .unwrap();
        best_dflt = best_dflt.min(t0.elapsed().as_secs_f64());
        black_box(&*d);

        let t0 = Instant::now();
        openblas
            .execute_slices(exec, 1.0, (a, 0), (b, 0), (d, 0))
            .unwrap();
        best_ob = best_ob.min(t0.elapsed().as_secs_f64());
        black_box(&*d);
    }
    (best_dflt, best_ob)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads = BenchThreads::from_args();
    threads.verify();
    let reps = args
        .iter()
        .position(|a| a == "--reps")
        .map_or(5, |i| args[i + 1].parse::<usize>().unwrap());
    assert!(reps > 0);
    let verify_only = args.iter().any(|a| a == "--verify");

    let family_id = if args.iter().any(|a| a == "--wide") {
        OPENBLAS_WIDE_FAMILY.id
    } else {
        OPENBLAS_FAMILY.id
    };
    println!("case,threads,backend,best_s,gflops");
    for case in cases() {
        let (dflt, openblas, a, b, mut d) = build_plans(&case, family_id);
        eprintln!(
            "PLAN {} default={:?} openblas={:?}",
            case.name,
            dflt.report(),
            openblas.report()
        );
        let flops = 2.0 * case.m as f64 * case.n as f64 * case.k as f64;

        threads.with_exec(|exec, _| {
            // Verify at the actual budget, not just serial. FMA/blocking may
            // change rounding, so exact equality is not the numerical gate.
            let mut want = d.clone();
            dflt.execute_slices(exec, 1.0, (&a, 0), (&b, 0), (&mut want, 0))
                .unwrap();
            openblas
                .execute_slices(exec, 1.0, (&a, 0), (&b, 0), (&mut d, 0))
                .unwrap();
            let num: f64 = d.iter().zip(&want).map(|(x, y)| (x - y).powi(2)).sum();
            let den: f64 = want.iter().map(|x| x * x).sum();
            let residual = (num / den).sqrt();
            assert!(
                residual.is_finite() && residual <= 1e-10,
                "{}: residual={residual}",
                case.name
            );
            eprintln!(
                "CHECK {} threads={} rel_frob={residual:.3e}",
                case.name, threads.requested
            );
            if !verify_only {
                let (old, new) = measure(&dflt, &openblas, &a, &b, &mut d, exec, reps);
                for (name, seconds) in [("tprims", old), ("openblas", new)] {
                    println!(
                        "{},{},{},{:.9},{:.3}",
                        case.name,
                        threads.requested,
                        name,
                        seconds,
                        flops / seconds / 1e9
                    );
                }
            }
        });
    }
}
