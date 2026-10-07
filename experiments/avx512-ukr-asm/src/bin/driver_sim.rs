//! Where does the packed driver's ~45% go?
//!
//! Facts established elsewhere in this experiment: the micro-kernel runs at
//! ~52.7 GFLOP/s per core at 1T and 52.6 per core at 8T (no throttling), and
//! its rate barely moves when its panels come from DRAM instead of L1
//! (`stream.rs`: 52.7 -> 49.6 GFLOP/s at a 32 MB working set). Yet a real
//! 1024^3 GEMM reaches only ~230 GFLOP/s at 8T, i.e. ~55% of the kernel-only
//! aggregate. So the loss is *instruction work around the kernel*, not memory.
//!
//! This binary reproduces the packed driver's loop nest for a rectangular
//! GEMM -- pack a B panel per (jc,pc), pack A rows per (jc,pc,ic), call the
//! micro-kernel per tile, write the tile back -- with no library involved, and
//! reports GFLOP/s for ablations that switch each surrounding stage off. The
//! differences are the per-stage cost.
//!
//! It is a model, not the library: no masking, no complex methods, no direct-B,
//! no dynamic scheduling. It compares stages of *this* model to each other.
//!
//!     benchmarks/scripts/pinned.sh 4-11 -- ./target/release/driver_sim 8

use std::hint::black_box;
use std::time::Instant;

use avx512_ukr_asm::{intrinsic_ukr, Ukr, MR, NR};

const MC: usize = 264; // 11 * MR
const KC: usize = 256;
const NC: usize = 1536;
const M: usize = 2112; // 8 * MC
const N: usize = 1536; // 1 * NC
const K: usize = 1024; // 4 * KC

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

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Full,
    NoWriteback,
    NoPackA,
    NoPackB,
    KernelOnly,
}

/// One thread's row strip `[m_lo, m_hi)` of the whole nest.
fn strip(
    ukr: Ukr,
    mode: Mode,
    a: &[f64],
    b: &[f64],
    d: &mut [f64],
    m_lo: usize,
    m_hi: usize,
    scratch: &mut Scratch,
) {
    for jc in (0..N).step_by(NC) {
        let jc_len = NC.min(N - jc);
        for pc in (0..K).step_by(KC) {
            let pc_len = KC.min(K - pc);
            if mode != Mode::NoPackB && mode != Mode::KernelOnly {
                for p in 0..pc_len {
                    for j in 0..jc_len {
                        scratch.bp[p * jc_len + j] = b[(pc + p) + (jc + j) * K];
                    }
                }
            }
            let mut ic = m_lo;
            while ic < m_hi {
                let ic_len = MC.min(m_hi - ic);
                if mode != Mode::NoPackA && mode != Mode::KernelOnly {
                    for p in 0..pc_len {
                        for i in 0..ic_len {
                            scratch.ap[p * ic_len + i] = a[(ic + i) + (pc + p) * M];
                        }
                    }
                }
                for mi in (0..ic_len).step_by(MR) {
                    for nj in (0..jc_len).step_by(NR) {
                        let tile = &mut scratch.tile;
                        // SAFETY: panels sized for pc_len rows, tile MR x NR.
                        unsafe {
                            ukr(
                                pc_len,
                                scratch.ap.as_ptr().add(mi * pc_len),
                                scratch.bp.as_ptr().add(nj * pc_len),
                                tile.as_mut_ptr(),
                            )
                        };
                        if mode != Mode::NoWriteback && mode != Mode::KernelOnly {
                            // Accumulating write-back: reads and writes D, as the
                            // driver does when it is not the first k slab.
                            for j in 0..NR {
                                for i in 0..MR {
                                    let at = (ic + mi + i) + (jc + nj + j) * M;
                                    d[at] += scratch.tile[j * MR + i];
                                }
                            }
                        }
                    }
                }
                ic += MC;
            }
        }
    }
}

struct Scratch {
    ap: Vec<f64>,
    bp: Vec<f64>,
    tile: Vec<f64>,
}

fn main() {
    let threads: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let ukr = intrinsic_ukr();
    let a = fill(M * K, 0x2545_F491_4F6C_DD1D);
    let b = fill(K * N, 0x9E37_79B9_7F4A_7C15);
    let flops = 2.0 * M as f64 * N as f64 * K as f64;
    let strip_rows = M / threads;

    println!("threads={threads} m={M} n={N} k={K} mc={MC} kc={KC} nc={NC}");
    println!("mode,ms,gflops,vs_kernel_only");
    let mut baseline = 0.0f64;
    for mode in [
        Mode::KernelOnly,
        Mode::NoWriteback,
        Mode::NoPackB,
        Mode::NoPackA,
        Mode::Full,
    ] {
        let measure = || {
            let mut d = vec![0.0f64; M * N];
            // The closure captures the address, not the buffer, so the
            // disjoint-row writes below are the only aliasing rule in play.
            let base = d.as_mut_ptr() as usize;
            let mut run = || {
                std::thread::scope(|s| {
                    for t in 0..threads {
                        let (a, b) = (&a, &b);
                        let lo = t * strip_rows;
                        let hi = lo + strip_rows;
                        s.spawn(move || {
                            // SAFETY: the strips partition the rows of D, so the
                            // range each thread writes is touched by no other
                            // thread; the length is the one buffer's.
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(base as *mut f64, M * N)
                            };
                            let mut sc = Scratch {
                                ap: vec![0.0; MC * KC],
                                bp: vec![0.0; NC * KC],
                                tile: vec![0.0; MR * NR],
                            };
                            strip(ukr, mode, a, b, dst, lo, hi, &mut sc);
                        });
                    }
                });
            };
            // Time-based warm-up, not a fixed call count: a short burst after
            // the idle gate measures the clock ramp, and the modes differ by
            // enough in duration that a per-mode constant would bias the
            // comparison. See the worklog note.
            let deadline = Instant::now() + std::time::Duration::from_millis(1500);
            while Instant::now() < deadline {
                run();
            }
            let t = Instant::now();
            for _ in 0..3 {
                run();
            }
            black_box(&d[0]);
            t.elapsed().as_secs_f64() / 3.0
        };
        let secs = measure();
        let g = flops / secs / 1e9;
        if mode == Mode::KernelOnly {
            baseline = g;
        }
        println!(
            "{:?},{:.3},{g:.1},{:.3}",
            mode,
            secs * 1e3,
            g / baseline
        );
    }
}
