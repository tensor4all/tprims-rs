## Summary

The `tprims-contract` planner sends every problem that fuses to one strided batched GEMM without copying to **faer** (rule 3 in `plan/mod.rs`). In the #61 benchmark this is right for small and medium problems, but wrong for large c64 ones: at MPS chi=64 the planner picks faer and runs 1.17 times slower than the packed driver it already has. Add a size (and dtype) condition to rule 3, decided from a crossover sweep, so large problems take the packed driver.

This is gate 3 of the cpueinsum plan (#62).

## Evidence

Source: `experiments/three-engine-contract` on branch `issue-61-three-engine` (commits 16bf77ce, 696331cc, 0480107e, not yet pushed). Apple M5 Max, macOS 26.5.1, rustc 1.96.0, release with thin LTO and `codegen-units = 1`, **1 thread**, column-major operands, timed boundary = execution with a prebuilt plan and a preallocated output. Values are ratios to tensorcontract's prebuilt execution (`tc_exec`), session median of per-call medians over 3 sessions.

`--report` shows `algorithm=faer` for **every** step of every case in the corpus.

| case | dtype | `tp_exec` (planner = faer) | `tp_packed_exec` (forced packed) | winner |
| --- | --- | --- | --- | --- |
| MPS chain L=32, chi=4 | c64 | 0.55 | 1.01 | faer |
| MPS chain L=32, chi=8 | c64 | 0.53 | 0.96 | faer |
| MPS chain L=32, chi=16 | c64 | 0.81 | 0.95 | faer |
| MPS chain L=32, chi=32 | c64 | 1.02 | 0.99 | tie (within noise) |
| **MPS chain L=32, chi=64** | **c64** | **1.17** | **0.99** | **packed** |
| `ij,jk->ik`, n=32 | c64 | 0.87 | 0.96 | faer |
| `ij,jk->ik`, n=64 | f64 | 0.80 | 1.00 | faer |
| `ij,jk->ik`, `ik,kl->il`, n=64 | f64 | 0.79 | 1.04 | faer |
| `ikb,knb->inb`, n=16, b=16..256 | f64 | 0.77..0.80 | 0.94 | faer |

MPS steps are `ab,asc->bsc` (env chi x chi, site chi x 2 x chi) and `bsc,bsd->cd`, so chi=64 means GEMMs of roughly 64 x 128 x 64 and 64 x 64 x 128 in c64. The c64 crossover lies between chi=32 and chi=64. No f64 case in the corpus is large enough to show whether f64 has a crossover too.

Limits of this evidence: one machine, 1T only, whole-chain timing (no per-step split), and no f32 / c32 cases.

## Proposal

Order of work: implement first, measure last. The sweep runs once at the end, together with the cpueinsum gates (#62), not before the code exists.

1. **Rule change (implementation):** rule 3 picks faer only below a per-dtype threshold on the fused GEMM size (for example on `m * n * k` per batch item, or on the smallest of m, n, k). The threshold lives in `PlanConfig` so it can be overridden, and starts from a provisional value taken from the evidence above (c64: faer below chi=32 class sizes, packed above; other dtypes: unchanged until measured).
2. **Report:** `PlanReport` states the reason for the choice (below or above the threshold), so `--report` style probes can check it.
3. **Crossover sweep (at the end)**, as a `tprims-bench` row set per `PERFORMANCE_TIPS.md`:
   - dtypes f32, f64, c32, c64;
   - shape classes: plain GEMM, batched GEMM (`ikb,knb->inb`), and the two MPS step shapes;
   - sizes spanning both sides of the crossover (for c64 at least chi = 16, 24, 32, 48, 64, 96, 128);
   - 1T and 4T, on the M5 Max and on the shared EPYC workstation;
   - arms: planner choice and forced packed, both with a prebuilt plan.
   The sweep fixes the predicate and the per-dtype values.
4. Keep the threshold provisional and recorded with its measurement, as `AGENTS.md` requires for kernel and executor choices.

## Acceptance

- [ ] Sweep results recorded (CPU, rustc, flags, dtype, shapes, thread count, timed boundary) for both machines.
- [ ] MPS chi=64 c64 selects packed; chi <= 16 c64 and the small f64 cases keep faer.
- [ ] No case in the #61 corpus gets slower than the better of its two arms by more than the A/A noise floor.
- [ ] Threshold and evidence written into `docs/decision-log.md`.

## Open questions

- Whether the c64 loss comes from faer's complex GEMM path rather than from size alone (if so, the predicate may be dtype first, size second). The sweep over c32 / c64 vs f32 / f64 should tell.
- Whether the crossover moves with thread count; the rule may need the executor width, which the plan already receives at execution time.
