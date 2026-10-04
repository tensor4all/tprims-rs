## Summary

Create a new repository, **tensor4all/cpueinsum-rs**, and move tenferro-rs's non-BLAS (Rayon) CPU einsum/contraction path into it. cpueinsum sits **above** tprims: it depends on `tprims-contract` (and through it `tprims-kernel` and `tprims-exec`) and adds only what tprims does not have. Nothing is moved out of tprims.

The name says the scope: CPU only. GPU stays in tenferro.

## Motivation

The three-engine benchmark of #61 (branch `issue-61-three-engine`, `experiments/three-engine-contract`, Apple M5 Max, 1 thread) shows that tenferro's CPU contraction is not slow in its kernel but in its call path. Per step of an MPS chain at chi=4:

| layer | µs |
| --- | --- |
| kernel (prebuilt plan, preallocated output) | 0.46 |
| + output allocation and execute wrapper | +0.89 |
| + prepare (plan construction) | +2.44 |
| + ordinary routing | +0.05 |
| + string parsing of the subscripts | +0.58 |
| **tenferro ordinary einsum, total** | **4.42** |
| tensorcontract, plan + alloc + exec every call | 1.31 |

`tp_call` (tprims, plan + alloc + exec every call) is already on par with tensorcontract (4.53 vs 4.03 times `tc_exec` at chi=4, 2.60 vs 2.68 at chi=8; tenferro is 14.14 and 7.70). The prepare cost in tenferro comes, by code reading (not profiled), from building a generic N-ary contraction tree with per-call `HashMap`/`HashSet` even for two operands, and from computing the binary dot plan a second time.

`tprims-contract` already is the thin binary engine we need: integer labels (`Labels`, `Vec<i64>`) and `DotGeneral` front ends, a validated `Problem`, a reusable `Plan`, and strategy selection (packed, faer, elementwise).

## Layering

```text
tenferro-einsum    subscript parsing, path optimization, AD, traced, GPU, BLAS path
  ├─ BLAS enabled  existing BLAS path (permute + GEMM), unchanged
  └─ BLAS disabled delegates to cpueinsum
cpueinsum          N-ary contraction with an explicit order, intermediate buffers, stable API, gates
tprims-contract    binary contraction: Problem, Plan, packed / faer / elementwise selection
tprims-kernel      GEMM microkernels, packing, blocking
tprims-exec        borrowed host pool, Exec
```

## Scope

In cpueinsum:

- Binary contraction as a direct path over `tprims_contract::Plan`: no N-ary tree for two operands.
- N-ary contraction where the **caller supplies the contraction order** (a sequence of pairs), executed as a chain of binary contractions with managed intermediate buffers.
- Integer labels plus layouts as the only input form.
- A stable public API that tenferro calls.
- The benchmark gates below, kept as regression checks.

Out of scope, by decision:

- **Path optimization.** The optimal order depends on the whole network and belongs to the caller (tenferro traced, omeco, or the user).
- **String subscript notation.** Parsing stays in tenferro; it cost 0.58 µs per step in the measurement above.
- **BLAS.** A general contraction needs a permute copy before BLAS, while the packed driver packs arbitrary strides straight into microkernel panels. Nothing at or below cpueinsum uses BLAS, so vendor-owned BLAS threading never enters this stack. tenferro-einsum keeps its BLAS path as is.
- **GPU, AD, traced integration.** These stay in tenferro.

## Design rules

- No per-call `HashMap` or `HashSet` on the call path.
- Process-wide choices are resolved once (as tensorcontract does behind `OnceLock`); no `getenv` per call.
- Threading follows the tprims-exec contract: the host lends its Rayon pool, width-one work never touches the pool.
- cpueinsum depends on a deliberately small tprims surface (`Problem`, `Plan`, `Exec` and their configuration), and that surface is kept stable.

## Order of work

Implement the whole path first (cpueinsum, the planner rule of #63, the tenferro-einsum switch), with correctness tests at each step. Run the gates below once at the end, on the finished stack, rather than benchmarking each piece as it lands.

## Gates

Measured against tenferro built **without** BLAS, recorded with CPU, rustc, build flags, dtype, shapes, thread count and timed boundary:

1. **Call path:** plan + alloc + exec per call at or below about 1.3 µs per step on the MPS chi=4 case (tensorcontract level).
2. **No regression of prebuilt execution:** `tf_exec` equivalent must not get slower.
3. **Large problems pick packed:** at c64 chi=64 faer was 1.17 times slower than the packed driver while the planner chose faer. This is a fix in the `tprims-contract` planner (threshold on the copy-free fusion rule); cpueinsum's gate is that it depends on a tprims version containing the fix and that the case selects packed.
4. Results at 1T and 4T, following `PERFORMANCE_TIPS.md`.

## Prerequisites and open questions

- [ ] **strided version conflict:** tenferro pins strided v0.4.4, tprims a post-v0.4.4 main commit. Resolve before tenferro can depend on cpueinsum.
- [ ] **Publishing:** a crates.io release of cpueinsum requires `tprims-contract`, `tprims-kernel` and `tprims-exec` on crates.io too (crates.io packages cannot depend on git). `AGENTS.md` currently says not to publish packages from this research repository; decide whether these three crates become publishable parts, or whether tenferro pins cpueinsum by git rev for a while.
- [ ] **Stable surface:** agree on which tprims types cpueinsum may use, so experiments in tprims do not break it.
- [ ] **planner threshold** in `tprims-contract` (gate 3): #63.
- [ ] **Switch in tenferro-einsum:** where the BLAS / cpueinsum branch lives, and how the tenferro binary dot plan and N-ary tree code is retired on the non-BLAS path.
- [ ] Provenance: tprims code stays in tprims with Lukas Devos's history; cpueinsum only depends on it.

## Not decided here

Repository creation itself, crate names inside the repository (`cpueinsum`, possibly `cpueinsum-core`; both free on crates.io as of 2026-10-05), and the tenferro-side migration issue.
