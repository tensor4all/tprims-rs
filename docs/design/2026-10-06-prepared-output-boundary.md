# Prepared output storage boundary

Prerequisite for [cpueinsum-rs#4](https://github.com/tensor4all/cpueinsum-rs/issues/4)
and subsequently tenferro-rs#2004. The complete cross-repository design and
independent pre-review live in cpueinsum's
`docs/design/2026-10-06-prepared-binary-and-grouped.md`.

- Add safe slice accumulation and `MaybeUninit` output execution to the existing
  `Plan`. C mode, beta, source metadata and buffer spans are checked before writes.
  Absent/nonzero (including NaN) beta is rejected consistently in view, slice,
  batch and raw execution; beta zero never reads C/D values. Packed execution
  also uses D's origin/batch offsets for unused C at beta zero: forming an
  out-of-allocation `offset` through a missing C is invalid even without a read.
- A fresh-output return requires existing reduced-output injectivity, checked
  reduced M/N/H cardinality equal to the slice length, and exact shifted span
  `[0,len)`. Original D dimension products are not a coverage proof for repeated
  labels. Padding/diagonal holes are unsupported, not zero-filled. Failure and
  unwind never return an initialized slice; numerical failure may partially write.
- Elementwise overwrite/Separate leaves use existing generic strided operations
  with a `MaybeUninit` destination. In-place leaves require live values. Packed
  scratch/direct leaves must write before reading or forming initialized references;
  their unsafe kernel ABI makes this obligation explicit.
- Faer 0.24.4 row-major matvec forms destination references even at Replace.
  `PlanConfig::fresh_output` therefore prepares a packed alternative when needed,
  defaulting off to preserve the ordinary Faer plan's <=2 allocation gate. Fresh
  Faer execution without preparation returns typed Unsupported before even no-op
  shortcuts. Initialized execution keeps its original strategy. No zero-fill,
  result temporary, provider probe, execution-time planning or serial retry is used.
- `execution_route` and the driver share one unchanged packed partition-policy
  helper. Reports distinguish Empty, OutputOnly, Elementwise, Faer and exact packed
  kind/width; Faer/elementwise active widths are not invented. A query is not a
  lease, reservation or worker-ID promise. Same-pool worker SPMD refusal remains
  before writes, and concurrent external SPMD calls retain the blocking gate.
- Raw Faer execution without the capability requires initialized D, including
  at beta zero. A prepared raw plan selects the fresh route for write-only D:
  zero beta or a disjoint Separate C. Nonzero beta reading D needs live D.
  The C plan requests preparation because its contract permits write-only D.
  Consequently its ordinary default fresh GEMM may now enter through a packed
  SPMD broadcast, and a same-pool worker may receive `TAPP_ERROR_UNSUPPORTED`
  rather than an implicit serial retry. Pool-entry observations count both ordinary
  entries and broadcasts. This is an explicit safety/route change, not a speed claim.

Plans own immutable metadata only. Execution lends caller-owned pools/arenas;
this addition does not change workspace ownership or introduce an ambient pool.
