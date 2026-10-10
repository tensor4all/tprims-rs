# Matching the reference on the transposed-output contractions

Status: consolidated design, four pre-reviews taken into account. Nothing here is
implemented. Every claim marked *measured* was taken on this host with the pinned
binary and the opt-in phase instrument (`--features phase-timing`,
`TPRIMS_PHASE=1`); everything else is a design statement and is labelled as one.

## 1. The problem, measured

`abjc-cbka-kj` f64 16 MiB 1T: 43.1 ms per call, of which `pack_a` 27.6 ms (64%),
`kernel` 14.3, output stores 1.8. TBLIS, through its own packm/gemm kernels
registered into BLIS, does 24.8 ms. Layouts from the harness's own `la`/`lc`:
`A` is `(a s1, b s48, j s1920, c s92160)`; the output is `(c s1, b s48, n s1920,
a s76800)`. So in the m group the operand and the output are **transposed**: `A`
is `a`-fastest, the output is `c`-fastest.

Measured alternatives, all dead ends:

| what was tried | result |
| --- | --- |
| m ordered by the **input** stride | `pack_a` 27.6 -> 4.5 ms, but output stores 1.8 -> 30.2, total 54.2 |
| order the pack's visit within the tile (row sort) | identity permutation on this case; nothing changes |
| one-step prefetch of the next k/sliver | 29.4 against 28.1 ms |
| rotate the lane start per k step | `pack_a` 45.6 against 27.6 |
| blocking model, MC/KC/NC, orientation, partition, `align_c_lines`, MR sweep | no effect (mr=8: +5%) |
| the same case with a one-element-padded leading dimension | 23.3 ms, `pack_a` 9.7 - the phase, not the kernel |

An offline address model (per k-step, per panel, resolved MR=24/KC=48) counts 1.0
useful element per touched 64-byte line today against the 8.0 a contiguous read
allows, and 4.42M pages against 98k for an `48a x 8c` block. The measured
`--stress padded` column shows what recovering that is worth.

## 2. What the four pre-reviews established

1. **A pack-side change cannot fix it.** With output-ordered rows the pack reads
   one element per line; with input-ordered rows the output stores do. The panel
   layout (`out + (s*kc + p)*vr*r + t*r`) is consistent with the packer writing
   the row order it reads, so source-ordered packing is fine for every
   `PackFormat` and needs **no** `PackFn` change. The transposition therefore has
   to happen downstream, in the output path.
2. **A per-tile permutation cannot fix it.** A tile is 24 rows of one axis run;
   the earlier drafts' intra-tile row sort degenerates to the identity. The same
   argument kills a per-`MC`-block permutation too: for this case a source-linear
   `MC`-row block (input order `a,b,c` with extents 48,40,48) contains `c` only
   once per 1920 rows, so it cannot contain a destination-contiguous `c` run.
3. **Membership, not order, is the lever.** A block must span enough of **both**
   axes to hold a contiguous source run *and* a contiguous destination run: e.g.
   all 48 `c` for a few `(a,b)` pairs. That is a *multi-dimensional* (axis-aware)
   block, which is what the reference's outer blocking does and what this
   implementation does not have today: `static_grid.rs` walks consecutive slices
   of one flattened row scatter.
4. **Cross-K persistence forces an ownership design.** Today the static traversal
   is `jc -> pc -> all ic blocks`, each slab producing an overwrite or accumulate
   tile, with barriers placed in identical `(h,jc,pc)` loops. Accumulating a whole
   block before emitting means either keeping every owned row block across K
   (roughly `owned_rows x owned_columns`), or moving K inside a block round, or
   restricting eligibility (e.g. a single K slab). Dynamic scheduling compounds it:
   the claim counter resets each epoch, so the same job can move between workers
   between slabs.
5. **Workspace and admission are not free.** The arena rounds to powers of two, so
   an 83 KiB block costs 128 KiB of retained capacity; an out-of-place permutation
   needs a second region; the traversal of 349 blocks costs about 58 MB of extra
   traffic, not 29 MB; and the existing seam test proves *route* refusal, not a
   capacity refusal, which does not exist yet.
6. **Numerical claims must stay within the rules.** The project requires exact
   serial/static/dynamic/width equivalence for a fixed arithmetic configuration;
   the corpus and harness comparisons are tolerance-based (1e-4..1e-12) with an
   exact known-value check. Delaying `alpha`, `beta`, complex recombination and
   conjugation to a final emission changes rounding and signed-zero behaviour, so
   it must be defined, not assumed away.
7. **No promotion on a regression.** Input-priority ordering alone is 26% slower
   end to end; it can only be an experimental switch until the full path wins.

## 3. The design that follows

**Axis-aware outer blocks, with the transpose moved to a block-level output
scratch.**

- **Membership.** A block is `{ per-axis ranges over the oriented m role's axes,
  per-axis ranges over the n role's axes, batch, K slab }`, chosen so that it
  contains whole runs of the packed operand's fastest axis **and** the output's
  fastest axis - for the measured case, all 48 `c` and 10 `(a,b)` pairs
  (`MC = 480` rows, an `a`-run of 10 elements per `(b,c)`, a `c`-run of 48). The
  offline model is used to pick the shape, and eligibility is a conservative
  predicate with a fallback to today's path.
- **Panel rows** stay in the **output** order, so the kernel, the emitter and the
  write-back contract are untouched.
- **Packing** reads the operand in the operand's order and writes the panel in the
  panel's order; the existing scatter and block-scatter machinery does this
  unchanged (the pack side is already efficient once the row order is settled),
  which is why no pack descriptor and no ABI change are needed.
- **Output.** The kernel accumulates the block's K slabs into an `MC x NC`
  scratch; a block-level transpose reorders that scratch into the output's order,
  and the existing emitter writes `D` contiguously - the same work it does today,
  fed a block instead of a tile. Workspace: one extra block region, checked and
  retained at the next power of two.
- **Ownership.** One of the three options from §2.4 must be chosen and written
  down; my preference is the narrowest that can win: a block round with K inside
  it, so a worker keeps one scratch per owned block, with the barriers restated on
  common rounds including idle participants, and direct-B decided for the whole
  call before any allocation (with the new N path disabling it on doubt).

## 4. Gates

Correctness first: the local gate; then old/new equivalence established by an
independent oracle (tolerance) with the partition-bitwise property and
width/serial equality retained exactly for a fixed configuration; all formats,
dtypes, conjugations, separate/in-place C, signed scatters, per-axis tails,
multiple K slabs, batches, and every route. Mechanism gates: destination run
lengths after the block permutation, exactly-once accumulation across K and across
dynamic job migration, unequal static row strips with empty column groups,
fresh-output/no-read-before-write, beta-zero C, and full checked scratch
accounting. Non-regression: the 49-case TCCG suite at 1T/4T, both dtypes, `plan`
and forced `packed`, none/padded/ragged, plus the per-shape corpus. Attribution
with disjoint phase scopes and an uninstrumented headline, thresholds declared
beforehand. Then both campaign cells are re-recorded.

## 5. Order of work

1. The offline geometry model, to fix the block shape and the expected runs, and
   to reject shapes whose destination runs are not contiguous (this is what the
   last four reviews asked for before any code).
2. The axis-aware block enumerator and block-local scatters, behind an
   eligibility switch, with the existing suites green.
3. The block scratch and the block-level transpose with postponed emission, one
   K slab first (the narrowest eligibility), then the ownership round.
4. The gates, then the campaign re-measurement.

## 6. What this costs, honestly

Steps 2-4 touch the driver's enumerator, both schedulers, ownership and barriers,
capacity and admission, the emitter interface and the reporting fields. That is an
architectural change of the kind the project writes a design for, not a patch, and
it will span more than one session. The cheap alternative that is already measured
is the caller-side layout: the campaign's TCCG sizing rounds stride-1 extents to
multiples of 24, which makes this case's source stride exactly 180 pages; the
one-element-padded variant of the same corpus is 23.3 ms against the reference's
20.0 ms, with no library change at all.
