# Execution workspace/lease boundary and worker-context routing

Status: design + implementation for tensor4all/tprims-rs#70, which is the tprims-side
prerequisite for tensor4all/tenferro-rs#2004 (tenferro delegates CPU numerical execution
to cpueinsum/tprims, keeps only admission and resource ownership, and no longer installs
the whole session callback into the pool).

Scope: this document covers the two tprims changes only. cpueinsum's prepared binary
contract and grouped plan form are tensor4all/cpueinsum-rs#4. Tenferro's session
admission, affinity guard, and provider removal are #2004.

This revision follows a read-only pre-implementation review by `openai/gpt-6.1-sol`; the
disposition of every finding is in §5. The review was of an earlier revision that added a
`Plan::workspace_req()`, a `NoWorkspace` provider, an SPMD reservation, and a public
`Route` report. Those are **not** part of the design below.

## 1. Current behaviour

### 1.1 The plan owns mutable execution storage

`Plan<T>` owns an `ArenaProvider` (`crates/tprims-contract/src/plan/mod.rs:126`), and the
packed driver selects it as the fallback:

```rust
let workspace: &dyn WorkspaceProvider = exec.workspace().unwrap_or(&self.workspace);
```

(`plan/mod.rs:692`). `Exec::Serial` supplies no workspace (`crates/tprims-exec/src/exec.rs:104-109`),
so **the plan-owned arena is the normal 1-thread path**.

`WorkspaceProvider` is a `Sync` owner with three operations
(`crates/tprims-exec/src/workspace.rs:255-266`): `with_worker` (this thread's A block,
tile, and scatter scratch), `take_team` (the shared panel, the team scatter vectors and
the barriers, exclusive until the returned `TeamLease` drops), and `trim`.

This is an ownership and accounting problem for a host that caches plans:

- mutable execution storage is owned by an object the host treats as immutable cached
  metadata, so plan identity and resource identity are conflated;
- every caller thread that executes a cached plan can add a slot to the plan's arena, and
  there is no release path short of dropping the plan;
- the host cannot state or bound the retained bytes of a plan's arena, because they are
  not part of the plan's own description.

`Pool::borrow` creates a fresh `ArenaProvider` and a fresh SPMD mutex per call
(`crates/tprims-exec/src/pool.rs:56-69`), so a caller that constructs a wrapper per call
loses all reuse and, worse, gets a second coordination owner for the same `ThreadPool`
(one wrapper per pool is the documented rule, `exec.rs:182-194`).

### 1.2 Route selection happens during execution, and a declined SPMD silently serializes

The packed driver decides as it goes (`crates/tprims-contract/src/driver/mod.rs:539-567`,
`:779-790`, `:863-894`):

1. `lanes > 1` → barrier-free batch partition via `for_each_partition`;
2. `p == 1` → serial strip run;
3. otherwise → `broadcast(p, &cell)`, and **if that declines**, a serial strip run with
   no signal.

`batch::lanes` returns 1 unconditionally when the caller is a worker of the pool
(`driver/batch.rs:44-47`), even though barrier-free work from a same-pool worker is
valid: `Exec::install` runs in place on a same-pool worker (`exec.rs:128-142`) and
`for_each_partition` then spawns its lanes into that pool (`exec.rs:160-183`), where other
workers can steal them. So the worker gate is a policy choice, not a mechanism limit.

`broadcast` refuses when the caller is a worker, when the width exceeds the pool or the
budget, or when the context is serial (`exec.rs:225-238`), and it serializes concurrent
broadcasts on the pool with a blocking mutex (`exec.rs:243-249`).

The result is that a host which asked for a parallel route can get a fully serial one with
no error and no way to tell the two apart.

`pm == 1` is a separate case: the row strips are not cut, every column group has exactly
one thread, a group's slice of the packed `B` panel is written and read only by that
thread, and no barrier is taken (`driver/mod.rs:33-56`, and `req.barriers == 0` at
`driver/mod.rs:643-645`). A `pm == 1` call is therefore **barrier-free** even though the
driver currently still dispatches it through `broadcast`.

### 1.3 The worker-slot map does not protect slot contents

`ArenaProvider::slot` takes the slot map lock only to hand out a raw pointer
(`workspace.rs:368-377`), then `with_worker` mutates `slot.borrowed` and grows
`slot.a`/`slot.tile` **after releasing it** (`:379-402`). `retained_bytes` and `trim` read
and write those same fields while holding the lock (`:350-364`, `:440-449`), and a slot
being written by its owner is not excluded from either.

Concurrent `retained_bytes()` (a safe `&self` diagnostic) or `trim()` against a running
worker is therefore a data race on `cap`, `ptr` and `borrowed`, and `trim` can free a
buffer another thread is writing through. This is a pre-existing defect, not one this
change introduces; it becomes load-bearing once the provider and its accounting are part
of the host-facing contract, so it is fixed here.

## 2. Design

### 2.1 The execution resource carries the workspace; the plan owns none

- Delete the `workspace: ArenaProvider` field from `Plan` and the
  `unwrap_or(&self.workspace)` fallback. `Plan` becomes pure immutable metadata.
- Put the workspace on `Exec`, which already carries the pool for the parallel case:
  - `Exec::Serial` — no storage owner. A packed call allocates its scratch for the call
    and frees it (the driver's existing `workspace: None` path, `driver/mod.rs:735-746`).
  - `Exec::SerialWithWorkspace(&'a dyn WorkspaceProvider)` (new, `Exec` is
    `#[non_exhaustive]`) — serial execution with caller-owned reusable storage.
  - `Exec::Rayon { pool, budget }` — the pool's arena, unchanged.
  - `Exec::workspace(&self) -> Option<&'a dyn WorkspaceProvider>` returns the caller's
    provider for the serial variant and the pool's for the pooled one. `Exec`'s `Debug`
    is hand-written (a `&dyn WorkspaceProvider` is not `Debug`, so the derive would stop
    compiling); `Copy` and `Send` are unaffected. `with_budget` and every other `Serial`-branching
    method preserve the provider.
- `Plan::run` passes `exec.workspace()` to the driver. Ownership then has exactly one
  rule: **the party that wants reuse owns the provider**, and the plan never does.
- The C ABI's nonzero serial executor owns an `ArenaProvider` and runs with
  `Exec::serial_with_workspace`; the zero handle keeps call-local scratch
  (`crates/tprims-capi/src/executor.rs:143`, `:240`).
- `contract_batched` runs each item on "the outer context's workspace, serial" instead of
  a bare `Exec::Serial`, so a pooled batch does not allocate per item
  (`crates/tprims-contract/src/batch.rs:154`).

No `Plan::workspace_req()` is added. The storage one execution needs depends on the
resolved route (batch lanes, effective grid, active-width blocking, `beta` —
`driver/mod.rs:539-644`), not on the plan, so a plan-time accessor could only be a
misleading upper bound. No `NoWorkspace` provider is added either: `None` already means
"this call has no owner and allocates its own scratch", and an empty `ArenaProvider`
allocates nothing on a zero request (`workspace.rs:65-70`).

### 2.2 The arena is race-free: checkout, return, and exact accounting

- `with_worker` **removes** its boxed slot from the map under the lock and returns it when
  the callback ends (a guard re-inserts it on panic, and is armed before any growth so a
  preparation unwind cannot lose it). A re-entrant call on the same thread finds no slot
  and takes the existing fresh-buffer path.
- The rule is therefore: **a slot in the map is touched only under the map lock; a slot
  that is checked out is touched only by the thread that holds it.** `trim` and the stats
  snapshot see idle slots only, and nothing mutates a slot outside the map lock.
- `ArenaProvider::stats() -> WorkspaceStats { retained_bytes, leased_bytes }`, where
  `leased ⊆ retained`. Both are kept in one mutex-protected `Accounting { owned, leased }`
  so that the counters and the snapshot share one synchronization boundary: `owned` is the
  capacity the owner holds (worker A/tile/scatter, team panel/scatter/barriers, idle or in
  use) and `leased` is the part currently lent out. Capacity is measured with two helpers
  (`slot_bytes`, `set_bytes`) over the actual capacities, so a shrinking `Vec` or a
  geometry change that rebuilds the barrier vector is settled in both directions; a
  checkout bills its billed amount and its return settles `now - billed` (signed), so
  `leased` cannot go negative.
- Exactness claim: **`stats()` is exact whenever no execution is in flight** (both values
  are settled at checkout, return, `take_team`, lease drop and trim). During one, a
  checked-out slot is counted at the capacity it had when it was checked out, so
  `retained_bytes` is an unsettled estimate: while a call is in flight it holds
  the checked-out capacity, so it can lag the slot's growth or shrinkage until
  the call returns.
- Team-panel growth happens only through `TeamLease::panel`, and `Drop` subtracts exactly
  the amount the lease accounted. `TeamSet` stops being a public struct and `TeamLease`
  loses its `Deref`/`DerefMut`, so the bypass the review found (`lease.b.ensure(...)`,
  `workspace.rs:208-225`) is no longer expressible. `TeamLease` keeps `panel`, and gains
  narrow accessors for the scatter vector and the barriers; `PageBuf`, `TeamSet` and
  `PAGE` stop being re-exported from `tprims-exec`. Nothing is billed through `panel`
  alone: `take_team` bills the set's existing capacity, so a zero-byte request (a direct-B
  call reusing a panel-warmed set) is still accounted.
- A zero requirement allocates no A/tile/panel **payload**; the first call on a thread
  still creates its map entry, `Box` and thread-local handle, as it does today.
- Documented constraint: one map entry per caller thread that has ever used the provider,
  kept after `trim` (which releases the buffers, not the entries). A host that executes a
  cached plan from many short-lived threads pays a small per-thread metadata residue;
  bounding or reclaiming it needs a thread-exit hook and is not part of this change.

### 2.3 Route selection classifies the existing policy; it does not replace it

The geometry, lane and width policy is **not** rewritten. `batch::lanes`, `width_for`,
`partition_with`, the explicit-grid clamp, the batch balance rule and the blocking
selection stay exactly as they are. Two things change:

- the worker gate in `batch::lanes` (`driver/batch.rs:44-47`) is removed: a worker is a
  legitimate barrier-free caller;
- after the partition is final and **before any allocation or output write**, the driver
  classifies the route and dispatches it once:

| condition | route | dispatch |
|---|---|---|
| `lanes > 1` | barrier-free batch lanes | `for_each_partition(lanes, lane)` |
| `p == 1` | serial | inline strip run |
| `pm == 1` | barrier-free output cells | `for_each_partition(p, cell)` |
| `pm > 1`, caller is a worker of the pool | **typed error** (`ExecError::Unavailable`) | nothing runs, nothing is written |
| `pm > 1` | SPMD team | `broadcast(p, cell)` |

- The dispatch is the route: there is no "try the parallel one, then fall back" path. The
  removed fallback (`driver/mod.rs:863-894`) is gone. The worker check and the error
  propagation come before the lease, before any allocation and before
  `DynStats::note_call`, so an unavailable route changes no counter and writes nothing.
- This makes the private driver fallible: `execute_packed`/`execute_packed_instrumented`
  return `Result<(), ExecError>`, `Plan::run` propagates it with `?` (the
  `PreparedContraction` trait already documents `Error::Exec` "before any write",
  `api/backend.rs:141-146`), and the test adapters in `driver/tests/compat.rs` and
  `driver/tests/dynamic.rs` propagate it instead of discarding it. No public `Plan`,
  trait, batch or C ABI signature changes.
- The `p <= budget <= pool size` argument is about the context `Plan::run` normalizes
  (`with_budget`, `exec.rs:81-87`), not about an arbitrary hand-built `Exec::Rayon`.
- `p <= budget <= pool size` by construction (the grid is clamped to the budget,
  `driver/mod.rs:588-595`), so the only way SPMD is unavailable is the worker context.
  That check is a pure predicate, evaluated before the lease is taken.
- SPMD coordination is still the pool's blocking mutex, taken at execution time
  (`exec.rs:243-249`). There is **no** reservation and **no** "coordination busy" error:
  waiting for a concurrent broadcast on the same pool is the documented behaviour, and
  making it an error would need a lease held from resolution to execution.
- A barrier-free cell run at `pm == 1` needs no barrier (see §1.2), so dispatching it
  through `for_each_partition` rather than `broadcast` removes the last unnecessary call
  into the co-scheduling path; the arithmetic is unchanged.
- Bitwise identity with the serial path is preserved: barrier-free lanes run whole entries
  with the serial blocking, and barrier-free cells run strips with the existing
  active-width blocking (`rg.with_threads(p)`), each with its own buffers, exactly as
  today. Only the dispatch primitive changes.

### 2.4 What is observable

Nothing new is reported publicly. The route is already visible through
`PoolStats { entries, broadcasts, inline_runs }`: a serial run increments none, a
barrier-free run from outside the pool increments `entries`, the same run from a worker
increments `inline_runs`, and an SPMD run increments `broadcasts`. Tests assert the route
with those counters instead of inferring it from timing.

### 2.5 Not in this change

- **A barrier-free alternate for `pm > 1`.** Today a plan that needs the shared packed
  panel can only run SPMD; per-lane private packing would let it run barrier-free at the
  cost of duplicate packing work. That is an optimization with a measured tradeoff, so a
  worker-context call that needs it errors rather than being silently serialized.
- A `Plan::workspace_req()`, a `NoWorkspace` provider, an SPMD reservation, and a public
  route report (see §5).
- cpueinsum's prepared binary contract and grouped plan form (#4), and anything in
  tenferro (#2004).

### 2.6 Compatibility and consumers

`tprims-contract` and `tprims-exec` are unpublished. The consumers outside this
repository are cpueinsum (pinned to `fed859ac`, which predates this work) and the older
`tenferro-rs/ext/tenferro-cpu-tprims` extension. No compatibility shim is added. The public `Plan::execute_*` signatures do not change: the workspace travels
on `Exec`. What changes for a caller is that a serial call now allocates its scratch per
call unless it passes `Exec::serial_with_workspace`.

| Consumer | Change |
|---|---|
| `crates/tprims-contract/src/plan/mod.rs` | drop the arena field and the fallback; pass `exec.workspace()` |
| `crates/tprims-exec/src/exec.rs` | the `SerialWithWorkspace` variant, `workspace()` lifetime |
| `crates/tprims-exec/src/workspace.rs` | checkout/return, `WorkspaceStats`, private `TeamSet`, exact `TeamLease` accounting |
| `crates/tprims-exec/src/pool.rs` | `trim_workspace` unchanged; stats reachable through `workspace()` |
| `crates/tprims-contract/src/driver/mod.rs` | route classification, fallible entry points, typed error; no serial fallback |
| `crates/tprims-contract/src/driver/tests/compat.rs`, `tests/dynamic.rs` | adapters propagate the result; DynamicTiles refusal becomes a no-write error test |
| `crates/tprims-contract/src/driver/batch.rs` | remove the worker gate and its doc claim |
| `crates/tprims-contract/src/batch.rs` | items inherit the outer workspace, serially |
| `crates/tprims-capi/src/executor.rs`, `execute.rs` | the serial executor owns an arena |
| `crates/tprims-testkit` | `NaivePlan` ignores the provider; no signature change |
| `benchmarks/` (`threads.rs`, the serial entries) | keep one arena outside the timed loop for 1T rows |
| existing worker-path tests | batch worker calls now succeed barrier-free; SPMD worker calls now error with no write |
| cpueinsum (`Cargo.toml` pin, `src/plan.rs`) | same-revision bump; one-shot and reusable providers passed explicitly |
| `ext/tenferro-cpu-tprims` in tenferro | migrated or deleted by #2004, not here |

The no-write guarantee is stated per **single contraction**: the resolution above happens
before the packed contraction can write, and the output-only passes (`alpha == 0`, empty
`K`) are elementwise and have no unavailable route. An aggregate call that iterates
several contractions (`contract_batched`, cpueinsum's N-ary plan) keeps its existing
"validate everything, then execute" preflight, but does not promise route-error atomicity
across items; that needs every route resolved up front and is not part of this change.

## 3. Acceptance

- A `Plan` owns no mutable execution storage and a cached plan does not grow when it is
  executed from many short-lived threads.
- Serial execution with `Exec::serial_with_workspace` reuses buffers across calls (warm
  steady state allocates nothing); `stats()` reports retained and leased bytes; `trim`
  releases idle storage and never invalidates a live lease.
- Concurrent `stats()`/`trim()` against a running worker execution is not a data race
  (checked with a stress test, and by construction: a slot is either in the map or owned
  by exactly one thread).
- Route dispatch matches the table in §2.3, verified by `PoolStats` counters for:
  below-threshold serial, batch barrier-free from a coordinator, batch barrier-free from a
  same-pool worker, barrier-free cells at `pm == 1`, and SPMD.
- A worker-context call that needs `pm > 1` returns `ExecError::Unavailable` with **no
  partial write** (sentinel output unchanged, no counter moved), from both the packed and
  the DynamicTiles path.
- `stats()` is exact when nothing is in flight; `Exec` keeps `Copy`, `Send` and `Debug`, and
  `with_budget` never drops a serial workspace.
- Numerical results of the packed driver stay bitwise identical to the serial path at
  every width, as today. The faer and elementwise strategies are unchanged and are not
  covered by that claim (`tests/faer_schedule.rs:80-91` already tests numerical agreement
  for faer, not bitwise identity).
- `Pool::borrow` in a loop is gone from the steady state: one wrapper per `ThreadPool`.

## 4. Rollout

1. This repository: the design and the implementation above, in one PR.
2. Then cpueinsum (`fed859ac` → this revision) and its own prepared-contract work
   (tensor4all/cpueinsum-rs#4).
3. Then tenferro's session-admission change (#2004), which is what removes the need for
   the plan-owned arena in the first place.

## 5. Pre-review dispositions

`openai/gpt-6.1-sol`, read-only, against this repository at `cbb6edd`.

| # | Finding | Disposition |
|---|---|---|
| 1 | BLOCKER: checking "SPMD coordination free" at resolution and locking at execution races; holding the guard needs a reservation type | **Accepted.** The "coordination free" precondition is dropped: the pool mutex is taken at execution (blocking, as today) and there is no busy error or reservation (§2.3). |
| 2 | BLOCKER: the map mutex does not protect worker-slot contents | **Accepted and verified** (`workspace.rs:368-402` vs `:350-364`, `:440-449`). Fixed by checkout/return (§2.2). |
| 3 | IMPORTANT: a new resolver would replace the existing partition/width policy | **Accepted.** The policy is unchanged; only the worker gate is removed and the resulting geometry is classified (§2.3). |
| 4 | IMPORTANT: an argument-less `workspace_req()` is underdefined | **Accepted.** Not added (§2.1). |
| 5 | IMPORTANT: accounting misses scatter/barriers and the `lease.b.ensure` bypass can underflow; a per-thread residue remains | **Accepted.** `WorkspaceStats` is defined over the whole payload; growth goes through `panel()` only; `TeamLease` subtracts what it accounted; the per-thread residue is documented (§2.2). |
| 6 | IMPORTANT: a non-zero request through `NoWorkspace` is not an internal invariant | **Accepted.** No `NoWorkspace` type; `None` is the no-owner path (§2.1). |
| 7 | IMPORTANT: canonical execution surface and consumer migration scope are understated | **Partly accepted.** The `Exec`-carries-the-workspace choice keeps the `Plan` entries, the trait, `batch` and the testkit signatures unchanged; the C adapter's serial arena and cpueinsum's pin are in the migration list (§2.6). |
| 8 | IMPORTANT: partial-write contract for aggregate calls is undefined | **Accepted.** The no-write guarantee is limited to a single contraction and the aggregate behaviour is stated (§2.6). |
| 9 | IMPORTANT: the bitwise/route acceptance criteria must be limited to packed | **Accepted.** §3 states it. |
| — | Delete list (`workspace_req`, `NoWorkspace`, the new lane policy, the全-strategy report) | **Adopted**; see §2.1, §2.3, §2.5. |

A second read-only pass on this revision (`openai/gpt-6.1-sol`) found no blocker. Its five
findings are folded in: the hand-written `Debug` (§2.1); the single accounting boundary,
both-direction settlement and the quiescent-exactness wording (§2.2); the fallible driver
and the test adapters (§2.3); the payload-only zero-request wording (§2.2); and the
active-width blocking of barrier-free cells (§2.3). It also confirmed the checkout/return
design, the `pm == 1` classification (including DynamicTiles), the route table's width
argument, and the no-write ordering.
