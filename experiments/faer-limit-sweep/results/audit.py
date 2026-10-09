#!/usr/bin/env python3
"""Audit the faer-limit sweep's recorded rows against its own claims.

Every statement the sweep and the 2026-10-09 decision-log row make about these
rows is a check here, so a reader does not have to take them on trust:

  1. `mnk` equals the shape class's volume (`n**3` for the GEMM classes,
     `2 * chi**3` for the MPS steps).
  2. the `default` arm's routing matches `FaerLimit`'s defaults (c64 bounded at
     `1 << 17`, the other dtypes unbounded) — i.e. `algorithm == packed` exactly
     when the volume exceeds the bound.
  3. every session of a group agrees on the algorithm and the reason, so a
     comparison across sessions is comparing the same two routes.
  4. the groups where both arms run the same algorithm are exactly the c64 rows
     above that bound, which is the region where faer is unmeasured.
  5. outside that region, no 1T group has the packed arm faster than the
     planner's own choice **by more than the drift band**. The band comes from
     the second glob when one is given: the same group measured twice drifts by
     a median of a few percent and a p90 of about 15%, and a sign that flips
     between two runs is a tie, not a finding. Without a second glob the band is
     a conservative 2% and the report says so.
  6. the margins the row quotes, computed the way the manifest does (the median
     over sessions of each arm's ns, per group), printed per dtype because the
     row's numbers are per dtype.
  7. each session's guard said "idle before and after" and its CHECK lines report
     no mismatch.
  8. the run-to-run drift itself, when a second glob is given: how many groups
     were measured twice, the drift distribution, and how many flipped the sign
     of (packed - planner). That number is the band's provenance.

    python3 results/audit.py [glob] [compare_glob]

The default glob is `results/*session*.csv` beside this file; pass one to audit a
different set (for example the accumulating-C-mode run), and a second to compare
two recordings of the same configurations. Rows are grouped by every column
except `arm`, `ns`, `algorithm` and `reason`, so a `c_mode` column is audited per
mode without changes here; a comparison drops `c_mode` when only one side has it.
Exits non-zero if a check fails.
"""
from __future__ import annotations

import csv
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
BOUND = {"f32": None, "f64": None, "c32": None, "c64": 1 << 17}
GROUP_EXCLUDE = {"arm", "ns", "algorithm", "reason"}
# The dtype order the decision-log row reports in.
DTYPES = ("f32", "f64", "c32", "c64")


def load(paths: list[pathlib.Path]) -> list[dict]:
    rows = []
    for p in paths:
        lines = [l for l in p.read_text().splitlines() if not l.startswith("#")]
        rows += [r for r in csv.DictReader(lines) if r.get("class")]
    return rows


def volume(r: dict) -> int:
    cls, par = r["class"], r["params"]
    if cls == "gemm":
        n = int(par.split("=")[1])
        return n**3
    if cls == "gemm_batched":
        n = int(par.split(" ")[0].split("=")[1])
        return n**3
    chi = int(par.split("=")[1])
    return 2 * chi**3


def gname(g: tuple, keys: list[str]) -> str:
    """`class params [mode]` for a group tuple, for the report lines."""
    parts = [f"{g[keys.index('class')]} {g[keys.index('params')]}"]
    if "c_mode" in keys:
        parts.append(f"[{g[keys.index('c_mode')]}]")
    return " ".join(parts)


def main() -> int:
    args = sys.argv[1:]
    paths = sorted(HERE.glob(args[0] if args else "*session*.csv"))
    compare_paths = sorted(HERE.glob(args[1])) if len(args) > 1 else []
    if not paths:
        print("audit: no CSVs matched", file=sys.stderr)
        return 2
    rows = load(paths)
    failures: list[str] = []

    # 1. volume
    bad = [r for r in rows if int(r["mnk"]) != volume(r)]
    print(f"1. mnk vs the shape formula: {len(rows)} rows, {len(bad)} mismatches")
    failures += [f"volume mismatch {r['class']} {r['params']} mnk={r['mnk']}" for r in bad[:5]]

    # group every row, per arm, so the ratios are the manifest's medians
    keys = [k for k in rows[0] if k not in GROUP_EXCLUDE]
    groups: dict[tuple, dict[str, list[dict]]] = {}
    for r in rows:
        g = tuple(r[k] for k in keys)
        groups.setdefault(g, {}).setdefault(r["arm"], []).append(r)

    def med(g: tuple, arm: str) -> float:
        return statistics.median([float(x["ns"]) for x in groups[g][arm]])

    # 2. routing against the bound
    mis = []
    for g, arms in groups.items():
        for r in arms.get("default", []):
            b = BOUND[r["dtype"]]
            expect = b is not None and int(r["mnk"]) > b
            if expect != (r["algorithm"] == "packed"):
                mis.append(r)
    print(f"2. default routing vs FaerLimit defaults: {len(mis)} mismatches")
    failures += [
        f"routing {r['class']} {r['dtype']} {r['params']} mnk={r['mnk']} -> {r['algorithm']}/{r['reason']}"
        for r in mis[:5]
    ]

    # 3. every session of a group ran the same route
    drift = []
    for g, arms in groups.items():
        for arm, rs in arms.items():
            if len({(r["algorithm"], r["reason"]) for r in rs}) > 1:
                drift.append((g, arm))
    print(f"3. groups whose sessions disagree on the route: {len(drift)}")
    failures += [f"route drift in {gname(g, keys)}/{arm}" for g, arm in drift[:5]]

    # 4. and 5. the comparison, on medians
    comparable = {
        g: arms for g, arms in groups.items() if "default" in arms and "packed" in arms
        and arms["default"][0]["algorithm"] != arms["packed"][0]["algorithm"]
    }
    same = [g for g, arms in groups.items() if g not in comparable]
    not_c64_above = [
        g for g in same if not (g[keys.index("dtype")] == "c64" and int(g[keys.index("mnk")]) > (1 << 17))
    ]
    print(
        f"4. groups where both arms run the same route: {len(same)} of {len(groups)} "
        f"({len(not_c64_above)} not c64-above-the-bound)"
    )
    failures += [f"same-route group outside the blind region: {gname(g, keys)}" for g in not_c64_above[:5]]

    one_t = [g for g in comparable if g[keys.index("threads")] == "1"]

    # the drift band: the same configuration measured in two recordings
    def ratio(g: tuple, med_fn) -> float:
        return med_fn(g, "packed") / med_fn(g, "default")

    band, drift_line = 0.02, "2% (no second glob given; a conservative default)"
    pairs_by_width: dict[str, list[tuple]] = {}
    if compare_paths:
        other_rows = load(compare_paths)
        other_keys = [k for k in other_rows[0] if k not in GROUP_EXCLUDE]
        other: dict[tuple, dict[str, list[dict]]] = {}
        for r in other_rows:
            g = tuple(r[k] for k in other_keys)
            other.setdefault(g, {}).setdefault(r["arm"], []).append(r)

        def other_med(g: tuple, arm: str) -> float:
            return statistics.median([float(x["ns"]) for x in other[g][arm]])

        # Pair positionally within each side's own columns: `c_mode` exists on
        # one side only, and zipping the primary's keys against the other's
        # tuple would shift every column after it. A side without `c_mode`
        # recorded the overwrite form only, so pairing an `output` row against it
        # would measure the mode difference rather than the run-to-run drift.
        def key_of(ks: list[str], g: tuple) -> tuple:
            return tuple(v for k, v in zip(ks, g) if k in other_keys)

        shared_other = {key_of(other_keys, g): g for g in other}
        restrict_absent = "c_mode" not in other_keys and "c_mode" in keys
        for g in comparable:
            if restrict_absent and g[keys.index("c_mode")] != "absent":
                continue
            og = shared_other.get(key_of(keys, g))
            if og is None or "default" not in other[og] or "packed" not in other[og]:
                continue
            width = g[keys.index("threads")]
            pairs_by_width.setdefault(width, []).append((g, ratio(g, med), ratio(og, other_med)))
        if pairs_by_width.get("1"):
            drifts = sorted(abs(a - b) for _, a, b in pairs_by_width["1"])
            band = drifts[int(0.9 * (len(drifts) - 1))]
            drift_line = f"{band * 100:.0f}% (p90 of {len(drifts)} paired groups at 1T)"

    faster = [g for g in one_t if ratio(g, med) < 1.0 - band]
    ties = [g for g in one_t if ratio(g, med) < 1.0]
    print(
        f"5. 1T groups where packed beats the planner by more than the drift band "
        f"({drift_line}): {len(faster)} of {len(one_t)}; within the band (a tie): {len(ties)}"
    )
    for g in ties:
        print(f"   tie at {ratio(g, med):.3f}: {gname(g, keys)}")
    failures += [f"packed faster than the drift band at 1T: {gname(g, keys)}" for g in faster[:5]]

    # 6. the margins, per dtype
    print(f"6. margins packed/planner at 1T, median over sessions ({len(one_t)} groups):")
    modes = [""] if "c_mode" not in keys else sorted({g[keys.index("c_mode")] for g in one_t})
    for mode in modes:
        if mode:
            print(f"   c_mode={mode}:")
        for dt in DTYPES:
            sub = [
                g
                for g in one_t
                if g[keys.index("dtype")] == dt and (not mode or g[keys.index("c_mode")] == mode)
            ]
            if not sub:
                continue
            ratios = {g: med(g, "packed") / med(g, "default") for g in sub}
            lo = min(ratios, key=ratios.get)
            hi = max(ratios, key=ratios.get)
            pad = "      " if mode else "   "
            print(
                f"{pad}{dt}: {len(sub)} groups, {ratios[lo]:.2f} (min, {gname(lo, keys)}) "
                f"to {ratios[hi]:.2f} (max, {gname(hi, keys)})"
            )
            top = max(sub, key=lambda g: int(g[keys.index("mnk")]))
            print(
                f"{pad}   largest volume measured: {gname(top, keys)} at mnk "
                f"{top[keys.index('mnk')]}, ratio {ratios[top]:.2f}"
            )

    # 9. the fixed faer arm, when the recording has one: it exists to measure the
    # faer route above the c64 bound, where the planner's own choice is packed.
    if any("faer_forced" in arms for arms in groups.values()):
        forced = {
            g: arms
            for g, arms in groups.items()
            if "faer_forced" in arms and "packed" in arms
        }
        wrong = [g for g in forced if forced[g]["faer_forced"][0]["algorithm"] != "faer"]
        print(
            f"9. faer_forced arm: {len(forced)} groups, routed to faer in "
            f"{len(forced) - len(wrong)}"
        )
        failures += [f"faer_forced did not route to faer: {gname(g, keys)}" for g in wrong[:5]]
        one_t_forced = [g for g in forced if g[keys.index("threads")] == "1"]
        print(f"   packed/faer_forced at 1T ({len(one_t_forced)} groups), per dtype:")
        for dt in DTYPES:
            sub = [g for g in one_t_forced if g[keys.index("dtype")] == dt]
            if not sub:
                continue
            r = {g: med(g, "packed") / med(g, "faer_forced") for g in sub}
            lo, hi = min(r, key=r.get), max(r, key=r.get)
            top = max(sub, key=lambda g: int(g[keys.index("mnk")]))
            print(
                f"      {dt}: {len(sub)} groups, {r[lo]:.2f} (min, {gname(lo, keys)}) to "
                f"{r[hi]:.2f} (max, {gname(hi, keys)}); largest volume "
                f"{gname(top, keys)} at mnk {top[keys.index('mnk')]}, ratio {r[top]:.2f}"
            )
        # the number the bound's value turns on: above the c64 bound, is packed
        # still ahead of the faer route the arm forces?
        above = [
            g
            for g in one_t_forced
            if g[keys.index("dtype")] == "c64" and int(g[keys.index("mnk")]) > (1 << 17)
        ]
        for g in sorted(above, key=lambda g: int(g[keys.index("mnk")])):
            print(
                f"      c64 above the bound: {gname(g, keys)} mnk {g[keys.index('mnk')]} "
                f"packed/faer_forced {med(g, 'packed') / med(g, 'faer_forced'):.2f}"
            )

    # 7. the guards and the CHECK lines
    bad_sessions = []
    for csv_path in paths:
        stem = csv_path.name.replace(".csv", "")
        guard = csv_path.with_name(stem + ".guard")
        text = csv_path.with_name(stem + ".txt")
        ok_guard = guard.exists() and "idle before and after" in guard.read_text()
        ok_check = text.exists() and "CHECK" in text.read_text() and "MISMATCH" not in text.read_text()
        if not (ok_guard and ok_check):
            bad_sessions.append(stem)
    print(f"7. guards and CHECK lines: {len(paths) - len(bad_sessions)} of {len(paths)} sessions ok")
    failures += [f"{s}: guard or CHECK lines not ok" for s in bad_sessions]

    # 8. the run-to-run drift, which is where the band came from
    if pairs_by_width:
        for width in sorted(pairs_by_width):
            pairs = pairs_by_width[width]
            drifts = sorted(abs(a - b) for _, a, b in pairs)
            flipped = sum(1 for _, a, b in pairs if (a - 1) * (b - 1) < 0)
            ahead_both = sum(1 for _, a, b in pairs if a < 1.0 and b < 1.0)
            ahead_beyond = sum(1 for _, a, b in pairs if a < 1.0 - band and b < 1.0 - band)
            print(
                f"8. {width}T run-to-run drift over {len(pairs)} groups measured twice: "
                f"median {statistics.median(drifts):.3f}, p90 {drifts[int(0.9 * (len(drifts) - 1))]:.3f}, "
                f"max {max(drifts):.3f}, sign flips {flipped}, "
                f"packed ahead in both {ahead_both} (by more than the band {ahead_beyond})"
            )
            # Only the width the decision is about must have zero: at 4T the
            # reproducible packed leads are the finding, not a failure.
            if width == "1":
                failures += [
                    f"packed ahead in both recordings by more than the band at 1T: {gname(g, keys)}"
                    for g, a, b in pairs
                    if a < 1.0 - band and b < 1.0 - band
                ]
    else:
        print(
            "8. run-to-run drift: "
            + ("no pairs found" if compare_paths else "no second glob, so no paired groups")
        )

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nall audit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
