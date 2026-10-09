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
     planner's own choice — the substantive claim the decision-log row makes.
  6. the margins the row quotes, computed the way the manifest does (the median
     over sessions of each arm's ns, per group), printed per dtype because the
     row's numbers are per dtype.
  7. each session's guard said "idle before and after" and its CHECK lines report
     no mismatch.

    python3 results/audit.py [glob]

The default glob is `results/*session*.csv` beside this file; pass one to audit a
different set (for example the accumulating-C-mode run). Rows are grouped by
every column except `arm`, `ns`, `algorithm` and `reason`, so a `c_mode` column
is audited per mode without changes here. Exits non-zero if a check fails.
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
    faster = [g for g in one_t if med(g, "packed") < med(g, "default")]
    print(f"5. 1T groups where packed beats the planner's choice: {len(faster)} of {len(one_t)}")
    failures += [f"packed faster at 1T: {gname(g, keys)}" for g in faster[:5]]

    # 6. the margins, per dtype
    print(f"6. margins packed/planner at 1T, median over sessions ({len(one_t)} groups):")
    for dt in DTYPES:
        sub = [g for g in one_t if g[keys.index("dtype")] == dt]
        if not sub:
            continue
        ratios = {g: med(g, "packed") / med(g, "default") for g in sub}
        lo = min(ratios, key=ratios.get)
        hi = max(ratios, key=ratios.get)
        print(
            f"   {dt}: {len(sub)} groups, {ratios[lo]:.2f} (min, {gname(lo, keys)}) "
            f"to {ratios[hi]:.2f} (max, {gname(hi, keys)})"
        )
        # the row also quotes a bound per dtype: "up to mnk X" for the fast ones,
        # and for c64 the largest volume below the bound where faer ran at all
        top = max(sub, key=lambda g: int(g[keys.index("mnk")]))
        print(
            f"      largest volume measured: {gname(top, keys)} at mnk "
            f"{top[keys.index('mnk')]}, ratio {ratios[top]:.2f}"
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

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nall audit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
