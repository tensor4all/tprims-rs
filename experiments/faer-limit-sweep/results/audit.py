#!/usr/bin/env python3
"""Audit the faer-limit sweep's recorded rows against its own claims.

Every statement the sweep and the decision-log row make about these rows is a
check here, so a reader does not have to take them on trust:

  1. `mnk` equals the shape class's volume (`n**3` for the GEMM classes,
     `2 * chi**3` for the MPS steps).
  2. the `default` arm's routing matches `FaerLimit`'s defaults (c64 bounded at
     `1 << 17`, the other dtypes unbounded) — i.e. `algorithm == packed` exactly
     when the volume exceeds the bound.
  3. the rows where both arms ran the same algorithm are exactly the c64 rows
     above that bound, which is the region where faer is unmeasured.
  4. outside that region, no 1T row has the packed arm faster than the planner's
     own choice.
  5. the two arms agree (the CHECK lines) and the guard said "idle before and
     after" for every session in the results directory.

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

    # 2. routing against the bound
    mis = []
    for r in rows:
        if r["arm"] != "default":
            continue
        b = BOUND[r["dtype"]]
        expect = b is not None and int(r["mnk"]) > b
        if expect != (r["algorithm"] == "packed"):
            mis.append(r)
    print(f"2. default routing vs FaerLimit defaults: {len(mis)} mismatches")
    failures += [
        f"routing {r['class']} {r['dtype']} {r['params']} mnk={r['mnk']} -> {r['algorithm']}/{r['reason']}"
        for r in mis[:5]
    ]

    # 3. and 4. per-group comparisons
    groups: dict[tuple, dict[str, dict]] = {}
    keys = [k for k in rows[0] if k not in GROUP_EXCLUDE and k != "threads"]
    for r in rows:
        g = tuple(r[k] for k in keys) + (r["threads"],)
        groups.setdefault(g, {})[r["arm"]] = r
    same = [g for g, d in groups.items() if d["default"]["algorithm"] == d["packed"]["algorithm"]]
    not_c64_above = [
        g
        for g in same
        if not (g[keys.index("dtype")] == "c64" and int(groups[g]["default"]["mnk"]) > (1 << 17))
    ]
    print(
        f"3. groups where both arms run the same algorithm: {len(same)} of {len(groups)} "
        f"({len(not_c64_above)} not c64-above-the-bound)"
    )
    failures += [f"same-algorithm group outside the blind region: {g}" for g in not_c64_above[:5]]

    faster = []
    for g, d in groups.items():
        if groups[g]["default"]["algorithm"] != d["packed"]["algorithm"] and g[-1] == "1":
            faster.append((g, statistics.median([float(d["packed"]["ns"])]) < statistics.median([float(d["default"]["ns"])])))
    wins = [f for f, w in faster if w]
    print(f"4. 1T rows where packed beats the planner's choice: {len(wins)}")
    failures += [f"packed faster at 1T: {g}" for g in wins[:5]]

    # 5. the guards and the CHECK lines
    for csv_path in paths:
        stem = csv_path.name.replace(".csv", "")
        guard = csv_path.with_name(stem + ".guard")
        text = csv_path.with_name(stem + ".txt")
        ok_guard = guard.exists() and "idle before and after" in guard.read_text()
        ok_check = text.exists() and "CHECK" in text.read_text() and "MISMATCH" not in text.read_text()
        if not (ok_guard and ok_check):
            failures.append(f"{stem}: guard={'ok' if ok_guard else 'missing'} checks={'ok' if ok_check else 'bad'}")
    print(f"5. guards and CHECK lines: {len(paths) - len([f for f in failures if 'guard=' in f])} of {len(paths)} sessions ok")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nall audit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
