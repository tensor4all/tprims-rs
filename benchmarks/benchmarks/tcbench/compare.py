#!/usr/bin/env python3
"""Compare complete baseline/candidate A/A suites after summarize.py validates them."""
import csv
import math
import statistics
import sys
from pathlib import Path

baseline, candidate = map(Path, sys.argv[1:3])
threads_to_run = tuple(sys.argv[3].split(',')) if len(sys.argv) > 3 else ("1", "4", "8", "12")
assert threads_to_run in (("1", "4", "8", "12"), ("1", "4", "8"))

def load(root):
    rows = list(csv.DictReader((root / "comparison.csv").open()))
    data = {(r["size_mib"], r["dtype"], r["threads"], r["case"]): r for r in rows}
    assert len(rows) == len(data) == 49 * 2 * 2 * len(threads_to_run)
    return data

b, c = load(baseline), load(candidate)
assert b.keys() == c.keys()
fields = ["size_mib", "dtype", "threads", "case", "baseline_s", "candidate_s", "speedup", "baseline_AA", "candidate_AA", "above_noise_regression"]
results = []
for key in sorted(b):
    old, new = b[key], c[key]
    old_s = math.sqrt(float(old["plan_A_s"]) * float(old["plan_A2_s"]))
    new_s = math.sqrt(float(new["plan_A_s"]) * float(new["plan_A2_s"]))
    noise_b, noise_c = float(old["plan_AA_spread"]), float(new["plan_AA_spread"])
    regression = new_s / old_s - 1
    results.append(dict(zip(fields, [*key, old_s, new_s, old_s/new_s, noise_b, noise_c, regression > max(0.1, noise_b, noise_c)])))
with (candidate / "vs-baseline.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(results)
print("| MiB | dtype | T | baseline / candidate time |")
print("| --- | --- | --- | --- |")
for size in ("1", "16"):
    for dtype in ("f64", "c64"):
        for threads in threads_to_run:
            group = [r for r in results if (r["size_mib"], r["dtype"], r["threads"]) == (size, dtype, threads)]
            ratio = statistics.geometric_mean(r["speedup"] for r in group)
            print(f"| {size} | {dtype} | {threads} | {ratio:.3f} |")
print("\nPrimary cases (>=2x, exceeds both A/A spreads):")
for case in ("ajbc-ckba-jk", "ajbdc-ckbad-jk"):
    r = next(r for r in results if (r["size_mib"], r["dtype"], r["threads"], r["case"]) == ("16", "c64", "8", case))
    passes = r["speedup"] >= 2 and r["speedup"]-1 > max(r["baseline_AA"], r["candidate_AA"])
    print(case, f"{r['speedup']:.3f}x", "PASS" if passes else "FAIL")
regressions = [r for r in results if r["above_noise_regression"]]
print(f"\nIndividual >10% regressions above both A/A spreads: {len(regressions)}")
for r in regressions:
    print(r["size_mib"], r["dtype"], r["threads"], r["case"], f"{r['speedup']:.3f}x")
