"""Median over sessions of per-call median times; ratios vs tc_exec."""
import csv, glob, statistics, sys
from collections import defaultdict
t = defaultdict(list)
order = []
for f in sorted(glob.glob("results/session*.csv")):
    for r in csv.reader(l for l in open(f) if not l.startswith("#")):
        if r[0] == "case":
            continue
        t[(r[0], r[4])].append(float(r[5]))
        if r[0] not in order:
            order.append(r[0])
arms = ["tc_exec","tc_call","tp_exec","tp_packed_exec","tp_call","tf_exec","tf_call","tf_call_spc","tf_call_subs","tf_prep_exec","tf_plan_exec","tf_eager","tf_eager_scoped","tf_traced","tf_traced_nary"]
print("| case | tc_exec µs | " + " | ".join(a for a in arms[1:]) + " |")
print("|" + "---|" * len(arms) + "---|")
for c in order:
    base = statistics.median(t[(c, "tc_exec")])
    cells = [f"{base/1e3:.2f}"]
    for a in arms[1:]:
        v = statistics.median(t[(c, a)])
        spread = (max(t[(c, a)]) - min(t[(c, a)])) / v
        cells.append(f"{v/base:.2f}" + ("*" if spread > 0.1 else ""))
    print(f"| {c} | " + " | ".join(cells) + " |")
print("\nratios = arm / tc_exec (session median of per-call medians); * = cross-session spread > 10%")
