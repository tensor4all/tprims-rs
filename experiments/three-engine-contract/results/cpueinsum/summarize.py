"""cpueinsum final benchmark: session median of per-call medians, in ns per step."""
import csv, glob, os, statistics
from collections import defaultdict
here = os.path.dirname(os.path.abspath(__file__))
t = defaultdict(list)
steps = {}
order = []
for f in sorted(glob.glob(os.path.join(here, "session*.csv"))):
    for r in csv.reader(l for l in open(f) if not l.startswith("#")):
        if r[0] == "case":
            continue
        t[(r[0], r[4])].append(float(r[5]))
        steps[r[0]] = int(r[2])
        if r[0] not in order:
            order.append(r[0])
arms = ["tc_exec", "tp_exec", "ce_exec", "tf_exec", "tc_call", "tp_call", "ce_call", "tf_call"]
print("| case | " + " | ".join(arms) + " | ce_call/tc_call |")
print("|---|" + "---|" * (len(arms) + 1))
for c in order:
    cells = []
    for a in arms:
        v = statistics.median(t[(c, a)])
        spread = (max(t[(c, a)]) - min(t[(c, a)])) / v
        cells.append(f"{v / steps[c]:.0f}" + ("*" if spread > 0.1 else ""))
    ratio = statistics.median(t[(c, "ce_call")]) / statistics.median(t[(c, "tc_call")])
    print(f"| {c} | " + " | ".join(cells) + f" | {ratio:.2f} |")
print("\nns per pairwise step (session median of per-call medians); * = cross-session spread > 10%")
