#!/usr/bin/env python3
"""The per-shape ratios of the three-engine corpus, in the original figure's form.

The figure Lukas measured was tenferro / tensorcontract speedup per case, so > 1
means tensorcontract was faster. This prints that, plus the tprims ratios, plus
the per-call overhead each engine pays on top of its own prebuilt execution
(`*_call - *_exec`), which is what decides the small cases.

    python3 results/findings.py [n]

`n` is how many sessions to fold in (default: all `results/session*.csv`). Values
are the median over sessions of the per-session median.
"""
from __future__ import annotations

import csv
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
PAIRS = [("tc", "tc_exec", "tc_call"), ("tp", "tp_exec", "tp_call"), ("tf", "tf_exec", "tf_call")]


def load(paths: list[pathlib.Path]) -> dict[tuple[str, str], float]:
    per_session: dict[tuple[str, str], list[float]] = {}
    for path in paths:
        lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
        for row in csv.DictReader(lines):
            per_session.setdefault((row["case"], row["arm"]), []).append(float(row["median_ns"]))
    return {k: statistics.median(v) for k, v in per_session.items()}


def main() -> None:
    sessions = sorted(HERE.glob("session[0-9]*.csv")) if sys.argv[1:] == [] else [
        HERE / f"session{i}.csv" for i in range(1, int(sys.argv[1]) + 1)
    ]
    med = load(sessions)
    cases = sorted({c for c, _ in med})
    dtype_of = {}
    for path in sessions:
        lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
        for row in csv.DictReader(lines):
            dtype_of[row["case"]] = row["dtype"]
    print(f"sessions: {', '.join(p.name for p in sessions)}")
    print()

    print("## Ratios to tensorcontract's prebuilt execution")
    print()
    print("`tc_exec` is the denominator; `<1` means tprims or tenferro ran the same")
    print("program faster than tensorcontract, at the same boundary.")
    print()
    header = ["case", "dtype", "tc_exec ns"]
    for _, _, _ in PAIRS:
        pass
    cols = ["tp_exec", "tf_exec", "tc_call", "tp_call", "tf_call"]
    print("| " + " | ".join(header + cols) + " |")
    print("|" + "---|" * (len(header) + len(cols)))
    for c in cases:
        base = med.get((c, "tc_exec"))
        cells = [f"{med[(c, a)] / base:.2f}" if (c, a) in med and base else "-" for a in cols]
        print(f"| `{c}` | {dtype_of[c]} | {base:.0f} | " + " | ".join(cells) + " |")
    print()

    print("## What each engine pays per call on top of its own prebuilt execution")
    print()
    print("`*_call - *_exec`, per program call, and per pairwise step of the MPS")
    print("chains (64 steps at L=32). This is host work: planning, allocation and")
    print("routing, not the kernel.")
    print()
    print("| case | dtype | tc ns/call | tp ns/call | tf ns/call | tc ns/step | tp ns/step | tf ns/step |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for c in cases:
        if "mps" not in c:
            continue
        steps = 64
        row = [med[(c, call)] - med[(c, exe)] for _, exe, call in PAIRS]
        dt = dtype_of[c]
        print(
            f"| `{c}` | {dt} | {row[0]:.0f} | {row[1]:.0f} | {row[2]:.0f} | "
            f"{row[0] / steps:.0f} | {row[1] / steps:.0f} | {row[2] / steps:.0f} |"
        )
    print()

    print("## The largest c64 case, where the packed driver is the slower one")
    print()
    print("| arm | ns/call | ratio to `tc_exec` |")
    print("|---|---:|---:|")
    c = "mps_chain_L32_chi64"
    base = med[(c, "tc_exec")]
    for arm in ["tc_exec", "tc_call", "tp_exec", "tp_call", "tf_exec", "tf_call"]:
        if (c, arm) in med:
            print(f"| `{arm}` | {med[(c, arm)]:.0f} | {med[(c, arm)] / base:.2f} |")


if __name__ == "__main__":
    main()
