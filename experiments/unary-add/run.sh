#!/usr/bin/env bash
# Record one cell of the unary-add probe: three sessions, each an idle-gated,
# pinned, one-thread run through the measured checkout's own guard.
#
#   run.sh [sessions]
#
# One CPU, one measurement at a time. The guard is `benchmarks/scripts/pinned.sh`
# of the checkout this probe is built against, so its log records the window,
# the threshold and the SMT siblings it checked.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
sessions=${1:-3}
idle=$root/benchmarks/scripts/idle_cpus.py
pinned=$root/benchmarks/scripts/pinned.sh
out=$here/results
mkdir -p "$out"

cargo build --release --manifest-path "$here/Cargo.toml" >&2

for s in $(seq 1 "$sessions"); do
    until cpu=$(python3 "$idle" pick 1 --seconds 3 2>/dev/null); do
        echo "run.sh: no idle CPU; waiting" >&2
        sleep 30
    done
    echo "run.sh: session $s on cpu$cpu" >&2
    bash "$pinned" "$cpu" -- "$here/target/release/unary-add" \
        --reps 5 --prime-ms 500 --csv "$out/session$s.csv" \
        > "$out/session$s.txt" 2> "$out/session$s.guard"
done

python3 - "$out" "$sessions" <<'PY'
import csv, pathlib, statistics, sys
out, sessions = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rows = {}
for s in range(1, sessions + 1):
    for r in csv.DictReader((out / f"session{s}.csv").open()):
        rows.setdefault((r["shape"], r["dtype"], r["arm"]), []).append(float(r["ns"]))
print(f"| shape | dtype | arm | ns/call (median of {sessions} sessions) | spread |")
print("|---|---|---|---:|---:|")
for (shape, dtype, arm), v in sorted(rows.items()):
    med = statistics.median(v)
    spread = (max(v) - min(v)) / med if med else 0.0
    print(f"| `{shape}` | {dtype} | {arm} | {med:.0f} | {100 * spread:.0f}% |")
PY
