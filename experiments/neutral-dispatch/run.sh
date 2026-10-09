#!/usr/bin/env bash
# Record the neutral-dispatch cell: three sessions, each idle-gated, pinned to
# four CPUs of one L3 domain, one measurement at a time.
#
#   run.sh [sessions]
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
    until cpus=$(python3 "$idle" pick 4 --seconds 3 2>/dev/null); do
        echo "run.sh: no idle 4-CPU domain; waiting" >&2
        sleep 30
    done
    echo "run.sh: session $s on cpus $cpus" >&2
    bash "$pinned" "$cpus" -- "$here/target/release/neutral-dispatch" \
        --reps 5 --prime-ms 500 --csv "$out/session$s.csv" \
        > "$out/session$s.txt" 2> "$out/session$s.guard"
done

python3 - "$out" "$sessions" <<'PY'
import csv, pathlib, statistics, sys
out, sessions = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rows = {}
for s in range(1, sessions + 1):
    for r in csv.DictReader((out / f"session{s}.csv").open()):
        rows.setdefault((r["case"], r["dtype"], r["threads"], r["arm"]), []).append(float(r["ns"]))
print(f"| case | dtype | T | arm | ns (median of {sessions}) | spread |")
print("|---|---|---|---|---:|---:|")
for (case, dtype, t, arm), v in sorted(rows.items()):
    med = statistics.median(v)
    spread = (max(v) - min(v)) / med if med else 0.0
    print(f"| `{case}` | {dtype} | {t} | {arm} | {med:.0f} | {100 * spread:.0f}% |")
print()
print("| case | dtype | T | prepare trait/concrete | exec trait/concrete |")
print("|---|---|---|---:|---:|")
keys = {(c, d, t) for (c, d, t, _) in rows}
for (case, dtype, t) in sorted(keys):
    g = lambda a: statistics.median(rows[(case, dtype, t, a)])
    print(f"| `{case}` | {dtype} | {t} | {g('prepare_trait') / g('prepare_concrete'):.2f} | {g('exec_trait') / g('exec_concrete'):.2f} |")
PY
