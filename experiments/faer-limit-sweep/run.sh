#!/usr/bin/env bash
# Record the faer/packed crossover sweep: three sessions, each running a 1T and
# a 4T sub-run on idle CPU sets picked from the measured checkout's own
# `idle_cpus.py`, through that checkout's `pinned.sh`.
#
#   run.sh [sessions] [set]
#
# One binary records every case, both C modes (overwrite `absent`, accumulating
# `output`) and the arms of the requested set; each sub-run writes
# `results/<set>-session$s.{csv,txt,guard}`. One measurement at a time. Every
# sub-run is its own guarded process, and its guard log must say the cores were
# idle before and after; a sub-run that does not is discarded and retried.
#
# Sets
#   triarm  default, packed and faer_forced (the default).
#   cmode   default and packed only, the two-arm set the earlier
#           `cmode-session*` files were recorded with.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
sessions=${1:-3}
setname=${2:-triarm}
case "$setname" in
    triarm) arms=default,packed,faer_forced ;;
    cmode) arms=default,packed ;;
    *) echo "run.sh: unknown set '$setname' (triarm|cmode)" >&2; exit 2 ;;
esac
prefix=$setname-session
idle=$root/benchmarks/scripts/idle_cpus.py
pinned=$root/benchmarks/scripts/pinned.sh
target=${CARGO_TARGET_DIR:-$root/target-sweep}
bin=$target/release/faer-limit-sweep
out=$here/results
mkdir -p "$out"

CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-16} CARGO_TARGET_DIR=$target \
    cargo build --release --manifest-path "$here/Cargo.toml" >&2

# One guarded, pinned process. The 1T arm runs on `pick 1` and the 4T arm on
# `pick 4` (the pool is sized to that set; the binary asserts the effective
# width against Cpus_allowed_list).
run_pinned() {
    local s=$1 t=$2 attempt cpu
    for attempt in 1 2 3 4 5; do
        if ! cpu=$(python3 "$idle" pick "$t" --seconds 3 2>/dev/null); then
            echo "run.sh: no idle ${t}-CPU set; waiting" >&2
            sleep 30
            continue
        fi
        echo "run.sh: session $s, ${t}T on cpus $cpu" >&2
        if bash "$pinned" "$cpu" -- "$bin" --threads "$t" --arms "$arms" \
            --reps 5 --prime-ms 500 --csv "$out/.tmp.$s.$t.csv" \
            > "$out/.tmp.$s.$t.txt" 2> "$out/.tmp.$s.$t.guard" \
            && grep -q "idle before and after" "$out/.tmp.$s.$t.guard"; then
            {
                echo "=== session $s, ${t}T, cpus=$cpu ==="
                cat "$out/.tmp.$s.$t.guard"
            } >> "$out/$prefix$s.guard"
            cat "$out/.tmp.$s.$t.txt" >> "$out/$prefix$s.txt"
            if [ -s "$out/$prefix$s.csv" ]; then
                tail -n +2 "$out/.tmp.$s.$t.csv" >> "$out/$prefix$s.csv"
            else
                cat "$out/.tmp.$s.$t.csv" >> "$out/$prefix$s.csv"
            fi
            rm -f "$out/.tmp.$s.$t.csv" "$out/.tmp.$s.$t.txt" "$out/.tmp.$s.$t.guard"
            return 0
        fi
        echo "run.sh: session $s ${t}T discarded (no idle guard); retrying" >&2
        rm -f "$out/.tmp.$s.$t.csv" "$out/.tmp.$s.$t.txt" "$out/.tmp.$s.$t.guard"
        sleep 5
    done
    echo "run.sh: session $s ${t}T produced no valid guarded run" >&2
    return 1
}

for s in $(seq 1 "$sessions"); do
    : > "$out/$prefix$s.csv"
    : > "$out/$prefix$s.txt"
    : > "$out/$prefix$s.guard"
    run_pinned "$s" 1
    run_pinned "$s" 4
done

python3 - "$out" "$sessions" "$setname" <<'PY'
import csv, pathlib, statistics, sys

out, sessions, set_ = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
# key -> list of per-session ns; order of first appearance is the corpus order.
runs = {}
meta = {}
for s in range(1, sessions + 1):
    for r in csv.DictReader((out / f"{set_}-session{s}.csv").open()):
        key = (r["c_mode"], r["threads"], r["class"], r["dtype"], r["params"], r["arm"])
        runs.setdefault(key, []).append(float(r["ns"]))
        meta[key] = r

def med(key):
    return statistics.median(runs[key])

def spread(key):
    v = runs[key]
    m = statistics.median(v)
    return (max(v) - min(v)) / m if m else 0.0

# Group rows by case, keeping corpus order.
seen = []
for (c_mode, threads, cls, dtype, params, arm) in runs:
    row = (c_mode, threads, cls, dtype, params)
    if row not in seen:
        seen.append(row)

print()
if set_ == "triarm":
    print(
        "| c_mode | threads | class | dtype | params | mnk | default | packed | "
        "faer_forced | ns default | ns packed | ns faer_forced | packed/faer_forced | spread |"
    )
    print("|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|")
    for (c_mode, threads, cls, dtype, params) in seen:
        d = (c_mode, threads, cls, dtype, params, "default")
        p = (c_mode, threads, cls, dtype, params, "packed")
        f = (c_mode, threads, cls, dtype, params, "faer_forced")
        nd, np_, nf = med(d), med(p), med(f)
        ratio = np_ / nf if nf else float("inf")
        sd = max(spread(d), spread(p), spread(f))
        print(
            f"| {c_mode} | {threads} | `{cls}` | {dtype} | {params} | {meta[d]['mnk']} | "
            f"{meta[d]['algorithm']} | {meta[p]['algorithm']} | {meta[f]['algorithm']} | "
            f"{nd:.0f} | {np_:.0f} | {nf:.0f} | {ratio:.3f} | {100 * sd:.0f}% |"
        )
else:
    print(
        "| c_mode | threads | class | dtype | params | mnk | default | packed | "
        "ns default | ns packed | default/packed | spread |"
    )
    print("|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|")
    for (c_mode, threads, cls, dtype, params) in seen:
        d = (c_mode, threads, cls, dtype, params, "default")
        p = (c_mode, threads, cls, dtype, params, "packed")
        nd, np_ = med(d), med(p)
        ratio = nd / np_ if np_ else float("inf")
        sd = max(spread(d), spread(p))
        print(
            f"| {c_mode} | {threads} | `{cls}` | {dtype} | {params} | {meta[d]['mnk']} | "
            f"{meta[d]['algorithm']} | {meta[p]['algorithm']} | {nd:.0f} | {np_:.0f} | "
            f"{ratio:.3f} | {100 * sd:.0f}% |"
        )
PY
