#!/usr/bin/env bash
# Grid-only experiment, fixed before measurements in scaling worklog.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: grid-diagnostic.sh OUT NATIVE_BIN' >&2; exit 2; }
out=$(realpath -m "$1"); bin=$(realpath "$2")
root=$(cd "$(dirname "$0")/../../.." && pwd); cd "$root"
mkdir -p "$out"
export LD_LIBRARY_PATH="${TBLIS_ROOT:?}/lib:${TBLIS_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
{ date -Iseconds; sha256sum "$bin"; git rev-parse HEAD; rustc -Vv; lscpu -e=CPU,CORE,SOCKET,CACHE; } > "$out/manifest.txt"
git diff > "$out/source.patch"
for stage in verify A A2; do
    for size in 1 16; do
        for t in 1 4 8; do
            cpus="4-$((t+3))"
            case "$t" in
                1) grids='1x1';;
                4) grids='4x1 1x4 2x2';;
                8) grids='8x1 1x8 4x2 2x4';;
            esac
            [[ $stage != A2 ]] || grids=$(printf '%s\n' $grids | tac | tr '\n' ' ')
            for grid in $grids; do
                for filter in abcijk ajb; do
                    stem="$out/$stage-${size}m-${t}t-${grid}-${filter}"
                    if [[ $stage == verify ]]; then
                        env TCBENCH_PARTITION="$grid" bash benchmarks/scripts/pinned.sh "$cpus" -- "$bin" verify --threads "$t" --size "$size" --dtype f64,c64 --case "$filter" > "$stem.txt" 2> "$stem.guard"
                    else
                        env TCBENCH_PARTITION="$grid" bash benchmarks/scripts/pinned.sh "$cpus" -- "$bin" run --threads "$t" --size "$size" --dtype f64,c64 --case "$filter" --engines packed --reps 5 --csv "$stem.pending.csv" > "$stem.txt" 2> "$stem.guard"
                        mv "$stem.pending.csv" "$stem.csv"
                    fi
                done
            done
        done
    done
done
