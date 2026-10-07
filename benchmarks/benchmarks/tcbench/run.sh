#!/usr/bin/env bash
# Fixed native three-provider protocol; see .agents/skills/tprims-benchmark.
# Run sequentially, never beside another benchmark/build.
# Usage: run.sh OUT CPU1 CPU4 CPU8 CPU12
set -euo pipefail
[[ $# == 5 ]] || { echo 'usage: run.sh OUT CPU1 CPU4 CPU8 CPU12' >&2; exit 2; }
script=$(realpath "$0")
root=$(cd "$(dirname "$script")/../../.." && pwd)
out=$(realpath -m "$1"); shift
mkdir -p "$out"
cd "$root"
: "${TBLIS_ROOT:?set TBLIS_ROOT to the native Release TBLIS install}"
export RUSTFLAGS='-C target-cpu=native'
export LD_LIBRARY_PATH="$TBLIS_ROOT/lib:$TBLIS_ROOT/lib64:${LD_LIBRARY_PATH:-}"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-8}
# A saved baseline binary avoids rebuilding it from candidate sources.
if [[ -z ${TCBENCH_BIN:-} ]]; then
    cargo build --release --locked -p tprims-bench --bin tcbench --features upstream,tblis
fi
bin="${TCBENCH_BIN:-${CARGO_TARGET_DIR:-$root/target}/release/tcbench}"
{
    date -Iseconds
    git rev-parse HEAD
    git status --short
    rustc -Vv
    printf 'RUSTFLAGS=%s\nprofile=release\nthreads=1,4,8,12\ncpus=%s;%s;%s;%s\nTBLIS_ROOT=%s\n' "$RUSTFLAGS" "$1" "$2" "$3" "$4" "$TBLIS_ROOT"
    sha256sum "$bin" Cargo.lock "$script"
    lscpu
    lscpu -e=CPU,CORE,SOCKET,CACHE
    ps -eo pid,user,psr,%cpu,comm --sort=-%cpu | head
    "$bin" info
} > "$out/manifest.txt"
git diff > "$out/source.patch"
pinned="$root/benchmarks/scripts/pinned.sh"
# Complete correctness gate at both sizes and all budgets, before any timing.
for size in 1 16; do
    i=1
    for threads in 1 4 8 12; do
        cpus=${!i}; i=$((i+1))
        bash "$pinned" "$cpus" -- "$bin" verify --size "$size" --dtype f64,c64 --threads "$threads" > "$out/verify-${size}m-${threads}t.txt" 2> "$out/verify-${size}m-${threads}t.guard"
    done
done
# Same complete suite twice, minutes apart. Only promote CSVs from valid runs.
for sample in A A2; do
    for size in 1 16; do
        i=1
        for threads in 1 4 8 12; do
            cpus=${!i}; i=$((i+1))
            stem="$out/${sample}-${size}m-${threads}t"
            bash "$pinned" "$cpus" -- "$bin" run --size "$size" --dtype f64,c64 --threads "$threads" --reps 5 --engines plan,packed,upstream,tblis --csv "$stem.pending.csv" > "$stem.txt" 2> "$stem.guard"
            mv "$stem.pending.csv" "$stem.csv"
        done
    done
done
