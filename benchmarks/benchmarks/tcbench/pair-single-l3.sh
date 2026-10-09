#!/usr/bin/env bash
# Explicit single-L3 follow-up; does NOT replace failed full 12T experiments.
# Usage: pair-single-l3.sh OUT BASELINE CANDIDATE CPU1 CPU4 CPU8
set -euo pipefail
[[ $# == 6 ]] || { echo 'usage: pair-single-l3.sh OUT BASELINE CANDIDATE CPU1 CPU4 CPU8' >&2; exit 2; }
root=$(cd "$(dirname "$0")/../../.." && pwd)
out=$(realpath -m "$1"); baseline=$(realpath "$2"); candidate=$(realpath "$3"); shift 3
cd "$root"
mkdir -p "$out/baseline" "$out/candidate"
{
    date -Iseconds
    git rev-parse HEAD
    rustc -Vv
    sha256sum "$baseline" "$candidate"
    lscpu -e=CPU,CORE,SOCKET,CACHE
    printf 'scope=all49 cases,f64/c64,1/16MiB,1/4/8T; NOT a 12T result\ncpus=%s;%s;%s\n' "$1" "$2" "$3"
} > "$out/manifest.txt"
git diff > "$out/candidate.patch"
for arm in baseline candidate; do
    bin=${!arm}
    i=1
    for t in 1 4 8; do
        cpus=${!i}; i=$((i+1))
        taskset -c "$cpus" "$bin" info --threads "$t" > "$out/$arm/info-${t}t.txt"
    done
    for size in 1 16; do
        i=1
        for t in 1 4 8; do
            cpus=${!i}; i=$((i+1))
            bash benchmarks/scripts/pinned.sh "$cpus" -- "$bin" verify --threads "$t" --size "$size" --dtype f64,c64 > "$out/$arm/verify-${size}m-${t}t.txt" 2> "$out/$arm/verify-${size}m-${t}t.guard"
        done
    done
done
for sample in A A2; do
    arms='baseline candidate'; [[ $sample == A2 ]] && arms='candidate baseline'
    for size in 1 16; do
        i=1
        for t in 1 4 8; do
            cpus=${!i}; i=$((i+1))
            for arm in $arms; do
                bin=${!arm}; stem="$out/$arm/${sample}-${size}m-${t}t"
                bash benchmarks/scripts/pinned.sh "$cpus" -- "$bin" run --threads "$t" --size "$size" --dtype f64,c64 --reps 5 --engines plan,packed,upstream --csv "$stem.pending.csv" > "$stem.txt" 2> "$stem.guard"
                mv "$stem.pending.csv" "$stem.csv"
            done
        done
    done
done
for arm in baseline candidate; do
    python3 benchmarks/benchmarks/tcbench/summarize.py "$out/$arm" 1,4,8 > "$out/$arm/summary.md"
done
python3 benchmarks/benchmarks/tcbench/compare.py "$out/baseline" "$out/candidate" 1,4,8 > "$out/comparison.txt"
