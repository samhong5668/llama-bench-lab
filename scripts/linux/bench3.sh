#!/bin/bash
# Three-way Linux comparison, alternating: llama.app vs a native build vs a -march=x86-64-v3
# build. Same discipline as everywhere else - warm each binary once, then alternate.
#
#   scripts/linux/bench3.sh <model.gguf> [rounds]
#
# Defaults measure the deployment workload (Nemotron with experts on the CPU, E-core pinned).
# To compare CPU backends instead, which is far less noisy, override BENCH_ARGS with a small
# model and -ngl 0 so all compute lands on the CPU:
#
#   BENCH_ARGS="-p 0 -n 64 -ngl 0 -t 6 -r 3" scripts/linux/bench3.sh qwen2.5-0.5b-*.gguf 5
#
# All three slots are overridable and hold any binary: LLAMA_APP, LLAMA_NATIVE, LLAMA_V3, with
# BENCH_LABELS renaming the columns. To compare three local builds instead of including
# llama.app:
#
#   LLAMA_APP=build-linux-native/bin/llama-bench #   LLAMA_NATIVE=build-linux-v3/bin/llama-bench #   LLAMA_V3=build-linux-v3vnni/bin/llama-bench #   BENCH_LABELS="native v3 v3vnni" BENCH_ARGS="-p 0 -n 64 -ngl 0 -t 6 -r 3" #   scripts/linux/bench3.sh <model.gguf> 5
set -u
MODEL="${1:?usage: bench3.sh <model.gguf> [rounds]}"
ROUNDS="${2:-4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

APP="${LLAMA_APP:-$HOME/.local/bin/llama}"
NATIVE="${LLAMA_NATIVE:-$REPO_ROOT/build-linux-native/bin/llama-bench}"
V3="${LLAMA_V3:-$REPO_ROOT/build-linux-v3/bin/llama-bench}"
LABELS="${BENCH_LABELS:-llama.app native v3}"

export CUDA_VISIBLE_DEVICES=0
# Pinned to the E-cores: the README shows this collapses run-to-run spread from ~12% to ~2.4%,
# which is what makes a few-percent difference between builds measurable at all.
ARGS="${BENCH_ARGS:--p 0 -n 128 -ngl 99 -ncmoe 10 -t 6 -C 0x3F0 --cpu-strict 1 -r 10}"

# Parse -o json rather than the md table: avg_ts is model-agnostic and needs no column counting.
ts() { grep -oE '"(avg|stddev)_ts": *[0-9.]+' | awk -F: '{ printf "%.2f%s", $2, (NR % 2 ? " +/- " : "") }'; }

for b in "$APP" "$NATIVE" "$V3"; do
    [ -x "$b" ] || { echo "not executable: $b" >&2; exit 1; }
done

# llama.app is a single multi-tool `llama` that takes a `bench` subcommand; a release or local
# build is `llama-bench` itself. Detected by name, so any slot can hold any of them.
run() {
    if [ "$(basename "$1")" = "llama" ]; then "$1" bench -m "$MODEL" $ARGS -o json 2>/dev/null
    else                                      "$1"       -m "$MODEL" $ARGS -o json 2>/dev/null
    fi
}

echo "args: $ARGS"
echo "warming each binary once, with the same arguments used for measurement"
for b in "$APP" "$NATIVE" "$V3"; do run "$b" >/dev/null; done
echo

set -- $LABELS
for r in $(seq 1 "$ROUNDS"); do
    printf "round%s" "$r"
    i=1
    for b in "$APP" "$NATIVE" "$V3"; do
        eval "label=\${$i}"
        printf "   %s %-18s" "$label" "$(run "$b" | ts)"
        i=$((i + 1))
    done
    echo
done
