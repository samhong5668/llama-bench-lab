#!/bin/bash
# Linux counterpart of the Windows comparison. llama.cpp publishes no Linux CUDA release
# binary, so the only two CUDA options on Linux are the llama.app prebuilt and a local build -
# which is exactly what this compares.
#
# Same discipline as the Windows runs: warm every binary once before taking a number, then
# alternate them so neither is systematically penalised by running later.
#
#   scripts/bench_linux.sh <model.gguf> [rounds]

set -u
MODEL="${1:?usage: bench_linux.sh <model.gguf> [rounds]}"
ROUNDS="${2:-4}"

PREBUILT="${LLAMA_PREBUILT:-$HOME/.local/bin/llama}"
LOCAL="${LLAMA_LOCAL:-$HOME/llama-lab/build/bin/llama-bench}"

for b in "$PREBUILT" "$LOCAL"; do
    [ -x "$b" ] || { echo "not executable: $b" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES=0
ARGS="-p 0 -n 128 -ngl 99 -ncmoe 10 -t 10 -r 10"

ts() { grep nemotron | awk -F'|' '{gsub(/ /,"",$(NF-1)); print $(NF-1)}'; }

echo "prebuilt: $("$PREBUILT" --version 2>&1 | head -1)"
echo "local:    $("$LOCAL" --version 2>&1 | grep -oE 'b?[0-9]+ \(.*\)' | head -1)"
echo

echo "warming each binary once (numbers from a binary's first run are not comparable)"
"$PREBUILT" bench -m "$MODEL" -p 0 -n 16 -ngl 99 -ncmoe 10 -r 1 -o md >/dev/null 2>&1
"$LOCAL"          -m "$MODEL" -p 0 -n 16 -ngl 99 -ncmoe 10 -r 1 -o md >/dev/null 2>&1
echo

for r in $(seq 1 "$ROUNDS"); do
    printf "round%s  prebuilt " "$r"
    printf "%-16s" "$("$PREBUILT" bench -m "$MODEL" $ARGS -o md 2>/dev/null | ts)"
    printf "  local "
    "$LOCAL" -m "$MODEL" $ARGS -o md 2>/dev/null | ts
done
