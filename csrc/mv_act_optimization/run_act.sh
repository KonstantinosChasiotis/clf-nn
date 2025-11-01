#!/usr/bin/env bash
set -e

echo "Compiling activation benchmark…"
gcc -O3 -march=native -std=c99 -Wall -Wno-unused-function -DTRACEFLOPS \
    main.c \
    mv_act_bench.c mv_act_impls.c mv_act_setup.c \
    ../mv_act.c ../flops.c \
    -lm \
    -o activation_benchmark
echo "Compiled."

if [ $# -lt 1 ]; then
  echo "Usage: $0 <mode: 0|1>"
  exit 1
fi

MODE=$1
C=16       # example in/out channels
I_IN=8     # number of input blades
I_FULL=16  # full algebra blades
K=4        # kernel size
BATCHES=(16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
CSV=perfplot_activation.csv

if [ "$MODE" = "0" ]; then
  echo "Fixed-input activation benchmark (batch=32)…"
  ./activation_benchmark 0 $C $I_IN $I_FULL $K 32
  exit 0
fi

if [ "$MODE" != "1" ]; then
  echo "Invalid mode. Use 0 or 1."
  exit 1
fi

echo "Gathering function count…"
FUNC_COUNT=$(./activation_benchmark --count)
echo "→ Found $FUNC_COUNT functions."

# start fresh
> "$CSV"

for idx in $(seq 0 $((FUNC_COUNT-1))); do
  NAME=$(./activation_benchmark --name $idx)

  # header for this function
  echo "$NAME"              >> "$CSV"
  echo "batch,cycles,flops,bytes_read,bytes_written" >> "$CSV"

  for B in "${BATCHES[@]}"; do
    # console progress
    printf "  %-15s batch=%5d … " "$NAME" "$B"
    # run and capture the single-line CSV output from main.c
    ./activation_benchmark 1 $C $I_IN $I_FULL $K $B $idx >> "$CSV"
    echo "ok"
  done

  # blank line between functions
  echo >> "$CSV"
done

echo "All done. Results in $CSV"

echo "Plotting all functions…"
python3 ../../benchmarks/plot_performance.py "$CSV" \
    "Multivector Activation Performance" activation_perf.png
echo "Saved plot to activation_perf.png"
