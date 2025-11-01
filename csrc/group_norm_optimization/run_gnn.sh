#!/usr/bin/env bash
set -e

echo "Compiling group‐norm benchmark…"
gcc -O3 -march=native -std=c99 -Wall -Wno-unused-function \
    -DTRACEFLOPS \
    main.c \
    gnn_bench.c gnn_impls.c gnn_setup.c \
    ../clifford_groupnorm.c ../flops.c \
    -o groupnorm_benchmark -lm
echo "Compiled."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <mode: 0|1>"
    exit 1
fi

MODE=$1

C=16
D=8
I=8
NUM_GROUPS=4

BATCHES=(16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
CSV=perfplot_groupnorm.csv

if [ "$MODE" = "0" ]; then
  echo "Fixed‐input group‐norm benchmark (batch=32)…"
  ./groupnorm_benchmark 0 $C $D $I $NUM_GROUPS 32
  exit 0
fi

if [ "$MODE" != "1" ]; then
  echo "Invalid mode. Use 0 or 1."
  exit 1
fi

echo "Gathering function count…"
FUNC_COUNT=$(./groupnorm_benchmark --count)
echo "→ Found $FUNC_COUNT functions."

> "$CSV"

for idx in $(seq 0 $((FUNC_COUNT-1))); do
  NAME=$(./groupnorm_benchmark --name $idx)
  echo "$NAME"              >> "$CSV"
  echo "batch,cycles,flops,bytes_read,bytes_written" >> "$CSV"

  for B in "${BATCHES[@]}"; do
    printf "  %-15s batch=%5d … " "$NAME" "$B"
    ./groupnorm_benchmark 1 $C $D $I $NUM_GROUPS $B $idx >> "$CSV"
    echo "ok"
  done

  echo >> "$CSV"
done

echo "All done. Results in $CSV"

echo "Plotting all functions…"
python3 ../../benchmarks/plot_performance.py "$CSV" \
     "Clifford Group Norm Perfomance" groupnorm_plot.png
echo "Saved plot to groupnorm_plot.png"
