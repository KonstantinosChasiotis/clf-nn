#!/bin/bash
set -e

echo "Compiling…"

gcc -O3 -march=native -std=c99 -Wall -Wno-unused-function \
    -DTRACEFLOPS \
    main.c \
    linear_bench.c \
    linear_impls.c \
    linear_setup.c \
    ../clifford_linear.c \
    ../flops.c \
    -o linear_benchmark

echo "Done."

if [ $# -lt 1 ]; then
  echo "Usage: $0 <mode: 0|1>"
  exit 1
fi

MODE=$1
DIM=2
CHANNELS=16
BATCHES=(16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536)
CSV=baseline_perfplot_linear.csv

if [ "$MODE" = "0" ]; then
  echo "Fixed-input benchmark:"
  ./linear_benchmark 0 $DIM $CHANNELS 32
  exit 0
fi

if [ "$MODE" != "1" ]; then
  echo "Invalid mode. Use 0 or 1."
  exit 1
fi

# mode == 1:
echo "Gathering function count…"
FUNC_COUNT=$(./linear_benchmark --count)
echo "→ Found ${FUNC_COUNT} functions."

# start fresh
> "$CSV"

for idx in $(seq 0 $((FUNC_COUNT-1))); do
  NAME=$(./linear_benchmark --name $idx)
  echo -e "\n${NAME}"    >> "$CSV"
  echo "batch,cycles,flops,bytes_read,bytes_written" >> "$CSV"

  for B in "${BATCHES[@]}"; do
    printf "  %-15s batch=%5d … " "$NAME" "$B"
    ./linear_benchmark 1 $DIM $CHANNELS $B $idx >> "$CSV"
    echo "ok"
  done
done

echo "All done. Results in $CSV"

echo "Plotting all functions…"
python3 ../../benchmarks/plot_performance.py baseline_perfplot_linear.csv \
     "Clifford Linear Performance" perf_plot.png
echo "Saved plot to perf_plot.png"

