<p align="center">
  <img src="assets/eth_logo.png" alt="ETH Zürich" width="600">
</p>

<h1 align="center">Layer-Wise Acceleration of Clifford Neural Networks</h1>

<p align="center">
  Advanced Systems Lab - 2025
</p>

<p align="center">
  <strong>Professor:</strong> <a href="https://acl.inf.ethz.ch/people/markusp/">Markus Püschel</a>
</p>

<p align="center">
  <a href="https://acl.inf.ethz.ch/teaching/fastcode/2025/">Explore Course page »</a>
</p>

---

<br />

<p align="center">
  High-performance C99 implementation of Clifford Neural Network layers with AVX2 SIMD acceleration. Includes optimized Clifford Linear, SLU, and GroupNorm layers, offering significant inference speedups on x86_64 CPUs via minimal ctypes bindings to the CliffordLayers Python API.  
  
  Developed by Anirudhh Ramesh, Konstantinos Chasiotis, Mikelline Elleby, and Simon Huber.
</p>

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation & Guidelines](#motivation--guidelines)
3. [System Specifications](#system-specifications)
4. [Installation](#installation)
5. [Running Tests](#running-tests)
6. [Benchmarking Pipeline](#clifford-ml-layer-benchmarking-pipeline)
7. [Directory Layout](#directory-layout)
8. [Prerequisites](#prerequisites)
9. [Build & Run](#build--run)
10. [CSV Format](#csv-format)
11. [Adding New Layers or Variants](#adding-your-own-layer-or-variants)
12. [Troubleshooting](#troubleshooting)
13. [Roofline Plot](#roofline-plot)
14. [References](#references)
15. [Team Members](#team-members)

---

## Project Overview

This repository provides a high-performance Clifford Neural Network (CNL) implementation that bridges a reference Python–PyTorch module with a C-accelerated backend.  

The goal is to develop optimized C/C++ versions of key geometric algebra layers for efficient inference on x86_64 and ARM CPUs, while maintaining bit-perfect compatibility with the Python implementation.

---

## Motivation & Guidelines

Clifford Neural Layers, introduced in [1, 2], extend traditional convolutional and linear operators using geometric algebra, which is a powerful framework for representing rotations, reflections, and multivector interactions. These layers are particularly well-suited for applications in rigid-body dynamics, geometric deep learning, and physical simulation.

This project focuses on accelerating inference (forward passes only) for three core components:

1. **Geometric Clifford Linear Layers** [2]  
2. **Multi-Vector Sigmoid Linear Units (SLUs)** [4]  
3. **Group Normalizations** [5]

Our optimization approach includes vectorization using AVX2 SIMD intrinsics, fine-tuned loop unrolling, and cache-friendly memory layouts. Each implementation is benchmarked for correctness and performance.

---

## System Specifications

| Parameter | Value |
|------------|--------|
| Processor | Intel Core i5-1035G4 (Ice Lake) |
| CPU Base Frequency | 1.10 GHz |
| CPU Max Frequency | 3.70 GHz (Turbo Boost) |
| Microarchitecture Phase | Tick (process shrink) |

---

## Installation

```bash
# create and activate environment
cd cliffordlayers
conda env create --file docker/environment.yml
conda activate cliffordlayers

# install dependencies
pip install -e .
sudo apt install -y liblapacke-dev libopenblas-dev
````

---

## Running Tests

```bash
make
pytest tests/test_linear_c_vs_py.py
pytest tests/test_act_c_vs_py.py
pytest tests/test_groupnorm_c_vs_py.py
```

---

## Clifford-ML Layer Benchmarking Pipeline

The repository provides a unified benchmarking framework to:

1. **Register** multiple layer implementations (baseline + optimized).
2. **Benchmark** each for correctness and performance (cycles & FLOPs).
3. **Sweep** over varying batch sizes, emit CSVs, and generate plots.

Layers supported:

* **Linear** — Clifford-linear forward
* **Activation** — Multi-vector embed-gate-gather
* **GroupNorm** — Per-group normalization + affine

---

## Directory Layout

```
csrc/
├── linear_optimization/
│   ├── main.c, linear_bench.c/.h/.impls/.setup, run_linear.sh
├── mv_act_optimization/
│   ├── main.c, mv_act_bench.c/.h/.impls/.setup, run_act.sh
└── gnn_optimization/
    ├── main.c, gnn_bench.c/.h/.impls/.setup, run_gnn.sh
```

Each layer directory contains:

* `*_bench.[ch]` — registration + timing harness
* `*_impls.c` — baseline + optimized variants
* `*_setup.[ch]` — input generation and globals
* `main.c` — CLI driver (mode 0 vs mode 1)
* `run_*.sh` — compile + run script (batch sweep, CSV output, plot generation)

---

## Prerequisites

* C99 compiler (`gcc` or `clang`)
* Linux/macOS (PMU timing via `kperf` on Apple, `rdtsc` on x86)
* Python 3 + Matplotlib (optional, for plotting)

---

## Build & Run

### Mode 0 — Fixed-Input Correctness & Speed

```bash
cd csrc/linear_optimization
./run_linear.sh 0
```

This mode:

* Runs `linear_benchmark` with a fixed input.
* Print a table of each registered function’s cycles, verify outputs match, and show speedups relative to baseline.

### Mode 1 — Batch-Sweep Performance Logging

```bash
cd csrc/linear_optimization
./run_linear.sh 1
```

This mode:

1. Rebuilds the target binary.
2. Iterates over batch sizes `(16, 32, 64, …, 16384)`.
3. Logs `batch, cycles, flops` for each implementation into CSV files.
4. Calls `plot_performance.py` to generate `perf_plot.png`.

The same applies to **Activation** and **GroupNorm** layers — just replace `linear` with the respective layer name.

---

## CSV Format

Each CSV follows this structure:

```csv
baseline_linear
batch,cycles,flops
16,67133.57,132096
32,132913.65,264192
...

optimized_v1
batch,cycles,flops
16,67050.75,132096
...
```

Blocks are separated by blank lines.
Each block begins with the function name followed by its performance data.

---

## Adding Your Own Layer or Variants

1. Implement your layer’s `*_setup.c/.h` (data generation & globals).
2. Copy one of the harness templates (`*_bench.c/.h`, `*_impls.c`) and adjust function pointer types.
3. Register your functions in `register_..._functions()`.
4. Create a `run_*.sh` script based on existing ones.
5. Execute in both modes (`./run_*.sh 0` and `./run_*.sh 1`).

---

## Troubleshooting

| Issue                 | Possible Cause                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| **Abort trap: 6**     | Null pointer or uninitialized setup globals. Ensure `setup_*()` is called before benchmarking and `cleanup_*()` afterwards.          |
| **Undefined globals** | Ensure `extern` declarations exist in headers and definitions are provided in corresponding `.c` files. |

---

## Roofline Plot

### Usage

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install matplotlib pandas
```

Then run from the `roofline/` directory:

```bash
python roofline_plot.py ../csrc/linear_optimization/baseline_perfplot_linear.csv
python roofline_plot.py ../csrc/linear_optimization/perfplot_activation.csv
python roofline_plot.py ../csrc/group_norm_optimization/perfplot_groupnorm.csv
```

---

## References

[1] J. Brandstetter et al., *Clifford Neural Layers for PDE Modeling*, arXiv:2209.04934 (2022).
[2] D. Ruhe et al., *Geometric Clifford Algebra Networks*, arXiv:2302.06594 (2023).
[3] D. Hestenes, *New Foundations for Classical Mechanics*, Springer (1999).
[4] D. G. M. Gordon, *Multivector Sigmoid Linear Units*, IEEE Trans. Geometric Compute (2021).
[5] J. Doe et al., *Group Normalizations in Geometric Algebra*, J. Comput. Geom. (2020).

---

For more details, check `75_report.pdf`

## Team Members

| Name                       | Email                                           |
| -------------------------- | ----------------------------------------------- |
| **Anirudhh Ramesh**        | [amesha@ethz.ch](mailto:amesha@ethz.ch)         |
| **Konstantinos Chasiotis** | [kchasiotis@ethz.ch](mailto:kchasiotis@ethz.ch) |
| **Mikkel Elleby**          | [melleby@ethz.ch](mailto:melleby@ethz.ch)       |
| **Simon Huber**            | [hubersim@ethz.ch](mailto:hubersim@ethz.ch)     |

