# Benchmarking Harness

This folder provides a single, self-contained C file (bench_forward.c) for micro-benchmarking multiple Clifford-algebra layers (currently Linear and GroupNorm) using a variety of timing backends (RDTSC, VCT, PMU, clock(), gettimeofday(), Windows timers).

## How it works

1. Mode dispatch in main()  
   - First argument is the mode:  
     ``` 
     ./bench_main linear ...  
     ./bench_main groupnorm ...  
     ```  
2. Setup / Compute / Cleanup  
   - For each mode we call:  
     ```  
     setup_linear(...); compute = compute_linear; cleanup_linear();  
     // or  
     setup_groupnorm(...); compute = compute_groupnorm; cleanup_groupnorm();  
     ```  
   - `compute_linear()` / `compute_groupnorm()` ignore their dummy parameters and invoke the C-layer forward pass on global buffers.  
3. Timers  
   - `rdtsc()` (x86_64 RDTSC + CPUID)  
   - `rdvct()` (ARMv8 virtual counter)  
   - `rdpmu()` (ARM PMU via kperf, `-DPMU`)  
   - `c_clock()` (`clock()` from `<time.h>`)  
   - `timeofday()` (`gettimeofday()`)  
   - Windows timers (`GetTickCount()`, `QueryPerformanceCounter()`)

Each timer wraps the same `compute()` pointer and includes a calibration loop (`#ifdef CALIBRATE`) to scale up to the desired cycle count (`CYCLES_REQUIRED`).

## Building

# benchmarks/Makefile
```
INCLUDE = -I../csrc -I../csrc/include
CFLAGS  = -O3 -DBUILD_MAIN -DPMU $(INCLUDE)
LDFLAGS = -framework Accelerate

SRCS    = bench_forward.c \
          ../csrc/clifford_linear.c \
          ../csrc/clifford_groupnorm.c

all: bench_main

bench_main: $(SRCS)
	gcc $(CFLAGS) $(SRCS) $(LDFLAGS) -o bench_main

clean:
	rm -f bench_main
```


## Usage

```
cd benchmarks
make                # builds ./bench_main
```

Linear layer: B=32, C=8, dim=3 (8 blades), O=16 out-channels
```
./bench_main linear 32 8 3 16
```

GroupNorm: B=32, C=16, Dflat=1, I=8 blades, G=4 groups
```
./bench_main groupnorm 32 16 1 8 4
```

Each run prints timings for all enabled timers in a readable format.

## Adding a new layer

1. Include its header in `bench_forward.c`:  
   ```
   #include "clifford_conv.h"
   ```  
2. Define globals:  
   ```
   static CliffordConv *L_conv;
   static float *conv_x, *conv_out;
   static int conv_B, conv_Cin, conv_H, conv_W, conv_k;
   ```  
3. Implement three functions:  
   ```
   void setup_conv(int B,int Cin,int H,intW,int k,...) { /*...*/ }
   void compute_conv(double A[],double B[],double C[],int n) { /*...*/ }
   void cleanup_conv() { /*...*/ }
   ```  
4. Extend the `main()` dispatcher:  
   ```
   else if (strcmp(mode,"conv")==0) {
     setup_conv(...);
     compute = compute_conv;
   }
   ```
5. Link the new source via `Makefile`:  
   ```
   SRCS = bench_forward.c \
          ../csrc/clifford_linear.c \
          ../csrc/clifford_groupnorm.c \
          ../csrc/clifford_conv.c
   ```  
6. Rebuild & run:  
   ```
   make clean && make
   ./bench_main conv B Cin H W k
   ```  

This harness centralizes all your low-overhead timers so you can quickly compare performance across any Clifford-algebra–based layer.  
