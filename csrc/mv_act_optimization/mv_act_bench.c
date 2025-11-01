#include "mv_act_bench.h"
#include "mv_act_setup.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef PMU
  #include "../kperf.h"
#endif
#ifdef __x86_64__
  #include "../tsc_x86.h"
#endif

#define MAX_FUNCS       32
#define EPSILON         1e-5f
#define NUM_RUNS        10
#define CYCLES_REQUIRED 100000000.0
#define CALIBRATE
#define REP             10

static activation_func registered[MAX_FUNCS];
static const char *names[MAX_FUNCS];
static int func_count = 0;

void add_activation_function(activation_func f, const char* name) {
    if (func_count >= MAX_FUNCS) {
        fprintf(stderr, "Max activation functions reached!\n");
        exit(1);
    }
    registered[func_count] = f;
    names[func_count]       = name;
    func_count++;
}

int get_registered_count() {
    return func_count;
}

const char* get_registered_name(int idx) {
    return names[idx];
}

activation_func get_registered_function(int idx) {
    return registered[idx];
}

static double __benchmark_act(activation_func f, int B) {
    int i, num_runs = NUM_RUNS;
    double cycles = 0.0;

#ifdef __x86_64__
    myInt64 start;
  #ifdef CALIBRATE
    while (num_runs < (1 << 14)) {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i) {
            f(
              act_x,
              act_B, act_C,
              act_I_in, act_I_full,
              act_input_blades,
              act_weight, act_K,
              act_kernel_blades,
              act_bias,
              act_out
            );
        }
        cycles = (double)stop_tsc(start);
        if (cycles >= CYCLES_REQUIRED) break;
        num_runs <<= 1;
    }
  #endif

    cycles = 0.0;
    for (int rep = 0; rep < REP; ++rep) {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i) {
            f(
              act_x,
              act_B, act_C,
              act_I_in, act_I_full,
              act_input_blades,
              act_weight, act_K,
              act_kernel_blades,
              act_bias,
              act_out
            );
        }
        cycles += (double)stop_tsc(start) / num_runs;
    }
    return cycles / REP;
#endif

#ifdef PMU
    kperf_init();
    struct performance_counters startperf, endperf;
  #ifdef CALIBRATE
    while (num_runs < (1 << 14)) {
        startperf = kperf_get_counters();
        for (i = 0; i < num_runs; ++i) {
            f(
              act_x,
              act_B, act_C,
              act_I_in, act_I_full,
              act_input_blades,
              act_weight, act_K,
              act_kernel_blades,
              act_bias,
              act_out
            );
        }
        endperf = kperf_get_counters();
        double tot = (double)(endperf.cycles - startperf.cycles);
        if (tot >= CYCLES_REQUIRED) break;
        num_runs <<= 1;
    }
  #endif

    cycles = 0.0;
    for (int rep = 0; rep < REP; ++rep) {
        startperf = kperf_get_counters();
        for (i = 0; i < num_runs; ++i) {
            f(
              act_x,
              act_B, act_C,
              act_I_in, act_I_full,
              act_input_blades,
              act_weight, act_K,
              act_kernel_blades,
              act_bias,
              act_out
            );
        }
        endperf = kperf_get_counters();
        cycles += (double)(endperf.cycles - startperf.cycles) / num_runs;
    }
    return cycles / REP;
#endif

    return -1.0;
}

double benchmark_act_by_index(int idx, int B) {
    return __benchmark_act(registered[idx], B);
}

void run_all_act_benchmarks(int B) {
    int cnt = get_registered_count();
    if (cnt == 0) {
        fprintf(stderr, "No activation functions registered!\n");
        return;
    }

    size_t out_len = (size_t)act_B * act_C * act_I_in;
    float *ref_out = malloc(out_len * sizeof(float));

    registered[0](
      act_x,
      act_B, act_C,
      act_I_in, act_I_full,
      act_input_blades,
      act_weight, act_K,
      act_kernel_blades,
      act_bias,
      act_out
    );
    memcpy(ref_out, act_out, out_len * sizeof(float));

    double baseline_cycles = __benchmark_act(registered[0], B);
    printf("Baseline %-20s: recorded as reference\n", names[0]);

    for (int i = 0; i < cnt; ++i) {
        registered[i](
          act_x,
          act_B, act_C,
          act_I_in, act_I_full,
          act_input_blades,
          act_weight, act_K,
          act_kernel_blades,
          act_bias,
          act_out
        );
        int ok = 1;
        for (size_t j = 0; j < out_len; ++j) {
            if (fabsf(act_out[j] - ref_out[j]) > EPSILON) {
                ok = 0; break;
            }
        }

        double cycles = __benchmark_act(registered[i], B);
        double speedup = baseline_cycles / cycles;

        printf(
          "Function %-20s: %8.2f cycles [%s] | Speedup: %.2fx\n",
          names[i],
          cycles,
          ok ? "OK" : "ERROR",
          speedup
        );
    }

    free(ref_out);
}
