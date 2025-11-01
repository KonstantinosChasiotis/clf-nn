#include "linear_bench.h"
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

#define MAX_FUNCS 32
#define EPSILON 1e-5
#define CYCLES_REQUIRED 1e8
#define NUM_RUNS 10
#define CALIBRATE
#define REP 10

typedef struct {
    linear_func func;
    const char* name;
} LinearImpl;

static LinearImpl registered[MAX_FUNCS];
static int count = 0;

static double benchmark(linear_func f, CliffordLinear* L, const float* x, float* out, int B) {
    int i, num_runs;
    double cycles;

#ifdef __x86_64__
    myInt64 start;
    num_runs = NUM_RUNS;

    #ifdef CALIBRATE
    while (num_runs < (1 << 14)) {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i) {
            f(L, x, B, out);
        }
        cycles = (double)stop_tsc(start);
        if (cycles >= CYCLES_REQUIRED)
            break;
        num_runs *= 2;
    }
    #endif

    cycles = 0;
    for (int rep = 0; rep < REP; ++rep) {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i) {
            f(L, x, B, out);
        }
        cycles += (double)stop_tsc(start) / num_runs;
    }
    return cycles / REP;
#endif

#ifdef PMU
    kperf_init();
    struct performance_counters startperf, endperf;
    num_runs = NUM_RUNS;

    #ifdef CALIBRATE
    while (num_runs < (1 << 14)) {
        startperf = kperf_get_counters();
        for (i = 0; i < num_runs; ++i) {
            f(L, x, B, out);
        }
        endperf = kperf_get_counters();
        double total = (double)(endperf.cycles - startperf.cycles);
        if (total >= CYCLES_REQUIRED)
            break;
        num_runs *= 2;
    }
    #endif

    cycles = 0;
    for (int rep = 0; rep < REP; ++rep) {
        startperf = kperf_get_counters();
        for (i = 0; i < num_runs; ++i) {
            f(L, x, B, out);
        }
        endperf = kperf_get_counters();
        cycles += (double)(endperf.cycles - startperf.cycles) / num_runs;
    }
    return cycles / REP;
#endif
}

void add_linear_function(linear_func f, const char* name) {
    if (count >= MAX_FUNCS) {
        fprintf(stderr, "Max functions reached!\n");
        exit(1);
    }
    registered[count].func = f;
    registered[count].name = name;
    count++;
}

int outputs_match(const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > EPSILON) return 0;
    }
    return 1;
}

int get_registered_count() {
    return count;
}

const char* get_registered_name(int i) {
    return registered[i].name;
}

linear_func get_registered_function(int idx) {
    if (idx < 0 || idx >= count) return NULL;
    return registered[idx].func;
}

double benchmark_linear_by_index(int i, CliffordLinear* L, const float* x, float* out, int B) {
    return benchmark(registered[i].func, L, x, out, B);
}


void run_all_linear_benchmarks(CliffordLinear* L, const float* x, float* out_ref, int B) {

    int out_size = B * L->sig.n_blades * L->out_channels;
    float* out_temp = malloc(sizeof(float) * out_size);

    double baseline_cycles = benchmark(registered[0].func, L, x, out_ref, B);
    printf("Baseline %-20s: recorded as reference\n", registered[0].name);

    for (int i = 0; i < count; i++) {
        double cycles = benchmark(registered[i].func, L, x, out_temp, B);
        int ok = outputs_match(out_temp, out_ref, out_size);
        double speedup = baseline_cycles / cycles;
    
        printf("Function %-20s: %8.2f cycles [%s] | Speedup: %.2fx\n",
               registered[i].name, cycles, ok ? "OK" : "ERROR", speedup);
    }

    free(out_temp);
}
