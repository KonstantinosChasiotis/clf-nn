#include "gnn_bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef PMU
#include "../kperf.h"
#endif
#ifdef __x86_64__
#include "../tsc_x86.h"
#endif

#define MAX_FUNCS 32
#define EPSILON 1e-5f
#define NUM_RUNS 10
#define CYCLES_REQUIRED 1e8
#define CALIBRATE
#define REP 10

static groupnorm_func registered[MAX_FUNCS];
static const char *names[MAX_FUNCS];
static int count = 0;

static void clifford_groupnorm_zero_globals(
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    bool scaling
    ){

    //basic constants
    int dim0 = (B*C)/num_groups;
    //int group_size = C/num_groups;

    if (running){
        size_t running_mean_size = _I * dim0;
        memset(gn_running_mean_temp, 0, running_mean_size * sizeof(float));
        size_t running_cov_size = _I* _I * dim0;
        memset(gn_running_cov_temp, 0, running_cov_size * sizeof(float));
    }
    if (scaling){
        size_t weight_size = _I* _I * dim0;
        memset(gn_weight_temp, 0, weight_size * sizeof(float));
        size_t bias_size = _I * dim0;
        memset(gn_bias_temp, 0, bias_size * sizeof(float));
        memset(gn_weight_perm_temp, 0, weight_size * sizeof(float));
    }
    size_t x_size = dim0 * num_groups * D * _I;
    memset(gn_X_temp, 0, x_size * sizeof(float));
    size_t mean_size = dim0 * _I;
    memset(gn_mean_temp, 0, mean_size * sizeof(float));
    size_t cov_size = dim0*_I*_I;
    memset(gn_cov_temp, 0, cov_size * sizeof(float));
    size_t max_size = dim0;
    memset(gn_max_temp, 0, max_size * sizeof(float));
    return;
}

void add_groupnorm_function(groupnorm_func f, const char *name)
{
    if (count >= MAX_FUNCS)
    {
        fprintf(stderr, "Max functions reached!\n");
        exit(1);
    }
    registered[count] = f;
    names[count] = name;
    count++;
}

int get_registered_count() { return count; }
const char *get_registered_name(int i) { return names[i]; }
groupnorm_func get_registered_function(int i) { return registered[i]; }

static double benchmark_gn(groupnorm_func f, int B)
{
    int i, num_runs;
    double cycles = 0.0;

#ifdef __x86_64__
    myInt64 start;
    num_runs = NUM_RUNS;
#ifdef CALIBRATE
    while (num_runs < (1 << 14))
    {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i)
        {
            f(
                gn_x, gn_B, gn_C, gn_D, gn_I,
                gn_num_groups, gn_running,
                gn_running_mean, gn_running_cov,
                gn_scaling, gn_weight, gn_bias,
                gn_training, gn_momentum, gn_eps,
                gn_x_norm);
        }
        cycles = (double)stop_tsc(start);
        if (cycles >= CYCLES_REQUIRED)
            break;
        num_runs <<= 1;
    }
#endif
    cycles = 0.0;
    for (int rep = 0; rep < REP; ++rep)
    {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i)
        {
            f(
                gn_x, gn_B, gn_C, gn_D, gn_I,
                gn_num_groups, gn_running,
                gn_running_mean, gn_running_cov,
                gn_scaling, gn_weight, gn_bias,
                gn_training, gn_momentum, gn_eps,
                gn_x_norm);
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
    while (num_runs < (1 << 14))
    {
        startperf = kperf_get_counters();
        for (i = 0; i < num_runs; ++i)
        {
            f(
                gn_x, gn_B, gn_C, gn_D, gn_I,
                gn_num_groups, gn_running,
                gn_running_mean, gn_running_cov,
                gn_scaling, gn_weight, gn_bias,
                gn_training, gn_momentum, gn_eps,
                gn_x_norm);
        }
        endperf = kperf_get_counters();
        double tot = (double)(endperf.cycles - startperf.cycles);
        if (tot >= CYCLES_REQUIRED)
            break;
        num_runs <<= 1;
    }
#endif
    cycles = 0.0;
    for (int rep = 0; rep < REP; ++rep)
    {
        startperf = kperf_get_counters();
        for (i = 0; i < num_runs; ++i)
        {
            f(
                gn_x, gn_B, gn_C, gn_D, gn_I,
                gn_num_groups, gn_running,
                gn_running_mean, gn_running_cov,
                gn_scaling, gn_weight, gn_bias,
                gn_training, gn_momentum, gn_eps,
                gn_x_norm);
        }
        endperf = kperf_get_counters();
        cycles += (double)(endperf.cycles - startperf.cycles) / num_runs;
    }
    return cycles / REP;
#endif
}

double benchmark_groupnorm_by_index(int idx, int B)
{
    (void)B; 
    return benchmark_gn(registered[idx], B);
}

static int outputs_match(const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; ++i){
        if (fabsf(a[i] - b[i]) > EPSILON || 
            (isnan(a[i]) && !isnan(b[i]))||
            (!isnan(a[i]) && isnan(b[i]))
            ){
            printf("missmatch at: %ld , %f, %f\n", i, a[i], b[i]);
            return 0;
        }
    }
    return 1;
}

void run_all_groupnorm_benchmarks(int B)
{
    int cnt = get_registered_count();
    if (!cnt)
    {
        fprintf(stderr, "No functions registered!\n");
        return;
    }

    size_t len = (size_t)gn_B * gn_C * gn_D * gn_I;
    float *ref = malloc(len * sizeof(float));

    //clifford_groupnorm_zero_globals(gn_B, gn_C, gn_D, gn_I, gn_num_groups, gn_running, gn_scaling);
    registered[0](
        gn_x, gn_B, gn_C, gn_D, gn_I,
        gn_num_groups, gn_running,
        gn_running_mean, gn_running_cov,
        gn_scaling, gn_weight, gn_bias,
        gn_training, gn_momentum, gn_eps,
        gn_x_norm);
    memcpy(ref, gn_x_norm, len * sizeof(float));

    double base_cycles = benchmark_gn(registered[0], B);
    printf("Baseline %-20s: recorded as reference\n", names[0]);

    for (int i = 0; i < cnt; ++i)
    {
        //clifford_groupnorm_zero_globals(gn_B, gn_C, gn_D, gn_I, gn_num_groups, gn_running, gn_scaling);
        registered[i](
            gn_x, gn_B, gn_C, gn_D, gn_I,
            gn_num_groups, gn_running,
            gn_running_mean, gn_running_cov,
            gn_scaling, gn_weight, gn_bias,
            gn_training, gn_momentum, gn_eps,
            gn_x_norm);
        int ok = outputs_match(gn_x_norm, ref, len);

        double cycles = benchmark_gn(registered[i], B);
        double speed = base_cycles / cycles;
        printf("Function %-20s: %8.2f cycles [%s] | Speedup: %.2fx\n",
               names[i], cycles, ok ? "OK" : "ERROR", speed);
    }
    free(ref);
}
