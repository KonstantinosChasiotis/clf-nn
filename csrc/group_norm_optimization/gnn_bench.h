#ifndef GROUPNORM_BENCH_H
#define GROUPNORM_BENCH_H

#include "gnn_setup.h"
#include <stdint.h>

// signature of a groupnorm function
typedef void (*groupnorm_func)(
    const float *x,
    int B,
    int C,
    int D,
    int I,
    int num_groups,
    bool running,
    float *running_mean,
    float *running_cov,
    bool scaling,
    const float *weight,
    const float *bias,
    bool training,
    float momentum,
    float eps,
    float *x_norm
);

void register_groupnorm_functions();
void add_groupnorm_function(groupnorm_func f, const char* name);
int  get_registered_count();
const char* get_registered_name(int idx);
groupnorm_func get_registered_function(int idx);

double benchmark_groupnorm_by_index(int idx, int B);

void run_all_groupnorm_benchmarks(int B);

#endif // GROUPNORM_BENCH_H