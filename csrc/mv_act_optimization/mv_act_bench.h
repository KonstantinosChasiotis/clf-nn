#ifndef ACTIVATION_BENCH_H
#define ACTIVATION_BENCH_H

#include "mv_act_setup.h"
#include <stdint.h>

typedef void (*activation_func)(
    const float *x,
    int           B,
    int           C,
    int           I_in,
    int           I_full,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
);

void register_activation_functions();
void add_activation_function(activation_func f, const char* name);
int  get_registered_count();
const char* get_registered_name(int idx);
activation_func get_registered_function(int idx);

double benchmark_act_by_index(int idx, int B);

void run_all_act_benchmarks(int B);

#endif // ACTIVATION_BENCH_H