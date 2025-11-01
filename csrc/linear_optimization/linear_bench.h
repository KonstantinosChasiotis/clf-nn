#ifndef LINEAR_BENCH_H
#define LINEAR_BENCH_H

#include "../clifford_linear.h"

typedef void (*linear_func)(const CliffordLinear* L, const float* x, int B, float* out);

void register_linear_functions();  
void add_linear_function(linear_func f, const char* name);
void run_all_linear_benchmarks(CliffordLinear* L, const float* x, float* out_ref, int B);
int get_registered_count();
linear_func get_registered_function(int idx);
const char* get_registered_name(int i);
double benchmark_linear_by_index(int i, CliffordLinear* L, const float* x, float* out, int B);

#endif
