#ifndef CLIFFORD_GROUPNORM_H
#define CLIFFORD_GROUPNORM_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>
#include <math.h>

void clifford_groupnorm( //wrapper (if you want to understand how to benchmark groupnorm, read this definition)
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float *running_mean_orig,
    float *running_cov_orig,
    bool scaling,
    const float *weight_orig,
    const float *bias_orig,
    bool training,
    float momentum,
    float eps,
    float *x_norm);

void clifford_groupnorm_alloctemp(
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float **running_mean_temp,
    float **running_cov_temp,
    bool scaling,
    float **weight_temp,
    float **weight_perm_temp,
    float **bias_temp,
    float **mean_temp,
    float **X_temp,
    float **cov_temp,
    float **max_temp);

void clifford_groupnorm_freetemp(
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float **running_mean_temp,
    float **running_cov_temp,
    bool scaling,
    float **weight_temp,
    float **weight_perm_temp,
    float **bias_temp,
    float **mean_temp,
    float **X_temp,
    float **cov_temp,
    float **max_temp);

void clifford_groupnorm_baseline(
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float *running_mean_orig,
    float *running_mean_temp,
    float *running_cov_orig,
    float *running_cov_temp,
    float *mean_temp,
    float *X_temp,
    float *cov_temp,
    float *max_temp,
    bool scaling,
    const float *weight_orig,
    float *weight_temp,
    float *weight_perm_temp,
    const float *bias_orig,
    float *bias_temp,
    bool training,
    float momentum,
    float eps,
    float *x_norm);

#endif // CLIFFORD_GROUPNORM_H
