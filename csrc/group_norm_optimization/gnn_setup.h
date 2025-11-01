#ifndef GNN_SETUP_H
#define GNN_SETUP_H

#include <stdbool.h>
#include <math.h>


extern int    gn_B, gn_C, gn_D, gn_I, gn_num_groups;
extern bool   gn_running, gn_scaling, gn_training;
extern float  gn_momentum, gn_eps;

extern float *gn_x;
extern float *gn_x_norm;
extern float *gn_running_mean;
extern float *gn_running_cov;
extern float *gn_weight;
extern float *gn_bias;

extern float *gn_running_mean_temp;
extern float *gn_running_cov_temp;
extern float *gn_weight_temp;
extern float *gn_weight_perm_temp;
extern float *gn_bias_temp;
extern float *gn_mean_temp;
extern float *gn_X_temp;
extern float *gn_cov_temp;
extern float *gn_max_temp;

void setup_groupnorm(
    int B, int C, int D, int I, int num_groups,
    bool running, bool scaling, bool training,
    float momentum, float eps
);

void cleanup_groupnorm(void);

#endif // GNN_SETUP_H
