#include "gnn_setup.h"
#include "../clifford_groupnorm.h"
#include <stdlib.h>
#include <time.h>

int    gn_B, gn_C, gn_D, gn_I, gn_num_groups;
bool   gn_running, gn_scaling, gn_training;
float  gn_momentum, gn_eps;

float *gn_x;
float *gn_x_norm;
float *gn_running_mean;
float *gn_running_cov;
float *gn_weight;
float *gn_bias;

float *gn_running_mean_temp;
float *gn_running_cov_temp;
float *gn_weight_temp;
float *gn_weight_perm_temp;
float *gn_bias_temp;
float *gn_mean_temp;
float *gn_X_temp;
float *gn_cov_temp;
float *gn_max_temp;

void setup_groupnorm(
    int B, int C, int D, int I, int num_groups,
    bool running, bool scaling, bool training,
    float momentum, float eps
) {
    gn_B = B;
    gn_C = C;
    gn_D = D;
    gn_I = I;
    gn_num_groups = num_groups;
    gn_running  = running;
    gn_scaling  = scaling;
    gn_training = training;
    gn_momentum = momentum;
    gn_eps      = eps;

    srand((unsigned)time(NULL));

    size_t N = (size_t)B * C * D * I;
    gn_x      = malloc(N * sizeof(float));
    gn_x_norm = malloc(N * sizeof(float));
    for (size_t i = 0; i < N; ++i) {
        gn_x[i] = rand() / (float)RAND_MAX;
    }

    int grp_channels = C / num_groups;
    size_t mean_sz = (size_t)I * grp_channels;
    size_t cov_sz  = (size_t)I * I * grp_channels;

    gn_running_mean = malloc(mean_sz * sizeof(float));
    gn_running_cov  = malloc(cov_sz  * sizeof(float));
    memset(gn_running_mean, 0, mean_sz * sizeof(float));
    memset(gn_running_cov,  0, cov_sz  * sizeof(float));

    gn_weight = malloc(cov_sz * sizeof(float));
    gn_bias   = malloc(mean_sz * sizeof(float));
    for (size_t i = 0; i < cov_sz;  ++i) gn_weight[i] = rand()/(float)RAND_MAX;
    for (size_t i = 0; i < mean_sz; ++i) gn_bias[i]   = rand()/(float)RAND_MAX;

    clifford_groupnorm_alloctemp(
        B, C, D, I, num_groups, running,
        &gn_running_mean_temp,
        &gn_running_cov_temp,
        scaling,
        &gn_weight_temp,
        &gn_weight_perm_temp,
        &gn_bias_temp,
        &gn_mean_temp,
        &gn_X_temp,
        &gn_cov_temp,
        &gn_max_temp
    );
}

void cleanup_groupnorm(void) {

    clifford_groupnorm_freetemp(
        gn_B, gn_C, gn_D, gn_I, gn_num_groups, gn_running,
        &gn_running_mean_temp,
        &gn_running_cov_temp,
        gn_scaling,
        &gn_weight_temp,
        &gn_weight_perm_temp,
        &gn_bias_temp,
        &gn_mean_temp,
        &gn_X_temp,
        &gn_cov_temp,
        &gn_max_temp
    );

    free(gn_x);
    free(gn_x_norm);
    free(gn_running_mean);
    free(gn_running_cov);
    free(gn_weight);
    free(gn_bias);
}
