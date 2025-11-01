#include "mv_act_setup.h"
#include <stdlib.h>
#include <time.h>

float *act_x;
float *act_out;
int   *act_input_blades;
float *act_weight;
int   *act_kernel_blades;
float *act_bias;
int act_B, act_C, act_I_in, act_I_full, act_K;

void setup_act(int B, int C, int I_in, int I_full, int K) {
    act_B = B;
    act_C = C;
    act_I_in = I_in;
    act_I_full = I_full;
    act_K = K;

    srand((unsigned)time(NULL));

    size_t N_in = (size_t)B * C * I_in;

    act_x = malloc(N_in   * sizeof(float));
    act_out = malloc(N_in   * sizeof(float));
    for (size_t i = 0; i < N_in; ++i) {
        act_x[i] = rand() / (float)RAND_MAX;
    }

    act_input_blades = malloc(I_in * sizeof(int));
    for (int i = 0; i < I_in; ++i) {
        act_input_blades[i] = i % I_full;
    }

    act_weight = malloc((size_t)C * K * sizeof(float));
    act_kernel_blades = malloc(K * sizeof(int));
    act_bias = malloc(C * sizeof(float));

    for (int i = 0; i < C * K; ++i) {
        act_weight[i] = rand() / (float)RAND_MAX;
    }

    for (int i = 0; i < K; ++i) {
        act_kernel_blades[i] = rand() % I_full;
    }

    for (int c = 0; c < C; ++c) {
        act_bias[c] = rand() / (float)RAND_MAX;
    }
}

void cleanup_act(void) {
    free(act_x);
    free(act_out);
    free(act_input_blades);
    free(act_weight);
    free(act_kernel_blades);
    free(act_bias);
}