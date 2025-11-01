#include "linear_setup.h"
#include <stdio.h>
#include <stdlib.h>

CliffordLinear* L_lin;
float* lin_x;
float* lin_out;

void setup_linear(int B, int C, int dim, int O, int *g_sig)
{
    int I = 1 << dim;

    lin_x = malloc(sizeof(float) * B * C * I);
    lin_out = malloc(sizeof(float) * B * O * I);

    for (int i = 0, N = B * C * I; i < N; i++)
        lin_x[i] = rand() / (float)RAND_MAX;

    L_lin = clifford_linear_create(g_sig, dim, C, O, true);
    if (!L_lin) {
        fprintf(stderr, "Error: clifford_linear_create failed (dim=%d)\n", dim);
        exit(1);
    }

    size_t nW = (size_t)I * O * C;
    for (size_t i = 0; i < nW; i++)
        L_lin->weight[i] = rand() / (float)RAND_MAX;

    size_t nB = (size_t)I * O;
    for (size_t i = 0; i < nB; i++)
        L_lin->bias[i] = rand() / (float)RAND_MAX;
}