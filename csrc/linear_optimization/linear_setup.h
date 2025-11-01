#ifndef LINEAR_SETUP_H
#define LINEAR_SETUP_H

#include "../clifford_linear.h"

extern CliffordLinear* L_lin;
extern float* lin_x;
extern float* lin_out;

void setup_linear(int B, int C, int dim, int O, int* g_sig);

#endif