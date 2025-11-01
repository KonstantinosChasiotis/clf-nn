#ifndef ACTIVATION_SETUP_H
#define ACTIVATION_SETUP_H

#include "../mv_act.h"
#include <stddef.h>
#include <stdint.h>

extern float *act_x;
extern float *act_out;
extern int   *act_input_blades;
extern float *act_weight;
extern int   *act_kernel_blades;
extern float *act_bias;
extern int act_B, act_C, act_I_in, act_I_full, act_K;

void setup_act(int B, int C, int I_in, int I_full, int K);

void cleanup_act(void);

#endif // ACTIVATION_SETUP_H