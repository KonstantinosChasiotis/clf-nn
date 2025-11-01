#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "mv_act.h"
#include "flops.h"

/**
 * Multivector activation (linear) across blade dimension:
 * 
 * SO WHAT WE ARE ESSENTIALLY DOING : 
 *   out[b,c,p,i] = x[b,c,p,i] * sigmoid(
 *       sum_{k=0..K-1} weight[c*K + k] * x[b,c,p, kernel_blades[k]]
 *   )
 * 
 * 
 *
 * Todo : need to update that we also need the input_blades 
 * Parameters:
 *   x               - input data [B][C][L][I]
 *   B, C, L, I      - batch size, channels,length,num blades
 *   weight          - conv weight [C][K]
 *   K               - number of kernel blades
 *   kernel_blades   - blade indices [K]
 *   bias            - bias
 *   out             - output buffer [B][C][L][I]
 */

void mv_act_forward(
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
) {
    // 1) Allocate + zero the embedded multivector buffer v[B*C*I_full]
    size_t Vsize = (size_t)B * C * I_full;
    float *v = (float*)malloc(Vsize * sizeof(float));
    memset(v, 0, Vsize * sizeof(float));

    // 2) EMBED: scatter each x[b,c,i] → v[b,c, input_blades[i]]
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        size_t xc = (size_t)b*C*I_in + c*I_in;
        size_t vc = (size_t)b*C*I_full + c*I_full;
        for (int i = 0; i < I_in; ++i) {
          int blade = input_blades[i];
          v[vc + blade] = x[xc + i];
        }
      }
    }

    // 3) CONV+SIGMOID: one gate per (b,c)
    float *gate = (float*)malloc((size_t)B * C * sizeof(float));
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        float acc = bias[c];                   // start with bias
        const float *w = weight + (size_t)c * K;
        FLOP(2);
        size_t vc = (size_t)b*C*I_full + c*I_full;
        for (int k = 0; k < K; ++k) {
          int blade = kernel_blades[k];
          if (blade >= 0 && blade < I_full) {
            acc += w[k] * v[vc + blade];
            FLOP(2); // multiply + add
          }
        }
        gate[b*C + c] = 1.0f / (1.0f + expf(-acc));
        FLOP(4); // exp + divide
      }
    }

    // 4) GET: for each (b,c), multiply embedded slots by gate and gather back
    for (int b = 0; b < B; ++b) {
      for (int c = 0; c < C; ++c) {
        float g = gate[b*C + c];
        size_t vc = (size_t)b*C*I_full + c*I_full;
        size_t xo = (size_t)b*C*I_in  + c*I_in;
        for (int i = 0; i < I_in; ++i) {
          int blade = input_blades[i];
          out[xo + i] = v[vc + blade] * g;
            FLOP(1); // multiply by gate
        }
      }
    }

    free(gate);
    free(v);
}
