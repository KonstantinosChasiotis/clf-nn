#ifndef MV_ACT_H
#define MV_ACT_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Full multivector activation (linear aggregation with bias).
 *
 * x:               [B][C][I_in]             raw per‐blade scalars
 * B, C, I_in:      batch, channels, num input blades
 * I_full:          number of blades in the algebra
 * input_blades:    length‐I_in array of blade indices
 * weight:          [C][K]                   per‐channel conv kernels
 * K:               kernel size (num taps)
 * kernel_blades:   length‐K array of blade indices to convolve over
 * bias:            [C]                      per‐channel bias
 * out:             [B][C][I_in]             final gated outputs
 */
void mv_act_forward(
    const float *x,             // [B][C][I_in]
    int           B,
    int           C,
    int           I_in,
    int           I_full,
    const int    *input_blades, // [I_in]
    const float  *weight,       // [C][K]
    int           K,
    const int    *kernel_blades,// [K]
    const float  *bias,         // [C]
    float        *out           // [B][C][I_in]
);

#ifdef __cplusplus
}
#endif
#endif // MV_ACT_H
