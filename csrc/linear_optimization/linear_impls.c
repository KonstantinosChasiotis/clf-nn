#include "linear_bench.h"
#include "../clifford_linear.h"
#include "../flops.h"
#include <stdlib.h>    
#include <stdint.h> 
#include <immintrin.h> // For AVX2 intrinsics
#include <stdio.h>     // For fprintf, and for commented-out debug prints
#include <stdio.h>     // For fprintf in case of errors (though not strictly used in final proposal)
#include <math.h>      // For fmaf


// --------------------------------------------------------------------
// Scratch buffer for flattening
// --------------------------------------------------------------------
static float  *scratch_Xf = NULL;
static size_t  scratch_sz = 0;
static inline float* ensure_scratch(size_t needed) {
    if (scratch_sz < needed) {
        free(scratch_Xf);
        scratch_Xf = malloc(needed * sizeof *scratch_Xf);
        scratch_sz = needed;
    }
    return scratch_Xf;
}

// 1. =========================== Baseline ===========================

// --------------------------------------------------------------------
// 1) baseline_linear
// --------------------------------------------------------------------
void baseline_linear(const CliffordLinear* L, const float* x, int B, float* out) {
    clifford_linear_forward(L, x, B, out);
}


// 2. =========================== Basic Scalar Optimizations ===========================

// --------------------------------------------------------------------
// 2) optimized_linear_simple
// --------------------------------------------------------------------
void optimized_linear_v1(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        /* flatten */
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f; if (L->bias) FLOP(1);
                float *Krow = K + (size_t)r * d_flat;

                int d = 0, lim = (int)d_flat - 3;
                for (; d < lim; d += 4) {
                    acc +=
                      Krow[d  ] * Xf[d  ]
                    + Krow[d+1] * Xf[d+1]
                    + Krow[d+2] * Xf[d+2]
                    + Krow[d+3] * Xf[d+3];
                    FLOP(8);
                }
                for (; d < (int)d_flat; ++d) {
                    acc += Krow[d] * Xf[d]; FLOP(2);
                }

                outb[oc * I + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 3) optimized_linear_v2
// --------------------------------------------------------------------
void optimized_linear_v2(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float a0=0,a1=0,a2=0,a3=0;
                if (L->bias) { a0 = L->bias[r]; FLOP(1); }
                float *Krow = K + (size_t)r * d_flat;

                int d=0, lim = ((int)d_flat/4)*4;
                for (; d < lim; d += 4) {
                    a0 += Krow[d+0]*Xf[d+0];
                    a1 += Krow[d+1]*Xf[d+1];
                    a2 += Krow[d+2]*Xf[d+2];
                    a3 += Krow[d+3]*Xf[d+3];
                    FLOP(8);
                }
                float sum = a0+a1+a2+a3; FLOP(3);
                for (; d < (int)d_flat; ++d) {
                    sum += Krow[d]*Xf[d]; FLOP(2);
                }
                outb[oc * I + i_out] = sum;
            }
        }
    }
}

// --------------------------------------------------------------------
// 4) optimized_linear_v3
// --------------------------------------------------------------------
void optimized_linear_v3(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float a0=0,a1=0,a2=0,a3=0;
                if (L->bias) { a0 = L->bias[r]; FLOP(1); }
                float *Krow = K + (size_t)r * d_flat;

                int d=0, lim = ((int)d_flat/4)*4;
                for (; d < lim; d += 4) {
                    a0 = fmaf(Krow[d+0], Xf[d+0], a0);
                    a1 = fmaf(Krow[d+1], Xf[d+1], a1);
                    a2 = fmaf(Krow[d+2], Xf[d+2], a2);
                    a3 = fmaf(Krow[d+3], Xf[d+3], a3);
                    FLOP(8);
                }
                float sum = a0+a1+a2+a3; FLOP(3);
                for (; d < (int)d_flat; ++d) {
                    sum = fmaf(Krow[d], Xf[d], sum); FLOP(2);
                }
                outb[oc * I + i_out] = sum;
            }
        }
    }
}

// --------------------------------------------------------------------
// 5) optimized_linear_v4
// --------------------------------------------------------------------
void optimized_linear_v4(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f;
                if (L->bias) FLOP(1);

                float *Krow = K + (size_t)r * d_flat;
                const int TILE = 256;
                for (size_t base = 0; base < d_flat; base += TILE) {
                    size_t end = base + TILE;
                    if (end > d_flat) end = d_flat;
                    float a0=0,a1=0,a2=0,a3=0;
                    size_t t = base, lim = base + (((end-base)/4)*4);
                    for (; t < lim; t += 4) {
                        a0 = fmaf(Krow[t+0], Xf[t+0], a0);
                        a1 = fmaf(Krow[t+1], Xf[t+1], a1);
                        a2 = fmaf(Krow[t+2], Xf[t+2], a2);
                        a3 = fmaf(Krow[t+3], Xf[t+3], a3);
                        FLOP(8);
                    }
                    float sum = a0+a1+a2+a3; FLOP(3);
                    for (; t < end; ++t) {
                        sum = fmaf(Krow[t], Xf[t], sum); FLOP(2);
                    }
                    acc += sum; FLOP(1);
                }
                outb[oc * I + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 6) optimized_linear_v5 (alias of v4 for experimentation)
// --------------------------------------------------------------------
void optimized_linear_v5(
    const CliffordLinear* L,
    const float*          x,
    int                   B,
    float*                out
) {
    optimized_linear_v4(L, x, B, out);
}

// --------------------------------------------------------------------
// 7) optimized_linear_v6_avx
// --------------------------------------------------------------------
static inline float horizontal_sum_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s,s);
    s = _mm_hadd_ps(s,s);
    return _mm_cvtss_f32(s);
}
void optimized_linear_v6_avx(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f; if (L->bias) FLOP(1);
                float *Krow = K + (size_t)r * d_flat;

                __m256 vec = _mm256_setzero_ps();
                int d = 0, lim = ((int)d_flat/8)*8;
                for (; d < lim; d += 8) {
                    __m256 kv = _mm256_loadu_ps(Krow + d);
                    __m256 xv = _mm256_loadu_ps(Xf   + d);
                    vec = _mm256_fmadd_ps(kv, xv, vec); FLOP(16);
                }
                float sum = horizontal_sum_m256(vec); FLOP(7);
                acc += sum; FLOP(1);
                for (; d < (int)d_flat; ++d) {
                    acc = fmaf(Krow[d], Xf[d], acc); FLOP(2);
                }
                outb[oc * I + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 8) optimized_linear_v7_avx_k_accum
// --------------------------------------------------------------------
void optimized_linear_v7_avx_k_accum(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f; if (L->bias) FLOP(1);
                float *Krow = K + (size_t)r * d_flat;

                __m256 va = _mm256_setzero_ps();
                __m256 vb = _mm256_setzero_ps();
                int d=0, lim16=((int)d_flat/16)*16;
                for (; d < lim16; d += 16) {
                    __m256 k0=_mm256_loadu_ps(Krow+d+ 0), x0=_mm256_loadu_ps(Xf+d+ 0);
                    va=_mm256_fmadd_ps(k0,x0,va); FLOP(16);
                    __m256 k1=_mm256_loadu_ps(Krow+d+ 8), x1=_mm256_loadu_ps(Xf+d+ 8);
                    vb=_mm256_fmadd_ps(k1,x1,vb); FLOP(16);
                }
                if (d < ((int)d_flat/8)*8) {
                    __m256 k0=_mm256_loadu_ps(Krow+d), x0=_mm256_loadu_ps(Xf+d);
                    va=_mm256_fmadd_ps(k0,x0,va); FLOP(16);
                    d+=8;
                }
                __m256 vs=_mm256_add_ps(va,vb); FLOP(8);
                float sum = horizontal_sum_m256(vs); FLOP(7);
                acc += sum; FLOP(1);
                for (; d < (int)d_flat; ++d) {
                    acc = fmaf(Krow[d], Xf[d], acc); FLOP(2);
                }
                outb[oc * I + i_out] = acc;
            }
        }
    }
}

// --------------------------------------------------------------------
// 9) optimized_linear_v8_avx_more_unroll
// --------------------------------------------------------------------
void optimized_linear_v8_avx_more_unroll(
    const CliffordLinear* __restrict L,
    const float*          __restrict x,
    int                           B,
    float*                __restrict out
) {
    const int I     = L->sig.n_blades;
    const int C_in  = L->in_channels;
    const int C_out = L->out_channels;
    const size_t d_flat = (size_t)C_in * I;
    const size_t n_flat = (size_t)C_out * I;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    static float *K = NULL;
    static int     last_sz = 0;
    int needed = d_flat * n_flat;
    if (last_sz != needed) {
        free(K);
        int nb_unused;
        L->get_kernel(L->weight, L->sig.g, C_in, C_out, &nb_unused, &K);
        K = (float*)__builtin_assume_aligned(K, 64);
        last_sz = needed;
    }

    float *Xf = ensure_scratch(d_flat);

    for (int b = 0; b < B; ++b) {
        const float *xb = x + (size_t)b * d_flat;
        for (int i_in = 0; i_in < I; ++i_in) {
            size_t dst = (size_t)i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[(size_t)ic * I + i_in];
            }
        }

        float *outb = out + (size_t)b * n_flat;
        for (int i_out = 0; i_out < I; ++i_out) {
            for (int oc = 0; oc < C_out; ++oc) {
                int r = i_out * C_out + oc;
                float acc = L->bias ? L->bias[r] : 0.0f; if (L->bias) FLOP(1);
                float *Krow = K + (size_t)r * d_flat;

                __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
                __m256 a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
                int d=0, lim32=((int)d_flat/32)*32;
                for (; d < lim32; d += 32) {
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+ 0),
                                      _mm256_loadu_ps(Xf  +d+ 0),a0); FLOP(16);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+ 8),
                                      _mm256_loadu_ps(Xf  +d+ 8),a1); FLOP(16);
                    a2=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+16),
                                      _mm256_loadu_ps(Xf  +d+16),a2); FLOP(16);
                    a3=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+24),
                                      _mm256_loadu_ps(Xf  +d+24),a3); FLOP(16);
                }
                int lim16=((int)d_flat/16)*16;
                if (d < lim16) {
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+ 0),
                                      _mm256_loadu_ps(Xf  +d+ 0),a0); FLOP(16);
                    a1=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d+ 8),
                                      _mm256_loadu_ps(Xf  +d+ 8),a1); FLOP(16);
                    d+=16;
                }
                int lim8=((int)d_flat/8)*8;
                if (d < lim8) {
                    a0=_mm256_fmadd_ps(_mm256_loadu_ps(Krow+d),
                                      _mm256_loadu_ps(Xf  +d),a0); FLOP(16);
                    d+=8;
                }
                __m256 vs = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3)); FLOP(24);
                float sum = horizontal_sum_m256(vs); FLOP(7);
                acc += sum; FLOP(1);
                for (; d < (int)d_flat; ++d) {
                    acc = fmaf(Krow[d], Xf[d], acc); FLOP(2);
                }
                outb[oc * I + i_out] = acc;
            }
        }
    }
}


// 4. =========================== Advanced Optimizations ===========================

void optimized_linear_1d_kernel(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    // weights_view[output_blade][input_blade] = which weight index to use
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    // g_view[output_blade][input_blade] = which g coefficient to apply
    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                float accumulator = 0.0f;

                // Add bias if present
                if (L->bias) {
                    accumulator = L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Accumulate across all input channels and input blades
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        // Get the appropriate weight using lookup table
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = L->weight[weight_idx * C_out * C_in + out_ch * C_in + in_ch];
                        
                        // Get input value: x[batch * (C_in * I) + in_ch * I + in_blade]
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Get the appropriate g coefficient using lookup table
                        float g_coeff = g_view[out_blade][in_blade];
                        
                        // Accumulate: weight * input * g_coefficient
                        accumulator += weight * input_val * g_coeff;
                        FLOP(3); // 2 multiplies + 1 add
                    }
                }

                // Store result: out[batch * (C_out * I) + out_ch * I + out_blade]
                out[batch * (C_out * I) + out_ch * I + out_blade] = accumulator;
            }
        }
    }
}

void optimized_linear_1d_kernel_reordered_weights(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
                if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    // This gives us unit stride access when iterating through blades
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int out_ch = 0; out_ch < C_out; ++out_ch) {
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            for (int blade = 0; blade < I; ++blade) {
                // Original: L->weight[blade * C_out * C_in + out_ch * C_in + in_ch]
                // New:      weights_reorg[out_ch * C_in * I + in_ch * I + blade]
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                float accumulator = 0.0f;

                // Add bias if present
                if (L->bias) {
                    accumulator = L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Accumulate across all input channels and input blades
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    // Get base pointer for this out_ch/in_ch pair - unit stride access from here
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    const float *input_slice = x + batch * (C_in * I) + in_ch * I;
                    
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        // Get the appropriate weight using lookup table - now unit stride!
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                        
                        // Get input value - also unit stride
                        float input_val = input_slice[in_blade];
                        
                        // Get the appropriate g coefficient using lookup table
                        float g_coeff = g_view[out_blade][in_blade];
                        
                        // Accumulate: weight * input * g_coefficient
                        accumulator += weight * input_val * g_coeff;
                        FLOP(3); // 2 multiplies + 1 add
                    }
                }

                // Store result: out[batch * (C_out * I) + out_ch * I + out_blade]
                out[batch * (C_out * I) + out_ch * I + out_blade] = accumulator;
            }
        }
    }

    free(weights_reorg);
}

void optimized_linear_1d_kernel_reordered_weights_factored_g_accum(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int out_ch = 0; out_ch < C_out; ++out_ch) {
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            for (int blade = 0; blade < I; ++blade) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                
                // Accumulator per input blade - accumulate weight*input without g_coeff
                float blade_accumulator[I]; // For I=2, this is just 2 floats
                for (int i = 0; i < I; ++i) {
                    blade_accumulator[i] = 0.0f;
                }

                // Accumulate across all input channels (without g_coeff)
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    const float *input_slice = x + batch * (C_in * I) + in_ch * I;
                    
                    // Hottest loop - no g_coeff lookup here!
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                        float input_val = input_slice[in_blade];
                        
                        // Just accumulate weight * input (no g_coeff multiplication)
                        blade_accumulator[in_blade] += weight * input_val;
                        FLOP(2); // 1 multiply + 1 add
                    }
                }

                // Now apply g_coeff and reduce to final result
                float final_accumulator = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    float g_coeff = g_view[out_blade][in_blade];
                    final_accumulator += blade_accumulator[in_blade] * g_coeff;
                    FLOP(2); // 1 multiply + 1 add
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

void optimized_linear_1d_kernel_reordered_weights_factored_g_accum_fma(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int out_ch = 0; out_ch < C_out; ++out_ch) {
        for (int in_ch = 0; in_ch < C_in; ++in_ch) {
            for (int blade = 0; blade < I; ++blade) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                
                // Accumulator per input blade - accumulate weight*input without g_coeff
                float blade_accumulator[I]; // For I=2, this is just 2 floats
                for (int i = 0; i < I; ++i) {
                    blade_accumulator[i] = 0.0f;
                }

                // Accumulate across all input channels (without g_coeff)
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    const float *input_slice = x + batch * (C_in * I) + in_ch * I;
                    
                    // Hottest loop - no g_coeff lookup here! - OPTIMIZED WITH FMAF
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                        float input_val = input_slice[in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    float g_coeff = g_view[out_blade][in_blade];
                    // Use fmaf for fused multiply-add operation
                    final_accumulator = fmaf(blade_accumulator[in_blade], g_coeff, final_accumulator);
                    FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

void optimized_linear_1d_kernel_reordered_source_weights_factored_g_accum_fma(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 2) {
        // printf("This implementation is only valid for #blades = 2\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];

    // Lookup tables for 1D Clifford algebra
    int weights_view[2][2] = {
        {0, 1},  // For output blade 0: use w0 for input blade 0, w1 for input blade 1
        {1, 0}   // For output blade 1: use w1 for input blade 0, w0 for input blade 1
    };

    float g_view[2][2] = {
        {1.0f, g0},   // For output blade 0: multiply by 1 for input blade 0, g0 for input blade 1
        {1.0f, 1.0f}  // For output blade 1: multiply by 1 for both input blades
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                
                // Accumulator per input blade - accumulate weight*input without g_coeff
                float blade_accumulator[I]; // For I=2, this is just 2 floats
                for (int i = 0; i < I; ++i) {
                    blade_accumulator[i] = 0.0f;
                }

                // Accumulate across all input channels (without g_coeff)
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    const float *input_slice = x + batch * (C_in * I) + in_ch * I;
                    
                    // Hottest loop - no g_coeff lookup here! - OPTIMIZED WITH FMAF
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                        float input_val = input_slice[in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    float g_coeff = g_view[out_blade][in_blade];
                    // Use fmaf for fused multiply-add operation
                    final_accumulator = fmaf(blade_accumulator[in_blade], g_coeff, final_accumulator);
                    FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    float g_view[8][8] = {
        {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
        {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
        {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
        {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
        {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
        {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
        {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    float blade_accumulator[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {

                // Accumulate across all input channels (without g_coeff)
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    // Likely heavily-strided weight matrix cache accesses
                    int weight_idx = weights_view[out_blade][in_blade];
                    
                    for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                        const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                        float weight = weight_slice[weight_idx];
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    float g_coeff = g_view[out_blade][in_blade];
                    // Use fmaf for fused multiply-add operation
                    final_accumulator = fmaf(blade_accumulator[in_blade], g_coeff, final_accumulator);
                    blade_accumulator[in_blade] = 0.0f;
                    FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

// Likely bad striding of in_channel for each input leading to far more cycles than one-time input/output flattening striding.
void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use

    // Given the matrix is symmetric, we can load in the weight once and multiply by the weights_view inverse? Some form of unrolling?
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    float g_view[8][8] = {
        {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
        {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
        {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
        {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
        {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
        {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
        {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // float blade_accumulator[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {

                float blade_acc0 = 0.0f, blade_acc1 = 0.0f, blade_acc2 = 0.0f, blade_acc3 = 0.0f;
                float blade_acc4 = 0.0f, blade_acc5 = 0.0f, blade_acc6 = 0.0f, blade_acc7 = 0.0f;

                // Accumulate across all input channels (without g_coeff)
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    // Likely heavily-strided weight matrix cache accesses
                    int weight_idx = weights_view[out_blade][in_blade];
                    
                    for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                        const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                        float weight = weight_slice[weight_idx];
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        switch (in_blade) {
                            case 0: blade_acc0 = fmaf(weight, input_val, blade_acc0); break;
                            case 1: blade_acc1 = fmaf(weight, input_val, blade_acc1); break;
                            case 2: blade_acc2 = fmaf(weight, input_val, blade_acc2); break;
                            case 3: blade_acc3 = fmaf(weight, input_val, blade_acc3); break;
                            case 4: blade_acc4 = fmaf(weight, input_val, blade_acc4); break;
                            case 5: blade_acc5 = fmaf(weight, input_val, blade_acc5); break;
                            case 6: blade_acc6 = fmaf(weight, input_val, blade_acc6); break;
                            case 7: blade_acc7 = fmaf(weight, input_val, blade_acc7); break;
                        }
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator_1 = 0.0f;
                float final_accumulator_2 = 0.0f;
                float final_accumulator_3 = 0.0f;
                float final_accumulator_4 = 0.0f;
                final_accumulator_1 = fmaf(blade_acc0, g_view[out_blade][0], final_accumulator_1);
                final_accumulator_2 = fmaf(blade_acc1, g_view[out_blade][1], final_accumulator_2);
                final_accumulator_3 = fmaf(blade_acc2, g_view[out_blade][2], final_accumulator_3);
                final_accumulator_4 = fmaf(blade_acc3, g_view[out_blade][3], final_accumulator_4);
                final_accumulator_1 = fmaf(blade_acc4, g_view[out_blade][4], final_accumulator_1);
                final_accumulator_2 = fmaf(blade_acc5, g_view[out_blade][5], final_accumulator_2);
                final_accumulator_3 = fmaf(blade_acc6, g_view[out_blade][6], final_accumulator_3);
                final_accumulator_4 = fmaf(blade_acc7, g_view[out_blade][7], final_accumulator_4);

                float final_accumulator = final_accumulator_1 + final_accumulator_2 + final_accumulator_3 + final_accumulator_4;
                FLOP(16); // 8 FMAs, so 8 multiplies + 8 adds

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

// Likely bad striding of in_channel for each input leading to far more cycles than one-time input/output flattening striding.
void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use

    // Given the matrix is symmetric, we can load in the weight once and multiply by the weights_view inverse? Some form of unrolling?
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    float g_view[8][8] = {
        {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
        {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
        {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
        {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
        {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
        {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
        {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    float blade_accumulator[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {

                // float blade_acc0 = 0.0f, blade_acc1 = 0.0f, blade_acc2 = 0.0f, blade_acc3 = 0.0f;
                // float blade_acc4 = 0.0f, blade_acc5 = 0.0f, blade_acc6 = 0.0f, blade_acc7 = 0.0f;

                // Accumulate across all input channels (without g_coeff)
                for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                    
                    const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                    
                    for (int in_blade = 0; in_blade < I; ++in_blade) {
                        // Likely heavily-strided weight matrix cache accesses
                        int weight_idx = weights_view[out_blade][in_blade];
                        float weight = weight_slice[weight_idx];
                    
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);

                        // Use fmaf for fused multiply-add operation
                        // switch (in_blade) {
                        //     case 0: blade_acc0 = fmaf(weight, input_val, blade_acc0); break;
                        //     case 1: blade_acc1 = fmaf(weight, input_val, blade_acc1); break;
                        //     case 2: blade_acc2 = fmaf(weight, input_val, blade_acc2); break;
                        //     case 3: blade_acc3 = fmaf(weight, input_val, blade_acc3); break;
                        //     case 4: blade_acc4 = fmaf(weight, input_val, blade_acc4); break;
                        //     case 5: blade_acc5 = fmaf(weight, input_val, blade_acc5); break;
                        //     case 6: blade_acc6 = fmaf(weight, input_val, blade_acc6); break;
                        //     case 7: blade_acc7 = fmaf(weight, input_val, blade_acc7); break;
                        // }
                        FLOP(2); // FMA counts as 2 operations (1 multiply + 1 add)
                    }
                }

                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                // float final_accumulator_1 = 0.0f;
                // float final_accumulator_2 = 0.0f;
                // float final_accumulator_3 = 0.0f;
                // float final_accumulator_4 = 0.0f;
                // final_accumulator_1 = fmaf(blade_acc0, g_view[out_blade][0], final_accumulator_1);
                // final_accumulator_2 = fmaf(blade_acc1, g_view[out_blade][1], final_accumulator_2);
                // final_accumulator_3 = fmaf(blade_acc2, g_view[out_blade][2], final_accumulator_3);
                // final_accumulator_4 = fmaf(blade_acc3, g_view[out_blade][3], final_accumulator_4);
                // final_accumulator_1 = fmaf(blade_acc4, g_view[out_blade][4], final_accumulator_1);
                // final_accumulator_2 = fmaf(blade_acc5, g_view[out_blade][5], final_accumulator_2);
                // final_accumulator_3 = fmaf(blade_acc6, g_view[out_blade][6], final_accumulator_3);
                // final_accumulator_4 = fmaf(blade_acc7, g_view[out_blade][7], final_accumulator_4);

                float final_accumulator = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    final_accumulator = fmaf(blade_accumulator[in_blade], g_view[out_blade][in_blade], final_accumulator);
                }
                FLOP(16); // 8 FMAs, so 8 multiplies + 8 adds

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

// Likely bad striding of in_channel for each input leading to far more cycles than one-time input/output flattening striding.
void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use

    // Given the matrix is symmetric, we can load in the weight once and multiply by the weights_view inverse? Some form of unrolling?
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    float g_view[8][8] = {
        {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
        {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
        {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
        {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
        {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
        {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
        {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    float blade_accumulator[8] = {0}; // For I=8, this is 8 floats
    float blade_accumulator_inverse[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            int half_I = I / 2;
            for (int out_blade = 0; out_blade < half_I; ++out_blade) {

                float blade_acc0 = 0.0f, blade_acc1 = 0.0f, blade_acc2 = 0.0f, blade_acc3 = 0.0f;
                float blade_acc4 = 0.0f, blade_acc5 = 0.0f, blade_acc6 = 0.0f, blade_acc7 = 0.0f;

                // Accumulate across all input channels (without g_coeff)
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    int weight_idx = weights_view[out_blade][in_blade];
                    int weight_idx_inverse = weights_view[out_blade][I - in_blade - 1];
                    
                    for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                        const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                        
                        float weight = weight_slice[weight_idx];
                        float weight_inverse = weight_slice[weight_idx_inverse];
                    
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        blade_accumulator_inverse[in_blade] = fmaf(weight_inverse, input_val, blade_accumulator_inverse[in_blade]);

                        FLOP(4); // 2 FMAs, so 2 multiplies + 2 adds
                    }
                }


                // TODO: Fully-unroll this based on the in_blade and thus multiply only with g (otherwise just return or negate!)
                // Now apply g_coeff and reduce to final result - OPTIMIZED WITH FMAF
                float final_accumulator = 0.0f;
                float final_accumulator_inverse = 0.0f;
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    final_accumulator = fmaf(blade_accumulator[in_blade], g_view[out_blade][in_blade], final_accumulator);
                    final_accumulator_inverse = fmaf(blade_accumulator_inverse[in_blade], g_view[I - out_blade - 1][in_blade], final_accumulator_inverse);

                    blade_accumulator[in_blade] = 0.0f;
                    blade_accumulator_inverse[in_blade] = 0.0f;
                    FLOP(4); // 2 FMAs, so 2 multiplies + 2 adds
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    final_accumulator_inverse += L->bias[(I - out_blade - 1) * C_out + out_ch];
                    FLOP(2);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
                out[batch * (C_out * I) + out_ch * I + (I - out_blade - 1)] = final_accumulator_inverse;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

void optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse_unrolled_g(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use

    // Given the matrix is symmetric, we can load in the weight once and multiply by the weights_view inverse? Some form of unrolling?
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    // float g_view[8][8] = {
    //     {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
    //     {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
    //     {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
    //     {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
    //     {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
    //     {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
    //     {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
    //     {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    // };

    // Reorganize weights from [blades][out_channels][in_channels] to [out_channels][in_channels][blades]
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    float blade_accumulator[8] = {0}; // For I=8, this is 8 floats
    float blade_accumulator_inverse[8] = {0}; // For I=8, this is 8 floats

    // Main computation loops
    for (int batch = 0; batch < B; ++batch) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            int half_I = I / 2;
            for (int out_blade = 0; out_blade < half_I; ++out_blade) {

                float blade_acc0 = 0.0f, blade_acc1 = 0.0f, blade_acc2 = 0.0f, blade_acc3 = 0.0f;
                float blade_acc4 = 0.0f, blade_acc5 = 0.0f, blade_acc6 = 0.0f, blade_acc7 = 0.0f;

                // Accumulate across all input channels (without g_coeff)
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    int weight_idx = weights_view[out_blade][in_blade];
                    int weight_idx_inverse = weights_view[out_blade][I - in_blade - 1];
                    
                    for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                        const float *weight_slice = weights_reorg + out_ch * C_in * I + in_ch * I;
                        
                        float weight = weight_slice[weight_idx];
                        float weight_inverse = weight_slice[weight_idx_inverse];
                    
                        float input_val = x[batch * (C_in * I) + in_ch * I + in_blade];
                        
                        // Use fmaf for fused multiply-add operation
                        blade_accumulator[in_blade] = fmaf(weight, input_val, blade_accumulator[in_blade]);
                        blade_accumulator_inverse[in_blade] = fmaf(weight_inverse, input_val, blade_accumulator_inverse[in_blade]);

                        FLOP(4); // 2 FMAs, so 2 multiplies + 2 adds
                    }
                }

                // Replace the TODO section with explicit g-coefficient multiplications
                // Now apply g_coeff and reduce to final result - FULLY UNROLLED AND OPTIMIZED
                float final_accumulator = 0.0f;
                float final_accumulator_inverse = 0.0f;

                switch (out_blade) {
                    case 0: // blade 0: {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2}
                            // blade 7: {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f}
                        final_accumulator = blade_accumulator[0] +                          // 1.0f
                                        blade_accumulator[1] * g0 +                      // g0
                                        blade_accumulator[2] * g1 +                      // g1
                                        blade_accumulator[3] * g2 +                      // g2
                                        blade_accumulator[4] * (-g0g1) +                 // -g0g1
                                        blade_accumulator[5] * (-g0g2) +                 // -g0g2
                                        blade_accumulator[6] * (-g1g2) +                 // -g1g2
                                        blade_accumulator[7] * (-g0g1g2);                // -g0g1g2
                        
                        final_accumulator_inverse = blade_accumulator_inverse[0] +          // 1.0f
                                                blade_accumulator_inverse[1] +           // 1.0f (no multiply!)
                                                (-blade_accumulator_inverse[2]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[3] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[4] +           // 1.0f (no multiply!)
                                                (-blade_accumulator_inverse[5]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[6] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[7];            // 1.0f (no multiply!)
                        FLOP(6 + 0); // 6 multiplies for normal, 0 for inverse (only adds/negates)
                        break;

                    case 1: // blade 1: {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2}
                            // blade 6: {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0}
                        final_accumulator = blade_accumulator[0] +                          // 1.0f (no multiply!)
                                        blade_accumulator[1] +                           // 1.0f (no multiply!)
                                        blade_accumulator[2] * (-g1) +                   // -g1
                                        blade_accumulator[3] * (-g2) +                   // -g2
                                        blade_accumulator[4] * g1 +                      // g1
                                        blade_accumulator[5] * g2 +                      // g2
                                        blade_accumulator[6] * (-g1g2) +                 // -g1g2
                                        blade_accumulator[7] * (-g1g2);                  // -g1g2
                        
                        final_accumulator_inverse = blade_accumulator_inverse[0] +          // 1.0f (no multiply!)
                                                blade_accumulator_inverse[1] * g0 +      // g0
                                                blade_accumulator_inverse[2] +           // 1.0f (no multiply!)
                                                (-blade_accumulator_inverse[3]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[4] * (-g0) +   // -g0
                                                blade_accumulator_inverse[5] * g0 +      // g0
                                                blade_accumulator_inverse[6] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[7] * g0;       // g0
                        FLOP(6 + 4); // 6 multiplies for normal, 4 for inverse
                        break;

                    case 2: // blade 2: {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2}
                            // blade 5: {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1}
                        final_accumulator = blade_accumulator[0] +                          // 1.0f (no multiply!)
                                        blade_accumulator[1] * g0 +                      // g0
                                        blade_accumulator[2] +                           // 1.0f (no multiply!)
                                        blade_accumulator[3] * (-g2) +                   // -g2
                                        blade_accumulator[4] * (-g0) +                   // -g0
                                        blade_accumulator[5] * g0g2 +                    // g0g2
                                        blade_accumulator[6] * g2 +                      // g2
                                        blade_accumulator[7] * g0g2;                     // g0g2
                        
                        final_accumulator_inverse = blade_accumulator_inverse[0] +          // 1.0f (no multiply!)
                                                blade_accumulator_inverse[1] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[2] * (-g1) +   // -g1
                                                (-blade_accumulator_inverse[3]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[4] * g1 +      // g1
                                                blade_accumulator_inverse[5] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[6] * (-g1) +   // -g1
                                                blade_accumulator_inverse[7] * (-g1);    // -g1
                        FLOP(6 + 4); // 6 multiplies for normal, 4 for inverse
                        break;

                    case 3: // blade 3: {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1}
                            // blade 4: {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2}
                        final_accumulator = blade_accumulator[0] +                          // 1.0f (no multiply!)
                                        blade_accumulator[1] * g0 +                      // g0
                                        blade_accumulator[2] * g1 +                      // g1
                                        blade_accumulator[3] +                           // 1.0f (no multiply!)
                                        blade_accumulator[4] * (-g0g1) +                 // -g0g1
                                        blade_accumulator[5] * (-g0) +                   // -g0
                                        blade_accumulator[6] * (-g1) +                   // -g1
                                        blade_accumulator[7] * (-g0g1);                  // -g0g1
                        
                        final_accumulator_inverse = blade_accumulator_inverse[0] +          // 1.0f (no multiply!)
                                                blade_accumulator_inverse[1] +           // 1.0f (no multiply!)
                                                (-blade_accumulator_inverse[2]) +        // -1.0f (just negate!)
                                                blade_accumulator_inverse[3] * g2 +      // g2
                                                blade_accumulator_inverse[4] +           // 1.0f (no multiply!)
                                                blade_accumulator_inverse[5] * (-g2) +   // -g2
                                                blade_accumulator_inverse[6] * g2 +      // g2
                                                blade_accumulator_inverse[7] * g2;       // g2
                        FLOP(6 + 4); // 6 multiplies for normal, 4 for inverse
                        break;
                }

                // Clear accumulators for next iteration
                for (int i = 0; i < I; ++i) {
                    blade_accumulator[i] = 0.0f;
                    blade_accumulator_inverse[i] = 0.0f;
                }

                // Add bias if present
                if (L->bias) {
                    final_accumulator += L->bias[out_blade * C_out + out_ch];
                    final_accumulator_inverse += L->bias[(I - out_blade - 1) * C_out + out_ch];
                    FLOP(2);
                }

                // Store result
                out[batch * (C_out * I) + out_ch * I + out_blade] = final_accumulator;
                out[batch * (C_out * I) + out_ch * I + (I - out_blade - 1)] = final_accumulator_inverse;
            }
        }
    }
    
    // Clean up weights memory
    free(weights_reorg);
}

void optimized_linear_3d_kernel_flattened_input(
    const CliffordLinear * __restrict L,
    const float          * __restrict x,    // x[b*(C_in*I) + c*I + i]
    int                    B,
    float                * __restrict out)  // out[b*(C_out*I) + c*I + i]
{
    const int I      = L->sig.n_blades;
    const int C_in   = L->in_channels;
    const int C_out  = L->out_channels;

    if (I != 8) {
        // printf("This implementation is only valid for #blades = 8 (3D Clifford)\n");
        return;
    }

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * C_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * C_out * C_in * sizeof(float);
    size_t output_bytes = B * C_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * C_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    const float g0 = L->sig.g[0];
    const float g1 = L->sig.g[1];
    const float g2 = L->sig.g[2];

    // Lookup tables for 3D Clifford algebra (8x8 matrices)
    // Permutation matrix P - which weight index to use
    int weights_view[8][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7},  // output blade 0
        {1, 0, 4, 5, 2, 3, 7, 6},  // output blade 1
        {2, 4, 0, 6, 1, 7, 3, 5},  // output blade 2
        {3, 5, 6, 0, 7, 1, 2, 4},  // output blade 3
        {4, 2, 1, 7, 0, 6, 5, 3},  // output blade 4
        {5, 3, 7, 1, 6, 0, 4, 2},  // output blade 5
        {6, 7, 3, 2, 5, 4, 0, 1},  // output blade 6
        {7, 6, 5, 4, 3, 2, 1, 0}   // output blade 7
    };

    // Coefficient matrix C - scalar multipliers
    // Pre-compute all g coefficient combinations
    float g0g1 = g0 * g1;
    float g0g2 = g0 * g2;
    float g1g2 = g1 * g2;
    float g0g1g2 = g0 * g1 * g2;

    float g_view[8][8] = {
        {1.0f, g0, g1, g2, -g0g1, -g0g2, -g1g2, -g0g1g2},  // output blade 0
        {1.0f, 1.0f, -g1, -g2, g1, g2, -g1g2, -g1g2},      // output blade 1
        {1.0f, g0, 1.0f, -g2, -g0, g0g2, g2, g0g2},        // output blade 2
        {1.0f, g0, g1, 1.0f, -g0g1, -g0, -g1, -g0g1},      // output blade 3
        {1.0f, 1.0f, -1.0f, g2, 1.0f, -g2, g2, g2},        // output blade 4
        {1.0f, 1.0f, -g1, -1.0f, g1, 1.0f, -g1, -g1},      // output blade 5
        {1.0f, g0, 1.0f, -1.0f, -g0, g0, 1.0f, g0},        // output blade 6
        {1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f} // output blade 7
    };

    const int d_flat = C_in * I;      // input‐features per sample

    // 1) Flatten weights from [blades][output][input] to [output][input][blades] (before batch loop)
    float *weights_reorg = malloc(C_out * C_in * I * sizeof(float));
    for (int blade = 0; blade < I; ++blade) {
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                weights_reorg[out_ch * C_in * I + in_ch * I + blade] = 
                    L->weight[blade * C_out * C_in + out_ch * C_in + in_ch];
            }
        }
    }

    // Initialize output to zero
    for (int i = 0; i < B * C_out * I; ++i) {
        out[i] = 0.0f;
    }

    // 2) Process each sample
    for (int b = 0; b < B; ++b) {
        
        // 2a) Flatten x[b] into [blades][input_channels] layout
        float Xf[d_flat];
        const float *xb = x + b * (C_in * I);
        for (int i_in = 0; i_in < I; ++i_in) {
            int dst = i_in * C_in;
            for (int ic = 0; ic < C_in; ++ic) {
                Xf[dst + ic] = xb[ic * I + i_in];
            }
        }

        // 2b) For each output channel and output blade
        for (int out_ch = 0; out_ch < C_out; ++out_ch) {
            for (int out_blade = 0; out_blade < I; ++out_blade) {
                
                float total_acc = 0.0f;
                
                // Add bias if present
                if (L->bias) {
                    total_acc = L->bias[out_blade * C_out + out_ch];
                    FLOP(1);
                }

                // For each input blade, accumulate contribution
                for (int in_blade = 0; in_blade < I; ++in_blade) {
                    
                    // Get which weight to use from permutation matrix
                    int weight_idx = weights_view[out_blade][in_blade];
                    
                    // Get geometric coefficient
                    float g_coeff = g_view[out_blade][in_blade];
                    
                    // Accumulate across all input channels for this blade pair
                    float blade_acc = 0.0f;
                    for (int in_ch = 0; in_ch < C_in; ++in_ch) {
                        // Get weight: weights_reorg[out_ch][in_ch][weight_idx]
                        float weight = weights_reorg[out_ch * C_in * I + in_ch * I + weight_idx];
                        
                        // Get input: Xf[in_blade][in_ch]
                        float input_val = Xf[in_blade * C_in + in_ch];
                        
                        blade_acc = fmaf(weight, input_val, blade_acc);
                        FLOP(2); // FMA counts as 2 operations
                    }
                    
                    // Apply geometric coefficient and add to total
                    total_acc = fmaf(g_coeff, blade_acc, total_acc);
                    FLOP(2); // FMA counts as 2 operations
                }

                // Store result: out[batch][out_ch][out_blade]
                out[b * (C_out * I) + out_ch * I + out_blade] = total_acc;
            }
        }
    }
    
    // Clean up memory
    free(weights_reorg);
}

void register_linear_functions() {
    // 1. Baseline
    add_linear_function(baseline_linear, "baseline_linear");
    
    // // 2. Basic Scalar Optimizations
    // add_linear_function(optimized_linear_v1, "optimized_v1_kernel_caching"); //kernel caching
    // add_linear_function(optimized_linear_v2, "optimized_v2_k_accum");
    // add_linear_function(optimized_linear_v3, "optimized_v3_fma_k_accum");
    // // add_linear_function(optimized_linear_v4, "optimized_v4_tiled_fma_k_accum");
    // // add_linear_function(optimized_linear_v5, "optimized_v5_scalar_peak"); //Best scalar performance
    
    // // 3. Vectorized versions
    // add_linear_function(optimized_linear_v6_avx, "optimized_v6_avx");
    // add_linear_function(optimized_linear_v7_avx_k_accum, "optimized_v7_avx_k_accum"); //Using vector k accumulators
    // add_linear_function(optimized_linear_v8_avx_more_unroll, "optimized_v8_avx_more_unroll"); //Use further vector k accumulators

    // 4. Advanced optimizations (dimension specific, no kernel construction)
    // 1D specific kernel
    // add_linear_function(optimized_linear_1d_kernel, "optimized_linear_1d_kernel");
    // add_linear_function(optimized_linear_1d_kernel_reordered_weights, "optimized_linear_1d_kernel_reordered_weights");
    // add_linear_function(optimized_linear_1d_kernel_reordered_weights_factored_g_accum, "optimized_linear_1d_kernel_reordered_weights_factored_g_accum");
    // add_linear_function(optimized_linear_1d_kernel_reordered_weights_factored_g_accum_fma, "optimized_linear_1d_kernel_reordered_weights_factored_g_accum_fma");
    // add_linear_function(optimized_linear_1d_kernel_reordered_source_weights_factored_g_accum_fma, "optimized_linear_1d_kernel_reordered_source_weights_factored_g_accum_fma");

    // // 3D specific kernel
    add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma, "no_kernel_v1");
    add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum, "no_kernel_v2");
    // add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner, "optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner");
    // add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner_inverse, "optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_blade_inner_inverse");
    add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse, "no_kernel_v3");
    add_linear_function(optimized_linear_3d_kernel_reordered_weights_factored_g_accum_fma_k_accum_inverse_unrolled_g, "no_kernel_v4");
    // add_linear_function(optimized_linear_3d_kernel_flattened_input, "optimized_linear_3d_kernel_flattened_input");
}
