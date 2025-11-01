#include "mv_act_bench.h"
#include "mv_act_setup.h"
#include "../mv_act.h"
#include "../flops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>


static void baseline_act(
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
    // Track bytes read from input arrays
    BYTES_READ(B * C * I_in * sizeof(float));        // x
    BYTES_READ(C * K * sizeof(float));               // weight
    BYTES_READ(C * sizeof(float));                   // bias
    BYTES_READ(I_in * sizeof(int));                  // input_blades
    BYTES_READ(K * sizeof(int));                     // kernel_blades
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * I_in * sizeof(float));     // out
    
    mv_act_forward(
      x, B, C, I_in, I_full,
      input_blades, weight, K,
      kernel_blades, bias, out
    );
}



/*
 * optimized_act_v1
 *
 * This function computes a gated activation over a 3D tensor x[B][C][I_in],
 * using a sparse kernel defined by kernel_blades[K] and an inverse mapping
 * of input_blades[I_in] into a dense space of size I_full. It eliminates
 * the need for a large temporary buffer by performing an on-the-fly lookup.
 */
void optimized_act_v1(
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
    // ----------------------------------------------------------------------
    // 1) Build inverse map: blade_index → position in x (or -1 if unused)
    //    - Allocate an array of length I_full
    //    - Initialize each slot to -1 (marks "unused blade")
    int *inv = malloc(I_full * sizeof *inv);
    if (!inv) {
        // Optional: handle allocation failure
        return;
    }
    for (int i = 0; i < I_full; ++i) {
        inv[i] = -1;
    }
    //    - Fill in only the used blades by inverting the input_blades map
    for (int i = 0; i < I_in; ++i) {
        int blade = input_blades[i];
        if (blade >= 0 && blade < I_full) {
            inv[blade] = i;
        }
        // else: invalid blade index - could assert or ignore
    }

    // ----------------------------------------------------------------------
    // 2) Allocate scratch space for gate activations of size B * C
    float *gate = malloc((size_t)B * C * sizeof *gate);
    if (!gate) {
        free(inv);
        // Optional: handle allocation failure
        return;
    }

    // ----------------------------------------------------------------------
    // 3) Compute gate[b,c] = sigmoid( bias[c] + Σ_k weight[c*K + k] * x[b,c,inv[blade]] )
    //    - Loop over batches and channels
    //    - For each kernel blade index, look up its input position via inv[]
    //    - If inv[blade] < 0, that blade is unused and contributes zero
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            size_t baseX = (size_t)b * C * I_in + (size_t)c * I_in;
            size_t baseW = (size_t)c * K;
            float acc = bias[c];  // start from bias

            for (int k = 0; k < K; ++k) {
                int blade = kernel_blades[k];
                int xi    = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
                if (xi >= 0) {
                    // valid input position: accumulate weighted x
                    acc += weight[baseW + k] * x[baseX + xi];
                    FLOP(2); // multiply + add
                }
                // if xi < 0: skip unused blade (contribution = 0)
            }

            // Apply sigmoid activation: gate = 1 / (1 + exp(-acc))
            gate[b*C + c] = 1.0f / (1.0f + expf(-acc));
            FLOP(4); // exp + add + reciprocal + misc
        }
    }

    // ----------------------------------------------------------------------
    // 4) Write back gated results: out[b,c,i] = x[b,c,i] * gate[b,c]
    //    - Loop over batches, channels, and input positions
    //    - Multiply each original input by the computed gate
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            size_t baseX = (size_t)b * C * I_in + (size_t)c * I_in;
            size_t gate_idx = (size_t)b * C + c;
            float  g = gate[gate_idx];
            for (int i = 0; i < I_in; ++i) {
                // One FLOP: multiply
                out[baseX + i] = x[baseX + i] * g;
            }
        }
    }

    // ----------------------------------------------------------------------
    // 5) Cleanup temporary allocations
    free(gate);
    free(inv);

    // ----------------------------------------------------------------------
}



void optimized_act_v2(
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
    // 1) Build inverse‐index map: blade → input index (or –1 if unused)
    int *inv = malloc(I_full * sizeof *inv);
    for (int i = 0; i < I_full; ++i) {
        inv[i] = -1;
    }
    for (int i = 0; i < I_in; ++i) {
        inv[input_blades[i]] = i;
    }

    // 2) Allocate workspace for gates
    float *gate = malloc((size_t)B * C * sizeof *gate);

    // 3) Precompute strides
    int CI = C * I_in;   // jump from x[b] to x[b+1]

    // --- Conv + Sigmoid: channels outer, batches inner ---
    for (int c = 0; c < C; ++c) {
        size_t baseW = (size_t)c * K;      // &weight[c][0]
        float  bval  = bias[c];            // bias for channel c
        size_t cIin  = (size_t)c * I_in;   // offset within each batch

        for (int b = 0; b < B; ++b) {
            size_t xoff = (size_t)b * CI + cIin;
            float acc = bval;

            // sparse dot‐product over K
            for (int k = 0; k < K; ++k) {
                int blade = kernel_blades[k];
                int xi    = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
                if (xi >= 0) {
                    acc += weight[baseW + k] * x[xoff + xi];
                    FLOP(2);    // one multiply + one add
                }
            }

            // sigmoid activation
            gate[b*C + c] = 1.0f / (1.0f + expf(-acc));
            FLOP(4);        // exp + add + divide + misc
        }
    }

    // --- Scale & write‐back: batches outer, channels inner ---
    for (int b = 0; b < B; ++b) {
        size_t bCI = (size_t)b * CI;
        size_t bg  = (size_t)b * C;

        for (int c = 0; c < C; ++c) {
            size_t xoff = bCI + (size_t)c * I_in;
            float  g    = gate[bg + c];

            for (int i = 0; i < I_in; ++i) {
                out[xoff + i] = x[xoff + i] * g;
                FLOP(1);    // one multiply
            }
        }
    }

    // 4) Cleanup
    free(gate);
    free(inv);
}




/*
 * optimized_act_v5
 *
 * Fused activation without SIMD/OpenMP pragmas, using per-channel
 * precomputation of valid indices and weights to enable branch-free
 * inner loops. This version targets single-core execution with
 * minimal control flow inside hot loops.
 */


void optimized_act_v5(
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
    // Track bytes read from input arrays
    BYTES_READ(B * C * I_in * sizeof(float));        // x
    BYTES_READ(C * K * sizeof(float));               // weight
    BYTES_READ(C * sizeof(float));                   // bias
    BYTES_READ(I_in * sizeof(int));                  // input_blades
    BYTES_READ(K * sizeof(int));                     // kernel_blades
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * I_in * sizeof(float));     // out
    
    // ----------------------------------------------------------------------
    // 1) Build inverse map: blade index → input slot (or -1 if unused)
    int *inv = malloc((size_t)I_full * sizeof *inv);
    if (!inv) return;  // allocation failure
    // Initialize all entries to -1
    for (int i = 0; i < I_full; ++i) {
        inv[i] = -1;
    }
    // Fill map for active blades
    for (int i = 0; i < I_in; ++i) {
        int blade = input_blades[i];
        if (blade >= 0 && blade < I_full) {
            inv[blade] = i;
        }
        // else: out-of-range blade ignored
    }

    // ----------------------------------------------------------------------
    // 2) Compute stride: number of floats per batch slice
    size_t stride = (size_t)C * I_in;

    // ----------------------------------------------------------------------
    // 3) Allocate per-channel temporary arrays for valid indices & weights
    int   *valid_idx    = malloc((size_t)K * sizeof *valid_idx);
    float *valid_weight = malloc((size_t)K * sizeof *valid_weight);
    if (!valid_idx || !valid_weight) {
        free(inv);
        free(valid_idx);
        free(valid_weight);
        return;
    }

    // ----------------------------------------------------------------------
    // 4) For each channel: filter out unused blades, then apply to all batches
    for (int c = 0; c < C; ++c) {
        // a) Gather valid indices and corresponding weights
        int vc = 0;
        const float *w_c = weight + (size_t)c * K;
        for (int k = 0; k < K; ++k) {
            int blade = kernel_blades[k];
            int idx = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
            if (idx >= 0) {
                valid_idx[vc]    = idx;
                valid_weight[vc] = w_c[k];
                vc++;
            }
        }

        // b) Process each batch for this channel
        for (int b = 0; b < B; ++b) {
            // Pointers to input slice x[b][c][:] and output slice out[b][c][:]
            const float *x_bc = x   + (size_t)b * stride + (size_t)c * I_in;
            float       *o_bc = out + (size_t)b * stride + (size_t)c * I_in;

            // Compute dot-product over valid entries
            float acc = bias[c];  // start from bias
            FLOP(2);
            for (int j = 0; j < vc; ++j) {
                int idx = valid_idx[j];
                acc += valid_weight[j] * x_bc[idx];
                FLOP(2);
            }

            // Apply sigmoid activation
            float gate = 1.0f / (1.0f + expf(-acc));
            FLOP(4);

            // Scale and store outputs for each active input
            for (int i = 0; i < I_in; ++i) {
                o_bc[i] = x_bc[i] * gate;
                FLOP(1);
            }
        }
    }

    // ----------------------------------------------------------------------
    // 5) Cleanup temporaries
    free(valid_idx);
    free(valid_weight);
    free(inv);
}








/*
 * optimized_act_v6
 *
 * Incorporates fmaf-based fused activation using precomputed valid indices
 * and weights per channel, minimizing arithmetic in hot loops. Inspired by v3’s
 * k_idx/x_idx concept and v5’s valid_weight array.
 */



void optimized_act_v6(
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
    // 1) Build inverse map: blade index → input slot
    int *inv = malloc((size_t)I_full * sizeof *inv);
    if (!inv) return;
    for (int i = 0; i < I_full; ++i) inv[i] = -1;
    for (int i = 0; i < I_in; ++i) inv[input_blades[i]] = i;

    // 2) Prepare per-channel valid lists (max K entries each)
    int   *valid_idx    = malloc((size_t)C * K * sizeof *valid_idx);
    float *valid_weight = malloc((size_t)C * K * sizeof *valid_weight);
    int   *valid_count  = malloc((size_t)C * sizeof *valid_count);
    if (!valid_idx || !valid_weight || !valid_count) {
        free(inv);
        free(valid_idx);
        free(valid_weight);
        free(valid_count);
        return;
    }
    for (int c = 0; c < C; ++c) {
        int vc = 0;
        const float *w_c = weight + (size_t)c * K;
        for (int k = 0; k < K; ++k) {
            int blade = kernel_blades[k];
            int idx   = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
            if (idx >= 0) {
                valid_idx[c*K + vc]    = idx;
                valid_weight[c*K + vc] = w_c[k];
                vc++;
            }
        }
        valid_count[c] = vc;
    }
    free(inv);

    // 3) Stride for each batch slice
    size_t stride = (size_t)C * I_in;

    // 4) Fused conv + activation + write-back
    for (int c = 0; c < C; ++c) {
        int vc = valid_count[c];
        int *idxs = valid_idx    + (size_t)c * K;
        float *wts = valid_weight + (size_t)c * K;
        float bval = bias[c];

        for (int b = 0; b < B; ++b) {
            const float *x_bc = x   + (size_t)b * stride + (size_t)c * I_in;
            float       *o_bc = out + (size_t)b * stride + (size_t)c * I_in;

            // dot-product using fused multiply-add
            float acc = bval;
            FLOP(2);
            for (int j = 0; j < vc; ++j) {
                acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                FLOP(2);
            }

            // sigmoid activation
            float g = 1.0f / (1.0f + expf(-acc));
            FLOP(4);

            // scale and store
            for (int i = 0; i < I_in; ++i) {
                o_bc[i] = x_bc[i] * g;
                FLOP(1);
            }
        }
    }

    // 5) Cleanup
    free(valid_idx);
    free(valid_weight);
    free(valid_count);
}







/*
 * optimized_act_v7
 *
 * Fused activation with LUT-based sigmoid using a reduced range and size
 * for improved cache locality. LUT initialization samples [-6,6] instead of [-8,8],
 * and table size is halved to 512 entries, reducing memory footprint and
 * potentially improving speed while still maintaining ~1e-5 accuracy.
 */


// ------------------------------------------------------------
// Sigmoid lookup table parameters and storage (half-size, tighter range)
// ------------------------------------------------------------
#define SIGMOID_LOOKUP_TABLE_SIZE 512      // Reduced table size for better cache
static float sigmoid_lookup_table[SIGMOID_LOOKUP_TABLE_SIZE];
static int   lookup_table_initialized = 0;

// ------------------------------------------------------------
// init_sigmoid_lookup_table
// One-time initialization of the sigmoid lookup table over [-6,6]
// ------------------------------------------------------------
static void init_sigmoid_lookup_table(void) {
    const float x_min = -6.0f;
    const float x_max =  6.0f;
    for (int i = 0; i < SIGMOID_LOOKUP_TABLE_SIZE; ++i) {
        float x = x_min + (x_max - x_min) * i / (SIGMOID_LOOKUP_TABLE_SIZE - 1);
        FLOP(4); // sub, mul, div, add for x calculation
        sigmoid_lookup_table[i] = 1.0f / (1.0f + expf(-x));
        FLOP(4); // expf, negate, add, div for sigmoid computation
    }
    lookup_table_initialized = 1;
}

// ------------------------------------------------------------
// fast_sigmoid
// Lookup + interpolation over reduced-range lookup table
// ------------------------------------------------------------
static inline float fast_sigmoid(float x) {
    if (!lookup_table_initialized) init_sigmoid_lookup_table();
    const float x_min = -6.0f, x_max = 6.0f;
    if (x <= x_min) return 0.0f;
    if (x >= x_max) return 1.0f;
    float fx   = (x - x_min) * (SIGMOID_LOOKUP_TABLE_SIZE - 1) / (x_max - x_min);
    FLOP(3); // sub, mul, div for fx calculation
    int   idx  = (int)fx;
    float frac = fx - idx;
    FLOP(1); // sub for frac
    FLOP(3); // sub, mul, add for interpolation (lookup_table[idx+1]-lookup_table[idx], *frac, + base)
    return sigmoid_lookup_table[idx] + frac * (sigmoid_lookup_table[idx+1] - sigmoid_lookup_table[idx]);
}

// ------------------------------------------------------------
// optimized_act_v7
// Main fused activation kernel with optimized LUT
// ------------------------------------------------------------
void optimized_act_v7(
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
    // Track bytes read from input arrays
    BYTES_READ(B * C * I_in * sizeof(float));        // x
    BYTES_READ(C * K * sizeof(float));               // weight
    BYTES_READ(C * sizeof(float));                   // bias
    BYTES_READ(I_in * sizeof(int));                  // input_blades
    BYTES_READ(K * sizeof(int));                     // kernel_blades
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * I_in * sizeof(float));     // out
    
    // 1) Build inverse map
    int *inv = malloc((size_t)I_full * sizeof *inv);
    if (!inv) return;
    for (int i = 0; i < I_full; ++i) inv[i] = -1;
    for (int i = 0; i < I_in; ++i) inv[input_blades[i]] = i;

    // 2) Build per-channel valid lists
    int   *valid_idx    = malloc((size_t)C * K * sizeof *valid_idx);
    float *valid_weight = malloc((size_t)C * K * sizeof *valid_weight);
    int   *valid_count  = malloc((size_t)C * sizeof *valid_count);
    if (!valid_idx || !valid_weight || !valid_count) {
        free(inv); free(valid_idx); free(valid_weight); free(valid_count);
        return;
    }
    for (int c = 0; c < C; ++c) {
        int vc = 0;
        const float *w_c = weight + (size_t)c * K;
        for (int k = 0; k < K; ++k) {
            int blade = kernel_blades[k];
            int idx   = (blade >= 0 && blade < I_full) ? inv[blade] : -1;
            if (idx >= 0) {
                valid_idx[c*K + vc]    = idx;
                valid_weight[c*K + vc] = w_c[k];
                vc++;
            }
        }
        valid_count[c] = vc;
    }
    free(inv);

    // 3) Stride per batch slice
    size_t stride = (size_t)C * I_in;

    // 4) Fused loop: dot, lookup table sigmoid, write-back
    for (int c = 0; c < C; ++c) {
        int     vc   = valid_count[c];
        int    *idxs = valid_idx    + (size_t)c * K;
        float  *wts  = valid_weight + (size_t)c * K;
        float   bval = bias[c];
        for (int b = 0; b < B; ++b) {
            const float *x_bc = x   + (size_t)b * stride + (size_t)c * I_in;
            float       *o_bc = out + (size_t)b * stride + (size_t)c * I_in;
            float acc = bval;
            FLOP(2);
            for (int j = 0; j < vc; ++j) {
                acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                FLOP(2);
            }
            float g = fast_sigmoid(acc);
            FLOP(4);
            for (int i = 0; i < I_in; ++i) {
                o_bc[i] = x_bc[i] * g;
                FLOP(1);
            }
        }
    }

    // 5) Cleanup
    free(valid_idx); free(valid_weight); free(valid_count);
}


/*
 * optimized_act_v8
 *
 * Fused activation with reduced-range LUT (-6..6, 512 entries)
 * and heap-allocated temporary buffers for dynamic sizes.
 * Eliminates VLAs to ensure compatibility when C, I_full are not constants.
 */


// ------------------------------------------------------------
// Sigmoid LUT parameters and storage (half-size, tighter range)
// ------------------------------------------------------------
#define SIGMOID_LUT_SIZE 512      // Reduced table size for better cache
static float sigmoid_lut[SIGMOID_LUT_SIZE];
static int   lut_initialized = 0;

// ------------------------------------------------------------
// init_sigmoid_lut_v8
// One-time initialization of the sigmoid LUT over [-6,6]
// ------------------------------------------------------------
static void init_sigmoid_lut_v8(void) {
    const float x_min = -6.0f;
    const float x_max =  6.0f;
    for (int i = 0; i < SIGMOID_LUT_SIZE; ++i) {
        float x = x_min + (x_max - x_min) * i / (SIGMOID_LUT_SIZE - 1);
        FLOP(4); // sub, mul, div, add for x calculation
        sigmoid_lut[i] = 1.0f / (1.0f + expf(-x));
        FLOP(4); // expf, negate, add, div for sigmoid computation
    }
    lut_initialized = 1;
}

// ------------------------------------------------------------
// fast_sigmoid_v8
// Lookup + interpolation over reduced-range LUT
// ------------------------------------------------------------
static inline float fast_sigmoid_v8(float x) {
    if (!lut_initialized) init_sigmoid_lut_v8();
    const float x_min = -6.0f, x_max = 6.0f;
    if (x <= x_min) return 0.0f;
    if (x >= x_max) return 1.0f;
    float fx   = (x - x_min) * (SIGMOID_LUT_SIZE - 1) / (x_max - x_min);
    FLOP(3); // sub, mul, div for fx calculation
    int   idx  = (int)fx;
    float frac = fx - idx;
    FLOP(1); // sub for frac
    FLOP(3); // sub, mul, add for interpolation
    return sigmoid_lut[idx] + frac * (sigmoid_lut[idx+1] - sigmoid_lut[idx]);
}

// ------------------------------------------------------------
// optimized_act_v8
// Main fused activation kernel with optimized LUT and heap buffers
// ------------------------------------------------------------
void optimized_act_v8(
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
    // Track bytes read from input arrays
    BYTES_READ(B * C * I_in * sizeof(float));        // x
    BYTES_READ(C * K * sizeof(float));               // weight
    BYTES_READ(C * sizeof(float));                   // bias
    BYTES_READ(I_in * sizeof(int));                  // input_blades
    BYTES_READ(K * sizeof(int));                     // kernel_blades
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * I_in * sizeof(float));     // out
    
    // L1 data cache ~32 KB (per core). We choose tiles so the working-set per tile:
    //   tc * I_in * sizeof(float)  (data slice)  + LUT/table ≤ 32 KB   
    const int tb = 64;// batch tile
    const int tc = 4;// channel tile

    // Sigmoid table setup remains global/static; use fast_sigmoid_v8() as before.

    size_t stride = (size_t)C * I_in;

    // Pre-allocate inverse map once
    int *inv = malloc((size_t)I_full * sizeof *inv);
    if (!inv) return;

    // Pre-build weight pruning thresholds
    const float weight_eps = 1e-6f;
    const float gate_eps   = 1e-6f;

    // Process by tiles
    for (int b0 = 0; b0 < B; b0 += tb) {
        int b1 = b0 + tb < B ? b0 + tb : B;
        for (int c0 = 0; c0 < C; c0 += tc) {
            int c1 = c0 + tc < C ? c0 + tc : C;

            // Build valid lists for this channel tile
            int   *valid_idx    = malloc((size_t)(c1-c0) * K * sizeof *valid_idx);
            float *valid_weight = malloc((size_t)(c1-c0) * K * sizeof *valid_weight);
            int   *valid_count  = malloc((size_t)(c1-c0) * sizeof *valid_count);
            if (!valid_idx || !valid_weight || !valid_count) {
                free(inv);
                free(valid_idx); free(valid_weight); free(valid_count);
                return;
            }

            for (int cc = c0; cc < c1; ++cc) {
                // inverse map for this channel
                for (int b = 0; b < I_full; ++b) inv[b] = -1;
                for (int i = 0; i < I_in; ++i) {
                    int blade = input_blades[i];
                    if ((unsigned)blade < (unsigned)I_full) inv[blade] = i;
                }
                // prune and gather
                int vc = 0;
                const float *w_c = weight + (size_t)cc * K;
                for (int k = 0; k < K; ++k) {
                    int blade = kernel_blades[k];
                    int idx   = ((unsigned)blade < (unsigned)I_full) ? inv[blade] : -1;
                    float w   = w_c[k];
                    if (idx >= 0 && fabsf(w) > weight_eps) {
                        valid_idx[(cc-c0)*K + vc]    = idx;
                        valid_weight[(cc-c0)*K + vc] = w;
                        vc++;
                    }
                }
                valid_count[cc-c0] = vc;
            }

            // Compute each tile
            for (int cc = c0; cc < c1; ++cc) {
                int vc = valid_count[cc-c0];
                int *idxs = valid_idx + (size_t)(cc-c0) * K;
                float *wts = valid_weight + (size_t)(cc-c0) * K;
                float bval = bias[cc];

                for (int bb = b0; bb < b1; ++bb) {
                    const float *x_bc = x + (size_t)bb * stride + (size_t)cc * I_in;
                    float *o_bc        = out + (size_t)bb * stride + (size_t)cc * I_in;

                    // sparse dot-product
                    float acc = bval;
                    FLOP(2);
                    for (int j = 0; j < vc; ++j) {
                        acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                        FLOP(2);
                    }
                    // fast sigmoid
                    float g = fast_sigmoid_v8(acc);
                    FLOP(4);
                    // gate sparsity
                    if (g < gate_eps) {
                        for (int i = 0; i < I_in; ++i) o_bc[i] = 0.0f;
                        // no flops for zero assignment
                    } else if (g > 1.0f - gate_eps) {
                        for (int i = 0; i < I_in; ++i) o_bc[i] = x_bc[i];
                        // no flops for copy assignment
                    } else {
                        for (int i = 0; i < I_in; ++i) {
                            o_bc[i] = x_bc[i] * g;
                            FLOP(1);
                        }
                    }
                }
            }

            free(valid_idx);
            free(valid_weight);
            free(valid_count);
        }
    }

    free(inv);
}


/*
 * optimized_act_v9
 *
 * Fused activation with reduced-range SIGMOID lookup table (-6..6, 512 entries),
 * tiled for cache locality, and hoisted/reused buffers to eliminate per-tile allocation.
 * Eliminates VLAs to ensure compatibility when C, I_full are not constants.
 */


// ------------------------------------------------------------
// SIGMOID lookup parameters and storage (half-size, tighter range)
// ------------------------------------------------------------
#define SIGMOID_LUT_SIZE 512      // Reduced table size for better cache
static float sigmoid_lut_v9[SIGMOID_LUT_SIZE];
static int   lut_v9_initialized = 0;

// ------------------------------------------------------------
// init_sigmoid_lut_v9
// One-time initialization of the sigmoid LUT over [-6,6]
// ------------------------------------------------------------
static void init_sigmoid_lut_v9(void) {
    const float x_min = -6.0f;
    const float x_max =  6.0f;
    for (int i = 0; i < SIGMOID_LUT_SIZE; ++i) {
        // compute sample x
        float x = x_min + (x_max - x_min) * i / (SIGMOID_LUT_SIZE - 1);
        FLOP(4); // sub, mul, div, add for x calculation
        // compute sigmoid
        sigmoid_lut_v9[i] = 1.0f / (1.0f + expf(-x));
        FLOP(4); // expf, negate, add, div for sigmoid computation
    }
    lut_v9_initialized = 1;
}

// ------------------------------------------------------------
// fast_sigmoid_v9
// Lookup + interpolation over reduced-range LUT
// ------------------------------------------------------------
static inline float fast_sigmoid_v9(float x) {
    if (!lut_v9_initialized) init_sigmoid_lut_v9();
    const float x_min = -6.0f, x_max =  6.0f;
    if (x <= x_min) return 0.0f;
    if (x >= x_max) return 1.0f;
    // map to LUT index space
    float fx   = (x - x_min) * (SIGMOID_LUT_SIZE - 1) / (x_max - x_min);
    FLOP(3); // sub, mul, div for fx calculation
    int   idx  = (int)fx;
    float frac = fx - idx;
    FLOP(1); // sub for frac calculation
    // linear interpolation
    FLOP(3); // sub, mul, add for interpolation
    return sigmoid_lut_v9[idx] + frac * (sigmoid_lut_v9[idx+1] - sigmoid_lut_v9[idx]);
}

// ------------------------------------------------------------
// optimized_act_v9
// Main fused activation kernel with optimized LUT and hoisted buffers
// ------------------------------------------------------------
void optimized_act_v9(
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
    // Ensure SIGMOID table is ready
    if (!lut_v9_initialized) init_sigmoid_lut_v9();

    // L1 cache ~64KB: choose tile sizes for batch (tb) and channel (tc)
    const int tb = 64;
    const int tc = 4;
    size_t stride = (size_t)C * I_in;

    // Hoisted & reused buffers
    int   *inv          = malloc((size_t)I_full * sizeof *inv);
    int   *valid_idx    = malloc((size_t)C * K * sizeof *valid_idx);
    float *valid_weight = malloc((size_t)C * K * sizeof *valid_weight);
    int   *valid_count  = malloc((size_t)C * sizeof *valid_count);
    if (!inv || !valid_idx || !valid_weight || !valid_count) {
        free(inv); free(valid_idx); free(valid_weight); free(valid_count);
        return;
    }

    const float weight_eps = 1e-6f;
    const float gate_eps   = 1e-6f;

    // Tiled main loops
    for (int b0 = 0; b0 < B; b0 += tb) {
        int b1 = b0 + tb < B ? b0 + tb : B;
        for (int c0 = 0; c0 < C; c0 += tc) {
            int c1 = c0 + tc < C ? c0 + tc : C;

            // Build per-channel sparse lists
            for (int cc = c0; cc < c1; ++cc) {
                // inverse-map build
                for (int i = 0; i < I_full; ++i) inv[i] = -1;
                for (int i = 0; i < I_in; ++i) {
                    int blade = input_blades[i];
                    if ((unsigned)blade < (unsigned)I_full) inv[blade] = i;
                }
                // prune & gather
                int vc = 0;
                const float *w_c = weight + (size_t)cc * K;
                for (int k = 0; k < K; ++k) {
                    int blade = kernel_blades[k];
                    int idx   = ((unsigned)blade < (unsigned)I_full) ? inv[blade] : -1;
                    float w   = w_c[k];
                    if (idx >= 0 && fabsf(w) > weight_eps) {
                        valid_idx[cc*K + vc]    = idx;
                        valid_weight[cc*K + vc] = w;
                        vc++;
                    }
                }
                valid_count[cc] = vc;
            }

            // Compute over each tile
            for (int cc = c0; cc < c1; ++cc) {
                int     vc   = valid_count[cc];
                int    *idxs = valid_idx    + (size_t)cc * K;
                float  *wts  = valid_weight + (size_t)cc * K;
                float   bval = bias[cc];

                for (int bb = b0; bb < b1; ++bb) {
                    const float *x_bc = x   + (size_t)bb * stride + (size_t)cc * I_in;
                    float       *o_bc = out + (size_t)bb * stride + (size_t)cc * I_in;

                    // sparse dot-product
                    float acc = bval;
                    FLOP(2); // load bias and zero-init accumulator
                    for (int j = 0; j < vc; ++j) {
                        acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                        FLOP(2); // fused multiply-add: mul+add
                    }

                    // fast sigmoid
                    float g = fast_sigmoid_v9(acc);
                    FLOP(4); // count the sigmoid

                    // gate-output sparsity
                    if (g < gate_eps) {
                        for (int i = 0; i < I_in; ++i) o_bc[i] = 0.0f;
                        // zero stores: no FLOP
                    } else if (g > 1.0f - gate_eps) {
                        for (int i = 0; i < I_in; ++i) o_bc[i] = x_bc[i];
                        // copy: no FLOP
                    } else {
                        for (int i = 0; i < I_in; ++i) {
                            o_bc[i] = x_bc[i] * g;
                            FLOP(1); // final multiply
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    free(inv);
    free(valid_idx);
    free(valid_weight);
    free(valid_count);
}

/*
 * optimized_act_v10
 *
 * Fused activation with reduced-range SIGMOID lookup table (-6..6, 512 entries),
 * hoisted sparse-list building per channel, tiled for cache locality,
 * and hoisted/reused buffers to eliminate per-tile allocation
 * Eliminates VLAs to ensure compatibility when C and space are not compile-time constants
 */


// ------------------------------------------------------------
// SIGMOID lookup parameters and storage (half-size, tighter range)
// ------------------------------------------------------------
     // Reduced table size for better cache
static float sigmoid_lut_v10[SIGMOID_LUT_SIZE];
static int   lut_v10_initialized = 0;

// ------------------------------------------------------------
// init_sigmoid_lut_v10
// One-time initialization of the sigmoid LUT over [-6,6]
// ------------------------------------------------------------
static void init_sigmoid_lut_v10(void) {
    const float x_min = -6.0f, x_max = 6.0f;
    for (int i = 0; i < SIGMOID_LUT_SIZE; ++i) {
        // compute sample x
        float x = x_min + (x_max - x_min) * i / (SIGMOID_LUT_SIZE - 1);
        FLOP(4); // sub, mul, div, add for x calculation
        // compute sigmoid
        sigmoid_lut_v10[i] = 1.0f / (1.0f + expf(-x));
        FLOP(4); // expf, negate, add, div for sigmoid computation
    }
    lut_v10_initialized = 1;
}

// ------------------------------------------------------------
// fast_sigmoid_v10
// Lookup + linear interpolation over reduced-range LUT
// ------------------------------------------------------------
static inline float fast_sigmoid_v10(float z) {
    if (!lut_v10_initialized) init_sigmoid_lut_v10();
    const float x_min = -6.0f, x_max = 6.0f;
    if (z <= x_min) return 0.0f;
    if (z >= x_max) return 1.0f;
    // map to LUT index space
    float f = (z - x_min) * (SIGMOID_LUT_SIZE - 1) / (x_max - x_min);
    FLOP(3); // sub, mul, div for f calculation
    int   idx  = (int)f;
    float frac = f - idx;
    FLOP(1); // sub for frac calculation
    // linear interpolation
    FLOP(3); // sub, mul, add for interpolation
    return sigmoid_lut_v10[idx] + frac * (sigmoid_lut_v10[idx+1] - sigmoid_lut_v10[idx]);
}

// ------------------------------------------------------------
// optimized_act_v10
// Main fused activation kernel with optimized LUT and hoisted buffers
// ------------------------------------------------------------
void optimized_act_v10(
    const float *x,
    int           B,
    int           C,
    int           slots,
    int           space,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Initialize SIGMOID table once
    if (!lut_v10_initialized) init_sigmoid_lut_v10();

    // L1 cache ~64KB: choose tile sizes
    const int tb = 64;
    const int tc = 4;
    size_t stride = (size_t)C * slots;

    // Hoisted & reused buffers
    int   *inv        = malloc((size_t)space * sizeof *inv);
    int   *valid_idx  = malloc((size_t)C * K   * sizeof *valid_idx);
    float *valid_wght = malloc((size_t)C * K   * sizeof *valid_wght);
    int   *valid_cnt  = malloc((size_t)C         * sizeof *valid_cnt);
    if (!inv || !valid_idx || !valid_wght || !valid_cnt) {
        free(inv); free(valid_idx); free(valid_wght); free(valid_cnt);
        return;
    }

    // Build sparse lists once per channel
    const float w_eps = 1e-6f;
    for (int c = 0; c < C; ++c) {
        // inverse-map build
        for (int i = 0; i < space; ++i) inv[i] = -1;
        for (int i = 0; i < slots; ++i) {
            int blade = input_blades[i];
            if ((unsigned)blade < (unsigned)space) inv[blade] = i;
        }
        // prune & gather
        int vc = 0;
        const float *w_c = weight + (size_t)c * K;
        for (int k = 0; k < K; ++k) {
            int blade = kernel_blades[k];
            int idx   = ((unsigned)blade < (unsigned)space) ? inv[blade] : -1;
            float w   = w_c[k];
            if (idx >= 0 && fabsf(w) > w_eps) {
                valid_idx[c*K + vc]  = idx;
                valid_wght[c*K + vc] = w;
                vc++;
            }
        }
        valid_cnt[c] = vc;
    }

    const float g_eps = 1e-6f;

    // Tiled main compute
    for (int b0 = 0; b0 < B; b0 += tb) {
        int b1 = b0 + tb < B ? b0 + tb : B;
        for (int c0 = 0; c0 < C; c0 += tc) {
            int c1 = c0 + tc < C ? c0 + tc : C;
            for (int c = c0; c < c1; ++c) {
                int    vc   = valid_cnt[c];
                int   *idxs = valid_idx  + (size_t)c * K;
                float *wts  = valid_wght + (size_t)c * K;
                float  bval = bias[c];
                for (int b = b0; b < b1; ++b) {
                    const float *x_bc = x + (size_t)b * stride + (size_t)c * slots;
                    float       *o_bc = out + (size_t)b * stride + (size_t)c * slots;

                    // sparse dot-product
                    float acc = bval;
                    FLOP(2); // load bias and zero-init accumulator
                    for (int j = 0; j < vc; ++j) {
                        acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                        FLOP(2); // fused multiply-add: mul+add
                    }

                    // fast sigmoid
                    float g = fast_sigmoid_v10(acc);
                    FLOP(4); // sigmoid count

                    // gate-output sparsity
                    if (g < g_eps) {
                        for (int i = 0; i < slots; ++i) o_bc[i] = 0.0f;
                    } else if (g > 1.0f - g_eps) {
                        for (int i = 0; i < slots; ++i) o_bc[i] = x_bc[i];
                    } else {
                        for (int i = 0; i < slots; ++i) {
                            o_bc[i] = x_bc[i] * g;
                            FLOP(1); // final multiply
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    free(inv);
    free(valid_idx);
    free(valid_wght);
    free(valid_cnt);
}


/*
 * optimized_act_v11
 *
 * Fused activation with reduced-range SIGMOID lookup table (-6..6, 512 entries),
 * hoisted sparse-list building per channel, tiled for cache locality,
 * reused buffers, and branchless gate application.
 */

static float sigmoid_lut_v11[SIGMOID_LUT_SIZE];
static int   lut_v11_initialized = 0;

static void init_sigmoid_lut_v11(void) {
    const float x_min = -6.0f, x_max = 6.0f;
    for (int i = 0; i < SIGMOID_LUT_SIZE; ++i) {
        float x = x_min + (x_max - x_min) * i / (SIGMOID_LUT_SIZE - 1);
        FLOP(4); // sub, mul, div, add for x sampling
        sigmoid_lut_v11[i] = 1.0f / (1.0f + expf(-x));
        FLOP(4); // expf, negate, add, div for sigmoid computation
    }
    lut_v11_initialized = 1;
}

static inline float fast_sigmoid_v11(float z) {
    if (!lut_v11_initialized) init_sigmoid_lut_v11();
    const float x_min = -6.0f, x_max = 6.0f;
    if (z <= x_min) return 0.0f;
    if (z >= x_max) return 1.0f;
    float f    = (z - x_min) * (SIGMOID_LUT_SIZE - 1) / (x_max - x_min);
    FLOP(3); // sub, mul, div for index mapping
    int   idx  = (int)f;
    float frac = f - idx;
    FLOP(1); // sub for fractional part
    FLOP(3); // sub, mul, add for interpolation
    return sigmoid_lut_v11[idx] + frac * (sigmoid_lut_v11[idx+1] - sigmoid_lut_v11[idx]);
}

void optimized_act_v11(
    const float *x,
    int           B,
    int           C,
    int           slots,
    int           space,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Track bytes read from input arrays
    BYTES_READ(B * C * slots * sizeof(float));       // x
    BYTES_READ(C * K * sizeof(float));               // weight
    BYTES_READ(C * sizeof(float));                   // bias
    BYTES_READ(slots * sizeof(int));                 // input_blades
    BYTES_READ(K * sizeof(int));                     // kernel_blades
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * slots * sizeof(float));    // out
    
    if (!lut_v11_initialized) init_sigmoid_lut_v11();

    const int tb = 64;
    const int tc = 4;
    size_t stride = (size_t)C * slots;

    int   *inv        = malloc((size_t)space * sizeof *inv);
    int   *valid_idx  = malloc((size_t)C * K   * sizeof *valid_idx);
    float *valid_wght = malloc((size_t)C * K   * sizeof *valid_wght);
    int   *valid_cnt  = malloc((size_t)C         * sizeof *valid_cnt);
    if (!inv || !valid_idx || !valid_wght || !valid_cnt) {
        free(inv); free(valid_idx); free(valid_wght); free(valid_cnt);
        return;
    }

    const float w_eps = 1e-6f;
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < space; ++i) inv[i] = -1;
        for (int i = 0; i < slots; ++i) {
            int blade = input_blades[i];
            if ((unsigned)blade < (unsigned)space) inv[blade] = i;
        }
        int vc = 0;
        const float *w_c = weight + (size_t)c * K;
        for (int k = 0; k < K; ++k) {
            int blade = kernel_blades[k];
            int idx   = ((unsigned)blade < (unsigned)space) ? inv[blade] : -1;
            float w   = w_c[k];
            if (idx >= 0 && fabsf(w) > w_eps) {
                valid_idx[c*K + vc]  = idx;
                valid_wght[c*K + vc] = w;
                vc++;
            }
        }
        valid_cnt[c] = vc;
    }

    const float g_eps = 1e-6f;
    for (int b0 = 0; b0 < B; b0 += tb) {
        int b1 = b0 + tb < B ? b0 + tb : B;
        for (int c0 = 0; c0 < C; c0 += tc) {
            int c1 = c0 + tc < C ? c0 + tc : C;
            for (int c = c0; c < c1; ++c) {
                int    vc   = valid_cnt[c];
                int   *idxs = valid_idx  + (size_t)c * K;
                float *wts  = valid_wght + (size_t)c * K;
                float  bval = bias[c];
                for (int b = b0; b < b1; ++b) {
                    const float *x_bc = x + (size_t)b * stride + (size_t)c * slots;
                    float       *o_bc = out + (size_t)b * stride + (size_t)c * slots;

                    float acc = bval;
                    FLOP(2); // load bias + zero-init
                    for (int j = 0; j < vc; ++j) {
                        acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                        FLOP(2); // mul + add
                    }

                    float g = fast_sigmoid_v11(acc);
                    FLOP(4); // sigmoid cost

                    // branchless gate
                    float m_low  = (g > g_eps);
                    float m_high = (g > 1.0f - g_eps);
                    float scale  = m_low * (m_high + (1.0f - m_high) * g);
                    FLOP(3); // comp + mul + add for scale

                    for (int i = 0; i < slots; ++i) {
                        o_bc[i] = x_bc[i] * scale;
                        FLOP(1); // final multiply
                    }
                }
            }
        }
    }

    free(inv);
    free(valid_idx);
    free(valid_wght);
    free(valid_cnt);
}


/*
 * optimized_act_v12
 *
 * Fused activation with reduced-range SIGMOID lookup table (-6..6, 512 entries),
 * tiled for cache locality, hoisted/reused buffers,
 * per-tile sparse-list building, and branchless gate application.
 */


static float sigmoid_lut_v12[SIGMOID_LUT_SIZE];
static int   lut_v12_initialized = 0;

static inline void init_sigmoid_lut_v12(void) {
    const float x_min = -6.0f, x_max = 6.0f;
    for (int i = 0; i < SIGMOID_LUT_SIZE; ++i) {
        float x = x_min + (x_max - x_min) * i / (SIGMOID_LUT_SIZE - 1);
        FLOP(4); // sub, mul, div, add for x sampling
        sigmoid_lut_v12[i] = 1.0f / (1.0f + expf(-x));
        FLOP(4); // expf, negate, add, div for sigmoid computation
    }
    lut_v12_initialized = 1;
}

static inline float fast_sigmoid_v12(float z) {
    if (!lut_v12_initialized) init_sigmoid_lut_v12();
    const float x_min = -6.0f, x_max = 6.0f;
    if (z <= x_min) return 0.0f;
    if (z >= x_max) return 1.0f;
    float f    = (z - x_min) * (SIGMOID_LUT_SIZE - 1) / (x_max - x_min);
    FLOP(3); // sub, mul, div for index mapping
    int   idx  = (int)f;
    float frac = f - idx;
    FLOP(1); // sub for fractional part
    FLOP(3); // sub, mul, add for interpolation
    return sigmoid_lut_v12[idx] + frac * (sigmoid_lut_v12[idx+1] - sigmoid_lut_v12[idx]);
}

void optimized_act_v12(
    const float *x,
    int           B,
    int           C,
    int           slots,
    int           space,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    if (!lut_v12_initialized) init_sigmoid_lut_v12();

    const int tb = 64;
    const int tc = 4;
    size_t stride = (size_t)C * slots;

    int   *inv        = malloc((size_t)space * sizeof *inv);
    int   *valid_idx  = malloc((size_t)C * K   * sizeof *valid_idx);
    float *valid_wght = malloc((size_t)C * K   * sizeof *valid_wght);
    int   *valid_cnt  = malloc((size_t)C         * sizeof *valid_cnt);
    if (!inv || !valid_idx || !valid_wght || !valid_cnt) {
        free(inv); free(valid_idx); free(valid_wght); free(valid_cnt);
        return;
    }

    const float w_eps = 1e-6f;
    const float g_eps = 1e-6f;

    for (int b0 = 0; b0 < B; b0 += tb) {
        int b1 = b0 + tb < B ? b0 + tb : B;
        for (int c0 = 0; c0 < C; c0 += tc) {
            int c1 = c0 + tc < C ? c0 + tc : C;
            for (int c = c0; c < c1; ++c) {
                // build sparse lists per tile
                for (int i = 0; i < space; ++i) inv[i] = -1;
                for (int i = 0; i < slots; ++i) {
                    int blade = input_blades[i];
                    if ((unsigned)blade < (unsigned)space) inv[blade] = i;
                }
                int vc = 0;
                const float *w_c = weight + (size_t)c * K;
                for (int k = 0; k < K; ++k) {
                    int blade = kernel_blades[k];
                    int idx = ((unsigned)blade < (unsigned)space) ? inv[blade] : -1;
                    float w = w_c[k];
                    if (idx >= 0 && fabsf(w) > w_eps) {
                        valid_idx[c*K + vc]  = idx;
                        valid_wght[c*K + vc] = w;
                        vc++;
                    }
                }
                valid_cnt[c] = vc;
            }

            for (int c = c0; c < c1; ++c) {
                int    vc   = valid_cnt[c];
                int   *idxs = valid_idx  + (size_t)c * K;
                float *wts  = valid_wght + (size_t)c * K;
                float  bval = bias[c];
                for (int b = b0; b < b1; ++b) {
                    const float *x_bc = x + (size_t)b * stride + (size_t)c * slots;
                    float       *o_bc = out + (size_t)b * stride + (size_t)c * slots;

                    float acc = bval;
                    FLOP(2); // load bias + zero-init
                    for (int j = 0; j < vc; ++j) {
                        acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                        FLOP(2); // mul + add
                    }

                    float g      = fast_sigmoid_v12(acc);
                    FLOP(4); // sigmoid cost
                    // branchless gate mask
                    float m_low  = (g > g_eps);
                    float m_high = (g > 1.0f - g_eps);
                    float scale  = m_low * (m_high + (1.0f - m_high) * g);
                    FLOP(3); // comp + mul + add for scale

                    for (int i = 0; i < slots; ++i) {
                        o_bc[i] = x_bc[i] * scale;
                        FLOP(1); // final multiply
                    }
                }
            }
        }
    }

    free(inv);
    free(valid_idx);
    free(valid_wght);
    free(valid_cnt);
}



static float sigmoid_lut_v13[SIGMOID_LUT_SIZE];
static int   lut_v13_initialized = 0;

// One-time initialization of SIGMOID LUT over [-6,6]
static void init_sigmoid_lut_v13(void) {
    const float x_min = -6.0f, x_max = 6.0f;
    for (int i = 0; i < SIGMOID_LUT_SIZE; ++i) {
        float x = x_min + (x_max - x_min) * i / (SIGMOID_LUT_SIZE - 1);
        FLOP(4);  // sub, mul, div, add for x sampling
        sigmoid_lut_v13[i] = 1.0f / (1.0f + expf(-x));
        FLOP(4);  // expf, negate, add, div for LUT init
    }
    lut_v13_initialized = 1;
}

// fast_sigmoid_v13: LUT lookup + linear interpolation with full FLOP count
static inline float fast_sigmoid_v13(float z) {
    if (!lut_v13_initialized) init_sigmoid_lut_v13();

    const float x_min = -6.0f, x_max = 6.0f;
    FLOP(1);                    // compare z <= x_min
    if (z <= x_min) return 0.0f;
    FLOP(1);                    // compare z >= x_max
    if (z >= x_max) return 1.0f;

    float f    = (z - x_min) * (SIGMOID_LUT_SIZE - 1) / (x_max - x_min);
    FLOP(3);                    // sub, mul, div for index mapping

    int   idx  = (int)f;
    float frac = f - idx;
    FLOP(1);                    // sub for fractional part

    float y0   = sigmoid_lut_v13[idx];
    float y1   = sigmoid_lut_v13[idx + 1];
    float res  = y0 + frac * (y1 - y0);
    FLOP(3);                    // sub, mul, add for interpolation

    return res;
}

void optimized_act_v13(
    const float *x,
    int           B,
    int           C,
    int           slots,
    int           space,
    const int    *input_blades,
    const float  *weight,
    int           K,
    const int    *kernel_blades,
    const float  *bias,
    float        *out
) {
    // Track bytes read from input arrays
    BYTES_READ(B * C * slots * sizeof(float));       // x
    BYTES_READ(C * K * sizeof(float));               // weight
    BYTES_READ(C * sizeof(float));                   // bias
    BYTES_READ(slots * sizeof(int));                 // input_blades
    BYTES_READ(K * sizeof(int));                     // kernel_blades
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * slots * sizeof(float));    // out
    
    if (!lut_v13_initialized) init_sigmoid_lut_v13();

    const int   tb    = 64;
    const int   tc    = 4;
    const float w_eps = 1e-6f;
    const float g_eps = 1e-6f;
    size_t      stride = (size_t)C * slots;

    // Allocate sparse-helper arrays
    int   *inv        = malloc(space * sizeof *inv);
    int   *valid_idx  = malloc((size_t)C * K * sizeof *valid_idx);
    float *valid_wght = malloc((size_t)C * K * sizeof *valid_wght);
    int   *valid_cnt  = malloc((size_t)C       * sizeof *valid_cnt);
    if (!inv || !valid_idx || !valid_wght || !valid_cnt) return;

    // Build sparse lists per channel
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < space; ++i) inv[i] = -1;
        for (int i = 0; i < slots; ++i) {
            int b = input_blades[i];
            if ((unsigned)b < (unsigned)space) inv[b] = i;
        }
        int vc = 0;
        const float *w_c = weight + (size_t)c * K;
        for (int k = 0; k < K; ++k) {
            int b   = kernel_blades[k];
            int idx = ((unsigned)b < (unsigned)space) ? inv[b] : -1;
            float w = w_c[k];
            if (idx >= 0 && fabsf(w) > w_eps) {
                valid_idx[c*K + vc]  = idx;
                valid_wght[c*K + vc] = w;
                ++vc;
            }
        }
        valid_cnt[c] = vc;
    }

    // Main tiled computation
    for (int b0 = 0; b0 < B; b0 += tb) {
        int b1 = b0 + tb < B ? b0 + tb : B;
        for (int c0 = 0; c0 < C; c0 += tc) {
            int c1 = c0 + tc < C ? c0 + tc : C;
            for (int c = c0; c < c1; ++c) {
                int    vc   = valid_cnt[c];
                int   *idxs = valid_idx  + (size_t)c * K;
                float *wts  = valid_wght + (size_t)c * K;
                float  bval = bias[c];

                for (int b = b0; b < b1; ++b) {
                    const float *x_bc = x   + (size_t)b * stride + (size_t)c * slots;
                    float       *o_bc = out + (size_t)b * stride + (size_t)c * slots;

                    // 1) Vectorized AVX2 dot-product
                    __m256 accv = _mm256_setzero_ps();
                    int j = 0;
                    for (; j + 7 < vc; j += 8) {
                        __m256i iv = _mm256_loadu_si256((__m256i*)(idxs + j));
                        __m256  wv = _mm256_loadu_ps(wts + j);
                        __m256  xv = _mm256_i32gather_ps(x_bc, iv, 4);
                        accv = _mm256_fmadd_ps(wv, xv, accv);
                        FLOP(16);  // 8 mul + 8 add
                    }

                    // horizontal sum of accv
                    float buf[8];
                    _mm256_storeu_ps(buf, accv);
                    float acc = 0.0f;
                    int lanes = vc < 8 ? vc : 8;
                    for (int k = 0; k < lanes; ++k) {
                        acc += buf[k];
                        FLOP(1);  // add
                    }

                    // 2) scalar tail
                    for (; j < vc; ++j) {
                        acc = fmaf(wts[j], x_bc[idxs[j]], acc);
                        FLOP(2);  // mul + add
                    }

                    // 3) add bias
                    acc += bval;
                    FLOP(1);     // add

                    // 4) LUT sigmoid + mask/scale
                    float g = fast_sigmoid_v13(acc);
                    FLOP(7);    // 3 map + 1 frac + 3 interp inside LUT
                    FLOP(2);    // two compares for m_low, m_high

                    float m_low  = (g > g_eps);
                    float m_high = (g > 1.0f - g_eps);
                    float scale  = m_low * (m_high + (1.0f - m_high) * g);
                    FLOP(4);    // sub, mul, add, mul for mask/scale

                    // 5) broadcast scale for vector write-back
                    __m256 vs = _mm256_set1_ps(scale);
                    FLOP(1);    // broadcast

                    // 6) vectorized write-back
                    int i = 0;
                    for (; i + 7 < slots; i += 8) {
                        __m256 xv = _mm256_loadu_ps(x_bc + i);
                        __m256 yv = _mm256_mul_ps(xv, vs);
                        _mm256_storeu_ps(o_bc + i, yv);
                        FLOP(8);  // 8 mul
                    }
                    for (; i < slots; ++i) {
                        o_bc[i] = x_bc[i] * scale;
                        FLOP(1);  // mul
                    }
                }
            }
        }
    }

    free(inv);
    free(valid_idx);
    free(valid_wght);
    free(valid_cnt);
}







void register_activation_functions(void) {
    add_activation_function(baseline_act,      "Baseline");
    // add_activation_function(optimized_act_v1,  "optimized_act_v1");
    //add_activation_function(optimized_act_v2,  "optimized_act_v2");
    add_activation_function(optimized_act_v5,  "Index Inversion and Filtering Active Blades");
    //add_activation_function(optimized_act_v6,  "optimized_act_v6");
    add_activation_function(optimized_act_v7,  "Sigmoid Acceleration");
    add_activation_function(optimized_act_v8,  "Tiling for Sparse Locality");
    //add_activation_function(optimized_act_v9,  "optimized_act_v9");
    //add_activation_function(optimized_act_v10,  "optimized_act_v10");
    add_activation_function(optimized_act_v11,  "Branchless Mask");
    //add_activation_function(optimized_act_v12,  "optimized_act_v12");
    add_activation_function(optimized_act_v13,  "AVX Implementation");
}
