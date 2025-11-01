#include "clifford_linear.h"
#include <string.h>
#include <math.h>
#include "flops.h"

/*
* Kernel generators: Each allocates a float array of shape
* (out_ch * n_blades) × (in_ch * n_blades)
* The kernel is a matrix of the form: 
* K = weight * g * product rule
* where weight is the learnable weight matrix, g is the metric 
* coefficient, and the product rule is the algebaic sign from 
* multiplying two blades.
* The values of the kernel are the coefficients of the linear 
* combinations of the input channels that are used to create the output blades. 
* The number of blades is 2^dim
*/

/*
* 1D Clifford kernel:
* The kernel for 1 dimension (i.e. complex numbers) is a 2x2 matrix. 
* k0 = [ w[0], g[0]*w[1] ]
* k1 = [ w[1], w[0] ]
* where w[0] and w[1] are the two input channels.
* The output is a 2D matrix of shape (out_ch * 2) × (in_ch * 2).
* The first row is the first blade, and the second row is the second blade.
*
* Here the combinations are:
* x         1          e1
* 1         1          e1
* e1        e1         g0
* which corresponds to the following multiplication rules:
* scalar, scalar -> scalar
* scalar, vector -> vector
* vector, scalar -> vector
* vector, vector -> scalar
*/

static void get_1d_clifford_kernel(
    const float *w, 
    const float *g, 
    int in_ch,
    int out_ch,
    int *out_nblades,
    float **out_matrix)
{
    int nB = 2;
    int rows = out_ch * nB; // number of rows
    int cols = in_ch * nB; // number of columns
    float *K = malloc(sizeof(float) * rows * cols);

    float g0 = g[0];
    
    // We index elements as shown below, as we flatten a (blade, channel) pair into 1D index for row/col. Thus K[row*C+col] 
    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {

            float w_scalar = w[(0 * out_ch + oc) * in_ch + ic]; 
            float w_vector = w[(1 * out_ch + oc) * in_ch + ic]; 

            // Determine the coefficients of the linear combinations of the weights and the metric that give us a scalar output => y0 = w0*x0 + g0*w1*x1 
            K[(0 * out_ch + oc) * cols + (0 * in_ch + ic)] = w_scalar;    
            K[(0 * out_ch + oc) * cols + (1 * in_ch + ic)] = g0 * w_vector; 

            // Determine the coefficients of the linear combinations of the weights and the metric that give us a vector output => y1 = w1*x0 + w0*x1
            K[(1 * out_ch + oc) * cols + (0 * in_ch + ic)] = w_vector;     
            K[(1 * out_ch + oc) * cols + (1 * in_ch + ic)] = w_scalar;    
        }
    }

    *out_nblades = nB;
    *out_matrix = K;
}

/*
* 2D Clifford kernel:
* The kernel for 2 dimensions (i.e. quaternions) is a 4x4 matrix.
* k0 = [w[0], g[0]*w[1], g[1]*w[2], -g[0]*g[1]*w[3]]
* k1 = [w[1], w[0], -g[1]*w[3], g[1]*w[2]]
* k2 = [w[2], g[0]*w[3], w[0], -g[0]*w[1]]
* k3 = [w[3], w[2], -w[1], w[0]]
*
* Here the combinations are:
* x         1          e1         e2         e3
* 1         1          e1         e2         e3
* e1        e1         g0        g1         -g0*g1
* e2        e2         -g1       g0         g1
* e3        e3         g0*g1     -g0       -g1
* which corresponds to the following multiplication rules:
*/

static void get_2d_clifford_kernel(
    const float *w, 
    const float *g, 
    int in_ch,
    int out_ch,
    int *out_nblades,
    float **out_matrix)
{
    int nB = 4;
    int rows = out_ch * nB;
    int cols = in_ch * nB;
    float *K = malloc(sizeof(float) * rows * cols);

    float g0 = g[0];
    float g1 = g[1];

    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            float w0 = w[(0 * out_ch + oc) * in_ch + ic];
            float w1 = w[(1 * out_ch + oc) * in_ch + ic];
            float w2 = w[(2 * out_ch + oc) * in_ch + ic];
            float w3 = w[(3 * out_ch + oc) * in_ch + ic];

            K[(0 * out_ch + oc) * cols + (0 * in_ch + ic)] = w0;
            K[(0 * out_ch + oc) * cols + (1 * in_ch + ic)] = g0 * w1;
            K[(0 * out_ch + oc) * cols + (2 * in_ch + ic)] = g1 * w2;
            K[(0 * out_ch + oc) * cols + (3 * in_ch + ic)] = -g0 * g1 * w3;

            
            K[(1 * out_ch + oc) * cols + (0 * in_ch + ic)] = w1;
            K[(1 * out_ch + oc) * cols + (1 * in_ch + ic)] = w0;
            K[(1 * out_ch + oc) * cols + (2 * in_ch + ic)] = -g1 * w3;
            K[(1 * out_ch + oc) * cols + (3 * in_ch + ic)] = g1 * w2;

            K[(2 * out_ch + oc) * cols + (0 * in_ch + ic)] = w2;
            K[(2 * out_ch + oc) * cols + (1 * in_ch + ic)] = g0 * w3;
            K[(2 * out_ch + oc) * cols + (2 * in_ch + ic)] = w0;
            K[(2 * out_ch + oc) * cols + (3 * in_ch + ic)] = -g0 * w1;

            K[(3 * out_ch + oc) * cols + (0 * in_ch + ic)] = w3;
            K[(3 * out_ch + oc) * cols + (1 * in_ch + ic)] = w2;
            K[(3 * out_ch + oc) * cols + (2 * in_ch + ic)] = -w1;
            K[(3 * out_ch + oc) * cols + (3 * in_ch + ic)] = w0;
        }
    }

    *out_nblades = nB;
    *out_matrix = K;
}

/*
* 3D Clifford kernel:
* The kernel for 3 dimensions (i.e. octonions) is an 8x8 matrix.
* k0 = [w[0], g[0]*w[1], g[1]*w[2], g[2]*w[3], -g[0]*g[1]*w[4], -g[0]*g[2]*w[5], -g[1]*g[2]*w[6], -g[0]*g[1]*g[2]*w[7]]
* k1 = [w[1], w[0], -g[1]*w[4], -g[2]*w[5], g[1]*w[2], g[2]*w[3], -g[1]*g[2]*w[7], -g[0]*g[2]*w[6]]
* k2 = [w[2], g[0]*w[3], w[0], -g[2]*w[6], -g[0]*w[1], g[0]*w[4], g[1]*w[5], -g[1]*g[2]*w[7]]
* k3 = [w[3], g[0]*w[4], g[1]*w[5], w[0], -g[0]*w[2], -g[1]*w[6], w[1], -g[0]*g[1]*w[7]]
* k4 = [w[4], g[0]*w[5], g[1]*w[6], g[2]*w[7], w[0], -g[0]*g[1]*w[1], -g[0]*g[2]*w[2], -g[1]*g[2]*w[3]]
* k5 = [w[5], g[0]*w[6], g[1]*w[7], -g[2]*w[4], w[1], -g[0]*g[1]*w[2], w[0], -g[0]*g[2]*w[3]]
* k6 = [w[6], g[0]*w[7], -g[2]*w[5], -g[1]*w[4], w[2], g[0]*w[3], w[1], -g[1]*g[2]*w[0]]
* k7 = [w[7], -g[2]*w[6], -g[1]*w[5], -g[0]*w[4], w[3], -g[0]*g[1]*w[2], w[2], w[1]]
*
* Here the combinations are:
* x         1          e1         e2         e3         e4         e5         e6         e7
* 1         1          e1         e2         e3         e4         e5         e6         e7
* e1        e1         g0        g1         g2         -g0*g1    -g0*g2    -g1*g2    -g0*g1*g2
* e2        e2         -g1       g0         g1         g1        -g0*g2    -g1*g2    -g0*g1*g2
* e3        e3         g2*g0     -g2       g0         -g0*g1    -g1*g2    -g0       -g1*g2
* e4        e4         -g0*g1    -g0*g2    -g1*g2    g0        -g0       -g1       -g2
* e5        e5         -g0*g2    -g1*g2    -g0       -g1       g1        -g1       -g0
* e6        e6         -g1*g2    -g0       -g1       -g2       -g1       g0        -g0
* e7        e7         -g0*g1*g2 -g0*g1*g2 -g1*g2    -g0       -g0       -g1       g0
*/

static void get_3d_clifford_kernel(
    const float *w, 
    const float *g, 
    int in_ch,
    int out_ch,
    int *out_nblades,
    float **out_matrix)
{
    int nB = 8;
    int rows = out_ch * nB;
    int cols = in_ch * nB;
    float *K = malloc(sizeof(float) * rows * cols);

    float g0 = g[0], g1 = g[1], g2 = g[2];

    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {

            float w0 = w[(0 * out_ch + oc) * in_ch + ic]; 
            float w1 = w[(1 * out_ch + oc) * in_ch + ic]; 
            float w2 = w[(2 * out_ch + oc) * in_ch + ic]; 
            float w3 = w[(3 * out_ch + oc) * in_ch + ic]; 
            float w4 = w[(4 * out_ch + oc) * in_ch + ic]; 
            float w5 = w[(5 * out_ch + oc) * in_ch + ic]; 
            float w6 = w[(6 * out_ch + oc) * in_ch + ic]; 
            float w7 = w[(7 * out_ch + oc) * in_ch + ic]; 

            K[(0*out_ch+oc)*cols + (0*in_ch+ic)] = w0;
            K[(0*out_ch+oc)*cols + (1*in_ch+ic)] = g0 * w1;
            K[(0*out_ch+oc)*cols + (2*in_ch+ic)] = g1 * w2;
            K[(0*out_ch+oc)*cols + (3*in_ch+ic)] = g2 * w3;
            K[(0*out_ch+oc)*cols + (4*in_ch+ic)] = -g0*g1 * w4;
            K[(0*out_ch+oc)*cols + (5*in_ch+ic)] = -g0*g2 * w5;
            K[(0*out_ch+oc)*cols + (6*in_ch+ic)] = -g1*g2 * w6;
            K[(0*out_ch+oc)*cols + (7*in_ch+ic)] = -g0*g1*g2 * w7;

            K[(1*out_ch+oc)*cols + (0*in_ch+ic)] = w1;
            K[(1*out_ch+oc)*cols + (1*in_ch+ic)] = w0;
            K[(1*out_ch+oc)*cols + (2*in_ch+ic)] = -g1 * w4;
            K[(1*out_ch+oc)*cols + (3*in_ch+ic)] = -g2 * w5;
            K[(1*out_ch+oc)*cols + (4*in_ch+ic)] = w2 * g1;
            K[(1*out_ch+oc)*cols + (5*in_ch+ic)] = w3 * g2;
            K[(1*out_ch+oc)*cols + (6*in_ch+ic)] = -g1*g2 * w7;
            K[(1*out_ch+oc)*cols + (7*in_ch+ic)] = -g1*g2 * w6;

            K[(2*out_ch+oc)*cols + (0*in_ch+ic)] = w2;
            K[(2*out_ch+oc)*cols + (1*in_ch+ic)] = g0 * w4;
            K[(2*out_ch+oc)*cols + (2*in_ch+ic)] = w0;
            K[(2*out_ch+oc)*cols + (3*in_ch+ic)] = -g2 * w6;
            K[(2*out_ch+oc)*cols + (4*in_ch+ic)] = -g0 * w1;
            K[(2*out_ch+oc)*cols + (5*in_ch+ic)] = w7 * g0*g2;
            K[(2*out_ch+oc)*cols + (6*in_ch+ic)] = w3 * g2;
            K[(2*out_ch+oc)*cols + (7*in_ch+ic)] = w5 * g0*g2;

            K[(3*out_ch+oc)*cols + (0*in_ch+ic)] = w3;
            K[(3*out_ch+oc)*cols + (1*in_ch+ic)] = g0 * w5;
            K[(3*out_ch+oc)*cols + (2*in_ch+ic)] = g1 * w6;
            K[(3*out_ch+oc)*cols + (3*in_ch+ic)] = w0;
            K[(3*out_ch+oc)*cols + (4*in_ch+ic)] = -g0*g1 * w7;
            K[(3*out_ch+oc)*cols + (5*in_ch+ic)] = -g0 * w1;
            K[(3*out_ch+oc)*cols + (6*in_ch+ic)] = -g1 * w2;
            K[(3*out_ch+oc)*cols + (7*in_ch+ic)] = -g0*g1 * w4;

            K[(4*out_ch+oc)*cols + (0*in_ch+ic)] = w4;
            K[(4*out_ch+oc)*cols + (1*in_ch+ic)] = w2;
            K[(4*out_ch+oc)*cols + (2*in_ch+ic)] = -w1;
            K[(4*out_ch+oc)*cols + (3*in_ch+ic)] = g2 * w7;
            K[(4*out_ch+oc)*cols + (4*in_ch+ic)] = w0;
            K[(4*out_ch+oc)*cols + (5*in_ch+ic)] = -g2 * w6;
            K[(4*out_ch+oc)*cols + (6*in_ch+ic)] = g2 * w5;
            K[(4*out_ch+oc)*cols + (7*in_ch+ic)] = g2 * w3;

            K[(5*out_ch+oc)*cols + (0*in_ch+ic)] = w5;
            K[(5*out_ch+oc)*cols + (1*in_ch+ic)] = w3;
            K[(5*out_ch+oc)*cols + (2*in_ch+ic)] = -g1 * w7;
            K[(5*out_ch+oc)*cols + (3*in_ch+ic)] = -w1;
            K[(5*out_ch+oc)*cols + (4*in_ch+ic)] = g1 * w6;
            K[(5*out_ch+oc)*cols + (5*in_ch+ic)] = w0;
            K[(5*out_ch+oc)*cols + (6*in_ch+ic)] = -g1 * w4;
            K[(5*out_ch+oc)*cols + (7*in_ch+ic)] = -g1 * w2;

            K[(6*out_ch+oc)*cols + (0*in_ch+ic)] = w6;
            K[(6*out_ch+oc)*cols + (1*in_ch+ic)] = g0 * w7;
            K[(6*out_ch+oc)*cols + (2*in_ch+ic)] = w3;
            K[(6*out_ch+oc)*cols + (3*in_ch+ic)] = -w2;
            K[(6*out_ch+oc)*cols + (4*in_ch+ic)] = -g0 * w5;
            K[(6*out_ch+oc)*cols + (5*in_ch+ic)] = g0 * w4;
            K[(6*out_ch+oc)*cols + (6*in_ch+ic)] = w0;
            K[(6*out_ch+oc)*cols + (7*in_ch+ic)] = g0 * w1;

            K[(7*out_ch+oc)*cols + (0*in_ch+ic)] = w7;
            K[(7*out_ch+oc)*cols + (1*in_ch+ic)] = w6;
            K[(7*out_ch+oc)*cols + (2*in_ch+ic)] = -w5;
            K[(7*out_ch+oc)*cols + (3*in_ch+ic)] = w4;
            K[(7*out_ch+oc)*cols + (4*in_ch+ic)] = w3;
            K[(7*out_ch+oc)*cols + (5*in_ch+ic)] = -w2;
            K[(7*out_ch+oc)*cols + (6*in_ch+ic)] = w1;
            K[(7*out_ch+oc)*cols + (7*in_ch+ic)] = w0;
        }
    }

    *out_nblades = nB;
    *out_matrix = K;
}

// Initialize signature struct.
void clifford_signature_init(CliffordSignature *sig, const int *g_in, int dim)
{
    sig->dim = dim;
    sig->n_blades = 1 << dim;
    sig->g = malloc(sizeof(float) * dim);
    for (int i = 0; i < dim; i++)
    {
        sig->g[i] = (float)g_in[i];
    }
}

// Create and zero‐initialize weights/bias.
CliffordLinear *clifford_linear_create(
    const int *g_in,
    int dim,
    int in_ch,
    int out_ch,
    bool use_bias)
{
    CliffordLinear *L = malloc(sizeof(*L));
    L->in_channels = in_ch;
    L->out_channels = out_ch;
    clifford_signature_init(&L->sig, g_in, dim);

    if (dim == 1) {
        L->get_kernel = get_1d_clifford_kernel;
    } else if (dim == 2) {
        L->get_kernel = get_2d_clifford_kernel;
    } else if (dim == 3) {
        L->get_kernel = get_3d_clifford_kernel;
    } else {
        free(L);
        return NULL;
    }

    int B = L->sig.n_blades * out_ch * in_ch;
    L->weight = calloc(B, sizeof(float));
    if (use_bias) {
        L->bias = calloc(L->sig.n_blades * out_ch, sizeof(float));
    }
    else {
        L->bias = NULL;
    }
    return L;
}

// Free all memory inside and the struct itself.
void clifford_linear_destroy(CliffordLinear *L)
{
    if (!L){
        return;
    }

    free(L->sig.g);
    free(L->weight);
    free(L->bias);
    free(L);
}

// Performs: out[b] = K * flatten(x[b]) + bias, reshaped back
void clifford_linear_forward(
    const CliffordLinear *L,
    const float *x,
    int B,
    float *out)
{
    int I = L->sig.n_blades; // number blades
    int c_in = L->in_channels; // number of input channels
    int c_out = L->out_channels; // number of output channels
    int d_flat = c_in * I; // number of input features
    int n_flat = c_out * I; // number of output features
    int nb; 
    float *K;

    // Track bytes for input, weight, and output data
    size_t input_bytes = B * c_in * I * sizeof(float);
    size_t weight_bytes = L->sig.n_blades * c_out * c_in * sizeof(float);
    size_t output_bytes = B * c_out * I * sizeof(float);

    BYTES_READ(input_bytes + weight_bytes);
    if (L->bias) {
        BYTES_READ(L->sig.n_blades * c_out * sizeof(float));
    }
    BYTES_WRITTEN(output_bytes);

    // Flatten x
    float *x_f = malloc(sizeof(float) * B * d_flat);
    for (int b = 0; b < B; b++)
    {
        for (int i = 0; i < I; i++)
        {
            for (int c = 0; c < c_in; c++)
            {
                x_f[b * d_flat + i * c_in + c] = x[b * (c_in * I) + c * I + i];
            }
        }
    }
    
    // Create kernel
    L->get_kernel(L->weight, L->sig.g, c_in, c_out, &nb, &K);


    // Perform matrix multiplication to get y 
    float *y_f = malloc(sizeof(float) * B * n_flat);
    for (int b = 0; b < B; b++)
    {
        for (int r = 0; r < n_flat; r++)
        {
            float acc = 0.f;
            for (int d = 0; d < d_flat; d++)
            {
                acc += K[r * d_flat + d] * x_f[b * d_flat + d];
                FLOP(2);
            }
            if (L->bias)
            {
                FLOP(1);
                acc += L->bias[r];
            }
            y_f[b * n_flat + r] = acc;
        }
    }
    
    // Reshape back
    for (int b = 0; b < B; b++)
    {
        for (int i = 0; i < I; i++)
        {
            for (int c = 0; c < c_out; c++)
            {
                out[b * (c_out * I) + c * I + i] =
                    y_f[b * n_flat + i * c_out + c];
            }
        }
    }

    free(x_f);
    free(K);
    free(y_f);
}
