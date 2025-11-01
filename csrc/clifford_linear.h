/* 
* Header file for Clifford Linear layer
* This file contains the definition of the CliffordLinear struct and
* the function prototypes for creating, destroying, and using the layer.
*/

#ifndef CLIFFORD_LINEAR_H
#define CLIFFORD_LINEAR_H

#include <stdlib.h>
#include <stdbool.h>


typedef struct
{
    int dim;      
    int n_blades; // 2^dim
    float *g;     
} CliffordSignature;

/// Main linear‐layer object.
typedef struct
{
    CliffordSignature sig;
    int in_channels;
    int out_channels;
    float *weight; 
    float *bias; 


    void (*get_kernel)(
        const float *w, 
        const float *g, 
        int in_ch,
        int out_ch,
        int *out_nblades,
        float **out_matrix 
    );
} CliffordLinear;

// Initialize a signature from an integer array g_in of length dim (1,2,3).
void clifford_signature_init(CliffordSignature *sig, const int *g_in, int dim);

// Create a new CliffordLinear; allocates weight & bias (zeroed).
CliffordLinear *clifford_linear_create(
    const int *g_in,
    int dim,
    int in_channels,
    int out_channels,
    bool use_bias);

// Free all memory inside and the struct itself.
void clifford_linear_destroy(CliffordLinear *L);

// Run the forward pass:
void clifford_linear_forward(
    const CliffordLinear *L,
    const float *x,
    int B,
    float *out);

#endif // CLIFFORD_LINEAR_H
