#include "gnn_bench.h"
#include "gnn_setup.h"
#include "../clifford_groupnorm.h"
#include "../flops.h"
#include <stdlib.h>
#include <string.h>

#include <assert.h>
#include <stdlib.h>
#include <stdint.h> 
#include <immintrin.h> // For AVX2 intrinsics
#include <stdio.h>     // For fprintf, and for commented-out debug prints
#include <stdio.h>     // For fprintf in case of errors (though not strictly used in final proposal)
#include <math.h>      // For fmaf



//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// README section is at the end of the file (above the registered functions)
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



//baseline with memory alloc
/*
static void baseline_groupnorm(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {
    clifford_groupnorm(
        x, B, C, D, I, num_groups,
        running,
        running_mean_orig,
        running_cov_orig,
        scaling,
        weight_orig,
        bias_orig,
        training,
        momentum,
        eps,
        x_norm
    );
}
*/
static void  baseline_groupnorm(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           _I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
){
    // Track bytes read from input arrays
    BYTES_READ(B * C * D * _I * sizeof(float));      // x
    if (running) {
        BYTES_READ(C * _I * sizeof(float));          // running_mean_orig
        BYTES_READ(C * _I * _I * sizeof(float));     // running_cov_orig
    }
    if (scaling) {
        BYTES_READ((C / num_groups) * _I * _I * sizeof(float));     // weight_orig
        BYTES_READ((C / num_groups) * _I * _I * sizeof(float));     // bias_orig
    }
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * D * _I * sizeof(float));   // x_norm
    if (running && training) {
        BYTES_WRITTEN(C * _I * sizeof(float));       // running_mean_orig (updated)
        BYTES_WRITTEN(C * _I * _I * sizeof(float));  // running_cov_orig (updated)
    }
    

    //basic constants
    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    //blow up running mean and cov along the batch dim
    if ( running ){
        for(int i = 0; i < _I; i++){ //COST: 0 flops
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_running_mean_temp[i*group_size*B + j*B + l] = running_mean_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_running_cov_temp[i*_I*group_size*B + ii*group_size*B + j*B + l] = running_cov_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
    }

    //mean
    if (training || (!running)) {
        int mean_size = dim0 * _I;
        for (int i = 0; i < mean_size; i++){
            gn_mean_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < _I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                        FLOP(1);
                        gn_mean_temp[i*_I + l] += x[i*num_groups*D*_I + j*D*_I + k*_I + l];
                    }
                }
            }
        }
        for(int i = 0; i < dim0; i++){
            for(int l = 0; l < _I; l++){ //COST: dim0*I
                FLOP(1);
                gn_mean_temp[i*_I + l] /= (D*num_groups);
            }
        }
        if (running){ //update running mean
            for(int i = 0; i < _I; i++){
                for(int j = 0; j < dim0; j++){
                    FLOP(2);
                    FLOP(1);
                    gn_running_mean_temp[i*dim0 + j] += momentum *(gn_mean_temp[j*_I + i] - gn_running_mean_temp[i*dim0 + j]);
                }
            }
        }
    } else { //if not training use running mean
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < dim0; j++){
                    gn_mean_temp[j*_I + i] = gn_running_mean_temp[i*dim0 + j];
            }
        }
    }

    //subtract mean form x
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    FLOP(1);
                    x_norm[i*num_groups*D*_I + j*D*_I + k*_I + l] = x[i*num_groups*D*_I + j*D*_I + k*_I + l] - gn_mean_temp[i*_I + l];
                }
            }
        }
    }

    //permute: dim0, num_groups, D, _I -> dim0, _I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    gn_X_temp[i*_I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*_I + j*D*_I + k*_I + l];
                }
            }
        }
    }

    //calc cov
    if (training || (!running)){
        int cov_size = dim0*_I*_I;
        for (int i = 0; i < cov_size; i++){
            gn_cov_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < _I; j++){ //row selector
                for(int l = 0; l < _I; l++){ //col selector
                    for(int k = 0; k < num_groups*D; k++){
                        FLOP(1);
                        FLOP(1);
                        gn_cov_temp[i*_I*_I + j*_I + l] += gn_X_temp[i*_I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*_I + k*_I + l];
        }}}}
        //scale all of cov
        for(int i = 0; i < cov_size; i++){
            FLOP(1);
            gn_cov_temp[i] /= (num_groups * D);
        }
        if (running) { //upadate running cov
            for(int i = 0; i < _I; i++){            
                for(int ii = 0; ii < _I; ii++){
                    for(int j = 0; j < dim0; j++){
                        FLOP(2);
                        FLOP(1);
                        gn_running_cov_temp[i*_I*dim0 + ii*dim0 + j] += momentum *(gn_cov_temp[j*_I*_I + i*_I + ii] - gn_running_cov_temp[i*_I*dim0 + ii*dim0 + j]);
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < _I; i++){ //if not training use running cov
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < dim0; j++){
                    gn_cov_temp[j*_I*_I + i*_I + ii] = gn_running_cov_temp[i*_I*dim0 + ii*dim0 + j];
                }
            }
        }
    }

    //write runnning mean and cov back
    if( running ){
        //mean
        for(int i = 0; i < _I*group_size; i++){
            running_mean_orig[i] = 0;
        }
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    FLOP(1);
                    running_mean_orig[i*group_size + j] += gn_running_mean_temp[i*group_size*B + l*group_size + j];
                }
            }
        }
        for(int i = 0; i < _I*group_size; i++){
            FLOP(1);
            running_mean_orig[i] /= B;
        }
        //cov
        for(int i = 0; i < _I*_I*group_size; i++){
            running_cov_orig[i] = 0;
        }
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        FLOP(1);
                        running_cov_orig[i*_I*group_size + ii*group_size + j] += gn_running_cov_temp[i*_I*group_size*B + ii*group_size*B + l*group_size + j];
                    }
                }
            }
        }
        for(int i = 0; i < _I*_I*group_size; i++){
            FLOP(1);
            running_cov_orig[i] /= B;
        }

    }

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*_I*_I];
        for(int j = 1; j < _I*_I; j++){
            gn_max_temp[i] = gn_cov_temp[i*_I*_I + j] > gn_max_temp[i] ? gn_cov_temp[i*_I*_I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < _I; j++){
            FLOP(2);
            gn_cov_temp[i*_I*_I + j*_I + j] += gn_max_temp[i]*eps;
    }}

    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */

    for(int i = 0; i < dim0; i++){
        int n = _I;
        float *a = &gn_cov_temp[i*_I*_I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }


    //solve triangular
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*_I*_I);
                    float *vec = x_norm + (i*num_groups*D*_I + j*D*_I + k*_I);
                    for(int r = _I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = _I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*_I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum / mat[r*_I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*_I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    }

    //scaling or just copy over
    if ( scaling ){
        //permute weight from (_I0, _I1, dim0) to (dim0, _I0, _I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < _I*_I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*_I*_I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*_I*_I;
                    float *xv = x_norm + i*num_groups*D*_I + j*D*_I + l*_I;
                    float *b = gn_bias_temp + i*_I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < _I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < _I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*_I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < _I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v0(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    //blow up running mean and cov along the batch dim
    if ( running ){
        for(int i = 0; i < I; i++){ //COST: 0 flops
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                     gn_running_mean_temp[i*group_size*B + j*B + l] = running_mean_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                         gn_running_cov_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = running_cov_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
    }

    //mean
    if (training || (!running)) {
        int mean_size = dim0 * I;
        for (int i = 0; i < mean_size; i++){
            gn_mean_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                        FLOP(1);
                        gn_mean_temp[i*I + l] += x[i*num_groups*D*I + j*D*I + k*I + l];
                    }
                }
            }
        }
        for(int i = 0; i < dim0; i++){
            for(int l = 0; l < I; l++){ //COST: dim0*I
                FLOP(1);
                gn_mean_temp[i*I + l] /= (D*num_groups);
            }
        }
        if (running){ //update running mean
            for(int i = 0; i < I; i++){
                for(int j = 0; j < dim0; j++){
                    FLOP(2);
                    FLOP(1);
                     gn_running_mean_temp[i*dim0 + j] += momentum *(gn_mean_temp[j*I + i] -  gn_running_mean_temp[i*dim0 + j]);
                }
            }
        }
    } else { //if not training use running mean
        for(int i = 0; i < I; i++){
            for(int j = 0; j < dim0; j++){
                    gn_mean_temp[j*I + i] =  gn_running_mean_temp[i*dim0 + j];
            }
        }
    }

    //subtract mean form x
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    FLOP(1);
                    x_norm[i*num_groups*D*I + j*D*I + k*I + l] = x[i*num_groups*D*I + j*D*I + k*I + l] - gn_mean_temp[i*I + l];
                }
            }
        }
    }

    //permute: dim0, num_groups, D, I -> dim0, I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*I + j*D*I + k*I + l];
                }
            }
        }
    }

    //calc cov
    if (training || (!running)){
        int cov_size = dim0*I*I;
        for (int i = 0; i < cov_size; i++){
            gn_cov_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < I; j++){ //row selector
                for(int l = 0; l < I; l++){ //col selector
                    for(int k = 0; k < num_groups*D; k++){
                        FLOP(1);
                        FLOP(1);
                        gn_cov_temp[i*I*I + j*I + l] += gn_X_temp[i*I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*I + k*I + l];
        }}}}
        //scale all of cov
        for(int i = 0; i < cov_size; i++){
            FLOP(1);
            gn_cov_temp[i] /= (num_groups * D);
        }
        if (running) { //upadate running cov
            for(int i = 0; i < I; i++){            
                for(int ii = 0; ii < I; ii++){
                    for(int j = 0; j < dim0; j++){
                        FLOP(2);
                        FLOP(1);
                         gn_running_cov_temp[i*I*dim0 + ii*dim0 + j] += momentum *(gn_cov_temp[j*I*I + i*I + ii] -  gn_running_cov_temp[i*I*dim0 + ii*dim0 + j]);
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < I; i++){ //if not training use running cov
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < dim0; j++){
                    gn_cov_temp[j*I*I + i*I + ii] =  gn_running_cov_temp[i*I*dim0 + ii*dim0 + j];
                }
            }
        }
    }

    //write runnning mean and cov back
    if( running ){
        //mean
        for(int i = 0; i < I*group_size; i++){
            running_mean_orig[i] = 0;
        }
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    FLOP(1);
                    running_mean_orig[i*group_size + j] +=  gn_running_mean_temp[i*group_size*B + l*group_size + j];
                }
            }
        }
        for(int i = 0; i < I*group_size; i++){
            FLOP(1);
            running_mean_orig[i] /= B;
        }
        //cov
        for(int i = 0; i < I*I*group_size; i++){
            running_cov_orig[i] = 0;
        }
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        FLOP(1);
                        running_cov_orig[i*I*group_size + ii*group_size + j] +=  gn_running_cov_temp[i*I*group_size*B + ii*group_size*B + l*group_size + j];
                    }
                }
            }
        }
        for(int i = 0; i < I*I*group_size; i++){
            FLOP(1);
            running_cov_orig[i] /= B;
        }

    }

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}

    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */

    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    }

    //scaling or just copy over
    if ( scaling ){
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v1(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D

    //mean
    /*
    int mean_size = dim0 * I;
    for (int i = 0; i < mean_size; i++){
        gn_mean_temp[i] = 0.0f;
    }
    */
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
                /*
                for(int l = 0; l < I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                    gn_mean_temp[i*I + l] += x[i*num_groups*D*I + j*D*I + k*I + l];
                }
                */
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }
    /*
    for(int i = 0; i < dim0; i++){
        for(int l = 0; l < I; l++){ //COST: dim0*I
            FLOP(1);
            gn_mean_temp[i*I + l] /= (D*num_groups);
        }
    }
    */

    //subtract mean form x
    /*
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    FLOP(1);
                    x_norm[i*num_groups*D*I + j*D*I + k*I + l] = x[i*num_groups*D*I + j*D*I + k*I + l] - gn_mean_temp[i*I + l];
                }
            }
        }
    }
    */

    //permute: dim0, num_groups, D, I -> dim0, I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*I + j*D*I + k*I + l];
                }
            }
        }
    }

    //calc cov
    int cov_size = dim0*I*I;
    for (int i = 0; i < cov_size; i++){
        gn_cov_temp[i] = 0.0f;
    }
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            for(int l = 0; l < I; l++){ //col selector
                for(int k = 0; k < num_groups*D; k++){
                    FLOP(1);
                    FLOP(1);
                    gn_cov_temp[i*I*I + j*I + l] += gn_X_temp[i*I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*I + k*I + l];
    }}}}
    //scale all of cov
    for(int i = 0; i < cov_size; i++){
        FLOP(1);
        gn_cov_temp[i] /= (num_groups * D);
    }

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v2(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    /*
    //permute: dim0, num_groups, D, I -> dim0, I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*I + j*D*I + k*I + l];
                }
            }
        }
    }
    */

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum = _mm256_setzero_ps();
            for(int k = 0; k < num_groups*D; k++){
                __m256 j_part = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 l_part = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                FLOP(16);
                accum = _mm256_fmadd_ps(j_part, l_part, accum);
            }
            FLOP(8);
            accum = _mm256_mul_ps(accum, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum);
        }
    }
    /*
    int cov_size = dim0*I*I;
    for (int i = 0; i < cov_size; i++){
        gn_cov_temp[i] = 0.0f;
    }
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            for(int l = 0; l < I; l++){ //col selector
                for(int k = 0; k < num_groups*D; k++){
                    FLOP(1);
                    FLOP(1);
                    gn_cov_temp[i*I*I + j*I + l] += gn_X_temp[i*I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*I + k*I + l];
    }}}}
    //scale all of cov
    for(int i = 0; i < cov_size; i++){
        FLOP(1);
        gn_cov_temp[i] /= (num_groups * D);
    }
    */

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v3(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {
    // Track bytes read from input arrays
    BYTES_READ(B * C * D * I * sizeof(float));       // x
    if (running) {
        BYTES_READ(C * I * sizeof(float));           // running_mean_orig
        BYTES_READ(C * I * I * sizeof(float));       // running_cov_orig
    }
    if (scaling) {
        BYTES_READ((C / num_groups) * I * I * sizeof(float));       // weight_orig
        BYTES_READ((C / num_groups) * I * I * sizeof(float));       // bias_orig
    }
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * D * I * sizeof(float));    // x_norm
    if (running && training) {
        BYTES_WRITTEN(C * I * sizeof(float));        // running_mean_orig (updated)
        BYTES_WRITTEN(C * I * I * sizeof(float));    // running_cov_orig (updated)
    }
    

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D
    assert(dim0 >= 8); //use this different now so needs to be at least 8 long

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum = _mm256_setzero_ps();
            for(int k = 0; k < num_groups*D; k++){
                __m256 j_part = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 l_part = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                FLOP(16);
                accum = _mm256_fmadd_ps(j_part, l_part, accum);
            }
            FLOP(8);
            accum = _mm256_mul_ps(accum, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }
    
    /*
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
        }
    }
    */

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v4(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D
    assert(dim0 >= 8); //use this different now so needs to be at least 8 long
    assert((D*num_groups)%8 == 0); //needs to be divisible for backsub vectorization

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum = _mm256_setzero_ps();
            for(int k = 0; k < num_groups*D; k++){
                __m256 j_part = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 l_part = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                FLOP(16);
                accum = _mm256_fmadd_ps(j_part, l_part, accum);
            }
            FLOP(8);
            accum = _mm256_mul_ps(accum, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }
    
    /*
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
        }
    }
    */

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    //FLOP(1);
                    //FLOP(1);
                    //rowsum += mat[r*I + c]*vec[c];
                    __m256 vec_elem = _mm256_loadu_ps(&vec[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + c]);
                    FLOP(16);
                    rowsum = _mm256_fmadd_ps(mat_elem, vec_elem, rowsum);
                }
                //FLOP(1);
                //rowsum = vec[r] - rowsum;
                __m256 vec_elem = _mm256_loadu_ps(&vec[r*8]);
                FLOP(8);
                rowsum = _mm256_sub_ps(vec_elem, rowsum);
                //FLOP(1);
                //vec[r] = rowsum * mat[r*I + r];
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res = _mm256_mul_ps(rowsum, mat_elem);
                _mm256_storeu_ps(&vec[r*8], res);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    /*
    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }
    */
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v5(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D
    assert(dim0 >= 8); //use this different now so needs to be at least 8 long
    assert((D*num_groups)%8 == 0); //needs to be divisible for backsub vectorization

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum = _mm256_setzero_ps();
            for(int k = 0; k < num_groups*D; k++){
                __m256 j_part = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 l_part = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                FLOP(16);
                accum = _mm256_fmadd_ps(j_part, l_part, accum);
            }
            FLOP(8);
            accum = _mm256_mul_ps(accum, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }
    
    /*
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
        }
    }
    */

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        /*
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
        */
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem = _mm256_loadu_ps(&vec[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum = _mm256_fmadd_ps(mat_elem, vec_elem, rowsum);
                }
                __m256 vec_elem = _mm256_loadu_ps(&vec[r*8]);
                FLOP(8);
                rowsum = _mm256_sub_ps(vec_elem, rowsum);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res = _mm256_mul_ps(rowsum, mat_elem);
                _mm256_storeu_ps(&vec[r*8], res);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v6(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long
    assert((D*num_groups)%8 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem = _mm256_loadu_ps(&vec[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum = _mm256_fmadd_ps(mat_elem, vec_elem, rowsum);
                }
                __m256 vec_elem = _mm256_loadu_ps(&vec[r*8]);
                FLOP(8);
                rowsum = _mm256_sub_ps(vec_elem, rowsum);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res = _mm256_mul_ps(rowsum, mat_elem);
                _mm256_storeu_ps(&vec[r*8], res);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v7(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%8 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    /*
    // Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    // In-place Cholesky decomposition for L in A = L L^T where L is a
    // Ilower triangular matrix.  Uses the Cholesky-Crout algorithm,
    // Ibased on Stoer & Bulirsch "Introduction to numerical analysis"
    // ISect. 4.3, but with the loops for i = j and i != j separated and
    // Ichanged to access only the lower triangular part of the matrix.
    // IOutput is computed column by column so the accesses in the
    // Iinnermost loop (indexed by k) are sequential in memory. 
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            x = a[j*n+j];  //A_jj 

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  // L_jj
            FLOP(1);
            r = 1.0 / x;

            // i != j 
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  // A_ij 

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                }
                FLOP(1);
                a[i*n+j] = x * r;  // L_ij = x / L_jj 
            }
        }
    }
    */

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem = _mm256_loadu_ps(&vec[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum = _mm256_fmadd_ps(mat_elem, vec_elem, rowsum);
                }
                __m256 vec_elem = _mm256_loadu_ps(&vec[r*8]);
                FLOP(8);
                rowsum = _mm256_sub_ps(vec_elem, rowsum);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res = _mm256_mul_ps(rowsum, mat_elem);
                _mm256_storeu_ps(&vec[r*8], res);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v8(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%8 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    //__m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8 + 30*8);
            __m256 r = _mm256_rsqrt_ps(x);

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }
    /*
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I; k++){ //go through cov mat
                for(int l = 0; l < I; l++){ //go through cov mat
                    gn_cov_temp[i*I*I + j*I*I + (l*I+k)] = gn_X_temp[i*I*I + (k*I+l)*8 + j];
                }
            }
        }
    }
    */

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem = _mm256_loadu_ps(&vec[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum = _mm256_fmadd_ps(mat_elem, vec_elem, rowsum);
                }
                __m256 vec_elem = _mm256_loadu_ps(&vec[r*8]);
                FLOP(8);
                rowsum = _mm256_sub_ps(vec_elem, rowsum);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res = _mm256_mul_ps(rowsum, mat_elem);
                _mm256_storeu_ps(&vec[r*8], res);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v9(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%8 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        //__m256 accum = _mm256_setzero_ps();
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            //__m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*I]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
            
            //FLOP(8);
            //accum = _mm256_add_ps(accum, row);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        //FLOP(8);
        //accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }
    /*
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }
    */

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }
    /*
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I; k++){ //go through cov mat
                for(int l = 0; l < I; l++){ //go through cov mat
                    gn_cov_temp[i*I*I + j*I*I + (l*I+k)] = gn_X_temp[i*I*I + (k*I+l)*8 + j];
                }
            }
        }
    }
    */

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem = _mm256_loadu_ps(&vec[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum = _mm256_fmadd_ps(mat_elem, vec_elem, rowsum);
                }
                __m256 vec_elem = _mm256_loadu_ps(&vec[r*8]);
                FLOP(8);
                rowsum = _mm256_sub_ps(vec_elem, rowsum);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res = _mm256_mul_ps(rowsum, mat_elem);
                _mm256_storeu_ps(&vec[r*8], res);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v10(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%8 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        /*
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        */
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }
    /*
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I; k++){ //go through cov mat
                for(int l = 0; l < I; l++){ //go through cov mat
                    gn_cov_temp[i*I*I + j*I*I + (l*I+k)] = gn_X_temp[i*I*I + (k*I+l)*8 + j];
                }
            }
        }
    }
    */

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem = _mm256_loadu_ps(&vec[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum = _mm256_fmadd_ps(mat_elem, vec_elem, rowsum);
                }
                __m256 vec_elem = _mm256_loadu_ps(&vec[r*8]);
                FLOP(8);
                rowsum = _mm256_sub_ps(vec_elem, rowsum);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res = _mm256_mul_ps(rowsum, mat_elem);
                _mm256_storeu_ps(&vec[r*8], res);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v11(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%16 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 16){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v12(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {
    // Track bytes read from input arrays
    BYTES_READ(B * C * D * I * sizeof(float));       // x
    if (running) {
        BYTES_READ(C * I * sizeof(float));           // running_mean_orig
        BYTES_READ(C * I * I * sizeof(float));       // running_cov_orig
    }
    if (scaling) {
        BYTES_READ((C / num_groups) * I * I * sizeof(float));       // weight_orig
        BYTES_READ((C / num_groups) * I * I * sizeof(float));       // bias_orig
    }
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * D * I * sizeof(float));    // x_norm
    if (running && training) {
        BYTES_WRITTEN(C * I * sizeof(float));        // running_mean_orig (updated)
        BYTES_WRITTEN(C * I * I * sizeof(float));    // running_cov_orig (updated)
    }
    

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v13(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    //assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    if(I == 8){
        //mean
        __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
        for(int i = 0; i < dim0; i++){
            //calc mean
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
                __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
                __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
                __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
                __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
                __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
                __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
                __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

                FLOP(8);
                accum0 = _mm256_add_ps(accum0, row0);
                FLOP(8);
                accum1 = _mm256_add_ps(accum1, row1);
                FLOP(8);
                accum2 = _mm256_add_ps(accum2, row2);
                FLOP(8);
                accum3 = _mm256_add_ps(accum3, row3);
                FLOP(8);
                accum4 = _mm256_add_ps(accum4, row4);
                FLOP(8);
                accum5 = _mm256_add_ps(accum5, row5);
                FLOP(8);
                accum6 = _mm256_add_ps(accum6, row6);
                FLOP(8);
                accum7 = _mm256_add_ps(accum7, row7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, mean_scale);
            
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                    FLOP(8);
                    row = _mm256_sub_ps(row, accum0);
                    _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
                }
            }
        }

        //calc cov
        __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < I; j++){ //row selector
                __m256 accum0 = _mm256_setzero_ps();
                __m256 accum1 = _mm256_setzero_ps();
                __m256 accum2 = _mm256_setzero_ps();
                __m256 accum3 = _mm256_setzero_ps();
                __m256 accum4 = _mm256_setzero_ps();
                __m256 accum5 = _mm256_setzero_ps();
                __m256 accum6 = _mm256_setzero_ps();
                __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
                for(int k = 0; k < num_groups*D; k+=8){
                    __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                    __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                    __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                    __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                    __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                    __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                    __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                    __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                    __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                    __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                    __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                    __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                    __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                    __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                    __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                    __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                    FLOP(16);
                    accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                    FLOP(16);
                    accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                    FLOP(16);
                    accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                    FLOP(16);
                    accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                    FLOP(16);
                    accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                    FLOP(16);
                    accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                    FLOP(16);
                    accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                    FLOP(16);
                    accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
                }
                FLOP(8);
                accum0 = _mm256_add_ps(accum0, accum1);
                FLOP(8);
                accum2 = _mm256_add_ps(accum2, accum3);
                FLOP(8);
                accum4 = _mm256_add_ps(accum4, accum5);
                FLOP(8);
                accum6 = _mm256_add_ps(accum6, accum7);
                FLOP(8);
                accum0 = _mm256_add_ps(accum0, accum2);
                FLOP(8);
                accum4 = _mm256_add_ps(accum4, accum6);
                FLOP(8);
                accum0 = _mm256_add_ps(accum0, accum4);
                FLOP(8);
                accum0 = _mm256_mul_ps(accum0, cov_scale);
                _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
            }
        }

        for(int i = 0; i < dim0; i++){
            __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
            __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
            __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
            __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
            __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
            __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
            __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
            __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

            maxv0 = _mm256_max_ps(maxv0, maxv1);
            maxv2 = _mm256_max_ps(maxv2, maxv3);
            maxv4 = _mm256_max_ps(maxv4, maxv5);
            maxv6 = _mm256_max_ps(maxv6, maxv7);
            maxv0 = _mm256_max_ps(maxv0, maxv2);
            maxv4 = _mm256_max_ps(maxv4, maxv6);
            maxv0 = _mm256_max_ps(maxv0, maxv4);
            _mm256_storeu_ps(gn_max_temp, maxv0);
            float max = gn_max_temp[0];
            for(int j = 1; j < I; j++){
                max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
            }
            FLOP(1);
            max = max*eps;
            for(int j = 0; j < I; j++){
                FLOP(1);
                gn_cov_temp[i*I*I + j*I + j] += max;
            }
        }



    } else if (I == 4) { //2D
        //mean
        int mean_size = dim0 * I;
        for (int i = 0; i < mean_size; i++){
            gn_mean_temp[i] = 0.0f;
        }
        
        for(int i = 0; i < dim0; i++){
            //calc mean
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                        gn_mean_temp[i*I + l] += x[i*num_groups*D*I + j*D*I + k*I + l];
                    }
                }
            }
        }
        for(int i = 0; i < dim0; i++){
            for(int l = 0; l < I; l++){ //COST: dim0*I
                FLOP(1);
                gn_mean_temp[i*I + l] /= (D*num_groups);
            }
        }
        

        //subtract mean form x
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < I; l++){
                        FLOP(1);
                        x_norm[i*num_groups*D*I + j*D*I + k*I + l] = x[i*num_groups*D*I + j*D*I + k*I + l] - gn_mean_temp[i*I + l];
                    }
                }
            }
        }

        //permute: dim0, num_groups, D, I -> dim0, I, num_groups, D
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < I; l++){
                        gn_X_temp[i*I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*I + j*D*I + k*I + l];
                    }
                }
            }
        }

        //calc cov
        int cov_size = dim0*I*I;
        for (int i = 0; i < cov_size; i++){
            gn_cov_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < I; j++){ //row selector
                for(int l = 0; l < I; l++){ //col selector
                    for(int k = 0; k < num_groups*D; k++){
                        FLOP(1);
                        FLOP(1);
                        gn_cov_temp[i*I*I + j*I + l] += gn_X_temp[i*I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*I + k*I + l];
        }}}}
        //scale all of cov
        for(int i = 0; i < cov_size; i++){
            FLOP(1);
            gn_cov_temp[i] /= (num_groups * D);
        }

        //get max for each groups cov matrix
        for(int i = 0; i < dim0; i++){
            gn_max_temp[i] = gn_cov_temp[i*I*I];
            for(int j = 1; j < I*I; j++){
                gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
            }
        }

        //add pertubation to diagonal
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < I; j++){
                FLOP(2);
                gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
        }}


    } else if( I == 2 ){ // 1D
        //mean
        int mean_size = dim0 * I;
        for (int i = 0; i < mean_size; i++){
            gn_mean_temp[i] = 0.0f;
        }
        
        for(int i = 0; i < dim0; i++){
            //calc mean
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                        gn_mean_temp[i*I + l] += x[i*num_groups*D*I + j*D*I + k*I + l];
                    }
                }
            }
        }
        for(int i = 0; i < dim0; i++){
            for(int l = 0; l < I; l++){ //COST: dim0*I
                FLOP(1);
                gn_mean_temp[i*I + l] /= (D*num_groups);
            }
        }
        

        //subtract mean form x
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < I; l++){
                        FLOP(1);
                        x_norm[i*num_groups*D*I + j*D*I + k*I + l] = x[i*num_groups*D*I + j*D*I + k*I + l] - gn_mean_temp[i*I + l];
                    }
                }
            }
        }

        //permute: dim0, num_groups, D, I -> dim0, I, num_groups, D
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < I; l++){
                        gn_X_temp[i*I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*I + j*D*I + k*I + l];
                    }
                }
            }
        }

        //calc cov
        int cov_size = dim0*I*I;
        for (int i = 0; i < cov_size; i++){
            gn_cov_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < I; j++){ //row selector
                for(int l = 0; l < I; l++){ //col selector
                    for(int k = 0; k < num_groups*D; k++){
                        FLOP(1);
                        FLOP(1);
                        gn_cov_temp[i*I*I + j*I + l] += gn_X_temp[i*I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*I + k*I + l];
        }}}}
        //scale all of cov
        for(int i = 0; i < cov_size; i++){
            FLOP(1);
            gn_cov_temp[i] /= (num_groups * D);
        }

        //get max for each groups cov matrix
        for(int i = 0; i < dim0; i++){
            gn_max_temp[i] = gn_cov_temp[i*I*I];
            for(int j = 1; j < I*I; j++){
                gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
            }
        }

        //add pertubation to diagonal
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < I; j++){
                FLOP(2);
                gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
        }}
    } else {
        assert( I == 2 ||
                I == 4 ||
                I == 8);
    }


    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }


    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*8 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*8 + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v14(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 4); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)
   
    //mean
    __m128 mean_scale = _mm_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m128 accum0 = _mm_setzero_ps();
        __m128 accum1 = _mm_setzero_ps();
        __m128 accum2 = _mm_setzero_ps();
        __m128 accum3 = _mm_setzero_ps();
        __m128 accum4 = _mm_setzero_ps();
        __m128 accum5 = _mm_setzero_ps();
        __m128 accum6 = _mm_setzero_ps();
        __m128 accum7 = _mm_setzero_ps(); // use 8 accumulators because _mm_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m128 row0 = _mm_loadu_ps(&x[i*num_groups*D*I + k*I]);
            __m128 row1 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+1)*I]);
            __m128 row2 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+2)*I]);
            __m128 row3 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+3)*I]);
            __m128 row4 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+4)*I]);
            __m128 row5 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+5)*I]);
            __m128 row6 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+6)*I]);
            __m128 row7 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+7)*I]);

            FLOP(4);
            accum0 = _mm_add_ps(accum0, row0);
            FLOP(4);
            accum1 = _mm_add_ps(accum1, row1);
            FLOP(4);
            accum2 = _mm_add_ps(accum2, row2);
            FLOP(4);
            accum3 = _mm_add_ps(accum3, row3);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, row4);
            FLOP(4);
            accum5 = _mm_add_ps(accum5, row5);
            FLOP(4);
            accum6 = _mm_add_ps(accum6, row6);
            FLOP(4);
            accum7 = _mm_add_ps(accum7, row7);
        }
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum1);
        FLOP(4);
        accum2 = _mm_add_ps(accum2, accum3);
        FLOP(4);
        accum4 = _mm_add_ps(accum4, accum5);
        FLOP(4);
        accum6 = _mm_add_ps(accum6, accum7);
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum2);
        FLOP(4);
        accum4 = _mm_add_ps(accum4, accum6);
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum4);
        FLOP(4);
        accum0 = _mm_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m128 row = _mm_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(4);
                row = _mm_sub_ps(row, accum0);
                _mm_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m128 cov_scale = _mm_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m128 accum0 = _mm_setzero_ps();
            __m128 accum1 = _mm_setzero_ps();
            __m128 accum2 = _mm_setzero_ps();
            __m128 accum3 = _mm_setzero_ps();
            __m128 accum4 = _mm_setzero_ps();
            __m128 accum5 = _mm_setzero_ps();
            __m128 accum6 = _mm_setzero_ps();
            __m128 accum7 = _mm_setzero_ps(); // use 8 accumulators because _mm_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m128 j_part0 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m128 j_part1 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m128 j_part2 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m128 j_part3 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m128 j_part4 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m128 j_part5 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m128 j_part6 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m128 j_part7 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m128 l_part0 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m128 l_part1 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m128 l_part2 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m128 l_part3 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m128 l_part4 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m128 l_part5 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m128 l_part6 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m128 l_part7 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(8);
                accum0 = _mm_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(8);
                accum1 = _mm_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(8);
                accum2 = _mm_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(8);
                accum3 = _mm_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(8);
                accum4 = _mm_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(8);
                accum5 = _mm_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(8);
                accum6 = _mm_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(8);
                accum7 = _mm_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum1);
            FLOP(4);
            accum2 = _mm_add_ps(accum2, accum3);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, accum5);
            FLOP(4);
            accum6 = _mm_add_ps(accum6, accum7);
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum2);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, accum6);
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum4);
            FLOP(4);
            accum0 = _mm_mul_ps(accum0, cov_scale);
            _mm_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }
    
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}


    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }


    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*8 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*8 + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v15(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 2); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps(); // use 2 accumulators with 4 rows each
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*I]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*I]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        __m256 accum0_high = _mm256_permute2f128_ps(accum0, accum0, 0x01);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum0_high);
        __m256 accum0_even_odd = _mm256_shuffle_ps(accum0, accum0, (0x01 << 6) | (0x00 << 4) | (0x03 << 2) | 0x02);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum0_even_odd);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale); //4 copies of the 2 mean floats
        

        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k += 4){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    float cov_scale = 1.0f / ((float)(D*num_groups));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            float accum0_col0 = 0.0f;
            float accum0_col1 = 0.0f;
            float accum1_col0 = 0.0f;
            float accum1_col1 = 0.0f;
            float accum2_col0 = 0.0f;
            float accum2_col1 = 0.0f;
            float accum3_col0 = 0.0f;
            float accum3_col1 = 0.0f; // use 4 accumulators
            for(int k = 0; k < num_groups*D; k+=4){
                float j_part0 = x_norm[i*num_groups*D*I + k*I + j];
                float j_part1 = x_norm[i*num_groups*D*I + (k+1)*I + j];
                float j_part2 = x_norm[i*num_groups*D*I + (k+2)*I + j];
                float j_part3 = x_norm[i*num_groups*D*I + (k+3)*I + j];

                FLOP(8);
                accum0_col0 = fmaf(j_part0, x_norm[i*num_groups*D*I + k*I], accum0_col0);
                accum1_col0 = fmaf(j_part1, x_norm[i*num_groups*D*I + (k+1)*I], accum1_col0);
                accum2_col0 = fmaf(j_part2, x_norm[i*num_groups*D*I + (k+2)*I], accum2_col0);
                accum3_col0 = fmaf(j_part3, x_norm[i*num_groups*D*I + (k+3)*I], accum3_col0);

                FLOP(8);
                accum0_col1 = fmaf(j_part0, x_norm[i*num_groups*D*I + k*I + 1], accum0_col1);
                accum1_col1 = fmaf(j_part1, x_norm[i*num_groups*D*I + (k+1)*I + 1], accum1_col1);
                accum2_col1 = fmaf(j_part2, x_norm[i*num_groups*D*I + (k+2)*I + 1], accum2_col1);
                accum3_col1 = fmaf(j_part3, x_norm[i*num_groups*D*I + (k+3)*I + 1], accum3_col1);
            }
            accum0_col0 = accum0_col0 + accum1_col0;
            accum2_col0 = accum2_col0 + accum3_col0;
            accum0_col0 = accum0_col0 + accum2_col0;

            accum0_col1 = accum0_col1 + accum1_col1;
            accum2_col1 = accum2_col1 + accum3_col1;
            accum0_col1 = accum0_col1 + accum2_col1;

            gn_cov_temp[i*I*I + j*I] = accum0_col0 * cov_scale;
            gn_cov_temp[i*I*I + j*I + 1] = accum0_col1 * cov_scale;
        }
    }
    

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}
    


    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }


    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*8 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*8 + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v16(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*32 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec[c*32]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec[c*32 + 8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec[c*32 + 16]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec[c*32 + 24]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec[r*32]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec[r*32 + 8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec[r*32 + 16]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec[r*32 + 24]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec[r*32], res0);
                _mm256_storeu_ps(&vec[r*32 + 8], res1);
                _mm256_storeu_ps(&vec[r*32 + 16], res2);
                _mm256_storeu_ps(&vec[r*32 + 24], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*32 + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v17(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            int pos = (j%3)*32*I;
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[pos + l*32 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp; //32xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec[c*32]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec[c*32 + 8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec[c*32 + 16]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec[c*32 + 24]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec[r*32]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec[r*32 + 8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec[r*32 + 16]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec[r*32 + 24]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec[r*32], res0);
                _mm256_storeu_ps(&vec[r*32 + 8], res1);
                _mm256_storeu_ps(&vec[r*32 + 16], res2);
                _mm256_storeu_ps(&vec[r*32 + 24], res3);
            }
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[pos + l*32 + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v18(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){

        //just a transpose
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + i*I + ii] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        /*
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }

        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }
        */

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v19(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //mean
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //cholesky decomp

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){

        //just a transpose
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + ii*I + i] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        

        //mat mult and add
        //float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *WT = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    //1) load in b as base
                    __m256 accum = _mm256_loadu_ps(b);
                    
                    //2) iterate W cols with fma (note we have W^T)
                    for(int ii = 0; ii < I; ii++){
                        __m256 col = _mm256_loadu_ps(&WT[ii*I]);
                        __m256 xv_part = _mm256_broadcast_ss(&xv[ii]);
                        FLOP(16);
                        accum = _mm256_fmadd_ps(col, xv_part, accum);
                    }

                    //3) store back result
                    _mm256_storeu_ps(xv, accum);
                    
                    /*
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                    */
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v20(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {
    // Track bytes read from input arrays
    BYTES_READ(B * C * D * I * sizeof(float));       // x
    if (running) {
        BYTES_READ(C * I * sizeof(float));           // running_mean_orig
        BYTES_READ(C * I * I * sizeof(float));       // running_cov_orig
    }
    if (scaling) {
        BYTES_READ((C / num_groups) * I * I * sizeof(float));       // weight_orig
        BYTES_READ((C / num_groups) * I * I * sizeof(float));       // bias_orig
    }
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * D * I * sizeof(float));    // x_norm
    if (running && training) {
        BYTES_WRITTEN(C * I * sizeof(float));        // running_mean_orig (updated)
        BYTES_WRITTEN(C * I * I * sizeof(float));    // running_cov_orig (updated)
    }
    

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //CENTERING
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //CALCULATE COV
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    //MAXIMUM
    for(int i = 0; i < dim0; i++){
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //CHOLESKY

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //BACK SUBSTITUTION
    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //SCALING
    if ( scaling ){

        //just a transpose
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + ii*I + i] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        

        //mat mult and add
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *WT = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    //1) load in b as base
                    //4 accumulators
                    __m256 accum0 = _mm256_loadu_ps(b);
                    __m256 accum1 = _mm256_setzero_ps();
                    __m256 accum2 = _mm256_setzero_ps();
                    __m256 accum3 = _mm256_setzero_ps();
                    
                    //2) iterate W cols with fma (note we have W^T)
                    //use each of the 4 accum to do 2 cols
                    for(int ii = 0; ii < 2; ii++){
                        __m256 col0 = _mm256_loadu_ps(&WT[(ii*4 + 0)*I]);
                        __m256 col1 = _mm256_loadu_ps(&WT[(ii*4 + 1)*I]);
                        __m256 col2 = _mm256_loadu_ps(&WT[(ii*4 + 2)*I]);
                        __m256 col3 = _mm256_loadu_ps(&WT[(ii*4 + 3)*I]);

                        __m256 xv_part0 = _mm256_broadcast_ss(&xv[(ii*4 + 0)]);
                        __m256 xv_part1 = _mm256_broadcast_ss(&xv[(ii*4 + 1)]);
                        __m256 xv_part2 = _mm256_broadcast_ss(&xv[(ii*4 + 2)]);
                        __m256 xv_part3 = _mm256_broadcast_ss(&xv[(ii*4 + 3)]);

                        FLOP(16);
                        accum0 = _mm256_fmadd_ps(col0, xv_part0, accum0);
                        FLOP(16);
                        accum1 = _mm256_fmadd_ps(col1, xv_part1, accum1);
                        FLOP(16);
                        accum2 = _mm256_fmadd_ps(col2, xv_part2, accum2); 
                        FLOP(16);
                        accum3 = _mm256_fmadd_ps(col3, xv_part3, accum3);
                    }

                    //reduction on accums
                    FLOP(8);
                    accum0 = _mm256_add_ps(accum0, accum1);
                    FLOP(8);
                    accum2 = _mm256_add_ps(accum2, accum3);
                    FLOP(8);
                    accum0 = _mm256_add_ps(accum0, accum2);

                    //3) store back result
                    _mm256_storeu_ps(xv, accum0);
                    
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v21(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 4); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)
   
    //CENTERING
    __m128 mean_scale = _mm_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m128 accum0 = _mm_setzero_ps();
        __m128 accum1 = _mm_setzero_ps();
        __m128 accum2 = _mm_setzero_ps();
        __m128 accum3 = _mm_setzero_ps();
        __m128 accum4 = _mm_setzero_ps();
        __m128 accum5 = _mm_setzero_ps();
        __m128 accum6 = _mm_setzero_ps();
        __m128 accum7 = _mm_setzero_ps(); // use 8 accumulators because _mm_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m128 row0 = _mm_loadu_ps(&x[i*num_groups*D*I + k*I]);
            __m128 row1 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+1)*I]);
            __m128 row2 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+2)*I]);
            __m128 row3 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+3)*I]);
            __m128 row4 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+4)*I]);
            __m128 row5 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+5)*I]);
            __m128 row6 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+6)*I]);
            __m128 row7 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+7)*I]);

            FLOP(4);
            accum0 = _mm_add_ps(accum0, row0);
            FLOP(4);
            accum1 = _mm_add_ps(accum1, row1);
            FLOP(4);
            accum2 = _mm_add_ps(accum2, row2);
            FLOP(4);
            accum3 = _mm_add_ps(accum3, row3);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, row4);
            FLOP(4);
            accum5 = _mm_add_ps(accum5, row5);
            FLOP(4);
            accum6 = _mm_add_ps(accum6, row6);
            FLOP(4);
            accum7 = _mm_add_ps(accum7, row7);
        }
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum1);
        FLOP(4);
        accum2 = _mm_add_ps(accum2, accum3);
        FLOP(4);
        accum4 = _mm_add_ps(accum4, accum5);
        FLOP(4);
        accum6 = _mm_add_ps(accum6, accum7);
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum2);
        FLOP(4);
        accum4 = _mm_add_ps(accum4, accum6);
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum4);
        FLOP(4);
        accum0 = _mm_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m128 row = _mm_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(4);
                row = _mm_sub_ps(row, accum0);
                _mm_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //CALCULATE COV
    __m128 cov_scale = _mm_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m128 accum0 = _mm_setzero_ps();
            __m128 accum1 = _mm_setzero_ps();
            __m128 accum2 = _mm_setzero_ps();
            __m128 accum3 = _mm_setzero_ps();
            __m128 accum4 = _mm_setzero_ps();
            __m128 accum5 = _mm_setzero_ps();
            __m128 accum6 = _mm_setzero_ps();
            __m128 accum7 = _mm_setzero_ps(); // use 8 accumulators because _mm_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m128 j_part0 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m128 j_part1 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m128 j_part2 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m128 j_part3 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m128 j_part4 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m128 j_part5 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m128 j_part6 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m128 j_part7 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m128 l_part0 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m128 l_part1 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m128 l_part2 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m128 l_part3 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m128 l_part4 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m128 l_part5 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m128 l_part6 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m128 l_part7 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(8);
                accum0 = _mm_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(8);
                accum1 = _mm_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(8);
                accum2 = _mm_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(8);
                accum3 = _mm_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(8);
                accum4 = _mm_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(8);
                accum5 = _mm_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(8);
                accum6 = _mm_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(8);
                accum7 = _mm_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum1);
            FLOP(4);
            accum2 = _mm_add_ps(accum2, accum3);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, accum5);
            FLOP(4);
            accum6 = _mm_add_ps(accum6, accum7);
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum2);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, accum6);
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum4);
            FLOP(4);
            accum0 = _mm_mul_ps(accum0, cov_scale);
            _mm_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }
    
    //MAXIMUM
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}


    //CHOLESKY

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }


    //BACK SUBSTITUTION
    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*8 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*8 + k];
                }
            }
        }
    }

    //SCALING
    if ( scaling ){

        //just a transpose
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + ii*I + i] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        

        //mat mult and add
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *WT = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    //1) load in b as base
                    //4 accumulators
                    __m128 accum0 = _mm_loadu_ps(b);
                    __m128 accum1 = _mm_setzero_ps();
                    
                    //2) iterate W cols with fma (note we have W^T)
                    //use each of the 4 accum to do 2 cols
                    for(int ii = 0; ii < 2; ii++){
                        __m128 col0 = _mm_loadu_ps(&WT[(ii*2 + 0)*I]);
                        __m128 col1 = _mm_loadu_ps(&WT[(ii*2 + 1)*I]);

                        __m128 xv_part0 = _mm_broadcast_ss(&xv[(ii*2 + 0)]);
                        __m128 xv_part1 = _mm_broadcast_ss(&xv[(ii*2 + 1)]);

                        FLOP(8);
                        accum0 = _mm_fmadd_ps(col0, xv_part0, accum0);
                        FLOP(8);
                        accum1 = _mm_fmadd_ps(col1, xv_part1, accum1);
                    }

                    //reduction on accums
                    FLOP(4);
                    accum0 = _mm_add_ps(accum0, accum1);

                    //3) store back result
                    _mm_storeu_ps(xv, accum0);
                    
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v22(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 2); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //CENTERING
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps(); // use 2 accumulators with 4 rows each
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*I]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*I]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        __m256 accum0_high = _mm256_permute2f128_ps(accum0, accum0, 0x01);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum0_high);
        __m256 accum0_even_odd = _mm256_shuffle_ps(accum0, accum0, (0x01 << 6) | (0x00 << 4) | (0x03 << 2) | 0x02);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum0_even_odd);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale); //4 copies of the 2 mean floats
        

        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k += 4){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //CACULATE COV
    float cov_scale = 1.0f / ((float)(D*num_groups));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            float accum0_col0 = 0.0f;
            float accum0_col1 = 0.0f;
            float accum1_col0 = 0.0f;
            float accum1_col1 = 0.0f;
            float accum2_col0 = 0.0f;
            float accum2_col1 = 0.0f;
            float accum3_col0 = 0.0f;
            float accum3_col1 = 0.0f; // use 4 accumulators
            for(int k = 0; k < num_groups*D; k+=4){
                float j_part0 = x_norm[i*num_groups*D*I + k*I + j];
                float j_part1 = x_norm[i*num_groups*D*I + (k+1)*I + j];
                float j_part2 = x_norm[i*num_groups*D*I + (k+2)*I + j];
                float j_part3 = x_norm[i*num_groups*D*I + (k+3)*I + j];

                FLOP(8);
                accum0_col0 = fmaf(j_part0, x_norm[i*num_groups*D*I + k*I], accum0_col0);
                accum1_col0 = fmaf(j_part1, x_norm[i*num_groups*D*I + (k+1)*I], accum1_col0);
                accum2_col0 = fmaf(j_part2, x_norm[i*num_groups*D*I + (k+2)*I], accum2_col0);
                accum3_col0 = fmaf(j_part3, x_norm[i*num_groups*D*I + (k+3)*I], accum3_col0);

                FLOP(8);
                accum0_col1 = fmaf(j_part0, x_norm[i*num_groups*D*I + k*I + 1], accum0_col1);
                accum1_col1 = fmaf(j_part1, x_norm[i*num_groups*D*I + (k+1)*I + 1], accum1_col1);
                accum2_col1 = fmaf(j_part2, x_norm[i*num_groups*D*I + (k+2)*I + 1], accum2_col1);
                accum3_col1 = fmaf(j_part3, x_norm[i*num_groups*D*I + (k+3)*I + 1], accum3_col1);
            }
            accum0_col0 = accum0_col0 + accum1_col0;
            accum2_col0 = accum2_col0 + accum3_col0;
            accum0_col0 = accum0_col0 + accum2_col0;

            accum0_col1 = accum0_col1 + accum1_col1;
            accum2_col1 = accum2_col1 + accum3_col1;
            accum0_col1 = accum0_col1 + accum2_col1;

            gn_cov_temp[i*I*I + j*I] = accum0_col0 * cov_scale;
            gn_cov_temp[i*I*I + j*I + 1] = accum0_col1 * cov_scale;
        }
    }
    
    //MAXIMUM
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}
    


    //CHOLESKY

    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int i, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(i = j+1; i < n; i++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(i*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(i*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(i*n+j)], tostore);
            }
        }
    }

    
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //BACK SUBSTITUTION
    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*8 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*8 + k];
                }
            }
        }
    }
    

    //SCALING
    if ( scaling ){

        //just a transpose
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + i*I + ii] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        

        //mat mult and add
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    float row0 = b[0];
                    float row1 = b[1];

                    FLOP(4);
                    row0 = fmaf(W[0], xv[0], row0);
                    row1 = fmaf(W[2], xv[0], row1);

                    FLOP(4);
                    row0 = fmaf(W[1], xv[1], row0);
                    row1 = fmaf(W[3], xv[1], row1);

                    xv[0] = row0;
                    xv[1] = row1;
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v23(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //CENTERING
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //CALCULATE COV
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    //MAXIMUM
    for(int i = 0; i < dim0; i++){
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //CHOLESKY

    /*
    //reshape
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
    }
    */
    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int ii, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(ii = j+1; ii < n; ii++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(ii*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(ii*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(ii*n+j)], tostore);
            }
        }
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    /*
    //reshape back
    for(int i = 0; i < dim0; i+=8){ //jump 8 cov mats
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }
    */

    //BACK SUBSTITUTION
    /*
    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    */
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    /*
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    */
    
    //SCALING
    if ( scaling ){

        //just a transpose
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + ii*I + i] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        

        //mat mult and add
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *WT = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    //1) load in b as base
                    //4 accumulators
                    __m256 accum0 = _mm256_loadu_ps(b);
                    __m256 accum1 = _mm256_setzero_ps();
                    __m256 accum2 = _mm256_setzero_ps();
                    __m256 accum3 = _mm256_setzero_ps();
                    
                    //2) iterate W cols with fma (note we have W^T)
                    //use each of the 4 accum to do 2 cols
                    for(int ii = 0; ii < 2; ii++){
                        __m256 col0 = _mm256_loadu_ps(&WT[(ii*4 + 0)*I]);
                        __m256 col1 = _mm256_loadu_ps(&WT[(ii*4 + 1)*I]);
                        __m256 col2 = _mm256_loadu_ps(&WT[(ii*4 + 2)*I]);
                        __m256 col3 = _mm256_loadu_ps(&WT[(ii*4 + 3)*I]);

                        __m256 xv_part0 = _mm256_broadcast_ss(&xv[(ii*4 + 0)]);
                        __m256 xv_part1 = _mm256_broadcast_ss(&xv[(ii*4 + 1)]);
                        __m256 xv_part2 = _mm256_broadcast_ss(&xv[(ii*4 + 2)]);
                        __m256 xv_part3 = _mm256_broadcast_ss(&xv[(ii*4 + 3)]);

                        FLOP(16);
                        accum0 = _mm256_fmadd_ps(col0, xv_part0, accum0);
                        FLOP(16);
                        accum1 = _mm256_fmadd_ps(col1, xv_part1, accum1);
                        FLOP(16);
                        accum2 = _mm256_fmadd_ps(col2, xv_part2, accum2); 
                        FLOP(16);
                        accum3 = _mm256_fmadd_ps(col3, xv_part3, accum3);
                    }

                    //reduction on accums
                    FLOP(8);
                    accum0 = _mm256_add_ps(accum0, accum1);
                    FLOP(8);
                    accum2 = _mm256_add_ps(accum2, accum3);
                    FLOP(8);
                    accum0 = _mm256_add_ps(accum0, accum2);

                    //3) store back result
                    _mm256_storeu_ps(xv, accum0);
                    
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v24(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 4); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)
   
    //CENTERING
    __m128 mean_scale = _mm_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m128 accum0 = _mm_setzero_ps();
        __m128 accum1 = _mm_setzero_ps();
        __m128 accum2 = _mm_setzero_ps();
        __m128 accum3 = _mm_setzero_ps();
        __m128 accum4 = _mm_setzero_ps();
        __m128 accum5 = _mm_setzero_ps();
        __m128 accum6 = _mm_setzero_ps();
        __m128 accum7 = _mm_setzero_ps(); // use 8 accumulators because _mm_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m128 row0 = _mm_loadu_ps(&x[i*num_groups*D*I + k*I]);
            __m128 row1 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+1)*I]);
            __m128 row2 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+2)*I]);
            __m128 row3 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+3)*I]);
            __m128 row4 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+4)*I]);
            __m128 row5 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+5)*I]);
            __m128 row6 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+6)*I]);
            __m128 row7 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+7)*I]);

            FLOP(4);
            accum0 = _mm_add_ps(accum0, row0);
            FLOP(4);
            accum1 = _mm_add_ps(accum1, row1);
            FLOP(4);
            accum2 = _mm_add_ps(accum2, row2);
            FLOP(4);
            accum3 = _mm_add_ps(accum3, row3);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, row4);
            FLOP(4);
            accum5 = _mm_add_ps(accum5, row5);
            FLOP(4);
            accum6 = _mm_add_ps(accum6, row6);
            FLOP(4);
            accum7 = _mm_add_ps(accum7, row7);
        }
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum1);
        FLOP(4);
        accum2 = _mm_add_ps(accum2, accum3);
        FLOP(4);
        accum4 = _mm_add_ps(accum4, accum5);
        FLOP(4);
        accum6 = _mm_add_ps(accum6, accum7);
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum2);
        FLOP(4);
        accum4 = _mm_add_ps(accum4, accum6);
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum4);
        FLOP(4);
        accum0 = _mm_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m128 row = _mm_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(4);
                row = _mm_sub_ps(row, accum0);
                _mm_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //CALCULATE COV
    __m128 cov_scale = _mm_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m128 accum0 = _mm_setzero_ps();
            __m128 accum1 = _mm_setzero_ps();
            __m128 accum2 = _mm_setzero_ps();
            __m128 accum3 = _mm_setzero_ps();
            __m128 accum4 = _mm_setzero_ps();
            __m128 accum5 = _mm_setzero_ps();
            __m128 accum6 = _mm_setzero_ps();
            __m128 accum7 = _mm_setzero_ps(); // use 8 accumulators because _mm_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m128 j_part0 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m128 j_part1 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m128 j_part2 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m128 j_part3 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m128 j_part4 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m128 j_part5 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m128 j_part6 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m128 j_part7 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m128 l_part0 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m128 l_part1 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m128 l_part2 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m128 l_part3 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m128 l_part4 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m128 l_part5 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m128 l_part6 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m128 l_part7 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(8);
                accum0 = _mm_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(8);
                accum1 = _mm_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(8);
                accum2 = _mm_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(8);
                accum3 = _mm_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(8);
                accum4 = _mm_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(8);
                accum5 = _mm_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(8);
                accum6 = _mm_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(8);
                accum7 = _mm_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum1);
            FLOP(4);
            accum2 = _mm_add_ps(accum2, accum3);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, accum5);
            FLOP(4);
            accum6 = _mm_add_ps(accum6, accum7);
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum2);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, accum6);
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum4);
            FLOP(4);
            accum0 = _mm_mul_ps(accum0, cov_scale);
            _mm_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }
    
    //MAXIMUM
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}


    //CHOLESKY

    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int ii, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(ii = j+1; ii < n; ii++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(ii*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(ii*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(ii*n+j)], tostore);
            }
        }
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //BACK SUBSTITUTION
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*8 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*8 + k];
                }
            }
        }
    }

    //SCALING
    if ( scaling ){

        //just a transpose
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + ii*I + i] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        

        //mat mult and add
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *WT = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    //1) load in b as base
                    //4 accumulators
                    __m128 accum0 = _mm_loadu_ps(b);
                    __m128 accum1 = _mm_setzero_ps();
                    
                    //2) iterate W cols with fma (note we have W^T)
                    //use each of the 4 accum to do 2 cols
                    for(int ii = 0; ii < 2; ii++){
                        __m128 col0 = _mm_loadu_ps(&WT[(ii*2 + 0)*I]);
                        __m128 col1 = _mm_loadu_ps(&WT[(ii*2 + 1)*I]);

                        __m128 xv_part0 = _mm_broadcast_ss(&xv[(ii*2 + 0)]);
                        __m128 xv_part1 = _mm_broadcast_ss(&xv[(ii*2 + 1)]);

                        FLOP(8);
                        accum0 = _mm_fmadd_ps(col0, xv_part0, accum0);
                        FLOP(8);
                        accum1 = _mm_fmadd_ps(col1, xv_part1, accum1);
                    }

                    //reduction on accums
                    FLOP(4);
                    accum0 = _mm_add_ps(accum0, accum1);

                    //3) store back result
                    _mm_storeu_ps(xv, accum0);
                    
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v25(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 2); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //CENTERING
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps(); // use 2 accumulators with 4 rows each
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*I]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*I]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        __m256 accum0_high = _mm256_permute2f128_ps(accum0, accum0, 0x01);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum0_high);
        __m256 accum0_even_odd = _mm256_shuffle_ps(accum0, accum0, (0x01 << 6) | (0x00 << 4) | (0x03 << 2) | 0x02);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum0_even_odd);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale); //4 copies of the 2 mean floats
        

        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k += 4){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //CACULATE COV
    float cov_scale = 1.0f / ((float)(D*num_groups));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            float accum0_col0 = 0.0f;
            float accum0_col1 = 0.0f;
            float accum1_col0 = 0.0f;
            float accum1_col1 = 0.0f;
            float accum2_col0 = 0.0f;
            float accum2_col1 = 0.0f;
            float accum3_col0 = 0.0f;
            float accum3_col1 = 0.0f; // use 4 accumulators
            for(int k = 0; k < num_groups*D; k+=4){
                float j_part0 = x_norm[i*num_groups*D*I + k*I + j];
                float j_part1 = x_norm[i*num_groups*D*I + (k+1)*I + j];
                float j_part2 = x_norm[i*num_groups*D*I + (k+2)*I + j];
                float j_part3 = x_norm[i*num_groups*D*I + (k+3)*I + j];

                FLOP(8);
                accum0_col0 = fmaf(j_part0, x_norm[i*num_groups*D*I + k*I], accum0_col0);
                accum1_col0 = fmaf(j_part1, x_norm[i*num_groups*D*I + (k+1)*I], accum1_col0);
                accum2_col0 = fmaf(j_part2, x_norm[i*num_groups*D*I + (k+2)*I], accum2_col0);
                accum3_col0 = fmaf(j_part3, x_norm[i*num_groups*D*I + (k+3)*I], accum3_col0);

                FLOP(8);
                accum0_col1 = fmaf(j_part0, x_norm[i*num_groups*D*I + k*I + 1], accum0_col1);
                accum1_col1 = fmaf(j_part1, x_norm[i*num_groups*D*I + (k+1)*I + 1], accum1_col1);
                accum2_col1 = fmaf(j_part2, x_norm[i*num_groups*D*I + (k+2)*I + 1], accum2_col1);
                accum3_col1 = fmaf(j_part3, x_norm[i*num_groups*D*I + (k+3)*I + 1], accum3_col1);
            }
            accum0_col0 = accum0_col0 + accum1_col0;
            accum2_col0 = accum2_col0 + accum3_col0;
            accum0_col0 = accum0_col0 + accum2_col0;

            accum0_col1 = accum0_col1 + accum1_col1;
            accum2_col1 = accum2_col1 + accum3_col1;
            accum0_col1 = accum0_col1 + accum2_col1;

            gn_cov_temp[i*I*I + j*I] = accum0_col0 * cov_scale;
            gn_cov_temp[i*I*I + j*I + 1] = accum0_col1 * cov_scale;
        }
    }
    
    //MAXIMUM
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}
    


    //CHOLESKY
    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        //float x, r;
        int ii, j, k;
        // Loop over columns
        for(j = 0; j < n; j++) {
            // i = j
            //x = a[j*n+j];  //A_jj 
            __m256 x = _mm256_loadu_ps(&a[8*(j*n+j)]);
            
            for(k = 0; k < j; k++){
                //FLOP(2);
                //x -= a[j*n+k] * a[j*n+k];  // L_jk L_jk 
                __m256 temp = _mm256_loadu_ps(&a[8*(j*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp);
            }

            //if(x < 0)
            //    printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            //FLOP(30);
            //x = sqrt(x);
            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            //a[j*n+j] = x;  // L_jj
            _mm256_storeu_ps(&a[8*(j*n+j)], x);

            //FLOP(1);
            //r = 1.0 / x;
            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            

            // i != j 
            for(ii = j+1; ii < n; ii++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(ii*n+j)]);  // A_ij 

                for(k = 0; k < j; k++){
                    //FLOP(2);
                    //x -= a[i*n+k] * a[j*n+k];  // L_ik L_ij 
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(ii*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(j*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1);
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                //FLOP(1);
                //a[i*n+j] = x * r;  // L_ij = x / L_jj 
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r);
                _mm256_storeu_ps(&a[8*(ii*n+j)], tostore);
            }
        }
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //BACK SUBSTITUTION
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*8 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
        for(int j = 0; j < num_groups*D; j += 32){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec0 = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            float *vec1 = gn_X_temp + (i*num_groups*D*I + (j+8)*I); //8xI vector
            float *vec2 = gn_X_temp + (i*num_groups*D*I + (j+16)*I); //8xI vector
            float *vec3 = gn_X_temp + (i*num_groups*D*I + (j+24)*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec0[c*8]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec1[c*8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec2[c*8]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec3[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec0[r*8]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec1[r*8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec2[r*8]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec3[r*8]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec0[r*8], res0);
                _mm256_storeu_ps(&vec1[r*8], res1);
                _mm256_storeu_ps(&vec2[r*8], res2);
                _mm256_storeu_ps(&vec3[r*8], res3);
            }
        }
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*8 + k];
                }
            }
        }
    }
    

    //SCALING
    if ( scaling ){

        //just a transpose
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + i*I + ii] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        

        //mat mult and add
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    float row0 = b[0];
                    float row1 = b[1];

                    FLOP(4);
                    row0 = fmaf(W[0], xv[0], row0);
                    row1 = fmaf(W[2], xv[0], row1);

                    FLOP(4);
                    row0 = fmaf(W[1], xv[1], row0);
                    row1 = fmaf(W[3], xv[1], row1);

                    xv[0] = row0;
                    xv[1] = row1;
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v26(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {
    // Track bytes read from input arrays
    BYTES_READ(B * C * D * I * sizeof(float));       // x
    if (running) {
        BYTES_READ(C * I * sizeof(float));           // running_mean_orig
        BYTES_READ(C * I * I * sizeof(float));       // running_cov_orig
    }
    if (scaling) {
        BYTES_READ((C / num_groups) * I * I * sizeof(float));       // weight_orig
        BYTES_READ((C / num_groups) * I * I * sizeof(float));       // bias_orig
    }
    
    // Track bytes written to output
    BYTES_WRITTEN(B * C * D * I * sizeof(float));    // x_norm
    if (running && training) {
        BYTES_WRITTEN(C * I * sizeof(float));        // running_mean_orig (updated)
        BYTES_WRITTEN(C * I * I * sizeof(float));    // running_cov_orig (updated)
    }
    

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //CENTERING
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();
        __m256 accum4 = _mm256_setzero_ps();
        __m256 accum5 = _mm256_setzero_ps();
        __m256 accum6 = _mm256_setzero_ps();
        __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*8]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+1)*8]);
            __m256 row2 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+2)*8]);
            __m256 row3 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+3)*8]);
            __m256 row4 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*8]);
            __m256 row5 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+5)*8]);
            __m256 row6 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+6)*8]);
            __m256 row7 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+7)*8]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, row2);
            FLOP(8);
            accum3 = _mm256_add_ps(accum3, row3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, row4);
            FLOP(8);
            accum5 = _mm256_add_ps(accum5, row5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, row6);
            FLOP(8);
            accum7 = _mm256_add_ps(accum7, row7);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        FLOP(8);
        accum2 = _mm256_add_ps(accum2, accum3);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum5);
        FLOP(8);
        accum6 = _mm256_add_ps(accum6, accum7);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum2);
        FLOP(8);
        accum4 = _mm256_add_ps(accum4, accum6);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum4);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //CALCULATE COV
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum0 = _mm256_setzero_ps();
            __m256 accum1 = _mm256_setzero_ps();
            __m256 accum2 = _mm256_setzero_ps();
            __m256 accum3 = _mm256_setzero_ps();
            __m256 accum4 = _mm256_setzero_ps();
            __m256 accum5 = _mm256_setzero_ps();
            __m256 accum6 = _mm256_setzero_ps();
            __m256 accum7 = _mm256_setzero_ps(); // use 8 accumulators because _mm256_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m256 j_part0 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 j_part1 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m256 j_part2 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m256 j_part3 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m256 j_part4 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m256 j_part5 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m256 j_part6 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m256 j_part7 = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m256 l_part0 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m256 l_part1 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m256 l_part2 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m256 l_part3 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m256 l_part4 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m256 l_part5 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m256 l_part6 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m256 l_part7 = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(16);
                accum0 = _mm256_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(16);
                accum1 = _mm256_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(16);
                accum2 = _mm256_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(16);
                accum3 = _mm256_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(16);
                accum4 = _mm256_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(16);
                accum5 = _mm256_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(16);
                accum6 = _mm256_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(16);
                accum7 = _mm256_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum1);
            FLOP(8);
            accum2 = _mm256_add_ps(accum2, accum3);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum5);
            FLOP(8);
            accum6 = _mm256_add_ps(accum6, accum7);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum2);
            FLOP(8);
            accum4 = _mm256_add_ps(accum4, accum6);
            FLOP(8);
            accum0 = _mm256_add_ps(accum0, accum4);
            FLOP(8);
            accum0 = _mm256_mul_ps(accum0, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }

    //MAXIMUM
    for(int i = 0; i < dim0; i++){
        __m256 maxv0 = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        __m256 maxv1 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 8]);
        __m256 maxv2 = _mm256_loadu_ps(&gn_cov_temp[i*I*I + 16]);
        __m256 maxv3 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 24);
        __m256 maxv4 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 32);
        __m256 maxv5 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 40);
        __m256 maxv6 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 48);
        __m256 maxv7 = _mm256_loadu_ps(&gn_cov_temp[i*I*I] + 56);

        maxv0 = _mm256_max_ps(maxv0, maxv1);
        maxv2 = _mm256_max_ps(maxv2, maxv3);
        maxv4 = _mm256_max_ps(maxv4, maxv5);
        maxv6 = _mm256_max_ps(maxv6, maxv7);
        maxv0 = _mm256_max_ps(maxv0, maxv2);
        maxv4 = _mm256_max_ps(maxv4, maxv6);
        maxv0 = _mm256_max_ps(maxv0, maxv4);
        _mm256_storeu_ps(gn_max_temp, maxv0);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }

    //CHOLESKY
    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        int ii, jj, k;
        // Loop over columns
        for(jj = 0; jj < n; jj++) {
            // i = j
            __m256 x = _mm256_loadu_ps(&a[8*(jj*n+jj)]); //A_jj 
            
            for(k = 0; k < jj; k++){
                __m256 temp = _mm256_loadu_ps(&a[8*(jj*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp); // L_jk L_jk
            }

            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            _mm256_storeu_ps(&a[8*(jj*n+jj)], x); // L_jj

            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            
            // i != j 
            for(ii = jj+1; ii < n; ii++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(ii*n+jj)]);  // A_ij 

                for(k = 0; k < jj; k++){
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(ii*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(jj*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1); // L_ik L_ij 
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r); // L_ij = x / L_jj 
                _mm256_storeu_ps(&a[8*(ii*n+jj)], tostore);
            }
        }
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //BACK SUBSTITUTION
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
        for(int j = 0; j < num_groups*D; j += 32){
            int pos = (j%3)*32*I;
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[pos + l*32 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp; //32xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec[c*32]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec[c*32 + 8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec[c*32 + 16]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec[c*32 + 24]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec[r*32]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec[r*32 + 8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec[r*32 + 16]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec[r*32 + 24]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec[r*32], res0);
                _mm256_storeu_ps(&vec[r*32 + 8], res1);
                _mm256_storeu_ps(&vec[r*32 + 16], res2);
                _mm256_storeu_ps(&vec[r*32 + 24], res3);
            }
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[pos + l*32 + k];
                }
            }
        }
    }
    
    
    //SCALING
    if ( scaling ){
        //just a transpose (no replication)
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + ii*I + i] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        
        //mat mult and add bias
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *WT = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    //1) load in b as base
                    //4 accumulators
                    __m256 accum0 = _mm256_loadu_ps(b);
                    __m256 accum1 = _mm256_setzero_ps();
                    __m256 accum2 = _mm256_setzero_ps();
                    __m256 accum3 = _mm256_setzero_ps();
                    
                    //2) iterate W cols with fma (note we have W^T)
                    //use each of the 4 accum to do 2 cols
                    for(int ii = 0; ii < 2; ii++){
                        __m256 col0 = _mm256_loadu_ps(&WT[(ii*4 + 0)*I]);
                        __m256 col1 = _mm256_loadu_ps(&WT[(ii*4 + 1)*I]);
                        __m256 col2 = _mm256_loadu_ps(&WT[(ii*4 + 2)*I]);
                        __m256 col3 = _mm256_loadu_ps(&WT[(ii*4 + 3)*I]);

                        __m256 xv_part0 = _mm256_broadcast_ss(&xv[(ii*4 + 0)]);
                        __m256 xv_part1 = _mm256_broadcast_ss(&xv[(ii*4 + 1)]);
                        __m256 xv_part2 = _mm256_broadcast_ss(&xv[(ii*4 + 2)]);
                        __m256 xv_part3 = _mm256_broadcast_ss(&xv[(ii*4 + 3)]);

                        FLOP(16);
                        accum0 = _mm256_fmadd_ps(col0, xv_part0, accum0);
                        FLOP(16);
                        accum1 = _mm256_fmadd_ps(col1, xv_part1, accum1);
                        FLOP(16);
                        accum2 = _mm256_fmadd_ps(col2, xv_part2, accum2); 
                        FLOP(16);
                        accum3 = _mm256_fmadd_ps(col3, xv_part3, accum3);
                    }

                    //reduction on accums
                    FLOP(8);
                    accum0 = _mm256_add_ps(accum0, accum1);
                    FLOP(8);
                    accum2 = _mm256_add_ps(accum2, accum3);
                    FLOP(8);
                    accum0 = _mm256_add_ps(accum0, accum2);

                    //3) store back result
                    _mm256_storeu_ps(xv, accum0);
                    
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v27(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 4); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)
   
    //CENTERING
    __m128 mean_scale = _mm_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m128 accum0 = _mm_setzero_ps();
        __m128 accum1 = _mm_setzero_ps();
        __m128 accum2 = _mm_setzero_ps();
        __m128 accum3 = _mm_setzero_ps();
        __m128 accum4 = _mm_setzero_ps();
        __m128 accum5 = _mm_setzero_ps();
        __m128 accum6 = _mm_setzero_ps();
        __m128 accum7 = _mm_setzero_ps(); // use 8 accumulators because _mm_fmadd_ps has latency 4 and intel tp 0.5
        for(int k = 0; k < num_groups*D; k+=8){
            __m128 row0 = _mm_loadu_ps(&x[i*num_groups*D*I + k*I]);
            __m128 row1 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+1)*I]);
            __m128 row2 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+2)*I]);
            __m128 row3 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+3)*I]);
            __m128 row4 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+4)*I]);
            __m128 row5 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+5)*I]);
            __m128 row6 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+6)*I]);
            __m128 row7 = _mm_loadu_ps(&x[i*num_groups*D*I + (k+7)*I]);

            FLOP(4);
            accum0 = _mm_add_ps(accum0, row0);
            FLOP(4);
            accum1 = _mm_add_ps(accum1, row1);
            FLOP(4);
            accum2 = _mm_add_ps(accum2, row2);
            FLOP(4);
            accum3 = _mm_add_ps(accum3, row3);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, row4);
            FLOP(4);
            accum5 = _mm_add_ps(accum5, row5);
            FLOP(4);
            accum6 = _mm_add_ps(accum6, row6);
            FLOP(4);
            accum7 = _mm_add_ps(accum7, row7);
        }
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum1);
        FLOP(4);
        accum2 = _mm_add_ps(accum2, accum3);
        FLOP(4);
        accum4 = _mm_add_ps(accum4, accum5);
        FLOP(4);
        accum6 = _mm_add_ps(accum6, accum7);
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum2);
        FLOP(4);
        accum4 = _mm_add_ps(accum4, accum6);
        FLOP(4);
        accum0 = _mm_add_ps(accum0, accum4);
        FLOP(4);
        accum0 = _mm_mul_ps(accum0, mean_scale);
        
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m128 row = _mm_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(4);
                row = _mm_sub_ps(row, accum0);
                _mm_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //CALCULATE COV
    __m128 cov_scale = _mm_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m128 accum0 = _mm_setzero_ps();
            __m128 accum1 = _mm_setzero_ps();
            __m128 accum2 = _mm_setzero_ps();
            __m128 accum3 = _mm_setzero_ps();
            __m128 accum4 = _mm_setzero_ps();
            __m128 accum5 = _mm_setzero_ps();
            __m128 accum6 = _mm_setzero_ps();
            __m128 accum7 = _mm_setzero_ps(); // use 8 accumulators because _mm_fmadd_ps has latency 4 and intel tp 0.5
            for(int k = 0; k < num_groups*D; k+=8){
                __m128 j_part0 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m128 j_part1 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+1)*I + j]);
                __m128 j_part2 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+2)*I + j]);
                __m128 j_part3 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+3)*I + j]);
                __m128 j_part4 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+4)*I + j]);
                __m128 j_part5 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+5)*I + j]);
                __m128 j_part6 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+6)*I + j]);
                __m128 j_part7 = _mm_broadcast_ss(&x_norm[i*num_groups*D*I + (k+7)*I + j]);

                __m128 l_part0 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                __m128 l_part1 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+1)*I]);
                __m128 l_part2 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+2)*I]);
                __m128 l_part3 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+3)*I]);
                __m128 l_part4 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+4)*I]);
                __m128 l_part5 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+5)*I]);
                __m128 l_part6 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+6)*I]);
                __m128 l_part7 = _mm_loadu_ps(&x_norm[i*num_groups*D*I + (k+7)*I]);

                FLOP(8);
                accum0 = _mm_fmadd_ps(j_part0, l_part0, accum0);
                FLOP(8);
                accum1 = _mm_fmadd_ps(j_part1, l_part1, accum1);
                FLOP(8);
                accum2 = _mm_fmadd_ps(j_part2, l_part2, accum2);
                FLOP(8);
                accum3 = _mm_fmadd_ps(j_part3, l_part3, accum3);
                FLOP(8);
                accum4 = _mm_fmadd_ps(j_part4, l_part4, accum4);
                FLOP(8);
                accum5 = _mm_fmadd_ps(j_part5, l_part5, accum5);
                FLOP(8);
                accum6 = _mm_fmadd_ps(j_part6, l_part6, accum6);
                FLOP(8);
                accum7 = _mm_fmadd_ps(j_part7, l_part7, accum7);
            }
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum1);
            FLOP(4);
            accum2 = _mm_add_ps(accum2, accum3);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, accum5);
            FLOP(4);
            accum6 = _mm_add_ps(accum6, accum7);
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum2);
            FLOP(4);
            accum4 = _mm_add_ps(accum4, accum6);
            FLOP(4);
            accum0 = _mm_add_ps(accum0, accum4);
            FLOP(4);
            accum0 = _mm_mul_ps(accum0, cov_scale);
            _mm_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum0);
        }
    }
    
    //MAXIMUM
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}


    //CHOLESKY
    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        int ii, jj, k;
        // Loop over columns
        for(jj = 0; jj < n; jj++) {
            // i = j
            __m256 x = _mm256_loadu_ps(&a[8*(jj*n+jj)]); //A_jj 
            
            for(k = 0; k < jj; k++){
                __m256 temp = _mm256_loadu_ps(&a[8*(jj*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp); // L_jk L_jk
            }

            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            _mm256_storeu_ps(&a[8*(jj*n+jj)], x); // L_jj

            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            
            // i != j 
            for(ii = jj+1; ii < n; ii++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(ii*n+jj)]);  // A_ij 

                for(k = 0; k < jj; k++){
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(ii*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(jj*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1); // L_ik L_ij 
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r); // L_ij = x / L_jj 
                _mm256_storeu_ps(&a[8*(ii*n+jj)], tostore);
            }
        }
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //BACK SUBSTITUTION
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
        for(int j = 0; j < num_groups*D; j += 32){
            int pos = (j%3)*32*I;
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[pos + l*32 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp; //32xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec[c*32]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec[c*32 + 8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec[c*32 + 16]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec[c*32 + 24]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec[r*32]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec[r*32 + 8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec[r*32 + 16]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec[r*32 + 24]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec[r*32], res0);
                _mm256_storeu_ps(&vec[r*32 + 8], res1);
                _mm256_storeu_ps(&vec[r*32 + 16], res2);
                _mm256_storeu_ps(&vec[r*32 + 24], res3);
            }
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[pos + l*32 + k];
                }
            }
        }
    }

    //SCALING
    if ( scaling ){
        //just a transpose (no replications)
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + ii*I + i] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        

        //mat mult and add
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *WT = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    //1) load in b as base
                    //2 accumulators
                    __m128 accum0 = _mm_loadu_ps(b);
                    __m128 accum1 = _mm_setzero_ps();
                    
                    //2) iterate W cols with fma (note we have W^T)
                    //use each of the 2 accum to do 2 cols
                    for(int ii = 0; ii < 2; ii++){
                        __m128 col0 = _mm_loadu_ps(&WT[(ii*2 + 0)*I]);
                        __m128 col1 = _mm_loadu_ps(&WT[(ii*2 + 1)*I]);

                        __m128 xv_part0 = _mm_broadcast_ss(&xv[(ii*2 + 0)]);
                        __m128 xv_part1 = _mm_broadcast_ss(&xv[(ii*2 + 1)]);

                        FLOP(8);
                        accum0 = _mm_fmadd_ps(col0, xv_part0, accum0);
                        FLOP(8);
                        accum1 = _mm_fmadd_ps(col1, xv_part1, accum1);
                    }

                    //reduction on accums
                    FLOP(4);
                    accum0 = _mm_add_ps(accum0, accum1);

                    //3) store back result
                    _mm_storeu_ps(xv, accum0);
                    
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v28(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 2); //optimized for 3D (2D or 1D need addaptions to the cov calculation)
    assert(dim0 >= 8); //use gn_max_temp different now so needs to be at least 8 long (would need adaption but could also be changed)
    assert(dim0 % 8 == 0); //for cholesky decomp with avx (could be lifted with rollup)
    assert(D*num_groups > I); //for the cholesky decomp reshape, so it fits into gn_X_temp
    assert((D*num_groups)%32 == 0); //needs to be divisible for backsub vectorization, also needed for the accumulators in cov calculation (could be fixed with rollup)

    //CENTERING
    __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps(); // use 2 accumulators with 4 rows each
        for(int k = 0; k < num_groups*D; k+=8){
            __m256 row0 = _mm256_loadu_ps(&x[i*num_groups*D*I + k*I]);
            __m256 row1 = _mm256_loadu_ps(&x[i*num_groups*D*I + (k+4)*I]);

            FLOP(8);
            accum0 = _mm256_add_ps(accum0, row0);
            FLOP(8);
            accum1 = _mm256_add_ps(accum1, row1);
        }
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum1);
        __m256 accum0_high = _mm256_permute2f128_ps(accum0, accum0, 0x01);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum0_high);
        __m256 accum0_even_odd = _mm256_shuffle_ps(accum0, accum0, (0x01 << 6) | (0x00 << 4) | (0x03 << 2) | 0x02);
        FLOP(8);
        accum0 = _mm256_add_ps(accum0, accum0_even_odd);
        FLOP(8);
        accum0 = _mm256_mul_ps(accum0, mean_scale); //4 copies of the 2 mean floats
        

        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k += 4){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum0);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //CACULATE COV
    float cov_scale = 1.0f / ((float)(D*num_groups));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            float accum0_col0 = 0.0f;
            float accum0_col1 = 0.0f;
            float accum1_col0 = 0.0f;
            float accum1_col1 = 0.0f;
            float accum2_col0 = 0.0f;
            float accum2_col1 = 0.0f;
            float accum3_col0 = 0.0f;
            float accum3_col1 = 0.0f; // use 4 accumulators
            for(int k = 0; k < num_groups*D; k+=4){
                float j_part0 = x_norm[i*num_groups*D*I + k*I + j];
                float j_part1 = x_norm[i*num_groups*D*I + (k+1)*I + j];
                float j_part2 = x_norm[i*num_groups*D*I + (k+2)*I + j];
                float j_part3 = x_norm[i*num_groups*D*I + (k+3)*I + j];

                FLOP(8);
                accum0_col0 = fmaf(j_part0, x_norm[i*num_groups*D*I + k*I], accum0_col0);
                accum1_col0 = fmaf(j_part1, x_norm[i*num_groups*D*I + (k+1)*I], accum1_col0);
                accum2_col0 = fmaf(j_part2, x_norm[i*num_groups*D*I + (k+2)*I], accum2_col0);
                accum3_col0 = fmaf(j_part3, x_norm[i*num_groups*D*I + (k+3)*I], accum3_col0);

                FLOP(8);
                accum0_col1 = fmaf(j_part0, x_norm[i*num_groups*D*I + k*I + 1], accum0_col1);
                accum1_col1 = fmaf(j_part1, x_norm[i*num_groups*D*I + (k+1)*I + 1], accum1_col1);
                accum2_col1 = fmaf(j_part2, x_norm[i*num_groups*D*I + (k+2)*I + 1], accum2_col1);
                accum3_col1 = fmaf(j_part3, x_norm[i*num_groups*D*I + (k+3)*I + 1], accum3_col1);
            }
            accum0_col0 = accum0_col0 + accum1_col0;
            accum2_col0 = accum2_col0 + accum3_col0;
            accum0_col0 = accum0_col0 + accum2_col0;

            accum0_col1 = accum0_col1 + accum1_col1;
            accum2_col1 = accum2_col1 + accum3_col1;
            accum0_col1 = accum0_col1 + accum2_col1;

            gn_cov_temp[i*I*I + j*I] = accum0_col0 * cov_scale;
            gn_cov_temp[i*I*I + j*I + 1] = accum0_col1 * cov_scale;
        }
    }
    
    //MAXIMUM
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}
    


    //CHOLESKY
    __m256 one = _mm256_set1_ps(1.0f);
    for(int i = 0; i < dim0; i+=8){
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_X_temp[i*I*I + k*8 + j] = gn_cov_temp[i*I*I + j*I*I + k];
            }
        }
        int n = I;
        float *a = &gn_X_temp[i*I*I];
        int ii, jj, k;
        // Loop over columns
        for(jj = 0; jj < n; jj++) {
            // i = j
            __m256 x = _mm256_loadu_ps(&a[8*(jj*n+jj)]); //A_jj 
            
            for(k = 0; k < jj; k++){
                __m256 temp = _mm256_loadu_ps(&a[8*(jj*n+k)]);
                FLOP(8);
                temp = _mm256_mul_ps(temp, temp);
                FLOP(8);
                x = _mm256_sub_ps(x, temp); // L_jk L_jk
            }

            FLOP(8*30);
            x = _mm256_sqrt_ps(x);

            _mm256_storeu_ps(&a[8*(jj*n+jj)], x); // L_jj

            FLOP(8);
            __m256 r = _mm256_div_ps(one, x);
            
            // i != j 
            for(ii = jj+1; ii < n; ii++) {
                __m256 x1 = _mm256_loadu_ps(&a[8*(ii*n+jj)]);  // A_ij 

                for(k = 0; k < jj; k++){
                    __m256 temp0 = _mm256_loadu_ps(&a[8*(ii*n+k)]);
                    __m256 temp1 = _mm256_loadu_ps(&a[8*(jj*n+k)]);
                    FLOP(8);
                    temp0 = _mm256_mul_ps(temp0, temp1); // L_ik L_ij 
                    FLOP(8);
                    x1 = _mm256_sub_ps(x1, temp0);
                }
                FLOP(8);
                __m256 tostore = _mm256_mul_ps(x1, r); // L_ij = x / L_jj 
                _mm256_storeu_ps(&a[8*(ii*n+jj)], tostore);
            }
        }
        for(int j = 0; j < 8; j++){ //jump form cov mat to cov mat
            for(int k = 0; k < I*I; k ++){ //go through cov mat
                gn_cov_temp[i*I*I + j*I*I + k] = gn_X_temp[i*I*I + k*8 + j];
            }
        }
    }

    //BACK SUBSTITUTION
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
        for(int j = 0; j < num_groups*D; j += 32){
            int pos = (j%3)*32*I;
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[pos + l*32 + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp; //32xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                __m256 rowsum0 = _mm256_setzero_ps();
                __m256 rowsum1 = _mm256_setzero_ps();
                __m256 rowsum2 = _mm256_setzero_ps();
                __m256 rowsum3 = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem0 = _mm256_loadu_ps(&vec[c*32]);
                    __m256 vec_elem1 = _mm256_loadu_ps(&vec[c*32 + 8]);
                    __m256 vec_elem2 = _mm256_loadu_ps(&vec[c*32 + 16]);
                    __m256 vec_elem3 = _mm256_loadu_ps(&vec[c*32 + 24]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum0 = _mm256_fmadd_ps(mat_elem, vec_elem0, rowsum0);
                    FLOP(16);
                    rowsum1 = _mm256_fmadd_ps(mat_elem, vec_elem1, rowsum1);
                    FLOP(16);
                    rowsum2 = _mm256_fmadd_ps(mat_elem, vec_elem2, rowsum2);
                    FLOP(16);
                    rowsum3 = _mm256_fmadd_ps(mat_elem, vec_elem3, rowsum3);
                }
                __m256 vec_elem0 = _mm256_loadu_ps(&vec[r*32]);
                __m256 vec_elem1 = _mm256_loadu_ps(&vec[r*32 + 8]);
                __m256 vec_elem2 = _mm256_loadu_ps(&vec[r*32 + 16]);
                __m256 vec_elem3 = _mm256_loadu_ps(&vec[r*32 + 24]);
                FLOP(8);
                rowsum0 = _mm256_sub_ps(vec_elem0, rowsum0);
                FLOP(8);
                rowsum1 = _mm256_sub_ps(vec_elem1, rowsum1);
                FLOP(8);
                rowsum2 = _mm256_sub_ps(vec_elem2, rowsum2);
                FLOP(8);
                rowsum3 = _mm256_sub_ps(vec_elem3, rowsum3);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res0 = _mm256_mul_ps(rowsum0, mat_elem);
                FLOP(8);
                __m256 res1 = _mm256_mul_ps(rowsum1, mat_elem);
                FLOP(8);
                __m256 res2 = _mm256_mul_ps(rowsum2, mat_elem);
                FLOP(8);
                __m256 res3 = _mm256_mul_ps(rowsum3, mat_elem);
                _mm256_storeu_ps(&vec[r*32], res0);
                _mm256_storeu_ps(&vec[r*32 + 8], res1);
                _mm256_storeu_ps(&vec[r*32 + 16], res2);
                _mm256_storeu_ps(&vec[r*32 + 24], res3);
            }
            for (int k = 0; k < 32; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[pos + l*32 + k];
                }
            }
        }
    }
    

    //SCALING
    if ( scaling ){
        //just a transpose (no replication)
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                        gn_weight_temp[j*I*I + i*I + ii] = weight_orig[i*I*group_size + ii*group_size + j];
                }
            }
        }

        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }

        //mat mult and add
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_temp + (i%group_size)*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;

                    float row0 = b[0];
                    float row1 = b[1];

                    FLOP(4);
                    row0 = fmaf(W[0], xv[0], row0);
                    row1 = fmaf(W[2], xv[0], row1);

                    FLOP(4);
                    row0 = fmaf(W[1], xv[1], row0);
                    row1 = fmaf(W[3], xv[1], row1);

                    xv[0] = row0;
                    xv[1] = row1;
                }
            }
        }
    }

    return;
}



static void optimized_groupnorm_all_dim(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    switch (I){
        case 8:
            optimized_groupnorm_v26(
                                        x,
                                        B,
                                        C,
                                        D,
                                        I,
                                        num_groups,
                                        running,
                                        running_mean_orig,
                                        running_cov_orig,
                                        scaling,
                                        weight_orig,
                                        bias_orig,
                                        training,
                                        momentum,
                                        eps,
                                        x_norm
                                    );
            break;
        case 4:
            optimized_groupnorm_v27(
                                        x,
                                        B,
                                        C,
                                        D,
                                        I,
                                        num_groups,
                                        running,
                                        running_mean_orig,
                                        running_cov_orig,
                                        scaling,
                                        weight_orig,
                                        bias_orig,
                                        training,
                                        momentum,
                                        eps,
                                        x_norm
                                    );
            break;
        case 2:
            optimized_groupnorm_v28(
                                        x,
                                        B,
                                        C,
                                        D,
                                        I,
                                        num_groups,
                                        running,
                                        running_mean_orig,
                                        running_cov_orig,
                                        scaling,
                                        weight_orig,
                                        bias_orig,
                                        training,
                                        momentum,
                                        eps,
                                        x_norm
                                    );
            break;
        default:
            printf("Invalid dimension value I=%d\n", I);
    }

    return;
}

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//README section
/*
    - The baseline versions were developed in /csrc/clifford_groupnorm.c. For the optimization we moved
      to this file and started to use global variables for the _temp arrays
    - Starting with the python version the variable num_groups was adapted. But from the way the varaible
      is used, it seems that this variable refers to the size of a group. An example that would suggest this is
      that when we sum the mean we sum over num_groups*D multivectors, but rather one would expect to sum over
      "size of one group"*D when calculating the mean within one group. Whether the naming as it is now is "right"
      or "wrong" is not clear. This is mentioned here to aid understanding.
    - Versions v1 - v12, v16 - v20, v23 and v26 are 3D only
    - Versions v14, v21, v24 and v27 are 2D only
    - Versions v15, v22, v25 and v28 are 1D only
    - Versions baseline, v0, v13 and all_dim work on 1D, 2D and 3D
    - Versions v26 (3D), v27 (2D) and v28 (1D) are the most optimzed versions for each dimension
*/
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


// ——— Register them in the order you want them tested ———
void register_groupnorm_functions(void) {
    add_groupnorm_function(baseline_groupnorm,      "baseline");
    /*
    add_groupnorm_function(optimized_groupnorm_v0,  "optimized_groupnorm_v0"); //invert diag back sub
    add_groupnorm_function(optimized_groupnorm_v1,  "optimized_groupnorm_v1"); //mean vectorized
    add_groupnorm_function(optimized_groupnorm_v2,  "optimized_groupnorm_v2"); //cov vectorized
    */
    add_groupnorm_function(optimized_groupnorm_v3,  "mean_cov_max opt"); //max permute temp vars
    /*
    add_groupnorm_function(optimized_groupnorm_v4,  "optimized_groupnorm_v4"); //vectorize backsubstitution
    add_groupnorm_function(optimized_groupnorm_v5,  "optimized_groupnorm_v5"); //dont transpose cholesky decomp
    add_groupnorm_function(optimized_groupnorm_v6,  "optimized_groupnorm_v6"); //8 accumulators for cov
    add_groupnorm_function(optimized_groupnorm_v7,  "optimized_groupnorm_v7"); //avx for cholesky decomp
    //add_groupnorm_function(optimized_groupnorm_v8,  "optimized_groupnorm_v8"); //rsqrt (not accurate enough)
    add_groupnorm_function(optimized_groupnorm_v9,  "optimized_groupnorm_v9"); //8 accumulators for mean (base v7)
    add_groupnorm_function(optimized_groupnorm_v10, "optimized_groupnorm_v10"); //reduction for max
    add_groupnorm_function(optimized_groupnorm_v11, "optimized_groupnorm_v11"); //backsub manual 2 way unroll
    */
    add_groupnorm_function(optimized_groupnorm_v12, "decomp_backsub opt"); //backsub manual 4 way unroll

    //add_groupnorm_function(optimized_groupnorm_v13, "optimized_groupnorm_v13"); //v12 with unoptimized 1D and 2D support (strong 3D slowdown)
    //add_groupnorm_function(optimized_groupnorm_v14, "optimized_groupnorm_v14"); //optimzed 2D only (base v12)
    //add_groupnorm_function(optimized_groupnorm_v15, "optimized_groupnorm_v15"); //optimzed 1D only (base v12)
    
    //add_groupnorm_function(optimized_groupnorm_v16, "optimized_groupnorm_v16"); //backsub manual 4 way unroll better layout (base v12)
    //add_groupnorm_function(optimized_groupnorm_v17, "optimized_groupnorm_v17");   //fuse reshape and backsub (worse for 32)

    //add_groupnorm_function(optimized_groupnorm_v18, "optimized_groupnorm_v18"); //no scaling weight temp array blowup (based v12)
    //add_groupnorm_function(optimized_groupnorm_v19, "optimized_groupnorm_v19"); //vectorize scaling
    add_groupnorm_function(optimized_groupnorm_v20, "scaling opt"); //vectorized scaling 4 accumlators
    //add_groupnorm_function(optimized_groupnorm_v21, "optimized_groupnorm_v21"); //2D version, scaling optimization (base v14) (scaling base v20)
    //add_groupnorm_function(optimized_groupnorm_v22, "optimized_groupnorm_v22"); //1D version, scaling optimization (base v15) (scaling base v21)

    //add_groupnorm_function(optimized_groupnorm_v23, "optimized_groupnorm_v23"); //fuse permutes into step main loops to improve cache locality (base v20)
    //add_groupnorm_function(optimized_groupnorm_v24, "optimized_groupnorm_v24"); //2D version, fuse permutes (base v21) (decomp, backsub base v23)
    //add_groupnorm_function(optimized_groupnorm_v25, "optimized_groupnorm_v25"); //1D version, fuse permutes (base v22) (decomp, backsub base v23)

    add_groupnorm_function(optimized_groupnorm_v26, "cache opt"); //add the reshape form v17 (base v23)
    //add_groupnorm_function(optimized_groupnorm_v27, "optimized_groupnorm_v27"); //2D version, add v26 Backsub permute (base v24) (backsub base v26)
    //add_groupnorm_function(optimized_groupnorm_v28, "optimized_groupnorm_v28"); //1D version, add v26 Backsub permute (base v25) (backsub base v26)

    //add_groupnorm_function(optimized_groupnorm_all_dim, "optimized_groupnorm_all_dim"); //switch to best version of each dimention
}
