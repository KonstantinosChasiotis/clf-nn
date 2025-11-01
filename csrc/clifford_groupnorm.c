#include "clifford_groupnorm.h"
#include "flops.h"

void clifford_groupnorm( //wrapper around with memory allocation around clifford_groupnorm_temp
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float *running_mean_orig,
    float *running_cov_orig,
    bool scaling,
    const float *weight_orig,
    const float *bias_orig,
    bool training,
    float momentum,
    float eps,
    float *x_norm){
    
    float * running_mean_temp = NULL;
    float * running_cov_temp = NULL;
    float * weight_temp = NULL;
    float * bias_temp = NULL;
    float * weight_perm_temp = NULL;
    float * X_temp = NULL;
    float * mean_temp = NULL;
    float * cov_temp = NULL;
    float * max_temp = NULL;

    clifford_groupnorm_alloctemp(
        B,
        C,
        D,
        _I,
        num_groups,
        running,
        &running_mean_temp,
        &running_cov_temp,
        scaling,
        &weight_temp,
        &weight_perm_temp,
        &bias_temp,
        &mean_temp,
        &X_temp,
        &cov_temp,
        &max_temp);
        
    clifford_groupnorm_baseline(
    //clifford_groupnorm_invdiag(
        x,
        B,
        C,
        D,
        _I,
        num_groups,
        running,
        running_mean_orig,
        running_mean_temp,
        running_cov_orig,
        running_cov_temp,
        mean_temp,
        X_temp,
        cov_temp,
        max_temp,
        scaling,
        weight_orig,
        weight_temp,
        weight_perm_temp,
        bias_orig,
        bias_temp,
        training,
        momentum,
        eps,
        x_norm);

    clifford_groupnorm_freetemp(
        B,
        C,
        D,
        _I,
        num_groups,
        running,
        &running_mean_temp,
        &running_cov_temp,
        scaling,
        &weight_temp,
        &weight_perm_temp,
        &bias_temp,
        &mean_temp,
        &X_temp,
        &cov_temp,
        &max_temp);
        
    return;
}

void clifford_groupnorm_alloctemp(
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float **running_mean_temp,
    float **running_cov_temp,
    bool scaling,
    float **weight_temp,
    float **weight_perm_temp,
    float **bias_temp,
    float **mean_temp,
    float **X_temp,
    float **cov_temp,
    float **max_temp){

    //basic constants
    int dim0 = (B*C)/num_groups;
    //int group_size = C/num_groups;

    //allocate temp memory (zeroed!!!)
    *running_mean_temp = NULL;
    *running_cov_temp = NULL;
    if (running){
        size_t running_mean_size = _I * dim0;
        *running_mean_temp = malloc(running_mean_size * sizeof(float));
        size_t running_cov_size = _I* _I * dim0;
        *running_cov_temp = malloc(running_cov_size * sizeof(float));
    }
    *weight_temp = NULL;
    *bias_temp = NULL;
    *weight_perm_temp = NULL;
    if (scaling){
        size_t weight_size = _I* _I * dim0;
        *weight_temp = malloc(weight_size * sizeof(float));
        size_t bias_size = _I * dim0;
        *bias_temp = malloc(bias_size * sizeof(float));
        *weight_perm_temp = malloc(weight_size * sizeof(float));
    }
    size_t x_size = dim0 * num_groups * D * _I;
    *X_temp = malloc(x_size * sizeof(float));
    size_t mean_size = dim0 * _I;
    *mean_temp = malloc(mean_size * sizeof(float));
    size_t cov_size = dim0*_I*_I;
    *cov_temp = malloc(cov_size * sizeof(float));
    size_t max_size = dim0;
    *max_temp = malloc(max_size * sizeof(float));
    return;
}

void clifford_groupnorm_freetemp(
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float **running_mean_temp,
    float **running_cov_temp,
    bool scaling,
    float **weight_temp,
    float **weight_perm_temp,
    float **bias_temp,
    float **mean_temp,
    float **X_temp,
    float **cov_temp,
    float **max_temp){

    if (running){
        free(*running_mean_temp);
        free(*running_cov_temp);
    }

    if (scaling){
        free(*weight_temp);
        free(*bias_temp);
        free(*weight_perm_temp);
    }

    //free all the temp arrays
    free(*max_temp);
    free(*cov_temp);
    free(*X_temp);
    free(*mean_temp);
    
    return;
}   

void clifford_groupnorm_baseline(
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float *running_mean_orig,
    float *running_mean_temp,
    float *running_cov_orig,
    float *running_cov_temp,
    float *mean_temp,
    float *X_temp,
    float *cov_temp,
    float *max_temp,
    bool scaling,
    const float *weight_orig,
    float *weight_temp,
    float *weight_perm_temp,
    const float *bias_orig,
    float *bias_temp,
    bool training,
    float momentum,
    float eps,
    float *x_norm){

    //basic constants
    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    //blow up running mean and cov along the batch dim
    if ( running ){
        for(int i = 0; i < _I; i++){ //COST: 0 flops
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    running_mean_temp[i*group_size*B + j*B + l] = running_mean_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        running_cov_temp[i*_I*group_size*B + ii*group_size*B + j*B + l] = running_cov_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
    }

    //mean
    if (training || (!running)) {
        int mean_size = dim0 * _I;
        for (int i = 0; i < mean_size; i++){
            mean_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < _I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                        FLOP(1);
                        mean_temp[i*_I + l] += x[i*num_groups*D*_I + j*D*_I + k*_I + l];
                    }
                }
            }
        }
        for(int i = 0; i < dim0; i++){
            for(int l = 0; l < _I; l++){ //COST: dim0*I
                FLOP(1);
                mean_temp[i*_I + l] /= (D*num_groups);
            }
        }
        if (running){ //update running mean
            for(int i = 0; i < _I; i++){
                for(int j = 0; j < dim0; j++){
                    FLOP(2);
                    FLOP(1);
                    running_mean_temp[i*dim0 + j] += momentum *(mean_temp[j*_I + i] - running_mean_temp[i*dim0 + j]);
                }
            }
        }
    } else { //if not training use running mean
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < dim0; j++){
                    mean_temp[j*_I + i] = running_mean_temp[i*dim0 + j];
            }
        }
    }

    //subtract mean form x
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    FLOP(1);
                    x_norm[i*num_groups*D*_I + j*D*_I + k*_I + l] = x[i*num_groups*D*_I + j*D*_I + k*_I + l] - mean_temp[i*_I + l];
                }
            }
        }
    }

    //permute: dim0, num_groups, D, _I -> dim0, _I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    X_temp[i*_I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*_I + j*D*_I + k*_I + l];
                }
            }
        }
    }

    //calc cov
    if (training || (!running)){
        int cov_size = dim0*_I*_I;
        for (int i = 0; i < cov_size; i++){
            cov_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < _I; j++){ //row selector
                for(int l = 0; l < _I; l++){ //col selector
                    for(int k = 0; k < num_groups*D; k++){
                        FLOP(1);
                        FLOP(1);
                        cov_temp[i*_I*_I + j*_I + l] += X_temp[i*_I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*_I + k*_I + l];
        }}}}
        //scale all of cov
        for(int i = 0; i < cov_size; i++){
            FLOP(1);
            cov_temp[i] /= (num_groups * D);
        }
        if (running) { //upadate running cov
            for(int i = 0; i < _I; i++){            
                for(int ii = 0; ii < _I; ii++){
                    for(int j = 0; j < dim0; j++){
                        FLOP(2);
                        FLOP(1);
                        running_cov_temp[i*_I*dim0 + ii*dim0 + j] += momentum *(cov_temp[j*_I*_I + i*_I + ii] - running_cov_temp[i*_I*dim0 + ii*dim0 + j]);
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < _I; i++){ //if not training use running cov
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < dim0; j++){
                    cov_temp[j*_I*_I + i*_I + ii] = running_cov_temp[i*_I*dim0 + ii*dim0 + j];
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
                    running_mean_orig[i*group_size + j] += running_mean_temp[i*group_size*B + l*group_size + j];
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
                        running_cov_orig[i*_I*group_size + ii*group_size + j] += running_cov_temp[i*_I*group_size*B + ii*group_size*B + l*group_size + j];
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
        max_temp[i] = cov_temp[i*_I*_I];
        for(int j = 1; j < _I*_I; j++){
            max_temp[i] = cov_temp[i*_I*_I + j] > max_temp[i] ? cov_temp[i*_I*_I + j] : max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < _I; j++){
            FLOP(1);
            cov_temp[i*_I*_I + j*_I + j] += max_temp[i]*eps;
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
        float *a = &cov_temp[i*_I*_I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++)
            x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
            x = a[i*n+j];  /* A_ij */

            for(k = 0; k < j; k++)
                x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */

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
                    float *mat = cov_temp + (i*_I*_I);
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
                        weight_temp[i*_I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
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
                weight_perm_temp[j*_I*_I + i] = weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = weight_perm_temp + i*_I*_I;
                    float *xv = x_norm + i*num_groups*D*_I + j*D*_I + l*_I;
                    float *b = bias_temp + i*_I;
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
