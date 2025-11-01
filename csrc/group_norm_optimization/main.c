// main.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "gnn_bench.h"
#include "gnn_setup.h"
#include "../flops.h"

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --count\n"
        "  %s --name <idx>\n"
        "  %s 0 <C> <D> <I> <groups> <batch>\n"
        "  %s 1 <C> <D> <I> <groups> <batch> <func_idx>\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    register_groupnorm_functions();

    if (argc == 2 && strcmp(argv[1], "--count")==0) {
        printf("%d\n", get_registered_count());
        return 0;
    }
    if (argc == 3 && strcmp(argv[1], "--name")==0) {
        int idx = atoi(argv[2]), cnt = get_registered_count();
        if (idx<0||idx>=cnt) {
            fprintf(stderr,"Invalid function index (0–%d)\n",cnt-1);
            return 1;
        }
        printf("%s\n", get_registered_name(idx));
        return 0;
    }

    if (argc<7) {
        print_usage(argv[0]);
        return 1;
    }

    int mode   = atoi(argv[1]);
    int C      = atoi(argv[2]);
    int D      = atoi(argv[3]);
    int I      = atoi(argv[4]);
    int groups = atoi(argv[5]);
    int B, which;

    if (mode==0) {
        if (argc!=7) { print_usage(argv[0]); return 1; }
        B     = atoi(argv[6]);
        which = -1;
    } else if (mode==1) {
        if (argc!=8) { print_usage(argv[0]); return 1; }
        B     = atoi(argv[6]);
        which = atoi(argv[7]);
    } else {
        print_usage(argv[0]);
        return 1;
    }

    int cnt = get_registered_count();
    if (mode==1 && (which<0||which>=cnt)) {
        fprintf(stderr,"Function index must be in [0–%d]\n",cnt-1);
        return 1;
    }

    const bool  running  = false;
    const bool  scaling  = true;
    const bool  training = true;
    const float momentum = 0.1f;
    const float eps      = 1e-5f;

    if (mode==0) {
        setup_groupnorm(B,C,D,I,groups,running,scaling,training,momentum,eps);
        run_all_groupnorm_benchmarks(B);
        cleanup_groupnorm();
    } else {
        setup_groupnorm(B,C,D,I,groups,running,scaling,training,momentum,eps);

        total_flops = 0;
        total_bytes_read = 0;
        total_bytes_written = 0;
        groupnorm_func f = get_registered_function(which);
        FLOP_ENABLED = 1;
        f( gn_x,
           gn_B,gn_C,gn_D,gn_I,gn_num_groups,
           gn_running,    gn_running_mean,
           gn_running_cov,
           gn_scaling,
           gn_weight,     gn_bias,
           gn_training,   gn_momentum,gn_eps,
           gn_x_norm );
        FLOP_ENABLED = 0;
        uint64_t fl = total_flops;
        uint64_t bytes_read = total_bytes_read;
        uint64_t bytes_written = total_bytes_written;

        double cycles = benchmark_groupnorm_by_index(which, B);
        printf("%d,%.2f,%llu,%llu,%llu\n", B, cycles, 
               (unsigned long long)fl,
               (unsigned long long)bytes_read,
               (unsigned long long)bytes_written);

        cleanup_groupnorm();
    }

    return 0;
}
