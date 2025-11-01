#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear_bench.h"
#include "linear_setup.h"
#include "../flops.h"

int main(int argc, char** argv) {
    // Register all functions up front for --count and --name flags
    register_linear_functions();

    // --count: print number of registered functions
    if (argc == 2 && strcmp(argv[1], "--count") == 0) {
        printf("%d\n", get_registered_count());
        return 0;
    }

    // --name <idx>: print name of function at index
    if (argc == 3 && strcmp(argv[1], "--name") == 0) {
        int idx = atoi(argv[2]);
        int cnt = get_registered_count();
        if (idx < 0 || idx >= cnt) {
            fprintf(stderr, "Invalid function index (must be 0–%d)\n", cnt - 1);
            return 1;
        }
        printf("%s\n", get_registered_name(idx));
        return 0;
    }

    // Standard usage: <mode> <dim> <channels> <batch> [func_idx]
    if (argc < 5) {
        printf("Usage: %s <mode: 0|1> <dim> <channels> <batch> [func_idx]\n", argv[0]);
        return 1;
    }

    int mode = atoi(argv[1]);
    int dim  = atoi(argv[2]);
    int C    = atoi(argv[3]);
    int B    = atoi(argv[4]);
    int which_func = (argc >= 6) ? atoi(argv[5]) : -1;

    // Prepare geometry array
    int* g = malloc(dim * sizeof(int));
    if (!g) { 
        perror("malloc g"); 
        return 1; 
    }
    for (int i = 0; i < dim; i++) {
        g[i] = 1;
    }

    // Setup the linear layer
    setup_linear(B, C, dim, C, g);

    // Reference output buffer (used only in mode 0)
    float* out_ref = malloc(sizeof(float) * B * L_lin->out_channels * L_lin->sig.n_blades);
    if (!out_ref) { perror("malloc out_ref"); return 1; }

    if (mode == 0) {
        // Fixed-input benchmarking: compare all registered functions
        run_all_linear_benchmarks(L_lin, lin_x, out_ref, B);

    } else if (mode == 1) {
        // Performance logging: one line of CSV per invocation
        int cnt = get_registered_count();
        if (which_func < 0 || which_func >= cnt) {
            fprintf(stderr, "Must pass a valid function index (0–%d) for mode 1\n", cnt - 1);
            return 1;
        }

        size_t out_size = (size_t)B * L_lin->sig.n_blades * L_lin->out_channels;
        float* out_temp = malloc(out_size * sizeof(float));
        if (!out_temp) { perror("malloc out_temp"); return 1; }

        linear_func f = get_registered_function(which_func);
        FLOP_ENABLED = 1;
        total_flops  = 0;
        f(L_lin, lin_x, B, out_temp);
        FLOP_ENABLED = 0;
        uint64_t measured_flops = total_flops;
        uint64_t p_bytes_read = 0;

        double cycles = benchmark_linear_by_index(which_func, L_lin, lin_x, out_temp, B);

        uint64_t final_bytes_read = 0;
        final_bytes_read = total_bytes_read;

        printf("%d,%.2f,%llu,%llu,%llu\n",
               B,
               cycles,
               (unsigned long long)measured_flops,
               (unsigned long long)final_bytes_read,
               (unsigned long long)total_bytes_written);

        free(out_temp);

    } else {
        fprintf(stderr, "Invalid mode. Use 0 or 1\n");
        return 1;
    }

    // Cleanup
    clifford_linear_destroy(L_lin);
    free(lin_x);
    // Note: lin_out is not allocated here, so do not free it
    free(out_ref);
    free(g);

    return 0;
}
