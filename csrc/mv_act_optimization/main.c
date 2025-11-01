// main.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "mv_act_bench.h"
#include "mv_act_setup.h"
#include "../flops.h"

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --count\n"
        "  %s --name <idx>\n"
        "  %s 0 <C> <I_in> <I_full> <K> <batch>\n"
        "  %s 1 <C> <I_in> <I_full> <K> <batch> <func_idx>\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv) {

    register_activation_functions();

    if (argc == 2 && strcmp(argv[1], "--count") == 0) {
        printf("%d\n", get_registered_count());
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "--name") == 0) {
        int idx = atoi(argv[2]);
        int cnt = get_registered_count();
        if (idx < 0 || idx >= cnt) {
            fprintf(stderr, "Invalid function index (0–%d)\n", cnt - 1);
            return 1;
        }
        printf("%s\n", get_registered_name(idx));
        return 0;
    }

    if (argc < 7) {
        print_usage(argv[0]);
        return 1;
    }

    int mode   = atoi(argv[1]);
    int C      = atoi(argv[2]);
    int I_in   = atoi(argv[3]);
    int I_full = atoi(argv[4]);
    int K      = atoi(argv[5]);
    int B, which;

    if (mode == 0) {
        if (argc != 7) {
            print_usage(argv[0]);
            return 1;
        }
        B = atoi(argv[6]);
        which = -1;
    }
    else if (mode == 1) {
        if (argc != 8) {
            print_usage(argv[0]);
            return 1;
        }
        B     = atoi(argv[6]);
        which = atoi(argv[7]);
    }
    else {
        print_usage(argv[0]);
        return 1;
    }

    int cnt = get_registered_count();
    if (mode == 1 && (which < 0 || which >= cnt)) {
        fprintf(stderr, "Function index must be in [0–%d]\n", cnt - 1);
        return 1;
    }

    if (mode == 0) {
        setup_act(B, C, I_in, I_full, K);
        run_all_act_benchmarks(B);
        cleanup_act();
    }
    else {
        setup_act(B, C, I_in, I_full, K);

        total_flops = 0;
        total_bytes_read = 0;
        total_bytes_written = 0;
        activation_func f = get_registered_function(which);
        FLOP_ENABLED = 1;
        f(
          act_x,
          act_B, act_C,
          act_I_in, act_I_full,
          act_input_blades,
          act_weight, act_K,
          act_kernel_blades,
          act_bias,
          act_out
        );
        FLOP_ENABLED = 0;
        uint64_t fl = total_flops;
        uint64_t bytes_read = total_bytes_read;
        uint64_t bytes_written = total_bytes_written;

        double cycles = benchmark_act_by_index(which, B);

        printf("%d,%.2f,%llu,%llu,%llu\n",
               B,
               cycles,
               (unsigned long long)fl,
               (unsigned long long)bytes_read,
               (unsigned long long)bytes_written);

        cleanup_act();
    }

    return 0;
}
