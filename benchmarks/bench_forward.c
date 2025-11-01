#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#ifndef WIN32
#include <sys/time.h>
#endif

#ifdef __x86_64__
#include "tsc_x86.h"
#endif

#ifdef __aarch64__
#include "vct_arm.h"

#ifdef PMU
#include "kperf.h"
#endif
#endif

#include "clifford_linear.h"
#include "clifford_groupnorm.h"
#include "mv_act.h"

#define NUM_RUNS 1
#define CYCLES_REQUIRED 1e8
#define FREQUENCY 3.22e9
#define CALIBRATE

void (*compute)(double[], double[], double[], int);

// Linear Layer Variables
static CliffordLinear *L_lin;
static float *lin_x, *lin_out;
static int lin_B, lin_C, lin_D, lin_O;
static int *lin_g;

// Group Norm Layer Variables
static float *gn_x, *gn_out;
static int gn_B, gn_C, gn_Dflat, gn_I, gn_G;
static bool gn_has_running, gn_has_affine;
static float *gn_running_mean, *gn_running_cov, *gn_weight, *gn_bias;

//act vars
static float *act_x, *act_out;
static int act_B, act_C, act_L, act_I;
static float * act_weight;
static int act_K;
static int * act_kernel_blades;

void setup_act(int B, int C, int L, int _I, int K)
{
    act_B = B;
    act_C = C;
    act_L = L;
    act_I = _I;
    act_K = K;

    size_t N = (size_t)B * C * L * _I;
    act_x = malloc(N * sizeof(float));
    act_out = malloc(N * sizeof(float));

    for (size_t i = 0; i < N; i++)
    {
        act_x[i] = rand() / (float)RAND_MAX;
    }

    act_weight = malloc(C * K * sizeof(float));
    for (size_t i = 0; i < C * K; i++)
    {
        act_weight[i] = rand() / (float)RAND_MAX;
    }
    act_kernel_blades = malloc(K * sizeof(int));
    for (size_t i = 0; i < K; i++)
    {
        act_kernel_blades[i] = 1; //SH: simplificaiton but has to do for now
    }
}

void compute_act(double A[], double B[], double C[], int n)
{
    mv_act_forward(
        act_x,
        act_B, act_C, act_L, act_I,
        act_weight,
        act_K,
        act_kernel_blades,
        act_out
    );
}

void cleanup_act()
{
    free(act_x);
    free(act_out);
    free(act_weight);
    free(act_kernel_blades);
}


void setup_linear(int B, int C, int dim, int O, int *g_sig)
{
    lin_B = B;
    lin_C = C;
    lin_O = O;
    lin_g = g_sig;

    int _I = 1 << dim; //this needs to be _I since the libs have I defined as a makro
    lin_D = _I;

    lin_x = malloc(sizeof(float) * B * C * _I);
    lin_out = malloc(sizeof(float) * B * O * _I);

    for (int i = 0, N = B * C * _I; i < N; i++)
        lin_x[i] = rand() / (float)RAND_MAX;

    L_lin = clifford_linear_create(g_sig, dim, C, O, true);
    if (!L_lin)
    {
        fprintf(stderr, "Error: clifford_linear_create failed (dim=%d)\n", dim);
        exit(1);
    }

    size_t nW = (size_t)_I * O * C;
    for (size_t i = 0; i < nW; i++)
        L_lin->weight[i] = rand() / (float)RAND_MAX;

    size_t nB = (size_t)_I * O;
    for (size_t i = 0; i < nB; i++)
        L_lin->bias[i] = rand() / (float)RAND_MAX;
}

void compute_linear(double A[], double B[], double C[], int n)
{
    clifford_linear_forward(L_lin, (const float *)lin_x, lin_B, lin_out);
}

void cleanup_linear()
{
    free(lin_x);
    free(lin_out);
    clifford_linear_destroy(L_lin);
}

void setup_groupnorm(int B, int C, int Dflat, int _I, int G)
{
    gn_B = B;
    gn_C = C;
    gn_Dflat = Dflat;
    gn_I = _I;
    gn_G = G;
    gn_has_running = false;
    gn_has_affine = false;

    size_t N = (size_t)B * C * Dflat * _I;
    gn_x = malloc(N * sizeof(float));
    gn_out = malloc(N * sizeof(float));

    for (size_t i = 0; i < N; i++)
    {
        gn_x[i] = rand() / (float)RAND_MAX;
    }
}

void compute_groupnorm(double A[], double B[], double C[], int n)
{
    clifford_groupnorm_forward(
        gn_x,
        gn_B,
        gn_C,
        gn_Dflat,
        gn_I,
        gn_G,
        gn_has_running,
        gn_running_mean,
        gn_running_cov,
        gn_has_affine,
        gn_weight,
        gn_bias,
        true,
        0.1f,
        1e-5f,
        gn_out);
}

void cleanup_groupnorm()
{
    free(gn_x);
    free(gn_out);
}

/*
 * Timing function based on the TimeStep Counter of the CPU.
 */
#ifdef __x86_64__
double rdtsc(double A[], double B[], double C[], int n)
{
    int i, num_runs;
    myInt64 cycles;
    myInt64 start;
    num_runs = NUM_RUNS;

    /*
     * The CPUID instruction serializes the pipeline.
     * Using it, we can create execution barriers around the code we want to time.
     * The calibrate section is used to make the computation large enough so as to
     * avoid measurements bias due to the timing overhead.
     */
#ifdef CALIBRATE
    while (num_runs < (1 << 14))
    {
        start = start_tsc();
        for (i = 0; i < num_runs; ++i)
        {
            compute(A, B, C, n);
        }
        cycles = stop_tsc(start);

        if (cycles >= CYCLES_REQUIRED)
            break;

        num_runs *= 2;
    }
#endif

    start = start_tsc();
    for (i = 0; i < num_runs; ++i)
    {
        compute(A, B, C, n);
    }

    cycles = stop_tsc(start) / num_runs;
    return (double)cycles;
}
#endif

#ifdef __aarch64__
double rdvct(double A[], double B[], double C[], int n)
{
    int i, num_runs;
    TIMESTAMP cycles;
    TIMESTAMP start;
    num_runs = NUM_RUNS;

    /*
     * The CPUID instruction serializes the pipeline.
     * Using it, we can create execution barriers around the code we want to time.
     * The calibrate section is used to make the computation large enough so as to
     * avoid measurements bias due to the timing overhead.
     */
#ifdef CALIBRATE
    while (num_runs < (1 << 14))
    {
        start = start_vct();
        for (i = 0; i < num_runs; ++i)
        {
            compute(A, B, C, n);
        }
        cycles = stop_vct(start);

        if (cycles >= CYCLES_REQUIRED)
            break;

        num_runs *= 2;
    }
#endif

    start = start_vct();
    for (i = 0; i < num_runs; ++i)
    {
        compute(A, B, C, n);
    }

    cycles = stop_vct(start) / num_runs;
    return (double)cycles;
}

#ifdef PMU
double rdpmu(double A[], double B[], double C[], int n)
{
    kperf_init();
    int i, num_runs;
    struct performance_counters startperf, endperf;
    num_runs = NUM_RUNS;

    /*
     * The CPUID instruction serializes the pipeline.
     * Using it, we can create execution barriers around the code we want to time.
     * The calibrate section is used to make the computation large enough so as to
     * avoid measurements bias due to the timing overhead.
     */
#ifdef CALIBRATE
    while (num_runs < (1 << 14))
    {
        startperf = kperf_get_counters();
        for (i = 0; i < num_runs; ++i)
        {
            compute(A, B, C, n);
        }
        endperf = kperf_get_counters();
        double cycles = endperf.cycles - startperf.cycles;
        if (cycles >= CYCLES_REQUIRED)
            break;

        num_runs *= 2;
    }
#endif

    startperf = kperf_get_counters();
    for (i = 0; i < num_runs; ++i)
    {
        compute(A, B, C, n);
    }

    endperf = kperf_get_counters();
    double cycles = (endperf.cycles - startperf.cycles) / num_runs;
    return (double)cycles;
}
#endif

#endif

double c_clock(double A[], double B[], double C[], int n)
{
    int i, num_runs;
    double cycles;
    clock_t start, end;

    num_runs = NUM_RUNS;
#ifdef CALIBRATE
    while (num_runs < (1 << 14))
    {
        start = clock();
        for (i = 0; i < num_runs; ++i)
        {
            compute(A, B, C, n);
        }
        end = clock();

        cycles = (double)(end - start);

        // Same as in c_clock: CYCLES_REQUIRED should be expressed accordingly to the order of magnitude of CLOCKS_PER_SEC
        if (cycles >= CYCLES_REQUIRED / (FREQUENCY / CLOCKS_PER_SEC))
            break;

        num_runs *= 2;
    }
#endif

    start = clock();
    for (i = 0; i < num_runs; ++i)
    {
        compute(A, B, C, n);
    }
    end = clock();

    return (double)(end - start) / num_runs;
}

#ifndef WIN32
double timeofday(double A[], double B[], double C[], int n)
{
    int i, num_runs;
    double cycles;
    struct timeval start, end;

    num_runs = NUM_RUNS;
#ifdef CALIBRATE
    while (num_runs < (1 << 14))
    {
        gettimeofday(&start, NULL);
        for (i = 0; i < num_runs; ++i)
        {
            compute(A, B, C, n);
        }
        gettimeofday(&end, NULL);

        cycles = (double)((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6) * FREQUENCY;

        if (cycles >= CYCLES_REQUIRED)
            break;

        num_runs *= 2;
    }
#endif

    gettimeofday(&start, NULL);
    for (i = 0; i < num_runs; ++i)
    {
        compute(A, B, C, n);
    }
    gettimeofday(&end, NULL);

    return (double)((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6) / num_runs;
}

#else

double gettickcount(double A[], double B[], double C[], int n)
{
    int i, num_runs;
    double cycles, start, end;

    num_runs = NUM_RUNS;
#ifdef CALIBRATE
    while (num_runs < (1 << 14))
    {
        start = (double)GetTickCount();
        for (i = 0; i < num_runs; ++i)
        {
            compute(A, B, C, n);
        }
        end = (double)GetTickCount();

        cycles = (end - start) * FREQUENCY / 1e3; // end-start provides a measurement in the order of milliseconds

        if (cycles >= CYCLES_REQUIRED)
            break;

        num_runs *= 2;
    }
#endif

    start = (double)GetTickCount();
    for (i = 0; i < num_runs; ++i)
    {
        compute(A, B, C, n);
    }
    end = (double)GetTickCount();

    return (end - start) / num_runs;
}

double queryperfcounter(double A[], double B[], double C[], int n, LARGE_INTEGER f)
{
    int i, num_runs;
    double cycles;
    LARGE_INTEGER start, end;

    num_runs = NUM_RUNS;
#ifdef CALIBRATE
    while (num_runs < (1 << 14))
    {
        QueryPerformanceCounter(&start);
        for (i = 0; i < num_runs; ++i)
        {
            compute(A, B, C, n);
        }
        QueryPerformanceCounter(&end);

        cycles = (double)(end.QuadPart - start.QuadPart);

        // Same as in c_clock: CYCLES_REQUIRED should be expressed accordingly to the order of magnitude of f
        if (cycles >= CYCLES_REQUIRED / (FREQUENCY / f.QuadPart))
            break;

        num_runs *= 2;
    }
#endif

    QueryPerformanceCounter(&start);
    for (i = 0; i < num_runs; ++i)
    {
        compute(A, B, C, n);
    }
    QueryPerformanceCounter(&end);

    return (double)(end.QuadPart - start.QuadPart) / num_runs;
}

#endif

#ifdef BUILD_MAIN
int main(int argc, char **argv)
{
    int B, C, O, dim, _I, Dflat, G, L, K;

    if (argc < 2)
    {
        fprintf(stderr, "Usage:\n"
                        "  %s linear B C dim O sig_len\n"
                        "  %s groupnorm B C Dflat I G\n"
                        "  %s act B C L I K\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    compute = NULL;

    if (strcmp(mode, "linear") == 0)
    {
        if (argc != 6)
        {
            fprintf(stderr, "linear needs 5 args\n");
            return 1;
        }

        B = atoi(argv[2]);
        C = atoi(argv[3]);
        dim = atoi(argv[4]);
        O = atoi(argv[5]);

        int g_sig[dim];
        for (int i = 0; i < dim; i++)
            g_sig[i] = 1;

        setup_linear(B, C, dim, O, g_sig);
        compute = compute_linear;
    }
    else if (strcmp(mode, "groupnorm") == 0)
    {
        if (argc != 7)
        {
            fprintf(stderr, "groupnorm needs 6 args\n");
            return 1;
        }

        B = atoi(argv[2]);
        C = atoi(argv[3]);
        Dflat = atoi(argv[4]);
        _I = atoi(argv[5]);
        G = atoi(argv[6]);

        setup_groupnorm(B, C, Dflat, _I, G);
        compute = compute_groupnorm;
    } else if (strcmp(mode, "act") == 0)
    {
        if (argc != 7)
        {
            fprintf(stderr, "act needs 6 args\n");
            return 1;
        }

        B = atoi(argv[2]);
        C = atoi(argv[3]);
        L = atoi(argv[4]);
        _I = atoi(argv[5]);
        K = atoi(argv[6]);

        setup_act(B, C, L, _I, K);
        compute = compute_act;
    }
    else
    {
        fprintf(stderr, "Unknown mode '%s'\n", mode);
        return 1;
    }

#ifdef __x86_64__
    {
        double r = rdtsc(NULL, NULL, NULL, 0);
        printf("RDTSC instruction:\n %lf cycles measured => %lf seconds, assuming frequency is %lf MHz.\n\n",
               r, r / FREQUENCY, FREQUENCY / 1e6);
    }
#endif

#ifdef __aarch64__
    {
        double v = rdvct(NULL, NULL, NULL, 0);
        printf("VCT instruction:\n %lf cycles measured => %lf seconds, assuming VCT freq is %lf MHz.\n\n",
               v, v / (get_vct_freq()), get_vct_freq() / 1e6);
    }
#ifdef PMU
    {
        double p = rdpmu(NULL, NULL, NULL, 0);
        printf("PMU instruction:\n %lf cycles measured => %lf seconds, assuming frequency is %lf MHz.\n\n",
               p, p / FREQUENCY, FREQUENCY / 1e6);
    }
#endif
#endif

    {
        double c = c_clock(NULL, NULL, NULL, 0);
        printf("C clock() function:\n %lf cycles measured => %lf seconds (CLOCKS_PER_SEC = %lf MHz).\n\n",
               c, c / CLOCKS_PER_SEC, CLOCKS_PER_SEC / 1e6);
    }

#ifndef WIN32
    {
        double t = timeofday(NULL, NULL, NULL, 0);
        printf("C gettimeofday():\n %lf seconds measured\n\n", t);
    }
#else
    {
        double t = gettickcount(NULL, NULL, NULL, 0);
        printf("Windows GetTickCount():\n %lf milliseconds measured\n\n", t);
    }
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        double q = queryperfcounter(NULL, NULL, NULL, 0, freq);
        printf("Windows QueryPerformanceCounter():\n %lf cycles measured => %lf seconds @ %lf MHz\n\n",
               q, q / freq.QuadPart, freq.QuadPart / 1000.0);
    }
#endif

    if (compute == compute_linear) {       
        cleanup_linear();
    } else if (compute == compute_groupnorm){ 
        cleanup_groupnorm();
    } else if (compute == compute_act){ 
        cleanup_act();
    }
    printf("Done.\n");
    return 0;
}
#endif