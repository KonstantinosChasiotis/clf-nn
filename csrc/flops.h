#ifndef FLOPS_H
#define FLOPS_H

#include <stdint.h>

// counts all FLOP(n) when FLOP_ENABLED is non-zero
extern uint64_t total_flops;
extern uint64_t total_bytes_read;
extern uint64_t total_bytes_written;
extern int      FLOP_ENABLED;

#ifdef TRACEFLOPS
#include <stdio.h>

#ifndef FLOPPATH
#define FLOPPATH "flopslog.csv"
#endif

// only add to total_flops if enabled
#define FLOP(n)                                          \
  do {                                                   \
    if (FLOP_ENABLED)                                    \
      total_flops += (uint64_t)(n);                      \
  } while (0)

#define BYTES_READ(n)                                    \
  do {                                                   \
    if (FLOP_ENABLED)                                    \
      total_bytes_read += (uint64_t)(n);                 \
  } while (0)

#define BYTES_WRITTEN(n)                                 \
  do {                                                   \
    if (FLOP_ENABLED)                                    \
      total_bytes_written += (uint64_t)(n);             \
  } while (0)

static inline void flops_readout(FILE *fp, int B) {
    if (fp != NULL) {
        fprintf(fp, "%d,%llu,%llu,%llu\n", B, 
                (unsigned long long)total_flops,
                (unsigned long long)total_bytes_read,
                (unsigned long long)total_bytes_written);
    } else {
        fprintf(stderr, "Error writing performance data to file.\n");
    }
}

#else

// when TRACEFLOPS is off, do nothing
#define FLOP(n)            ((void)0)
#define BYTES_READ(n)      ((void)0)
#define BYTES_WRITTEN(n)   ((void)0)
#define flops_readout(fp,B) ((void)0)

#endif // TRACEFLOPS

#endif // FLOPS_H
