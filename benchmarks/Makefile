
INCLUDE = -I../csrc

CC       := gcc
CFLAGS   := -O3 -DPMU -fPIC -Wno-alloc-size-larger-than -DBUILD_MAIN  $(INCLUDE)
LDFLAGS  :=  -llapacke -lopenblas -lm

SRCS    = bench_forward.c ../csrc/clifford_linear.c ../csrc/clifford_groupnorm.c ../csrc/mv_act.c

all: bench_main

bench_main: $(SRCS)
	gcc $(CFLAGS) $(SRCS) -o bench_main $(LDFLAGS)

clean:
	rm -f bench_main
