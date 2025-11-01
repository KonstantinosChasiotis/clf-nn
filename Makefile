# Makefile

# Compiler and flags
CC       := gcc
CFLAGS   := -O3 -g -fPIC -mavx2 -mfma -Wno-alloc-size-larger-than -Wno-aggressive-loop-optimizations -std=c99
LDFLAGS  := -shared

# Library suffix based on OS
SO_SUFFIX := .so
ifeq ($(OS),Windows_NT)
  SO_SUFFIX := .dll
endif

# Paths
CSRCDIR    := csrc
LIBDIR     := $(CSRCDIR)/build

# Targets
.PHONY: all clean
all: \
  $(LIBDIR)/libclifford_groupnorm$(SO_SUFFIX) \
  $(LIBDIR)/libclifford_linear$(SO_SUFFIX) \
  $(LIBDIR)/libmv_act$(SO_SUFFIX)

# Build rule for each .c → .so
$(LIBDIR)/lib%$(SO_SUFFIX): $(CSRCDIR)/%.c | $(LIBDIR)
	@echo "Building C shared library → $@"
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# Ensure output directory exists
$(LIBDIR):
	mkdir -p $(LIBDIR)

clean:
	@echo "Cleaning up…"
	rm -rf $(LIBDIR)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache

