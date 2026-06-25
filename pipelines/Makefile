# Audio_ALG/pipelines/Makefile
#
# Linear AEC -> NR -> RES Pipeline (with HPF pre-processing)
#
# Version A (malloc): uses submodule libs (lib/aec, lib/nr)
# Version B (static): uses SE/ repos on feature/static-memory branch
#
# Usage:
#   make libs          # Build libraries (submodule, for Version A)
#   make libs-static   # Build libraries (SE/ repos, for Version B)
#   make               # Build both versions

# Version A: submodule paths
AEC_DIR = ../lib/aec/c_impl
NR_DIR  = ../lib/nr/c_impl

# Version B: SE/ repo paths (feature/static-memory branch)
AEC_SE_DIR = ../../AEC/c_impl
NR_SE_DIR  = ../../NR/c_impl

CC ?= gcc

CFLAGS = -Wall -Wextra -O2 -std=c99 \
         -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
         -I$(NR_DIR)/include

CFLAGS_STATIC = -Wall -Wextra -O2 -std=c99 \
                -I$(AEC_SE_DIR)/include -I$(AEC_SE_DIR)/example \
                -I$(NR_SE_DIR)/include

LIBS = -L$(AEC_DIR)/bin -laec -L$(NR_DIR)/bin -lmmse_lsa -lm
LIBS_STATIC = -L$(AEC_SE_DIR)/bin -laec -L$(NR_SE_DIR)/bin -lmmse_lsa -lm

.PHONY: all clean libs libs-static

all: aec_nr_pipeline

aec_nr_pipeline: aec_nr_pipeline.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

# NOTE: the aec_nr_pipeline_static (static-memory pool) target has no source yet
# (aec_nr_pipeline_static.c is unwritten), so it was removed from `all` to make
# `make` work. Re-add the rule here + to `all` when the source is written.

clean:
	rm -f aec_nr_pipeline aec_nr_pipeline_static

libs:
	$(MAKE) -C $(AEC_DIR) lib
	$(MAKE) -C $(NR_DIR) lib

libs-static:
	$(MAKE) -C $(AEC_SE_DIR) lib
	$(MAKE) -C $(NR_SE_DIR) lib
