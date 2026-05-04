# Audio_ALG/pipelines/Makefile
#
# Linear AEC (HPF built-in) -> NR -> external RES pipeline.
# AEC v3.8.2 API; HPF is now part of AEC, so no separate libhpf dep.
#
# Usage:
#   make libs          # Build AEC + NR libraries first
#   make               # Build pipeline executable

AEC_DIR = ../lib/aec/c_impl
NR_DIR  = ../lib/nr/c_impl

CC ?= gcc
# AEC v3.8.2 mandates -ffp-contract=off for bit-exact behaviour; pipeline
# inherits the same flag for consistency.
CFLAGS = -Wall -Wextra -O2 -ffp-contract=off -std=c99 \
         -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
         -I$(NR_DIR)/include

LIBS = -L$(AEC_DIR)/bin -laec -L$(NR_DIR)/bin -lmmse_lsa -lm

.PHONY: all clean libs

all: aec_nr_pipeline

aec_nr_pipeline: aec_nr_pipeline.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f aec_nr_pipeline aec_nr_pipeline_static

libs:
	$(MAKE) -C $(AEC_DIR) lib
	$(MAKE) -C $(NR_DIR) lib
