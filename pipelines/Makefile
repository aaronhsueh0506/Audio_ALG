# Audio_ALG/pipelines/Makefile
#
# Linear AEC -> NR -> RES Pipeline
#
# Usage:
#   make libs          # Build AEC and NR libraries first
#   make               # Build pipeline executable

AEC_DIR = ../lib/aec/c_impl
NR_DIR  = ../lib/nr/c_impl

CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 \
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
