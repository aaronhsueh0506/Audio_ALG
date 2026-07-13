# Audio_ALG/pipelines/Makefile
#
# AEC(linear) -> echo-aware NR -> RES pipeline — two versions of the SAME
# algorithm (see aec_nr_pipeline.c's header for the DSP chain):
#
#   aec_nr_pipeline         (Version A: malloc)         — heap-allocated modules
#   aec_nr_pipeline_static  (Version B: static memory)  — one caller pool, no
#                                                          malloc after init
#
# Both link against the submodule libs (lib/aec, lib/nr) + the shared
# audio_common FFT/fast_math layer (../../audio_common relative to this dir).
#
# Usage:
#   make                       # builds libs (BACKEND=kiss) + both binaries
#   make BACKEND=ne10          # NE10 backend (rebuild libs first: make clean-libs)
#   make EXTRA_CFLAGS="-DAEC_FAST_MATH"   # see the EXTRA_CFLAGS block below
#   make clean                 # remove the two pipeline binaries
#   make clean-libs             # also clean lib/aec, lib/nr, audio_common build dirs

AEC_DIR = ../lib/aec/c_impl
NR_DIR  = ../lib/nr/c_impl

# audio_common sits two levels up from pipelines/ (SE/Audio_ALG/pipelines -> SE/audio_common).
# lib/aec/c_impl and lib/nr/c_impl each resolve it themselves from their own
# location (../../../../audio_common from c_impl/, verified to exist as a
# submodule checkout); AC_DIR here is for OUR OWN CFLAGS/LDFLAGS and is passed
# through explicitly to the sub-makes so all three always agree on the path.
AC_DIR ?= ../../audio_common
BACKEND ?= kiss

CC  ?= gcc
CXX ?= c++

# EXTRA_CFLAGS — pass-through to the AEC/NR lib sub-builds AND this pipeline's
# own compile (both submodule Makefiles already do `CFLAGS += $(EXTRA_CFLAGS)`,
# so one EXTRA_CFLAGS on this top-level `make` reaches every .o for the build):
#
#   make EXTRA_CFLAGS="-DAEC_FAST_MATH"
#     Opt-in fast matched-filter arithmetic in lib/aec's delay estimator
#     (float32 NEON-style accumulation instead of the bit-exact reference
#     dot products). Sampled-quality-verified: ~zero cost to output quality
#     on the cases spot-checked so far. Diverges from the bit-exact Python
#     reference — default build (no EXTRA_CFLAGS) stays byte-identical.
#     See lib/aec/c_impl/src/delay_aec3.c.
#
#   make EXTRA_CFLAGS="-DAEC_DELAY_DS_FACTOR=8"
#     Selects AEC3's own ds8 decimation for the delay estimator (matched-
#     filter taps 512->256, sub-block rate 16->8), roughly halving that
#     stage's cost. FS-REGRESSION RISK — coarsens delay resolution from 4 to
#     8 samples; needs a full 800-case AECMOS gate before shipping. Default
#     (4) stays byte-identical. See lib/aec/c_impl/include/delay_aec3.h.
#
#   Flags combine: make EXTRA_CFLAGS="-DAEC_FAST_MATH -DAEC_DELAY_DS_FACTOR=8"
#
# NOTE: switching EXTRA_CFLAGS on an already-built tree needs `make clean-libs`
# first — object files aren't flag-tagged, so a stale non-flagged .o (or a
# stale flagged one, if you're turning a flag back off) will silently persist
# otherwise.
EXTRA_CFLAGS ?=

CFLAGS = -Wall -Wextra -O2 -std=gnu99 \
         -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
         -I$(NR_DIR)/include \
         -I$(AC_DIR)/include
CFLAGS += $(EXTRA_CFLAGS)

# The NE10 archive carries one C++ TU (NE10's generic-radix FFT kernel); a
# plain-C pipeline binary linking against it needs the C++ runtime driver.
LINK = $(CC)
LDFLAGS_EXTRA =
ifeq ($(BACKEND),ne10)
  LINK = $(CXX)
  LDFLAGS_EXTRA = -lc++
endif

# Link order matters: pipeline.o, libaec.a, libmmse_lsa.a, libaudio_common.a, -lm
LIBS = $(AEC_DIR)/bin/libaec.a \
       $(NR_DIR)/bin/libmmse_lsa.a \
       $(AC_DIR)/bin/$(BACKEND)/libaudio_common.a
LDFLAGS = $(LIBS) -lm $(LDFLAGS_EXTRA)

.PHONY: all clean clean-libs libs

all: libs aec_nr_pipeline aec_nr_pipeline_static

# Build the shared archive + both submodule libs for the selected BACKEND.
# AC_DIR is passed through explicitly so lib/aec and lib/nr resolve the SAME
# audio_common checkout this Makefile uses (their own AC_DIR fallback also
# resolves correctly at the submodule depth: ../../../../audio_common from
# c_impl/, but pass it anyway to be explicit/robust).
libs:
	$(MAKE) -C $(AC_DIR) BACKEND=$(BACKEND) lib
	$(MAKE) -C $(AEC_DIR) lib BACKEND=$(BACKEND) AC_DIR=$(abspath $(AC_DIR)) EXTRA_CFLAGS="$(EXTRA_CFLAGS)"
	$(MAKE) -C $(NR_DIR) lib BACKEND=$(BACKEND) AC_DIR=$(abspath $(AC_DIR)) EXTRA_CFLAGS="$(EXTRA_CFLAGS)"

# Compile with $(CC) (always a plain-C driver, -std=gnu99 valid) then link with
# $(LINK) (c++ for BACKEND=ne10 — the NE10 archive's one C++ TU needs it). A
# C++ driver invoked directly on a .c source rejects -std=gnu99 and treats the
# file as C++, so the compile and link steps must stay separate here (mirrors
# how lib/aec and lib/nr's own Makefiles split .o-compile from $(LINK)).
aec_nr_pipeline.o: aec_nr_pipeline.c
	$(CC) $(CFLAGS) -c -o $@ $<

aec_nr_pipeline_static.o: aec_nr_pipeline_static.c
	$(CC) $(CFLAGS) -c -o $@ $<

aec_nr_pipeline: aec_nr_pipeline.o | libs
	$(LINK) -o $@ $< $(LDFLAGS)

aec_nr_pipeline_static: aec_nr_pipeline_static.o | libs
	$(LINK) -o $@ $< $(LDFLAGS)

clean:
	rm -f aec_nr_pipeline aec_nr_pipeline_static aec_nr_pipeline.o aec_nr_pipeline_static.o

clean-libs: clean
	$(MAKE) -C $(AEC_DIR) clean
	$(MAKE) -C $(NR_DIR) clean
	$(MAKE) -C $(AC_DIR) clean
