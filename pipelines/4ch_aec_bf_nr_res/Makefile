# Four-channel traditional AEC -> spatial -> mono NR/RES pipeline.
#
# This Makefile lives beside the 4-channel implementation; the parent pipelines/Makefile is the mono
# AEC -> NR producer/runner. The archive built here holds only the reusable four-lane linear core.
# The standard and ULCNet wrappers are linked directly into application binaries; AEC, NR, common,
# DOA, BF and spatial kernels remain component libs.
#
# Common invocations:
#   make SIMD=1
#   make SIMD=0 test
#   make audio_pipeline_4ch_raw

CC ?= cc
CXX ?= c++
AR ?= ar
RANLIB ?= ranlib

PIPELINES_DIR := ..
ALG_DIR := ../..
AEC_DIR ?= $(ALG_DIR)/lib/aec/c_impl
NR_DIR ?= $(ALG_DIR)/lib/nr/c_impl
AUDIOLIB_DIR ?= ../../../audiolib
COMMON_DIR ?= ../../../audiolib/common
ULCNET_DIR ?= $(ALG_DIR)/AIAEC/Align_ULCNet
DFN2_DIR ?= $(ALG_DIR)/AINR/DeepFilterNet2

# Align-ULCNet deployment grid, same knob and same values as pipelines/
# Makefile. One binary serves one grid (the C structs embed the STFT/WOLA
# tables and scratch), so it is a build parameter. Kept out of EXTRA_CFLAGS
# on purpose: that is forwarded verbatim to the AEC/NR/common/spatial
# producer sub-makes, none of which contain ULCNet code, and would split
# their object caches for nothing. Applied only to the ULCNet TUs below and
# folded into CFG_SIG_PAYLOAD so the two grids never share an object dir.
ULCNET_GRID ?= 16k
ifeq ($(ULCNET_GRID),16k)
ULCNET_GRID_CFLAGS = -DULCNET_MODEL_IO_SR=16000 -DULCNET_MODEL_IO_N_FFT=512
else ifeq ($(ULCNET_GRID),48k)
ULCNET_GRID_CFLAGS = -DULCNET_MODEL_IO_SR=48000 -DULCNET_MODEL_IO_N_FFT=1024
else
$(error ULCNET_GRID must be 16k or 48k, got '$(ULCNET_GRID)')
endif

BACKEND = ne10
SIMD ?= 1
WERROR = 0
NO_STDIO = 0
DEBUG = 0
TOOLCHAIN_CHECK = 1
EXTRA_CFLAGS =
# PROFILE=1 turns on per-hop stage timing across the whole chain: this wrapper's own stages and,
# through the PROFILE knob forwarded to the lib/aec build, the four summed off the lanes. Off
# otherwise, so a release build takes no clock reads; FourAecNrResLastTiming's doc comment
# (4aec_nr_res.h) carries the field list.

# PROFILE_CFLAGS goes into CFLAGS rather than EXTRA_CFLAGS, as SIMD does: EXTRA_CFLAGS is forwarded
# to every producer sub-make and folded into CFG_SIG, so a -D smuggled through it reached lib/aec
# twice, splitting identical objects across two keys and re-keying producers that never use the
# macro. CFLAGS here is `override CFLAGS :=`, so a command-line assignment cannot drop the knob.
# Both -D land in this Makefile's CFLAGS only; lib/aec gets AEC_STAGE_TIMING from the PROFILE knob
# forwarded by name in FWD.

# AEC_STAGE_TIMING is repeated on purpose: this layer's own TUs test it, to decide whether to copy
# the lanes' record out and what the tests may assert about it; deriving both flags from PROFILE
# keeps the two views consistent. A lib/aec pin without the PROFILE knob degrades to the one-half
# state: the wrapper's stages measure and the AEC block reads zero.
PROFILE ?= 0
ifeq ($(PROFILE),1)
PROFILE_CFLAGS := -DFOUR_AEC_NR_RES_STAGE_TIMING=1 -DAEC_STAGE_TIMING=1
else
PROFILE_CFLAGS :=
endif
EXTRA_LDFLAGS =
CPPFLAGS ?=
CXXFLAGS ?=
LDFLAGS ?=

define require_bool
ifneq ($$(words $$($(1))),1)
$$(error $(1) must be exactly 0 or 1, got '$$($(1))')
endif
ifneq ($$(filter-out 0 1,$$($(1))),)
$$(error $(1) must be exactly 0 or 1, got '$$($(1))')
endif
endef

ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): ne10 is the only FFT backend)
endif

# Announce the resolved backend on every build (same MAKECMDGOALS guard as
# common's own banner: print-bin-dir/print-obj-dir/print-lib-path are
# consumed via `$$(make -s ... print-lib-path)` command substitution
# elsewhere, and this must not add stray stdout to that captured value).
ifeq ($(filter print-%,$(MAKECMDGOALS)),)
$(info Audio_ALG/4ch_aec_bf_nr_res: FFT backend = $(BACKEND))
endif

$(eval $(call require_bool,SIMD))
$(eval $(call require_bool,PROFILE))
$(eval $(call require_bool,WERROR))
$(eval $(call require_bool,NO_STDIO))
$(eval $(call require_bool,DEBUG))
$(eval $(call require_bool,TOOLCHAIN_CHECK))

# These values become shell command arguments in the two-phase producer
# resolver.  Match the sibling Makefiles' supported surface and reject the
# unsafe/ambiguous inputs that would also undermine the FP policy.
ifneq ($(findstring ;,$(EXTRA_CFLAGS) $(EXTRA_LDFLAGS)),)
$(error EXTRA_CFLAGS/EXTRA_LDFLAGS must not contain ';')
endif
ifneq ($(filter -Ofast -ffast-math -funsafe-math-optimizations -ffp-contract=%,$(CFLAGS) $(CPPFLAGS) $(EXTRA_CFLAGS)),)
$(error fast-math and caller-supplied -ffp-contract flags are unsupported)
endif

ifeq ($(DEBUG),1)
OPT_CFLAGS := -O0 -g
else
OPT_CFLAGS := -O2
endif

ifeq ($(SIMD),0)
SIMD_CFLAGS := -DSIMD_KERNELS_FORCE_SCALAR
else
SIMD_CFLAGS :=
endif

ifeq ($(WERROR),1)
WERROR_CFLAGS := -Werror
else
WERROR_CFLAGS :=
endif

override CFLAGS := -Wall -Wextra $(OPT_CFLAGS) -std=gnu99 \
          -I. -I$(PIPELINES_DIR) -I$(PIPELINES_DIR)/4ch_alignulcnet \
          -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
          -I$(NR_DIR)/include -I$(COMMON_DIR)/include -I$(COMMON_DIR) \
          -I$(AUDIOLIB_DIR)/common/utility \
          -I$(AUDIOLIB_DIR)/doa \
          -I$(AUDIOLIB_DIR)/bf \
          -I$(DFN2_DIR) \
          $(CPPFLAGS) $(EXTRA_CFLAGS) $(SIMD_CFLAGS) $(PROFILE_CFLAGS) \
          -DAUDIO_PIPELINE_BACKEND_STR=\"$(BACKEND)\" \
          $(WERROR_CFLAGS) -ffp-contract=off -fno-math-errno
LIB_CFLAGS := $(CFLAGS)

LINK := $(CXX)
LDLIBS := $(EXTRA_LDFLAGS) $(LDFLAGS) -lm

FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) PROFILE=$(PROFILE) WERROR=$(WERROR) \
      CC='$(CC)' CXX='$(CXX)' AR='$(AR)' RANLIB='$(RANLIB)' \
      EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) \
      TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) DEBUG=$(DEBUG)

# Resolve every producer before constructing this consumer's CFG identity.
ifeq ($(COMMON_LIB),)
.PHONY: all test libs lib libaudio_pipeline_4ch.a lib4aec_dfn_res.a \
        audio_pipeline_4ch_raw 4aec_nr_res_static audio_pipeline_4ch_static 4ch_alignulcnet \
        4ch_aec_bf_dfn_res \
        test_4aec_nr_res test_audio_pipeline_4ch test_4aec_dfn_res \
        test_audio_pipeline_4ch_ulcnet dump_delay_parity \
        print-bin-dir print-obj-dir print-lib-path audit-no-stdio \
        test-third-party
all test libs lib libaudio_pipeline_4ch.a lib4aec_dfn_res.a \
audio_pipeline_4ch_raw 4aec_nr_res_static audio_pipeline_4ch_static 4ch_alignulcnet \
4ch_aec_bf_dfn_res \
test_4aec_nr_res test_audio_pipeline_4ch test_4aec_dfn_res \
test_audio_pipeline_4ch_ulcnet dump_delay_parity \
print-bin-dir print-obj-dir print-lib-path audit-no-stdio \
test-third-party:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: cannot resolve common archive" >&2; exit 1; }; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 test -n "$$aec" || { echo "FATAL: cannot resolve AEC archive" >&2; exit 1; }; \
	 nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 test -n "$$nr" || { echo "FATAL: cannot resolve NR archive" >&2; exit 1; }; \
	 spatial="$$($(MAKE) -s --no-print-directory -C $(AUDIOLIB_DIR) $(FWD) print-spatial-common-lib-path)"; \
	 doa="$$($(MAKE) -s --no-print-directory -C $(AUDIOLIB_DIR) $(FWD) print-doa-lib-path)"; \
	 bf="$$($(MAKE) -s --no-print-directory -C $(AUDIOLIB_DIR) $(FWD) print-bf-lib-path)"; \
	 test -n "$$spatial" || { echo "FATAL: cannot resolve spatial-common archive" >&2; exit 1; }; \
	 test -n "$$doa" || { echo "FATAL: cannot resolve DOA archive" >&2; exit 1; }; \
	 test -n "$$bf" || { echo "FATAL: cannot resolve BF archive" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory $@ COMMON_LIB="$$ac" AEC_LIB="$$aec" NR_LIB="$$nr" \
	   SPATIAL_COMMON_LIB="$$spatial" DOA_LIB="$$doa" BF_LIB="$$bf"
else
ifneq ($(words $(COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(BF_LIB)),6)
$(error COMMON_LIB, AEC_LIB, NR_LIB, SPATIAL_COMMON_LIB, DOA_LIB, and BF_LIB must be supplied together)
endif
COMMON_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(COMMON_LIB))))
AEC_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(AEC_LIB))))
NR_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(NR_LIB))))
SPATIAL_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(SPATIAL_COMMON_LIB))))
endif

SRCS := 4aec_nr_res.c audio_pipeline_4ch.c \
        ../common/se_config.c ../common/se_anr_stage.c \
        ../4ch_aec_bf_dfn_res/4aec_dfn_res.c ../dfn_res_stage.c ../dfn_rate_bridge.c \
        $(DFN2_DIR)/dfn2_process.c $(DFN2_DIR)/dfn2_model_io.c \
        $(DFN2_DIR)/dfn2_prepost.c \
        ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.c \
        audio_pipeline_4ch_raw.c mcab_sample_driver.c \
        4aec_nr_res_static.c audio_pipeline_4ch_static.c \
        tests/test_4aec_nr_res.c tests/test_mcab_sample_driver.c \
        ../4ch_aec_bf_dfn_res/tests/test_4aec_dfn_res.c \
        ../4ch_aec_bf_dfn_res/main.c \
        tests/test_audio_pipeline_4ch.c tests/dump_delay_parity.c \
        ../4ch_alignulcnet/tests/test_audio_pipeline_4ch_ulcnet.c \
        $(ULCNET_DIR)/ulcnet_accelerator_adapter.c \
        $(ULCNET_DIR)/ulcnet_prepost.c \
        ../4ch_alignulcnet/main.c $(ULCNET_DIR)/ulcnet_process.c \
        $(ULCNET_DIR)/ulcnet_model_io.c
# The same CC command can select a different SDK through PATH.
# Include effective pointer/FP/ISA predefines to avoid cross-ABI cache reuse.
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) ULCNET_GRID=$(ULCNET_GRID) CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) LDFLAGS=$(LDLIBS) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) COMMON_PRODUCER_SIG=$(COMMON_PRODUCER_SIG) AEC_PRODUCER_SIG=$(AEC_PRODUCER_SIG) NR_PRODUCER_SIG=$(NR_PRODUCER_SIG) SPATIAL_PRODUCER_SIG=$(SPATIAL_PRODUCER_SIG) SRCS=$(sort $(SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR := $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR := $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)
PIPELINE_LIB := $(BIN_DIR)/libaudio_pipeline_4ch.a
DFN_PIPELINE_LIB := $(BIN_DIR)/lib4aec_dfn_res.a

# Only the reusable 4-lane linear core and the shared configuration/NR stage
# code it uses are archived. Standard and ULCNet application wrappers are
# linked as objects into their respective binaries.
PIPELINE_OBJS := $(OBJ_DIR)/4aec_nr_res.o $(OBJ_DIR)/se_config.o $(OBJ_DIR)/se_anr_stage.o
DFN_PIPELINE_OBJS := $(OBJ_DIR)/4aec_dfn_res.o \
                     $(OBJ_DIR)/dfn_res_stage.o \
                     $(OBJ_DIR)/dfn_rate_bridge.o \
                     $(OBJ_DIR)/dfn2_process.o \
                     $(OBJ_DIR)/dfn2_model_io.o \
                     $(OBJ_DIR)/dfn2_prepost.o

.DEFAULT_GOAL := all
.PHONY: _cfg_guard FORCE clean clean-all

_cfg_guard:
	@test -n "$(CC_TARGET_SIG)" && test "$(CC_TARGET_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine compiler target; CC preprocessing with the configured flags must succeed." >&2; \
	  exit 1; }
	@if [ "$(TOOLCHAIN_CHECK)" = 1 ]; then \
	  cc_m="$$($(CC) -dumpmachine 2>/dev/null)"; \
	  cxx_m="$$($(CXX) -dumpmachine 2>/dev/null)"; \
	  test -n "$$cc_m" && test "$$cc_m" = "$$cxx_m" || { \
	    echo "FATAL: CC/CXX target mismatch (CC=$$cc_m, CXX=$$cxx_m)" >&2; exit 1; }; \
	fi
	@mkdir -p "$(OBJ_DIR)" "$(BIN_DIR)"
	@for d in "$(OBJ_DIR)" "$(BIN_DIR)"; do \
	  manifest="$$d/config.manifest"; \
	  if [ -f "$$manifest" ]; then \
	    test "$$(cat "$$manifest")" = "$(CFG_SIG_PAYLOAD)" || { \
	      echo "FATAL: CFG_SIG collision in $$manifest" >&2; exit 1; }; \
	  else \
	    tmp="$$d/.config.manifest.tmp.$$$$"; \
	    printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$tmp"; mv -f "$$tmp" "$$manifest"; \
	  fi; \
	done

$(OBJ_DIR)/4aec_nr_res.o: 4aec_nr_res.c 4aec_nr_res.h 4aec_nr_res_internal.h ../common/se_config.h ../common/se_anr_stage.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/se_config.o: ../common/se_config.c ../common/se_config.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/se_anr_stage.o: ../common/se_anr_stage.c ../common/se_anr_stage.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/4aec_dfn_res.o: ../4ch_aec_bf_dfn_res/4aec_dfn_res.c ../4ch_aec_bf_dfn_res/4aec_dfn_res.h 4aec_nr_res.h ../dfn_res_stage.h ../dfn_rate_bridge.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I../4ch_aec_bf_dfn_res -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn_res_stage.o: ../dfn_res_stage.c ../dfn_res_stage.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn_rate_bridge.o: ../dfn_rate_bridge.c ../dfn_rate_bridge.h ../dfn_res_stage.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn2_process.o: $(DFN2_DIR)/dfn2_process.c $(DFN2_DIR)/dfn2_process.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn2_model_io.o: $(DFN2_DIR)/dfn2_model_io.c $(DFN2_DIR)/dfn2_model_io.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn2_prepost.o: $(DFN2_DIR)/dfn2_prepost.c $(DFN2_DIR)/dfn2_prepost.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_4aec_dfn_res.o: ../4ch_aec_bf_dfn_res/tests/test_4aec_dfn_res.c ../4ch_aec_bf_dfn_res/4aec_dfn_res.h ../tests/dfn_test_fixture.h | _cfg_guard
	$(CC) $(CFLAGS) -I../4ch_aec_bf_dfn_res -MD -MP -c -o $@ $<

$(OBJ_DIR)/4ch_aec_bf_dfn_res_main.o: ../4ch_aec_bf_dfn_res/main.c mcab_sample_driver.h ../4ch_aec_bf_dfn_res/4aec_dfn_res.h ../dfn_example_args.h ../dfn_unit_partition.h | _cfg_guard
	$(CC) $(CFLAGS) -I../4ch_aec_bf_dfn_res -MD -MP -c -o $@ $<

$(OBJ_DIR)/audio_pipeline_4ch.o: audio_pipeline_4ch.c audio_pipeline_4ch.h 4aec_nr_res.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/audio_pipeline_4ch_ulcnet.o: ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.c ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.h audio_pipeline_4ch.h 4aec_nr_res.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_process.o: $(ULCNET_DIR)/ulcnet_process.c $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_model_io.o: $(ULCNET_DIR)/ulcnet_model_io.c $(ULCNET_DIR)/ulcnet_model_io.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

# ulcnet_prepost.o: the AIAEC-owned pre/post class the accelerator adapter
# drives (the same third TU pipelines/Makefile links for the mono
# application). Anything linking the adapter links this with it.
$(OBJ_DIR)/ulcnet_prepost.o: $(ULCNET_DIR)/ulcnet_prepost.c $(ULCNET_DIR)/ulcnet_prepost.h $(ULCNET_DIR)/ulcnet_process.h $(ULCNET_DIR)/ulcnet_model_io.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_accelerator_adapter.o: $(ULCNET_DIR)/ulcnet_accelerator_adapter.c $(ULCNET_DIR)/ulcnet_accelerator_adapter.h $(ULCNET_DIR)/ulcnet_prepost.h $(ULCNET_DIR)/ulcnet_model_io.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/4ch_alignulcnet_main.o: ../4ch_alignulcnet/main.c mcab_sample_driver.h ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.h $(ULCNET_DIR)/ulcnet_accelerator_adapter.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/audio_pipeline_4ch_raw.o: audio_pipeline_4ch_raw.c audio_pipeline_4ch.h mcab_sample_driver.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/mcab_sample_driver.o: mcab_sample_driver.c mcab_sample_driver.h audio_pipeline_4ch.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/4aec_nr_res_static.o: 4aec_nr_res_static.c 4aec_nr_res.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/audio_pipeline_4ch_static.o: audio_pipeline_4ch_static.c audio_pipeline_4ch.h mcab_sample_driver.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_4aec_nr_res.o: tests/test_4aec_nr_res.c 4aec_nr_res.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_audio_pipeline_4ch.o: tests/test_audio_pipeline_4ch.c audio_pipeline_4ch.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dump_delay_parity.o: tests/dump_delay_parity.c 4aec_nr_res.h 4aec_nr_res_internal.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_audio_pipeline_4ch_ulcnet.o: ../4ch_alignulcnet/tests/test_audio_pipeline_4ch_ulcnet.c ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.h audio_pipeline_4ch.h 4aec_nr_res.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(PIPELINE_LIB): $(PIPELINE_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && \
	  $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "libaudio_pipeline_4ch.a [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(DFN_PIPELINE_LIB): $(DFN_PIPELINE_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && \
	  $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "lib4aec_dfn_res.a [$(BACKEND), SIMD=$(SIMD)] -> $@"

ifneq ($(COMMON_LIB),)
.PHONY: all test libs lib libaudio_pipeline_4ch.a lib4aec_dfn_res.a \
        audio_pipeline_4ch_raw 4aec_nr_res_static audio_pipeline_4ch_static 4ch_alignulcnet \
        4ch_aec_bf_dfn_res \
        test_4aec_nr_res test_audio_pipeline_4ch test_4aec_dfn_res \
        test_audio_pipeline_4ch_ulcnet dump_delay_parity \
        audit-no-stdio test-third-party

all: $(PIPELINE_LIB) $(DFN_PIPELINE_LIB) $(BIN_DIR)/audio_pipeline_4ch_raw \
     $(BIN_DIR)/4aec_nr_res_static $(BIN_DIR)/audio_pipeline_4ch_static \
     $(BIN_DIR)/4ch_aec_bf_dfn_res

# The 4ch_alignulcnet application shares this Makefile (its wrapper compiles
# against this directory's core), so its acceptance test and board skeleton
# run here too: this directory's `make test` is the one-stop gate, mirroring
# the mono arrangement in pipelines/Makefile.
test: test-third-party test-mcab-sample-driver \
      $(BIN_DIR)/test_4aec_nr_res \
      $(BIN_DIR)/test_4aec_dfn_res \
      $(BIN_DIR)/test_audio_pipeline_4ch $(BIN_DIR)/4aec_nr_res_static \
      $(BIN_DIR)/audio_pipeline_4ch_static \
      $(BIN_DIR)/test_audio_pipeline_4ch_ulcnet $(BIN_DIR)/4ch_alignulcnet \
      $(BIN_DIR)/4ch_aec_bf_dfn_res
	"$(abspath $(BIN_DIR))/test_4aec_nr_res"
	"$(abspath $(BIN_DIR))/test_4aec_dfn_res"
	"$(abspath $(BIN_DIR))/test_audio_pipeline_4ch"
	"$(abspath $(BIN_DIR))/test_audio_pipeline_4ch_ulcnet"
	"$(abspath $(BIN_DIR))/4aec_nr_res_static" --sample-rate 16000
	"$(abspath $(BIN_DIR))/4aec_nr_res_static" --sample-rate 48000
	"$(abspath $(BIN_DIR))/audio_pipeline_4ch_static" --sample-rate 16000
	"$(abspath $(BIN_DIR))/audio_pipeline_4ch_static" --sample-rate 48000
	"$(abspath $(BIN_DIR))/4ch_alignulcnet"
	"$(abspath $(BIN_DIR))/4ch_alignulcnet" --delay-num-filters 1
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res"
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --delay-num-filters 1 --cng --atten-lim 12
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --sample-rate 48000
# Malformed numbers, and the 8 kHz grid the core does not offer, are refused
# at parse time with the documented exit code 2.
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --sample-rate 16000Hz >/dev/null 2>&1; test $$? -eq 2
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --sample-rate 8000 >/dev/null 2>&1; test $$? -eq 2
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --atten-lim 12dB >/dev/null 2>&1; test $$? -eq 2

lib libaudio_pipeline_4ch.a: $(PIPELINE_LIB)
lib4aec_dfn_res.a: $(DFN_PIPELINE_LIB)
libs: $(COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(BF_LIB) $(PIPELINE_LIB) $(DFN_PIPELINE_LIB)
audio_pipeline_4ch_raw: $(BIN_DIR)/audio_pipeline_4ch_raw
4aec_nr_res_static: $(BIN_DIR)/4aec_nr_res_static
audio_pipeline_4ch_static: $(BIN_DIR)/audio_pipeline_4ch_static
4ch_alignulcnet: $(BIN_DIR)/4ch_alignulcnet
4ch_aec_bf_dfn_res: $(BIN_DIR)/4ch_aec_bf_dfn_res
test_4aec_nr_res: $(BIN_DIR)/test_4aec_nr_res
test_4aec_dfn_res: $(BIN_DIR)/test_4aec_dfn_res
test_audio_pipeline_4ch: $(BIN_DIR)/test_audio_pipeline_4ch
test_audio_pipeline_4ch_ulcnet: $(BIN_DIR)/test_audio_pipeline_4ch_ulcnet
# CSV dumper for the C/Python per-hop alignment parity test
# (tests/test_delay_parity.py builds and runs it); not part of `make test`,
# which only runs self-checking binaries.
dump_delay_parity: $(BIN_DIR)/dump_delay_parity

# Same audit as the mono full stack: no archive may reference the stdio set
# (printf family, stderr, fputc, fwrite).
audit-no-stdio: $(PIPELINE_LIB) $(DFN_PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) \
                $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(BF_LIB)
	@test "$(NO_STDIO)" = 1 || { \
	  echo "FATAL: audit-no-stdio must be invoked with NO_STDIO=1" >&2; \
	  exit 1; \
	}
	@pat='^_?(fprintf|vfprintf|printf|puts|fputs|stderr|__stderrp|fputc|fwrite)$$'; \
	 for archive in "$(PIPELINE_LIB)" "$(DFN_PIPELINE_LIB)" "$(AEC_LIB)" "$(NR_LIB)" "$(COMMON_LIB)" \
	                "$(SPATIAL_COMMON_LIB)" "$(DOA_LIB)" "$(BF_LIB)"; do \
	  hits=$$(nm -A "$$archive" | awk -v re="$$pat" '{ n=$$NF; if (n ~ re) print }'); \
	  if [ -n "$$hits" ]; then echo "FAIL: stdio symbols in $$archive" >&2; echo "$$hits" >&2; exit 1; fi; \
	 done; \
	 echo "4ch pipeline stack: NO_STDIO audit PASS"

FORCE:

$(COMMON_LIB): FORCE
	+$(MAKE) -C $(COMMON_DIR) $(FWD) lib
	@test -f '$@' || { echo "FATAL: common did not produce $@" >&2; exit 1; }

$(AEC_LIB): FORCE
	+$(MAKE) -C $(AEC_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB=$(COMMON_LIB) lib
	@test -f '$@' || { echo "FATAL: AEC did not produce $@" >&2; exit 1; }

$(NR_LIB): FORCE
	+$(MAKE) -C $(NR_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB=$(COMMON_LIB) lib
	@test -f '$@' || { echo "FATAL: NR did not produce $@" >&2; exit 1; }

$(SPATIAL_COMMON_LIB): FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) libspatial_common.a
	@test -f '$@' || { echo "FATAL: spatial-common did not produce $@" >&2; exit 1; }

$(DOA_LIB): FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) libdoa.a
	@test -f '$@' || { echo "FATAL: DOA did not produce $@" >&2; exit 1; }

$(BF_LIB): FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) libbf.a
	@test -f '$@' || { echo "FATAL: BF did not produce $@" >&2; exit 1; }

$(BIN_DIR)/test_4aec_nr_res: $(OBJ_DIR)/test_4aec_nr_res.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_4aec_nr_res.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)

$(BIN_DIR)/test_4aec_dfn_res: $(OBJ_DIR)/test_4aec_dfn_res.o $(DFN_PIPELINE_LIB) $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_4aec_dfn_res.o $(DFN_PIPELINE_LIB) $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)

$(BIN_DIR)/dump_delay_parity: $(OBJ_DIR)/dump_delay_parity.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/dump_delay_parity.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)

# The spatial dependencies (SRP-PHAT, BF, VAD) live in audiolib and test themselves there; this
# target only invokes audiolib's tests.
test-third-party: FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) test

$(BIN_DIR)/test_audio_pipeline_4ch: $(OBJ_DIR)/test_audio_pipeline_4ch.o $(OBJ_DIR)/audio_pipeline_4ch.o $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_audio_pipeline_4ch.o $(OBJ_DIR)/audio_pipeline_4ch.o $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)

ULCNET_4CH_OBJS := $(OBJ_DIR)/audio_pipeline_4ch_ulcnet.o \
                   $(OBJ_DIR)/audio_pipeline_4ch.o \
                   $(OBJ_DIR)/ulcnet_process.o

# The wrapper itself needs no ulcnet_model_io symbol (it only READS the
# descriptor a runtime published, so a board integrating the wrapper alone
# gains no new link dependency); the test links it to BUILD descriptors for
# the far-input contract gate.
$(BIN_DIR)/test_audio_pipeline_4ch_ulcnet: $(OBJ_DIR)/test_audio_pipeline_4ch_ulcnet.o $(ULCNET_4CH_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_audio_pipeline_4ch_ulcnet.o $(ULCNET_4CH_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)

# The adapter drives the pre/post class, so the application links
# ulcnet_prepost.o with it (the test above does not link the adapter).
ULCNET_ADAPTER_OBJS := $(OBJ_DIR)/ulcnet_accelerator_adapter.o \
                       $(OBJ_DIR)/ulcnet_prepost.o \
                       $(OBJ_DIR)/ulcnet_model_io.o

$(BIN_DIR)/4ch_alignulcnet: $(OBJ_DIR)/4ch_alignulcnet_main.o $(OBJ_DIR)/mcab_sample_driver.o $(ULCNET_ADAPTER_OBJS) $(ULCNET_4CH_OBJS) $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/4ch_alignulcnet_main.o $(OBJ_DIR)/mcab_sample_driver.o $(ULCNET_ADAPTER_OBJS) $(ULCNET_4CH_OBJS) $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)

# The 4-channel DFN2 application wraps the core alone (the beamformer (bf) seam stays
# external, as in 4aec_nr_res_static), so it links the DFN archive and the
# core archive, not the spatial libraries.
$(BIN_DIR)/4ch_aec_bf_dfn_res: $(OBJ_DIR)/4ch_aec_bf_dfn_res_main.o $(OBJ_DIR)/mcab_sample_driver.o $(DFN_PIPELINE_LIB) $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/4ch_aec_bf_dfn_res_main.o $(OBJ_DIR)/mcab_sample_driver.o $(DFN_PIPELINE_LIB) $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)

$(BIN_DIR)/4aec_nr_res_static: $(OBJ_DIR)/4aec_nr_res_static.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/4aec_nr_res_static.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)

$(BIN_DIR)/audio_pipeline_4ch_static: $(OBJ_DIR)/audio_pipeline_4ch_static.o $(OBJ_DIR)/audio_pipeline_4ch.o $(OBJ_DIR)/mcab_sample_driver.o $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/audio_pipeline_4ch_static.o $(OBJ_DIR)/audio_pipeline_4ch.o $(OBJ_DIR)/mcab_sample_driver.o $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)

$(BIN_DIR)/audio_pipeline_4ch_raw: $(OBJ_DIR)/audio_pipeline_4ch_raw.o $(OBJ_DIR)/audio_pipeline_4ch.o $(OBJ_DIR)/mcab_sample_driver.o $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/audio_pipeline_4ch_raw.o $(OBJ_DIR)/audio_pipeline_4ch.o $(OBJ_DIR)/mcab_sample_driver.o $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) $(LDLIBS)
endif

ifeq ($(COMMON_LIB),)
# Query targets are dispatched above.
else
print-bin-dir:
	@echo $(abspath $(BIN_DIR))
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))
print-lib-path:
	@echo $(abspath $(PIPELINE_LIB))
endif

clean:
	rm -rf "$(OBJ_ROOT)" "$(BIN_ROOT)"

clean-all: clean
	$(MAKE) -C $(AUDIOLIB_DIR) clean
	$(MAKE) -C $(AEC_DIR) clean
	$(MAKE) -C $(NR_DIR) clean
	$(MAKE) -C $(COMMON_DIR) clean

-include $(wildcard $(OBJ_DIR)/*.d)

# test-grids: runs both Align-ULCNet deployment grids in one command (`test` only exercises the grid
# ULCNET_GRID selects). ULCNET_GRID is in CFG_SIG, so each grid has its own obj/bin directory.
# Defined outside the COMMON_LIB bootstrap conditionals: it needs no producer archive itself and each
# recursive `test` bootstraps its own.
.PHONY: test-grids
test-grids:
	$(MAKE) ULCNET_GRID=16k test
	$(MAKE) ULCNET_GRID=48k test

# The sample runner has no algorithm-library link dependency.
.PHONY: test-mcab-sample-driver
test-mcab-sample-driver: $(BIN_DIR)/test_mcab_sample_driver
	"$(abspath $(BIN_DIR))/test_mcab_sample_driver"
$(BIN_DIR)/test_mcab_sample_driver: tests/test_mcab_sample_driver.c mcab_sample_driver.c mcab_sample_driver.h | _cfg_guard
	$(CC) $(CFLAGS) -o $@ tests/test_mcab_sample_driver.c mcab_sample_driver.c -lm
