# Four-channel linear AEC + beamformer flow (mcab) and the examples composed over it.
#
# This Makefile lives beside the 4-channel implementation; the parent pipelines/Makefile builds the
# mono examples. libaudio_pipeline_mcab.a holds the flow layer, which drives the lane and
# estimator objects the application creates; mcab_sample_flow.o (the application-side init in one
# place) and mcab_sample_beam.o (SRP-PHAT/BF) are sample helpers; the post-beam RES (lib/aec), NR, comfort noise, synthesis and the neural post-filters are
# separate libraries the example binaries chain.
# AEC, NR, common, DOA, BF and spatial kernels remain component libs.
#
# Producers: audiolib's libaudio_common.a, lib/aec's libaec.a, lib/nr's libanr.a and audiolib's
# spatial kernels (libspatial_common.a, libdoa.a, libbf.a, built by one audiolib invocation and
# treated as one producer, "spatial"). Every goal is dispatched on its own (phase 1) and resolves
# the keyed archive path of the producers it needs; those identities, and only those, enter its
# CFG_SIG, so the object/binary directory of a goal does not change when a producer it never
# uses is reconfigured, and that producer need not exist:
#   libaudio_pipeline_mcab.a, lib   common, aec         (mcab.o references only lib/aec)
#   test_mcab                       common, aec, nr     (its reference chain calls the NR)
#   every other goal                common, aec, nr, spatial
# A goal built in a smaller scope lands in a different bin/obj directory than the same goal built
# by `make all` or `make test`. print-lib-path prints the flow archive of the smaller scope;
# print-bin-dir and print-obj-dir print the directory of `make all`, and PRINT_FOR=<goal> prints
# the one `make <goal>` uses (PRINT_FOR=all for the archive of `make all`):
#   make test_mcab && "$(make -s print-bin-dir PRINT_FOR=test_mcab)/test_mcab"
# Phase 2 (producer archives supplied) refuses a goal whose producers it did not receive.
#
# Common invocations:
#   make SIMD=1
#   make SIMD=0 test
#   make mcab_raw
#   make libaudio_pipeline_mcab.a   # needs only common and lib/aec (NR_DIR may be absent)

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

# DeepFilterNet2 lookahead pair, same knobs and rules as pipelines/Makefile: a pair other than the
# headers' default 1/1 becomes -D flags on every unit compiled here, the shipped pair adds none,
# and the pair is folded into CFG_SIG_PAYLOAD. (No DFN3 unit is compiled here.)
DFN2_MASK_LOOKAHEAD ?= 1
DFN2_DF_LOOKAHEAD ?= 1
$(if $(filter-out 1,$(words $(DFN2_MASK_LOOKAHEAD)))$(filter-out 0 1 2,$(DFN2_MASK_LOOKAHEAD)),\
  $(error DFN2_MASK_LOOKAHEAD must be 0, 1 or 2, got '$(DFN2_MASK_LOOKAHEAD)'))
$(if $(filter-out 1,$(words $(DFN2_DF_LOOKAHEAD)))$(filter-out 0 1 2 3 4,$(DFN2_DF_LOOKAHEAD)),\
  $(error DFN2_DF_LOOKAHEAD must be 0..4 (DF_ORDER - 1), got '$(DFN2_DF_LOOKAHEAD)'))
DFN_PAIR_CFLAGS :=
ifneq ($(DFN2_MASK_LOOKAHEAD)/$(DFN2_DF_LOOKAHEAD),1/1)
DFN_PAIR_CFLAGS += -DDFN2_MASK_LOOKAHEAD=$(DFN2_MASK_LOOKAHEAD) -DDFN2_DF_LOOKAHEAD=$(DFN2_DF_LOOKAHEAD)
endif

BACKEND = ne10
SIMD ?= 1
WERROR = 0
NO_STDIO = 0
DEBUG = 0
TOOLCHAIN_CHECK = 1
EXTRA_CFLAGS =
# PROFILE=1 turns on per-hop stage timing across the whole chain: the module's own stages and,
# through the PROFILE knob forwarded to the lib/aec build, the four summed off the lanes. Off
# otherwise, so a release build takes no clock reads; McabLastTiming's doc comment (mcab.h)
# carries the field list.

# PROFILE_CFLAGS goes into CFLAGS rather than EXTRA_CFLAGS, as SIMD does: EXTRA_CFLAGS is forwarded
# to every producer sub-make and folded into CFG_SIG, so a -D smuggled through it reached lib/aec
# twice, splitting identical objects across two keys and re-keying producers that never use the
# macro. CFLAGS here is `override CFLAGS :=`, so a command-line assignment cannot drop the knob.
# Both -D land in this Makefile's CFLAGS only; lib/aec gets AEC_STAGE_TIMING from the PROFILE knob
# forwarded by name in FWD.

# AEC_STAGE_TIMING is repeated on purpose: this layer's own TUs test it, to decide whether to copy
# the lanes' record out and what the tests may assert about it; deriving both flags from PROFILE
# keeps the two views consistent. A lib/aec pin without the PROFILE knob degrades to the one-half
# state: the module's stages measure and the AEC block reads zero.
PROFILE ?= 0
ifeq ($(PROFILE),1)
PROFILE_CFLAGS := -DMCAB_STAGE_TIMING=1 -DAEC_STAGE_TIMING=1
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

# lib/nr's headers are on the include path only when this goal received libanr.a (see the producer
# scope below): a goal that needs no NR neither names NR_DIR on its compile lines nor keys on it.
# audiolib's include directories (common, doa, bf) stay on every goal: they are source directories,
# and test_mcab reaches the sample beam headers through tests/mcab_test_rig.h without linking them.
NR_INC := $(if $(NR_LIB),-I$(NR_DIR)/include)

override CFLAGS := -Wall -Wextra $(OPT_CFLAGS) -std=gnu99 \
          -I. -I$(PIPELINES_DIR) \
          -I$(AEC_DIR)/include \
          $(NR_INC) -I$(COMMON_DIR)/include -I$(COMMON_DIR) \
          -I$(AUDIOLIB_DIR)/common/utility \
          -I$(AUDIOLIB_DIR)/doa \
          -I$(AUDIOLIB_DIR)/bf \
          -I$(DFN2_DIR) \
          $(CPPFLAGS) $(EXTRA_CFLAGS) $(SIMD_CFLAGS) $(PROFILE_CFLAGS) \
          -DAUDIO_PIPELINE_BACKEND_STR=\"$(BACKEND)\" \
          $(WERROR_CFLAGS) -ffp-contract=off -fno-math-errno
ifneq ($(DFN_PAIR_CFLAGS),)
override CFLAGS += $(DFN_PAIR_CFLAGS)
endif
LIB_CFLAGS := $(CFLAGS)

LINK := $(CXX)
LDLIBS := $(EXTRA_LDFLAGS) $(LDFLAGS) -lm

FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) PROFILE=$(PROFILE) WERROR=$(WERROR) \
      CC='$(CC)' CXX='$(CXX)' AR='$(AR)' RANLIB='$(RANLIB)' \
      EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) \
      TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) DEBUG=$(DEBUG)

# Producer scope. Each goal is dispatched on its own and resolves only the producers in its
# scope (PRODUCERS_ALL unless a NEEDS_<goal> line below narrows it); audiolib's libaudio_common.a
# is in every scope because the keyed paths of lib/aec and lib/nr fold it in. A goal that links an
# archive, or whose archive consumer links it, lists that producer; the rest are neither
# resolved nor keyed. aec = lib/aec's libaec.a, nr = lib/nr's libanr.a, spatial = audiolib's
# libspatial_common.a + libdoa.a + libbf.a (one audiolib invocation builds the three).
PRODUCERS_ALL := aec nr spatial
NEEDS_lib := aec
NEEDS_libaudio_pipeline_mcab.a := aec
NEEDS_test_mcab := aec nr

DISPATCH_GOALS := all test libs lib libaudio_pipeline_mcab.a libdfn_res_stage.a \
        mcab_raw mcab_external_static mcab_static 4ch_alignulcnet \
        4ch_aec_bf_dfn_res \
        test_mcab test_mcab_sample_beam test_4ch_dfn \
        test_4ch_ulcnet dump_delay_parity \
        print-bin-dir print-obj-dir print-lib-path audit-no-stdio \
        test-third-party test-mcab-sample-driver

# The query goals report the directory of another goal's build: PRINT_FOR=<goal> names it. Without
# it print-bin-dir and print-obj-dir report `make all` and print-lib-path the flow archive of
# `make libaudio_pipeline_mcab.a`.
PRINT_FOR ?=
ifneq ($(PRINT_FOR),)
ifneq ($(words $(PRINT_FOR)),1)
$(error PRINT_FOR must name exactly one goal, got '$(PRINT_FOR)')
endif
ifeq ($(filter $(PRINT_FOR),$(filter-out print-%,$(DISPATCH_GOALS))),)
$(error PRINT_FOR='$(PRINT_FOR)' is not a goal of this Makefile)
endif
endif

goal_scope = $(strip $(if $(filter print-lib-path,$(1)),$(or $(PRINT_FOR),libaudio_pipeline_mcab.a),$(if $(filter print-bin-dir print-obj-dir,$(1)),$(or $(PRINT_FOR),all),$(1))))
goal_needs = $(if $(filter undefined,$(origin NEEDS_$(call goal_scope,$(1)))),$(PRODUCERS_ALL),$(NEEDS_$(call goal_scope,$(1))))

# Resolve the producers of the goal's scope before constructing this consumer's CFG identity.
ifeq ($(COMMON_LIB),)
.PHONY: $(DISPATCH_GOALS)
RESOLVE_AEC = aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 test -n "$$aec" || { echo "FATAL: cannot resolve AEC archive" >&2; exit 1; };
RESOLVE_NR = nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 test -n "$$nr" || { echo "FATAL: cannot resolve NR archive" >&2; exit 1; };
RESOLVE_SPATIAL = spatial="$$($(MAKE) -s --no-print-directory -C $(AUDIOLIB_DIR) $(FWD) print-spatial-common-lib-path)"; \
	 doa="$$($(MAKE) -s --no-print-directory -C $(AUDIOLIB_DIR) $(FWD) print-doa-lib-path)"; \
	 bf="$$($(MAKE) -s --no-print-directory -C $(AUDIOLIB_DIR) $(FWD) print-bf-lib-path)"; \
	 test -n "$$spatial" || { echo "FATAL: cannot resolve spatial-common archive" >&2; exit 1; }; \
	 test -n "$$doa" || { echo "FATAL: cannot resolve DOA archive" >&2; exit 1; }; \
	 test -n "$$bf" || { echo "FATAL: cannot resolve BF archive" >&2; exit 1; };
$(DISPATCH_GOALS):
	+@ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: cannot resolve common archive" >&2; exit 1; }; \
	 $(if $(filter aec,$(call goal_needs,$@)),$(RESOLVE_AEC)) \
	 $(if $(filter nr,$(call goal_needs,$@)),$(RESOLVE_NR)) \
	 $(if $(filter spatial,$(call goal_needs,$@)),$(RESOLVE_SPATIAL)) \
	 $(MAKE) --no-print-directory $@ COMMON_LIB="$$ac" \
	   $(if $(filter aec,$(call goal_needs,$@)),AEC_LIB="$$aec") \
	   $(if $(filter nr,$(call goal_needs,$@)),NR_LIB="$$nr") \
	   $(if $(filter spatial,$(call goal_needs,$@)),SPATIAL_COMMON_LIB="$$spatial" DOA_LIB="$$doa" BF_LIB="$$bf")
else
ifneq ($(words $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(BF_LIB)),$(if $(SPATIAL_COMMON_LIB)$(DOA_LIB)$(BF_LIB),3,0))
$(error SPATIAL_COMMON_LIB, DOA_LIB and BF_LIB must be supplied together)
endif
HAVE_PRODUCERS := $(if $(AEC_LIB),aec) $(if $(NR_LIB),nr) $(if $(SPATIAL_COMMON_LIB),spatial)
# A goal runs only with the producers of its scope: a missing one is named here, at parse time,
# instead of surfacing as an unresolved symbol or header at the end of a compile or link.
$(foreach g,$(filter $(DISPATCH_GOALS),$(or $(MAKECMDGOALS),all)),\
  $(foreach p,$(call goal_needs,$(g)),\
    $(if $(filter $(p),$(HAVE_PRODUCERS)),,\
      $(error '$(g)' needs the $(p) producer archive(s), which this invocation did not receive (AEC_LIB / NR_LIB / SPATIAL_COMMON_LIB+DOA_LIB+BF_LIB); run plain 'make $(g)' so the dispatch resolves them))))
COMMON_PRODUCER_SIG  := $(notdir $(patsubst %/,%,$(dir $(COMMON_LIB))))
AEC_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(AEC_LIB))))
NR_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(NR_LIB))))
SPATIAL_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(SPATIAL_COMMON_LIB))))
endif

# The producer identities of this goal's scope, in the payload's fixed order. The common archive
# is always present; a producer outside the scope has no field, so it cannot change the sig.
PRODUCER_SIGS := COMMON_PRODUCER_SIG=$(COMMON_PRODUCER_SIG)
ifneq ($(AEC_LIB),)
PRODUCER_SIGS += AEC_PRODUCER_SIG=$(AEC_PRODUCER_SIG)
endif
ifneq ($(NR_LIB),)
PRODUCER_SIGS += NR_PRODUCER_SIG=$(NR_PRODUCER_SIG)
endif
ifneq ($(SPATIAL_COMMON_LIB),)
PRODUCER_SIGS += SPATIAL_PRODUCER_SIG=$(SPATIAL_PRODUCER_SIG)
endif

SRCS := mcab.c mcab_sample_flow.c mcab_sample_beam.c \
        ../dfn/dfn_res_stage.c ../dfn/dfn_rate_bridge.c \
        $(DFN2_DIR)/dfn2_process.c $(DFN2_DIR)/dfn2_model_io.c \
        $(DFN2_DIR)/dfn2_prepost.c \
        ../ulcnet/ulcnet_nn_stage.c \
        mcab_raw.c mcab_sample_driver.c \
        mcab_external_static.c mcab_static.c \
        tests/test_mcab.c tests/test_mcab_sample_driver.c \
        ../dfn/4ch_aec_bf_dfn_res/tests/test_4ch_dfn.c \
        ../dfn/4ch_aec_bf_dfn_res/main.c \
        tests/test_mcab_sample_beam.c tests/dump_delay_parity.c \
        ../ulcnet/4ch_alignulcnet/tests/test_4ch_ulcnet.c \
        $(ULCNET_DIR)/ulcnet_accelerator_adapter.c \
        $(ULCNET_DIR)/ulcnet_prepost.c \
        ../ulcnet/4ch_alignulcnet/main.c $(ULCNET_DIR)/ulcnet_process.c \
        $(ULCNET_DIR)/ulcnet_model_io.c
# The same CC command can select a different SDK through PATH.
# Include effective pointer/FP/ISA predefines to avoid cross-ABI cache reuse.
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) ULCNET_GRID=$(ULCNET_GRID) DFN2_LOOKAHEAD=$(DFN2_MASK_LOOKAHEAD)/$(DFN2_DF_LOOKAHEAD) CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) LDFLAGS=$(LDLIBS) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) $(PRODUCER_SIGS) SRCS=$(sort $(SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR := $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR := $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)
PIPELINE_LIB := $(BIN_DIR)/libaudio_pipeline_mcab.a
DFN_PIPELINE_LIB := $(BIN_DIR)/libdfn_res_stage.a

# The module: the shared delay, the four-lane flow and the fuse. The lane configs, the
# beamformer helper, the post-beam stages and the NN stages are not part of it.
PIPELINE_OBJS := $(OBJ_DIR)/mcab.o
SAMPLE_FLOW_OBJ := $(OBJ_DIR)/mcab_sample_flow.o
SAMPLE_BEAM_OBJ := $(OBJ_DIR)/mcab_sample_beam.o
DFN_PIPELINE_OBJS := $(OBJ_DIR)/dfn_res_stage.o \
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

$(OBJ_DIR)/mcab.o: mcab.c mcab.h mcab_internal.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

# Sample-level helpers over the flow; not in the module archive.
$(OBJ_DIR)/mcab_sample_flow.o: mcab_sample_flow.c mcab_sample_flow.h mcab.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/mcab_sample_beam.o: mcab_sample_beam.c mcab_sample_beam.h mcab.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn_res_stage.o: ../dfn/dfn_res_stage.c ../dfn/dfn_res_stage.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn_rate_bridge.o: ../dfn/dfn_rate_bridge.c ../dfn/dfn_rate_bridge.h ../dfn/dfn_res_stage.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn2_process.o: $(DFN2_DIR)/dfn2_process.c $(DFN2_DIR)/dfn2_process.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn2_model_io.o: $(DFN2_DIR)/dfn2_model_io.c $(DFN2_DIR)/dfn2_model_io.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dfn2_prepost.o: $(DFN2_DIR)/dfn2_prepost.c $(DFN2_DIR)/dfn2_prepost.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

# ../dfn/4ch_aec_bf_dfn_res/main.c and its tests/test_4ch_dfn.c compose the unit with
# ../dfn/dfn_hop.h.
$(OBJ_DIR)/test_4ch_dfn.o: ../dfn/4ch_aec_bf_dfn_res/tests/test_4ch_dfn.c mcab.h mcab_sample_flow.h mcab_sample_beam.h mcab_sample_post.h tests/conv_chain_mcab.h tests/mcab_test_rig.h ../dfn/dfn_hop.h ../dfn/tests/dfn_test_fixture.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/4ch_aec_bf_dfn_res_main.o: ../dfn/4ch_aec_bf_dfn_res/main.c mcab_sample_driver.h mcab.h mcab_sample_flow.h mcab_sample_beam.h mcab_sample_post.h ../dfn/dfn_hop.h ../dfn/dfn_example_args.h ../dfn/dfn_unit_partition.h ../common/example_cli.h ../dfn/dfn_hop_pool.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# ../ulcnet/4ch_alignulcnet/main.c and its tests/test_4ch_ulcnet.c compose the unit with the
# ULCNet NN stage example.
$(OBJ_DIR)/ulcnet_nn_stage.o: ../ulcnet/ulcnet_nn_stage.c ../ulcnet/ulcnet_nn_stage.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

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

$(OBJ_DIR)/4ch_alignulcnet_main.o: ../ulcnet/4ch_alignulcnet/main.c mcab_sample_driver.h mcab.h mcab_sample_flow.h mcab_sample_beam.h ../ulcnet/ulcnet_nn_stage.h ../common/example_cli.h $(ULCNET_DIR)/ulcnet_accelerator_adapter.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_4ch_ulcnet.o: ../ulcnet/4ch_alignulcnet/tests/test_4ch_ulcnet.c mcab.h mcab_sample_flow.h mcab_sample_beam.h ../ulcnet/ulcnet_nn_stage.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) $(ULCNET_GRID_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/mcab_raw.o: mcab_raw.c mcab.h mcab_sample_flow.h mcab_sample_beam.h mcab_sample_driver.h mcab_sample_post.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_mcab_sample_driver.o: tests/test_mcab_sample_driver.c mcab_sample_driver.h mcab.h mcab_sample_flow.h mcab_sample_beam.h ../common/example_cli.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/mcab_sample_driver.o: mcab_sample_driver.c mcab_sample_driver.h mcab.h mcab_sample_flow.h mcab_sample_beam.h ../common/example_cli.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/mcab_external_static.o: mcab_external_static.c mcab.h mcab_sample_post.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/mcab_static.o: mcab_static.c mcab.h mcab_sample_flow.h mcab_sample_beam.h mcab_sample_driver.h mcab_sample_post.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_mcab.o: tests/test_mcab.c mcab.h mcab_internal.h mcab_sample_flow.h tests/conv_chain_mcab.h tests/mcab_test_rig.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_mcab_sample_beam.o: tests/test_mcab_sample_beam.c mcab.h mcab_sample_flow.h mcab_sample_beam.h tests/conv_chain_mcab.h tests/mcab_test_rig.h mcab_sample_post.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dump_delay_parity.o: tests/dump_delay_parity.c mcab.h mcab_internal.h mcab_sample_flow.h tests/mcab_test_rig.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(PIPELINE_LIB): $(PIPELINE_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && \
	  $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "libaudio_pipeline_mcab.a [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(DFN_PIPELINE_LIB): $(DFN_PIPELINE_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && \
	  $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "libdfn_res_stage.a [$(BACKEND), SIMD=$(SIMD)] -> $@"

ifneq ($(COMMON_LIB),)
.PHONY: all test libs lib libaudio_pipeline_mcab.a libdfn_res_stage.a \
        mcab_raw mcab_external_static mcab_static 4ch_alignulcnet \
        4ch_aec_bf_dfn_res \
        test_mcab test_mcab_sample_beam test_4ch_dfn \
        test_4ch_ulcnet dump_delay_parity \
        audit-no-stdio test-third-party test-mcab-sample-driver

all: $(PIPELINE_LIB) $(DFN_PIPELINE_LIB) \
     $(BIN_DIR)/mcab_raw $(BIN_DIR)/mcab_external_static $(BIN_DIR)/mcab_static \
     $(BIN_DIR)/4ch_aec_bf_dfn_res

# The 4ch_alignulcnet application shares this Makefile (it composes the unit with the NN stage
# example), so its acceptance test and board skeleton run here too: this directory's `make test`
# is the one-stop gate, mirroring the mono arrangement in pipelines/Makefile.
test: test-third-party test-mcab-sample-driver \
      $(BIN_DIR)/test_mcab \
      $(BIN_DIR)/test_4ch_dfn \
      $(BIN_DIR)/test_mcab_sample_beam $(BIN_DIR)/mcab_raw \
      $(BIN_DIR)/mcab_external_static \
      $(BIN_DIR)/mcab_static \
      $(BIN_DIR)/test_4ch_ulcnet $(BIN_DIR)/4ch_alignulcnet \
      $(BIN_DIR)/4ch_aec_bf_dfn_res
	"$(abspath $(BIN_DIR))/test_mcab"
	"$(abspath $(BIN_DIR))/test_4ch_dfn"
	"$(abspath $(BIN_DIR))/test_mcab_sample_beam" "$(abspath $(BIN_DIR))" "$(abspath $(OBJ_DIR))"
	"$(abspath $(BIN_DIR))/test_4ch_ulcnet"
	"$(abspath $(BIN_DIR))/mcab_external_static" --sample-rate 16000
	"$(abspath $(BIN_DIR))/mcab_external_static" --sample-rate 48000
	"$(abspath $(BIN_DIR))/mcab_static" --sample-rate 16000
	"$(abspath $(BIN_DIR))/mcab_static" --sample-rate 48000
# The command lines of mcab_static and mcab_external_static: bad numbers exit 2, refused configs 1.
	bash tests/test_cli_strict.sh "$(abspath $(BIN_DIR))"
	"$(abspath $(BIN_DIR))/4ch_alignulcnet"
	"$(abspath $(BIN_DIR))/4ch_alignulcnet" --delay-num-filters 1
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res"
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --delay-num-filters 1 --cng --atten-lim 12
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --sample-rate 48000
# Malformed numbers, and the 8 kHz grid the core does not offer, are refused
# at parse time with the documented exit code 2 (the ULCNet example too).
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --sample-rate 16000Hz >/dev/null 2>&1; test $$? -eq 2
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --sample-rate 8000 >/dev/null 2>&1; test $$? -eq 2
	"$(abspath $(BIN_DIR))/4ch_aec_bf_dfn_res" --atten-lim 12dB >/dev/null 2>&1; test $$? -eq 2
	"$(abspath $(BIN_DIR))/4ch_alignulcnet" --delay-num-filters 3x >/dev/null 2>&1; test $$? -eq 2
	"$(abspath $(BIN_DIR))/4ch_alignulcnet" --delay-mode fixed --fixed-delay 12x >/dev/null 2>&1; test $$? -eq 2
# Empty, leading or trailing garbage and out-of-int-range numbers.
	for v in "" " 3" "3 " x3 +-1 99999999999 999999999999999999999999; do "$(abspath $(BIN_DIR))/4ch_alignulcnet" --delay-num-filters "$$v" >/dev/null 2>&1; test $$? -eq 2 || exit 1; done

lib libaudio_pipeline_mcab.a: $(PIPELINE_LIB)
libdfn_res_stage.a: $(DFN_PIPELINE_LIB)
libs: $(COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(BF_LIB) $(PIPELINE_LIB) $(DFN_PIPELINE_LIB) \
      $(SAMPLE_FLOW_OBJ) $(SAMPLE_BEAM_OBJ)
mcab_raw: $(BIN_DIR)/mcab_raw
mcab_external_static: $(BIN_DIR)/mcab_external_static
mcab_static: $(BIN_DIR)/mcab_static
4ch_alignulcnet: $(BIN_DIR)/4ch_alignulcnet
4ch_aec_bf_dfn_res: $(BIN_DIR)/4ch_aec_bf_dfn_res
test_mcab: $(BIN_DIR)/test_mcab
test_4ch_dfn: $(BIN_DIR)/test_4ch_dfn
test_mcab_sample_beam: $(BIN_DIR)/test_mcab_sample_beam
test_4ch_ulcnet: $(BIN_DIR)/test_4ch_ulcnet
# CSV dumper for the C/Python per-hop alignment parity test
# (tests/test_delay_parity.py builds and runs it); not part of `make test`,
# which only runs self-checking binaries.
dump_delay_parity: $(BIN_DIR)/dump_delay_parity

# Same audit as the mono full stack: no archive may reference the stdio set
# (printf family, stderr, fputc, fwrite). The module's archive carries the module TU.
audit-no-stdio: $(PIPELINE_LIB) $(DFN_PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB) \
                $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(BF_LIB) $(SAMPLE_FLOW_OBJ) $(SAMPLE_BEAM_OBJ)
	@test "$(NO_STDIO)" = 1 || { \
	  echo "FATAL: audit-no-stdio must be invoked with NO_STDIO=1" >&2; \
	  exit 1; \
	}
	@pat='^_?(fprintf|vfprintf|printf|puts|fputs|stderr|__stderrp|fputc|fwrite)$$'; \
	 for archive in "$(PIPELINE_LIB)" "$(DFN_PIPELINE_LIB)" "$(AEC_LIB)" "$(NR_LIB)" "$(COMMON_LIB)" \
	                "$(SPATIAL_COMMON_LIB)" "$(DOA_LIB)" "$(BF_LIB)" "$(SAMPLE_FLOW_OBJ)" "$(SAMPLE_BEAM_OBJ)"; do \
	  hits=$$(nm -A "$$archive" | awk -v re="$$pat" '{ n=$$NF; if (n ~ re) print }'); \
	  if [ -n "$$hits" ]; then echo "FAIL: stdio symbols in $$archive" >&2; echo "$$hits" >&2; exit 1; fi; \
	 done; \
	 echo "4ch pipeline stack: NO_STDIO audit PASS"

FORCE:

$(COMMON_LIB): FORCE
	+$(MAKE) -C $(COMMON_DIR) $(FWD) lib
	@test -f '$@' || { echo "FATAL: common did not produce $@" >&2; exit 1; }

ifneq ($(AEC_LIB),)
$(AEC_LIB): FORCE
	+$(MAKE) -C $(AEC_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB=$(COMMON_LIB) lib
	@test -f '$@' || { echo "FATAL: AEC did not produce $@" >&2; exit 1; }
endif

ifneq ($(NR_LIB),)
$(NR_LIB): FORCE
	+$(MAKE) -C $(NR_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB=$(COMMON_LIB) lib
	@test -f '$@' || { echo "FATAL: NR did not produce $@" >&2; exit 1; }
endif

ifneq ($(SPATIAL_COMMON_LIB),)
$(SPATIAL_COMMON_LIB): FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) libspatial_common.a
	@test -f '$@' || { echo "FATAL: spatial-common did not produce $@" >&2; exit 1; }

$(DOA_LIB): FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) libdoa.a
	@test -f '$@' || { echo "FATAL: DOA did not produce $@" >&2; exit 1; }

$(BF_LIB): FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) libbf.a
	@test -f '$@' || { echo "FATAL: BF did not produce $@" >&2; exit 1; }
endif


# The spatial dependencies (SRP-PHAT, BF, VAD) live in audiolib and test themselves there; this
# target only invokes audiolib's tests.
test-third-party: FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) test

# The module needs only lib/aec (and NR/common for the post stages). FLOW_LIBS adds the sample
# builder of the application-side init (mcab_sample_flow.o); UNIT_LIBS adds the sample beamformer
# helper, which pulls SRP-PHAT and BF. Sample objects first, then the module, DOA, BF, spatial
# kernels, AEC, NR, common.
CORE_LIBS := $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB)
FLOW_LIBS := $(SAMPLE_FLOW_OBJ) $(CORE_LIBS)
UNIT_LIBS := $(SAMPLE_BEAM_OBJ) $(SAMPLE_FLOW_OBJ) $(PIPELINE_LIB) $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(COMMON_LIB)

$(BIN_DIR)/test_mcab: $(OBJ_DIR)/test_mcab.o $(FLOW_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_mcab.o $(FLOW_LIBS) $(LDLIBS)

$(BIN_DIR)/dump_delay_parity: $(OBJ_DIR)/dump_delay_parity.o $(FLOW_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/dump_delay_parity.o $(FLOW_LIBS) $(LDLIBS)

$(BIN_DIR)/test_mcab_sample_beam: $(OBJ_DIR)/test_mcab_sample_beam.o $(UNIT_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_mcab_sample_beam.o $(UNIT_LIBS) $(LDLIBS)

# The NN stage example and the ULCNet framing objects the stage links.
ULCNET_4CH_OBJS := $(OBJ_DIR)/ulcnet_nn_stage.o \
                   $(OBJ_DIR)/ulcnet_process.o

# The stage itself needs no ulcnet_model_io symbol (it only READS the
# descriptor a runtime published, so a board integrating the stage alone
# gains no new link dependency); the test links it to BUILD descriptors for
# the far-input contract gate.
$(BIN_DIR)/test_4ch_ulcnet: $(OBJ_DIR)/test_4ch_ulcnet.o $(ULCNET_4CH_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(UNIT_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_4ch_ulcnet.o $(ULCNET_4CH_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(UNIT_LIBS) $(LDLIBS)

# The adapter drives the pre/post class, so the application links
# ulcnet_prepost.o with it (the test above does not link the adapter).
ULCNET_ADAPTER_OBJS := $(OBJ_DIR)/ulcnet_accelerator_adapter.o \
                       $(OBJ_DIR)/ulcnet_prepost.o \
                       $(OBJ_DIR)/ulcnet_model_io.o

$(BIN_DIR)/4ch_alignulcnet: $(OBJ_DIR)/4ch_alignulcnet_main.o $(OBJ_DIR)/mcab_sample_driver.o $(ULCNET_ADAPTER_OBJS) $(ULCNET_4CH_OBJS) $(UNIT_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/4ch_alignulcnet_main.o $(OBJ_DIR)/mcab_sample_driver.o $(ULCNET_ADAPTER_OBJS) $(ULCNET_4CH_OBJS) $(UNIT_LIBS) $(LDLIBS)

# The 4-channel DFN2 example and its test compose the unit and an AecRes of their own, so they
# link the DFN archive ahead of the unit's archives.
$(BIN_DIR)/4ch_aec_bf_dfn_res: $(OBJ_DIR)/4ch_aec_bf_dfn_res_main.o $(OBJ_DIR)/mcab_sample_driver.o $(DFN_PIPELINE_LIB) $(UNIT_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/4ch_aec_bf_dfn_res_main.o $(OBJ_DIR)/mcab_sample_driver.o $(DFN_PIPELINE_LIB) $(UNIT_LIBS) $(LDLIBS)

$(BIN_DIR)/test_4ch_dfn: $(OBJ_DIR)/test_4ch_dfn.o $(DFN_PIPELINE_LIB) $(UNIT_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_4ch_dfn.o $(DFN_PIPELINE_LIB) $(UNIT_LIBS) $(LDLIBS)

# mcab_external_static spells the application-side init out and needs the module only.
$(BIN_DIR)/mcab_external_static: $(OBJ_DIR)/mcab_external_static.o $(CORE_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/mcab_external_static.o $(CORE_LIBS) $(LDLIBS)

$(BIN_DIR)/mcab_static: $(OBJ_DIR)/mcab_static.o $(OBJ_DIR)/mcab_sample_driver.o $(UNIT_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/mcab_static.o $(OBJ_DIR)/mcab_sample_driver.o $(UNIT_LIBS) $(LDLIBS)

$(BIN_DIR)/mcab_raw: $(OBJ_DIR)/mcab_raw.o $(OBJ_DIR)/mcab_sample_driver.o $(UNIT_LIBS) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/mcab_raw.o $(OBJ_DIR)/mcab_sample_driver.o $(UNIT_LIBS) $(LDLIBS)

# The sample runner has no algorithm-library link dependency.
test-mcab-sample-driver: $(BIN_DIR)/test_mcab_sample_driver
	"$(abspath $(BIN_DIR))/test_mcab_sample_driver"
$(BIN_DIR)/test_mcab_sample_driver: $(OBJ_DIR)/test_mcab_sample_driver.o $(OBJ_DIR)/mcab_sample_driver.o | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_mcab_sample_driver.o $(OBJ_DIR)/mcab_sample_driver.o $(LDLIBS)
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
