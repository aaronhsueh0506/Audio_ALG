# Audio_ALG/pipelines/samples/Makefile
#
# Reference examples only -- see each source file's own top-of-file comment
# (e.g. mcab_demo.c) for what it demonstrates and what production machinery
# it deliberately omits. Nothing built here is linked into any of the three
# production pipeline targets (4ch_aec_bf_nr_res, 4ch_aec_bf_dfn_res,
# 4ch_alignulcnet); this directory is a sibling, not a dependency, of theirs.
#
# Same two-phase recursive-make producer dispatch as the other pipeline
# Makefiles in this repo (see pipelines/Makefile and
# 4ch_aec_bf_nr_res/Makefile for the long-form explanation), trimmed to the
# three producers this directory's demos actually need: audiolib's own
# libaudio_common.a, lib/aec's libaec.a, and audiolib's
# libspatial_common.a/libdoa.a/libbf.a.
#
# Usage:
#   make                  # build mcab_demo
#   make SIMD=0           # scalar spatial/SIMD kernels
#   make test             # build + run mcab_demo (self-checking: nonzero
#                          # exit on any NaN/degenerate output)
#   make clean            # remove this dir's own obj/bin trees

CC ?= cc
AR ?= ar
RANLIB ?= ranlib

ALG_DIR := ../..
AEC_DIR ?= $(ALG_DIR)/lib/aec/c_impl
AUDIOLIB_DIR ?= ../../../audiolib
COMMON_DIR ?= ../../../audiolib/common

AUDIO_PIPELINE_EMPTY :=
AUDIO_PIPELINE_SPACE := $(AUDIO_PIPELINE_EMPTY) $(AUDIO_PIPELINE_EMPTY)
ifneq ($(findstring $(AUDIO_PIPELINE_SPACE),$(CURDIR)),)
$(error this build root path contains whitespace ('$(CURDIR)'), which is not supported -- re-run from a path with no spaces.)
endif

BACKEND = ne10
SIMD ?= 1
WERROR = 0
DEBUG = 0
TOOLCHAIN_CHECK = 1
EXTRA_CFLAGS =
EXTRA_LDFLAGS =
CPPFLAGS ?=
LDFLAGS ?=
NO_STDIO = 0

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

ifeq ($(filter print-%,$(MAKECMDGOALS)),)
$(info Audio_ALG/pipelines/samples: FFT backend = $(BACKEND))
endif

$(eval $(call require_bool,SIMD))
$(eval $(call require_bool,WERROR))
$(eval $(call require_bool,DEBUG))
$(eval $(call require_bool,TOOLCHAIN_CHECK))

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
          -I. -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
          -I$(COMMON_DIR)/include -I$(COMMON_DIR) \
          -I$(AUDIOLIB_DIR)/common/utility \
          -I$(AUDIOLIB_DIR)/doa -I$(AUDIOLIB_DIR)/bf \
          $(CPPFLAGS) $(EXTRA_CFLAGS) $(SIMD_CFLAGS) \
          $(WERROR_CFLAGS) -ffp-contract=off -fno-math-errno
LDLIBS := $(EXTRA_LDFLAGS) $(LDFLAGS) -lm

FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) \
      CC='$(CC)' AR='$(AR)' RANLIB='$(RANLIB)' \
      EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) \
      TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) DEBUG=$(DEBUG)

ifeq ($(COMMON_LIB),)
.PHONY: all test libs lib mcab_demo print-bin-dir print-obj-dir
all test libs lib mcab_demo print-bin-dir print-obj-dir:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: cannot resolve common archive" >&2; exit 1; }; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$$ac" print-lib-path)"; \
	 test -n "$$aec" || { echo "FATAL: cannot resolve AEC archive" >&2; exit 1; }; \
	 spatial="$$($(MAKE) -s --no-print-directory -C $(AUDIOLIB_DIR) $(FWD) print-spatial-common-lib-path)"; \
	 doa="$$($(MAKE) -s --no-print-directory -C $(AUDIOLIB_DIR) $(FWD) print-doa-lib-path)"; \
	 bf="$$($(MAKE) -s --no-print-directory -C $(AUDIOLIB_DIR) $(FWD) print-bf-lib-path)"; \
	 test -n "$$spatial" || { echo "FATAL: cannot resolve spatial-common archive" >&2; exit 1; }; \
	 test -n "$$doa" || { echo "FATAL: cannot resolve DOA archive" >&2; exit 1; }; \
	 test -n "$$bf" || { echo "FATAL: cannot resolve BF archive" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory $@ COMMON_LIB="$$ac" AEC_LIB="$$aec" \
	   SPATIAL_COMMON_LIB="$$spatial" DOA_LIB="$$doa" BF_LIB="$$bf"
else
ifneq ($(words $(COMMON_LIB) $(AEC_LIB) $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(BF_LIB)),5)
$(error COMMON_LIB, AEC_LIB, SPATIAL_COMMON_LIB, DOA_LIB, and BF_LIB must be supplied together)
endif
COMMON_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(COMMON_LIB))))
AEC_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(AEC_LIB))))
SPATIAL_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(SPATIAL_COMMON_LIB))))
endif

# The same CC command can select a different SDK through PATH.
# Include effective pointer/FP/ISA predefines to avoid cross-ABI cache reuse.
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) AR=$(AR) RANLIB=$(RANLIB) CFLAGS=$(CFLAGS) \
  LDFLAGS=$(LDLIBS) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) \
  DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) \
  COMMON_PRODUCER_SIG=$(COMMON_PRODUCER_SIG) AEC_PRODUCER_SIG=$(AEC_PRODUCER_SIG) \
  SPATIAL_PRODUCER_SIG=$(SPATIAL_PRODUCER_SIG)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR := $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR := $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

.DEFAULT_GOAL := all
.PHONY: _cfg_guard FORCE clean

FORCE:

$(COMMON_LIB): FORCE
	+$(MAKE) -C $(COMMON_DIR) $(FWD) lib
	@test -f '$@' || { echo "FATAL: common did not produce $@" >&2; exit 1; }

$(AEC_LIB): FORCE
	+$(MAKE) -C $(AEC_DIR) $(FWD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB=$(COMMON_LIB) lib
	@test -f '$@' || { echo "FATAL: AEC did not produce $@" >&2; exit 1; }

$(SPATIAL_COMMON_LIB): FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) libspatial_common.a
	@test -f '$@' || { echo "FATAL: spatial-common did not produce $@" >&2; exit 1; }

$(DOA_LIB): FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) libdoa.a
	@test -f '$@' || { echo "FATAL: DOA did not produce $@" >&2; exit 1; }

$(BF_LIB): FORCE
	+$(MAKE) -C $(AUDIOLIB_DIR) $(FWD) libbf.a
	@test -f '$@' || { echo "FATAL: BF did not produce $@" >&2; exit 1; }

_cfg_guard:
	@test -n "$(CC_TARGET_SIG)" && test "$(CC_TARGET_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine compiler target; CC preprocessing with the configured flags must succeed." >&2; \
	  exit 1; }
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

$(OBJ_DIR)/mcab_demo.o: mcab_demo.c | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

ifneq ($(COMMON_LIB),)
.PHONY: all test libs lib mcab_demo

$(BIN_DIR)/mcab_demo: $(OBJ_DIR)/mcab_demo.o $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(COMMON_LIB) | _cfg_guard
	$(CC) -o $@ $(OBJ_DIR)/mcab_demo.o $(DOA_LIB) $(BF_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(COMMON_LIB) $(LDLIBS)

all: $(BIN_DIR)/mcab_demo

mcab_demo: $(BIN_DIR)/mcab_demo

# Self-checking: mcab_demo itself exits nonzero on any NaN/zero/degenerate
# result (see mcab_demo.c's own isfinite()/final-summary checks), so running
# it IS the test.
test: $(BIN_DIR)/mcab_demo
	"$(abspath $(BIN_DIR))/mcab_demo"

libs lib: $(COMMON_LIB) $(AEC_LIB) $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(BF_LIB)

print-bin-dir:
	@echo "$(BIN_DIR)"
print-obj-dir:
	@echo "$(OBJ_DIR)"
endif

clean:
	rm -rf obj bin
