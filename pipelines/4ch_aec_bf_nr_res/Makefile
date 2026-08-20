# Four-channel traditional AEC -> spatial -> mono NR/RES pipeline.
#
# This Makefile intentionally lives beside the 4-channel implementation.  The
# parent pipelines/Makefile remains the mono AEC -> NR producer/runner. The
# archive below contains only the reusable four-lane linear core.
# Standard and ULCNet wrappers are linked directly into application binaries;
# AEC, NR, audio_common, DOA, GSC, and spatial kernels remain component libs.
#
# Common invocations:
#   make BACKEND=kiss SIMD=1
#   make BACKEND=ne10 SIMD=1
#   make BACKEND=kiss SIMD=0 test
#   make BACKEND=ne10 SIMD=0 test
#   make audio_pipeline_4ch_raw

CC ?= cc
CXX ?= c++
AR ?= ar
RANLIB ?= ranlib

PIPELINES_DIR := ..
ALG_DIR := ../..
AEC_DIR ?= $(ALG_DIR)/lib/aec/c_impl
NR_DIR ?= $(ALG_DIR)/lib/nr/c_impl
THIRD_PARTY_DIR ?= ../third_party
AC_DIR ?= ../../../audio_common
ULCNET_DIR ?= $(ALG_DIR)/AIAEC/Align_ULCNet

BACKEND = ne10
SIMD ?= 1
WERROR = 0
NO_STDIO = 0
DEBUG = 0
TOOLCHAIN_CHECK = 1
EXTRA_CFLAGS =
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

ifneq ($(words $(BACKEND)),1)
$(error BACKEND must be exactly kiss or ne10, got '$(BACKEND)')
endif
ifneq ($(filter-out kiss ne10,$(BACKEND)),)
$(error BACKEND must be exactly kiss or ne10, got '$(BACKEND)')
endif

# Announce the resolved backend on every build (same MAKECMDGOALS guard as
# audio_common's own banner: print-bin-dir/print-obj-dir/print-lib-path are
# consumed via `$$(make -s ... print-lib-path)` command substitution
# elsewhere, and this must not add stray stdout to that captured value).
ifeq ($(filter print-%,$(MAKECMDGOALS)),)
$(info Audio_ALG/4ch_aec_bf_nr_res: FFT backend = $(BACKEND) -- switch with `make BACKEND=kiss` or `make BACKEND=ne10`)
endif

$(eval $(call require_bool,SIMD))
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
          -I$(NR_DIR)/include -I$(AC_DIR)/include \
          -I$(THIRD_PARTY_DIR)/utility \
          -I$(THIRD_PARTY_DIR)/doa \
          -I$(THIRD_PARTY_DIR)/GSC \
          $(CPPFLAGS) $(EXTRA_CFLAGS) $(SIMD_CFLAGS) \
          -DAUDIO_PIPELINE_BACKEND_STR=\"$(BACKEND)\" \
          $(WERROR_CFLAGS) -ffp-contract=off
LIB_CFLAGS := $(CFLAGS)

LINK := $(CC)
ifeq ($(BACKEND),ne10)
LINK := $(CXX)
endif
LDLIBS := $(EXTRA_LDFLAGS) $(LDFLAGS) -lm

FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) \
      CC='$(CC)' CXX='$(CXX)' AR='$(AR)' RANLIB='$(RANLIB)' \
      EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) \
      TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) DEBUG=$(DEBUG)

# Resolve every producer before constructing this consumer's CFG identity.
ifeq ($(AC_LIB),)
.PHONY: all test libs lib libaudio_pipeline_4ch.a \
        audio_pipeline_4ch_raw 4aec_nr_res_static audio_pipeline_4ch_static 4ch_alignulcnet \
        test_4aec_nr_res test_audio_pipeline_4ch \
        test_audio_pipeline_4ch_ulcnet dump_delay_parity \
        print-bin-dir print-obj-dir print-lib-path audit-no-stdio \
        test-third-party
all test libs lib libaudio_pipeline_4ch.a \
audio_pipeline_4ch_raw 4aec_nr_res_static audio_pipeline_4ch_static 4ch_alignulcnet \
test_4aec_nr_res test_audio_pipeline_4ch \
test_audio_pipeline_4ch_ulcnet dump_delay_parity \
print-bin-dir print-obj-dir print-lib-path audit-no-stdio \
test-third-party:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) $(FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: cannot resolve audio_common archive" >&2; exit 1; }; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 test -n "$$aec" || { echo "FATAL: cannot resolve AEC archive" >&2; exit 1; }; \
	 nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 test -n "$$nr" || { echo "FATAL: cannot resolve NR archive" >&2; exit 1; }; \
	 spatial="$$($(MAKE) -s --no-print-directory -C $(THIRD_PARTY_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) print-spatial-common-lib-path)"; \
	 doa="$$($(MAKE) -s --no-print-directory -C $(THIRD_PARTY_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) print-doa-lib-path)"; \
	 gsc="$$($(MAKE) -s --no-print-directory -C $(THIRD_PARTY_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) print-gsc-lib-path)"; \
	 test -n "$$spatial" || { echo "FATAL: cannot resolve spatial-common archive" >&2; exit 1; }; \
	 test -n "$$doa" || { echo "FATAL: cannot resolve DOA archive" >&2; exit 1; }; \
	 test -n "$$gsc" || { echo "FATAL: cannot resolve GSC archive" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory $@ AC_LIB="$$ac" AEC_LIB="$$aec" NR_LIB="$$nr" \
	   SPATIAL_COMMON_LIB="$$spatial" DOA_LIB="$$doa" GSC_LIB="$$gsc"
else
ifneq ($(words $(AC_LIB) $(AEC_LIB) $(NR_LIB) $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(GSC_LIB)),6)
$(error AC_LIB, AEC_LIB, NR_LIB, SPATIAL_COMMON_LIB, DOA_LIB, and GSC_LIB must be supplied together)
endif
AC_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(AC_LIB))))
AEC_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(AEC_LIB))))
NR_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(NR_LIB))))
SPATIAL_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(SPATIAL_COMMON_LIB))))
endif

SRCS := 4aec_nr_res.c audio_pipeline_4ch.c \
        ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.c \
        audio_pipeline_4ch_raw.c \
        4aec_nr_res_static.c audio_pipeline_4ch_static.c \
        tests/test_4aec_nr_res.c \
        tests/test_audio_pipeline_4ch.c tests/dump_delay_parity.c \
        ../4ch_alignulcnet/tests/test_audio_pipeline_4ch_ulcnet.c \
        $(ULCNET_DIR)/ulcnet_accelerator_adapter.c \
        ../4ch_alignulcnet/main.c $(ULCNET_DIR)/ulcnet_process.c \
        $(ULCNET_DIR)/ulcnet_model_io.c
CFG_SIG_PAYLOAD := CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) LDFLAGS=$(LDLIBS) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) AC_PRODUCER_SIG=$(AC_PRODUCER_SIG) AEC_PRODUCER_SIG=$(AEC_PRODUCER_SIG) NR_PRODUCER_SIG=$(NR_PRODUCER_SIG) SPATIAL_PRODUCER_SIG=$(SPATIAL_PRODUCER_SIG) SRCS=$(sort $(SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR := $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR := $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)
PIPELINE_LIB := $(BIN_DIR)/libaudio_pipeline_4ch.a

# Only the reusable 4-lane linear core is archived. Standard and ULCNet
# application wrappers are linked as objects into their respective binaries.
PIPELINE_OBJS := $(OBJ_DIR)/4aec_nr_res.o

.DEFAULT_GOAL := all
.PHONY: _cfg_guard FORCE clean clean-all

_cfg_guard:
	@if [ "$(BACKEND)" = ne10 ] && [ "$(TOOLCHAIN_CHECK)" = 1 ]; then \
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

$(OBJ_DIR)/4aec_nr_res.o: 4aec_nr_res.c 4aec_nr_res.h 4aec_nr_res_internal.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/audio_pipeline_4ch.o: audio_pipeline_4ch.c audio_pipeline_4ch.h 4aec_nr_res.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/audio_pipeline_4ch_ulcnet.o: ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.c ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.h audio_pipeline_4ch.h 4aec_nr_res.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_process.o: $(ULCNET_DIR)/ulcnet_process.c $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_model_io.o: $(ULCNET_DIR)/ulcnet_model_io.c $(ULCNET_DIR)/ulcnet_model_io.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_accelerator_adapter.o: $(ULCNET_DIR)/ulcnet_accelerator_adapter.c $(ULCNET_DIR)/ulcnet_accelerator_adapter.h $(ULCNET_DIR)/ulcnet_model_io.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/4ch_alignulcnet_main.o: ../4ch_alignulcnet/main.c ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.h $(ULCNET_DIR)/ulcnet_accelerator_adapter.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/audio_pipeline_4ch_raw.o: audio_pipeline_4ch_raw.c audio_pipeline_4ch.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/4aec_nr_res_static.o: 4aec_nr_res_static.c 4aec_nr_res.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/audio_pipeline_4ch_static.o: audio_pipeline_4ch_static.c audio_pipeline_4ch.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_4aec_nr_res.o: tests/test_4aec_nr_res.c 4aec_nr_res.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_audio_pipeline_4ch.o: tests/test_audio_pipeline_4ch.c audio_pipeline_4ch.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/dump_delay_parity.o: tests/dump_delay_parity.c 4aec_nr_res.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_audio_pipeline_4ch_ulcnet.o: ../4ch_alignulcnet/tests/test_audio_pipeline_4ch_ulcnet.c ../4ch_alignulcnet/audio_pipeline_4ch_ulcnet.h audio_pipeline_4ch.h 4aec_nr_res.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(PIPELINE_LIB): $(PIPELINE_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && \
	  $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "libaudio_pipeline_4ch.a [$(BACKEND), SIMD=$(SIMD)] -> $@"

ifneq ($(AC_LIB),)
.PHONY: all test libs lib libaudio_pipeline_4ch.a \
        audio_pipeline_4ch_raw 4aec_nr_res_static audio_pipeline_4ch_static 4ch_alignulcnet \
        test_4aec_nr_res test_audio_pipeline_4ch \
        test_audio_pipeline_4ch_ulcnet dump_delay_parity \
        audit-no-stdio test-third-party

all: $(PIPELINE_LIB) $(BIN_DIR)/audio_pipeline_4ch_raw \
     $(BIN_DIR)/4aec_nr_res_static $(BIN_DIR)/audio_pipeline_4ch_static

# The 4ch_alignulcnet application shares this Makefile (its wrapper compiles
# against this directory's core), so its acceptance test and board skeleton
# run here too: this directory's `make test` is the one-stop gate, mirroring
# the mono arrangement in pipelines/Makefile.
test: test-third-party \
      $(BIN_DIR)/test_4aec_nr_res \
      $(BIN_DIR)/test_audio_pipeline_4ch $(BIN_DIR)/4aec_nr_res_static \
      $(BIN_DIR)/audio_pipeline_4ch_static \
      $(BIN_DIR)/test_audio_pipeline_4ch_ulcnet $(BIN_DIR)/4ch_alignulcnet
	"$(abspath $(BIN_DIR))/test_4aec_nr_res"
	"$(abspath $(BIN_DIR))/test_audio_pipeline_4ch"
	"$(abspath $(BIN_DIR))/test_audio_pipeline_4ch_ulcnet"
	"$(abspath $(BIN_DIR))/4aec_nr_res_static" --sample-rate 16000
	"$(abspath $(BIN_DIR))/4aec_nr_res_static" --sample-rate 48000
	"$(abspath $(BIN_DIR))/audio_pipeline_4ch_static" --sample-rate 16000
	"$(abspath $(BIN_DIR))/audio_pipeline_4ch_static" --sample-rate 48000
	"$(abspath $(BIN_DIR))/4ch_alignulcnet"
	"$(abspath $(BIN_DIR))/4ch_alignulcnet" --delay-num-filters 1

lib libaudio_pipeline_4ch.a: $(PIPELINE_LIB)
libs: $(AC_LIB) $(AEC_LIB) $(NR_LIB) $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(GSC_LIB) $(PIPELINE_LIB)
audio_pipeline_4ch_raw: $(BIN_DIR)/audio_pipeline_4ch_raw
4aec_nr_res_static: $(BIN_DIR)/4aec_nr_res_static
audio_pipeline_4ch_static: $(BIN_DIR)/audio_pipeline_4ch_static
4ch_alignulcnet: $(BIN_DIR)/4ch_alignulcnet
test_4aec_nr_res: $(BIN_DIR)/test_4aec_nr_res
test_audio_pipeline_4ch: $(BIN_DIR)/test_audio_pipeline_4ch
test_audio_pipeline_4ch_ulcnet: $(BIN_DIR)/test_audio_pipeline_4ch_ulcnet
# CSV dumper for the C/Python per-hop alignment parity test
# (tests/test_delay_parity.py builds and runs it); not part of `make test`,
# which only runs self-checking binaries.
dump_delay_parity: $(BIN_DIR)/dump_delay_parity

# Match the established mono full-stack audit tiers. NE10 is the board
# deliverable and must be clear of the extended S2 set. Vendored KISS retains
# compiler-lowered stderr/fputc/fwrite references in its allocation-failure
# logger, so the host/reference backend is checked against S1 only.
audit-no-stdio: $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) \
                $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(GSC_LIB)
	@test "$(NO_STDIO)" = 1 || { \
	  echo "FATAL: audit-no-stdio must be invoked with NO_STDIO=1" >&2; \
	  exit 1; \
	}
	@s1='^_?(fprintf|vfprintf|printf|puts|fputs)$$'; \
	 s2='^_?(fprintf|vfprintf|printf|puts|fputs|stderr|__stderrp|fputc|fwrite)$$'; \
	 pat="$$s1"; [ "$(BACKEND)" = ne10 ] && pat="$$s2"; \
	 for archive in "$(PIPELINE_LIB)" "$(AEC_LIB)" "$(NR_LIB)" "$(AC_LIB)" \
	                "$(SPATIAL_COMMON_LIB)" "$(DOA_LIB)" "$(GSC_LIB)"; do \
	  hits=$$(nm -A "$$archive" | awk -v re="$$pat" '{ n=$$NF; if (n ~ re) print }'); \
	  if [ -n "$$hits" ]; then echo "FAIL: stdio symbols in $$archive" >&2; echo "$$hits" >&2; exit 1; fi; \
	 done; \
	 echo "4ch pipeline stack: NO_STDIO audit PASS ($$pat tier)"

FORCE:

$(AC_LIB): FORCE
	+$(MAKE) -C $(AC_DIR) $(FWD) lib
	@test -f '$@' || { echo "FATAL: audio_common did not produce $@" >&2; exit 1; }

$(AEC_LIB): FORCE
	+$(MAKE) -C $(AEC_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB=$(AC_LIB) lib
	@test -f '$@' || { echo "FATAL: AEC did not produce $@" >&2; exit 1; }

$(NR_LIB): FORCE
	+$(MAKE) -C $(NR_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB=$(AC_LIB) lib
	@test -f '$@' || { echo "FATAL: NR did not produce $@" >&2; exit 1; }

$(SPATIAL_COMMON_LIB): FORCE
	+$(MAKE) -C $(THIRD_PARTY_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) libspatial_common.a
	@test -f '$@' || { echo "FATAL: spatial-common did not produce $@" >&2; exit 1; }

$(DOA_LIB): FORCE
	+$(MAKE) -C $(THIRD_PARTY_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) libdoa.a
	@test -f '$@' || { echo "FATAL: DOA did not produce $@" >&2; exit 1; }

$(GSC_LIB): FORCE
	+$(MAKE) -C $(THIRD_PARTY_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) libgsc.a
	@test -f '$@' || { echo "FATAL: GSC did not produce $@" >&2; exit 1; }

$(BIN_DIR)/test_4aec_nr_res: $(OBJ_DIR)/test_4aec_nr_res.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_4aec_nr_res.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDLIBS)

$(BIN_DIR)/dump_delay_parity: $(OBJ_DIR)/dump_delay_parity.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/dump_delay_parity.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDLIBS)

# The spatial dependencies test themselves now: their sources, their
# GSC_TESTING build and their deployable-archive audit all live in
# third_party/, so this directory only has to invoke them.
test-third-party: FORCE
	+$(MAKE) -C $(THIRD_PARTY_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) test

$(BIN_DIR)/test_audio_pipeline_4ch: $(OBJ_DIR)/test_audio_pipeline_4ch.o $(OBJ_DIR)/audio_pipeline_4ch.o $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_audio_pipeline_4ch.o $(OBJ_DIR)/audio_pipeline_4ch.o $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDLIBS)

ULCNET_4CH_OBJS := $(OBJ_DIR)/audio_pipeline_4ch_ulcnet.o \
                   $(OBJ_DIR)/audio_pipeline_4ch.o \
                   $(OBJ_DIR)/ulcnet_process.o

# The wrapper itself needs no ulcnet_model_io symbol (it only READS the
# descriptor a runtime published, so a board integrating the wrapper alone
# gains no new link dependency); the test links it to BUILD descriptors for
# the far-input contract gate.
$(BIN_DIR)/test_audio_pipeline_4ch_ulcnet: $(OBJ_DIR)/test_audio_pipeline_4ch_ulcnet.o $(ULCNET_4CH_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_audio_pipeline_4ch_ulcnet.o $(ULCNET_4CH_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDLIBS)

$(BIN_DIR)/4ch_alignulcnet: $(OBJ_DIR)/4ch_alignulcnet_main.o $(OBJ_DIR)/ulcnet_accelerator_adapter.o $(OBJ_DIR)/ulcnet_model_io.o $(ULCNET_4CH_OBJS) $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/4ch_alignulcnet_main.o $(OBJ_DIR)/ulcnet_accelerator_adapter.o $(OBJ_DIR)/ulcnet_model_io.o $(ULCNET_4CH_OBJS) $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDLIBS)

$(BIN_DIR)/4aec_nr_res_static: $(OBJ_DIR)/4aec_nr_res_static.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/4aec_nr_res_static.o $(PIPELINE_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDLIBS)

$(BIN_DIR)/audio_pipeline_4ch_static: $(OBJ_DIR)/audio_pipeline_4ch_static.o $(OBJ_DIR)/audio_pipeline_4ch.o $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/audio_pipeline_4ch_static.o $(OBJ_DIR)/audio_pipeline_4ch.o $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDLIBS)

$(BIN_DIR)/audio_pipeline_4ch_raw: $(OBJ_DIR)/audio_pipeline_4ch_raw.o $(OBJ_DIR)/audio_pipeline_4ch.o $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/audio_pipeline_4ch_raw.o $(OBJ_DIR)/audio_pipeline_4ch.o $(PIPELINE_LIB) $(DOA_LIB) $(GSC_LIB) $(SPATIAL_COMMON_LIB) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDLIBS)
endif

ifeq ($(AC_LIB),)
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
	$(MAKE) -C $(THIRD_PARTY_DIR) clean
	$(MAKE) -C $(AEC_DIR) clean
	$(MAKE) -C $(NR_DIR) clean
	$(MAKE) -C $(AC_DIR) clean

-include $(wildcard $(OBJ_DIR)/*.d)
