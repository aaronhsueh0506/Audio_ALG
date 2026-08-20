# Reusable spatial DSP and gain-staging libraries used by Audio_ALG pipelines.
#
#   make BACKEND=kiss|ne10 SIMD=0|1
#
# BACKEND follows audio_common/AEC/NR's build identity even though these
# frequency-domain algorithms do not execute an FFT themselves.  Their public
# API uses audio_common's backend-neutral Complex type, so the same archives'
# headers work with either fft_wrapper implementation.  SIMD selects the
# shared scalar/NEON spatial kernels independently of the FFT backend.

CC ?= cc
CXX ?= c++
AR ?= ar
RANLIB ?= ranlib

AC_DIR ?= ../../../audio_common
AC_DIR_ABS := $(abspath $(AC_DIR))
BACKEND = ne10
SIMD ?= 1
WERROR = 0
NO_STDIO = 0
DEBUG = 0
TOOLCHAIN_CHECK = 1
EXTRA_CFLAGS =
EXTRA_LDFLAGS =
CPPFLAGS +=
CXXFLAGS +=

ifneq ($(words $(BACKEND)),1)
$(error BACKEND must be exactly kiss or ne10, got '$(BACKEND)')
endif
ifneq ($(filter-out kiss ne10,$(BACKEND)),)
$(error BACKEND must be exactly kiss or ne10, got '$(BACKEND)')
endif

define require_bool
ifneq ($$(words $$($(1))),1)
$$(error $(1) must be exactly 0 or 1, got '$$($(1))')
endif
ifneq ($$(filter-out 0 1,$$($(1))),)
$$(error $(1) must be exactly 0 or 1, got '$$($(1))')
endif
endef
$(eval $(call require_bool,SIMD))
$(eval $(call require_bool,WERROR))
$(eval $(call require_bool,NO_STDIO))
$(eval $(call require_bool,DEBUG))
$(eval $(call require_bool,TOOLCHAIN_CHECK))

ifneq ($(findstring ;,$(EXTRA_CFLAGS) $(EXTRA_LDFLAGS)),)
$(error EXTRA_CFLAGS/EXTRA_LDFLAGS must not contain ';')
endif
ifneq ($(filter -Ofast -ffast-math -funsafe-math-optimizations -ffp-contract=%,$(CFLAGS) $(CPPFLAGS) $(EXTRA_CFLAGS)),)
$(error fast-math and caller-supplied -ffp-contract flags are unsupported)
endif

ifeq ($(SIMD),0)
SIMD_CFLAGS := -DSIMD_KERNELS_FORCE_SCALAR
else
SIMD_CFLAGS :=
endif

ifeq ($(DEBUG),1)
OPT_CFLAGS := -O0 -g
else
OPT_CFLAGS := -O2
endif

ifeq ($(WERROR),1)
WERROR_CFLAGS := -Werror
else
WERROR_CFLAGS :=
endif

override CFLAGS := -Wall -Wextra $(OPT_CFLAGS) -std=gnu99 \
                   -Iutility -Idoa -IGSC -IVAD -Igain -I$(AC_DIR_ABS)/include \
                   $(CPPFLAGS) $(EXTRA_CFLAGS) $(SIMD_CFLAGS) \
                   $(WERROR_CFLAGS) -ffp-contract=off

SRCS = utility/complex.c utility/spatial_simd.c \
       doa/doa_smoother.c doa/steering.c doa/srp.c GSC/gsc.c \
       VAD/masker.c VAD/vad.c VAD/vad_api.c \
       gain/fix_gain.c gain/nr_gain.c gain/post_gain.c
CFG_SIG_PAYLOAD := CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) SRCS=$(sort $(SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR = $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

SPATIAL_COMMON_LIB = $(BIN_DIR)/libspatial_common.a
DOA_LIB = $(BIN_DIR)/libdoa.a
GSC_LIB = $(BIN_DIR)/libgsc.a
VAD_LIB = $(BIN_DIR)/libvad.a
GAIN_LIB = $(BIN_DIR)/libgain.a

# Every deployable archive, in one place: `all`, `libs` and the NO_STDIO
# audit all read this, so a new archive cannot be added to the build while
# quietly escaping the gate.
ARCHIVES = $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(GSC_LIB) $(VAD_LIB) \
           $(GAIN_LIB)

# The archives above need only audio_common's HEADERS. The test BINARY links,
# so it needs the archive itself.
#
# ⚠ audio_common keys its output directory on FIFTEEN variables, not just
# BACKEND/SIMD -- DEBUG=1 and NO_STDIO=1 each land in a different hash dir
# (measured). Forwarding a subset would resolve, and link, the WRONG archive
# with no diagnostic. So every knob this Makefile accepts is forwarded, and
# AC_LIB is an input a caller can supply directly (the 4ch Makefile already
# has the resolved path in hand and passes it).
# Same set the 4ch Makefile forwards to its producers. CFLAGS/CXXFLAGS are
# deliberately absent: audio_common refuses them from the command line because
# overriding would drop its own FP-policy and backend appends.
AC_FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) \
         CC='$(CC)' CXX='$(CXX)' AR='$(AR)' RANLIB='$(RANLIB)' \
         EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) \
         TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) DEBUG=$(DEBUG)
AC_LIB ?= $(shell $(MAKE) -s --no-print-directory -C $(AC_DIR_ABS) $(AC_FWD) print-lib-path)
ifeq ($(strip $(AC_LIB)),)
$(error cannot resolve the audio_common archive; pass AC_LIB= explicitly)
endif

# print-lib-path only ECHOES; nothing builds the archive. Without this rule the
# test link would reference a file no target ever asked anyone to produce.
$(AC_LIB): FORCE
	+$(MAKE) -C $(AC_DIR_ABS) $(AC_FWD) lib
	@test -f '$@' || { echo "FATAL: audio_common did not produce $@" >&2; exit 1; }

SPATIAL_COMMON_OBJS = $(OBJ_DIR)/complex.o $(OBJ_DIR)/spatial_simd.o
DOA_OBJS = $(OBJ_DIR)/doa_smoother.o $(OBJ_DIR)/steering.o $(OBJ_DIR)/srp.o
GSC_OBJS = $(OBJ_DIR)/gsc.o
VAD_OBJS = $(OBJ_DIR)/masker.o $(OBJ_DIR)/vad.o $(OBJ_DIR)/vad_api.o
GAIN_OBJS = $(OBJ_DIR)/fix_gain.o $(OBJ_DIR)/nr_gain.o \
            $(OBJ_DIR)/post_gain.o

.DEFAULT_GOAL := all
.PHONY: all libs lib libspatial_common.a libdoa.a libgsc.a libvad.a \
        libgain.a clean \
        test test_spatial_third_party audit-no-gsc-test-symbols \
        print-bin-dir print-obj-dir print-spatial-common-lib-path \
        print-doa-lib-path print-gsc-lib-path print-vad-lib-path \
        print-gain-lib-path \
        audit-no-stdio _cfg_guard

all libs lib: $(ARCHIVES)
libspatial_common.a: $(SPATIAL_COMMON_LIB)
libdoa.a: $(DOA_LIB)
libgsc.a: $(GSC_LIB)
libvad.a: $(VAD_LIB)
libgain.a: $(GAIN_LIB)

FORCE:
.PHONY: FORCE

_cfg_guard:
	@if [ "$(BACKEND)" = "ne10" ] && [ "$(TOOLCHAIN_CHECK)" = "1" ]; then \
	  cc_m="$$($(CC) -dumpmachine 2>/dev/null)"; cxx_m="$$($(CXX) -dumpmachine 2>/dev/null)"; \
	  if [ -z "$$cc_m" ] || [ "$$cc_m" != "$$cxx_m" ]; then \
	    echo "FATAL: CC/CXX target mismatch (CC=$$cc_m, CXX=$$cxx_m)" >&2; exit 1; \
	  fi; \
	fi
	@mkdir -p "$(OBJ_DIR)" "$(BIN_DIR)"
	@if [ -f "$(OBJ_DIR)/config.manifest" ]; then \
	  test "$$(cat "$(OBJ_DIR)/config.manifest")" = "$(CFG_SIG_PAYLOAD)" || { \
	    echo "FATAL: spatial CFG_SIG collision in $(OBJ_DIR)" >&2; exit 1; }; \
	else \
	  tmp="$(OBJ_DIR)/.config.manifest.tmp.$$$$"; \
	  printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$tmp"; \
	  mv -f "$$tmp" "$(OBJ_DIR)/config.manifest"; \
	fi

$(OBJ_DIR)/complex.o: utility/complex.c utility/complex.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/spatial_simd.o: utility/spatial_simd.c utility/spatial_simd.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/doa_smoother.o: doa/doa_smoother.c doa/doa_smoother.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/steering.o: doa/steering.c doa/steering.h doa/srp.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/srp.o: doa/srp.c doa/srp.h utility/complex.h utility/spatial_simd.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/masker.o: VAD/masker.c VAD/masker.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/vad.o: VAD/vad.c VAD/vad.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/vad_api.o: VAD/vad_api.c VAD/vad_api.h VAD/masker.h VAD/vad.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/gsc.o: GSC/gsc.c GSC/gsc.h utility/complex.h utility/spatial_simd.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/fix_gain.o: gain/fix_gain.c gain/fix_gain.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/nr_gain.o: gain/nr_gain.c gain/nr_gain.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/post_gain.o: gain/post_gain.c gain/post_gain.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_spatial_third_party.o: tests/test_spatial_third_party.c \
        GSC/gsc.h GSC/gsc_test_hooks.h VAD/vad_api.h gain/fix_gain.h \
        gain/nr_gain.h gain/post_gain.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# The GSC_TESTING build of gsc.c, compiled ONLY here. audit-no-gsc-test-symbols
# below proves the deployable archive never picks it up.
$(OBJ_DIR)/gsc_test.o: GSC/gsc.c GSC/gsc.h utility/complex.h \
        utility/spatial_simd.h | _cfg_guard
	$(CC) $(CFLAGS) -DGSC_TESTING=1 -MD -MP -c -o $@ $<

$(BIN_DIR)/test_spatial_third_party: $(OBJ_DIR)/test_spatial_third_party.o \
        $(OBJ_DIR)/gsc_test.o $(DOA_LIB) $(SPATIAL_COMMON_LIB) $(VAD_LIB) \
        $(GAIN_LIB) $(AC_LIB) | _cfg_guard
	$(CC) $(CFLAGS) -o $@ $(OBJ_DIR)/test_spatial_third_party.o \
		$(OBJ_DIR)/gsc_test.o $(DOA_LIB) $(SPATIAL_COMMON_LIB) $(VAD_LIB) \
		$(GAIN_LIB) $(AC_LIB) -lm

audit-no-gsc-test-symbols: $(GSC_LIB)
	@if nm "$(GSC_LIB)" | grep -q 'gsc_test_'; then \
	  echo "FAIL: test-only symbol present in deployable $(GSC_LIB)" >&2; \
	  exit 1; \
	fi
	@echo "audit-no-gsc-test-symbols: clean"

test: audit-no-gsc-test-symbols $(BIN_DIR)/test_spatial_third_party
	"$(abspath $(BIN_DIR))/test_spatial_third_party"

test_spatial_third_party: $(BIN_DIR)/test_spatial_third_party

$(SPATIAL_COMMON_LIB): $(SPATIAL_COMMON_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "spatial-common [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(DOA_LIB): $(DOA_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "doa [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(GSC_LIB): $(GSC_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "gsc [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(VAD_LIB): $(VAD_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "vad [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(GAIN_LIB): $(GAIN_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "gain [$(BACKEND), SIMD=$(SIMD)] -> $@"

print-bin-dir:
	@echo $(abspath $(BIN_DIR))
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))
print-spatial-common-lib-path:
	@echo $(abspath $(SPATIAL_COMMON_LIB))
print-doa-lib-path:
	@echo $(abspath $(DOA_LIB))
print-gsc-lib-path:
	@echo $(abspath $(GSC_LIB))
print-vad-lib-path:
	@echo $(abspath $(VAD_LIB))
print-gain-lib-path:
	@echo $(abspath $(GAIN_LIB))

audit-no-stdio: libs
	@for lib in $(ARCHIVES); do \
	  hits=$$(nm -A "$$lib" | awk '{ n=$$NF; if (n ~ /^_?(fprintf|vfprintf|printf|puts|fputs|stderr|__stderrp|fputc|fwrite)$$/) print }'); \
	  if [ -n "$$hits" ]; then echo "FAIL: stdio symbols in $$lib"; echo "$$hits"; exit 1; fi; \
	done
	@echo "spatial libraries: NO_STDIO audit PASS"

clean:
	rm -rf "$(OBJ_ROOT)" "$(BIN_ROOT)"

-include $(wildcard $(OBJ_DIR)/*.d)
