# Reusable spatial DSP libraries used by Audio_ALG pipelines.
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

AC_DIR ?= ../../../../audio_common
AC_DIR_ABS := $(abspath $(AC_DIR))
BACKEND = kiss
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
                   -Iutility -Idoa -IGSC -I$(AC_DIR_ABS)/include \
                   $(CPPFLAGS) $(EXTRA_CFLAGS) $(SIMD_CFLAGS) \
                   $(WERROR_CFLAGS) -ffp-contract=off

SRCS = utility/complex.c utility/spatial_simd.c \
       doa/doa_smoother.c doa/steering.c doa/srp.c GSC/gsc.c
CFG_SIG_PAYLOAD := CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) SRCS=$(sort $(SRCS))
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR = $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

SPATIAL_COMMON_LIB = $(BIN_DIR)/libspatial_common.a
DOA_LIB = $(BIN_DIR)/libdoa.a
GSC_LIB = $(BIN_DIR)/libgsc.a

SPATIAL_COMMON_OBJS = $(OBJ_DIR)/complex.o $(OBJ_DIR)/spatial_simd.o
DOA_OBJS = $(OBJ_DIR)/doa_smoother.o $(OBJ_DIR)/steering.o $(OBJ_DIR)/srp.o
GSC_OBJS = $(OBJ_DIR)/gsc.o

.DEFAULT_GOAL := all
.PHONY: all libs lib libspatial_common.a libdoa.a libgsc.a clean \
        print-bin-dir print-obj-dir print-spatial-common-lib-path \
        print-doa-lib-path print-gsc-lib-path audit-no-stdio _cfg_guard

all libs lib: $(SPATIAL_COMMON_LIB) $(DOA_LIB) $(GSC_LIB)
libspatial_common.a: $(SPATIAL_COMMON_LIB)
libdoa.a: $(DOA_LIB)
libgsc.a: $(GSC_LIB)

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

$(OBJ_DIR)/gsc.o: GSC/gsc.c GSC/gsc.h utility/complex.h utility/spatial_simd.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(SPATIAL_COMMON_LIB): $(SPATIAL_COMMON_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "spatial-common [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(DOA_LIB): $(DOA_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "doa [$(BACKEND), SIMD=$(SIMD)] -> $@"

$(GSC_LIB): $(GSC_OBJS) | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "gsc [$(BACKEND), SIMD=$(SIMD)] -> $@"

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

audit-no-stdio: libs
	@for lib in "$(SPATIAL_COMMON_LIB)" "$(DOA_LIB)" "$(GSC_LIB)"; do \
	  hits=$$(nm -A "$$lib" | awk '{ n=$$NF; if (n ~ /^_?(fprintf|vfprintf|printf|puts|fputs|stderr|__stderrp|fputc|fwrite)$$/) print }'); \
	  if [ -n "$$hits" ]; then echo "FAIL: stdio symbols in $$lib"; echo "$$hits"; exit 1; fi; \
	done
	@echo "spatial libraries: NO_STDIO audit PASS"

clean:
	rm -rf "$(OBJ_ROOT)" "$(BIN_ROOT)"

-include $(wildcard $(OBJ_DIR)/*.d)
