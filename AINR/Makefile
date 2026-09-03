CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
# The floating-point contract and project include paths are invariants of the
# C/Python parity boundary. Keep them even when a board build overrides
# CFLAGS on the command line.
override CFLAGS += -ffp-contract=off -I.
LDFLAGS ?=
SIMD ?= 1
override SIMD := $(strip $(SIMD))
BACKEND ?= ne10

# The FFT (and Complex type) come from audio_common -- the cross-product
# base library every port already carries; the models keep only their own
# DSP on top of it.
AC_DIR ?= $(abspath ../../audio_common)
AC_LIB = $(shell $(MAKE) -s -C $(AC_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) 	lib print-lib-path | tail -1)
override CFLAGS += -I$(AC_DIR)/include

# -Werror opt-in, the same control name and shape as AIAEC/Makefile and
# pipelines/Makefile. `override` because CFLAGS may be a command-line
# variable here too.
WERROR ?= 0
ifeq ($(WERROR),1)
override CFLAGS += -Werror
endif

ifneq ($(words $(SIMD)),1)
$(error SIMD must be exactly 0 or 1)
endif
ifneq ($(filter-out 0 1,$(SIMD)),)
$(error SIMD must be exactly 0 or 1)
endif

SIMD_CPPFLAGS :=
ifeq ($(SIMD),0)
SIMD_CPPFLAGS += -DSIMD_KERNELS_FORCE_SCALAR
endif

# --- Configuration-keyed outputs --------------------------------------------
# Every knob that changes what a binary or object contains is folded into a
# signature, and everything this file builds lives under build/simd<N>-<sig>/.
# Without this, `make SIMD=0 lib` after a SIMD=1 build reports "Nothing to
# be done" (or re-archives a mix of scalar and NEON objects), and a WERROR=1
# run reuses a binary that was never compiled with -Werror. `make
# print-lib-path` prints the archive for the current configuration.
BUILD_DIR := build
CFG_SIG_PAYLOAD := CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) SIMD=$(SIMD) SIMD_CPPFLAGS=$(SIMD_CPPFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) AC_LIB=$(AC_LIB)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
OBJ_DIR := $(BUILD_DIR)/simd$(SIMD)-$(CFG_SIG)
# dfn2_prepost.c is not referenced by tests/test_c_prepost.c. It is in
# SOURCES so the contract binary compiles it under both SIMD settings; its
# behavioural gate is tests/test_dfn2_prepost_c.py and its shipped form is
# the archive below.
SOURCES := tests/test_c_prepost.c \
	DeepFilterNet2/dfn2_process.c \
	DeepFilterNet2/dfn2_model_io.c \
	DeepFilterNet2/dfn2_prepost.c \
	GTCRN/gtcrn_process.c
HEADERS := \
	DeepFilterNet2/dfn2_process.h \
	DeepFilterNet2/dfn2_model_io.h \
	DeepFilterNet2/dfn2_prepost.h \
	GTCRN/gtcrn_process.h
TEST_BIN := $(OBJ_DIR)/test_c_prepost
SCALAR_BIN := $(OBJ_DIR)/test_c_prepost_scalar

# The AINR release archive, the counterpart of AIAEC's libaiaec_prepost.a:
# DeepFilterNet2 (including its pre/post class) and GTCRN. RNNoise-ERB is
# NOT in it -- it has its own Makefile under RNNoise-ERB/ and no archive of
# its own (its process.c is compiled directly), so a board using it adds that
# source to its build. The contract binaries above compile the same sources
# whole-program rather than linking this.
LIB := $(OBJ_DIR)/libainr_prepost.a
LIB_OBJECTS := $(OBJ_DIR)/dfn2_process.o \
	$(OBJ_DIR)/dfn2_model_io.o \
	$(OBJ_DIR)/dfn2_prepost.o \
	$(OBJ_DIR)/gtcrn_process.o
VPATH := DeepFilterNet2:GTCRN

.PHONY: all lib print-lib-path test test-simd clean _cfg_guard

all: lib $(TEST_BIN)

lib: $(LIB)

print-lib-path:
	@echo "$(abspath $(LIB))"

# rm first: `ar rcs` updates in place and never prunes, so an object dropped
# from LIB_OBJECTS would otherwise linger in the archive under the same key.
$(LIB): $(LIB_OBJECTS)
	rm -f "$@" && $(AR) rcs "$@" $(LIB_OBJECTS)

_cfg_guard:
	@mkdir -p "$(OBJ_DIR)"
	@m="$(OBJ_DIR)/config.manifest"; \
	if [ -f "$$m" ] && [ "$$(cat "$$m")" != "$(CFG_SIG_PAYLOAD)" ]; then \
	  echo "FATAL: configuration signature collision in $$m" >&2; \
	  echo "  stored : $$(cat "$$m")" >&2; \
	  echo "  current: $(CFG_SIG_PAYLOAD)" >&2; \
	  exit 1; \
	fi; \
	printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$m"

$(OBJ_DIR)/%.o: %.c | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -MMD -MP -c -o "$@" "$<"

-include $(LIB_OBJECTS:.o=.d)

# The contract binaries are whole-program builds, so they list what they
# read: the model sources and headers, audio_common's archive and headers.
AC_HEADERS := $(wildcard $(AC_DIR)/include/*.h)

$(TEST_BIN): $(SOURCES) $(HEADERS) $(AC_LIB) $(AC_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(SOURCES) "$(AC_LIB)" $(LDFLAGS) -lm -o "$@"

$(SCALAR_BIN): $(SOURCES) $(HEADERS) $(AC_LIB) $(AC_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) -DSIMD_KERNELS_FORCE_SCALAR $(SOURCES) "$(AC_LIB)" \
		$(LDFLAGS) -lm -o "$@"

test: $(TEST_BIN)
	"$(TEST_BIN)"

test-simd: $(TEST_BIN) $(SCALAR_BIN)
	@auto_output="$$($(TEST_BIN))"; \
	scalar_output="$$($(SCALAR_BIN))"; \
	auto_digest="$${auto_output##*digest=}"; \
	scalar_digest="$${scalar_output##*digest=}"; \
	printf '%s\n%s\n' "$$auto_output" "$$scalar_output"; \
	test "$$auto_digest" = "$$scalar_digest" || { \
		echo "FAIL: SIMD/scalar digest mismatch" >&2; exit 1; \
	}

clean:
	rm -rf "$(BUILD_DIR)"
