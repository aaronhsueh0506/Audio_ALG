CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
# The floating-point contract and project include paths are invariants of the
# C/Python parity boundary. Keep them even when a board build overrides
# CFLAGS on the command line.
override CFLAGS += -ffp-contract=off -I.
LDFLAGS ?=
SIMD ?= 1
BACKEND ?= ne10

# The FFT (and Complex type) come from audio_common -- the cross-product
# base library every port already carries; the models keep only their own
# DSP on top of it.
AC_DIR ?= $(abspath ../../audio_common)
AC_LIB = $(shell $(MAKE) -s -C $(AC_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) 	lib print-lib-path | tail -1)
override CFLAGS += -I$(AC_DIR)/include

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

BUILD_DIR := build
SOURCES := tests/test_c_prepost.c \
	DeepFilterNet2/dfn2_process.c \
	DeepFilterNet2/dfn2_model_io.c \
	GTCRN/gtcrn_process.c
HEADERS := \
	DeepFilterNet2/dfn2_process.h \
	DeepFilterNet2/dfn2_model_io.h \
	GTCRN/gtcrn_process.h
TEST_BIN := $(BUILD_DIR)/test_c_prepost_simd$(SIMD)
SCALAR_BIN := $(BUILD_DIR)/test_c_prepost_scalar

.PHONY: all test test-simd clean

all: $(TEST_BIN)

$(BUILD_DIR):
	mkdir -p "$@"

$(TEST_BIN): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(SOURCES) "$(AC_LIB)" $(LDFLAGS) -lm -o "$@"

$(SCALAR_BIN): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
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
