CC ?= cc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -I.
LDFLAGS ?=
SIMD ?= 1

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
	DeepFilterNet3/dfn3_process.c \
	GTCRN/gtcrn_process.c
HEADERS := \
	DeepFilterNet2/dfn2_process.h \
	DeepFilterNet2/dfn2_model_io.h \
	DeepFilterNet3/dfn3_process.h \
	GTCRN/gtcrn_process.h
TEST_BIN := $(BUILD_DIR)/test_c_prepost_simd$(SIMD)
SCALAR_BIN := $(BUILD_DIR)/test_c_prepost_scalar

.PHONY: all test test-simd clean

all: $(TEST_BIN)

$(BUILD_DIR):
	mkdir -p "$@"

$(TEST_BIN): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(SOURCES) $(LDFLAGS) -lm -o "$@"

$(SCALAR_BIN): $(SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DSIMD_KERNELS_FORCE_SCALAR $(SOURCES) \
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
