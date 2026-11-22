CC  ?= cc
CXX ?= c++
AR  ?= ar
RANLIB ?= ranlib
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
# EXTRA_CFLAGS is the supported hook for a caller's own flags (e.g. a cross-compile -arch/--target); same
# convention as the sibling stream Makefiles (stream/ainr/dfn2, stream/ainr).
EXTRA_CFLAGS ?=
override CFLAGS += -ffp-contract=off -fno-math-errno -I. -I../../common $(EXTRA_CFLAGS)
LDFLAGS ?=
SIMD ?= 1
override SIMD := $(strip $(SIMD))
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): ne10 is the only FFT backend)
endif
include ../../../lib/deps.mk

$(eval $(call CHECK_01,SIMD))
WERROR ?= 0
ifeq ($(WERROR),1)
override CFLAGS += -Werror
endif

$(eval $(call RESOLVE_COMMON_LIB))
override CFLAGS += -I$(COMMON_DIR)/include -I$(COMMON_DIR)

SIMD_CPPFLAGS :=
ifeq ($(SIMD),0)
SIMD_CPPFLAGS += -DSIMD_KERNELS_FORCE_SCALAR
endif

BUILD_DIR := build
override CC_TARGET_SIG := $(TARGET_ABI_SIGNATURE)
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) SIMD=$(SIMD) SIMD_CPPFLAGS=$(SIMD_CPPFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) COMMON_LIB=$(COMMON_LIB)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
OBJ_DIR := $(BUILD_DIR)/simd$(SIMD)-$(CFG_SIG)

LIB := $(OBJ_DIR)/libainr_bridge.a
LIB_OBJECTS := $(OBJ_DIR)/ainr_bridge.o
HEADERS := ainr_bridge.h ../../common/stream_mem.h
COMMON_HEADERS := $(wildcard $(COMMON_DIR)/include/*.h $(COMMON_DIR)/utility/*.h)

TEST_BIN := $(OBJ_DIR)/test_ainr_bridge

.PHONY: all lib print-lib-path print-cflags print-sources test audit-no-stdio clean _cfg_guard

all: lib

lib: $(LIB)

print-lib-path:
	@echo "$(abspath $(LIB))"

print-cflags:
	@printf '%s\n' "$(CFLAGS) $(SIMD_CPPFLAGS)"

print-sources:
	@printf '%s\n' "ainr_bridge.c"

$(LIB): $(LIB_OBJECTS)
	rm -f "$@" && $(AR) rcs "$@" $(LIB_OBJECTS)

_cfg_guard:
	$(CFG_GUARD_RECIPE)

$(OBJ_DIR)/%.o: %.c $(HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -MMD -MP -c -o "$@" "$<"

-include $(LIB_OBJECTS:.o=.d)

# The test links the sources directly (not the archive) and audiolib; no DFN2, no ERB tables.
$(TEST_BIN): tests/test_ainr_bridge.c ainr_bridge.c $(HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) tests/test_ainr_bridge.c ainr_bridge.c "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$@"

test: $(TEST_BIN)
	"$(TEST_BIN)"

# audit-no-stdio: $(LIB) has no undefined stdio reference; recipe in lib/deps.mk (MASK_ONLY does not exist here).
audit-no-stdio: MASK_ONLY := n/a
audit-no-stdio: $(LIB)
	$(AUDIT_NO_STDIO_RECIPE)

clean:
	rm -rf "$(BUILD_DIR)"
