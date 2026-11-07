CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
# Same floating-point/include invariants as AINR/Makefile.
override CFLAGS += -ffp-contract=off -fno-math-errno -I. -I../../../lib/ainr/dfn2 -I../../common
LDFLAGS ?=
SIMD ?= 1
override SIMD := $(strip $(SIMD))
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): ne10 is the only FFT backend)
endif

# MASK_ONLY selects the DFN2 variant this archive and its stream objects are built against; the two
# variants must never share an OBJ_DIR (DFN2_VARIANT_SYMBOL makes mixing them a link error, not a
# silent bug -- see dfn2_process.h), so MASK_ONLY is folded into CFG_SIG like SIMD is.
MASK_ONLY ?= 0
override MASK_ONLY := $(strip $(MASK_ONLY))
ifneq ($(words $(MASK_ONLY)),1)
$(error MASK_ONLY must be exactly 0 or 1)
endif
ifneq ($(filter-out 0 1,$(MASK_ONLY)),)
$(error MASK_ONLY must be exactly 0 or 1)
endif
MASK_CPPFLAGS :=
ifeq ($(MASK_ONLY),1)
MASK_CPPFLAGS += -DDFN2_MASK_ONLY
endif

# NO_STDIO=1 drops dfn2_stream_erb_file.o (the only <stdio.h> TU here) from the archive and from
# the smoke build; see audit-no-stdio.
NO_STDIO ?= 0
override NO_STDIO := $(strip $(NO_STDIO))
ifneq ($(words $(NO_STDIO)),1)
$(error NO_STDIO must be exactly 0 or 1)
endif
ifneq ($(filter-out 0 1,$(NO_STDIO)),)
$(error NO_STDIO must be exactly 0 or 1)
endif

include ../../../lib/deps.mk
COMMON_LIB := $(shell $(MAKE) -s -C $(COMMON_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) 	lib print-lib-path | tail -1)
override CFLAGS += -I$(COMMON_DIR)/include -I$(COMMON_DIR)

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

BUILD_DIR := build
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) SIMD=$(SIMD) SIMD_CPPFLAGS=$(SIMD_CPPFLAGS) MASK_ONLY=$(MASK_ONLY) MASK_CPPFLAGS=$(MASK_CPPFLAGS) NO_STDIO=$(NO_STDIO) BACKEND=$(BACKEND) WERROR=$(WERROR) COMMON_LIB=$(COMMON_LIB)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
OBJ_DIR := $(BUILD_DIR)/simd$(SIMD)-mask$(MASK_ONLY)-$(CFG_SIG)

# The DFN2 core, recompiled here under this OBJ_DIR's MASK_CPPFLAGS (the lib/ainr/dfn2 archive
# AINR/Makefile builds is always the full variant; a stream build needs the matching variant's
# objects of its own, same source, different flags, never sharing an object directory).
CORE_SOURCES := ../../../lib/ainr/dfn2/dfn2_process.c \
	../../../lib/ainr/dfn2/dfn2_model_io.c \
	../../../lib/ainr/dfn2/dfn2_prepost.c
CORE_HEADERS := ../../../lib/ainr/dfn2/dfn2_grid.h \
	../../../lib/ainr/dfn2/dfn2_process.h \
	../../../lib/ainr/dfn2/dfn2_model_io.h \
	../../../lib/ainr/dfn2/dfn2_model_io_priv.h \
	../../../lib/ainr/dfn2/dfn2_prepost.h

STREAM_SOURCES := dfn2_stream.c
ifneq ($(NO_STDIO),1)
STREAM_SOURCES += dfn2_stream_erb_file.c
endif
STREAM_HEADERS := dfn2_stream.h dfn2_stream_internal.h ../../common/stream_mem.h

LIB := $(OBJ_DIR)/libdfn2_stream.a
LIB_OBJECTS := $(OBJ_DIR)/dfn2_process.o $(OBJ_DIR)/dfn2_model_io.o $(OBJ_DIR)/dfn2_prepost.o \
	$(OBJ_DIR)/dfn2_stream.o
ifneq ($(NO_STDIO),1)
LIB_OBJECTS += $(OBJ_DIR)/dfn2_stream_erb_file.o
endif
VPATH := ../../../lib/ainr/dfn2

SMOKE_BIN := $(OBJ_DIR)/smoke_dfn2_stream
SMOKE_SOURCES := tests/smoke_dfn2_stream.c $(CORE_SOURCES) dfn2_stream.c

# Needs the file-loader TU (stdio): never built when NO_STDIO=1.
ERB_FILE_CHECK_BIN := $(OBJ_DIR)/check_erb_file
ERB_FILE_CHECK_SOURCES := tests/check_erb_file.c $(CORE_SOURCES) dfn2_stream.c dfn2_stream_erb_file.c

# Compares dfn2_stream against an independent re-implementation of the board sample's own loop,
# built directly against lib/ainr/dfn2 + audiolib (never against dfn2_stream.c's internals); see
# the file's own header comment. Memory-init only, so it never needs dfn2_stream_erb_file.c.
VERIFY_VS_SAMPLE_BIN := $(OBJ_DIR)/verify_vs_sample
VERIFY_VS_SAMPLE_SOURCES := tests/verify_vs_sample.c $(CORE_SOURCES) dfn2_stream.c

.PHONY: all lib print-lib-path print-cflags print-sources test smoke check-erb-file verify-vs-sample audit-no-stdio clean _cfg_guard

all: lib

lib: $(LIB)

print-lib-path:
	@echo "$(abspath $(LIB))"

print-cflags:
	@printf '%s\n' "$(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS)"

print-sources:
	@printf '%s\n' "$(CORE_SOURCES) $(STREAM_SOURCES)"

$(LIB): $(LIB_OBJECTS)
	rm -f "$@" && $(AR) rcs "$@" $(LIB_OBJECTS)

_cfg_guard:
	@test -n "$(CC_TARGET_SIG)" && test "$(CC_TARGET_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine compiler target; CC preprocessing with the configured flags must succeed." >&2; \
	  exit 1; }
	@mkdir -p "$(OBJ_DIR)"
	@m="$(OBJ_DIR)/config.manifest"; \
	if [ -f "$$m" ] && [ "$$(cat "$$m")" != "$(CFG_SIG_PAYLOAD)" ]; then \
	  echo "FATAL: configuration signature collision in $$m" >&2; \
	  echo "  stored : $$(cat "$$m")" >&2; \
	  echo "  current: $(CFG_SIG_PAYLOAD)" >&2; \
	  exit 1; \
	fi; \
	printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$m"

$(OBJ_DIR)/%.o: %.c $(CORE_HEADERS) $(STREAM_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) -MMD -MP -c -o "$@" "$<"

-include $(LIB_OBJECTS:.o=.d)

COMMON_HEADERS := $(wildcard $(COMMON_DIR)/include/*.h $(COMMON_DIR)/utility/*.h)

$(SMOKE_BIN): $(SMOKE_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(SMOKE_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

smoke: $(SMOKE_BIN)
	"$(SMOKE_BIN)"

$(ERB_FILE_CHECK_BIN): $(ERB_FILE_CHECK_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
ifeq ($(NO_STDIO),1)
	$(error check-erb-file needs dfn2_stream_erb_file.c; run with NO_STDIO=0)
endif
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(ERB_FILE_CHECK_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

check-erb-file: $(ERB_FILE_CHECK_BIN)
	"$(ERB_FILE_CHECK_BIN)"

$(VERIFY_VS_SAMPLE_BIN): $(VERIFY_VS_SAMPLE_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(VERIFY_VS_SAMPLE_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

verify-vs-sample: $(VERIFY_VS_SAMPLE_BIN)
	"$(VERIFY_VS_SAMPLE_BIN)"
	"$(VERIFY_VS_SAMPLE_BIN)" --inject-shift

test: smoke check-erb-file verify-vs-sample

# audit-no-stdio: builds the NO_STDIO=1 archive and asserts, via `nm -u`, that dfn2_stream.o alone
# (never dfn2_stream_erb_file.o, which legitimately uses stdio) references no stdio symbol.
audit-no-stdio:
	@$(MAKE) --no-print-directory NO_STDIO=1 MASK_ONLY=$(MASK_ONLY) SIMD=$(SIMD) lib >/dev/null
	@obj_dir="$$($(MAKE) -s --no-print-directory NO_STDIO=1 MASK_ONLY=$(MASK_ONLY) SIMD=$(SIMD) print-lib-path)"; \
	obj_dir="$$(dirname "$$obj_dir")"; \
	undef="$$(nm -u "$$obj_dir/dfn2_stream.o" 2>/dev/null | grep -Ei 'fopen|fread|fclose|fwrite|fprintf|printf|puts' || true)"; \
	if [ -n "$$undef" ]; then \
	  echo "FAIL: dfn2_stream.o references stdio symbols:" >&2; echo "$$undef" >&2; exit 1; \
	fi; \
	echo "audit-no-stdio: PASS ($$obj_dir/dfn2_stream.o has no stdio references)"

clean:
	rm -rf "$(BUILD_DIR)"
