CC  ?= cc
CXX ?= c++
AR  ?= ar
RANLIB ?= ranlib
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
# EXTRA_CFLAGS is the supported hook for a caller's own flags (e.g. a cross-compile -arch/--target):
# it reaches both this directory's own compiles and the audiolib sub-make below (as that
# Makefile's own EXTRA_CFLAGS), same convention as pipelines/Makefile's FWD. Plain CFLAGS is never
# forwarded to the sub-make -- audiolib's Makefile treats a command-line CFLAGS override as fatal
# (it would silently drop that Makefile's own FP-contraction policy and NE10 defines), and this
# directory's own CFLAGS carries -I paths meaningless there.
EXTRA_CFLAGS ?=
# Same floating-point/include invariants as AINR/Makefile.
override CFLAGS += -ffp-contract=off -fno-math-errno -I. -I../../../lib/ainr/dfn2 -I../../common $(EXTRA_CFLAGS)
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

# NO_STDIO=1 compiles out dfn2_stream.c's file-loader entry points (the only stdio in this SDK)
# via -DDFN2_STREAM_NO_STDIO; see audit-no-stdio.
NO_STDIO ?= 0
override NO_STDIO := $(strip $(NO_STDIO))
ifneq ($(words $(NO_STDIO)),1)
$(error NO_STDIO must be exactly 0 or 1)
endif
ifneq ($(filter-out 0 1,$(NO_STDIO)),)
$(error NO_STDIO must be exactly 0 or 1)
endif
NO_STDIO_CPPFLAGS :=
ifeq ($(NO_STDIO),1)
NO_STDIO_CPPFLAGS += -DDFN2_STREAM_NO_STDIO
endif

WERROR ?= 0
ifeq ($(WERROR),1)
override CFLAGS += -Werror
endif

include ../../../lib/deps.mk
# RESOLVE_COMMON_LIB (from lib/deps.mk) resolves COMMON_LIB via a sub-make forwarding
# BACKEND/SIMD/TOOLCHAIN_FWD and validates it -- a cross-compile (different CC/CXX, or
# EXTRA_CFLAGS carrying e.g. -arch arm64e) must build a COMMON_LIB for the SAME target, not
# silently fall back to the host default and fail to link only once an external consumer tries to
# use it; exercised by hand with a cross CC (e.g. CC=clang EXTRA_CFLAGS="-arch arm64e"), no local
# test here since it needs an actual alternate toolchain available. Skipped outright (not just the
# validation) for a plain `make clean` -- see RESOLVE_COMMON_LIB's own comment.
$(eval $(call RESOLVE_COMMON_LIB))
override CFLAGS += -I$(COMMON_DIR)/include -I$(COMMON_DIR)

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
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) SIMD=$(SIMD) SIMD_CPPFLAGS=$(SIMD_CPPFLAGS) MASK_ONLY=$(MASK_ONLY) MASK_CPPFLAGS=$(MASK_CPPFLAGS) NO_STDIO=$(NO_STDIO) NO_STDIO_CPPFLAGS=$(NO_STDIO_CPPFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) COMMON_LIB=$(COMMON_LIB)
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
STREAM_HEADERS := dfn2_stream.h ../../common/stream_mem.h ../../common/stream_run.h
TEST_HEADERS := tests/erb_fixture.h

LIB := $(OBJ_DIR)/libdfn2_stream.a
LIB_OBJECTS := $(OBJ_DIR)/dfn2_process.o $(OBJ_DIR)/dfn2_model_io.o $(OBJ_DIR)/dfn2_prepost.o \
	$(OBJ_DIR)/dfn2_stream.o

VPATH := ../../../lib/ainr/dfn2

SMOKE_BIN := $(OBJ_DIR)/smoke_dfn2_stream
SMOKE_SOURCES := tests/smoke_dfn2_stream.c $(CORE_SOURCES) dfn2_stream.c

# Needs the file-loader entry points (stdio), so never built when NO_STDIO=1.
ERB_FILE_CHECK_BIN := $(OBJ_DIR)/check_erb_file
ERB_FILE_CHECK_SOURCES := tests/check_erb_file.c $(CORE_SOURCES) dfn2_stream.c

# Compares dfn2_stream against an independent re-implementation of the board sample's own loop,
# built directly against lib/ainr/dfn2 + audiolib (never against dfn2_stream.c's internals); see
# the file's own header comment.
VERIFY_VS_SAMPLE_BIN := $(OBJ_DIR)/verify_vs_sample
VERIFY_VS_SAMPLE_SOURCES := tests/verify_vs_sample.c $(CORE_SOURCES) dfn2_stream.c

# Memory-mode checks always; the file-mode ones too unless NO_STDIO (the test compiles under both).
ERB_PARTITION_BIN := $(OBJ_DIR)/erb_partition
ERB_PARTITION_SOURCES := tests/erb_partition.c $(CORE_SOURCES) dfn2_stream.c

# Run-status table of dfn2_stream_post_process() against a stand-in accelerator that keeps its own state.
FAIL_TABLE_BIN := $(OBJ_DIR)/fail_table
FAIL_TABLE_SOURCES := tests/fail_table.c $(CORE_SOURCES) dfn2_stream.c

POOL_SIZE_BIN := $(OBJ_DIR)/pool_size
POOL_SIZE_SOURCES := tests/pool_size.c $(CORE_SOURCES) dfn2_stream.c

# The freq_only context against a full one; memory mode always, file mode too unless NO_STDIO.
FREQ_ONLY_BIN := $(OBJ_DIR)/freq_only
FREQ_ONLY_SOURCES := tests/freq_only.c $(CORE_SOURCES) dfn2_stream.c

.PHONY: all lib print-lib-path print-cflags print-sources test smoke check-erb-file verify-vs-sample fail-table pool-size freq-only erb-partition test-variant-guard check-header-cxx audit-no-stdio clean _cfg_guard

all: lib

lib: $(LIB)

print-lib-path:
	@echo "$(abspath $(LIB))"

print-cflags:
	@printf '%s\n' "$(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(NO_STDIO_CPPFLAGS)"

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
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(NO_STDIO_CPPFLAGS) -MMD -MP -c -o "$@" "$<"

-include $(LIB_OBJECTS:.o=.d)

COMMON_HEADERS := $(wildcard $(COMMON_DIR)/include/*.h $(COMMON_DIR)/utility/*.h)

$(SMOKE_BIN): $(SMOKE_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(TEST_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(NO_STDIO_CPPFLAGS) $(SMOKE_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

smoke: $(SMOKE_BIN)
	"$(SMOKE_BIN)"

$(ERB_FILE_CHECK_BIN): $(ERB_FILE_CHECK_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(TEST_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
ifeq ($(NO_STDIO),1)
	$(error check-erb-file needs the file-loader entry points; run with NO_STDIO=0)
endif
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(NO_STDIO_CPPFLAGS) $(ERB_FILE_CHECK_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

check-erb-file: $(ERB_FILE_CHECK_BIN)
	"$(ERB_FILE_CHECK_BIN)"

$(VERIFY_VS_SAMPLE_BIN): $(VERIFY_VS_SAMPLE_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(TEST_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(NO_STDIO_CPPFLAGS) $(VERIFY_VS_SAMPLE_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

verify-vs-sample: $(VERIFY_VS_SAMPLE_BIN)
	"$(VERIFY_VS_SAMPLE_BIN)"
	"$(VERIFY_VS_SAMPLE_BIN)" --inject-shift

$(FAIL_TABLE_BIN): $(FAIL_TABLE_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(TEST_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(NO_STDIO_CPPFLAGS) $(FAIL_TABLE_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

fail-table: $(FAIL_TABLE_BIN)
	"$(FAIL_TABLE_BIN)"

$(POOL_SIZE_BIN): $(POOL_SIZE_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(TEST_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(NO_STDIO_CPPFLAGS) $(POOL_SIZE_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

pool-size: $(POOL_SIZE_BIN)
	"$(POOL_SIZE_BIN)"

$(FREQ_ONLY_BIN): $(FREQ_ONLY_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(TEST_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(NO_STDIO_CPPFLAGS) $(FREQ_ONLY_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

freq-only: $(FREQ_ONLY_BIN)
	"$(FREQ_ONLY_BIN)"

$(ERB_PARTITION_BIN): $(ERB_PARTITION_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(TEST_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) $(NO_STDIO_CPPFLAGS) $(ERB_PARTITION_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

erb-partition: $(ERB_PARTITION_BIN)
	"$(ERB_PARTITION_BIN)"

test: smoke check-erb-file verify-vs-sample fail-table pool-size freq-only erb-partition test-variant-guard check-header-cxx

# check-header-cxx: a C++ consumer of the DFN2 headers (tests/cxx_consumer.cpp) must reference no mangled symbol;
# a generated probe calling one C++-linkage function must show one, so the scan is shown able to fail.
CXX_PROBE_FLAGS = -std=c++11 -Wall -Wextra -Werror $(filter -I% -D%,$(CFLAGS)) $(MASK_CPPFLAGS)
check-header-cxx: tests/cxx_consumer.cpp $(CORE_HEADERS) $(STREAM_HEADERS)
	@mkdir -p "$(OBJ_DIR)"
	$(CXX) $(CXX_PROBE_FLAGS) -c tests/cxx_consumer.cpp -o "$(OBJ_DIR)/cxx_consumer.o"
	@printf 'int dfn2_cxx_control(int v);\nint dfn2_cxx_probe(int v) { return dfn2_cxx_control(v); }\n' > "$(OBJ_DIR)/cxx_control.cpp"
	$(CXX) $(CXX_PROBE_FLAGS) -c "$(OBJ_DIR)/cxx_control.cpp" -o "$(OBJ_DIR)/cxx_consumer_control.o"
	@mangled="$$(nm -u "$(OBJ_DIR)/cxx_consumer.o" | grep -E '(^|[ ])_?_Z' || true)"; \
	if [ -n "$$mangled" ]; then echo "FAIL: C++ linkage leaked from the DFN2 headers:" >&2; echo "$$mangled" >&2; exit 1; fi; \
	nm -u "$(OBJ_DIR)/cxx_consumer.o" | grep -Eq '(^|[ ])_?dfn2_state_mem_size$$' || { echo "FAIL: dfn2_state_mem_size not referenced with C linkage" >&2; exit 1; }; \
	nm -u "$(OBJ_DIR)/cxx_consumer_control.o" | grep -Eq '(^|[ ])_?_Z[0-9]+dfn2_cxx_control' || { echo "FAIL: positive control shows no mangled reference" >&2; exit 1; }; \
	echo "check-header-cxx: PASS (MASK_ONLY=$(MASK_ONLY): every DFN2 call from C++ has C linkage)"

# test-variant-guard: builds BOTH full and mask-only archives, then links an external-consumer TU
# (tests/variant_consumer.c, never the SDK's own .c files) against each in all four combinations.
# The two mismatched pairings must fail to link (dfn2_stream.h's variant-suffixed public symbols --
# see its own comment); the two matched pairings must link AND run a few hops of real audio.
# TOOLCHAIN_FWD comes from lib/deps.mk (included above): CC/CXX/EXTRA_CFLAGS/WERROR, shared with
# the COMMON_LIB resolution below instead of each spelling out the same four vars again.
test-variant-guard:
	@$(MAKE) --no-print-directory MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@$(MAKE) --no-print-directory MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@full_lib="$$($(MAKE) -s --no-print-directory MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	full_cflags="$$($(MAKE) -s --no-print-directory MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-cflags)"; \
	lite_lib="$$($(MAKE) -s --no-print-directory MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	lite_cflags="$$($(MAKE) -s --no-print-directory MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-cflags)"; \
	test -n "$$full_lib" && test -n "$$lite_lib" || { echo "FAIL: print-lib-path returned nothing" >&2; exit 1; }; \
	full_obj="$$(dirname "$$full_lib")"; lite_obj="$$(dirname "$$lite_lib")"; \
	$(CC) $$full_cflags -c tests/variant_consumer.c -o "$$full_obj/variant_consumer_full.o" || exit 1; \
	$(CC) $$lite_cflags -c tests/variant_consumer.c -o "$$lite_obj/variant_consumer_lite.o" || exit 1; \
	if $(CC) $$full_cflags "$$full_obj/variant_consumer_full.o" "$$lite_lib" "$(COMMON_LIB)" -lm \
	     -o "$$full_obj/vg_full_consumer_vs_lite_archive" 2>/dev/null; then \
	  echo "FAIL: full consumer linked against the mask-only archive (must be refused)" >&2; exit 1; \
	fi; \
	echo "test-variant-guard: full consumer vs mask-only archive correctly refused to link"; \
	if $(CC) $$lite_cflags "$$lite_obj/variant_consumer_lite.o" "$$full_lib" "$(COMMON_LIB)" -lm \
	     -o "$$lite_obj/vg_lite_consumer_vs_full_archive" 2>/dev/null; then \
	  echo "FAIL: mask-only consumer linked against the full archive (must be refused)" >&2; exit 1; \
	fi; \
	echo "test-variant-guard: mask-only consumer vs full archive correctly refused to link"; \
	$(CC) $$full_cflags "$$full_obj/variant_consumer_full.o" "$$full_lib" "$(COMMON_LIB)" -lm \
	     -o "$$full_obj/vg_full_consumer_vs_full_archive" || exit 1; \
	"$$full_obj/vg_full_consumer_vs_full_archive" || exit 1; \
	$(CC) $$lite_cflags "$$lite_obj/variant_consumer_lite.o" "$$lite_lib" "$(COMMON_LIB)" -lm \
	     -o "$$lite_obj/vg_lite_consumer_vs_lite_archive" || exit 1; \
	"$$lite_obj/vg_lite_consumer_vs_lite_archive" || exit 1; \
	for spec in "$$full_lib::_mask_only" "$$lite_lib:_mask_only:"; do \
	  lib="$${spec%%:*}"; rest="$${spec#*:}"; var="$${rest%%:*}"; other="$${rest#*:}"; \
	  for sym in dfn2_stream_pre_process dfn2_stream_post_process dfn2_stream_pre_process_freq_dual dfn2_stream_post_process_freq; do \
	    nm "$$lib" 2>/dev/null | grep -Eq " T _?$${sym}$${var}$$" || { echo "FAIL: $${sym}$${var} not exported by $$lib" >&2; exit 1; }; \
	    if nm "$$lib" 2>/dev/null | grep -Eq " T _?$${sym}$${other}$$"; then echo "FAIL: $${sym}$${other} is exported by $$lib (wrong variant)" >&2; exit 1; fi; \
	  done; \
	done; \
	echo "test-variant-guard: variant-suffixed hop entries present, no cross-variant entry exported"; \
	echo "test-variant-guard: PASS (both matched pairings link and run)"

# audit-no-stdio: builds the NO_STDIO=1 archive and asserts, via `nm -u`, that dfn2_stream.o
# references no stdio symbol (the file-loader entry points are compiled out under that flag).
audit-no-stdio:
	@$(MAKE) --no-print-directory NO_STDIO=1 MASK_ONLY=$(MASK_ONLY) SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@nostdio_lib="$$($(MAKE) -s --no-print-directory NO_STDIO=1 MASK_ONLY=$(MASK_ONLY) SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	obj_dir="$$(dirname "$$nostdio_lib")"; \
	undef="$$(nm -u "$$obj_dir/dfn2_stream.o" 2>/dev/null | grep -Ei 'fopen|fread|fclose|fwrite|fprintf|printf|puts' || true)"; \
	if [ -n "$$undef" ]; then \
	  echo "FAIL: dfn2_stream.o references stdio symbols:" >&2; echo "$$undef" >&2; exit 1; \
	fi; \
	echo "audit-no-stdio: PASS ($$obj_dir/dfn2_stream.o has no stdio references)"

clean:
	rm -rf "$(BUILD_DIR)"
