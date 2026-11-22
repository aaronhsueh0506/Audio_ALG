CC  ?= cc
CXX ?= c++
AR  ?= ar
RANLIB ?= ranlib
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
# Same hook/convention as the per-model stream Makefiles (dfn2 and, eventually, gtcrn/rnnoise_erb/
# dfn3): reaches this directory's own compiles and every sub-make below.
EXTRA_CFLAGS ?=
override CFLAGS += -ffp-contract=off -fno-math-errno -I. -Idfn2 -I../../lib/ainr/dfn2 -I../common $(EXTRA_CFLAGS)
LDFLAGS ?=
SIMD ?= 1
override SIMD := $(strip $(SIMD))
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): ne10 is the only FFT backend)
endif

include ../../lib/deps.mk

# MASK_ONLY selects the DFN2 variant ainr.o and dfn2/'s own archive are both built against -- same
# knob and OBJ_DIR discipline as dfn2/Makefile's (ainr.c has its own #ifdef DFN2_MASK_ONLY catalog).
MASK_ONLY ?= 0
override MASK_ONLY := $(strip $(MASK_ONLY))
$(eval $(call CHECK_01,MASK_ONLY))
MASK_CPPFLAGS :=
ifeq ($(MASK_ONLY),1)
MASK_CPPFLAGS += -DDFN2_MASK_ONLY
endif

WERROR ?= 0
ifeq ($(WERROR),1)
override CFLAGS += -Werror
endif

$(eval $(call RESOLVE_COMMON_LIB))
override CFLAGS += -I$(COMMON_DIR)/include -I$(COMMON_DIR)

$(eval $(call CHECK_01,SIMD))

SIMD_CPPFLAGS :=
ifeq ($(SIMD),0)
SIMD_CPPFLAGS += -DSIMD_KERNELS_FORCE_SCALAR
endif

# Forwarded to the dfn2/ sub-make: same shape as pipelines/Makefile's own FWD / stream/aiaec's FWD.
FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) MASK_ONLY=$(MASK_ONLY) $(TOOLCHAIN_FWD)

BUILD_DIR := build
override CC_TARGET_SIG := $(TARGET_ABI_SIGNATURE)
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) SIMD=$(SIMD) SIMD_CPPFLAGS=$(SIMD_CPPFLAGS) MASK_ONLY=$(MASK_ONLY) MASK_CPPFLAGS=$(MASK_CPPFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) COMMON_LIB=$(COMMON_LIB)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
OBJ_DIR := $(BUILD_DIR)/simd$(SIMD)-mask$(MASK_ONLY)-$(CFG_SIG)

# ainr.c dispatches by model_id (DFN2 full/lite today) -- it lives here, one level above dfn2/,
# instead of inside its own build. It only needs dfn2/'s headers, never recompiles dfn2's core.
STREAM_SOURCES := ainr.c
STREAM_HEADERS := ainr.h ainr_product.h dfn2_grids.h dfn2/dfn2_stream.h ../../lib/ainr/dfn2/dfn2_grid.h ../common/stream_mem.h ../common/stream_range.h ../common/stream_run.h
LIB := $(OBJ_DIR)/libainr.a
LIB_OBJECTS := $(OBJ_DIR)/ainr.o
VPATH := . ../common

.PHONY: all lib print-lib-path print-cflags print-sources test dfn2-lib \
	test-ainr-dfn2 test-ainr-freq test-ainr-latency test-ainr-blocks test-ainr-freq-nonfinite \
	test-ainr-state-generation test-ainr-pool-gates test-ainr-product test-ainr-graph-io \
	test-link-only-ainr work-dumper graph-io-dumper audit-no-stdio clean _cfg_guard

all: lib

lib: $(LIB)

print-lib-path:
	@echo "$(abspath $(LIB))"

print-cflags:
	@printf '%s\n' "$(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS)"

print-sources:
	@printf '%s\n' "$(STREAM_SOURCES)"

$(LIB): $(LIB_OBJECTS) | _cfg_guard
	rm -f "$@"
	$(AR) rcs "$@" $(LIB_OBJECTS)

_cfg_guard:
	$(CFG_GUARD_RECIPE)

$(OBJ_DIR)/%.o: %.c $(STREAM_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) -MMD -MP -c -o "$@" "$<"

-include $(LIB_OBJECTS:.o=.d)

COMMON_HEADERS := $(wildcard $(COMMON_DIR)/include/*.h $(COMMON_DIR)/utility/*.h)

# dfn2-lib: sub-make dfn2/ for its own archive (same orchestration shape as stream/aiaec's
# deepvqe-lib/ulcnet-lib and pipelines/Makefile's COMMON_LIB/AEC_LIB/NR_LIB resolution), never
# recompiling dfn2's core sources into this directory. Resolved lazily inside recipes, not as a
# top-level := shell call, so a plain `make clean`/`print-cflags` here never triggers a sub-build.
dfn2-lib:
	+$(MAKE) --no-print-directory -C dfn2 $(FWD) lib

# Host-only ainr.h/ainr.c regression suite (no NPU): compiles ainr.c's own sources directly, same
# convention dfn2/Makefile used for its own smoke/verify-vs-sample binaries.
CORE_SOURCES := ../../lib/ainr/dfn2/dfn2_process.c \
	../../lib/ainr/dfn2/dfn2_model_io.c \
	../../lib/ainr/dfn2/dfn2_prepost.c
CORE_HEADERS := ../../lib/ainr/dfn2/dfn2_grid.h \
	../../lib/ainr/dfn2/dfn2_process.h \
	../../lib/ainr/dfn2/dfn2_model_io.h \
	../../lib/ainr/dfn2/dfn2_model_io_priv.h \
	../../lib/ainr/dfn2/dfn2_prepost.h

# The test fixtures and the sample-side headers the suites include (ERB file reader, graph rows).
TEST_FIXTURE := tests/ainr_test_fixture.h dfn2/tests/erb_fixture.h dfn2_erb_file.h graph_io_dfn2.h \
	../common/graph_io.h ../common/stream_bind.h

AINR_TEST_BIN := $(OBJ_DIR)/test_ainr_dfn2
AINR_TEST_SOURCES := tests/test_ainr_dfn2.c $(CORE_SOURCES) dfn2/dfn2_stream.c ainr.c

$(AINR_TEST_BIN): $(AINR_TEST_SOURCES) $(TEST_FIXTURE) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
ifeq ($(MASK_ONLY),1)
	$(error test-ainr-dfn2 is full-variant only (needs real catalog grids to cross-check against \
dfn2_stream_* directly) -- run with MASK_ONLY=0, or use test-link-only-ainr which already covers \
the lite build, or dfn2/Makefile's own test-variant-guard)
endif
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) \
		$(AINR_TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$@"

test-ainr-dfn2: $(AINR_TEST_BIN)
	"$(AINR_TEST_BIN)"

# Host-only regression suite for the "_freq" family against the PCM family (tests/test_ainr_freq.c's own
# top comment) -- same build shape as test-ainr-dfn2 above (full-variant only, same reason: needs real
# catalog grids to cross-check against).
AINR_FREQ_TEST_BIN := $(OBJ_DIR)/test_ainr_freq
AINR_FREQ_TEST_SOURCES := tests/test_ainr_freq.c $(CORE_SOURCES) dfn2/dfn2_stream.c ainr.c

$(AINR_FREQ_TEST_BIN): $(AINR_FREQ_TEST_SOURCES) $(TEST_FIXTURE) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
ifeq ($(MASK_ONLY),1)
	$(error test-ainr-freq is full-variant only (needs real catalog grids to cross-check \
against) -- run with MASK_ONLY=0)
endif
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) \
		$(AINR_FREQ_TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$@"

test-ainr-freq: $(AINR_FREQ_TEST_BIN)
	"$(AINR_FREQ_TEST_BIN)"

# Variant-agnostic host suites (they pick the full or the lite grids by DFN2_MASK_ONLY): the
# stream latency through both families (test_ainr_latency.c) and the other single-purpose suites below.
GENERIC_TEST_SOURCES := $(CORE_SOURCES) dfn2/dfn2_stream.c ainr.c

define GENERIC_TEST_RULE
$$(OBJ_DIR)/$(1): tests/$(1).c $$(TEST_FIXTURE) $$(GENERIC_TEST_SOURCES) $$(CORE_HEADERS) $$(STREAM_HEADERS) $$(COMMON_LIB) $$(COMMON_HEADERS) | _cfg_guard
	$$(CC) $$(CFLAGS) $$(SIMD_CPPFLAGS) $$(MASK_CPPFLAGS) tests/$(1).c $$(GENERIC_TEST_SOURCES) \
		"$$(COMMON_LIB)" $$(LDFLAGS) -lm -o "$$@"
endef
$(eval $(call GENERIC_TEST_RULE,test_ainr_latency))
$(eval $(call GENERIC_TEST_RULE,test_ainr_freq_nonfinite))
$(eval $(call GENERIC_TEST_RULE,test_ainr_state_generation))
$(eval $(call GENERIC_TEST_RULE,test_ainr_pool_gates))
$(eval $(call GENERIC_TEST_RULE,test_ainr_atten_lim))

test-ainr-latency: $(OBJ_DIR)/test_ainr_latency
	"$(OBJ_DIR)/test_ainr_latency"

test-ainr-freq-nonfinite: $(OBJ_DIR)/test_ainr_freq_nonfinite
	"$(OBJ_DIR)/test_ainr_freq_nonfinite"

test-ainr-state-generation: $(OBJ_DIR)/test_ainr_state_generation
	"$(OBJ_DIR)/test_ainr_state_generation"

test-ainr-pool-gates: $(OBJ_DIR)/test_ainr_pool_gates
	"$(OBJ_DIR)/test_ainr_pool_gates"

test-ainr-atten-lim: $(OBJ_DIR)/test_ainr_atten_lim
	"$(OBJ_DIR)/test_ainr_atten_lim"

# test-ainr-blocks: the mask_lookahead 0 model (id AINR_MODEL_ID_DFN2_FULL_L0): the block stream of every N > 1 row
# against the N == 1 stream. Full variant only.
test-ainr-blocks: tests/test_ainr_blocks.c $(TEST_FIXTURE) $(GENERIC_TEST_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
ifeq ($(MASK_ONLY),1)
	$(error test-ainr-blocks is full-variant only (the mask_lookahead 0 model has no lite build); run with MASK_ONLY=0)
endif
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) -DAINR_PRODUCT_MODEL_ID=3 tests/test_ainr_blocks.c \
		$(GENERIC_TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$(OBJ_DIR)/test_ainr_blocks"
	"$(OBJ_DIR)/test_ainr_blocks"

# work-dumper: builds tests/dump_work.c for this variant and prints the binary path. It prints the init result
# and the StreamWork views (counts, shapes, element counts) of one config from real preprocess calls, for the
# Python pins and the sample model-config check to compare against the exporter.
$(OBJ_DIR)/dump_work: tests/dump_work.c $(TEST_FIXTURE) $(GENERIC_TEST_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) tests/dump_work.c $(GENERIC_TEST_SOURCES) \
		"$(COMMON_LIB)" $(LDFLAGS) -lm -o "$@"

work-dumper: $(OBJ_DIR)/dump_work
	@echo "$(abspath $(OBJ_DIR))/dump_work"

# test-ainr-product: ainr.c compiled with a different AINR_PRODUCT_MODEL_ID (the product binding, ainr_product.h)
# in extra test binaries, one per spec "<test> <id>" (tests/test_ainr_product.c): test 1 the variant's own model,
# test 2 an id of the other archive variant (refused), test 3 the mask_lookahead 0 model (id 3, full only). The full
# build runs ids 1 and 3 and the lite id 2 as the other variant's; the lite build its own id 2 and the full ids 1 and
# 3 as the other variant's.
ifeq ($(MASK_ONLY),1)
PRODUCT_SPECS := "1 2" "2 1" "2 3"
else
PRODUCT_SPECS := "1 1" "2 2" "3 3"
endif
test-ainr-product: tests/test_ainr_product.c $(TEST_FIXTURE) $(GENERIC_TEST_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	@set -e; for spec in $(PRODUCT_SPECS); do \
	  set -- $$spec; test=$$1; id=$$2; \
	  $(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) -DAINR_TEST_PRODUCT=$$test -DAINR_PRODUCT_MODEL_ID=$$id \
	    tests/test_ainr_product.c $(GENERIC_TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm \
	    -o "$(OBJ_DIR)/test_ainr_product_$$test"; \
	  "$(OBJ_DIR)/test_ainr_product_$$test"; \
	done

# test-ainr-graph-io: the DFN2 graph rows (graph_io_dfn2.h) against the library of their model and the StreamWork of
# real preprocess calls (tests/test_ainr_graph_io.c): this variant's own model, and for the full build also the
# mask_lookahead 0 model (AINR_MODEL_ID_DFN2_FULL_L0), which owns the 32 kHz and the 48 kHz mask_lookahead 0 rows.
# The port check of the model-config header links ../common/stream_bind.c (samples only, not the archive).
ifeq ($(MASK_ONLY),1)
GRAPH_IO_EXTRA_SPECS :=
else
GRAPH_IO_EXTRA_SPECS := "1 -DAINR_PRODUCT_MODEL_ID=3"
endif
test-ainr-graph-io: tests/test_ainr_graph_io.c $(TEST_FIXTURE) $(GENERIC_TEST_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	@set -e; for spec in "0 " $(GRAPH_IO_EXTRA_SPECS); do \
	  set -- $$spec; tag=$$1; shift; \
	  $(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) "$$@" tests/test_ainr_graph_io.c $(GENERIC_TEST_SOURCES) \
	    ../common/stream_bind.c "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$(OBJ_DIR)/test_ainr_graph_io_$$tag"; \
	  "$(OBJ_DIR)/test_ainr_graph_io_$$tag"; \
	done

# graph-io-dumper: tests/dump_graph_io.c prints the rows of graph_io_dfn2.h and the tables' layout constants;
# the Python pin AINR/tests/test_dfn2_port_order.py compares them with a fresh export. Prints the binary path.
$(OBJ_DIR)/dump_graph_io: tests/dump_graph_io.c graph_io_dfn2.h ../common/graph_io.h ../common/stream_bind.c ../common/stream_bind.h | _cfg_guard
	$(CC) $(CFLAGS) tests/dump_graph_io.c ../common/stream_bind.c -lm -o "$@"

graph-io-dumper: $(OBJ_DIR)/dump_graph_io
	@echo "$(abspath $(OBJ_DIR))/dump_graph_io"

# test-link-only-ainr: builds BOTH full and mask-only dfn2/ archives (sub-make, as ainr.c is built here), then
# links tests/link_only_ainr.c -- which #includes ONLY ainr.h -- against each; both must complete a real round trip.
AINR_SYMBOLS := ainr_get_mem_size ainr_init ainr_get_mem_size_freq ainr_init_freq

test-link-only-ainr:
	@$(MAKE) --no-print-directory -C dfn2 MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@$(MAKE) --no-print-directory -C dfn2 MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@$(MAKE) --no-print-directory MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@$(MAKE) --no-print-directory MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@full_dfn2_lib="$$($(MAKE) -s --no-print-directory -C dfn2 MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	lite_dfn2_lib="$$($(MAKE) -s --no-print-directory -C dfn2 MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	full_ainr_lib="$$($(MAKE) -s --no-print-directory MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	full_cflags="$$($(MAKE) -s --no-print-directory MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-cflags)"; \
	lite_ainr_lib="$$($(MAKE) -s --no-print-directory MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	lite_cflags="$$($(MAKE) -s --no-print-directory MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-cflags)"; \
	test -n "$$full_ainr_lib" && test -n "$$lite_ainr_lib" || { echo "FAIL: print-lib-path returned nothing" >&2; exit 1; }; \
	full_obj="$$(dirname "$$full_ainr_lib")"; lite_obj="$$(dirname "$$lite_ainr_lib")"; \
	$(CC) $$full_cflags tests/link_only_ainr.c "$$full_ainr_lib" "$$full_dfn2_lib" "$(COMMON_LIB)" -lm \
	     -o "$$full_obj/link_only_ainr_full" || exit 1; \
	"$$full_obj/link_only_ainr_full" || exit 1; \
	$(CC) $$lite_cflags -DAINR_TEST_EXPECT_LITE tests/link_only_ainr.c "$$lite_ainr_lib" "$$lite_dfn2_lib" "$(COMMON_LIB)" -lm \
	     -o "$$lite_obj/link_only_ainr_lite" || exit 1; \
	"$$lite_obj/link_only_ainr_lite" || exit 1; \
	for lib in "$$full_ainr_lib" "$$lite_ainr_lib"; do \
	  for sym in $(AINR_SYMBOLS); do \
	    nm "$$lib" 2>/dev/null | grep -Eq " T _?$$sym$$" || { echo "FAIL: $$sym not exported by $$lib" >&2; exit 1; }; \
	  done; \
	  if nm "$$lib" 2>/dev/null | grep -Eq " [Tt] _?stream_check_ports$$"; then \
	    echo "FAIL: $$lib carries the sample-side port check (stream_bind.o)" >&2; exit 1; fi; \
	done; \
	echo "test-link-only-ainr: PASS (both archives export ainr_* and behave as documented)"

# test-ainr-dfn2 needs real catalog grids to cross-check against dfn2_stream_* directly and is
# full-variant only (see its own $(error) guard above) -- the lite build skips it here rather than
# aborting the whole composite target (test-link-only-ainr always builds/exercises both variants
# internally regardless of this directory's own MASK_ONLY).
ifeq ($(MASK_ONLY),1)
test: test-ainr-latency test-ainr-freq-nonfinite test-ainr-state-generation test-ainr-pool-gates test-ainr-atten-lim \
	test-ainr-product test-ainr-graph-io test-link-only-ainr
else
test: test-ainr-dfn2 test-ainr-freq test-ainr-latency test-ainr-blocks test-ainr-freq-nonfinite \
	test-ainr-state-generation test-ainr-pool-gates test-ainr-atten-lim test-ainr-product test-ainr-graph-io test-link-only-ainr
endif

# audit-no-stdio: $(LIB) (this MASK_ONLY variant) has no undefined stdio reference; recipe in lib/deps.mk.
audit-no-stdio: $(LIB)
	$(AUDIT_NO_STDIO_RECIPE)

clean:
	rm -rf "$(BUILD_DIR)"
	+$(MAKE) --no-print-directory -C dfn2 clean
