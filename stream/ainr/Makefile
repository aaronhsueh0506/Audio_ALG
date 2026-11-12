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

# MASK_ONLY selects the DFN2 variant ainr.o and dfn2/'s own archive are both built against -- same
# knob and OBJ_DIR discipline as dfn2/Makefile's (ainr.c has its own #ifdef DFN2_MASK_ONLY catalog).
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

# ainr.c's only entry point (ainr_init_from_erb_file/ainr_init) is file-mode only -- it calls
# dfn2_stream_init_from_erb_file(), itself compiled out under NO_STDIO=1 in dfn2/. ainr.o is
# excluded from this directory's own LIB_OBJECTS under NO_STDIO=1, not just internally guarded, so
# a NO_STDIO build never even attempts to compile/link it -- see audit-no-stdio below.
NO_STDIO ?= 0
override NO_STDIO := $(strip $(NO_STDIO))
ifneq ($(words $(NO_STDIO)),1)
$(error NO_STDIO must be exactly 0 or 1)
endif
ifneq ($(filter-out 0 1,$(NO_STDIO)),)
$(error NO_STDIO must be exactly 0 or 1)
endif

WERROR ?= 0
ifeq ($(WERROR),1)
override CFLAGS += -Werror
endif

include ../../lib/deps.mk
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

# Forwarded to the dfn2/ sub-make: same shape as pipelines/Makefile's own FWD / stream/aiaec's FWD.
FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) MASK_ONLY=$(MASK_ONLY) NO_STDIO=$(NO_STDIO) $(TOOLCHAIN_FWD)

BUILD_DIR := build
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) SIMD=$(SIMD) SIMD_CPPFLAGS=$(SIMD_CPPFLAGS) MASK_ONLY=$(MASK_ONLY) MASK_CPPFLAGS=$(MASK_CPPFLAGS) NO_STDIO=$(NO_STDIO) BACKEND=$(BACKEND) WERROR=$(WERROR) COMMON_LIB=$(COMMON_LIB)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
OBJ_DIR := $(BUILD_DIR)/simd$(SIMD)-mask$(MASK_ONLY)-$(CFG_SIG)

# ainr.c dispatches by model_id (DFN2 full/lite today) -- it lives here, one level above dfn2/,
# instead of inside its own build. It only needs dfn2/'s headers, never recompiles dfn2's core.
STREAM_SOURCES :=
STREAM_HEADERS := ainr.h ainr_test.h ainr_product.h dfn2/dfn2_stream.h ../../lib/ainr/dfn2/dfn2_grid.h ../common/stream_mem.h
LIB := $(OBJ_DIR)/libainr.a
LIB_OBJECTS :=
ifeq ($(NO_STDIO),0)
STREAM_SOURCES += ainr.c
LIB_OBJECTS += $(OBJ_DIR)/ainr.o
endif
VPATH := . ../common

.PHONY: all lib print-lib-path print-cflags print-sources test dfn2-lib \
	test-ainr-dfn2 test-ainr-spec-freq test-ainr-work-shapes test-ainr-tail test-ainr-freq-nonfinite \
	test-ainr-state-generation test-ainr-pool-gates test-ainr-product test-ainr-model-io \
	test-link-only-ainr work-dumper model-io-dumper audit-no-stdio clean _cfg_guard

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
ifeq ($(strip $(LIB_OBJECTS)),)
# NO_STDIO=1: ainr.o is excluded above and there is nothing else in this directory's own archive
# (unlike dfn2/, whose LIB_OBJECTS always has its core objects regardless of NO_STDIO) -- write a
# minimal valid empty ar archive (just the format's magic header) rather than invoking $(AR) with
# no file operands, which errors out on this host's ar.
	printf '!<arch>\n' > "$@"
else
	$(AR) rcs "$@" $(LIB_OBJECTS)
endif

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
# convention dfn2/Makefile used for its own smoke/verify-vs-sample binaries. Never built under
# NO_STDIO=1, same reason ainr.o itself is excluded above.
CORE_SOURCES := ../../lib/ainr/dfn2/dfn2_process.c \
	../../lib/ainr/dfn2/dfn2_model_io.c \
	../../lib/ainr/dfn2/dfn2_prepost.c
CORE_HEADERS := ../../lib/ainr/dfn2/dfn2_grid.h \
	../../lib/ainr/dfn2/dfn2_process.h \
	../../lib/ainr/dfn2/dfn2_model_io.h \
	../../lib/ainr/dfn2/dfn2_model_io_priv.h \
	../../lib/ainr/dfn2/dfn2_prepost.h

AINR_TEST_BIN := $(OBJ_DIR)/test_ainr_dfn2
AINR_TEST_SOURCES := tests/test_ainr_dfn2.c $(CORE_SOURCES) dfn2/dfn2_stream.c ainr.c

$(AINR_TEST_BIN): $(AINR_TEST_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
ifeq ($(NO_STDIO),1)
	$(error test-ainr-dfn2 needs the file-loader entry points; run with NO_STDIO=0)
endif
ifeq ($(MASK_ONLY),1)
	$(error test-ainr-dfn2 is full-variant only (needs real catalog grids to cross-check against \
dfn2_stream_* directly) -- run with MASK_ONLY=0, or use test-link-only-ainr which already covers \
the lite build, or dfn2/Makefile's own test-variant-guard)
endif
# AINR_ERB_MATRIX_DIR points ainr_init() at this config's OBJ_DIR (tests/test_ainr_dfn2.c).
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) \
		-DAINR_ERB_MATRIX_DIR='"$(OBJ_DIR)"' \
		$(AINR_TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$@"

test-ainr-dfn2: $(AINR_TEST_BIN)
	"$(AINR_TEST_BIN)"

# Host-only regression suite for the "_spec"/"_freq" families (tests/test_ainr_spec_freq.c's own
# top comment) -- same build shape as test-ainr-dfn2 above (full-variant only, same reason: needs
# real catalog grids to cross-check against). AINR_ERB_MATRIX_DIR is unused by this suite (every init goes through
# the explicit-path *_from_erb_file entry points) but passed anyway for build-command symmetry.
AINR_SPEC_FREQ_TEST_BIN := $(OBJ_DIR)/test_ainr_spec_freq
AINR_SPEC_FREQ_TEST_SOURCES := tests/test_ainr_spec_freq.c $(CORE_SOURCES) dfn2/dfn2_stream.c ainr.c

$(AINR_SPEC_FREQ_TEST_BIN): $(AINR_SPEC_FREQ_TEST_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
ifeq ($(NO_STDIO),1)
	$(error test-ainr-spec-freq needs the file-loader entry points; run with NO_STDIO=0)
endif
ifeq ($(MASK_ONLY),1)
	$(error test-ainr-spec-freq is full-variant only (needs real catalog grids to cross-check \
against) -- run with MASK_ONLY=0)
endif
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) \
		$(AINR_SPEC_FREQ_TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$@"

test-ainr-spec-freq: $(AINR_SPEC_FREQ_TEST_BIN)
	"$(AINR_SPEC_FREQ_TEST_BIN)"

# Variant-agnostic host suites (they pick the full or the lite catalog rows by DFN2_MASK_ONLY): the
# AinrWork shapes of every registered config through real preprocess calls (test_ainr_work_shapes.c), and the
# end-of-input tail through all three families (test_ainr_tail.c). Never built under NO_STDIO.
GENERIC_TEST_SOURCES := $(CORE_SOURCES) dfn2/dfn2_stream.c ainr.c

define GENERIC_TEST_RULE
$$(OBJ_DIR)/$(1): tests/$(1).c $$(GENERIC_TEST_SOURCES) $$(CORE_HEADERS) $$(STREAM_HEADERS) $$(COMMON_LIB) $$(COMMON_HEADERS) | _cfg_guard
ifeq ($$(NO_STDIO),1)
	$$(error $(1) needs the file-loader entry points; run with NO_STDIO=0)
endif
	$$(CC) $$(CFLAGS) $$(SIMD_CPPFLAGS) $$(MASK_CPPFLAGS) tests/$(1).c $$(GENERIC_TEST_SOURCES) \
		"$$(COMMON_LIB)" $$(LDFLAGS) -lm -o "$$@"
endef
$(eval $(call GENERIC_TEST_RULE,test_ainr_work_shapes))
$(eval $(call GENERIC_TEST_RULE,test_ainr_tail))
$(eval $(call GENERIC_TEST_RULE,test_ainr_freq_nonfinite))
$(eval $(call GENERIC_TEST_RULE,test_ainr_state_generation))
$(eval $(call GENERIC_TEST_RULE,test_ainr_pool_gates))
$(eval $(call GENERIC_TEST_RULE,test_ainr_atten_lim))

test-ainr-work-shapes: $(OBJ_DIR)/test_ainr_work_shapes
	"$(OBJ_DIR)/test_ainr_work_shapes"

test-ainr-tail: $(OBJ_DIR)/test_ainr_tail
	"$(OBJ_DIR)/test_ainr_tail"

test-ainr-freq-nonfinite: $(OBJ_DIR)/test_ainr_freq_nonfinite
	"$(OBJ_DIR)/test_ainr_freq_nonfinite"

test-ainr-state-generation: $(OBJ_DIR)/test_ainr_state_generation
	"$(OBJ_DIR)/test_ainr_state_generation"

test-ainr-pool-gates: $(OBJ_DIR)/test_ainr_pool_gates
	"$(OBJ_DIR)/test_ainr_pool_gates"

test-ainr-atten-lim: $(OBJ_DIR)/test_ainr_atten_lim
	"$(OBJ_DIR)/test_ainr_atten_lim"

# work-dumper: builds tests/dump_work.c for this variant and prints the binary path. It prints the init result
# and the AinrWork views (counts, shapes, element counts) of one config from real preprocess calls, for the
# Python pins and the sample model-config check to compare against the exporter.
$(OBJ_DIR)/dump_work: tests/dump_work.c $(GENERIC_TEST_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
ifeq ($(NO_STDIO),1)
	$(error work-dumper needs the file-loader entry points; run with NO_STDIO=0)
endif
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) tests/dump_work.c $(GENERIC_TEST_SOURCES) \
		"$(COMMON_LIB)" $(LDFLAGS) -lm -o "$@"

work-dumper: $(OBJ_DIR)/dump_work
	@echo "$(abspath $(OBJ_DIR))/dump_work"

# test-ainr-product: ainr.c compiled with a different AINR_PRODUCT_MODEL_ID (the product binding,
# ainr_product.h) in three extra test binaries -- the only way a test reaches the quantization
# alternates, since the customer config has no model field. Ids per archive variant: alternate 2
# (16x8), alternate 1 (8x8, not registered), and the other variant's default id.
ifeq ($(MASK_ONLY),1)
PRODUCT_ID_ALT2 := 21
PRODUCT_ID_ALT1 := 20
PRODUCT_ID_OTHER := 1
else
PRODUCT_ID_ALT2 := 11
PRODUCT_ID_ALT1 := 10
PRODUCT_ID_OTHER := 2
endif
test-ainr-product: | _cfg_guard
ifeq ($(NO_STDIO),1)
	$(error test-ainr-product needs the file-loader entry points; run with NO_STDIO=0)
endif
	@set -e; for spec in "1 $(PRODUCT_ID_ALT2)" "2 $(PRODUCT_ID_ALT1)" "3 $(PRODUCT_ID_OTHER)"; do \
	  set -- $$spec; \
	  $(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) -DAINR_TEST_PRODUCT=$$1 -DAINR_PRODUCT_MODEL_ID=$$2 \
	    tests/test_ainr_product.c $(GENERIC_TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm \
	    -o "$(OBJ_DIR)/test_ainr_product_$$1"; \
	  "$(OBJ_DIR)/test_ainr_product_$$1"; \
	done

# test-ainr-model-io: the DFN2 graph rows (ai3_model_io_dfn2.h) against the AinrWork of real preprocess calls
# for this variant, with the default product id and the three others of ainr_product.h (alt2, alt1, the other
# variant's). The port check of the model-config header links ../common/stream_bind.c (samples only, not the archive).
test-ainr-model-io: $(GENERIC_TEST_SOURCES) tests/test_ainr_model_io.c ai3_model_io_dfn2.h ../common/ai3_model_io.h $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
ifeq ($(NO_STDIO),1)
	$(error test-ainr-model-io needs the file-loader entry points; run with NO_STDIO=0)
endif
	@set -e; for spec in "0 " "1 -DAINR_PRODUCT_MODEL_ID=$(PRODUCT_ID_ALT2)" "2 -DAINR_PRODUCT_MODEL_ID=$(PRODUCT_ID_ALT1)" \
	    "3 -DAINR_PRODUCT_MODEL_ID=$(PRODUCT_ID_OTHER)"; do \
	  set -- $$spec; tag=$$1; shift; \
	  $(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(MASK_CPPFLAGS) "$$@" tests/test_ainr_model_io.c $(GENERIC_TEST_SOURCES) \
	    ../common/stream_bind.c "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$(OBJ_DIR)/test_ainr_model_io_$$tag"; \
	  "$(OBJ_DIR)/test_ainr_model_io_$$tag"; \
	done

# model-io-dumper: tests/dump_model_io.c prints the rows of ai3_model_io_dfn2.h and the tables' layout constants;
# the Python pin AINR/tests/test_dfn2_port_order.py compares them with a fresh export. Prints the binary path.
$(OBJ_DIR)/dump_model_io: tests/dump_model_io.c ai3_model_io_dfn2.h ../common/ai3_model_io.h ../common/stream_bind.c ../common/stream_bind.h | _cfg_guard
	$(CC) $(CFLAGS) tests/dump_model_io.c ../common/stream_bind.c -lm -o "$@"

model-io-dumper: $(OBJ_DIR)/dump_model_io
	@echo "$(abspath $(OBJ_DIR))/dump_model_io"

# test-link-only-ainr: builds BOTH full and mask-only dfn2/ archives (sub-make, as ainr.c is built here), then
# links tests/link_only_ainr.c -- which #includes ONLY ainr.h -- against each; both must complete a real round trip.
AINR_SYMBOLS := ainr_get_mem_size ainr_init_from_erb_file ainr_init ainr_get_mem_size_spec \
	ainr_init_spec_from_erb_file ainr_init_spec ainr_get_mem_size_freq ainr_init_freq_from_erb_file \
	ainr_init_freq

test-link-only-ainr:
ifeq ($(NO_STDIO),1)
	$(error test-link-only-ainr needs the file-loader entry points; run with NO_STDIO=0)
endif
	@$(MAKE) --no-print-directory -C dfn2 MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@$(MAKE) --no-print-directory -C dfn2 MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@$(MAKE) --no-print-directory MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@$(MAKE) --no-print-directory MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@full_dfn2_lib="$$($(MAKE) -s --no-print-directory -C dfn2 MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	lite_dfn2_lib="$$($(MAKE) -s --no-print-directory -C dfn2 MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	full_ainr_info="$$($(MAKE) -s --no-print-directory MASK_ONLY=0 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path print-cflags)"; \
	lite_ainr_info="$$($(MAKE) -s --no-print-directory MASK_ONLY=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path print-cflags)"; \
	full_ainr_lib="$$(printf '%s\n' "$$full_ainr_info" | sed -n 1p)"; \
	full_cflags="$$(printf '%s\n' "$$full_ainr_info" | sed -n 2p)"; \
	lite_ainr_lib="$$(printf '%s\n' "$$lite_ainr_info" | sed -n 1p)"; \
	lite_cflags="$$(printf '%s\n' "$$lite_ainr_info" | sed -n 2p)"; \
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
test: test-ainr-work-shapes test-ainr-tail test-ainr-freq-nonfinite test-ainr-state-generation test-ainr-pool-gates test-ainr-atten-lim \
	test-ainr-product test-ainr-model-io test-link-only-ainr
else
test: test-ainr-dfn2 test-ainr-spec-freq test-ainr-work-shapes test-ainr-tail test-ainr-freq-nonfinite \
	test-ainr-state-generation test-ainr-pool-gates test-ainr-atten-lim test-ainr-product test-ainr-model-io test-link-only-ainr
endif

# audit-no-stdio: builds the NO_STDIO=1 archive here and asserts, via `nm`, that it exports no
# ainr_* symbol at all (ainr.o is excluded from LIB_OBJECTS under NO_STDIO=1, its only entry point
# is file-mode) -- with a positive control (the NO_STDIO=0 build) proving the check can actually
# detect a present symbol, so an empty result isn't silently vacuous.
audit-no-stdio:
	@$(MAKE) --no-print-directory NO_STDIO=1 MASK_ONLY=$(MASK_ONLY) SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@$(MAKE) --no-print-directory NO_STDIO=0 MASK_ONLY=$(MASK_ONLY) SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@nostdio_lib="$$($(MAKE) -s --no-print-directory NO_STDIO=1 MASK_ONLY=$(MASK_ONLY) SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	stdio_lib="$$($(MAKE) -s --no-print-directory NO_STDIO=0 MASK_ONLY=$(MASK_ONLY) SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	ainr_pattern='[Tt] _?ainr_'; \
	stdio_ainr_syms="$$(nm "$$stdio_lib" 2>/dev/null | grep -iE "$$ainr_pattern" || true)"; \
	if [ -z "$$stdio_ainr_syms" ]; then \
	  echo "FAIL: positive control broken -- $$stdio_lib (NO_STDIO=0) exports no ainr_* symbols either, so the check below cannot prove anything" >&2; exit 1; \
	fi; \
	nostdio_ainr_syms="$$(nm "$$nostdio_lib" 2>/dev/null | grep -iE "$$ainr_pattern" || true)"; \
	if [ -n "$$nostdio_ainr_syms" ]; then \
	  echo "FAIL: $$nostdio_lib exports ainr_* symbols under NO_STDIO=1 (should be absent):" >&2; echo "$$nostdio_ainr_syms" >&2; exit 1; \
	fi; \
	echo "audit-no-stdio: PASS ($$nostdio_lib exports no ainr_* -- confirmed detectable via $$stdio_lib's positive control)"

clean:
	rm -rf "$(BUILD_DIR)"
	+$(MAKE) --no-print-directory -C dfn2 clean
