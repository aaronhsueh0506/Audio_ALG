CC  ?= cc
CXX ?= c++
AR  ?= ar
RANLIB ?= ranlib
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Wpedantic
# Same hook/convention as the per-model stream Makefiles (deepvqe_s, ulcnet): reaches this
# directory's own compiles and every sub-make below.
EXTRA_CFLAGS ?=
override CFLAGS += -ffp-contract=off -fno-math-errno -I. -Ideepvqe_s -I../../lib/aiaec/deepvqe_s -Iulcnet -I../../lib/aiaec/ulcnet -I../common $(EXTRA_CFLAGS)
LDFLAGS ?=
SIMD ?= 1
override SIMD := $(strip $(SIMD))
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): ne10 is the only FFT backend)
endif

# aiaec.c has no file-loader TU -- nothing here needs stdio. Same build-matrix knob as every
# other stream Makefile.
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

# AIAEC_MODEL: which model(s) aiaec.c is compiled for -- both (default), deepvqe (DeepVQE-S only),
# or align (Align-ULCNet only). A single-model build's aiaec.o references none of the other model's
# deepvqe_stream_*/ulcnet_stream_* symbols at all (see aiaec.c's own top comment); make
# test-single-model proves both single-model variants link and run, and that a product bound to the
# model the build lacks is refused before anything else. aiaec.h is identical across every variant --
# this knob only changes aiaec.c's own preprocessor flags. The product binding (aiaec_product.h)
# defaults to the one enabled model, or DeepVQE-S when both are.
AIAEC_MODEL ?= both
override AIAEC_MODEL := $(strip $(AIAEC_MODEL))
ifeq ($(AIAEC_MODEL),both)
AIAEC_MODEL_CPPFLAGS := -DAIAEC_ENABLE_DEEPVQE_S=1 -DAIAEC_ENABLE_ALIGN_ULCNET=1
else ifeq ($(AIAEC_MODEL),deepvqe)
AIAEC_MODEL_CPPFLAGS := -DAIAEC_ENABLE_DEEPVQE_S=1 -DAIAEC_ENABLE_ALIGN_ULCNET=0
else ifeq ($(AIAEC_MODEL),align)
AIAEC_MODEL_CPPFLAGS := -DAIAEC_ENABLE_DEEPVQE_S=0 -DAIAEC_ENABLE_ALIGN_ULCNET=1
else
$(error AIAEC_MODEL must be one of: both, deepvqe, align (got '$(AIAEC_MODEL)'))
endif
override CFLAGS += $(AIAEC_MODEL_CPPFLAGS)

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

# Forwarded to every child model directory's Makefile (deepvqe_s, ulcnet): same shape as
# pipelines/Makefile's own FWD, scoped to what those Makefiles actually accept.
FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) NO_STDIO=$(NO_STDIO) $(TOOLCHAIN_FWD)

BUILD_DIR := build
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) SIMD=$(SIMD) SIMD_CPPFLAGS=$(SIMD_CPPFLAGS) NO_STDIO=$(NO_STDIO) BACKEND=$(BACKEND) WERROR=$(WERROR) AIAEC_MODEL=$(AIAEC_MODEL) COMMON_LIB=$(COMMON_LIB)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
OBJ_DIR := $(BUILD_DIR)/simd$(SIMD)-$(CFG_SIG)

# aiaec.c dispatches across model directories (DeepVQE-S and Align-ULCNet) -- it lives here, one
# level above both, instead of inside either one's own build. It only needs their headers, never
# their core sources: no CORE_SOURCES in this Makefile.
STREAM_SOURCES := aiaec.c
STREAM_HEADERS := aiaec.h aiaec_internal.h aiaec_product.h deepvqe_s/deepvqe_stream.h ../../lib/aiaec/deepvqe_s/deepvqe_prepost.h \
	ulcnet/ulcnet_stream.h ../../lib/aiaec/ulcnet/ulcnet_prepost.h \
	../../lib/aiaec/ulcnet/ulcnet_model_io.h ../common/stream_mem.h

LIB := $(OBJ_DIR)/libaiaec.a
LIB_OBJECTS := $(OBJ_DIR)/aiaec.o
VPATH := .

.PHONY: all lib print-lib-path print-cflags print-sources test deepvqe-lib ulcnet-lib \
	test-aiaec-catalog test-aiaec-block test-aiaec-spec-freq test-aiaec-ports test-aiaec-product test-aiaec-packet test-aiaec-model-io link-only-aiaec \
	bind-dumpers model-io-dumper packet-driver test-single-model audit-no-stdio clean _cfg_guard

all: lib

lib: $(LIB)

print-lib-path:
	@echo "$(abspath $(LIB))"

print-cflags:
	@printf '%s\n' "$(CFLAGS) $(SIMD_CPPFLAGS)"

print-sources:
	@printf '%s\n' "$(STREAM_SOURCES)"

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

$(OBJ_DIR)/%.o: %.c $(STREAM_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -MMD -MP -c -o "$@" "$<"

-include $(LIB_OBJECTS:.o=.d)

COMMON_HEADERS := $(wildcard $(COMMON_DIR)/include/*.h $(COMMON_DIR)/utility/*.h)

# deepvqe-lib/ulcnet-lib: sub-make each model directory for its own archive, same orchestration
# shape as pipelines/Makefile's COMMON_LIB/AEC_LIB/NR_LIB resolution (print-lib-path then lib),
# never recompiling a model's core sources into this directory. ulcnet-lib is built here because
# aiaec.o references ulcnet_stream_* in a build that enables both models. Resolved lazily inside
# recipes (never as a top-level := shell call) so a plain
# `make clean`/`print-cflags` here does not trigger a full sub-build, same reasoning as
# lib/deps.mk's RESOLVE_COMMON_LIB skipping itself for clean.
deepvqe-lib:
	+$(MAKE) --no-print-directory -C deepvqe_s $(FWD) lib

ulcnet-lib:
	+$(MAKE) --no-print-directory -C ulcnet $(FWD) lib

# Model cores and streams that any binary compiling aiaec.c from source needs: aiaec.c references
# both models' streams (in a build that enables both), exactly like linking libaiaec.a does.
DEEPVQE_CORE_SOURCES := ../../lib/aiaec/deepvqe_s/deepvqe_process.c \
	../../lib/aiaec/deepvqe_s/deepvqe_prepost.c
DEEPVQE_CORE_HEADERS := ../../lib/aiaec/deepvqe_s/deepvqe_process.h \
	../../lib/aiaec/deepvqe_s/deepvqe_prepost.h
ULCNET_CORE_SOURCES := ../../lib/aiaec/ulcnet/ulcnet_model_io.c \
	../../lib/aiaec/ulcnet/ulcnet_prepost.c
ULCNET_CORE_HEADERS := ../../lib/aiaec/ulcnet/ulcnet_model_io.h \
	../../lib/aiaec/ulcnet/ulcnet_process.h \
	../../lib/aiaec/ulcnet/ulcnet_prepost.h
ULCNET_STREAM_SOURCES := ulcnet/ulcnet_stream.c
ULCNET_STREAM_HEADERS := ulcnet/ulcnet_stream.h ulcnet/ulcnet_stream_internal.h

TEST_SOURCES := $(DEEPVQE_CORE_SOURCES) deepvqe_s/deepvqe_stream.c \
	$(ULCNET_CORE_SOURCES) $(ULCNET_STREAM_SOURCES) aiaec.c
TEST_DEPS := $(DEEPVQE_CORE_HEADERS) $(ULCNET_CORE_HEADERS) $(ULCNET_STREAM_HEADERS) \
	$(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS)

# The product a test binary is built for is chosen by compiling aiaec.c with
# -DAIAEC_PRODUCT_MODEL_ID=<id> (aiaec_product.h) -- the only way a test reaches the other model;
# no customer field selects a model. AIAEC_TEST_CATALOG (aiaec.c) substitutes placeholder graph
# paths so a test can drive the gated API through rows the production catalog does not deliver; it
# is passed only to the targets below that say so, never to `lib` or the catalog/link-only tests.
PRODUCTS := deepvqe_s=1 align_ulcnet=10

# One test source built and run once per product. $(1) test name, $(2) extra compile flags.
define RUN_PER_PRODUCT
	@set -e; for spec in $(PRODUCTS); do \
	  tag=$${spec%%=*}; id=$${spec##*=}; \
	  $(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(2) -DAIAEC_PRODUCT_MODEL_ID=$$id -Ideepvqe_s -Iulcnet \
	    tests/$(1).c $(TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$(OBJ_DIR)/$(1)_$$tag"; \
	  "$(OBJ_DIR)/$(1)_$$tag"; \
	done
endef

# test_aiaec_catalog: the PRODUCTION catalog (no AIAEC_TEST_CATALOG): refusal order, the 16k/256
# rows without a delivered graph, and the one delivered 16k/512 row.
test-aiaec-catalog: $(TEST_DEPS) | _cfg_guard
	$(call RUN_PER_PRODUCT,test_aiaec_catalog,)

# test_aiaec_block: the block entry end to end against deepvqe_stream_*/ulcnet_stream_* driven
# directly, with the test catalog.
test-aiaec-block: $(TEST_DEPS) | _cfg_guard
	$(call RUN_PER_PRODUCT,test_aiaec_block,-DAIAEC_TEST_CATALOG)

# test_aiaec_spec_freq: the _spec and _freq families against the PCM block family and an external
# AcIstft/AcStft, the tail drain, per-family info and pool sizes, with the test catalog.
test-aiaec-spec-freq: $(TEST_DEPS) | _cfg_guard
	$(call RUN_PER_PRODUCT,test_aiaec_spec_freq,-DAIAEC_TEST_CATALOG)

# test_aiaec_packet: Align-ULCNet's packet graph (N=8, layout 19) against a host accelerator that implements the
# graph's frame_reset semantics; the Align-ULCNet product only (the test catalog opens its N=8 row).
test-aiaec-packet: $(TEST_DEPS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -DAIAEC_TEST_CATALOG -DAIAEC_PRODUCT_MODEL_ID=10 -Ideepvqe_s -Iulcnet \
	  tests/test_aiaec_packet.c $(TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$(OBJ_DIR)/test_aiaec_packet"
	"$(OBJ_DIR)/test_aiaec_packet"

# test_aiaec_ports: the AiaecWork signal and head views of real pre calls against the exporters' tables, with the test catalog.
test-aiaec-ports: $(TEST_DEPS) | _cfg_guard
	$(call RUN_PER_PRODUCT,test_aiaec_ports,-DAIAEC_TEST_CATALOG)

# test_aiaec_product: the product binding (1 DeepVQE-S, 2 Align-ULCNet, 3 an id naming no model);
# builds 1 and 2 report the pool size of the same config, which must differ.
test-aiaec-product: $(TEST_DEPS) | _cfg_guard
	@set -e; for spec in "1 1" "2 10" "3 999"; do \
	  set -- $$spec; \
	  $(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -DAIAEC_TEST_CATALOG -DAIAEC_TEST_PRODUCT=$$1 -DAIAEC_PRODUCT_MODEL_ID=$$2 \
	    -Ideepvqe_s -Iulcnet tests/test_aiaec_product.c $(TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm \
	    -o "$(OBJ_DIR)/test_aiaec_product_$$1"; \
	  "$(OBJ_DIR)/test_aiaec_product_$$1" > "$(OBJ_DIR)/test_aiaec_product_$$1.out"; \
	  cat "$(OBJ_DIR)/test_aiaec_product_$$1.out"; \
	done; \
	h1="$$(grep '^pool ' "$(OBJ_DIR)/test_aiaec_product_1.out" | sed 's/product=[0-9]* //')"; \
	h2="$$(grep '^pool ' "$(OBJ_DIR)/test_aiaec_product_2.out" | sed 's/product=[0-9]* //')"; \
	[ -n "$$h1" ] && [ -n "$$h2" ] && [ "$$h1" != "$$h2" ] || { echo "FAIL: the two products' pool sizes are equal or missing ($$h1 / $$h2)" >&2; exit 1; }; \
	echo "test-aiaec-product: PASS (pool sizes differ: $$h1 vs $$h2)"

# test-aiaec-model-io: the graph rows (ai3_model_io_deepvqes.h, ai3_model_io_alignulcnet.h) against the
# AiaecWork of real pre calls, per product, with the test catalog and the production one. The port check of the
# model-config header links ../common/stream_bind.c (samples only, not the archive).
test-aiaec-model-io: $(TEST_DEPS) tests/test_aiaec_model_io.c ai3_model_io_deepvqes.h ai3_model_io_alignulcnet.h ../common/ai3_model_io.h | _cfg_guard
	@set -e; for spec in $(PRODUCTS); do \
	  tag=$${spec%%=*}; id=$${spec##*=}; \
	  for cat in test production; do \
	    defs=""; [ "$$cat" = test ] && defs="-DAIAEC_TEST_CATALOG"; \
	    $(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $$defs -DAIAEC_PRODUCT_MODEL_ID=$$id -Ideepvqe_s -Iulcnet \
	      tests/test_aiaec_model_io.c $(TEST_SOURCES) ../common/stream_bind.c "$(COMMON_LIB)" $(LDFLAGS) -lm \
	      -o "$(OBJ_DIR)/test_aiaec_model_io_$${tag}_$$cat"; \
	    "$(OBJ_DIR)/test_aiaec_model_io_$${tag}_$$cat"; \
	  done; \
	done

# model-io-dumper: tests/dump_model_io.c prints the rows of both model-config headers and their layout constants;
# AIAEC/tests/test_aiaec_bind_info_pin.py compares them with a fresh export. Prints the binary path.
$(OBJ_DIR)/dump_model_io: tests/dump_model_io.c ai3_model_io_deepvqes.h ai3_model_io_alignulcnet.h ../common/ai3_model_io.h ../common/stream_bind.c ../common/stream_bind.h | _cfg_guard
	$(CC) $(CFLAGS) tests/dump_model_io.c ../common/stream_bind.c -lm -o "$@"

model-io-dumper: $(OBJ_DIR)/dump_model_io
	@echo "$(abspath $(OBJ_DIR))/dump_model_io"

# bind-dumpers: builds tests/dump_bind_info.c once per product with the test catalog and prints the
# two binary paths (deepvqe_s first). AIAEC/tests/test_aiaec_bind_info_pin.py runs them and compares
# the AiaecWork signal and head views of a real pre call with the ONNX export.
bind-dumpers: $(TEST_DEPS) | _cfg_guard
	@set -e; for spec in $(PRODUCTS); do \
	  tag=$${spec%%=*}; id=$${spec##*=}; \
	  $(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -DAIAEC_TEST_CATALOG -DAIAEC_PRODUCT_MODEL_ID=$$id -Ideepvqe_s -Iulcnet \
	    tests/dump_bind_info.c $(TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$(OBJ_DIR)/dump_bind_info_$$tag"; \
	done
	@echo "$(abspath $(OBJ_DIR))/dump_bind_info_deepvqe_s"
	@echo "$(abspath $(OBJ_DIR))/dump_bind_info_align_ulcnet"

# packet-driver: tests/packet_driver.c (the Align-ULCNet _freq family behind a binary stdin/stdout protocol, N chosen
# per request) with the test catalog; prints the binary path for AIAEC/tests/test_aiaec_packet_ort.py.
packet-driver: $(TEST_DEPS) | _cfg_guard
	@$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -DAIAEC_TEST_CATALOG -DAIAEC_PRODUCT_MODEL_ID=10 -Ideepvqe_s -Iulcnet \
	  tests/packet_driver.c $(TEST_SOURCES) "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$(OBJ_DIR)/packet_driver"
	@echo "$(abspath $(OBJ_DIR))/packet_driver"

# link_only_aiaec: only #includes aiaec.h and links against the already-built libaiaec.a +
# deepvqe_s's own libdeepvqe_stream.a + ulcnet's own libulcnet_stream.a (not recompiling any of
# their sources) -- confirms the shipped archives actually export the customer entry points and
# that deepvqe_stream_*/ulcnet_stream_* are unaffected, same spirit as stream/ainr/dfn2's
# link_only_ainr.c. libulcnet_stream.a is in the link because a build that enables both models
# references ulcnet_stream_* from aiaec.o whichever model the product binds.
LINK_ONLY_AIAEC_BIN := $(OBJ_DIR)/link_only_aiaec

# Struct-taking entries aiaec.h exports.
AIAEC_STRUCT_SYMS := aiaec_get_mem_size aiaec_init aiaec_preprocess \
	aiaec_get_mem_size_spec aiaec_init_spec aiaec_preprocess_spec \
	aiaec_get_mem_size_freq aiaec_init_freq aiaec_preprocess_freq

$(LINK_ONLY_AIAEC_BIN): tests/link_only_aiaec.c aiaec.h $(LIB) deepvqe-lib ulcnet-lib $(COMMON_LIB) | _cfg_guard
	@dv="$$($(MAKE) -s --no-print-directory -C deepvqe_s $(FWD) print-lib-path)"; \
	ul="$$($(MAKE) -s --no-print-directory -C ulcnet $(FWD) print-lib-path)"; \
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) tests/link_only_aiaec.c "$(LIB)" "$$dv" "$$ul" "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

link-only-aiaec: $(LINK_ONLY_AIAEC_BIN)
	"$(LINK_ONLY_AIAEC_BIN)"
	@dv="$$($(MAKE) -s --no-print-directory -C deepvqe_s $(FWD) print-lib-path)"; \
	ul="$$($(MAKE) -s --no-print-directory -C ulcnet $(FWD) print-lib-path)"; \
	for sym in $(AIAEC_STRUCT_SYMS); do \
	  nm "$(LIB)" 2>/dev/null | grep -Eq " T _?$$sym$$" || { echo "FAIL: $$sym not exported by $(LIB)" >&2; exit 1; }; \
	done; \
	for sym in aiaec_status_text aiaec_postprocess aiaec_postprocess_spec aiaec_postprocess_freq; do \
	  nm "$(LIB)" 2>/dev/null | grep -Eq " T _?$$sym$$" || { echo "FAIL: $$sym not exported by $(LIB)" >&2; exit 1; }; \
	done; \
	if nm "$(LIB)" 2>/dev/null | grep -Eq 'stream_check_ports|stream_port_shape'; then \
	  echo "FAIL: $(LIB) carries the sample-side port check (stream_bind)" >&2; exit 1; \
	fi; \
	nm "$$dv" 2>/dev/null | grep -Eq ' T _?deepvqe_stream_post_process_freq$$' || { echo "FAIL: deepvqe_stream_post_process_freq not exported by $$dv" >&2; exit 1; }; \
	for sym in ulcnet_stream_pre_process ulcnet_stream_pre_process_f32 ulcnet_stream_pre_process_freq ulcnet_stream_pre_process_packet_f32 ulcnet_stream_pre_process_packet_freq ulcnet_stream_get_mem_size ulcnet_prepost_frame_open ulcnet_prepost_init ulcnet_model_io_prepare ulcnet_model_io_prepare_packet ulcnet_model_io_descriptor_validate; do \
	  nm "$$ul" 2>/dev/null | grep -Eq " T _?$$sym$$" || { echo "FAIL: $$sym not exported by $$ul" >&2; exit 1; }; \
	done; \
	nm "$$ul" 2>/dev/null | grep -Eq ' T _?ulcnet_stream_restart_mask$$' || { echo "FAIL: ulcnet_stream_restart_mask not exported by $$ul" >&2; exit 1; }; \
	nm "$$dv" 2>/dev/null | grep -Eq ' T _?deepvqe_stream_init$$' || { echo "FAIL: deepvqe_stream_init not exported by $$dv" >&2; exit 1; }; \
	nm "$$ul" 2>/dev/null | grep -Eq ' T _?ulcnet_stream_init$$' || { echo "FAIL: ulcnet_stream_init not exported by $$ul" >&2; exit 1; }; \
	echo "link-only-aiaec: PASS ($(LIB) exports aiaec_*, $$dv exports deepvqe_stream_*, $$ul exports ulcnet_stream_*)"

# The per-product tests above compile aiaec.c with both models enabled and reach the second model
# through -DAIAEC_PRODUCT_MODEL_ID; they are not the right fixtures under a single-model
# AIAEC_MODEL, and silently skipping sections inside them would hide real coverage loss. A
# single-model `test` runs only `lib` (confirms aiaec.o itself still compiles clean under this
# variant) and says plainly what covers that variant's contract: test-single-model.
ifeq ($(AIAEC_MODEL),both)
test: test-aiaec-catalog test-aiaec-block test-aiaec-spec-freq test-aiaec-ports test-aiaec-product test-aiaec-packet test-aiaec-model-io link-only-aiaec ulcnet-lib
else
test: lib
	@echo "test: AIAEC_MODEL=$(AIAEC_MODEL) -- the per-product tests and link-only-aiaec assume both models and are skipped; run 'make test-single-model' for this variant's coverage (built with AIAEC_MODEL=both, which exercises AIAEC_MODEL=deepvqe and =align internally)."
endif

# test-single-model: proves the AIAEC_MODEL=deepvqe and AIAEC_MODEL=align single-model variants
# (see the AIAEC_MODEL knob's own comment above) each compile an aiaec.o with none of the OTHER
# model's deepvqe_stream_*/ulcnet_stream_* symbols (nm -u, against the same object `make lib`
# would archive -- no AIAEC_TEST_CATALOG on that compile), that the real public block API runs a
# normal hop and a forced-bypass hop end to end for the enabled model, and that a product bound to
# the OTHER model (-DAIAEC_PRODUCT_MODEL_ID=<its id>) is refused with AIAEC_ERR_MODEL_ID by every
# entry (see tests/test_single_model.c). Deliberately NOT a
# dependency of `test` above (same reasoning as audit-no-stdio): it always exercises both
# variants via its own recursive sub-make regardless of the AIAEC_MODEL this invocation used, so
# folding it into the default target would silently double- or triple-build aiaec.o on every
# plain `make test`.
test-single-model:
	@$(MAKE) --no-print-directory AIAEC_MODEL=deepvqe SIMD=$(SIMD) $(TOOLCHAIN_FWD) _test_single_model_variant EXPECT_OTHER=ulcnet OTHER_ID=10
	@$(MAKE) --no-print-directory AIAEC_MODEL=align SIMD=$(SIMD) $(TOOLCHAIN_FWD) _test_single_model_variant EXPECT_OTHER=deepvqe OTHER_ID=1

.PHONY: _test_single_model_variant
_test_single_model_variant: $(OBJ_DIR)/aiaec.o
	@set -e; case "$(AIAEC_MODEL)" in \
	  deepvqe) sources="$(DEEPVQE_CORE_SOURCES) deepvqe_s/deepvqe_stream.c" ;; \
	  align)   sources="$(ULCNET_CORE_SOURCES) $(ULCNET_STREAM_SOURCES)" ;; \
	  *) echo "FATAL: _test_single_model_variant needs AIAEC_MODEL=deepvqe or align, got '$(AIAEC_MODEL)'" >&2; exit 1 ;; \
	esac; \
	undef_other="$$(nm -u "$(OBJ_DIR)/aiaec.o" 2>/dev/null | grep -i '$(EXPECT_OTHER)' || true)"; \
	if [ -n "$$undef_other" ]; then \
	  echo "FAIL: $(AIAEC_MODEL)-only aiaec.o references $(EXPECT_OTHER)_* symbols:" >&2; echo "$$undef_other" >&2; exit 1; \
	fi; \
	bin="$(OBJ_DIR)/test_single_model"; \
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -DAIAEC_TEST_CATALOG -Ideepvqe_s -Iulcnet \
	  tests/test_single_model.c $$sources aiaec.c "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$$bin"; \
	"$$bin"; \
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -DAIAEC_TEST_CATALOG -DAIAEC_TEST_OTHER_BOUND=1 -DAIAEC_PRODUCT_MODEL_ID=$(OTHER_ID) \
	  -Ideepvqe_s -Iulcnet tests/test_single_model.c $$sources aiaec.c "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$$bin.other"; \
	"$$bin.other"; \
	echo "test-single-model ($(AIAEC_MODEL)): PASS (no $(EXPECT_OTHER)_* symbols, round trip + bound-model refusal verified)"

# audit-no-stdio: aiaec.o itself has no stdio references (same convention as every other stream
# Makefile's audit-no-stdio, scoped to just this directory's own object -- deepvqe_s/ and ulcnet/
# each audit their own).
audit-no-stdio:
	@$(MAKE) --no-print-directory NO_STDIO=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@obj_dir="$$($(MAKE) -s --no-print-directory NO_STDIO=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	obj_dir="$$(dirname "$$obj_dir")"; \
	undef="$$(nm -u "$$obj_dir/aiaec.o" 2>/dev/null | grep -Ei 'fopen|fread|fclose|fwrite|fprintf|printf|puts' || true)"; \
	if [ -n "$$undef" ]; then \
	  echo "FAIL: aiaec.o references stdio symbols:" >&2; echo "$$undef" >&2; exit 1; \
	fi; \
	echo "audit-no-stdio: PASS ($$obj_dir/aiaec.o has no stdio references)"

clean:
	rm -rf "$(BUILD_DIR)"
	+$(MAKE) --no-print-directory -C deepvqe_s clean
	+$(MAKE) --no-print-directory -C ulcnet clean
	+$(MAKE) --no-print-directory -C front clean
