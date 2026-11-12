CC  ?= cc
CXX ?= c++
AR  ?= ar
RANLIB ?= ranlib
CFLAGS ?= -O2 -std=gnu99 -Wall -Wextra
EXTRA_CFLAGS ?=
LDFLAGS ?=
SIMD ?= 1
override SIMD := $(strip $(SIMD))
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): ne10 is the only FFT backend)
endif
ifneq ($(words $(SIMD)),1)
$(error SIMD must be exactly 0 or 1)
endif
ifneq ($(filter-out 0 1,$(SIMD)),)
$(error SIMD must be exactly 0 or 1)
endif
PROFILE ?= 0
TOOLCHAIN_CHECK ?= 1
DEBUG ?= 0
NO_STDIO ?= 0
WERROR ?= 0
ifeq ($(WERROR),1)
override CFLAGS += -Werror
endif

# The AEC repo is a sibling dependency (lib/deps.mk resolves AEC_DIR / COMMON_DIR); this directory
# is the only part of stream/aiaec that links it. libaiaec.a has no AEC dependency.
include ../../../lib/deps.mk
$(eval $(call RESOLVE_COMMON_LIB))

override CFLAGS += -ffp-contract=off -fno-math-errno -I. -I.. -I../../common \
	-I$(AEC_DIR)/include -I$(COMMON_DIR)/include -I$(COMMON_DIR) $(EXTRA_CFLAGS)
SIMD_CPPFLAGS :=
ifeq ($(SIMD),0)
SIMD_CPPFLAGS += -DSIMD_KERNELS_FORCE_SCALAR
endif
ifeq ($(PROFILE),1)
SIMD_CPPFLAGS += -DAEC_STAGE_TIMING=1
endif

# Same forwarding as pipelines/Makefile: COMMON_DIR/COMMON_LIB must reach the AEC sub-make so its
# keyed archive path equals the one resolved here.
AEC_FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) PROFILE=$(PROFILE) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' \
	EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) DEBUG=$(DEBUG) \
	COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB="$(COMMON_LIB)"
PARENT_FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) $(TOOLCHAIN_FWD)

ifneq ($(MAKECMDGOALS),clean)
AEC_LIB := $(shell $(MAKE) -s --no-print-directory -C $(AEC_DIR) $(AEC_FWD) print-lib-path | tail -1)
ifeq ($(strip $(AEC_LIB)),)
$(error AEC_LIB resolution failed (AEC_DIR=$(AEC_DIR)))
endif
endif

BUILD_DIR := build
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
AEC_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(AEC_LIB))))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) SIMD=$(SIMD) SIMD_CPPFLAGS=$(SIMD_CPPFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) COMMON_LIB=$(COMMON_LIB) AEC_PRODUCER_SIG=$(AEC_PRODUCER_SIG)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
OBJ_DIR := $(BUILD_DIR)/simd$(SIMD)-$(CFG_SIG)

LIB := $(OBJ_DIR)/libaiaec_front.a
FRONT_HEADERS := aiaec_front.h aiaec_front_internal.h aiaec_front_test.h ../aiaec.h ../aiaec_internal.h ../../common/stream_mem.h

# The Align-ULCNet build for the front integration tests: aiaec.c built for that one model with
# placeholder graph paths (AIAEC_TEST_CATALOG), same source list as ../Makefile's single-model variant.
ALIGN_SOURCES := ../aiaec.c ../../../lib/aiaec/ulcnet/ulcnet_model_io.c ../../../lib/aiaec/ulcnet/ulcnet_prepost.c \
	../ulcnet/ulcnet_stream.c
ALIGN_CPPFLAGS := -DAIAEC_ENABLE_DEEPVQE_S=0 -DAIAEC_ENABLE_ALIGN_ULCNET=1 -DAIAEC_PRODUCT_MODEL_ID=10 \
	-DAIAEC_TEST_CATALOG -I../deepvqe_s -I../../../lib/aiaec/deepvqe_s -I../ulcnet -I../../../lib/aiaec/ulcnet

.PHONY: all lib print-lib-path test test-front-events test-front-scene test-front-block test-front-oneshot \
	audit-libaiaec-no-aec clean _cfg_guard _aec_lib

all: lib
lib: $(LIB)
print-lib-path:
	@echo "$(abspath $(LIB))"

_aec_lib:
	+$(MAKE) --no-print-directory -C $(AEC_DIR) $(AEC_FWD) lib

_cfg_guard:
	@test -n "$(CC_TARGET_SIG)" && test "$(CC_TARGET_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine compiler target." >&2; exit 1; }
	@mkdir -p "$(OBJ_DIR)"
	@m="$(OBJ_DIR)/config.manifest"; \
	if [ -f "$$m" ] && [ "$$(cat "$$m")" != "$(CFG_SIG_PAYLOAD)" ]; then \
	  echo "FATAL: configuration signature collision in $$m" >&2; exit 1; \
	fi; \
	printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$m"

$(OBJ_DIR)/aiaec_front.o: aiaec_front.c $(FRONT_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -MMD -MP -c -o "$@" "$<"
-include $(OBJ_DIR)/aiaec_front.d

$(OBJ_DIR)/aiaec_front_preprocess.o: aiaec_front_preprocess.c $(FRONT_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -MMD -MP -c -o "$@" "$<"
-include $(OBJ_DIR)/aiaec_front_preprocess.d

$(LIB): $(OBJ_DIR)/aiaec_front.o $(OBJ_DIR)/aiaec_front_preprocess.o
	rm -f "$@" && $(AR) rcs "$@" $^ && $(RANLIB) "$@"

# Tests link the C++ driver: the audiolib archive carries one C++ TU.
define BUILD_TEST
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(2) tests/$(1).c $(3) $(LIB) "$(AEC_LIB)" "$(COMMON_LIB)" $(LDFLAGS) -lm -o "$(OBJ_DIR)/$(1)"
	"$(OBJ_DIR)/$(1)"
endef

test-front-events: $(LIB) _aec_lib
	$(call BUILD_TEST,test_front_events,,)

test-front-scene: $(LIB) _aec_lib
	$(call BUILD_TEST,test_front_scene,,)

test-front-block: $(LIB) _aec_lib $(ALIGN_SOURCES) ../aiaec.h ../aiaec_product.h
	$(call BUILD_TEST,test_front_block,$(ALIGN_CPPFLAGS),$(ALIGN_SOURCES))

test-front-oneshot: $(LIB) _aec_lib $(ALIGN_SOURCES) ../aiaec.h ../aiaec_product.h
	$(call BUILD_TEST,test_front_oneshot,$(ALIGN_CPPFLAGS),$(ALIGN_SOURCES))

# libaiaec.a must not contain or reference any symbol the AEC archive defines (checked for each
# AIAEC_MODEL variant). The same check run against libaiaec_front.a must find some: otherwise the
# gate could not fail.
audit-libaiaec-no-aec: $(LIB) _aec_lib
	@set -e; \
	aec_syms="$(OBJ_DIR)/aec_defined.txt"; \
	nm -g "$(AEC_LIB)" 2>/dev/null | awk 'NF==3 && $$2 != "U" && $$2 != "u" {print $$3}' | sort -u > "$$aec_syms"; \
	test -s "$$aec_syms" || { echo "FAIL: no AEC symbols read from $(AEC_LIB)" >&2; exit 1; }; \
	overlap() { nm -g "$$1" 2>/dev/null | awk 'NF>=2 {print $$NF}' | sort -u | comm -12 - "$$aec_syms"; }; \
	front_hits="$$(overlap "$(LIB)")"; \
	test -n "$$front_hits" || { echo "FAIL: control: libaiaec_front.a shows no AEC symbols; the audit cannot fail" >&2; exit 1; }; \
	defined="$$(nm -g "$(LIB)" 2>/dev/null | awk 'NF==3 && $$2 != "U" && $$2 != "u" {print $$3}' | grep -Ev '^_?aiaec_front_' || true)"; \
	test -z "$$defined" || { echo "FAIL: libaiaec_front.a defines non-front symbols: $$defined" >&2; exit 1; }; \
	for model in both deepvqe align; do \
	  $(MAKE) --no-print-directory -C .. AIAEC_MODEL=$$model $(PARENT_FWD) lib >/dev/null; \
	  lib="$$($(MAKE) -s --no-print-directory -C .. AIAEC_MODEL=$$model $(PARENT_FWD) print-lib-path)"; \
	  hits="$$(overlap "$$lib")"; \
	  test -z "$$hits" || { echo "FAIL: $$lib shares symbols with the AEC archive:" >&2; echo "$$hits" >&2; exit 1; }; \
	  bad="$$(nm -g "$$lib" 2>/dev/null | awk 'NF>=2 {print $$NF}' | grep -E '^_?(aec|delay_aec)_' || true)"; \
	  test -z "$$bad" || { echo "FAIL: $$lib has aec_* symbols: $$bad" >&2; exit 1; }; \
	  echo "audit-libaiaec-no-aec ($$model): PASS ($$lib)"; \
	done; \
	echo "audit-libaiaec-no-aec: PASS (control: libaiaec_front.a overlaps the AEC archive on $$(echo "$$front_hits" | wc -l | tr -d ' ') symbols)"

test: test-front-events test-front-scene test-front-block test-front-oneshot audit-libaiaec-no-aec

clean:
	rm -rf "$(BUILD_DIR)"
