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
# Same floating-point/include invariants as AIAEC/Makefile.
override CFLAGS += -ffp-contract=off -fno-math-errno -I. -I../../../lib/aiaec/deepvqe_s -I../../common $(EXTRA_CFLAGS)
LDFLAGS ?=
SIMD ?= 1
override SIMD := $(strip $(SIMD))
BACKEND ?= ne10
ifneq ($(BACKEND),ne10)
$(error BACKEND must be ne10 (got '$(BACKEND)'): ne10 is the only FFT backend)
endif

# DeepVQE-S has no file-loader TU (no ERB matrices, unlike DFN2) -- nothing here needs stdio at
# all. NO_STDIO is kept as a build-matrix knob for consistency with the other stream Makefiles
# (and audit-no-stdio below checks it holds), but it changes nothing in this directory's sources.
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

include ../../../lib/deps.mk
# RESOLVE_COMMON_LIB (from lib/deps.mk) resolves COMMON_LIB via a sub-make forwarding
# BACKEND/SIMD/TOOLCHAIN_FWD and validates it -- a cross-compile (different CC/CXX, or
# EXTRA_CFLAGS carrying e.g. -arch arm64e) must build a COMMON_LIB for the SAME target, not
# silently fall back to the host default and fail to link only once an external consumer tries to
# use it. Skipped outright (not just the validation) for a plain `make clean` -- see
# RESOLVE_COMMON_LIB's own comment.
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
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) AR=$(AR) CFLAGS=$(CFLAGS) LDFLAGS=$(LDFLAGS) SIMD=$(SIMD) SIMD_CPPFLAGS=$(SIMD_CPPFLAGS) NO_STDIO=$(NO_STDIO) BACKEND=$(BACKEND) WERROR=$(WERROR) COMMON_LIB=$(COMMON_LIB)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
OBJ_DIR := $(BUILD_DIR)/simd$(SIMD)-$(CFG_SIG)

# The DeepVQE-S core, recompiled here under this OBJ_DIR (mirroring how stream/ainr/dfn2/Makefile
# recompiles lib/ainr/dfn2's core under its own OBJ_DIR): same two sources AIAEC/Makefile's own
# OBJECTS list uses for this model, now resolved from lib/aiaec/deepvqe_s/.
CORE_SOURCES := ../../../lib/aiaec/deepvqe_s/deepvqe_process.c \
	../../../lib/aiaec/deepvqe_s/deepvqe_prepost.c
CORE_HEADERS := ../../../lib/aiaec/deepvqe_s/deepvqe_process.h \
	../../../lib/aiaec/deepvqe_s/deepvqe_prepost.h

STREAM_SOURCES := deepvqe_stream.c
STREAM_HEADERS := deepvqe_stream.h deepvqe_stream_internal.h ../../common/stream_mem.h

LIB := $(OBJ_DIR)/libdeepvqe_stream.a
LIB_OBJECTS := $(OBJ_DIR)/deepvqe_process.o $(OBJ_DIR)/deepvqe_prepost.o $(OBJ_DIR)/deepvqe_stream.o
VPATH := ../../../lib/aiaec/deepvqe_s

SMOKE_BIN := $(OBJ_DIR)/smoke_deepvqe_stream
SMOKE_SOURCES := tests/smoke_deepvqe_stream.c $(CORE_SOURCES) deepvqe_stream.c

# Compares deepvqe_stream against an independent re-implementation of the board sample's own
# loop, built directly against lib/aiaec/deepvqe_s + audiolib (never against deepvqe_stream.c's
# internals); see the file's own header comment.
VERIFY_VS_SAMPLE_BIN := $(OBJ_DIR)/verify_vs_sample
VERIFY_VS_SAMPLE_SOURCES := tests/verify_vs_sample.c $(CORE_SOURCES) deepvqe_stream.c

.PHONY: all lib print-lib-path print-cflags print-sources test smoke verify-vs-sample audit-no-stdio clean _cfg_guard

all: lib

lib: $(LIB)

print-lib-path:
	@echo "$(abspath $(LIB))"

print-cflags:
	@printf '%s\n' "$(CFLAGS) $(SIMD_CPPFLAGS)"

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
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) -MMD -MP -c -o "$@" "$<"

-include $(LIB_OBJECTS:.o=.d)

COMMON_HEADERS := $(wildcard $(COMMON_DIR)/include/*.h $(COMMON_DIR)/utility/*.h)

$(SMOKE_BIN): $(SMOKE_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(SMOKE_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

smoke: $(SMOKE_BIN)
	"$(SMOKE_BIN)"

$(VERIFY_VS_SAMPLE_BIN): $(VERIFY_VS_SAMPLE_SOURCES) $(CORE_HEADERS) $(STREAM_HEADERS) $(COMMON_LIB) $(COMMON_HEADERS) | _cfg_guard
	$(CC) $(CFLAGS) $(SIMD_CPPFLAGS) $(VERIFY_VS_SAMPLE_SOURCES) "$(COMMON_LIB)" \
		$(LDFLAGS) -lm -o "$@"

verify-vs-sample: $(VERIFY_VS_SAMPLE_BIN)
	"$(VERIFY_VS_SAMPLE_BIN)"
	"$(VERIFY_VS_SAMPLE_BIN)" --inject-shift
	"$(VERIFY_VS_SAMPLE_BIN)" --ola-matters

test: smoke verify-vs-sample

# audit-no-stdio: builds the archive and asserts, via `nm -u`, that deepvqe_stream.o references
# no stdio symbol (nothing in this model needs stdio at all, unlike DFN2's file loader).
# TOOLCHAIN_FWD comes from lib/deps.mk (included above): CC/CXX/EXTRA_CFLAGS/WERROR, shared with
# the COMMON_LIB resolution below instead of each spelling out the same four vars again.
audit-no-stdio:
	@$(MAKE) --no-print-directory NO_STDIO=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) lib >/dev/null
	@obj_dir="$$($(MAKE) -s --no-print-directory NO_STDIO=1 SIMD=$(SIMD) $(TOOLCHAIN_FWD) print-lib-path)"; \
	obj_dir="$$(dirname "$$obj_dir")"; \
	undef="$$(nm -u "$$obj_dir/deepvqe_stream.o" 2>/dev/null | grep -Ei 'fopen|fread|fclose|fwrite|fprintf|printf|puts' || true)"; \
	if [ -n "$$undef" ]; then \
	  echo "FAIL: deepvqe_stream.o references stdio symbols:" >&2; echo "$$undef" >&2; exit 1; \
	fi; \
	echo "audit-no-stdio: PASS ($$obj_dir/deepvqe_stream.o has no stdio references)"

clean:
	rm -rf "$(BUILD_DIR)"
