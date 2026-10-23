# RNNoise-ERB test wiring.
#
#   make test-tables    build + run BOTH layers and fail on any FAIL:
#                         1. canonical (default mode): byte-exact memcmp of
#                            rnnoise_tables_gen.h against an independent
#                            recompute.
#                         2. portable (-DRNN_TABLES_PORTABLE): mathematical-
#                            contract checks plus ULP bounds (256 ULP
#                            recompute-vs-table, 4096 ULP Hann mirror pair).
#
# Build policy:
#   - CFLAGS/FP_POLICY cannot be overridden from the command line or by
#     `make -e`; EXTRA_CFLAGS is the supported hook.
#   - CFLAGS/EXTRA_CFLAGS must use a restricted character set and may not carry
#     -Ofast, -ffast-math or -ffp-contract=; -ffp-contract=off is appended last.
#   - Binaries live in build/<cfg-sig>/, keyed by the build configuration, with
#     a config.manifest collision guard. BUILD_DIR may be absolute.

CC ?= cc
COMMON_DIR ?= $(abspath ../../../audiolib/common)
COMMON_FWD = SIMD=$(SIMD) CC='$(CC)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)'
COMMON_LIB ?= $(shell $(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) print-lib-path)
COMMON_INCLUDES = -I"$(COMMON_DIR)/include" -I"$(COMMON_DIR)"

SIMD ?= 1
ifneq ($(words $(SIMD)),1)
$(error SIMD must be exactly 0 or 1, got '$(SIMD)')
endif
ifneq ($(filter-out 0 1,$(SIMD)),)
$(error SIMD must be exactly 0 or 1, got '$(SIMD)')
endif
ifeq ($(SIMD),0)
SIMD_CFLAGS := -DSIMD_KERNELS_FORCE_SCALAR
else
SIMD_CFLAGS :=
endif
CFLAGS = -O2 -Wall -Wextra
CFLAGS += -I.
CFLAGS += $(EXTRA_CFLAGS)
CFLAGS += $(SIMD_CFLAGS)

# --- Policy literals, assigned before the override check below: under `make -e`
# the check only sees an environment override of a variable once its own
# assignment has been parsed.
FP_POLICY := -ffp-contract=off -fno-math-errno
# The allow-list character class is a bare literal for the same reason.
FP_ALLOWED_CHARS_RE := ^[A-Za-z0-9_./=,+ -]*$$

# --- Override rejection: a command-line or `-e` value for these would drop the
# FP policy or widen the allow-list.
$(foreach v,CFLAGS FP_POLICY FP_ALLOWED_CHARS_RE,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy) or neuter the FP-policy allow-list; use EXTRA_CFLAGS instead)))

# --- FP-policy allow-list. A single quote is rejected first by a Make-native
# check: the allow-list below embeds $(CFLAGS) single-quoted in a $(shell) call,
# and a stray quote would end that string early.
ifneq ($(findstring ',$(CFLAGS)),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains a single-quote character; this repo requires plain, unquoted, whitespace-separated compiler argv tokens built only from the allowed character set (see the allow-list below) -- a single quote specifically must be rejected by this Make-native check BEFORE the allow-list's own $(shell) call can safely single-quote-embed the rest of the text)
endif

# Allow-list: letters, digits, underscore, dot, slash, equals, comma, plus, minus
# and space. LC_ALL=C keeps the bracket ranges locale-independent.
FP_ALLOWLIST_RC := $(shell printf '%s' '$(CFLAGS)' | LC_ALL=C grep -Eq '$(FP_ALLOWED_CHARS_RE)'; echo $$?)
ifneq ($(FP_ALLOWLIST_RC),0)
# Error path only: name the offending characters.
FP_DISALLOWED_CHARS := $(shell printf '%s' '$(CFLAGS)' | LC_ALL=C tr -d 'A-Za-z0-9_./=,+ -' | fold -w1 | LC_ALL=C sort -u | tr -d '\n')
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains a character outside the allowed set [A-Za-z0-9_./=,+- ] (letters/digits/underscore/dot/slash/equals/comma/plus/minus, plus space as the token separator) -- disallowed character(s) found: "$(FP_DISALLOWED_CHARS)"; this repo requires plain, unquoted, whitespace-separated compiler argv tokens built only from that set -- remove/replace the disallowed character(s) shown above in CFLAGS/EXTRA_CFLAGS)
endif

# FP_ALLOWLIST_RC is computed from the final CFLAGS, so its override check runs
# after the assignment.
$(foreach v,FP_ALLOWLIST_RC,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently defeat the FP-policy allow-list check above, leaving the real CFLAGS content unvalidated; this variable is internal to that check and has no supported external use)))

# FP-policy conflict check with exact-token matching: $(filter) compares whole
# words, so a value that merely contains "-Ofast" as a substring is accepted
# while a bare -Ofast / -ffast-math / -ffp-contract=<x> is rejected.
FP_CONFLICT_FLAGS := $(filter -Ofast -ffast-math -ffp-contract=%,$(CFLAGS))
ifneq ($(FP_CONFLICT_FLAGS),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains $(FP_CONFLICT_FLAGS); this repo pins -ffp-contract=off; remove $(FP_CONFLICT_FLAGS) from CFLAGS/EXTRA_CFLAGS)
endif

# FP_CONFLICT_FLAGS is derived from the final CFLAGS, so its override check
# runs after the assignment.
$(foreach v,FP_CONFLICT_FLAGS,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently defeat the FP-policy conflict check above, leaving the real CFLAGS content unvalidated; this variable is internal to that check and has no supported external use)))

CFLAGS    += $(FP_POLICY)

# Config-keyed build dir and collision guard.
# BUILD_DIR may be relative or absolute; recipes that run a built binary use
# $(abspath $(BUILD_DIR)) so both forms resolve, and quote it because the
# checkout path may contain spaces.
# The same CC command can select a different SDK through PATH.
# Include effective pointer/FP/ISA predefines to avoid cross-ABI cache reuse.
override CC_TARGET_SIG := $(shell sh $(COMMON_DIR)/scripts/target_abi_signature.sh $(CC) $(CPPFLAGS) $(CFLAGS))
CFG_SIG_PAYLOAD := CC_TARGET_SIG=$(CC_TARGET_SIG) CC=$(CC) CFLAGS=$(CFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) SIMD=$(SIMD) COMMON_DIR=$(abspath $(COMMON_DIR)) COMMON_LIB=$(COMMON_LIB)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
BUILD_DIR = build/$(CFG_SIG)
PY_TEST_DIR = tests

.PHONY: test test-tables test-features test-simd test-feature-contract test-feature-python test-loss-python test-framing clean _cfg_guard FORCE
_cfg_guard:
	@test -n "$(CC_TARGET_SIG)" && test "$(CC_TARGET_SIG)" != unavailable || { \
	  echo "FATAL: cannot determine compiler target; CC preprocessing with the configured flags must succeed." >&2; \
	  exit 1; }
	@mkdir -p $(BUILD_DIR)
	@m="$(BUILD_DIR)/config.manifest"; \
	if [ -f "$$m" ]; then \
	  old="$$(cat "$$m")"; \
	  if [ "$$old" != "$(CFG_SIG_PAYLOAD)" ]; then \
	    echo "FATAL: CFG_SIG collision or coverage gap in $$m" >&2; \
	    echo "  stored : $$old" >&2; \
	    echo "  current: $(CFG_SIG_PAYLOAD)" >&2; \
	    exit 1; \
	  fi; \
	else \
	  tmp="$$m.tmp.$$$$"; printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$tmp"; mv -f "$$tmp" "$$m"; \
	fi

test-tables: $(BUILD_DIR)/test_rnnoise_tables $(BUILD_DIR)/test_rnnoise_tables_portable
	@echo "== rnnoise tables drift-guard: layer 1 (canonical, byte-exact) =="
	"$(abspath $(BUILD_DIR))/test_rnnoise_tables"
	@echo "== rnnoise tables drift-guard: layer 2 (portable, math contracts) =="
	"$(abspath $(BUILD_DIR))/test_rnnoise_tables_portable"

test: test-tables test-feature-contract test-features test-simd test-framing

test-feature-contract:
	@echo "== rnnoise config/Python/C feature contract =="
	$(PYTHON) $(PY_TEST_DIR)/test_feature_contract.py

test-features: $(BUILD_DIR)/test_rnnoise_features
	@echo "== rnnoise feature-state contract =="
	"$(abspath $(BUILD_DIR))/test_rnnoise_features"

test-framing: $(BUILD_DIR)/test_rnnoise_framing
	"$(abspath $(BUILD_DIR))/test_rnnoise_framing"

test-simd: $(BUILD_DIR)/test_rnnoise_simd_auto $(BUILD_DIR)/test_rnnoise_simd_scalar
	@auto="$$( "$(abspath $(BUILD_DIR))/test_rnnoise_simd_auto" )"; \
	 scalar="$$( "$(abspath $(BUILD_DIR))/test_rnnoise_simd_scalar" )"; \
	 test "$$auto" = "$$scalar" || { \
	   echo "FAIL: RNNoise SIMD/scalar digest mismatch: $$auto != $$scalar" >&2; \
	   exit 1; \
	 }; \
	 echo "PASS: RNNoise SIMD/scalar local DSP (shared FFT fixed) digest $$auto"

# Requires the training Python environment (torch + numpy); kept out of
# `make test` so the C checks run on hosts without PyTorch.
#
# PYTHON is overridable so the project's virtualenv can be used:
#   make test-python PYTHON=../../.venv/bin/python3
PYTHON ?= python3

test-python: test-feature-python test-loss-python test-calibration

test-feature-python:
	$(PYTHON) $(PY_TEST_DIR)/test_feature_norm.py
	COMMON_DIR="$(abspath $(COMMON_DIR))" RNNOISE_TEST_SIMD=$(SIMD) $(PYTHON) $(PY_TEST_DIR)/test_python_c_features.py

test-loss-python:
	$(PYTHON) $(PY_TEST_DIR)/test_loss.py

# The ramp fit is shared with DeepFilterNet2 (dataset_gen/calibration.py), so
# its test lives at AINR/tests/ rather than under this project.
test-calibration:
	@echo "== normaliser-init calibration method (shared) =="
	$(PYTHON) ../tests/test_norm_init_calibration.py

$(BUILD_DIR)/test_rnnoise_tables: test/test_rnnoise_tables.c rnnoise_tables_gen.h process.h | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -o $@ test/test_rnnoise_tables.c -lm

$(BUILD_DIR)/test_rnnoise_tables_portable: test/test_rnnoise_tables.c rnnoise_tables_gen.h process.h | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -DRNN_TABLES_PORTABLE -o $@ test/test_rnnoise_tables.c -lm

$(BUILD_DIR)/test_rnnoise_features: test/test_rnnoise_features.c process.c process.h rnnoise_tables_gen.h test/rnnoise_test_framing.h $(COMMON_LIB) | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -o $@ test/test_rnnoise_features.c process.c $(COMMON_LIB) -lm

$(BUILD_DIR)/test_rnnoise_simd_auto: test/test_rnnoise_simd.c process.c process.h rnnoise_tables_gen.h test/rnnoise_test_framing.h $(COMMON_LIB) | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -o $@ test/test_rnnoise_simd.c process.c $(COMMON_LIB) -lm

$(BUILD_DIR)/test_rnnoise_simd_scalar: test/test_rnnoise_simd.c process.c process.h rnnoise_tables_gen.h test/rnnoise_test_framing.h $(COMMON_LIB) | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -DSIMD_KERNELS_FORCE_SCALAR -o $@ test/test_rnnoise_simd.c process.c $(COMMON_LIB) -lm

$(BUILD_DIR)/test_rnnoise_framing: test/test_rnnoise_framing.c process.c process.h rnnoise_tables_gen.h test/rnnoise_test_framing.h $(COMMON_LIB) | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -o $@ test/test_rnnoise_framing.c process.c $(COMMON_LIB) -lm

$(COMMON_LIB): FORCE
	$(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) lib
	@test -f '$@' || { echo "FATAL: common archive missing: $@" >&2; exit 1; }

FORCE:

clean:
	rm -rf build
