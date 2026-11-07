# RNNoise-ERB test wiring.
#
#   make test           the C checks below plus the feature contract
#   make test-tables    ERB matrix files against an independent recompute (library-free)
#   make test-features  feature reference; registered state, non-finite input, attenuation limit
#                       and pool descriptor contracts
#   make test-framing   the framing contract of process.h on several grids
#   make test-simd      the pre/post chain digest of the SIMD and scalar builds must be equal
#
# ERB_FWD and ERB_INV are the two headerless float32 files the C tests read (forward
# [n_bins][n_erb], inverse [n_erb][n_bins]). The defaults are written by tests/erb_fixture.py
# (standard-library Python, no torch or numpy), so `make test` needs only a C compiler and
# python3; ERB_FWD=FILE ERB_INV=FILE reads other files instead.
# test-erb-python (part of test-python, training environment) pins the fixture to train.py
# and to export_erb_matrix.py.
#
# Build policy:
#   - CFLAGS/FP_POLICY cannot be overridden from the command line or by
#     `make -e`; EXTRA_CFLAGS is the supported hook.
#   - CFLAGS/EXTRA_CFLAGS must use a restricted character set and may not carry
#     -Ofast, -ffast-math or -ffp-contract=; -ffp-contract=off is appended last.
#   - Binaries live in build/<cfg-sig>/, keyed by the build configuration, with
#     a config.manifest collision guard. BUILD_DIR may be absolute.

CC ?= cc
# COMMON_DIR comes from the one shared lib/deps.mk (see lib/audiolib.pin): the sibling audiolib
# dev checkout by default, still a caller-overridable `?=`.
include ../../lib/deps.mk
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

.PHONY: test test-tables test-features test-simd test-feature-contract test-feature-python test-loss-python test-erb-python test-framing clean _cfg_guard FORCE
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

PYTHON ?= python3
ERB_FWD ?= $(BUILD_DIR)/erb_fwd.bin
ERB_INV ?= $(BUILD_DIR)/erb_inv.bin

test: test-tables test-feature-contract test-features test-framing test-simd

test-tables: $(BUILD_DIR)/test_rnnoise_tables $(ERB_FWD) $(ERB_INV)
	@echo "== rnnoise ERB matrix drift guard =="
	"$(abspath $(BUILD_DIR))/test_rnnoise_tables" "$(abspath $(ERB_FWD))" "$(abspath $(ERB_INV))"

test-feature-contract:
	@echo "== rnnoise config/Python/C feature contract =="
	COMMON_DIR="$(abspath $(COMMON_DIR))" RNNOISE_TEST_SIMD=$(SIMD) $(PYTHON) -B $(PY_TEST_DIR)/test_feature_contract.py

test-features: $(BUILD_DIR)/test_rnnoise_features $(ERB_FWD) $(ERB_INV)
	@echo "== rnnoise feature-state contract =="
	"$(abspath $(BUILD_DIR))/test_rnnoise_features" "$(abspath $(ERB_FWD))" "$(abspath $(ERB_INV))"

test-framing: $(BUILD_DIR)/test_rnnoise_framing
	"$(abspath $(BUILD_DIR))/test_rnnoise_framing"

test-simd: $(BUILD_DIR)/test_rnnoise_simd_auto $(BUILD_DIR)/test_rnnoise_simd_scalar $(ERB_FWD) $(ERB_INV)
	@auto="$$( "$(abspath $(BUILD_DIR))/test_rnnoise_simd_auto" "$(abspath $(ERB_FWD))" "$(abspath $(ERB_INV))" )" || exit 1; \
	 scalar="$$( "$(abspath $(BUILD_DIR))/test_rnnoise_simd_scalar" "$(abspath $(ERB_FWD))" "$(abspath $(ERB_INV))" )" || exit 1; \
	 test -n "$$auto" && test "$$auto" = "$$scalar" || { \
	   echo "FAIL: RNNoise SIMD/scalar digest mismatch: '$$auto' != '$$scalar'" >&2; \
	   exit 1; \
	 }; \
	 echo "PASS: RNNoise SIMD/scalar local DSP (shared FFT fixed) digest $$auto"

# test-python needs the training Python environment (torch + numpy); `make test` does not.
#
# PYTHON is overridable so the project's virtualenv can be used:
#   make test-python PYTHON=../../.venv/bin/python3
test-python: test-feature-python test-loss-python test-calibration test-erb-python

test-feature-python:
	$(PYTHON) -B $(PY_TEST_DIR)/test_feature_norm.py
	COMMON_DIR="$(abspath $(COMMON_DIR))" RNNOISE_TEST_SIMD=$(SIMD) $(PYTHON) -B $(PY_TEST_DIR)/test_python_c_features.py

test-erb-python:
	$(PYTHON) -B $(PY_TEST_DIR)/test_erb_fixture.py

test-loss-python:
	$(PYTHON) -B $(PY_TEST_DIR)/test_loss.py

# The ramp fit is shared with DeepFilterNet2 (dataset_gen/calibration.py), so
# its test lives at AINR/tests/ rather than under this project.
test-calibration:
	@echo "== normaliser-init calibration method (shared) =="
	$(PYTHON) -B ../tests/test_norm_init_calibration.py

$(BUILD_DIR)/erb_fwd.bin: $(PY_TEST_DIR)/erb_fixture.py config.ini | _cfg_guard
	$(PYTHON) -B $(PY_TEST_DIR)/erb_fixture.py --config config.ini --fwd $@

$(BUILD_DIR)/erb_inv.bin: $(PY_TEST_DIR)/erb_fixture.py config.ini | _cfg_guard
	$(PYTHON) -B $(PY_TEST_DIR)/erb_fixture.py --config config.ini --inv $@

$(BUILD_DIR)/test_rnnoise_tables: test/test_rnnoise_tables.c test/rnnoise_test_erb.h | _cfg_guard
	$(CC) $(CFLAGS) -o $@ test/test_rnnoise_tables.c -lm

RN_TEST_DEPS = ../../lib/ainr/rnnoise_erb/process.c ../../lib/ainr/rnnoise_erb/process.h test/rnnoise_test_support.h test/rnnoise_test_erb.h $(COMMON_LIB)

$(BUILD_DIR)/test_rnnoise_features: test/test_rnnoise_features.c $(RN_TEST_DEPS) | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -o $@ test/test_rnnoise_features.c ../../lib/ainr/rnnoise_erb/process.c $(COMMON_LIB) -lm

$(BUILD_DIR)/test_rnnoise_simd_auto: test/test_rnnoise_simd.c $(RN_TEST_DEPS) | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -o $@ test/test_rnnoise_simd.c ../../lib/ainr/rnnoise_erb/process.c $(COMMON_LIB) -lm

$(BUILD_DIR)/test_rnnoise_simd_scalar: test/test_rnnoise_simd.c $(RN_TEST_DEPS) | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -DSIMD_KERNELS_FORCE_SCALAR -o $@ test/test_rnnoise_simd.c ../../lib/ainr/rnnoise_erb/process.c $(COMMON_LIB) -lm

$(BUILD_DIR)/test_rnnoise_framing: test/test_rnnoise_framing.c $(RN_TEST_DEPS) | _cfg_guard
	$(CC) $(CFLAGS) $(COMMON_INCLUDES) -o $@ test/test_rnnoise_framing.c ../../lib/ainr/rnnoise_erb/process.c $(COMMON_LIB) -lm

$(COMMON_LIB): FORCE
	$(MAKE) -s --no-print-directory -C $(COMMON_DIR) $(COMMON_FWD) lib
	@test -f '$@' || { echo "FATAL: common archive missing: $@" >&2; exit 1; }

FORCE:

clean:
	rm -rf build
