# RNNoise-ERB table drift-guard wiring: the original flat build/ had no
# compiler/flags identity, so `make CC=false CFLAGS=... test-tables` silently
# re-ran a STALE binary from an earlier config -- reported PASS without ever
# invoking CC=false -- and a command-line CFLAGS= could replace
# -ffp-contract=off outright.
#
#   make test-tables    build + run BOTH layers and fail on any FAIL:
#                         1. canonical (default mode): byte-exact memcmp of
#                            rnnoise_tables_gen.h against an independent
#                            recompute -- the pinned-toolchain contract.
#                         2. portable (-DRNN_TABLES_PORTABLE): mathematical-
#                            contract checks + measured-then-margined ULP
#                            bounds (256 ULP recompute-vs-table garbage
#                            detector; 4096 ULP Hann mirror-pair symmetry --
#                            two DIFFERENT bounds for two different checks,
#                            see the test source).
#
# Same discipline as the four big Makefiles (lightweight version):
#   - CFLAGS/FP_POLICY cannot be overridden on the command line OR via
#     `make -e` + an environment variable of the same name (origin gate,
#     with FP_POLICY's own bare-literal `:=` defined early enough for the
#     origin check to see an `-e` override -- see its own comment below);
#     EXTRA_CFLAGS is the supported hook, folded in BEFORE the repo-pinned
#     -ffp-contract=off (appended last, conflict-rejected).
#   - CFLAGS/EXTRA_CFLAGS text is validated against a character allow-list
#     (letters/digits/underscore/dot/slash/equals/comma/plus/minus/space)
#     before any conflict check runs, closing the quote/glob/tilde/redirect/
#     process-substitution bypass class a substring-based flag check alone
#     cannot see (see the "FP-policy allow-list" block below).
#   - Binaries live in build/<cfg-sig>/ keyed by CC+CFLAGS+EXTRA_CFLAGS, with
#     a config.manifest collision guard -- a config change always lands in a
#     fresh directory and must genuinely compile there; a stale binary from
#     another config can never satisfy this target again. BUILD_DIR may be
#     overridden to an absolute path; the recipes that run a built binary
#     resolve it via $(abspath $(BUILD_DIR)) so this works whether BUILD_DIR
#     is relative or absolute (see the BUILD_DIR comment below).

CC ?= cc
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

# --- Bare-literal policy flags, defined early (`make -e` environment-override
# gap; same fix already applied in the sibling C-library Makefiles
# Makefiles for the identical bug -- see audio_common's Makefile "Bare-literal
# policy flags" comment for the general mechanism). FP_POLICY and
# FP_ALLOWED_CHARS_RE are both plain, dependency-free `:=` literals (no
# reference to CFLAGS or anything computed above) that MUST be assigned
# before the "Command-line override rejection" foreach below queries
# $(origin ...) on them: GNU Make only flips a variable's origin from plain
# "environment" to "environment override" at the point THIS SAME-NAME
# assignment is actually parsed (and, under `-e`, silently voided in the
# environment's favor) -- checking before that assignment ever runs means the
# check can never observe the flipped state. Confirmed empirically: with
# FP_POLICY's bare literal previously sitting AFTER this foreach (its only
# OTHER job -- folding it into CFLAGS, LAST, after conflict detection --
# stays exactly at its original position below), `env
# FP_POLICY=-ffp-contract=fast make -e -n test-features` used to sail
# straight through with no error, and the printed compile line genuinely
# carried -ffp-contract=fast (this repo's own pinned policy silently
# overridden). There is exactly one `:=` definition of FP_POLICY in this
# file, just relocated; its CFLAGS append stays exactly where it already was.
FP_POLICY := -ffp-contract=off
# FP_ALLOWED_CHARS_RE (the FP-policy allow-list's character class, added
# alongside the allow-list redesign further below) has the exact same
# `make -e` exposure and is a bare literal for the same reason -- see the
# allow-list block below for what it's used for.
FP_ALLOWED_CHARS_RE := ^[A-Za-z0-9_./=,+ -]*$$

# --- Command-line override rejection (same design as the four main
# Makefiles' analogous gate -- see audio_common's Makefile). A
# command-line CFLAGS=/FP_POLICY= would silently drop the FP policy below;
# a command-line/`-e` FP_ALLOWED_CHARS_RE= could silently widen the
# allow-list below (e.g. to `.*`) and neuter it.
$(foreach v,CFLAGS FP_POLICY FP_ALLOWED_CHARS_RE,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy) or neuter the FP-policy allow-list; use EXTRA_CFLAGS instead)))

# --- FP-policy allow-list (character-based; same redesign as audio_common's
# Makefile -- see its "REDESIGN" comment for the full writeup of the
# quote/glob/tilde/redirect/process-substitution bypass class a
# character-by-character deny-list structurally cannot keep up with). This
# file never had ANY character-based check before this -- only the three raw
# -Ofast/-ffast-math/-ffp-contract= findstring conflict checks further below.
#
# Single-quote rejection MUST run first, as a pure $(findstring) Make-native
# check with zero shell involvement: the allow-list below embeds $(CFLAGS)
# SINGLE-QUOTED inside a $(shell ...) call, and a literal single quote in
# CFLAGS would close that quoted string early and hand whatever follows to
# the shell as live syntax. Inside a single-quoted POSIX shell string every
# OTHER character is literal (double quote, backslash, backtick, "$(",
# semicolon, pipe, ampersand all inert there), so a lone single quote is the
# only character that can break this embedding -- kept as the sole survivor
# of the old-style deny-list check for that structural reason, not merely
# because it's also outside the allow-list below (it is rejected by the
# allow-list too).
ifneq ($(findstring ',$(CFLAGS)),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains a single-quote character; this repo requires plain, unquoted, whitespace-separated compiler argv tokens built only from the allowed character set (see the allow-list below) -- a single quote specifically must be rejected by this Make-native check BEFORE the allow-list's own $(shell) call can safely single-quote-embed the rest of the text)
endif

# The allow-list itself (same character set as audio_common's Makefile):
# letters, digits, underscore, dot, slash, equals, comma, plus, minus, plus
# space as the token separator between flags. Closes the redirection/glob/
# tilde/process-substitution bypass class (>, <, <(...), *, ?, [, ], ~) that
# a character-by-character deny-list cannot keep up with -- this file had no
# character check of any kind before this. LC_ALL=C avoids locale-dependent
# bracket-expression surprises in the [A-Za-z0-9...] ranges.
FP_ALLOWLIST_RC := $(shell printf '%s' '$(CFLAGS)' | LC_ALL=C grep -Eq '$(FP_ALLOWED_CHARS_RE)'; echo $$?)
ifneq ($(FP_ALLOWLIST_RC),0)
# Diagnostic-only second $(shell) call (only ever runs on the already-doomed
# error path above, never on a passing build): deletes every ALLOWED
# character from the same single-quote-embedded text, leaving only the
# disallowed ones, then dedupes/sorts them into a short, stable string -- so
# the $(error ...) below names the SPECIFIC offending character(s) instead of
# a generic "invalid format" message.
FP_DISALLOWED_CHARS := $(shell printf '%s' '$(CFLAGS)' | LC_ALL=C tr -d 'A-Za-z0-9_./=,+ -' | fold -w1 | LC_ALL=C sort -u | tr -d '\n')
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains a character outside the allowed set [A-Za-z0-9_./=,+- ] (letters/digits/underscore/dot/slash/equals/comma/plus/minus, plus space as the token separator) -- disallowed character(s) found: "$(FP_DISALLOWED_CHARS)"; this repo requires plain, unquoted, whitespace-separated compiler argv tokens built only from that set -- remove/replace the disallowed character(s) shown above in CFLAGS/EXTRA_CFLAGS)
endif

# Command-line/`-e` override rejection for the allow-list's own internal
# variable (same gap class as audio_common's Makefile's second foreach):
# FP_ALLOWLIST_RC is computed from the fully-assembled CFLAGS via a
# $(shell) call, so (unlike FP_POLICY/FP_ALLOWED_CHARS_RE above) it cannot be
# relocated to the early bare-literal foreach -- its own override-rejection
# check must run AFTER its real assignment so the origin-flip (plain
# "command line", or under `-e` "environment override") has already happened
# by the time $(origin ...) is queried here.
$(foreach v,FP_ALLOWLIST_RC,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently defeat the FP-policy allow-list check above, leaving the real CFLAGS content unvalidated; this variable is internal to that check and has no supported external use)))

# FP-policy conflict detection + last-position append, hardened to
# EXACT-TOKEN matching -- same fix as audio_common's Makefile's
# FP_CONFLICT_FLAGS, see its comment for the full writeup. The original
# three $(findstring -Ofast,...)/$(findstring -ffast-math,...)/
# $(findstring -ffp-contract=,...) checks matched on SUBSTRING, not whole
# compiler argv token: a legitimate flag value that merely CONTAINS one of
# these substrings without BEING the flag -- e.g. -DROUND9_NOTE=-Ofastness,
# a harmless macro definition -- was incorrectly rejected as a policy
# conflict. Confirmed empirically: `env
# EXTRA_CFLAGS='-DROUND9_NOTE=-Ofastness' make -n test-features` used to
# fail with "FP policy conflict: EXTRA_CFLAGS contains -Ofast" even though
# -Ofast never appears as its own argv token anywhere in CFLAGS.
#
# $(filter PATTERN...,TEXT) instead splits CFLAGS on whitespace and matches
# each PATTERN against a WHOLE word only (% is $(filter)'s own single
# wildcard, used here for -ffp-contract=<anything>) -- it can never match a
# flag value that merely embeds one of these substrings inside a longer
# token, while a real bare -Ofast / -ffast-math / -ffp-contract=<x> token is
# still caught exactly as before.
FP_CONFLICT_FLAGS := $(filter -Ofast -ffast-math -ffp-contract=%,$(CFLAGS))
ifneq ($(FP_CONFLICT_FLAGS),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains $(FP_CONFLICT_FLAGS); this repo pins -ffp-contract=off; remove $(FP_CONFLICT_FLAGS) from CFLAGS/EXTRA_CFLAGS)
endif

# Command-line/`-e` override rejection for FP_CONFLICT_FLAGS itself (same gap
# class as FP_ALLOWLIST_RC above, and the same one audio_common's Makefile
# closes for its own FP_CONFLICT_FLAGS): confirmed empirically that, without
# this guard, `make FP_CONFLICT_FLAGS= EXTRA_CFLAGS=-Ofast -n test-features`
# sails straight through with no error, and the printed compile line
# genuinely carries -Ofast -- this file's own pinned FP policy silently
# defeated by overriding the check's OWN internal variable rather than
# CFLAGS/EXTRA_CFLAGS directly. FP_CONFLICT_FLAGS is computed from the
# fully-assembled CFLAGS via $(filter), so (like FP_ALLOWLIST_RC) it cannot
# be relocated to the early bare-literal foreach; this check must run AFTER
# its real assignment so the origin-flip has already happened by the time
# $(origin ...) is queried here.
$(foreach v,FP_CONFLICT_FLAGS,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently defeat the FP-policy conflict check above, leaving the real CFLAGS content unvalidated; this variable is internal to that check and has no supported external use)))

CFLAGS    += $(FP_POLICY)

# Config-keyed build dir + collision guard (lightweight _cfg_guard).
#
# BUILD_DIR is intentionally a caller-overridable knob (`make
# BUILD_DIR=/some/path ...`), so it may be either relative (the default,
# `build/<cfg-sig>`) or an absolute path. The recipes below that actually
# EXECUTE a built binary (test-tables/test-features) must resolve it via
# $(abspath $(BUILD_DIR)) rather than a hardcoded `./$(BUILD_DIR)/<binary>`
# prefix: when BUILD_DIR is already absolute, `./$(BUILD_DIR)/binary` becomes
# `.//tmp/abs_test_dir/binary` -- since the leading path component is `.`,
# not `/`, that doubled slash collapses to a path RELATIVE to cwd
# (`./tmp/abs_test_dir/...`), not the absolute path the binary was actually
# built at, causing "No such file or directory" at real (non-dry-run)
# execution (`make -n` alone never surfaces this -- it only prints the
# broken recipe text without running it). $(abspath ...) is correct for
# BOTH a relative and an already-absolute BUILD_DIR, and the result is kept
# double-quoted in the recipe since this checkout's own path may contain
# spaces.
CFG_SIG_PAYLOAD := CC=$(CC) CFLAGS=$(CFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) SIMD=$(SIMD)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)
BUILD_DIR = build/$(CFG_SIG)
PY_TEST_DIR = tests

.PHONY: test test-tables test-features test-simd test-feature-contract test-feature-python test-loss-python clean _cfg_guard
_cfg_guard:
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

test: test-tables test-feature-contract test-features test-simd

test-feature-contract:
	@echo "== rnnoise config/Python/C feature contract =="
	$(PYTHON) $(PY_TEST_DIR)/test_feature_contract.py

test-features: $(BUILD_DIR)/test_rnnoise_features
	@echo "== rnnoise feature-state contract =="
	"$(abspath $(BUILD_DIR))/test_rnnoise_features"

test-simd: $(BUILD_DIR)/test_rnnoise_simd_auto $(BUILD_DIR)/test_rnnoise_simd_scalar
	@auto="$$( "$(abspath $(BUILD_DIR))/test_rnnoise_simd_auto" )"; \
	 scalar="$$( "$(abspath $(BUILD_DIR))/test_rnnoise_simd_scalar" )"; \
	 test "$$auto" = "$$scalar" || { \
	   echo "FAIL: RNNoise SIMD/scalar digest mismatch: $$auto != $$scalar" >&2; \
	   exit 1; \
	 }; \
	 echo "PASS: RNNoise SIMD/scalar full preprocessing digest $$auto"

# Requires the training Python environment (torch + numpy).  Kept separate from
# `make test` so the dependency-free embedded C checks remain runnable on hosts
# that intentionally do not install PyTorch.  `test-calibration` belongs here
# for the same reason -- it was briefly wired into `test`, which silently gave
# the default target a torch dependency it is specifically designed not to have.
#
# PYTHON is overridable because a bare `python3` resolves to whatever is on
# PATH (often /usr/bin/python3, which has no torch) rather than the project's
# virtualenv:  make test-python PYTHON=../../.venv/bin/python3
PYTHON ?= python3

test-python: test-feature-python test-loss-python test-calibration

test-feature-python:
	$(PYTHON) $(PY_TEST_DIR)/test_feature_norm.py
	$(PYTHON) $(PY_TEST_DIR)/test_python_c_features.py

test-loss-python:
	$(PYTHON) $(PY_TEST_DIR)/test_loss.py

# The ramp fit is shared with DeepFilterNet2 (dataset_gen/calibration.py), so
# its test lives at AINR/tests/ rather than under this project.
test-calibration:
	@echo "== normaliser-init calibration method (shared) =="
	$(PYTHON) ../tests/test_norm_init_calibration.py

$(BUILD_DIR)/test_rnnoise_tables: test/test_rnnoise_tables.c rnnoise_tables_gen.h process.h | _cfg_guard
	$(CC) $(CFLAGS) -o $@ test/test_rnnoise_tables.c -lm

$(BUILD_DIR)/test_rnnoise_tables_portable: test/test_rnnoise_tables.c rnnoise_tables_gen.h process.h | _cfg_guard
	$(CC) $(CFLAGS) -DRNN_TABLES_PORTABLE -o $@ test/test_rnnoise_tables.c -lm

$(BUILD_DIR)/test_rnnoise_features: test/test_rnnoise_features.c process.c process.h rnnoise_tables_gen.h | _cfg_guard
	$(CC) $(CFLAGS) -o $@ test/test_rnnoise_features.c process.c -lm

$(BUILD_DIR)/test_rnnoise_simd_auto: test/test_rnnoise_simd.c process.c process.h rnnoise_tables_gen.h | _cfg_guard
	$(CC) $(CFLAGS) -o $@ test/test_rnnoise_simd.c process.c -lm

$(BUILD_DIR)/test_rnnoise_simd_scalar: test/test_rnnoise_simd.c process.c process.h rnnoise_tables_gen.h | _cfg_guard
	$(CC) $(CFLAGS) -DSIMD_KERNELS_FORCE_SCALAR -o $@ test/test_rnnoise_simd.c process.c -lm

clean:
	rm -rf build
