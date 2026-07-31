#!/bin/sh
# run_makefile_build_safety_test.sh - permanent regression test for four
# Makefile build-safety fixes in ../Makefile (this component has no build
# dependency on sibling projects -- these are RNNoise-ERB's OWN Makefile
# mechanics, fixed to match the pattern already shipped in the four main
# repos' Makefiles):
#
# 1. `make -e` FP_POLICY origin-check-ordering bug: the bare
#    `FP_POLICY := -ffp-contract=off` literal used to be defined AFTER the
#    "Command-line override rejection" $(foreach) that checks
#    $(origin FP_POLICY) -- GNU Make only flips a variable's origin from
#    plain "environment" to "environment override" at the point THAT
#    SAME-NAME assignment is actually parsed (and, under `-e`, silently
#    voided in the environment's favor), so checking before the assignment
#    ever ran meant the check could never observe the flipped state.
#    Reproduced: `env FP_POLICY=-ffp-contract=fast make -e -n test-features`
#    used to succeed, with the printed compile line actually carrying
#    -ffp-contract=fast (this repo's own pinned FP policy silently
#    overridden). Fixed by moving only the bare literal ahead of the
#    foreach; the later `CFLAGS += $(FP_POLICY)` append (after conflict
#    detection) is untouched.
#
# 2. BUILD_DIR=/absolute/path execution-path bug: the test-tables/
#    test-features recipes used to invoke the built binary as
#    `./$(BUILD_DIR)/binary_name`. When BUILD_DIR is already absolute (e.g.
#    BUILD_DIR=/tmp/abs_test_dir), `./$(BUILD_DIR)/binary` becomes
#    `.//tmp/abs_test_dir/binary` -- the leading component is `.`, not `/`,
#    so the doubled slash collapses to a path RELATIVE to cwd
#    (`./tmp/abs_test_dir/...`), not the absolute path the binary was
#    actually built at, producing "No such file or directory" at real
#    (non-dry-run) execution -- `make -n` alone never surfaces this, since
#    it only prints the broken recipe text without running it. Fixed by
#    invoking `$(abspath $(BUILD_DIR))/binary_name` instead, which is
#    correct whether BUILD_DIR is relative or already absolute.
#
# 3. FP-policy allow-list (this Makefile had ZERO character-based
#    protection before this fix -- not even an old deny-list): a
#    quote/glob/tilde/redirect/process-substitution bypass of the plain
#    -Ofast/-ffast-math/-ffp-contract= findstring conflict checks could
#    reach a real shell unrejected. Fixed by adding the same
#    character-class ALLOW-list mechanism already shipped in
#    the sibling C-library Makefiles (adapted for this
#    file's simpler CFLAGS-only variable set): a Make-native single-quote
#    rejection (protects the allow-list's own single-quote-embedded
#    $(shell) call), then a `grep -E` allow-list over
#    [A-Za-z0-9_./=,+ -], with the specific disallowed character(s) named
#    in the error.
#
# 4. -Ofast/-ffast-math/-ffp-contract= SUBSTRING false-positive: the three
#    conflict checks used $(findstring <flag>,$(CFLAGS)), which matches
#    <flag> as a SUBSTRING anywhere in the text, not as a whole compiler
#    argv token. A legitimate flag whose VALUE merely CONTAINS one of these
#    substrings without BEING the flag -- e.g. EXTRA_CFLAGS=
#    -DROUND9_NOTE=-Ofastness, a harmless macro definition -- was
#    incorrectly rejected as a policy conflict. Reproduced: `env
#    EXTRA_CFLAGS='-DROUND9_NOTE=-Ofastness' make -n test-features` used to
#    fail with "FP policy conflict: EXTRA_CFLAGS contains -Ofast" even
#    though -Ofast never appears as its own argv token. Fixed by switching
#    to $(filter -Ofast -ffast-math -ffp-contract=%,$(CFLAGS)) (same design
#    as audio_common's Makefile's FP_CONFLICT_FLAGS): $(filter) splits on
#    whitespace and matches each pattern against a WHOLE word only, so it
#    can never match a flag value that merely embeds one of these
#    substrings inside a longer token, while a real bare -Ofast/
#    -ffast-math/-ffp-contract=<x> token is still caught exactly as before.
#    The new FP_CONFLICT_FLAGS variable this introduces got the same
#    command-line/`-e` override-rejection guard as FP_ALLOWLIST_RC above
#    (found live during this same fix: without it, `make FP_CONFLICT_FLAGS=
#    EXTRA_CFLAGS=-Ofast` silently defeated the whole conflict check).
#
# Style/convention follows the repository's other C-library policy tests.
# (this repo has no test/ directory of its own before this file; tests/
# already exists here but holds the Python feature/loss tests, a different
# thing).
#
# Scratch-isolated throughout (real fix, not just documentation): this
# script rsync's the ENTIRE RNNoise-ERB directory tree (Makefile + all .c/.h
# sources this Makefile depends on) into a fresh mktemp scratch working
# copy at startup, cd's into THAT copy, and runs every single test case
# below -- default BUILD_DIR, relative BUILD_DIR, absolute BUILD_DIR, and
# all the character-bypass cases -- only inside the scratch copy. The real
# checkout this script was invoked from is never cd'd into for any recipe
# invocation and has no file created, modified, or deleted by any scenario
# below, period (verified by a before/after directory-tree snapshot diff --
# see the component's review notes). `build/` is excluded from the copy on
# purpose: excluding it forces every scratch run to genuinely (re)compile
# from source rather than silently reusing a config.manifest/binary that
# happened to get copied over, and it means there is no real build/<sig>
# directory in the checkout for this script to ever land in even by
# accident. A single `rm -rf "$SCRATCH"` cleanup covers the whole scratch
# tree (working copy included), so there is no separate throwaway-directory
# special case left to get the "does it already exist for a real reason"
# check wrong.
#
# Usage: ./test/run_makefile_build_safety_test.sh
#   (from anywhere; locates this component's root via the script's own
#   location, no assumption about caller cwd -- but only to know what to
#   rsync FROM; every actual test runs from the scratch copy)
#
# Exit code 0 + "run_makefile_build_safety_test: ALL PASS" means every case
# behaved as expected. Nonzero + at least one "FAIL" line means a
# regression.

set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Explicit template (same fix applied throughout the sibling repos' other
# test/run_*.sh scripts): a bare `mktemp -d` with no template argument
# ignores $TMPDIR on some hosts, silently defeating any caller that sets
# TMPDIR to observe this script's scratch-dir lifecycle. Also deliberately
# space-free (macOS default $TMPDIR is under /var/folders/..., no spaces)
# even though this component's own real checkout path is not.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/rnnoise-erb-makefile-safety.XXXXXX")" || {
    echo "FATAL: mktemp failed to create a scratch directory" >&2
    exit 1
}
[ -n "$SCRATCH" ] && [ -d "$SCRATCH" ] || {
    echo "FATAL: mktemp reported success but SCRATCH is empty or not a directory -- refusing to proceed (would otherwise fall through to a bare /workdir)" >&2
    exit 1
}
WORKDIR="$SCRATCH/workdir"
cleanup() {
    [ -n "$SCRATCH" ] && [ -d "$SCRATCH" ] && rm -rf "$SCRATCH"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$WORKDIR"
# --exclude='/build': never copy the real checkout's build artifacts into
# the scratch copy (see comment above -- forces a genuine fresh compile
# every run and keeps the real build/ directory wholly irrelevant to this
# script from this point on).
if ! rsync -a --exclude='/build' "$REPO_ROOT/" "$WORKDIR/"; then
    echo "FATAL: rsync of $REPO_ROOT into scratch workdir $WORKDIR failed -- aborting before any test runs (real checkout untouched)" >&2
    exit 1
fi
cd "$WORKDIR" || { echo "FATAL: cd into scratch workdir $WORKDIR failed -- aborting" >&2; exit 1; }
RELATIVE_BUILD_DIR=".rnnoise_erb_test_relative_build_scratch"

MAKE_BIN=${MAKE:-make}

PASS_COUNT=0
FAIL_COUNT=0
pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); echo "  FAIL: $1"; }

echo "=== Section 1: make -e FP_POLICY origin-check-ordering bug ==="

# The exact repro from the bug report. `-n` (dry run) is sufficient and
# side-effect-free: the origin-check foreach is parse-time Makefile text
# ($(error ...)), evaluated before any recipe (even a dry-run-printed one)
# would run.
LOG1="$SCRATCH/log_e_fp_policy"
if env FP_POLICY=-ffp-contract=fast "$MAKE_BIN" -e -n test-features >"$LOG1" 2>&1; then
    fail "1a: env FP_POLICY=-ffp-contract=fast make -e -n test-features unexpectedly SUCCEEDED (must be rejected)"
    cat "$LOG1" >&2
else
    if grep -q "cannot be overridden" "$LOG1" && grep -q "FP_POLICY" "$LOG1" && grep -q "environment override" "$LOG1"; then
        pass "1a: env FP_POLICY=-ffp-contract=fast make -e -n test-features correctly FAILS, identifying FP_POLICY as an environment override"
    else
        fail "1a: env FP_POLICY=-ffp-contract=fast make -e -n test-features failed but did NOT identify FP_POLICY/environment override specifically"
        cat "$LOG1" >&2
    fi
fi

# Prior (round-5) behaviour must not regress: a plain command-line
# CFLAGS=/FP_POLICY= override was already rejected before this fix (a
# different Make origin, "command line", which is position-independent and
# was never the bug) -- confirm both still fail.
LOG2="$SCRATCH/log_cmdline_cflags"
if "$MAKE_BIN" CFLAGS=-O3 -n test-features >"$LOG2" 2>&1; then
    fail "1b: make CFLAGS=-O3 -n test-features unexpectedly SUCCEEDED (command-line override rejection regressed)"
    cat "$LOG2" >&2
else
    grep -q "cannot be overridden" "$LOG2" \
        && pass "1b: make CFLAGS=-O3 -n test-features correctly FAILS, mentioning 'cannot be overridden' (command-line override rejection unaffected by this fix)" \
        || { fail "1b: make CFLAGS=-O3 -n test-features failed but did NOT mention 'cannot be overridden'"; cat "$LOG2" >&2; }
fi

LOG3="$SCRATCH/log_cmdline_fp_policy"
if "$MAKE_BIN" FP_POLICY=-ffp-contract=fast -n test-features >"$LOG3" 2>&1; then
    fail "1c: make FP_POLICY=-ffp-contract=fast -n test-features unexpectedly SUCCEEDED (command-line override rejection regressed)"
    cat "$LOG3" >&2
else
    grep -q "cannot be overridden" "$LOG3" \
        && pass "1c: make FP_POLICY=-ffp-contract=fast -n test-features correctly FAILS, mentioning 'cannot be overridden'" \
        || { fail "1c: make FP_POLICY=-ffp-contract=fast -n test-features failed but did NOT mention 'cannot be overridden'"; cat "$LOG3" >&2; }
fi

echo "=== Section 2: BUILD_DIR=/absolute/path real (non-dry-run) execution ==="

# The exact repro's dry-run form, to confirm the recipe TEXT is now
# unambiguous (no bare `./$(BUILD_DIR)` anywhere).
ABS_BUILD_DIR="$SCRATCH/abs_test_dir"
LOG4="$SCRATCH/log_dryrun_absdir"
"$MAKE_BIN" -n BUILD_DIR="$ABS_BUILD_DIR" test-tables >"$LOG4" 2>&1
if grep -qF "./$ABS_BUILD_DIR/" "$LOG4"; then
    fail "2a: make -n BUILD_DIR=$ABS_BUILD_DIR test-tables recipe text still contains the buggy './\$(BUILD_DIR)/' pattern"
    cat "$LOG4" >&2
elif grep -qF "$ABS_BUILD_DIR/test_rnnoise_tables" "$LOG4"; then
    pass "2a: make -n BUILD_DIR=$ABS_BUILD_DIR test-tables recipe text invokes the binary by its real absolute path (no './' collapse)"
else
    fail "2a: make -n BUILD_DIR=$ABS_BUILD_DIR test-tables recipe text did not contain the expected absolute invocation at all"
    cat "$LOG4" >&2
fi

# The real thing: actually build AND run (not a dry run) with an absolute
# BUILD_DIR, since the bug specifically only manifests at real execution.
LOG5="$SCRATCH/log_real_absdir"
if "$MAKE_BIN" BUILD_DIR="$ABS_BUILD_DIR" test-tables >"$LOG5" 2>&1; then
    if grep -q "PASS: rnnoise_tables_gen.h byte-identical" "$LOG5" && grep -q "PASS (portable)" "$LOG5"; then
        pass "2b: make BUILD_DIR=$ABS_BUILD_DIR test-tables (real execution) succeeds and both layers report PASS"
    else
        fail "2b: make BUILD_DIR=$ABS_BUILD_DIR test-tables (real execution) exited 0 but did not report both expected PASS lines"
        cat "$LOG5" >&2
    fi
else
    fail "2b: make BUILD_DIR=$ABS_BUILD_DIR test-tables (real execution) unexpectedly FAILED (the absolute-BUILD_DIR execution-path bug may have regressed)"
    cat "$LOG5" >&2
fi

# Confirm the binary that ran really lives at the absolute path (not a
# cwd-relative collapse of it), and that no bogus relative directory
# mirroring the collapsed pattern was created under this component's cwd.
if [ -x "$ABS_BUILD_DIR/test_rnnoise_tables" ] && [ -x "$ABS_BUILD_DIR/test_rnnoise_tables_portable" ]; then
    pass "2c: both built binaries actually exist, executable, at the absolute BUILD_DIR path"
else
    fail "2c: built binaries are missing at the absolute BUILD_DIR path ($ABS_BUILD_DIR)"
fi
BOGUS_RELATIVE_DIR="./${ABS_BUILD_DIR#/}"
if [ -e "$BOGUS_RELATIVE_DIR" ]; then
    fail "2d: a bogus cwd-relative directory matching the OLD './\$(BUILD_DIR)' collapse pattern was created ($BOGUS_RELATIVE_DIR) -- the abspath fix did not take effect"
    rm -rf "$BOGUS_RELATIVE_DIR"
else
    pass "2d: no bogus cwd-relative directory was created by the absolute-BUILD_DIR run (the old './\$(BUILD_DIR)' collapse pattern does not reappear)"
fi

# test-features gets the same absolute-BUILD_DIR real-execution check.
ABS_BUILD_DIR2="$SCRATCH/abs_test_dir_features"
LOG6="$SCRATCH/log_real_absdir_features"
if "$MAKE_BIN" BUILD_DIR="$ABS_BUILD_DIR2" test-features >"$LOG6" 2>&1; then
    grep -q "PASS: log_erb_dfn_mean_cplx_unit_0_4k_v3" "$LOG6" \
        && pass "2e: make BUILD_DIR=$ABS_BUILD_DIR2 test-features (real execution) succeeds and reports PASS" \
        || { fail "2e: make BUILD_DIR=$ABS_BUILD_DIR2 test-features (real execution) exited 0 but did not report the expected PASS line"; cat "$LOG6" >&2; }
else
    fail "2e: make BUILD_DIR=$ABS_BUILD_DIR2 test-features (real execution) unexpectedly FAILED"
    cat "$LOG6" >&2
fi

# Regression: a RELATIVE BUILD_DIR override (never buggy, but must keep
# working after the abspath rewrite) still resolves and runs correctly.
rm -rf "$RELATIVE_BUILD_DIR"
LOG7="$SCRATCH/log_real_relativedir"
if "$MAKE_BIN" BUILD_DIR="$RELATIVE_BUILD_DIR" test-tables >"$LOG7" 2>&1; then
    grep -q "PASS: rnnoise_tables_gen.h byte-identical" "$LOG7" && grep -q "PASS (portable)" "$LOG7" \
        && pass "2f: make BUILD_DIR=$RELATIVE_BUILD_DIR (relative) test-tables still succeeds after the abspath rewrite" \
        || { fail "2f: make BUILD_DIR=$RELATIVE_BUILD_DIR (relative) test-tables exited 0 but did not report both expected PASS lines"; cat "$LOG7" >&2; }
else
    fail "2f: make BUILD_DIR=$RELATIVE_BUILD_DIR (relative) test-tables unexpectedly FAILED"
    cat "$LOG7" >&2
fi

# Plain, unmodified invocation (no BUILD_DIR override at all) must still
# work exactly as before -- uses the default build/<cfg-sig>/ directory
# resolved relative to cwd, which by this point is the scratch WORKDIR
# copy (cd'd into above), never the real checkout's own build/ directory.
LOG8="$SCRATCH/log_plain_test_tables"
if "$MAKE_BIN" test-tables >"$LOG8" 2>&1; then
    grep -q "PASS: rnnoise_tables_gen.h byte-identical" "$LOG8" && grep -q "PASS (portable)" "$LOG8" \
        && pass "2g: plain 'make test-tables' (no BUILD_DIR override) still succeeds" \
        || { fail "2g: plain 'make test-tables' exited 0 but did not report both expected PASS lines"; cat "$LOG8" >&2; }
else
    fail "2g: plain 'make test-tables' unexpectedly FAILED"
    cat "$LOG8" >&2
fi

LOG9="$SCRATCH/log_plain_test_features"
if "$MAKE_BIN" test-features >"$LOG9" 2>&1; then
    grep -q "PASS: log_erb_dfn_mean_cplx_unit_0_4k_v3" "$LOG9" \
        && pass "2h: plain 'make test-features' (no BUILD_DIR override) still succeeds" \
        || { fail "2h: plain 'make test-features' exited 0 but did not report the expected PASS line"; cat "$LOG9" >&2; }
else
    fail "2h: plain 'make test-features' unexpectedly FAILED"
    cat "$LOG9" >&2
fi

echo "=== Section 3: FP-policy allow-list -- quote/glob/tilde/redirect/process-substitution bypass set ==="

# helper: run one EXTRA_CFLAGS negative case (dry run -- the FP-policy
# checks are parse-time $(error)s, so -n exercises them without ever
# letting a dangerous character reach a real shell recipe) and assert both
# that it FAILS and that the expected marker text appears in the log.
check_rejected() {
    label="$1"; flag_value="$2"; must_contain="$3"
    log="$SCRATCH/log_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_')"
    if env EXTRA_CFLAGS="$flag_value" "$MAKE_BIN" -n test-features >"$log" 2>&1; then
        fail "3.$label: EXTRA_CFLAGS='$flag_value' unexpectedly SUCCEEDED (must be rejected)"
        cat "$log" >&2
        return
    fi
    if grep -q "FP policy conflict" "$log" && grep -qF "$must_contain" "$log"; then
        pass "3.$label: EXTRA_CFLAGS='$flag_value' correctly FAILS, identifying \"$must_contain\""
    else
        fail "3.$label: EXTRA_CFLAGS='$flag_value' failed but did NOT identify \"$must_contain\" specifically"
        cat "$log" >&2
    fi
}

check_rejected "single-quoted"      "'-Ofast'"        "single-quote"
check_rejected "double-quoted"      '"-ffast-math"'   'found: """'
check_rejected "quote-split"        "-O'f'ast"        "single-quote"
check_rejected "response-file"      '@flags.rsp'      'found: "@"'
check_rejected "glob-star"          '-O*t'            'found: "*"'
check_rejected "glob-question"     '-Ofas?'          'found: "?"'
check_rejected "glob-brackets"      '-Ofas[t]'        'found: "[]"'
check_rejected "tilde"              '~/pwned'         'found: "~"'
check_rejected "redirect-out"       '-I>/tmp/evil'    'found: ">"'
check_rejected "redirect-in"        '-I</etc/passwd'  'found: "<"'
check_rejected "process-subst"      '<(echo hi)'      'found: "()<"'

# Positive control: a harmless EXTRA_CFLAGS built only from the allowed
# character set must NOT be rejected, and must produce a real, working
# build (scratch BUILD_DIR, real execution -- not a dry run, so this also
# proves the allow-list doesn't merely let parsing continue but actually
# still produces a correct binary).
POS_BUILD_DIR="$SCRATCH/pos_ctrl_build"
LOG_POS="$SCRATCH/log_positive_control"
if env EXTRA_CFLAGS=-DSCRATCH_TEST_DEFINE=1 "$MAKE_BIN" BUILD_DIR="$POS_BUILD_DIR" test-features >"$LOG_POS" 2>&1; then
    grep -q "PASS: log_erb_dfn_mean_cplx_unit_0_4k_v3" "$LOG_POS" \
        && pass "3.positive-control: EXTRA_CFLAGS=-DSCRATCH_TEST_DEFINE=1 (allowed characters only) builds and runs correctly" \
        || { fail "3.positive-control: EXTRA_CFLAGS=-DSCRATCH_TEST_DEFINE=1 build succeeded but did not report the expected PASS line"; cat "$LOG_POS" >&2; }
else
    fail "3.positive-control: EXTRA_CFLAGS=-DSCRATCH_TEST_DEFINE=1 (allowed characters only) unexpectedly FAILED (false-positive on a harmless token)"
    cat "$LOG_POS" >&2
fi

# The allow-list's own internal variables (FP_ALLOWED_CHARS_RE,
# FP_ALLOWLIST_RC) must themselves be immune to a command-line/`-e`
# override -- otherwise `make FP_ALLOWED_CHARS_RE='.*' EXTRA_CFLAGS=';rm'`
# would silently neuter the whole allow-list.
check_override_rejected() {
    varname="$1"; varvalue="$2"; payload="$3"; use_dash_e="$4"
    log="$SCRATCH/log_override_$(printf '%s' "$varname" | tr -c 'A-Za-z0-9' '_')_$use_dash_e"
    if [ "$use_dash_e" = "dashe" ]; then
        if env "$varname=$varvalue" "$MAKE_BIN" -e -n EXTRA_CFLAGS="$payload" test-features >"$log" 2>&1; then
            fail "3.override.$varname($use_dash_e): env $varname=$varvalue make -e -n unexpectedly SUCCEEDED (must be rejected)"
            cat "$log" >&2
            return
        fi
    else
        if "$MAKE_BIN" -n "$varname=$varvalue" EXTRA_CFLAGS="$payload" test-features >"$log" 2>&1; then
            fail "3.override.$varname($use_dash_e): make $varname=$varvalue -n unexpectedly SUCCEEDED (must be rejected)"
            cat "$log" >&2
            return
        fi
    fi
    if grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
        pass "3.override.$varname($use_dash_e): correctly FAILS, mentioning '$varname cannot be overridden'"
    else
        fail "3.override.$varname($use_dash_e): failed but did NOT mention '$varname cannot be overridden'"
        cat "$log" >&2
    fi
}
check_override_rejected "FP_ALLOWED_CHARS_RE" ".*" "-O2;rm" cmdline
check_override_rejected "FP_ALLOWED_CHARS_RE" ".*" "-O2;rm" dashe
check_override_rejected "FP_ALLOWLIST_RC"     "0"  "-O2;rm" cmdline
check_override_rejected "FP_ALLOWLIST_RC"     "0"  "-O2;rm" dashe

# Prior (pre-allow-list) behaviour must not regress: a real, unquoted
# -Ofast/-ffast-math/-ffp-contract=<x> token is still rejected by the
# original three findstring conflict checks.
LOG10="$SCRATCH/log_plain_ofast"
if env EXTRA_CFLAGS=-Ofast "$MAKE_BIN" -n test-features >"$LOG10" 2>&1; then
    fail "3.regression: env EXTRA_CFLAGS=-Ofast make -n test-features unexpectedly SUCCEEDED (must be rejected)"
    cat "$LOG10" >&2
else
    grep -q "FP policy conflict" "$LOG10" && grep -q -- "-Ofast" "$LOG10" \
        && pass "3.regression: env EXTRA_CFLAGS=-Ofast make -n test-features correctly FAILS (pre-existing conflict check unaffected)" \
        || { fail "3.regression: env EXTRA_CFLAGS=-Ofast make -n test-features failed but did NOT identify -Ofast specifically"; cat "$LOG10" >&2; }
fi

echo "=== Section 4: -Ofast/-ffast-math/-ffp-contract= exact-token matching (no substring false positives) ==="

# The exact Codex-reported false positive: a legitimate flag whose VALUE
# merely CONTAINS "-Ofast" as a substring (a macro definition, not the
# compiler flag itself) must build successfully now that the conflict check
# is $(filter)-based (whole-word match) instead of $(findstring)-based
# (substring match). Real (non-dry-run) build with a scratch BUILD_DIR, so
# this also proves the allow-list/exact-token fix doesn't merely let parsing
# continue but actually still produces a correct binary.
FIND2_BUILD_DIR="$SCRATCH/find2_build"
LOG11="$SCRATCH/log_ofast_substring_false_positive"
if env EXTRA_CFLAGS='-DROUND9_NOTE=-Ofastness' "$MAKE_BIN" BUILD_DIR="$FIND2_BUILD_DIR" test-features >"$LOG11" 2>&1; then
    grep -q "PASS: log_erb_dfn_mean_cplx_unit_0_4k_v3" "$LOG11" \
        && pass "4a: EXTRA_CFLAGS='-DROUND9_NOTE=-Ofastness' (contains \"-Ofast\" as a substring, not the flag) builds and runs correctly (false positive fixed)" \
        || { fail "4a: EXTRA_CFLAGS='-DROUND9_NOTE=-Ofastness' build succeeded but did not report the expected PASS line"; cat "$LOG11" >&2; }
else
    fail "4a: EXTRA_CFLAGS='-DROUND9_NOTE=-Ofastness' unexpectedly FAILED (substring false positive regressed)"
    cat "$LOG11" >&2
fi

# Regression: a real, bare -Ofast / -ffast-math / -ffp-contract=<x> token
# (the exact-token match's true positives) must still be rejected -- one
# case per conflicting flag (the pre-existing 3.regression case above only
# ever covered -Ofast).
check_real_conflict_rejected() {
    label="$1"; flag="$2"
    log="$SCRATCH/log_real_conflict_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_')"
    if env EXTRA_CFLAGS="$flag" "$MAKE_BIN" -n test-features >"$log" 2>&1; then
        fail "4.$label: EXTRA_CFLAGS='$flag' unexpectedly SUCCEEDED (must be rejected)"
        cat "$log" >&2
        return
    fi
    if grep -q "FP policy conflict" "$log" && grep -qF -- "$flag" "$log"; then
        pass "4.$label: EXTRA_CFLAGS='$flag' correctly FAILS, identifying \"$flag\" (exact-token match still catches the real flag)"
    else
        fail "4.$label: EXTRA_CFLAGS='$flag' failed but did NOT identify \"$flag\" specifically"
        cat "$log" >&2
    fi
}
check_real_conflict_rejected "ofast"          "-Ofast"
check_real_conflict_rejected "fast-math"      "-ffast-math"
check_real_conflict_rejected "fp-contract"    "-ffp-contract=fast"

# The exact-token check's own internal variable (FP_CONFLICT_FLAGS) must
# itself be immune to a command-line/`-e` override -- otherwise `make
# FP_CONFLICT_FLAGS= EXTRA_CFLAGS=-Ofast` would silently defeat the whole
# conflict check while a real -Ofast still reaches the compiler (confirmed
# empirically as a genuine gap before the matching override-rejection guard
# was added alongside FP_CONFLICT_FLAGS itself).
check_override_rejected "FP_CONFLICT_FLAGS" "" "-Ofast" cmdline
check_override_rejected "FP_CONFLICT_FLAGS" "" "-Ofast" dashe

echo
echo "TOTAL: $((PASS_COUNT + FAIL_COUNT))  PASS: $PASS_COUNT  FAIL: $FAIL_COUNT"
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "run_makefile_build_safety_test: ALL PASS"
    exit 0
else
    echo "run_makefile_build_safety_test: FAIL"
    exit 1
fi
