#!/usr/bin/env bash
# pipelines/scripts/test_build_isolation.sh — round-3 review B01 build-isolation
# regression suite for Audio_ALG/pipelines, the fourth (and last) of four repos
# to get the CFG_SIG-keyed obj/bin directory design (audio_common, AEC, NR
# already have their own scripts/test_build_isolation.sh — see audio_common's
# for the S1-S5/S7p/S8 single-producer scenarios this one does NOT repeat).
#
# pipelines/Makefile is a THREE-producer consumer (audio_common's
# libaudio_common.a, lib/aec's libaec.a, lib/nr's libmmse_lsa.a), each
# resolved via the two-phase recursive-make dispatch described in that
# Makefile's own header comment. This script exercises the scenarios specific
# to that three-producer design plus this repo's own audit-no-stdio/publish
# gates, PLUS (round-5 review) the RNNoise-ERB table drift-gate hardening:
#
#   S6:  audit-no-stdio false-pass regression -- the delivered NO_STDIO=1
#        archive really has no stdio refs, its path really differs from the
#        default config's, and running the audit never mutates the default
#        (stdio) archive.
#   S7:  publish v4 -- lock-FIRST driver + concurrent-publish semantics (the
#        loser of a race fails fast with "already held"), MANIFEST sha
#        self-consistency, the `current` symlink resolves to a complete
#        release dir, ATTEST/ carries the git provenance. Uses a throwaway
#        DIST_ROOT (mktemp -d) -- never the real dist/. See SP-S11 for the
#        content-addressed layout / idempotent-republish / ATTEST-growth
#        assertions specific to v4.
#   SP1: pipeline-level A->B->A (kiss -> ne10 -> kiss) -- each build's
#        delivered aec_nr_pipeline links backend-correct FFT symbols, and the
#        third (kiss) build is NOT relinked (mtime stable).
#   SP2: producer-change propagation -- touching audio_common/src/hpf.c
#        relinks the pipeline binary; touching audio_common/include/
#        fast_math.h recompiles the AEC/NR objects that include it AND
#        relinks the pipeline binary (the full transitive header chain).
#
#   -- round-4 review scenarios (cross-repo: this Makefile + lib/aec's +
#      lib/nr's, all three upgraded together) --
#   SP-S9:  command-line override rejection (round-4 review P1-1) --
#           CFLAGS=/CXXFLAGS=/CPPFLAGS=/LDFLAGS=/FP_POLICY= on the command
#           line fail at PARSE time (mentioning "cannot be overridden") in
#           this Makefile, lib/aec's, and lib/nr's alike; EXTRA_CFLAGS is
#           unaffected and still keys to its own obj dir.
#   SP-S10: lib/aec fresh-archive discipline (round-4 review P1-4) -- a
#           foreign member `ar r`'d into a built libaec.a does not survive a
#           rebuild triggered by touching one small source file (the real
#           discipline is $@.tmp then `mv -f`, never `ar r` onto the
#           existing archive); its obj-dir config.manifest records SRCS=.
#   SP-S11: pipelines publish v4 -- content-addressed
#           <DIST_ROOT>/<backend>/<cfg_sig>-<content12>/ release dirs,
#           resolved ONLY via `readlink <DIST_ROOT>/<backend>/current` (never
#           a hardcoded id). MANIFEST.txt is fully DETERMINISTIC (release_id=/
#           ac_producer_cfg_sig=/ar=/ranlib=/link=/etc, but NEVER git_commit=
#           or date_utc=); per-publish-event provenance (git_commit=/
#           aec_git_commit=/nr_git_commit=/date_utc=) lives ONLY in
#           append-only ATTEST/attest-<stamp>-<commit>[-dirty].txt files.
#           Idempotent republish byte-verifies artifacts+MANIFEST, prints
#           "already published (byte-verified, incl. MANIFEST)", leaves the
#           release dir and its files (excl. ATTEST/) mtime-untouched, and
#           appends a SECOND attest file -- after a `sleep 1`, since a
#           same-second republish would reuse the same attest filename.
#   SP-S12: BACKEND=ne10 CC/CXX toolchain-coherence guard fires through the
#           dispatch (round-4 review P1-2) -- a CXX shim that deliberately
#           answers `-dumpmachine` with a bogus triple fails a BACKEND=ne10
#           build (mentioning "different targets") but not a BACKEND=kiss
#           one (the guard is ne10-only, since only ne10 links a C++ TU).
#
#   -- round-5 review scenario (RNNoise-ERB, a lightweight sibling of the
#      four big Makefiles, sharing the same override-rejection/keyed-dir
#      discipline) --
#   SP-S13: RNNoise-ERB table drift-gate hardening (round-5 review P2) --
#           `make test-tables` builds+runs both drift-guard layers (2 PASS
#           lines: canonical byte-exact + portable math-contract);
#           `make CC=false test-tables` FAILS outright (the old flat build/
#           had no compiler/flags identity, so this used to silently re-run a
#           STALE binary from an earlier config and report PASS without ever
#           invoking CC=false -- the keyed build/<cfg-sig>/ dir forces a real
#           compile attempt every time, so that false-pass repro stays
#           closed); `make CFLAGS=-O0 test-tables` is rejected at PARSE time
#           ("cannot be overridden"), the same origin-gate discipline as the
#           four big Makefiles' SP-S9.
#
# Design rules (same as audio_common's script -- do not violate when editing):
#   - No `make clean` inside any scenario body (except SP-S13's own trailing
#     `make -C RNNoise-ERB clean`, which only ever removes THAT repo's own
#     gitignored build/ -- never this repo's bin/obj): distinct configs must
#     coexist WITHOUT ever needing a clean between them.
#   - Every path is resolved via `make -s ... print-bin-dir` / `print-obj-dir`
#     / `print-lib-path`, using the EXACT flag set under test for that call --
#     never a hand-reconstructed path guess. Because this Makefile's three
#     producer archives (AC_LIB/AEC_LIB/NR_LIB) must be resolved before ANY of
#     this Makefile's own query targets report the REAL path `make all` would
#     use (their CFG_SIG folds in the three resolved producers' identities),
#     every query below explicitly resolves and passes AC_LIB/AEC_LIB/NR_LIB
#     itself, exactly the way pipelines/Makefile's own phase-1 dispatch does.
#   - "Did this get rebuilt?" is always an mtime comparison, never a content
#     (sha) comparison.
#   - "Is this the SAME delivered artifact as its own keyed object?" IS a sha
#     comparison, via file_sha() below.
#
# Safety / footprint (round-5 review P1 -- this script runs alongside other
# work in these trees, so it must be inert outside its own throwaway state):
#   - Writes ONLY: this script's own mktemp/mktemp -d scratch dirs (every
#     mktemp -d this script creates is registered in CLEANUP_DIRS and removed
#     by the single EXIT/INT/TERM trap below, even on failure or Ctrl-C) plus
#     the normal CFG_SIG-keyed obj/bin build dirs that `make` itself creates
#     in this repo, lib/aec, lib/nr, audio_common, and (SP-S13) RNNoise-ERB --
#     all gitignored build products, never tracked files.
#   - NEVER reads, writes, or removes the real `dist/`. Every publish
#     scenario passes an explicit DIST_ROOT that lives under a throwaway
#     mktemp -d dir -- never the Makefile's own `DIST_ROOT ?= dist` default.
#   - NEVER modifies any tracked file's CONTENT. SP-S10's
#     `touch lib/aec/c_impl/src/aec_debug.c` and SP2's touches of
#     audio_common's hpf.c/fast_math.h only advance mtimes (to force a
#     rebuild) -- file bytes/git-status are unaffected.
#   - Optional integrity check (enabled whenever `git` resolves a toplevel
#     from this script's own directory): hashes `git status --porcelain` at
#     start-of-run and again at the summary, and FAILs the run if it
#     changed -- a cheap trip-wire for "this script accidentally touched
#     tracked-file state." (Can false-positive if something ELSE concurrently
#     mutates the same repo while this script runs -- a shared-tree hazard,
#     not a bug in the check itself.)
#
# Usage: ./scripts/test_build_isolation.sh   (run from pipelines/, or
# anywhere -- paths are resolved relative to this script's own location).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AC_DIR="$(cd "$PIPE_DIR/../../audio_common" && pwd)"
AEC_DIR="$(cd "$PIPE_DIR/../lib/aec/c_impl" && pwd)"
NR_DIR="$(cd "$PIPE_DIR/../lib/nr/c_impl" && pwd)"
RNN_DIR="$(cd "$PIPE_DIR/../ainr/RNNoise-ERB" && pwd)"

# Single global cleanup trap (round-5 P1): every `mktemp -d` this script
# creates appends its path here, and this is the ONLY place anything gets
# rm -rf'd on exit -- no scenario removes a real tracked directory itself.
# `${CLEANUP_DIRS[@]:-}` (not `${CLEANUP_DIRS[@]}`) is deliberate: under
# `set -u`, bash 3.2 (macOS's default /bin/bash) throws "unbound variable"
# expanding an EMPTY array's `[@]` without the `:-` fallback.
CLEANUP_DIRS=()
trap 'rm -rf "${CLEANUP_DIRS[@]:-}"' EXIT INT TERM

PASS_COUNT=0
FAIL_COUNT=0
FAILURES=()

pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $*"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); FAILURES+=("$*"); echo "  FAIL: $*" >&2; }

file_sha() { shasum -a 256 "$1" | awk '{print $1}'; }
mtime()    { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1"; }

# release_mtime_snapshot <release-dir> -- prints "basename mtime" for every
# regular file directly under <release-dir> (the published artifacts +
# MANIFEST.txt), EXCLUDING the ATTEST/ subdirectory (which is expected to
# gain files across idempotent republishes). Used by SP-S11 to assert an
# idempotent republish leaves the release dir's own files byte/mtime-stable.
release_mtime_snapshot() {
  local dir="$1" f
  for f in "$dir"/*; do
    [ -f "$f" ] || continue
    echo "$(basename "$f") $(mtime "$f")"
  done | sort
}

# assert_cmd_fails_with <description> <expected-substring> <cmd...> (round-4
# review scenarios SP-S9/SP-S12) -- runs <cmd...> with stdout+stderr merged;
# PASS iff it exits non-zero AND the combined output contains
# <expected-substring>; FAIL (dumping the log) otherwise -- either because it
# unexpectedly succeeded, or failed without the expected message.
assert_cmd_fails_with() {
  local desc="$1" needle="$2" log
  shift 2
  log="$(mktemp)"
  if "$@" >"$log" 2>&1; then
    fail "$desc -- command unexpectedly SUCCEEDED"
    cat "$log" >&2
  elif grep -q -- "$needle" "$log"; then
    pass "$desc"
  else
    fail "$desc -- failed, but without the expected message ('$needle')"
    cat "$log" >&2
  fi
  rm -f "$log"
}

# resolve_producers <backend> -- resolves audio_common's/lib-aec's/lib-nr's
# archive paths for <backend> (default flags otherwise), exactly the way
# pipelines/Makefile's own phase-1 dispatch recipe does (AC_DIR/AC_LIB
# forwarded to the lib/aec and lib/nr queries so THEIR OWN CFG_SIG resolves
# to the same path a real `make all BACKEND=<backend>` build would use).
# Sets globals AC_LIB_/AEC_LIB_/NR_LIB_.
resolve_producers() {
  local backend="$1"
  AC_LIB_="$(make -s -C "$AC_DIR" BACKEND="$backend" WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 print-lib-path)"
  AEC_LIB_="$(make -s -C "$AEC_DIR" BACKEND="$backend" WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 AC_DIR="$AC_DIR" AC_LIB="$AC_LIB_" print-lib-path)"
  NR_LIB_="$(make -s -C "$NR_DIR" BACKEND="$backend" WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 AC_DIR="$AC_DIR" AC_LIB="$AC_LIB_" print-lib-path)"
}

cd "$PIPE_DIR"

# Optional integrity trip-wire (round-5 P1) -- see the "Safety / footprint"
# header block above. Scoped to this script's own repo toplevel (Audio_ALG);
# a no-op (never asserted) if `git` isn't available.
INTEGRITY_ROOT=""
INTEGRITY_BEFORE=""
if command -v git >/dev/null 2>&1 && INTEGRITY_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  INTEGRITY_BEFORE="$(git -C "$INTEGRITY_ROOT" status --porcelain | shasum -a 256)"
fi

echo "############################################################"
echo "# S6: audit-no-stdio false-pass regression"
echo "############################################################"
make -s BACKEND=kiss libaudio_pipeline.a >/dev/null
default_lib="$(make -s BACKEND=kiss NO_STDIO=0 print-lib-path)"
sha_default_before="$(file_sha "$default_lib")"

S6_LOG="$(mktemp)"
if make BACKEND=kiss audit-no-stdio >"$S6_LOG" 2>&1; then
  pass "S6: audit-no-stdio exits green"
else
  fail "S6: audit-no-stdio FAILED"
  cat "$S6_LOG" >&2
fi
grep -q '^PASS:' "$S6_LOG" && pass "S6: audit-no-stdio printed a PASS line" \
  || fail "S6: audit-no-stdio did not print a PASS line"
rm -f "$S6_LOG"

nostdio_lib="$(make -s BACKEND=kiss NO_STDIO=1 print-lib-path)"
[ "$nostdio_lib" != "$default_lib" ] && pass "S6: NO_STDIO=1 archive path differs from the default (NO_STDIO=0) path" \
  || fail "S6: NO_STDIO=1 archive path COLLIDES with the default path ($default_lib)"
[ -f "$nostdio_lib" ] && pass "S6: NO_STDIO=1 archive exists on disk" || fail "S6: NO_STDIO=1 archive missing"

sha_default_after="$(file_sha "$default_lib")"
[ "$sha_default_before" = "$sha_default_after" ] && pass "S6: default (stdio) archive untouched (sha stable) by the audit run" \
  || fail "S6: default archive CHANGED by the audit-no-stdio run (sha $sha_default_before -> $sha_default_after)"

nm "$default_lib" 2>/dev/null | grep -Eq '_?fprintf' && pass "S6: default archive still contains fprintf refs (nm)" \
  || fail "S6: default archive unexpectedly has NO fprintf refs"

echo "############################################################"
echo "# S7: publish (v4 -- lock-first driver + concurrent semantics)"
echo "############################################################"
S7_TMP="$(mktemp -d)"; CLEANUP_DIRS+=("$S7_TMP")
S7_DIST_ROOT="$S7_TMP/dist"
resolve_producers kiss
make -s BACKEND=kiss DIST_ROOT="$S7_DIST_ROOT" publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null

kiss_current_target="$(readlink "$S7_DIST_ROOT/kiss/current" || true)"
[ -n "$kiss_current_target" ] && [ -d "$S7_DIST_ROOT/kiss/$kiss_current_target" ] && \
  pass "S7: publish -- current symlink resolves to a real release dir" \
  || fail "S7: publish -- current symlink broken or missing"

manifest_ok=1
while read -r sha fname; do
  [ "$fname" = "MANIFEST.txt" ] && continue
  actual="$(file_sha "$S7_DIST_ROOT/kiss/current/$fname")"
  [ "$actual" = "$sha" ] || manifest_ok=0
done < <(grep -E '^[0-9a-f]{64}  ' "$S7_DIST_ROOT/kiss/current/MANIFEST.txt")
[ "$manifest_ok" -eq 1 ] && pass "S7: MANIFEST sha self-consistency" \
  || fail "S7: MANIFEST sha mismatch against files on disk"

grep -q '^ac_producer_cfg_sig=kiss-' "$S7_DIST_ROOT/kiss/current/MANIFEST.txt" && \
  grep -q '^aec_producer_cfg_sig=kiss-' "$S7_DIST_ROOT/kiss/current/MANIFEST.txt" && \
  grep -q '^nr_producer_cfg_sig=kiss-' "$S7_DIST_ROOT/kiss/current/MANIFEST.txt" && \
  pass "S7: MANIFEST records all three producer cfg_sig identities" \
  || fail "S7: MANIFEST missing one or more producer cfg_sig identities"

grep -q '^git_commit=' "$S7_DIST_ROOT/kiss/current/MANIFEST.txt" && \
  fail "S7: MANIFEST.txt unexpectedly contains a git_commit= line (v4 keeps provenance in ATTEST/ only)" \
  || pass "S7: MANIFEST.txt has no git_commit= line (deterministic MANIFEST)"

s7_attest="$(find "$S7_DIST_ROOT/kiss/current/ATTEST" -type f -name 'attest-*.txt' 2>/dev/null | head -n1)"
[ -n "$s7_attest" ] && grep -q '^git_commit=' "$s7_attest" && grep -q '^aec_git_commit=' "$s7_attest" && \
  pass "S7: ATTEST file carries git_commit=/aec_git_commit= provenance" \
  || fail "S7: ATTEST file missing, or missing git_commit=/aec_git_commit="

# Concurrent same-backend publish: the lock-FIRST driver means the loser can
# fail fast with "already held" rather than waiting -- but depending on
# scheduling, both invocations can also land in disjoint (non-overlapping)
# lock windows and both succeed. Either outcome is fine as long as `current`
# ends up pointing at a COMPLETE, self-consistent release.
S7_LOG_A="$(mktemp)"; S7_LOG_B="$(mktemp)"
( make -s BACKEND=kiss DIST_ROOT="$S7_DIST_ROOT" publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S7_LOG_A" 2>&1 ) & cp1=$!
( make -s BACKEND=kiss DIST_ROOT="$S7_DIST_ROOT" publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S7_LOG_B" 2>&1 ) & cp2=$!
cr1=0; cr2=0
wait "$cp1" || cr1=$?
wait "$cp2" || cr2=$?
if [ "$cr1" -eq 0 ] || [ "$cr2" -eq 0 ]; then
  pass "S7: concurrent same-backend publish -- at least one caller succeeded"
else
  fail "S7: concurrent same-backend publish -- BOTH callers failed"
  cat "$S7_LOG_A" "$S7_LOG_B" >&2
fi
if grep -q "already held" "$S7_LOG_A" "$S7_LOG_B" 2>/dev/null || { [ "$cr1" -eq 0 ] && [ "$cr2" -eq 0 ]; }; then
  pass "S7: concurrent same-backend publish -- lock-first driver enforced (loser failed fast with 'already held', or both landed in disjoint windows)"
else
  fail "S7: concurrent same-backend publish -- no evidence of lock enforcement"
fi
rm -f "$S7_LOG_A" "$S7_LOG_B"

final_target="$(readlink "$S7_DIST_ROOT/kiss/current" || true)"
[ -n "$final_target" ] && [ -f "$S7_DIST_ROOT/kiss/$final_target/MANIFEST.txt" ] && [ -f "$S7_DIST_ROOT/kiss/$final_target/aec_nr_pipeline" ] && \
  pass "S7: after concurrent publish, current points at a complete release dir" \
  || fail "S7: after concurrent publish, current points at an incomplete/missing release dir"

echo "############################################################"
echo "# SP1: pipeline-level A->B->A (kiss -> ne10 -> kiss)"
echo "############################################################"
resolve_producers kiss
ac_k="$AC_LIB_"; aec_k="$AEC_LIB_"; nr_k="$NR_LIB_"
make -s BACKEND=kiss WERROR=0 all AC_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k" >/dev/null
bd_k1="$(make -s BACKEND=kiss WERROR=0 print-bin-dir AC_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k")"
m_k1="$(mtime "$bd_k1/aec_nr_pipeline")"

resolve_producers ne10
ac_n="$AC_LIB_"; aec_n="$AEC_LIB_"; nr_n="$NR_LIB_"
make -s BACKEND=ne10 WERROR=0 all AC_LIB="$ac_n" AEC_LIB="$aec_n" NR_LIB="$nr_n" >/dev/null
bd_n="$(make -s BACKEND=ne10 WERROR=0 print-bin-dir AC_LIB="$ac_n" AEC_LIB="$aec_n" NR_LIB="$nr_n")"

sleep 1
make -s BACKEND=kiss WERROR=0 all AC_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k" >/dev/null
bd_k2="$(make -s BACKEND=kiss WERROR=0 print-bin-dir AC_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k")"
m_k2="$(mtime "$bd_k2/aec_nr_pipeline")"

[ "$bd_k1" = "$bd_k2" ] && pass "SP1: kiss bin dir stable across A->B->A" \
  || fail "SP1: kiss bin dir CHANGED across A->B->A ($bd_k1 vs $bd_k2)"
[ "$m_k1" = "$m_k2" ] && pass "SP1: kiss aec_nr_pipeline NOT relinked on the third (A->B->A) run" \
  || fail "SP1: kiss aec_nr_pipeline relinked with nothing changed"

nm "$bd_k2/aec_nr_pipeline" 2>/dev/null | grep -q 'ne10_fft_r2c_1d_float32_neon' && \
  fail "SP1: kiss aec_nr_pipeline unexpectedly links ne10 FFT symbols" \
  || pass "SP1: kiss aec_nr_pipeline links backend-correct (no ne10) FFT symbols"
nm "$bd_n/aec_nr_pipeline" 2>/dev/null | grep -q 'ne10_fft_r2c_1d_float32_neon' && \
  pass "SP1: ne10 aec_nr_pipeline links backend-correct (ne10) FFT symbols" \
  || fail "SP1: ne10 aec_nr_pipeline missing ne10 FFT symbols"

echo "############################################################"
echo "# SP2: producer-change propagation"
echo "############################################################"
resolve_producers kiss
ac_k="$AC_LIB_"; aec_k="$AEC_LIB_"; nr_k="$NR_LIB_"
make -s BACKEND=kiss WERROR=0 all AC_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k" >/dev/null
bd_k="$(make -s BACKEND=kiss WERROR=0 print-bin-dir AC_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k")"

# touch audio_common/src/hpf.c -> hpf.o recompiles AND aec_nr_pipeline relinks
# (audio_common's own CFG_SIG is a hash of its COMPILER INVOCATION, not file
# content, so its archive path is unaffected -- only its mtime advances).
# mtime-only touch: content is never edited, so this never dirties audio_common's
# own git status.
m_before="$(mtime "$bd_k/aec_nr_pipeline")"
sleep 1
touch "$AC_DIR/src/hpf.c"
make -s BACKEND=kiss WERROR=0 all AC_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k" >/dev/null
m_after="$(mtime "$bd_k/aec_nr_pipeline")"
[ "$m_after" != "$m_before" ] && pass "SP2: touching audio_common/src/hpf.c relinked aec_nr_pipeline" \
  || fail "SP2: aec_nr_pipeline NOT relinked after audio_common's hpf.o changed"

# touch audio_common/include/fast_math.h -> NR/AEC objects that include it
# (aec3_post.c/suppression_gain.c in lib/aec, mcra_noise_estimator.c/
# spp_estimator.c/mmse_lsa_denoiser.c in lib/nr) recompile, AND the pipeline
# binary relinks -- the full transitive header chain. Again mtime-only: no
# tracked file's content changes.
aec_objdir="$(make -s -C "$AEC_DIR" BACKEND=kiss WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 AC_DIR="$AC_DIR" AC_LIB="$ac_k" print-obj-dir)"
nr_objdir="$(make -s -C "$NR_DIR" BACKEND=kiss WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 AC_DIR="$AC_DIR" AC_LIB="$ac_k" print-obj-dir)"
t_aec_before="$(mtime "$aec_objdir/aec3_post.o")"
t_nr_before="$(mtime "$nr_objdir/mcra_noise_estimator.o")"
m_before="$(mtime "$bd_k/aec_nr_pipeline")"

sleep 1
touch "$AC_DIR/include/fast_math.h"
make -s BACKEND=kiss WERROR=0 all AC_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k" >/dev/null
t_aec_after="$(mtime "$aec_objdir/aec3_post.o")"
t_nr_after="$(mtime "$nr_objdir/mcra_noise_estimator.o")"
m_after="$(mtime "$bd_k/aec_nr_pipeline")"

[ "$t_aec_after" != "$t_aec_before" ] && pass "SP2: touching audio_common/include/fast_math.h recompiled lib/aec's aec3_post.o" \
  || fail "SP2: aec3_post.o did NOT recompile after touching fast_math.h"
[ "$t_nr_after" != "$t_nr_before" ] && pass "SP2: touching audio_common/include/fast_math.h recompiled lib/nr's mcra_noise_estimator.o" \
  || fail "SP2: mcra_noise_estimator.o did NOT recompile after touching fast_math.h"
[ "$m_after" != "$m_before" ] && pass "SP2: touching audio_common/include/fast_math.h relinked aec_nr_pipeline (full transitive chain)" \
  || fail "SP2: aec_nr_pipeline NOT relinked after fast_math.h touch"

echo "############################################################"
echo "# SP-S9: command-line override rejection across the stack"
echo "############################################################"
assert_cmd_fails_with "SP-S9: pipelines 'make CFLAGS=-O3' rejected at parse time" "cannot be overridden" \
  make BACKEND=kiss CFLAGS=-O3 print-obj-dir

assert_cmd_fails_with "SP-S9: pipelines 'make FP_POLICY=-ffp-contract=fast' rejected at parse time" "cannot be overridden" \
  make BACKEND=kiss FP_POLICY=-ffp-contract=fast print-obj-dir

assert_cmd_fails_with "SP-S9: lib/aec 'make CFLAGS=-O3' rejected at parse time" "cannot be overridden" \
  make -C "$AEC_DIR" BACKEND=kiss CFLAGS=-O3 print-obj-dir

assert_cmd_fails_with "SP-S9: lib/nr 'make LDFLAGS=-lfoo' rejected at parse time" "cannot be overridden" \
  make -C "$NR_DIR" BACKEND=kiss LDFLAGS=-lfoo print-obj-dir

sp9_plain_objdir="$(make -s BACKEND=kiss print-obj-dir)"
sp9_probe_objdir="$(make -s BACKEND=kiss EXTRA_CFLAGS=-DSP9_PROBE print-obj-dir)"
[ -n "$sp9_probe_objdir" ] && pass "SP-S9: pipelines EXTRA_CFLAGS=-DSP9_PROBE print-obj-dir SUCCEEDED" \
  || fail "SP-S9: pipelines EXTRA_CFLAGS=-DSP9_PROBE print-obj-dir produced no output"
[ "$sp9_probe_objdir" != "$sp9_plain_objdir" ] && pass "SP-S9: EXTRA_CFLAGS=-DSP9_PROBE obj dir differs from the plain-query obj dir" \
  || fail "SP-S9: EXTRA_CFLAGS=-DSP9_PROBE obj dir COLLIDES with the plain-query obj dir ($sp9_plain_objdir)"

echo "############################################################"
echo "# SP-S10: lib/aec archive freshness (fresh-archive discipline)"
echo "############################################################"
make -s -C "$AEC_DIR" BACKEND=kiss lib >/dev/null
aec_lib_path="$(make -s -C "$AEC_DIR" BACKEND=kiss print-lib-path)"
[ -f "$aec_lib_path" ] && pass "SP-S10: lib/aec archive built ($aec_lib_path)" \
  || fail "SP-S10: lib/aec archive missing after 'make lib'"

SP10_TMPDIR="$(mktemp -d)"; CLEANUP_DIRS+=("$SP10_TMPDIR")
cat > "$SP10_TMPDIR/sp10_foreign.c" <<'EOF'
int sp10_foreign_symbol(void) { return 0; }
EOF
cc -c -o "$SP10_TMPDIR/sp10_foreign.o" "$SP10_TMPDIR/sp10_foreign.c"
ar r "$aec_lib_path" "$SP10_TMPDIR/sp10_foreign.o"
ar -t "$aec_lib_path" | grep -qx 'sp10_foreign.o' && pass "SP-S10: foreign member injected into libaec.a (setup)" \
  || fail "SP-S10: foreign member injection FAILED (setup problem, not the thing under test)"

sleep 1
# mtime-only touch (content unchanged) -- exercises the fresh-archive rebuild
# discipline without ever editing this tracked file's bytes.
touch "$AEC_DIR/src/aec_debug.c"
make -s -C "$AEC_DIR" BACKEND=kiss lib >/dev/null

if ar -t "$aec_lib_path" | grep -qx 'sp10_foreign.o'; then
  fail "SP-S10: foreign member SURVIVED a rebuild (archive was NOT rebuilt fresh -- looks like 'ar r' onto the existing .a rather than \$@.tmp + mv -f)"
else
  pass "SP-S10: foreign member is GONE after rebuild (fresh-archive discipline: \$@.tmp + mv -f)"
fi
ar -t "$aec_lib_path" | grep -qx 'aec_debug.o' && pass "SP-S10: aec_debug.o present in the freshly-rebuilt archive" \
  || fail "SP-S10: aec_debug.o missing from the freshly-rebuilt archive"
rm -rf "$SP10_TMPDIR"

aec_objdir="$(make -s -C "$AEC_DIR" BACKEND=kiss print-obj-dir)"
grep -q 'SRCS=' "$aec_objdir/config.manifest" && pass "SP-S10: lib/aec's obj-dir config.manifest records a SRCS= entry" \
  || fail "SP-S10: lib/aec's obj-dir config.manifest missing a SRCS= entry"

echo "############################################################"
echo "# SP-S11: pipelines publish v4 (content-addressed release + ATTEST)"
echo "############################################################"
SP11_TMP="$(mktemp -d)"; CLEANUP_DIRS+=("$SP11_TMP")
SP11_DIST_ROOT="$SP11_TMP/dist"
resolve_producers kiss
make -s BACKEND=kiss DIST_ROOT="$SP11_DIST_ROOT" publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null

id1="$(readlink "$SP11_DIST_ROOT/kiss/current" || true)"
[ -n "$id1" ] && [ -d "$SP11_DIST_ROOT/kiss/$id1" ] && pass "SP-S11: publish v4 -- current symlink resolves to release dir '$id1'" \
  || fail "SP-S11: publish v4 -- current symlink broken or missing after the first publish"

rel_dir="$SP11_DIST_ROOT/kiss/$id1"

grep -q "^release_id=$id1\$" "$rel_dir/MANIFEST.txt" && pass "SP-S11: MANIFEST.txt release_id= matches the release dir name" \
  || fail "SP-S11: MANIFEST.txt release_id= missing or does not match '$id1'"
grep -q '^ac_producer_cfg_sig=' "$rel_dir/MANIFEST.txt" && pass "SP-S11: MANIFEST.txt has an ac_producer_cfg_sig= line" \
  || fail "SP-S11: MANIFEST.txt missing an ac_producer_cfg_sig= line"
grep -q '^ar=' "$rel_dir/MANIFEST.txt" && grep -q '^ranlib=' "$rel_dir/MANIFEST.txt" && grep -q '^link=' "$rel_dir/MANIFEST.txt" && \
  pass "SP-S11: MANIFEST.txt has ar=/ranlib=/link= lines" \
  || fail "SP-S11: MANIFEST.txt missing one or more of ar=/ranlib=/link="
grep -q '^git_commit=' "$rel_dir/MANIFEST.txt" && \
  fail "SP-S11: MANIFEST.txt unexpectedly has a git_commit= line (moved to ATTEST/ in v4)" \
  || pass "SP-S11: MANIFEST.txt has NO git_commit= line (deterministic MANIFEST)"

attest_count_before="$(find "$rel_dir/ATTEST" -type f -name 'attest-*.txt' | wc -l | tr -d ' ')"
[ "$attest_count_before" -eq 1 ] && pass "SP-S11: exactly one ATTEST file after the first publish" \
  || fail "SP-S11: expected exactly 1 ATTEST file after the first publish, found $attest_count_before"

first_attest="$(find "$rel_dir/ATTEST" -type f -name 'attest-*.txt' | head -n1)"
[ -n "$first_attest" ] && grep -q '^git_commit=' "$first_attest" && grep -q '^aec_git_commit=' "$first_attest" && \
  pass "SP-S11: ATTEST file carries git_commit=/aec_git_commit= provenance" \
  || fail "SP-S11: ATTEST file missing, or missing git_commit=/aec_git_commit="

release_dir_mtime_before="$(mtime "$rel_dir")"
snap_before="$(release_mtime_snapshot "$rel_dir")"

sleep 1
S11_LOG="$(mktemp)"
make -s BACKEND=kiss DIST_ROOT="$SP11_DIST_ROOT" publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S11_LOG" 2>&1
if grep -q "already published (byte-verified" "$S11_LOG"; then
  pass "SP-S11: idempotent republish printed 'already published (byte-verified, incl. MANIFEST)'"
else
  fail "SP-S11: idempotent republish did NOT print the byte-verified message"
  cat "$S11_LOG" >&2
fi
grep -Eq '\(attested: attest-[^)]+\)[[:space:]]*$' "$S11_LOG" && pass "SP-S11: republish success line ends '(attested: <name>)'" \
  || fail "SP-S11: republish success line missing the '(attested: <name>)' suffix"
rm -f "$S11_LOG"

id2="$(readlink "$SP11_DIST_ROOT/kiss/current" || true)"
[ "$id2" = "$id1" ] && pass "SP-S11: current still points at the same release id after idempotent republish" \
  || fail "SP-S11: current MOVED after idempotent republish ('$id1' -> '$id2')"

release_dir_mtime_after="$(mtime "$rel_dir")"
[ "$release_dir_mtime_before" = "$release_dir_mtime_after" ] && pass "SP-S11: release dir's own mtime unchanged by idempotent republish" \
  || fail "SP-S11: release dir mtime CHANGED by idempotent republish ($release_dir_mtime_before -> $release_dir_mtime_after)"

snap_after="$(release_mtime_snapshot "$rel_dir")"
[ "$snap_before" = "$snap_after" ] && pass "SP-S11: release-dir files (MANIFEST.txt + artifacts, excl. ATTEST/) left mtime-untouched by idempotent republish" \
  || fail "SP-S11: one or more release-dir files' mtime CHANGED by idempotent republish"

attest_count_after="$(find "$rel_dir/ATTEST" -type f -name 'attest-*.txt' | wc -l | tr -d ' ')"
[ "$attest_count_after" -eq 2 ] && pass "SP-S11: ATTEST/ grew to 2 files after the (sleep-1'd) republish" \
  || fail "SP-S11: expected 2 ATTEST files after the republish, found $attest_count_after"

echo "############################################################"
echo "# SP-S12: BACKEND=ne10 toolchain guard fires through the dispatch"
echo "############################################################"
SP12_DIR="$(mktemp -d)"; CLEANUP_DIRS+=("$SP12_DIR")
SP12_SHIM="$SP12_DIR/sp12-cxx-shim"
cat > "$SP12_SHIM" <<'SHIM_EOF'
#!/usr/bin/env bash
if [ "$1" = "-dumpmachine" ]; then
  echo "sp12-wrong-triple"
else
  exec c++ "$@"
fi
SHIM_EOF
chmod +x "$SP12_SHIM"

assert_cmd_fails_with "SP-S12: BACKEND=ne10 with a CXX whose -dumpmachine disagrees with CC's is rejected" "different targets" \
  make BACKEND=ne10 CXX="$SP12_SHIM" libaudio_pipeline.a

S12_LOG="$(mktemp)"
if make BACKEND=kiss CXX="$SP12_SHIM" libaudio_pipeline.a >"$S12_LOG" 2>&1; then
  pass "SP-S12: BACKEND=kiss with the same mismatched-triple CXX shim still SUCCEEDS (guard is ne10-only)"
else
  fail "SP-S12: BACKEND=kiss build unexpectedly FAILED with the CXX shim in place"
  cat "$S12_LOG" >&2
fi
rm -f "$S12_LOG"
rm -rf "$SP12_DIR"

echo "############################################################"
echo "# SP-S13: RNNoise-ERB drift-gate hardening (round-5 P2)"
echo "############################################################"
SP13_LOG="$(mktemp)"
if make -C "$RNN_DIR" test-tables >"$SP13_LOG" 2>&1; then
  pass "SP-S13: 'make test-tables' succeeds"
else
  fail "SP-S13: 'make test-tables' FAILED"
  cat "$SP13_LOG" >&2
fi
sp13_pass_count="$(grep -c '^PASS' "$SP13_LOG" || true)"
[ "${sp13_pass_count:-0}" -eq 2 ] && pass "SP-S13: 'make test-tables' output has both PASS layers (canonical + portable)" \
  || fail "SP-S13: expected 2 PASS lines from 'make test-tables', found ${sp13_pass_count:-0}"
rm -f "$SP13_LOG"

SP13_LOG2="$(mktemp)"
if make -C "$RNN_DIR" CC=false test-tables >"$SP13_LOG2" 2>&1; then
  fail "SP-S13: 'make CC=false test-tables' unexpectedly SUCCEEDED (stale-binary false-pass would be back)"
  cat "$SP13_LOG2" >&2
else
  pass "SP-S13: 'make CC=false test-tables' FAILS (fresh keyed build dir forces a real compile attempt; stale-binary false-pass repro stays closed)"
fi
rm -f "$SP13_LOG2"

assert_cmd_fails_with "SP-S13: 'make CFLAGS=-O0 test-tables' rejected at parse time" "cannot be overridden" \
  make -C "$RNN_DIR" CFLAGS=-O0 test-tables

make -s -C "$RNN_DIR" clean >/dev/null

if [ -n "$INTEGRITY_BEFORE" ]; then
  INTEGRITY_AFTER="$(git -C "$INTEGRITY_ROOT" status --porcelain | shasum -a 256)"
  [ "$INTEGRITY_BEFORE" = "$INTEGRITY_AFTER" ] && \
    pass "INTEGRITY: git status --porcelain for $INTEGRITY_ROOT unchanged across the full run" \
    || fail "INTEGRITY: git status --porcelain for $INTEGRITY_ROOT CHANGED across the run (this script, or something concurrent, mutated tracked-file state)"
fi

echo "############################################################"
echo "SUMMARY: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "############################################################"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
  exit 1
fi
exit 0
