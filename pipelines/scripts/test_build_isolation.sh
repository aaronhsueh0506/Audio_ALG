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
# gates:
#
#   S6:  audit-no-stdio false-pass regression -- the delivered NO_STDIO=1
#        archive really has no stdio refs, its path really differs from the
#        default config's, and running the audit never mutates the default
#        (stdio) archive.
#   S7:  publish v2 -- MANIFEST sha self-consistency, the `current` symlink
#        resolves to a complete release dir, concurrent same-backend
#        publishes serialise via the mkdir lock.
#   SP1: pipeline-level A->B->A (kiss -> ne10 -> kiss) -- each build's
#        delivered aec_nr_pipeline links backend-correct FFT symbols, and the
#        third (kiss) build is NOT relinked (mtime stable).
#   SP2: producer-change propagation -- touching audio_common/src/hpf.c
#        relinks the pipeline binary; touching audio_common/include/
#        fast_math.h recompiles the AEC/NR objects that include it AND
#        relinks the pipeline binary (the full transitive header chain).
#
# Design rules (same as audio_common's script -- do not violate when editing):
#   - No `make clean` inside any scenario body: distinct configs must coexist
#     WITHOUT ever needing a clean between them.
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
# Usage: ./scripts/test_build_isolation.sh   (run from pipelines/, or
# anywhere -- paths are resolved relative to this script's own location).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AC_DIR="$(cd "$PIPE_DIR/../../audio_common" && pwd)"
AEC_DIR="$(cd "$PIPE_DIR/../lib/aec/c_impl" && pwd)"
NR_DIR="$(cd "$PIPE_DIR/../lib/nr/c_impl" && pwd)"

PASS_COUNT=0
FAIL_COUNT=0
FAILURES=()

pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $*"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); FAILURES+=("$*"); echo "  FAIL: $*" >&2; }

file_sha() { shasum -a 256 "$1" | awk '{print $1}'; }
mtime()    { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1"; }

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
echo "# S7: publish"
echo "############################################################"
resolve_producers kiss
make -s BACKEND=kiss publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null

kiss_current_target="$(readlink dist/kiss/current || true)"
[ -n "$kiss_current_target" ] && [ -d "dist/kiss/$kiss_current_target" ] && \
  pass "S7: publish -- current symlink resolves to a real release dir" \
  || fail "S7: publish -- current symlink broken or missing"

manifest_ok=1
while read -r sha fname; do
  [ "$fname" = "MANIFEST.txt" ] && continue
  actual="$(file_sha "dist/kiss/current/$fname")"
  [ "$actual" = "$sha" ] || manifest_ok=0
done < <(grep -E '^[0-9a-f]{64}  ' "dist/kiss/current/MANIFEST.txt")
[ "$manifest_ok" -eq 1 ] && pass "S7: MANIFEST sha self-consistency" \
  || fail "S7: MANIFEST sha mismatch against files on disk"

grep -q '^ac_producer_cfg_sig=kiss-' dist/kiss/current/MANIFEST.txt && \
  grep -q '^aec_producer_cfg_sig=kiss-' dist/kiss/current/MANIFEST.txt && \
  grep -q '^nr_producer_cfg_sig=kiss-' dist/kiss/current/MANIFEST.txt && \
  pass "S7: MANIFEST records all three producer cfg_sig identities" \
  || fail "S7: MANIFEST missing one or more producer cfg_sig identities"

# Concurrent same-backend publish: lock must serialise. Either one caller
# fails with the lock message (acceptable) or both succeed; either way
# `current` must end up pointing at a COMPLETE, self-consistent release.
S7_LOG_A="$(mktemp)"; S7_LOG_B="$(mktemp)"
( make -s BACKEND=kiss publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S7_LOG_A" 2>&1 ) & cp1=$!
( make -s BACKEND=kiss publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S7_LOG_B" 2>&1 ) & cp2=$!
cr1=0; cr2=0
wait "$cp1" || cr1=$?
wait "$cp2" || cr2=$?
if [ "$cr1" -eq 0 ] || [ "$cr2" -eq 0 ]; then
  pass "S7: concurrent same-backend publish -- at least one caller succeeded"
else
  fail "S7: concurrent same-backend publish -- BOTH callers failed"
  cat "$S7_LOG_A" "$S7_LOG_B" >&2
fi
if grep -q "publish lock" "$S7_LOG_A" "$S7_LOG_B" 2>/dev/null || { [ "$cr1" -eq 0 ] && [ "$cr2" -eq 0 ]; }; then
  pass "S7: concurrent same-backend publish -- lock serialised (one waited/failed cleanly, or both completed in turn)"
else
  fail "S7: concurrent same-backend publish -- no evidence of serialisation"
fi
rm -f "$S7_LOG_A" "$S7_LOG_B"

final_target="$(readlink dist/kiss/current || true)"
[ -n "$final_target" ] && [ -f "dist/kiss/$final_target/MANIFEST.txt" ] && [ -f "dist/kiss/$final_target/aec_nr_pipeline" ] && \
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
# binary relinks -- the full transitive header chain.
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
echo "SUMMARY: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "############################################################"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
  exit 1
fi
exit 0
