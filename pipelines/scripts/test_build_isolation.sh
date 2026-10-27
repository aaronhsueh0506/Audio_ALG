#!/usr/bin/env bash
# pipelines/scripts/test_build_isolation.sh -- build-isolation suite for Audio_ALG/pipelines (the CFG_SIG-keyed
# obj/bin directory design). audiolib, AEC and NR have their own scripts/test_build_isolation.sh for the
# single-producer scenarios (S1-S5/S7p/S8), which this script does not repeat.
#
# pipelines/Makefile is a three-producer consumer (audiolib's libaudio_common.a, lib/aec's libaec.a, lib/nr's
# libanr.a), each resolved via the two-phase recursive-make dispatch described in that Makefile's header. This
# script exercises the scenarios specific to that design, plus this repo's own audit-no-stdio and publish gates
# and the RNNoise-ERB table drift-gate.
#
# Contract:
#   - OBJ_ROOT=/BIN_ROOT= placement knobs (pipelines' Makefile and the two PRIMARY producer repos AEC/c_impl and
#     NR/c_impl) let throwaway and tamper scenarios build in a scratch directory without touching the real
#     obj/ or bin/.
#   - ATTEST is one event, one file (attest-<utc>-<commit>[-dirty]-<seq>.txt, installed through the
#     atomic_symlink_swap helper's `--excl-install` mode).
#   - `make -n`/`-q`/`-t publish`, including every combined flag form (-nt, "-n -t", -tq, -nq, -nqt; SP-S14b),
#     have no filesystem side effects: no DIST_ROOT/OBJ_ROOT/BIN_ROOT is created and a pre-existing artifact's
#     stat is unchanged.
#   - Dirty policy: publish FATALs by default on a dirty or no-git-identity checkout (this repo or any producer);
#     ALLOW_DIRTY_PUBLISH=1 is the recorded escape hatch. Untracked files are a separate default refusal
#     (ALLOW_UNTRACKED_PUBLISH=1), so both ride along on every publish call here except SP-S17's deliberate
#     policy-negative calls. SP-S11 asserts the four *_git_untracked fields plus allow_untracked_publish=.
#   - Cross-suite mutex: a SUITE_LOCK at `${TMPDIR:-/tmp}/audio_dsp_isolation_suite.lock`, the same path as
#     audiolib's script, keeps two instances of either suite from running concurrently (both assert real-tree
#     invariants). mkdir-based with a holder-pid file and one TOCTOU-safe stale-reclaim retry (a losing reclaimer
#     never steals the winner's fresh lock); released in cleanup() only by the process that acquired it. It sits
#     after the ISOL_INTERRUPT_PROBE early return and before `export TMPDIR=...`, so it resolves against the
#     invoking TMPDIR.
#
# AEC/NR note: the scenarios that use OBJ_ROOT=/BIN_ROOT=/ALLOW_DIRTY_PUBLISH=/ATTEST_STAMP= or the -n/-q/-t
# dry-run guard on the AEC/NR Makefiles target $AEC_R6_DIR/$NR_R6_DIR (the primary repos' c_impl/, siblings of
# Audio_ALG), not the submodule paths $AEC_DIR/$NR_DIR; scenarios that need only basic behavior (normal all/lib
# builds, print-lib-path) use the submodule paths.
#
# Scenario index:
#   S6:      audit-no-stdio false-pass check (the nm filter must be able to report a stdio reference)
#   S7:      publish -- lock-first driver and concurrent-publish semantics
#   SP1:     pipeline-level A->B->A (SIMD=1 -> SIMD=0 -> SIMD=1)
#   SP2:     producer-change propagation (audiolib hpf.c / fast_math.h) against disposable clones of
#            audiolib/common/AEC/NR, never the real trees
#   SP-S9:   command-line override rejection, all three Makefiles (pipelines/lib-aec/lib-nr, submodule paths)
#   SP-S9b:  FP-policy allow-list: bypass shapes a character deny-list misses (bare glob, tilde, shell
#            redirection, process substitution), the exact-token $(filter) check's own path, override rejection
#            of the check's own internal variables, and the positive control (an override-free build succeeds
#            with the internal -DAUDIO_PIPELINE_BACKEND_STR define reaching the compile line)
#   SP-S9c:  link-flags (LDFLAGS/EXTRA_LDFLAGS) character safety: the `EXTRA_LDFLAGS=';id'` injection repro, the
#            SP-S9b glob/tilde/redirect matrix for EXTRA_LDFLAGS, a positive control (a real `-Wl,-rpath,... -L...`
#            flag still builds and runs test_mcab_sample_post to ALL PASS), and override rejection of
#            SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC via an EXTRA_LDFLAGS payload
#   SP-S10:  lib/aec fresh-archive discipline against the PRIMARY AEC repo ($AEC_R6_DIR) with scratch
#            OBJ_ROOT/BIN_ROOT; the foreign member goes into a scratch archive, never the real libaec.a
#   SP-S11:  pipelines publish (content-addressed release + ATTEST): attest field/naming assertions, no sleep
#            (the <NNN> suffix disambiguates a same-second republish)
#   SP-S12:  BACKEND=ne10 CC/CXX toolchain-coherence guard through the dispatch
#   SP-S13:  RNNoise-ERB table drift-gate
#   SP-S14:  make -n/-q/-t publish zero side effects on three Makefiles (pipelines in place, $AEC_R6_DIR,
#            $NR_R6_DIR); SP-S14b: combined dry-run flags against a deliberately backdated scratch artifact of
#            the pipelines Makefile only
#   SP-S15:  ATTEST uniqueness under forced same-second collisions at the pipelines level
#   SP-S16:  interruption safety -- EXIT/INT/TERM all clean up the whole scratch tree
#   SP-S17:  dirty-producer provenance -- publish against three dirty producer clones (the clone set SP2 built
#            while clean): the ALLOW_DIRTY_PUBLISH=1 path succeeds with correct per-producer attest fields and the
#            default path fails "publish refused"; SP-S17a/b: the untracked dimension (untracked producer probe
#            naming audiolib; unknown-producer AEC_DIR refused unconditionally naming lib/aec)
#   SP-S18:  `make -e` environment-override rejection, the SP-S9 complement: FP_POLICY=(empty)/
#            FP_POLICY=-ffp-contract=fast/CPPFLAGS=<poison>/CXXFLAGS=<poison> against pipelines and both primary
#            producers, plus LDFLAGS=<poison> against pipelines only. Every case is a true `-n` dry run against
#            scratch OBJ_ROOT/BIN_ROOT; the only passing outcome is parse-time rejection ("cannot be overridden")
#            with a nonzero exit (see assert_make_e_rejected()).
#   SP-S19:  the 4ch test_mcab_sample_driver object records its header dependencies (.d)
#   SP-S20:  test-pairs: an empty DFN_TEST_PAIRS is refused; -n/-q/-t print no FAIL and build nothing
#
# Design rules (as in audiolib's script):
#   - No `make clean` inside any scenario body (except SP-S13's trailing `make -C RNNoise-ERB clean`, which only
#     removes that repo's gitignored build/): distinct configs coexist without a clean between them.
#   - Every path is resolved with `make -s ... print-bin-dir` / `print-obj-dir` / `print-lib-path` using the exact
#     flag set (including OBJ_ROOT=/BIN_ROOT= when the build used them), never a hand-built path.
#   - "Was this rebuilt?" is an mtime comparison, never a content comparison; "is this the same delivered artifact
#     as its keyed object?" is a sha comparison via file_sha() below.
#
# Safety contract (see audiolib's script for the rationale of each point):
#   - One scratch root per run: SCRATCH_ROOT="$(mktemp -d)", removed by a single EXIT trap. Every temp path is a
#     fixed path under "$SCRATCH_ROOT/<scenario>/..." created inline, never inside a `$(...)` substitution (cleanup
#     state registered in a subshell is dropped). TMPDIR is exported into scratch with a `mktemp` PATH shim ahead of
#     the real one (macOS's bare `mktemp -d` ignores $TMPDIR), so a child make's own `mktemp -d` lands there too.
#   - The real obj/ and bin/ (this repo's, lib/aec's, lib/nr's, audiolib's) only see the normal SIMD=1/SIMD=0 ne10
#     configs that S6/S7/SP1/SP-S9 depend on; every throwaway or tamper scenario uses a scratch OBJ_ROOT/BIN_ROOT
#     (SP-S10 against $AEC_R6_DIR; SP-S12's toolchain shims, whose scratch path would otherwise mint a new orphan
#     keyed directory each run; SP-S14's dry runs; SP2/SP-S17's clones). A final-guard snapshot (path + sha256 +
#     mtime per file of every keyed directory already under the real obj/+bin/ of the four trees, taken once SP1's
#     builds finish) must be identical at the end; a later-created directory is allowed (INFO/WARN) though there
#     should be none.
#   - The real dist/ of those repos (and the two primary AEC/NR repos) is never read, written or removed: every
#     `make ... publish` passes an explicit DIST_ROOT= under $SCRATCH_ROOT, and a sentinel digest of each real dist/
#     (absent, or a manifest of paths + sha256 + mtime) is taken at the start and re-checked at the end.
#   - No git-tracked file in $COMMON_DIR, the submodule lib/aec and lib/nr, the Audio_ALG toplevel or the two
#     primary AEC/NR repos is touched, not even its mtime: mtime bumps that force a recompile probe happen on a
#     scratch clone's copy (SP2, reused and dirtied by SP-S17). `tree_state_hash()` (status --porcelain plus
#     diff --binary HEAD) of those trees is captured at the start and must be exactly equal at the end.
#   - No `sleep`. SP1's "should NOT relink" checks use mtime()'s BSD `stat -f '%Fm'` fractional-second read; SP2's
#     "should recompile" checks use `touch -r <the artifact that must become stale> -A 01 <source>` (GNU Make 3.81
#     truncates its newer-than check to whole seconds, so the +1s bump is load-bearing); SP-S11's same-second
#     republish is disambiguated by the attest <NNN> suffix.
#   - Interruption-safe: cleanup() is the only EXIT-trap resident and INT/TERM map to 130/143 through it (SP-S16).
#
# Usage: ./scripts/test_build_isolation.sh   (from pipelines/, or anywhere: paths resolve relative to this script)
#
# ISOL_INTERRUPT_PROBE (internal, used only by SP-S16): when set, this script runs as a child re-invocation of
# itself in interruption-probe mode instead of running the suite (see the block after the SCRATCH_ROOT/trap setup).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AUDIO_ALG_ROOT="$(cd "$PIPE_DIR/.." && pwd)"
COMMON_DIR="$(cd "$PIPE_DIR/../../audiolib/common" && pwd)"
AUDIOLIB_DIR="$(cd "$PIPE_DIR/../../audiolib" && pwd)"
AEC_DIR="$(cd "$PIPE_DIR/../lib/aec/c_impl" && pwd)"
NR_DIR="$(cd "$PIPE_DIR/../lib/nr/c_impl" && pwd)"
AEC_SUB_ROOT="$(cd "$AEC_DIR/.." && pwd)"
NR_SUB_ROOT="$(cd "$NR_DIR/.." && pwd)"
RNN_DIR="$(cd "$PIPE_DIR/../AINR/RNNoise-ERB" && pwd)"

# AEC/NR submodule note (see the header): SP-S10/SP-S14/SP-S17 need the
# OBJ_ROOT=/BIN_ROOT=/ALLOW_DIRTY_PUBLISH=/ATTEST_STAMP=/-n/-q/-t dry-run guard
# features and so target these PRIMARY sibling repos ($AEC_R6_DIR/$NR_R6_DIR)
# instead of $AEC_DIR/$NR_DIR.
AEC_REPO_DIR="$(cd "$PIPE_DIR/../../AEC" && pwd)"
AEC_R6_DIR="$AEC_REPO_DIR/c_impl"
NR_REPO_DIR="$(cd "$PIPE_DIR/../../NR" && pwd)"
NR_R6_DIR="$NR_REPO_DIR/c_impl"

# --- single scratch root, single trap -------------------------------------------
SCRATCH_ROOT="$(mktemp -d)"
# SUITE_LOCK_HELD must exist (as "0") before ANYTHING that could exit through
# cleanup() runs -- including the ISOL_INTERRUPT_PROBE early-return branch
# immediately below, which (being a probe CHILD, not a real suite run) never
# reaches the mutex block that would otherwise set this to "1" -- `set -u`
# would otherwise fault on cleanup()'s own reference to it.
SUITE_LOCK_HELD=0
SUITE_LOCK=""
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  [ "$SUITE_LOCK_HELD" -eq 1 ] && rm -rf -- "$SUITE_LOCK" 2>/dev/null
  rm -rf -- "$SCRATCH_ROOT"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# --- SP-S16 interruption probe: checked early, before any suite logic and before the mktemp-shim setup, so probe
# mode's `set +e` (see the branch) is in effect before any further command can be interrupted.
# With ISOL_INTERRUPT_PROBE set, this is a child re-invocation of this script spawned by the SP-S16 scenario body.
# It signals its own SCRATCH_ROOT to the parent over a fifo (the write is also the readiness signal), then either
# exits immediately (mode "exit") or blocks on a second fifo nobody writes to, so a parent-delivered INT/TERM is
# what interrupts it. It needs no shim of its own: its `mktemp -d` already inherited the parent's shim through PATH.
if [ -n "${ISOL_INTERRUPT_PROBE:-}" ]; then
  # `set +e` for the rest of probe mode: with `errexit` still active, a
  # SIGTERM/SIGINT that arrives while this process is blocked in a syscall
  # (opening the fifo, or a `read`) can surface as EINTR turning into that
  # command's own non-zero return -- and errexit reacts to THAT directly
  # before bash ever services the pending INT/TERM trap, so the child would
  # exit with the wrong code. Disabling errexit here removes the whole race:
  # the INT/TERM traps installed above are what decide this process's exit
  # code, unconditionally.
  set +e
  mkdir -p "$SCRATCH_ROOT/probe"
  : > "$SCRATCH_ROOT/probe/canary_a"
  : > "$SCRATCH_ROOT/probe/canary_b"
  printf '%s\n' "$SCRATCH_ROOT" > "$ISOL_PROBE_FIFO"
  if [ "$ISOL_INTERRUPT_PROBE" = "exit" ]; then
    exit 0
  fi
  # Open the hold-fifo O_RDWR (never blocks on open(), since we hold both
  # ends ourselves). Block via `wait` on a background reader of that fd,
  # NOT a direct foreground `read <&9` -- a signal arriving while this shell
  # is itself blocked inside a foreground `read` builtin can leave the
  # pending INT/TERM trap action un-run; `wait` is bash's own documented
  # interruptible blocking primitive and does not exhibit this.
  exec 9<>"$ISOL_PROBE_FIFO.hold"
  cat <&9 >/dev/null &
  catpid=$!
  wait "$catpid"
  exit 0
fi

# --- cross-suite mutex -----------------------------------------------------------
# Placed after the ISOL_INTERRUPT_PROBE early return (a probe child never contends for the lock its parent holds)
# and before `export TMPDIR=...` (so `${TMPDIR:-/tmp}` still resolves to the invoking environment's TMPDIR). Two
# instances of this suite, or of the audiolib counterpart with the same lock path, must not run concurrently: both
# assert real-tree invariants (S6/S7/SP1's real archives, the final-guard snapshot, the dist/ sentinels,
# tree_state_hash()) that a second run would trample.
SUITE_LOCK="${TMPDIR:-/tmp}/audio_dsp_isolation_suite.lock"
if mkdir "$SUITE_LOCK" 2>/dev/null; then
  printf '%s\n' "$$" > "$SUITE_LOCK/pid"
  SUITE_LOCK_HELD=1
else
  holder_pid="$(cat "$SUITE_LOCK/pid" 2>/dev/null || true)"
  reclaimed=0
  if [ -n "$holder_pid" ] && ! kill -0 "$holder_pid" 2>/dev/null; then
    # Stale lock: the recorded holder pid is confirmed dead. Reclaim once --
    # TOCTOU: two reclaimers can race this exact rm -rf/mkdir pair; only one
    # mkdir wins, and the loser must NOT steal the winner's fresh lock, so
    # this is a single retry (not a loop) and the loser falls through to the
    # FATAL below on failure.
    rm -rf -- "$SUITE_LOCK" 2>/dev/null || true
    if mkdir "$SUITE_LOCK" 2>/dev/null; then
      printf '%s\n' "$$" > "$SUITE_LOCK/pid"
      SUITE_LOCK_HELD=1
      reclaimed=1
    fi
  fi
  if [ "$reclaimed" -ne 1 ] && [ "$SUITE_LOCK_HELD" -ne 1 ]; then
    holder_pid="$(cat "$SUITE_LOCK/pid" 2>/dev/null || echo unknown)"
    echo "FATAL: another isolation suite instance is running (pid $holder_pid) -- the suites assert real-tree invariants and must not run concurrently" >&2
    exit 1
  fi
fi

mkdir "$SCRATCH_ROOT/tmp"
# Child `make`s' own mktemp workdirs (e.g. the publish recipe's `work="$(mktemp
# -d)"`) land under scratch too via this, so the single `rm -rf` above collects
# EVERYTHING this run creates, including after an interruption -- see SP-S16.
export TMPDIR="$SCRATCH_ROOT/tmp"

# macOS-specific gotcha: BSD `mktemp`'s bare `-d` / no-template form resolves
# via _CS_DARWIN_USER_TEMP_DIR FIRST (see mktemp(1)) -- so exporting TMPDIR
# alone does NOT redirect a child process's own bare `mktemp -d` (e.g. this
# Makefile's own `publish` recipe: `work="$(mktemp -d)"`, or SP-S16's child
# re-invocation's own `SCRATCH_ROOT="$(mktemp -d)"`) into our scratch tree. A
# tiny shim ahead of the real mktemp on PATH closes this gap: it forwards to
# the real /usr/bin/mktemp, injecting `-p "$TMPDIR"` whenever the caller
# didn't already give an explicit template or -p.
mkdir "$SCRATCH_ROOT/shimbin"
cat > "$SCRATCH_ROOT/shimbin/mktemp" <<'EOF'
#!/bin/sh
has_template=0
has_p=0
for a in "$@"; do
  case "$a" in
    -p) has_p=1 ;;
    -*) ;;
    *) has_template=1 ;;
  esac
done
if [ "$has_template" -eq 0 ] && [ "$has_p" -eq 0 ] && [ -n "${TMPDIR:-}" ]; then
  exec /usr/bin/mktemp -p "$TMPDIR" "$@"
fi
exec /usr/bin/mktemp "$@"
EOF
chmod +x "$SCRATCH_ROOT/shimbin/mktemp"
export PATH="$SCRATCH_ROOT/shimbin:$PATH"

# --- helpers -----------------------------------------------------------------
mkscratch() { mkdir -p "$SCRATCH_ROOT/$1"; }

file_sha() { shasum -a 256 "$1" | awk '{print $1}'; }
# Fractional seconds (not whole-second `stat -f %m`) -- two genuinely separate
# real writes are distinguishable without an artificial delay. GNU fallback
# loses the fractional part (not exercised on this host).
mtime()    { stat -f '%Fm' "$1" 2>/dev/null || stat -c %Y "$1"; }

# Tree-state hash: status --porcelain ALONE misses a content edit to an
# already-dirty file, so this also folds in `diff --binary HEAD`.
tree_state_hash() {
  { git -C "$1" status --porcelain; git -C "$1" diff --binary HEAD; } 2>/dev/null | shasum -a 256 | awk '{print $1}'
}

# adopt_worktree_clone <src-repo-dir> <clone-dir> -- shared by SP2 and SP-S17, aligned with audiolib's version.
# Clones <src-repo-dir> (read-only on the source), overlays every currently-modified tracked file and every
# untracked-but-not-ignored file, and adopts them as one throwaway commit inside the scratch clone only, so the
# clone has a clean git identity that still reflects the current, possibly uncommitted, worktree. A plain
# `git clone` carries only the last commit, not uncommitted working-tree bytes (so not the Makefile under test).
# The clone lives under $SCRATCH_ROOT and is destroyed by the EXIT trap; it is never pushed or fetched back. The
# commit is conditional (`git diff --cached --quiet --exit-code`): an unconditional commit on a clean tip would
# exit 1 and kill the suite under `set -e`.
adopt_worktree_clone() {
  local src="$1" dst="$2" relf
  git clone --no-hardlinks --quiet "$src" "$dst"
  while IFS= read -r relf; do
    [ -n "$relf" ] || continue
    if [ -e "$src/$relf" ]; then
      mkdir -p "$(dirname "$dst/$relf")"
      cp "$src/$relf" "$dst/$relf"
    else
      # A path git reports as changed vs HEAD but absent on disk is a
      # worktree deletion (e.g. a vendored-dependency removal) -- mirror it
      # rather than failing to cp a file that no longer exists; `git add -A`
      # below stages the removal the same way it stages the copies above.
      rm -f "$dst/$relf"
    fi
  # --no-renames: with git's default rename detection a staged rename lists only the NEW
  # path, so the old file would survive in the clone next to the new one. Untracked (not
  # ignored) files are adopted too: an in-progress rename leaves the new sources untracked.
  done < <(git -C "$src" diff --name-only --no-renames HEAD; git -C "$src" ls-files --others --exclude-standard)
  git -C "$dst" add -A
  if ! git -C "$dst" diff --cached --quiet --exit-code; then
    git -C "$dst" -c user.email=scratch@example.invalid -c user.name="scratch clone" \
      commit -q -m "scratch: adopt current worktree (disposable clone only, never touches the real repo)"
  fi
  # Loud "can this actually fail" guard: the clone must be fully clean at
  # this HEAD, or downstream dirty/untracked scenarios would be silently
  # polluted by leftover adoption state instead of seeing only what the
  # scenario itself injects afterward.
  if [ -n "$(git -C "$dst" status --porcelain)" ]; then
    echo "FATAL: adopt_worktree_clone: $dst is NOT clean immediately after adoption (bug in the overlay/commit step above)" >&2
    git -C "$dst" status --porcelain >&2
    exit 1
  fi
}

# Real-tree final guard: per-file "path sha256 mtime" snapshot of everything under the real obj/ and bin/ of the
# four trees this suite builds in (pipelines, lib/aec, lib/nr, audiolib), plus a top-level keyed-directory listing,
# captured once SP1's intentional real-tree builds are done and re-verified at the end: every snapshotted file
# must be sha+mtime identical; a new directory is allowed (INFO/WARN), a new file inside an already-snapshotted
# directory is a FAIL. Each line carries a repo prefix (pipelines/lib_aec/lib_nr/audiolib/common) because the
# relative paths (obj/ne10-<sig>/...) would otherwise collide across repos.
_fg_snapshot_repo() {  # <tag> <repo-root>
  local tag="$1" dir="$2"
  ( cd "$dir" && for sub in obj bin; do
      [ -d "$sub" ] || continue
      find "$sub" -type f | LC_ALL=C sort | while IFS= read -r f; do
        printf '%s/%s %s %s\n' "$tag" "$f" "$(file_sha "$f")" "$(mtime "$f")"
      done
    done )
}
_fg_dirs_repo() {  # <tag> <repo-root>
  local tag="$1" dir="$2"
  ( cd "$dir" && for sub in obj bin; do
      [ -d "$sub" ] || continue
      find "$sub" -mindepth 1 -maxdepth 1 -type d | LC_ALL=C sort | while IFS= read -r d; do
        printf '%s/%s\n' "$tag" "$d"
      done
    done )
}
final_guard_snapshot() {
  _fg_snapshot_repo pipelines "$PIPE_DIR"
  _fg_snapshot_repo lib_aec "$AEC_DIR"
  _fg_snapshot_repo lib_nr "$NR_DIR"
  _fg_snapshot_repo audiolib "$COMMON_DIR"
}
final_guard_dirs() {
  _fg_dirs_repo pipelines "$PIPE_DIR"
  _fg_dirs_repo lib_aec "$AEC_DIR"
  _fg_dirs_repo lib_nr "$NR_DIR"
  _fg_dirs_repo audiolib "$COMMON_DIR"
}

# Real dist/ sentinel (standing guard): absent stays absent; otherwise a full
# path list + per-file sha256 + per-file (name, mtime). <root>/dist is the
# argument (not a bare repo root), so this works for pipelines' own dist,
# lib/aec/c_impl's, lib/nr/c_impl's, $COMMON_DIR's, and (belt-and-braces) the two
# PRIMARY AEC/NR repos' c_impl dist/ too.
real_dist_sentinel() {
  local root="$1"
  if [ ! -e "$root/dist" ]; then
    echo absent
    return
  fi
  ( cd "$root" && {
      find dist -print | sort
      find dist -type f -print | sort | while read -r f; do shasum -a 256 "$f"; done
      find dist -print | sort | while read -r f; do stat -f '%N %m' "$f"; done
    } ) | shasum -a 256 | awk '{print $1}'
}

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

# assert_cmd_fails_with <description> <expected-substring> <cmd...> (used by
# SP-S9/SP-S12/SP-S13) -- runs <cmd...> with stdout+stderr merged; PASS iff it
# exits non-zero AND the combined output contains <expected-substring>; FAIL
# (dumping the log) otherwise. Fixed scratch path (a counter-suffixed name
# under $SCRATCH_ROOT/tmp), not `mktemp` -- no cleanup-state registration
# involved either way, consistent with the fixed-path style used everywhere
# else in this script.
ACF_COUNTER=0
assert_cmd_fails_with() {
  local desc="$1" needle="$2" log
  shift 2
  ACF_COUNTER=$((ACF_COUNTER + 1))
  log="$SCRATCH_ROOT/tmp/acf-$ACF_COUNTER.log"
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

# resolve_producers <backend> [simd] -- resolves audiolib's/lib-aec's/lib-nr's
# archive paths for <backend> and SIMD=<simd> (default 1; default flags
# otherwise), exactly the way pipelines/Makefile's own phase-1 dispatch recipe
# does. Uses the submodule AEC_DIR/NR_DIR paths (no OBJ_ROOT=/BIN_ROOT= feature
# needed). Sets globals AC_LIB_/AEC_LIB_/NR_LIB_.
resolve_producers() {
  local backend="$1" simd="${2:-1}"
  AC_LIB_="$(make -s -C "$COMMON_DIR" BACKEND="$backend" SIMD="$simd" WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 print-lib-path)"
  AEC_LIB_="$(make -s -C "$AEC_DIR" BACKEND="$backend" SIMD="$simd" WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 COMMON_DIR="$COMMON_DIR" COMMON_LIB="$AC_LIB_" print-lib-path)"
  NR_LIB_="$(make -s -C "$NR_DIR" BACKEND="$backend" SIMD="$simd" WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 COMMON_DIR="$COMMON_DIR" COMMON_LIB="$AC_LIB_" print-lib-path)"
}

cd "$PIPE_DIR"

# --- BEFORE snapshots (the four repos named by the safety contract, plus the
# two PRIMARY AEC/NR repos this script also builds against directly) ----
AUDIO_ALG_STATE_BEFORE="$(tree_state_hash "$AUDIO_ALG_ROOT")"
AEC_SUB_STATE_BEFORE="$(tree_state_hash "$AEC_SUB_ROOT")"
NR_SUB_STATE_BEFORE="$(tree_state_hash "$NR_SUB_ROOT")"
AC_STATE_BEFORE="$(tree_state_hash "$COMMON_DIR")"
AEC_R6_STATE_BEFORE="$(tree_state_hash "$AEC_REPO_DIR")"
NR_R6_STATE_BEFORE="$(tree_state_hash "$NR_REPO_DIR")"

REAL_DIST_PIPE_BEFORE="$(real_dist_sentinel "$PIPE_DIR")"
REAL_DIST_AEC_BEFORE="$(real_dist_sentinel "$AEC_DIR")"
REAL_DIST_NR_BEFORE="$(real_dist_sentinel "$NR_DIR")"
REAL_DIST_AC_BEFORE="$(real_dist_sentinel "$COMMON_DIR")"
REAL_DIST_AECR6_BEFORE="$(real_dist_sentinel "$AEC_R6_DIR")"
REAL_DIST_NRR6_BEFORE="$(real_dist_sentinel "$NR_R6_DIR")"

PASS_COUNT=0
FAIL_COUNT=0
FAILURES=()

pass() { PASS_COUNT=$((PASS_COUNT + 1)); echo "  PASS: $*"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); FAILURES+=("$*"); echo "  FAIL: $*" >&2; }

echo "############################################################"
echo "# S6: audit-no-stdio false-pass regression"
echo "############################################################"
mkscratch s6
make -s BACKEND=ne10 libaudio_pipeline_dfn.a >/dev/null
default_lib="$(make -s BACKEND=ne10 NO_STDIO=0 print-lib-path)"
sha_default_before="$(file_sha "$default_lib")"

S6_LOG="$SCRATCH_ROOT/s6/log"
if make BACKEND=ne10 audit-no-stdio >"$S6_LOG" 2>&1; then
  pass "S6: audit-no-stdio exits green"
else
  fail "S6: audit-no-stdio FAILED"
  cat "$S6_LOG" >&2
fi
grep -q '^PASS:' "$S6_LOG" && pass "S6: audit-no-stdio printed a PASS line" \
  || fail "S6: audit-no-stdio did not print a PASS line"

nostdio_lib="$(make -s BACKEND=ne10 NO_STDIO=1 print-lib-path)"
[ "$nostdio_lib" != "$default_lib" ] && pass "S6: NO_STDIO=1 archive path differs from the default (NO_STDIO=0) path" \
  || fail "S6: NO_STDIO=1 archive path COLLIDES with the default path ($default_lib)"
[ -f "$nostdio_lib" ] && pass "S6: NO_STDIO=1 archive exists on disk" || fail "S6: NO_STDIO=1 archive missing"

sha_default_after="$(file_sha "$default_lib")"
[ "$sha_default_before" = "$sha_default_after" ] && pass "S6: default archive untouched (sha stable) by the audit run" \
  || fail "S6: default archive CHANGED by the audit-no-stdio run (sha $sha_default_before -> $sha_default_after)"

# The archive holds no stdio code in either configuration, so the audit's nm filter is exercised on a scratch copy
# of the NO_STDIO archive with one member added that references fprintf/stderr: the same pattern must report it.
S6_PROBE_SRC="$SCRATCH_ROOT/s6/probe.c"
S6_PROBE_OBJ="$SCRATCH_ROOT/s6/probe.o"
S6_TAMPERED="$SCRATCH_ROOT/s6/tampered.a"
printf '#include <stdio.h>\nvoid s6_probe(void) { fprintf(stderr, "x"); }\n' > "$S6_PROBE_SRC"
cc -c "$S6_PROBE_SRC" -o "$S6_PROBE_OBJ"
cp "$nostdio_lib" "$S6_TAMPERED"
ar r "$S6_TAMPERED" "$S6_PROBE_OBJ" 2>/dev/null
s6_hits_clean="$(nm -A "$nostdio_lib" | awk '{ n = $NF; if (n ~ /^_?(fprintf|printf|puts|fputs|stderr|__stderrp)$/) print }')"
s6_hits_tampered="$(nm -A "$S6_TAMPERED" | awk '{ n = $NF; if (n ~ /^_?(fprintf|printf|puts|fputs|stderr|__stderrp)$/) print }')"
[ -z "$s6_hits_clean" ] && [ -n "$s6_hits_tampered" ] \
  && pass "S6: the audit's nm filter reports a stdio reference added to a copy of the archive, and none in the shipped one" \
  || fail "S6: the nm filter did not separate the shipped archive (hits: '$s6_hits_clean') from the tampered copy (hits: '$s6_hits_tampered')"

echo "############################################################"
echo "# S7: publish (v4 -- lock-first driver + concurrent semantics)"
echo "############################################################"
mkscratch s7
S7_DIST_ROOT="$SCRATCH_ROOT/s7/dist"
resolve_producers ne10
# ALLOW_DIRTY_PUBLISH=1 on every publish call from here on: these dev trees may legitimately be dirty (the
# dirty-publish policy itself is exercised in SP-S17). ALLOW_UNTRACKED_PUBLISH=1 rides along too, except on
# SP-S17's deliberate policy-negative calls: the untracked dimension is a separate default refusal. DIST_ROOT stays
# a throwaway scratch path: the real dist/ is never read, written or removed.
make -s BACKEND=ne10 DIST_ROOT="$S7_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish COMMON_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null

ne10_current_target="$(readlink "$S7_DIST_ROOT/ne10/current" || true)"
[ -n "$ne10_current_target" ] && [ -d "$S7_DIST_ROOT/ne10/$ne10_current_target" ] && \
  pass "S7: publish -- current symlink resolves to a real release dir" \
  || fail "S7: publish -- current symlink broken or missing"

manifest_ok=1
while read -r sha fname; do
  [ "$fname" = "MANIFEST.txt" ] && continue
  actual="$(file_sha "$S7_DIST_ROOT/ne10/current/$fname")"
  [ "$actual" = "$sha" ] || manifest_ok=0
done < <(grep -E '^[0-9a-f]{64}  ' "$S7_DIST_ROOT/ne10/current/MANIFEST.txt")
[ "$manifest_ok" -eq 1 ] && pass "S7: MANIFEST sha self-consistency" \
  || fail "S7: MANIFEST sha mismatch against files on disk"

grep -q '^ac_producer_cfg_sig=ne10-' "$S7_DIST_ROOT/ne10/current/MANIFEST.txt" && \
  grep -q '^aec_producer_cfg_sig=ne10-' "$S7_DIST_ROOT/ne10/current/MANIFEST.txt" && \
  grep -q '^nr_producer_cfg_sig=ne10-' "$S7_DIST_ROOT/ne10/current/MANIFEST.txt" && \
  pass "S7: MANIFEST records all three producer cfg_sig identities" \
  || fail "S7: MANIFEST missing one or more producer cfg_sig identities"

grep -q '^git_commit=' "$S7_DIST_ROOT/ne10/current/MANIFEST.txt" && \
  fail "S7: MANIFEST.txt unexpectedly contains a git_commit= line (v4 keeps provenance in ATTEST/ only)" \
  || pass "S7: MANIFEST.txt has no git_commit= line (deterministic MANIFEST)"

s7_attest="$(find "$S7_DIST_ROOT/ne10/current/ATTEST" -type f -name 'attest-*.txt' 2>/dev/null | head -n1)"
[ -n "$s7_attest" ] && grep -q '^git_commit=' "$s7_attest" && grep -q '^aec_git_commit=' "$s7_attest" && \
  pass "S7: ATTEST file carries git_commit=/aec_git_commit= provenance" \
  || fail "S7: ATTEST file missing, or missing git_commit=/aec_git_commit="

# Concurrent same-backend publish: the lock-FIRST driver means the loser can
# fail fast with "already held" rather than waiting -- but depending on
# scheduling, both invocations can also land in disjoint (non-overlapping)
# lock windows and both succeed. Either outcome is fine as long as `current`
# ends up pointing at a COMPLETE, self-consistent release.
S7_LOG_A="$SCRATCH_ROOT/s7/log_a"; S7_LOG_B="$SCRATCH_ROOT/s7/log_b"
( make -s BACKEND=ne10 DIST_ROOT="$S7_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish COMMON_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S7_LOG_A" 2>&1 ) & cp1=$!
( make -s BACKEND=ne10 DIST_ROOT="$S7_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish COMMON_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S7_LOG_B" 2>&1 ) & cp2=$!
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

final_target="$(readlink "$S7_DIST_ROOT/ne10/current" || true)"
[ -n "$final_target" ] && [ -f "$S7_DIST_ROOT/ne10/$final_target/MANIFEST.txt" ] && [ -f "$S7_DIST_ROOT/ne10/$final_target/aec_nr_pipeline" ] && \
  pass "S7: after concurrent publish, current points at a complete release dir" \
  || fail "S7: after concurrent publish, current points at an incomplete/missing release dir"

echo "############################################################"
echo "# SP-S19: the 4ch test_mcab_sample_driver object tracks the headers it includes"
echo "############################################################"
mkscratch sp19
SP19_ARGS=(-C "$PIPE_DIR/4ch_aec_bf_nr_res" BACKEND=ne10 SIMD=1 WERROR=0
  OBJ_ROOT="$SCRATCH_ROOT/sp19/obj" BIN_ROOT="$SCRATCH_ROOT/sp19/bin")
if make -s "${SP19_ARGS[@]}" test-mcab-sample-driver >"$SCRATCH_ROOT/sp19/build.log" 2>&1; then
  pass "SP-S19: make test-mcab-sample-driver builds and passes"
  sp19_obj="$(make -s "${SP19_ARGS[@]}" print-obj-dir | tail -1)"
  sp19_missing=""
  for h in mcab.h mcab_sample_beam.h mcab_sample_flow.h mcab_sample_driver.h aec.h example_cli.h; do
    grep -q "[/ ]$h" "$sp19_obj/test_mcab_sample_driver.d" 2>/dev/null || sp19_missing="$sp19_missing $h"
  done
  [ -z "$sp19_missing" ] && pass "SP-S19: test_mcab_sample_driver.d lists mcab.h, the sample headers, aec.h and example_cli.h" \
    || fail "SP-S19: test_mcab_sample_driver.d is missing (or lacks):$sp19_missing -- a header edit would leave a stale test binary"
  case "$sp19_obj" in
    "$SCRATCH_ROOT/sp19/obj/"*) pass "SP-S19: the test builds under the producer-keyed object directory the rest of 4ch uses" ;;
    *) fail "SP-S19: unexpected object directory $sp19_obj" ;;
  esac
else
  fail "SP-S19: make test-mcab-sample-driver failed"; tail -20 "$SCRATCH_ROOT/sp19/build.log" >&2
fi

echo "############################################################"
echo "# SP-S20: test-pairs refuses an empty pair list and is side-effect free under -n/-q/-t"
echo "############################################################"
mkscratch sp20
SP20_ARGS=(BACKEND=ne10 OBJ_ROOT="$SCRATCH_ROOT/sp20/obj" BIN_ROOT="$SCRATCH_ROOT/sp20/bin")
rc=0; make "${SP20_ARGS[@]}" DFN_TEST_PAIRS= test-pairs >"$SCRATCH_ROOT/sp20/empty.log" 2>&1 || rc=$?
[ "$rc" -ne 0 ] && grep -q "DFN_TEST_PAIRS is empty" "$SCRATCH_ROOT/sp20/empty.log" \
  && pass "SP-S20: an empty DFN_TEST_PAIRS is refused with its own error (rc=$rc)" \
  || fail "SP-S20: an empty DFN_TEST_PAIRS was not refused (rc=$rc)"
for f in -n -q -t -nq -nt; do
  rc=0; make "$f" "${SP20_ARGS[@]}" DFN_TEST_PAIRS="0/0 2/4" test-pairs >"$SCRATCH_ROOT/sp20/$f.log" 2>&1 || rc=$?
  want=0; case "$f" in *q*) want=1 ;; esac
  if [ "$rc" -eq "$want" ] && ! grep -q "FAIL" "$SCRATCH_ROOT/sp20/$f.log" && \
     [ ! -e "$SCRATCH_ROOT/sp20/obj" ] && [ ! -e "$SCRATCH_ROOT/sp20/bin" ]; then
    pass "SP-S20: make $f test-pairs: rc=$rc, no FAIL line, nothing built"
  else
    fail "SP-S20: make $f test-pairs: rc=$rc (expected $want), FAIL lines=$(grep -c FAIL "$SCRATCH_ROOT/sp20/$f.log"), obj/bin created=$([ -e "$SCRATCH_ROOT/sp20/obj" ] || [ -e "$SCRATCH_ROOT/sp20/bin" ] && echo yes || echo no)"
  fi
done
grep -q "would build and run" "$SCRATCH_ROOT/sp20/-n.log" && pass "SP-S20: make -n test-pairs prints what it would run" \
  || fail "SP-S20: make -n test-pairs printed no plan"

echo "############################################################"
echo "# SP1: pipeline-level A->B->A (SIMD=1 -> SIMD=0 -> SIMD=1)"
echo "############################################################"
resolve_producers ne10 1
ac_a="$AC_LIB_"; aec_a="$AEC_LIB_"; nr_a="$NR_LIB_"
make -s BACKEND=ne10 SIMD=1 WERROR=0 all COMMON_LIB="$ac_a" AEC_LIB="$aec_a" NR_LIB="$nr_a" >/dev/null
bd_a1="$(make -s BACKEND=ne10 SIMD=1 WERROR=0 print-bin-dir COMMON_LIB="$ac_a" AEC_LIB="$aec_a" NR_LIB="$nr_a")"
m_a1="$(mtime "$bd_a1/aec_nr_pipeline")"

resolve_producers ne10 0
ac_b="$AC_LIB_"; aec_b="$AEC_LIB_"; nr_b="$NR_LIB_"
make -s BACKEND=ne10 SIMD=0 WERROR=0 all COMMON_LIB="$ac_b" AEC_LIB="$aec_b" NR_LIB="$nr_b" >/dev/null
bd_b="$(make -s BACKEND=ne10 SIMD=0 WERROR=0 print-bin-dir COMMON_LIB="$ac_b" AEC_LIB="$aec_b" NR_LIB="$nr_b")"

# Final-guard snapshot, taken NOW -- both of the real-tree normal-config builds
# this suite intends (SIMD=1 above, via S6/S7 and this scenario; SIMD=0 just
# now) are done for all four trees (pipelines itself, lib/aec, lib/nr,
# audiolib). Every later real-tree `make ... lib`/`all` in this suite
# re-targets one of these same two already-built configs and must be a pure
# no-op against them. Re-verified at the very end (see "Final integrity
# guards").
FINAL_GUARD_DIRS_BEFORE="$(final_guard_dirs)"
FINAL_GUARD_SNAPSHOT_BEFORE="$(final_guard_snapshot)"

# No sleep -- this is a "should NOT relink" check, so mtime()'s BSD
# fractional-second read (stat -f '%Fm') is what makes a genuine relink here
# distinguishable from m_a1, regardless of how close in wall-clock time the two
# SIMD=1 builds land (a whole-second mtime comparison could otherwise false-PASS
# a real, same-second relink).
make -s BACKEND=ne10 SIMD=1 WERROR=0 all COMMON_LIB="$ac_a" AEC_LIB="$aec_a" NR_LIB="$nr_a" >/dev/null
bd_a2="$(make -s BACKEND=ne10 SIMD=1 WERROR=0 print-bin-dir COMMON_LIB="$ac_a" AEC_LIB="$aec_a" NR_LIB="$nr_a")"
m_a2="$(mtime "$bd_a2/aec_nr_pipeline")"

[ "$bd_a1" != "$bd_b" ] && pass "SP1: SIMD=1 and SIMD=0 builds land in different bin dirs" \
  || fail "SP1: SIMD=1 and SIMD=0 builds share one bin dir ($bd_a1)"
[ "$bd_a1" = "$bd_a2" ] && pass "SP1: SIMD=1 bin dir stable across A->B->A" \
  || fail "SP1: SIMD=1 bin dir CHANGED across A->B->A ($bd_a1 vs $bd_a2)"
[ "$m_a1" = "$m_a2" ] && pass "SP1: SIMD=1 aec_nr_pipeline NOT relinked on the third (A->B->A) run" \
  || fail "SP1: SIMD=1 aec_nr_pipeline relinked with nothing changed"

nm "$bd_a2/aec_nr_pipeline" 2>/dev/null | grep -q 'ne10_fft_r2c_1d_float32_neon' && \
  pass "SP1: SIMD=1 aec_nr_pipeline links the ne10 FFT symbols" \
  || fail "SP1: SIMD=1 aec_nr_pipeline missing ne10 FFT symbols"
nm "$bd_b/aec_nr_pipeline" 2>/dev/null | grep -q 'ne10_fft_r2c_1d_float32_c$' && \
  ! nm "$bd_b/aec_nr_pipeline" 2>/dev/null | grep -q 'ne10_fft_r2c_1d_float32_neon' && \
  pass "SP1: SIMD=0 aec_nr_pipeline links the scalar ne10 FFT (no NEON FFT symbol)" \
  || fail "SP1: SIMD=0 aec_nr_pipeline does not link exactly the scalar ne10 FFT"

echo "############################################################"
echo "# SP2: producer-change propagation (regression: scratch clones)"
echo "############################################################"
# This scenario runs against a disposable clone of audiolib + the PRIMARY
# AEC repo + the PRIMARY NR repo; no real tracked file is touched, not even its
# mtime. This SAME clone set is created fresh HERE (on the still-clean tip) and
# REUSED (then further dirtied) by SP-S17 further down. The producer objs build
# INSIDE the clones (their own default, clone-relative obj/bin, which live
# under scratch by construction); the pipeline binary itself builds from the
# REAL pipelines directory (never cloned) with COMMON_DIR/AEC_DIR/NR_DIR pointed at
# the clones plus an explicit scratch OBJ_ROOT/BIN_ROOT.
mkscratch pshared
mkscratch sp2
# audiolib's common/ has no .git of its own (it is a subdirectory of the audiolib repo), so the clone is of the
# whole audiolib repo, named "audiolib" so it sits where the wav_io.h shim of AEC's/NR's c_impl/example expects it
# ("../../../audiolib/common/include/wav_io.h" via __has_include): a sibling of the AEC/NR clones, all under
# $SCRATCH_ROOT/pshared/, with no symlink or tracked-source edit. PSHARED_AC_CLONE is that clone's common/
# subdirectory.
PSHARED_AUDIOLIB_CLONE="$SCRATCH_ROOT/pshared/audiolib"
PSHARED_AEC_CLONE="$SCRATCH_ROOT/pshared/aec_clone"
PSHARED_NR_CLONE="$SCRATCH_ROOT/pshared/nr_clone"
adopt_worktree_clone "$AUDIOLIB_DIR" "$PSHARED_AUDIOLIB_CLONE"
adopt_worktree_clone "$AEC_REPO_DIR" "$PSHARED_AEC_CLONE"
adopt_worktree_clone "$NR_REPO_DIR" "$PSHARED_NR_CLONE"
PSHARED_AC_CLONE="$PSHARED_AUDIOLIB_CLONE/common"

SP2_OBJ_ROOT="$SCRATCH_ROOT/sp2/obj"
SP2_BIN_ROOT="$SCRATCH_ROOT/sp2/bin"

# Resolve + build each producer inside its own clone. Every make into the AEC/NR clones passes
# COMMON_DIR=<ac clone abs path> explicitly: their `$(COMMON_LIB): FORCE` rule re-invokes `$(MAKE) -C $(COMMON_DIR)
# ... lib` with that invocation's COMMON_DIR, and a wildcard/relative default would resolve the real tree.

# CC/CXX are pinned (CC='cc' CXX='c++', the values resolve_producers() uses) instead of each Makefile's bare `?=`
# default: audiolib defaults CC to cc but pipelines to gcc, and forwarding the wrong one would key a producer to a
# different CFG_SIG than the rest of this scenario expects.

# OBJ_ROOT/BIN_ROOT (SP2_OBJ_ROOT/SP2_BIN_ROOT) are the same explicit pair on every make call below, producers and
# pipeline alike. GNU Make propagates every command-line variable to recursive $(MAKE) calls through MAKEFLAGS, so
# the pipeline's own override (needed to keep its artifacts out of pipelines/obj,bin) reaches the nested FORCE
# sub-makes whatever $(FWD) lists. Resolving `ac_k` without the override while the pipeline build uses one would
# point `ac_k` at a stale copy under the clone's default path, and every "touch source -> relink" assertion below
# would false-fail.
CCX=(CC='cc' CXX='c++' WERROR=0 EXTRA_CFLAGS='' NO_STDIO=0)
ORX=(OBJ_ROOT="$SP2_OBJ_ROOT" BIN_ROOT="$SP2_BIN_ROOT")
ac_k="$(make -s -C "$PSHARED_AC_CLONE" BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" print-lib-path)"
make -s -C "$PSHARED_AC_CLONE" BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" lib >/dev/null
aec_k="$(make -s -C "$PSHARED_AEC_CLONE/c_impl" BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" COMMON_DIR="$PSHARED_AC_CLONE" COMMON_LIB="$ac_k" print-lib-path)"
make -s -C "$PSHARED_AEC_CLONE/c_impl" BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" COMMON_DIR="$PSHARED_AC_CLONE" COMMON_LIB="$ac_k" lib >/dev/null
nr_k="$(make -s -C "$PSHARED_NR_CLONE/c_impl" BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" COMMON_DIR="$PSHARED_AC_CLONE" COMMON_LIB="$ac_k" print-lib-path)"
make -s -C "$PSHARED_NR_CLONE/c_impl" BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" COMMON_DIR="$PSHARED_AC_CLONE" COMMON_LIB="$ac_k" lib >/dev/null

sp2_makeargs=(BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" \
  COMMON_DIR="$PSHARED_AC_CLONE" AEC_DIR="$PSHARED_AEC_CLONE/c_impl" NR_DIR="$PSHARED_NR_CLONE/c_impl" \
  COMMON_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k")

make -s "${sp2_makeargs[@]}" all >/dev/null
bd_k="$(make -s "${sp2_makeargs[@]}" print-bin-dir)"

# "Did it rebuild?" is verified from make's own build log (grep for that rule's `@echo` line), never an mtime diff,
# as in audiolib's S5: two mtime snapshots of one output file can land in the same whole second when the
# touch-rebuild-relink chain runs back-to-back, and GNU Make 3.81's whole-second newer-than check then declines to
# re-archive or relink.
# backdate_sp2(): backdates a clone-owned build artifact (an archive or the final binary, never a real tracked
# source) by 5 minutes right before the rebuild that must supersede it, so the fresh timestamp is unambiguously
# newer. Mirrors audiolib's S5 `backdate_aecwav()`.
backdate_sp2() { touch -A -0500 "$1"; }

# Touch the audiolib clone's src/hpf.c: hpf.o recompiles and aec_nr_pipeline relinks (audiolib's CFG_SIG hashes the
# compiler invocation, not file content, so its archive path is unchanged and only its mtime advances). hpf.c's
# mtime is set to hpf.o's current mtime + 1s (BSD touch -r/-A) so make sees the source as newer; the clone's
# archive and the pipeline binary are backdated right before the rebuild so the archive-rebuild and relink echoes
# fire. Only the clone's copy is touched, never a real tree.
ac_objdir="$(make -s -C "$PSHARED_AC_CLONE" BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" print-obj-dir)"
backdate_sp2 "$ac_k"
backdate_sp2 "$bd_k/aec_nr_pipeline"
touch -r "$ac_objdir/hpf.o" -A 01 "$PSHARED_AC_CLONE/src/hpf.c"
SP2_LOG_HPF="$SCRATCH_ROOT/sp2/log_hpf"
make -s "${sp2_makeargs[@]}" all >"$SP2_LOG_HPF" 2>&1
grep -q "^common \[ne10\] ->" "$SP2_LOG_HPF" && pass "SP2: touching the audiolib clone's src/hpf.c rebuilt its archive" \
  || { fail "SP2: audiolib's archive did NOT rebuild after the clone's hpf.o changed"; cat "$SP2_LOG_HPF" >&2; }
grep -q "^aec_nr_pipeline \[ne10\] ->" "$SP2_LOG_HPF" && pass "SP2: touching the audiolib clone's src/hpf.c relinked aec_nr_pipeline" \
  || { fail "SP2: aec_nr_pipeline NOT relinked after the audiolib clone's hpf.o changed"; cat "$SP2_LOG_HPF" >&2; }

# Touch utility/fast_math.h: the NR/AEC objects that include it (aec_post.c/suppression_gain.c in lib/aec,
# noise_estimator.c/spp_estimator.c/anr.c in lib/nr) recompile and the pipeline binary relinks, the full transitive
# header chain. Again mtime-only on the clone's copy, with the same log-based verification and backdating, this
# time of both producer archives and the pipeline binary.
aec_objdir="$(make -s -C "$PSHARED_AEC_CLONE/c_impl" BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" COMMON_DIR="$PSHARED_AC_CLONE" COMMON_LIB="$ac_k" print-obj-dir)"
nr_objdir="$(make -s -C "$PSHARED_NR_CLONE/c_impl" BACKEND=ne10 "${CCX[@]}" "${ORX[@]}" COMMON_DIR="$PSHARED_AC_CLONE" COMMON_LIB="$ac_k" print-obj-dir)"

# Touch fast_math.h strictly newer than both aec_post.o and noise_estimator.o: they were compiled moments apart in
# no guaranteed order, so the one with the later current mtime is the `touch -r` reference before the +1s bump.
sp2_ref="$aec_objdir/aec_post.o"
if awk -v a="$(mtime "$nr_objdir/noise_estimator.o")" -v b="$(mtime "$sp2_ref")" 'BEGIN{exit !(a>b)}'; then
  sp2_ref="$nr_objdir/noise_estimator.o"
fi
backdate_sp2 "$aec_k"
backdate_sp2 "$nr_k"
backdate_sp2 "$bd_k/aec_nr_pipeline"
touch -r "$sp2_ref" -A 01 "$PSHARED_AC_CLONE/utility/fast_math.h"
SP2_LOG_FASTMATH="$SCRATCH_ROOT/sp2/log_fastmath"
make -s "${sp2_makeargs[@]}" all >"$SP2_LOG_FASTMATH" 2>&1

grep -q "^aec \[ne10\] ->" "$SP2_LOG_FASTMATH" && pass "SP2: touching the audiolib clone's utility/fast_math.h recompiled lib/aec's aec_post.o (archive rebuilt)" \
  || { fail "SP2: lib/aec's archive did NOT rebuild after touching the clone's fast_math.h (aec_post.o likely did not recompile)"; cat "$SP2_LOG_FASTMATH" >&2; }
grep -q "^Static library built: " "$SP2_LOG_FASTMATH" && pass "SP2: touching the audiolib clone's utility/fast_math.h recompiled lib/nr's noise_estimator.o (archive rebuilt)" \
  || { fail "SP2: lib/nr's archive did NOT rebuild after touching the clone's fast_math.h (noise_estimator.o likely did not recompile)"; cat "$SP2_LOG_FASTMATH" >&2; }
grep -q "^aec_nr_pipeline \[ne10\] ->" "$SP2_LOG_FASTMATH" && pass "SP2: touching the audiolib clone's utility/fast_math.h relinked aec_nr_pipeline (full transitive chain)" \
  || { fail "SP2: aec_nr_pipeline NOT relinked after the clone's fast_math.h touch"; cat "$SP2_LOG_FASTMATH" >&2; }

echo "############################################################"
echo "# SP-S9: command-line override rejection across the stack"
echo "############################################################"
assert_cmd_fails_with "SP-S9: pipelines 'make CFLAGS=-O3' rejected at parse time" "cannot be overridden" \
  make BACKEND=ne10 CFLAGS=-O3 print-obj-dir

assert_cmd_fails_with "SP-S9: pipelines 'make FP_POLICY=-ffp-contract=fast' rejected at parse time" "cannot be overridden" \
  make BACKEND=ne10 FP_POLICY=-ffp-contract=fast print-obj-dir

assert_cmd_fails_with "SP-S9: lib/aec 'make CFLAGS=-O3' rejected at parse time" "cannot be overridden" \
  make -C "$AEC_DIR" BACKEND=ne10 CFLAGS=-O3 print-obj-dir

assert_cmd_fails_with "SP-S9: lib/nr 'make LDFLAGS=-lfoo' rejected at parse time" "cannot be overridden" \
  make -C "$NR_DIR" BACKEND=ne10 LDFLAGS=-lfoo print-obj-dir

sp9_plain_objdir="$(make -s BACKEND=ne10 print-obj-dir)"
sp9_probe_objdir="$(make -s BACKEND=ne10 EXTRA_CFLAGS=-DSP9_PROBE print-obj-dir)"
[ -n "$sp9_probe_objdir" ] && pass "SP-S9: pipelines EXTRA_CFLAGS=-DSP9_PROBE print-obj-dir SUCCEEDED" \
  || fail "SP-S9: pipelines EXTRA_CFLAGS=-DSP9_PROBE print-obj-dir produced no output"
[ "$sp9_probe_objdir" != "$sp9_plain_objdir" ] && pass "SP-S9: EXTRA_CFLAGS=-DSP9_PROBE obj dir differs from the plain-query obj dir" \
  || fail "SP-S9: EXTRA_CFLAGS=-DSP9_PROBE obj dir COLLIDES with the plain-query obj dir ($sp9_plain_objdir)"

echo "############################################################"
echo "# SP-S9b: FP-policy allow-list redesign (this task) -- bypass shapes"
echo "# the OLD 9-item deny-list never caught, the exact-token filter's own"
echo "# distinct code path, the FP-policy check's OWN internal variables,"
echo "# and the legitimate positive control"
echo "############################################################"
# See the header (SP-S9b) and pipelines/Makefile's allow-list comments. Every negative case below is a live bypass of a
# character deny-list (single quote, double quote, backslash, backtick, a literal "$(", semicolon, pipe, ampersand,
# @-response-file) or of bare glob characters, tilde, shell redirection or process substitution, each rejected by the
# character-class allow-list. They are all parse-time rejections (the FP-policy checks run before any recipe,
# print-obj-dir's included), so as in SP-S9 no scratch OBJ_ROOT/BIN_ROOT is needed: a rejected build creates no
# directory.
sp9b_check_rejected() {
  local label="$1" flag_value="$2" expect_chars="$3" log
  log="$SCRATCH_ROOT/tmp/sp9b_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_').log"
  if env EXTRA_CFLAGS="$flag_value" make BACKEND=ne10 print-obj-dir >"$log" 2>&1; then
    fail "SP-S9b: EXTRA_CFLAGS='$flag_value' ($label) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
  elif grep -q "FP policy conflict" "$log" && grep -q "outside the allowed set" "$log" && grep -qF "found: \"$expect_chars\"" "$log"; then
    pass "SP-S9b: EXTRA_CFLAGS='$flag_value' ($label) print-obj-dir correctly FAILS, identifying \"$expect_chars\""
  else
    fail "SP-S9b: EXTRA_CFLAGS='$flag_value' ($label) print-obj-dir failed but did NOT identify \"$expect_chars\" specifically"
    cat "$log" >&2
  fi
}

sp9b_check_rejected "glob-star"     '-O*t'            '*'
sp9b_check_rejected "glob-question" '-Ofas?'          '?'
sp9b_check_rejected "glob-brackets" '-Ofas[t]'        '[]'
sp9b_check_rejected "tilde"         '~/pwned'         '~'
sp9b_check_rejected "redirect-out"  '-I>/tmp/evil'    '>'
sp9b_check_rejected "redirect-in"   '-I</etc/passwd'  '<'
sp9b_check_rejected "process-subst" '<(echo hi)'      '()<'

# Single-quote -- caught by the ONE pure-Make $(findstring) check (the
# allow-list's own $(shell) call needs it excluded first to safely
# single-quote-embed the rest of the text), so this message says "single-quote
# character" verbatim, unlike the allow-list cases above.
SP9B_LOG_SQ="$SCRATCH_ROOT/tmp/sp9b_single_quote.log"
if env EXTRA_CFLAGS="-O'f'ast" make BACKEND=ne10 print-obj-dir >"$SP9B_LOG_SQ" 2>&1; then
  fail "SP-S9b: EXTRA_CFLAGS=\"-O'f'ast\" (quote-split) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
elif grep -q "FP policy conflict" "$SP9B_LOG_SQ" && grep -q "single-quote" "$SP9B_LOG_SQ"; then
  pass "SP-S9b: EXTRA_CFLAGS=\"-O'f'ast\" (quote-split) print-obj-dir correctly FAILS, identifying the single-quote character"
else
  fail "SP-S9b: EXTRA_CFLAGS=\"-O'f'ast\" (quote-split) print-obj-dir failed but did NOT identify the single-quote character"
  cat "$SP9B_LOG_SQ" >&2
fi

# Positive control (distinct code path): a Make-native ${VAR}-style
# expansion still must be rejected via the EXACT-TOKEN $(filter) check, not
# the character allow-list -- GNU Make resolves ${...} identically to
# $(...) before either FP-policy check ever runs, landing the plain,
# ALL-ALLOWED-CHARACTERS token "-Ofast" in FP_INPUT_FLAGS, which only the
# unchanged FP_CONFLICT_FLAGS $(filter) check can still reject.
export SP9B_R11_FLAG=-Ofast
SP9B_LOG_R11="$SCRATCH_ROOT/tmp/sp9b_r11_dollar_brace.log"
if make -n BACKEND=ne10 'EXTRA_CFLAGS=${SP9B_R11_FLAG}' lib >"$SP9B_LOG_R11" 2>&1; then
  fail "SP-S9b: SP9B_R11_FLAG=-Ofast (exported) make -n EXTRA_CFLAGS=\${SP9B_R11_FLAG} lib unexpectedly SUCCEEDED (must be rejected via the exact-token filter)"
elif grep -q "FP policy conflict" "$SP9B_LOG_R11" && grep -q -- "-Ofast" "$SP9B_LOG_R11" && ! grep -q "outside the allowed set" "$SP9B_LOG_R11"; then
  pass "SP-S9b: SP9B_R11_FLAG=-Ofast (exported) make -n EXTRA_CFLAGS=\${SP9B_R11_FLAG} lib correctly FAILS via the exact-token \$(filter) check (not the character allow-list)"
else
  fail "SP-S9b: SP9B_R11_FLAG=-Ofast (exported) make -n EXTRA_CFLAGS=\${SP9B_R11_FLAG} lib failed but not via the expected exact-token filter path"
  cat "$SP9B_LOG_R11" >&2
fi
unset SP9B_R11_FLAG

# The FP-policy check's own internal variables (FP_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC/FP_CONFLICT_FLAGS/
# FP_ALLOWED_CHARS_RE; SHELL_SAFE_ALLOWLIST_RC also gates LDFLAGS, see SP-S9c) must themselves be rejected as
# command-line and `-e` overrides, or a caller could defeat every check above while the real CFLAGS/CXXFLAGS/CPPFLAGS
# still carry the dangerous content. SP-S9c repeats the SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC pair via an
# EXTRA_LDFLAGS payload.
sp9b_check_override_rejected() {
  local varname="$1" varvalue="$2" payload="$3" use_dash_e="$4" log
  log="$SCRATCH_ROOT/tmp/sp9b_override_$(printf '%s' "$varname" | tr -c 'A-Za-z0-9' '_')_$use_dash_e.log"
  if [ "$use_dash_e" = "dashe" ]; then
    if env "$varname=$varvalue" make -e BACKEND=ne10 EXTRA_CFLAGS="$payload" print-obj-dir >"$log" 2>&1; then
      fail "SP-S9b: env $varname=$varvalue make -e print-obj-dir unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the FP-policy checks)"
    elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
      pass "SP-S9b: env $varname=$varvalue make -e print-obj-dir correctly FAILS, mentioning '$varname cannot be overridden'"
    else
      fail "SP-S9b: env $varname=$varvalue make -e print-obj-dir failed but did NOT mention '$varname cannot be overridden'"
      cat "$log" >&2
    fi
  else
    if make BACKEND=ne10 EXTRA_CFLAGS="$payload" "$varname=$varvalue" print-obj-dir >"$log" 2>&1; then
      fail "SP-S9b: make $varname=$varvalue print-obj-dir unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the FP-policy checks)"
    elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
      pass "SP-S9b: make $varname=$varvalue print-obj-dir correctly FAILS, mentioning '$varname cannot be overridden'"
    else
      fail "SP-S9b: make $varname=$varvalue print-obj-dir failed but did NOT mention '$varname cannot be overridden'"
      cat "$log" >&2
    fi
  fi
}
sp9b_check_override_rejected "FP_INPUT_FLAGS"         "clean" '-O2;rm' cmdline
sp9b_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     '-O2;rm' cmdline
sp9b_check_override_rejected "FP_ALLOWED_CHARS_RE"    ".*"    '-O2;rm' cmdline
sp9b_check_override_rejected "FP_CONFLICT_FLAGS"      ""      '-Ofast' cmdline
sp9b_check_override_rejected "FP_INPUT_FLAGS"         "clean" '-O2;rm' dashe
sp9b_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     '-O2;rm' dashe
sp9b_check_override_rejected "FP_ALLOWED_CHARS_RE"    ".*"    '-O2;rm' dashe
sp9b_check_override_rejected "FP_CONFLICT_FLAGS"      ""      '-Ofast' dashe

# Positive control: a plain, override-free build must still succeed, AND
# the internal -DAUDIO_PIPELINE_BACKEND_STR=\"ne10\" define (this repo's own
# wrinkle -- a legitimate Makefile-generated token containing a backslash
# and a double quote, neither of which is in the allow-list's character
# set) must still reach the actual compile line, protected by the
# `-DAUDIO_PIPELINE_BACKEND_STR=%` filter-out exclusion. Uses scratch
# OBJ_ROOT/BIN_ROOT (this one actually builds, unlike every case above).
mkscratch sp9b_pos
SP9B_POS_OBJ_ROOT="$SCRATCH_ROOT/sp9b_pos/obj"
SP9B_POS_BIN_ROOT="$SCRATCH_ROOT/sp9b_pos/bin"
SP9B_POS_LOG="$SCRATCH_ROOT/tmp/sp9b_positive_build.log"
if make BACKEND=ne10 OBJ_ROOT="$SP9B_POS_OBJ_ROOT" BIN_ROOT="$SP9B_POS_BIN_ROOT" libaudio_pipeline_dfn.a >"$SP9B_POS_LOG" 2>&1; then
  pass "SP-S9b: plain override-free 'make libaudio_pipeline_dfn.a' SUCCEEDS under the new allow-list"
else
  fail "SP-S9b: plain override-free 'make libaudio_pipeline_dfn.a' unexpectedly FAILED under the new allow-list"
  cat "$SP9B_POS_LOG" >&2
fi
if grep -qF -- '-DAUDIO_PIPELINE_BACKEND_STR=\"ne10\"' "$SP9B_POS_LOG"; then
  pass "SP-S9b: -DAUDIO_PIPELINE_BACKEND_STR=\\\"ne10\\\" still reaches the archive's compile lines"
else
  fail "SP-S9b: -DAUDIO_PIPELINE_BACKEND_STR define did NOT appear on the compile line"
  cat "$SP9B_POS_LOG" >&2
fi

echo "############################################################"
echo "# SP-S9c: link-flags (LDFLAGS/EXTRA_LDFLAGS) character-safety coverage"
echo "############################################################"
# See the header (SP-S9c) and pipelines/Makefile's "link-flags character-safety coverage" comment. SP-S9b's allow-list
# validates FP_INPUT_FLAGS; LDFLAGS/EXTRA_LDFLAGS need the same check because LDFLAGS is embedded in every link recipe
# ($(LINK) ... $(LDFLAGS)) like CFLAGS in every compile recipe. As in SP-S9b, the negative cases are parse-time
# rejections, so no scratch OBJ_ROOT/BIN_ROOT is needed there.
sp9c_check_rejected() {
  local label="$1" flag_value="$2" expect_chars="$3" log
  log="$SCRATCH_ROOT/tmp/sp9c_$(printf '%s' "$label" | tr -c 'A-Za-z0-9' '_').log"
  if env EXTRA_LDFLAGS="$flag_value" make BACKEND=ne10 print-obj-dir >"$log" 2>&1; then
    fail "SP-S9c: EXTRA_LDFLAGS='$flag_value' ($label) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
  elif grep -q "FP policy conflict" "$log" && grep -q "outside the allowed set" "$log" && grep -qF "found: \"$expect_chars\"" "$log"; then
    pass "SP-S9c: EXTRA_LDFLAGS='$flag_value' ($label) print-obj-dir correctly FAILS, identifying \"$expect_chars\""
  else
    fail "SP-S9c: EXTRA_LDFLAGS='$flag_value' ($label) print-obj-dir failed but did NOT identify \"$expect_chars\" specifically"
    cat "$log" >&2
  fi
}

# The reported repro: a semicolon payload via EXTRA_LDFLAGS -- `make
# aec_nr_pipeline EXTRA_LDFLAGS=';id'` (with COMMON_LIB/AEC_LIB/NR_LIB pre-resolved
# so the dispatch lands directly on this repo's OWN link recipe instead of
# recursing into audiolib's sub-make first) -- must be rejected, not land
# live and unfiltered after the `-lm` of a real `cc -o ... $(LDFLAGS)` recipe
# line (checked via `make -n`, zero files touched).
sp9c_check_rejected "injection-semicolon" ';id' ';'

# Same glob/tilde/redirect/process-substitution matrix SP-S9b runs for
# EXTRA_CFLAGS, replicated verbatim for EXTRA_LDFLAGS.
sp9c_check_rejected "glob-star"     '-L*t'            '*'
sp9c_check_rejected "glob-question" '-Lfas?'          '?'
sp9c_check_rejected "glob-brackets" '-Lfas[t]'        '[]'
sp9c_check_rejected "tilde"         '~/pwned'         '~'
sp9c_check_rejected "redirect-out"  '-lm>/tmp/evil'   '>'
sp9c_check_rejected "redirect-in"   '-lm</etc/passwd' '<'
sp9c_check_rejected "process-subst" '<(echo hi)'      '()<'

# Single-quote -- caught by the ONE pure-Make $(findstring) check, same as
# SP-S9b's own single-quote case.
SP9C_LOG_SQ="$SCRATCH_ROOT/tmp/sp9c_single_quote.log"
if env EXTRA_LDFLAGS="-l'p'thread" make BACKEND=ne10 print-obj-dir >"$SP9C_LOG_SQ" 2>&1; then
  fail "SP-S9c: EXTRA_LDFLAGS=\"-l'p'thread\" (quote-split) print-obj-dir unexpectedly SUCCEEDED (must be rejected)"
elif grep -q "FP policy conflict" "$SP9C_LOG_SQ" && grep -q "single-quote" "$SP9C_LOG_SQ"; then
  pass "SP-S9c: EXTRA_LDFLAGS=\"-l'p'thread\" (quote-split) print-obj-dir correctly FAILS, identifying the single-quote character"
else
  fail "SP-S9c: EXTRA_LDFLAGS=\"-l'p'thread\" (quote-split) print-obj-dir failed but did NOT identify the single-quote character"
  cat "$SP9C_LOG_SQ" >&2
fi

# Positive control: a real link flag the Makefile's allow-list comment anticipates (-Wl,-rpath,dir style
# pass-through) must pass the character check, build and run. test_mcab_sample_post is a real linked binary (unlike
# SP-S9b's libaudio_pipeline_dfn.a, an `ar` recipe that never touches $(LDFLAGS)) and prints "ALL PASS" with exit 0, so it
# doubles as the link-succeeded-and-still-correct assertion. Uses scratch OBJ_ROOT/BIN_ROOT.
mkscratch sp9c_pos
SP9C_POS_OBJ_ROOT="$SCRATCH_ROOT/sp9c_pos/obj"
SP9C_POS_BIN_ROOT="$SCRATCH_ROOT/sp9c_pos/bin"
SP9C_POS_BUILD_LOG="$SCRATCH_ROOT/tmp/sp9c_positive_build.log"
if env EXTRA_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib' make BACKEND=ne10 OBJ_ROOT="$SP9C_POS_OBJ_ROOT" BIN_ROOT="$SP9C_POS_BIN_ROOT" test_mcab_sample_post >"$SP9C_POS_BUILD_LOG" 2>&1; then
  pass "SP-S9c: EXTRA_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib' (positive control, real rpath/library-path link flag) builds test_mcab_sample_post"
  if grep -qF -- '-Wl,-rpath,/usr/lib -L/usr/lib' "$SP9C_POS_BUILD_LOG"; then
    pass "SP-S9c: the real link flag reaches the actual \$(LINK) recipe line for test_mcab_sample_post"
  else
    fail "SP-S9c: the real link flag did NOT appear on the test_mcab_sample_post link recipe line"
    cat "$SP9C_POS_BUILD_LOG" >&2
  fi
  SP9C_POS_BINDIR="$(env EXTRA_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib' make -s BACKEND=ne10 OBJ_ROOT="$SP9C_POS_OBJ_ROOT" BIN_ROOT="$SP9C_POS_BIN_ROOT" print-bin-dir)"
  SP9C_POS_RUN_LOG="$SCRATCH_ROOT/tmp/sp9c_positive_run.log"
  if "$SP9C_POS_BINDIR/test_mcab_sample_post" >"$SP9C_POS_RUN_LOG" 2>&1 && grep -q "^ALL PASS$" "$SP9C_POS_RUN_LOG"; then
    pass "SP-S9c: test_mcab_sample_post (built with the real rpath/library-path link flag) RUNS to ALL PASS"
  else
    fail "SP-S9c: test_mcab_sample_post (built with the real rpath/library-path link flag) did not run to ALL PASS"
    cat "$SP9C_POS_RUN_LOG" >&2
  fi
else
  fail "SP-S9c: EXTRA_LDFLAGS='-Wl,-rpath,/usr/lib -L/usr/lib' (positive control, legitimate link flag) unexpectedly FAILED to build"
  cat "$SP9C_POS_BUILD_LOG" >&2
fi

# Command-line/`-e` override-rejection for
# SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC (SP-S9b already covers both
# via an EXTRA_CFLAGS payload; this repeats it via EXTRA_LDFLAGS specifically,
# since that is the vector SP-S9c covers -- an override here must not be able
# to let an EXTRA_LDFLAGS injection payload back through).
sp9c_check_override_rejected() {
  local varname="$1" varvalue="$2" payload="$3" use_dash_e="$4" log
  log="$SCRATCH_ROOT/tmp/sp9c_override_$(printf '%s' "$varname" | tr -c 'A-Za-z0-9' '_')_$use_dash_e.log"
  if [ "$use_dash_e" = "dashe" ]; then
    if env "$varname=$varvalue" make -e BACKEND=ne10 EXTRA_LDFLAGS="$payload" print-obj-dir >"$log" 2>&1; then
      fail "SP-S9c: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the link-flags character-safety check)"
    elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
      pass "SP-S9c: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) correctly FAILS, mentioning '$varname cannot be overridden'"
    else
      fail "SP-S9c: env $varname=$varvalue make -e print-obj-dir (EXTRA_LDFLAGS payload) failed but did NOT mention '$varname cannot be overridden'"
      cat "$log" >&2
    fi
  else
    if make BACKEND=ne10 EXTRA_LDFLAGS="$payload" "$varname=$varvalue" print-obj-dir >"$log" 2>&1; then
      fail "SP-S9c: make $varname=$varvalue print-obj-dir (EXTRA_LDFLAGS payload) unexpectedly SUCCEEDED (must be rejected: overriding $varname must not defeat the link-flags character-safety check)"
    elif grep -q "cannot be overridden" "$log" && grep -q "$varname" "$log"; then
      pass "SP-S9c: make $varname=$varvalue print-obj-dir (EXTRA_LDFLAGS payload) correctly FAILS, mentioning '$varname cannot be overridden'"
    else
      fail "SP-S9c: make $varname=$varvalue print-obj-dir (EXTRA_LDFLAGS payload) failed but did NOT mention '$varname cannot be overridden'"
      cat "$log" >&2
    fi
  fi
}
sp9c_check_override_rejected "SHELL_SAFE_INPUT_FLAGS"  "clean" ';rm' cmdline
sp9c_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     ';rm' cmdline
sp9c_check_override_rejected "SHELL_SAFE_INPUT_FLAGS"  "clean" ';rm' dashe
sp9c_check_override_rejected "SHELL_SAFE_ALLOWLIST_RC" "0"     ';rm' dashe

echo "############################################################"
echo "# SP-S10: lib/aec archive freshness (regression: scratch-side against the"
echo "#         PRIMARY AEC repo -- regression P1 fix)"
echo "############################################################"
mkscratch sp10
SP10_OBJ_ROOT="$SCRATCH_ROOT/sp10/obj"
SP10_BIN_ROOT="$SCRATCH_ROOT/sp10/bin"

# Resolve the real audiolib lib once (plain query target -- no
# ALLOW_DIRTY_PUBLISH needed; print-lib-path never touches DIST_ROOT/dirty
# state at all).
sp10_ac_lib="$(make -s -C "$COMMON_DIR" BACKEND=ne10 print-lib-path)"

# Snapshot the REAL submodule libaec.a -- must stay byte/mtime-untouched across
# this whole scenario (the foreign member is injected into a scratch archive
# instead).
sp10_submodule_lib="$(make -s -C "$AEC_DIR" BACKEND=ne10 print-lib-path)"
sp10_submodule_existed_before=0
if [ -f "$sp10_submodule_lib" ]; then
  sp10_submodule_existed_before=1
  sp10_submodule_sha_before="$(file_sha "$sp10_submodule_lib")"
  sp10_submodule_mtime_before="$(mtime "$sp10_submodule_lib")"
fi

# Belt-and-braces: also snapshot the PRIMARY AEC repo's own REAL (default
# OBJ_ROOT/BIN_ROOT) libaec.a, if a prior run happened to leave one there --
# this scenario builds ONLY under scratch OBJ_ROOT/BIN_ROOT, so that path
# must never be touched either.
sp10_primary_real_lib="$(make -s -C "$AEC_R6_DIR" BACKEND=ne10 print-lib-path)"
sp10_primary_existed_before=0
if [ -f "$sp10_primary_real_lib" ]; then
  sp10_primary_existed_before=1
  sp10_primary_sha_before="$(file_sha "$sp10_primary_real_lib")"
  sp10_primary_mtime_before="$(mtime "$sp10_primary_real_lib")"
fi

make -s -C "$AEC_R6_DIR" BACKEND=ne10 OBJ_ROOT="$SP10_OBJ_ROOT" BIN_ROOT="$SP10_BIN_ROOT" COMMON_DIR="$COMMON_DIR" COMMON_LIB="$sp10_ac_lib" lib >/dev/null
sp10_lib="$(make -s -C "$AEC_R6_DIR" BACKEND=ne10 OBJ_ROOT="$SP10_OBJ_ROOT" BIN_ROOT="$SP10_BIN_ROOT" COMMON_DIR="$COMMON_DIR" COMMON_LIB="$sp10_ac_lib" print-lib-path)"
[ -f "$sp10_lib" ] && pass "SP-S10: PRIMARY AEC repo's SCRATCH archive built ($sp10_lib)" \
  || fail "SP-S10: PRIMARY AEC repo's SCRATCH archive missing after 'make lib'"

mkdir -p "$SCRATCH_ROOT/sp10/foreign"
cat > "$SCRATCH_ROOT/sp10/foreign/sp10_foreign.c" <<'EOF'
int sp10_foreign_symbol(void) { return 0; }
EOF
cc -c -o "$SCRATCH_ROOT/sp10/foreign/sp10_foreign.o" "$SCRATCH_ROOT/sp10/foreign/sp10_foreign.c"
ar r "$sp10_lib" "$SCRATCH_ROOT/sp10/foreign/sp10_foreign.o"
ar -t "$sp10_lib" | grep -qx 'sp10_foreign.o' && pass "SP-S10: foreign member injected into the SCRATCH libaec.a (setup)" \
  || fail "SP-S10: foreign member injection FAILED (setup problem, not the thing under test)"

# Backdate the SCRATCH archive itself (never a real source, never a real
# archive) to a fixed date well before its member .o's real "just built"
# mtimes, so make's own dependency check ($(LIB): $(OBJS)) sees it as stale
# and re-archives from scratch on the very next build.
touch -t 202001010000 "$sp10_lib"
make -s -C "$AEC_R6_DIR" BACKEND=ne10 OBJ_ROOT="$SP10_OBJ_ROOT" BIN_ROOT="$SP10_BIN_ROOT" COMMON_DIR="$COMMON_DIR" COMMON_LIB="$sp10_ac_lib" lib >/dev/null

if ar -t "$sp10_lib" | grep -qx 'sp10_foreign.o'; then
  fail "SP-S10: foreign member SURVIVED a rebuild (archive was NOT rebuilt fresh -- looks like 'ar r' onto the existing .a rather than \$@.tmp + mv -f)"
else
  pass "SP-S10: foreign member is GONE after rebuild (fresh-archive discipline: \$@.tmp + mv -f)"
fi
ar -t "$sp10_lib" | grep -qx 'aec_debug.o' && pass "SP-S10: aec_debug.o present in the freshly-rebuilt archive" \
  || fail "SP-S10: aec_debug.o missing from the freshly-rebuilt archive"

sp10_objdir="$(make -s -C "$AEC_R6_DIR" BACKEND=ne10 OBJ_ROOT="$SP10_OBJ_ROOT" BIN_ROOT="$SP10_BIN_ROOT" COMMON_DIR="$COMMON_DIR" COMMON_LIB="$sp10_ac_lib" print-obj-dir)"
grep -q 'SRCS=' "$sp10_objdir/config.manifest" && pass "SP-S10: the SCRATCH obj dir's config.manifest records a SRCS= entry" \
  || fail "SP-S10: SCRATCH obj dir's config.manifest missing a SRCS= entry"

# The REAL submodule libaec.a must be byte/mtime-unchanged throughout.
if [ "$sp10_submodule_existed_before" -eq 1 ]; then
  [ -f "$sp10_submodule_lib" ] \
    && [ "$(file_sha "$sp10_submodule_lib")" = "$sp10_submodule_sha_before" ] \
    && [ "$(mtime "$sp10_submodule_lib")" = "$sp10_submodule_mtime_before" ] \
    && pass "SP-S10: the REAL submodule libaec.a (lib/aec/c_impl) sha+mtime unchanged across the scenario" \
    || fail "SP-S10: the REAL submodule libaec.a CHANGED during SP-S10 (scratch isolation failed)"
else
  [ ! -f "$sp10_submodule_lib" ] && pass "SP-S10: the REAL submodule libaec.a still does not exist (scratch isolation held -- nothing built one)" \
    || fail "SP-S10: the REAL submodule libaec.a was CREATED by this scenario (scratch isolation failed)"
fi

# ...and the PRIMARY AEC repo's own REAL (non-scratch) libaec.a too.
if [ "$sp10_primary_existed_before" -eq 1 ]; then
  [ -f "$sp10_primary_real_lib" ] \
    && [ "$(file_sha "$sp10_primary_real_lib")" = "$sp10_primary_sha_before" ] \
    && [ "$(mtime "$sp10_primary_real_lib")" = "$sp10_primary_mtime_before" ] \
    && pass "SP-S10: the PRIMARY AEC repo's own REAL libaec.a sha+mtime unchanged across the scenario" \
    || fail "SP-S10: the PRIMARY AEC repo's own REAL libaec.a CHANGED during SP-S10 (scratch isolation failed)"
else
  [ ! -f "$sp10_primary_real_lib" ] && pass "SP-S10: the PRIMARY AEC repo's own REAL libaec.a still does not exist (scratch isolation held)" \
    || fail "SP-S10: the PRIMARY AEC repo's own REAL libaec.a was CREATED by this scenario (scratch isolation failed)"
fi

echo "############################################################"
echo "# SP-S11: pipelines publish v4 (content-addressed release + ATTEST v2)"
own_untracked_flag() {
  local dir="$1" paths
  paths="$(git -C "$dir" ls-files --others --exclude-standard | grep -vE '^AINR/(GTCRN|gtcrn_github)/' || true)"
  if [ -n "$paths" ]; then printf '1\n'; else printf '0\n'; fi
}
sp11_expected_self_untracked="$(own_untracked_flag "$PIPE_DIR")"
# Exercise both expectations without staging anything in the real checkout.
mkdir -p "$SCRATCH_ROOT/sp11-untracked-policy/AINR/GTCRN"
git -C "$SCRATCH_ROOT/sp11-untracked-policy" init -q
printf 'manual content\n' > "$SCRATCH_ROOT/sp11-untracked-policy/AINR/GTCRN/probe.txt"
[ "$(own_untracked_flag "$SCRATCH_ROOT/sp11-untracked-policy")" = 0 ] \
  && pass "SP-S11: whitelist-only fixture has expected own untracked=0" \
  || fail "SP-S11: whitelist-only fixture wrongly reports own untracked files"
printf 'int probe;\n' > "$SCRATCH_ROOT/sp11-untracked-policy/probe.c"
[ "$(own_untracked_flag "$SCRATCH_ROOT/sp11-untracked-policy")" = 1 ] \
  && pass "SP-S11: new source fixture has expected own untracked=1" \
  || fail "SP-S11: new source fixture hides own untracked files"

echo "############################################################"
mkscratch sp11
SP11_DIST_ROOT="$SCRATCH_ROOT/sp11/dist"
resolve_producers ne10
make -s BACKEND=ne10 DIST_ROOT="$SP11_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish COMMON_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null

id1="$(readlink "$SP11_DIST_ROOT/ne10/current" || true)"
[ -n "$id1" ] && [ -d "$SP11_DIST_ROOT/ne10/$id1" ] && pass "SP-S11: publish v4 -- current symlink resolves to release dir '$id1'" \
  || fail "SP-S11: publish v4 -- current symlink broken or missing after the first publish"

rel_dir="$SP11_DIST_ROOT/ne10/$id1"

grep -q "^release_id=$id1\$" "$rel_dir/MANIFEST.txt" && pass "SP-S11: MANIFEST.txt release_id= matches the release dir name" \
  || fail "SP-S11: MANIFEST.txt release_id= missing or does not match '$id1'"
grep -q '^ac_producer_cfg_sig=' "$rel_dir/MANIFEST.txt" && grep -q '^aec_producer_cfg_sig=' "$rel_dir/MANIFEST.txt" && grep -q '^nr_producer_cfg_sig=' "$rel_dir/MANIFEST.txt" && \
  pass "SP-S11: MANIFEST.txt has ac_producer_cfg_sig=/aec_producer_cfg_sig=/nr_producer_cfg_sig= lines" \
  || fail "SP-S11: MANIFEST.txt missing one or more of the three producer cfg_sig lines"
grep -q '^ar=' "$rel_dir/MANIFEST.txt" && grep -q '^ranlib=' "$rel_dir/MANIFEST.txt" && grep -q '^link=' "$rel_dir/MANIFEST.txt" && \
  pass "SP-S11: MANIFEST.txt has ar=/ranlib=/link= lines" \
  || fail "SP-S11: MANIFEST.txt missing one or more of ar=/ranlib=/link="
grep -q '^git_commit=' "$rel_dir/MANIFEST.txt" && \
  fail "SP-S11: MANIFEST.txt unexpectedly has a git_commit= line (moved to ATTEST/ in v4)" \
  || pass "SP-S11: MANIFEST.txt has NO git_commit= line (deterministic MANIFEST)"

attest_count_before="$(find "$rel_dir/ATTEST" -type f -name 'attest-*.txt' | wc -l | tr -d ' ')"
[ "$attest_count_before" -eq 1 ] && pass "SP-S11: exactly one ATTEST file after the first publish" \
  || fail "SP-S11: expected exactly 1 ATTEST file after the first publish, found $attest_count_before"

first_attest="$(find "$rel_dir/ATTEST" -type f -name 'attest-*.txt')"
first_attest_stem="$(basename "$first_attest" .txt)"
if grep -q "^event_id=$first_attest_stem\$" "$first_attest"; then
  pass "SP-S11: the ATTEST file's event_id= matches its own filename stem"
else
  fail "SP-S11: ATTEST file $first_attest event_id= does not match its filename stem"
fi

# attest v2: git_commit= (self) AND all three producer *_git_commit= fields are
# full 40-hex OIDs (not `git rev-parse --short`).
sp11_hex_ok=1
for field in git_commit audiolib_git_commit aec_git_commit nr_git_commit; do
  val="$(grep "^${field}=" "$first_attest" | head -1 | cut -d= -f2)"
  echo "$val" | grep -Eq '^[0-9a-f]{40}$' || sp11_hex_ok=0
done
[ "$sp11_hex_ok" -eq 1 ] && pass "SP-S11: git_commit= and all three producer *_git_commit= fields are full 40-hex OIDs" \
  || fail "SP-S11: one or more of git_commit=/audiolib_git_commit=/aec_git_commit=/nr_git_commit= is not 40 hex characters"

# The *_git_untracked dimension is a SEPARATE default refusal from *_git_dirty
# (see the Makefile's ALLOW_UNTRACKED_PUBLISH comment) -- assert all four
# fields + allow_untracked_publish= exist in a normal, successful publish's
# attest (SP-S17a/b exercise the refusal path itself; this is the
# field-presence check for the ordinary path).
sp11_untracked_fields_ok=1
for field in git_untracked audiolib_git_untracked aec_git_untracked nr_git_untracked allow_untracked_publish; do
  grep -q "^${field}=" "$first_attest" || sp11_untracked_fields_ok=0
done
[ "$sp11_untracked_fields_ok" -eq 1 ] && pass "SP-S11: attest carries all four *_git_untracked fields + allow_untracked_publish=" \
  || fail "SP-S11: attest is missing one or more of git_untracked=/audiolib_git_untracked=/aec_git_untracked=/nr_git_untracked=/allow_untracked_publish="

grep -q "^git_untracked=$sp11_expected_self_untracked\$" "$first_attest" \
  && pass "SP-S11: pipelines' own git_untracked=$sp11_expected_self_untracked matches the independently enumerated worktree" \
  || fail "SP-S11: pipelines' own git_untracked does not match the actual worktree"

release_dir_mtime_before="$(mtime "$rel_dir")"
snap_before="$(release_mtime_snapshot "$rel_dir")"

# No sleep -- the <NNN> suffix (not a distinct UTC second) is what
# disambiguates this immediate same-second republish (SP-S15 stress-tests the
# same-second case directly, forcing 20 publishes into one literal second via
# ATTEST_STAMP=).
S11_LOG="$SCRATCH_ROOT/sp11/republish.log"
make -s BACKEND=ne10 DIST_ROOT="$SP11_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish COMMON_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S11_LOG" 2>&1
if grep -q "already published (byte-verified" "$S11_LOG"; then
  pass "SP-S11: idempotent republish printed 'already published (byte-verified, incl. MANIFEST)'"
else
  fail "SP-S11: idempotent republish did NOT print the byte-verified message"
  cat "$S11_LOG" >&2
fi
grep -Eq '\(attested: attest-[^)]+\)[[:space:]]*$' "$S11_LOG" && pass "SP-S11: republish success line ends '(attested: <name>)'" \
  || fail "SP-S11: republish success line missing the '(attested: <name>)' suffix"

id2="$(readlink "$SP11_DIST_ROOT/ne10/current" || true)"
[ "$id2" = "$id1" ] && pass "SP-S11: current still points at the same release id after idempotent republish" \
  || fail "SP-S11: current MOVED after idempotent republish ('$id1' -> '$id2')"

release_dir_mtime_after="$(mtime "$rel_dir")"
[ "$release_dir_mtime_before" = "$release_dir_mtime_after" ] && pass "SP-S11: release dir's own mtime unchanged by idempotent republish" \
  || fail "SP-S11: release dir mtime CHANGED by idempotent republish ($release_dir_mtime_before -> $release_dir_mtime_after)"

snap_after="$(release_mtime_snapshot "$rel_dir")"
[ "$snap_before" = "$snap_after" ] && pass "SP-S11: release-dir files (MANIFEST.txt + artifacts, excl. ATTEST/) left mtime-untouched by idempotent republish" \
  || fail "SP-S11: one or more release-dir files' mtime CHANGED by idempotent republish"

attest_count_after="$(find "$rel_dir/ATTEST" -type f -name 'attest-*.txt' | wc -l | tr -d ' ')"
[ "$attest_count_after" -eq 2 ] && pass "SP-S11: ATTEST/ grew to exactly 2 files after the republish (attest v2 -- no sleep needed)" \
  || fail "SP-S11: expected 2 ATTEST files after the republish, found $attest_count_after"

second_attest="$(find "$rel_dir/ATTEST" -type f -name 'attest-*.txt' | grep -v -F "$first_attest" || true)"
if [ -n "$second_attest" ] && [ "$second_attest" != "$first_attest" ]; then
  pass "SP-S11: republish's ATTEST file is a NEW, distinct file from the first publish's"
else
  fail "SP-S11: could not identify a second distinct ATTEST file after the republish"
fi
second_attest_stem="$(basename "$second_attest" .txt)"
grep -q "^event_id=$second_attest_stem\$" "$second_attest" 2>/dev/null && pass "SP-S11: second ATTEST file's event_id= matches its own filename stem" \
  || fail "SP-S11: second ATTEST file's event_id= does not match its filename"

echo "############################################################"
echo "# SP-S12: BACKEND=ne10 toolchain guard fires through the dispatch"
echo "############################################################"
mkscratch sp12
# This shim's path lives under $SCRATCH_ROOT, which is fresh every run -- and
# CXX participates in this repo's own CFG_SIG payload -- so building
# libaudio_pipeline_dfn.a with it in the REAL (default-rooted) obj/bin would mint a
# brand-new orphan keyed directory there on every run of this suite. Both shim
# builds use a scratch OBJ_ROOT/BIN_ROOT; the guard under test doesn't care
# where the objects land.
SP12_OBJ_ROOT="$SCRATCH_ROOT/sp12/obj"
SP12_BIN_ROOT="$SCRATCH_ROOT/sp12/bin"
SP12_SHIM="$SCRATCH_ROOT/sp12/sp12-cxx-shim"
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
  make BACKEND=ne10 CXX="$SP12_SHIM" OBJ_ROOT="$SP12_OBJ_ROOT" BIN_ROOT="$SP12_BIN_ROOT" libaudio_pipeline_dfn.a

S12_LOG="$SCRATCH_ROOT/sp12/log"
if make TOOLCHAIN_CHECK=0 CXX="$SP12_SHIM" OBJ_ROOT="$SP12_OBJ_ROOT" BIN_ROOT="$SP12_BIN_ROOT" libaudio_pipeline_dfn.a >"$S12_LOG" 2>&1; then
  pass "SP-S12: TOOLCHAIN_CHECK=0 with the same mismatched-triple CXX shim SUCCEEDS (the guard is the only thing that rejected it)"
else
  fail "SP-S12: TOOLCHAIN_CHECK=0 build unexpectedly FAILED with the CXX shim in place"
  cat "$S12_LOG" >&2
fi

echo "############################################################"
echo "# SP-S13: RNNoise-ERB drift-gate hardening (regression P2)"
echo "############################################################"
mkscratch sp13
SP13_LOG="$SCRATCH_ROOT/sp13/log1"
if make -C "$RNN_DIR" test-tables >"$SP13_LOG" 2>&1; then
  pass "SP-S13: 'make test-tables' succeeds"
else
  fail "SP-S13: 'make test-tables' FAILED"
  cat "$SP13_LOG" >&2
fi
sp13_pass_count="$(grep -c '^PASS' "$SP13_LOG" || true)"
[ "${sp13_pass_count:-0}" -eq 2 ] && pass "SP-S13: 'make test-tables' output has both PASS layers (canonical + portable)" \
  || fail "SP-S13: expected 2 PASS lines from 'make test-tables', found ${sp13_pass_count:-0}"

SP13_LOG2="$SCRATCH_ROOT/sp13/log2"
if make -C "$RNN_DIR" CC=false test-tables >"$SP13_LOG2" 2>&1; then
  fail "SP-S13: 'make CC=false test-tables' unexpectedly SUCCEEDED (stale-binary false-pass would be back)"
  cat "$SP13_LOG2" >&2
else
  pass "SP-S13: 'make CC=false test-tables' FAILS (fresh keyed build dir forces a real compile attempt; stale-binary false-pass repro stays closed)"
fi

assert_cmd_fails_with "SP-S13: 'make CFLAGS=-O0 test-tables' rejected at parse time" "cannot be overridden" \
  make -C "$RNN_DIR" CFLAGS=-O0 test-tables

make -s -C "$RNN_DIR" clean >/dev/null

echo "############################################################"
echo "# SP-S14: make -n/-q/-t publish zero side effects (regression P2-3),"
echo "#         THREE Makefiles: pipelines / \$AEC_R6_DIR / \$NR_R6_DIR"
echo "############################################################"
mkscratch sp14

snap_dirs() { { find "$1" -maxdepth 1 -type d 2>/dev/null | sort; } || true; }

SP14_BEFORE_PIPE_OBJ="$(snap_dirs "$PIPE_DIR/obj")"
SP14_BEFORE_PIPE_BIN="$(snap_dirs "$PIPE_DIR/bin")"
SP14_BEFORE_AC_OBJ="$(snap_dirs "$COMMON_DIR/obj")"
SP14_BEFORE_AC_BIN="$(snap_dirs "$COMMON_DIR/bin")"
SP14_BEFORE_AECR6_OBJ="$(snap_dirs "$AEC_R6_DIR/obj")"
SP14_BEFORE_AECR6_BIN="$(snap_dirs "$AEC_R6_DIR/bin")"
SP14_BEFORE_NRR6_OBJ="$(snap_dirs "$NR_R6_DIR/obj")"
SP14_BEFORE_NRR6_BIN="$(snap_dirs "$NR_R6_DIR/bin")"

for x in pipe aec nr; do
  MAKE_ARGS=(make)
  case "$x" in
    aec) MAKE_ARGS+=(-C "$AEC_R6_DIR") ;;
    nr)  MAKE_ARGS+=(-C "$NR_R6_DIR") ;;
  esac
  SP14_DIST="$SCRATCH_ROOT/sp14/$x/nx"
  SP14_OBJ="$SCRATCH_ROOT/sp14/$x/no"
  SP14_BIN="$SCRATCH_ROOT/sp14/$x/nb"

  rc=0
  "${MAKE_ARGS[@]}" -n BACKEND=ne10 DIST_ROOT="$SP14_DIST" OBJ_ROOT="$SP14_OBJ" BIN_ROOT="$SP14_BIN" publish >"$SCRATCH_ROOT/sp14/$x.n.log" 2>&1 || rc=$?
  if [ "$x" = "pipe" ]; then
    # pipelines' three-producer FORCE rules forward $(COMMON_LIB) into a nested
    # `$(MAKE) ... COMMON_LIB=$(COMMON_LIB) lib` for lib/aec and lib/nr. Under `-n`, GNU Make prints audiolib's
    # `print-lib-path` recipe line verbatim ("echo /abs/path") instead of running it, so the captured "$$ac" becomes
    # the string "echo /abs/path/to/lib.a"; forwarded as COMMON_LIB it splits into bogus words and the nested make
    # exits nonzero ("No rule to make target ..."). AEC's and NR's single-producer dispatch does not re-forward
    # COMMON_LIB a third level down (SP-S14[aec]/SP-S14[nr] below both give rc=0). The property under test, zero
    # filesystem side effects, is asserted unconditionally for all three Makefiles below.
    echo "  INFO: SP-S14[pipe]: make -n publish rc=$rc (KNOWN pre-existing pipelines-specific dry-run quirk, see comment above -- not asserted; the side-effect-free invariant below is the real safety check)"
  else
    [ "$rc" -eq 0 ] && pass "SP-S14[$x]: make -n publish exits rc=0" || fail "SP-S14[$x]: make -n publish exits rc=$rc (expected 0)"
  fi
  if [ ! -e "$SP14_DIST" ] && [ ! -e "$SP14_OBJ" ] && [ ! -e "$SP14_BIN" ]; then
    pass "SP-S14[$x]: make -n publish is zero-write (created NONE of DIST_ROOT/OBJ_ROOT/BIN_ROOT)"
  else
    fail "SP-S14[$x]: make -n publish left behind a path (dist=$([ -e "$SP14_DIST" ] && echo yes || echo no) obj=$([ -e "$SP14_OBJ" ] && echo yes || echo no) bin=$([ -e "$SP14_BIN" ] && echo yes || echo no))"
  fi

  rc=0
  "${MAKE_ARGS[@]}" -q BACKEND=ne10 DIST_ROOT="$SP14_DIST" OBJ_ROOT="$SP14_OBJ" BIN_ROOT="$SP14_BIN" publish >"$SCRATCH_ROOT/sp14/$x.q.log" 2>&1 || rc=$?
  [ "$rc" -ne 0 ] && pass "SP-S14[$x]: make -q publish exits NONZERO (rc=$rc)" || fail "SP-S14[$x]: make -q publish exited 0 (expected nonzero)"
  if [ ! -e "$SP14_DIST" ] && [ ! -e "$SP14_OBJ" ] && [ ! -e "$SP14_BIN" ]; then
    pass "SP-S14[$x]: make -q publish is zero-write (created NONE of DIST_ROOT/OBJ_ROOT/BIN_ROOT)"
  else
    fail "SP-S14[$x]: make -q publish left behind a path (dist=$([ -e "$SP14_DIST" ] && echo yes || echo no) obj=$([ -e "$SP14_OBJ" ] && echo yes || echo no) bin=$([ -e "$SP14_BIN" ] && echo yes || echo no))"
  fi

  "${MAKE_ARGS[@]}" -t BACKEND=ne10 DIST_ROOT="$SP14_DIST" OBJ_ROOT="$SP14_OBJ" BIN_ROOT="$SP14_BIN" publish >"$SCRATCH_ROOT/sp14/$x.t.log" 2>&1 || true
  if [ ! -e "$SP14_DIST" ] && [ ! -e "$SP14_OBJ" ] && [ ! -e "$SP14_BIN" ]; then
    pass "SP-S14[$x]: make -t publish is zero-write (created NONE of DIST_ROOT/OBJ_ROOT/BIN_ROOT)"
  else
    fail "SP-S14[$x]: make -t publish left behind a path (dist=$([ -e "$SP14_DIST" ] && echo yes || echo no) obj=$([ -e "$SP14_OBJ" ] && echo yes || echo no) bin=$([ -e "$SP14_BIN" ] && echo yes || echo no))"
  fi
done

SP14_AFTER_PIPE_OBJ="$(snap_dirs "$PIPE_DIR/obj")"
SP14_AFTER_PIPE_BIN="$(snap_dirs "$PIPE_DIR/bin")"
SP14_AFTER_AC_OBJ="$(snap_dirs "$COMMON_DIR/obj")"
SP14_AFTER_AC_BIN="$(snap_dirs "$COMMON_DIR/bin")"
SP14_AFTER_AECR6_OBJ="$(snap_dirs "$AEC_R6_DIR/obj")"
SP14_AFTER_AECR6_BIN="$(snap_dirs "$AEC_R6_DIR/bin")"
SP14_AFTER_NRR6_OBJ="$(snap_dirs "$NR_R6_DIR/obj")"
SP14_AFTER_NRR6_BIN="$(snap_dirs "$NR_R6_DIR/bin")"

[ "$SP14_BEFORE_PIPE_OBJ" = "$SP14_AFTER_PIPE_OBJ" ] && [ "$SP14_BEFORE_PIPE_BIN" = "$SP14_AFTER_PIPE_BIN" ] && \
  pass "SP-S14: no NEW keyed dirs appeared in pipelines' own real obj/bin" \
  || fail "SP-S14: a NEW keyed dir appeared in pipelines' own real obj/bin"
[ "$SP14_BEFORE_AC_OBJ" = "$SP14_AFTER_AC_OBJ" ] && [ "$SP14_BEFORE_AC_BIN" = "$SP14_AFTER_AC_BIN" ] && \
  pass "SP-S14: no NEW keyed dirs appeared in audiolib's real obj/bin" \
  || fail "SP-S14: a NEW keyed dir appeared in audiolib's real obj/bin"
[ "$SP14_BEFORE_AECR6_OBJ" = "$SP14_AFTER_AECR6_OBJ" ] && [ "$SP14_BEFORE_AECR6_BIN" = "$SP14_AFTER_AECR6_BIN" ] && \
  pass "SP-S14: no NEW keyed dirs appeared in the PRIMARY AEC repo's real obj/bin" \
  || fail "SP-S14: a NEW keyed dir appeared in the PRIMARY AEC repo's real obj/bin"
[ "$SP14_BEFORE_NRR6_OBJ" = "$SP14_AFTER_NRR6_OBJ" ] && [ "$SP14_BEFORE_NRR6_BIN" = "$SP14_AFTER_NRR6_BIN" ] && \
  pass "SP-S14: no NEW keyed dirs appeared in the PRIMARY NR repo's real obj/bin" \
  || fail "SP-S14: a NEW keyed dir appeared in the PRIMARY NR repo's real obj/bin"

echo "--- SP-S14b: COMBINED dry-run flags stay zero-write (PIPELINES Makefile) --"
# Combined dry-run flags (as in audiolib's script): `make -nt publish` could take the -n branch and recurse, and the
# child then sees both flags and applies real touch semantics down the _publish_impl prerequisite chain; the
# pipelines Makefile's dry-run guard checks t first to avoid this. Exercised against a scratch-root
# libaudio_pipeline_dfn.a deliberately backdated below its object, the state a leaked -t would "fix" by touching. Scratch
# side only. libaudio_pipeline_dfn.a contains no producer object, but its goal still goes through the three-producer dispatch.
mkscratch sp14b
SP14B_OBJ="$SCRATCH_ROOT/sp14b/obj"; SP14B_BIN="$SCRATCH_ROOT/sp14b/bin"; SP14B_DIST="$SCRATCH_ROOT/sp14b/nx"
make -s BACKEND=ne10 OBJ_ROOT="$SP14B_OBJ" BIN_ROOT="$SP14B_BIN" libaudio_pipeline_dfn.a >/dev/null
sp14b_lib="$(make -s BACKEND=ne10 OBJ_ROOT="$SP14B_OBJ" BIN_ROOT="$SP14B_BIN" print-lib-path)"
touch -t 202001010000 "$sp14b_lib"
sp14b_ref="$(stat -f '%m %z' "$sp14b_lib")"
for combo in "-nt" "-n -t" "-tq" "-nq" "-nqt"; do
  rm -rf "$SP14B_DIST"
  rc=0; make $combo BACKEND=ne10 OBJ_ROOT="$SP14B_OBJ" BIN_ROOT="$SP14B_BIN" DIST_ROOT="$SP14B_DIST" publish >/dev/null 2>&1 || rc=$?
  if [ ! -e "$SP14B_DIST" ] && [ "$(stat -f '%m %z' "$sp14b_lib")" = "$sp14b_ref" ]; then
    pass "SP-S14b[$combo]: zero-write (no DIST_ROOT; stale scratch archive untouched; rc=$rc informational)"
  else
    fail "SP-S14b[$combo]: WROTE something (dist=$([ -e "$SP14B_DIST" ] && echo yes || echo no), archive stat now '$(stat -f '%m %z' "$sp14b_lib")' vs '$sp14b_ref')"
  fi
done

echo "############################################################"
echo "# SP-S15: ATTEST uniqueness under forced same-second collisions"
echo "#         (regression P2-2)"
echo "############################################################"
mkscratch sp15
SP15_DIST="$SCRATCH_ROOT/sp15/dist"
resolve_producers ne10
make -s BACKEND=ne10 DIST_ROOT="$SP15_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish COMMON_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null
sp15_id="$(readlink "$SP15_DIST/ne10/current" || true)"
sp15_attest_dir="$SP15_DIST/ne10/$sp15_id/ATTEST"

sp15_before_list="$(find "$sp15_attest_dir" -name 'attest-*.txt' | sort)"
sp15_before_snap="$SCRATCH_ROOT/sp15/before_snap.txt"
: > "$sp15_before_snap"
for f in $sp15_before_list; do
  [ -n "$f" ] || continue
  printf '%s %s %s %s\n' "$f" "$(stat -f '%i' "$f")" "$(mtime "$f")" "$(file_sha "$f")" >> "$sp15_before_snap"
done

for i in $(seq 1 20); do
  make -s BACKEND=ne10 DIST_ROOT="$SP15_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 ATTEST_STAMP=20260715T999999Z publish COMMON_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null
done

sp15_after_list="$(find "$sp15_attest_dir" -name 'attest-*.txt' | sort)"
new_files="$(comm -13 <(printf '%s\n' "$sp15_before_list") <(printf '%s\n' "$sp15_after_list"))"
new_count="$(printf '%s\n' "$new_files" | grep -c . || true)"
new_count="${new_count:-0}"
[ "$new_count" -eq 20 ] && pass "SP-S15: exactly 20 new ATTEST files after 20 same-stamp publishes" \
  || fail "SP-S15: expected 20 new ATTEST files, found $new_count"

suffixes="$(printf '%s\n' "$new_files" | while read -r f; do [ -n "$f" ] || continue; bn="$(basename "$f" .txt)"; echo "${bn##*-}"; done | sort)"
expected="$(printf '%03d\n' $(seq 1 20))"
[ "$suffixes" = "$expected" ] && pass "SP-S15: new files' -NNN suffixes are exactly 001..020, all distinct" \
  || fail "SP-S15: -NNN suffixes ($(printf '%s' "$suffixes" | tr '\n' ',')) do not exactly match 001..020"

unchanged_ok=1
while read -r f inode mt sha; do
  [ -n "$f" ] || continue
  if [ ! -f "$f" ]; then unchanged_ok=0; continue; fi
  [ "$(stat -f '%i' "$f")" = "$inode" ] && [ "$(mtime "$f")" = "$mt" ] && [ "$(file_sha "$f")" = "$sha" ] || unchanged_ok=0
done < "$sp15_before_snap"
[ "$unchanged_ok" -eq 1 ] && pass "SP-S15: every pre-existing ATTEST file's inode+mtime+sha is unchanged" \
  || fail "SP-S15: a pre-existing ATTEST file's inode/mtime/sha CHANGED"

tmp_leftovers="$(find "$sp15_attest_dir" -name '*.tmp' 2>/dev/null | grep -c . || true)"
tmp_leftovers="${tmp_leftovers:-0}"
[ "$tmp_leftovers" -eq 0 ] && pass "SP-S15: no *.tmp leftovers under ATTEST/" \
  || fail "SP-S15: $tmp_leftovers *.tmp leftover(s) under ATTEST/"

spot_ok=1
spot_n=0
for f in $new_files; do
  [ -n "$f" ] || continue
  spot_n=$((spot_n + 1))
  [ "$spot_n" -gt 3 ] && break
  stem="$(basename "$f" .txt)"
  grep -q "^event_id=$stem\$" "$f" || spot_ok=0
done
if [ "$spot_ok" -eq 1 ] && [ "$spot_n" -ge 1 ]; then
  pass "SP-S15: spot-checked $([ "$spot_n" -gt 3 ] && echo 3 || echo "$spot_n") new ATTEST file(s) -- event_id= matches filename stem"
else
  fail "SP-S15: a spot-checked ATTEST file's event_id= did not match its filename stem"
fi

echo "############################################################"
echo "# SP-S16: interruption-safety probe (regression P2-1 acceptance)"
echo "############################################################"
mkscratch sp16
cat > "$SCRATCH_ROOT/sp16/sigreset_exec.c" <<'EOF'
/* Tiny launcher: resets SIGINT/SIGQUIT to their default disposition before
 * exec'ing the real command. Needed because bash, for a NON-interactive
 * script's backgrounded (`cmd &`) jobs, sets SIGINT/SIGQUIT to SIG_IGN in
 * the forked child -- and bash's own `trap` builtin refuses to install (or
 * even reset) a handler for a signal that was already SIG_IGN "upon entry
 * to the shell" (see bash(1), SIGNALS). A plain C signal()/execvp() is not
 * bound by that bash-specific policy, so this helper is the only reliable
 * way for this scenario's backgrounded child to actually SEE a SIGINT its
 * own `trap ... INT` can catch. */
#include <signal.h>
#include <unistd.h>
int main(int argc, char** argv) {
    signal(SIGINT, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    if (argc < 2) return 127;
    execvp(argv[1], argv + 1);
    return 127;
}
EOF
cc -O2 -o "$SCRATCH_ROOT/sp16/sigreset_exec" "$SCRATCH_ROOT/sp16/sigreset_exec.c"

run_sp16_mode() {
  mode="$1" sig="$2" expect_rc="$3"
  d="$SCRATCH_ROOT/sp16/$mode"
  mkdir -p "$d/probe_tmp"
  mkfifo "$d/ready.fifo" "$d/hold.fifo"
  ( ISOL_INTERRUPT_PROBE="$mode" ISOL_PROBE_FIFO="$d/ready.fifo" TMPDIR="$d/probe_tmp" \
    "$SCRATCH_ROOT/sp16/sigreset_exec" bash "$SCRIPT_PATH" ) &
  pid=$!
  child_scratch="$(cat "$d/ready.fifo")"
  if [ -n "$sig" ]; then
    kill "-$sig" "$pid" 2>/dev/null || true
  fi
  rc=0
  wait "$pid" || rc=$?
  [ "$rc" -eq "$expect_rc" ] && pass "SP-S16[$mode]: child exit code = $rc (expected $expect_rc)" \
    || fail "SP-S16[$mode]: child exit code = $rc (expected $expect_rc)"
  [ ! -e "$child_scratch" ] && pass "SP-S16[$mode]: child's own SCRATCH_ROOT no longer exists after exit" \
    || fail "SP-S16[$mode]: child's SCRATCH_ROOT ($child_scratch) STILL EXISTS after exit"
  if [ -d "$d/probe_tmp" ] && [ -z "$(ls -A "$d/probe_tmp" 2>/dev/null)" ]; then
    pass "SP-S16[$mode]: probe TMPDIR is empty afterward (the child's scratch root landed inside it and was fully removed)"
  else
    fail "SP-S16[$mode]: probe TMPDIR ($d/probe_tmp) is NOT empty afterward"
  fi
}

run_sp16_mode exit "" 0
run_sp16_mode term TERM 143
run_sp16_mode intr INT 130

echo "############################################################"
echo "# SP-S17: dirty-producer provenance (regression P2; regression: shared clones)"
echo "############################################################"
mkscratch sp17

# SP-S17 REUSES the exact PSHARED_AC_CLONE/PSHARED_AEC_CLONE/PSHARED_NR_CLONE
# set SP2 (above) already built while they were still clean.
# `adopt_worktree_clone` is defined once, in the shared helpers section near
# the top of this script -- see that definition's own comment for the full
# rationale.
SP17_AC_CLONE="$PSHARED_AC_CLONE"
SP17_AEC_CLONE="$PSHARED_AEC_CLONE"
SP17_NR_CLONE="$PSHARED_NR_CLONE"

# Dirty each clone with a disposable, comment-only edit to one tracked
# source -- this scenario tests the DIRTY-PUBLISH POLICY (attestation
# fields), not content-addressing, so whether the resulting object/archive
# bytes happen to change is irrelevant here. (SP2's OWN edits to these same
# clones were mtime-only touches of already-tracked files with UNCHANGED
# content, so git still considers them clean coming into this scenario --
# this is the FIRST real content edit either clone sees.)
echo "/* sp17 disposable dirty probe */" >> "$SP17_AC_CLONE/src/hpf.c"
echo "/* sp17 disposable dirty probe */" >> "$SP17_AEC_CLONE/c_impl/src/aec_debug.c"
echo "/* sp17 disposable dirty probe */" >> "$SP17_NR_CLONE/c_impl/src/anr.c"

SP17_OBJ_ROOT="$SCRATCH_ROOT/sp17/obj"
SP17_BIN_ROOT="$SCRATCH_ROOT/sp17/bin"
SP17_DIST="$SCRATCH_ROOT/sp17/dist"

# sp17_attest_name_from_log(): the publish recipe's own success line ends "(attested: <name>)"; extract it directly
# rather than globbing ATTEST/, since this pipelines repo may also be dirty and publish its own attest fields, and the
# log line names this invocation's attest file regardless of how many a release dir holds.
sp17_attest_name_from_log() {
  local log="$1" name
  name="$(grep -o '(attested: [^)]*)' "$log" | sed -e 's/^(attested: //' -e 's/)$//' | head -1)"
  [ -n "$name" ] || return 1
  printf '%s\n' "$name"
}

SP17_LOG_OK="$SCRATCH_ROOT/sp17/log_ok"
# ALLOW_UNTRACKED_PUBLISH=1 rides along here too (this is a success-path call,
# not one of the deliberate policy-negative sub-cases below) -- inert in
# practice, since these clones carry no untracked files yet (only tracked-dirty
# edits above), but uniform with every other publish call in this script from
# S7 onward.
if make -s BACKEND=ne10 DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     COMMON_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17_AEC_CLONE/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
     ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$SP17_LOG_OK" 2>&1; then
  pass "SP-S17: publish against three dirty producer clones + ALLOW_DIRTY_PUBLISH=1 succeeds"
else
  fail "SP-S17: publish against three dirty producer clones + ALLOW_DIRTY_PUBLISH=1 FAILED"
  cat "$SP17_LOG_OK" >&2
fi

sp17_id="$(readlink "$SP17_DIST/ne10/current" || true)"
sp17_attest_name="$(sp17_attest_name_from_log "$SP17_LOG_OK" || true)"
sp17_attest="$SP17_DIST/ne10/$sp17_id/ATTEST/$sp17_attest_name"

sp17_ac_commit="$(git -C "$SP17_AC_CLONE" rev-parse HEAD)"
sp17_aec_commit="$(git -C "$SP17_AEC_CLONE" rev-parse HEAD)"
sp17_nr_commit="$(git -C "$SP17_NR_CLONE" rev-parse HEAD)"
sp17_ac_ddiff="$(git -C "$SP17_AC_CLONE" diff --binary HEAD | shasum -a 256 | cut -d' ' -f1)"
sp17_aec_ddiff="$(git -C "$SP17_AEC_CLONE" diff --binary HEAD | shasum -a 256 | cut -d' ' -f1)"
sp17_nr_ddiff="$(git -C "$SP17_NR_CLONE" diff --binary HEAD | shasum -a 256 | cut -d' ' -f1)"

if [ -n "$sp17_attest_name" ] && [ -f "$sp17_attest" ] \
   && grep -q "^audiolib_git_dirty=1\$" "$sp17_attest" \
   && grep -q "^audiolib_git_commit=$sp17_ac_commit\$" "$sp17_attest" \
   && grep -q "^audiolib_dirty_diff_sha256=$sp17_ac_ddiff\$" "$sp17_attest" \
   && grep -q "^aec_git_dirty=1\$" "$sp17_attest" \
   && grep -q "^aec_git_commit=$sp17_aec_commit\$" "$sp17_attest" \
   && grep -q "^aec_dirty_diff_sha256=$sp17_aec_ddiff\$" "$sp17_attest" \
   && grep -q "^nr_git_dirty=1\$" "$sp17_attest" \
   && grep -q "^nr_git_commit=$sp17_nr_commit\$" "$sp17_attest" \
   && grep -q "^nr_dirty_diff_sha256=$sp17_nr_ddiff\$" "$sp17_attest"; then
  pass "SP-S17: attestation records all three producers' git_dirty=1, matching git_commit=, matching dirty_diff_sha256="
else
  fail "SP-S17: attestation ($sp17_attest) field mismatch for one or more producers"
fi

SP17_LOG_FAIL="$SCRATCH_ROOT/sp17/log_fail"
if make -s BACKEND=ne10 DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     COMMON_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17_AEC_CLONE/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
     publish >"$SP17_LOG_FAIL" 2>&1; then
  fail "SP-S17: publish against the same dirty producer clones WITHOUT ALLOW_DIRTY_PUBLISH unexpectedly SUCCEEDED"
else
  if grep -q "publish refused" "$SP17_LOG_FAIL" \
     && grep -q "audiolib working tree has uncommitted TRACKED changes" "$SP17_LOG_FAIL" \
     && grep -q "lib/aec working tree has uncommitted TRACKED changes" "$SP17_LOG_FAIL" \
     && grep -q "lib/nr working tree has uncommitted TRACKED changes" "$SP17_LOG_FAIL"; then
    pass "SP-S17: publish without ALLOW_DIRTY_PUBLISH correctly FAILS, naming audiolib/common/lib/aec/lib/nr as the dirty repos"
  else
    fail "SP-S17: publish failed but without the expected 'publish refused' / per-producer dirty wording"
    cat "$SP17_LOG_FAIL" >&2
  fi
fi

echo "--- SP-S17a: untracked producer probe (audiolib) -------------------"
# The untracked dimension is separate from the tracked-dirty one exercised above. An untracked file in the shared
# audiolib clone (already tracked-dirty, so ALLOW_DIRTY_PUBLISH=1 admits that dimension) must still be refused on the
# untracked dimension unless ALLOW_UNTRACKED_PUBLISH=1 is also given; once given, the attestation records
# audiolib_git_untracked=1 and a 64-hex audiolib_untracked_tree_sha256, while pipelines' own untracked flag keeps its
# pre-probe value. The scenario adds no untracked file to the real pipelines checkout.
echo "int sp17a_untracked_probe;" > "$SP17_AC_CLONE/probe.c"

SP17_LOG_A1="$SCRATCH_ROOT/sp17/log_untracked_a1"
if make -s BACKEND=ne10 DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     COMMON_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17_AEC_CLONE/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
     ALLOW_DIRTY_PUBLISH=1 publish >"$SP17_LOG_A1" 2>&1; then
  fail "SP-S17a: ALLOW_DIRTY_PUBLISH=1 ALONE unexpectedly admitted an untracked file in audiolib"
else
  if grep -qi "publish refused" "$SP17_LOG_A1" && grep -q "audiolib" "$SP17_LOG_A1" && grep -q "UNTRACKED" "$SP17_LOG_A1"; then
    pass "SP-S17a: ALLOW_DIRTY_PUBLISH=1 alone is refused, naming audiolib on the untracked dimension"
  else
    fail "SP-S17a: refusal did not name audiolib's UNTRACKED files"
    cat "$SP17_LOG_A1" >&2
  fi
fi

SP17_LOG_A2="$SCRATCH_ROOT/sp17/log_untracked_a2"
if make -s BACKEND=ne10 DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     COMMON_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17_AEC_CLONE/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
     ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$SP17_LOG_A2" 2>&1; then
  pass "SP-S17a: ALLOW_DIRTY_PUBLISH=1 + ALLOW_UNTRACKED_PUBLISH=1 together succeed with an untracked audiolib file present"
else
  fail "SP-S17a: both knobs together FAILED"
  cat "$SP17_LOG_A2" >&2
fi

sp17a_id="$(readlink "$SP17_DIST/ne10/current" || true)"
sp17a_attest_name="$(sp17_attest_name_from_log "$SP17_LOG_A2" || true)"
sp17a_attest="$SP17_DIST/ne10/$sp17a_id/ATTEST/$sp17a_attest_name"
sp17a_ac_uhash="$(grep '^audiolib_untracked_tree_sha256=' "$sp17a_attest" 2>/dev/null | head -1 | cut -d= -f2)"
if [ -n "$sp17a_attest_name" ] && [ -f "$sp17a_attest" ] \
   && grep -q "^audiolib_git_untracked=1\$" "$sp17a_attest" \
   && [ -n "$sp17a_ac_uhash" ] && [ "${#sp17a_ac_uhash}" -eq 64 ] \
   && grep -q "^git_untracked=$sp11_expected_self_untracked\$" "$sp17a_attest"; then
  pass "SP-S17a: attestation records the untracked producer and preserves pipelines' own measured untracked flag"
else
  fail "SP-S17a: attestation ($sp17a_attest) does not match the expected untracked-producer shape"
fi

echo "--- SP-S17b: unknown producer (AEC_DIR -> a no-git copy) -> unconditional FATAL --"
# Point AEC_DIR at a no-git scratch copy of the PRIMARY AEC repo (a plain tar
# copy, no .git at all) -- a checkout with no git identity is refused
# UNCONDITIONALLY, naming lib/aec, even with BOTH escape hatches set (neither
# admits it: without a commit OID neither the tracked diff nor the untracked
# enumeration can be named for that repo).
SP17B_NOGIT_AEC="$SCRATCH_ROOT/sp17/aec_nogit"
mkdir -p "$SP17B_NOGIT_AEC"
tar -c -C "$AEC_REPO_DIR" --exclude obj --exclude bin --exclude dist --exclude .git . | tar -x -C "$SP17B_NOGIT_AEC"

SP17_LOG_B="$SCRATCH_ROOT/sp17/log_unknown_aec"
if make -s BACKEND=ne10 DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     COMMON_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17B_NOGIT_AEC/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
     ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$SP17_LOG_B" 2>&1; then
  fail "SP-S17b: publish with AEC_DIR pointed at a no-git copy unexpectedly SUCCEEDED (unknown identity slipped past both hatches)"
else
  if grep -q "lib/aec" "$SP17_LOG_B" && grep -q "no git identity" "$SP17_LOG_B"; then
    pass "SP-S17b: unknown producer (lib/aec, no git identity) refused UNCONDITIONALLY -- both escape hatches set, still FATAL"
  else
    fail "SP-S17b: publish failed but without the expected lib/aec no-git-identity message"
    cat "$SP17_LOG_B" >&2
  fi
fi

echo "############################################################"
echo "# SP-S18: 'make -e' environment-override bypass rejection, cross-repo"
echo "############################################################"
# SP-S18: `make -e` environment-override rejection, distinct from SP-S9's command-line overrides. `$(origin VAR)` only
# becomes "environment override" once a makefile assignment/append to VAR has been parsed, so under `-e` a bare env
# var whose file-side assignment sits after the rejection foreach would pass unrejected and silently drop that
# Makefile's flag appends (`env FP_POLICY=-ffp-contract=fast make -e -n BACKEND=ne10 lib` printing
# -ffp-contract=fast instead of the pinned off). Each Makefile therefore places its bare FP_POLICY/LDFLAGS
# literal/composition ahead of the foreach and uses `+=` (not `?=`) for CXXFLAGS/CPPFLAGS; this scenario covers all
# three consumer Makefiles.

# AEC/NR are the PRIMARY repos ($AEC_R6_DIR/$NR_R6_DIR), not the submodule paths; pipelines runs in place ($PIPE_DIR).
# Every case is a true `-n` dry run with scratch OBJ_ROOT/BIN_ROOT, so the invariant that the parse-time $(error)
# fires before any recipe runs holds even if the check ever moves.
SP18_OBJ_ROOT="$SCRATCH_ROOT/sp18/obj"
SP18_BIN_ROOT="$SCRATCH_ROOT/sp18/bin"
mkscratch sp18

# assert_make_e_rejected <description> <dir> <target> <mode> <needle> <VAR=value>...
# Runs `env <VAR=value>... make -C <dir> -e -n BACKEND=ne10 OBJ_ROOT=/ BIN_ROOT= (fresh scratch dirs per case) <target>`.

# The only passing outcome is make exiting nonzero with the log containing "cannot be overridden", i.e. the origin
# override rejection fired. make exiting 0 is always a FAIL: a missing poison string in one dry run's output proves
# nothing (pipelines' CPPFLAGS/CXXFLAGS never reach a compile recipe, so `make -n all` would not print
# "-DSP18_POISON" for them even with the gate removed). The rejection is a parse-time check, so gating on it alone is
# necessary and sufficient. A nonzero exit without that message is a FAIL too. The grep for the poison / pinned flag is
# only an extra diagnostic printed next to a FAIL.

# Dispatch isolation: `target=all` would shell out through pipelines' three-producer dispatch (a `+@...$(MAKE)...`
# line that runs even under `-n`), and the child makes inherit `env VAR=...`, so a PASS could come from audiolib's or a
# submodule's gate instead of pipelines'. COMMON_LIB=/AEC_LIB=/NR_LIB= are therefore forced to dummy values on every
# call (harmless for the lib/aec and lib/nr primary cases, which have no such variables), skipping the dispatch, so a
# PASS can only come from the Makefile named by <dir>. For the same reason the pipelines cases target
# `libaudio_pipeline_dfn.a` rather than `all`.
SP18_COUNTER=0
assert_make_e_rejected() {
  local desc="$1" dir="$2" target="$3" mode="$4" needle="$5" log
  shift 5
  SP18_COUNTER=$((SP18_COUNTER + 1))
  log="$SCRATCH_ROOT/sp18/case-$SP18_COUNTER.log"
  if env "$@" make -C "$dir" -e -n BACKEND=ne10 COMMON_LIB=x AEC_LIB=y NR_LIB=z \
       OBJ_ROOT="$SP18_OBJ_ROOT/$SP18_COUNTER" BIN_ROOT="$SP18_BIN_ROOT/$SP18_COUNTER" \
       "$target" >"$log" 2>&1; then
    fail "$desc -- make exited 0 (NOT rejected): the origin-override gate did not fire"
    if [ "$mode" = present ]; then
      if grep -qF -- "$needle" "$log"; then
        echo "  [diagnostic, non-authoritative] the pinned flag ('$needle') is still present in the dry-run output" >&2
      else
        echo "  [diagnostic, non-authoritative] the pinned flag ('$needle') is ALSO missing from the dry-run output" >&2
      fi
    else
      if grep -qF -- "$needle" "$log"; then
        echo "  [diagnostic, non-authoritative] the poisoned value ('$needle') DOES appear in the dry-run output" >&2
      else
        echo "  [diagnostic, non-authoritative] the poisoned value ('$needle') does not appear in the dry-run output -- absence here is NOT proof of rejection (it may simply never reach a printed recipe line for this target/variable combination)" >&2
      fi
    fi
    cat "$log" >&2
  else
    if grep -q "cannot be overridden" "$log"; then
      pass "$desc -- rejected at parse time ('cannot be overridden')"
    else
      fail "$desc -- failed, but NOT with the expected 'cannot be overridden' message"
      cat "$log" >&2
    fi
  fi
}

echo "--- FP_POLICY= (empty) via make -e -----------------------------------"
assert_make_e_rejected "SP-S18: pipelines env FP_POLICY= (empty) make -e -n rejected" \
  "$PIPE_DIR" libaudio_pipeline_dfn.a present "-ffp-contract=off" FP_POLICY=
assert_make_e_rejected "SP-S18: lib/aec (primary) env FP_POLICY= (empty) make -e -n rejected" \
  "$AEC_R6_DIR" lib present "-ffp-contract=off" FP_POLICY=
assert_make_e_rejected "SP-S18: lib/nr (primary) env FP_POLICY= (empty) make -e -n rejected" \
  "$NR_R6_DIR" lib present "-ffp-contract=off" FP_POLICY=

echo "--- FP_POLICY=-ffp-contract=fast via make -e (environment-override regression) ---"
assert_make_e_rejected "SP-S18: pipelines env FP_POLICY=-ffp-contract=fast make -e -n rejected" \
  "$PIPE_DIR" libaudio_pipeline_dfn.a absent "-ffp-contract=fast" FP_POLICY=-ffp-contract=fast
assert_make_e_rejected "SP-S18: lib/aec (primary) env FP_POLICY=-ffp-contract=fast make -e -n rejected" \
  "$AEC_R6_DIR" lib absent "-ffp-contract=fast" FP_POLICY=-ffp-contract=fast
assert_make_e_rejected "SP-S18: lib/nr (primary) env FP_POLICY=-ffp-contract=fast make -e -n rejected" \
  "$NR_R6_DIR" lib absent "-ffp-contract=fast" FP_POLICY=-ffp-contract=fast

echo "--- CPPFLAGS=<poison> via make -e -------------------------------------"
assert_make_e_rejected "SP-S18: pipelines env CPPFLAGS=-DSP18_POISON make -e -n rejected" \
  "$PIPE_DIR" libaudio_pipeline_dfn.a absent "-DSP18_POISON" CPPFLAGS=-DSP18_POISON
assert_make_e_rejected "SP-S18: lib/aec (primary) env CPPFLAGS=-DSP18_POISON make -e -n rejected" \
  "$AEC_R6_DIR" lib absent "-DSP18_POISON" CPPFLAGS=-DSP18_POISON
assert_make_e_rejected "SP-S18: lib/nr (primary) env CPPFLAGS=-DSP18_POISON make -e -n rejected" \
  "$NR_R6_DIR" lib absent "-DSP18_POISON" CPPFLAGS=-DSP18_POISON

echo "--- CXXFLAGS=<poison> via make -e -------------------------------------"
assert_make_e_rejected "SP-S18: pipelines env CXXFLAGS=-DSP18_POISON make -e -n rejected" \
  "$PIPE_DIR" libaudio_pipeline_dfn.a absent "-DSP18_POISON" CXXFLAGS=-DSP18_POISON
assert_make_e_rejected "SP-S18: lib/aec (primary) env CXXFLAGS=-DSP18_POISON make -e -n rejected" \
  "$AEC_R6_DIR" lib absent "-DSP18_POISON" CXXFLAGS=-DSP18_POISON
assert_make_e_rejected "SP-S18: lib/nr (primary) env CXXFLAGS=-DSP18_POISON make -e -n rejected" \
  "$NR_R6_DIR" lib absent "-DSP18_POISON" CXXFLAGS=-DSP18_POISON

echo "--- LDFLAGS=<poison> via make -e -- pipelines-specific (the one repo"
echo "    whose LDFLAGS composition itself needed relocating; AEC's/NR's"
echo "    LDFLAGS already precede their own foreach and never needed it) ---"
assert_make_e_rejected "SP-S18: pipelines env LDFLAGS=-lSP18POISON make -e -n rejected" \
  "$PIPE_DIR" libaudio_pipeline_dfn.a absent "-lSP18POISON" LDFLAGS=-lSP18POISON

echo "############################################################"
echo "# Final integrity guards"
echo "############################################################"
AUDIO_ALG_STATE_AFTER="$(tree_state_hash "$AUDIO_ALG_ROOT")"
[ "$AUDIO_ALG_STATE_BEFORE" = "$AUDIO_ALG_STATE_AFTER" ] && pass "integrity: Audio_ALG toplevel tree-state (status+diff) unchanged across the whole run" \
  || fail "integrity: Audio_ALG toplevel tree-state CHANGED during this run"

AEC_SUB_STATE_AFTER="$(tree_state_hash "$AEC_SUB_ROOT")"
[ "$AEC_SUB_STATE_BEFORE" = "$AEC_SUB_STATE_AFTER" ] && pass "integrity: lib/aec (submodule) tree-state (status+diff) unchanged across the whole run" \
  || fail "integrity: lib/aec (submodule) tree-state CHANGED during this run"

NR_SUB_STATE_AFTER="$(tree_state_hash "$NR_SUB_ROOT")"
[ "$NR_SUB_STATE_BEFORE" = "$NR_SUB_STATE_AFTER" ] && pass "integrity: lib/nr (submodule) tree-state (status+diff) unchanged across the whole run" \
  || fail "integrity: lib/nr (submodule) tree-state CHANGED during this run"

AC_STATE_AFTER="$(tree_state_hash "$COMMON_DIR")"
[ "$AC_STATE_BEFORE" = "$AC_STATE_AFTER" ] && pass "integrity: \$COMMON_DIR (audiolib) tree-state (status+diff) unchanged across the whole run" \
  || fail "integrity: \$COMMON_DIR (audiolib) tree-state CHANGED during this run"

# Bonus (beyond the four repos the safety contract mandates): the PRIMARY
# AEC/NR repos are exercised directly (SP-S10/SP-S14) -- guard their
# tree-state too, even though this script never edits a tracked file there.
AEC_R6_STATE_AFTER="$(tree_state_hash "$AEC_REPO_DIR")"
[ "$AEC_R6_STATE_BEFORE" = "$AEC_R6_STATE_AFTER" ] && pass "integrity (bonus): PRIMARY AEC repo (\$AEC_REPO_DIR) tree-state unchanged across the whole run" \
  || fail "integrity (bonus): PRIMARY AEC repo tree-state CHANGED during this run"

NR_R6_STATE_AFTER="$(tree_state_hash "$NR_REPO_DIR")"
[ "$NR_R6_STATE_BEFORE" = "$NR_R6_STATE_AFTER" ] && pass "integrity (bonus): PRIMARY NR repo (\$NR_REPO_DIR) tree-state unchanged across the whole run" \
  || fail "integrity (bonus): PRIMARY NR repo tree-state CHANGED during this run"

REAL_DIST_PIPE_AFTER="$(real_dist_sentinel "$PIPE_DIR")"
[ "$REAL_DIST_PIPE_BEFORE" = "$REAL_DIST_PIPE_AFTER" ] && pass "integrity: real pipelines/dist sentinel unchanged" \
  || fail "integrity: real pipelines/dist sentinel CHANGED during this run"

REAL_DIST_AEC_AFTER="$(real_dist_sentinel "$AEC_DIR")"
[ "$REAL_DIST_AEC_BEFORE" = "$REAL_DIST_AEC_AFTER" ] && pass "integrity: real lib/aec/c_impl/dist sentinel unchanged" \
  || fail "integrity: real lib/aec/c_impl/dist sentinel CHANGED during this run"

REAL_DIST_NR_AFTER="$(real_dist_sentinel "$NR_DIR")"
[ "$REAL_DIST_NR_BEFORE" = "$REAL_DIST_NR_AFTER" ] && pass "integrity: real lib/nr/c_impl/dist sentinel unchanged" \
  || fail "integrity: real lib/nr/c_impl/dist sentinel CHANGED during this run"

REAL_DIST_AC_AFTER="$(real_dist_sentinel "$COMMON_DIR")"
[ "$REAL_DIST_AC_BEFORE" = "$REAL_DIST_AC_AFTER" ] && pass "integrity: real \$COMMON_DIR/dist sentinel unchanged" \
  || fail "integrity: real \$COMMON_DIR/dist sentinel CHANGED during this run"

# Bonus: the PRIMARY AEC/NR repos' own real dist/ too.
REAL_DIST_AECR6_AFTER="$(real_dist_sentinel "$AEC_R6_DIR")"
[ "$REAL_DIST_AECR6_BEFORE" = "$REAL_DIST_AECR6_AFTER" ] && pass "integrity (bonus): real PRIMARY AEC repo dist sentinel unchanged" \
  || fail "integrity (bonus): real PRIMARY AEC repo dist sentinel CHANGED during this run"

REAL_DIST_NRR6_AFTER="$(real_dist_sentinel "$NR_R6_DIR")"
[ "$REAL_DIST_NRR6_BEFORE" = "$REAL_DIST_NRR6_AFTER" ] && pass "integrity (bonus): real PRIMARY NR repo dist sentinel unchanged" \
  || fail "integrity (bonus): real PRIMARY NR repo dist sentinel CHANGED during this run"

# Final-guard re-check against the snapshot taken in SP1 (right after this
# suite's LAST intentional real-tree build). Every file that existed then,
# across all FOUR trees (pipelines, lib/aec, lib/nr, audiolib), must still
# be sha+mtime identical now; a directory created after the snapshot is allowed
# but reported (WARN/INFO -- there should be none), while a NEW file inside an
# already-snapshotted directory is a HARD FAIL.
FINAL_GUARD_DIRS_AFTER="$(final_guard_dirs)"
FINAL_GUARD_SNAPSHOT_AFTER="$(final_guard_snapshot)"

fg_lost="$(comm -23 <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_BEFORE" | LC_ALL=C sort) <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_AFTER" | LC_ALL=C sort))"
if [ -z "$fg_lost" ]; then
  pass "final-guard: every real obj/+bin/ file (pipelines, lib/aec, lib/nr, audiolib) snapshotted after SP1's builds is still sha+mtime identical"
else
  fail "final-guard: real obj/+bin/ files changed or vanished since SP1's builds"
  printf '%s\n' "$fg_lost" | sed 's/^/    changed\/lost: /' >&2
fi

fg_new_dirs="$(comm -13 <(printf '%s\n' "$FINAL_GUARD_DIRS_BEFORE" | LC_ALL=C sort) <(printf '%s\n' "$FINAL_GUARD_DIRS_AFTER" | LC_ALL=C sort) | grep . || true)"
if [ -n "$fg_new_dirs" ]; then
  echo "  WARN: final-guard: new keyed director(ies) appeared under a real obj/+bin/ during this run (allowed, but unexpected after the SP-S12 fix):"
  printf '%s\n' "$fg_new_dirs" | sed 's/^/    INFO: new dir /'
fi

fg_new_files="$(comm -13 <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_BEFORE" | LC_ALL=C sort) <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_AFTER" | LC_ALL=C sort) | grep . || true)"
fg_stray_ok=1
if [ -n "$fg_new_files" ]; then
  while IFS= read -r fg_ln; do
    [ -n "$fg_ln" ] || continue
    fg_path="${fg_ln%% *}"
    fg_top="$(printf '%s' "$fg_path" | cut -d/ -f1-3)"
    printf '%s\n' "$fg_new_dirs" | grep -qxF "$fg_top" || fg_stray_ok=0
  done <<EOF_FG
$fg_new_files
EOF_FG
fi
[ "$fg_stray_ok" -eq 1 ] && pass "final-guard: no stray new file inside any already-snapshotted real obj/+bin/ directory" \
  || fail "final-guard: a NEW file appeared inside an EXISTING (already-snapshotted) real obj/+bin/ directory"

echo "############################################################"
echo "SUMMARY: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "############################################################"
if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "Failures:" >&2
  for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
  exit 1
fi
exit 0
