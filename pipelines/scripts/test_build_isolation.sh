#!/usr/bin/env bash
# pipelines/scripts/test_build_isolation.sh -- round-3/4/5/6/7 review
# build-isolation regression suite for Audio_ALG/pipelines, the fourth (and
# last) of four repos to get the CFG_SIG-keyed obj/bin directory design
# (audio_common, AEC, NR already have their own scripts/test_build_isolation.sh
# -- see audio_common's for the S1-S5/S7p/S8 single-producer scenarios this
# one does NOT repeat).
#
# pipelines/Makefile is a THREE-producer consumer (audio_common's
# libaudio_common.a, lib/aec's libaec.a, lib/nr's libmmse_lsa.a), each
# resolved via the two-phase recursive-make dispatch described in that
# Makefile's own header comment. This script exercises the scenarios specific
# to that three-producer design plus this repo's own audit-no-stdio/publish
# gates, the round-5 RNNoise-ERB table drift-gate hardening, and now the
# round-6 review:
#
#   - P1: OBJ_ROOT=/BIN_ROOT= placement knobs (pipelines' own Makefile, PLUS
#     the two PRIMARY producer repos AEC/c_impl and NR/c_impl -- see the
#     "round-6 submodule caveat" note below), so throwaway/tamper scenarios
#     drive a scratch-directory build of the real worktree without ever
#     touching the real obj/ or bin/.
#   - P2-1: this script's OWN temp management -- fixed SCRATCH_ROOT tree +
#     one trap, never a CLEANUP_DIRS array mutated from inside a `$(...)`
#     command substitution (a subshell whose variable changes never make it
#     back to the parent shell that owns the EXIT trap -- the actual round-5
#     bug class this replaces).
#   - P2-2: ATTEST is one-event-one-file
#     (attest-<utc>-<commit>[-dirty]-<seq>.txt, installed via the
#     atomic_symlink_swap helper's `--excl-install` mode).
#   - P2-3: `make -n`/`-q`/`-t publish` must have ZERO filesystem side effects.
#   - P2 (dirty policy): publish FATALs by default on a dirty/no-git-identity
#     checkout (this repo OR any of the three producers); ALLOW_DIRTY_PUBLISH=1
#     is the recorded escape hatch.
#
# ...and now the round-7 review (mirrors audio_common's own round-7 rewrite;
# see that script for the full rationale on each point -- reused as directly
# as possible here, generalized from ONE producer to the three this repo
# resolves plus this repo's own tree):
#
#   - Cross-suite mutex: a SUITE_LOCK under `${TMPDIR:-/tmp}/
#     audio_dsp_isolation_suite.lock` -- the EXACT SAME lock path/name
#     audio_common's own script uses -- refuses to let two instances of
#     EITHER suite run concurrently (both assert real-tree invariants a
#     second concurrent run would trample). mkdir-based, holder-pid file,
#     single TOCTOU-safe stale-reclaim retry, released in cleanup() only by
#     the process that actually acquired it. Placed AFTER the
#     ISOL_INTERRUPT_PROBE early-return (a probe child never contends for
#     its own parent's lock) and BEFORE `export TMPDIR=...` (so it resolves
#     against the INVOKING environment's TMPDIR, not the scratch-local one
#     this script is about to export over it).
#   - Final-guard real-tree snapshot: once this suite's OWN intended
#     real-tree builds are done (captured right after SP1's kiss+ne10
#     builds), a per-file path+sha256+fractional-mtime snapshot of every
#     keyed directory already under the REAL obj/ and bin/ of the FOUR trees
#     this suite builds in (pipelines itself, lib/aec, lib/nr, audio_common)
#     is re-verified byte-for-byte identical at the very end. A later-created
#     directory is allowed (INFO/WARN, never a FAIL) -- should be none, since
#     the one pre-existing leak (SP-S12's toolchain-shim probe, previously
#     landing in the REAL tree because its CFG_SIG-participating CXX shim
#     path is scratch-fresh every run) is fixed below to build under a
#     scratch OBJ_ROOT/BIN_ROOT too, same as SP-S10/SP-S14 already did.
#   - SP2 (producer-change propagation) no longer mtime-touches REAL
#     audio_common/lib/aec/lib/nr sources: it now runs against a disposable
#     clone EACH of audio_common, the PRIMARY AEC repo, and the PRIMARY NR
#     repo (via `adopt_worktree_clone()`), built via their own default
#     (clone-relative, therefore scratch) obj/bin, with the pipeline binary
#     itself built from the REAL pipelines directory but with
#     AC_DIR/AEC_DIR/NR_DIR pointed at the clones and an explicit scratch
#     OBJ_ROOT/BIN_ROOT. SP-S17 (below) REUSES this exact clone set (created
#     once, while still clean, by SP2) and dirties it further -- one clone
#     build instead of two separate ones.
#   - SP-S14 gains a COMBINED dry-run flag matrix (-nt/"-n -t"/-tq/-nq/-nqt)
#     against a deliberately BACKDATED scratch artifact of the PIPELINES
#     Makefile (the S15b pattern from audio_common's script) -- every combo
#     must leave the artifact's stat unchanged and create no
#     DIST_ROOT/OBJ_ROOT/BIN_ROOT.
#   - SP-S17 gains the untracked-file dimension at the pipelines level: an
#     untracked producer probe (ALLOW_DIRTY_PUBLISH=1 alone still refuses,
#     naming audio_common on the untracked dimension; +
#     ALLOW_UNTRACKED_PUBLISH=1 succeeds with audio_common_git_untracked=1 +
#     a 64-hex audio_common_untracked_tree_sha256, while pipelines' OWN
#     git_untracked stays 0 -- the ainr/GTCRN|gtcrn_github whitelist holding)
#     and an unknown-producer probe (AEC_DIR pointed at a no-git scratch
#     copy is refused UNCONDITIONALLY, naming lib/aec, even with BOTH knobs
#     set).
#   - ALLOW_UNTRACKED_PUBLISH=1 now rides alongside ALLOW_DIRTY_PUBLISH=1 on
#     every publish call in this script (the untracked dimension is a
#     SEPARATE default refusal from the tracked-dirty one -- see the
#     Makefile's own knob comment), EXCEPT SP-S17's own deliberate
#     policy-negative calls (which test a knob's absence on purpose).
#   - SP-S11's attest assertions are extended for the four *_git_untracked
#     fields (git_untracked/audio_common_git_untracked/aec_git_untracked/
#     nr_git_untracked) + allow_untracked_publish=.
#
# ROUND-6 SUBMODULE CAVEAT (read this before touching AEC/NR-facing scenarios):
#   lib/aec and lib/nr are checked out here as SUBMODULES still pinned at
#   their ROUND-5 commit (publish v4, no OBJ_ROOT=/BIN_ROOT=/
#   ALLOW_DIRTY_PUBLISH=/ATTEST_STAMP=) -- the round-6 Makefile edits for AEC
#   and NR live in the PRIMARY repos (../../AEC, ../../NR, siblings of
#   Audio_ALG) and will be synced into the submodules AFTER this task. Any
#   scenario that needs round-6-ONLY AEC/NR Makefile behavior therefore
#   targets $AEC_R6_DIR/$NR_R6_DIR (the primary repos' c_impl/, resolved
#   below) instead of the submodule $AEC_DIR/$NR_DIR -- once the submodule
#   pin is bumped post-task, $AEC_R6_DIR/$NR_R6_DIR and $AEC_DIR/$NR_DIR
#   become equivalent and this indirection stops mattering. Scenarios that
#   only need round-5-stable behavior (normal all/lib builds, print-lib-path)
#   keep using the submodule paths, matching how this script has always
#   worked.
#
# Scenario index:
#   S6:      audit-no-stdio false-pass regression
#   S7:      publish v4 -- lock-FIRST driver + concurrent-publish semantics
#   SP1:     pipeline-level A->B->A (kiss -> ne10 -> kiss)
#   SP2:     producer-change propagation (audio_common hpf.c / fast_math.h --
#            round-7: against disposable clones, never the real trees; see
#            the round-7 bullet list above)
#   SP-S9:   command-line override rejection (round-4 P1-1), all three
#            Makefiles (pipelines/lib-aec/lib-nr, submodule paths -- stable
#            since round-4, no round-6-only feature needed)
#   SP-S10:  lib/aec fresh-archive discipline (round-4 P1-4) -- round-6:
#            rewritten against the PRIMARY AEC repo ($AEC_R6_DIR) with
#            scratch OBJ_ROOT/BIN_ROOT (the round-5 version of this scenario
#            injected a foreign member directly into the REAL SUBMODULE
#            libaec.a -- a round-6 P1 finding this fixes)
#   SP-S11:  pipelines publish v4 (content-addressed release + ATTEST) --
#            round-6: attest v2 field/naming assertions, no sleep (the
#            <NNN> suffix disambiguates a same-second republish)
#   SP-S12:  BACKEND=ne10 CC/CXX toolchain-coherence guard through the dispatch
#   SP-S13:  RNNoise-ERB table drift-gate hardening (round-5 P2)
#   SP-S14:  make -n/-q/-t publish zero side effects, THREE Makefiles --
#            pipelines (in-place), $AEC_R6_DIR, $NR_R6_DIR (round-6 P2-3);
#            round-7 SP-S14b: COMBINED dry-run flags (-nt/"-n -t"/-tq/-nq/
#            -nqt) against a deliberately backdated scratch artifact of the
#            PIPELINES Makefile only (the S15b pattern)
#   SP-S15:  ATTEST uniqueness under forced same-second collisions at the
#            pipelines level (round-6 P2-2)
#   SP-S16:  interruption-safety probe -- EXIT/INT/TERM all clean up the
#            whole scratch tree (round-6 P2-1 acceptance test)
#   SP-S17:  dirty-producer provenance -- publish against three DIRTY
#            producer clones (audio_common/AEC/NR, round-7: the SAME clone
#            set SP2 built while clean), ALLOW_DIRTY_PUBLISH=1 path succeeds
#            with correct per-producer attest fields, the default (no
#            override) path FAILS "publish refused" (round-6 P2); round-7
#            SP-S17a/b: the untracked-file dimension (untracked producer
#            probe naming audio_common; unknown-producer AEC_DIR refused
#            unconditionally naming lib/aec)
#
# Design rules (same as audio_common's script -- do not violate when editing):
#   - No `make clean` inside any scenario body (except SP-S13's own trailing
#     `make -C RNNoise-ERB clean`, which only ever removes THAT repo's own
#     gitignored build/ -- never this repo's bin/obj): distinct configs must
#     coexist WITHOUT ever needing a clean between them.
#   - Every path is resolved via `make -s ... print-bin-dir` / `print-obj-dir`
#     / `print-lib-path`, using the EXACT flag set (INCLUDING OBJ_ROOT=/
#     BIN_ROOT= when the build under test used them) under test for that
#     call -- never a hand-reconstructed path guess.
#   - "Did this get rebuilt?" is an mtime comparison, never a content (sha)
#     comparison.
#   - "Is this the SAME delivered artifact as its own keyed object?" IS a sha
#     comparison, via file_sha() below.
#
# Round-7 safety contract (supersedes the round-6 version of this comment --
# mirrors audio_common's script, see that script for the full rationale on
# each point):
#   - ONE scratch root for the entire run: SCRATCH_ROOT="$(mktemp -d)",
#     removed by a single EXIT trap; every temp file/dir this script uses is
#     a FIXED path under "$SCRATCH_ROOT/<scenario>/...", created inline --
#     never inside a `$(...)` command substitution (registering cleanup
#     state from inside a subshell silently drops it -- the round-5 P2-1 bug
#     class this replaces the old CLEANUP_DIRS array with). TMPDIR is
#     exported into the scratch tree, with a `mktemp` PATH shim ahead of the
#     real one (macOS's bare `mktemp -d` resolves via
#     _CS_DARWIN_USER_TEMP_DIR FIRST, ignoring $TMPDIR, unless given an
#     explicit -p/template) -- so a child `make`'s own `work="$(mktemp -d)"`
#     (the publish recipe) lands under scratch too.
#   - A cross-suite mutex (SUITE_LOCK="${TMPDIR:-/tmp}/
#     audio_dsp_isolation_suite.lock" -- computed from the INVOKING
#     environment's TMPDIR, BEFORE this script's own TMPDIR export, and the
#     EXACT SAME lock path/name audio_common's own script uses) refuses to
#     let two instances of this suite -- or of the audio_common counterpart
#     -- run concurrently. mkdir-based, holder-pid file, a single
#     TOCTOU-safe stale-reclaim retry (a losing reclaimer never steals the
#     winner's fresh lock), released in cleanup() only by the process that
#     actually acquired it. Placed AFTER the ISOL_INTERRUPT_PROBE
#     early-return (below) so a probe CHILD never contends for its own
#     parent's lock.
#   - Real obj/ and bin/ (this repo's own, AND the submodule lib/aec's/
#     lib/nr's, AND audio_common's) only ever see the NORMAL kiss/ne10
#     configs S6/S7/SP1/SP-S9's real-tree assertions depend on (SP2, round-6's
#     other real-tree normal-config builder, now runs inside scratch clones
#     instead -- see below). Every throwaway/tamper scenario that needs a
#     scratch OBJ_ROOT/BIN_ROOT uses one (SP-S10 against $AEC_R6_DIR; SP-S12's
#     toolchain-shim probes, round-7: moved to scratch -- the CXX shim's
#     scratch-fresh path participates in CFG_SIG, so it used to mint a new
#     orphan keyed directory in the REAL obj/bin on every single run; SP-S14's
#     dry runs; SP-S17's producer clones, whose own obj/bin land under
#     SCRATCH_ROOT by construction, being inside the clone directories under
#     $SCRATCH_ROOT). A final-guard snapshot (path + sha256 + `stat -f %Fm`
#     per file, taken once SP1's kiss+ne10 builds finish, of every keyed
#     directory already under the real obj/+bin/ of the four trees this
#     suite builds in -- pipelines, lib/aec, lib/nr, audio_common) is
#     re-verified byte-for-byte identical at the very end; a later-created
#     directory is allowed (INFO/WARN) -- after the SP-S12 fix there should
#     be none.
#   - The real dist/ (pipelines' own, lib/aec's, lib/nr's, audio_common's,
#     and -- belt-and-braces -- the two PRIMARY AEC/NR repos' own) is never
#     read, written, or removed: every `make ... publish` passes an explicit
#     DIST_ROOT= under $SCRATCH_ROOT. A sentinel digest of each real dist/
#     (absent, or a full manifest of paths + sha256 + mtime) is captured at
#     the very start and re-checked at the very end.
#   - No git-tracked file, in $AC_DIR, the submodule lib/aec/lib/nr, the
#     Audio_ALG toplevel, OR the two PRIMARY AEC/NR repos, is EVER touched --
#     not its content, not even its mtime (round-7 removes the round-6
#     allowance for a handful of mtime-only touches on real tracked sources
#     entirely). Every mtime bump this script performs to force a recompile
#     probe (hpf.c, fast_math.h) now happens against a SCRATCH CLONE's copy
#     of that source (SP2, reused and further dirtied by SP-S17), or an
#     already clone-local file (SP-S17's untracked/unknown-producer probes).
#     `tree_state_hash()` (status --porcelain + diff --binary HEAD, not
#     status alone) is captured for the Audio_ALG toplevel, lib/aec, lib/nr,
#     and $AC_DIR (the four repos the task's own safety contract names) --
#     plus, belt-and-braces, the two PRIMARY AEC/NR repos this script also
#     builds directly against -- at the start and re-checked at the end,
#     asserting exact equality (not just "still dirty") -- any script bug
#     that touched a real tracked file's content OR mtime-affecting metadata
#     would still have to leave `git status --porcelain`/`git diff`
#     themselves unchanged to pass, which a content change cannot do.
#   - No `sleep` anywhere. SP1's "should NOT relink" checks rely on
#     mtime()'s BSD `stat -f '%Fm'` fractional-seconds read (two genuinely
#     different real writes are distinguishable without an artificial
#     delay); SP2's "should recompile" checks use a deterministic
#     `touch -r <the artifact that must become stale> -A 01 <source>` bump
#     (GNU Make 3.81 truncates its own prerequisite-newer-than-target check
#     to whole seconds, so the +1s bump is load-bearing, not just extra
#     caution); SP-S11's same-second republish is disambiguated by the
#     attest v2 <NNN> suffix, not a delay.
#   - Interruption-safe by construction: cleanup() is the ONLY EXIT-trap
#     resident, INT/TERM map to 130/143 (both routed back through the same
#     cleanup()) -- SP-S16 tests exactly this.
#   - Zero-write dry-runs: `make -n`/`-q`/`-t publish`, INCLUDING every
#     combined flag form (-nt/"-n -t"/-tq/-nq/-nqt -- SP-S14b), create no
#     DIST_ROOT/OBJ_ROOT/BIN_ROOT and leave a pre-existing artifact's stat
#     unchanged.
#
# Usage: ./scripts/test_build_isolation.sh   (run from pipelines/, or
# anywhere -- paths are resolved relative to this script's own location).
#
# ISOL_INTERRUPT_PROBE (internal, used only by SP-S16): when set, this
# script runs as a child re-invocation of itself in "interruption probe"
# mode instead of running the suite -- see the block immediately after the
# SCRATCH_ROOT/trap setup below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
PIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
AUDIO_ALG_ROOT="$(cd "$PIPE_DIR/.." && pwd)"
AC_DIR="$(cd "$PIPE_DIR/../../audio_common" && pwd)"
AEC_DIR="$(cd "$PIPE_DIR/../lib/aec/c_impl" && pwd)"
NR_DIR="$(cd "$PIPE_DIR/../lib/nr/c_impl" && pwd)"
AEC_SUB_ROOT="$(cd "$AEC_DIR/.." && pwd)"
NR_SUB_ROOT="$(cd "$NR_DIR/.." && pwd)"
RNN_DIR="$(cd "$PIPE_DIR/../ainr/RNNoise-ERB" && pwd)"

# Round-6 submodule caveat (see header comment above): lib/aec and lib/nr
# above are still pinned at round-5. The round-6 Makefile edits live in
# these PRIMARY sibling repos instead; SP-S10/SP-S14/SP-S17 specifically
# need round-6-only features (OBJ_ROOT=/BIN_ROOT=/ALLOW_DIRTY_PUBLISH=/
# ATTEST_STAMP=/the -n/-q/-t dry-run guard) and so target these instead of
# $AEC_DIR/$NR_DIR. Once the submodule pin is bumped post-task,
# $AEC_R6_DIR/$NR_R6_DIR and $AEC_DIR/$NR_DIR become equivalent and this
# indirection stops mattering.
AEC_REPO_DIR="$(cd "$PIPE_DIR/../../AEC" && pwd)"
AEC_R6_DIR="$AEC_REPO_DIR/c_impl"
NR_REPO_DIR="$(cd "$PIPE_DIR/../../NR" && pwd)"
NR_R6_DIR="$NR_REPO_DIR/c_impl"

# --- round-6 review P2-1: single scratch root, single trap ------------------
SCRATCH_ROOT="$(mktemp -d)"
# SUITE_LOCK_HELD must exist (as "0") before ANYTHING that could exit through
# cleanup() runs -- including the ISOL_INTERRUPT_PROBE early-return branch
# immediately below, which (being a probe CHILD, not a real suite run) never
# reaches the round-7 mutex block that would otherwise set this to "1" --
# `set -u` would otherwise fault on cleanup()'s own reference to it.
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

# --- SP-S16 interruption probe: MUST be checked early, before any suite logic
# (round-6 review P2-1 acceptance test), and BEFORE the mktemp-shim setup
# below: probe mode's `set +e` (see the comment inside the branch) needs to
# take effect before ANY further command runs in this process, since a
# signal can arrive during the shim's own file writes just as easily as
# during the probe's own fifo dance -- moving the whole shim setup after
# this check keeps probe mode's error-handling window as small as possible.
# When invoked with ISOL_INTERRUPT_PROBE set, this is a CHILD re-invocation
# of this very script, spawned by the SP-S16 scenario body further down. It
# signals its own SCRATCH_ROOT back to the parent over a fifo (the write is
# also the readiness signal), then either exits immediately (mode "exit") or
# blocks forever on a second fifo that nobody ever writes to, so a
# parent-delivered INT/TERM is what interrupts it -- exactly mirroring how
# this script could be interrupted mid-scenario in real use. It does NOT
# need its own mktemp shim (below): its own `mktemp -d` for SCRATCH_ROOT
# above already inherited the PARENT's shim via PATH (set before the parent
# spawned it), which is what lands this SCRATCH_ROOT inside the parent's
# chosen probe TMPDIR in the first place.
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

# --- round-7 review P2b: cross-suite mutex -----------------------------------
# Placed immediately after the ISOL_INTERRUPT_PROBE early-return above (so a
# probe CHILD -- itself a full re-invocation of this script, spawned by
# SP-S16 -- never reaches this and never contends for the lock its own
# PARENT already holds) and BEFORE `export TMPDIR=...` below (so
# `${TMPDIR:-/tmp}` here still resolves to the INVOKING environment's TMPDIR,
# not the scratch-local one this script is about to export over it). Two
# instances of this suite -- or of the audio_common counterpart, which uses
# this EXACT SAME lock path/name -- must never run concurrently: both assert
# real-tree invariants (S6/S7/SP1's real kiss/ne10 archives, the final-guard
# snapshot below, the dist/ sentinels, tree_state_hash()) that a second
# concurrent run's own real-tree activity would trample.
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
# Fractional seconds (round-6: replaces whole-second `stat -f %m`) -- two
# genuinely separate real writes are distinguishable without an artificial
# delay. GNU fallback loses the fractional part (not exercised on this host).
mtime()    { stat -f '%Fm' "$1" 2>/dev/null || stat -c %Y "$1"; }

# Tree-state hash (round-6 review): status --porcelain ALONE misses a content
# edit to an already-dirty file, so this also folds in `diff --binary HEAD`.
tree_state_hash() {
  { git -C "$1" status --porcelain; git -C "$1" diff --binary HEAD; } 2>/dev/null | shasum -a 256 | awk '{print $1}'
}

# adopt_worktree_clone <src-repo-dir> <clone-dir> (round-7 review P1b,
# hoisted here from SP-S17 so SP2 -- which now runs before SP-S17 and shares
# its clone set -- can use it too; aligned verbatim with audio_common's own
# shared version of this helper) -- clone <src-repo-dir> (fully read-only on
# the source -- the explicitly allowed exception to "no mutating git commands
# against real repos"), then overlay every currently-MODIFIED TRACKED file on
# top and adopt them as ONE throwaway commit INSIDE THE SCRATCH CLONE ONLY, so
# the clone gets a genuinely CLEAN git identity that still reflects today's
# (possibly uncommitted) worktree content. A plain `git clone` alone only
# reproduces the source's last COMMIT -- it transfers committed objects,
# never uncommitted working-tree bytes -- so a bare clone would carry the OLD
# Makefile instead of the round-7 one under test. The clone is a disposable
# git repository living entirely under $SCRATCH_ROOT, destroyed by the EXIT
# trap; its own history never touches the source repo's .git in any way (a
# distinct clone, never pushed/fetched back).
#
# The commit is CONDITIONAL -- `git add -A`, then commit only if
# `git diff --cached --quiet --exit-code` says something is actually staged
# -- an unconditional commit would exit 1 (nothing to commit) on a clean tip
# and kill the whole suite under `set -e`.
adopt_worktree_clone() {
  local src="$1" dst="$2" relf
  git clone --no-hardlinks --quiet "$src" "$dst"
  while IFS= read -r relf; do
    [ -n "$relf" ] || continue
    mkdir -p "$(dirname "$dst/$relf")"
    cp "$src/$relf" "$dst/$relf"
  done < <(git -C "$src" diff --name-only HEAD)
  git -C "$dst" add -A
  if ! git -C "$dst" diff --cached --quiet --exit-code; then
    git -C "$dst" -c user.email=scratch@example.invalid -c user.name="scratch clone" \
      commit -q -m "scratch: adopt current worktree (disposable clone only, never touches the real repo)"
  fi
}

# Real-tree final guard (round-7 review P3): per-file "path sha256 mtime"
# snapshot of everything under the REAL obj/ and bin/ of the four trees this
# suite builds in (pipelines itself, lib/aec, lib/nr, audio_common), plus a
# separate top-level keyed-directory listing -- captured once SP1's own
# intentional real-tree builds (kiss + ne10) are done, and re-verified at the
# very end: every snapshotted file must still be sha+mtime identical; a
# genuinely NEW directory is allowed (INFO/WARN, not FAIL -- but after the
# SP-S12 fix there should be none), while a new file inside an
# ALREADY-snapshotted directory is a FAIL. Each line is tagged with a short
# repo prefix (pipelines/lib_aec/lib_nr/audio_common) joined onto the
# relative path -- the four trees' relative paths (obj/kiss-<sig>/...) would
# otherwise collide across repos once combined into one snapshot set.
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
  _fg_snapshot_repo audio_common "$AC_DIR"
}
final_guard_dirs() {
  _fg_dirs_repo pipelines "$PIPE_DIR"
  _fg_dirs_repo lib_aec "$AEC_DIR"
  _fg_dirs_repo lib_nr "$NR_DIR"
  _fg_dirs_repo audio_common "$AC_DIR"
}

# Real dist/ sentinel (round-6 review, standing guard): absent stays absent;
# otherwise a full path list + per-file sha256 + per-file (name, mtime).
# <root>/dist is the argument (not a bare repo root), so this works for
# pipelines' own dist, lib/aec/c_impl's, lib/nr/c_impl's, $AC_DIR's, and
# (belt-and-braces) the two PRIMARY AEC/NR repos' c_impl dist/ too.
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

# assert_cmd_fails_with <description> <expected-substring> <cmd...> (round-4
# review scenarios SP-S9/SP-S12/SP-S13) -- runs <cmd...> with stdout+stderr
# merged; PASS iff it exits non-zero AND the combined output contains
# <expected-substring>; FAIL (dumping the log) otherwise. Fixed scratch path
# (a counter-suffixed name under $SCRATCH_ROOT/tmp), not `mktemp` -- no
# cleanup-state registration involved either way, consistent with the
# fixed-path style used everywhere else in this rewrite.
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

# resolve_producers <backend> -- resolves audio_common's/lib-aec's/lib-nr's
# archive paths for <backend> (default flags otherwise), exactly the way
# pipelines/Makefile's own phase-1 dispatch recipe does. Submodule AEC_DIR/
# NR_DIR paths -- this is round-5-stable producer resolution, no round-6
# feature needed. Sets globals AC_LIB_/AEC_LIB_/NR_LIB_.
resolve_producers() {
  local backend="$1"
  AC_LIB_="$(make -s -C "$AC_DIR" BACKEND="$backend" WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 print-lib-path)"
  AEC_LIB_="$(make -s -C "$AEC_DIR" BACKEND="$backend" WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 AC_DIR="$AC_DIR" AC_LIB="$AC_LIB_" print-lib-path)"
  NR_LIB_="$(make -s -C "$NR_DIR" BACKEND="$backend" WERROR=0 CC='cc' CXX='c++' EXTRA_CFLAGS='' NO_STDIO=0 AC_DIR="$AC_DIR" AC_LIB="$AC_LIB_" print-lib-path)"
}

cd "$PIPE_DIR"

# --- BEFORE snapshots (round-6 review: 4 repos mandated by the safety
# contract, plus 2 bonus repos this script also builds against directly) ----
AUDIO_ALG_STATE_BEFORE="$(tree_state_hash "$AUDIO_ALG_ROOT")"
AEC_SUB_STATE_BEFORE="$(tree_state_hash "$AEC_SUB_ROOT")"
NR_SUB_STATE_BEFORE="$(tree_state_hash "$NR_SUB_ROOT")"
AC_STATE_BEFORE="$(tree_state_hash "$AC_DIR")"
AEC_R6_STATE_BEFORE="$(tree_state_hash "$AEC_REPO_DIR")"
NR_R6_STATE_BEFORE="$(tree_state_hash "$NR_REPO_DIR")"

REAL_DIST_PIPE_BEFORE="$(real_dist_sentinel "$PIPE_DIR")"
REAL_DIST_AEC_BEFORE="$(real_dist_sentinel "$AEC_DIR")"
REAL_DIST_NR_BEFORE="$(real_dist_sentinel "$NR_DIR")"
REAL_DIST_AC_BEFORE="$(real_dist_sentinel "$AC_DIR")"
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
make -s BACKEND=kiss libaudio_pipeline.a >/dev/null
default_lib="$(make -s BACKEND=kiss NO_STDIO=0 print-lib-path)"
sha_default_before="$(file_sha "$default_lib")"

S6_LOG="$SCRATCH_ROOT/s6/log"
if make BACKEND=kiss audit-no-stdio >"$S6_LOG" 2>&1; then
  pass "S6: audit-no-stdio exits green"
else
  fail "S6: audit-no-stdio FAILED"
  cat "$S6_LOG" >&2
fi
grep -q '^PASS:' "$S6_LOG" && pass "S6: audit-no-stdio printed a PASS line" \
  || fail "S6: audit-no-stdio did not print a PASS line"

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
mkscratch s7
S7_DIST_ROOT="$SCRATCH_ROOT/s7/dist"
resolve_producers kiss
# round-6: ALLOW_DIRTY_PUBLISH=1 on every publish call in this script from
# here on -- these dev trees are legitimately dirty (this very round-6/7
# rewrite is itself uncommitted, same as audio_common's/AEC's/NR's own
# round-6/7 edits); the dirty-publish POLICY itself is exercised on purpose
# in SP-S17. round-7: ALLOW_UNTRACKED_PUBLISH=1 rides along on every one of
# those same calls from here on too (EXCEPT SP-S17's own deliberate
# policy-negative calls, which test a knob's absence on purpose) -- the
# untracked dimension is a SEPARATE default refusal from the tracked-dirty
# one (see the Makefile's own ALLOW_UNTRACKED_PUBLISH comment), and a dev
# tree can legitimately carry untracked files that have nothing to do with
# what each of these scenarios actually tests. DIST_ROOT stays a throwaway
# scratch path: the real dist/ is never read, written, or removed.
make -s BACKEND=kiss DIST_ROOT="$S7_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null

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
S7_LOG_A="$SCRATCH_ROOT/s7/log_a"; S7_LOG_B="$SCRATCH_ROOT/s7/log_b"
( make -s BACKEND=kiss DIST_ROOT="$S7_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S7_LOG_A" 2>&1 ) & cp1=$!
( make -s BACKEND=kiss DIST_ROOT="$S7_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S7_LOG_B" 2>&1 ) & cp2=$!
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

# round-7 review P3: final-guard snapshot, taken NOW -- both of the real-tree
# normal-config builds this suite intends (kiss above, via S6/S7 and this
# scenario; ne10 just now) are done for all four trees (pipelines itself,
# lib/aec, lib/nr, audio_common). Every later real-tree `make ... lib`/`all`
# in this suite re-targets one of these same two already-built configs and
# must be a pure no-op against them. Re-verified at the very end (see "Final
# integrity guards").
FINAL_GUARD_DIRS_BEFORE="$(final_guard_dirs)"
FINAL_GUARD_SNAPSHOT_BEFORE="$(final_guard_snapshot)"

# round-6: no sleep -- this is a "should NOT relink" check, so mtime()'s BSD
# fractional-second read (stat -f '%Fm') is what makes a genuine relink here
# distinguishable from m_k1, regardless of how close in wall-clock time the
# two kiss builds land (a whole-second mtime comparison could otherwise
# false-PASS a real, same-second relink).
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
echo "# SP2: producer-change propagation (round-7: scratch clones)"
echo "############################################################"
# round-7 review P2b: this scenario used to `touch` REAL audio_common
# sources (src/hpf.c, include/fast_math.h) -- round-7 removes even a
# mtime-only touch of any real tracked file. Runs against a disposable clone
# of audio_common + the PRIMARY AEC repo + the PRIMARY NR repo instead. This
# SAME clone set is created fresh HERE (on the still-clean tip) and REUSED
# (then further dirtied) by SP-S17 further down -- one clone build instead
# of two separate ones. The producer objs build INSIDE the clones (their own
# default, clone-relative obj/bin, which live under scratch by construction);
# the pipeline binary itself builds from the REAL pipelines directory (never
# cloned) with AC_DIR/AEC_DIR/NR_DIR pointed at the clones plus an explicit
# scratch OBJ_ROOT/BIN_ROOT.
mkscratch pshared
mkscratch sp2
PSHARED_AC_CLONE="$SCRATCH_ROOT/pshared/ac_clone"
PSHARED_AEC_CLONE="$SCRATCH_ROOT/pshared/aec_clone"
PSHARED_NR_CLONE="$SCRATCH_ROOT/pshared/nr_clone"
adopt_worktree_clone "$AC_DIR" "$PSHARED_AC_CLONE"
adopt_worktree_clone "$AEC_REPO_DIR" "$PSHARED_AEC_CLONE"
adopt_worktree_clone "$NR_REPO_DIR" "$PSHARED_NR_CLONE"
# AEC's/NR's own c_impl/example/wav_io.h shim locates audio_common's
# include/wav_io.h via a HARDCODED relative __has_include path
# ("../../../audio_common/include/wav_io.h" from c_impl/example/) -- it
# expects audio_common as a SIBLING of the repo root. Neither clone lives
# next to a real "audio_common" (both under $SCRATCH_ROOT/pshared/), so a
# symlink named "audio_common", sibling to both, pointing at the audio_common
# clone, satisfies that lookup for both without editing any tracked source.
ln -s "$PSHARED_AC_CLONE" "$SCRATCH_ROOT/pshared/audio_common"

SP2_OBJ_ROOT="$SCRATCH_ROOT/sp2/obj"
SP2_BIN_ROOT="$SCRATCH_ROOT/sp2/bin"

# Resolve + build each producer INSIDE its own clone. CRITICAL: every make
# into the AEC/NR clones passes AC_DIR=<ac clone abs path> explicitly --
# their own `$(AC_LIB): FORCE` rule re-invokes `$(MAKE) -C $(AC_DIR) ...
# lib` using THAT invocation's own AC_DIR, and leaving it to a
# wildcard/relative default would resolve the wrong (real) tree,
# reintroducing exactly the real-tree writes this round-7 change removes.
#
# CC/CXX are pinned EXPLICITLY here (CC='cc' CXX='c++', the SAME literal
# values resolve_producers() uses throughout this script) rather than left to
# each Makefile's own bare `?=` default -- found empirically while validating
# this rewrite: audio_common's own bare default is `CC ?= cc`, but pipelines'
# own bare default is `CC ?= gcc`; forwarding the wrong one anywhere would key
# a producer to a different CFG_SIG than the rest of this scenario expects.
#
# OBJ_ROOT/BIN_ROOT (SP2_OBJ_ROOT/SP2_BIN_ROOT) are likewise the SAME
# EXPLICIT pair on every single make call below -- producer AND pipeline
# alike -- rather than the producers' own default (clone-relative) obj/bin.
# This is load-bearing, not stylistic: GNU Make automatically propagates
# EVERY command-line-origin variable (not just the ones $(FWD) explicitly
# lists) to every recursive $(MAKE) sub-invocation via MAKEFLAGS ("Communi-
# cating Variables to a Sub-make" in the GNU Make manual) -- so the
# PIPELINE's own OBJ_ROOT/BIN_ROOT override (needed so ITS artifacts never
# land in the real pipelines/obj,bin) is UNAVOIDABLY inherited by the nested
# `$(AC_LIB): FORCE`/`$(AEC_LIB): FORCE`/`$(NR_LIB): FORCE` sub-makes too,
# regardless of what $(FWD) lists -- so each producer's OWN build ends up
# under THIS SAME root either way, never its own clone-relative default.
# Found empirically while validating this rewrite (via `make -d`): a first
# attempt resolved `ac_k` with NO OBJ_ROOT/BIN_ROOT override (so it pointed
# at the clone's own default obj/bin), then separately ran the pipeline
# build WITH an OBJ_ROOT/BIN_ROOT override -- the nested sub-make's ACTUAL
# audio_common archive silently landed under the inherited scratch root
# instead, while `ac_k` kept pointing at a stale, never-rebuilt copy under
# the clone's own default path (masked because that stale file already
# existed from an earlier manual pre-build, so the FORCE rule's own
# `test -f '$@'` assertion still passed) -- every "touch source -> relink"
# assertion below false-failed as a result. Passing the identical
# OBJ_ROOT=/BIN_ROOT= explicitly everywhere removes the mismatch outright
# instead of fighting MAKEFLAGS propagation.
CCX=(CC='cc' CXX='c++' WERROR=0 EXTRA_CFLAGS='' NO_STDIO=0)
ORX=(OBJ_ROOT="$SP2_OBJ_ROOT" BIN_ROOT="$SP2_BIN_ROOT")
ac_k="$(make -s -C "$PSHARED_AC_CLONE" BACKEND=kiss "${CCX[@]}" "${ORX[@]}" print-lib-path)"
make -s -C "$PSHARED_AC_CLONE" BACKEND=kiss "${CCX[@]}" "${ORX[@]}" lib >/dev/null
aec_k="$(make -s -C "$PSHARED_AEC_CLONE/c_impl" BACKEND=kiss "${CCX[@]}" "${ORX[@]}" AC_DIR="$PSHARED_AC_CLONE" AC_LIB="$ac_k" print-lib-path)"
make -s -C "$PSHARED_AEC_CLONE/c_impl" BACKEND=kiss "${CCX[@]}" "${ORX[@]}" AC_DIR="$PSHARED_AC_CLONE" AC_LIB="$ac_k" lib >/dev/null
nr_k="$(make -s -C "$PSHARED_NR_CLONE/c_impl" BACKEND=kiss "${CCX[@]}" "${ORX[@]}" AC_DIR="$PSHARED_AC_CLONE" AC_LIB="$ac_k" print-lib-path)"
make -s -C "$PSHARED_NR_CLONE/c_impl" BACKEND=kiss "${CCX[@]}" "${ORX[@]}" AC_DIR="$PSHARED_AC_CLONE" AC_LIB="$ac_k" lib >/dev/null

sp2_makeargs=(BACKEND=kiss "${CCX[@]}" "${ORX[@]}" \
  AC_DIR="$PSHARED_AC_CLONE" AEC_DIR="$PSHARED_AEC_CLONE/c_impl" NR_DIR="$PSHARED_NR_CLONE/c_impl" \
  AC_LIB="$ac_k" AEC_LIB="$aec_k" NR_LIB="$nr_k")

make -s "${sp2_makeargs[@]}" all >/dev/null
bd_k="$(make -s "${sp2_makeargs[@]}" print-bin-dir)"

# "Did it rebuild?" is verified from make's own build LOG (grep for that
# rule's `@echo` line), never an mtime diff -- exactly audio_common's own S5
# technique (see that scenario's comment for the full rationale). Found
# empirically while validating THIS rewrite: comparing two mtime snapshots
# of the SAME output file, taken moments apart by THIS SCRIPT, can land in
# the very same whole second when the whole touch-rebuild-relink chain runs
# fast and back-to-back inside a scratch clone (no human/sleep delay between
# steps) -- a real false-negative (reproduced directly: a genuinely
# recompiled hpf.o, and a genuinely rebuilt archive, both landing in the
# same integer second as their OWN prior build, so GNU Make 3.81's own
# whole-second-truncated newer-than check declines to re-archive/relink,
# even though a higher-resolution clock shows a real, later timestamp).
# backdate_sp2(): backdates a CLONE-owned build artifact (an archive or the
# final binary -- never a real tracked source) by a safe margin (5 minutes)
# immediately before the rebuild that must supersede it, so that rebuild's
# fresh "now" timestamp is unambiguously newer regardless of which whole
# second it lands in -- removing the collision instead of hoping to avoid
# it. Mirrors audio_common's S5 `backdate_aecwav()` helper.
backdate_sp2() { touch -A -0500 "$1"; }

# touch the audio_common CLONE's src/hpf.c -> hpf.o recompiles AND
# aec_nr_pipeline relinks (audio_common's own CFG_SIG is a hash of its
# COMPILER INVOCATION, not file content, so its archive path is unaffected --
# only its mtime advances). Deterministic strictly-newer bump (not `sleep`):
# hpf.c's mtime is set to hpf.o's CURRENT mtime + 1s (BSD touch -r/-A),
# guaranteeing make itself sees the source as newer than the object. The
# CLONE's own archive AND the pipeline binary are then backdated (freely
# mutable clone/scratch artifacts) right before the rebuild that must
# supersede both, so the archive-rebuild and relink echoes are guaranteed to
# fire regardless of whole-second collisions. mtime-only touch of hpf.c: no
# real tree, tracked or otherwise, is ever touched.
ac_objdir="$(make -s -C "$PSHARED_AC_CLONE" BACKEND=kiss "${CCX[@]}" "${ORX[@]}" print-obj-dir)"
backdate_sp2 "$ac_k"
backdate_sp2 "$bd_k/aec_nr_pipeline"
touch -r "$ac_objdir/hpf.o" -A 01 "$PSHARED_AC_CLONE/src/hpf.c"
SP2_LOG_HPF="$SCRATCH_ROOT/sp2/log_hpf"
make -s "${sp2_makeargs[@]}" all >"$SP2_LOG_HPF" 2>&1
grep -q "^audio_common \[kiss\] ->" "$SP2_LOG_HPF" && pass "SP2: touching the audio_common clone's src/hpf.c rebuilt its archive" \
  || { fail "SP2: audio_common's archive did NOT rebuild after the clone's hpf.o changed"; cat "$SP2_LOG_HPF" >&2; }
grep -q "^aec_nr_pipeline \[kiss\] ->" "$SP2_LOG_HPF" && pass "SP2: touching the audio_common clone's src/hpf.c relinked aec_nr_pipeline" \
  || { fail "SP2: aec_nr_pipeline NOT relinked after the audio_common clone's hpf.o changed"; cat "$SP2_LOG_HPF" >&2; }

# touch include/fast_math.h -> NR/AEC objects that include it
# (aec3_post.c/suppression_gain.c in lib/aec, mcra_noise_estimator.c/
# spp_estimator.c/mmse_lsa_denoiser.c in lib/nr) recompile, AND the pipeline
# binary relinks -- the full transitive header chain. Again mtime-only, and
# again against the CLONE's copy: no tracked file's content OR mtime changes
# in any real repo. Same log-based verification + backdating as above --
# this time backdating BOTH producer archives (lib/aec's, lib/nr's) plus the
# pipeline binary.
aec_objdir="$(make -s -C "$PSHARED_AEC_CLONE/c_impl" BACKEND=kiss "${CCX[@]}" "${ORX[@]}" AC_DIR="$PSHARED_AC_CLONE" AC_LIB="$ac_k" print-obj-dir)"
nr_objdir="$(make -s -C "$PSHARED_NR_CLONE/c_impl" BACKEND=kiss "${CCX[@]}" "${ORX[@]}" AC_DIR="$PSHARED_AC_CLONE" AC_LIB="$ac_k" print-obj-dir)"

# touch fast_math.h strictly newer than BOTH aec3_post.o AND
# mcra_noise_estimator.o -- the two were compiled moments apart in the
# original build above, in no guaranteed order, so pick whichever of the
# two has the LATER current mtime as the touch -r reference before bumping
# +1s, guaranteeing fast_math.h ends up newer than both regardless of which
# one happened to compile later (immune to whichever way it rounds).
sp2_ref="$aec_objdir/aec3_post.o"
if awk -v a="$(mtime "$nr_objdir/mcra_noise_estimator.o")" -v b="$(mtime "$sp2_ref")" 'BEGIN{exit !(a>b)}'; then
  sp2_ref="$nr_objdir/mcra_noise_estimator.o"
fi
backdate_sp2 "$aec_k"
backdate_sp2 "$nr_k"
backdate_sp2 "$bd_k/aec_nr_pipeline"
touch -r "$sp2_ref" -A 01 "$PSHARED_AC_CLONE/include/fast_math.h"
SP2_LOG_FASTMATH="$SCRATCH_ROOT/sp2/log_fastmath"
make -s "${sp2_makeargs[@]}" all >"$SP2_LOG_FASTMATH" 2>&1

grep -q "^aec \[kiss\] ->" "$SP2_LOG_FASTMATH" && pass "SP2: touching the audio_common clone's include/fast_math.h recompiled lib/aec's aec3_post.o (archive rebuilt)" \
  || { fail "SP2: lib/aec's archive did NOT rebuild after touching the clone's fast_math.h (aec3_post.o likely did not recompile)"; cat "$SP2_LOG_FASTMATH" >&2; }
grep -q "^Static library built: " "$SP2_LOG_FASTMATH" && pass "SP2: touching the audio_common clone's include/fast_math.h recompiled lib/nr's mcra_noise_estimator.o (archive rebuilt)" \
  || { fail "SP2: lib/nr's archive did NOT rebuild after touching the clone's fast_math.h (mcra_noise_estimator.o likely did not recompile)"; cat "$SP2_LOG_FASTMATH" >&2; }
grep -q "^aec_nr_pipeline \[kiss\] ->" "$SP2_LOG_FASTMATH" && pass "SP2: touching the audio_common clone's include/fast_math.h relinked aec_nr_pipeline (full transitive chain)" \
  || { fail "SP2: aec_nr_pipeline NOT relinked after the clone's fast_math.h touch"; cat "$SP2_LOG_FASTMATH" >&2; }

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
echo "# SP-S10: lib/aec archive freshness (round-6: scratch-side against the"
echo "#         PRIMARY AEC repo -- round-6 P1 fix)"
echo "############################################################"
mkscratch sp10
SP10_OBJ_ROOT="$SCRATCH_ROOT/sp10/obj"
SP10_BIN_ROOT="$SCRATCH_ROOT/sp10/bin"

# Resolve the real audio_common lib once (plain query target -- no
# ALLOW_DIRTY_PUBLISH needed; print-lib-path never touches DIST_ROOT/dirty
# state at all).
sp10_ac_lib="$(make -s -C "$AC_DIR" BACKEND=kiss print-lib-path)"

# Snapshot the REAL submodule libaec.a -- must stay byte/mtime-untouched
# across this whole scenario. The round-5 version of this scenario injected
# the foreign member directly into THIS file (a round-6 P1 finding this
# rewrite fixes by never touching it at all).
sp10_submodule_lib="$(make -s -C "$AEC_DIR" BACKEND=kiss print-lib-path)"
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
sp10_primary_real_lib="$(make -s -C "$AEC_R6_DIR" BACKEND=kiss print-lib-path)"
sp10_primary_existed_before=0
if [ -f "$sp10_primary_real_lib" ]; then
  sp10_primary_existed_before=1
  sp10_primary_sha_before="$(file_sha "$sp10_primary_real_lib")"
  sp10_primary_mtime_before="$(mtime "$sp10_primary_real_lib")"
fi

make -s -C "$AEC_R6_DIR" BACKEND=kiss OBJ_ROOT="$SP10_OBJ_ROOT" BIN_ROOT="$SP10_BIN_ROOT" AC_DIR="$AC_DIR" AC_LIB="$sp10_ac_lib" lib >/dev/null
sp10_lib="$(make -s -C "$AEC_R6_DIR" BACKEND=kiss OBJ_ROOT="$SP10_OBJ_ROOT" BIN_ROOT="$SP10_BIN_ROOT" AC_DIR="$AC_DIR" AC_LIB="$sp10_ac_lib" print-lib-path)"
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
make -s -C "$AEC_R6_DIR" BACKEND=kiss OBJ_ROOT="$SP10_OBJ_ROOT" BIN_ROOT="$SP10_BIN_ROOT" AC_DIR="$AC_DIR" AC_LIB="$sp10_ac_lib" lib >/dev/null

if ar -t "$sp10_lib" | grep -qx 'sp10_foreign.o'; then
  fail "SP-S10: foreign member SURVIVED a rebuild (archive was NOT rebuilt fresh -- looks like 'ar r' onto the existing .a rather than \$@.tmp + mv -f)"
else
  pass "SP-S10: foreign member is GONE after rebuild (fresh-archive discipline: \$@.tmp + mv -f)"
fi
ar -t "$sp10_lib" | grep -qx 'aec_debug.o' && pass "SP-S10: aec_debug.o present in the freshly-rebuilt archive" \
  || fail "SP-S10: aec_debug.o missing from the freshly-rebuilt archive"

sp10_objdir="$(make -s -C "$AEC_R6_DIR" BACKEND=kiss OBJ_ROOT="$SP10_OBJ_ROOT" BIN_ROOT="$SP10_BIN_ROOT" AC_DIR="$AC_DIR" AC_LIB="$sp10_ac_lib" print-obj-dir)"
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
echo "############################################################"
mkscratch sp11
SP11_DIST_ROOT="$SCRATCH_ROOT/sp11/dist"
resolve_producers kiss
make -s BACKEND=kiss DIST_ROOT="$SP11_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null

id1="$(readlink "$SP11_DIST_ROOT/kiss/current" || true)"
[ -n "$id1" ] && [ -d "$SP11_DIST_ROOT/kiss/$id1" ] && pass "SP-S11: publish v4 -- current symlink resolves to release dir '$id1'" \
  || fail "SP-S11: publish v4 -- current symlink broken or missing after the first publish"

rel_dir="$SP11_DIST_ROOT/kiss/$id1"

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

# round-6 attest v2: git_commit= (self) AND all three producer *_git_commit=
# fields are full 40-hex OIDs (round-5 used `git rev-parse --short`).
sp11_hex_ok=1
for field in git_commit audio_common_git_commit aec_git_commit nr_git_commit; do
  val="$(grep "^${field}=" "$first_attest" | head -1 | cut -d= -f2)"
  echo "$val" | grep -Eq '^[0-9a-f]{40}$' || sp11_hex_ok=0
done
[ "$sp11_hex_ok" -eq 1 ] && pass "SP-S11: git_commit= and all three producer *_git_commit= fields are full 40-hex OIDs" \
  || fail "SP-S11: one or more of git_commit=/audio_common_git_commit=/aec_git_commit=/nr_git_commit= is not 40 hex characters"

# round-7 review P2a/item 7: the *_git_untracked dimension is a SEPARATE
# default refusal from *_git_dirty (see the Makefile's ALLOW_UNTRACKED_PUBLISH
# comment) -- assert all four fields + allow_untracked_publish= exist in a
# normal, successful publish's attest (SP-S17a/b exercise the refusal path
# itself; this is the field-presence regression for the ordinary path).
sp11_untracked_fields_ok=1
for field in git_untracked audio_common_git_untracked aec_git_untracked nr_git_untracked allow_untracked_publish; do
  grep -q "^${field}=" "$first_attest" || sp11_untracked_fields_ok=0
done
[ "$sp11_untracked_fields_ok" -eq 1 ] && pass "SP-S11: attest carries all four *_git_untracked fields + allow_untracked_publish=" \
  || fail "SP-S11: attest is missing one or more of git_untracked=/audio_common_git_untracked=/aec_git_untracked=/nr_git_untracked=/allow_untracked_publish="

grep -q "^git_untracked=0\$" "$first_attest" && pass "SP-S11: pipelines' own git_untracked=0 (ainr/GTCRN|gtcrn_github whitelist holds)" \
  || fail "SP-S11: pipelines' own git_untracked= is not 0 (whitelist not filtering, or a genuine untracked file is present)"

release_dir_mtime_before="$(mtime "$rel_dir")"
snap_before="$(release_mtime_snapshot "$rel_dir")"

# round-6: no sleep -- the <NNN> suffix (not a distinct UTC second) is what
# disambiguates this immediate same-second republish (SP-S15 stress-tests
# the same-second case directly, forcing 20 publishes into one literal
# second via ATTEST_STAMP=).
S11_LOG="$SCRATCH_ROOT/sp11/republish.log"
make -s BACKEND=kiss DIST_ROOT="$SP11_DIST_ROOT" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >"$S11_LOG" 2>&1
if grep -q "already published (byte-verified" "$S11_LOG"; then
  pass "SP-S11: idempotent republish printed 'already published (byte-verified, incl. MANIFEST)'"
else
  fail "SP-S11: idempotent republish did NOT print the byte-verified message"
  cat "$S11_LOG" >&2
fi
grep -Eq '\(attested: attest-[^)]+\)[[:space:]]*$' "$S11_LOG" && pass "SP-S11: republish success line ends '(attested: <name>)'" \
  || fail "SP-S11: republish success line missing the '(attested: <name>)' suffix"

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
# round-7 review P2b (parity with audio_common's own S12 fix): this shim's
# path lives under $SCRATCH_ROOT, which is fresh every run -- and CXX
# participates in this repo's own CFG_SIG payload -- so building
# libaudio_pipeline.a with it in the REAL (default-rooted) obj/bin used to
# mint a brand-new orphan keyed directory there on every single run of this
# suite. Both shim builds now use a scratch OBJ_ROOT/BIN_ROOT instead; the
# guard under test doesn't care where the objects land.
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
  make BACKEND=ne10 CXX="$SP12_SHIM" OBJ_ROOT="$SP12_OBJ_ROOT" BIN_ROOT="$SP12_BIN_ROOT" libaudio_pipeline.a

S12_LOG="$SCRATCH_ROOT/sp12/log"
if make BACKEND=kiss CXX="$SP12_SHIM" OBJ_ROOT="$SP12_OBJ_ROOT" BIN_ROOT="$SP12_BIN_ROOT" libaudio_pipeline.a >"$S12_LOG" 2>&1; then
  pass "SP-S12: BACKEND=kiss with the same mismatched-triple CXX shim still SUCCEEDS (guard is ne10-only)"
else
  fail "SP-S12: BACKEND=kiss build unexpectedly FAILED with the CXX shim in place"
  cat "$S12_LOG" >&2
fi

echo "############################################################"
echo "# SP-S13: RNNoise-ERB drift-gate hardening (round-5 P2)"
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
echo "# SP-S14: make -n/-q/-t publish zero side effects (round-6 P2-3),"
echo "#         THREE Makefiles: pipelines / \$AEC_R6_DIR / \$NR_R6_DIR"
echo "############################################################"
mkscratch sp14

snap_dirs() { { find "$1" -maxdepth 1 -type d 2>/dev/null | sort; } || true; }

SP14_BEFORE_PIPE_OBJ="$(snap_dirs "$PIPE_DIR/obj")"
SP14_BEFORE_PIPE_BIN="$(snap_dirs "$PIPE_DIR/bin")"
SP14_BEFORE_AC_OBJ="$(snap_dirs "$AC_DIR/obj")"
SP14_BEFORE_AC_BIN="$(snap_dirs "$AC_DIR/bin")"
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
  "${MAKE_ARGS[@]}" -n BACKEND=kiss DIST_ROOT="$SP14_DIST" OBJ_ROOT="$SP14_OBJ" BIN_ROOT="$SP14_BIN" publish >"$SCRATCH_ROOT/sp14/$x.n.log" 2>&1 || rc=$?
  if [ "$x" = "pipe" ]; then
    # Discovered while validating this rewrite (round-6 P2-3, pipelines-
    # specific, pre-existing -- confirmed by a standalone `make -n publish`
    # repro outside this script entirely, so it is a genuine Makefile
    # characteristic, not a bug in this test): pipelines' own THREE-producer
    # FORCE rules forward $(AC_LIB) into a FURTHER nested
    # `$(MAKE) ... AC_LIB=$(AC_LIB) lib` sub-make call for lib/aec and
    # lib/nr. Under `-n`, GNU Make PRINTS (rather than executes)
    # audio_common's own `print-lib-path` recipe line VERBATIM -- including
    # its `echo` text -- instead of running it (reproduced directly:
    # `make -n -s -C audio_common print-lib-path` prints "echo /abs/path"
    # instead of just "/abs/path"), so the captured "$$ac" becomes the
    # literal STRING "echo /abs/path/to/lib.a" (containing a space) instead
    # of just the path. Forwarding THAT corrupted value on as
    # AC_LIB=$(AC_LIB) into the nested lib/aec or lib/nr sub-make then
    # splits it into extra bogus command-line words/goals, and that nested
    # make exits nonzero ("No rule to make target ..."). AEC's and NR's own
    # single-producer dispatch never re-forwards AC_LIB into a THIRD level
    # of nesting this way, so they don't hit it (see SP-S14[aec]/
    # SP-S14[nr] immediately below, both rc=0) -- this is specific to
    # pipelines' three-producer design. A Makefile fix is out of this
    # script's remit (only this test script may be modified); the actual
    # safety property this scenario exists to verify -- ZERO filesystem
    # side effects -- still holds regardless and is asserted unconditionally
    # for all three Makefiles right below, so this quirk does not weaken
    # the guarantee under test.
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
  "${MAKE_ARGS[@]}" -q BACKEND=kiss DIST_ROOT="$SP14_DIST" OBJ_ROOT="$SP14_OBJ" BIN_ROOT="$SP14_BIN" publish >"$SCRATCH_ROOT/sp14/$x.q.log" 2>&1 || rc=$?
  [ "$rc" -ne 0 ] && pass "SP-S14[$x]: make -q publish exits NONZERO (rc=$rc)" || fail "SP-S14[$x]: make -q publish exited 0 (expected nonzero)"
  if [ ! -e "$SP14_DIST" ] && [ ! -e "$SP14_OBJ" ] && [ ! -e "$SP14_BIN" ]; then
    pass "SP-S14[$x]: make -q publish is zero-write (created NONE of DIST_ROOT/OBJ_ROOT/BIN_ROOT)"
  else
    fail "SP-S14[$x]: make -q publish left behind a path (dist=$([ -e "$SP14_DIST" ] && echo yes || echo no) obj=$([ -e "$SP14_OBJ" ] && echo yes || echo no) bin=$([ -e "$SP14_BIN" ] && echo yes || echo no))"
  fi

  "${MAKE_ARGS[@]}" -t BACKEND=kiss DIST_ROOT="$SP14_DIST" OBJ_ROOT="$SP14_OBJ" BIN_ROOT="$SP14_BIN" publish >"$SCRATCH_ROOT/sp14/$x.t.log" 2>&1 || true
  if [ ! -e "$SP14_DIST" ] && [ ! -e "$SP14_OBJ" ] && [ ! -e "$SP14_BIN" ]; then
    pass "SP-S14[$x]: make -t publish is zero-write (created NONE of DIST_ROOT/OBJ_ROOT/BIN_ROOT)"
  else
    fail "SP-S14[$x]: make -t publish left behind a path (dist=$([ -e "$SP14_DIST" ] && echo yes || echo no) obj=$([ -e "$SP14_OBJ" ] && echo yes || echo no) bin=$([ -e "$SP14_BIN" ] && echo yes || echo no))"
  fi
done

SP14_AFTER_PIPE_OBJ="$(snap_dirs "$PIPE_DIR/obj")"
SP14_AFTER_PIPE_BIN="$(snap_dirs "$PIPE_DIR/bin")"
SP14_AFTER_AC_OBJ="$(snap_dirs "$AC_DIR/obj")"
SP14_AFTER_AC_BIN="$(snap_dirs "$AC_DIR/bin")"
SP14_AFTER_AECR6_OBJ="$(snap_dirs "$AEC_R6_DIR/obj")"
SP14_AFTER_AECR6_BIN="$(snap_dirs "$AEC_R6_DIR/bin")"
SP14_AFTER_NRR6_OBJ="$(snap_dirs "$NR_R6_DIR/obj")"
SP14_AFTER_NRR6_BIN="$(snap_dirs "$NR_R6_DIR/bin")"

[ "$SP14_BEFORE_PIPE_OBJ" = "$SP14_AFTER_PIPE_OBJ" ] && [ "$SP14_BEFORE_PIPE_BIN" = "$SP14_AFTER_PIPE_BIN" ] && \
  pass "SP-S14: no NEW keyed dirs appeared in pipelines' own real obj/bin" \
  || fail "SP-S14: a NEW keyed dir appeared in pipelines' own real obj/bin"
[ "$SP14_BEFORE_AC_OBJ" = "$SP14_AFTER_AC_OBJ" ] && [ "$SP14_BEFORE_AC_BIN" = "$SP14_AFTER_AC_BIN" ] && \
  pass "SP-S14: no NEW keyed dirs appeared in audio_common's real obj/bin" \
  || fail "SP-S14: a NEW keyed dir appeared in audio_common's real obj/bin"
[ "$SP14_BEFORE_AECR6_OBJ" = "$SP14_AFTER_AECR6_OBJ" ] && [ "$SP14_BEFORE_AECR6_BIN" = "$SP14_AFTER_AECR6_BIN" ] && \
  pass "SP-S14: no NEW keyed dirs appeared in the PRIMARY AEC repo's real obj/bin" \
  || fail "SP-S14: a NEW keyed dir appeared in the PRIMARY AEC repo's real obj/bin"
[ "$SP14_BEFORE_NRR6_OBJ" = "$SP14_AFTER_NRR6_OBJ" ] && [ "$SP14_BEFORE_NRR6_BIN" = "$SP14_AFTER_NRR6_BIN" ] && \
  pass "SP-S14: no NEW keyed dirs appeared in the PRIMARY NR repo's real obj/bin" \
  || fail "SP-S14: a NEW keyed dir appeared in the PRIMARY NR repo's real obj/bin"

echo "--- SP-S14b: COMBINED dry-run flags stay zero-write (PIPELINES Makefile) --"
# round-7 follow-up (the S15b pattern from audio_common's script): a combined
# `make -nt publish` could take the -n branch and recurse; the recursed
# child then sees BOTH flags and GNU make applies REAL touch semantics down
# the _publish_impl prerequisite chain -- the pipelines Makefile's own
# dry-run guard checks t FIRST specifically to avoid this (see that target's
# own comment). Exercised against a scratch-root libaudio_pipeline.a whose
# archive is deliberately BACKDATED below its own object: exactly the state
# a leaked -t would "fix" by touching. Scratch-side only -- the real tree's
# artifacts are covered by the SP1 final-guard snapshot and must never be
# backdated. libaudio_pipeline.a needs no producer archive at all, so no
# AC_LIB/AEC_LIB/NR_LIB resolution is needed for this check.
mkscratch sp14b
SP14B_OBJ="$SCRATCH_ROOT/sp14b/obj"; SP14B_BIN="$SCRATCH_ROOT/sp14b/bin"; SP14B_DIST="$SCRATCH_ROOT/sp14b/nx"
make -s BACKEND=kiss OBJ_ROOT="$SP14B_OBJ" BIN_ROOT="$SP14B_BIN" libaudio_pipeline.a >/dev/null
sp14b_lib="$(make -s BACKEND=kiss OBJ_ROOT="$SP14B_OBJ" BIN_ROOT="$SP14B_BIN" print-lib-path)"
touch -t 202001010000 "$sp14b_lib"
sp14b_ref="$(stat -f '%m %z' "$sp14b_lib")"
for combo in "-nt" "-n -t" "-tq" "-nq" "-nqt"; do
  rm -rf "$SP14B_DIST"
  rc=0; make $combo BACKEND=kiss OBJ_ROOT="$SP14B_OBJ" BIN_ROOT="$SP14B_BIN" DIST_ROOT="$SP14B_DIST" publish >/dev/null 2>&1 || rc=$?
  if [ ! -e "$SP14B_DIST" ] && [ "$(stat -f '%m %z' "$sp14b_lib")" = "$sp14b_ref" ]; then
    pass "SP-S14b[$combo]: zero-write (no DIST_ROOT; stale scratch archive untouched; rc=$rc informational)"
  else
    fail "SP-S14b[$combo]: WROTE something (dist=$([ -e "$SP14B_DIST" ] && echo yes || echo no), archive stat now '$(stat -f '%m %z' "$sp14b_lib")' vs '$sp14b_ref')"
  fi
done

echo "############################################################"
echo "# SP-S15: ATTEST uniqueness under forced same-second collisions"
echo "#         (round-6 P2-2)"
echo "############################################################"
mkscratch sp15
SP15_DIST="$SCRATCH_ROOT/sp15/dist"
resolve_producers kiss
make -s BACKEND=kiss DIST_ROOT="$SP15_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null
sp15_id="$(readlink "$SP15_DIST/kiss/current" || true)"
sp15_attest_dir="$SP15_DIST/kiss/$sp15_id/ATTEST"

sp15_before_list="$(find "$sp15_attest_dir" -name 'attest-*.txt' | sort)"
sp15_before_snap="$SCRATCH_ROOT/sp15/before_snap.txt"
: > "$sp15_before_snap"
for f in $sp15_before_list; do
  [ -n "$f" ] || continue
  printf '%s %s %s %s\n' "$f" "$(stat -f '%i' "$f")" "$(mtime "$f")" "$(file_sha "$f")" >> "$sp15_before_snap"
done

for i in $(seq 1 20); do
  make -s BACKEND=kiss DIST_ROOT="$SP15_DIST" ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 ATTEST_STAMP=20260715T999999Z publish AC_LIB="$AC_LIB_" AEC_LIB="$AEC_LIB_" NR_LIB="$NR_LIB_" >/dev/null
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
echo "# SP-S16: interruption-safety probe (round-6 P2-1 acceptance)"
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
echo "# SP-S17: dirty-producer provenance (round-6 P2; round-7: shared clones)"
echo "############################################################"
mkscratch sp17

# round-7 review P2b/item 3: SP-S17 no longer clones its OWN three producer
# checkouts -- it REUSES the exact PSHARED_AC_CLONE/PSHARED_AEC_CLONE/
# PSHARED_NR_CLONE set SP2 (above) already built while they were still
# clean, one clone build instead of two separate ones. `adopt_worktree_clone`
# itself is now defined once, in the shared helpers section near the top of
# this script (hoisted from here so SP2, which now runs first, can use it
# too) -- see that definition's own comment for the full rationale.
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
echo "/* sp17 disposable dirty probe */" >> "$SP17_NR_CLONE/c_impl/src/mmse_lsa_denoiser.c"

SP17_OBJ_ROOT="$SCRATCH_ROOT/sp17/obj"
SP17_BIN_ROOT="$SCRATCH_ROOT/sp17/bin"
SP17_DIST="$SCRATCH_ROOT/sp17/dist"

# sp17_attest_name_from_log(): the publish recipe's own success line ends
# "(attested: <name>)" -- extract it directly rather than globbing ATTEST/
# (this pipelines repo is ALSO dirty right now -- uncommitted round-6
# changes, by this task's own design -- so it publishes its own attest
# fields too; the log line unambiguously names THIS invocation's own attest
# file regardless of how many total attest files a release dir holds).
sp17_attest_name_from_log() {
  local log="$1" name
  name="$(grep -o '(attested: [^)]*)' "$log" | sed -e 's/^(attested: //' -e 's/)$//' | head -1)"
  [ -n "$name" ] || return 1
  printf '%s\n' "$name"
}

SP17_LOG_OK="$SCRATCH_ROOT/sp17/log_ok"
# round-7 item 6: ALLOW_UNTRACKED_PUBLISH=1 rides along here too (this is a
# success-path call, not one of the deliberate policy-negative sub-cases
# below) -- inert in practice, since these clones carry no untracked files
# yet (only tracked-dirty edits above), but uniform with every other publish
# call in this script from S7 onward.
if make -s BACKEND=kiss DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     AC_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17_AEC_CLONE/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
     ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$SP17_LOG_OK" 2>&1; then
  pass "SP-S17: publish against three dirty producer clones + ALLOW_DIRTY_PUBLISH=1 succeeds"
else
  fail "SP-S17: publish against three dirty producer clones + ALLOW_DIRTY_PUBLISH=1 FAILED"
  cat "$SP17_LOG_OK" >&2
fi

sp17_id="$(readlink "$SP17_DIST/kiss/current" || true)"
sp17_attest_name="$(sp17_attest_name_from_log "$SP17_LOG_OK" || true)"
sp17_attest="$SP17_DIST/kiss/$sp17_id/ATTEST/$sp17_attest_name"

sp17_ac_commit="$(git -C "$SP17_AC_CLONE" rev-parse HEAD)"
sp17_aec_commit="$(git -C "$SP17_AEC_CLONE" rev-parse HEAD)"
sp17_nr_commit="$(git -C "$SP17_NR_CLONE" rev-parse HEAD)"
sp17_ac_ddiff="$(git -C "$SP17_AC_CLONE" diff --binary HEAD | shasum -a 256 | cut -d' ' -f1)"
sp17_aec_ddiff="$(git -C "$SP17_AEC_CLONE" diff --binary HEAD | shasum -a 256 | cut -d' ' -f1)"
sp17_nr_ddiff="$(git -C "$SP17_NR_CLONE" diff --binary HEAD | shasum -a 256 | cut -d' ' -f1)"

if [ -n "$sp17_attest_name" ] && [ -f "$sp17_attest" ] \
   && grep -q "^audio_common_git_dirty=1\$" "$sp17_attest" \
   && grep -q "^audio_common_git_commit=$sp17_ac_commit\$" "$sp17_attest" \
   && grep -q "^audio_common_dirty_diff_sha256=$sp17_ac_ddiff\$" "$sp17_attest" \
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
if make -s BACKEND=kiss DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     AC_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17_AEC_CLONE/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
     publish >"$SP17_LOG_FAIL" 2>&1; then
  fail "SP-S17: publish against the same dirty producer clones WITHOUT ALLOW_DIRTY_PUBLISH unexpectedly SUCCEEDED"
else
  if grep -q "publish refused" "$SP17_LOG_FAIL" \
     && grep -q "audio_common working tree has uncommitted TRACKED changes" "$SP17_LOG_FAIL" \
     && grep -q "lib/aec working tree has uncommitted TRACKED changes" "$SP17_LOG_FAIL" \
     && grep -q "lib/nr working tree has uncommitted TRACKED changes" "$SP17_LOG_FAIL"; then
    pass "SP-S17: publish without ALLOW_DIRTY_PUBLISH correctly FAILS, naming audio_common/lib/aec/lib/nr as the dirty repos"
  else
    fail "SP-S17: publish failed but without the expected 'publish refused' / per-producer dirty wording"
    cat "$SP17_LOG_FAIL" >&2
  fi
fi

echo "--- SP-S17a: untracked producer probe (audio_common) -------------------"
# round-7 review P2a/item 5a: the untracked dimension is SEPARATE from the
# tracked-dirty one already exercised above. An untracked file in the SHARED
# audio_common clone (already tracked-dirty from the edit above, so
# ALLOW_DIRTY_PUBLISH=1 alone already admits THAT dimension) must still be
# refused on the UNTRACKED dimension unless ALLOW_UNTRACKED_PUBLISH=1 is
# ALSO given -- and once given, the attestation records
# audio_common_git_untracked=1 + a 64-hex audio_common_untracked_tree_sha256,
# while pipelines' OWN git_untracked stays 0 (the ainr/GTCRN|gtcrn_github
# whitelist holding, since this repo's own tree never gained an untracked
# file from this scenario).
echo "int sp17a_untracked_probe;" > "$SP17_AC_CLONE/probe.c"

SP17_LOG_A1="$SCRATCH_ROOT/sp17/log_untracked_a1"
if make -s BACKEND=kiss DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     AC_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17_AEC_CLONE/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
     ALLOW_DIRTY_PUBLISH=1 publish >"$SP17_LOG_A1" 2>&1; then
  fail "SP-S17a: ALLOW_DIRTY_PUBLISH=1 ALONE unexpectedly admitted an untracked file in audio_common"
else
  if grep -qi "publish refused" "$SP17_LOG_A1" && grep -q "audio_common" "$SP17_LOG_A1" && grep -q "UNTRACKED" "$SP17_LOG_A1"; then
    pass "SP-S17a: ALLOW_DIRTY_PUBLISH=1 alone is refused, naming audio_common on the untracked dimension"
  else
    fail "SP-S17a: refusal did not name audio_common's UNTRACKED files"
    cat "$SP17_LOG_A1" >&2
  fi
fi

SP17_LOG_A2="$SCRATCH_ROOT/sp17/log_untracked_a2"
if make -s BACKEND=kiss DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     AC_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17_AEC_CLONE/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
     ALLOW_DIRTY_PUBLISH=1 ALLOW_UNTRACKED_PUBLISH=1 publish >"$SP17_LOG_A2" 2>&1; then
  pass "SP-S17a: ALLOW_DIRTY_PUBLISH=1 + ALLOW_UNTRACKED_PUBLISH=1 together succeed with an untracked audio_common file present"
else
  fail "SP-S17a: both knobs together FAILED"
  cat "$SP17_LOG_A2" >&2
fi

sp17a_id="$(readlink "$SP17_DIST/kiss/current" || true)"
sp17a_attest_name="$(sp17_attest_name_from_log "$SP17_LOG_A2" || true)"
sp17a_attest="$SP17_DIST/kiss/$sp17a_id/ATTEST/$sp17a_attest_name"
sp17a_ac_uhash="$(grep '^audio_common_untracked_tree_sha256=' "$sp17a_attest" 2>/dev/null | head -1 | cut -d= -f2)"
if [ -n "$sp17a_attest_name" ] && [ -f "$sp17a_attest" ] \
   && grep -q "^audio_common_git_untracked=1\$" "$sp17a_attest" \
   && [ -n "$sp17a_ac_uhash" ] && [ "${#sp17a_ac_uhash}" -eq 64 ] \
   && grep -q "^git_untracked=0\$" "$sp17a_attest"; then
  pass "SP-S17a: attestation shows audio_common_git_untracked=1 + 64-hex audio_common_untracked_tree_sha256, pipelines' own git_untracked=0"
else
  fail "SP-S17a: attestation ($sp17a_attest) does not match the expected untracked-producer shape"
fi

echo "--- SP-S17b: unknown producer (AEC_DIR -> a no-git copy) -> unconditional FATAL --"
# round-7 review item 5b: point AEC_DIR at a no-git scratch copy of the
# PRIMARY AEC repo (a plain tar copy, no .git at all) -- a checkout with no
# git identity is refused UNCONDITIONALLY, naming lib/aec, even with BOTH
# escape hatches set (neither admits it: without a commit OID neither the
# tracked diff nor the untracked enumeration can be named for that repo).
SP17B_NOGIT_AEC="$SCRATCH_ROOT/sp17/aec_nogit"
mkdir -p "$SP17B_NOGIT_AEC"
tar -c -C "$AEC_REPO_DIR" --exclude obj --exclude bin --exclude dist --exclude .git . | tar -x -C "$SP17B_NOGIT_AEC"

SP17_LOG_B="$SCRATCH_ROOT/sp17/log_unknown_aec"
if make -s BACKEND=kiss DIST_ROOT="$SP17_DIST" OBJ_ROOT="$SP17_OBJ_ROOT" BIN_ROOT="$SP17_BIN_ROOT" \
     AC_DIR="$SP17_AC_CLONE" AEC_DIR="$SP17B_NOGIT_AEC/c_impl" NR_DIR="$SP17_NR_CLONE/c_impl" \
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

AC_STATE_AFTER="$(tree_state_hash "$AC_DIR")"
[ "$AC_STATE_BEFORE" = "$AC_STATE_AFTER" ] && pass "integrity: \$AC_DIR (audio_common) tree-state (status+diff) unchanged across the whole run" \
  || fail "integrity: \$AC_DIR (audio_common) tree-state CHANGED during this run"

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

REAL_DIST_AC_AFTER="$(real_dist_sentinel "$AC_DIR")"
[ "$REAL_DIST_AC_BEFORE" = "$REAL_DIST_AC_AFTER" ] && pass "integrity: real \$AC_DIR/dist sentinel unchanged" \
  || fail "integrity: real \$AC_DIR/dist sentinel CHANGED during this run"

# Bonus: the PRIMARY AEC/NR repos' own real dist/ too.
REAL_DIST_AECR6_AFTER="$(real_dist_sentinel "$AEC_R6_DIR")"
[ "$REAL_DIST_AECR6_BEFORE" = "$REAL_DIST_AECR6_AFTER" ] && pass "integrity (bonus): real PRIMARY AEC repo dist sentinel unchanged" \
  || fail "integrity (bonus): real PRIMARY AEC repo dist sentinel CHANGED during this run"

REAL_DIST_NRR6_AFTER="$(real_dist_sentinel "$NR_R6_DIR")"
[ "$REAL_DIST_NRR6_BEFORE" = "$REAL_DIST_NRR6_AFTER" ] && pass "integrity (bonus): real PRIMARY NR repo dist sentinel unchanged" \
  || fail "integrity (bonus): real PRIMARY NR repo dist sentinel CHANGED during this run"

# round-7 review P3: final-guard re-check against the snapshot taken in SP1
# (right after this suite's LAST intentional real-tree build). Every file
# that existed then, across all FOUR trees (pipelines, lib/aec, lib/nr,
# audio_common), must still be sha+mtime identical now; a directory created
# after the snapshot is allowed but reported (WARN/INFO -- after the SP-S12
# fix there should be none), while a NEW file inside an already-snapshotted
# directory is a hard FAIL.
FINAL_GUARD_DIRS_AFTER="$(final_guard_dirs)"
FINAL_GUARD_SNAPSHOT_AFTER="$(final_guard_snapshot)"

fg_lost="$(comm -23 <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_BEFORE" | LC_ALL=C sort) <(printf '%s\n' "$FINAL_GUARD_SNAPSHOT_AFTER" | LC_ALL=C sort))"
if [ -z "$fg_lost" ]; then
  pass "final-guard: every real obj/+bin/ file (pipelines, lib/aec, lib/nr, audio_common) snapshotted after SP1's builds is still sha+mtime identical"
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
