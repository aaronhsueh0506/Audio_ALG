# Audio_ALG/pipelines/Makefile
#
# AEC(linear) -> echo-aware NR -> RES pipeline — two versions of the SAME
# algorithm (see mono_aec_nr_res/main.c's header for the DSP chain):
#
#   aec_nr_pipeline         (Version A: malloc)         — heap-allocated modules
#   aec_nr_pipeline_static  (Version B: static memory)  — one caller pool, no
#                                                          malloc after init
#
# Both are now thin CLI shells (arg parsing + WAV I/O + diagnostics) over
# mono_aec_nr_res/audio_pipeline.h/.c — the pool-sizing/carving/per-hop-
# processing logic itself is reusable through audio_pipeline.h. The two host
# runners link it from libaudio_pipeline.a; the ULCNet application links its
# wrapper/pre-post objects directly rather than adding another wrapper archive.
#
# All three link against the submodule libs (lib/aec, lib/nr) + the shared
# audio_common FFT/fast_math layer (../../audio_common relative to this dir).
#
# =============================================================================
# Build isolation — the fourth and last of four repos
# =============================================================================
# This repo is a CONSUMER of THREE producer archives (audio_common's
# libaudio_common.a, lib/aec's libaec.a, lib/nr's libmmse_lsa.a), each of
# which is itself CFG_SIG-keyed (see those repos' own Makefiles). The design
# here is the same two-phase recursive-make dispatch AEC/NR already use for
# their ONE producer (audio_common), extended to resolve THREE:
#
#   Phase 1 (AC_LIB unset — the normal case for any top-level `make ...`):
#   a goal that needs any producer archive gets a recipe that queries, IN
#   ORDER, audio_common's, then lib/aec's, then lib/nr's own `print-lib-path`
#   (forwarding this invocation's exact BACKEND/WERROR/CC/CXX/EXTRA_CFLAGS/
#   NO_STDIO — and, for the lib/aec and lib/nr queries, ALSO the just-resolved
#   AC_DIR/AC_LIB, so THEIR OWN two-phase dispatch resolves directly instead
#   of re-deriving it) and re-invokes itself with AC_LIB/AEC_LIB/NR_LIB set.
#
#   Phase 2 (AC_LIB known): the REAL rules are defined. Each producer archive
#   is a FORCE-recipe target (always re-invokes that producer's own build,
#   which is a no-op if already up to date, then asserts the archive exists)
#   and a NORMAL (not order-only) prerequisite of every rule that links
#   against it — so relinking happens iff a producer's archive mtime actually
#   advanced.
#
# This repo's own objects/binaries are ALSO hash-keyed the same way
# (obj|bin/<backend>-<cfg_sig>/) — CFG_SIG folds in this Makefile's own full
# flag coverage (CC/CXX/AR/RANLIB/CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/
# EXTRA_CFLAGS/EXTRA_LDFLAGS/BACKEND/WERROR/NO_STDIO/DEBUG) PLUS the three
# resolved producers' own config identities (AC_PRODUCER_SIG/AEC_PRODUCER_SIG/
# NR_PRODUCER_SIG — each just the last path component of that producer's
# resolved archive dir, e.g. "kiss-1234567890", never the full path) — so a
# pipeline build against a differently-configured producer keys to its own
# separate directory too, instead of risking silent config skew.
#
# Usage:
#   make                       # resolves+builds all three producer archives
#                              # (BACKEND=ne10, the default everywhere) + both
#                              # pipeline binaries
#   make BACKEND=kiss          # portable, bit-reproducible KISS reference
#                              # backend, pass explicitly -- every axis this
#                              # file and the three producers' own Makefiles
#                              # key on (BACKEND, EXTRA_CFLAGS, WERROR,
#                              # NO_STDIO, ...) lands in a fresh, separate
#                              # obj|bin directory automatically; no manual
#                              # clean needed, ever
#   make EXTRA_CFLAGS="..."    # forwarded to all three producers AND this
#                              # pipeline's own compile
#   make WERROR=1              # opt-in: -Werror for this file's own TUs
#   make test                  # build + run mono API and board-adapter tests
#
# The four-channel pipeline has an intentionally independent producer at
# 4ch_aec_bf_nr_res/Makefile.  Keeping it there prevents its DOA/GSC dependencies and
# source set from changing this mono pipeline's build identity.
#   make NO_STDIO=1 libaudio_pipeline.a   # board-firmware stdio-less build of
#                              # JUST the library object (see the NO_STDIO
#                              # comment further down) -- does not link, so it
#                              # needs none of the three producer archives
#   make audit-no-stdio        # PASS/FAIL gate: proves the delivered
#                              # NO_STDIO=1 libaudio_pipeline.a really has no
#                              # fprintf/printf/puts/fputs/stderr refs, and
#                              # that its keyed path differs from the default
#                              # (stdio) config's path
#   make print-bin-dir | print-obj-dir | print-lib-path
#                              # report THIS invocation's own keyed paths (see
#                              # the query-targets comment further down for
#                              # how to get the exact path `make all` itself
#                              # would produce)
#   make publish               # copy this build's artifacts to
#                              # dist/<backend>/<cfg_sig>/ + update
#                              # dist/<backend>/current (see that section)
#   make libs                  # compat alias: builds just the three producer
#                              # archives (no pipeline objects/binaries)
#   make clean                 # remove this dir's own obj/bin trees + any
#                              # legacy flat binaries/archive left over from
#                              # before this directory layout existed
#   make clean-all             # also clean lib/aec, lib/nr, audio_common build
#                              # dirs (clean-libs is a compatibility alias)

# =============================================================================
# Whitespace build-root fail-fast guard (this task)
# =============================================================================
# This Makefile forwards several resolved directory paths (AC_DIR, and
# transitively AEC_DIR/NR_DIR) to sub-`make` invocations as BARE, UNQUOTED
# shell words on a recipe line -- e.g. `AC_DIR=$(abspath $(AC_DIR))` in the
# two-phase dispatch / FORCE rules / query targets / audit-no-stdio-stack
# recipes further down this file. GNU Make has no concept of shell quoting
# for text like this: it is substituted verbatim into the recipe's command
# line and handed to $(SHELL) as-is. If the resolved path contains a space
# (as THIS checkout's own absolute path can and does -- confirmed: the real
# checkout root here contains both a literal space and CJK characters), the
# shell word-splits that unquoted argument on the space, silently breaking
# `AC_DIR=/that/whole/path` into several separate words -- which the
# sub-`make` on that same line then parses as EXTRA COMMAND-LINE ARGUMENTS
# (stray target names, or a lone "-"), producing a cryptic, deep "No such
# file or directory" / "No rule to make target ..." failure inside some
# unrelated sub-invocation, nowhere near this actual cause (verified
# empirically against this exact checkout).
#
# Making whitespace paths actually WORK (quoting every such forward
# consistently, end to end, across all four repos in this stack) is
# explicitly OUT OF SCOPE here -- a known, separately-tracked, pre-existing
# environment limitation, not something to fix in this pass. This guard only
# makes the FAILURE ITSELF immediate and legible: checked as close to the top
# of this file as possible -- before AC_DIR/BACKEND/CC/CXX/... are even given
# their defaults, and long before any of the recipes described above can run
# -- so a whitespace-containing invocation is rejected right here, with a
# clear, one-line explanation, instead of failing later, cryptically, deep
# inside some sub-make this file spawns.
#
# $(CURDIR) is GNU Make's own notion of "the directory this invocation is
# running in" (post any `-C` processing) -- exactly the "invocation's own
# working/build-root path" this guard is meant to check: AC_DIR/AEC_DIR/
# NR_DIR are always resolved RELATIVE to this same directory (see their own
# defaults just below), so they share whatever space-or-not-space ancestor
# path this invocation is already running under. A literal space has no
# direct token form in GNU Make function syntax (a bare space in a function
# argument list is a separator, not data), so it is built the standard way:
# an empty variable, doubled with itself around one space.
AUDIO_PIPELINE_EMPTY :=
AUDIO_PIPELINE_SPACE := $(AUDIO_PIPELINE_EMPTY) $(AUDIO_PIPELINE_EMPTY)
ifneq ($(findstring $(AUDIO_PIPELINE_SPACE),$(CURDIR)),)
$(error this build root path contains whitespace ('$(CURDIR)'), which is not supported -- re-run from a path with no spaces. (Whitespace build-root paths are a known, pre-existing limitation shared by all four repos in this stack -- audio_common, lib/aec, lib/nr, and this repo -- not something this pass fixes; this check only fails fast and clearly, here, instead of a cryptic "No such file or directory" deep inside an unrelated sub-invocation.))
endif

AEC_DIR = ../lib/aec/c_impl
NR_DIR  = ../lib/nr/c_impl

# Align-ULCNet C pre/post (STFT/WOLA + UlcnetModel callback boundary) --
# consumed by the audio_pipeline_ulcnet variant below. Header-only include
# path plus one TU (ulcnet_process.c) compiled by THIS Makefile (that file
# is not in any producer archive). Not part of CFG_SIG's caller-varying
# axes: a fixed repo-relative constant, like AEC_DIR/NR_DIR above.
AIAEC_DIR = ../AIAEC
ULCNET_DIR = $(AIAEC_DIR)/Align_ULCNet

# audio_common sits two levels up from pipelines/ (SE/Audio_ALG/pipelines -> SE/audio_common).
# lib/aec/c_impl and lib/nr/c_impl each resolve it themselves from their own
# location (../../../../audio_common from c_impl/, verified to exist as a
# submodule checkout); AC_DIR here is for OUR OWN CFLAGS/-I path and is passed
# through explicitly (as AC_DIR=$(abspath $(AC_DIR))) to the lib/aec and
# lib/nr sub-makes so all three always agree on the exact same audio_common
# checkout.
AC_DIR ?= ../../audio_common
BACKEND ?= ne10

# Announce the resolved backend on every build (same MAKECMDGOALS guard as
# audio_common's own banner: print-bin-dir/print-obj-dir/print-lib-path are
# consumed via `$$(make -s ... print-lib-path)` command substitution
# elsewhere, and this must not add stray stdout to that captured value).
ifeq ($(filter print-%,$(MAKECMDGOALS)),)
$(info Audio_ALG/pipelines: FFT backend = $(BACKEND) -- switch with `make BACKEND=kiss` or `make BACKEND=ne10`)
endif

CC  ?= gcc
CXX ?= c++
AR      ?= ar
RANLIB  ?= ranlib

# CFG_SIG payload-coverage placeholders: CXXFLAGS/
# CPPFLAGS are not consumed by this Makefile today (no C++ TU of its own),
# but the CFG_SIG payload below unconditionally includes them so the payload
# string has the same stable shape as audio_common's/AEC's/NR's, and so a
# future use of any of them here keys builds correctly instead of silently
# aliasing with a build that didn't set it. EXTRA_LDFLAGS/NO_STDIO are real
# here (see LDFLAGS/LIB_CFLAGS below).
#
# CXXFLAGS and CPPFLAGS are also two of the five names the "Command-line
# override rejection" foreach further below rejects a command-line/`-e`
# override of (the same class of bug
# audio_common's/AEC's/NR's CXXFLAGS/CPPFLAGS had, all fixed the same way).
# `?=` only ever assigns when the variable's origin is still "undefined" --
# if either already has a value from the environment, `?=` is a complete
# no-op (GNU Make never attempts the assignment at all), so the origin never
# flips to "environment override" under `make -e` no matter how early this
# line sits. `+=` fixes this while keeping the externally-visible behaviour
# identical: appending an empty string is a no-op on the VALUE (a
# plain-environment CXXFLAGS/CPPFLAGS still folds in normally -- origin
# becomes "file" -- and still participates in CFG_SIG_PAYLOAD unchanged),
# but `+=` is an unconditional assignment ATTEMPT, so under `-e` GNU Make's
# origin bookkeeping now correctly flips it to "environment override", which
# the foreach rejects. See audio_common's/AEC's/NR's Makefile for the
# identical fix and full writeup.
CXXFLAGS      +=
CPPFLAGS      +=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

# DEBUG is real here too: this Makefile has no
# DEBUG-gated code path of its OWN, but DEBUG is forwarded via $(FWD) (below)
# to every producer sub-make (audio_common, lib/aec, lib/nr) -- most directly
# lib/nr's, whose Makefile rejects any non-0/1 DEBUG outright. Without this
# Makefile resolving/validating its OWN DEBUG explicitly, a sub-make would
# fall through to inheriting whatever the calling shell's environment
# happens to have exported for DEBUG (e.g. a stray `DEBUG=release` left over
# from an unrelated tool), silently building the wrong config downstream --
# or, since lib/nr's guard landed, hard-failing deep inside a producer
# sub-build instead of erroring here with proper locality. Same exact-match
# guard shape as lib/nr's: `ifneq` twice (not a
# single $(filter DEBUG,0 1) check) so a multi-word value like `DEBUG='0 1'`
# or one containing `%` can't slip through -- see lib/nr's Makefile for the
# full rationale.
ifneq ($(DEBUG),0)
ifneq ($(DEBUG),1)
$(error DEBUG must be exactly 0 or 1 (got DEBUG='$(DEBUG)') -- pass DEBUG=0 \
  (default) or DEBUG=1. Any other value -- including one inherited from a \
  stray exported DEBUG environment variable -- used to silently pass \
  through to producer sub-makes (e.g. lib/nr's, which now rejects it) \
  instead of erroring here at the point Audio_ALG itself is invoked; it no \
  longer does)
endif
endif

# TOOLCHAIN_CHECK=0 skips the CC/CXX -dumpmachine target-coherence guard in
# _cfg_guard below. Forwarded to all three producer
# sub-makes via $(FWD); participates in CFG_SIG.
TOOLCHAIN_CHECK ?= 1

# DIST_ROOT / HOSTCC: where `make publish` writes releases
# (test scripts point this at a throwaway mktemp dir so they never touch the
# real dist/), and the HOST compiler for the publish-time atomic
# symlink-swap helper (never $(CC): that may be a cross compiler). Neither
# is in CFG_SIG -- neither affects what is built. See audio_common's
# Makefile for the full writeups.
#
# ALLOW_DIRTY_PUBLISH: `make
# publish` REFUSES by default when THIS repo (Audio_ALG) or any of the three
# producers (audio_common, lib/aec, lib/nr) has uncommitted TRACKED changes
# (this repo: `git status --porcelain -uno`; each producer: `git -C <dir>
# status --porcelain -uno`, same -uno scope -- see _publish_impl) or is not
# a git checkout at all (commit unknown) -- a release whose ATTEST
# provenance cannot name the exact source state of the whole stack is not a
# board deliverable. A checkout with NO git identity at all is refused
# UNCONDITIONALLY, for every one of the four repos -- neither knob below
# admits it (provenance is impossible without a commit OID to name), and
# this is checked before any dirty/untracked flag is even computed for that
# repo. ALLOW_DIRTY_PUBLISH=1 is the explicit escape hatch for dev
# publishes on an otherwise-identified checkout; it is recorded in the
# attestation together with a sha256 of each dirty repo's `git diff
# --binary HEAD`. It covers TRACKED changes ONLY -- untracked files have
# their own dimension below (the tracked diff hash says
# nothing about untracked bytes, so lumping the two under one flag let two
# different untracked source states publish with identical provenance).
# Not in CFG_SIG: it never changes what is built, only whether publish
# proceeds.
#
# ALLOW_UNTRACKED_PUBLISH: publish also REFUSES by default
# when this repo (outside the whitelist below) or any of the three
# producers contains ANY untracked, non-ignored file (this repo: `git
# ls-files --others --exclude-standard` minus the whitelist; each producer,
# NO whitelist: `git -C <dir> ls-files --others --exclude-standard`;
# gitignored build output is excluded by design and never part of this
# provenance) -- an untracked source a (possibly clean-tracked) Makefile
# references can change the artifact without leaving a trace in the
# tracked diff. ALLOW_UNTRACKED_PUBLISH=1 is the separate escape hatch; it
# records untracked_tree_sha256 per repo that had untracked=1: sha256 over
# sorted FIXED-FIELD per-file records emitted by the `hash_untracked()`
# helper at the top of _publish_impl -- "L <sha256(path)> <sha256(readlink
# output)>" for a symlink, "F <mode> <sha256(path)> <sha256(content)>" for
# a regular file -- every variable-length value is itself hashed, so
# records concatenate unambiguously (raw space-joined path/target strings
# were collision-prone: "a b"->"c" vs "a"->"b c" encoded identically) and
# any stat/readlink/shasum failure downgrades the entry to an X record =>
# FATAL, never an empty hash (fail-closed). A path that is neither a
# regular file nor a symlink (an embedded git checkout, a fifo) cannot be
# hashed deterministically and is always refused, with a diagnostic naming
# both the path and the repo it came from.
#
# THIS repo's untracked check is WHITELISTED BY PREFIX (the three producers
# get NO such whitelist -- an untracked file there is always a violation):
# `git ls-files --others --exclude-standard` output with any path matching
# `^AINR/(GTCRN|gtcrn_github)/` removed (one `grep -vE`, `|| true` guards
# the empty-list case). This working tree permanently carries those two
# directories as user-owned, never-staged content (see workflow docs);
# without the whitelist, an otherwise byte-identical publish would refuse
# (or, under ALLOW_UNTRACKED_PUBLISH=1, would have its provenance hash
# perturbed by) files that have nothing to do with the released artifact.
# Both prefixes stay whitelisted for the SAME two reasons, one per
# directory: `GTCRN/` is a plain directory that may gain further untracked
# files over time (still must never count against a publish);
# `gtcrn_github/` is an EMBEDDED git checkout (its own `.git/`, not a
# registered submodule) and so currently collapses to a single opaque
# `AINR/gtcrn_github/` line in `git ls-files --others` output rather than
# one line per contained file -- which is ALSO exactly the "neither a
# regular file nor a symlink" case the X-record type-guard in
# `hash_untracked()` would otherwise FATAL on as unhashable. If that inner
# `.git` ever disappears, the same prefix will instead match a whole tree
# of ordinary per-file untracked entries -- covered either way by matching
# on the prefix rather than the single current line. Neither knob is in
# CFG_SIG: they never change what is built, only whether publish proceeds.
#
# ATTEST_STAMP (TEST-ONLY): overrides the UTC timestamp
# embedded in the attestation FILENAME so the isolation tests can
# deterministically force same-second publish collisions and prove the
# no-clobber retry path. Never use it on a real publish -- though note a
# forged stamp is within the already-accepted threat model: attestations
# are unsigned and provide traceability, NOT authenticity against an
# attacker with filesystem access. Not in CFG_SIG or MANIFEST.
DIST_ROOT ?= dist
HOSTCC    ?= cc
ALLOW_DIRTY_PUBLISH ?= 0
ALLOW_UNTRACKED_PUBLISH ?= 0
ATTEST_STAMP ?=

# EXTRA_CFLAGS — pass-through to the AEC/NR/audio_common producer builds AND
# this pipeline's own compile (all three producers' own Makefiles already do
# `CFLAGS += $(EXTRA_CFLAGS)`, so one EXTRA_CFLAGS on this top-level `make`
# reaches every .o for the whole build). The fast matched-filter arithmetic
# and delay-estimator duty-cycling are built in unconditionally (no flag) —
# see lib/aec/c_impl/src/delay_aec3.c.
#
# Switching EXTRA_CFLAGS (or BACKEND, or WERROR, or NO_STDIO) never needs a
# manual `make clean-libs` first: this Makefile's own objects/binaries are
# hash-keyed by config (see the OBJ_DIR/BIN_DIR/CFG_SIG section below), and
# lib/aec's, lib/nr's, and audio_common's own Makefiles independently key
# their objects/archives off the same axes -- a stale, differently-flagged
# .o/.a simply can't exist under the new path a config produces.
# `make clean-libs`/`clean-all` remain available but are no longer required
# for correctness.
EXTRA_CFLAGS ?=
# PROFILE=1 turns on the per-hop stage timing across everything this Makefile
# builds: the mono pipeline's own nr/post/synth here, and lib/aec's own stages
# through the same knob forwarded by name. Off by default so a release build
# takes none of the clock reads; each component's own header carries its
# count, and this file deliberately does not restate it. Folded into CFG_SIG
# both by name and through CFLAGS, so a profile build and a release build
# occupy different obj/bin directories and cannot hand each other a stale
# object. Setting only one component's -D by hand is legible rather than
# broken -- the other half simply reads 0.
PROFILE ?= 0
ifneq ($(words $(PROFILE)),1)
$(error PROFILE must be exactly 0 or 1, got '$(PROFILE)')
endif
ifneq ($(filter-out 0 1,$(PROFILE)),)
$(error PROFILE must be exactly 0 or 1, got '$(PROFILE)')
endif
# Set through PROFILE_CFLAGS -> CFLAGS rather than EXTRA_CFLAGS, mirroring
# SIMD right below. EXTRA_CFLAGS is forwarded verbatim to the producer
# sub-makes and folded into CFG_SIG, so smuggling the -D through it made the
# flag arrive twice whenever a consumer that also has PROFILE drove this
# build, splitting byte-identical objects across two keys, and re-keyed
# producers that never reference the macro at all. CFLAGS is already
# protected from a command-line assignment (see the FP policy guard below),
# so the knob cannot be silently dropped either.
# Both -D land in THIS Makefile's CFLAGS, which is not forwarded: lib/aec gets
# its own AEC_STAGE_TIMING from the PROFILE knob forwarded by name in FWD
# below, so the flag reaches its compile once rather than twice.
#
# AEC_STAGE_TIMING is repeated here on purpose. It is not only the library's
# switch -- this layer's own translation units test it, to decide whether to
# copy the lanes' record out and what the tests may assert about it. Leaving
# it out built a consumer that believed the library had not measured while it
# had: every AEC stage reported zero with the cost showing up as
# unattributed, and the test that should have caught it agreed with the
# broken behaviour instead. Both sides deriving from PROFILE is what keeps
# the two views from disagreeing.
#
# A lib/aec pin older than the PROFILE knob has nothing to receive and
# degrades to the documented one-half state -- the wrapper's stages measure,
# the AEC block reads zero -- which the --timing report shows as real numbers
# beside a row of zeros rather than hiding.
ifeq ($(PROFILE),1)
PROFILE_CFLAGS := -DAUDIO_PIPELINE_STAGE_TIMING=1 -DAEC_STAGE_TIMING=1
else
PROFILE_CFLAGS :=
endif

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

# Deliberate BYTE-CHANGE when -ffp-contract=off first landed here (clang had
# been contracting a handful of mul+add pairs in the pipeline glue, e.g. the
# |E|^2 hoist) — the pipeline render baselines were regenerated at that
# commit. Keeping the flag uniform means future SIMD/EMA-style code added to
# pipeline TUs inherits the repo-wide bit-exactness discipline instead of
# silently depending on the compiler's contraction mood. The
# flag itself is appended further down (see the "FP-contraction policy"
# section below, after EXTRA_CFLAGS/WERROR are folded in) rather than here —
# it was moved from this early position (BEFORE EXTRA_CFLAGS)
# to the true end of the CFLAGS assembly, so a caller's EXTRA_CFLAGS can
# never land after it and override it.
CFLAGS = -Wall -Wextra -O2 -std=gnu99 \
         -I. \
         -Imono_aec_nr_res -Imono_alignulcnet -I$(ULCNET_DIR) \
         -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
         -I$(NR_DIR)/include \
         -I$(AC_DIR)/include -I$(AC_DIR)/lib/kiss_fft
# AUDIO_PIPELINE_BACKEND_STR: compile-time FFT backend identity baked into
# audio_pipeline.c's AudioPipelineMemReq.backend field (F20) -- matches
# whichever libaudio_common.a variant this build links (BACKEND= above), so
# the descriptor never has to be told separately which backend produced it.
CFLAGS += -DAUDIO_PIPELINE_BACKEND_STR=\"$(BACKEND)\"
CFLAGS += $(EXTRA_CFLAGS)
CFLAGS += $(SIMD_CFLAGS)
CFLAGS += $(PROFILE_CFLAGS)

# -Werror opt-in for this Makefile's OWN TUs (default off; CI can flip it
# on with `make WERROR=1`). Vendor code (audio_common's NE10/KISS) is built by
# audio_common's own Makefile, not by rules in this file, so no separate
# warning isolation is needed on this side.
WERROR ?= 0
ifeq ($(WERROR),1)
  CFLAGS += -Werror
endif

# NO_STDIO=1: compiles audio_pipeline.o with
# -DAUDIO_PIPELINE_NO_STDIO, which turns every AP_LOG_ERR() diagnostic in
# audio_pipeline.c into a no-op and drops its <stdio.h> include entirely --
# see that file's "No-stdio build gate" section. This is a board-firmware
# concern (a stdio-less image, or one that forbids the stdio symbol set
# outright), so it is scoped to LIB_CFLAGS (used ONLY by the
# libaudio_pipeline.a object below) rather than the shared CFLAGS: the two
# CLIs (aec_nr_pipeline / aec_nr_pipeline_static) are host tools that always
# do their own WAV-I/O stdio regardless, and test_audio_pipeline.c prints
# its own PASS/FAIL via stdio unconditionally -- neither references
# AUDIO_PIPELINE_NO_STDIO, so there is nothing to gain (and a real loss of
# CLI diagnostics) from adding the define to their compiles.
LIB_CFLAGS = $(CFLAGS)
ifeq ($(NO_STDIO),1)
  LIB_CFLAGS += -DAUDIO_PIPELINE_NO_STDIO
endif

# --- FP-contraction policy: unified across all four repos. This flag
# USED TO be the fourth token of the base CFLAGS assignment
# above (before -I$(AEC_DIR)/include etc., and well before EXTRA_CFLAGS/
# WERROR were folded in) -- so a caller passing EXTRA_CFLAGS=-ffp-contract=fast
# (or -Ofast/-ffast-math, which imply contraction) would land LATER on the
# compile line and silently win. Moved here -- appended LAST, after every
# other CFLAGS append in this file (the base -I flags, -DAUDIO_PIPELINE_
# BACKEND_STR, EXTRA_CFLAGS, WERROR), and BEFORE CFG_SIG_PAYLOAD is computed
# below -- so nothing can override it any more. LIB_CFLAGS = $(CFLAGS) above
# is a recursive (lazily-expanded) variable: its own definition is the
# literal text "$(CFLAGS)" (plus the conditional -DAUDIO_PIPELINE_NO_STDIO
# token), re-expanded against CFLAGS's CURRENT value every time LIB_CFLAGS
# is used in a recipe -- so appending FP_POLICY to CFLAGS alone (below)
# already reaches LIB_CFLAGS automatically; appending it to LIB_CFLAGS a
# second time here would duplicate the flag on every libaudio_pipeline.a
# compile line (harmless in effect, since both instances say "off", but
# noisy and worth avoiding). See audio_common's Makefile for the full
# cross-repo design writeup and scripts/audit_fp_contract.sh (in
# audio_common) for the disassembly-level verification.
#
# --- Bare-literal policy flag, defined early (`make -e`
# environment-override gap) --------------------------------------------------
# FP_POLICY is a plain, dependency-free `:=` literal (no reference to CFLAGS
# or anything else computed above) that used to be defined much further down
# this file, immediately ahead of CFG_SIG_PAYLOAD -- for a reason that still
# applies and is unaffected by this move (see the comment above: FP_POLICY's
# CFLAGS append still needs to happen AFTER every other CFLAGS append in
# this file, and BEFORE CFG_SIG_PAYLOAD is computed -- both still true with
# only the bare literal relocated here). FP_POLICY is ALSO listed in the
# "Command-line override rejection" foreach directly below, and that
# combination used to be a real gap under `make -e`, empirically confirmed:
# `env FP_POLICY=-ffp-contract=fast make -e -n BACKEND=kiss lib` used to
# sail straight through with no error (rc=0), silently dropping
# -ffp-contract=off from the dry-run compile line for audio_pipeline.o (see
# the foreach's own comment for the full mechanism -- $(origin FP_POLICY)
# only flips from plain "environment" to "environment override" once a
# makefile assignment to FP_POLICY has actually been PARSED, which never
# happened before the foreach ran while the bare `:=` sat all the way down
# at its old position, well after this foreach). Duplicating only the bare
# literal up here -- ahead of the foreach in file order -- fixes it; the
# original site further below still holds its OTHER job unchanged (`CFLAGS
# += $(FP_POLICY)`, appended LAST), so there is exactly one `:=` definition
# of FP_POLICY in this file, just moved. Same fix audio_common/AEC/NR
# already applied to their own Makefiles.
FP_POLICY := -ffp-contract=off

# FP_ALLOWED_CHARS_RE (added alongside the FP-policy allow-list redesign
# below -- replicated from audio_common's/AEC's/NR's identical fix, see the
# "REDESIGN" comment further down for the full writeup) is a SECOND bare,
# dependency-free `:=` literal with the exact same `make -e` exposure as
# FP_POLICY above -- it lives here, fully relocated (this is its ONLY
# definition in this file), so its origin is already final by the time the
# foreach below queries it. The regex it holds is documented in full where
# it is actually used (the FP-policy allow-list block, further below).
FP_ALLOWED_CHARS_RE := ^[A-Za-z0-9_./=,+ -]*$$

# LDFLAGS is just link FLAGS (-lm plus whatever EXTRA_LDFLAGS adds) -- NOT
# the three producer archive paths. Those are resolved dynamically (see the
# two-phase dispatch below) and appended directly to each link recipe's own
# prerequisite list / command line, in the fixed order the old hardcoded
# LIBS variable used: pipeline.o, libaudio_pipeline.a, libaec.a,
# libmmse_lsa.a, libaudio_common.a, -lm. libaudio_pipeline.a
# holds ONLY the pipeline TU (audio_pipeline.o) -- it is not a bundled/fat
# archive, so it still needs libaec.a/libmmse_lsa.a/libaudio_common.a linked
# after it, same as when audio_pipeline.c's logic was file-local code inside
# each CLI's own .o.
#
# (`make -e` gap fix: this composition USED to sit further
# down this file, after the "Command-line override rejection" foreach below
# -- the one repo of the four where this was true (AEC's and NR's own
# LDFLAGS compositions both already precede their own foreach, so neither
# needed this move). Same mechanism as FP_POLICY's bare literal above:
# `$(origin LDFLAGS)` only flips to "environment override" once a makefile
# assignment to LDFLAGS has actually been PARSED, so
# `env LDFLAGS=<poison> make -e -n ... lib` used to sail through the foreach
# untouched while LDFLAGS's own defining lines sat below it, unparsed at
# foreach-time. Relocating the two lines here -- verbatim, same relative
# order, nothing else about them changed -- fixes it; nothing between here
# and their old position reads $(LDFLAGS) (the two-phase archive-path
# dispatch below forwards BACKEND/WERROR/CC/CXX/EXTRA_CFLAGS/NO_STDIO/
# TOOLCHAIN_CHECK/DEBUG via $(FWD), never LDFLAGS; the only other readers
# are CFG_SIG_PAYLOAD and the four link recipes, both further down still.)
LDFLAGS = -lm
LDFLAGS += $(EXTRA_LDFLAGS)

# --- Command-line override rejection ------------------------------------
# See audio_common's Makefile for the full writeup: a command-line
# CFLAGS=/CXXFLAGS=/CPPFLAGS=/LDFLAGS=/FP_POLICY=/FP_ALLOWED_CHARS_RE= silently
# disables every one of this file's own assignments/appends to that variable
# (here that would strip -ffp-contract=off, -DAUDIO_PIPELINE_BACKEND_STR, the
# -I paths, and -lm) while still producing artifacts -- and a command-line/
# `-e` override of FP_ALLOWED_CHARS_RE specifically could widen the allowed
# character set (e.g. to `.*`, matching anything) and silently neuter the
# allow-list below. EXTRA_CFLAGS / EXTRA_LDFLAGS are the supported hooks.
$(foreach v,CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FP_POLICY FP_ALLOWED_CHARS_RE,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy, backend define, include paths); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead)))

# Conflict detection (widened to CXXFLAGS/CPPFLAGS + the
# quote/escape/response-file rejection gate, mirroring
# audio_common's/AEC's/NR's identical fix): reject outright rather than let a
# caller silently re-enable contraction. Checked against the fully-assembled
# $(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) (each already includes EXTRA_CFLAGS folded
# in above, AND any CFLAGS=/CXXFLAGS=/CPPFLAGS= command-line override, since a
# command-line-set value is what every prior `+=`/`=` in this file appended
# to) -- BEFORE FP_POLICY is appended below and BEFORE CFG_SIG is computed, so
# a rejected build never creates an obj/bin directory.
#
# CORRECTED UNDERSTANDING (this used to check $(CFLAGS) alone, reasoning that
# this repo has no C++ TUs of its own so CXXFLAGS/CPPFLAGS "didn't matter" --
# the same gap exists in AEC's/NR's Makefiles too: CXXFLAGS/CPPFLAGS are
# still two of the five names the
# "Command-line override rejection" foreach above rejects a command-line/`-e`
# override of, and both already participate in CFG_SIG_PAYLOAD above -- a
# PLAIN ENVIRONMENT CXXFLAGS or CPPFLAGS is deliberately allowed to fold in
# normally via that same foreach's origin logic, never going through
# EXTRA_CFLAGS at all, so `env CXXFLAGS=-Ofast make BACKEND=kiss lib` used to
# sail straight through this block entirely: no check here ever inspected
# $(CXXFLAGS), so the day either variable starts feeding a real compile
# recipe in this file, a live -Ofast/-ffast-math/-ffp-contract=<x> could ride
# in completely unchecked. The check below inspects all three variables
# together so no combination of CFLAGS/CXXFLAGS/CPPFLAGS -- command-line-
# folded, EXTRA_CFLAGS-folded, or a plain environment value -- can carry any
# of these three patterns past this gate. This position (after every
# CFLAGS/CXXFLAGS/CPPFLAGS append earlier in this file and before FP_POLICY's
# own append just below) is the same position already relied on to see
# CFLAGS's fully-composed value, and is equally late enough for CXXFLAGS and
# CPPFLAGS: neither is ever appended to anywhere else in this file (each is a
# single bare `+=` at the top of this file -- see that line's own comment),
# both fully resolved well before this point.
#
# (exact-token matching, not substring matching, applied here from the start)
# $(findstring ...) does a plain SUBSTRING search over the whole concatenated
# text, so it false-positives on any token that merely CONTAINS one of these
# patterns as part of a larger identifier -- e.g. a harmless
# `-DROUND9_NOTE=-Ofastness` macro definition is not the compiler flag
# -Ofast at all, yet a naive $(findstring -Ofast,...) check (as this block
# used to be, and as -DTEXT=ffast-math would equally false-positive on
# $(findstring -ffast-math,...)) would reject it anyway. $(filter
# PATTERN...,TEXT) instead splits TEXT on whitespace into WORDS and matches
# each whole word against PATTERN (a trailing %% is a single wildcard, used
# here for -ffp-contract=<anything>), so it only ever matches a complete flag
# token, never a substring buried inside an unrelated token.
#
# EXCLUSION, specific to this repo: unlike audio_common's/AEC's/NR's CFLAGS,
# THIS file's own CFLAGS append (above) unconditionally includes one
# Makefile-INTERNAL token that legitimately contains a backslash AND a
# double-quote character: `-DAUDIO_PIPELINE_BACKEND_STR=\"$(BACKEND)\"`. That
# backslash-escaped quoting is the entire POINT of the flag (audio_pipeline.c
# reads AUDIO_PIPELINE_BACKEND_STR as a real C string literal and compares it
# against "kiss"/"ne10" via strcmp() -- see that flag's own comment above);
# the shell that runs this Makefile's compile recipes unescapes `\"` to a
# literal `"` so the compiler receives a genuinely quoted string token, the
# same well-known idiom countless Makefiles use to inject a stringified
# #define. Verified empirically: without this exclusion, a plain, override-
# free `make all` (nothing external supplied) tripped the double-quote/
# backslash rejection checks below on this file's OWN default CFLAGS, before
# any caller-supplied EXTRA_CFLAGS/CXXFLAGS/CPPFLAGS ever entered the
# picture -- a false positive against this repo's own well-formed, unchanging
# default build, not a caller-supplied shell-quoting bypass attempt (the
# actual threat this gate exists to catch). This token can never carry a
# caller-supplied conflicting flag past the gate below either: it is
# Makefile-generated text with a fixed, known shape
# (`-DAUDIO_PIPELINE_BACKEND_STR=` followed by whatever $(BACKEND) is), never
# itself one of -Ofast/-ffast-math/-ffp-contract=%, so excluding it from the
# scan changes nothing about what this gate is able to catch. $(filter-out
# PATTERN,TEXT) removes every whole WORD in TEXT that matches PATTERN (same
# whole-word semantics as the $(filter) conflict check further below), so
# only this one exact-prefix token is ever removed -- every other word in
# CFLAGS/CXXFLAGS/CPPFLAGS (base flags, EXTRA_CFLAGS, and any command-line/
# environment override of any of the three) is still scanned in full below.
#
# RE-CONFIRMED for the allow-list redesign below (this task): the allow-list
# ([A-Za-z0-9_./=,+ -], see FP_ALLOWED_CHARS_RE) is JUST AS strict about
# backslash and double-quote as the old deny-list was -- neither character is
# in the allowed set, so a naive allow-list scan over the UNFILTERED CFLAGS
# would reject this repo's own default build exactly the same way the old
# deny-list did. This exclusion therefore still has to run first and still
# has to feed the SAME FP_INPUT_FLAGS into every check below (the single-quote
# gate, the allow-list's own $(shell) call, and the FP_CONFLICT_FLAGS filter),
# not just the ones the old deny-list happened to have -- re-verified
# empirically (this task): without it, a plain override-free `make all` trips
# the allow-list's generic "outside the allowed set" $(error) on this file's
# own default CFLAGS (naming `\` and `"` as the disallowed characters), before
# any caller-supplied input is involved. Nothing about the exclusion's SHAPE
# changed -- it is still scoped to the one exact-prefix token
# `-DAUDIO_PIPELINE_BACKEND_STR=%`, Makefile-generated with a fixed, known
# shape, never itself something a caller controls, so it cannot be used to
# smuggle a real conflicting/unsafe flag past either the old deny-list or the
# new allow-list.
FP_INPUT_FLAGS := $(filter-out -DAUDIO_PIPELINE_BACKEND_STR=%,$(CFLAGS) $(CXXFLAGS) $(CPPFLAGS))

# --- link-flags character-safety coverage (replicated from
# audio_common's identical fix -- see audio_common's Makefile for the full
# writeup this comment summarizes): the allow-list redesign below originally
# validated FP_INPUT_FLAGS only -- CFLAGS/CXXFLAGS/CPPFLAGS -- leaving LDFLAGS
# (built from a plain `-lm` literal plus whatever EXTRA_LDFLAGS folds in, see
# the `LDFLAGS += $(EXTRA_LDFLAGS)` line above) completely unchecked, even
# though LDFLAGS is embedded in every real link recipe below ($(LINK) ...
# $(LDFLAGS), see the four aec_nr_pipeline*/test_audio_pipeline/
# example_board_adapter link recipes further down) exactly the same way
# CFLAGS/CXXFLAGS are embedded in every compile recipe. Confirmed empirically
# (this task): `make aec_nr_pipeline EXTRA_LDFLAGS=';id'` (with AC_LIB/
# AEC_LIB/NR_LIB pre-resolved so the two-phase producer dispatch further down
# this file lands directly on this repo's OWN link recipe instead of
# recursing into audio_common's -- already-fixed -- sub-make, which would
# otherwise reject the very same payload first and mask this file's own gap)
# reached a real `cc -o ... $(LDFLAGS)` recipe line with the `;id` payload
# sitting live and unfiltered after the `-lm` -- confirmed via `make -n`
# (dry-run only, zero files touched) -- the exact same class of exposure the
# allow-list below exists to close for CFLAGS/CXXFLAGS/CPPFLAGS, just left
# open on this one variable. Fixed by introducing this union: the two checks
# that only care about CHARACTER safety (the single-quote pre-check and the
# allow-list itself, immediately below) now run against
# SHELL_SAFE_INPUT_FLAGS instead of FP_INPUT_FLAGS directly, so LDFLAGS/
# EXTRA_LDFLAGS content goes through the exact same grep-based gate with no
# separate/duplicated $(shell) invocation. The FP-semantic conflict check
# (FP_CONFLICT_FLAGS, further below -- -Ofast/-ffast-math/-ffp-contract=% are
# meaningless as LINK flags, not just unnecessary to check there) stays
# scoped to FP_INPUT_FLAGS alone, unchanged. The `-DAUDIO_PIPELINE_BACKEND_STR=%`
# exclusion above is unaffected: it is already applied to FP_INPUT_FLAGS
# before this union folds LDFLAGS in, and LDFLAGS itself never carries that
# internal token, so the exclusion still applies to exactly the one token it
# always did, on the CFLAGS/CXXFLAGS/CPPFLAGS side only.
SHELL_SAFE_INPUT_FLAGS := $(FP_INPUT_FLAGS) $(LDFLAGS)

# $(filter) directly above does whole-WORD matching against
# GNU Make's OWN text representation of CFLAGS/CXXFLAGS/CPPFLAGS -- but Make
# has zero concept of shell quoting. A value like EXTRA_CFLAGS="'-Ofast'" is
# stored by Make as the literal 9-character text '-Ofast', quote characters
# included, which is not equal to the bare word -Ofast, so $(filter) can
# never match it -- yet when this same text later appears on an actual
# compile recipe line and that line is handed to $(SHELL), the shell strips
# the quote characters and the compiler genuinely receives -Ofast as a real
# argv token. Three confirmed-live bypass shapes: EXTRA_CFLAGS="'-Ofast'"
# (single-quoted), EXTRA_CFLAGS='"-ffast-math"' (double-quoted), and
# EXTRA_CFLAGS="-O'f'ast" (quote-split mid-token -- the shell concatenates
# the pieces back into -Ofast once it strips the quotes). A fourth family,
# @response-file syntax (EXTRA_CFLAGS=@flags.rsp), is unrelated to quoting
# but equally invisible to a plain whole-word check: many compilers treat a
# bare @file argv token as "read more arguments from this file", which could
# inject anything at all with zero text ever appearing in CFLAGS/
# EXTRA_CFLAGS itself.
#
# REDESIGN (replicated from audio_common's/AEC's/NR's identical fix -- see
# audio_common's Makefile for the full writeup this comment summarizes): the
# character-by-character DENY-list this block used to be (single quote,
# double quote, backslash, backtick, a literal "$(" sequence, semicolon,
# pipe, ampersand, and an @-prefixed response-file token) is structurally
# unable to keep up -- a live audit found it does NOT catch bare glob
# characters (*, ?, [, ]), tilde (~), or -- most seriously -- real shell
# REDIRECTION (>, <) or process substitution (<(...)): none of these are
# Make's own syntax, so Make cannot pre-resolve or reject them the way it
# happens to catch ${VAR}-style expansions (GNU Make treats ${...}
# identically to $(...) and resolves it itself before any check runs -- a
# different, unrelated mechanism from the deny-list). If a recipe line
# containing an unbanned '>' or '<' is ever actually executed by a real
# shell (not a dry run), it silently redirects the compiler invocation's own
# stdout/stdin to an arbitrary path -- a real, unconditional risk,
# independent of any -Ofast/-ffast-math/-ffp-contract semantics. Separately,
# the old "$(" entry was likely DEAD CODE for well-formed input all along:
# `make ... 'CFLAGS=-O2 $(shell echo INJECTED)'` lets GNU Make's OWN parser
# resolve the well-formed $(...) reference at command-line-parsing time --
# "INJECTED" lands in CFLAGS as plain text, no literal "$(" substring ever
# reaches the check -- while an UNBALANCED `$(unbalanced` (no closing paren)
# makes Make itself abort with its own "unterminated variable reference.
# Stop." before any makefile-level check runs at all. Either way the
# findstring-based "$(" check never actually fires for the cases it was
# meant to catch.
#
# Replaced here with a single ALLOW-list covering the input as a whole. GNU
# Make has no native regex/character-class function, so this necessarily
# shells out to `grep -E` -- a $(shell ...) call, the same technique this
# file already uses for CFG_SIG's cksum computation below. Embedding
# untrusted, Make-supplied text inside a constructed shell command is
# exactly the kind of thing that can reintroduce the very problem being
# fixed if done carelessly, so the embedding strategy is deliberately narrow
# and was verified directly (not assumed) by audio_common's own phase of
# this fix: SHELL_SAFE_INPUT_FLAGS (CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS, fully-
# assembled -- see the "link-flags character-safety coverage" comment above
# for why LDFLAGS is folded in here too) is embedded SINGLE-QUOTED in the
# shell command text below (`printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | ...`).
# Inside a single-quoted POSIX shell string, EVERY character is literal except
# a single quote itself -- double quote, backslash, backtick, "$(", semicolon,
# pipe, and ampersand are all completely inert there. So the ONLY character
# that can break this embedding is a literal single quote in
# SHELL_SAFE_INPUT_FLAGS itself (which would close the quoted string early and
# hand whatever follows to the shell as live syntax) -- which is why the
# single-quote check below is the ONE survivor from the old 9-item deny-list,
# kept for this entirely different, structural reason (safe shell-embedding),
# not merely because it's also a disallowed character (the allow-list below
# would reject it either way). It MUST run first, and MUST stay a pure
# $(findstring) check with zero shell involvement, so it can gate the
# shell-embedding step that follows it.
ifneq ($(findstring ',$(SHELL_SAFE_INPUT_FLAGS)),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a single-quote character; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from the allowed character set (see the allow-list below) -- a single quote specifically must be rejected by this Make-native check BEFORE the allow-list's own $(shell) call can safely single-quote-embed the rest of the text)
endif

# The allow-list itself. Allowed set: letters, digits, underscore, dot,
# slash, equals, comma, plus, minus -- plus space (the token SEPARATOR
# between flags; SHELL_SAFE_INPUT_FLAGS is always multiple whitespace-joined
# tokens, never a single one, so space must be permitted even though it is
# not part of any individual flag's own character set). Chosen (same audit
# audio_common's/AEC's/NR's Makefile performed for its own identical
# allow-list) by checking every legitimate flag THIS Makefile's own
# invocations actually use: -O2/-Wall/-Wextra/-std=gnu99, -I$(AEC_DIR)/
# include/-I$(AEC_DIR)/example/-I$(NR_DIR)/include/-I$(AC_DIR)/include
# (letters/digits/underscore, and slash/dot for the paths), and (once the
# `-DAUDIO_PIPELINE_BACKEND_STR=%` exclusion above has removed the one known
# internal token that would otherwise trip it) every other flag this file
# appends by default -- every one of these tokens is built ONLY from this
# set. Comma and plus are included for headroom beyond this Makefile's
# current own usage (e.g. -Wl,-rpath,dir / -Wa,--option pass-through flags a
# consumer might legitimately need), on the same audit basis: neither
# character appears anywhere in this Makefile's own flags today, so
# including them cannot regress anything currently working, and both are
# inert to the shell (unlike the characters actually excluded below).
#
# LC_ALL=C avoids any locale-dependent surprises in the [A-Za-z0-9...]
# ranges (verified identical under C locale on this host's BSD grep/tr; the
# same invocation is expected to behave identically under GNU grep/tr on a
# Linux host, since POSIX bracket-expression ranges are locale-defined by
# design and C locale pins them to the unambiguous ASCII ordering on both).
#
# Every one of the FORMER 9-item deny-list's targets (single quote, double
# quote, backslash, backtick, a literal "$(", semicolon, pipe, ampersand,
# @-response-file) is a character NOT in this allowed set, so all of them
# are rejected by construction now. This repo has no standalone shell-test
# script for this check of its own separate from
# scripts/test_build_isolation.sh (see that script's S-series additions for
# this task) -- only the single-quote check above survives from the old
# deny-list, for the shell-embedding reason given there, not because the
# allow-list fails to cover it (a lone single quote is ALSO rejected by the
# allow-list on its own).
#
# FP_ALLOWED_CHARS_RE itself is NOT defined here -- it is a bare, early
# `:=` literal (see the "Bare-literal policy flag" / "Command-line override
# rejection" comments near the top of this file, where it is protected
# against a command-line/`-e` override the exact same way FP_POLICY is).
# Only referenced here.
SHELL_SAFE_ALLOWLIST_RC := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C grep -Eq '$(FP_ALLOWED_CHARS_RE)'; echo $$?)
ifneq ($(SHELL_SAFE_ALLOWLIST_RC),0)
# Diagnostic-only second $(shell) call (only ever runs on the already-doomed
# error path above, never on a passing build): deletes every ALLOWED
# character from the same single-quote-embedded text, leaving only the
# disallowed ones, then dedupes/sorts them into a short, stable string --
# so the $(error ...) below names the SPECIFIC offending character(s)
# (e.g. ">" or "~" or "*") instead of a generic "invalid format" message.
SHELL_SAFE_DISALLOWED_CHARS := $(shell printf '%s' '$(SHELL_SAFE_INPUT_FLAGS)' | LC_ALL=C tr -d 'A-Za-z0-9_./=,+ -' | fold -w1 | LC_ALL=C sort -u | tr -d '\n')
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS contains a character outside the allowed set [A-Za-z0-9_./=,+- ] (letters/digits/underscore/dot/slash/equals/comma/plus/minus, plus space as the token separator) -- disallowed character(s) found: "$(SHELL_SAFE_DISALLOWED_CHARS)"; this repo requires plain, unquoted, whitespace-separated compiler/linker argv tokens built only from that set -- remove/replace the disallowed character(s) shown above in CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/EXTRA_CFLAGS/EXTRA_LDFLAGS)
endif

FP_CONFLICT_FLAGS := $(filter -Ofast -ffast-math -ffp-contract=%,$(FP_INPUT_FLAGS))
ifneq ($(FP_CONFLICT_FLAGS),)
$(error FP policy conflict: CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS contains $(FP_CONFLICT_FLAGS); this repo pins -ffp-contract=off; remove $(FP_CONFLICT_FLAGS) from CFLAGS/CXXFLAGS/CPPFLAGS/EXTRA_CFLAGS)
endif

# Command-line/`-e` override rejection for the FP-policy/shell-safety checks'
# remaining FOUR internal variables (replicated from audio_common's/AEC's/
# NR's identical fix, found live while audio_common implemented the allow-list
# above; SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC added, same treatment,
# when the character-safety check was widened to
# LDFLAGS -- see the "link-flags character-safety coverage" comment above):
# FP_INPUT_FLAGS/SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC/
# FP_CONFLICT_FLAGS are each computed from the fully-assembled CFLAGS/
# CXXFLAGS/CPPFLAGS (SHELL_SAFE_INPUT_FLAGS/SHELL_SAFE_ALLOWLIST_RC also fold
# in the fully-assembled LDFLAGS; SHELL_SAFE_ALLOWLIST_RC via a $(shell) call
# keyed off SHELL_SAFE_INPUT_FLAGS) -- unlike FP_POLICY/FP_ALLOWED_CHARS_RE
# above, none of the four is a bare, dependency-free literal, so none can be
# relocated earlier to ride in the "Command-line override rejection" foreach
# near the top of this file (that foreach's own comment explains why
# bare-literal-and-early is required for the `-e` case). But each has the
# EXACT SAME command-line-override exposure as every name in that foreach:
# e.g. `make ... EXTRA_LDFLAGS=';echo INJECTED' SHELL_SAFE_ALLOWLIST_RC=0`
# (or the equivalent with SHELL_SAFE_INPUT_FLAGS=<anything clean>,
# FP_INPUT_FLAGS=<anything clean>, or FP_CONFLICT_FLAGS=) would silently win
# over this file's own computed value and turn every ifneq/$(error) gate
# above into a complete no-op -- while the REAL $(CFLAGS)/$(CXXFLAGS)/
# $(CPPFLAGS)/$(LDFLAGS) used in every actual compile/link recipe still carry
# the dangerous, unvalidated content unmodified. Fixed the same way as the
# `-e` gap elsewhere in this file, mirrored for a LATE-computed (not
# early-literal) variable: the override-rejection check must itself run
# AFTER the variable's own real assignment (never before), so the
# origin-flip -- for both plain "command line" and, under `-e`, "environment
# override" -- has already happened by the time $(origin ...) is queried
# here. This is why this second foreach sits HERE, immediately after
# FP_CONFLICT_FLAGS (the last of the four to be assigned), rather than folded
# into the earlier foreach (which runs too early in the file for any of
# these four to have been assigned yet). SHELL_SAFE_DISALLOWED_CHARS (like
# FP_DISALLOWED_CHARS before it) is deliberately NOT included here: it is
# pure diagnostic text folded into the $(error ...) message on the failure
# path only, never a condition any ifneq tests, so overriding it cannot
# bypass anything -- at worst it would garble the reported character list on
# a build that was going to fail anyway.
$(foreach v,FP_INPUT_FLAGS SHELL_SAFE_INPUT_FLAGS SHELL_SAFE_ALLOWLIST_RC FP_CONFLICT_FLAGS,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently defeat the FP-policy/shell-safety allow-list/conflict checks above, leaving the real CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS content unvalidated; this variable is internal to those checks and has no supported external use)))

# (`make -e` gap fix: FP_POLICY's own bare `:=` literal now
# lives earlier in this file, directly above the LDFLAGS composition and the
# "Command-line override rejection" foreach -- see that literal's "Bare-
# literal policy flag" comment for why. Only FP_POLICY's OTHER job --
# appending it to CFLAGS, LAST, exactly as the "FP-contraction policy"
# comment above describes -- still lives at this position; there is exactly
# one `:=` definition of FP_POLICY in this file, not two.)
CFLAGS     += $(FP_POLICY)

# The NE10 archive carries one C++ TU (NE10's generic-radix FFT kernel); a
# plain-C pipeline binary linking against it uses the C++ DRIVER, which
# supplies the correct C++ runtime itself (libc++ on macOS/clang, libstdc++
# on GNU/Linux gcc). The old hardcoded `-lc++` alongside it was a macOS-ism
# (duplicate there, a link error on GNU/Linux) and is gone. LINK
# participates in CFG_SIG below.
LINK = $(CC)
ifeq ($(BACKEND),ne10)
  LINK = $(CXX)
endif

.DEFAULT_GOAL := all

# =============================================================================
# Consumer two-phase, THREE-producer resolution
# =============================================================================
# Problem this replaces: the old Makefile hardcoded
#   LIBS = ./libaudio_pipeline.a $(AEC_DIR)/bin/libaec.a \
#          $(NR_DIR)/bin/libmmse_lsa.a $(AC_DIR)/bin/$(BACKEND)/libaudio_common.a
# and a `libs:` phony that unconditionally ran three sub-makes with only
# BACKEND/EXTRA_CFLAGS/WERROR forwarded. Two problems: (1) all three
# producers' own bin/ dirs are now CFG_SIG-keyed, so there is no fixed path
# to hardcode any more; (2) config skew -- a producer built with a different
# CC/CXX/NO_STDIO than THIS invocation's own could be silently linked in
# without that showing up in this repo's own build identity.
#
# The fix is the same two-phase recursive-make dispatch AEC/NR use for their
# ONE producer, extended to THREE, driven by whether AC_LIB is already known:
#
#   Phase 1 (AC_LIB unset): a goal that needs any producer archive gets a
#   recipe that resolves audio_common's archive path first (needed by both
#   lib/aec's and lib/nr's OWN two-phase dispatch -- see the AC_DIR=/AC_LIB=
#   forwarding below), then lib/aec's, then lib/nr's, and re-invokes itself
#   with all three set.
#
#   Phase 2 (AC_LIB known): the REAL rules are defined, with $(AC_LIB)/
#   $(AEC_LIB)/$(NR_LIB) as NORMAL (not order-only) prerequisites of the
#   binaries that link against them -- so relinking happens iff a producer's
#   archive mtime actually advanced.
#
# Goals that never need any producer archive (`libaudio_pipeline.a` -- this
# `clean`/`clean-all` and the help target get their real rules OUTSIDE this
# dispatch entirely, further down this file. `libaudio_pipeline.a` (and the
# audit targets that build+inspect it) DO go through the dispatch: the
# delivered archive is the one a consumer links, and consumers always
# resolve the producer identities -- so the archive's keyed path must fold
# them in too (a bare no-producer build would land in a different directory
# than every query reports).
#
# `make -n`: see AEC's/NR's Makefile comments for why the dispatch recipe's
# sub-`make` calls are genuinely spawned even under `make -n` (GNU Make
# always executes, never merely prints, a recipe line that textually
# mentions $(MAKE)) yet still create zero files (verified empirically): -n
# propagates to every child `make` via MAKEFLAGS regardless of how it was
# invoked, so each child's own recipes are themselves just printed.
#
# lib/aec's and lib/nr's own Makefiles fold AC_LIB into THEIR OWN CFG_SIG
# (as AC_PRODUCER_SIG) REGARDLESS of which of their targets is being built --
# even their `lib` target, which itself never links audio_common's archive
# (see those Makefiles: AC_PRODUCER_SIG is computed unconditionally whenever
# AC_LIB is non-empty, before their own CFG_SIG_PAYLOAD is assembled). So
# AC_DIR=$(abspath $(AC_DIR)) AC_LIB=<resolved path> MUST be forwarded to
# every lib/aec / lib/nr sub-make below (both the query calls and the real
# `lib` build calls in the FORCE rules further down) -- otherwise a `lib`
# build invoked without AC_LIB would land in a DIFFERENT (AC_PRODUCER_SIG-less)
# keyed directory than the one this Makefile's query call resolved, and the
# "does $(AEC_LIB) exist" assertion in that FORCE rule would fail.
FWD = BACKEND=$(BACKEND) SIMD=$(SIMD) PROFILE=$(PROFILE) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) DEBUG=$(DEBUG)

ifeq ($(AC_LIB),)
.PHONY: all test libs _publish_impl aec_nr_pipeline aec_nr_pipeline_static mono_alignulcnet test_audio_pipeline test_audio_pipeline_ulcnet test_ulcnet_accelerator_adapter example-adapter libaudio_pipeline.a
all test libs _publish_impl aec_nr_pipeline aec_nr_pipeline_static mono_alignulcnet test_audio_pipeline test_audio_pipeline_ulcnet test_ulcnet_accelerator_adapter example-adapter libaudio_pipeline.a:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) $(FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audio_common archive path (AC_DIR=$(AC_DIR) -- is audio_common present as a sibling of Audio_ALG?)" >&2; exit 1; }; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 test -n "$$aec" || { echo "FATAL: could not resolve lib/aec archive path (AEC_DIR=$(AEC_DIR))" >&2; exit 1; }; \
	 nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 test -n "$$nr" || { echo "FATAL: could not resolve lib/nr archive path (NR_DIR=$(NR_DIR))" >&2; exit 1; }; \
	 $(MAKE) --no-print-directory $@ AC_LIB="$$ac" AEC_LIB="$$aec" NR_LIB="$$nr"
else
# Producer-identity components of this consumer's own CFG_SIG (see AEC's/
# NR's Makefiles for the full rationale): the LAST path component of each
# resolved archive's directory (e.g. "kiss-2139547423"), NEVER the full
# (absolute) path -- so a pipeline build against a DIFFERENTLY-configured
# producer keys to its own separate obj/bin directory here too, instead of
# risking silent config skew. AC_DIR/AEC_DIR/NR_DIR themselves (abspaths,
# used only as -C arguments to the sub-makes) never appear in any payload or
# manifest -- only these short backend-sig components do.
AC_PRODUCER_SIG  := $(notdir $(patsubst %/,%,$(dir $(AC_LIB))))
AEC_PRODUCER_SIG := $(notdir $(patsubst %/,%,$(dir $(AEC_LIB))))
NR_PRODUCER_SIG  := $(notdir $(patsubst %/,%,$(dir $(NR_LIB))))
endif

# --- Hash-keyed object/binary directory + full-coverage CFG_SIG ------------
# obj/<backend>-<sig> AND bin/<backend>-<sig> -- see audio_common's Makefile
# for the full design rationale (identical mechanism, applied here). Payload
# coverage (USER-MANDATED): CC CXX AR RANLIB CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
# EXTRA_CFLAGS EXTRA_LDFLAGS BACKEND WERROR NO_STDIO DEBUG, PLUS (consumer-
# specific, THREE producers) AC_PRODUCER_SIG/AEC_PRODUCER_SIG/NR_PRODUCER_SIG
# -- each empty when its _LIB var isn't resolved yet (e.g. for goals like
# `libaudio_pipeline.a`/`audit-no-stdio` that never touch any producer).
#
# This MUST be defined after CFLAGS/LDFLAGS are fully assembled and after the
# AC_LIB ifeq/else above resolves the three PRODUCER_SIGs, but before the
# first rule below that mentions $(OBJ_DIR)/$(BIN_DIR)/anything derived from
# them -- GNU Make expands a rule's target/prerequisite list immediately
# when parsed, not deferred like a recipe body, so a forward reference here
# would silently resolve using CFG_SIG's value at parse time (empty),
# landing objects in a hash-less directory instead of the real one.
#
# Deliberately NOT $(strip)'d (false-miss OK, false-hit never -- see
# audio_common's Makefile for the full rationale). cksum (POSIX) computes
# the hash; `cut -d' ' -f1` isolates the checksum from cksum's
# "<checksum> <byte-count>" stdout.
# Payload additions beyond the base set: LINK, TOOLCHAIN_CHECK (the
# guard-skip knob must not alias a guarded build), SRCS (this
# Makefile's own compiled sources -- the archive's one member TU plus the
# CLI/test TUs -- so a source-set change keys to a fresh directory).
CFG_SIG_PAYLOAD := CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) LDFLAGS=$(LDFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) SIMD=$(SIMD) PROFILE=$(PROFILE) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) AC_PRODUCER_SIG=$(AC_PRODUCER_SIG) AEC_PRODUCER_SIG=$(AEC_PRODUCER_SIG) NR_PRODUCER_SIG=$(NR_PRODUCER_SIG) SRCS=$(sort mono_aec_nr_res/audio_pipeline.c mono_aec_nr_res/main.c mono_aec_nr_res/static_main.c mono_aec_nr_res/tests/test_audio_pipeline.c mono_aec_nr_res/example_board_adapter.c mono_alignulcnet/audio_pipeline_ulcnet.c mono_alignulcnet/tests/test_audio_pipeline_ulcnet.c $(ULCNET_DIR)/ulcnet_accelerator_adapter.c $(ULCNET_DIR)/ulcnet_process.c $(ULCNET_DIR)/ulcnet_model_io.c $(AIAEC_DIR)/tests/test_ulcnet_accelerator_adapter.c mono_alignulcnet/main.c)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

# OBJ_ROOT / BIN_ROOT: where THIS repo's own keyed
# obj/bin trees live (defaults: obj/ and bin/ in this directory -- the
# default expansion is byte-identical to the previous hardcoded paths).
# Same rationale as DIST_ROOT above: a placement knob, never a build-content
# knob, so deliberately NOT in CFG_SIG. The isolation test scripts use these
# to run tamper/injection scenarios against a scratch-directory build of the
# real worktree, so the REAL obj/ and bin/ artifacts here are never
# modified -- not even transiently. (Each of the three producers --
# audio_common, lib/aec, lib/nr -- keys its OWN obj/bin trees independently;
# this pair only covers this Makefile's own objects/binaries.)
OBJ_ROOT ?= obj
BIN_ROOT ?= bin
OBJ_DIR  = $(OBJ_ROOT)/$(BACKEND)-$(CFG_SIG)
BIN_DIR  = $(BIN_ROOT)/$(BACKEND)-$(CFG_SIG)

# --- CFG_SIG collision guard + directory creation ---------------------------
# Single order-only prerequisite for every compile/link/archive rule below.
# See audio_common's Makefile for the full design writeup (identical
# mechanism, applied here) -- write config.manifest first time, verify (hard
# error on mismatch = CFG_SIG collision or payload coverage gap) every time
# after, atomically (temp file + mv).
.PHONY: _cfg_guard
_cfg_guard:
	@if [ "$(BACKEND)" = "ne10" ] && [ "$(TOOLCHAIN_CHECK)" = "1" ]; then \
	  cc_m="$$($(CC) -dumpmachine 2>/dev/null)"; cxx_m="$$($(CXX) -dumpmachine 2>/dev/null)"; \
	  if [ -z "$$cc_m" ] || [ "$$cc_m" != "$$cxx_m" ]; then \
	    echo "FATAL: CC/CXX compile for different targets (CC='$(CC)' -> '$$cc_m', CXX='$(CXX)' -> '$$cxx_m')." >&2; \
	    echo "  BACKEND=ne10 links a C++ TU (audio_common's NE10 archive); pass the FULL toolchain:" >&2; \
	    echo "    make BACKEND=ne10 CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ AR=aarch64-linux-gnu-ar RANLIB=aarch64-linux-gnu-ranlib" >&2; \
	    echo "  (TOOLCHAIN_CHECK=0 skips this check; it participates in CFG_SIG.)" >&2; \
	    exit 1; \
	  fi; \
	fi
	@mkdir -p $(OBJ_DIR) $(BIN_DIR)
	@for d in $(OBJ_DIR) $(BIN_DIR); do \
	  m="$$d/config.manifest"; \
	  if [ -f "$$m" ]; then \
	    old="$$(cat "$$m")"; \
	    if [ "$$old" != "$(CFG_SIG_PAYLOAD)" ]; then \
	      echo "FATAL: CFG_SIG collision or coverage gap in $$m" >&2; \
	      echo "  CFG_SIG=$(CFG_SIG)" >&2; \
	      echo "  stored : $$old" >&2; \
	      echo "  current: $(CFG_SIG_PAYLOAD)" >&2; \
	      exit 1; \
	    fi; \
	  else \
	    tmp="$$d/.config.manifest.tmp.$$$$"; \
	    printf '%s\n' "$(CFG_SIG_PAYLOAD)" > "$$tmp"; \
	    mv -f "$$tmp" "$$m"; \
	  fi; \
	done

# -MD -MP: emits $(OBJ_DIR)/<name>.d per object (header dependency tracking)
# -- included near the bottom of this file via `-include $(wildcard $(OBJ_DIR)/*.d)`.
$(OBJ_DIR)/aec_nr_pipeline.o: mono_aec_nr_res/main.c mono_aec_nr_res/audio_pipeline.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/aec_nr_pipeline_static.o: mono_aec_nr_res/static_main.c mono_aec_nr_res/audio_pipeline.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# audio_pipeline.o / libaudio_pipeline.a: the pool-sizing/
# carving/per-hop-processing library TU, linkable independently of either
# CLI -- see audio_pipeline.h. The archive holds ONLY this one object (not a
# fat archive bundling libaec/libmmse_lsa/libaudio_common); those stay
# separate libraries a consumer links alongside it, same as both CLIs do.
# Compiled with $(LIB_CFLAGS) (not the shared $(CFLAGS)) so NO_STDIO=1 only
# ever affects THIS object -- see the NO_STDIO comment near CFLAGS above.
$(OBJ_DIR)/audio_pipeline.o: mono_aec_nr_res/audio_pipeline.c mono_aec_nr_res/audio_pipeline.h mono_aec_nr_res/pipeline_dims.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

# test_audio_pipeline: standalone acceptance test for audio_pipeline.h itself
# (create-vs-init byte equality incl. a dirty/poisoned pool, destroy
# idempotence, misaligned/undersized pool rejection, sample-rate whitelist
# rejection) -- see test_audio_pipeline.c's own header for detail.
$(OBJ_DIR)/test_audio_pipeline.o: mono_aec_nr_res/tests/test_audio_pipeline.c mono_aec_nr_res/audio_pipeline.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# Align-ULCNet pipeline variant objects. ulcnet_process.o compiles the
# AIAEC-owned C pre/post TU (centered sqrt-Hann STFT / WOLA / model
# boundary) -- read-only from this Makefile's perspective, never edited
# here. It transforms through audio_common's fft_wrapper (so BACKEND=
# kiss/ne10 genuinely selects the ULCNet FFT backend); the -I$(AC_DIR)/
# include already in $(CFLAGS) resolves that header, and -I$(ULCNET_DIR)
# resolves its own. The pipeline TU and its test also need the ULCNET
# include path for ulcnet_process.h (via audio_pipeline_ulcnet.h).
$(OBJ_DIR)/ulcnet_process.o: $(ULCNET_DIR)/ulcnet_process.c $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_model_io.o: $(ULCNET_DIR)/ulcnet_model_io.c $(ULCNET_DIR)/ulcnet_model_io.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/ulcnet_accelerator_adapter.o: $(ULCNET_DIR)/ulcnet_accelerator_adapter.c $(ULCNET_DIR)/ulcnet_accelerator_adapter.h $(ULCNET_DIR)/ulcnet_model_io.h $(ULCNET_DIR)/ulcnet_process.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/mono_alignulcnet_main.o: mono_alignulcnet/main.c mono_alignulcnet/audio_pipeline_ulcnet.h $(ULCNET_DIR)/ulcnet_accelerator_adapter.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

$(OBJ_DIR)/test_ulcnet_accelerator_adapter.o: $(AIAEC_DIR)/tests/test_ulcnet_accelerator_adapter.c $(ULCNET_DIR)/ulcnet_accelerator_adapter.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

# audio_pipeline_ulcnet.o: the AEC(linear) -> Align-ULCNet post-filter
# pipeline TU (audio_pipeline_ulcnet.h). It is deliberately not a member of
# libaudio_pipeline.a. ULCNet applications link this object and the AIAEC
# pre/post object directly alongside libaec/libaudio_common, avoiding a
# second application-specific wrapper archive.
$(OBJ_DIR)/audio_pipeline_ulcnet.o: mono_alignulcnet/audio_pipeline_ulcnet.c mono_alignulcnet/audio_pipeline_ulcnet.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

# test_audio_pipeline_ulcnet: standalone acceptance test for
# audio_pipeline_ulcnet.h (identity E2E one-hop-latency proof, model
# callback counting/reset policy, fail-open + delay-gating, pool/descriptor
# rejection, create-vs-init parity, NULL-model == identity-model) -- see
# tests/test_audio_pipeline_ulcnet.c's own header for detail.
$(OBJ_DIR)/test_audio_pipeline_ulcnet.o: mono_alignulcnet/tests/test_audio_pipeline_ulcnet.c mono_alignulcnet/audio_pipeline_ulcnet.h | _cfg_guard
	$(CC) $(CFLAGS) -I$(ULCNET_DIR) -MD -MP -c -o $@ $<

# example_board_adapter: REFERENCE ONLY host simulation of a board-side
# common-memory-manager integration -- see that
# file's own header. Not the real board adapter; demonstrates the calling
# sequence/error handling/descriptor contract only.
$(OBJ_DIR)/example_board_adapter.o: mono_aec_nr_res/example_board_adapter.c mono_aec_nr_res/audio_pipeline.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

LIB_TARGET = $(BIN_DIR)/libaudio_pipeline.a

# `libaudio_pipeline.a` needs only THIS repo's own object (audio_pipeline.o)
# -- no producer archive is linked into it -- so, per the dispatch design
# above, it's a real rule living entirely outside the AC_LIB dispatch (never
# shells out to any of the three producers, works even if none of them are
# buildable). The phony alias keeps the pre-B01 flat-filename invocation
# (`make libaudio_pipeline.a`) working.
.PHONY: libaudio_pipeline.a
libaudio_pipeline.a: $(LIB_TARGET)

# Fresh-archive discipline: always built from scratch
# into a temp file and renamed into place -- never `ar` onto an existing
# archive (which only adds/replaces members and would keep a removed source's
# stale member forever). See audio_common's Makefile for the full writeup.
# One chained shell line with a PID-suffixed temp: two
# same-config processes racing this rule each write their OWN temp, final mv
# is atomic last-writer-wins. See audio_common's Makefile.
$(LIB_TARGET): $(OBJ_DIR)/audio_pipeline.o | _cfg_guard
	t="$@.tmp.$$$$" && rm -f "$$t" && $(AR) rcs "$$t" $^ && $(RANLIB) "$$t" && mv -f "$$t" $@
	@echo "libaudio_pipeline.a [$(BACKEND)] -> $@"

# =============================================================================
# Phase 2 real rules: only defined once AC_LIB (and, transitively, AEC_LIB/
# NR_LIB, always set together by phase 1's re-invoke) is known.
# =============================================================================
ifneq ($(AC_LIB),)
.PHONY: all test libs aec_nr_pipeline aec_nr_pipeline_static mono_alignulcnet test_audio_pipeline test_audio_pipeline_ulcnet test_ulcnet_accelerator_adapter example-adapter example_board_adapter FORCE
all: $(BIN_DIR)/aec_nr_pipeline $(BIN_DIR)/aec_nr_pipeline_static $(BIN_DIR)/mono_alignulcnet

# test: also builds + runs example_board_adapter --
# wired into the SAME gate as test_audio_pipeline (not left standalone-only)
# because it is clean: green on both backends, exits 0 on full success, and
# its own failure modes (a positive-flow step or negative-demonstration
# check failing) are exactly the kind of regression this gate exists to
# catch. `make example-adapter` (below) remains available standalone too.
test: $(BIN_DIR)/test_audio_pipeline $(BIN_DIR)/example_board_adapter $(BIN_DIR)/test_audio_pipeline_ulcnet $(BIN_DIR)/test_ulcnet_accelerator_adapter $(BIN_DIR)/mono_alignulcnet
	"$(abspath $(BIN_DIR))/test_audio_pipeline"
	"$(abspath $(BIN_DIR))/example_board_adapter"
	"$(abspath $(BIN_DIR))/test_audio_pipeline_ulcnet"
	"$(abspath $(BIN_DIR))/test_ulcnet_accelerator_adapter"
	"$(abspath $(BIN_DIR))/mono_alignulcnet"
	"$(abspath $(BIN_DIR))/mono_alignulcnet" --delay-num-filters 1

# example-adapter: build + RUN the REFERENCE ONLY
# board-adapter host simulation (example_board_adapter.c) standalone -- same
# build-then-run shape as `test:` above. Its own exit code is the smoke-test
# signal: nonzero if any positive-flow step or negative-demonstration check
# failed, so a red `make example-adapter` fails the build the same way a red
# `make test` does.
example-adapter: $(BIN_DIR)/example_board_adapter
	"$(abspath $(BIN_DIR))/example_board_adapter"

# Root-name alias for the binary itself (mirrors test_audio_pipeline: below)
# -- `example-adapter` (above) is the build-AND-run entry point; this is just
# "build the binary" for whoever wants that split.
example_board_adapter: $(BIN_DIR)/example_board_adapter

# libs: compat alias for the old `libs:` phony (which used to build the
# three producer archives directly with hardcoded paths/flags). Building the
# archives is now an automatic side effect of the FORCE rules below, so this
# just asserts all three exist for whoever's tooling still invokes `make libs`.
libs: $(AC_LIB) $(AEC_LIB) $(NR_LIB)

# Root-name phony aliases: keep the pre-keyed-directory
# invocation (`make aec_nr_pipeline`, etc.) working -- the real artifact now
# lives at $(BIN_DIR)/<name>, not flat in this directory.
aec_nr_pipeline: $(BIN_DIR)/aec_nr_pipeline
aec_nr_pipeline_static: $(BIN_DIR)/aec_nr_pipeline_static
mono_alignulcnet: $(BIN_DIR)/mono_alignulcnet
test_audio_pipeline: $(BIN_DIR)/test_audio_pipeline
test_audio_pipeline_ulcnet: $(BIN_DIR)/test_audio_pipeline_ulcnet
test_ulcnet_accelerator_adapter: $(BIN_DIR)/test_ulcnet_accelerator_adapter

FORCE:

# Each producer archive is a FORCE-recipe target: always re-invokes that
# producer's own build (a no-op if it's already up to date -- that
# producer's own Makefile decides that), but only actually touches the
# archive's mtime if it really rebuilt something, so the binaries below
# only relink when it did. AC_DIR=$(abspath $(AC_DIR)) AC_LIB=$(AC_LIB) is
# forwarded to the lib/aec/lib/nr sub-makes even though `lib` itself never
# links audio_common's archive -- see the two-phase dispatch comment above
# for why this is required for their CFG_SIG (hence output path) to match
# what the query call above already resolved.
$(AC_LIB): FORCE
	+$(MAKE) -C $(AC_DIR) $(FWD) lib
	@test -f '$@' || { echo "ERROR: audio_common did not produce $@ (query/build divergence -- supported forwarded controls are BACKEND/SIMD/WERROR/CC/CXX/EXTRA_CFLAGS/NO_STDIO/TOOLCHAIN_CHECK)"; exit 1; }

$(AEC_LIB): FORCE
	+$(MAKE) -C $(AEC_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB=$(AC_LIB) lib
	@test -f '$@' || { echo "ERROR: lib/aec did not produce $@ (query/build divergence)"; exit 1; }

$(NR_LIB): FORCE
	+$(MAKE) -C $(NR_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB=$(AC_LIB) lib
	@test -f '$@' || { echo "ERROR: lib/nr did not produce $@ (query/build divergence)"; exit 1; }

# Compile with $(CC) (always a plain-C driver, -std=gnu99 valid) then link
# with $(LINK) (c++ for BACKEND=ne10 -- the NE10 archive's one C++ TU needs
# it). Link order matters (see the LDFLAGS comment above): pipeline.o,
# libaudio_pipeline.a, libaec.a, libmmse_lsa.a, libaudio_common.a, -lm.
$(BIN_DIR)/aec_nr_pipeline: $(OBJ_DIR)/aec_nr_pipeline.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/aec_nr_pipeline.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDFLAGS)
	@echo "aec_nr_pipeline [$(BACKEND)] -> $@"

$(BIN_DIR)/aec_nr_pipeline_static: $(OBJ_DIR)/aec_nr_pipeline_static.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/aec_nr_pipeline_static.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDFLAGS)
	@echo "aec_nr_pipeline_static [$(BACKEND)] -> $@"

$(BIN_DIR)/test_audio_pipeline: $(OBJ_DIR)/test_audio_pipeline.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_audio_pipeline.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDFLAGS)
	@echo "test_audio_pipeline [$(BACKEND)] -> $@"

# ULCNet applications link their wrapper/pre-post objects directly. They are
# applications, not additional libraries layered over the component libs.
ULCNET_PIPELINE_OBJS = $(OBJ_DIR)/audio_pipeline_ulcnet.o $(OBJ_DIR)/ulcnet_process.o

# The wrapper itself needs no ulcnet_model_io symbol (it only READS the
# descriptor a runtime published, so a board integrating the wrapper alone
# gains no new link dependency); the test links it to BUILD descriptors for
# the far-input contract gate.
$(BIN_DIR)/test_audio_pipeline_ulcnet: $(OBJ_DIR)/test_audio_pipeline_ulcnet.o $(ULCNET_PIPELINE_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(AEC_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_audio_pipeline_ulcnet.o $(ULCNET_PIPELINE_OBJS) $(OBJ_DIR)/ulcnet_model_io.o $(AEC_LIB) $(AC_LIB) $(LDFLAGS)
	@echo "test_audio_pipeline_ulcnet [$(BACKEND)] -> $@"

$(BIN_DIR)/mono_alignulcnet: $(OBJ_DIR)/mono_alignulcnet_main.o $(OBJ_DIR)/ulcnet_accelerator_adapter.o $(OBJ_DIR)/ulcnet_model_io.o $(ULCNET_PIPELINE_OBJS) $(AEC_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/mono_alignulcnet_main.o $(OBJ_DIR)/ulcnet_accelerator_adapter.o $(OBJ_DIR)/ulcnet_model_io.o $(ULCNET_PIPELINE_OBJS) $(AEC_LIB) $(AC_LIB) $(LDFLAGS)
	@echo "mono_alignulcnet [$(BACKEND)] -> $@"

$(BIN_DIR)/test_ulcnet_accelerator_adapter: $(OBJ_DIR)/test_ulcnet_accelerator_adapter.o $(OBJ_DIR)/ulcnet_accelerator_adapter.o $(OBJ_DIR)/ulcnet_model_io.o | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/test_ulcnet_accelerator_adapter.o $(OBJ_DIR)/ulcnet_accelerator_adapter.o $(OBJ_DIR)/ulcnet_model_io.o $(LDFLAGS)

$(BIN_DIR)/example_board_adapter: $(OBJ_DIR)/example_board_adapter.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/example_board_adapter.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDFLAGS)
	@echo "example_board_adapter [$(BACKEND)] -> $@"
endif

# --- Query targets -----------------------------------------------------
# Report THIS invocation's own build identity (empty AC_PRODUCER_SIG/
# AEC_PRODUCER_SIG/NR_PRODUCER_SIG unless AC_LIB/AEC_LIB/NR_LIB were
# explicitly passed in). Parse-time stdout purity: nothing above this point
# ever unconditionally $(info)/$(warning)s, so these targets' stdout is
# exactly their one recipe-time @echo line -- safe to capture with
# `$(MAKE) -s ... print-lib-path` the way the dispatch recipe above (and
# lib/aec's/lib/nr's own Makefiles, and this file's audit-no-stdio/publish
# recipes) all do. To get the exact path `make all` itself would produce
# (which DOES fold in the three resolved producers' identities), resolve
# AC_LIB/AEC_LIB/NR_LIB the same way phase 1 does and pass them explicitly.
.PHONY: print-bin-dir print-obj-dir print-lib-path
# print-bin-dir / print-lib-path / print-obj-dir all dispatch through the same
# three-producer phase-1 resolution as the build goals, so a bare
# `make -s print-<x>` prints the location that configuration actually uses
# (this repo's archive, binaries AND objects only exist in the
# resolved-producer directory). print-obj-dir did NOT dispatch before, so it
# answered for a different config signature than the one the build used --
# which made it useless for anything that has to find a just-built object.
# Passing AC_LIB/AEC_LIB/NR_LIB explicitly skips the query sub-makes.
ifeq ($(AC_LIB),)
print-bin-dir print-lib-path print-obj-dir:
	+@ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) $(FWD) print-lib-path)"; \
	 test -n "$$ac" || { echo "FATAL: could not resolve audio_common archive path (AC_DIR=$(AC_DIR))" >&2; exit 1; }; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) $(FWD) AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 $(MAKE) -s --no-print-directory $@ AC_LIB="$$ac" AEC_LIB="$$aec" NR_LIB="$$nr"
else
print-bin-dir:
	@echo $(abspath $(BIN_DIR))
print-lib-path:
	@echo $(abspath $(LIB_TARGET))
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))
endif

# test-fp-contract: disassemble the pipeline objects that run windowed
# overlap-add and assert they emitted no fma-class instruction. Part of the
# release gate: -ffp-contract=off is a compile flag, fma-free is a property of
# the generated code, and only this says which one actually happened. The
# script builds each object itself and queries print-obj-dir from the same
# configuration, so it can never audit a stale signature's leftovers.
.PHONY: test-fp-contract
test-fp-contract:
	@scripts/audit_fp_contract.sh $(BACKEND)

# audit-no-stdio (adapted to the keyed-directory layout):
# builds libaudio_pipeline.a with NO_STDIO=1 (current BACKEND=$(BACKEND))
# and asserts, via `nm` over the DELIVERED keyed archive itself (not an
# intermediate object file), that none of fprintf/printf/puts/fputs/stderr/
# __stderrp made it into it -- plus that the NO_STDIO=1
# archive's resolved path actually differs from the default (NO_STDIO=0)
# config's path, i.e. the two configs cannot silently collide/alias. `ar -t`
# confirms the archive still holds exactly the one member (audio_pipeline.o)
# nm's hits are attributed to. Pattern/style mirrors
# ../../audio_common/scripts/audit_alloc_symbols.sh (leading-underscore-
# tolerant regex, last-whitespace-field match against nm's NAME column, so
# it works the same under macOS/BSD nm and GNU nm). This IS a pass/fail gate
# (prints PASS/FAIL, non-zero exit on FAIL).
.PHONY: audit-no-stdio
audit-no-stdio:
	@echo "audit-no-stdio: building libaudio_pipeline.a (BACKEND=$(BACKEND), NO_STDIO=1)"
	@$(MAKE) --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 libaudio_pipeline.a >/dev/null
	@lib_nostdio="$$($(MAKE) -s --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 print-lib-path)"; \
	 lib_default="$$($(MAKE) -s --no-print-directory BACKEND=$(BACKEND) NO_STDIO=0 print-lib-path)"; \
	 if [ "$$lib_nostdio" = "$$lib_default" ]; then \
	   echo "FATAL: NO_STDIO=1 archive path COLLIDES with the default (NO_STDIO=0) archive path ($$lib_nostdio)" >&2; \
	   exit 1; \
	 fi; \
	 if [ ! -f "$$lib_nostdio" ]; then \
	   echo "FAIL: $$lib_nostdio not found after building libaudio_pipeline.a with NO_STDIO=1"; \
	   exit 1; \
	 fi; \
	 ar -t "$$lib_nostdio" | grep -q '^audio_pipeline\.o$$' || { echo "FAIL: $$lib_nostdio missing audio_pipeline.o member"; exit 1; }; \
	 echo "audit-no-stdio: nm -A $$lib_nostdio | grep -E '^_?(fprintf|printf|puts|fputs|stderr|__stderrp)$$' (NAME column)"; \
	 hits=$$(nm -A "$$lib_nostdio" | awk '{ name = $$NF; if (name ~ /^_?(fprintf|printf|puts|fputs|stderr|__stderrp)$$/) print }'); \
	 if [ -n "$$hits" ]; then \
	   echo "FAIL: forbidden stdio symbol(s) found in $$lib_nostdio (NO_STDIO=1 build):"; \
	   echo "$$hits"; \
	   exit 1; \
	 fi; \
	 echo "PASS: no fprintf/printf/puts/fputs/stderr/__stderrp symbols in $$lib_nostdio (NO_STDIO=1); path differs from default-config path $$lib_default"

# audit-no-stdio-stack (full-image gate): builds ALL FOUR
# delivered archives with NO_STDIO=1 (the dispatch forwards NO_STDIO to
# audio_common / lib/aec / lib/nr -- lib/aec compiles its trace/debug TU out
# entirely under this knob), nm-gates EACH delivered archive, then links
# test_no_stdio_stack.c (a minimal main that runs the whole
# query->init_ex->process->reset->destroy flow and uses no stdio itself --
# exit code is its only output) and nm-gates the EXECUTABLE, so a stdio
# symbol contributed by any archive fails the audit at the image level.
# Strictness tiers (calibrated, same as lib/aec's audit): the S1 set
# (fprintf/vfprintf/printf/puts/fputs) must be absent on BOTH backends at
# every level; the extended S2 set (+stderr/__stderrp/fputc/fwrite) is
# additionally enforced for BACKEND=ne10 -- the board deliverable -- end to
# end. BACKEND=kiss retains __stderrp/fputc/fwrite refs from the VENDORED
# KISS FFT's malloc-failure error macro (kiss_fft_log.h); that path is
# unreachable on the pool path (allocator-hook-proven zero-heap) and kiss is
# the host/reference backend, so it is a documented calibration, not a pass.
.PHONY: audit-no-stdio-stack
audit-no-stdio-stack:
	@echo "audit-no-stdio-stack: building the full stack (BACKEND=$(BACKEND), NO_STDIO=1)"
	@$(MAKE) --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 libaudio_pipeline.a >/dev/null
	@pl="$$($(MAKE) -s --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 print-lib-path)"; \
	 ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 print-lib-path)"; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 $(MAKE) --no-print-directory -C $(AC_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 lib >/dev/null; \
	 $(MAKE) --no-print-directory -C $(AEC_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" lib >/dev/null; \
	 $(MAKE) --no-print-directory -C $(NR_DIR) BACKEND=$(BACKEND) SIMD=$(SIMD) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" lib >/dev/null; \
	 s1='^_?(fprintf|vfprintf|printf|puts|fputs)$$'; \
	 s2='^_?(fprintf|vfprintf|printf|puts|fputs|stderr|__stderrp|fputc|fwrite)$$'; \
	 pat="$$s1"; [ "$(BACKEND)" = ne10 ] && pat="$$s2"; \
	 for lib in "$$pl" "$$aec" "$$nr" "$$ac"; do \
	   test -f "$$lib" || { echo "FAIL: archive not found: $$lib"; exit 1; }; \
	   hits=$$(nm -A "$$lib" | awk -v re="$$pat" '{ n=$$NF; if (n ~ re) print }'); \
	   if [ -n "$$hits" ]; then echo "FAIL: stdio symbol(s) in delivered archive $$lib:"; echo "$$hits"; exit 1; fi; \
	   echo "  archive clean: $$lib"; \
	 done; \
	 bd="$$($(MAKE) -s --no-print-directory BACKEND=$(BACKEND) NO_STDIO=1 print-bin-dir)"; \
	 mkdir -p "$$bd"; \
	 $(CC) $(CFLAGS) -DAUDIO_PIPELINE_NO_STDIO -c -o "$$bd/test_no_stdio_stack.o" mono_aec_nr_res/tests/test_no_stdio_stack.c; \
	 $(LINK) -o "$$bd/test_no_stdio_stack" "$$bd/test_no_stdio_stack.o" "$$pl" "$$aec" "$$nr" "$$ac" -lm; \
	 hits=$$(nm "$$bd/test_no_stdio_stack" | awk -v re="$$pat" '{ n=$$NF; if (n ~ re) print }'); \
	 if [ -n "$$hits" ]; then echo "FAIL: stdio symbol(s) in the linked EXECUTABLE:"; echo "$$hits"; exit 1; fi; \
	 "$$bd/test_no_stdio_stack" || { rc=$$?; echo "FAIL: test_no_stdio_stack exit $$rc"; exit 1; }; \
	 echo "PASS: full NO_STDIO=1 stack -- 4 delivered archives + linked executable clean ($$pat tier), runtime exit 0"

# =============================================================================
# publish v2: immutable per-config release under dist/
# -- same design as audio_common's/AEC's/NR's publish (see audio_common's
# Makefile for the full writeup): dist/<backend>/<cfg_sig>/ + atomic
# dist/<backend>/current symlink, a per-backend mkdir-lock serialising
# concurrent publishes, MANIFEST.txt with relative paths + per-file sha256 +
# this build's three resolved producer identities (AC_PRODUCER_SIG/
# AEC_PRODUCER_SIG/NR_PRODUCER_SIG) + the git commit of each of the three
# sibling/submodule repos (audio_common, lib/aec, lib/nr) via `git -C`, on
# top of this repo's (Audio_ALG's) own commit. PUBLISH_ARTIFACTS needs
# $(AC_LIB)/$(AEC_LIB)/$(NR_LIB) to exist, so `publish` goes through the same
# two-phase, three-producer dispatch as `all`/`test`/`libs` above when they
# aren't already known (see the ifeq($(AC_LIB),) block near the top of this
# file, which lists `publish` alongside all/test/libs).
# =============================================================================
# publish v4: immutable, content-addressed release under $(DIST_ROOT) --
# same design as audio_common's publish; see that Makefile for the full
# layout/semantics writeup. Additions here (mirroring audio_common's,
# extended to FOUR repos):
#   - ATTEST is one-event-one-file: EXACTLY one new
#     attest-<utc>-<commit>[-dirty]-<seq>.txt per publish EVENT (incl.
#     idempotent republishes), installed via the atomic_symlink_swap
#     helper's `--excl-install` mode (write-temp + link(2), the atomic
#     no-clobber equivalent of O_CREAT|O_EXCL) -- a same-second name
#     collision regenerates the event id with the next <seq> (capped at
#     999) and retries; an existing attestation is NEVER overwritten
#     (a prior cp+mv -f approach let same-second republishes silently
#     clobber the prior record).
#   - Every commit recorded is the FULL 40-hex git OID (never abbreviated),
#     for this repo (pipelines/Audio_ALG) AND all three producers
#     (audio_common, lib/aec, lib/nr) -- "which exact checkout of the whole
#     stack published this" lives in ATTEST/, not in MANIFEST (which stays
#     deterministic: config + tool identities + per-file SHA-256 only, so
#     the same release id always has the same MANIFEST bytes and republish
#     verifies it byte-for-byte).
#   - Unknown identity (c) is checked FIRST, per repo, and is refused
#     UNCONDITIONALLY -- no ALLOW_DIRTY_PUBLISH/ALLOW_UNTRACKED_PUBLISH
#     combination admits it, because without a commit OID neither the
#     tracked diff nor the untracked enumeration can be named for that
#     repo. Only once every one of the four repos (this repo AND all
#     three producers) has resolved to a real commit does this recipe
#     compute (a) TRACKED-dirty (`git status --porcelain -uno`, identical
#     -uno scope for all four) and (b) UNTRACKED (`git ls-files --others
#     --exclude-standard`; THIS repo's list has `AINR/GTCRN/` and
#     `AINR/gtcrn_github/` whitelisted BY PREFIX -- permanent user-owned
#     content never staged for a publish -- the three producers get no
#     such whitelist, so any untracked file there is always a violation).
#     Ordering is gate-then-hash: the dirty gate runs for all
#     four repos, then the untracked gate for all four, and ONLY AFTER
#     BOTH PASS does this recipe hash anything -- no hash work happens on
#     a repo that is about to FATAL anyway. The two escape hatches remain
#     separate and orthogonal: ALLOW_DIRTY_PUBLISH=1 admits tracked
#     changes and records a dirty_diff_sha256 (sha256 of `git diff
#     --binary HEAD`) per dirty repo; ALLOW_UNTRACKED_PUBLISH=1 admits
#     untracked files and records an untracked_tree_sha256 per repo with
#     untracked=1, computed by the shared `hash_untracked()` helper
#     (defined once, at the top of _publish_impl) over sorted FIXED-FIELD
#     records -- "L <sha256(path)> <sha256(readlink output)>" for a
#     symlink, "F <mode> <sha256(path)> <sha256(content)>" for a regular
#     file -- every variable-length value is itself hashed so records
#     concatenate unambiguously (raw space-joined path/target strings were
#     collision-prone) -- the tracked diff hash says nothing
#     about untracked bytes, so two different untracked source states must
#     not share a provenance record. An untracked path that is neither a
#     regular file nor a symlink (an embedded git checkout, a fifo), or
#     that a stat/readlink/shasum call fails on, cannot be hashed
#     deterministically and is always refused (fail-closed), with a
#     diagnostic naming both the path and the repo it came from.
#   - Attestations are UNSIGNED: they provide traceability, not
#     authenticity -- anyone with filesystem access could forge one, so do
#     not present ATTEST/ as tamper-proof under an attacker model
#     (MANIFEST byte-verification is the integrity check; ATTEST is the
#     provenance log).
# `publish` is the locked driver (a MAKEFLAGS word-scan makes
# `-n`/`-q`/`-t` fully ZERO-WRITE -- no DIST_ROOT, no lock, no
# stage/helper/MANIFEST/ATTEST writes, and no artifact mtime changes; `-t`
# is checked FIRST so a COMBINED invocation like `make -nt publish` takes
# the touch no-op path instead of recursing for `-n` -- see that target's
# own comment for the full word-scan detail); _publish_impl is INTERNAL
# (direct invocation skips the lock) and goes through the same
# three-producer phase-1 dispatch as `all`/`test`.
# =============================================================================
PUBLISH_ARTIFACTS = $(LIB_TARGET) $(BIN_DIR)/aec_nr_pipeline $(BIN_DIR)/aec_nr_pipeline_static $(BIN_DIR)/mono_alignulcnet

# Release gate (deliberately NOT wired into `test:` -- it runs a full real
# publish): publishes into a throwaway DIST_ROOT and asserts every
# PUBLISH_ARTIFACTS basename appears in the resulting MANIFEST.txt. Guards
# the "new artifact built but never shipped" failure mode: adding a target
# without extending PUBLISH_ARTIFACTS makes this gate fail instead of
# silently publishing an incomplete set. The ULCNet application binary is
# published directly; there is no separate application wrapper archive.
.PHONY: test-publish-manifest
test-publish-manifest:
	@tmp=$$(mktemp -d) || exit 1; \
	 $(MAKE) --no-print-directory publish DIST_ROOT=$$tmp ALLOW_DIRTY_PUBLISH=1 >/dev/null 2>&1 || \
	   { echo "FAIL: publish into throwaway DIST_ROOT failed"; rm -rf "$$tmp"; exit 1; }; \
	 manifest=$$(find "$$tmp" -name MANIFEST.txt | head -1); \
	 test -n "$$manifest" || { echo "FAIL: no MANIFEST.txt produced"; rm -rf "$$tmp"; exit 1; }; \
	 rc=0; \
	 for a in $(notdir $(PUBLISH_ARTIFACTS)); do \
	   grep -q "$$a" "$$manifest" || { echo "FAIL: $$a missing from MANIFEST.txt"; rc=1; }; \
	 done; \
	 [ $$rc -eq 0 ] && echo "publish manifest OK ($(words $(PUBLISH_ARTIFACTS)) artifacts verified)"; \
	 rm -rf "$$tmp"; exit $$rc
DIST_BACKEND_DIR  = $(DIST_ROOT)/$(BACKEND)
DIST_LOCK         = $(DIST_ROOT)/.lock.$(BACKEND)

.PHONY: publish
# Dry-run guard: this
# recipe mentions $(MAKE), so GNU make executes it for real under -n/-q/-t
# (all three share that rule) -- which used to mkdir $(DIST_ROOT)
# persistently and take/release the real publish lock on every dry run.
# The word-scan below detects those flags in MAKEFLAGS and branches:
#   -t  prints a one-line note and exits 0 WITHOUT recursing (recursing
#       lets standard touch semantics bump the mtimes of this
#       repo's own artifact prerequisites (the pipeline objects/archive/
#       binaries) -- real writes. publish is a phony ACTION, not a build
#       product; touching its artifact prerequisites without attesting
#       would fabricate state, so touch mode is an explicit no-op here).
#       Checked FIRST: in a COMBINED invocation like `make -nt publish`,
#       recursing for -n would hand the child make both flags and GNU make
#       then really applies touch semantics (reproduced) -- so any t wins
#       over n, conservatively;
#   -q  exits 1 directly (question mode's documented "needs updating"
#       answer -- publish is phony, so it always would run -- no output);
#   -n  exec a recursion-only path (the child make inherits -n and merely
#       PRINTS _publish_impl's commands).
# All three are now ZERO-WRITE: no DIST_ROOT, no lock, no stage/helper/
# MANIFEST/ATTEST writes, and no artifact mtime changes. See audio_common's
# Makefile for the full MAKEFLAGS word-scan writeup (verified against GNU
# make 3.81: 17 flag combinations incl. long forms: bundled no-dash short
# flags may appear as the first word ("n", "sn") OR as separate dashed
# words ("-n" after a long option); long options start with "--"; a
# literal "--" separates flags from command-line variable overrides, which
# can themselves contain n/q/t -- BACKEND=ne10 -- hence the explicit
# skips). Normal invocations keep the original lock-FIRST discipline
# unchanged.
publish:
	+@has_n=0; has_q=0; has_t=0; \
	for w in $(MAKEFLAGS); do \
	  case "$$w" in --) break ;; --*|*=*) continue ;; -n|-q|-t) ;; -*) continue ;; esac; \
	  case "$$w" in *n*) has_n=1 ;; esac; \
	  case "$$w" in *q*) has_q=1 ;; esac; \
	  case "$$w" in *t*) has_t=1 ;; esac; \
	done; \
	if [ "$$has_t" = 1 ]; then echo "publish: -t (touch) is a no-op here -- publish is an action, not a build product; nothing was touched"; exit 0; fi; \
	if [ "$$has_q" = 1 ]; then exit 1; fi; \
	if [ "$$has_n" = 1 ]; then exec $(MAKE) --no-print-directory _publish_impl; fi; \
	mkdir -p $(DIST_ROOT); \
	if ! mkdir "$(DIST_LOCK)" 2>/dev/null; then \
	  echo "FATAL: publish lock $(DIST_LOCK) already held (concurrent 'make publish BACKEND=$(BACKEND)'?)" >&2; \
	  exit 1; \
	fi; \
	trap 'rmdir "$(DIST_LOCK)" 2>/dev/null' EXIT INT TERM; \
	$(MAKE) --no-print-directory _publish_impl

# _publish_impl phase-2: attest v2 records FOUR repos'
# provenance -- this repo (Audio_ALG/pipelines) AND all three producers
# (audio_common, lib/aec, lib/nr) -- each as a FULL 40-hex git commit OID
# plus a dirty flag, so a release's ATTEST/ directory names the exact
# source state of the entire stack, not just this repo.
#
# hash_untracked() (defined at the very top of the recipe below, VERBATIM
# the same helper audio_common's Makefile defines): takes a path PREFIX as
# $1, reads a newline-separated untracked-path list on stdin, sorts it
# (`LC_ALL=C sort`), and emits one FIXED-FIELD record per path -- "L
# <sha256(path)> <sha256(readlink output)>" for a symlink, "F <mode>
# <sha256(path)> <sha256(content)>" for a regular file, "X <path>" for
# anything else (neither file nor symlink, or any stat/readlink/shasum
# failure -- fail-closed, never an empty hash). Every variable-length
# value is itself hashed before being placed in the record, so records
# concatenate unambiguously (a raw space-joined "path target" string is
# collision-prone: "a b"->"c" vs "a"->"b c" would otherwise encode
# identically). Called FOUR times below, once per repo, each into its OWN
# record file: this repo as `hash_untracked "./"` over the
# WHITELIST-FILTERED list, each producer as `hash_untracked "$(AC_DIR)/"` /
# `"$(AEC_DIR)/"` / `"$(NR_DIR)/"` over its raw (unfiltered) list -- the
# prefix is how the SAME helper resolves a path relative to a DIFFERENT
# repo root without duplicating the L/F/X logic four times. Each repo's
# untracked_tree_sha256 (`uhash`/`ac_uhash`/`aec_uhash`/`nr_uhash`) is then
# `shasum -a 256` of that ONE record file.
#
# Unknown identity is checked FIRST, per repo, and is an UNCONDITIONAL
# FATAL -- no ALLOW_DIRTY_PUBLISH/ALLOW_UNTRACKED_PUBLISH combination
# admits it, because without a commit OID neither the tracked diff nor the
# untracked enumeration can be named for that repo. Each producer dir is
# still guarded with `[ -z "$(X_DIR)" ] || [ ! -d "$(X_DIR)" ]` before any
# `git -C` call (an EMPTY or missing dir variable would otherwise make
# `git -C ''`/`git -C <missing>` fail in a way that doesn't clearly name
# the producer, or silently misattribute this repo's own state to it) --
# a missing/non-existent producer directory FATALs by name exactly like a
# present-but-non-git one does. Only once every one of the four repos has
# resolved to a real commit does this recipe compute that repo's
# dirty/untracked flags -- both are then plain 0/1, never a third
# "unknown" state, since "unknown" already exited above.
#
# Ordering (gate-then-hash): the dirty gate (all four
# repos) runs FIRST, then the untracked gate (all four repos) -- each a
# hard FATAL, by name, per offending repo, unless its escape hatch is set
# -- and ONLY AFTER BOTH GATES PASS does this recipe do any hashing: the
# four `git diff`-based dirty_diff_sha256 hashes (each only for a repo
# with dirty=1), then the four hash_untracked() record files (each only
# for a repo with untracked=1). No hash work happens on a repo that is
# about to FATAL anyway.
#
# Dirty policy: publish REFUSES by default when this
# repo OR any producer has uncommitted TRACKED changes. ALLOW_DIRTY_
# PUBLISH=1 is the escape hatch; the attestation then records
# allow_dirty_publish=1 plus, for each repo that was dirty, a
# dirty_diff_sha256 (sha256 of `git diff --binary HEAD`) so the deviation
# itself is traceable.
#
# Untracked policy: publish separately REFUSES by default
# when this repo (outside its whitelist) OR any producer contains any
# untracked file. ALLOW_UNTRACKED_PUBLISH=1 is the separate escape hatch;
# the attestation then records allow_untracked_publish=1 plus, for each
# repo with untracked=1, an untracked_tree_sha256 from hash_untracked()
# above. Any X record in that repo's record file is always a hard FATAL
# naming the repo -- "an untracked path in <repo> cannot be hashed
# deterministically ... and so cannot enter the provenance record" -- a
# publish must never proceed as though it could account for a source it
# can't.
ifneq ($(AC_LIB),)
.PHONY: _publish_impl
_publish_impl: $(PUBLISH_ARTIFACTS)
	@set -e; \
	work="$$(mktemp -d)"; \
	tmp="$(DIST_BACKEND_DIR)/.stage.$$$$"; \
	trap 'rm -rf "$$tmp" "$$work"' EXIT INT TERM; \
	hash_untracked() { \
	  LC_ALL=C sort | while IFS= read -r p; do \
	    fp="$$1$$p"; \
	    ph="$$(printf '%s' "$$p" | shasum -a 256 | cut -d' ' -f1)"; \
	    [ "$${#ph}" -eq 64 ] || { printf 'X %s\n' "$$p"; continue; }; \
	    if [ -L "$$fp" ]; then \
	      tgt="$$(readlink "$$fp")" || { printf 'X %s\n' "$$p"; continue; }; \
	      th="$$(printf '%s' "$$tgt" | shasum -a 256 | cut -d' ' -f1)"; \
	      [ "$${#th}" -eq 64 ] || { printf 'X %s\n' "$$p"; continue; }; \
	      printf 'L %s %s\n' "$$ph" "$$th"; \
	    elif [ -f "$$fp" ]; then \
	      m="$$(stat -f '%Lp' "$$fp" 2>/dev/null || stat -c '%a' "$$fp" 2>/dev/null)"; \
	      ch="$$(shasum -a 256 "$$fp" 2>/dev/null | cut -d' ' -f1)"; \
	      { [ -n "$$m" ] && [ "$${#ch}" -eq 64 ]; } || { printf 'X %s\n' "$$p"; continue; }; \
	      printf 'F %s %s %s\n' "$$m" "$$ph" "$$ch"; \
	    else printf 'X %s\n' "$$p"; fi; \
	  done; \
	}; \
	commit="$$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- pipelines (this repo) has no git identity here (not a git checkout)." >&2; \
	  echo "  Without an identity the attestation cannot name the source state OR enumerate untracked" >&2; \
	  echo "  content, so NO escape hatch admits this. Publish from a real checkout or a scratch clone" >&2; \
	  echo "  with a disposable commit instead." >&2; \
	  exit 1; \
	fi; \
	dirty=0; [ -n "$$(git status --porcelain -uno 2>/dev/null)" ] && dirty=1; \
	unt_list="$$(git ls-files --others --exclude-standard 2>/dev/null | grep -vE '^AINR/(GTCRN|gtcrn_github)/' || true)"; \
	untracked=0; [ -n "$$unt_list" ] && untracked=1; \
	if [ -z "$(AC_DIR)" ] || [ ! -d "$(AC_DIR)" ]; then \
	  echo "FATAL: publish refused -- audio_common has no git identity (not a git checkout, or AC_DIR=$(AC_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	ac_commit="$$(git -C "$(AC_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$ac_commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- audio_common has no git identity (not a git checkout, or AC_DIR=$(AC_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	ac_dirty=0; [ -n "$$(git -C "$(AC_DIR)" status --porcelain -uno 2>/dev/null)" ] && ac_dirty=1; \
	ac_unt_list="$$(git -C "$(AC_DIR)" ls-files --others --exclude-standard 2>/dev/null)"; \
	ac_untracked=0; [ -n "$$ac_unt_list" ] && ac_untracked=1; \
	if [ -z "$(AEC_DIR)" ] || [ ! -d "$(AEC_DIR)" ]; then \
	  echo "FATAL: publish refused -- lib/aec has no git identity (not a git checkout, or AEC_DIR=$(AEC_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	aec_commit="$$(git -C "$(AEC_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$aec_commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- lib/aec has no git identity (not a git checkout, or AEC_DIR=$(AEC_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	aec_dirty=0; [ -n "$$(git -C "$(AEC_DIR)" status --porcelain -uno 2>/dev/null)" ] && aec_dirty=1; \
	aec_unt_list="$$(git -C "$(AEC_DIR)" ls-files --others --exclude-standard 2>/dev/null)"; \
	aec_untracked=0; [ -n "$$aec_unt_list" ] && aec_untracked=1; \
	if [ -z "$(NR_DIR)" ] || [ ! -d "$(NR_DIR)" ]; then \
	  echo "FATAL: publish refused -- lib/nr has no git identity (not a git checkout, or NR_DIR=$(NR_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	nr_commit="$$(git -C "$(NR_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$nr_commit" = "unknown" ]; then \
	  echo "FATAL: publish refused -- lib/nr has no git identity (not a git checkout, or NR_DIR=$(NR_DIR) not present); no escape hatch admits this." >&2; \
	  exit 1; \
	fi; \
	nr_dirty=0; [ -n "$$(git -C "$(NR_DIR)" status --porcelain -uno 2>/dev/null)" ] && nr_dirty=1; \
	nr_unt_list="$$(git -C "$(NR_DIR)" ls-files --others --exclude-standard 2>/dev/null)"; \
	nr_untracked=0; [ -n "$$nr_unt_list" ] && nr_untracked=1; \
	adp=0; [ "$(ALLOW_DIRTY_PUBLISH)" = "1" ] && adp=1; \
	aup=0; [ "$(ALLOW_UNTRACKED_PUBLISH)" = "1" ] && aup=1; \
	if [ "$$adp" != "1" ]; then \
	  bad=0; \
	  if [ "$$dirty" = "1" ]; then echo "FATAL: publish refused -- pipelines (this repo) working tree has uncommitted TRACKED changes." >&2; bad=1; fi; \
	  if [ "$$ac_dirty" = "1" ]; then echo "FATAL: publish refused -- audio_common working tree has uncommitted TRACKED changes." >&2; bad=1; fi; \
	  if [ "$$aec_dirty" = "1" ]; then echo "FATAL: publish refused -- lib/aec working tree has uncommitted TRACKED changes." >&2; bad=1; fi; \
	  if [ "$$nr_dirty" = "1" ]; then echo "FATAL: publish refused -- lib/nr working tree has uncommitted TRACKED changes." >&2; bad=1; fi; \
	  if [ "$$bad" = "1" ]; then \
	    echo "  Commit (or clone a clean checkout) for a board deliverable, or pass ALLOW_DIRTY_PUBLISH=1" >&2; \
	    echo "  for a dev publish -- the escape hatch is recorded in the attestation (covers TRACKED changes only)." >&2; \
	    exit 1; \
	  fi; \
	fi; \
	if [ "$$aup" != "1" ]; then \
	  bad=0; \
	  if [ "$$untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- pipelines (this repo) contains UNTRACKED files outside the AINR/GTCRN|gtcrn_github whitelist:" >&2; \
	    printf '%s\n' "$$unt_list" | head -5 | sed 's/^/    /' >&2; \
	    bad=1; \
	  fi; \
	  if [ "$$ac_untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- audio_common contains UNTRACKED files:" >&2; \
	    printf '%s\n' "$$ac_unt_list" | head -5 | sed 's/^/    /' >&2; \
	    bad=1; \
	  fi; \
	  if [ "$$aec_untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- lib/aec contains UNTRACKED files:" >&2; \
	    printf '%s\n' "$$aec_unt_list" | head -5 | sed 's/^/    /' >&2; \
	    bad=1; \
	  fi; \
	  if [ "$$nr_untracked" = "1" ]; then \
	    echo "FATAL: publish refused -- lib/nr contains UNTRACKED files:" >&2; \
	    printf '%s\n' "$$nr_unt_list" | head -5 | sed 's/^/    /' >&2; \
	    bad=1; \
	  fi; \
	  if [ "$$bad" = "1" ]; then \
	    echo "  Track/remove them for a board deliverable, or pass ALLOW_UNTRACKED_PUBLISH=1 -- the escape" >&2; \
	    echo "  hatch is recorded in the attestation with untracked_tree_sha256 over their exact contents." >&2; \
	    exit 1; \
	  fi; \
	fi; \
	ddiff=""; [ "$$dirty" = "1" ] && ddiff="$$(git diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	ac_ddiff=""; [ "$$ac_dirty" = "1" ] && ac_ddiff="$$(git -C "$(AC_DIR)" diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	aec_ddiff=""; [ "$$aec_dirty" = "1" ] && aec_ddiff="$$(git -C "$(AEC_DIR)" diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	nr_ddiff=""; [ "$$nr_dirty" = "1" ] && nr_ddiff="$$(git -C "$(NR_DIR)" diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	uhash=""; \
	if [ "$$untracked" = "1" ]; then \
	  printf '%s\n' "$$unt_list" | hash_untracked "./" > "$$work/untracked.records"; \
	  if grep -q '^X ' "$$work/untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in pipelines (this repo) cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  uhash="$$(shasum -a 256 < "$$work/untracked.records" | cut -d' ' -f1)"; \
	fi; \
	ac_uhash=""; \
	if [ "$$ac_untracked" = "1" ]; then \
	  printf '%s\n' "$$ac_unt_list" | hash_untracked "$(AC_DIR)/" > "$$work/ac_untracked.records"; \
	  if grep -q '^X ' "$$work/ac_untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in audio_common cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/ac_untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  ac_uhash="$$(shasum -a 256 < "$$work/ac_untracked.records" | cut -d' ' -f1)"; \
	fi; \
	aec_uhash=""; \
	if [ "$$aec_untracked" = "1" ]; then \
	  printf '%s\n' "$$aec_unt_list" | hash_untracked "$(AEC_DIR)/" > "$$work/aec_untracked.records"; \
	  if grep -q '^X ' "$$work/aec_untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in lib/aec cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/aec_untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  aec_uhash="$$(shasum -a 256 < "$$work/aec_untracked.records" | cut -d' ' -f1)"; \
	fi; \
	nr_uhash=""; \
	if [ "$$nr_untracked" = "1" ]; then \
	  printf '%s\n' "$$nr_unt_list" | hash_untracked "$(NR_DIR)/" > "$$work/nr_untracked.records"; \
	  if grep -q '^X ' "$$work/nr_untracked.records"; then \
	    echo "FATAL: publish refused -- an untracked path in lib/nr cannot be hashed deterministically (embedded git checkout / special file / I/O failure) and so cannot enter the provenance record:" >&2; \
	    grep '^X ' "$$work/nr_untracked.records" | sed 's/^X /    /' >&2; \
	    exit 1; \
	  fi; \
	  nr_uhash="$$(shasum -a 256 < "$$work/nr_untracked.records" | cut -d' ' -f1)"; \
	fi; \
	dateutc="$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	stamp='$(ATTEST_STAMP)'; [ -n "$$stamp" ] || stamp="$$(date -u +%Y%m%dT%H%M%SZ)"; \
	$(HOSTCC) -O2 -o "$$work/swapln" $(AC_DIR)/tools/atomic_symlink_swap.c || { echo "FATAL: could not build the atomic symlink-swap helper with HOSTCC=$(HOSTCC) (pass HOSTCC=<host compiler>)" >&2; exit 1; }; \
	mkdir -p "$(DIST_BACKEND_DIR)"; \
	rm -rf "$$tmp"; mkdir -p "$$tmp"; \
	cp $(PUBLISH_ARTIFACTS) "$$tmp/"; \
	shas="$$( cd "$$tmp" && for f in *; do shasum -a 256 "$$f"; done )"; \
	content12="$$(printf '%s\n' "$$shas" | sort | shasum -a 256 | cut -c1-12)"; \
	rel_id="$(CFG_SIG)-$$content12"; \
	rel_dir="$(DIST_BACKEND_DIR)/$$rel_id"; \
	{ echo "repo=pipelines"; \
	  echo "release_id=$$rel_id"; \
	  echo "backend=$(BACKEND)"; \
	  echo "cfg_sig=$(CFG_SIG)"; \
	  echo "ac_producer_cfg_sig=$(AC_PRODUCER_SIG)"; \
	  echo "aec_producer_cfg_sig=$(AEC_PRODUCER_SIG)"; \
	  echo "nr_producer_cfg_sig=$(NR_PRODUCER_SIG)"; \
	  echo "cc=$(CC)"; echo "cxx=$(CXX)"; echo "ar=$(AR)"; echo "ranlib=$(RANLIB)"; echo "link=$(LINK)"; \
	  echo "cflags=$(CFLAGS)"; \
	  echo "extra_cflags=$(EXTRA_CFLAGS)"; \
	  echo "werror=$(WERROR)"; echo "no_stdio=$(NO_STDIO)"; echo "debug=$(DEBUG)"; \
	  echo "$$shas"; \
	} > "$$tmp/MANIFEST.txt"; \
	suffix=""; [ "$$dirty" = "1" ] && suffix="-dirty"; \
	if [ -d "$$rel_dir" ]; then \
	  for f in "$$tmp"/*; do \
	    bn="$$(basename "$$f")"; \
	    cmp -s "$$f" "$$rel_dir/$$bn" || { echo "FATAL: existing release $$rel_id differs from staged content for $$bn (content-address collision, or the stored release/MANIFEST was modified on disk)" >&2; exit 1; }; \
	  done; \
	  rm -rf "$$tmp"; \
	  echo "pipelines: $$rel_id already published (byte-verified, incl. MANIFEST) -- appending attestation + repointing current"; \
	else \
	  mkdir "$$tmp/ATTEST"; \
	  mv "$$tmp" "$$rel_dir"; \
	fi; \
	mkdir -p "$$rel_dir/ATTEST"; \
	n=1; \
	while :; do \
	  attest_name="attest-$$stamp-$$commit$$suffix-$$(printf '%03d' "$$n").txt"; \
	  { echo "event_id=$${attest_name%.txt}"; \
	    echo "repo=pipelines"; \
	    echo "release_id=$$rel_id"; \
	    echo "cfg_sig=$(CFG_SIG)"; \
	    echo "backend=$(BACKEND)"; \
	    echo "git_commit=$$commit"; echo "git_dirty=$$dirty"; \
	    [ -n "$$ddiff" ] && echo "dirty_diff_sha256=$$ddiff"; \
	    echo "git_untracked=$$untracked"; \
	    [ -n "$$uhash" ] && echo "untracked_tree_sha256=$$uhash"; \
	    echo "audio_common_git_commit=$$ac_commit"; echo "audio_common_git_dirty=$$ac_dirty"; \
	    [ -n "$$ac_ddiff" ] && echo "audio_common_dirty_diff_sha256=$$ac_ddiff"; \
	    echo "audio_common_git_untracked=$$ac_untracked"; \
	    [ -n "$$ac_uhash" ] && echo "audio_common_untracked_tree_sha256=$$ac_uhash"; \
	    echo "aec_git_commit=$$aec_commit"; echo "aec_git_dirty=$$aec_dirty"; \
	    [ -n "$$aec_ddiff" ] && echo "aec_dirty_diff_sha256=$$aec_ddiff"; \
	    echo "aec_git_untracked=$$aec_untracked"; \
	    [ -n "$$aec_uhash" ] && echo "aec_untracked_tree_sha256=$$aec_uhash"; \
	    echo "nr_git_commit=$$nr_commit"; echo "nr_git_dirty=$$nr_dirty"; \
	    [ -n "$$nr_ddiff" ] && echo "nr_dirty_diff_sha256=$$nr_ddiff"; \
	    echo "nr_git_untracked=$$nr_untracked"; \
	    [ -n "$$nr_uhash" ] && echo "nr_untracked_tree_sha256=$$nr_uhash"; \
	    echo "date_utc=$$dateutc"; \
	    echo "allow_dirty_publish=$$adp"; \
	    echo "allow_untracked_publish=$$aup"; \
	  } > "$$work/attest.txt"; \
	  rc=0; "$$work/swapln" --excl-install "$$work/attest.txt" "$$rel_dir/ATTEST/$$attest_name" || rc=$$?; \
	  [ "$$rc" -eq 0 ] && break; \
	  [ "$$rc" -eq 2 ] || { echo "FATAL: could not install attestation $$attest_name" >&2; exit 1; }; \
	  n=$$((n+1)); \
	  [ "$$n" -le 999 ] || { echo "FATAL: 999 attestation name collisions for $$stamp-$$commit (runaway?)" >&2; exit 1; }; \
	done; \
	"$$work/swapln" "$$rel_id" "$(DIST_BACKEND_DIR)/current"; \
	echo "pipelines: published $(BACKEND)/$$rel_id -> $(DIST_BACKEND_DIR)/current (attested: $$attest_name)"
endif

# clean: this dir's own hash-keyed obj/bin trees (every config, not just the
# one BACKEND/flags this invocation resolves to) PLUS any legacy flat
# binaries/archive left over from before this directory layout existed (an
# old checkout, or a manual build from an older revision of this file).
# Uses the OBJ_ROOT/BIN_ROOT knobs so `clean` with an override
# scrubs THAT tree, never the real obj/ and bin/. dist/ is NOT removed
# (publish is a deliberate release, not a build cache).
clean:
	rm -rf $(OBJ_ROOT) $(BIN_ROOT)
	rm -f aec_nr_pipeline aec_nr_pipeline_static libaudio_pipeline.a test_audio_pipeline

# clean-all: full nuke -- this dir's own artifacts plus all three
# submodule/shared build trees (lib/aec, lib/nr, audio_common). clean-libs is
# kept as a backward-compatible alias (same effect) since existing
# tooling/docs call it by that name.
clean-all: clean
	$(MAKE) -C $(AEC_DIR) clean
	$(MAKE) -C $(NR_DIR) clean
	$(MAKE) -C $(AC_DIR) clean

clean-libs: clean-all

.PHONY: clean clean-all clean-libs

# --- Header dependency tracking ----------------------------------------------
# See audio_common's Makefile for the full rationale (-MD not -MMD, -MP
# guards against deleted-header stale-.d errors). $(wildcard ...) makes this
# evaluate to nothing (not an error) before any objects exist yet for this
# CFG_SIG.
-include $(wildcard $(OBJ_DIR)/*.d)
