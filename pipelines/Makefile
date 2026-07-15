# Audio_ALG/pipelines/Makefile
#
# AEC(linear) -> echo-aware NR -> RES pipeline — two versions of the SAME
# algorithm (see aec_nr_pipeline.c's header for the DSP chain):
#
#   aec_nr_pipeline         (Version A: malloc)         — heap-allocated modules
#   aec_nr_pipeline_static  (Version B: static memory)  — one caller pool, no
#                                                          malloc after init
#
# Both are now thin CLI shells (arg parsing + WAV I/O + diagnostics) over
# audio_pipeline.h/.c (review F20) — the pool-sizing/carving/per-hop-
# processing logic itself, built into libaudio_pipeline.a, linkable on its
# own by a board's memory manager. See pipelines/README.md's "Board
# Integration" section.
#
# All three link against the submodule libs (lib/aec, lib/nr) + the shared
# audio_common FFT/fast_math layer (../../audio_common relative to this dir).
#
# =============================================================================
# Build isolation (round-3 review B01) — the fourth and last of four repos
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
#                              # (BACKEND=kiss) + both pipeline binaries
#   make BACKEND=ne10          # NE10 backend -- every axis this file and the
#                              # three producers' own Makefiles key on
#                              # (BACKEND, EXTRA_CFLAGS, WERROR, NO_STDIO, ...)
#                              # lands in a fresh, separate obj|bin directory
#                              # automatically; no manual clean needed, ever
#   make EXTRA_CFLAGS="..."    # forwarded to all three producers AND this
#                              # pipeline's own compile
#   make WERROR=1              # opt-in: -Werror for this file's own TUs
#   make test                  # build + run test_audio_pipeline (audio_pipeline.h
#                              # acceptance tests -- see that file's header)
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

AEC_DIR = ../lib/aec/c_impl
NR_DIR  = ../lib/nr/c_impl

# audio_common sits two levels up from pipelines/ (SE/Audio_ALG/pipelines -> SE/audio_common).
# lib/aec/c_impl and lib/nr/c_impl each resolve it themselves from their own
# location (../../../../audio_common from c_impl/, verified to exist as a
# submodule checkout); AC_DIR here is for OUR OWN CFLAGS/-I path and is passed
# through explicitly (as AC_DIR=$(abspath $(AC_DIR))) to the lib/aec and
# lib/nr sub-makes so all three always agree on the exact same audio_common
# checkout.
AC_DIR ?= ../../audio_common
BACKEND ?= kiss

CC  ?= gcc
CXX ?= c++
AR      ?= ar
RANLIB  ?= ranlib

# CFG_SIG payload-coverage placeholders (round-3 review B01): CXXFLAGS/
# CPPFLAGS/DEBUG are not consumed by this Makefile today (no C++ TU of its
# own, no DEBUG-gated code path here), but the CFG_SIG payload below
# unconditionally includes them so the payload string has the same stable
# shape as audio_common's/AEC's/NR's, and so a future use of any of them here
# keys builds correctly instead of silently aliasing with a build that didn't
# set it. EXTRA_LDFLAGS/NO_STDIO are real here (see LDFLAGS/LIB_CFLAGS below).
CXXFLAGS      ?=
CPPFLAGS      ?=
EXTRA_LDFLAGS ?=
NO_STDIO      ?= 0
DEBUG         ?= 0

# TOOLCHAIN_CHECK=0 skips the CC/CXX -dumpmachine target-coherence guard in
# _cfg_guard below (round-4 review P1-2). Forwarded to all three producer
# sub-makes via $(FWD); participates in CFG_SIG.
TOOLCHAIN_CHECK ?= 1

# DIST_ROOT / HOSTCC (round-5 review): where `make publish` writes releases
# (test scripts point this at a throwaway mktemp dir so they never touch the
# real dist/), and the HOST compiler for the publish-time atomic
# symlink-swap helper (never $(CC): that may be a cross compiler). Neither
# is in CFG_SIG -- neither affects what is built. See audio_common's
# Makefile for the full writeups.
#
# ALLOW_DIRTY_PUBLISH (round-6 review P2): `make publish` REFUSES by default
# when THIS repo (Audio_ALG) or any of the three producers (audio_common,
# lib/aec, lib/nr) has uncommitted tracked changes or is not a git checkout
# at all (commit unknown) -- a release whose ATTEST provenance cannot name
# the exact source state of the whole stack is not a board deliverable.
# ALLOW_DIRTY_PUBLISH=1 is the explicit escape hatch for dev publishes; it
# is recorded in the attestation together with a sha256 of each dirty
# repo's `git diff --binary HEAD` (see _publish_impl's own comment for why
# THIS repo's dirty check ignores untracked files while the three
# producers' don't) so the deviation itself is traceable. Not in CFG_SIG:
# it never changes what is built, only whether publish proceeds.
#
# ATTEST_STAMP (round-6 review P2, TEST-ONLY): overrides the UTC timestamp
# embedded in the attestation FILENAME so the isolation tests can
# deterministically force same-second publish collisions and prove the
# no-clobber retry path. Never use it on a real publish -- though note a
# forged stamp is within the already-accepted threat model: attestations
# are unsigned and provide traceability, NOT authenticity against an
# attacker with filesystem access. Not in CFG_SIG or MANIFEST.
DIST_ROOT ?= dist
HOSTCC    ?= cc
ALLOW_DIRTY_PUBLISH ?= 0
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

# Deliberate BYTE-CHANGE when -ffp-contract=off first landed here (clang had
# been contracting a handful of mul+add pairs in the pipeline glue, e.g. the
# |E|^2 hoist) — the pipeline render baselines were regenerated at that
# commit. Keeping the flag uniform means future SIMD/EMA-style code added to
# pipeline TUs inherits the repo-wide bit-exactness discipline instead of
# silently depending on the compiler's contraction mood (review F12). The
# flag itself is appended further down (see the "FP-contraction policy"
# section below, after EXTRA_CFLAGS/WERROR are folded in) rather than here —
# round-3 review B04 moved it from this early position (BEFORE EXTRA_CFLAGS)
# to the true end of the CFLAGS assembly, so a caller's EXTRA_CFLAGS can
# never land after it and override it.
CFLAGS = -Wall -Wextra -O2 -std=gnu99 \
         -I$(AEC_DIR)/include -I$(AEC_DIR)/example \
         -I$(NR_DIR)/include \
         -I$(AC_DIR)/include
# AUDIO_PIPELINE_BACKEND_STR: compile-time FFT backend identity baked into
# audio_pipeline.c's AudioPipelineMemReq.backend field (F20) -- matches
# whichever libaudio_common.a variant this build links (BACKEND= above), so
# the descriptor never has to be told separately which backend produced it.
CFLAGS += -DAUDIO_PIPELINE_BACKEND_STR=\"$(BACKEND)\"
CFLAGS += $(EXTRA_CFLAGS)

# -Werror opt-in for this Makefile's OWN TUs (default off; CI can flip it
# on with `make WERROR=1`). Vendor code (audio_common's NE10/KISS) is built by
# audio_common's own Makefile, not by rules in this file, so no separate
# warning isolation is needed on this side.
WERROR ?= 0
ifeq ($(WERROR),1)
  CFLAGS += -Werror
endif

# NO_STDIO=1 (re-review R09): compiles audio_pipeline.o with
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

# --- FP-contraction policy: unified across all four repos (round-3 review
# B04). This flag USED TO be the fourth token of the base CFLAGS assignment
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
# --- Command-line override rejection (round-4 review P1-1) ------------------
# See audio_common's Makefile for the full writeup: a command-line
# CFLAGS=/CXXFLAGS=/CPPFLAGS=/LDFLAGS=/FP_POLICY= silently disables every one
# of this file's own assignments/appends to that variable (here that would
# strip -ffp-contract=off, -DAUDIO_PIPELINE_BACKEND_STR, the -I paths, and
# -lm) while still producing artifacts. EXTRA_CFLAGS / EXTRA_LDFLAGS are the
# supported hooks.
$(foreach v,CFLAGS CXXFLAGS CPPFLAGS LDFLAGS FP_POLICY,\
  $(if $(findstring command,$(origin $(v)))$(findstring override,$(origin $(v))),\
    $(error $(v) cannot be overridden (origin: $(origin $(v))) -- it would silently drop this Makefile's own flag appends (FP policy, backend define, include paths); use EXTRA_CFLAGS / EXTRA_LDFLAGS instead)))

# Conflict detection (round-3 review B04): reject outright rather than let
# a caller silently re-enable contraction. Checked against the
# fully-assembled $(CFLAGS) (already includes EXTRA_CFLAGS; a command-line
# CFLAGS= override is rejected outright just above) BEFORE FP_POLICY is
# appended below and BEFORE CFG_SIG is computed, so a rejected build never
# creates an obj/bin directory.
ifneq ($(findstring -Ofast,$(CFLAGS)),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -Ofast; this repo pins -ffp-contract=off; remove -Ofast from EXTRA_CFLAGS)
endif
ifneq ($(findstring -ffast-math,$(CFLAGS)),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from EXTRA_CFLAGS)
endif
ifneq ($(findstring -ffp-contract=,$(CFLAGS)),)
$(error FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffp-contract=; this repo pins -ffp-contract=off; remove -ffp-contract= from EXTRA_CFLAGS)
endif

FP_POLICY := -ffp-contract=off
CFLAGS     += $(FP_POLICY)

# The NE10 archive carries one C++ TU (NE10's generic-radix FFT kernel); a
# plain-C pipeline binary linking against it uses the C++ DRIVER, which
# supplies the correct C++ runtime itself (libc++ on macOS/clang, libstdc++
# on GNU/Linux gcc). The old hardcoded `-lc++` alongside it was a macOS-ism
# (duplicate there, a link error on GNU/Linux) and is gone (round-4 review
# P1-2). LINK participates in CFG_SIG below (P2-1).
LINK = $(CC)
ifeq ($(BACKEND),ne10)
  LINK = $(CXX)
endif

# LDFLAGS is now just link FLAGS (-lm plus whatever EXTRA_LDFLAGS adds) --
# NOT the three producer archive paths. Those are resolved dynamically (see
# the two-phase dispatch below) and appended directly to each link recipe's
# own prerequisite list / command line, in the fixed order the old
# hardcoded LIBS variable used: pipeline.o, libaudio_pipeline.a, libaec.a,
# libmmse_lsa.a, libaudio_common.a, -lm. libaudio_pipeline.a (review F20)
# holds ONLY the pipeline TU (audio_pipeline.o) -- it is not a bundled/fat
# archive, so it still needs libaec.a/libmmse_lsa.a/libaudio_common.a linked
# after it, same as when audio_pipeline.c's logic was file-local code inside
# each CLI's own .o.
LDFLAGS = -lm
LDFLAGS += $(EXTRA_LDFLAGS)

.DEFAULT_GOAL := all

# =============================================================================
# Consumer two-phase, THREE-producer resolution (round-3 review B01)
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
FWD = BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=$(NO_STDIO) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK)

ifeq ($(AC_LIB),)
.PHONY: all test libs _publish_impl aec_nr_pipeline aec_nr_pipeline_static test_audio_pipeline example-adapter libaudio_pipeline.a
all test libs _publish_impl aec_nr_pipeline aec_nr_pipeline_static test_audio_pipeline example-adapter libaudio_pipeline.a:
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
# Round-4 review payload additions: LINK (P2-1), TOOLCHAIN_CHECK (the
# guard-skip knob must not alias a guarded build), SRCS (P1-4: this
# Makefile's own compiled sources -- the archive's one member TU plus the
# CLI/test TUs -- so a source-set change keys to a fresh directory).
CFG_SIG_PAYLOAD := CC=$(CC) CXX=$(CXX) AR=$(AR) RANLIB=$(RANLIB) LINK=$(LINK) CFLAGS=$(CFLAGS) CXXFLAGS=$(CXXFLAGS) CPPFLAGS=$(CPPFLAGS) LDFLAGS=$(LDFLAGS) EXTRA_CFLAGS=$(EXTRA_CFLAGS) EXTRA_LDFLAGS=$(EXTRA_LDFLAGS) BACKEND=$(BACKEND) WERROR=$(WERROR) NO_STDIO=$(NO_STDIO) DEBUG=$(DEBUG) TOOLCHAIN_CHECK=$(TOOLCHAIN_CHECK) AC_PRODUCER_SIG=$(AC_PRODUCER_SIG) AEC_PRODUCER_SIG=$(AEC_PRODUCER_SIG) NR_PRODUCER_SIG=$(NR_PRODUCER_SIG) SRCS=$(sort audio_pipeline.c aec_nr_pipeline.c aec_nr_pipeline_static.c test_audio_pipeline.c example_board_adapter.c)
CFG_SIG := $(shell printf '%s' "$(CFG_SIG_PAYLOAD)" | cksum | cut -d' ' -f1)

# OBJ_ROOT / BIN_ROOT (round-6 review P1): where THIS repo's own keyed
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

# --- CFG_SIG collision guard + directory creation (round-3 review B01) -----
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
$(OBJ_DIR)/aec_nr_pipeline.o: aec_nr_pipeline.c audio_pipeline.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

$(OBJ_DIR)/aec_nr_pipeline_static.o: aec_nr_pipeline_static.c audio_pipeline.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# audio_pipeline.o / libaudio_pipeline.a (review F20): the pool-sizing/
# carving/per-hop-processing library TU, linkable independently of either
# CLI -- see audio_pipeline.h. The archive holds ONLY this one object (not a
# fat archive bundling libaec/libmmse_lsa/libaudio_common); those stay
# separate libraries a consumer links alongside it, same as both CLIs do.
# Compiled with $(LIB_CFLAGS) (not the shared $(CFLAGS)) so NO_STDIO=1 only
# ever affects THIS object -- see the NO_STDIO comment near CFLAGS above.
$(OBJ_DIR)/audio_pipeline.o: audio_pipeline.c audio_pipeline.h pipeline_dims.h | _cfg_guard
	$(CC) $(LIB_CFLAGS) -MD -MP -c -o $@ $<

# test_audio_pipeline: standalone acceptance test for audio_pipeline.h itself
# (create-vs-init byte equality incl. a dirty/poisoned pool, destroy
# idempotence, misaligned/undersized pool rejection, sample-rate whitelist
# rejection) -- see test_audio_pipeline.c's own header for detail.
$(OBJ_DIR)/test_audio_pipeline.o: test_audio_pipeline.c audio_pipeline.h | _cfg_guard
	$(CC) $(CFLAGS) -MD -MP -c -o $@ $<

# example_board_adapter: REFERENCE ONLY host simulation of a board-side
# common-memory-manager integration (round-3 review B02/R04) -- see that
# file's own header. Not the real board adapter; demonstrates the calling
# sequence/error handling/descriptor contract only.
$(OBJ_DIR)/example_board_adapter.o: example_board_adapter.c audio_pipeline.h | _cfg_guard
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

# Fresh-archive discipline (round-4 review P1-4): always built from scratch
# into a temp file and renamed into place -- never `ar` onto an existing
# archive (which only adds/replaces members and would keep a removed source's
# stale member forever). See audio_common's Makefile for the full writeup.
# One chained shell line with a PID-suffixed temp (round-5 review P2): two
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
.PHONY: all test libs aec_nr_pipeline aec_nr_pipeline_static test_audio_pipeline example-adapter example_board_adapter FORCE
all: $(BIN_DIR)/aec_nr_pipeline $(BIN_DIR)/aec_nr_pipeline_static

# test: also builds + runs example_board_adapter (round-3 review B02/R04) --
# wired into the SAME gate as test_audio_pipeline (not left standalone-only)
# because it is clean: green on both backends, exits 0 on full success, and
# its own failure modes (a positive-flow step or negative-demonstration
# check failing) are exactly the kind of regression this gate exists to
# catch. `make example-adapter` (below) remains available standalone too.
test: $(BIN_DIR)/test_audio_pipeline $(BIN_DIR)/example_board_adapter
	./$(BIN_DIR)/test_audio_pipeline
	./$(BIN_DIR)/example_board_adapter

# example-adapter (round-3 review B02/R04): build + RUN the REFERENCE ONLY
# board-adapter host simulation (example_board_adapter.c) standalone -- same
# build-then-run shape as `test:` above. Its own exit code is the smoke-test
# signal: nonzero if any positive-flow step or negative-demonstration check
# failed, so a red `make example-adapter` fails the build the same way a red
# `make test` does.
example-adapter: $(BIN_DIR)/example_board_adapter
	./$(BIN_DIR)/example_board_adapter

# Root-name alias for the binary itself (mirrors test_audio_pipeline: below)
# -- `example-adapter` (above) is the build-AND-run entry point; this is just
# "build the binary" for whoever wants that split.
example_board_adapter: $(BIN_DIR)/example_board_adapter

# libs: compat alias for the old `libs:` phony (which used to build the
# three producer archives directly with hardcoded paths/flags). Building the
# archives is now an automatic side effect of the FORCE rules below, so this
# just asserts all three exist for whoever's tooling still invokes `make libs`.
libs: $(AC_LIB) $(AEC_LIB) $(NR_LIB)

# Root-name phony aliases (round-3 review B01): keep the pre-keyed-directory
# invocation (`make aec_nr_pipeline`, etc.) working -- the real artifact now
# lives at $(BIN_DIR)/<name>, not flat in this directory.
aec_nr_pipeline: $(BIN_DIR)/aec_nr_pipeline
aec_nr_pipeline_static: $(BIN_DIR)/aec_nr_pipeline_static
test_audio_pipeline: $(BIN_DIR)/test_audio_pipeline

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
	@test -f '$@' || { echo "ERROR: audio_common did not produce $@ (query/build divergence -- only BACKEND/WERROR/CC/CXX/EXTRA_CFLAGS/NO_STDIO overrides are supported)"; exit 1; }

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

$(BIN_DIR)/example_board_adapter: $(OBJ_DIR)/example_board_adapter.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) | _cfg_guard
	$(LINK) -o $@ $(OBJ_DIR)/example_board_adapter.o $(LIB_TARGET) $(AEC_LIB) $(NR_LIB) $(AC_LIB) $(LDFLAGS)
	@echo "example_board_adapter [$(BACKEND)] -> $@"
endif

# --- Query targets (round-3 review B01) -------------------------------------
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
# print-bin-dir / print-lib-path dispatch through the same three-producer
# phase-1 resolution as the build goals, so bare `make -s print-bin-dir` /
# `make -s print-lib-path` print the exact delivery locations (this repo's
# archive AND binaries only exist in the resolved-producer directory).
# Passing AC_LIB/AEC_LIB/NR_LIB explicitly skips the query sub-makes.
ifeq ($(AC_LIB),)
print-bin-dir print-lib-path:
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
endif
print-obj-dir:
	@echo $(abspath $(OBJ_DIR))

# audit-no-stdio (re-review R09, adapted to the keyed-directory layout):
# builds libaudio_pipeline.a with NO_STDIO=1 (current BACKEND=$(BACKEND))
# and asserts, via `nm` over the DELIVERED keyed archive itself (not an
# intermediate object file), that none of fprintf/printf/puts/fputs/stderr/
# __stderrp made it into it -- plus (round-3 review B01) that the NO_STDIO=1
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

# audit-no-stdio-stack (round-3 review B03, full-image gate): builds ALL FOUR
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
	 ac="$$($(MAKE) -s --no-print-directory -C $(AC_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 print-lib-path)"; \
	 aec="$$($(MAKE) -s --no-print-directory -C $(AEC_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 nr="$$($(MAKE) -s --no-print-directory -C $(NR_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" print-lib-path)"; \
	 $(MAKE) --no-print-directory -C $(AC_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 lib >/dev/null; \
	 $(MAKE) --no-print-directory -C $(AEC_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" lib >/dev/null; \
	 $(MAKE) --no-print-directory -C $(NR_DIR) BACKEND=$(BACKEND) WERROR=$(WERROR) CC='$(CC)' CXX='$(CXX)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' NO_STDIO=1 AC_DIR=$(abspath $(AC_DIR)) AC_LIB="$$ac" lib >/dev/null; \
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
	 $(CC) $(CFLAGS) -DAUDIO_PIPELINE_NO_STDIO -c -o "$$bd/test_no_stdio_stack.o" test_no_stdio_stack.c; \
	 $(LINK) -o "$$bd/test_no_stdio_stack" "$$bd/test_no_stdio_stack.o" "$$pl" "$$aec" "$$nr" "$$ac" -lm; \
	 hits=$$(nm "$$bd/test_no_stdio_stack" | awk -v re="$$pat" '{ n=$$NF; if (n ~ re) print }'); \
	 if [ -n "$$hits" ]; then echo "FAIL: stdio symbol(s) in the linked EXECUTABLE:"; echo "$$hits"; exit 1; fi; \
	 "$$bd/test_no_stdio_stack" || { rc=$$?; echo "FAIL: test_no_stdio_stack exit $$rc"; exit 1; }; \
	 echo "PASS: full NO_STDIO=1 stack -- 4 delivered archives + linked executable clean ($$pat tier), runtime exit 0"

# =============================================================================
# publish v2: immutable per-config release under dist/ (round-3 review B01)
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
# publish v4: immutable, content-addressed release under $(DIST_ROOT)
# (round-3 B01 -> round-4 P2-2 -> round-5 P2 -> round-6 P2) -- same design
# as audio_common's publish; see that Makefile for the full layout/
# semantics writeup. Round-6 additions here (mirroring audio_common's,
# extended to FOUR repos):
#   - ATTEST is one-event-one-file: EXACTLY one new
#     attest-<utc>-<commit>[-dirty]-<seq>.txt per publish EVENT (incl.
#     idempotent republishes), installed via the atomic_symlink_swap
#     helper's `--excl-install` mode (write-temp + link(2), the atomic
#     no-clobber equivalent of O_CREAT|O_EXCL) -- a same-second name
#     collision regenerates the event id with the next <seq> (capped at
#     999) and retries; an existing attestation is NEVER overwritten
#     (round-5 used cp+mv -f, so same-second republishes silently
#     clobbered the prior record).
#   - Every commit recorded is the FULL 40-hex git OID (never abbreviated),
#     for this repo (pipelines/Audio_ALG) AND all three producers
#     (audio_common, lib/aec, lib/nr) -- "which exact checkout of the whole
#     stack published this" lives in ATTEST/, not in MANIFEST (which stays
#     deterministic: config + tool identities + per-file SHA-256 only, so
#     the same release id always has the same MANIFEST bytes and republish
#     verifies it byte-for-byte).
#   - Dirty policy: publish REFUSES by default when this repo OR any
#     producer has uncommitted tracked changes or no git identity at all.
#     This repo's own dirty check uses `git status --porcelain -uno`
#     (untracked files EXCLUDED -- this working tree permanently carries
#     user-owned untracked directories, e.g. ainr/GTCRN/ and
#     ainr/gtcrn_github/, that must never make an otherwise-clean checkout
#     look dirty); the three producers use the FULL `git status
#     --porcelain` (untracked DOES count -- they carry no such permanent
#     cruft, so an untracked source file is a legitimate dirtiness signal
#     there). ALLOW_DIRTY_PUBLISH=1 is the dev escape hatch, recorded in
#     the attestation together with a dirty_diff_sha256 (sha256 of
#     `git diff --binary HEAD`) PER dirty repo, so each deviation is
#     itself traceable.
#   - Attestations are UNSIGNED: they provide traceability, not
#     authenticity -- anyone with filesystem access could forge one, so do
#     not present ATTEST/ as tamper-proof under an attacker model
#     (MANIFEST byte-verification is the integrity check; ATTEST is the
#     provenance log).
# `publish` is the locked driver (round-6: also a MAKEFLAGS word-scan so
# `-n`/`-q`/`-t` are side-effect-free -- no DIST_ROOT, no lock, no
# stage/helper/MANIFEST/ATTEST writes; see that target's own comment for
# the word-scan detail); _publish_impl is INTERNAL (direct invocation skips
# the lock) and goes through the same three-producer phase-1 dispatch as
# `all`/`test`.
# =============================================================================
PUBLISH_ARTIFACTS = $(LIB_TARGET) $(BIN_DIR)/aec_nr_pipeline $(BIN_DIR)/aec_nr_pipeline_static
DIST_BACKEND_DIR  = $(DIST_ROOT)/$(BACKEND)
DIST_LOCK         = $(DIST_ROOT)/.lock.$(BACKEND)

.PHONY: publish
# Dry-run guard (round-6 review P2-3): this recipe mentions $(MAKE), so GNU
# make executes it for real under -n/-q/-t (all three share that rule) --
# which used to mkdir $(DIST_ROOT) persistently and take/release the real
# publish lock on every dry run. The word-scan below detects those flags in
# MAKEFLAGS and branches: -n/-t exec a recursion-only path (the child make
# inherits the flag and merely prints / applies standard -t touch semantics
# to the impl chain); -q exits 1 directly (question mode's documented
# "needs updating" answer -- publish is phony, so it always would run --
# with no output). Either way: no DIST_ROOT, no lock, no stage/helper/
# MANIFEST/ATTEST writes. See audio_common's Makefile for the full MAKEFLAGS
# word-scan writeup (verified against GNU make 3.81: 17 flag combinations
# incl. long forms: bundled no-dash short flags may appear as the first
# word ("n", "sn") OR as separate dashed words ("-n" after a long option);
# long options start with "--"; a literal "--" separates flags from
# command-line variable overrides, which can themselves contain n/q/t
# -- BACKEND=ne10 -- hence the explicit skips). Normal invocations keep the
# round-5 lock-FIRST discipline unchanged.
publish:
	+@has_n=0; has_q=0; has_t=0; \
	for w in $(MAKEFLAGS); do \
	  case "$$w" in --) break ;; --*|*=*) continue ;; -n|-q|-t) ;; -*) continue ;; esac; \
	  case "$$w" in *n*) has_n=1 ;; esac; \
	  case "$$w" in *q*) has_q=1 ;; esac; \
	  case "$$w" in *t*) has_t=1 ;; esac; \
	done; \
	if [ "$$has_n" = 1 ] || [ "$$has_t" = 1 ]; then exec $(MAKE) --no-print-directory _publish_impl; fi; \
	if [ "$$has_q" = 1 ]; then exit 1; fi; \
	mkdir -p $(DIST_ROOT); \
	if ! mkdir "$(DIST_LOCK)" 2>/dev/null; then \
	  echo "FATAL: publish lock $(DIST_LOCK) already held (concurrent 'make publish BACKEND=$(BACKEND)'?)" >&2; \
	  exit 1; \
	fi; \
	trap 'rmdir "$(DIST_LOCK)" 2>/dev/null' EXIT INT TERM; \
	$(MAKE) --no-print-directory _publish_impl

# _publish_impl phase-2 (round-6 review P2): attest v2 records FOUR repos'
# provenance -- this repo (Audio_ALG/pipelines) AND all three producers
# (audio_common, lib/aec, lib/nr) -- each as a FULL 40-hex git commit OID
# plus a dirty flag, so a release's ATTEST/ directory names the exact
# source state of the entire stack, not just this repo.
#
# Self dirty check uses `git status --porcelain -uno` (untracked files
# EXCLUDED): this working tree permanently carries user-owned untracked
# directories (ainr/GTCRN/, ainr/gtcrn_github/) that are never staged for a
# publish and must never make an otherwise-clean checkout look dirty --
# only TRACKED changes count against THIS repo. The three producers use
# the FULL `git status --porcelain` (untracked files DO count): they carry
# no such permanent cruft, so an untracked source file there IS a
# legitimate dirtiness signal (e.g. a new .c not yet `git add`ed).
#
# Each producer dir is guarded with `[ -n "$(X_DIR)" ] && [ -d "$(X_DIR)" ]`
# before any `git -C` call: an EMPTY dir variable would make `git -C ''`
# silently operate on the CURRENT repo (this one) instead of failing,
# which would misattribute this repo's own commit/dirty state to that
# producer.
#
# Dirty policy (round-6 review P2): publish REFUSES by default when this
# repo OR any producer has uncommitted tracked changes (or no git identity
# at all -- commit unknown). ALLOW_DIRTY_PUBLISH=1 is the escape hatch;
# the attestation then records allow_dirty_publish=1 plus, for each repo
# that was dirty, a dirty_diff_sha256 (sha256 of `git diff --binary HEAD`)
# so the deviation itself is traceable.
ifneq ($(AC_LIB),)
.PHONY: _publish_impl
_publish_impl: $(PUBLISH_ARTIFACTS)
	@set -e; \
	work="$$(mktemp -d)"; \
	tmp="$(DIST_BACKEND_DIR)/.stage.$$$$"; \
	trap 'rm -rf "$$tmp" "$$work"' EXIT INT TERM; \
	commit="$$(git rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$commit" = "unknown" ]; then dirty=unknown; else \
	  dirty=0; [ -n "$$(git status --porcelain -uno 2>/dev/null)" ] && dirty=1; \
	fi; \
	ddiff=""; [ "$$dirty" = "1" ] && ddiff="$$(git diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	if [ -n "$(AC_DIR)" ] && [ -d "$(AC_DIR)" ]; then \
	  ac_commit="$$(git -C "$(AC_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  if [ "$$ac_commit" = "unknown" ]; then ac_dirty=unknown; else \
	    ac_dirty=0; [ -n "$$(git -C "$(AC_DIR)" status --porcelain 2>/dev/null)" ] && ac_dirty=1; \
	  fi; \
	else ac_commit=unknown; ac_dirty=unknown; fi; \
	ac_ddiff=""; [ "$$ac_dirty" = "1" ] && ac_ddiff="$$(git -C "$(AC_DIR)" diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	if [ -n "$(AEC_DIR)" ] && [ -d "$(AEC_DIR)" ]; then \
	  aec_commit="$$(git -C "$(AEC_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  if [ "$$aec_commit" = "unknown" ]; then aec_dirty=unknown; else \
	    aec_dirty=0; [ -n "$$(git -C "$(AEC_DIR)" status --porcelain 2>/dev/null)" ] && aec_dirty=1; \
	  fi; \
	else aec_commit=unknown; aec_dirty=unknown; fi; \
	aec_ddiff=""; [ "$$aec_dirty" = "1" ] && aec_ddiff="$$(git -C "$(AEC_DIR)" diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	if [ -n "$(NR_DIR)" ] && [ -d "$(NR_DIR)" ]; then \
	  nr_commit="$$(git -C "$(NR_DIR)" rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  if [ "$$nr_commit" = "unknown" ]; then nr_dirty=unknown; else \
	    nr_dirty=0; [ -n "$$(git -C "$(NR_DIR)" status --porcelain 2>/dev/null)" ] && nr_dirty=1; \
	  fi; \
	else nr_commit=unknown; nr_dirty=unknown; fi; \
	nr_ddiff=""; [ "$$nr_dirty" = "1" ] && nr_ddiff="$$(git -C "$(NR_DIR)" diff --binary HEAD 2>/dev/null | shasum -a 256 | cut -d' ' -f1)"; \
	adp=0; [ "$(ALLOW_DIRTY_PUBLISH)" = "1" ] && adp=1; \
	if [ "$$adp" != "1" ]; then \
	  bad=0; \
	  if [ "$$dirty" = "unknown" ]; then echo "FATAL: publish refused -- pipelines (this repo) has no git identity here (not a git checkout)." >&2; bad=1; \
	  elif [ "$$dirty" != "0" ]; then echo "FATAL: publish refused -- pipelines (this repo) working tree is dirty (uncommitted tracked changes)." >&2; bad=1; fi; \
	  if [ "$$ac_dirty" = "unknown" ]; then echo "FATAL: publish refused -- audio_common has no git identity (not a git checkout, or AC_DIR=$(AC_DIR) not present)." >&2; bad=1; \
	  elif [ "$$ac_dirty" != "0" ]; then echo "FATAL: publish refused -- audio_common working tree is dirty (uncommitted changes)." >&2; bad=1; fi; \
	  if [ "$$aec_dirty" = "unknown" ]; then echo "FATAL: publish refused -- lib/aec has no git identity (not a git checkout, or AEC_DIR=$(AEC_DIR) not present)." >&2; bad=1; \
	  elif [ "$$aec_dirty" != "0" ]; then echo "FATAL: publish refused -- lib/aec working tree is dirty (uncommitted changes)." >&2; bad=1; fi; \
	  if [ "$$nr_dirty" = "unknown" ]; then echo "FATAL: publish refused -- lib/nr has no git identity (not a git checkout, or NR_DIR=$(NR_DIR) not present)." >&2; bad=1; \
	  elif [ "$$nr_dirty" != "0" ]; then echo "FATAL: publish refused -- lib/nr working tree is dirty (uncommitted changes)." >&2; bad=1; fi; \
	  if [ "$$bad" = "1" ]; then \
	    echo "  Commit (or clone a clean checkout) for a board deliverable, or pass ALLOW_DIRTY_PUBLISH=1" >&2; \
	    echo "  for a dev publish -- the escape hatch is recorded in the attestation." >&2; \
	    exit 1; \
	  fi; \
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
	    echo "audio_common_git_commit=$$ac_commit"; echo "audio_common_git_dirty=$$ac_dirty"; \
	    [ -n "$$ac_ddiff" ] && echo "audio_common_dirty_diff_sha256=$$ac_ddiff"; \
	    echo "aec_git_commit=$$aec_commit"; echo "aec_git_dirty=$$aec_dirty"; \
	    [ -n "$$aec_ddiff" ] && echo "aec_dirty_diff_sha256=$$aec_ddiff"; \
	    echo "nr_git_commit=$$nr_commit"; echo "nr_git_dirty=$$nr_dirty"; \
	    [ -n "$$nr_ddiff" ] && echo "nr_dirty_diff_sha256=$$nr_ddiff"; \
	    echo "date_utc=$$dateutc"; \
	    echo "allow_dirty_publish=$$adp"; \
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
# binaries/archive left over from before this directory layout existed (a
# pre-B01 checkout, or a manual build from an older revision of this file).
# Uses the OBJ_ROOT/BIN_ROOT knobs (round-6) so `clean` with an override
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

# --- Header dependency tracking (round-3 review B01) ------------------------
# See audio_common's Makefile for the full rationale (-MD not -MMD, -MP
# guards against deleted-header stale-.d errors). $(wildcard ...) makes this
# evaluate to nothing (not an error) before any objects exist yet for this
# CFG_SIG.
-include $(wildcard $(OBJ_DIR)/*.d)
