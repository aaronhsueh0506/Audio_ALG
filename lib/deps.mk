# Audio_ALG/lib/deps.mk
#
# Single source of truth for where this repo's three external producers -- AEC, NR and audiolib --
# are read from, and at which commit each is pinned (lib/aec.pin, lib/nr.pin, lib/audiolib.pin).
# Included by every Makefile that needs AEC_DIR / NR_DIR / AUDIOLIB_DIR / COMMON_DIR
# (pipelines/Makefile, pipelines/4ch/Makefile, AINR/Makefile,
# AINR/RNNoise-ERB/Makefile, AIAEC/Makefile, stream/ainr/dfn2/Makefile,
# stream/aiaec/deepvqe_s/Makefile, stream/aiaec/ulcnet/Makefile) instead of each hardcoding its
# own relative path.
# Every value here is `?=`, so a caller that already set one on its command line or in its
# environment keeps that value untouched; this file only supplies the default.
#
# Default resolution, as of the dependency-path-convergence pass: the sibling dev checkouts this
# repo is normally developed alongside -- <Audio_ALG root>/../AEC, ../NR, ../audiolib (the
# "Recommended sole dev working directory" column of the dependency table: SE/AEC, SE/NR,
# SE/audiolib, each using its own c_impl/ for AEC and NR). The `lib/aec` and `lib/nr` git
# submodules (see .gitmodules) still exist, still carry a gitlink and still build standalone; they
# are simply no longer where a default build looks first. Point AEC_DIR/NR_DIR back at them
# explicitly (`make AEC_DIR=$(ALG_ROOT)/lib/aec/c_impl ...`) for a build that must use the
# submodule checkout instead.
#
# This file does not fetch or validate anything by itself outside its own two targets below:
#   make -f lib/deps.mk bootstrap-deps    # clone whichever of the three sibling repos is missing,
#                                          # at its pinned commit (AEC_URL/NR_URL/AUDIOLIB_URL,
#                                          # each overridable -- e.g. to a local path, for an
#                                          # offline/scratch bootstrap)
#   make -f lib/deps.mk check-deps-pins   # report (not enforce -- pipelines/Makefile's check-pins
#                                          # is what fails a build) each of AEC_DIR/NR_DIR/
#                                          # AUDIOLIB_DIR against its pin, and, while they still
#                                          # exist, the lib/aec and lib/nr gitlinks against the same
#                                          # pins (the step-5 "gitlink migration" readiness gate)
#   make -f lib/deps.mk print-deps-sh     # emit the four resolved directories as
#                                          # `eval`-able `KEY=value` shell assignments, so a shell
#                                          # script resolves them the same way a Makefile does
#                                          # instead of reimplementing the resolution itself

# This file's own absolute directory, computed before the three .pin files are included: an
# `include` changes $(MAKEFILE_LIST)'s last word to the included file, so capturing
# $(lastword $(MAKEFILE_LIST)) has to happen first, with `:=` (not `=`) so it is captured once,
# here, rather than re-evaluated lazily after the last word has moved on.
DEPS_MK_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
AUDIO_ALG_ROOT := $(patsubst %/,%,$(abspath $(DEPS_MK_DIR)/..))

include $(DEPS_MK_DIR)/aec.pin
include $(DEPS_MK_DIR)/nr.pin
include $(DEPS_MK_DIR)/audiolib.pin

AEC_REPO_DIR  ?= $(abspath $(AUDIO_ALG_ROOT)/../AEC)
NR_REPO_DIR   ?= $(abspath $(AUDIO_ALG_ROOT)/../NR)
AUDIOLIB_DIR  ?= $(abspath $(AUDIO_ALG_ROOT)/../audiolib)

AEC_DIR     ?= $(AEC_REPO_DIR)/c_impl
NR_DIR      ?= $(NR_REPO_DIR)/c_impl
# Not $(AUDIOLIB_DIR)/common: some callers (pipelines/4ch/Makefile's SP-S21/SP-S23
# isolation tests) override AUDIOLIB_DIR alone to simulate the spatial (doa/bf) libs being
# unavailable while the common FFT archive must still resolve -- deriving one from the other would
# make that override silently take COMMON_DIR down with it.
COMMON_DIR  ?= $(abspath $(AUDIO_ALG_ROOT)/../audiolib/common)

# Shared by every includer that forwards its own toolchain/flag choices to a COMMON_DIR (or other
# producer) sub-make, instead of each one spelling out the same CC/CXX/AR/RANLIB/EXTRA_CFLAGS/
# WERROR list -- a plain variable, lazily expanded at the point of use, so it costs nothing to a
# Makefile that never references it and picks up whatever values that Makefile has assigned by the
# time it's actually used (definition order relative to this `include` doesn't matter). AR/RANLIB
# matter for a cross build: without them the producer's own Makefile falls back to its own AR/
# RANLIB default (the host archiver), which a cross toolchain's .a output cannot link against.
# Defaulted here, not just in each includer: an includer that forwards AR/RANLIB without also
# defaulting them forwards an empty override that breaks the producer's own `ar`/`ranlib` calls --
# confirmed to have broken AINR/AIAEC/stream/* builds this way until every leaf Makefile was
# audited; defaulting once here closes the gap for any future includer too.
AR      ?= ar
RANLIB  ?= ranlib
TOOLCHAIN_FWD = CC='$(CC)' CXX='$(CXX)' AR='$(AR)' RANLIB='$(RANLIB)' EXTRA_CFLAGS='$(EXTRA_CFLAGS)' WERROR=$(WERROR)

# Shared by every includer that needs the audiolib common archive's path: resolves COMMON_LIB via
# a sub-make (forwarding BACKEND/SIMD/TOOLCHAIN_FWD, which the caller must already have set) and
# validates the result -- a cross-compile that can't build there, or a broken/pre-bootstrap
# sibling checkout, fails loudly instead of silently handing an empty or stale path to whatever
# recipe links against it. Call via `$(eval $(call RESOLVE_COMMON_LIB))` once BACKEND/SIMD/
# TOOLCHAIN_FWD are in scope (order relative to this `include` doesn't matter: $(eval ...) runs
# at the point it's written, same as any other top-level directive, so put the call after those
# three are assigned, not before).
#
# The entire resolution -- the sub-make call AND both validations -- is skipped outright when
# `clean` is the ONLY requested goal: that target never touches COMMON_LIB (just rm -rf a build
# directory), so it must keep working even when audiolib can't be resolved at all (a fresh
# checkout before bootstrap-deps has run, or a broken sibling). Skipping only the two $(error)
# checks and leaving the $(shell ...) sub-make call itself unconditional would still shell out and
# attempt to build audiolib as a side effect of a plain `make clean` -- wasted work, and still a
# point of failure the clean goal has no reason to depend on. `make clean <other-goal>` is NOT
# skipped, since <other-goal> may still need COMMON_LIB.
define RESOLVE_COMMON_LIB
ifneq ($$(MAKECMDGOALS),clean)
COMMON_LIB := $$(shell $$(MAKE) -s -C $$(COMMON_DIR) BACKEND=$$(BACKEND) SIMD=$$(SIMD) $$(TOOLCHAIN_FWD) \
	lib print-lib-path | tail -1)
ifeq ($$(strip $$(COMMON_LIB)),)
$$(error COMMON_LIB resolution failed: the audiolib sub-make (-C $$(COMMON_DIR)) produced no path -- check CC='$$(CC)' CXX='$$(CXX)' can build there)
endif
ifeq ($$(wildcard $$(COMMON_LIB)),)
$$(error COMMON_LIB=$$(COMMON_LIB) does not exist after the audiolib sub-make reported it built)
endif
endif
endef

.PHONY: print-deps-sh bootstrap-deps check-deps-pins

print-deps-sh:
	@printf 'AEC_DIR=%s\nNR_DIR=%s\nAUDIOLIB_DIR=%s\nCOMMON_DIR=%s\n' '$(AEC_DIR)' '$(NR_DIR)' '$(AUDIOLIB_DIR)' '$(COMMON_DIR)'

# Clones each of AEC_REPO_DIR / NR_REPO_DIR / AUDIOLIB_DIR from its pinned URL at its pinned
# commit, but only when that directory does not already exist -- an existing checkout at a
# different commit is left alone and reported (as a mismatch) by check-deps-pins, per step 4's
# "if an existing checkout is at a different version, report expected vs actual and leave the
# developer's working tree alone". AEC_URL/NR_URL/AUDIOLIB_URL may be overridden (e.g. to a local
# path or an already-cloned repo) to bootstrap without reaching the network.
#
# Clone + checkout happen in a staging directory next to the final one, moved into place only
# once `checkout` to the exact pinned commit has actually succeeded: a `git clone && git checkout`
# run directly against the final directory leaves a real, non-empty checkout behind even when the
# checkout half fails (wrong commit, network drop mid-fetch), and the plain `[ -d DIR ]` guard
# above would then see that directory on a retry and report "already exists, left alone" --
# success, at the wrong HEAD. Staging makes a failed attempt leave nothing at the final path, so a
# retry clones again instead of trusting a half-finished checkout.
#
# Two concurrent `bootstrap-deps` runs (two build jobs, or this and a sub-make's own forwarded
# call) both installing the SAME not-yet-existing DIR would otherwise both pass a plain `[ -d DIR ]`
# check, both clone+checkout successfully, and then race on `mv stage DIR`: whichever `mv` runs
# second finds DIR already exists (the first `mv` having just created it) and -- `mv` onto an
# existing, non-empty directory moves the SOURCE *inside* it, it does not replace it or fail --
# silently leaves a second, untracked, nested checkout at DIR/$(basename stage) (confirmed: this
# is exactly what reproduces). Each of the three installs below is therefore serialized by its own
# mkdir-based lock (mkdir is atomic across processes on every POSIX filesystem this targets), but
# only when DIR doesn't already exist -- an already-prepared dependency reports and returns without
# ever touching the lock, so a read-only parent directory (lock creation impossible, but nothing
# needs creating) doesn't fail a build that has nothing to bootstrap. Once in the lock, `[ -d DIR ]`
# is re-checked AFTER acquiring it, not before -- a loser of the race then correctly sees the
# winner's now-finished DIR and reports "already exists", instead of still racing it. A failed
# `mkdir` is always retried (up to the 30-minute timeout below) rather than diagnosed by whether the
# lock directory is momentarily absent: that absence is indistinguishable from "the winner just
# released it" (confirmed: widening this exact window turned most concurrent runs into false
# failures), so only sustained failure to acquire the lock -- not a single missing instant -- is
# reported as stuck/misconfigured.
BOOTSTRAP_LOCK_WAIT = @if [ -d '$(2)' ]; then \
	  echo "bootstrap-deps: $(1) checkout $(2) already exists, left alone"; \
	else \
	  lock='$(2).bootstrap-lock'; tries=0; \
	  while ! mkdir "$$lock" 2>/dev/null; do \
	    tries=$$((tries + 1)); \
	    if [ "$$tries" -ge 1800 ]; then \
	      echo "bootstrap-deps: FAIL could not acquire $$lock after 30 minutes (another bootstrap-deps stuck on a very slow clone, a crashed one left the lock behind -- remove it by hand if so -- or the parent directory is not writable)" >&2; \
	      exit 1; \
	    fi; \
	    sleep 1; \
	  done; \
	  trap 'rmdir "$$lock" 2>/dev/null' EXIT; \
	  if [ -d '$(2)' ]; then \
	    echo "bootstrap-deps: $(1) checkout $(2) already exists, left alone"; \
	  else \
	    echo "bootstrap-deps: cloning $(1) $(4) from $(3) -> $(2)"; \
	    stage='$(2).bootstrap-staging.'"$$$$"; rm -rf "$$stage"; \
	    if git clone --quiet '$(3)' "$$stage" && git -C "$$stage" checkout --quiet '$(4)'; then \
	      mv "$$stage" '$(2)'; \
	    else \
	      echo "bootstrap-deps: FAIL could not clone+checkout $(1) $(4) from $(3)" >&2; rm -rf "$$stage"; exit 1; \
	    fi; \
	  fi; \
	fi

bootstrap-deps:
	$(call BOOTSTRAP_LOCK_WAIT,AEC,$(AEC_REPO_DIR),$(AEC_URL),$(AEC_COMMIT))
	$(call BOOTSTRAP_LOCK_WAIT,NR,$(NR_REPO_DIR),$(NR_URL),$(NR_COMMIT))
	$(call BOOTSTRAP_LOCK_WAIT,audiolib,$(AUDIOLIB_DIR),$(AUDIOLIB_URL),$(AUDIOLIB_COMMIT))

# Report each pinned dependency against the directory a build actually resolves to -- AEC_DIR/
# NR_DIR/COMMON_DIR, honoring any override of those directly (not AEC_REPO_DIR/NR_REPO_DIR/
# AUDIOLIB_DIR, which stay at their own default if a caller overrides AEC_DIR/NR_DIR/COMMON_DIR
# instead and would then silently report the unused default's identity) -- AND, while the lib/aec
# and lib/nr submodules still exist, their gitlinks against the same pins -- this second half is
# the step-5 readiness gate: step 5 (migrating the two gitlinks onto these pins, and retiring the
# now-redundant one of the two) is not proposed until every row below agrees. pipelines/Makefile's
# check-pins target is what actually gates a real build/publish; this target is a standalone,
# caller-invoked report with the same exit-nonzero-on-mismatch behavior.
check-deps-pins:
	@ok=1; \
	have_aec="$$(git -C '$(AEC_DIR)' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$have_aec" = "$(AEC_COMMIT)" ]; then echo "check-deps-pins: AEC ($(AEC_DIR)) $$have_aec matches lib/aec.pin"; \
	else echo "check-deps-pins: FAIL AEC ($(AEC_DIR)) is $$have_aec, lib/aec.pin expects $(AEC_COMMIT)" >&2; ok=0; fi; \
	have_nr="$$(git -C '$(NR_DIR)' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$have_nr" = "$(NR_COMMIT)" ]; then echo "check-deps-pins: NR ($(NR_DIR)) $$have_nr matches lib/nr.pin"; \
	else echo "check-deps-pins: FAIL NR ($(NR_DIR)) is $$have_nr, lib/nr.pin expects $(NR_COMMIT)" >&2; ok=0; fi; \
	have_ac="$$(git -C '$(AUDIOLIB_DIR)' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$have_ac" = "$(AUDIOLIB_COMMIT)" ]; then echo "check-deps-pins: audiolib ($(AUDIOLIB_DIR)) $$have_ac matches lib/audiolib.pin"; \
	else echo "check-deps-pins: FAIL audiolib ($(AUDIOLIB_DIR)) is $$have_ac, lib/audiolib.pin expects $(AUDIOLIB_COMMIT)" >&2; ok=0; fi; \
	have_common="$$(git -C '$(COMMON_DIR)' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	if [ "$$have_common" = "$(AUDIOLIB_COMMIT)" ]; then echo "check-deps-pins: audiolib common ($(COMMON_DIR)) $$have_common matches lib/audiolib.pin"; \
	else echo "check-deps-pins: FAIL audiolib common ($(COMMON_DIR)) is $$have_common, lib/audiolib.pin expects $(AUDIOLIB_COMMIT)" >&2; ok=0; fi; \
	if [ -e '$(AUDIO_ALG_ROOT)/lib/aec/.git' ]; then \
	  gl_rec="$$(git -C '$(AUDIO_ALG_ROOT)' ls-files -s -- lib/aec 2>/dev/null | awk '$$3 == 0 {print $$2}')"; \
	  gl_have="$$(git -C '$(AUDIO_ALG_ROOT)/lib/aec' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  if [ "$$gl_rec" = "$$gl_have" ] && [ "$$gl_have" = "$(AEC_COMMIT)" ]; then \
	    echo "check-deps-pins: lib/aec gitlink $$gl_have matches lib/aec.pin (step 5 ready on this axis)"; \
	  else \
	    echo "check-deps-pins: NOTE lib/aec gitlink is $$gl_have (index records $$gl_rec), lib/aec.pin expects $(AEC_COMMIT) -- step 5 not ready"; \
	  fi; \
	else \
	  echo "check-deps-pins: lib/aec submodule not initialized (not needed for the default sibling build)"; \
	fi; \
	if [ -e '$(AUDIO_ALG_ROOT)/lib/nr/.git' ]; then \
	  gl_rec="$$(git -C '$(AUDIO_ALG_ROOT)' ls-files -s -- lib/nr 2>/dev/null | awk '$$3 == 0 {print $$2}')"; \
	  gl_have="$$(git -C '$(AUDIO_ALG_ROOT)/lib/nr' rev-parse HEAD 2>/dev/null || echo unknown)"; \
	  if [ "$$gl_rec" = "$$gl_have" ] && [ "$$gl_have" = "$(NR_COMMIT)" ]; then \
	    echo "check-deps-pins: lib/nr gitlink $$gl_have matches lib/nr.pin (step 5 ready on this axis)"; \
	  else \
	    echo "check-deps-pins: NOTE lib/nr gitlink is $$gl_have (index records $$gl_rec), lib/nr.pin expects $(NR_COMMIT) -- step 5 not ready"; \
	  fi; \
	else \
	  echo "check-deps-pins: lib/nr submodule not initialized (not needed for the default sibling build)"; \
	fi; \
	[ "$$ok" = 1 ] || exit 1

# Every includer puts `include lib/deps.mk` before its own first real target, so without this,
# GNU Make's "first target in the makefile becomes .DEFAULT_GOAL" rule would silently make
# print-deps-sh the default instead of the includer's own (all / _cfg_guard / ...). Clearing
# .DEFAULT_GOAL makes Make pick it up again from the next target the includer defines.
.DEFAULT_GOAL :=
